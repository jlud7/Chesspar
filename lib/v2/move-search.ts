/**
 * Legal-move beam search — the core inference step.
 *
 * From the previous FEN, enumerate every legal move via chess.js. For
 * each move, compute its "touched-square template" (the set of squares
 * whose occupancy changes when this move is played). Score each candidate
 * by how well that template matches the observed diff. Promotion,
 * castling, and en passant get explicit templates because they touch
 * unusual squares (rook hop on castle, captured pawn on EP, piece-class
 * change on promotion).
 *
 * This step is where the +2–3pp gain in the PDFs comes from. A ~95%
 * per-square classifier still leaves ~5% per-move error if you argmax;
 * legality + beam search collapses that 5% to <1% because the prior is
 * worth ~7 bits — far more than any single classifier improvement.
 */

import { Chess, type Move } from "chess.js";
import type { DiffResult, MoveCandidate, Square } from "./types";
import { indexToSquare, squareToIndex } from "./types";

export type ScoringOptions = {
  /**
   * Slack for the touched-square match: a candidate is admitted if its
   * touched-set ⊆ (changed-set ∪ N slack squares). Default 1 — the
   * diff detector occasionally drops one of two changed squares.
   */
  slack?: number;
  /**
   * Bonus added to the diff-alignment subscore when each touched square
   * is among the top-K of the changed list. The PDFs note that scoring
   * over "top-3 source × top-3 destination" beats greedy argmax.
   */
  rankBonus?: number;
};

export type ClassifierEvidence = {
  /**
   * Per-square log P(piece on square) under the trained classifier.
   * Length 64, row-major from a8. Optional — when absent we fall back
   * to a uniform prior and rely entirely on diff alignment.
   */
  destLogProb?: Map<Square, number>;
};

/**
 * Enumerate legal candidates from `prevFen` whose touched-square set is
 * consistent with `diff.changedSquares`, then score and rank them.
 */
export function searchLegalMoves(
  prevFen: string,
  diff: DiffResult,
  classifier?: ClassifierEvidence,
  opts: ScoringOptions = {},
): MoveCandidate[] {
  const slack = opts.slack ?? 1;
  const rankBonus = opts.rankBonus ?? 0.1;

  const game = new Chess(prevFen);
  const legal = game.moves({ verbose: true });
  const changed = new Set(diff.changedSquares);
  // Pre-compute changed-square rank table: ratio in [0, 1] of how high
  // each changed square sits in the diff's sorted list. Used for the
  // rank bonus.
  const changedRank = new Map<Square, number>();
  diff.changedSquares.forEach((sq, i) =>
    changedRank.set(sq, 1 - i / Math.max(1, diff.changedSquares.length)),
  );

  const candidates: MoveCandidate[] = [];
  for (const move of legal) {
    const template = touchedSquaresFor(move);
    // Admit if (template ⊆ changed) ∨ (template \ changed has ≤ slack).
    const missing = template.filter((sq) => !changed.has(sq)).length;
    if (missing > slack) continue;

    const sim = new Chess(prevFen);
    sim.move({ from: move.from, to: move.to, promotion: move.promotion });
    const resultingFen = sim.fen();

    // --- Score components ---
    // (1) Diff alignment: how much of the total delta mass lives in the
    //     template? Higher = better fit. Computed by summing perSquareDelta
    //     over the template and normalizing by the total mass of changed
    //     squares.
    let templateMass = 0;
    let totalChangedMass = 0;
    for (const sq of template) {
      templateMass += diff.perSquareDelta[squareToIndex(sq)];
    }
    for (const sq of diff.changedSquares) {
      totalChangedMass += diff.perSquareDelta[squareToIndex(sq)];
    }
    const diffAlign =
      totalChangedMass > 0 ? templateMass / totalChangedMass : 0;
    // Plus a small bonus when each touched square also sits high in
    // the changed-square ranking.
    let rankSum = 0;
    for (const sq of template) {
      rankSum += changedRank.get(sq) ?? 0;
    }
    const diffAlignment = diffAlign + (rankBonus * rankSum) / template.length;

    // (2) Classifier log P: only on the destination square. Per the PDFs
    //     we don't classify all 64 — just the destination, which is the
    //     one with new evidence (capture/move) the diff alone can't
    //     resolve (e.g., promotion piece identity).
    const destSq = move.to as Square;
    const classifierLogP = classifier?.destLogProb?.get(destSq) ?? 0;

    // (3) Engine prior — filled in by a higher layer if Stockfish is on.
    //     Kept at 0 here; the move-pipeline orchestrator may add it.
    const enginePrior = 0;

    const score = diffAlignment + classifierLogP + enginePrior;
    candidates.push({
      san: move.san,
      uci: move.from + move.to + (move.promotion ?? ""),
      fromSquare: move.from as Square,
      toSquare: move.to as Square,
      resultingFen,
      touchedSquares: template,
      score,
      subscores: { diffAlignment, classifierLogP, enginePrior },
    });
  }
  // Best (highest score) first. The top-1/top-2 margin feeds the
  // confidence model in `confidence.ts`.
  candidates.sort((a, b) => b.score - a.score);
  return candidates;
}

/**
 * Touched-square template for a chess.js `Move`. These are the squares
 * whose occupancy changes when the move is played — what the diff
 * detector should see.
 *
 *   - normal move: [from, to]
 *   - capture: [from, to] (same, since `to` just changes colour)
 *   - castle: [from, to, rook-from, rook-to] (4 squares)
 *   - en passant: [from, to, captured-pawn-square] (3 squares)
 *   - promotion: [from, to] (the diff is identical to a normal move
 *     in occupancy terms; piece identity is resolved by the classifier)
 */
export function touchedSquaresFor(move: Move): Square[] {
  const sqs: Square[] = [move.from as Square, move.to as Square];
  // Castling: chess.js flags it as 'k' (kingside) or 'q' (queenside).
  if (move.flags.includes("k")) {
    const rank = move.from[1];
    sqs.push(`h${rank}` as Square, `f${rank}` as Square);
    return sqs;
  }
  if (move.flags.includes("q")) {
    const rank = move.from[1];
    sqs.push(`a${rank}` as Square, `d${rank}` as Square);
    return sqs;
  }
  // En passant: the captured pawn sits on the same rank as the moving
  // pawn's source. chess.js sets the 'e' flag.
  if (move.flags.includes("e")) {
    const file = move.to[0];
    const rank = move.from[1];
    sqs.push(`${file}${rank}` as Square);
    return sqs;
  }
  return sqs;
}

/**
 * Convenience: human-readable summary of why each candidate scored as
 * it did. Used by the debug overlay so the user can see what the model
 * was thinking.
 */
export function explainCandidate(c: MoveCandidate): string {
  const parts = [
    `align=${c.subscores.diffAlignment.toFixed(2)}`,
    `dest=${c.subscores.classifierLogP.toFixed(2)}`,
    `engine=${c.subscores.enginePrior.toFixed(2)}`,
  ];
  return `${c.san} (${parts.join(", ")} = ${c.score.toFixed(2)})`;
}

/** Reduce a candidate list to its top-N. Stable, preserves rank order. */
export function topN(cs: MoveCandidate[], n: number): MoveCandidate[] {
  return cs.slice(0, n);
}

export { indexToSquare, squareToIndex };
