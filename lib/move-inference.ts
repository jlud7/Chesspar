import { Chess, type Move } from "chess.js";
import type { Occupancy } from "./occupancy";

export type SquareDiff = {
  square: string;
  before: Occupancy;
  after: Occupancy;
};

export type InferResult =
  | { kind: "matched"; move: Move; updatedFen: string; diff: SquareDiff[] }
  | { kind: "ambiguous"; candidates: Move[]; diff: SquareDiff[] }
  | { kind: "none"; diff: SquareDiff[]; legalMoveCount: number };

export type FuzzyCandidate = {
  move: Move;
  updatedFen: string;
  /** Count of squares where expected occupancy disagrees with observed. */
  mismatch: number;
  /**
   * Penalty score: mismatch with each cell weighted by 1 − its classifier
   * confidence, so confidently-classified disagreements cost more than
   * marginal ones.
   */
  weightedMismatch: number;
};

export type FuzzyInferResult = {
  diff: SquareDiff[];
  /** All legal moves ranked best-first by weighted mismatch. */
  ranked: FuzzyCandidate[];
  /** The top candidate's classical result kind. */
  kind: "matched" | "ambiguous" | "none";
  /** Best-pick fully-applied move (only when `kind === 'matched'`). */
  pick?: { move: Move; updatedFen: string };
};

/**
 * Render a FEN as a length-64 array of {empty,white,black}, indexed
 * row-major from a8 (idx 0) to h1 (idx 63). This matches the layout that
 * `extractSquareCrops` produces from a rectified board.
 */
export function fenToOccupancy(fen: string): Occupancy[] {
  const game = new Chess(fen);
  const board = game.board();
  const out: Occupancy[] = [];
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const cell = board[r][f];
      if (!cell) out.push("empty");
      else out.push(cell.color === "w" ? "white" : "black");
    }
  }
  return out;
}

export function diffOccupancy(
  before: Occupancy[],
  after: Occupancy[],
): SquareDiff[] {
  const out: SquareDiff[] = [];
  for (let i = 0; i < 64; i++) {
    if (before[i] !== after[i]) {
      const file = "abcdefgh"[i % 8];
      const rank = 8 - Math.floor(i / 8);
      out.push({ square: `${file}${rank}`, before: before[i], after: after[i] });
    }
  }
  return out;
}

/**
 * Strict inference: return the unique legal move whose result EXACTLY
 * matches the observed occupancy. Kept for backwards-compat with the
 * existing /detect inspector — the production pipeline now prefers
 * `inferMoveFuzzy`, which tolerates a handful of misclassifications.
 */
export function inferMove(prevFen: string, observed: Occupancy[]): InferResult {
  if (observed.length !== 64) {
    throw new Error(
      `inferMove expects 64 occupancy entries, got ${observed.length}`,
    );
  }
  const game = new Chess(prevFen);
  const prevOcc = fenToOccupancy(prevFen);
  const diff = diffOccupancy(prevOcc, observed);
  const legal = game.moves({ verbose: true });
  const matches: { move: Move; updatedFen: string }[] = [];

  for (const move of legal) {
    const sim = new Chess(prevFen);
    try {
      sim.move({ from: move.from, to: move.to, promotion: move.promotion });
    } catch {
      continue;
    }
    const expected = fenToOccupancy(sim.fen());
    if (occupancyEquals(observed, expected)) {
      matches.push({ move, updatedFen: sim.fen() });
    }
  }

  if (matches.length === 1) {
    return {
      kind: "matched",
      move: matches[0].move,
      updatedFen: matches[0].updatedFen,
      diff,
    };
  }
  if (matches.length > 1) {
    return { kind: "ambiguous", candidates: matches.map((m) => m.move), diff };
  }
  return { kind: "none", diff, legalMoveCount: legal.length };
}

/**
 * Fuzzy inference, with **frame-to-frame diff matching** as the primary
 * strategy.
 *
 * The classifier can mis-classify the same cell consistently across
 * frames — e.g., always labelling a particular empty square as "white"
 * because of piece-top bleed from an adjacent rank. Comparing this frame
 * directly to the FEN-derived "expected" occupancy treats those
 * consistent errors as mismatches, drowning the real move in classifier
 * noise.
 *
 * Comparing this frame to the *previous observed frame* fixes this: a
 * consistent error cancels out (the same wrong label appears in both
 * frames). Only cells that genuinely changed between the two photos
 * register as differences. We then match those against each legal move's
 * expected before→after delta.
 *
 * If `previousObserved` is not supplied (e.g., the very first move and
 * no calibration-frame classification was kept), we fall back to the
 * FEN-derived baseline — same behaviour as the strict matcher.
 */
export function inferMoveFuzzy(
  prevFen: string,
  observed: Occupancy[],
  options: {
    previousObserved?: Occupancy[];
    confidences?: number[];
    /**
     * Per-cell pixel-level change between the previous and current
     * rectified frames. Cells with high delta are likely involved in the
     * move regardless of how the classifier labelled them — robust
     * tiebreaker when the classifier misses a small pawn.
     */
    cellDeltas?: number[];
  } = {},
): FuzzyInferResult {
  if (observed.length !== 64) {
    throw new Error(
      `inferMoveFuzzy expects 64 occupancy entries, got ${observed.length}`,
    );
  }
  const game = new Chess(prevFen);
  const fenPrev = fenToOccupancy(prevFen);
  const baseline = options.previousObserved ?? fenPrev;
  if (baseline.length !== 64) {
    throw new Error(
      `inferMoveFuzzy: previousObserved must be 64 entries, got ${baseline.length}`,
    );
  }
  const diff = diffOccupancy(baseline, observed);
  const legal = game.moves({ verbose: true });
  const conf = options.confidences ?? new Array(64).fill(0.7);
  const deltas = options.cellDeltas;
  // Pre-compute baselines for the delta-aware reward. Pixel deltas above
  // a noise floor count as evidence; we use the 60th-percentile delta as
  // a heuristic noise band (most cells don't change in a typical move).
  let deltaUseful = false;
  let deltaNoise = 0;
  let deltaMax = 0;
  if (deltas && deltas.length === 64) {
    const sorted = [...deltas].sort((a, b) => a - b);
    deltaNoise = sorted[Math.floor(sorted.length * 0.6)] || 0;
    deltaMax = sorted[sorted.length - 1] || 0;
    if (deltaMax > Math.max(10, deltaNoise * 2)) deltaUseful = true;
  }
  const deltaGate = Math.max(8, deltaNoise * 1.5);

  const candidates: FuzzyCandidate[] = [];
  for (const move of legal) {
    const sim = new Chess(prevFen);
    try {
      sim.move({ from: move.from, to: move.to, promotion: move.promotion });
    } catch {
      continue;
    }
    const fenAfter = fenToOccupancy(sim.fen());

    // Touched-cell set: cells where the move actually changes the
    // occupancy. Usual moves touch 2 cells; castling 4; en-passant 3.
    const touchedSet = new Set<number>();
    for (let i = 0; i < 64; i++) {
      if (fenPrev[i] !== fenAfter[i]) touchedSet.add(i);
    }

    let mismatch = 0;
    let weighted = 0;
    let touchedDeltaSum = 0;
    let untouchedDeltaPenalty = 0;
    let touchedLowDeltaPenalty = 0;
    for (let i = 0; i < 64; i++) {
      const touched = touchedSet.has(i);
      const predicted = touched ? fenAfter[i] : baseline[i];
      if (predicted !== observed[i]) {
        mismatch++;
        weighted += 0.3 + conf[i];
      }
      if (deltaUseful && deltas) {
        const d = deltas[i];
        if (touched) {
          touchedDeltaSum += d;
          if (d < deltaGate) touchedLowDeltaPenalty += 1.5;
        } else if (d > deltaGate && !isAdjacent(i, touchedSet)) {
          // Spurious high delta — only counts as evidence against if
          // the cell isn't immediately adjacent to a touched square
          // (immediate neighbours commonly bleed via piece-top parallax).
          untouchedDeltaPenalty += 0.25 * (d / deltaGate);
        }
      }
    }

    if (deltaUseful) {
      // Pull score down (better) when touched cells have strong delta
      // and untouched cells are quiet.
      weighted -= touchedDeltaSum / Math.max(deltaGate, 1) / 2;
      weighted += touchedLowDeltaPenalty + untouchedDeltaPenalty;
    }

    candidates.push({
      move,
      updatedFen: sim.fen(),
      mismatch,
      weightedMismatch: weighted,
    });
  }
  candidates.sort(
    (a, b) =>
      a.weightedMismatch - b.weightedMismatch || a.mismatch - b.mismatch,
  );

  if (candidates.length === 0) {
    return { diff, ranked: [], kind: "none" };
  }
  const best = candidates[0];
  const next = candidates[1];
  const margin = next ? next.weightedMismatch - best.weightedMismatch : Infinity;

  if (best.mismatch === 0) {
    return matched(diff, candidates);
  }
  if (best.mismatch <= 1 && best.weightedMismatch <= 1.4) {
    return matched(diff, candidates);
  }
  if (best.mismatch <= 2 && margin >= 0.7) {
    return matched(diff, candidates);
  }
  if (best.mismatch <= 3 && margin >= 1.5) {
    return matched(diff, candidates);
  }
  if (next && next.mismatch === best.mismatch && margin < 0.5) {
    return { diff, ranked: candidates, kind: "ambiguous" };
  }
  return { diff, ranked: candidates, kind: "none" };
}

function matched(
  diff: SquareDiff[],
  candidates: FuzzyCandidate[],
): FuzzyInferResult {
  const best = candidates[0];
  return {
    diff,
    ranked: candidates,
    kind: "matched",
    pick: { move: best.move, updatedFen: best.updatedFen },
  };
}

function occupancyEquals(a: Occupancy[], b: Occupancy[]): boolean {
  for (let i = 0; i < 64; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function isAdjacent(idx: number, set: Set<number>): boolean {
  const r = Math.floor(idx / 8);
  const f = idx % 8;
  for (let dr = -1; dr <= 1; dr++) {
    for (let df = -1; df <= 1; df++) {
      if (dr === 0 && df === 0) continue;
      const nr = r + dr;
      const nf = f + df;
      if (nr < 0 || nr >= 8 || nf < 0 || nf >= 8) continue;
      if (set.has(nr * 8 + nf)) return true;
    }
  }
  return false;
}
