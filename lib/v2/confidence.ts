/**
 * Calibrated confidence + abstention policy.
 *
 * Per the PDFs, abstaining on 1–3% of moves with a "tap to confirm"
 * fallback is the difference between a 99.7%+ user-visible quality
 * system and a silent ~99% one. Silent errors poison every later diff;
 * abstention is a feature.
 *
 * Inputs (all in the same struct):
 *   - top-1 score and top-1/top-2 margin from the beam search
 *   - sharpest-frame Laplacian variance (high = clear capture)
 *   - max per-square delta of the changed set (high = strong evidence)
 *   - number of changed squares (2 = simple move, 3 = EP, 4 = castle,
 *     >4 = something is wrong)
 *
 * We don't have training data on day 1 so this is a hand-tuned logistic
 * with the structure recommended by the technical-plan PDF. As soon as
 * we have user-labeled data, swap the coefficients for a fitted model
 * without changing the calling surface.
 */

import type { MoveCandidate, DiffResult } from "./types";

export type ConfidenceFeatures = {
  top1Score: number;
  top1Top2Margin: number;
  /** Sharpness of the post-move frame. */
  laplacian: number;
  /** Max per-square delta of the changed set. */
  maxDelta: number;
  /** Sum of all changed-square deltas. */
  totalDelta: number;
  /** Number of changed squares the diff detector found. */
  changedCount: number;
  /** Whether the top candidate's template fully covers the changed set. */
  templateExact: boolean;
};

/**
 * Hand-tuned logistic: P(top-1 is correct).
 *
 * Coefficients chosen from the PDF's heuristic guidance:
 *   - intercept: -1.6 (a 50% base rate)
 *   - margin: large positive — the top-1/top-2 gap is the single most
 *     informative feature (a clear winner is almost always right)
 *   - templateExact: large positive — exact-match means no slack used
 *   - laplacian: moderate positive — sharp frames are more trustworthy
 *   - changedCount=2 bonus: simple moves are easier than castles
 */
export function pCorrect(f: ConfidenceFeatures): number {
  let z = -1.6;
  z += 2.8 * Math.tanh(f.top1Top2Margin); // 0 → 0, 0.5 → 0.46, 2 → 0.96, 5 → 0.99+
  z += 1.4 * Math.tanh(f.top1Score / 2);
  z += f.templateExact ? 1.2 : -0.6;
  // Laplacian is unbounded; tanh-squash with a sane scale.
  z += 0.8 * Math.tanh(f.laplacian / 400);
  z += 1.0 * Math.tanh(f.maxDelta * 3); // maxDelta in [0,1]; 0.4 → 0.83
  // Simple moves are far more reliable than rare-rule moves.
  if (f.changedCount === 2) z += 0.4;
  else if (f.changedCount === 3) z += 0.0;
  else if (f.changedCount === 4) z -= 0.2;
  else z -= 0.8; // 5+ changed = something is wrong (hand in frame?)
  return sigmoid(z);
}

/**
 * Bundle the features the calibrator needs from one move-detection
 * attempt. Convenience helper.
 */
export function buildFeatures(
  candidates: MoveCandidate[],
  diff: DiffResult,
): ConfidenceFeatures {
  const top1 = candidates[0];
  const top2 = candidates[1];
  const margin = top2 ? top1.score - top2.score : top1?.score ?? 0;
  const changedDeltas = diff.changedSquares.map(
    (sq) => diff.perSquareDelta[squareIndex(sq)],
  );
  const maxDelta = changedDeltas.length ? Math.max(...changedDeltas) : 0;
  const totalDelta = changedDeltas.reduce((a, b) => a + b, 0);
  const templateExact = top1
    ? top1.touchedSquares.every((sq) => diff.changedSquares.includes(sq))
    : false;
  return {
    top1Score: top1?.score ?? 0,
    top1Top2Margin: margin,
    laplacian: diff.frameSharpness,
    maxDelta,
    totalDelta,
    changedCount: diff.changedSquares.length,
    templateExact,
  };
}

function squareIndex(sq: string): number {
  const file = sq.charCodeAt(0) - "a".charCodeAt(0);
  const rank = 8 - parseInt(sq[1], 10);
  return rank * 8 + file;
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}
