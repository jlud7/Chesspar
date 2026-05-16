/**
 * Occupancy delta — find which 2–6 squares changed between two rectified
 * frames, in HSV-V space with per-square photometric normalization.
 *
 * Per the PDFs, this is the core insight of the diff-first pipeline: do
 * not classify all 64 squares. Just ask "which squares look different
 * from the cached pre-move frame?" The legality+beam-search downstream
 * then decides which legal move explains the change.
 *
 * Why HSV-V not RGB: shadows shift RGB significantly but preserve hue.
 * Working in the V channel after histogram normalization kills most
 * shadow false positives.
 */

import type { DiffResult, Square } from "./types";
import { indexToSquare } from "./types";

export type DiffOptions = {
  /** Canonical square edge in pixels (rectified board is `size × size`). */
  size: number;
  /**
   * Pad the inner crop by this fraction of the square edge — the
   * outer 18% is often the dark/light border which adds noise.
   */
  innerPad?: number;
  /**
   * Per-square delta threshold in [0, 1] for a square to be flagged.
   * Defaults to 0.12 per the PDFs; tune per device.
   */
  threshold?: number;
};

/**
 * Compute the per-square HSV-V delta between two rectified boards. The
 * top-K squares above `threshold` are returned as candidate-changed.
 */
export function computeOccupancyDelta(
  pre: HTMLCanvasElement,
  post: HTMLCanvasElement,
  opts: DiffOptions,
): DiffResult {
  const size = opts.size;
  const pad = opts.innerPad ?? 0.18;
  const threshold = opts.threshold ?? 0.12;
  if (pre.width !== size || pre.height !== size) {
    throw new Error(`pre canvas must be ${size}×${size}, got ${pre.width}×${pre.height}`);
  }
  if (post.width !== size || post.height !== size) {
    throw new Error(`post canvas must be ${size}×${size}, got ${post.width}×${post.height}`);
  }

  const preV = toVChannel(pre);
  const postV = toVChannel(post);

  // Photometric normalization: shift the post frame's per-row mean to
  // match the pre frame's per-row mean. Kills "someone turned on a lamp
  // mid-game" false positives without invalidating real piece changes.
  normalizeRowMeans(preV, postV, size);

  const square = size / 8;
  const innerOffset = Math.round(square * pad);
  const inner = Math.max(1, Math.round(square - 2 * innerOffset));

  const perSquare = new Array<number>(64);
  for (let idx = 0; idx < 64; idx++) {
    const r = Math.floor(idx / 8);
    const c = idx % 8;
    const x0 = Math.round(c * square + innerOffset);
    const y0 = Math.round(r * square + innerOffset);
    let sum = 0;
    let n = 0;
    for (let y = y0; y < y0 + inner; y++) {
      const rowPre = y * size;
      for (let x = x0; x < x0 + inner; x++) {
        sum += Math.abs(preV[rowPre + x] - postV[rowPre + x]);
        n++;
      }
    }
    // Normalize to [0, 1]: max-possible diff is 255 (V is uint8).
    perSquare[idx] = n === 0 ? 0 : sum / n / 255;
  }

  const changed: Square[] = [];
  for (let i = 0; i < 64; i++) {
    if (perSquare[i] > threshold) changed.push(indexToSquare(i));
  }
  // Sort by delta descending — beam search wants the highest-confidence
  // changed squares first.
  changed.sort((a, b) => {
    const ai = squareToIdx(a);
    const bi = squareToIdx(b);
    return perSquare[bi] - perSquare[ai];
  });
  // Cap at the top 6 — castling touches 4 squares; en passant 3; anything
  // beyond 6 candidates is probably noise (a hand still in frame, or a
  // global luminance shift the normalization didn't fully cancel).
  const capped = changed.slice(0, 6);

  return {
    changedSquares: capped,
    perSquareDelta: perSquare,
    frameSharpness: 0, // filled in by caller
  };
}

function squareToIdx(sq: Square): number {
  const file = sq.charCodeAt(0) - "a".charCodeAt(0);
  const rank = 8 - parseInt(sq[1], 10);
  return rank * 8 + file;
}

/**
 * Convert a canvas to an HSV-V uint8 buffer. V = max(R, G, B).
 *
 * V is shadow-tolerant in a way luminance isn't: a piece-shadow on an
 * empty square keeps the underlying hue but darkens R, G, and B
 * proportionally — V drops but not as much as overall luminance, and
 * the per-row normalization below brings it back into line.
 */
function toVChannel(canvas: HTMLCanvasElement): Uint8Array {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable for V channel");
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const out = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    out[j] = r > g ? (r > b ? r : b) : g > b ? g : b;
  }
  return out;
}

/**
 * Shift `post` so each horizontal row's mean equals the corresponding
 * row's mean in `pre`. Conservative: clamps the per-row shift to ±40
 * to avoid wiping out a row that legitimately changed (e.g., a major
 * piece moved into rank 4 from rank 1).
 */
function normalizeRowMeans(
  pre: Uint8Array,
  post: Uint8Array,
  size: number,
): void {
  for (let y = 0; y < size; y++) {
    let preSum = 0;
    let postSum = 0;
    const row = y * size;
    for (let x = 0; x < size; x++) {
      preSum += pre[row + x];
      postSum += post[row + x];
    }
    const shift = Math.max(-40, Math.min(40, (preSum - postSum) / size));
    if (shift === 0) continue;
    for (let x = 0; x < size; x++) {
      const v = post[row + x] + shift;
      post[row + x] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
}
