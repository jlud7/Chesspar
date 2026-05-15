export type Point = { x: number; y: number };

export type Matrix3x3 = readonly [
  readonly [number, number, number],
  readonly [number, number, number],
  readonly [number, number, number],
];

/**
 * Compute a 3x3 homography H mapping src[i] -> dst[i] (up to scale).
 * Requires exactly 4 point pairs in general position (no three collinear).
 *
 * Uses Direct Linear Transform: each pair contributes 2 rows to an 8x8
 * system that solves for the 8 free entries of H (h22 is fixed at 1).
 */
export function computeHomography(src: Point[], dst: Point[]): Matrix3x3 {
  if (src.length !== 4 || dst.length !== 4) {
    throw new Error("computeHomography needs exactly 4 source and 4 destination points");
  }
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = src[i];
    const { x: X, y: Y } = dst[i];
    A.push([x, y, 1, 0, 0, 0, -X * x, -X * y]);
    A.push([0, 0, 0, x, y, 1, -Y * x, -Y * y]);
    b.push(X);
    b.push(Y);
  }
  const h = solveLinearSystem(A, b);
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ];
}

/**
 * Least-squares homography fit for N ≥ 4 point pairs.
 *
 * Same DLT formulation as `computeHomography`, but for N pairs we build
 * a (2N × 8) system and solve it via the normal equations
 *   (Aᵀ A) h = Aᵀ b
 * which collapses to an 8x8 system solvable with the same Gaussian
 * elimination routine. With h22 fixed at 1, h has 8 free entries.
 *
 * Used by the chess-board detector to fit a perspective transform from
 * cell-space (file, rank) coordinates onto the centroids of every
 * detected dark square (~28-32 correspondences across the board). This
 * is what makes corner placement robust to phone-angle perspective
 * where a uniform-square grid fit shrinks or over-extends.
 */
export function computeHomographyLeastSquares(
  src: Point[],
  dst: Point[],
): Matrix3x3 {
  if (src.length !== dst.length) {
    throw new Error("src and dst must have the same length");
  }
  if (src.length < 4) {
    throw new Error("Need at least 4 point pairs");
  }
  // Build Aᵀ A (8x8) and Aᵀ b (8) directly without materialising A —
  // each row of A appears twice and the accumulation is symmetric.
  const ATA: number[][] = Array.from({ length: 8 }, () =>
    new Array<number>(8).fill(0),
  );
  const ATb: number[] = new Array<number>(8).fill(0);
  for (let i = 0; i < src.length; i++) {
    const { x, y } = src[i];
    const { x: X, y: Y } = dst[i];
    const row1 = [x, y, 1, 0, 0, 0, -X * x, -X * y];
    const row2 = [0, 0, 0, x, y, 1, -Y * x, -Y * y];
    for (let j = 0; j < 8; j++) {
      ATb[j] += row1[j] * X + row2[j] * Y;
      for (let k = 0; k < 8; k++) {
        ATA[j][k] += row1[j] * row1[k] + row2[j] * row2[k];
      }
    }
  }
  const h = solveLinearSystem(ATA, ATb);
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ];
}

/** Apply a homography H to a 2D point. */
export function applyHomography(H: Matrix3x3, p: Point): Point {
  const w = H[2][0] * p.x + H[2][1] * p.y + H[2][2];
  return {
    x: (H[0][0] * p.x + H[0][1] * p.y + H[0][2]) / w,
    y: (H[1][0] * p.x + H[1][1] * p.y + H[1][2]) / w,
  };
}

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M: number[][] = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    let pivotMag = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      const mag = Math.abs(M[r][col]);
      if (mag > pivotMag) {
        pivot = r;
        pivotMag = mag;
      }
    }
    if (pivotMag < 1e-12) {
      throw new Error("Degenerate point configuration (collinear or duplicate corners)");
    }
    if (pivot !== col) [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}
