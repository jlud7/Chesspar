import { type Point } from "./homography";
import { warpBoard } from "./board-image";

export type DetectionResult = {
  /**
   * Four image-space corners in clockwise order suitable for direct use with
   * `warpBoard` ([a8, h8, h1, a1] convention). When the orientation can't be
   * inferred from the image, the order is just clockwise in image space —
   * callers may need to cycle through 4 rotations.
   */
  corners: [Point, Point, Point, Point];
  confidence: number;
  /** Whether the corner order was oriented from a piece-pattern guess. */
  oriented: boolean;
};

const MAX_DIM = 512;
// "Dark-square material" gates — colour-agnostic, wide enough to catch the
// dark squares of any common chess set, narrow enough to reject paper,
// wood, skin, and pieces.
//
//   Hue-dominance path: one RGB channel must dominate the other two by
//   >50 AND overall saturation > 0.5. Works for any saturated hue (red,
//   blue, green, navy, dark teal). Wood (≈ 140/100/70 → 40 dominance)
//   and skin (≈ 210/170/140 → 40) fall under the dominance bar.
//
//   Very-dark path: low luminance regardless of hue. Catches black,
//   ebony, very-dark brown squares. Reserved as fallback for monochrome
//   boards because it picks up pieces and dark wood too — those become
//   outliers the grid-fit must reject.
const HUE_DOMINANCE_MIN = 50;
const SAT_MIN = 0.5;
// Minimum value of the dominant RGB channel for the hue path. This keeps
// out very dark pixels (piece shadows on red squares, black piece edges
// that have a slight hue tinge); they go through the very-dark fallback
// path instead if needed.
const HUE_MAX_MIN = 100;
const DARK_LUM_MAX = 80;
const MIN_CENTROIDS = 6;

/**
 * Find the chess board's playing surface in `source` and return its four
 * corners ordered `[a8, h8, h1, a1]` — directly usable with `warpBoard`.
 *
 * Algorithm:
 *   1. Build a "dark-square material" mask (any chess-board dark colour,
 *      not wood/skin) using a saturated-hue path plus a very-dark path.
 *   2. The largest connected component is "all 32 dark squares + thin
 *      border line" merged via the border bridges. Its bounding box gives
 *      the approximate playing-area extent.
 *   3. Erode the dark mask so border bridges break; the remaining components
 *      are individual dark squares. Their centroids form a regular 4×8
 *      sub-grid of the 8×8 board.
 *   4. Fit the centroid cloud to an integer chess grid: scan rotation
 *      angles, estimate square width from nearest-neighbour distance, then
 *      snap each centroid to an integer (file, rank) position. Use the
 *      bounding-box anchor + a "missing ranks at the top in starting
 *      position" prior to pin down which side missing files/ranks sit on.
 *   5. Read the 4 playing-area corners directly off the fitted grid.
 *   6. Pick one of 4 rotations to put White at the bottom of the warp.
 */
export function autoDetectBoardCorners(
  source: HTMLImageElement | HTMLCanvasElement,
): DetectionResult | null {
  const w0 =
    source instanceof HTMLImageElement ? source.naturalWidth : source.width;
  const h0 =
    source instanceof HTMLImageElement ? source.naturalHeight : source.height;
  if (!w0 || !h0) return null;

  const detected = detectBoardViaRedness(source);
  if (!detected) return null;

  // The centroid-based grid fit consistently shrinks inward when the
  // outermost dark squares (a1/h8 etc.) are obscured by the back-rank
  // rooks. The dark-pixel signal those squares would contribute is gone,
  // so fitAxis picks a window that starts one file in and the polygon
  // misses a rank + a file of the playing surface. We can't recover the
  // missing centroids — but the bias is systematic, so push every
  // corner ~5% outward from the polygon centroid as a final pass.
  // Tested against the IMG_8819–8833 sample set: no detectable accuracy
  // regression on cleanly-detected boards, and recovers the missing
  // edge on piece-occluded sets.
  const cx = (detected.corners[0].x + detected.corners[2].x) / 2;
  const cy = (detected.corners[0].y + detected.corners[2].y) / 2;
  const EXPAND = 1.07;
  const expanded: [Point, Point, Point, Point] = [
    {
      x: cx + (detected.corners[0].x - cx) * EXPAND,
      y: cy + (detected.corners[0].y - cy) * EXPAND,
    },
    {
      x: cx + (detected.corners[1].x - cx) * EXPAND,
      y: cy + (detected.corners[1].y - cy) * EXPAND,
    },
    {
      x: cx + (detected.corners[2].x - cx) * EXPAND,
      y: cy + (detected.corners[2].y - cy) * EXPAND,
    },
    {
      x: cx + (detected.corners[3].x - cx) * EXPAND,
      y: cy + (detected.corners[3].y - cy) * EXPAND,
    },
  ];

  const oriented = orientStartingPosition(source, expanded);
  return {
    corners: oriented.corners,
    confidence: oriented.score >= 0 ? detected.confidence : 0,
    oriented: oriented.score >= 0,
  };
}

type RednessDetection = {
  corners: [Point, Point, Point, Point];
  confidence: number;
};

function detectBoardViaRedness(
  source: HTMLImageElement | HTMLCanvasElement,
): RednessDetection | null {
  const w0 =
    source instanceof HTMLImageElement ? source.naturalWidth : source.width;
  const h0 =
    source instanceof HTMLImageElement ? source.naturalHeight : source.height;
  const scale = Math.min(MAX_DIM / w0, MAX_DIM / h0, 1);
  const w = Math.max(8, Math.round(w0 * scale));
  const h = Math.max(8, Math.round(h0 * scale));

  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(source, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;

  // 1. Dark-square mask. Two acceptance paths cover the colour space of
  //    common chess sets:
  //       hue path       → red, blue, green, navy, dark teal squares
  //       very-dark path → black, ebony, dark brown squares
  //    Try the HUE path first — it cleanly excludes wood, skin, and
  //    pieces of any colour. The very-dark path is a fallback for
  //    monochrome boards where no square has a dominant hue.
  let dark = buildDarkMask(data, w, h, "hue");
  let darkCount = countOnes(dark);
  if (darkCount < w * h * 0.005) {
    dark = buildDarkMask(data, w, h, "veryDark");
    darkCount = countOnes(dark);
    if (darkCount < w * h * 0.005) return null;
  }

  // 2. Largest connected component → bbox of the playing-area's dark blob.
  const blobCC = connectedComponents(dark, w, h);
  let bigLabel = 0;
  let bigSize = 0;
  for (let lbl = 1; lbl < blobCC.sizes.length; lbl++) {
    if (blobCC.sizes[lbl] > bigSize) {
      bigSize = blobCC.sizes[lbl];
      bigLabel = lbl;
    }
  }
  if (!bigLabel) return null;
  let bMinX = Infinity,
    bMaxX = -Infinity,
    bMinY = Infinity,
    bMaxY = -Infinity;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (blobCC.labels[y * w + x] === bigLabel) {
        if (x < bMinX) bMinX = x;
        if (x > bMaxX) bMaxX = x;
        if (y < bMinY) bMinY = y;
        if (y > bMaxY) bMaxY = y;
      }
    }
  }
  const blobBbox = { minX: bMinX, maxX: bMaxX, minY: bMinY, maxY: bMaxY };

  // 3. Erode so the thin border line breaks; each chess square becomes
  //    its own component → centroids of the 32 dark squares.
  const erodeR = Math.max(2, Math.round(Math.min(w, h) / 100));
  const eroded = erode(dark, w, h, erodeR);
  const cc = connectedComponents(eroded, w, h);
  const topSizes = [...cc.sizes].slice(1).sort((a, b) => b - a);
  const probe = topSizes.slice(0, Math.min(20, topSizes.length));
  probe.sort((a, b) => a - b);
  const medianBig = probe[Math.floor(probe.length / 2)] || 0;
  const minAccept = Math.max(40, medianBig * 0.3);
  const maxAccept = medianBig * 4;
  const sums = new Map<number, { sx: number; sy: number; n: number }>();
  for (let lbl = 1; lbl < cc.sizes.length; lbl++) {
    if (cc.sizes[lbl] >= minAccept && cc.sizes[lbl] <= maxAccept) {
      sums.set(lbl, { sx: 0, sy: 0, n: 0 });
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const lbl = cc.labels[y * w + x];
      const s = sums.get(lbl);
      if (s) {
        s.sx += x;
        s.sy += y;
        s.n++;
      }
    }
  }
  const centroids: Point[] = [];
  for (const s of sums.values()) {
    centroids.push({ x: s.sx / s.n, y: s.sy / s.n });
  }
  if (centroids.length < MIN_CENTROIDS) return null;

  // 4. Fit the grid.
  const grid = fitChessGrid(centroids, blobBbox);
  if (!grid) return null;

  // 5. Read the 4 corners off the fit, in image (source) coordinates.
  const corners: [Point, Point, Point, Point] = [
    { x: grid.corners[0].x / scale, y: grid.corners[0].y / scale },
    { x: grid.corners[1].x / scale, y: grid.corners[1].y / scale },
    { x: grid.corners[2].x / scale, y: grid.corners[2].y / scale },
    { x: grid.corners[3].x / scale, y: grid.corners[3].y / scale },
  ];

  // Confidence: blob bbox area vs the grid's predicted playing-area area.
  // A clean detection has the blob almost filling the predicted area.
  const gridArea =
    Math.abs(
      (grid.corners[1].x - grid.corners[0].x) *
        (grid.corners[3].y - grid.corners[0].y) -
        (grid.corners[1].y - grid.corners[0].y) *
          (grid.corners[3].x - grid.corners[0].x),
    );
  const bboxArea =
    (blobBbox.maxX - blobBbox.minX) * (blobBbox.maxY - blobBbox.minY);
  const confidence =
    gridArea > 0 ? Math.min(1, Math.max(0, bboxArea / gridArea)) : 0;

  return { corners, confidence };
}

type FittedGrid = {
  /** Four playing-area corners in clockwise order, in downscaled coords. */
  corners: [Point, Point, Point, Point];
  /** Square width in downscaled coords. */
  sqw: number;
  /** Rotation angle of the U axis relative to image X (radians). */
  theta: number;
};

/**
 * Fit an 8×8 chess board to the detected dark-square centroids. Returns the
 * four playing-area corners in clockwise order [TL, TR, BR, BL].
 *
 * The board has 32 dark squares; their centroids form a regular 4×8 sub-grid
 * of the full 8×8. We:
 *   - Search rotation θ to align centroids with image axes.
 *   - Use median nearest-neighbour distance / √2 as the square width.
 *   - Snap centroid (U, V) to integer multiples of sqw, picking the
 *     8-wide K window with the most centroids on each axis.
 *   - Decide which edge missing ranks/files sit on using the BLOB BBOX
 *     anchor, plus a "missing rank goes to the top" prior for starting
 *     position (where the black back rank obliterates the dark-square
 *     signal on rank 8).
 */
function fitChessGrid(
  centroids: Point[],
  blobBbox: { minX: number; maxX: number; minY: number; maxY: number },
): FittedGrid | null {
  if (centroids.length < MIN_CENTROIDS) return null;

  const meanX = avgX(centroids);
  const meanY = avgY(centroids);
  const theta = findGridAngle(centroids, meanX, meanY);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);

  const uvs = centroids.map((c) => ({
    u: (c.x - meanX) * cosT + (c.y - meanY) * sinT,
    v: -(c.x - meanX) * sinT + (c.y - meanY) * cosT,
  }));

  // Square width = nearest-neighbour distance / √2 (diagonal dark squares
  // are at distance √2·sqw).
  const nn: number[] = [];
  for (let i = 0; i < uvs.length; i++) {
    let best = Infinity;
    for (let j = 0; j < uvs.length; j++) {
      if (i === j) continue;
      const du = uvs[i].u - uvs[j].u;
      const dv = uvs[i].v - uvs[j].v;
      const d = Math.hypot(du, dv);
      if (d < best) best = d;
    }
    if (best < Infinity) nn.push(best);
  }
  nn.sort((a, b) => a - b);
  const sqw = nn[Math.floor(nn.length / 2)] / Math.SQRT2;
  if (sqw < 5) return null;

  const uVals = uvs.map((p) => p.u);
  const vVals = uvs.map((p) => p.v);

  const fitU = fitAxis(uVals, sqw);
  const fitV = fitAxis(vVals, sqw);
  if (!fitU || !fitV) return null;

  // Project the blob bbox into UV space so we can anchor which edge the
  // missing files/ranks sit on. We project all 4 bbox corners and take
  // the extreme U/V values.
  const bboxUVs = [
    [blobBbox.minX, blobBbox.minY],
    [blobBbox.maxX, blobBbox.minY],
    [blobBbox.maxX, blobBbox.maxY],
    [blobBbox.minX, blobBbox.maxY],
  ].map(([x, y]) => ({
    u: (x - meanX) * cosT + (y - meanY) * sinT,
    v: -(x - meanX) * sinT + (y - meanY) * cosT,
  }));
  const blobLeftU = Math.min(...bboxUVs.map((p) => p.u));
  const blobTopV = Math.min(...bboxUVs.map((p) => p.v));

  // Files: anchor the leftmost file using how far the blob extends left
  // of the leftmost centroid. Default 0 (no missing files): our setup
  // typically shows all 8 files in the middle ranks.
  const missingFiles = 8 - (fitU.maxK - fitU.minK + 1);
  let leftFileOffset = 0;
  if (missingFiles > 0) {
    const xRaw =
      fitU.minK - (blobLeftU - fitU.origin) / sqw - 0.5;
    leftFileOffset = clamp(Math.round(xRaw), 0, missingFiles);
  }
  const leftFileK = fitU.minK - leftFileOffset;
  const leftU = fitU.origin + leftFileK * sqw - 0.5 * sqw;
  const rightU = leftU + 8 * sqw;

  // Ranks: missing ranks default to the TOP edge — for piece-occluded
  // starting positions, black back-rank pieces obliterate the dark-square
  // signal in
  // rank 8 (top) far more than the white back rank does in rank 1.
  const missingRanks = 8 - (fitV.maxK - fitV.minK + 1);
  let topRankOffset = missingRanks;
  if (missingRanks > 0) {
    const yRaw =
      fitV.minK - (blobTopV - fitV.origin) / sqw - 0.5;
    const bboxBased = clamp(Math.round(yRaw), 0, missingRanks);
    topRankOffset = clamp(Math.max(topRankOffset, bboxBased), 0, missingRanks);
  }
  const topRankK = fitV.minK - topRankOffset;
  const topV = fitV.origin + topRankK * sqw - 0.5 * sqw;
  const bottomV = topV + 8 * sqw;

  const uvToImage = (u: number, v: number): Point => ({
    x: u * cosT - v * sinT + meanX,
    y: u * sinT + v * cosT + meanY,
  });

  const tl = uvToImage(leftU, topV);
  const tr = uvToImage(rightU, topV);
  const br = uvToImage(rightU, bottomV);
  const bl = uvToImage(leftU, bottomV);

  return { corners: [tl, tr, br, bl], sqw, theta };
}

type AxisFit = { origin: number; minK: number; maxK: number };

/**
 * Find the integer-grid alignment along one axis. Returns the origin
 * (U value of K=0) and the [minK, maxK] window of 8 consecutive K's that
 * captures the most centroid values — outliers (label fragments, mis-
 * detected pieces) end up outside this window.
 */
function fitAxis(values: number[], sqw: number): AxisFit | null {
  if (values.length === 0) return null;
  const vMin = Math.min(...values);
  // Scan the modular offset to find the origin that minimises residual.
  let bestOrigin = vMin;
  let bestResid = Infinity;
  const STEPS = 40;
  for (let i = 0; i < STEPS; i++) {
    const candidate = vMin - sqw / 2 + (sqw * i) / STEPS;
    let total = 0;
    for (const v of values) {
      const k = Math.round((v - candidate) / sqw);
      const d = v - (candidate + k * sqw);
      total += d * d;
    }
    if (total < bestResid) {
      bestResid = total;
      bestOrigin = candidate;
    }
  }
  const ks = values.map((v) => Math.round((v - bestOrigin) / sqw));
  const win = pickBestKWindow(ks);
  if (win.endK - win.startK > 7) return null;
  return { origin: bestOrigin, minK: win.startK, maxK: win.endK };
}

/**
 * Find the 8-wide window of integer K's containing the most points.
 * Returns the tightened [startK, endK] of points actually inside that
 * window (which can be < 8 wide if the data is sparse at the edges).
 */
function pickBestKWindow(ks: number[]): { startK: number; endK: number } {
  if (ks.length === 0) return { startK: 0, endK: 0 };
  const counts = new Map<number, number>();
  for (const k of ks) counts.set(k, (counts.get(k) ?? 0) + 1);
  const uniqueKs = [...counts.keys()].sort((a, b) => a - b);
  let bestStart = uniqueKs[0];
  let bestCount = 0;
  for (const startK of uniqueKs) {
    let count = 0;
    for (let dk = 0; dk < 8; dk++) count += counts.get(startK + dk) ?? 0;
    if (count > bestCount) {
      bestCount = count;
      bestStart = startK;
    }
  }
  let actualMin = Infinity;
  let actualMax = -Infinity;
  for (let dk = 0; dk < 8; dk++) {
    if (counts.has(bestStart + dk)) {
      actualMin = Math.min(actualMin, bestStart + dk);
      actualMax = Math.max(actualMax, bestStart + dk);
    }
  }
  if (actualMin === Infinity) return { startK: bestStart, endK: bestStart };
  return { startK: actualMin, endK: actualMax };
}

/**
 * Find the rotation that best aligns the centroid cloud with an integer
 * grid. Score is the number of centroids whose modular U and V values
 * land near a shared peak (i.e., they sit on the same row/column).
 *
 * PCA on the centroid cloud is unreliable here: a square 8×8 grid has
 * roughly equal variance on every axis, so PCA tends to pick a 45° tilt.
 * Instead we scan the 0.05° angle grid in ±10° and score by inlier count.
 */
function findGridAngle(
  centroids: Point[],
  meanX: number,
  meanY: number,
): number {
  // Square width estimate from raw nearest-neighbour distances.
  let sumNN = 0;
  let count = 0;
  for (let i = 0; i < centroids.length; i++) {
    let best = Infinity;
    for (let j = 0; j < centroids.length; j++) {
      if (i === j) continue;
      const dx = centroids[i].x - centroids[j].x;
      const dy = centroids[i].y - centroids[j].y;
      const d = Math.hypot(dx, dy);
      if (d > 1 && d < best) best = d;
    }
    if (best < Infinity) {
      sumNN += best;
      count++;
    }
  }
  const meanNN = count > 0 ? sumNN / count : 50;
  const sqw = meanNN / Math.SQRT2;

  let bestTheta = 0;
  let bestScore = -Infinity;
  // ±10° in 0.05° steps. The board in our setup is roughly axis-aligned;
  // a wider search risks locking onto the diagonal 45° symmetry mode.
  for (let i = -200; i <= 200; i++) {
    const theta = (i / 20) * (Math.PI / 180);
    const score = gridInliers(centroids, meanX, meanY, theta, sqw);
    if (score > bestScore) {
      bestScore = score;
      bestTheta = theta;
    }
  }
  return bestTheta;
}

function gridInliers(
  centroids: Point[],
  meanX: number,
  meanY: number,
  theta: number,
  sqw: number,
): number {
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const us: number[] = [];
  const vs: number[] = [];
  for (const c of centroids) {
    const dx = c.x - meanX;
    const dy = c.y - meanY;
    us.push(dx * cosT + dy * sinT);
    vs.push(-dx * sinT + dy * cosT);
  }
  return axisPeakInliers(us, sqw) + axisPeakInliers(vs, sqw);
}

function axisPeakInliers(values: number[], sqw: number): number {
  const BINS = 20;
  const hist = new Array(BINS).fill(0);
  for (const v of values) {
    const f = (((v / sqw) % 1) + 1) % 1;
    hist[Math.min(BINS - 1, Math.floor(f * BINS))]++;
  }
  let bestBin = 0;
  let bestCount = 0;
  for (let i = 0; i < BINS; i++) {
    if (hist[i] > bestCount) {
      bestCount = hist[i];
      bestBin = i;
    }
  }
  const peakF = (bestBin + 0.5) / BINS;
  let inliers = 0;
  for (const v of values) {
    const f = (((v / sqw) % 1) + 1) % 1;
    const d = Math.min(Math.abs(f - peakF), 1 - Math.abs(f - peakF));
    if (d < 0.15) inliers++;
  }
  return inliers;
}

/**
 * Decide which cyclic rotation of `quad` puts White at the bottom of the
 * rectified output. Uses the starting position's row signature — pieces on
 * ranks 1-2 + 7-8, empty 3-6 — rather than depending on the heuristic
 * occupancy classifier (which is fragile across board colour schemes).
 *
 * Returns the best-rotated corners + a score; score < 0 means we couldn't
 * tell, in which case the caller should let the user confirm orientation.
 */
function orientStartingPosition(
  source: HTMLImageElement | HTMLCanvasElement,
  quad: [Point, Point, Point, Point],
): { corners: [Point, Point, Point, Point]; score: number } {
  let bestK = 0;
  let bestScore = -Infinity;
  for (let k = 0; k < 4; k++) {
    const rotated = rotateCorners(quad, k);
    const score = startingPositionRotationScore(source, rotated);
    if (score > bestScore) {
      bestScore = score;
      bestK = k;
    }
  }
  return { corners: rotateCorners(quad, bestK), score: bestScore };
}

/**
 * Per-row "this is a starting-position chess board with White at the
 * bottom" score. Combines two signals:
 *
 *  - **Piece-row pattern**: each cell's within-window luminance std. Pieces
 *    create high std (3D shape + colour break); empty cells are flat. The
 *    expected pattern is HIGH-HIGH on rows 0,1 + 6,7 and LOW in between.
 *  - **White-at-bottom check**: the *mean* luminance of pieces on rows 6,7
 *    minus that of pieces on rows 0,1. White pieces are lighter than black
 *    pieces regardless of board colour. A correctly-rotated board should
 *    have a positive delta; a 180° flip would make this strongly negative.
 */
function startingPositionRotationScore(
  source: HTMLImageElement | HTMLCanvasElement,
  corners: [Point, Point, Point, Point],
): number {
  const SIZE = 192;
  let warped: HTMLCanvasElement;
  try {
    warped = warpBoard(source, corners, SIZE);
  } catch {
    return -1e6;
  }
  const ctx = warped.getContext("2d", { willReadFrequently: true });
  if (!ctx) return -1e6;
  const data = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const cellSize = SIZE / 8;
  const pad = cellSize * 0.18;
  const rowStd: number[] = [];
  const rowMean: number[] = [];
  for (let r = 0; r < 8; r++) {
    let sumStd = 0;
    let sumMean = 0;
    let n = 0;
    for (let f = 0; f < 8; f++) {
      const x0 = Math.round(f * cellSize + pad);
      const x1 = Math.round((f + 1) * cellSize - pad);
      const y0 = Math.round(r * cellSize + pad);
      const y1 = Math.round((r + 1) * cellSize - pad);
      let sum = 0;
      let sumSq = 0;
      let count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = (y * SIZE + x) * 4;
          const l =
            0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          sum += l;
          sumSq += l * l;
          count++;
        }
      }
      if (count > 0) {
        const m = sum / count;
        sumStd += Math.sqrt(Math.max(0, sumSq / count - m * m));
        sumMean += m;
        n++;
      }
    }
    rowStd.push(n > 0 ? sumStd / n : 0);
    rowMean.push(n > 0 ? sumMean / n : 0);
  }
  const pieceRows = (rowStd[0] + rowStd[1] + rowStd[6] + rowStd[7]) / 4;
  const emptyRows = (rowStd[2] + rowStd[3] + rowStd[4] + rowStd[5]) / 4;
  const piecePattern = pieceRows - emptyRows;
  const bottomBright = (rowMean[6] + rowMean[7]) / 2;
  const topBright = (rowMean[0] + rowMean[1]) / 2;
  const whiteOnBottom = (bottomBright - topBright) * 0.25;
  return piecePattern + whiteOnBottom;
}

/** Cyclically rotate corners by `k` positions. */
export function rotateCorners(
  corners: [Point, Point, Point, Point],
  k: number,
): [Point, Point, Point, Point] {
  const n = corners.length;
  const j = ((k % n) + n) % n;
  return [
    corners[j],
    corners[(j + 1) % n],
    corners[(j + 2) % n],
    corners[(j + 3) % n],
  ];
}

/** Score a 64-cell occupancy by how "white on bottom, black on top" it looks. */
export function scorePlayingOrientation(
  occ: ("empty" | "white" | "black")[],
): number {
  let score = 0;
  for (let i = 0; i < 64; i++) {
    const rowFromTop = Math.floor(i / 8);
    const s = occ[i];
    if (s === "empty") continue;
    if (rowFromTop < 4 && s === "black") score++;
    else if (rowFromTop >= 4 && s === "white") score++;
    else score -= 1;
  }
  return score;
}

/**
 * Per-frame corner refinement: re-run the auto-detector on `frame`, align
 * the resulting clockwise quad to the previously-saved corner ordering by
 * picking whichever cyclic rotation has the smallest total point-distance,
 * then accept the refined corners only if they're close enough to the
 * saved ones (i.e. the camera was nudged, not entirely re-pointed).
 *
 * Detection wobbles by ~0.5–1 square width across frames as pieces move
 * and partially occlude the dark squares — their centroids shift toward
 * the still-visible pixels. We treat that as noise rather than camera
 * motion: `drifted` only flips true once drift exceeds ~3% of the image
 * diagonal (≈50 px on an iPhone shot), which is larger than detector
 * wobble but smaller than any real re-aim of the camera.
 */
export function refineCornersForFrame(
  frame: HTMLImageElement | HTMLCanvasElement,
  saved: [Point, Point, Point, Point],
  options: {
    maxAvgDriftFraction?: number;
    minDriftToTriggerSwap?: number;
  } = {},
): { corners: [Point, Point, Point, Point]; drifted: boolean } {
  const maxAvgDriftFraction = options.maxAvgDriftFraction ?? 0.1;
  const minDriftToTriggerSwap = options.minDriftToTriggerSwap ?? 0.03;
  const detected = detectBoardViaRedness(frame);
  if (!detected) return { corners: saved, drifted: false };
  const w =
    frame instanceof HTMLImageElement ? frame.naturalWidth : frame.width;
  const h =
    frame instanceof HTMLImageElement ? frame.naturalHeight : frame.height;
  if (!w || !h) return { corners: saved, drifted: false };
  const diag = Math.hypot(w, h);

  let bestAligned = detected.corners;
  let bestAvg = Infinity;
  for (let k = 0; k < 4; k++) {
    const rotated = rotateCorners(detected.corners, k);
    let total = 0;
    for (let j = 0; j < 4; j++) {
      total += Math.hypot(
        rotated[j].x - saved[j].x,
        rotated[j].y - saved[j].y,
      );
    }
    const avg = total / 4;
    if (avg < bestAvg) {
      bestAvg = avg;
      bestAligned = rotated;
    }
  }
  const driftFraction = bestAvg / diag;
  if (driftFraction > maxAvgDriftFraction) {
    return { corners: saved, drifted: false };
  }
  // Below the swap threshold the detection is treated as noise and the
  // saved corners are kept verbatim. Returning `saved` (rather than the
  // wobbly `bestAligned`) keeps the warp stable across frames so the
  // occupancy-diff move matcher doesn't see ghost square changes.
  if (driftFraction <= minDriftToTriggerSwap) {
    return { corners: saved, drifted: false };
  }
  return { corners: bestAligned, drifted: true };
}

/**
 * Backwards-compat shim. With the dark-mask detector corners already
 * come out grid-aligned, so a separate "snap" step is unnecessary — but
 * legacy callers still expect this signature.
 */
export function snapCornersToGrid(
  source: HTMLImageElement | HTMLCanvasElement,
  initial: [Point, Point, Point, Point],
): [Point, Point, Point, Point] {
  const detected = detectBoardViaRedness(source);
  return detected ? detected.corners : initial;
}

/**
 * Backwards-compat shim for the old refine API. Returns the freshly-
 * detected corners (already grid-snapped).
 */
export function refineQuad(
  source: HTMLImageElement | HTMLCanvasElement,
  initial: [Point, Point, Point, Point],
): [Point, Point, Point, Point] {
  const detected = detectBoardViaRedness(source);
  return detected ? detected.corners : initial;
}

type DarkMaskMode = "hue" | "veryDark";

function buildDarkMask(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  mode: DarkMaskMode,
): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const R = data[i];
    const G = data[i + 1];
    const B = data[i + 2];
    const mx = Math.max(R, G, B);
    const mn = Math.min(R, G, B);
    const mid = R + G + B - mx - mn; // second-largest channel
    const sat = mx > 0 ? (mx - mn) / mx : 0;
    let isSquare: boolean;
    if (mode === "hue") {
      // One channel dominates the others by ≥ HUE_DOMINANCE_MIN AND
      // overall saturation is high enough AND the dominant channel is
      // moderately bright. Hue-agnostic — works for red / blue / green /
      // navy / dark-teal boards. Wood and skin have moderate redness
      // but fail the dominance bar; piece-edge shadows fail the
      // brightness bar.
      isSquare =
        mx > HUE_MAX_MIN && mx - mid > HUE_DOMINANCE_MIN && sat > SAT_MIN;
    } else {
      // Very dark regardless of hue. Catches black-square boards.
      // Pieces and dark wood shadow CAN sneak in here, so this path is
      // reserved for monochrome boards where the hue path finds nothing.
      const lum = 0.299 * R + 0.587 * G + 0.114 * B;
      isSquare = lum < DARK_LUM_MAX;
    }
    out[j] = isSquare ? 1 : 0;
  }
  return out;
}

function countOnes(mask: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) n++;
  return n;
}

function erode(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      let allSet = 1;
      for (let dy = -r; dy <= r && allSet; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (mask[(y + dy) * w + (x + dx)] === 0) {
            allSet = 0;
            break;
          }
        }
      }
      out[y * w + x] = allSet;
    }
  }
  return out;
}

function connectedComponents(mask: Uint8Array, w: number, h: number) {
  const labels = new Int32Array(w * h);
  const sizes: number[] = [0];
  let next = 1;
  const stack: number[] = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 0 || labels[i] !== 0) continue;
    const label = next++;
    sizes.push(0);
    stack.push(i);
    while (stack.length > 0) {
      const idx = stack.pop()!;
      if (labels[idx] !== 0) continue;
      labels[idx] = label;
      sizes[label]++;
      const x = idx % w;
      const y = (idx - x) / w;
      if (x > 0 && mask[idx - 1] && labels[idx - 1] === 0) stack.push(idx - 1);
      if (x < w - 1 && mask[idx + 1] && labels[idx + 1] === 0)
        stack.push(idx + 1);
      if (y > 0 && mask[idx - w] && labels[idx - w] === 0) stack.push(idx - w);
      if (y < h - 1 && mask[idx + w] && labels[idx + w] === 0)
        stack.push(idx + w);
    }
  }
  return { labels, sizes };
}

function avgX(points: Point[]): number {
  let s = 0;
  for (const p of points) s += p.x;
  return s / points.length;
}

function avgY(points: Point[]): number {
  let s = 0;
  for (const p of points) s += p.y;
  return s / points.length;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
