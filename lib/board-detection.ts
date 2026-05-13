import { applyHomography, computeHomography, type Point } from "./homography";
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

type Mask = Uint8Array;

const MAX_DIM = 320;
const VARIANCE_HALF_WINDOW = 2; // 5x5 window
const MIN_COMPONENT_FRACTION = 0.035;
const VARIANCE_PERCENTILE = 0.65;
const MIN_VARIANCE_ABSOLUTE = 60;
const GRADIENT_PERCENTILE = 0.6;
const MIN_GRADIENT_ABSOLUTE = 12;

/**
 * Find the chess board's playing surface in `source` and return its four
 * corners ordered `[a8, h8, h1, a1]` — directly usable with `warpBoard`.
 *
 * Pipeline:
 *   1. Coarse blob detection (local variance ∧ bidirectional gradient).
 *      Returns a clockwise quad enclosing the board area.
 *   2. Edge inset/outset refinement — each of the 4 sides can slide
 *      parallel to itself to snap onto the actual 8×8 boundary, peeling off
 *      label borders and the protruding tops of 3D pieces.
 *   3. Per-corner local search — tiny 2D wiggle on each corner to absorb
 *      perspective distortion that uniform edge slides can't handle.
 *   4. Orientation pick — try all 4 cyclic rotations of the quad and pick
 *      whichever rectified output most looks like a normal chess starting
 *      position (piece-occupied ranks 1-2 + 7-8, empty 3-6). Uses cell-level
 *      variance, NOT the heuristic classifier — that classifier needs the
 *      orientation to already be correct to work, so we can't depend on it.
 */
export function autoDetectBoardCorners(
  source: HTMLImageElement | HTMLCanvasElement,
): DetectionResult | null {
  const w0 =
    source instanceof HTMLImageElement ? source.naturalWidth : source.width;
  const h0 =
    source instanceof HTMLImageElement ? source.naturalHeight : source.height;
  if (!w0 || !h0) return null;

  const coarse = coarseBoardQuad(source);
  if (!coarse) return null;

  const refinedQuad = refineQuad(source, coarse.ordered);
  const oriented = orientStartingPosition(source, refinedQuad);
  return {
    corners: oriented.corners,
    confidence: oriented.score >= 0 ? Math.min(1, coarse.coverage) : 0,
    oriented: oriented.score >= 0,
  };
}

/** Coarse stage: variance/gradient masks → blob → convex hull → quad. */
function coarseBoardQuad(
  source: HTMLImageElement | HTMLCanvasElement,
): { ordered: [Point, Point, Point, Point]; coverage: number } | null {
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
  const img = ctx.getImageData(0, 0, w, h);

  const lum = new Float32Array(w * h);
  for (let i = 0, j = 0; i < img.data.length; i += 4, j++) {
    lum[j] =
      0.299 * img.data[i] +
      0.587 * img.data[i + 1] +
      0.114 * img.data[i + 2];
  }

  const dilateRadius = Math.max(1, Math.floor(Math.min(w, h) / 80));

  const variance = computeLocalVariance(lum, w, h, VARIANCE_HALF_WINDOW);
  const varThreshold = pickPercentileThreshold(
    variance,
    VARIANCE_PERCENTILE,
    MIN_VARIANCE_ABSOLUTE,
  );
  const varMask = thresholdMask(variance, varThreshold);

  const checker = computeCheckerSignal(lum, w, h);
  const chkThreshold = pickPercentileThreshold(
    checker,
    GRADIENT_PERCENTILE,
    MIN_GRADIENT_ABSOLUTE,
  );
  const chkMask = thresholdMask(checker, chkThreshold);

  const combined = andMask(varMask, chkMask);

  const candidates: Uint8Array[] = [combined, chkMask, varMask];
  let chosen: { ordered: Point[]; bestSize: number; quadArea: number } | null =
    null;
  for (const candidate of candidates) {
    const dilated = dilate(candidate, w, h, dilateRadius);
    const { labels, sizes } = connectedComponents(dilated, w, h);
    let bestLabel = 0;
    let bestSize = 0;
    for (let lbl = 1; lbl < sizes.length; lbl++) {
      if (sizes[lbl] > bestSize) {
        bestSize = sizes[lbl];
        bestLabel = lbl;
      }
    }
    if (!bestLabel || bestSize < w * h * MIN_COMPONENT_FRACTION) continue;
    const pts: Point[] = [];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (labels[y * w + x] === bestLabel) pts.push({ x, y });
      }
    }
    const hull = convexHull(pts);
    if (hull.length < 4) continue;
    const quad = simplifyToQuadrilateral(hull);
    if (quad.length !== 4) continue;
    const quadArea = polygonArea(quad);
    if (quadArea <= 0) continue;
    const ordered = orderClockwise(quad);
    chosen = { ordered, bestSize, quadArea };
    break;
  }

  if (!chosen) return null;
  const coverage = Math.min(1, chosen.bestSize / chosen.quadArea);
  const ordered = chosen.ordered.map((p) => ({
    x: p.x / scale,
    y: p.y / scale,
  })) as [Point, Point, Point, Point];
  return { ordered, coverage };
}

/**
 * Refine a coarse clockwise quad onto the actual 8×8 playing surface.
 *
 * Two phases:
 *  - Edge slides: each of the 4 edges shifts parallel to itself. We
 *    parameterize as a fraction of the polygon's average dimension; the
 *    search range is wide enough to cut off a typical label border AND to
 *    push out if the coarse quad undershot.
 *  - Corner wiggles: small per-corner search to mop up perspective effects
 *    the parallel slides can't handle.
 *
 * The score function evaluates whichever candidate quad best produces an
 * 8×8 chess grid in the rectified output (parity-consistent square means,
 * tight within-parity uniformity, high parity separation).
 */
export function refineQuad(
  source: HTMLImageElement | HTMLCanvasElement,
  initial: [Point, Point, Point, Point],
): [Point, Point, Point, Point] {
  // Cache: extract the source luminance once at a working resolution
  // (~512px on longest side) and score every candidate against that buffer.
  // Avoids the >100ms-per-call cost of getImageData on a full 4032x3024 iPhone
  // photo, which is what made the earlier per-warp scorer unusably slow.
  const cache = buildScoringCache(source);

  // Phase 1: edge slides. Coarse-to-fine — first a wide sweep at 5% steps,
  // then a refining sweep at 1.5% steps around the winner.
  let current = initial;
  current = edgeSlideSweep(cache, current, [-0.06, -0.025, 0, 0.025, 0.06]);
  current = edgeSlideSweep(
    cache,
    current,
    [-0.025, -0.0125, 0, 0.0125, 0.025],
  );

  // Phase 2: per-corner wiggle. Two passes through all 4 corners, 3×3 search
  // per corner per pass. Catches perspective effects the parallel slides
  // can't (e.g., the camera tilted more on one side than the other).
  const cornerWiggleRange = [-0.022, 0, 0.022];
  const cornerFineRange = [-0.011, 0, 0.011];
  for (const range of [cornerWiggleRange, cornerFineRange]) {
    for (let cornerIdx = 0; cornerIdx < 4; cornerIdx++) {
      let bestForCorner = current;
      let bestForCornerScore = scoreFromCache(cache, current);
      for (const dx of range) {
        for (const dy of range) {
          if (dx === 0 && dy === 0) continue;
          const candidate = wiggleCorner(current, cornerIdx, dx, dy);
          const score = scoreFromCache(cache, candidate);
          if (score > bestForCornerScore) {
            bestForCornerScore = score;
            bestForCorner = candidate;
          }
        }
      }
      current = bestForCorner;
    }
  }

  return current;
}

function edgeSlideSweep(
  cache: ScoringCache,
  base: [Point, Point, Point, Point],
  range: number[],
): [Point, Point, Point, Point] {
  let best = base;
  let bestScore = scoreFromCache(cache, base);
  for (const top of range) {
    for (const right of range) {
      for (const bottom of range) {
        for (const left of range) {
          if (top === 0 && right === 0 && bottom === 0 && left === 0) continue;
          const candidate = slideEdges(base, { top, right, bottom, left });
          if (!candidate) continue;
          const score = scoreFromCache(cache, candidate);
          if (score > bestScore) {
            bestScore = score;
            best = candidate;
          }
        }
      }
    }
  }
  return best;
}

type ScoringCache = {
  /** Source luminance, row-major, 0..255. */
  lum: Float32Array;
  width: number;
  height: number;
  /**
   * Scale factor from original image coords to cached buffer coords —
   * multiply caller-space corner coords by this before sampling.
   */
  scale: number;
};

const SCORING_CACHE_MAX = 512;

function buildScoringCache(
  source: HTMLImageElement | HTMLCanvasElement,
): ScoringCache {
  const w0 =
    source instanceof HTMLImageElement ? source.naturalWidth : source.width;
  const h0 =
    source instanceof HTMLImageElement ? source.naturalHeight : source.height;
  const scale = Math.min(SCORING_CACHE_MAX / w0, SCORING_CACHE_MAX / h0, 1);
  const w = Math.max(8, Math.round(w0 * scale));
  const h = Math.max(8, Math.round(h0 * scale));
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("buildScoringCache: failed to get 2D context");
  ctx.drawImage(source, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const lum = new Float32Array(w * h);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    lum[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return { lum, width: w, height: h, scale };
}

/**
 * Fast scoring that bypasses warpBoard entirely. Maps each cell's inner
 * window from "rectified 8×8 coordinates" back to source coordinates via
 * the homography, then nearest-neighbour samples from the cached luminance
 * buffer. ~16 samples per cell × 64 cells = ~1024 source reads per
 * evaluation — much faster than warping a full image and re-reading it.
 */
function scoreFromCache(
  cache: ScoringCache,
  corners: [Point, Point, Point, Point],
): number {
  const scaled = corners.map((p) => ({
    x: p.x * cache.scale,
    y: p.y * cache.scale,
  })) as [Point, Point, Point, Point];
  const H = computeHomography(
    [
      { x: 0, y: 0 },
      { x: 8, y: 0 },
      { x: 8, y: 8 },
      { x: 0, y: 8 },
    ],
    scaled,
  );
  const lightMeans: number[] = [];
  const darkMeans: number[] = [];
  const SAMPLES = 5;
  const innerPad = 0.18;
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      let sum = 0;
      let n = 0;
      for (let sy = 0; sy < SAMPLES; sy++) {
        for (let sx = 0; sx < SAMPLES; sx++) {
          const cellX =
            f + innerPad + ((1 - 2 * innerPad) * (sx + 0.5)) / SAMPLES;
          const cellY =
            r + innerPad + ((1 - 2 * innerPad) * (sy + 0.5)) / SAMPLES;
          const src = applyHomography(H, { x: cellX, y: cellY });
          const ix = Math.round(src.x);
          const iy = Math.round(src.y);
          if (
            ix < 0 ||
            iy < 0 ||
            ix >= cache.width ||
            iy >= cache.height
          ) {
            continue;
          }
          sum += cache.lum[iy * cache.width + ix];
          n++;
        }
      }
      const mean = n > 0 ? sum / n : 0;
      if ((r + f) % 2 === 0) lightMeans.push(mean);
      else darkMeans.push(mean);
    }
  }
  if (lightMeans.length === 0 || darkMeans.length === 0) return -1e6;
  const lightAvg = avg(lightMeans);
  const darkAvg = avg(darkMeans);
  const lightSd = stdev(lightMeans, lightAvg);
  const darkSd = stdev(darkMeans, darkAvg);
  const separation = Math.abs(lightAvg - darkAvg);
  return separation - 1.2 * (lightSd + darkSd);
}

/**
 * Slide each of the four edges of a clockwise quad inward (negative) or
 * outward (positive) by a fraction of the polygon's average side length.
 * Edges are identified by the corner they emanate from in the clockwise
 * walk: top edge runs corner 0→1, right 1→2, bottom 2→3, left 3→0.
 */
function slideEdges(
  quad: [Point, Point, Point, Point],
  amounts: { top: number; right: number; bottom: number; left: number },
): [Point, Point, Point, Point] | null {
  const cx = (quad[0].x + quad[1].x + quad[2].x + quad[3].x) / 4;
  const cy = (quad[0].y + quad[1].y + quad[2].y + quad[3].y) / 4;
  const sideLens = [0, 1, 2, 3].map((i) => {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    return Math.hypot(b.x - a.x, b.y - a.y);
  });
  const avgSide = (sideLens[0] + sideLens[1] + sideLens[2] + sideLens[3]) / 4;
  // Compute inward unit normals for each edge (toward centroid).
  function inwardNormal(i: number): { x: number; y: number } {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    const ex = b.x - a.x;
    const ey = b.y - a.y;
    const len = Math.hypot(ex, ey) || 1;
    // Two candidate normals; pick whichever points toward the centroid.
    const n1x = -ey / len;
    const n1y = ex / len;
    const midX = (a.x + b.x) / 2;
    const midY = (a.y + b.y) / 2;
    const sign = (cx - midX) * n1x + (cy - midY) * n1y > 0 ? 1 : -1;
    return { x: n1x * sign, y: n1y * sign };
  }
  const normals = [
    inwardNormal(0),
    inwardNormal(1),
    inwardNormal(2),
    inwardNormal(3),
  ];
  const slides = [amounts.top, amounts.right, amounts.bottom, amounts.left];
  // Shift each edge along its inward normal. A given corner is the
  // intersection of two consecutive shifted edges; we move it by the sum
  // of both shifts projected onto the corner's directions.
  function shiftedEdgeLine(i: number) {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    const n = normals[i];
    const offset = slides[i] * avgSide;
    return {
      p: { x: a.x + n.x * offset, y: a.y + n.y * offset },
      q: { x: b.x + n.x * offset, y: b.y + n.y * offset },
    };
  }
  const lines = [0, 1, 2, 3].map(shiftedEdgeLine);
  const out: [Point, Point, Point, Point] = [
    quad[0],
    quad[1],
    quad[2],
    quad[3],
  ];
  for (let i = 0; i < 4; i++) {
    // Corner i = intersection of edge (i-1) and edge i (mod 4).
    const a = lines[(i + 3) % 4];
    const b = lines[i];
    const pt = lineIntersection(a.p, a.q, b.p, b.q);
    if (!pt) return null;
    out[i] = pt;
  }
  // Sanity check: polygon must remain convex and non-degenerate.
  if (polygonArea(out) <= 0) return null;
  return out;
}

function wiggleCorner(
  quad: [Point, Point, Point, Point],
  idx: number,
  dxFrac: number,
  dyFrac: number,
): [Point, Point, Point, Point] {
  const sideLens = [0, 1, 2, 3].map((i) => {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    return Math.hypot(b.x - a.x, b.y - a.y);
  });
  const avgSide = (sideLens[0] + sideLens[1] + sideLens[2] + sideLens[3]) / 4;
  const dx = dxFrac * avgSide;
  const dy = dyFrac * avgSide;
  const out: [Point, Point, Point, Point] = [
    quad[0],
    quad[1],
    quad[2],
    quad[3],
  ];
  out[idx] = { x: quad[idx].x + dx, y: quad[idx].y + dy };
  return out;
}

function lineIntersection(
  p1: Point,
  p2: Point,
  p3: Point,
  p4: Point,
): Point | null {
  const x1 = p1.x,
    y1 = p1.y,
    x2 = p2.x,
    y2 = p2.y,
    x3 = p3.x,
    y3 = p3.y,
    x4 = p4.x,
    y4 = p4.y;
  const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(den) < 1e-9) return null;
  const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
  return { x: x1 + t * (x2 - x1), y: y1 + t * (y2 - y1) };
}


function avg(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function stdev(xs: number[], mean: number): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) {
    const d = x - mean;
    s += d * d;
  }
  return Math.sqrt(s / xs.length);
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
 *
 * The two signals together pin both "is this a chess starting position?"
 * and "which rotation puts White at the bottom?"
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
          const l = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
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
  // Piece-row pattern (HIGH std on outer rows, LOW std in middle).
  const pieceRows = (rowStd[0] + rowStd[1] + rowStd[6] + rowStd[7]) / 4;
  const emptyRows = (rowStd[2] + rowStd[3] + rowStd[4] + rowStd[5]) / 4;
  const piecePattern = pieceRows - emptyRows;
  // White-at-bottom signal: bottom two rows brighter than top two rows.
  // Scale to be in the same ballpark as piecePattern.
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
 */
export function refineCornersForFrame(
  frame: HTMLImageElement | HTMLCanvasElement,
  saved: [Point, Point, Point, Point],
  options: { maxAvgDriftFraction?: number } = {},
): { corners: [Point, Point, Point, Point]; drifted: boolean } {
  const maxAvgDriftFraction = options.maxAvgDriftFraction ?? 0.1;
  const coarse = coarseBoardQuad(frame);
  if (!coarse) return { corners: saved, drifted: false };
  const w =
    frame instanceof HTMLImageElement ? frame.naturalWidth : frame.width;
  const h =
    frame instanceof HTMLImageElement ? frame.naturalHeight : frame.height;
  if (!w || !h) return { corners: saved, drifted: false };
  const diag = Math.hypot(w, h);

  let bestAligned = coarse.ordered;
  let bestAvg = Infinity;
  for (let k = 0; k < 4; k++) {
    const rotated = rotateCorners(coarse.ordered, k);
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
  // Snap the saved corners with a fresh refinement against this frame.
  const refined = refineQuad(frame, bestAligned);
  return { corners: refined, drifted: driftFraction > 0.005 };
}

/**
 * Backwards-compat shim. Old API name kept so calls keep working; the
 * new `refineQuad` runs both edge slides and corner wiggles.
 */
export function snapCornersToGrid(
  source: HTMLImageElement | HTMLCanvasElement,
  initial: [Point, Point, Point, Point],
): [Point, Point, Point, Point] {
  return refineQuad(source, initial);
}

function andMask(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] && b[i] ? 1 : 0;
  return out;
}

function computeCheckerSignal(
  lum: Float32Array,
  w: number,
  h: number,
): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = Math.abs(lum[i + 1] - lum[i - 1]);
      const gy = Math.abs(lum[i + w] - lum[i - w]);
      out[i] = gx < gy ? gx : gy;
    }
  }
  return out;
}

function computeLocalVariance(
  lum: Float32Array,
  w: number,
  h: number,
  half: number,
): Float32Array {
  const out = new Float32Array(w * h);
  const winArea = (half * 2 + 1) ** 2;
  for (let y = half; y < h - half; y++) {
    for (let x = half; x < w - half; x++) {
      let sum = 0;
      let sumSq = 0;
      for (let dy = -half; dy <= half; dy++) {
        const row = (y + dy) * w + x;
        for (let dx = -half; dx <= half; dx++) {
          const v = lum[row + dx];
          sum += v;
          sumSq += v * v;
        }
      }
      const mean = sum / winArea;
      const vr = sumSq / winArea - mean * mean;
      out[y * w + x] = vr > 0 ? vr : 0;
    }
  }
  return out;
}

function pickPercentileThreshold(
  values: Float32Array,
  percentile: number,
  floor: number,
): number {
  const arr = Array.from(values);
  arr.sort((a, b) => a - b);
  const p = arr[Math.floor(arr.length * percentile)];
  return Math.max(p, floor);
}

function thresholdMask(variance: Float32Array, threshold: number): Mask {
  const out = new Uint8Array(variance.length);
  for (let i = 0; i < variance.length; i++) {
    if (variance[i] > threshold) out[i] = 1;
  }
  return out;
}

function dilate(mask: Mask, w: number, h: number, r: number): Mask {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x] === 0) continue;
      const y0 = Math.max(0, y - r);
      const y1 = Math.min(h - 1, y + r);
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      for (let yy = y0; yy <= y1; yy++) {
        const row = yy * w;
        for (let xx = x0; xx <= x1; xx++) out[row + xx] = 1;
      }
    }
  }
  return out;
}

function connectedComponents(mask: Mask, w: number, h: number) {
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

function convexHull(points: Point[]): Point[] {
  if (points.length < 3) return [...points];
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const lower: Point[] = [];
  for (const p of sorted) {
    while (
      lower.length >= 2 &&
      cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0
    ) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper: Point[] = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (
      upper.length >= 2 &&
      cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0
    ) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function cross(o: Point, a: Point, b: Point): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function simplifyToQuadrilateral(hull: Point[]): Point[] {
  if (hull.length <= 4) return [...hull];
  const pts = [...hull];
  while (pts.length > 4) {
    let bestIdx = 0;
    let bestLoss = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const prev = pts[(i - 1 + pts.length) % pts.length];
      const next = pts[(i + 1) % pts.length];
      const tri = Math.abs(
        prev.x * (pts[i].y - next.y) +
          pts[i].x * (next.y - prev.y) +
          next.x * (pts[i].y - prev.y),
      );
      if (tri < bestLoss) {
        bestLoss = tri;
        bestIdx = i;
      }
    }
    pts.splice(bestIdx, 1);
  }
  return pts;
}

function polygonArea(poly: Point[]): number {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i];
    const q = poly[(i + 1) % poly.length];
    a += p.x * q.y - q.x * p.y;
  }
  return a / 2;
}

function orderClockwise(quad: Point[]): [Point, Point, Point, Point] {
  if (quad.length !== 4) throw new Error("orderClockwise expects 4 points");
  const cx = quad.reduce((s, p) => s + p.x, 0) / 4;
  const cy = quad.reduce((s, p) => s + p.y, 0) / 4;
  const withAngle = quad.map((p) => ({
    p,
    angle: Math.atan2(p.y - cy, p.x - cx),
  }));
  withAngle.sort((a, b) => a.angle - b.angle);
  return [withAngle[0].p, withAngle[1].p, withAngle[2].p, withAngle[3].p];
}
