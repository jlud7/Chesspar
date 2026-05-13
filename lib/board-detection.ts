import type { Point } from "./homography";

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
 * Locate the chessboard quadrilateral in `source` using purely image stats.
 *
 * The board has alternating squares that produce a dense field of high local
 * luminance variance. Backgrounds (table, carpet, clock) generally don't.
 * We threshold the variance map, find the largest connected component, take
 * its convex hull, and simplify to a quadrilateral by iteratively removing
 * the vertex that contributes the least area.
 *
 * The 4 corners are returned in clockwise order. Their *chess-relative*
 * identity (which is a8 vs h8 vs h1 vs a1) is decided later by trying all
 * 4 cyclic assignments and picking whichever produces the most
 * starting-position-like classifier output.
 */
export function autoDetectBoardCorners(
  source: HTMLImageElement | HTMLCanvasElement,
): DetectionResult | null {
  const w0 =
    source instanceof HTMLImageElement ? source.naturalWidth : source.width;
  const h0 =
    source instanceof HTMLImageElement ? source.naturalHeight : source.height;
  if (!w0 || !h0) return null;

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

  // Strategy A: local-variance mask. Fast and works on most photos.
  const variance = computeLocalVariance(lum, w, h, VARIANCE_HALF_WINDOW);
  const varThreshold = pickPercentileThreshold(
    variance,
    VARIANCE_PERCENTILE,
    MIN_VARIANCE_ABSOLUTE,
  );
  const varMask = thresholdMask(variance, varThreshold);

  // Strategy B: checkerboard-specific mask — min(|gx|, |gy|) is high only
  // where there are edges in *both* directions, which is unique to grid
  // patterns. Filters out wood-grain and carpet that have edges in just
  // one direction.
  const checker = computeCheckerSignal(lum, w, h);
  const chkThreshold = pickPercentileThreshold(
    checker,
    GRADIENT_PERCENTILE,
    MIN_GRADIENT_ABSOLUTE,
  );
  const chkMask = thresholdMask(checker, chkThreshold);

  // AND the two masks — accept only pixels that are both texturally rough
  // AND have bidirectional gradient. Falls back to either alone if that
  // produces too small a region.
  const combined = andMask(varMask, chkMask);

  const candidates: Uint8Array[] = [combined, chkMask, varMask];

  let chosen: { ordered: Point[]; bestSize: number; quadArea: number } | null = null;
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

  const cornersOrig: [Point, Point, Point, Point] = chosen.ordered.map((p) => ({
    x: p.x / scale,
    y: p.y / scale,
  })) as [Point, Point, Point, Point];

  return {
    corners: cornersOrig,
    confidence: coverage,
    oriented: false,
  };
}

function andMask(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] && b[i] ? 1 : 0;
  return out;
}

/**
 * Per-pixel "is this part of a checkerboard?" signal: min(|gx|, |gy|).
 * High only when edges exist in both directions; near zero on smooth
 * surfaces, wood grain (edges in one direction), or out-of-focus regions.
 */
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

/** Score a 64-cell occupancy by how "white on bottom, black on top" it looks. */
export function scorePlayingOrientation(occ: ("empty" | "white" | "black")[]): number {
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

/**
 * Per-frame corner refinement: re-run the auto-detector on `frame`, align
 * the resulting clockwise quad to the previously-saved corner ordering by
 * picking whichever cyclic rotation has the smallest total point-distance,
 * then accept the refined corners only if they're close enough to the
 * saved ones (i.e. the camera was nudged, not entirely re-pointed).
 *
 * Returns the saved corners unchanged when auto-detect fails, when the
 * detected quad doesn't match closely, or when there's no reasonable
 * alignment. This makes the refinement strictly non-destructive — at
 * worst we keep what we had.
 */
export function refineCornersForFrame(
  frame: HTMLImageElement | HTMLCanvasElement,
  saved: [Point, Point, Point, Point],
  options: { maxAvgDriftFraction?: number } = {},
): { corners: [Point, Point, Point, Point]; drifted: boolean } {
  const maxAvgDriftFraction = options.maxAvgDriftFraction ?? 0.1;
  const detection = autoDetectBoardCorners(frame);
  if (!detection) return { corners: saved, drifted: false };
  const w =
    frame instanceof HTMLImageElement ? frame.naturalWidth : frame.width;
  const h =
    frame instanceof HTMLImageElement ? frame.naturalHeight : frame.height;
  if (!w || !h) return { corners: saved, drifted: false };
  const diag = Math.hypot(w, h);

  let bestAligned: [Point, Point, Point, Point] = detection.corners;
  let bestAvg = Infinity;
  for (let k = 0; k < 4; k++) {
    const rotated = rotateCorners(detection.corners, k);
    let total = 0;
    for (let j = 0; j < 4; j++) {
      total += Math.hypot(rotated[j].x - saved[j].x, rotated[j].y - saved[j].y);
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
  return { corners: bestAligned, drifted: driftFraction > 0.005 };
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
  return Math.abs(a) / 2;
}

function orderClockwise(quad: Point[]): [Point, Point, Point, Point] {
  if (quad.length !== 4) throw new Error("orderClockwise expects 4 points");
  const cx = quad.reduce((s, p) => s + p.x, 0) / 4;
  const cy = quad.reduce((s, p) => s + p.y, 0) / 4;
  const withAngle = quad.map((p) => ({
    p,
    angle: Math.atan2(p.y - cy, p.x - cx),
  }));
  // Image-space angles: -π (west) … π (east). Clockwise from top-left means
  // starting at the smallest angle (most negative-Y, leftward) and going CW.
  withAngle.sort((a, b) => a.angle - b.angle);
  return [
    withAngle[0].p,
    withAngle[1].p,
    withAngle[2].p,
    withAngle[3].p,
  ];
}
