import {
  applyHomography,
  computeHomographyLeastSquares,
  type Matrix3x3,
  type Point,
} from "./homography.ts";
import { extractSquareCrops, warpBoard } from "./board-image.ts";
import { classifyBoard } from "./occupancy.ts";

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
  const oriented = orientStartingPosition(source, detected.corners);
  return {
    corners: oriented.corners,
    confidence: oriented.score >= 0 ? detected.confidence : 0,
    oriented: oriented.score >= 0,
  };
}

export function autoDetectBoardCornersLegacy(
  source: HTMLImageElement | HTMLCanvasElement,
): DetectionResult | null {
  const w0 =
    source instanceof HTMLImageElement ? source.naturalWidth : source.width;
  const h0 =
    source instanceof HTMLImageElement ? source.naturalHeight : source.height;
  if (!w0 || !h0) return null;

  const detected = detectBoardViaRedness(source);
  if (!detected) return null;

  const expanded = expandQuad(detected.corners, 1.07);
  const refined = refineCornersByOuterRows(source, expanded);
  const oriented = orientStartingPosition(source, refined);
  return {
    corners: oriented.corners,
    confidence: oriented.score >= 0 ? detected.confidence : 0,
    oriented: oriented.score >= 0,
  };
}

/**
 * Async fetcher that returns the chessboard's bbox in original-image
 * pixel coords, or null if no bbox can be obtained. Injected by callers
 * so the detector stays test-friendly and doesn't import the worker
 * client directly.
 */
export type ChessboardBboxFetcher = (
  image: HTMLCanvasElement,
) => Promise<{ x1: number; y1: number; x2: number; y2: number } | null>;

/**
 * Detect playing-surface corners using Florence-2 as a calibration aid.
 *
 * Steps:
 *   1. Ask Florence-2 (via the supplied fetcher) for a bbox of the
 *      chessboard in the original image.
 *   2. Crop the image to that bbox onto an off-screen canvas.
 *   3. Run the existing `autoDetectBoardCorners` inside the clean crop —
 *      table, lamp, hands, and background are now outside the frame.
 *   4. Translate the corners back to original-image pixel coords.
 *
 * Always falls back to the no-Florence detector if any step fails — so
 * this strictly dominates the original entry point. The bbox itself is
 * used only as a localiser; the precise 4-corner geometry still comes
 * from the existing CV pipeline (now running on a much easier input).
 */
export async function detectBoardCornersViaFlorence(
  source: HTMLImageElement | HTMLCanvasElement,
  fetchBbox: ChessboardBboxFetcher,
): Promise<DetectionResult | null> {
  const fullCanvas = sourceToCanvas(source);
  if (!fullCanvas) return autoDetectBoardCorners(source);

  let bbox: { x1: number; y1: number; x2: number; y2: number } | null = null;
  try {
    bbox = await fetchBbox(fullCanvas);
  } catch {
    bbox = null;
  }
  if (!bbox) return autoDetectBoardCorners(source);

  // Clamp + pad slightly so the crop never excludes the playing surface
  // when Florence's bbox lands a pixel or two inside it.
  const pad = Math.max(4, Math.round(fullCanvas.width * 0.005));
  const cx1 = Math.max(0, Math.floor(bbox.x1 - pad));
  const cy1 = Math.max(0, Math.floor(bbox.y1 - pad));
  const cx2 = Math.min(fullCanvas.width, Math.ceil(bbox.x2 + pad));
  const cy2 = Math.min(fullCanvas.height, Math.ceil(bbox.y2 + pad));
  const cw = cx2 - cx1;
  const ch = cy2 - cy1;
  if (cw < 50 || ch < 50) return autoDetectBoardCorners(source);

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = cw;
  cropCanvas.height = ch;
  const cropCtx = cropCanvas.getContext("2d");
  if (!cropCtx) return autoDetectBoardCorners(source);
  cropCtx.drawImage(fullCanvas, cx1, cy1, cw, ch, 0, 0, cw, ch);

  const inner = autoDetectBoardCorners(cropCanvas);
  if (!inner) return autoDetectBoardCorners(source);

  return {
    corners: [
      { x: inner.corners[0].x + cx1, y: inner.corners[0].y + cy1 },
      { x: inner.corners[1].x + cx1, y: inner.corners[1].y + cy1 },
      { x: inner.corners[2].x + cx1, y: inner.corners[2].y + cy1 },
      { x: inner.corners[3].x + cx1, y: inner.corners[3].y + cy1 },
    ],
    confidence: inner.confidence,
    oriented: inner.oriented,
  };
}

function sourceToCanvas(
  source: HTMLImageElement | HTMLCanvasElement,
): HTMLCanvasElement | null {
  if (source instanceof HTMLCanvasElement) return source;
  const w = source.naturalWidth;
  const h = source.naturalHeight;
  if (!w || !h) return null;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(source, 0, 0);
  return c;
}

function expandQuad(
  corners: [Point, Point, Point, Point],
  scale: number,
): [Point, Point, Point, Point] {
  const cx = (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4;
  const cy = (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4;
  return [
    {
      x: cx + (corners[0].x - cx) * scale,
      y: cy + (corners[0].y - cy) * scale,
    },
    {
      x: cx + (corners[1].x - cx) * scale,
      y: cy + (corners[1].y - cy) * scale,
    },
    {
      x: cx + (corners[2].x - cx) * scale,
      y: cy + (corners[2].y - cy) * scale,
    },
    {
      x: cx + (corners[3].x - cx) * scale,
      y: cy + (corners[3].y - cy) * scale,
    },
  ];
}

/**
 * If the rectified board produced by `corners` has an outer rank/file
 * with almost no pieces but the next-in rank/file has many, the polygon
 * very likely missed that outer rank/file (a common failure when corner
 * pieces obscure the dark squares the centroid grid-fit relies on).
 * Expand that specific edge by ~0.8 cell widths and try again.
 * Iterates up to twice to handle a "missing rank 8 + file a" combo.
 *
 * Independent of starting position: works whenever the player is set up
 * with back-rank pieces. For verification we don't actually need it to
 * match the canonical layout — just to keep the playing area inside the
 * polygon.
 */
function refineCornersByOuterRows(
  source: HTMLImageElement | HTMLCanvasElement,
  initial: [Point, Point, Point, Point],
): [Point, Point, Point, Point] {
  let corners = initial;
  for (let iter = 0; iter < 2; iter++) {
    let warped: HTMLCanvasElement;
    try {
      warped = warpBoard(source, corners, 256);
    } catch {
      return corners;
    }
    const crops = extractSquareCrops(warped);
    if (crops.length !== 64) return corners;
    const occ = classifyBoard(crops).map((c) => c.state);

    const piecesIn = (start: number, stride: number, count: number): number => {
      let n = 0;
      for (let k = 0; k < count; k++) {
        if (occ[start + k * stride] !== "empty") n++;
      }
      return n;
    };

    // Top row = cells 0..7, next = cells 8..15
    const topOuter = piecesIn(0, 1, 8);
    const topInner = piecesIn(8, 1, 8);
    // Bottom row = cells 56..63, next = cells 48..55
    const bottomOuter = piecesIn(56, 1, 8);
    const bottomInner = piecesIn(48, 1, 8);
    // Left column = cells 0,8,16,...,56; next = cells 1,9,17,...,57
    const leftOuter = piecesIn(0, 8, 8);
    const leftInner = piecesIn(1, 8, 8);
    // Right column = cells 7,15,...,63; next = cells 6,14,...,62
    const rightOuter = piecesIn(7, 8, 8);
    const rightInner = piecesIn(6, 8, 8);

    // "missing" if outer < 2 pieces AND next row in has ≥ 4 — signals a
    // packed back rank just inside the polygon edge with empty cells at
    // the edge itself.
    const needTop = topOuter < 2 && topInner >= 4;
    const needBottom = bottomOuter < 2 && bottomInner >= 4;
    const needLeft = leftOuter < 2 && leftInner >= 4;
    const needRight = rightOuter < 2 && rightInner >= 4;

    if (!needTop && !needBottom && !needLeft && !needRight) return corners;

    // Polygon centroid for outward-direction reference.
    const polyCx = (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4;
    const polyCy = (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4;
    const cellSize =
      (Math.hypot(corners[1].x - corners[0].x, corners[1].y - corners[0].y) +
        Math.hypot(corners[2].x - corners[3].x, corners[2].y - corners[3].y)) /
      2 /
      8;
    const SHIFT = cellSize * 0.85;

    const outwardShift = (p1: Point, p2: Point): { dx: number; dy: number } => {
      // Outward normal to the edge from p1 to p2.
      const ex = p2.x - p1.x;
      const ey = p2.y - p1.y;
      const len = Math.hypot(ex, ey) || 1;
      const nx = -ey / len;
      const ny = ex / len;
      const midX = (p1.x + p2.x) / 2;
      const midY = (p1.y + p2.y) / 2;
      const dot = nx * (midX - polyCx) + ny * (midY - polyCy);
      const sign = dot >= 0 ? 1 : -1;
      return { dx: nx * sign * SHIFT, dy: ny * sign * SHIFT };
    };

    // Corners convention: [TL, TR, BR, BL] (a8, h8, h1, a1).
    let [tl, tr, br, bl]: [Point, Point, Point, Point] = [
      { ...corners[0] },
      { ...corners[1] },
      { ...corners[2] },
      { ...corners[3] },
    ];

    if (needTop) {
      const s = outwardShift(tl, tr);
      tl = { x: tl.x + s.dx, y: tl.y + s.dy };
      tr = { x: tr.x + s.dx, y: tr.y + s.dy };
    }
    if (needBottom) {
      const s = outwardShift(br, bl);
      br = { x: br.x + s.dx, y: br.y + s.dy };
      bl = { x: bl.x + s.dx, y: bl.y + s.dy };
    }
    if (needLeft) {
      const s = outwardShift(bl, tl);
      bl = { x: bl.x + s.dx, y: bl.y + s.dy };
      tl = { x: tl.x + s.dx, y: tl.y + s.dy };
    }
    if (needRight) {
      const s = outwardShift(tr, br);
      tr = { x: tr.x + s.dx, y: tr.y + s.dy };
      br = { x: br.x + s.dx, y: br.y + s.dy };
    }
    corners = [tl, tr, br, bl];
  }
  return corners;
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

  // 4. Use the grid fit ONLY to recover the board's tilt angle. The
  //    centroid-based fit reliably picks `theta` (the integer-grid
  //    alignment) even when the outermost dark squares are partially
  //    obscured by back-rank pieces — there are still 24+ unobscured
  //    interior dark squares whose centroids are clean.
  const grid = fitChessGrid(centroids, blobBbox);
  if (!grid) return null;

  // 5. Read the 4 corners off the fit, in image (source) coordinates.
  //    Earlier attempts to expand the polygon via dark-mask extents
  //    helped on the bundled photos (where back-rank pieces occlude
  //    corner squares uniformly) but introduced a worse failure on
  //    live phone shots: perspective distortion means the near-edge
  //    cells are physically larger in image space than the far-edge
  //    cells, so any "uniform 1.2-cell expansion" downward extends
  //    the polygon past the playing surface and into the table /
  //    chess clock / lap area. Until we solve the perspective issue
  //    properly (homography-fit grid or Hough-line edge detection)
  //    we stick with the raw centroid fit, which at worst clips a
  //    sliver of an occluded back rank rather than blowing out.
  const corners: [Point, Point, Point, Point] = [
    { x: grid.corners[0].x / scale, y: grid.corners[0].y / scale },
    { x: grid.corners[1].x / scale, y: grid.corners[1].y / scale },
    { x: grid.corners[2].x / scale, y: grid.corners[2].y / scale },
    { x: grid.corners[3].x / scale, y: grid.corners[3].y / scale },
  ];

  // Confidence = how much of the dark-blob bbox is inside the predicted
  // grid area. A clean detection has the blob almost filling the area.
  const gridArea = Math.abs(
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

/**
 * Rotated bounding rectangle of the UNION of multiple labelled
 * components, oriented by `theta`. Walks every pixel whose label is
 * in `acceptedLabels`, projects into the (u, v) coordinate system
 * rotated by theta, finds the four extreme corner points, expands
 * outward by `expandPx` to undo erosion shrink, and maps back to
 * image space.
 *
 * Works for both boards with dark border bridges (one giant component,
 * `acceptedLabels` has one entry) and boards with cream gaps between
 * squares (32 separate components — each square accepted by size
 * filter). The extreme points across the union ARE the playing-
 * surface corners as long as the outermost dark squares (a1/h1/a8/h8
 * plus rank-1/8 and file-a/h edge cells) contribute at least a few
 * pixels each.
 */
/**
 * Take the OUTER extent of `quadA` and `quadB` in the (u, v) frame
 * aligned with the detected grid orientation, capped so that any
 * outward expansion of `quadA` is no more than `maxOutward` pixels per
 * edge. Lets the extent quad rescue an occluded back rank from the
 * grid quad while preventing it from running away to a chess clock
 * or table edge that happens to pass the dark-mask filter.
 *
 * Both inputs are clockwise [TL, TR, BR, BL] (in the orientation the
 * grid fit declared — caller may rotate later via
 * `orientStartingPosition`).
 */
function clampedOuter(
  quadA: [Point, Point, Point, Point],
  quadB: [Point, Point, Point, Point],
  theta: number,
  maxOutward: number,
): [Point, Point, Point, Point] {
  const cx = (quadA[0].x + quadA[1].x + quadA[2].x + quadA[3].x) / 4;
  const cy = (quadA[0].y + quadA[1].y + quadA[2].y + quadA[3].y) / 4;
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const project = (p: Point): { u: number; v: number } => {
    const dx = p.x - cx;
    const dy = p.y - cy;
    return { u: dx * cosT + dy * sinT, v: -dx * sinT + dy * cosT };
  };
  const uvA = quadA.map(project);
  const uvB = quadB.map(project);
  const aMinU = Math.min(...uvA.map((p) => p.u));
  const aMaxU = Math.max(...uvA.map((p) => p.u));
  const aMinV = Math.min(...uvA.map((p) => p.v));
  const aMaxV = Math.max(...uvA.map((p) => p.v));
  const bMinU = Math.min(...uvB.map((p) => p.u));
  const bMaxU = Math.max(...uvB.map((p) => p.u));
  const bMinV = Math.min(...uvB.map((p) => p.v));
  const bMaxV = Math.max(...uvB.map((p) => p.v));
  // Outward = away from quadA. Clamp each side's adjustment.
  const minU = Math.max(bMinU, aMinU - maxOutward);
  const maxU = Math.min(bMaxU, aMaxU + maxOutward);
  const minV = Math.max(bMinV, aMinV - maxOutward);
  const maxV = Math.min(bMaxV, aMaxV + maxOutward);
  // But never SMALLER than A on any side.
  const finalMinU = Math.min(minU, aMinU);
  const finalMaxU = Math.max(maxU, aMaxU);
  const finalMinV = Math.min(minV, aMinV);
  const finalMaxV = Math.max(maxV, aMaxV);
  const toImage = (u: number, v: number): Point => ({
    x: u * cosT - v * sinT + cx,
    y: u * sinT + v * cosT + cy,
  });
  return [
    toImage(finalMinU, finalMinV),
    toImage(finalMaxU, finalMinV),
    toImage(finalMaxU, finalMaxV),
    toImage(finalMinU, finalMaxV),
  ];
}

function orientedExtentsForLabels(
  labels: Int32Array,
  acceptedLabels: Set<number>,
  w: number,
  h: number,
  theta: number,
  expandPx: number,
): [Point, Point, Point, Point] | null {
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (let i = 0; i < w * h; i++) {
    if (acceptedLabels.has(labels[i])) {
      const x = i % w;
      const y = (i - x) / w;
      sx += x;
      sy += y;
      n++;
    }
  }
  if (n === 0) return null;
  const cx = sx / n;
  const cy = sy / n;
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < w * h; i++) {
    if (!acceptedLabels.has(labels[i])) continue;
    const x = i % w;
    const y = (i - x) / w;
    const dx = x - cx;
    const dy = y - cy;
    const u = dx * cosT + dy * sinT;
    const v = -dx * sinT + dy * cosT;
    if (u < minU) minU = u;
    if (u > maxU) maxU = u;
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
  // Undo the inward shrink caused by erosion before component labelling.
  minU -= expandPx;
  maxU += expandPx;
  minV -= expandPx;
  maxV += expandPx;
  const toImage = (u: number, v: number): Point => ({
    x: u * cosT - v * sinT + cx,
    y: u * sinT + v * cosT + cy,
  });
  return [
    toImage(minU, minV), // TL (u-min, v-min)
    toImage(maxU, minV), // TR (u-max, v-min)
    toImage(maxU, maxV), // BR
    toImage(minU, maxV), // BL
  ];
}

/**
 * Per-edge gradient snap. For each of the polygon's four edges, walk
 * perpendicular to the edge in a ±W band looking for the strongest
 * luminance gradient transition (the red-square ↔ cream-border edge
 * is the strongest line in the image at the playing-surface boundary).
 * Move that edge to wherever the gradient peaks.
 *
 * The dark-blob rotated bbox is correct within ~1-2 pixels in most
 * cases, but a half-cell occluded by a rook can leave the blob extent
 * just inside the actual border. The gradient snap reliably closes
 * that gap because the red-cream transition is the highest-contrast
 * straight line in the corner neighbourhood, regardless of whether a
 * piece sits on the corner square.
 */
function snapQuadToEdges(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  quad: [Point, Point, Point, Point],
): [Point, Point, Point, Point] {
  // Build a luminance buffer for fast sampling.
  const lum = new Float32Array(w * h);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
    lum[j] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
  }
  const sample = (x: number, y: number): number => {
    const xi = Math.max(0, Math.min(w - 1, Math.round(x)));
    const yi = Math.max(0, Math.min(h - 1, Math.round(y)));
    return lum[yi * w + xi];
  };
  // Find the strongest gradient transition along the line from p1 to
  // p2, scanning ±band perpendicular. Returns the offset (in pixels)
  // from the original line that maximises mean |gradient|.
  const findBestOffset = (
    p1: Point,
    p2: Point,
    outwardSign: number,
    band: number,
    samples: number,
  ): number => {
    const ex = p2.x - p1.x;
    const ey = p2.y - p1.y;
    const len = Math.hypot(ex, ey) || 1;
    // Perpendicular outward direction.
    const nx = (-ey / len) * outwardSign;
    const ny = (ex / len) * outwardSign;
    let bestOffset = 0;
    let bestScore = -Infinity;
    for (let off = -band; off <= band; off += 1) {
      let total = 0;
      let count = 0;
      for (let s = 0; s < samples; s++) {
        const t = (s + 0.5) / samples;
        const baseX = p1.x + t * ex + off * nx;
        const baseY = p1.y + t * ey + off * ny;
        // Sample 3 px inward and 3 px outward; the difference is the
        // gradient across the edge.
        const inX = baseX - 3 * nx;
        const inY = baseY - 3 * ny;
        const outX = baseX + 3 * nx;
        const outY = baseY + 3 * ny;
        total += Math.abs(sample(inX, inY) - sample(outX, outY));
        count++;
      }
      const score = count > 0 ? total / count : 0;
      if (score > bestScore) {
        bestScore = score;
        bestOffset = off;
      }
    }
    return bestOffset;
  };
  // For each of the 4 edges, find the best offset and shift the two
  // endpoints along the outward normal.
  const shiftEdge = (
    p1: Point,
    p2: Point,
    offset: number,
    outwardSign: number,
  ): [Point, Point] => {
    const ex = p2.x - p1.x;
    const ey = p2.y - p1.y;
    const len = Math.hypot(ex, ey) || 1;
    const nx = (-ey / len) * outwardSign;
    const ny = (ex / len) * outwardSign;
    return [
      { x: p1.x + offset * nx, y: p1.y + offset * ny },
      { x: p2.x + offset * nx, y: p2.y + offset * ny },
    ];
  };
  // Compute polygon centroid to determine "outward" direction for each edge.
  const polyCx = (quad[0].x + quad[1].x + quad[2].x + quad[3].x) / 4;
  const polyCy = (quad[0].y + quad[1].y + quad[2].y + quad[3].y) / 4;
  const outwardSign = (p1: Point, p2: Point): number => {
    const ex = p2.x - p1.x;
    const ey = p2.y - p1.y;
    const len = Math.hypot(ex, ey) || 1;
    const nx = -ey / len;
    const ny = ex / len;
    const midX = (p1.x + p2.x) / 2;
    const midY = (p1.y + p2.y) / 2;
    const dot = nx * (midX - polyCx) + ny * (midY - polyCy);
    return dot >= 0 ? 1 : -1;
  };
  // Search band: ±6% of the polygon's mean side length.
  const meanSide =
    (Math.hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y) +
      Math.hypot(quad[2].x - quad[1].x, quad[2].y - quad[1].y) +
      Math.hypot(quad[3].x - quad[2].x, quad[3].y - quad[2].y) +
      Math.hypot(quad[0].x - quad[3].x, quad[0].y - quad[3].y)) /
    4;
  const band = Math.max(4, Math.round(meanSide * 0.06));
  const samples = 32;
  let [tl, tr, br, bl] = quad;
  // Edge a8→h8 (TL→TR), outward = "up" relative to polygon centroid.
  {
    const sign = outwardSign(tl, tr);
    const off = findBestOffset(tl, tr, sign, band, samples);
    [tl, tr] = shiftEdge(tl, tr, off, sign);
  }
  // Edge h8→h1 (TR→BR), outward = "right".
  {
    const sign = outwardSign(tr, br);
    const off = findBestOffset(tr, br, sign, band, samples);
    [tr, br] = shiftEdge(tr, br, off, sign);
  }
  // Edge h1→a1 (BR→BL), outward = "down".
  {
    const sign = outwardSign(br, bl);
    const off = findBestOffset(br, bl, sign, band, samples);
    [br, bl] = shiftEdge(br, bl, off, sign);
  }
  // Edge a1→a8 (BL→TL), outward = "left".
  {
    const sign = outwardSign(bl, tl);
    const off = findBestOffset(bl, tl, sign, band, samples);
    [bl, tl] = shiftEdge(bl, tl, off, sign);
  }
  return [tl, tr, br, bl];
}

type FittedGrid = {
  /** Four playing-area corners in clockwise order, in downscaled coords. */
  corners: [Point, Point, Point, Point];
  /** Square width in downscaled coords. */
  sqw: number;
  /** Rotation angle of the U axis relative to image X (radians). */
  theta: number;
  /** 0..8 — how many files of the playing surface had detected centroids. */
  missingFiles: number;
  /** 0..8 — how many ranks of the playing surface had detected centroids. */
  missingRanks: number;
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

  // --- Perspective-aware corner fit -----------------------------------
  // Replace the linear "corner = origin + 8 * sqw" formula with a proper
  // homography fit from cell-space → image space. Each detected dark
  // square contributes one correspondence: its (file+0.5, rank+0.5)
  // chess-cell-center coords ↦ its (image x, y) centroid. With ~28-32
  // points on the bundled photos and ≥15 even on heavily-occluded live
  // shots, the least-squares fit nails the perspective transform and
  // its application to (0,0), (8,0), (8,8), (0,8) gives the playing-
  // surface corners directly — no uniform-sqw assumption, no need to
  // pick which centroid is at the corner.
  //
  // Falls back to the linear formula on degenerate input (too few
  // points, collinear, or solver failure).
  const cellSrc: Point[] = [];
  const imgDst: Point[] = [];
  for (let i = 0; i < centroids.length; i++) {
    const u = uvs[i].u;
    const v = uvs[i].v;
    const fileK = Math.round((u - fitU.origin) / sqw);
    const rankK = Math.round((v - fitV.origin) / sqw);
    if (fileK < fitU.minK || fileK > fitU.maxK) continue;
    if (rankK < fitV.minK || rankK > fitV.maxK) continue;
    const file = fileK - leftFileK; // 0..7 in playing-surface space
    const rank = rankK - topRankK;
    if (file < 0 || file > 7) continue;
    if (rank < 0 || rank > 7) continue;
    // Reject ALL centroids that don't sit on a dark square — the chess
    // colouring is alternating, dark cells are (file+rank) even when
    // a1 is dark. We treat the leftFileK/topRankK basis as having a1
    // at (file=0, rank=7); per dark-square check that's (file+rank) odd.
    // (We don't strictly need to filter — wrong-parity centroids are
    // noise — but filtering tightens the fit.)
    if (((file + rank) & 1) !== 1) continue;
    cellSrc.push({ x: file + 0.5, y: rank + 0.5 });
    imgDst.push({ x: centroids[i].x, y: centroids[i].y });
  }

  const tryHomographyCorners = (): [Point, Point, Point, Point] | null => {
    if (cellSrc.length < 8) return null;
    let H: Matrix3x3;
    try {
      H = computeHomographyLeastSquares(cellSrc, imgDst);
    } catch {
      return null;
    }
    const corners: [Point, Point, Point, Point] = [
      applyHomography(H, { x: 0, y: 0 }),
      applyHomography(H, { x: 8, y: 0 }),
      applyHomography(H, { x: 8, y: 8 }),
      applyHomography(H, { x: 0, y: 8 }),
    ];
    // Sanity check — every corner must be a finite point. Solver can
    // produce NaN/Infinity on near-singular inputs.
    for (const p of corners) {
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return null;
    }
    // Sanity check — fitted polygon should be roughly the same size as
    // the centroid spread. If it's hugely larger (>10× the centroid
    // bbox area) the fit went rogue and we fall back.
    const polyArea = Math.abs(
      (corners[1].x - corners[0].x) * (corners[3].y - corners[0].y) -
        (corners[1].y - corners[0].y) * (corners[3].x - corners[0].x),
    );
    const cMinX = Math.min(...centroids.map((c) => c.x));
    const cMaxX = Math.max(...centroids.map((c) => c.x));
    const cMinY = Math.min(...centroids.map((c) => c.y));
    const cMaxY = Math.max(...centroids.map((c) => c.y));
    const centroidArea = (cMaxX - cMinX) * (cMaxY - cMinY);
    if (centroidArea > 0 && polyArea > centroidArea * 6) return null;
    return corners;
  };

  const hCorners = tryHomographyCorners();
  if (hCorners) {
    return {
      corners: hCorners,
      sqw,
      theta,
      missingFiles,
      missingRanks,
    };
  }

  // --- Fallback: linear corners (legacy path) -------------------------
  const tl = uvToImage(leftU, topV);
  const tr = uvToImage(rightU, topV);
  const br = uvToImage(rightU, bottomV);
  const bl = uvToImage(leftU, bottomV);

  return {
    corners: [tl, tr, br, bl],
    sqw,
    theta,
    missingFiles,
    missingRanks,
  };
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
): {
  corners: [Point, Point, Point, Point];
  drifted: boolean;
  detected: boolean;
  driftFraction?: number;
} {
  const maxAvgDriftFraction = options.maxAvgDriftFraction ?? 0.1;
  const minDriftToTriggerSwap = options.minDriftToTriggerSwap ?? 0.03;
  const detected = detectBoardViaRedness(frame);
  if (!detected) return { corners: saved, drifted: false, detected: false };
  const w =
    frame instanceof HTMLImageElement ? frame.naturalWidth : frame.width;
  const h =
    frame instanceof HTMLImageElement ? frame.naturalHeight : frame.height;
  if (!w || !h) return { corners: saved, drifted: false, detected: false };
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
    return { corners: saved, drifted: false, detected: true, driftFraction };
  }
  // Below the swap threshold the detection is treated as noise and the
  // saved corners are kept verbatim. Returning `saved` (rather than the
  // wobbly `bestAligned`) keeps the warp stable across frames so the
  // occupancy-diff move matcher doesn't see ghost square changes.
  if (driftFraction <= minDriftToTriggerSwap) {
    return { corners: saved, drifted: false, detected: true, driftFraction };
  }
  return { corners: bestAligned, drifted: true, detected: true, driftFraction };
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
