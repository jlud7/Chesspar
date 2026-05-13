export type Occupancy = "empty" | "white" | "black";

export type ClassifyResult = {
  state: Occupancy;
  confidence: number;
  /** Internal stats useful for debugging the threshold heuristics. */
  stats: {
    mean: number;
    median: number;
    std: number;
    range: number;
  };
};

/**
 * Heuristic per-square classifier.
 *
 * The classifier looks at the inner region of a rectified square crop and
 * decides whether it's empty or holds a white/black piece. Pieces add
 * texture (high local std + wider luminance range) and shift the median
 * luminance toward black or white. Empty squares stay flat, regardless of
 * whether they're a light or dark board square.
 *
 * Thresholds are tuned for typical phone photos of plastic chess sets on
 * mid-saturation boards. Calibration against the user's specific set will
 * improve accuracy; see `calibrateBaseline` (not yet implemented) for the
 * future path.
 */
const INNER_PAD = 0.15;
const EMPTY_STD = 18;
const EMPTY_RANGE = 60;
const BLACK_MAX_LUMA = 95;
const WHITE_MIN_LUMA = 145;

export function classifyCrop(crop: HTMLCanvasElement): ClassifyResult {
  const w = crop.width;
  const h = crop.height;
  const ctx = crop.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("classifyCrop: failed to get 2D context");
  const x0 = Math.floor(w * INNER_PAD);
  const x1 = Math.max(x0 + 1, Math.ceil(w * (1 - INNER_PAD)));
  const y0 = Math.floor(h * INNER_PAD);
  const y1 = Math.max(y0 + 1, Math.ceil(h * (1 - INNER_PAD)));
  const iw = x1 - x0;
  const ih = y1 - y0;
  const data = ctx.getImageData(x0, y0, iw, ih).data;
  const n = iw * ih;

  const lums = new Float32Array(n);
  let sum = 0;
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const l = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    lums[j] = l;
    sum += l;
  }
  const mean = sum / n;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = lums[i] - mean;
    sumSq += d * d;
  }
  const std = Math.sqrt(sumSq / n);

  const sorted = Array.from(lums).sort((a, b) => a - b);
  const median = sorted[n >> 1];
  const p10 = sorted[Math.floor(n * 0.1)];
  const p90 = sorted[Math.floor(n * 0.9)];
  const range = p90 - p10;
  const stats = { mean, median, std, range };

  if (std < EMPTY_STD && range < EMPTY_RANGE) {
    const confidence = 0.5 + 0.5 * clamp01(1 - std / EMPTY_STD);
    return { state: "empty", confidence, stats };
  }

  if (median < BLACK_MAX_LUMA) {
    const confidence =
      0.55 + 0.4 * clamp01((BLACK_MAX_LUMA - median) / BLACK_MAX_LUMA);
    return { state: "black", confidence, stats };
  }
  if (median > WHITE_MIN_LUMA) {
    const confidence =
      0.55 + 0.4 * clamp01((median - WHITE_MIN_LUMA) / (255 - WHITE_MIN_LUMA));
    return { state: "white", confidence, stats };
  }

  return {
    state: median > 120 ? "white" : "black",
    confidence: 0.35,
    stats,
  };
}

export function classifyBoard(crops: HTMLCanvasElement[]): ClassifyResult[] {
  if (crops.length !== 64) {
    throw new Error(`classifyBoard expects 64 crops, got ${crops.length}`);
  }
  return crops.map(classifyCrop);
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
