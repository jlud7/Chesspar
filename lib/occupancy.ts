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
 * Per-square colour + texture signature used by both the calibrated
 * classifier and the baseline builder.
 */
export type SquareStats = {
  meanR: number;
  meanG: number;
  meanB: number;
  meanL: number;
  std: number;
};

/**
 * Per-board baseline learned from a rectified starting-position frame.
 * Each entry is the average colour signature of the four buckets:
 * empty/white/black × on-light-square/on-dark-square.
 */
export type BaselineSignature = {
  emptyLight: SquareStats;
  emptyDark: SquareStats;
  whiteOnLight: SquareStats;
  whiteOnDark: SquareStats;
  blackOnLight: SquareStats;
  blackOnDark: SquareStats;
};

/** Index → true if a8/h1-style light square (a8 is light by chess convention). */
export function isLightSquare(idx: number): boolean {
  return ((Math.floor(idx / 8) + (idx % 8)) % 2) === 0;
}

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

/**
 * Compute the full RGB + std signature of a single square's inner region.
 * Same window as the heuristic classifier so the two paths stay comparable.
 */
export function computeSquareStats(crop: HTMLCanvasElement): SquareStats {
  const w = crop.width;
  const h = crop.height;
  const ctx = crop.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("computeSquareStats: failed to get 2D context");
  const x0 = Math.floor(w * INNER_PAD);
  const x1 = Math.max(x0 + 1, Math.ceil(w * (1 - INNER_PAD)));
  const y0 = Math.floor(h * INNER_PAD);
  const y1 = Math.max(y0 + 1, Math.ceil(h * (1 - INNER_PAD)));
  const iw = x1 - x0;
  const ih = y1 - y0;
  const data = ctx.getImageData(x0, y0, iw, ih).data;
  const n = iw * ih;
  let sumR = 0,
    sumG = 0,
    sumB = 0,
    sumL = 0,
    sumL2 = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const l = 0.299 * r + 0.587 * g + 0.114 * b;
    sumR += r;
    sumG += g;
    sumB += b;
    sumL += l;
    sumL2 += l * l;
  }
  const meanL = sumL / n;
  const std = Math.sqrt(Math.max(0, sumL2 / n - meanL * meanL));
  return {
    meanR: sumR / n,
    meanG: sumG / n,
    meanB: sumB / n,
    meanL,
    std,
  };
}

/**
 * Build a per-board calibration from a rectified starting-position frame.
 * The 32 empty squares (ranks 3-6) and the 16 white / 16 black piece
 * squares give us six bucket means (3 occupancy × 2 board-square colours).
 */
export function computeBaseline(crops: HTMLCanvasElement[]): BaselineSignature {
  if (crops.length !== 64) {
    throw new Error(`computeBaseline expects 64 crops, got ${crops.length}`);
  }
  const buckets = {
    el: [] as SquareStats[],
    ed: [] as SquareStats[],
    wl: [] as SquareStats[],
    wd: [] as SquareStats[],
    bl: [] as SquareStats[],
    bd: [] as SquareStats[],
  };
  for (let i = 0; i < 64; i++) {
    const row = Math.floor(i / 8); // 0 = rank 8 (top, Black), 7 = rank 1 (bottom, White)
    const light = isLightSquare(i);
    const s = computeSquareStats(crops[i]);
    if (row >= 2 && row <= 5) {
      if (light) buckets.el.push(s);
      else buckets.ed.push(s);
    } else if (row >= 6) {
      if (light) buckets.wl.push(s);
      else buckets.wd.push(s);
    } else {
      if (light) buckets.bl.push(s);
      else buckets.bd.push(s);
    }
  }
  return {
    emptyLight: averageStats(buckets.el),
    emptyDark: averageStats(buckets.ed),
    whiteOnLight: averageStats(buckets.wl),
    whiteOnDark: averageStats(buckets.wd),
    blackOnLight: averageStats(buckets.bl),
    blackOnDark: averageStats(buckets.bd),
  };
}

/**
 * Classify a single square against the per-board baseline. Picks whichever
 * of {empty, white-piece, black-piece} prototype is nearest in the
 * (R,G,B,std) feature space for the square's known light/dark colour.
 */
export function classifyCropCalibrated(
  crop: HTMLCanvasElement,
  baseline: BaselineSignature,
  squareIsLight: boolean,
): ClassifyResult {
  const stats = computeSquareStats(crop);
  const refs = squareIsLight
    ? {
        empty: baseline.emptyLight,
        white: baseline.whiteOnLight,
        black: baseline.blackOnLight,
      }
    : {
        empty: baseline.emptyDark,
        white: baseline.whiteOnDark,
        black: baseline.blackOnDark,
      };
  const dE = statsDistance(stats, refs.empty);
  const dW = statsDistance(stats, refs.white);
  const dB = statsDistance(stats, refs.black);
  let state: Occupancy = "empty";
  let best = dE;
  if (dW < best) {
    state = "white";
    best = dW;
  }
  if (dB < best) {
    state = "black";
    best = dB;
  }
  const sorted = [dE, dW, dB].sort((a, b) => a - b);
  const margin = sorted[1] > 0 ? 1 - sorted[0] / sorted[1] : 0;
  const confidence = clamp01(0.35 + margin);
  return {
    state,
    confidence,
    stats: { mean: stats.meanL, median: stats.meanL, std: stats.std, range: 0 },
  };
}

export function classifyBoardCalibrated(
  crops: HTMLCanvasElement[],
  baseline: BaselineSignature,
): ClassifyResult[] {
  if (crops.length !== 64) {
    throw new Error(`classifyBoardCalibrated expects 64 crops, got ${crops.length}`);
  }
  return crops.map((c, i) =>
    classifyCropCalibrated(c, baseline, isLightSquare(i)),
  );
}

function averageStats(list: SquareStats[]): SquareStats {
  if (list.length === 0) {
    return { meanR: 128, meanG: 128, meanB: 128, meanL: 128, std: 30 };
  }
  let r = 0,
    g = 0,
    b = 0,
    l = 0,
    s = 0;
  for (const x of list) {
    r += x.meanR;
    g += x.meanG;
    b += x.meanB;
    l += x.meanL;
    s += x.std;
  }
  const n = list.length;
  return {
    meanR: r / n,
    meanG: g / n,
    meanB: b / n,
    meanL: l / n,
    std: s / n,
  };
}

function statsDistance(a: SquareStats, b: SquareStats): number {
  const dR = a.meanR - b.meanR;
  const dG = a.meanG - b.meanG;
  const dB = a.meanB - b.meanB;
  const colourDist = Math.sqrt(dR * dR + dG * dG + dB * dB);
  const stdDist = Math.abs(a.std - b.std);
  return colourDist + 0.5 * stdDist;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
