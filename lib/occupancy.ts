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
 *
 * Tracks luminance + chroma (R-G, R-B) + within-square std + saturation.
 * Chroma is needed for non-standard board colours where the dark square is
 * coloured (red/green/blue) instead of true black — luminance alone can't
 * tell a red square from a dark piece on a cream square.
 */
export type SquareStats = {
  meanR: number;
  meanG: number;
  meanB: number;
  meanL: number;
  /** R - G — captures red/green hue shift. */
  chromaRG: number;
  /** R - B — captures red/blue hue shift. */
  chromaRB: number;
  /** Standard deviation of luminance within the crop. */
  std: number;
  /** Approximate saturation: max channel − min channel, averaged. */
  saturation: number;
};

/**
 * Per-board baseline learned from a rectified starting-position frame.
 * Each entry is the median colour signature of the four buckets we can
 * label from the starting layout:
 *   - empty on light parity / empty on dark parity
 *   - white piece on light / white piece on dark
 *   - black piece on light / black piece on dark
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
  return (Math.floor(idx / 8) + (idx % 8)) % 2 === 0;
}

// Fixed-threshold heuristic — only used before the per-board baseline is
// learned (i.e., the starting-frame detection step). Once we have a
// baseline, the calibrated classifier is dramatically more accurate.
const INNER_PAD = 0.18;
const EMPTY_STD = 18;
const EMPTY_RANGE = 60;
const BLACK_MAX_LUMA = 95;
const WHITE_MIN_LUMA = 145;

export function classifyCrop(crop: HTMLCanvasElement): ClassifyResult {
  const { stats } = readCropStats(crop);
  if (stats.std < EMPTY_STD) {
    const confidence = 0.5 + 0.5 * clamp01(1 - stats.std / EMPTY_STD);
    return { state: "empty", confidence, stats: legacyStats(stats) };
  }
  if (stats.meanL < BLACK_MAX_LUMA) {
    const confidence =
      0.55 + 0.4 * clamp01((BLACK_MAX_LUMA - stats.meanL) / BLACK_MAX_LUMA);
    return { state: "black", confidence, stats: legacyStats(stats) };
  }
  if (stats.meanL > WHITE_MIN_LUMA) {
    const confidence =
      0.55 +
      0.4 * clamp01((stats.meanL - WHITE_MIN_LUMA) / (255 - WHITE_MIN_LUMA));
    return { state: "white", confidence, stats: legacyStats(stats) };
  }
  return {
    state: stats.meanL > 120 ? "white" : "black",
    confidence: 0.35,
    stats: legacyStats(stats),
  };
}

export function classifyBoard(crops: HTMLCanvasElement[]): ClassifyResult[] {
  if (crops.length !== 64) {
    throw new Error(`classifyBoard expects 64 crops, got ${crops.length}`);
  }
  return crops.map(classifyCrop);
}

export function computeSquareStats(crop: HTMLCanvasElement): SquareStats {
  return readCropStats(crop).stats;
}

/**
 * Build a per-board calibration from a rectified starting-position frame.
 * The 32 empty squares (ranks 3-6) and the 16 white / 16 black piece
 * squares give us six bucket signatures. We use the *median* per channel
 * inside each bucket rather than the mean — robust to outliers caused by
 * piece-top overhang from adjacent rows, mat curl, or stray shadows.
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
    emptyLight: medianStats(buckets.el),
    emptyDark: medianStats(buckets.ed),
    whiteOnLight: medianStats(buckets.wl),
    whiteOnDark: medianStats(buckets.wd),
    blackOnLight: medianStats(buckets.bl),
    blackOnDark: medianStats(buckets.bd),
  };
}

/**
 * Classify a single square against the per-board baseline using a
 * pixel-vote scheme instead of cell-averaged statistics.
 *
 * Why pixel-vote: a small pawn occupies maybe 25% of its square; the rest
 * is empty board colour. If we average over the whole cell, the pawn's
 * signature is washed out — the cell looks ~empty. By voting per pixel
 * instead, we ask "how much of this cell looks like non-board content?"
 * which is robust to small pieces.
 *
 * For each pixel we compute its distance from the learned empty-board
 * colour for the cell's light/dark parity. Pixels far from the empty
 * baseline are tagged "non-board" and classified by luminance — brighter
 * than the baseline = part of a white piece, darker = black piece. The
 * cell vote is the dominant non-board class if there's enough non-board
 * content, else "empty".
 *
 * Falls back to the previous cell-averaged classifier when the pixel-vote
 * count is ambiguous — that legacy path handles tie-breaks via the median
 * piece prototype.
 */
export function classifyCropCalibrated(
  crop: HTMLCanvasElement,
  baseline: BaselineSignature,
  squareIsLight: boolean,
): ClassifyResult {
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
  // Pixel-vote provides "non-board" counts. Tight noise tolerance means
  // these counts only spike when an actual piece silhouette is present.
  const pv = pixelVoteCrop(crop, refs.empty);
  const stats = pv.stats;

  // Stats-distance to each prototype as the primary signal.
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

  // Pixel-vote overrides: only when pixel content is unambiguous. A clean
  // "lots of non-board pixels and clearly one colour" beats stats noise;
  // anything murkier defers to the stats distance.
  if (pv.nonBoardFrac > 0.16) {
    if (pv.whiteFrac > pv.blackFrac * 2 && pv.whiteFrac > 0.05) {
      state = "white";
    } else if (pv.blackFrac > pv.whiteFrac * 2 && pv.blackFrac > 0.05) {
      state = "black";
    }
  } else if (pv.nonBoardFrac < 0.04) {
    // Clearly empty — overrides "white/black" if the stats-distance
    // mis-picked due to lighting drift.
    state = "empty";
  }

  const sorted = [dE, dW, dB].sort((a, b) => a - b);
  const margin = sorted[1] > 0 ? 1 - sorted[0] / sorted[1] : 0;
  const confidence = clamp01(0.4 + 0.55 * margin);
  return {
    state,
    confidence,
    stats: legacyStats(stats),
  };
}

/**
 * Per-cell RMS colour distance between two rectified boards, computed
 * over each cell's inner pad. Larger values mean the cell's pixels are
 * very different between frames — strong evidence that a piece arrived,
 * left, or got swapped (capture, promotion).
 *
 * Robust signal: completely independent of which prototype each cell
 * matches, so it works even when the classifier mislabels a square.
 * Pair with `inferMoveFuzzy` via the `cellDeltas` option to upgrade
 * "this might be e4 or e3" to "this is e4 because the e4 cell *changed*".
 */
export function computeCellDeltas(
  prevCrops: HTMLCanvasElement[],
  currCrops: HTMLCanvasElement[],
): number[] {
  if (prevCrops.length !== 64 || currCrops.length !== 64) {
    throw new Error("computeCellDeltas needs 64 crops each");
  }
  const out = new Array<number>(64);
  for (let i = 0; i < 64; i++) {
    out[i] = cellDelta(prevCrops[i], currCrops[i]);
  }
  return out;
}

function cellDelta(prev: HTMLCanvasElement, curr: HTMLCanvasElement): number {
  // Sample at a low resolution so the deltas track piece-base appearance,
  // not high-frequency noise. Tighter inner window than the classifier
  // uses: piece bases sit at cell centres, but 3D piece tops *bleed* into
  // neighbouring cells' edges due to camera tilt. Keeping the delta window
  // away from the edges focuses on the piece's footprint instead of the
  // neighbour's overhang.
  const SIZE = 28;
  const c1 = downsample(prev, SIZE);
  const c2 = downsample(curr, SIZE);
  const inner = Math.floor(SIZE * 0.3);
  const ctx1 = c1.getContext("2d", { willReadFrequently: true });
  const ctx2 = c2.getContext("2d", { willReadFrequently: true });
  if (!ctx1 || !ctx2) return 0;
  const iw = SIZE - inner * 2;
  if (iw <= 0) return 0;
  const a = ctx1.getImageData(inner, inner, iw, iw).data;
  const b = ctx2.getImageData(inner, inner, iw, iw).data;
  let sumSq = 0;
  const px = a.length / 4;
  for (let i = 0; i < a.length; i += 4) {
    const dr = a[i] - b[i];
    const dg = a[i + 1] - b[i + 1];
    const db = a[i + 2] - b[i + 2];
    sumSq += dr * dr + dg * dg + db * db;
  }
  return Math.sqrt(sumSq / px);
}

function downsample(src: HTMLCanvasElement, size: number): HTMLCanvasElement {
  if (src.width === size && src.height === size) return src;
  const out = document.createElement("canvas");
  out.width = size;
  out.height = size;
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("downsample: failed to get 2D context");
  ctx.drawImage(src, 0, 0, size, size);
  return out;
}

export function classifyBoardCalibrated(
  crops: HTMLCanvasElement[],
  baseline: BaselineSignature,
): ClassifyResult[] {
  if (crops.length !== 64) {
    throw new Error(
      `classifyBoardCalibrated expects 64 crops, got ${crops.length}`,
    );
  }
  return crops.map((c, i) =>
    classifyCropCalibrated(c, baseline, isLightSquare(i)),
  );
}

/**
 * Per-pixel vote against the learned empty-board signature.
 *
 * Walks the inner-pad region of the crop; for each pixel computes its
 * Euclidean RGB distance to the empty-board mean and the per-channel
 * delta. Pixels whose distance exceeds a learned threshold count as
 * "non-board"; non-board pixels are further classified by luminance
 * relative to the board's empty mean — significantly brighter = part of a
 * white piece, significantly darker = part of a black piece.
 *
 * The threshold uses a fixed floor (24) plus 1.6× the board's own
 * within-cell std — adapts to noisy boards (e.g., grainy textures) without
 * letting clean boards have a too-permissive cutoff.
 */
function pixelVoteCrop(
  crop: HTMLCanvasElement,
  emptyRef: SquareStats,
): {
  stats: SquareStats;
  nonBoardFrac: number;
  whiteFrac: number;
  blackFrac: number;
} {
  const w = crop.width;
  const h = crop.height;
  const ctx = crop.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("pixelVoteCrop: failed to get 2D context");
  const x0 = Math.floor(w * INNER_PAD);
  const x1 = Math.max(x0 + 1, Math.ceil(w * (1 - INNER_PAD)));
  const y0 = Math.floor(h * INNER_PAD);
  const y1 = Math.max(y0 + 1, Math.ceil(h * (1 - INNER_PAD)));
  const iw = x1 - x0;
  const ih = y1 - y0;
  const data = ctx.getImageData(x0, y0, iw, ih).data;
  const total = iw * ih;

  // Tolerance for "this pixel looks like board". Aggressive floor avoids
  // false-positives from JPEG noise + shadows + sub-cell border bleed;
  // std-scaled component handles textured boards where the floor is loose.
  const colorTol = Math.max(45, emptyRef.std * 2.5);
  const colorTolSq = colorTol * colorTol;
  // Luminance jump for tagging a non-board pixel as part of a white vs
  // black piece. Pieces have very strong luminance contrast against board
  // colours of either parity, so the floor is set well above noise.
  const lumJump = Math.max(35, emptyRef.std * 2.0);

  let sumR = 0,
    sumG = 0,
    sumB = 0,
    sumL = 0,
    sumL2 = 0,
    sumSat = 0;
  let nonBoard = 0;
  let whiteish = 0;
  let blackish = 0;

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
    sumSat += Math.max(r, g, b) - Math.min(r, g, b);

    const dr = r - emptyRef.meanR;
    const dg = g - emptyRef.meanG;
    const db = b - emptyRef.meanB;
    if (dr * dr + dg * dg + db * db > colorTolSq) {
      nonBoard++;
      if (l > emptyRef.meanL + lumJump) whiteish++;
      else if (l < emptyRef.meanL - lumJump) blackish++;
    }
  }

  const meanR = sumR / total;
  const meanG = sumG / total;
  const meanB = sumB / total;
  const meanL = sumL / total;
  const std = Math.sqrt(Math.max(0, sumL2 / total - meanL * meanL));
  return {
    stats: {
      meanR,
      meanG,
      meanB,
      meanL,
      chromaRG: meanR - meanG,
      chromaRB: meanR - meanB,
      std,
      saturation: sumSat / total,
    },
    nonBoardFrac: nonBoard / total,
    whiteFrac: whiteish / total,
    blackFrac: blackish / total,
  };
}

/** Read the inner-pad colour + texture signature out of a crop. */
function readCropStats(crop: HTMLCanvasElement): {
  stats: SquareStats;
} {
  const w = crop.width;
  const h = crop.height;
  const ctx = crop.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("readCropStats: failed to get 2D context");
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
    sumL2 = 0,
    sumSat = 0;
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
    sumSat += Math.max(r, g, b) - Math.min(r, g, b);
  }
  const meanR = sumR / n;
  const meanG = sumG / n;
  const meanB = sumB / n;
  const meanL = sumL / n;
  const std = Math.sqrt(Math.max(0, sumL2 / n - meanL * meanL));
  return {
    stats: {
      meanR,
      meanG,
      meanB,
      meanL,
      chromaRG: meanR - meanG,
      chromaRB: meanR - meanB,
      std,
      saturation: sumSat / n,
    },
  };
}

function legacyStats(s: SquareStats): ClassifyResult["stats"] {
  return { mean: s.meanL, median: s.meanL, std: s.std, range: 0 };
}

/**
 * Median-per-channel reduction. Robust to a couple of outliers per bucket
 * — the bucket sizes are small (16 white pieces, 32 empty cells, etc.) so
 * a single mis-aligned square can pull a mean off significantly.
 */
function medianStats(list: SquareStats[]): SquareStats {
  if (list.length === 0) {
    return {
      meanR: 128,
      meanG: 128,
      meanB: 128,
      meanL: 128,
      chromaRG: 0,
      chromaRB: 0,
      std: 30,
      saturation: 0,
    };
  }
  function median(extract: (s: SquareStats) => number): number {
    const arr = list.map(extract).sort((a, b) => a - b);
    return arr[arr.length >> 1];
  }
  return {
    meanR: median((s) => s.meanR),
    meanG: median((s) => s.meanG),
    meanB: median((s) => s.meanB),
    meanL: median((s) => s.meanL),
    chromaRG: median((s) => s.chromaRG),
    chromaRB: median((s) => s.chromaRB),
    std: median((s) => s.std),
    saturation: median((s) => s.saturation),
  };
}

/**
 * Multi-feature distance. The weights are calibrated so each feature
 * contributes ~comparable variance in typical mixed lighting; in
 * particular, chroma and saturation get less weight than luminance because
 * white-balance shifts make them less stable than relative luminance.
 */
function statsDistance(a: SquareStats, b: SquareStats): number {
  const dL = (a.meanL - b.meanL) / 40;
  const dRG = (a.chromaRG - b.chromaRG) / 25;
  const dRB = (a.chromaRB - b.chromaRB) / 25;
  const dStd = (a.std - b.std) / 20;
  const dSat = (a.saturation - b.saturation) / 30;
  return Math.sqrt(
    dL * dL + dRG * dRG + dRB * dRB + dStd * dStd + 0.5 * dSat * dSat,
  );
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
