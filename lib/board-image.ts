import { computeHomography, type Point } from "@/lib/homography";

/**
 * Warp a source image into a square `size x size` rectified board.
 *
 * `corners` are the four image-space points in chess order:
 *   [a8, h8, h1, a1]
 * which become the corners (0,0), (size,0), (size,size), (0,size) in the
 * rectified output. The rectified image is laid out like a printed chess
 * diagram: a8 top-left, h1 bottom-right.
 *
 * Implementation uses inverse warping with bilinear sampling: we compute the
 * homography taking rectified coords -> source coords, then for each output
 * pixel sample the source.
 */
export function warpBoard(
  source: HTMLImageElement | HTMLCanvasElement,
  corners: [Point, Point, Point, Point],
  size: number,
): HTMLCanvasElement {
  const rectifiedCorners: [Point, Point, Point, Point] = [
    { x: 0, y: 0 },
    { x: size, y: 0 },
    { x: size, y: size },
    { x: 0, y: size },
  ];
  const H = computeHomography(rectifiedCorners, corners);

  const srcCanvas = sourceToCanvas(source);
  const sctx = srcCanvas.getContext("2d", { willReadFrequently: true });
  if (!sctx) throw new Error("Failed to acquire source 2D context");
  const sImg = sctx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
  const sData = sImg.data;
  const sW = sImg.width;
  const sH = sImg.height;

  const out = document.createElement("canvas");
  out.width = size;
  out.height = size;
  const octx = out.getContext("2d");
  if (!octx) throw new Error("Failed to acquire output 2D context");
  const oImg = octx.createImageData(size, size);
  const oData = oImg.data;

  const h00 = H[0][0], h01 = H[0][1], h02 = H[0][2];
  const h10 = H[1][0], h11 = H[1][1], h12 = H[1][2];
  const h20 = H[2][0], h21 = H[2][1], h22 = H[2][2];

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const w = h20 * x + h21 * y + h22;
      const sx = (h00 * x + h01 * y + h02) / w;
      const sy = (h10 * x + h11 * y + h12) / w;
      const oi = (y * size + x) * 4;
      if (sx < 0 || sy < 0 || sx > sW - 1 || sy > sH - 1) {
        oData[oi] = 0;
        oData[oi + 1] = 0;
        oData[oi + 2] = 0;
        oData[oi + 3] = 255;
        continue;
      }
      const x0 = sx | 0;
      const y0 = sy | 0;
      const x1 = x0 + 1 < sW ? x0 + 1 : x0;
      const y1 = y0 + 1 < sH ? y0 + 1 : y0;
      const fx = sx - x0;
      const fy = sy - y0;
      const i00 = (y0 * sW + x0) * 4;
      const i10 = (y0 * sW + x1) * 4;
      const i01 = (y1 * sW + x0) * 4;
      const i11 = (y1 * sW + x1) * 4;
      const w00 = (1 - fx) * (1 - fy);
      const w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy;
      const w11 = fx * fy;
      oData[oi] =
        sData[i00] * w00 + sData[i10] * w10 + sData[i01] * w01 + sData[i11] * w11;
      oData[oi + 1] =
        sData[i00 + 1] * w00 +
        sData[i10 + 1] * w10 +
        sData[i01 + 1] * w01 +
        sData[i11 + 1] * w11;
      oData[oi + 2] =
        sData[i00 + 2] * w00 +
        sData[i10 + 2] * w10 +
        sData[i01 + 2] * w01 +
        sData[i11 + 2] * w11;
      oData[oi + 3] = 255;
    }
  }
  octx.putImageData(oImg, 0, 0);
  return out;
}

/**
 * Slice a square rectified board into its 64 per-square crops.
 * Output is row-major from a8 (index 0) to h1 (index 63):
 *   index = (8 - rank) * 8 + fileIndex   where fileIndex: a=0..h=7
 */
export function extractSquareCrops(warped: HTMLCanvasElement): HTMLCanvasElement[] {
  const size = warped.width;
  if (warped.height !== size) {
    throw new Error("Rectified board must be square");
  }
  const sq = size / 8;
  const crops: HTMLCanvasElement[] = [];
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const sx = Math.round(f * sq);
      const sy = Math.round(r * sq);
      const sSize = Math.round((f + 1) * sq) - sx;
      const sSizeY = Math.round((r + 1) * sq) - sy;
      const c = document.createElement("canvas");
      c.width = sSize;
      c.height = sSizeY;
      const ctx = c.getContext("2d");
      if (!ctx) throw new Error("Failed to acquire crop 2D context");
      ctx.drawImage(warped, sx, sy, sSize, sSizeY, 0, 0, sSize, sSizeY);
      crops.push(c);
    }
  }
  return crops;
}

function sourceToCanvas(src: HTMLImageElement | HTMLCanvasElement): HTMLCanvasElement {
  if (src instanceof HTMLCanvasElement) return src;
  const c = document.createElement("canvas");
  c.width = src.naturalWidth;
  c.height = src.naturalHeight;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("Failed to acquire source 2D context");
  ctx.drawImage(src, 0, 0);
  return c;
}

export type Rotation = 0 | 90 | 180 | 270;

/**
 * Rotate a canvas by 0/90/180/270 degrees clockwise. Returns the input
 * untouched on 0; allocates a fresh canvas otherwise.
 */
export function rotateCanvas(
  canvas: HTMLCanvasElement,
  degrees: Rotation,
): HTMLCanvasElement {
  if (degrees === 0) return canvas;
  const out = document.createElement("canvas");
  if (degrees === 90 || degrees === 270) {
    out.width = canvas.height;
    out.height = canvas.width;
  } else {
    out.width = canvas.width;
    out.height = canvas.height;
  }
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("rotateCanvas: failed to get 2D context");
  ctx.save();
  ctx.translate(out.width / 2, out.height / 2);
  ctx.rotate((degrees * Math.PI) / 180);
  ctx.drawImage(canvas, -canvas.width / 2, -canvas.height / 2);
  ctx.restore();
  return out;
}

/**
 * Detect the rotation that puts WHITE PIECES at the bottom of the
 * rectified board, and apply it. Returns the (possibly identical)
 * canvas plus the rotation that was applied.
 *
 * Heuristic: white pieces are markedly brighter than black pieces, so
 * the back-rank slice with white's pieces is the brightest of the four
 * board edges. We sample the mean luminance of the outermost 2 ranks on
 * each side (top / bottom / left / right) and rotate so that the
 * brightest side ends up at the bottom of the canvas.
 *
 * This is a defence-in-depth check: the corner-ordering contract in
 * warpBoard already produces "a8 top-left → white at bottom" output
 * when corners are passed correctly, but mis-calibrated corners (board
 * captured sideways, or rotated 180° by manual taps) would otherwise
 * leak a wrong-orientation board into the VLM, which produces
 * confidently-wrong piece identifications. White-at-bottom matches what
 * VLMs have seen the most of during training.
 */
export function ensureWhiteAtBottom(
  warped: HTMLCanvasElement,
): { oriented: HTMLCanvasElement; rotationDeg: Rotation } {
  const size = warped.width;
  if (warped.height !== size) {
    return { oriented: warped, rotationDeg: 0 };
  }
  const ctx = warped.getContext("2d", { willReadFrequently: true });
  if (!ctx) return { oriented: warped, rotationDeg: 0 };
  const cell = size / 8;
  const strip = Math.max(2, Math.round(cell * 2));

  function meanLum(x0: number, y0: number, w: number, h: number): number {
    if (w <= 0 || h <= 0) return 0;
    const data = ctx!.getImageData(x0, y0, w, h).data;
    let s = 0;
    const n = data.length / 4;
    for (let i = 0; i < data.length; i += 4) {
      s += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    return s / n;
  }

  const top = meanLum(0, 0, size, strip);
  const bottom = meanLum(0, size - strip, size, strip);
  const left = meanLum(0, 0, strip, size);
  const right = meanLum(size - strip, 0, strip, size);

  // Each entry says: "if we rotate by R, this side becomes the new bottom".
  // CW rotation: original right → new bottom; 180 → top → new bottom; 270 (=90 CCW) → left → new bottom.
  const candidates: { rot: Rotation; lum: number }[] = [
    { rot: 0, lum: bottom },
    { rot: 90, lum: right },
    { rot: 180, lum: top },
    { rot: 270, lum: left },
  ];
  candidates.sort((a, b) => b.lum - a.lum);
  const best = candidates[0];
  if (best.rot === 0) return { oriented: warped, rotationDeg: 0 };
  return { oriented: rotateCanvas(warped, best.rot), rotationDeg: best.rot };
}
