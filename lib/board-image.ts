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
