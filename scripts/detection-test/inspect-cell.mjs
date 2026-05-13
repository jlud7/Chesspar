import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "../..");
const PHOTO_DIR = path.join(ROOT, "test_photos");
const OUT_DIR = path.join(__dirname, "out");

async function loadOriented(filePath) {
  const buf = await sharp(filePath).rotate().toFormat("jpeg").toBuffer();
  return await loadImage(buf);
}

const seedImg = await loadOriented(path.join(PHOTO_DIR, "IMG_8819.jpeg"));
const seedCanvas = createCanvas(1, 1);
globalThis.document = {
  createElement(tag) {
    if (tag !== "canvas") throw new Error(`unsupported`);
    return createCanvas(1, 1);
  },
};
globalThis.HTMLImageElement = class {};
globalThis.HTMLCanvasElement = class {};
Object.defineProperty(globalThis.HTMLImageElement, Symbol.hasInstance, {
  value: (v) => v && v.constructor === seedImg.constructor,
});
Object.defineProperty(globalThis.HTMLCanvasElement, Symbol.hasInstance, {
  value: (v) => v && v.constructor === seedCanvas.constructor,
});

const detection = await import("../../lib/board-detection.ts");
const boardImage = await import("../../lib/board-image.ts");

const img = await loadOriented(path.join(PHOTO_DIR, "IMG_8819.jpeg"));
const detect = detection.autoDetectBoardCorners(img);
if (!detect) throw new Error("detection failed");

// Render the warp at high resolution AND draw the inner-pad windows used
// for stat computation.
const SIZE = 1024;
const warped = boardImage.warpBoard(img, detect.corners, SIZE);
const c = createCanvas(SIZE, SIZE);
const ctx = c.getContext("2d");
ctx.drawImage(warped, 0, 0);
const cs = SIZE / 8;
const INNER = 0.18;
const x0frac = INNER;
const x1frac = 1 - INNER;
ctx.strokeStyle = "rgba(0,255,255,0.8)";
ctx.lineWidth = 2;
for (let r = 0; r < 8; r++) {
  for (let f = 0; f < 8; f++) {
    const x = f * cs + cs * x0frac;
    const y = r * cs + cs * x0frac;
    const w = cs * (x1frac - x0frac);
    const h = cs * (x1frac - x0frac);
    ctx.strokeRect(x, y, w, h);
  }
}
ctx.strokeStyle = "rgba(255,255,0,0.9)";
ctx.lineWidth = 1;
for (let i = 0; i <= 8; i++) {
  ctx.beginPath();
  ctx.moveTo(0, i * cs);
  ctx.lineTo(SIZE, i * cs);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(i * cs, 0);
  ctx.lineTo(i * cs, SIZE);
  ctx.stroke();
}
ctx.fillStyle = "rgba(255,255,255,0.9)";
ctx.strokeStyle = "rgba(0,0,0,0.95)";
ctx.lineWidth = 3;
ctx.font = "20px sans-serif";
const FILES = "abcdefgh";
for (let f = 0; f < 8; f++) {
  ctx.strokeText(FILES[f], f * cs + 6, 22);
  ctx.fillText(FILES[f], f * cs + 6, 22);
}
for (let r = 0; r < 8; r++) {
  ctx.strokeText(String(8 - r), 4, r * cs + cs / 2);
  ctx.fillText(String(8 - r), 4, r * cs + cs / 2);
}
await fs.writeFile(
  path.join(OUT_DIR, "IMG_8819_grid.png"),
  c.toBuffer("image/png"),
);
console.log("wrote IMG_8819_grid.png");
