// Dump rectified images so we can see what the VLM actually receives.
import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "../..");
const PHOTO_DIR = path.join(ROOT, "Test_Photos");
const OUT_DIR = path.join(__dirname, "rectified-out");

const ROTATE_QUARTERS = Number(process.env.ROTATE_QUARTERS ?? "1");
const CROP_LEFT = Number(process.env.CROP_LEFT ?? "0.22");

async function loadOriented(filePath) {
  let pipe = sharp(filePath).rotate();
  if (ROTATE_QUARTERS) pipe = pipe.rotate(ROTATE_QUARTERS * 90);
  const rotatedBuf = await pipe.toBuffer();
  const meta = await sharp(rotatedBuf).metadata();
  const W = meta.width;
  const H = meta.height;
  const left = Math.round(W * CROP_LEFT);
  const buf = await sharp(rotatedBuf)
    .extract({ left, top: 0, width: W - left, height: H })
    .toFormat("jpeg")
    .toBuffer();
  return await loadImage(buf);
}

const seedImg = await loadOriented(path.join(PHOTO_DIR, "IMG_8819.jpeg"));
const seedCanvas = createCanvas(1, 1);
globalThis.document = {
  createElement(tag) {
    if (tag !== "canvas") throw new Error(`unsupported: ${tag}`);
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

const boardImage = await import("../../lib/board-image.ts");

await fs.rm(OUT_DIR, { recursive: true, force: true });
await fs.mkdir(OUT_DIR, { recursive: true });

// Hard-coded corners from a previous calibrate run on IMG_8819 (rounded).
const CORNERS = [
  { x: 243, y: 63 }, // a8
  { x: 1155, y: 76 }, // h8
  { x: 1150, y: 1090 }, // h1
  { x: 238, y: 1127 }, // a1
];

function imageToCanvas(img) {
  const c = createCanvas(img.width, img.height);
  c.getContext("2d").drawImage(img, 0, 0);
  return c;
}

for (let i = 8819; i <= 8833; i++) {
  const name = `IMG_${i}.jpeg`;
  const img = await loadOriented(path.join(PHOTO_DIR, name));
  const cnv = imageToCanvas(img);
  const r1 = boardImage.warpBoardWithMargin(cnv, CORNERS, 512, 0.14);
  await fs.writeFile(path.join(OUT_DIR, `${name}-512.jpeg`), r1.toBuffer("image/jpeg", { quality: 0.88 }));
  const r2 = boardImage.warpBoardWithMargin(cnv, CORNERS, 1024, 0.14);
  await fs.writeFile(path.join(OUT_DIR, `${name}-1024.jpeg`), r2.toBuffer("image/jpeg", { quality: 0.9 }));
  console.log(`Wrote ${name}`);
}
console.log(`\nOpen ${OUT_DIR} to inspect.`);
