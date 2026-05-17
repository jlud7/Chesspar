// Dump the exact pre/post pair sent to identify-move for one transition,
// using the same calibrate corners the test rig would derive.
import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "../..");
const PHOTO_DIR = path.join(ROOT, "Test_Photos");
const OUT_DIR = path.join(__dirname, "pair-out");

const WORKER_URL =
  process.env.WORKER_URL ?? "https://chesspar-vlm.jamesleoluddy.workers.dev";
const ORIGIN = process.env.ORIGIN ?? "https://jlud7.github.io";
const RECTIFY_SIZE = Number(process.env.RECTIFY_SIZE ?? "1024");

const PRE_NAME = process.env.PRE ?? "IMG_8827.jpeg";
const POST_NAME = process.env.POST ?? "IMG_8828.jpeg";

async function loadOriented(filePath) {
  let pipe = sharp(filePath).rotate();
  pipe = pipe.rotate(90);
  const rotatedBuf = await pipe.toBuffer();
  const meta = await sharp(rotatedBuf).metadata();
  const W = meta.width;
  const H = meta.height;
  const left = Math.round(W * 0.22);
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

const { calibrateBoard } = await import("../../lib/v2/calibrate.ts");
const boardImage = await import("../../lib/board-image.ts");

await fs.rm(OUT_DIR, { recursive: true, force: true });
await fs.mkdir(OUT_DIR, { recursive: true });

function imageToCanvas(img) {
  const c = createCanvas(img.width, img.height);
  c.getContext("2d").drawImage(img, 0, 0);
  return c;
}

const startImg = await loadOriented(path.join(PHOTO_DIR, "IMG_8819.jpeg"));
const startCanvas = imageToCanvas(startImg);
const cal = await calibrateBoard({
  proxyUrl: WORKER_URL,
  image: startCanvas,
  origin: ORIGIN,
});
if (cal.kind !== "locked") {
  console.error(`CAL FAIL: ${cal.reason}`);
  process.exit(1);
}
console.log(`corners: ${cal.lock.corners.map((p) => `(${Math.round(p.x)},${Math.round(p.y)})`).join(" ")}`);

for (const name of [PRE_NAME, POST_NAME]) {
  const img = await loadOriented(path.join(PHOTO_DIR, name));
  const cnv = imageToCanvas(img);
  const r = boardImage.warpBoardWithMargin(cnv, cal.lock.corners, RECTIFY_SIZE, 0.14);
  await fs.writeFile(path.join(OUT_DIR, `${name}-rect.jpeg`), r.toBuffer("image/jpeg", { quality: 0.92 }));
  // Also save the source (rotated + cropped) for reference.
  await fs.writeFile(path.join(OUT_DIR, `${name}-raw.jpeg`), cnv.toBuffer("image/jpeg", { quality: 0.88 }));
  console.log(`wrote ${name}-rect.jpeg`);
}
