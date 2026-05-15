import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "../..");
const PHOTO_DIR = path.join(ROOT, "Test_Photos");
const REF_PATH = path.join(__dirname, "corner-reference.json");
const DETECTOR = process.env.DETECTOR === "legacy" ? "legacy" : "current";
const OUT_DIR = path.join(__dirname, "out", "corner-benchmark", DETECTOR);

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
    if (tag !== "canvas") throw new Error(`unsupported createElement: ${tag}`);
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
const reference = JSON.parse(await fs.readFile(REF_PATH, "utf8"));

const detectFn =
  DETECTOR === "legacy"
    ? detection.autoDetectBoardCornersLegacy
    : detection.autoDetectBoardCorners;

await fs.rm(OUT_DIR, { recursive: true, force: true });
await fs.mkdir(OUT_DIR, { recursive: true });

const photos = (await fs.readdir(PHOTO_DIR))
  .filter((n) => /\.(jpe?g|png)$/i.test(n))
  .sort();

const rows = [];
let totalAvg = 0;
let totalMax = 0;
let passCount = 0;

for (const name of photos) {
  const expected = reference.corners[name];
  if (!expected) throw new Error(`missing reference corners for ${name}`);
  const img = await loadOriented(path.join(PHOTO_DIR, name));
  const detected = detectFn(img);
  if (!detected) throw new Error(`detection failed for ${name}`);

  const diag = Math.hypot(img.width, img.height);
  const errs = expected.map((p, i) =>
    Math.hypot(
      detected.corners[i].x - p.x,
      detected.corners[i].y - p.y,
    ),
  );
  const avgPx = errs.reduce((sum, v) => sum + v, 0) / errs.length;
  const maxPx = Math.max(...errs);
  const avgDiagPct = (avgPx / diag) * 100;
  const maxDiagPct = (maxPx / diag) * 100;
  const withinSpec = maxDiagPct <= 2;

  totalAvg += avgDiagPct;
  totalMax += maxDiagPct;
  if (withinSpec) passCount++;

  const stem = name.replace(/\.[^.]+$/, "");
  const overlay = createCanvas(img.width, img.height);
  const octx = overlay.getContext("2d");
  octx.drawImage(img, 0, 0);

  const drawQuad = (corners, color, labelPrefix) => {
    octx.strokeStyle = color;
    octx.lineWidth = Math.max(4, img.width / 220);
    octx.beginPath();
    corners.forEach((p, i) => {
      if (i === 0) octx.moveTo(p.x, p.y);
      else octx.lineTo(p.x, p.y);
    });
    octx.closePath();
    octx.stroke();
    octx.font = `${Math.max(18, img.width / 42)}px sans-serif`;
    corners.forEach((p, i) => {
      octx.fillStyle = color;
      octx.beginPath();
      octx.arc(p.x, p.y, Math.max(8, img.width / 90), 0, Math.PI * 2);
      octx.fill();
      octx.fillStyle = "white";
      octx.strokeStyle = "black";
      octx.lineWidth = 3;
      octx.strokeText(`${labelPrefix}${i}`, p.x + 10, p.y - 10);
      octx.fillText(`${labelPrefix}${i}`, p.x + 10, p.y - 10);
    });
  };

  drawQuad(expected, "rgba(234, 179, 8, 0.95)", "R");
  drawQuad(detected.corners, "rgba(16, 185, 129, 0.95)", "D");

  await fs.writeFile(
    path.join(OUT_DIR, `${stem}_overlay.png`),
    overlay.toBuffer("image/png"),
  );
  const warped = boardImage.warpBoard(img, detected.corners, 384);
  await fs.writeFile(
    path.join(OUT_DIR, `${stem}_warp.png`),
    warped.toBuffer("image/png"),
  );

  rows.push({
    name,
    avgPx: +avgPx.toFixed(3),
    maxPx: +maxPx.toFixed(3),
    avgDiagPct: +avgDiagPct.toFixed(3),
    maxDiagPct: +maxDiagPct.toFixed(3),
    withinSpec,
  });
}

await fs.writeFile(
  path.join(OUT_DIR, "report.json"),
  JSON.stringify(
    {
      detector: DETECTOR,
      spec: "max corner error <= 2% of image diagonal",
      passCount,
      total: rows.length,
      meanAvgDiagPct: +(totalAvg / rows.length).toFixed(3),
      meanMaxDiagPct: +(totalMax / rows.length).toFixed(3),
      rows,
    },
    null,
    2,
  ),
);

console.log(`Detector: ${DETECTOR}`);
for (const row of rows) {
  console.log(
    `${row.name}: avg=${row.avgPx.toFixed(2)}px (${row.avgDiagPct.toFixed(2)}%), max=${row.maxPx.toFixed(2)}px (${row.maxDiagPct.toFixed(2)}%) ${row.withinSpec ? "PASS" : "FAIL"}`,
  );
}
console.log(
  `Summary: ${passCount}/${rows.length} within spec, mean max error ${(totalMax / rows.length).toFixed(2)}% diag`,
);
console.log(`Artifacts: ${OUT_DIR}`);
