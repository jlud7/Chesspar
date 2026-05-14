// Node-side smoke runner for the board-detection + move-inference pipeline.
// Loads each photo in test_photos/, shims the minimal DOM bits our lib code
// needs (createElement/canvas), runs detection + inference, and writes debug
// PNGs into scripts/detection-test/out/ so we can eyeball results without
// round-tripping through the browser.

import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "../..");
const PHOTO_DIR = path.join(ROOT, "Test_Photos");
const OUT_DIR = path.join(__dirname, "out");

// Match the browser pipeline's preprocessing: EXIF-rotate, then rotate to
// "white at bottom" canonical view, then crop the clock/hand clutter.
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
const occupancy = await import("../../lib/occupancy.ts");
const moveInference = await import("../../lib/move-inference.ts");
const { Chess } = await import("chess.js");

await fs.rm(OUT_DIR, { recursive: true, force: true });
await fs.mkdir(OUT_DIR, { recursive: true });

const photos = (await fs.readdir(PHOTO_DIR))
  .filter((n) => /\.(jpe?g|png)$/i.test(n))
  .sort();

let savedCorners = null;
let baseline = null;
let lastObservedOcc = null;
let lastCrops = null;
const game = new Chess();
const report = [];

for (let idx = 0; idx < photos.length; idx++) {
  const name = photos[idx];
  const stem = name.replace(/\.[^.]+$/, "");
  const img = await loadOriented(path.join(PHOTO_DIR, name));
  const entry = { name, idx };

  let corners;
  if (idx === 0) {
    const t0 = Date.now();
    const detect = detection.autoDetectBoardCorners(img);
    entry.detectMs = Date.now() - t0;
    if (!detect) {
      entry.ok = false;
      entry.reason = "detection failed";
      report.push(entry);
      console.log(`${name}: DETECT FAIL`);
      continue;
    }
    corners = detect.corners;
    savedCorners = corners;
    entry.confidence = +detect.confidence.toFixed(3);
  } else {
    // Reuse the calibration as-is. Per-frame corner refinement runs in the
    // capture flow but introduces frame-to-frame noise that hurts the
    // diff-based matcher on static photos like these.
    corners = savedCorners;
  }

  let warped, crops, occResults, occStates;
  try {
    warped = boardImage.warpBoard(img, corners, 384);
    crops = boardImage.extractSquareCrops(warped);
    if (baseline) {
      occResults = occupancy.classifyBoardCalibrated(crops, baseline);
    } else {
      occResults = occupancy.classifyBoard(crops);
    }
    occStates = occResults.map((c) => c.state);
  } catch (e) {
    entry.ok = false;
    entry.reason = `warp/classify failed: ${e.message}`;
    report.push(entry);
    console.log(`${name}: ${entry.reason}`);
    continue;
  }

  if (idx === 0) {
    baseline = occupancy.computeBaseline(crops);
    // Re-classify with the fresh baseline.
    occResults = occupancy.classifyBoardCalibrated(crops, baseline);
    occStates = occResults.map((c) => c.state);
    lastObservedOcc = occStates.slice();
    lastCrops = crops;
    entry.fen = game.fen();
    entry.ok = true;
    entry.note = "starting position — baseline learned";
  } else {
    const confidences = occResults.map((c) => c.confidence);
    const cellDeltas = occupancy.computeCellDeltas(lastCrops, crops);
    // Top cell deltas for debug — useful for understanding which squares
    // the pixel signal flagged as "really changed" between this frame
    // and the previous one.
    const FILES_AB = "abcdefgh";
    const topDeltas = cellDeltas
      .map((d, i) => ({
        sq: `${FILES_AB[i % 8]}${8 - Math.floor(i / 8)}`,
        d: +d.toFixed(1),
      }))
      .sort((a, b) => b.d - a.d)
      .slice(0, 6);
    entry.topDeltas = topDeltas;
    const result = moveInference.inferMoveFuzzy(game.fen(), occStates, {
      previousObserved: lastObservedOcc,
      confidences,
      cellDeltas,
    });
    if (result.kind === "matched" && result.pick) {
      game.move({
        from: result.pick.move.from,
        to: result.pick.move.to,
        promotion: result.pick.move.promotion ?? "q",
      });
      entry.san = result.pick.move.san;
      entry.ok = true;
      entry.fen = result.pick.updatedFen;
      entry.bestMismatch = result.ranked[0].mismatch;
      entry.bestWeighted = +result.ranked[0].weightedMismatch.toFixed(2);
      lastObservedOcc = occStates.slice();
      lastCrops = crops;
    } else {
      entry.ok = false;
      entry.reason = result.kind;
      entry.top3 = result.ranked.slice(0, 5).map((c) => ({
        san: c.move.san,
        mm: c.mismatch,
        w: +c.weightedMismatch.toFixed(2),
      }));
      entry.diff = result.diff.map(
        (d) => `${d.square}:${d.before[0]}→${d.after[0]}`,
      );
    }
  }

  // Debug outputs.
  await fs.writeFile(
    path.join(OUT_DIR, `${stem}_warp.png`),
    warped.toBuffer("image/png"),
  );

  const ov = createCanvas(warped.width, warped.height);
  const octx = ov.getContext("2d");
  octx.drawImage(warped, 0, 0);
  const cs = warped.width / 8;
  octx.lineWidth = 3;
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const state = occStates[r * 8 + f];
      octx.strokeStyle =
        state === "white"
          ? "rgba(255,255,255,0.95)"
          : state === "black"
            ? "rgba(0,0,0,0.95)"
            : "rgba(0,200,255,0.55)";
      octx.strokeRect(f * cs + 4, r * cs + 4, cs - 8, cs - 8);
    }
  }
  // File / rank labels for debugging.
  octx.font = `${Math.floor(cs / 6)}px sans-serif`;
  octx.fillStyle = "rgba(255,255,255,0.95)";
  octx.strokeStyle = "rgba(0,0,0,0.95)";
  octx.lineWidth = 3;
  for (let f = 0; f < 8; f++) {
    const ch = "abcdefgh"[f];
    octx.strokeText(ch, f * cs + 6, 18);
    octx.fillText(ch, f * cs + 6, 18);
  }
  for (let r = 0; r < 8; r++) {
    const rk = 8 - r;
    octx.strokeText(String(rk), 4, r * cs + cs / 2);
    octx.fillText(String(rk), 4, r * cs + cs / 2);
  }
  await fs.writeFile(
    path.join(OUT_DIR, `${stem}_occ.png`),
    ov.toBuffer("image/png"),
  );

  // Annotated original image with corners.
  {
    const c = createCanvas(img.width, img.height);
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    ctx.strokeStyle = "rgba(16,185,129,0.95)";
    ctx.lineWidth = Math.max(4, img.width / 250);
    ctx.beginPath();
    corners.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.closePath();
    ctx.stroke();
    const labels = ["a8", "h8", "h1", "a1"];
    ctx.font = `${Math.max(20, img.width / 30)}px sans-serif`;
    corners.forEach((p, i) => {
      ctx.fillStyle = "rgba(16,185,129,1)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(10, img.width / 80), 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "white";
      ctx.strokeStyle = "black";
      ctx.lineWidth = 4;
      ctx.strokeText(labels[i], p.x + 12, p.y - 12);
      ctx.fillText(labels[i], p.x + 12, p.y - 12);
    });
    await fs.writeFile(
      path.join(OUT_DIR, `${stem}_annot.png`),
      c.toBuffer("image/png"),
    );
  }

  report.push(entry);
  const status = entry.ok
    ? entry.san
      ? `MATCH ${entry.san}`
      : entry.note ?? "ok"
    : `FAIL: ${entry.reason}`;
  console.log(`[${idx.toString().padStart(2, "0")}] ${name}: ${status}`);
}

await fs.writeFile(
  path.join(OUT_DIR, "report.json"),
  JSON.stringify(report, null, 2),
);
console.log(`\nReport at ${path.join(OUT_DIR, "report.json")}`);
