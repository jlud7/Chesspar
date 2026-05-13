// Render a side-by-side debug grid: warp + per-cell crop + predicted class
// + chess.js ground-truth for the photo. Helps see exactly which cells are
// being misclassified.

import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";
import { Chess } from "chess.js";

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

const photoName = process.argv[2] ?? "IMG_8819.jpeg";
const moves = process.argv[3] ?? ""; // optional space-separated SAN moves to apply for ground-truth

const img = await loadOriented(path.join(PHOTO_DIR, photoName));
const detect = detection.autoDetectBoardCorners(img);
if (!detect) throw new Error("detection failed");
const corners = detect.corners;
const warped = boardImage.warpBoard(img, corners, 480);
const crops = boardImage.extractSquareCrops(warped);

// Use heuristic baseline-less classifier first to learn baseline.
const heuristic = occupancy.classifyBoard(crops);
const baseline = occupancy.computeBaseline(crops);
const calibrated = occupancy.classifyBoardCalibrated(crops, baseline);

console.log("Baseline signatures:");
for (const [name, sig] of Object.entries(baseline)) {
  console.log(
    `  ${name}: L=${sig.meanL.toFixed(0)} R=${sig.meanR.toFixed(0)} G=${sig.meanG.toFixed(0)} B=${sig.meanB.toFixed(0)} std=${sig.std.toFixed(1)} sat=${sig.saturation.toFixed(1)}`,
  );
}
// Dump per-cell stats for the middle rows (rank 3-6 = indexes 16..47).
const FILES = "abcdefgh";
console.log("\nPer-cell stats for empty ranks (3-6):");
for (let i = 16; i < 48; i++) {
  const sq = `${FILES[i % 8]}${8 - Math.floor(i / 8)}`;
  const s = occupancy.computeSquareStats(crops[i]);
  const isLight = occupancy.isLightSquare(i);
  const got = calibrated[i].state;
  console.log(
    `  ${sq} (${isLight ? "light" : "dark "}): L=${s.meanL.toFixed(0)} std=${s.std.toFixed(1)} sat=${s.saturation.toFixed(0)} -> ${got}`,
  );
}

const game = new Chess();
for (const m of moves.split(/\s+/).filter(Boolean)) {
  try {
    game.move(m);
  } catch {
    console.warn("invalid move", m);
  }
}
const expectedOcc = (() => {
  const board = game.board();
  const out = [];
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const cell = board[r][f];
      if (!cell) out.push("empty");
      else out.push(cell.color === "w" ? "white" : "black");
    }
  }
  return out;
})();

// Render a sheet showing each cell, its expected class, classifier output.
const COLS = 8;
const TILE = 80;
const PAD = 4;
const HEADER = 24;
const ROWS = 8;
const sheet = createCanvas(COLS * (TILE + PAD), ROWS * (TILE + HEADER + PAD));
const ctx = sheet.getContext("2d");
ctx.fillStyle = "#111";
ctx.fillRect(0, 0, sheet.width, sheet.height);
ctx.font = "11px sans-serif";

const files = "abcdefgh";
for (let r = 0; r < 8; r++) {
  for (let f = 0; f < 8; f++) {
    const idx = r * 8 + f;
    const sq = `${files[f]}${8 - r}`;
    const x = f * (TILE + PAD);
    const y = r * (TILE + HEADER + PAD);
    ctx.drawImage(crops[idx], x, y + HEADER, TILE, TILE);
    const exp = expectedOcc[idx];
    const got = calibrated[idx].state;
    const ok = exp === got;
    ctx.fillStyle = ok ? "#0a0" : "#a00";
    ctx.fillRect(x, y, TILE, HEADER);
    ctx.fillStyle = "white";
    ctx.fillText(
      `${sq} exp=${exp[0]} got=${got[0]} c=${calibrated[idx].confidence.toFixed(2)}`,
      x + 3,
      y + 15,
    );
  }
}
const outPath = path.join(
  OUT_DIR,
  `${photoName.replace(/\.[^.]+$/, "")}_cells.png`,
);
await fs.writeFile(outPath, sheet.toBuffer("image/png"));
console.log("wrote", outPath);

// Also count mismatches.
let wrong = 0;
const wrongList = [];
for (let i = 0; i < 64; i++) {
  if (expectedOcc[i] !== calibrated[i].state) {
    wrong++;
    const sq = `${files[i % 8]}${8 - Math.floor(i / 8)}`;
    wrongList.push(
      `${sq}: expected=${expectedOcc[i]} got=${calibrated[i].state} c=${calibrated[i].confidence.toFixed(2)}`,
    );
  }
}
console.log(`${wrong}/64 mismatches`);
for (const w of wrongList) console.log("  ", w);
