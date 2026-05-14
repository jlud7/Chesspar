import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import sharp from "sharp";

const ROOT = "/home/user/Chesspar";
const PHOTO_DIR = path.join(ROOT, "Test_Photos");

async function loadOriented(filePath) {
  let pipe = sharp(filePath).rotate().rotate(90);
  const rotatedBuf = await pipe.toBuffer();
  const meta = await sharp(rotatedBuf).metadata();
  const left = Math.round(meta.width * 0.22);
  const buf = await sharp(rotatedBuf).extract({ left, top: 0, width: meta.width - left, height: meta.height }).toFormat("jpeg").toBuffer();
  return loadImage(buf);
}

const seedImg = await loadOriented(path.join(PHOTO_DIR, "IMG_8819.jpeg"));
const seedCanvas = createCanvas(1, 1);
globalThis.document = { createElement: (t) => t === "canvas" ? createCanvas(1, 1) : (() => { throw new Error("nope"); })() };
globalThis.HTMLImageElement = class {};
globalThis.HTMLCanvasElement = class {};
Object.defineProperty(HTMLImageElement, Symbol.hasInstance, { value: (v) => v && v.constructor === seedImg.constructor });
Object.defineProperty(HTMLCanvasElement, Symbol.hasInstance, { value: (v) => v && v.constructor === seedCanvas.constructor });

const detection = await import(path.join(ROOT, "lib/board-detection.ts"));
const boardImage = await import(path.join(ROOT, "lib/board-image.ts"));
const occupancy = await import(path.join(ROOT, "lib/occupancy.ts"));
const moveInference = await import(path.join(ROOT, "lib/move-inference.ts"));
const vlmCellVerify = await import(path.join(ROOT, "lib/vlm-cell-verify.ts"));
const { Chess } = await import("chess.js");

const photos = (await fs.readdir(PHOTO_DIR)).filter(n => /\.(jpe?g|png)$/i.test(n)).sort();
const game = new Chess();
let savedCorners = null;
let baseline = null;
let lastObservedOcc = null;
let lastCrops = null;

for (let idx = 0; idx < photos.length; idx++) {
  const name = photos[idx];
  const img = await loadOriented(path.join(PHOTO_DIR, name));
  let corners;
  if (idx === 0) {
    const det = detection.autoDetectBoardCorners(img);
    if (!det) { console.log(`${name}: NO DETECT`); continue; }
    corners = det.corners;
    savedCorners = corners;
  } else {
    corners = savedCorners;
  }
  const warped = boardImage.warpBoard(img, corners, 384);
  const crops = boardImage.extractSquareCrops(warped);
  const occResults = baseline ? occupancy.classifyBoardCalibrated(crops, baseline) : occupancy.classifyBoard(crops);
  const occStates = occResults.map(c => c.state);
  const confidences = occResults.map(c => c.confidence);

  if (idx === 0) {
    baseline = occupancy.computeBaseline(crops);
    const occRes2 = occupancy.classifyBoardCalibrated(crops, baseline);
    lastObservedOcc = occRes2.map(c => c.state);
    lastCrops = crops;
    console.log(`${name}: baseline`);
    continue;
  }
  const cellDeltas = occupancy.computeCellDeltas(lastCrops, crops);
  const fenBefore = game.fen();
  const result = moveInference.inferMoveFuzzy(fenBefore, occStates, { previousObserved: lastObservedOcc, confidences, cellDeltas });
  if (result.kind === "matched" && result.pick) {
    game.move({ from: result.pick.move.from, to: result.pick.move.to, promotion: result.pick.move.promotion ?? "q" });
    lastObservedOcc = occStates.slice();
    lastCrops = crops;
    const top = result.ranked[0];
    const second = result.ranked[1];
    const margin = top && second ? second.weightedMismatch - top.weightedMismatch : Infinity;
    const topKSans = result.ranked.slice(0, 10).map(c => c.move.san);
    const disputed = vlmCellVerify.findDisputedSquares(fenBefore, topKSans);
    const COMP = 3; const compRanked = result.ranked.filter(c => c.weightedMismatch <= top.weightedMismatch + COMP); const topKFiltered = compRanked.slice(0,10).map(c => c.move.san); const disputed2 = vlmCellVerify.findDisputedSquares(fenBefore, topKFiltered); const cvFullyConfident = top.weightedMismatch <= 5 && disputed2.length === 0 && margin >= 1;
    console.log(`${name}: san=${result.pick.move.san}  topWeighted=${top.weightedMismatch.toFixed(2)}  margin=${margin.toFixed(2)}  disputed=[${disputed2.join(",")}] topK=[${topKSans.slice(0,4).join(",")}]  CVconfident=${cvFullyConfident}`);
  } else {
    console.log(`${name}: ${result.kind}`);
  }
}
