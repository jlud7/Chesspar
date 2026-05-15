import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";
import { Chess } from "chess.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "../..");
const PHOTO_DIR = path.join(ROOT, "Test_Photos");

const ROTATE_QUARTERS = Number(process.env.ROTATE_QUARTERS ?? "1");
const CROP_LEFT = Number(process.env.CROP_LEFT ?? "0.22");

const GROUND_TRUTH = [
  "e4", "e5", "Nf3", "Nc6", "Nc3", "d6", "b3", "Be7",
  "Ba3", "Nf6", "Nd5", "Nxd5", "exd5", "b6",
];

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

const photos = (await fs.readdir(PHOTO_DIR))
  .filter((n) => /\.(jpe?g|png)$/i.test(n))
  .sort();

const cvImgs = [];
for (const name of photos) {
  cvImgs.push(await loadOriented(path.join(PHOTO_DIR, name)));
}

const detectors = [
  ["current", detection.autoDetectBoardCorners],
  ["legacy", detection.autoDetectBoardCornersLegacy],
];

for (const [label, detectFn] of detectors) {
  const detect = detectFn(cvImgs[0]);
  if (!detect) {
    console.log(`\n[${label}] DETECT FAIL`);
    continue;
  }
  const corners = detect.corners;
  const seedWarp = boardImage.warpBoard(cvImgs[0], corners, 384);
  const seedCrops = boardImage.extractSquareCrops(seedWarp);
  const baseline = occupancy.computeBaseline(seedCrops);

  const game = new Chess();
  let matchedCorrect = 0;
  let top1Correct = 0;

  console.log(`\n[${label}]`);
  for (let i = 0; i < GROUND_TRUTH.length; i++) {
    const expected = GROUND_TRUTH[i];
    const before = cvImgs[i];
    const after = cvImgs[i + 1];

    const wBefore = boardImage.warpBoard(before, corners, 384);
    const cBefore = boardImage.extractSquareCrops(wBefore);
    const prevOcc = occupancy
      .classifyBoardCalibrated(cBefore, baseline)
      .map((c) => c.state);

    const wAfter = boardImage.warpBoard(after, corners, 384);
    const cAfter = boardImage.extractSquareCrops(wAfter);
    const occAfterResults = occupancy.classifyBoardCalibrated(cAfter, baseline);
    const occAfterStates = occAfterResults.map((c) => c.state);
    const confidences = occAfterResults.map((c) => c.confidence);
    const cellDeltas = occupancy.computeCellDeltas(cBefore, cAfter);

    const result = moveInference.inferMoveFuzzy(game.fen(), occAfterStates, {
      previousObserved: prevOcc,
      confidences,
      cellDeltas,
    });

    const matchedSan = result.kind === "matched" && result.pick
      ? result.pick.move.san
      : null;
    const top1San = result.ranked[0]?.move.san ?? null;
    if (matchedSan === expected) matchedCorrect++;
    if (top1San === expected) top1Correct++;

    console.log(
      `[${(i + 1).toString().padStart(2)}] expected=${expected.padEnd(6)} matched=${(matchedSan ?? "-").padEnd(6)} top1=${(top1San ?? "-").padEnd(6)} kind=${result.kind}`,
    );

    if (result.kind === "matched" && result.pick) {
      game.move({
        from: result.pick.move.from,
        to: result.pick.move.to,
        promotion: result.pick.move.promotion ?? "q",
      });
    }
  }

  console.log(
    `Matched accuracy: ${matchedCorrect}/${GROUND_TRUTH.length} = ${((matchedCorrect / GROUND_TRUTH.length) * 100).toFixed(0)}%`,
  );
  console.log(
    `Top-1 accuracy:   ${top1Correct}/${GROUND_TRUTH.length} = ${((top1Correct / GROUND_TRUTH.length) * 100).toFixed(0)}%`,
  );
}
