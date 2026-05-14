// Full constrained-search pipeline check: CV gives ranked candidates,
// VLM tie-breaks on the top-K. Mirrors what the deployed browser does.

import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "../..");
const PHOTO_DIR = path.join(ROOT, "Test_Photos");
const OUT_DIR = path.join(__dirname, "out-vlm");

const WORKER_URL =
  process.env.WORKER_URL ?? "https://chesspar-vlm.jamesleoluddy.workers.dev";
const ORIGIN = process.env.ORIGIN ?? "https://jlud7.github.io";
const MODEL = process.env.MODEL ?? "claude-opus-4-7";
const TOP_K = Number(process.env.TOP_K ?? "5");
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

const detection = await import("../../lib/board-detection.ts");
const boardImage = await import("../../lib/board-image.ts");
const occupancy = await import("../../lib/occupancy.ts");
const moveInference = await import("../../lib/move-inference.ts");
const { Chess } = await import("chess.js");

function canvasToB64(canvas) {
  const buf = canvas.toBuffer("image/jpeg", { quality: 0.85 });
  return buf.toString("base64");
}

function extractSan(raw, legalSans) {
  const trimmed = raw.trim();
  if (legalSans.includes(trimmed)) return trimmed;
  const last = trimmed.split(/\r?\n/).map((l) => l.trim()).filter(Boolean).pop() ?? "";
  const sorted = [...legalSans].sort((a, b) => b.length - a.length);
  for (const san of sorted) {
    const re = new RegExp(`(^|[^A-Za-z0-9])${san.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`);
    if (re.test(last)) return san;
  }
  for (const san of sorted) {
    const re = new RegExp(`(^|[^A-Za-z0-9])${san.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`);
    if (re.test(trimmed)) return san;
  }
  return null;
}

async function vlmTieBreak(beforeCanvas, afterCanvas, prevFen, candidatesSan) {
  const beforeB64 = canvasToB64(beforeCanvas);
  const afterB64 = canvasToB64(afterCanvas);
  const prompt = `Identify the chess move played between these two photos.

The photos show a chessboard from above with WHITE PIECES at the BOTTOM and BLACK PIECES at the TOP (standard chess view). Rank 1 is at the bottom, rank 8 at the top. File a is leftmost, file h is rightmost.

IMAGE 1 is the board BEFORE the move.
IMAGE 2 is the board AFTER the move.

Previous FEN (piece placement only): ${prevFen}

Candidate moves (exactly one was played — these are the moves the computer-vision pipeline thinks are most likely):
${candidatesSan.join(", ")}

Find which 1-3 squares look different between IMAGE 1 and IMAGE 2. Identify the piece (color and type) that moved and exactly which squares it moved from and to. Pick the unique candidate whose result produces those changes.

Reply with ONLY the SAN of the move, no preamble.`;
  const body = {
    model: MODEL,
    messages: [
      {
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: beforeB64 } },
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: afterB64 } },
          { type: "text", text: prompt },
        ],
      },
    ],
  };
  if (MODEL.includes("opus")) body.max_tokens = 4000;
  else { body.max_tokens = 64; body.temperature = 0; }
  const resp = await fetch(WORKER_URL.replace(/\/$/, "") + "/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: ORIGIN },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    return { error: `HTTP ${resp.status}: ${(await resp.text()).slice(0, 200)}` };
  }
  const data = await resp.json();
  const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
  return { raw, matched: extractSan(raw, candidatesSan) };
}

await fs.rm(OUT_DIR, { recursive: true, force: true });
await fs.mkdir(OUT_DIR, { recursive: true });

console.log(`Model: ${MODEL}, TOP_K: ${TOP_K}`);
console.log(`Ground truth: ${GROUND_TRUTH.join(" ")}\n`);

const photos = (await fs.readdir(PHOTO_DIR))
  .filter((n) => /\.(jpe?g|png)$/i.test(n))
  .sort();

let savedCorners = null;
let baseline = null;
let lastObservedOcc = null;
let lastCrops = null;
let lastWarped = null;
const game = new Chess();
let correct = 0;
const results = [];

for (let idx = 0; idx < photos.length; idx++) {
  const name = photos[idx];
  const img = await loadOriented(path.join(PHOTO_DIR, name));

  let corners;
  if (idx === 0) {
    const detect = detection.autoDetectBoardCorners(img);
    if (!detect) { console.log(`[${idx}] DETECT FAIL`); break; }
    corners = detect.corners;
    savedCorners = corners;
  } else {
    corners = savedCorners;
  }

  const warped = boardImage.warpBoard(img, corners, 384);
  const crops = boardImage.extractSquareCrops(warped);
  let occResults;
  if (baseline) {
    occResults = occupancy.classifyBoardCalibrated(crops, baseline);
  } else {
    occResults = occupancy.classifyBoard(crops);
  }
  const occStates = occResults.map((c) => c.state);

  if (idx === 0) {
    baseline = occupancy.computeBaseline(crops);
    occResults = occupancy.classifyBoardCalibrated(crops, baseline);
    const states = occResults.map((c) => c.state);
    lastObservedOcc = states.slice();
    lastCrops = crops;
    lastWarped = warped;
    console.log(`[ 0] starting position calibrated`);
    continue;
  }

  const expected = GROUND_TRUTH[idx - 1];
  const confidences = occResults.map((c) => c.confidence);
  const cellDeltas = occupancy.computeCellDeltas(lastCrops, crops);
  const result = moveInference.inferMoveFuzzy(game.fen(), occStates, {
    previousObserved: lastObservedOcc,
    confidences,
    cellDeltas,
  });

  // Take CV top-K candidates and VLM tie-break.
  const topK = result.ranked.slice(0, TOP_K).map((c) => c.move.san);
  const t0 = Date.now();
  let vlmPick = null;
  let vlmRaw = "";
  if (topK.length > 0) {
    try {
      const { matched, raw, error } = await vlmTieBreak(
        lastWarped,
        warped,
        game.fen().split(/\s+/)[0],
        topK,
      );
      if (error) vlmRaw = error;
      else { vlmPick = matched; vlmRaw = raw; }
    } catch (e) {
      vlmRaw = `THROWN: ${e.message ?? e}`;
    }
  }
  const ms = Date.now() - t0;

  let final;
  if (vlmPick) final = vlmPick;
  else final = result.ranked[0]?.move?.san ?? "?";

  const ok = final === expected;
  if (ok) correct += 1;
  game.move(final);
  lastObservedOcc = occStates.slice();
  lastCrops = crops;
  lastWarped = warped;

  console.log(
    `[${idx.toString().padStart(2)}] expected=${expected.padEnd(6)} cv_top=[${topK.join(",")}] vlm=${(vlmPick ?? "?").padEnd(6)} final=${final.padEnd(6)} ${ok ? "✓" : "✗"} (${ms}ms)`,
  );
  if (!ok) console.log(`     vlm raw: ${vlmRaw.slice(0, 200)}`);
  results.push({ idx, expected, topK, vlm: vlmPick, ok, vlmRaw });
}

console.log(`\nAccuracy: ${correct}/${GROUND_TRUTH.length} = ${((correct / GROUND_TRUTH.length) * 100).toFixed(0)}%`);
await fs.writeFile(path.join(OUT_DIR, "report.json"), JSON.stringify(results, null, 2));
