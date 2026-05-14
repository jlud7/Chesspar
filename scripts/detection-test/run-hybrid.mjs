// Hybrid: CV identifies top-K legal candidates from occupancy/delta;
// VLM picks from that short list using the FULL original photos (which
// have way more detail than 384px warps).

import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "../..");
const PHOTO_DIR = path.join(ROOT, "Test_Photos");

const WORKER_URL =
  process.env.WORKER_URL ?? "https://chesspar-vlm.jamesleoluddy.workers.dev";
const ORIGIN = process.env.ORIGIN ?? "https://jlud7.github.io";
const MODEL = process.env.MODEL ?? "claude-opus-4-7";
const TOP_K = Number(process.env.TOP_K ?? "6");
const ROTATE_QUARTERS = Number(process.env.ROTATE_QUARTERS ?? "1");
const CROP_LEFT = Number(process.env.CROP_LEFT ?? "0.22");
const PHOTO_MAX_DIM = Number(process.env.PHOTO_MAX_DIM ?? "1568");

const GROUND_TRUTH = [
  "e4", "e5", "Nf3", "Nc6", "Nc3", "d6", "b3", "Be7",
  "Ba3", "Nf6", "Nd5", "Nxd5", "exd5", "b6",
];

async function loadOriented(filePath, forCv = false) {
  let pipe = sharp(filePath).rotate();
  if (ROTATE_QUARTERS) pipe = pipe.rotate(ROTATE_QUARTERS * 90);
  const rotatedBuf = await pipe.toBuffer();
  const meta = await sharp(rotatedBuf).metadata();
  const W = meta.width;
  const H = meta.height;
  const left = Math.round(W * CROP_LEFT);
  let p = sharp(rotatedBuf).extract({ left, top: 0, width: W - left, height: H });
  if (!forCv) p = p.resize(PHOTO_MAX_DIM, PHOTO_MAX_DIM, { fit: "inside", withoutEnlargement: true });
  const buf = await p.toFormat("jpeg", { quality: forCv ? 95 : 88 }).toBuffer();
  return await loadImage(buf);
}

async function loadPhotoB64(filePath) {
  let pipe = sharp(filePath).rotate();
  if (ROTATE_QUARTERS) pipe = pipe.rotate(ROTATE_QUARTERS * 90);
  const rotatedBuf = await pipe.toBuffer();
  const meta = await sharp(rotatedBuf).metadata();
  const W = meta.width;
  const H = meta.height;
  const left = Math.round(W * CROP_LEFT);
  const buf = await sharp(rotatedBuf)
    .extract({ left, top: 0, width: W - left, height: H })
    .resize(PHOTO_MAX_DIM, PHOTO_MAX_DIM, { fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 88 })
    .toBuffer();
  return buf.toString("base64");
}

const seedImg = await loadOriented(path.join(PHOTO_DIR, "IMG_8819.jpeg"), true);
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

async function vlmFromCandidates(beforeB64, afterB64, prevFen, candidatesSan) {
  const prompt = `Identify the chess move played between these two photos.

The photos show a chessboard from above with WHITE PIECES at the BOTTOM and BLACK PIECES at the TOP (standard chess view). Rank 1 is at the bottom, rank 8 at the top. File a is leftmost, file h is rightmost.

IMAGE 1 is the board BEFORE the move; IMAGE 2 is the board AFTER the move.

Previous FEN (piece placement only): ${prevFen}

Candidate moves — exactly one was played. These are the computer-vision pipeline's top guesses:
${candidatesSan.join(", ")}

Look carefully at which 1-3 squares changed between IMAGE 1 and IMAGE 2. Identify the piece (color and type) and exactly which squares it moved from and to. Then pick the unique candidate above whose result produces those changes.

Reply with ONLY the SAN of the move, no preamble.`;
  const body = {
    model: MODEL,
    messages: [{
      role: "user",
      content: [
        { type: "image", source: { type: "base64", media_type: "image/jpeg", data: beforeB64 } },
        { type: "image", source: { type: "base64", media_type: "image/jpeg", data: afterB64 } },
        { type: "text", text: prompt },
      ],
    }],
  };
  if (MODEL.includes("opus")) body.max_tokens = 4000;
  else { body.max_tokens = 64; body.temperature = 0; }
  const resp = await fetch(WORKER_URL.replace(/\/$/, "") + "/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: ORIGIN },
    body: JSON.stringify(body),
  });
  if (!resp.ok) return { error: `HTTP ${resp.status}: ${(await resp.text()).slice(0, 200)}` };
  const data = await resp.json();
  const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
  return { raw, matched: extractSan(raw, candidatesSan) };
}

console.log(`Model: ${MODEL}, TOP_K: ${TOP_K}, photo_max_dim: ${PHOTO_MAX_DIM}`);
console.log(`Ground truth: ${GROUND_TRUTH.join(" ")}\n`);

const photos = (await fs.readdir(PHOTO_DIR))
  .filter((n) => /\.(jpe?g|png)$/i.test(n))
  .sort();

// Pre-load CV images and full-photo b64s.
const cvImgs = [];
const photoB64s = [];
for (const name of photos) {
  cvImgs.push(await loadOriented(path.join(PHOTO_DIR, name), true));
  photoB64s.push(await loadPhotoB64(path.join(PHOTO_DIR, name)));
}

// CV calibration on photo 0.
const detect = detection.autoDetectBoardCorners(cvImgs[0]);
if (!detect) { console.error("DETECT FAIL"); process.exit(1); }
const corners = detect.corners;

let baseline = null;
let lastObservedOcc = null;
let lastCrops = null;
// Bootstrap baseline + initial occupancy.
{
  const warped0 = boardImage.warpBoard(cvImgs[0], corners, 384);
  const crops0 = boardImage.extractSquareCrops(warped0);
  baseline = occupancy.computeBaseline(crops0);
  const occ0 = occupancy.classifyBoardCalibrated(crops0, baseline).map((c) => c.state);
  lastObservedOcc = occ0;
  lastCrops = crops0;
}

const reference = new Chess();
let correct = 0;

// EACH move is tested INDEPENDENTLY with the ground-truth-correct previous state
// (avoids compounding errors).
for (let i = 0; i < GROUND_TRUTH.length; i++) {
  const expected = GROUND_TRUTH[i];
  const before = cvImgs[i];
  const after = cvImgs[i + 1];
  const fullPrevFen = reference.fen();
  const prevFen = fullPrevFen.split(/\s+/)[0]; // piece-placement only for VLM prompt

  // CV: build prev-state crops from the before image so the diff is clean
  let prevCrops, prevOcc;
  {
    const w = boardImage.warpBoard(before, corners, 384);
    prevCrops = boardImage.extractSquareCrops(w);
    prevOcc = occupancy.classifyBoardCalibrated(prevCrops, baseline).map((c) => c.state);
  }
  const warpedAfter = boardImage.warpBoard(after, corners, 384);
  const cropsAfter = boardImage.extractSquareCrops(warpedAfter);
  const occResultsAfter = occupancy.classifyBoardCalibrated(cropsAfter, baseline);
  const occStatesAfter = occResultsAfter.map((c) => c.state);
  const confidences = occResultsAfter.map((c) => c.confidence);
  const cellDeltas = occupancy.computeCellDeltas(prevCrops, cropsAfter);
  const result = moveInference.inferMoveFuzzy(fullPrevFen, occStatesAfter, {
    previousObserved: prevOcc,
    confidences,
    cellDeltas,
  });
  const topK = result.ranked.slice(0, TOP_K).map((c) => c.move.san);

  const t0 = Date.now();
  let pick = null;
  let vlmRaw = "";
  if (topK.length > 0) {
    try {
      const r = await vlmFromCandidates(photoB64s[i], photoB64s[i + 1], prevFen, topK);
      if (r.error) vlmRaw = r.error;
      else { pick = r.matched; vlmRaw = r.raw; }
    } catch (e) {
      vlmRaw = `THROWN: ${e.message ?? e}`;
    }
  }
  const ms = Date.now() - t0;
  const final = pick ?? topK[0] ?? "?";
  const ok = final === expected;
  if (ok) correct += 1;
  console.log(
    `[${(i + 1).toString().padStart(2)}] expected=${expected.padEnd(6)} cv_top=[${topK.join(",")}] vlm=${(pick ?? "?").padEnd(6)} final=${final.padEnd(6)} ${ok ? "✓" : "✗"} (${ms}ms)`,
  );
  if (!ok) console.log(`     raw: ${vlmRaw.slice(0, 200)}`);
  // Advance ground-truth state regardless of pick.
  reference.move(expected);
}

console.log(`\nAccuracy: ${correct}/${GROUND_TRUTH.length} = ${((correct / GROUND_TRUTH.length) * 100).toFixed(0)}%`);
