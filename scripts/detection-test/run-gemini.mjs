// Gemini 2.5 Pro variant of the constrained-search pipeline.
// Uses your spec: CV gives candidate list, Gemini picks one.
//
// Requires GEMINI_API_KEY env var (get one free at
// https://aistudio.google.com/apikey ).

import { createCanvas, loadImage } from "canvas";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "../..");
const PHOTO_DIR = path.join(ROOT, "Test_Photos");

const API_KEY = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY;
if (!API_KEY) {
  console.error("Set GEMINI_API_KEY env var first. Get one at https://aistudio.google.com/apikey");
  process.exit(1);
}
const MODEL = process.env.MODEL ?? "gemini-2.5-pro";
const TOP_K = Number(process.env.TOP_K ?? "6");
const ROTATE_QUARTERS = Number(process.env.ROTATE_QUARTERS ?? "1");
const CROP_LEFT = Number(process.env.CROP_LEFT ?? "0.22");
const PHOTO_MAX_DIM = Number(process.env.PHOTO_MAX_DIM ?? "1568");
const CV_MARGIN = Number(process.env.CV_MARGIN ?? "0.3");

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
    .toFormat("jpeg", { quality: 95 })
    .toBuffer();
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

function extractSan(raw, legalSans) {
  const trimmed = raw.trim();
  if (legalSans.includes(trimmed)) return trimmed;
  const m = raw.match(/ANSWER\s*[:=]\s*([A-Za-z0-9+#=\-]+)/i);
  if (m) {
    for (const s of legalSans) if (s === m[1] || s.replace(/[+#]+$/, "") === m[1].replace(/[+#]+$/, "")) return s;
  }
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

async function geminiFromCandidates(beforeB64, afterB64, prevFen, candidatesSan) {
  const prompt = `Identify the chess move played between these two photos.

The photos show a chessboard from above with WHITE PIECES at the BOTTOM and BLACK PIECES at the TOP (standard chess view). Rank 1 is at the bottom, rank 8 at the top. File a is leftmost, file h is rightmost.

IMAGE 1 is the board BEFORE the move; IMAGE 2 is the board AFTER the move.

Previous FEN (piece placement only): ${prevFen}

Candidate moves — exactly ONE of these was played:
${candidatesSan.join(", ")}

Look carefully at IMAGE 2 and determine which candidate's resulting position matches what you see. Pay attention to:
- Which exact rank/file the moved piece ended up on (e.g. f6 vs g6 are different files)
- Piece type (pawn vs knight vs bishop) by shape

End your response with "ANSWER: <san>" on its own line. The SAN must be one of the candidates exactly as written.`;
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(MODEL)}:generateContent?key=${encodeURIComponent(API_KEY)}`;
  const body = {
    contents: [{
      parts: [
        { text: prompt },
        { inline_data: { mime_type: "image/jpeg", data: beforeB64 } },
        { inline_data: { mime_type: "image/jpeg", data: afterB64 } },
      ],
    }],
    generationConfig: { temperature: 0, maxOutputTokens: 2048 },
  };
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) return { error: `HTTP ${resp.status}: ${(await resp.text()).slice(0, 250)}` };
  const data = await resp.json();
  const raw = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  return { raw, matched: extractSan(raw, candidatesSan) };
}

console.log(`Model: ${MODEL}, TOP_K: ${TOP_K}, CV_MARGIN: ${CV_MARGIN}`);
console.log(`Ground truth: ${GROUND_TRUTH.join(" ")}\n`);

const photos = (await fs.readdir(PHOTO_DIR))
  .filter((n) => /\.(jpe?g|png)$/i.test(n))
  .sort();

const cvImgs = [];
const photoB64s = [];
for (const name of photos) {
  cvImgs.push(await loadOriented(path.join(PHOTO_DIR, name)));
  photoB64s.push(await loadPhotoB64(path.join(PHOTO_DIR, name)));
}

const detect = detection.autoDetectBoardCorners(cvImgs[0]);
if (!detect) { console.error("DETECT FAIL"); process.exit(1); }
const corners = detect.corners;

let baseline = null;
{
  const w = boardImage.warpBoard(cvImgs[0], corners, 384);
  const c = boardImage.extractSquareCrops(w);
  baseline = occupancy.computeBaseline(c);
}

const reference = new Chess();
let correct = 0;

for (let i = 0; i < GROUND_TRUTH.length; i++) {
  const expected = GROUND_TRUTH[i];
  const before = cvImgs[i];
  const after = cvImgs[i + 1];
  const fullPrevFen = reference.fen();
  const prevFen = fullPrevFen.split(/\s+/)[0];

  const wBefore = boardImage.warpBoard(before, corners, 384);
  const cBefore = boardImage.extractSquareCrops(wBefore);
  const prevOcc = occupancy.classifyBoardCalibrated(cBefore, baseline).map((c) => c.state);

  const wAfter = boardImage.warpBoard(after, corners, 384);
  const cAfter = boardImage.extractSquareCrops(wAfter);
  const occAfterResults = occupancy.classifyBoardCalibrated(cAfter, baseline);
  const occAfterStates = occAfterResults.map((c) => c.state);
  const confidences = occAfterResults.map((c) => c.confidence);
  const cellDeltas = occupancy.computeCellDeltas(cBefore, cAfter);
  const result = moveInference.inferMoveFuzzy(fullPrevFen, occAfterStates, {
    previousObserved: prevOcc,
    confidences,
    cellDeltas,
  });

  const topK = result.ranked.slice(0, TOP_K).map((c) => c.move.san);
  const topMismatch = result.ranked[0]?.weightedMismatch ?? Infinity;
  const secondMismatch = result.ranked[1]?.weightedMismatch ?? Infinity;
  const margin = secondMismatch - topMismatch;

  const t0 = Date.now();
  let pick = null;
  let raw = "";
  try {
    const r = await geminiFromCandidates(photoB64s[i], photoB64s[i + 1], prevFen, topK);
    if (r.error) raw = r.error;
    else { pick = r.matched; raw = r.raw ?? ""; }
  } catch (e) {
    raw = `THROWN: ${e.message ?? e}`;
  }
  const ms = Date.now() - t0;
  const final = pick ?? topK[0] ?? "?";
  const ok = final === expected;
  if (ok) correct += 1;
  console.log(
    `[${(i + 1).toString().padStart(2)}] expected=${expected.padEnd(6)} cv_top=[${topK.join(",")}] gemini=${(pick ?? "?").padEnd(6)} final=${final.padEnd(6)} ${ok ? "✓" : "✗"} margin=${margin.toFixed(2)} (${ms}ms)`,
  );
  if (!ok) console.log(`     raw: ${raw.slice(0, 300).replace(/\n/g, " | ")}`);
  reference.move(expected);
}

console.log(`\nAccuracy: ${correct}/${GROUND_TRUTH.length} = ${((correct / GROUND_TRUTH.length) * 100).toFixed(0)}%`);
