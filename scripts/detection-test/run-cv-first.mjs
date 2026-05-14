// CV-first hybrid: trust CV top-1 when its mismatch lead over top-2 is clear,
// VLM tie-breaks only when the CV margin is ambiguous.

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
// CV top-1 is trusted when its weightedMismatch is at least this much lower
// than top-2's weightedMismatch. The CV ranking is reasonably accurate on
// piece-color terms; the VLM is brought in only when the gap is small.
const CV_MARGIN = Number(process.env.CV_MARGIN ?? "0.5");

const GROUND_TRUTH = [
  "e4", "e5", "Nf3", "Nc6", "Nc3", "d6", "b3", "Be7",
  "Ba3", "Nf6", "Nd5", "Nxd5", "exd5", "b6",
];

// Set DEBUG_MOVES=8,10 (or DEBUG_MOVES=all) to dump per-square occupancy +
// delta data for the listed moves. Use this to diagnose which square the
// classifier is misreading on stubborn misses.
const DEBUG_MOVES = (() => {
  const raw = (process.env.DEBUG_MOVES ?? "").trim();
  if (!raw) return new Set();
  if (raw === "all" || raw === "*") return "all";
  return new Set(raw.split(/[,\s]+/).map((s) => Number(s)).filter((n) => !Number.isNaN(n)));
})();

const FILES = "abcdefgh";
function idxToSan(idx) {
  const file = FILES[idx % 8];
  const rank = 8 - Math.floor(idx / 8);
  return `${file}${rank}`;
}
function sanToIdx(san) {
  const file = FILES.indexOf(san[0]);
  const rank = Number(san[1]);
  if (file < 0 || !rank) return -1;
  return (8 - rank) * 8 + file;
}
function moveSquares(prevFen, san) {
  const sim = new Chess(prevFen);
  const m = sim.move(san);
  if (!m) return null;
  return { from: m.from, to: m.to, piece: m.piece, color: m.color };
}

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
  const prompt = `You're choosing between specific chess moves by looking at two photos.

The photos show a chessboard from above with WHITE PIECES at the BOTTOM and BLACK PIECES at the TOP. Rank 1 is at the bottom, rank 8 at the top. File a is leftmost, file h is rightmost.

IMAGE 1: before the move. IMAGE 2: after the move.

Previous position (piece placement): ${prevFen}

Possible moves (exactly ONE of these was played):
${candidatesSan.join(", ")}

Step 1: For each candidate move, write a one-line note of what it does ("Be7 = black bishop from f8 to e7; expect f8 empty, e7 black bishop").
Step 2: Look at IMAGE 2 and check each candidate's expected result. Eliminate those whose expected square contents disagree with what you see.
Step 3: Output the SAN of the surviving move.

End your response with "ANSWER: <san>" on its own line. The SAN must be one of the candidates exactly as written.`;
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
  else { body.max_tokens = 512; body.temperature = 0; }
  const resp = await fetch(WORKER_URL.replace(/\/$/, "") + "/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: ORIGIN },
    body: JSON.stringify(body),
  });
  if (!resp.ok) return { error: `HTTP ${resp.status}: ${(await resp.text()).slice(0, 200)}` };
  const data = await resp.json();
  const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
  // Prefer ANSWER: line
  const m = raw.match(/ANSWER\s*[:=]\s*([A-Za-z0-9+#=\-]+)/i);
  if (m) {
    for (const s of candidatesSan) if (s === m[1] || s.replace(/[+#]+$/, "") === m[1].replace(/[+#]+$/, "")) return { matched: s, raw };
  }
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
const detail = [];

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
  const cvConfident = topK.length > 0 && margin >= CV_MARGIN;

  let final;
  let pickedBy;
  let vlmRaw = "";
  if (cvConfident) {
    final = topK[0];
    pickedBy = `CV top-1 (margin ${margin.toFixed(2)})`;
  } else {
    try {
      const r = await vlmFromCandidates(photoB64s[i], photoB64s[i + 1], prevFen, topK);
      if (r.matched) {
        final = r.matched;
        pickedBy = `VLM (margin ${margin.toFixed(2)})`;
        vlmRaw = r.raw;
      } else {
        final = topK[0] ?? "?";
        pickedBy = `VLM-failed-fallback-CV (margin ${margin.toFixed(2)})`;
        vlmRaw = r.raw ?? r.error ?? "";
      }
    } catch (e) {
      final = topK[0] ?? "?";
      pickedBy = `VLM-threw-fallback-CV (margin ${margin.toFixed(2)})`;
      vlmRaw = `${e.message ?? e}`;
    }
  }

  const ok = final === expected;
  if (ok) correct += 1;
  console.log(
    `[${(i + 1).toString().padStart(2)}] expected=${expected.padEnd(6)} cv_top=[${topK.join(",")}] ${pickedBy} → ${final.padEnd(6)} ${ok ? "✓" : "✗"}`,
  );
  if (!ok && vlmRaw) console.log(`     raw: ${vlmRaw.slice(0, 300).replace(/\n/g, " | ")}`);

  const moveNum = i + 1;
  const shouldDebug =
    DEBUG_MOVES === "all" ||
    (DEBUG_MOVES instanceof Set && DEBUG_MOVES.has(moveNum)) ||
    (!ok && DEBUG_MOVES instanceof Set && DEBUG_MOVES.size === 0 && process.env.DEBUG_MISSES === "1");
  if (shouldDebug) {
    const expectedSqs = moveSquares(fullPrevFen, expected);
    const wrongSqs = !ok ? moveSquares(fullPrevFen, final) : null;
    const interesting = new Map(); // sanLabel -> { reason }
    function add(sq, reason) {
      if (!sq) return;
      const cur = interesting.get(sq);
      interesting.set(sq, cur ? `${cur}+${reason}` : reason);
    }
    if (expectedSqs) {
      add(expectedSqs.from, `EXP-from(${expectedSqs.color}${expectedSqs.piece})`);
      add(expectedSqs.to, `EXP-to(${expectedSqs.color}${expectedSqs.piece})`);
    }
    if (wrongSqs) {
      add(wrongSqs.from, `WRONG-from`);
      add(wrongSqs.to, `WRONG-to`);
    }
    console.log(`     --- DEBUG move ${moveNum} (${expected}${ok ? "" : ` → got ${final}`}) ---`);
    for (const [sq, why] of interesting) {
      const idx = sanToIdx(sq);
      const before = prevOcc[idx];
      const after = occAfterStates[idx];
      const conf = confidences[idx].toFixed(2);
      const delta = cellDeltas[idx].toFixed(1);
      console.log(
        `       ${sq.padEnd(2)} ${why.padEnd(28)} before=${before.padEnd(5)} after=${after.padEnd(5)} conf=${conf} delta=${delta}`,
      );
    }
    const top3 = result.ranked.slice(0, 3).map((c) =>
      `${c.move.san}(mm=${c.mismatch} w=${c.weightedMismatch.toFixed(2)})`,
    );
    console.log(`       scores: ${top3.join("  ")}`);
  }

  detail.push({ idx: i + 1, expected, topK, pickedBy, final, ok });
  reference.move(expected);
}

console.log(`\nAccuracy: ${correct}/${GROUND_TRUTH.length} = ${((correct / GROUND_TRUTH.length) * 100).toFixed(0)}%`);
