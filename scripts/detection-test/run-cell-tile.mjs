// Cell-tile VLM verifier offline runner.
//
// Pipeline:
//   1. Warp + classify per usual (lib/board-image + lib/occupancy).
//   2. Get top-K legal candidates via inferMoveFuzzy.
//   3. For each move, send only the disputed cells (squares where the
//      top-K candidates disagree on piece type) to the VLM as small
//      rectified tile crops with the target square outlined.
//   4. VLM identifies the piece on each highlighted square; we pick the
//      candidate whose after-state matches.
//
// Run:
//   node scripts/detection-test/run-cell-tile.mjs
//   WORKER_URL=https://... MODEL=claude-opus-4-7 node scripts/detection-test/run-cell-tile.mjs

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
// CV-confidence margin: if CV's top-1 weightedMismatch is at least this much
// lower than top-2 AND the top-K candidates agree on after-state (no
// disputed squares), skip the VLM call entirely.
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
const cellVerify = await import("../../lib/vlm-cell-verify.ts");
const { Chess } = await import("chess.js");

function canvasToBase64(canvas) {
  const url = canvas.toDataURL("image/jpeg", 0.9);
  const comma = url.indexOf(",");
  return comma >= 0 ? url.slice(comma + 1) : url;
}

// Free-function dispatcher to the worker proxy. Mirrors what
// verifyByCellTiles does internally for the browser path, but routes
// through the same /verify proxy as the existing offline scripts so we
// don't need to plumb keys into Node.
async function cellTileViaProxy(afterWarped, prevFen, candidatesSan) {
  const disputed = cellVerify.findDisputedSquares(prevFen, candidatesSan);
  if (disputed.length === 0) {
    return { matched: candidatesSan[0], raw: "(no disputed squares)", disputed };
  }
  const { oriented: afterOriented, rotationDeg } =
    boardImage.ensureWhiteAtBottom(afterWarped);
  if (rotationDeg !== 0) {
    console.log(`     [orientation] rotated rectified board by ${rotationDeg}° before VLM call`);
  }
  const tiles = disputed.map((sq) => ({
    square: sq,
    canvas: cellVerify.renderTile(afterOriented, sq),
  }));

  const candidateSummary = candidatesSan
    .map((san, i) => {
      const sim = new Chess(prevFen);
      let m;
      try { m = sim.move(san); } catch { /* */ }
      if (!m) return `  ${i + 1}. ${san}`;
      const color = m.color === "w" ? "white" : "black";
      const piece = ({ p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" })[m.piece] ?? m.piece;
      const cap = m.captured ? ` (captures ${m.captured})` : "";
      return `  ${i + 1}. ${san}  → ${color} ${piece} ${m.from}→${m.to}${cap}; expect ${m.to}=${color}-${piece}, ${m.from}=empty`;
    })
    .join("\n");

  const prompt = `You are identifying a chess move by checking specific squares on the AFTER-move board.

PREVIOUS POSITION (piece placement FEN): ${prevFen}

I will show you ${disputed.length} close-up tile${disputed.length === 1 ? "" : "s"}, each a small region of the AFTER-the-move board with ONE square OUTLINED IN RED. The red-outlined square is the one you must identify. The other squares in the tile are just context (immediate neighbours).

Tiles, in order: ${disputed.join(", ")}.

CANDIDATE MOVES (exactly ONE was played, ranked best-first by the upstream classifier):
${candidateSummary}

Your task:
  Step 1. For each tile, look ONLY at the red-outlined square and write a one-line observation in the form
            "${disputed[0]}: <empty | white-pawn | white-knight | white-bishop | white-rook | white-queen | white-king | black-pawn | black-knight | black-bishop | black-rook | black-queen | black-king>"
          If you cannot tell the piece type but can tell the colour, write "white-piece" or "black-piece". If a piece's TOP extends above the red square from an adjacent rank (3D piece-top bleed), do not count it — only what is *based* on the red square counts.

  Step 2. Cross-reference observations with each candidate's predicted after-state. Eliminate candidates that disagree.

  Step 3. Output the SAN of the surviving move.

Format your final line EXACTLY as:
ANSWER: <san>

The SAN must be one of the candidates above, written exactly as listed.`;

  const content = [];
  for (const tile of tiles) {
    content.push({
      type: "image",
      source: {
        type: "base64",
        media_type: "image/jpeg",
        data: canvasToBase64(tile.canvas),
      },
    });
  }
  content.push({ type: "text", text: prompt });

  const body = {
    model: MODEL,
    messages: [{ role: "user", content }],
  };
  if (MODEL.includes("opus")) body.max_tokens = 4000;
  else { body.max_tokens = 512; body.temperature = 0; }

  const resp = await fetch(WORKER_URL.replace(/\/$/, "") + "/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: ORIGIN },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    return {
      error: `HTTP ${resp.status}: ${(await resp.text()).slice(0, 200)}`,
      disputed,
    };
  }
  const data = await resp.json();
  const raw = data.content?.find((c) => c.type === "text")?.text ?? "";

  // Parse ANSWER line, fall back to last-line / longest-SAN match.
  const m = raw.match(/ANSWER\s*[:=]\s*([A-Za-z0-9+#=\-]+)/i);
  if (m) {
    const guess = m[1];
    for (const san of candidatesSan) {
      if (san === guess || san.replace(/[+#]+$/, "") === guess.replace(/[+#]+$/, "")) {
        return { matched: san, raw, disputed };
      }
    }
  }
  const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const last = lines[lines.length - 1] ?? "";
  const sorted = [...candidatesSan].sort((a, b) => b.length - a.length);
  for (const san of sorted) {
    const re = new RegExp(`(^|[^A-Za-z0-9])${san.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`);
    if (re.test(last)) return { matched: san, raw, disputed };
  }
  for (const san of sorted) {
    const re = new RegExp(`(^|[^A-Za-z0-9])${san.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`);
    if (re.test(raw)) return { matched: san, raw, disputed };
  }
  return { raw, disputed };
}

console.log(`Model: ${MODEL}, TOP_K: ${TOP_K}, CV_MARGIN: ${CV_MARGIN}`);
console.log(`Ground truth: ${GROUND_TRUTH.join(" ")}\n`);

const photos = (await fs.readdir(PHOTO_DIR))
  .filter((n) => /\.(jpe?g|png)$/i.test(n))
  .sort();

const cvImgs = [];
for (const name of photos) {
  cvImgs.push(await loadOriented(path.join(PHOTO_DIR, name)));
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
  const top = result.ranked[0];
  const second = result.ranked[1];
  const margin = top && second ? second.weightedMismatch - top.weightedMismatch : Infinity;

  const disputed = cellVerify.findDisputedSquares(fullPrevFen, topK);
  // CV fully confident only when NO disputed squares AND margin is wide.
  const cvFullyConfident =
    result.kind === "matched" &&
    result.pick != null &&
    disputed.length === 0 &&
    margin >= CV_MARGIN;

  let final;
  let pickedBy;
  let raw = "";
  if (cvFullyConfident) {
    final = topK[0];
    pickedBy = `CV top-1 (margin ${margin.toFixed(2)}, disputed=0)`;
  } else if (topK.length > 0) {
    try {
      const r = await cellTileViaProxy(wAfter, prevFen, topK);
      if (r.matched) {
        final = r.matched;
        pickedBy = `cell-tile VLM (margin ${margin.toFixed(2)}, disputed=${disputed.length}: ${disputed.join(",")})`;
        raw = r.raw ?? "";
      } else {
        final = topK[0] ?? "?";
        pickedBy = `cell-VLM failed → CV fallback (disputed=${disputed.length})`;
        raw = r.raw ?? r.error ?? "";
      }
    } catch (e) {
      final = topK[0] ?? "?";
      pickedBy = `cell-VLM threw → CV fallback`;
      raw = `${e.message ?? e}`;
    }
  } else {
    final = "?";
    pickedBy = "no candidates";
  }

  const ok = final === expected;
  if (ok) correct += 1;
  console.log(
    `[${(i + 1).toString().padStart(2)}] expected=${expected.padEnd(6)} cv_top=[${topK.join(",")}] ${pickedBy} → ${final.padEnd(6)} ${ok ? "✓" : "✗"}`,
  );
  if (!ok && raw) console.log(`     raw: ${raw.slice(0, 400).replace(/\n/g, " | ")}`);
  detail.push({ idx: i + 1, expected, topK, pickedBy, final, ok, raw });
  reference.move(expected);
}

console.log(`\nAccuracy: ${correct}/${GROUND_TRUTH.length} = ${((correct / GROUND_TRUTH.length) * 100).toFixed(0)}%`);

const OUT_DIR = path.join(__dirname, "out-cell-tile");
await fs.mkdir(OUT_DIR, { recursive: true });
await fs.writeFile(
  path.join(OUT_DIR, "report.json"),
  JSON.stringify(detail, null, 2),
);
console.log(`\nReport: ${path.relative(ROOT, path.join(OUT_DIR, "report.json"))}`);
