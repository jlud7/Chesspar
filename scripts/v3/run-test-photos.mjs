// Offline accuracy harness for the new v2 Chesspar pipeline.
// Loads Test_Photos/, runs calibrate (image 0) + identify-move (each pair),
// reports per-move correctness vs the known ground-truth game.
//
// Run: node scripts/v3/run-test-photos.mjs
//
// Optional env vars:
//   WORKER_URL     — Cloudflare worker root (default chesspar-vlm.jamesleoluddy.workers.dev)
//   ROTATE_QUARTERS — quarter-turns applied via sharp before sending (default 1)
//   CROP_LEFT      — fraction of width to crop from the left (default 0.22)
//   ONLY           — comma-separated stem list (e.g. "IMG_8819,IMG_8820") to run a subset

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

const ROTATE_QUARTERS = Number(process.env.ROTATE_QUARTERS ?? "1");
const CROP_LEFT = Number(process.env.CROP_LEFT ?? "0.22");
// When set to "1", each move evaluation uses the GROUND-TRUTH previous FEN
// instead of the chained one — so a single misclassification doesn't
// cascade into wrong legal-moves lists for subsequent moves. Useful for
// isolating per-move accuracy.
const RESET_FEN_PER_MOVE = process.env.RESET_FEN_PER_MOVE === "1";
const RECTIFY_SIZE = Number(process.env.RECTIFY_SIZE ?? "768");
const RECTIFY_QUALITY = Number(process.env.RECTIFY_QUALITY ?? "0.92");
const RECTIFY_MARGIN = Number(process.env.RECTIFY_MARGIN ?? "0.14");

// Documented ground truth — from scripts/test-with-ground-truth.ts:22-30.
// IMG_8819 = starting position; each subsequent photo applies one SAN.
const GROUND_TRUTH = [
  "e4", "e5", "Nf3", "Nc6",
  "Nc3", "d6", "b3", "Be7",
  "Ba3", "Nf6", "Nd5", "Nxd5",
  "exd5", "b6",
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

// Establish a canvas-compatible HTMLImageElement/HTMLCanvasElement shim
// so our lib code, which uses document.createElement("canvas") and
// instanceof checks, runs unchanged in Node.
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

// Force the new pipeline modules to load AFTER the DOM shim is in place.
const { calibrateBoard } = await import("../../lib/v2/calibrate.ts");
const { identifyMove } = await import("../../lib/v2/identify-move.ts");
const boardImage = await import("../../lib/board-image.ts");
const { Chess } = await import("chess.js");

const STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

function imageToCanvas(img) {
  const c = createCanvas(img.width, img.height);
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);
  return c;
}

async function main() {
  const files = (await fs.readdir(PHOTO_DIR))
    .filter((n) => /\.(jpe?g|png)$/i.test(n))
    .sort();
  const only = (process.env.ONLY ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  const photos = only.length
    ? files.filter((n) => only.some((stem) => n.startsWith(stem)))
    : files;

  console.log(`Worker: ${WORKER_URL}`);
  console.log(`Photos: ${photos.length}`);
  console.log(`Ground truth: ${GROUND_TRUTH.join(" ")}\n`);

  // ---- Calibration on photo[0] ----
  const t0 = Date.now();
  const startImg = await loadOriented(path.join(PHOTO_DIR, photos[0]));
  const startCanvas = imageToCanvas(startImg);
  const cal = await calibrateBoard({
    proxyUrl: WORKER_URL,
    image: startCanvas,
    origin: ORIGIN,
  });
  const calMs = Date.now() - t0;
  if (cal.kind !== "locked") {
    console.log(`CALIBRATE FAIL on ${photos[0]} (${calMs}ms): ${cal.reason}`);
    return;
  }
  console.log(
    `CALIBRATE ok on ${photos[0]} in ${calMs}ms` +
    (cal.isStartingPosition ? " · starting position confirmed" : " · NOT starting position"),
  );
  console.log(`  corners: ${cal.lock.corners.map((p) => `(${Math.round(p.x)},${Math.round(p.y)})`).join(" ")}`);

  let previousFen = STARTING_FEN;
  let previousRectified = cal.rectified;
  let previousRawCanvas = startCanvas;
  let correct = 0;
  let wrong = 0;
  let abstained = 0;
  let errored = 0;
  const game = new Chess(previousFen);
  const callLatencies = [];
  // Precompute ground-truth FEN history so we can optionally reset per move.
  const truthGame = new Chess(STARTING_FEN);
  const truthFens = [STARTING_FEN];
  for (const san of GROUND_TRUTH) {
    truthGame.move(san);
    truthFens.push(truthGame.fen());
  }

  for (let idx = 1; idx < photos.length; idx++) {
    const name = photos[idx];
    const expectedSan = GROUND_TRUTH[idx - 1];
    const img = await loadOriented(path.join(PHOTO_DIR, name));
    const imgCanvas = imageToCanvas(img);

    // Re-warp every frame with the saved lock so we always send a clean
    // rectified post-image to identify-move.
    const postRect = boardImage.warpBoardWithMargin(
      imgCanvas,
      cal.lock.corners,
      RECTIFY_SIZE,
      RECTIFY_MARGIN,
    );

    const fenForThisMove = RESET_FEN_PER_MOVE ? truthFens[idx - 1] : previousFen;
    const t = Date.now();
    const res = await identifyMove({
      proxyUrl: WORKER_URL,
      previousFen: fenForThisMove,
      preImage: previousRectified,
      postImage: postRect,
      rawPreImage: previousRawCanvas,
      rawPostImage: imgCanvas,
      origin: ORIGIN,
    });
    const dt = Date.now() - t;
    callLatencies.push(dt);

    let mark = "";
    if (res.kind === "matched") {
      if (res.san === expectedSan) {
        correct++;
        mark = "OK   ";
      } else {
        wrong++;
        mark = "WRONG";
      }
      if (RESET_FEN_PER_MOVE) {
        // Always advance the running game with the GROUND-TRUTH move so the
        // pre-image for the next call corresponds to the right position.
        previousFen = truthFens[idx];
      } else {
        try {
          game.move(res.san);
          previousFen = game.fen();
        } catch {
          mark = "ILLEG";
          // Leave previousFen unchanged.
        }
      }
      previousRectified = postRect;
      previousRawCanvas = imgCanvas;
      console.log(
        `[${String(idx).padStart(2)}] ${name} ${mark} · expected ${expectedSan}` +
        ` got ${res.san} (conf ${res.confidence.toFixed(2)}, ${dt}ms)` +
        (res.reason ? ` — ${res.reason}` : ""),
      );
    } else if (res.kind === "abstain") {
      abstained++;
      console.log(
        `[${String(idx).padStart(2)}] ${name} ABSTAIN · expected ${expectedSan} (${dt}ms) — ${res.reason}`,
      );
    } else {
      errored++;
      console.log(
        `[${String(idx).padStart(2)}] ${name} ERROR · expected ${expectedSan} (${dt}ms) — ${res.reason}`,
      );
    }
  }

  const total = correct + wrong + abstained + errored;
  callLatencies.sort((a, b) => a - b);
  const p50 = callLatencies[Math.floor(callLatencies.length / 2)] ?? 0;
  const p95 = callLatencies[Math.floor(callLatencies.length * 0.95)] ?? 0;
  console.log("");
  console.log("=== Summary ===");
  console.log(`Correct:   ${correct}/${total}`);
  console.log(`Wrong:     ${wrong}`);
  console.log(`Abstained: ${abstained}`);
  console.log(`Errored:   ${errored}`);
  console.log(`identify-move latency p50/p95: ${p50}ms / ${p95}ms`);
  console.log(`Calibrate latency: ${calMs}ms`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
