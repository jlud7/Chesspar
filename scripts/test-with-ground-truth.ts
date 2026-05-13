/**
 * Ground-truth-aware test. The actual moves played in Test_Photos are known
 * (the user confirmed: 1.e4 e5 2.Nf3 Nc6 3.Nc3 d6 4.b3 Be7 5.Ba3 Nf6 6.Nd5
 * Nxd5 7.exd5 b6). This script runs the photo-diff pipeline and reports
 * per-move correctness, plus consistency across multiple runs.
 */
import fs from "fs";
import path from "path";
import sharp from "sharp";
import { Chess, type Move } from "chess.js";

const PHOTOS_DIR = path.resolve(__dirname, "..", "Test_Photos");
const WORKER_URL =
  process.env.WORKER_URL ?? "https://chesspar-vlm.jamesleoluddy.workers.dev";
const ORIGIN = process.env.ORIGIN ?? "https://jlud7.github.io";
const MODEL = process.env.MODEL ?? "claude-opus-4-7";
const MAX_DIM = Number(process.env.MAX_DIM ?? "1280");
const MAX_TOKENS = Number(process.env.MAX_TOKENS ?? "4000");
const RUNS = Number(process.env.RUNS ?? "1");
const PROMPT_VARIANT = process.env.PROMPT ?? "structured";

const GROUND_TRUTH = [
  "e4", "e5",
  "Nf3", "Nc6",
  "Nc3", "d6",
  "b3", "Be7",
  "Ba3", "Nf6",
  "Nd5", "Nxd5",
  "exd5", "b6",
];

function buildPrompt(prevFen: string, legalSans: string[]): string {
  if (PROMPT_VARIANT === "simple") {
    return `Identify the chess move played between these two photos.

IMAGE 1: before. IMAGE 2: after. The board orientation in both photos is the same. The board may appear rotated 0/90/180/270° in the frame — use the printed rank-file labels on the board edges, or piece colors, to orient.

Previous FEN (piece placement only): ${prevFen}

Legal moves (exactly one was played):
${legalSans.join(", ")}

Reply with ONLY the SAN of the move. No preamble, no markdown, no explanation.`;
  }
  return `Two photographs of the same physical chessboard. The board orientation is identical between them; one chess move was played between IMAGE 1 (before) and IMAGE 2 (after).

PREVIOUS POSITION FEN (piece placement only): ${prevFen}

LEGAL MOVES — exactly one of these was played:
${legalSans.join(", ")}

Work through this carefully. The board may be rotated 0°/90°/180°/270° in the photo. Use the rank-file labels printed on the board AND the location of the white vs black pieces to orient.

REASONING STEPS (be brief):
1. State which image-edge corresponds to a-file, h-file, rank 1, rank 8.
2. Identify which SPECIFIC squares differ between IMAGE 1 and IMAGE 2 (a piece appeared, disappeared, or was replaced). Use square names (e.g. "e2 → empty, e4 → white pawn").
3. From the legal-moves list, pick the unique move whose result matches that change.

End your response with a single line "ANSWER: <san>" where <san> is exactly as it appears in the legal-moves list. The ANSWER line must be the very last line.`;
}

function extractAnswer(raw: string, legalSans: string[]): string | null {
  const lines = raw.split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i--) {
    const m = lines[i].match(/ANSWER\s*[:=]\s*([A-Za-z0-9+#=\-]+)/i);
    if (m) {
      for (const san of legalSans) {
        if (san === m[1]) return san;
      }
      const bare = m[1].replace(/[+#]+$/, "");
      for (const san of legalSans) {
        if (san.replace(/[+#]+$/, "") === bare) return san;
      }
    }
  }
  const lastLine = lines
    .map((l) => l.trim())
    .filter(Boolean)
    .pop() ?? "";
  const sorted = [...legalSans].sort((a, b) => b.length - a.length);
  for (const san of sorted) {
    const re = new RegExp(
      `(^|[^A-Za-z0-9])${san.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`,
    );
    if (re.test(lastLine)) return san;
  }
  for (const san of sorted) {
    const re = new RegExp(
      `(^|[^A-Za-z0-9])${san.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`,
    );
    if (re.test(raw)) return san;
  }
  return null;
}

async function loadResized(file: string): Promise<string> {
  const buf = await sharp(file)
    .rotate()
    .resize(MAX_DIM, MAX_DIM, { fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 88 })
    .toBuffer();
  return buf.toString("base64");
}

async function identifyMove(
  beforePath: string,
  afterPath: string,
  prevFen: string,
  legalSans: string[],
): Promise<{ raw: string; matched: string | null }> {
  const [beforeB64, afterB64] = await Promise.all([
    loadResized(beforePath),
    loadResized(afterPath),
  ]);
  const body: Record<string, unknown> = {
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/jpeg",
              data: beforeB64,
            },
          },
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/jpeg",
              data: afterB64,
            },
          },
          { type: "text", text: buildPrompt(prevFen, legalSans) },
        ],
      },
    ],
  };
  if (!MODEL.includes("opus")) body.temperature = 0;
  const resp = await fetch(WORKER_URL.replace(/\/$/, "") + "/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: ORIGIN },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const t = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status}: ${t.slice(0, 200)}`);
  }
  const data = (await resp.json()) as {
    content?: { type: string; text?: string }[];
  };
  const raw = (
    data.content?.find((c) => c.type === "text")?.text ?? ""
  ).trim();
  return { raw, matched: extractAnswer(raw, legalSans) };
}

async function runGame(): Promise<{ predicted: string[]; correctness: boolean[] }> {
  const files = fs
    .readdirSync(PHOTOS_DIR)
    .filter((f) => /\.(jpe?g|png|heic)$/i.test(f))
    .sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }),
    );
  const chess = new Chess();
  const predicted: string[] = [];
  const correctness: boolean[] = [];
  for (let i = 1; i < files.length; i++) {
    const before = path.join(PHOTOS_DIR, files[i - 1]);
    const after = path.join(PHOTOS_DIR, files[i]);
    const prevFen = chess.fen().split(/\s+/)[0];
    const legal = (chess.moves({ verbose: true }) as Move[]).map((m) => m.san);
    const groundTruth = GROUND_TRUTH[i - 1];
    const t0 = Date.now();
    try {
      const { matched, raw } = await identifyMove(before, after, prevFen, legal);
      const ms = Date.now() - t0;
      if (matched) {
        chess.move(matched);
        const ok = matched === groundTruth;
        predicted.push(matched);
        correctness.push(ok);
        console.log(
          `[${i}] expected=${groundTruth.padEnd(6)} got=${matched.padEnd(6)} ${ok ? "✓" : "✗"} (${ms}ms)`,
        );
        if (!ok) {
          console.log(`     raw: ${raw.slice(-200)}`);
        }
      } else {
        predicted.push("?");
        correctness.push(false);
        console.log(`[${i}] expected=${groundTruth} got=UNMATCHED ✗ (${ms}ms)`);
        console.log(`     raw: ${raw.slice(-200)}`);
        break;
      }
    } catch (e) {
      predicted.push("ERR");
      correctness.push(false);
      console.log(
        `[${i}] expected=${groundTruth} got=ERROR ✗ — ${e instanceof Error ? e.message : String(e)}`,
      );
      break;
    }
  }
  return { predicted, correctness };
}

async function main() {
  console.log(
    `Model: ${MODEL}, max_dim: ${MAX_DIM}, max_tokens: ${MAX_TOKENS}, prompt: ${PROMPT_VARIANT}, runs: ${RUNS}`,
  );
  console.log(`Ground truth: ${GROUND_TRUTH.join(" ")}\n`);

  const allResults: { predicted: string[]; correctness: boolean[] }[] = [];
  for (let r = 0; r < RUNS; r++) {
    if (RUNS > 1) console.log(`\n--- Run ${r + 1} ---`);
    const result = await runGame();
    allResults.push(result);
    const correct = result.correctness.filter(Boolean).length;
    const total = GROUND_TRUTH.length;
    console.log(`Run ${r + 1}: ${correct}/${total} moves correct`);
  }

  if (RUNS > 1) {
    console.log("\n=== Per-move correctness across runs ===");
    for (let i = 0; i < GROUND_TRUTH.length; i++) {
      const calls = allResults.map((r) => ({
        pred: r.predicted[i] ?? "—",
        ok: r.correctness[i] ?? false,
      }));
      console.log(
        `Move ${(i + 1).toString().padStart(2)}: expected=${GROUND_TRUTH[i].padEnd(6)}  ${calls
          .map((c) => `${c.pred.padEnd(6)}${c.ok ? "✓" : "✗"}`)
          .join("  ")}`,
      );
    }
    const totals = allResults.map(
      (r) => r.correctness.filter(Boolean).length,
    );
    console.log(`\nAccuracy by run: ${totals.join(", ")}`);
    const avg = totals.reduce((a, b) => a + b, 0) / totals.length;
    console.log(`Mean: ${avg.toFixed(1)} / ${GROUND_TRUTH.length}`);
  }
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(2);
});
