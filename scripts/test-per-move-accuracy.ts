/**
 * Per-move accuracy test. Each move is tested INDEPENDENTLY with the correct
 * prior position (from ground truth), so we measure raw per-call accuracy
 * without compounding errors.
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

const GROUND_TRUTH = [
  "e4", "e5",
  "Nf3", "Nc6",
  "Nc3", "d6",
  "b3", "Be7",
  "Ba3", "Nf6",
  "Nd5", "Nxd5",
  "exd5", "b6",
];

function prompt(prevFen: string, legalSans: string[]): string {
  return `Identify the chess move played between these two photos.

IMAGE 1: before. IMAGE 2: after. Board orientation is identical between them; the board may appear rotated 0/90/180/270° in the frame — use the printed rank-file labels and piece colors to orient.

Previous FEN (piece placement only): ${prevFen}

Legal moves (exactly one was played):
${legalSans.join(", ")}

Reply with ONLY the SAN of the move. No preamble, no markdown, no explanation.`;
}

function extractSan(raw: string, legalSans: string[]): string | null {
  const trimmed = raw.trim();
  if (legalSans.includes(trimmed)) return trimmed;
  const lines = trimmed.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const last = lines[lines.length - 1] ?? "";
  const sorted = [...legalSans].sort((a, b) => b.length - a.length);
  for (const san of sorted) {
    const re = new RegExp(
      `(^|[^A-Za-z0-9])${san.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`,
    );
    if (re.test(last)) return san;
  }
  for (const san of sorted) {
    const re = new RegExp(
      `(^|[^A-Za-z0-9])${san.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`,
    );
    if (re.test(trimmed)) return san;
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

async function callOnce(
  before: string,
  after: string,
  prevFen: string,
  legal: string[],
): Promise<{ raw: string; matched: string | null }> {
  const [b1, b2] = await Promise.all([loadResized(before), loadResized(after)]);
  const body: Record<string, unknown> = {
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "base64", media_type: "image/jpeg", data: b1 },
          },
          {
            type: "image",
            source: { type: "base64", media_type: "image/jpeg", data: b2 },
          },
          { type: "text", text: prompt(prevFen, legal) },
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
  const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
  return { raw, matched: extractSan(raw, legal) };
}

async function main() {
  const files = fs
    .readdirSync(PHOTOS_DIR)
    .filter((f) => /\.(jpe?g|png|heic)$/i.test(f))
    .sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }),
    );
  console.log(`Model: ${MODEL}, max_dim: ${MAX_DIM}, runs: ${RUNS}`);
  console.log(`Ground truth: ${GROUND_TRUTH.join(" ")}\n`);

  const reference = new Chess();
  const prevFens: string[] = [];
  const legalLists: string[][] = [];
  for (let i = 0; i < GROUND_TRUTH.length; i++) {
    prevFens.push(reference.fen().split(/\s+/)[0]);
    legalLists.push((reference.moves({ verbose: true }) as Move[]).map((m) => m.san));
    reference.move(GROUND_TRUTH[i]);
  }

  const perMoveResults: { correct: number; runs: string[] }[] = GROUND_TRUTH.map(() => ({
    correct: 0,
    runs: [],
  }));

  for (let r = 0; r < RUNS; r++) {
    console.log(`\n--- Run ${r + 1} ---`);
    for (let i = 0; i < GROUND_TRUTH.length; i++) {
      const before = path.join(PHOTOS_DIR, files[i]);
      const after = path.join(PHOTOS_DIR, files[i + 1]);
      try {
        const { matched } = await callOnce(
          before,
          after,
          prevFens[i],
          legalLists[i],
        );
        const ok = matched === GROUND_TRUTH[i];
        if (ok) perMoveResults[i].correct += 1;
        perMoveResults[i].runs.push(matched ?? "?");
        console.log(
          `[${i + 1}] expected=${GROUND_TRUTH[i].padEnd(6)} got=${(matched ?? "?").padEnd(6)} ${ok ? "✓" : "✗"}`,
        );
      } catch (e) {
        perMoveResults[i].runs.push("ERR");
        console.log(
          `[${i + 1}] expected=${GROUND_TRUTH[i]} ERROR: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    }
  }

  console.log("\n=== Per-move accuracy ===");
  for (let i = 0; i < GROUND_TRUTH.length; i++) {
    console.log(
      `Move ${(i + 1).toString().padStart(2)}: expected=${GROUND_TRUTH[i].padEnd(6)}  ${perMoveResults[i].correct}/${RUNS} correct  [${perMoveResults[i].runs.join(", ")}]`,
    );
  }
  const totalCorrect = perMoveResults.reduce((a, b) => a + b.correct, 0);
  const totalTries = perMoveResults.length * RUNS;
  console.log(
    `\nOverall: ${totalCorrect}/${totalTries} = ${((totalCorrect / totalTries) * 100).toFixed(0)}%`,
  );
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(2);
});
