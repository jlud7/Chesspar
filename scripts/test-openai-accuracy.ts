/**
 * Per-move accuracy test against GPT-5 via the worker /openai endpoint.
 */
import fs from "fs";
import path from "path";
import sharp from "sharp";
import { Chess, type Move } from "chess.js";

const PHOTOS_DIR = path.resolve(__dirname, "..", "Test_Photos");
const WORKER_URL =
  process.env.WORKER_URL ?? "https://chesspar-vlm.jamesleoluddy.workers.dev";
const ORIGIN = process.env.ORIGIN ?? "https://jlud7.github.io";
const MODEL = process.env.MODEL ?? "gpt-5";
const MAX_DIM = Number(process.env.MAX_DIM ?? "1568");
const MAX_TOKENS = Number(process.env.MAX_TOKENS ?? "4000");

const GROUND_TRUTH = [
  "e4", "e5", "Nf3", "Nc6", "Nc3", "d6", "b3", "Be7",
  "Ba3", "Nf6", "Nd5", "Nxd5", "exd5", "b6",
];

function prompt(prevFen: string, legalSans: string[]): string {
  return `Identify the chess move played between these two photos.

The chessboard is a checkered grid of red and white squares. IGNORE everything outside the board area (clock, hand, table surface). The board may be rotated 0/90/180/270° in the frame — use the printed rank-file labels on the board edges to orient.

IMAGE 1: before. IMAGE 2: after.

Previous FEN (piece placement only): ${prevFen}

Legal moves (exactly one was played):
${legalSans.join(", ")}

Find which 1-3 squares look different between IMAGE 1 and IMAGE 2, then pick the unique legal move that produces those changes.

Reply with ONLY the SAN of the move (e.g. "e4", "Nxf3"). No preamble.`;
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
    .jpeg({ quality: 90 })
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
  const content = [
    { type: "text", text: prompt(prevFen, legal) },
    { type: "image_url", image_url: { url: `data:image/jpeg;base64,${b1}` } },
    { type: "image_url", image_url: { url: `data:image/jpeg;base64,${b2}` } },
  ];
  const isGpt5 = MODEL.startsWith("gpt-5");
  const body: Record<string, unknown> = {
    model: MODEL,
    messages: [{ role: "user", content }],
  };
  if (isGpt5) {
    body.max_completion_tokens = MAX_TOKENS;
  } else {
    body.max_tokens = 64;
    body.temperature = 0;
  }
  const resp = await fetch(WORKER_URL.replace(/\/$/, "") + "/openai", {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: ORIGIN },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const t = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status}: ${t.slice(0, 200)}`);
  }
  const data = (await resp.json()) as {
    choices?: { message?: { content?: string } }[];
  };
  const raw = data.choices?.[0]?.message?.content ?? "";
  return { raw, matched: extractSan(raw, legal) };
}

async function main() {
  const files = fs
    .readdirSync(PHOTOS_DIR)
    .filter((f) => /\.(jpe?g|png|heic)$/i.test(f))
    .sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }),
    );
  console.log(`Model: ${MODEL}, max_dim: ${MAX_DIM}`);
  console.log(`Ground truth: ${GROUND_TRUTH.join(" ")}\n`);

  const reference = new Chess();
  const prevFens: string[] = [];
  const legalLists: string[][] = [];
  for (let i = 0; i < GROUND_TRUTH.length; i++) {
    prevFens.push(reference.fen().split(/\s+/)[0]);
    legalLists.push((reference.moves({ verbose: true }) as Move[]).map((m) => m.san));
    reference.move(GROUND_TRUTH[i]);
  }

  let correct = 0;
  const predictions: string[] = [];
  for (let i = 0; i < GROUND_TRUTH.length; i++) {
    const before = path.join(PHOTOS_DIR, files[i]);
    const after = path.join(PHOTOS_DIR, files[i + 1]);
    const t0 = Date.now();
    try {
      const { matched, raw } = await callOnce(before, after, prevFens[i], legalLists[i]);
      const ms = Date.now() - t0;
      const ok = matched === GROUND_TRUTH[i];
      if (ok) correct += 1;
      predictions.push(matched ?? "?");
      console.log(
        `[${(i + 1).toString().padStart(2)}] expected=${GROUND_TRUTH[i].padEnd(6)} got=${(matched ?? "?").padEnd(6)} ${ok ? "✓" : "✗"} (${ms}ms)`,
      );
      if (!ok) console.log(`     raw: ${raw.slice(-200)}`);
    } catch (e) {
      predictions.push("ERR");
      console.log(`[${i + 1}] ERROR: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  console.log(
    `\nAccuracy: ${correct}/${GROUND_TRUTH.length} = ${((correct / GROUND_TRUTH.length) * 100).toFixed(0)}%`,
  );
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(2);
});
