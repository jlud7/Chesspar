/**
 * 3-call majority voting with cropped photos and the v2 prompt.
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
const VOTES = Number(process.env.VOTES ?? "3");
const CROP_H = Number(process.env.CROP_H ?? "0.62");

const GROUND_TRUTH = [
  "e4", "e5", "Nf3", "Nc6", "Nc3", "d6", "b3", "Be7",
  "Ba3", "Nf6", "Nd5", "Nxd5", "exd5", "b6",
];

function prompt(prevFen: string, legalSans: string[]): string {
  return `Identify the chess move played between these two photos.

The chessboard may be rotated 0/90/180/270° in the frame — use the printed rank-file labels (1-8, a-h) on the board edges to orient yourself.

IMAGE 1: before. IMAGE 2: after.

Previous FEN: ${prevFen}

Legal moves (exactly one was played): ${legalSans.join(", ")}

Find which squares differ between IMAGE 1 and IMAGE 2, identify the piece type and exact rank/file. Pay attention to subtle differences like pawn on b3 vs d4 (different file), or knight on d4 vs d5 (different rank).

Reply with ONLY the SAN. No preamble.`;
}

function extractSan(raw: string, legalSans: string[]): string | null {
  const trimmed = raw.trim();
  if (legalSans.includes(trimmed)) return trimmed;
  const last = trimmed.split(/\r?\n/).map((l) => l.trim()).filter(Boolean).pop() ?? "";
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

async function loadCroppedResized(file: string): Promise<string> {
  const rotatedBuf = await sharp(file).rotate().toBuffer();
  const meta = await sharp(rotatedBuf).metadata();
  const W = meta.width ?? 0;
  const H = meta.height ?? 0;
  const buf = await sharp(rotatedBuf)
    .extract({ top: 0, left: 0, width: W, height: Math.round(H * CROP_H) })
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
): Promise<string | null> {
  const [b1, b2] = await Promise.all([
    loadCroppedResized(before),
    loadCroppedResized(after),
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
  const resp = await fetch(WORKER_URL.replace(/\/$/, "") + "/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: ORIGIN },
    body: JSON.stringify(body),
  });
  if (!resp.ok) return null;
  const data = (await resp.json()) as { content?: { type: string; text?: string }[] };
  const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
  return extractSan(raw, legal);
}

async function main() {
  const files = fs
    .readdirSync(PHOTOS_DIR)
    .filter((f) => /\.(jpe?g|png|heic)$/i.test(f))
    .sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }),
    );
  console.log(`Model: ${MODEL}, max_dim: ${MAX_DIM}, votes: ${VOTES}, crop_h: ${CROP_H}`);
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
  for (let i = 0; i < GROUND_TRUTH.length; i++) {
    const before = path.join(PHOTOS_DIR, files[i]);
    const after = path.join(PHOTOS_DIR, files[i + 1]);
    const votes: (string | null)[] = [];
    const t0 = Date.now();
    const promises: Promise<string | null>[] = [];
    for (let v = 0; v < VOTES; v++) {
      promises.push(callOnce(before, after, prevFens[i], legalLists[i]));
    }
    const results = await Promise.all(promises);
    for (const r of results) votes.push(r);
    const ms = Date.now() - t0;
    // Tally votes
    const tally = new Map<string, number>();
    for (const v of votes) {
      if (!v) continue;
      tally.set(v, (tally.get(v) ?? 0) + 1);
    }
    let winner: string | null = null;
    let bestCount = 0;
    for (const [san, count] of tally) {
      if (count > bestCount) {
        winner = san;
        bestCount = count;
      }
    }
    const ok = winner === GROUND_TRUTH[i];
    if (ok) correct += 1;
    console.log(
      `[${(i + 1).toString().padStart(2)}] expected=${GROUND_TRUTH[i].padEnd(6)} winner=${(winner ?? "?").padEnd(6)} votes=[${votes.join(",")}] ${ok ? "✓" : "✗"} (${ms}ms)`,
    );
  }
  console.log(
    `\nAccuracy: ${correct}/${GROUND_TRUTH.length} = ${((correct / GROUND_TRUTH.length) * 100).toFixed(0)}%`,
  );
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(2);
});
