/**
 * Crop the photos to just the chessboard before sending. The user's photos
 * have a clock + hand in the lower-right that may distract the model.
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
// Hard-coded crop region in % of the original image (top, left, width, height)
// Tuned against the user's specific test photos. The board sits in the upper-
// left area; clock + hand are in the lower-right.
const CROP_TOP = Number(process.env.CROP_TOP ?? "0.02");
const CROP_LEFT = Number(process.env.CROP_LEFT ?? "0.02");
const CROP_W = Number(process.env.CROP_W ?? "0.78");
const CROP_H = Number(process.env.CROP_H ?? "0.76");

const GROUND_TRUTH = [
  "e4", "e5", "Nf3", "Nc6", "Nc3", "d6", "b3", "Be7",
  "Ba3", "Nf6", "Nd5", "Nxd5", "exd5", "b6",
];

function prompt(prevFen: string, legalSans: string[]): string {
  return `Identify the chess move played between these two photos.

The chessboard is a checkered grid of red and white squares. The board may be rotated 0/90/180/270° in the frame — use the printed rank-file labels on the board edges (numbers 1-8, letters a-h) to orient yourself precisely.

IMAGE 1: before. IMAGE 2: after.

Previous FEN (piece placement only): ${prevFen}

Legal moves (exactly one was played):
${legalSans.join(", ")}

Procedure:
- Locate the rank/file labels printed on the board's edges. Determine which image-edge corresponds to which rank/file.
- Identify the SPECIFIC square coordinates (e.g. "e4", "g8") where things differ between IMAGE 1 and IMAGE 2.
- Pay close attention to piece TYPE (pawn vs knight vs bishop) and exact rank — a piece on d4 is on rank 4, on d5 is on rank 5 (different).
- Pick the unique legal move from the list whose result matches the observed changes.

Reply with ONLY the SAN of the move on its own line. No preamble.`;
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
  // Rotate first to a buffer so we can compute crop bounds from POST-rotation
  // dimensions.
  const rotatedBuf = await sharp(file).rotate().toBuffer();
  const meta = await sharp(rotatedBuf).metadata();
  const W = meta.width ?? 0;
  const H = meta.height ?? 0;
  const top = Math.max(0, Math.round(H * CROP_TOP));
  const left = Math.max(0, Math.round(W * CROP_LEFT));
  const width = Math.min(W - left, Math.round(W * CROP_W));
  const height = Math.min(H - top, Math.round(H * CROP_H));
  const buf = await sharp(rotatedBuf)
    .extract({ top, left, width, height })
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
  const data = (await resp.json()) as { content?: { type: string; text?: string }[] };
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
  console.log(
    `Model: ${MODEL}, max_dim: ${MAX_DIM}, crop: top=${CROP_TOP} left=${CROP_LEFT} w=${CROP_W} h=${CROP_H}`,
  );
  console.log(`Ground truth: ${GROUND_TRUTH.join(" ")}\n`);

  // Save sample crops to /tmp for inspection
  {
    const rotated = await sharp(path.join(PHOTOS_DIR, files[0])).rotate().toBuffer();
    const m = await sharp(rotated).metadata();
    const W = m.width!;
    const H = m.height!;
    await sharp(rotated)
      .extract({
        top: Math.round(H * CROP_TOP),
        left: Math.round(W * CROP_LEFT),
        width: Math.round(W * CROP_W),
        height: Math.round(H * CROP_H),
      })
      .jpeg({ quality: 92 })
      .toFile("/tmp/test-cropped-0.jpg");
  }
  console.log("Sample crop saved to /tmp/test-cropped-0.jpg\n");

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
    try {
      const { matched } = await callOnce(before, after, prevFens[i], legalLists[i]);
      const ok = matched === GROUND_TRUTH[i];
      if (ok) correct += 1;
      console.log(
        `[${(i + 1).toString().padStart(2)}] expected=${GROUND_TRUTH[i].padEnd(6)} got=${(matched ?? "?").padEnd(6)} ${ok ? "✓" : "✗"}`,
      );
    } catch (e) {
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
