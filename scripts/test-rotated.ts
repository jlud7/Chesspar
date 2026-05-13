/**
 * Pre-rotate so white-at-bottom (standard chess view), then crop the
 * clock + hand off the left side.
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
const ROTATE = Number(process.env.ROTATE ?? "90"); // degrees CW
const CROP_LEFT = Number(process.env.CROP_LEFT ?? "0.18"); // % to crop off left
const CROP_RIGHT = Number(process.env.CROP_RIGHT ?? "0.0"); // % to crop off right
const CROP_TOP = Number(process.env.CROP_TOP ?? "0.0");
const CROP_BOTTOM = Number(process.env.CROP_BOTTOM ?? "0.0");
const VOTES = Number(process.env.VOTES ?? "1");

const GROUND_TRUTH = [
  "e4", "e5", "Nf3", "Nc6", "Nc3", "d6", "b3", "Be7",
  "Ba3", "Nf6", "Nd5", "Nxd5", "exd5", "b6",
];

function prompt(prevFen: string, legalSans: string[]): string {
  return `Identify the chess move played between these two photos.

The photos show a chessboard from above with WHITE PIECES at the BOTTOM and BLACK PIECES at the TOP (the standard chess viewing orientation). Rank 1 is at the bottom of the image, rank 8 is at the top. File a is on the left, file h on the right.

IMAGE 1: before. IMAGE 2: after.

Previous FEN (piece placement only): ${prevFen}

Legal moves (exactly one was played):
${legalSans.join(", ")}

Find which 1-3 squares look different between IMAGE 1 and IMAGE 2, identify which piece moved (by color and type) and from which square to which, then pick the unique legal move that produces those changes.

Reply with ONLY the SAN of the move, no preamble.`;
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

async function loadProcessed(file: string): Promise<string> {
  // 1. honor EXIF rotation
  // 2. rotate to make white-at-bottom (sample photos need +90° CW)
  // 3. crop clutter (clock area)
  // 4. resize
  const exifRotated = await sharp(file).rotate().toBuffer();
  const standardOrientation = await sharp(exifRotated).rotate(ROTATE).toBuffer();
  const meta = await sharp(standardOrientation).metadata();
  const W = meta.width ?? 0;
  const H = meta.height ?? 0;
  const left = Math.max(0, Math.round(W * CROP_LEFT));
  const top = Math.max(0, Math.round(H * CROP_TOP));
  const width = Math.max(1, Math.round(W * (1 - CROP_LEFT - CROP_RIGHT)));
  const height = Math.max(1, Math.round(H * (1 - CROP_TOP - CROP_BOTTOM)));
  const buf = await sharp(standardOrientation)
    .extract({ left, top, width, height })
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
  attempt = 0,
): Promise<string | null> {
  try {
    const [b1, b2] = await Promise.all([loadProcessed(before), loadProcessed(after)]);
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
    if (!resp.ok) return null;
    const data = (await resp.json()) as { content?: { type: string; text?: string }[] };
    const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
    return extractSan(raw, legal);
  } catch (e) {
    if (attempt < 2) {
      await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
      return callOnce(before, after, prevFen, legal, attempt + 1);
    }
    throw e;
  }
}

async function main() {
  const files = fs
    .readdirSync(PHOTOS_DIR)
    .filter((f) => /\.(jpe?g|png|heic)$/i.test(f))
    .sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }),
    );
  console.log(
    `Model: ${MODEL}, rotate: ${ROTATE}°, crop_left: ${CROP_LEFT}, max_dim: ${MAX_DIM}, votes: ${VOTES}`,
  );
  // Save sample for inspection
  {
    const data = await loadProcessed(path.join(PHOTOS_DIR, files[0]));
    fs.writeFileSync("/tmp/rotated-cropped-0.jpg", Buffer.from(data, "base64"));
    console.log("Sample preprocessed image: /tmp/rotated-cropped-0.jpg\n");
  }
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
    const t0 = Date.now();
    const promises = Array.from({ length: VOTES }, () =>
      callOnce(before, after, prevFens[i], legalLists[i]),
    );
    const results = await Promise.all(promises);
    const tally = new Map<string, number>();
    for (const r of results) {
      if (!r) continue;
      tally.set(r, (tally.get(r) ?? 0) + 1);
    }
    let winner: string | null = null;
    let bestCount = 0;
    for (const [san, count] of tally) {
      if (count > bestCount) {
        winner = san;
        bestCount = count;
      }
    }
    const ms = Date.now() - t0;
    const ok = winner === GROUND_TRUTH[i];
    if (ok) correct += 1;
    console.log(
      `[${(i + 1).toString().padStart(2)}] expected=${GROUND_TRUTH[i].padEnd(6)} got=${(winner ?? "?").padEnd(6)} votes=[${results.join(",")}] ${ok ? "✓" : "✗"} (${ms}ms)`,
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
