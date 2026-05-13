/**
 * Test multiple prompt variants for per-move accuracy.
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
const MAX_DIM = Number(process.env.MAX_DIM ?? "1568");
const MAX_TOKENS = Number(process.env.MAX_TOKENS ?? "4000");
const VARIANT = process.env.VARIANT ?? "v1";

const GROUND_TRUTH = [
  "e4", "e5", "Nf3", "Nc6", "Nc3", "d6", "b3", "Be7",
  "Ba3", "Nf6", "Nd5", "Nxd5", "exd5", "b6",
];

function makePrompt(prevFen: string, legalSans: string[]): string {
  if (VARIANT === "v1") {
    // baseline
    return `Identify the chess move played between these two photos.

IMAGE 1: before. IMAGE 2: after. Board orientation is identical between them.

Previous FEN (piece placement only): ${prevFen}

Legal moves (exactly one was played):
${legalSans.join(", ")}

Reply with ONLY the SAN of the move.`;
  }
  if (VARIANT === "v2") {
    // Tell Claude to ignore clutter explicitly
    return `Identify the chess move played between these two photos.

The chessboard is a checkered grid of red and white squares. IGNORE everything outside the board area (clock, hand, table surface, background). The board may be rotated 0/90/180/270° in the frame — use the printed rank-file labels printed on the board edges to orient.

IMAGE 1: before. IMAGE 2: after.

Previous FEN (piece placement only): ${prevFen}

Legal moves (exactly one was played):
${legalSans.join(", ")}

Procedure:
- Look only at the 64 squares of the chess board.
- Find which 1-3 squares look different between IMAGE 1 and IMAGE 2.
- Pick the unique legal move that produces those changes.

Reply with ONLY the SAN of the move, no preamble.`;
  }
  if (VARIANT === "v3") {
    // Decompose into piece-identification steps
    return `Two photos of a chess game. One move was played between IMAGE 1 (before) and IMAGE 2 (after).

Previous FEN: ${prevFen}

Legal moves: ${legalSans.join(", ")}

Walk through these steps explicitly:
1. WHICH PIECE moved? Identify by COLOR (white = lighter pieces, black = darker) and TYPE (pawn=small, rook=castle-shaped, knight=horse-shaped, bishop=tall-mitred, queen=tall-crowned, king=cross-topped).
2. FROM which square? Use board labels (rank 1-8, file a-h printed on the edges).
3. TO which square?
4. Cross-check: which legal move matches "piece X from square Y to square Z"?

Last line of your response: just the SAN of the move, no other text.`;
  }
  if (VARIANT === "v4") {
    // Verify before image first, then identify diff
    return `Two photos of a chess game. ONE chess move was played between IMAGE 1 (before) and IMAGE 2 (after).

The board orientation between the two photos is identical. The board may be rotated in the frame; use rank/file labels on the board edges to orient.

Previous FEN (the position EXPECTED in IMAGE 1, piece placement only):
${prevFen}

Legal moves (the move that was played is EXACTLY one of these):
${legalSans.join(", ")}

Step 1: Look at IMAGE 1. Verify it matches the previous FEN above — every piece in the FEN should be on the board, in the expected squares. If you cannot verify this, say so.

Step 2: Compare to IMAGE 2 and identify which 1-3 squares CHANGED.

Step 3: From the legal-moves list, pick the unique move whose result explains those changes.

End with a line "ANSWER: <san>" on its own. The san must be one of the legal moves above, exactly as written.`;
  }
  throw new Error(`unknown variant ${VARIANT}`);
}

function extractSan(raw: string, legalSans: string[]): string | null {
  const trimmed = raw.trim();
  if (legalSans.includes(trimmed)) return trimmed;
  const lines = trimmed.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i--) {
    const m = lines[i].match(/ANSWER\s*[:=]\s*([A-Za-z0-9+#=\-]+)/i);
    if (m) {
      for (const san of legalSans) {
        if (san === m[1] || san.replace(/[+#]+$/, "") === m[1].replace(/[+#]+$/, "")) {
          return san;
        }
      }
    }
  }
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
  const body: Record<string, unknown> = {
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b1 } },
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b2 } },
          { type: "text", text: makePrompt(prevFen, legal) },
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
  console.log(`Variant: ${VARIANT}, Model: ${MODEL}, max_dim: ${MAX_DIM}`);
  console.log(`Ground truth: ${GROUND_TRUTH.join(" ")}\n`);

  const reference = new Chess();
  const prevFens: string[] = [];
  const legalLists: string[][] = [];
  for (let i = 0; i < GROUND_TRUTH.length; i++) {
    prevFens.push(reference.fen().split(/\s+/)[0]);
    legalLists.push((reference.moves({ verbose: true }) as Move[]).map((m) => m.san));
    reference.move(GROUND_TRUTH[i]);
  }

  const predictions: string[] = [];
  let correct = 0;
  for (let i = 0; i < GROUND_TRUTH.length; i++) {
    const before = path.join(PHOTOS_DIR, files[i]);
    const after = path.join(PHOTOS_DIR, files[i + 1]);
    try {
      const { matched } = await callOnce(before, after, prevFens[i], legalLists[i]);
      const ok = matched === GROUND_TRUTH[i];
      if (ok) correct += 1;
      predictions.push(matched ?? "?");
      console.log(
        `[${(i + 1).toString().padStart(2)}] expected=${GROUND_TRUTH[i].padEnd(6)} got=${(matched ?? "?").padEnd(6)} ${ok ? "✓" : "✗"}`,
      );
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
