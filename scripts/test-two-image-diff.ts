/**
 * Two-image diff approach: hand Claude the previous photo + current photo +
 * the legal-move list, ask which legal move explains the change. Claude
 * only needs to identify what *changed* between two photos, not read all
 * 64 squares correctly.
 *
 * Run:  npx tsx scripts/test-two-image-diff.ts
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

function buildPrompt(prevFen: string, legalSans: string[]): string {
  return `You are identifying which chess move was just played by comparing two photographs of the same physical chessboard.

IMAGE 1: the board BEFORE the move.
IMAGE 2: the board AFTER the move.

Both photos may be taken from any angle / any orientation — the board could be rotated 0°, 90°, 180°, or 270° in the frame, and the camera angle may vary. Use the rank/file labels printed on the board edges, or the location of the white vs black pieces, to orient yourself. The board orientation between the two photos is the same (same camera position).

PREVIOUS POSITION FEN (piece placement only): ${prevFen}

LEGAL MOVES — exactly one of these was played:
${legalSans.join(", ")}

Procedure:
1. Identify which square(s) changed between IMAGE 1 and IMAGE 2 (a piece appeared, disappeared, or was replaced by a different color).
2. Pick the unique legal move from the list above whose result explains exactly those changes.
3. Reply with ONLY that move's SAN, exactly as written in the list (e.g. "e4", "Nxf3", "O-O").

No explanation, no preamble, no markdown — just the SAN of the move.`;
}

async function loadResized(file: string): Promise<string> {
  const buf = await sharp(file)
    .rotate()
    .resize(MAX_DIM, MAX_DIM, { fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 85 })
    .toBuffer();
  return buf.toString("base64");
}

async function identifyMove(
  beforePath: string,
  afterPath: string,
  prevFen: string,
  legalSans: string[],
): Promise<{ raw: string; matched?: string }> {
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
  if (!MODEL.includes("opus")) body.temperature = 0.05;

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
  const lastLine = raw
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .pop() ?? "";
  for (const san of legalSans) {
    const re = new RegExp(
      `(^|[^A-Za-z0-9])${san.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`,
    );
    if (re.test(lastLine)) return { raw, matched: san };
  }
  for (const san of legalSans) {
    const re = new RegExp(
      `(^|[^A-Za-z0-9])${san.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`,
    );
    if (re.test(raw)) return { raw, matched: san };
  }
  return { raw };
}

async function main() {
  const files = fs
    .readdirSync(PHOTOS_DIR)
    .filter((f) => /\.(jpe?g|png|heic)$/i.test(f))
    .sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }),
    );
  console.log(`Photos: ${files.length}`);
  console.log(`Model: ${MODEL}, max_dim=${MAX_DIM}, max_tokens=${MAX_TOKENS}`);
  console.log("");

  const chess = new Chess();
  let resolved = 0;
  const failures: { idx: number; file: string; raw: string }[] = [];

  for (let i = 1; i < files.length; i++) {
    const before = path.join(PHOTOS_DIR, files[i - 1]);
    const after = path.join(PHOTOS_DIR, files[i]);
    const prevFen = chess.fen().split(/\s+/)[0];
    const legal = (chess.moves({ verbose: true }) as Move[]).map((m) => m.san);
    const t0 = Date.now();
    try {
      const { raw, matched } = await identifyMove(
        before,
        after,
        prevFen,
        legal,
      );
      const ms = Date.now() - t0;
      if (matched) {
        chess.move(matched);
        console.log(`[${i}] ${files[i]} → ${matched} (${ms}ms)`);
        resolved += 1;
      } else {
        console.log(
          `[${i}] ${files[i]} → UNMATCHED (${ms}ms)\n  raw: ${raw.slice(0, 200)}`,
        );
        failures.push({ idx: i, file: files[i], raw });
        break;
      }
    } catch (e) {
      console.log(`[${i}] ${files[i]} → ERROR: ${e instanceof Error ? e.message : String(e)}`);
      failures.push({
        idx: i,
        file: files[i],
        raw: e instanceof Error ? e.message : String(e),
      });
      break;
    }
  }

  console.log(`\n--- Summary ---`);
  console.log(`Resolved: ${resolved} / ${files.length - 1}`);
  console.log(`Final PGN: ${chess.pgn()}`);
  if (resolved === files.length - 1) {
    console.log("ALL MOVES RESOLVED");
    process.exit(0);
  } else {
    process.exit(1);
  }
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(2);
});
