/**
 * Structured-diff prompt: force Claude to enumerate which physical squares
 * changed (not "I think it was e4") before picking a move. The hope is that
 * grounding the answer in observed pixel changes will reduce hallucination
 * of plausible-but-wrong moves.
 *
 * Run multiple times to test consistency.
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
const MAX_TOKENS = Number(process.env.MAX_TOKENS ?? "6000");
const RUNS = Number(process.env.RUNS ?? "1");

function buildPrompt(prevFen: string, legalSans: string[]): string {
  return `Two photographs of the same physical chessboard. The board orientation in the photos is the same; only one chess move was made between them.

Your job: identify exactly which legal move was played.

PREVIOUS POSITION FEN (piece placement only): ${prevFen}

LEGAL MOVES — exactly one of these was played:
${legalSans.join(", ")}

CRITICAL: Work through this carefully. The photos may show the board from an unusual angle or orientation (rotated 0°/90°/180°/270°). Use the printed rank/file labels around the board edges, AND the location of the white vs black pieces, to map each square.

REQUIRED REASONING:
1. State the board orientation in the photo (which image edge corresponds to a-file, h-file, rank 1, rank 8).
2. In IMAGE 1 (before), identify the location of EACH piece visible. Verify this matches the previous FEN above.
3. In IMAGE 2 (after), identify the location of EACH piece visible.
4. List ONLY the squares whose contents differ between IMAGE 1 and IMAGE 2 (e.g. "e2 had white pawn, now empty; e4 was empty, now has white pawn").
5. Pick the unique legal move from the list whose result matches that change.

OUTPUT FORMAT:
After your reasoning, end your response with a line containing ONLY the SAN, on its own. Like:

ANSWER: e4

(replace e4 with the actual move). The ANSWER line must be the very last line.`;
}

function extractAnswer(raw: string, legalSans: string[]): string | null {
  const lines = raw.split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i--) {
    const m = lines[i].match(/ANSWER:\s*([A-Za-z0-9+#=\-]+)/);
    if (m) {
      const candidate = m[1].replace(/[+#]+$/g, "");
      for (const san of legalSans) {
        const sanBare = san.replace(/[+#]+$/g, "");
        if (sanBare === candidate || san === m[1]) return san;
      }
      // also allow regex within the answer line
      for (const san of legalSans) {
        const re = new RegExp(
          `(^|[^A-Za-z0-9])${san.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`,
        );
        if (re.test(lines[i])) return san;
      }
    }
  }
  // Fallback: search entire response for any SAN, prefer longest
  const sorted = [...legalSans].sort((a, b) => b.length - a.length);
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
    .jpeg({ quality: 90 })
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

async function runGame(runIdx: number): Promise<string[]> {
  const files = fs
    .readdirSync(PHOTOS_DIR)
    .filter((f) => /\.(jpe?g|png|heic)$/i.test(f))
    .sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }),
    );
  console.log(`\n=== Run ${runIdx + 1} ===`);
  const chess = new Chess();
  const moves: string[] = [];
  for (let i = 1; i < files.length; i++) {
    const before = path.join(PHOTOS_DIR, files[i - 1]);
    const after = path.join(PHOTOS_DIR, files[i]);
    const prevFen = chess.fen().split(/\s+/)[0];
    const legal = (chess.moves({ verbose: true }) as Move[]).map((m) => m.san);
    const t0 = Date.now();
    const { matched, raw } = await identifyMove(before, after, prevFen, legal);
    const ms = Date.now() - t0;
    if (matched) {
      chess.move(matched);
      console.log(`[${i}] ${files[i]} → ${matched} (${ms}ms)`);
      moves.push(matched);
    } else {
      console.log(
        `[${i}] ${files[i]} → UNMATCHED (${ms}ms)\n  raw: ${raw.slice(-300)}`,
      );
      moves.push("?");
      break;
    }
  }
  console.log(`Final PGN: ${chess.pgn()}`);
  return moves;
}

async function main() {
  console.log(`Model: ${MODEL}, max_dim: ${MAX_DIM}, max_tokens: ${MAX_TOKENS}`);
  console.log(`Runs: ${RUNS}\n`);
  const allRuns: string[][] = [];
  for (let r = 0; r < RUNS; r++) {
    allRuns.push(await runGame(r));
  }
  if (RUNS > 1) {
    console.log("\n=== Cross-run consistency ===");
    const max = Math.max(...allRuns.map((r) => r.length));
    for (let i = 0; i < max; i++) {
      const calls = allRuns.map((r) => r[i] ?? "—");
      const allSame = calls.every((c) => c === calls[0]) && calls[0] !== "?";
      console.log(
        `Move ${i + 1}: ${calls.join(" | ")}  ${allSame ? "✓" : "DIVERGENT"}`,
      );
    }
  }
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(2);
});
