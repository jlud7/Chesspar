/**
 * Per-cell VLM verifier for resolving back-rank piece-type ambiguity.
 *
 * The full-photo VLM path hallucinates on tilted-angle photos, and the
 * occupancy-only CV scorer can't distinguish moves that produce identical
 * empty/white/black diffs but differ in piece type (e.g. Be7 vs Ne7 —
 * both put a black piece on e7).
 *
 * This verifier uses the *rectified* (warped) board CV already produces
 * and sends the VLM small 3x3-cell tiles around just the squares where
 * the top candidates disagree on piece type. Each tile is a focused
 * close-up with the target cell outlined; the VLM's job collapses to
 * identifying what piece (if any) sits on each highlighted square, then
 * picking the candidate whose predicted after-state matches.
 *
 * Why this works:
 *  - Rectified tiles remove tilt + parallax, the main source of full-
 *    photo confusion.
 *  - Showing the 3x3 neighbourhood gives context for piece-top bleed
 *    without losing the per-square focus.
 *  - The BEFORE state is supplied via FEN (the model already knows it);
 *    only AFTER tiles need to be sent, halving the payload.
 *  - Candidate list is short and constrained — model is classifying,
 *    not reasoning.
 */

import { Chess } from "chess.js";
import { ensureWhiteAtBottom } from "./board-image";
import type { VlmProvider } from "./vlm";

const FILES = "abcdefgh";

export type CellVerifyArgs = {
  /** Rectified board, AFTER the move. a8 top-left, h1 bottom-right. */
  afterWarped: HTMLCanvasElement;
  /** Position before the move (full FEN). */
  prevFen: string;
  /** Top-K candidate moves in SAN, ranked best-first by CV. */
  candidatesSan: string[];
  /** Provider to use. */
  provider: VlmProvider;
  /** Direct API key (browser path). */
  apiKey?: string;
  /** Proxy URL (production path — worker holds the key). */
  proxyUrl?: string;
  /** Override model. Defaults: claude-opus-4-7 / gpt-5. */
  model?: string;
};

export type CellVerifyResult =
  | { kind: "matched"; san: string; raw: string; observations: CellObservation[] }
  | { kind: "rejected"; raw: string; reason: string; observations: CellObservation[] }
  | { kind: "error"; reason: string };

export type CellObservation = {
  square: string;
  /** The VLM's stated piece on this square in the AFTER image. */
  piece: string;
};

/**
 * Top-level entry. Builds the disputed-cell tile gallery, dispatches to
 * the configured provider, parses the response.
 */
export async function verifyByCellTiles(
  args: CellVerifyArgs,
): Promise<CellVerifyResult> {
  if (args.candidatesSan.length === 0) {
    return { kind: "error", reason: "no candidates" };
  }
  if (args.candidatesSan.length === 1) {
    return {
      kind: "matched",
      san: args.candidatesSan[0],
      raw: "(single candidate — no VLM call needed)",
      observations: [],
    };
  }

  const disputed = findDisputedSquares(args.prevFen, args.candidatesSan);
  if (disputed.length === 0) {
    return {
      kind: "matched",
      san: args.candidatesSan[0],
      raw: "(no disputed squares — candidates agree on after-state)",
      observations: [],
    };
  }

  // CRITICAL: Re-orient the rectified board to white-at-bottom BEFORE
  // cropping any tiles. VLMs are far more reliable on the canonical
  // chess viewing orientation; tiles cropped from a sideways/upside-down
  // warped board produce confidently wrong piece identifications.
  // Mis-calibrated corners (board captured at 90° from the player's
  // perspective, or tapped in the wrong order) would otherwise leak
  // through to the API call.
  const { oriented: afterOriented, rotationDeg } = ensureWhiteAtBottom(
    args.afterWarped,
  );
  if (rotationDeg !== 0 && typeof console !== "undefined") {
    console.warn(
      `verifyByCellTiles: rotated rectified board by ${rotationDeg}° to put white at bottom before VLM call`,
    );
  }

  const tiles: { square: string; canvas: HTMLCanvasElement }[] = [];
  for (const sq of disputed) {
    tiles.push({ square: sq, canvas: renderTile(afterOriented, sq) });
  }

  const prompt = buildPrompt(args.prevFen, args.candidatesSan, disputed);
  const dispatch = pickDispatch(args);
  const raw = await dispatch(prompt, tiles);
  return parseResponse(raw, args.candidatesSan);
}

/**
 * Squares where at least two candidates disagree about piece occupancy
 * in the AFTER state (using piece type, not just colour). Also includes
 * each candidate's source/destination cells so the VLM sees every action
 * square. Capped at 8 to keep the prompt + payload bounded.
 */
export function findDisputedSquares(
  prevFen: string,
  candidatesSan: string[],
): string[] {
  const interesting = new Set<number>();
  const perCandidateBoards: ((string | null)[])[] = [];
  for (const san of candidatesSan) {
    const sim = new Chess(prevFen);
    let move;
    try {
      move = sim.move(san);
    } catch {
      continue;
    }
    if (!move) continue;
    interesting.add(sanToIdx(move.from));
    interesting.add(sanToIdx(move.to));
    if (move.captured && move.flags.includes("e")) {
      // En-passant: capture square differs from `to`.
      const capRank = move.color === "w" ? "5" : "4";
      interesting.add(sanToIdx(move.to[0] + capRank));
    }
    perCandidateBoards.push(fenToBoardWithTypes(sim.fen()));
  }
  if (perCandidateBoards.length > 1) {
    for (let i = 0; i < 64; i++) {
      const seen = new Set<string | null>();
      for (const b of perCandidateBoards) seen.add(b[i]);
      if (seen.size > 1) interesting.add(i);
    }
  }
  return [...interesting].slice(0, 8).map(idxToSan);
}

/** Render a 3x3-cell context tile around `square`, target cell outlined. */
export function renderTile(
  warped: HTMLCanvasElement,
  square: string,
): HTMLCanvasElement {
  const size = warped.width;
  const cell = size / 8;
  const idx = sanToIdx(square);
  const row = Math.floor(idx / 8);
  const col = idx % 8;
  // Source rect: 3x3 cells centred on target, clamped to image.
  const sx0 = Math.max(0, (col - 1) * cell);
  const sy0 = Math.max(0, (row - 1) * cell);
  const sx1 = Math.min(size, (col + 2) * cell);
  const sy1 = Math.min(size, (row + 2) * cell);
  const sw = sx1 - sx0;
  const sh = sy1 - sy0;
  // Output upscaled so the VLM sees a generous close-up.
  const scale = 3; // 3x cells × 48 px × 3 = 432 px tile (board size dependent).
  const out = document.createElement("canvas");
  out.width = Math.round(sw * scale);
  out.height = Math.round(sh * scale);
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("renderTile: failed to get 2D context");
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(warped, sx0, sy0, sw, sh, 0, 0, out.width, out.height);
  // Outline the target cell in red.
  const tx0 = (col * cell - sx0) * scale;
  const ty0 = (row * cell - sy0) * scale;
  ctx.strokeStyle = "#ff2a2a";
  ctx.lineWidth = Math.max(3, Math.round(cell * scale * 0.06));
  ctx.strokeRect(tx0, ty0, cell * scale, cell * scale);
  // Square-name label, top-left of the highlight.
  ctx.font = `bold ${Math.round(cell * scale * 0.28)}px sans-serif`;
  ctx.fillStyle = "#ff2a2a";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 3;
  const labelX = tx0 + 4;
  const labelY = ty0 + Math.round(cell * scale * 0.3);
  ctx.strokeText(square, labelX, labelY);
  ctx.fillText(square, labelX, labelY);
  return out;
}

function buildPrompt(
  prevFen: string,
  candidatesSan: string[],
  disputed: string[],
): string {
  const candidateSummary = candidatesSan
    .map((san, i) => `  ${i + 1}. ${san}  → ${describeMove(prevFen, san)}`)
    .join("\n");
  return `You are identifying a chess move by checking specific squares on the AFTER-move board.

PREVIOUS POSITION (piece placement FEN): ${prevFen}

I will show you ${disputed.length} close-up tile${disputed.length === 1 ? "" : "s"}, each a small region of the AFTER-the-move board with one square OUTLINED IN RED. The red-outlined square is the one you must identify. The other squares in the tile are just context (the immediate neighbours of the target square).

Tiles, in order: ${disputed.join(", ")}.

CANDIDATE MOVES (exactly ONE of these was played, ranked best-first by the upstream classifier):
${candidateSummary}

Your task:
  Step 1. For each tile, look ONLY at the red-outlined square and write a one-line observation in the form
            "${disputed[0]}: <empty | white-pawn | white-knight | white-bishop | white-rook | white-queen | white-king | black-pawn | black-knight | black-bishop | black-rook | black-queen | black-king>"
          If you cannot tell the exact piece type but can tell the colour, write "white-piece" or "black-piece". If a piece's top extends above the red square from an adjacent rank (3D piece-top bleed), do not count it — only what is *based* on the red square counts.

  Step 2. Cross-reference your observations with each candidate's predicted after-state. Eliminate any candidate whose prediction disagrees with what you saw.

  Step 3. Output the SAN of the surviving move.

Format your final line EXACTLY as:
ANSWER: <san>

The SAN must be one of the candidates above, written exactly as listed.`;
}

function describeMove(prevFen: string, san: string): string {
  const sim = new Chess(prevFen);
  let move;
  try {
    move = sim.move(san);
  } catch {
    return "?";
  }
  if (!move) return "?";
  const color = move.color === "w" ? "white" : "black";
  const pieceName = pieceTypeName(move.piece);
  const extra = move.captured ? ` (captures ${pieceTypeName(move.captured)})` : "";
  return `${color} ${pieceName} ${move.from}→${move.to}${extra}; after move expect ${move.to} = ${color}-${pieceName}, ${move.from} = empty`;
}

function pieceTypeName(t: string): string {
  switch (t) {
    case "p": return "pawn";
    case "n": return "knight";
    case "b": return "bishop";
    case "r": return "rook";
    case "q": return "queen";
    case "k": return "king";
    default: return t;
  }
}

type DispatchFn = (
  prompt: string,
  tiles: { square: string; canvas: HTMLCanvasElement }[],
) => Promise<string>;

function pickDispatch(args: CellVerifyArgs): DispatchFn {
  const model =
    args.model ??
    (args.provider === "openai" ? "gpt-5" : "claude-opus-4-7");
  if (args.proxyUrl) {
    if (args.provider === "openai") {
      return openAiDispatch(
        args.proxyUrl.replace(/\/$/, "") + "/openai",
        {},
        model,
      );
    }
    return anthropicDispatch(
      args.proxyUrl.replace(/\/$/, "") + "/verify",
      {},
      model,
    );
  }
  if (!args.apiKey) {
    throw new Error("verifyByCellTiles: no apiKey or proxyUrl supplied");
  }
  if (args.provider === "openai") {
    return openAiDispatch(
      "https://api.openai.com/v1/chat/completions",
      { Authorization: `Bearer ${args.apiKey}` },
      model,
    );
  }
  if (args.provider === "anthropic") {
    return anthropicDispatch(
      "https://api.anthropic.com/v1/messages",
      {
        "x-api-key": args.apiKey,
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
      },
      model,
    );
  }
  throw new Error(`verifyByCellTiles: unsupported provider ${args.provider}`);
}

function anthropicDispatch(
  url: string,
  headers: Record<string, string>,
  model: string,
): DispatchFn {
  return async (prompt, tiles) => {
    const content: unknown[] = [];
    for (const tile of tiles) {
      content.push({
        type: "image",
        source: {
          type: "base64",
          media_type: "image/jpeg",
          data: canvasToBase64(tile.canvas),
        },
      });
    }
    content.push({ type: "text", text: prompt });
    const isOpus = model.includes("opus");
    const body: Record<string, unknown> = {
      model,
      max_tokens: isOpus ? 4000 : 256,
      messages: [{ role: "user", content }],
    };
    if (!isOpus) body.temperature = 0;
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`Anthropic HTTP ${resp.status}: ${text.slice(0, 200)}`);
    }
    const data = (await resp.json()) as {
      content?: { type: string; text?: string }[];
    };
    return data.content?.find((c) => c.type === "text")?.text ?? "";
  };
}

function openAiDispatch(
  url: string,
  headers: Record<string, string>,
  model: string,
): DispatchFn {
  return async (prompt, tiles) => {
    const content: unknown[] = [];
    for (const tile of tiles) {
      content.push({
        type: "image_url",
        image_url: {
          url: `data:image/jpeg;base64,${canvasToBase64(tile.canvas)}`,
        },
      });
    }
    content.push({ type: "text", text: prompt });
    const isGpt5 = model.startsWith("gpt-5");
    const body: Record<string, unknown> = {
      model,
      messages: [{ role: "user", content }],
    };
    if (isGpt5) body.max_completion_tokens = 4000;
    else {
      body.max_tokens = 256;
      body.temperature = 0;
    }
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`OpenAI HTTP ${resp.status}: ${text.slice(0, 200)}`);
    }
    const data = (await resp.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    return data.choices?.[0]?.message?.content ?? "";
  };
}

function parseResponse(
  raw: string,
  candidatesSan: string[],
): CellVerifyResult {
  const trimmed = raw.trim();
  const observations: CellObservation[] = [];
  for (const line of trimmed.split(/\r?\n/)) {
    const m = line.match(/^\s*([a-h][1-8])\s*[:=]\s*(.+?)\s*$/i);
    if (m) observations.push({ square: m[1].toLowerCase(), piece: m[2].trim() });
  }
  const answerMatch = trimmed.match(/ANSWER\s*[:=]\s*([A-Za-z0-9+#=\-]+)/i);
  if (answerMatch) {
    const guess = answerMatch[1];
    for (const san of candidatesSan) {
      if (san === guess) return { kind: "matched", san, raw, observations };
      if (san.replace(/[+#]+$/, "") === guess.replace(/[+#]+$/, "")) {
        return { kind: "matched", san, raw, observations };
      }
    }
  }
  // Fallback: search for any candidate SAN on the last non-empty line.
  const lines = trimmed.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const last = lines[lines.length - 1] ?? "";
  const byLength = [...candidatesSan].sort((a, b) => b.length - a.length);
  for (const san of byLength) {
    const re = new RegExp(`(^|[^A-Za-z0-9])${escapeRegExp(san)}([^A-Za-z0-9]|$)`);
    if (re.test(last)) return { kind: "matched", san, raw, observations };
  }
  for (const san of byLength) {
    const re = new RegExp(`(^|[^A-Za-z0-9])${escapeRegExp(san)}([^A-Za-z0-9]|$)`);
    if (re.test(trimmed)) return { kind: "matched", san, raw, observations };
  }
  return {
    kind: "rejected",
    raw,
    reason: "response did not match any candidate",
    observations,
  };
}

function fenToBoardWithTypes(fen: string): (string | null)[] {
  const game = new Chess(fen);
  const board = game.board();
  const out: (string | null)[] = [];
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const c = board[r][f];
      out.push(c ? `${c.color}${c.type}` : null);
    }
  }
  return out;
}

function sanToIdx(square: string): number {
  const file = FILES.indexOf(square[0]);
  const rank = Number(square[1]);
  return (8 - rank) * 8 + file;
}

function idxToSan(idx: number): string {
  return `${FILES[idx % 8]}${8 - Math.floor(idx / 8)}`;
}

function canvasToBase64(canvas: HTMLCanvasElement): string {
  const url = canvas.toDataURL("image/jpeg", 0.9);
  const comma = url.indexOf(",");
  return comma >= 0 ? url.slice(comma + 1) : url;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
