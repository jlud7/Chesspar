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
  | {
      kind: "matched";
      san: string;
      raw: string;
      observations: CellObservation[];
      via: "deterministic" | "answer-line" | "san-search";
      /** Match details when via === "deterministic". */
      matchDetails?: { matches: number; conflicts: number; margin: number };
    }
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
      via: "deterministic",
    };
  }

  const disputed = findDisputedSquares(args.prevFen, args.candidatesSan);
  if (disputed.length === 0) {
    return {
      kind: "matched",
      san: args.candidatesSan[0],
      raw: "(no disputed squares — candidates agree on after-state)",
      observations: [],
      via: "deterministic",
    };
  }

  // CRITICAL: Re-orient the rectified board to white-at-bottom BEFORE
  // cropping any tiles. VLMs are far more reliable on the canonical
  // chess viewing orientation; tiles cropped from a sideways/upside-down
  // warped board produce confidently wrong piece identifications. Pass
  // `prevFen` so the orientation check uses the FEN's actual piece
  // distribution rather than a generic luminance heuristic.
  const { oriented: afterOriented, rotationDeg } = ensureWhiteAtBottom(
    args.afterWarped,
    args.prevFen,
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
  return parseResponse(raw, args.candidatesSan, args.prevFen);
}

/**
 * Squares where the candidates' predicted after-states actually disagree
 * (by piece type — empty vs occupied, or knight vs bishop, etc.).
 *
 * We deliberately exclude squares all candidates agree on, even when
 * they're a candidate's source or destination — asking the VLM about
 * uncontested cells just wastes tokens and risks the model saying
 * something contradictory that pulls a clean pick into ambiguity.
 *
 * Ordering: cells where the most candidates disagree (highest cardinality
 * of distinct predicted piece-types) come first, so if we hit the cap the
 * most informative tiles survive. Cap is 8 — empirically enough to
 * decide every back-rank confusion case in the 14-move test set.
 */
export function findDisputedSquares(
  prevFen: string,
  candidatesSan: string[],
): string[] {
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
    perCandidateBoards.push(fenToBoardWithTypes(sim.fen()));
  }
  if (perCandidateBoards.length < 2) return [];

  const cellScore = new Map<number, number>();
  for (let i = 0; i < 64; i++) {
    const seen = new Set<string | null>();
    for (const b of perCandidateBoards) seen.add(b[i]);
    if (seen.size > 1) cellScore.set(i, seen.size);
  }
  const sorted = [...cellScore.entries()].sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1];
    return a[0] - b[0];
  });
  return sorted.slice(0, 8).map(([idx]) => idxToSan(idx));
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

export function buildPrompt(
  prevFen: string,
  candidatesSan: string[],
  disputed: string[],
): string {
  const candidateSummary = candidatesSan
    .map((san, i) => `  ${i + 1}. ${san}  → ${describeMove(prevFen, san)}`)
    .join("\n");
  const exampleObs = disputed
    .slice(0, 2)
    .map((sq, i) => `  ${sq}: ${i === 0 ? "black-bishop" : "empty"}`)
    .join("\n");
  return `You are identifying a chess move by reading specific squares on the AFTER-move board.

PREVIOUS POSITION (piece placement FEN): ${prevFen}

I will show you ${disputed.length} close-up tile${disputed.length === 1 ? "" : "s"}, each a small region of the AFTER-the-move board with ONE square OUTLINED IN RED. The red-outlined square is the one you must identify. The other 8 surrounding squares in the tile are just CONTEXT — do NOT report on them.

Tiles, in order: ${disputed.join(", ")}.

CANDIDATE MOVES (exactly ONE was played, ranked best-first):
${candidateSummary}

OUTPUT FORMAT — follow this STRICTLY:

For each tile, output EXACTLY one line in this form:
  <square>: <label>

Where <label> is EXACTLY one of:
  empty
  white-pawn, white-knight, white-bishop, white-rook, white-queen, white-king
  black-pawn, black-knight, black-bishop, black-rook, black-queen, black-king

If you are confident in the colour but not the piece type, use:
  white-piece  or  black-piece

Examples for two tiles:
${exampleObs}

IMPORTANT RULES for what counts as "on" the red square:
  - Only what is BASED on the red square counts. A piece's tall top may extend above the red square from the rank behind — DO NOT count that piece as being on the red square; the base of the piece tells you which square it sits on.
  - Look at where the bottom/base of each piece touches the board.
  - The red outline marks the EXACT cell of interest; ignore neighbouring pieces unless their base is INSIDE the outline.

After all ${disputed.length} observation lines, output exactly one final line:
  ANSWER: <san>

Where <san> is one of the candidates above written exactly as listed.`;
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

/**
 * Parse VLM output and pick the matching candidate DETERMINISTICALLY from
 * its per-square observations. The model's job is the easy part (identify
 * what's on each red-outlined square); we do the candidate-matching in
 * code. This decouples vision from decision: the model can identify
 * pieces correctly but pick the wrong SAN — happens often enough on tight
 * candidate sets — and our code still arrives at the right answer.
 *
 * Fallback hierarchy:
 *   1. Match candidates to per-square observations (preferred).
 *   2. Honour the VLM's explicit ANSWER line (when 1 is inconclusive).
 *   3. Last-resort: search for any candidate SAN in the response text.
 */
export function parseResponse(
  raw: string,
  candidatesSan: string[],
  prevFen: string,
): CellVerifyResult {
  const trimmed = raw.trim();
  const observations = parseObservations(trimmed);

  // --- Path 1: deterministic match from observations -------------------
  const deterministic = pickByObservations(observations, candidatesSan, prevFen);
  if (deterministic) {
    return {
      kind: "matched",
      san: deterministic.san,
      raw,
      observations,
      via: "deterministic",
      matchDetails: {
        matches: deterministic.matches,
        conflicts: deterministic.conflicts,
        margin: deterministic.margin,
      },
    };
  }

  // --- Path 2: ANSWER: <san> line --------------------------------------
  const answerMatch = trimmed.match(/ANSWER\s*[:=]\s*([A-Za-z0-9+#=\-]+)/i);
  if (answerMatch) {
    const guess = answerMatch[1];
    for (const san of candidatesSan) {
      if (san === guess) {
        return { kind: "matched", san, raw, observations, via: "answer-line" };
      }
      if (san.replace(/[+#]+$/, "") === guess.replace(/[+#]+$/, "")) {
        return { kind: "matched", san, raw, observations, via: "answer-line" };
      }
    }
  }

  // --- Path 3: any candidate SAN appears anywhere ----------------------
  const lines = trimmed.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const last = lines[lines.length - 1] ?? "";
  const byLength = [...candidatesSan].sort((a, b) => b.length - a.length);
  for (const san of byLength) {
    const re = new RegExp(`(^|[^A-Za-z0-9])${escapeRegExp(san)}([^A-Za-z0-9]|$)`);
    if (re.test(last)) {
      return { kind: "matched", san, raw, observations, via: "san-search" };
    }
  }
  for (const san of byLength) {
    const re = new RegExp(`(^|[^A-Za-z0-9])${escapeRegExp(san)}([^A-Za-z0-9]|$)`);
    if (re.test(trimmed)) {
      return { kind: "matched", san, raw, observations, via: "san-search" };
    }
  }
  return {
    kind: "rejected",
    raw,
    reason: "response did not match any candidate",
    observations,
  };
}

/**
 * Pull per-square observations out of a VLM response. Tolerant to wrappers
 * like "Square e7: ..." or " - e7 = black bishop." — we just scan each
 * line for a `<file><rank> <separator> <description>` pattern.
 */
function parseObservations(raw: string): CellObservation[] {
  const observations: CellObservation[] = [];
  const seen = new Set<string>();
  for (const line of raw.split(/\r?\n/)) {
    const m = line.match(
      /\b([a-h])([1-8])\b\s*[:=\-—]+\s*([A-Za-z][A-Za-z0-9 \-_/]+?)(?:[.,;]|$)/i,
    );
    if (!m) continue;
    const sq = (m[1] + m[2]).toLowerCase();
    if (seen.has(sq)) continue; // first mention wins
    seen.add(sq);
    observations.push({ square: sq, piece: m[3].trim() });
  }
  return observations;
}

type ParsedPiece =
  | { kind: "empty" }
  | { kind: "exact"; color: "w" | "b"; type: "p" | "n" | "b" | "r" | "q" | "k" }
  | { kind: "color"; color: "w" | "b" };

function parsePieceObservation(s: string): ParsedPiece | undefined {
  const t = s.toLowerCase().trim().replace(/[_/]+/g, "-");
  if (/^(empty|none|nothing|vacant|no\s*piece|blank)$/.test(t)) {
    return { kind: "empty" };
  }
  const colorMatch = t.match(/\b(white|black)\b/);
  if (!colorMatch) return undefined;
  const color = colorMatch[1] === "white" ? "w" : "b";
  const pieceMatch = t.match(/\b(pawn|knight|bishop|rook|queen|king)\b/);
  if (pieceMatch) {
    const map: Record<string, "p" | "n" | "b" | "r" | "q" | "k"> = {
      pawn: "p",
      knight: "n",
      bishop: "b",
      rook: "r",
      queen: "q",
      king: "k",
    };
    return { kind: "exact", color, type: map[pieceMatch[1]] };
  }
  return { kind: "color", color };
}

function pieceMatches(predicted: string | null, observed: ParsedPiece): boolean {
  if (observed.kind === "empty") return predicted === null;
  if (predicted === null) return false;
  if (observed.kind === "color") return predicted[0] === observed.color;
  return predicted[0] === observed.color && predicted[1] === observed.type;
}

/**
 * For each candidate, score how many observed squares its predicted
 * after-state matches. Returns the candidate with the highest net score
 * (matches − conflicts) when there's a CLEAR winner.
 *
 * A clear winner means:
 *   - at least 2 observed squares match its prediction, AND
 *   - matches > conflicts (more right than wrong), AND
 *   - score advantage over the runner-up is at least 2.
 *
 * The margin requirement is deliberately conservative: this matcher only
 * fires when it can override the upstream CV pick with high confidence.
 * Anything murkier falls through to the ANSWER-line and SAN-search paths.
 *
 * Why 2 for both: each observation that disagrees moves the score by 2
 * (one less match, one more conflict). A margin of 2 ≈ "off by one
 * observation" — robust to a single mis-ID by the VLM.
 */
function pickByObservations(
  observations: CellObservation[],
  candidatesSan: string[],
  prevFen: string,
): {
  san: string;
  matches: number;
  conflicts: number;
  margin: number;
} | null {
  if (observations.length === 0) return null;
  const parsed = new Map<string, ParsedPiece>();
  for (const obs of observations) {
    const p = parsePieceObservation(obs.piece);
    if (p) parsed.set(obs.square, p);
  }
  if (parsed.size === 0) return null;

  const scored: { san: string; score: number; matches: number; conflicts: number }[] = [];
  for (const san of candidatesSan) {
    const sim = new Chess(prevFen);
    let m;
    try {
      m = sim.move(san);
    } catch {
      continue;
    }
    if (!m) continue;
    const board = fenToBoardWithTypes(sim.fen());
    let matches = 0;
    let conflicts = 0;
    for (const [sq, observed] of parsed) {
      const idx = sanToIdx(sq);
      const predicted = board[idx];
      if (pieceMatches(predicted, observed)) matches++;
      else conflicts++;
    }
    scored.push({ san, score: matches - conflicts, matches, conflicts });
  }
  if (scored.length === 0) return null;
  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  const next = scored[1];
  const margin = next ? best.score - next.score : Infinity;
  if (best.matches < 2) return null;
  if (best.matches <= best.conflicts) return null;
  if (margin < 2) return null;
  return {
    san: best.san,
    matches: best.matches,
    conflicts: best.conflicts,
    margin,
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
