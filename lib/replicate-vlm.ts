/**
 * Per-move identifier — Claude Sonnet 4.6 via the /replicate/vlm worker.
 *
 * The worker proxies to `https://api.replicate.com/v1/models/${model}/
 * predictions` with `Prefer: wait`, so this call returns the model's
 * final output in a single round-trip (~1-2 s for Sonnet on a small
 * board crop). The browser never sees the Replicate token.
 *
 * Input we send the model: the *rectified* AFTER-move board (warped
 * top-down, white at the bottom), the previous FEN, and the legal-moves
 * list. We deliberately do NOT send the BEFORE image — the FEN encodes
 * it perfectly, and one image is cheaper + faster than two. The model's
 * job is reduced to: "which of these N legal moves was played, given
 * this picture of the resulting position?"
 *
 * The CV pipeline (`inferMoveFuzzy` in `lib/move-inference.ts`) remains
 * the local fallback when this call fails or returns an unparseable
 * SAN — callers chain the two.
 */

import { ensureWhiteAtBottom } from "./board-image";

export type ReplicateVlmInput = {
  /** Rectified warped board, AFTER the move. */
  afterCanvas: HTMLCanvasElement;
  /** Position before the move (piece-placement FEN is enough). */
  previousFen: string;
  /** Legal moves in SAN, exactly one of which was played. */
  legalMovesSan: string[];
  /** Worker root, e.g. https://chesspar-vlm.<account>.workers.dev. */
  proxyUrl: string;
  /** Optional override; defaults to anthropic/claude-sonnet-4-6. */
  model?: string;
  /** Optional override for max_tokens. Default 256 — output is just a SAN. */
  maxTokens?: number;
};

export type ReplicateVlmResult =
  | { kind: "matched"; san: string; raw: string; model: string }
  | { kind: "rejected"; raw: string; reason: string; model: string }
  | { kind: "error"; reason: string };

const DEFAULT_MODEL = "anthropic/claude-sonnet-4-6";

export async function identifyMoveViaReplicateVlm(
  input: ReplicateVlmInput,
): Promise<ReplicateVlmResult> {
  if (!input.proxyUrl) {
    return { kind: "error", reason: "no proxyUrl configured" };
  }
  if (input.legalMovesSan.length === 0) {
    return { kind: "error", reason: "no legal moves supplied" };
  }
  if (input.legalMovesSan.length === 1) {
    return {
      kind: "matched",
      san: input.legalMovesSan[0],
      raw: "(single legal move — no VLM call needed)",
      model: input.model ?? DEFAULT_MODEL,
    };
  }

  // White-at-bottom is the canonical chess view; the model is wildly
  // more reliable on this orientation than on a sideways or upside-down
  // rectified board (a real mis-calibration failure mode). The FEN-aware
  // check inside `ensureWhiteAtBottom` rotates by piece distribution, so
  // a botched corner ordering still yields a correctly-oriented image.
  const { oriented } = ensureWhiteAtBottom(
    input.afterCanvas,
    input.previousFen,
  );
  const dataUrl = oriented.toDataURL("image/jpeg", 0.9);

  const prompt = buildPrompt(input.previousFen, input.legalMovesSan);
  const model = input.model ?? DEFAULT_MODEL;
  const endpoint = input.proxyUrl.replace(/\/$/, "") + "/replicate/vlm";

  let resp: Response;
  try {
    resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        input: {
          prompt,
          image: dataUrl,
          max_tokens: input.maxTokens ?? 256,
        },
      }),
    });
  } catch (e) {
    return {
      kind: "error",
      reason: e instanceof Error ? e.message : String(e),
    };
  }
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    return {
      kind: "error",
      reason: `proxy HTTP ${resp.status}: ${text.slice(0, 200)}`,
    };
  }

  let body: ReplicatePredictionBody;
  try {
    body = (await resp.json()) as ReplicatePredictionBody;
  } catch (e) {
    return {
      kind: "error",
      reason: `bad JSON from worker: ${e instanceof Error ? e.message : e}`,
    };
  }
  if (body.status === "failed" || body.error) {
    return {
      kind: "error",
      reason: body.error ?? "replicate prediction failed",
    };
  }
  const raw = extractText(body.output).trim();
  if (!raw) {
    return { kind: "error", reason: "empty output from model" };
  }
  const matched = pickSan(raw, input.legalMovesSan);
  if (matched) {
    return { kind: "matched", san: matched, raw, model };
  }
  return {
    kind: "rejected",
    raw,
    reason: "response did not contain a legal SAN",
    model,
  };
}

function buildPrompt(prevFen: string, legalMovesSan: string[]): string {
  return `You are identifying a single chess move from a photograph of the resulting position.

PREVIOUS POSITION (piece placement FEN): ${prevFen}

The IMAGE shows the chessboard AFTER one chess move was played. The board is in standard view: WHITE pieces are at the BOTTOM (rank 1, 2), BLACK pieces are at the TOP (rank 7, 8). File a is on the LEFT, file h is on the RIGHT.

LEGAL MOVES — exactly ONE of these was played:
${legalMovesSan.join(", ")}

Identify which of those legal moves produced the position shown. Reasoning:
1. Compare the FEN's position to what you see in the image — which 1-4 squares differ?
2. Pick the unique legal move from the list whose result matches that change.

Reply with ONLY the SAN of the move, exactly as written in the legal-moves list (e.g. "e4", "Nxf3", "O-O", "exd5"). No preamble, no markdown, no quotes — just the SAN on a single line.`;
}

function pickSan(raw: string, legalMovesSan: string[]): string | null {
  const trimmed = raw.trim();
  // Prefer a clean last-line match (the prompt asks for SAN-on-a-single-line).
  const lines = trimmed
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  const sortedByLen = [...legalMovesSan].sort((a, b) => b.length - a.length);
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i];
    if (line === "") continue;
    for (const san of sortedByLen) {
      if (line === san) return san;
    }
    for (const san of sortedByLen) {
      const re = new RegExp(
        `(^|[^A-Za-z0-9])${escapeRegExp(san)}([^A-Za-z0-9]|$)`,
      );
      if (re.test(line)) return san;
    }
  }
  // Final pass: search the whole response.
  for (const san of sortedByLen) {
    const re = new RegExp(
      `(^|[^A-Za-z0-9])${escapeRegExp(san)}([^A-Za-z0-9]|$)`,
    );
    if (re.test(trimmed)) return san;
  }
  // Match without check/mate decorations as last resort.
  const bare = trimmed.replace(/[+#]+/g, "");
  for (const san of sortedByLen) {
    const sanBare = san.replace(/[+#]+/g, "");
    if (sanBare && bare.endsWith(sanBare)) return san;
  }
  return null;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

type ReplicatePredictionBody = {
  status?: string;
  error?: string | null;
  output?: unknown;
};

/**
 * The Replicate Anthropic models can return their text output in several
 * shapes depending on the model version:
 *   - plain string
 *   - array of strings (one per generation step)
 *   - object with a `text` field
 * We handle all three so a model-version bump doesn't break parsing.
 */
function extractText(output: unknown): string {
  if (typeof output === "string") return output;
  if (Array.isArray(output)) {
    return output.map((o) => extractText(o)).join("").trim();
  }
  if (output && typeof output === "object") {
    const o = output as Record<string, unknown>;
    if (typeof o.text === "string") return o.text;
    if (typeof o.output === "string") return o.output;
  }
  return "";
}
