/**
 * Vision-LM identifier for chess moves.
 *
 * The current design hands the model BOTH the rectified before-frame AND
 * after-frame plus the previous FEN and the full list of legal moves. The
 * model picks one — a constrained classification problem against ~40
 * candidates rather than open-ended chess reasoning. Two-image diff makes
 * it trivial for the model: "which of these legal moves explains this
 * before→after change?"
 *
 * All callers go through the swappable `VlmVerifier` shape so we can
 * A/B different providers on the same input.
 *
 * Every verifier funnels its rectified board through `ensureWhiteAtBottom`
 * before serialising to base64. Mis-calibrated corners (board rotated in
 * the camera frame, or four-tap done in the wrong order) would otherwise
 * leak a non-canonical orientation into the model, which is the single
 * worst input we can give it.
 */

import { ensureWhiteAtBottom } from "./board-image";

export type VlmVerifyInput = {
  previousFen: string;
  legalMovesSan: string[];
  /** Rectified top-down board image (a8 top-left, h1 bottom-right). */
  boardImage: HTMLCanvasElement;
  /** Rectified board image of the state BEFORE the move, if available. */
  previousBoardImage?: HTMLCanvasElement;
};

export type VlmVerifyResult =
  | { kind: "matched"; san: string; raw: string }
  | { kind: "rejected"; raw: string; reason: string }
  | { kind: "error"; reason: string };

export type VlmProvider = "gemini" | "openai" | "anthropic";

export interface VlmVerifier {
  readonly provider: VlmProvider;
  verify(input: VlmVerifyInput): Promise<VlmVerifyResult>;
}

const SINGLE_FRAME_PROMPT = (
  prevFen: string,
  legalMovesSan: string[],
): string => `You are identifying a chess move from a photo.

PREVIOUS POSITION FEN (piece placement only): ${prevFen}

LEGAL MOVES (the move that happened is exactly one of these):
${legalMovesSan.join(", ")}

The photo may be taken from any angle and the board may be rotated 0°, 90°, 180°, or 270° in the frame. Use the rank/file labels printed on the board edges, or the location of the white vs black pieces, to orient yourself.

Reply with ONLY the SAN notation of the move that happened, exactly as it appears in the legal-moves list above (e.g. "e4", "Nxf3", "O-O"). No explanation, no preamble, no markdown, no quotes — just the SAN on the last line.`;

const TWO_FRAME_PROMPT = (
  prevFen: string,
  legalMovesSan: string[],
): string => `Identify the chess move played between these two photos.

The photos show a chessboard with WHITE PIECES at the BOTTOM and BLACK PIECES at the TOP (standard chess view). Rank 1 is at the bottom, rank 8 at the top. File a is leftmost, file h is rightmost.

IMAGE 1: the board BEFORE the move.
IMAGE 2: the board AFTER the move.

PREVIOUS POSITION FEN (piece placement only): ${prevFen}

LEGAL MOVES — exactly one of these was played:
${legalMovesSan.join(", ")}

Find which 1-3 squares look different between IMAGE 1 and IMAGE 2. Identify which piece moved (by color and type — pawn, knight, bishop, rook, queen, king) and from which exact square to which exact square. Pay close attention to subtle differences: a piece on b3 vs d4 (different file), a knight on d4 vs d5 (different rank), a bishop on e7 vs g4 (different diagonal entirely).

Pick the unique legal move from the list whose result produces those changes. Reply with ONLY that move's SAN, exactly as written in the list (e.g. "e4", "Nxf3", "O-O"). No preamble, no markdown — just the SAN.`;

function makeGeminiVerifierFromUrl(url: string, model: string): VlmVerifier {
  return {
    provider: "gemini",
    async verify({ previousFen, legalMovesSan, boardImage, previousBoardImage }) {
      try {
        // Force the canonical chess view (white at bottom) before the
        // API call. Mis-calibrated corners can leak a rotated rectified
        // board into the VLM, which then identifies pieces against an
        // unfamiliar orientation and produces confidently wrong picks.
        // Pass `previousFen` so the FEN-aware orientation check fires.
        const { oriented: orientedAfter } = ensureWhiteAtBottom(
          boardImage,
          previousFen,
        );
        const orientedBefore = previousBoardImage
          ? ensureWhiteAtBottom(previousBoardImage, previousFen).oriented
          : undefined;
        const afterB64 = canvasToBase64(orientedAfter);
        const beforeB64 = orientedBefore
          ? canvasToBase64(orientedBefore)
          : null;
        const prompt = beforeB64
          ? TWO_FRAME_PROMPT(previousFen, legalMovesSan)
          : SINGLE_FRAME_PROMPT(previousFen, legalMovesSan);
        const parts: unknown[] = [{ text: prompt }];
        if (beforeB64) {
          parts.push({
            inline_data: { mime_type: "image/jpeg", data: beforeB64 },
          });
        }
        parts.push({
          inline_data: { mime_type: "image/jpeg", data: afterB64 },
        });
        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ parts }],
            generationConfig: { temperature: 0, maxOutputTokens: 2048 },
          }),
        });
        if (!response.ok) {
          const text = await response.text().catch(() => "");
          return {
            kind: "error",
            reason: `Gemini HTTP ${response.status}: ${text.slice(0, 160)}`,
          };
        }
        const data = (await response.json()) as {
          candidates?: { content?: { parts?: { text?: string }[] } }[];
        };
        const raw = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
        return resolveSan(raw, legalMovesSan);
      } catch (e) {
        return {
          kind: "error",
          reason: e instanceof Error ? e.message : String(e),
        };
      }
    },
  };
}

export function makeGeminiVerifier(
  apiKey: string,
  model = "gemini-2.5-pro",
): VlmVerifier {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`;
  return makeGeminiVerifierFromUrl(url, model);
}

/**
 * Gemini verifier that goes through the Cloudflare Worker /gemini endpoint.
 * The worker holds GEMINI_API_KEY as a secret; the browser never sees it.
 */
export function makeGeminiProxyVerifier(
  proxyUrl: string,
  model = "gemini-2.5-pro",
): VlmVerifier {
  const url = `${proxyUrl.replace(/\/$/, "")}/gemini?model=${encodeURIComponent(model)}`;
  return makeGeminiVerifierFromUrl(url, model);
}

type OpenAiCallArgs = {
  url: string;
  headers: Record<string, string>;
  model: string;
};

function buildOpenAiBody(
  model: string,
  content: unknown[],
): Record<string, unknown> {
  // GPT-5 uses the new "reasoning" parameters and rejects temperature/max_tokens.
  const isGpt5 = model.startsWith("gpt-5");
  const body: Record<string, unknown> = {
    model,
    messages: [{ role: "user", content }],
  };
  if (isGpt5) {
    body.max_completion_tokens = 4000;
  } else {
    body.max_tokens = 64;
    body.temperature = 0;
  }
  return body;
}

function makeOpenAiVerifierFromCall(args: OpenAiCallArgs): VlmVerifier {
  return {
    provider: "openai",
    async verify({ previousFen, legalMovesSan, boardImage, previousBoardImage }) {
      try {
        // Force the canonical chess view (white at bottom) before the
        // API call. Mis-calibrated corners can leak a rotated rectified
        // board into the VLM, which then identifies pieces against an
        // unfamiliar orientation and produces confidently wrong picks.
        // Pass `previousFen` so the FEN-aware orientation check fires.
        const { oriented: orientedAfter } = ensureWhiteAtBottom(
          boardImage,
          previousFen,
        );
        const orientedBefore = previousBoardImage
          ? ensureWhiteAtBottom(previousBoardImage, previousFen).oriented
          : undefined;
        const afterB64 = canvasToBase64(orientedAfter);
        const beforeB64 = orientedBefore
          ? canvasToBase64(orientedBefore)
          : null;
        const prompt = beforeB64
          ? TWO_FRAME_PROMPT(previousFen, legalMovesSan)
          : SINGLE_FRAME_PROMPT(previousFen, legalMovesSan);
        const content: unknown[] = [{ type: "text", text: prompt }];
        if (beforeB64) {
          content.push({
            type: "image_url",
            image_url: { url: `data:image/jpeg;base64,${beforeB64}` },
          });
        }
        content.push({
          type: "image_url",
          image_url: { url: `data:image/jpeg;base64,${afterB64}` },
        });
        const response = await fetch(args.url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...args.headers,
          },
          body: JSON.stringify(buildOpenAiBody(args.model, content)),
        });
        if (!response.ok) {
          const text = await response.text().catch(() => "");
          return {
            kind: "error",
            reason: `OpenAI HTTP ${response.status}: ${text.slice(0, 160)}`,
          };
        }
        const data = (await response.json()) as {
          choices?: { message?: { content?: string } }[];
        };
        const raw = data.choices?.[0]?.message?.content ?? "";
        return resolveSan(raw, legalMovesSan);
      } catch (e) {
        return {
          kind: "error",
          reason: e instanceof Error ? e.message : String(e),
        };
      }
    },
  };
}

export function makeOpenAiVerifier(
  apiKey: string,
  model = "gpt-5",
): VlmVerifier {
  return makeOpenAiVerifierFromCall({
    url: "https://api.openai.com/v1/chat/completions",
    headers: { Authorization: `Bearer ${apiKey}` },
    model,
  });
}

/**
 * OpenAI verifier that goes through the Cloudflare Worker /openai endpoint.
 * The worker holds OPENAI_API_KEY as a secret; the browser never sees it.
 */
export function makeOpenAiProxyVerifier(
  proxyUrl: string,
  model = "gpt-5",
): VlmVerifier {
  return makeOpenAiVerifierFromCall({
    url: proxyUrl.replace(/\/$/, "") + "/openai",
    headers: {},
    model,
  });
}

export function makeAnthropicVerifier(
  apiKey: string,
  model = "claude-opus-4-7",
): VlmVerifier {
  return {
    provider: "anthropic",
    async verify({ previousFen, legalMovesSan, boardImage, previousBoardImage }) {
      try {
        // Force the canonical chess view (white at bottom) before the
        // API call. Mis-calibrated corners can leak a rotated rectified
        // board into the VLM, which then identifies pieces against an
        // unfamiliar orientation and produces confidently wrong picks.
        // Pass `previousFen` so the FEN-aware orientation check fires.
        const { oriented: orientedAfter } = ensureWhiteAtBottom(
          boardImage,
          previousFen,
        );
        const orientedBefore = previousBoardImage
          ? ensureWhiteAtBottom(previousBoardImage, previousFen).oriented
          : undefined;
        const afterB64 = canvasToBase64(orientedAfter);
        const beforeB64 = orientedBefore
          ? canvasToBase64(orientedBefore)
          : null;
        const prompt = beforeB64
          ? TWO_FRAME_PROMPT(previousFen, legalMovesSan)
          : SINGLE_FRAME_PROMPT(previousFen, legalMovesSan);
        const content: unknown[] = [];
        if (beforeB64) {
          content.push({
            type: "image",
            source: {
              type: "base64",
              media_type: "image/jpeg",
              data: beforeB64,
            },
          });
        }
        content.push({
          type: "image",
          source: {
            type: "base64",
            media_type: "image/jpeg",
            data: afterB64,
          },
        });
        content.push({ type: "text", text: prompt });
        const response = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-api-key": apiKey,
            "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true",
          },
          body: JSON.stringify(
            buildAnthropicBody(model, content),
          ),
        });
        if (!response.ok) {
          const text = await response.text().catch(() => "");
          return {
            kind: "error",
            reason: `Anthropic HTTP ${response.status}: ${text.slice(0, 160)}`,
          };
        }
        const data = (await response.json()) as {
          content?: { type: string; text?: string }[];
        };
        const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
        return resolveSan(raw, legalMovesSan);
      } catch (e) {
        return {
          kind: "error",
          reason: e instanceof Error ? e.message : String(e),
        };
      }
    },
  };
}

/**
 * Build the Anthropic /v1/messages body. Opus 4.7 does its own chain-of-
 * thought, so it needs many more tokens than Sonnet to reach a final
 * answer — and doesn't accept `temperature` at all.
 */
function buildAnthropicBody(
  model: string,
  content: unknown[],
): Record<string, unknown> {
  const isOpus = model.includes("opus");
  const body: Record<string, unknown> = {
    model,
    max_tokens: isOpus ? 4000 : 64,
    messages: [{ role: "user", content }],
  };
  if (!isOpus) body.temperature = 0.05;
  return body;
}

/**
 * Anthropic verifier that goes through a Cloudflare Worker (or any compatible
 * proxy) instead of api.anthropic.com directly. The proxy holds the API key
 * server-side; the browser never sees it.
 *
 * Proxy contract: POST {proxyUrl}/verify with a body identical to the
 * Anthropic /v1/messages payload. The proxy attaches x-api-key and returns
 * the upstream response verbatim.
 */
export function makeAnthropicProxyVerifier(
  proxyUrl: string,
  model = "claude-opus-4-7",
): VlmVerifier {
  const endpoint = proxyUrl.replace(/\/$/, "") + "/verify";
  return {
    provider: "anthropic",
    async verify({ previousFen, legalMovesSan, boardImage, previousBoardImage }) {
      try {
        // Force the canonical chess view (white at bottom) before the
        // API call. Mis-calibrated corners can leak a rotated rectified
        // board into the VLM, which then identifies pieces against an
        // unfamiliar orientation and produces confidently wrong picks.
        // Pass `previousFen` so the FEN-aware orientation check fires.
        const { oriented: orientedAfter } = ensureWhiteAtBottom(
          boardImage,
          previousFen,
        );
        const orientedBefore = previousBoardImage
          ? ensureWhiteAtBottom(previousBoardImage, previousFen).oriented
          : undefined;
        const afterB64 = canvasToBase64(orientedAfter);
        const beforeB64 = orientedBefore
          ? canvasToBase64(orientedBefore)
          : null;
        const prompt = beforeB64
          ? TWO_FRAME_PROMPT(previousFen, legalMovesSan)
          : SINGLE_FRAME_PROMPT(previousFen, legalMovesSan);
        const content: unknown[] = [];
        if (beforeB64) {
          content.push({
            type: "image",
            source: {
              type: "base64",
              media_type: "image/jpeg",
              data: beforeB64,
            },
          });
        }
        content.push({
          type: "image",
          source: {
            type: "base64",
            media_type: "image/jpeg",
            data: afterB64,
          },
        });
        content.push({ type: "text", text: prompt });
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(buildAnthropicBody(model, content)),
        });
        if (!response.ok) {
          const text = await response.text().catch(() => "");
          return {
            kind: "error",
            reason: `Proxy HTTP ${response.status}: ${text.slice(0, 160)}`,
          };
        }
        const data = (await response.json()) as {
          content?: { type: string; text?: string }[];
        };
        const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
        return resolveSan(raw, legalMovesSan);
      } catch (e) {
        return {
          kind: "error",
          reason: e instanceof Error ? e.message : String(e),
        };
      }
    },
  };
}

export function makeVerifier(
  provider: VlmProvider,
  apiKey: string,
): VlmVerifier {
  if (provider === "gemini") return makeGeminiVerifier(apiKey);
  if (provider === "openai") return makeOpenAiVerifier(apiKey);
  return makeAnthropicVerifier(apiKey);
}

// =========================================================================
// VLM-based corner detection.
//
// The pure-CV detector (dark-square centroids → grid fit) is brittle when
// pieces obscure the corner dark squares — the polygon consistently shrinks
// inward and clips a file/rank of the playing surface. A vision-language
// model with spatial reasoning is far more robust here: it sees the whole
// board, ignores pieces, and identifies the outer corners directly. We use
// it as the primary detector during calibrate, with CV as the immediate
// fallback when the VLM proxy isn't configured.
// =========================================================================

export type CornerPoint = { x: number; y: number };

export type CornerDetectionResult =
  | {
      kind: "detected";
      /**
       * Four playing-surface corners in clockwise order `[a8, h8, h1, a1]`,
       * in image-pixel coordinates of the original (un-downscaled) image.
       * Directly compatible with `warpBoard`.
       */
      corners: [CornerPoint, CornerPoint, CornerPoint, CornerPoint];
      raw: string;
    }
  | { kind: "error"; reason: string };

export interface CornerDetector {
  readonly provider: VlmProvider;
  detectCorners(args: {
    image: HTMLCanvasElement;
  }): Promise<CornerDetectionResult>;
}

const CORNER_DETECTION_PROMPT = `You are looking at a top-down or angled photo of a chess board.

TASK: Identify the FOUR PHYSICAL CORNERS of the playing surface — the rectangular area that contains the 8×8 grid of alternating dark/light squares. Each corner is where two perpendicular sides of the playing surface meet (perspective may distort the angle in the image).

CRITICAL RULES:
1. The corner is at the OUTER EDGE of the playing surface, NOT on top of any piece. If a rook (or any piece) sits on the corner square, the corner is where the playing surface boundary continues OUTSIDE that square — not on the piece itself.
2. Include the FULL extent of the playing surface — all 8 ranks AND all 8 files. Back-row pieces (rooks, knights, etc.) often obscure the corner dark squares; the corner is STILL at the outer corner of those occupied squares, NOT pulled inward to where you can see clean board pattern.
3. Use the rank labels (1–8) and file labels (a–h) printed on the board edges to verify orientation if visible.
4. Do NOT use the paper border, table edge, chess-clock body, or any background as the boundary — only the 8×8 grid of squares.

OUTPUT — return ONLY this JSON, no preamble, no markdown, no explanation:
{"a8":{"x":0.000,"y":0.000},"h8":{"x":0.000,"y":0.000},"h1":{"x":0.000,"y":0.000},"a1":{"x":0.000,"y":0.000}}

Corner labels (use whichever rooks/back-rank pieces are visible to anchor):
- a8 = corner where Black's queenside rook is/should be
- h8 = corner where Black's kingside rook is/should be
- h1 = corner where White's kingside rook is/should be
- a1 = corner where White's queenside rook is/should be

x and y are fractions from 0.0 to 1.0 (x=0 left edge of image, x=1 right edge; y=0 top edge, y=1 bottom edge). Round to 3 decimals.

Before answering, mentally trace the OUTER PERIMETER of the playing surface — the boundary between the 8×8 checker pattern and everything else. The four corners are exactly where that perimeter changes direction by 90°. Make sure your four points include the full back ranks and full files, even when corner pieces sit there.`;

/**
 * Anthropic-proxy corner detector. Posts a downscaled photo to the same
 * /verify endpoint the move verifier uses, with a corner-detection prompt
 * that asks for normalised (0..1) corner positions labelled a8/h8/h1/a1.
 * The corners come back oriented (no separate `orientStartingPosition`
 * step needed) — the model identifies which physical corner is which.
 */
export function makeAnthropicProxyCornerDetector(
  proxyUrl: string,
  model = "claude-opus-4-7",
): CornerDetector {
  const endpoint = proxyUrl.replace(/\/$/, "") + "/verify";
  const isOpus = model.includes("opus");
  return {
    provider: "anthropic",
    async detectCorners({ image }) {
      try {
        // Match Anthropic's native vision resolution (≤1568 px longest
        // edge) — sending lower res throws away the spatial precision
        // we need for pixel-accurate corner placement.
        const scaled = downscaleForVlm(image, 1568);
        const b64 = canvasToBase64(scaled);
        const body: Record<string, unknown> = {
          model,
          // Opus does its own chain-of-thought; needs many more tokens
          // than Sonnet to reach a final answer.
          max_tokens: isOpus ? 4000 : 512,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "image",
                  source: {
                    type: "base64",
                    media_type: "image/jpeg",
                    data: b64,
                  },
                },
                { type: "text", text: CORNER_DETECTION_PROMPT },
              ],
            },
          ],
        };
        if (!isOpus) body.temperature = 0;
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!response.ok) {
          const text = await response.text().catch(() => "");
          return {
            kind: "error",
            reason: `Proxy HTTP ${response.status}: ${text.slice(0, 160)}`,
          };
        }
        const data = (await response.json()) as {
          content?: { type: string; text?: string }[];
        };
        const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
        return parseCornerResponse(raw, image.width, image.height);
      } catch (e) {
        return {
          kind: "error",
          reason: e instanceof Error ? e.message : String(e),
        };
      }
    },
  };
}

/**
 * Downscale a canvas so its longest edge ≤ `maxDim`, keeping aspect
 * ratio. Speeds up the VLM round-trip and stays inside the model's
 * per-image budget. We send JPEG so quality matters less than dimensions.
 */
function downscaleForVlm(
  source: HTMLCanvasElement,
  maxDim: number,
): HTMLCanvasElement {
  const w = source.width;
  const h = source.height;
  const longest = Math.max(w, h);
  if (longest <= maxDim) return source;
  const scale = maxDim / longest;
  const out = document.createElement("canvas");
  out.width = Math.max(1, Math.round(w * scale));
  out.height = Math.max(1, Math.round(h * scale));
  const ctx = out.getContext("2d");
  if (!ctx) return source;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, out.width, out.height);
  return out;
}

/**
 * Parse the model's reply into four pixel-space corners. Tolerates a bit
 * of slop in the response: leading/trailing whitespace, accidental code
 * fences, or stray prose around the JSON. Validates each corner has a
 * numeric x/y in [0, 1] before scaling to image pixels.
 */
function parseCornerResponse(
  raw: string,
  imgW: number,
  imgH: number,
): CornerDetectionResult {
  const trimmed = raw.trim();
  // Strip code fences if the model wrapped the JSON in ```json … ```.
  const fence = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  const candidate = fence ? fence[1] : trimmed;
  // Grab the first {...} block — protects against prose-then-JSON.
  const objMatch = candidate.match(/\{[\s\S]*\}/);
  if (!objMatch) {
    return { kind: "error", reason: `No JSON object in response: ${raw.slice(0, 120)}` };
  }
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(objMatch[0]) as Record<string, unknown>;
  } catch (e) {
    return {
      kind: "error",
      reason: `JSON parse failed: ${e instanceof Error ? e.message : String(e)}`,
    };
  }
  const readCorner = (key: "a8" | "h8" | "h1" | "a1"): CornerPoint | null => {
    const v = parsed[key] as { x?: unknown; y?: unknown } | undefined;
    if (!v) return null;
    const x = typeof v.x === "number" ? v.x : Number(v.x);
    const y = typeof v.y === "number" ? v.y : Number(v.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    if (x < -0.1 || x > 1.1 || y < -0.1 || y > 1.1) return null;
    return { x: x * imgW, y: y * imgH };
  };
  const a8 = readCorner("a8");
  const h8 = readCorner("h8");
  const h1 = readCorner("h1");
  const a1 = readCorner("a1");
  if (!a8 || !h8 || !h1 || !a1) {
    return {
      kind: "error",
      reason: `Missing/invalid corner in response: ${objMatch[0].slice(0, 160)}`,
    };
  }
  return { kind: "detected", corners: [a8, h8, h1, a1], raw };
}

function resolveSan(raw: string, legalMovesSan: string[]): VlmVerifyResult {
  const trimmed = raw.trim();
  if (!trimmed) {
    return { kind: "rejected", raw, reason: "empty response" };
  }
  if (legalMovesSan.includes(trimmed)) {
    return { kind: "matched", san: trimmed, raw };
  }
  const cleaned = trimmed.replace(/^["'`]+|["'`.,!?\s]+$/g, "");
  if (legalMovesSan.includes(cleaned)) {
    return { kind: "matched", san: cleaned, raw };
  }
  // For chain-of-thought responses (Opus), the final SAN usually sits on
  // the last non-empty line. Search there first, preferring the *longest*
  // legal-move match (so "Nxf3" beats "f3" if both appear).
  const lines = trimmed
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  const lastLine = lines[lines.length - 1] ?? "";
  const candidates = [...legalMovesSan].sort((a, b) => b.length - a.length);
  for (const candidate of candidates) {
    const re = new RegExp(
      `(^|[^A-Za-z0-9])${escapeRegExp(candidate)}([^A-Za-z0-9]|$)`,
    );
    if (re.test(lastLine)) {
      return { kind: "matched", san: candidate, raw };
    }
  }
  for (const candidate of candidates) {
    const re = new RegExp(
      `(^|[^A-Za-z0-9])${escapeRegExp(candidate)}([^A-Za-z0-9]|$)`,
    );
    if (re.test(trimmed)) {
      return { kind: "matched", san: candidate, raw };
    }
  }
  return {
    kind: "rejected",
    raw,
    reason: "response did not match any legal move",
  };
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function canvasToBase64(canvas: HTMLCanvasElement): string {
  const url = canvas.toDataURL("image/jpeg", 0.85);
  const comma = url.indexOf(",");
  return comma >= 0 ? url.slice(comma + 1) : url;
}
