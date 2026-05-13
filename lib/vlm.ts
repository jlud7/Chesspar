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
 */

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

/**
 * Result of reading a chess position directly from a photo into a FEN
 * piece-placement string. Used by the test-mode pipeline that bypasses
 * CV entirely and just asks the VLM "what position is this?".
 */
export type FenReadResult =
  | { kind: "fen"; fen: string; raw: string }
  | { kind: "unreadable"; raw: string }
  | { kind: "error"; reason: string };

const FEN_READ_PROMPT = `You are reading a chess position from a photograph of a real chessboard.

Reply with ONLY the FEN piece-placement field — just the board layout, nothing else. Example for the standard starting position:
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR

Conventions:
- Read with white at the bottom of the photo, black at the top.
- Write rank 8 first (top of the image), rank 1 last (bottom).
- Within each rank: a-file leftmost, h-file rightmost (from white's view).
- Pieces: K Q R B N P (white, uppercase) and k q r b n p (black, lowercase).
- Empty squares: digits 1-8 for consecutive empty squares within a rank.
- Ranks separated by /.

No explanation, no quotes, no markdown, no surrounding text — just the eight ranks separated by slashes.

If you cannot identify the position with high confidence (occluded pieces, blurry, ambiguous angle), reply with the single word UNREADABLE.`;

function isValidFenPlacement(s: string): boolean {
  const ranks = s.split("/");
  if (ranks.length !== 8) return false;
  for (const rank of ranks) {
    let count = 0;
    for (const ch of rank) {
      if (ch >= "1" && ch <= "8") count += Number(ch);
      else if ("KQRBNPkqrbnp".includes(ch)) count += 1;
      else return false;
    }
    if (count !== 8) return false;
  }
  return true;
}

function parseFenResponse(raw: string): FenReadResult {
  const trimmed = raw.trim();
  if (!trimmed) return { kind: "error", reason: "empty response" };
  if (/\bUNREADABLE\b/i.test(trimmed)) return { kind: "unreadable", raw };
  const cleaned = trimmed
    .replace(/```[a-z]*\n?|\n?```/gi, "")
    .replace(/^["'`]+|["'`.,!?]+$/g, "")
    .trim()
    .split(/\s+/)[0];
  if (!isValidFenPlacement(cleaned)) {
    return {
      kind: "error",
      reason: `not a valid FEN placement: ${cleaned.slice(0, 80)}`,
    };
  }
  return { kind: "fen", fen: cleaned, raw };
}

type AnthropicFenCallArgs = {
  url: string;
  extraHeaders: Record<string, string>;
  image: HTMLCanvasElement;
  model: string;
};

async function callAnthropicForFen(
  args: AnthropicFenCallArgs,
): Promise<FenReadResult> {
  try {
    const b64 = canvasToBase64(args.image);
    const response = await fetch(args.url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...args.extraHeaders },
      body: JSON.stringify({
        model: args.model,
        max_tokens: 80,
        temperature: 0.05,
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
              { type: "text", text: FEN_READ_PROMPT },
            ],
          },
        ],
      }),
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      return {
        kind: "error",
        reason: `HTTP ${response.status}: ${text.slice(0, 160)}`,
      };
    }
    const data = (await response.json()) as {
      content?: { type: string; text?: string }[];
    };
    const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
    return parseFenResponse(raw);
  } catch (e) {
    return {
      kind: "error",
      reason: e instanceof Error ? e.message : String(e),
    };
  }
}

export async function readFenViaProxy(
  proxyUrl: string,
  image: HTMLCanvasElement,
  model = "claude-sonnet-4-6",
): Promise<FenReadResult> {
  const url = proxyUrl.replace(/\/$/, "") + "/verify";
  return callAnthropicForFen({ url, extraHeaders: {}, image, model });
}

export async function readFenViaAnthropic(
  apiKey: string,
  image: HTMLCanvasElement,
  model = "claude-sonnet-4-6",
): Promise<FenReadResult> {
  return callAnthropicForFen({
    url: "https://api.anthropic.com/v1/messages",
    extraHeaders: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    image,
    model,
  });
}

const SINGLE_FRAME_PROMPT = (
  prevFen: string,
  legalMovesSan: string[],
): string => `You are identifying a chess move from a photo.

PREVIOUS POSITION FEN:
${prevFen}

LEGAL MOVES (the move that happened is exactly one of these):
${legalMovesSan.join(", ")}

The attached image is the rectified top-down view of the board AFTER the move was played. a8 is top-left, h1 is bottom-right (White's pieces are at the bottom).

Reply with ONLY the SAN notation of the move that just happened, exactly as it appears in the legal-moves list above (e.g. "e4", "Nxf3", "O-O", "Qxh7#"). No explanation, no quotes.`;

const TWO_FRAME_PROMPT = (
  prevFen: string,
  legalMovesSan: string[],
): string => `You are identifying which chess move was just played by comparing two photos.

IMAGE 1: the board BEFORE the move (top-down rectified view, a8 top-left, h1 bottom-right, White at bottom).
IMAGE 2: the board AFTER the move (same orientation).

PREVIOUS POSITION FEN:
${prevFen}

LEGAL MOVES — exactly one of these was played:
${legalMovesSan.join(", ")}

Compare the two images. Find the cells that changed. Pick the unique legal move whose result explains the change.

Reply with ONLY the SAN notation of the move that happened, exactly as it appears in the legal-moves list above (e.g. "e4", "Nxf3", "O-O", "Qxh7#"). No explanation, no quotes.`;

export function makeGeminiVerifier(
  apiKey: string,
  model = "gemini-2.5-pro",
): VlmVerifier {
  return {
    provider: "gemini",
    async verify({ previousFen, legalMovesSan, boardImage, previousBoardImage }) {
      try {
        const afterB64 = canvasToBase64(boardImage);
        const beforeB64 = previousBoardImage
          ? canvasToBase64(previousBoardImage)
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
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`;
        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ parts }],
            generationConfig: { temperature: 0.05, maxOutputTokens: 32 },
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

export function makeOpenAiVerifier(
  apiKey: string,
  model = "gpt-4o",
): VlmVerifier {
  return {
    provider: "openai",
    async verify({ previousFen, legalMovesSan, boardImage, previousBoardImage }) {
      try {
        const afterB64 = canvasToBase64(boardImage);
        const beforeB64 = previousBoardImage
          ? canvasToBase64(previousBoardImage)
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
        const response = await fetch(
          "https://api.openai.com/v1/chat/completions",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${apiKey}`,
            },
            body: JSON.stringify({
              model,
              max_tokens: 32,
              temperature: 0.05,
              messages: [{ role: "user", content }],
            }),
          },
        );
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

export function makeAnthropicVerifier(
  apiKey: string,
  model = "claude-sonnet-4-6",
): VlmVerifier {
  return {
    provider: "anthropic",
    async verify({ previousFen, legalMovesSan, boardImage, previousBoardImage }) {
      try {
        const afterB64 = canvasToBase64(boardImage);
        const beforeB64 = previousBoardImage
          ? canvasToBase64(previousBoardImage)
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
          body: JSON.stringify({
            model,
            max_tokens: 32,
            temperature: 0.05,
            messages: [{ role: "user", content }],
          }),
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
  model = "claude-sonnet-4-6",
): VlmVerifier {
  const endpoint = proxyUrl.replace(/\/$/, "") + "/verify";
  return {
    provider: "anthropic",
    async verify({ previousFen, legalMovesSan, boardImage, previousBoardImage }) {
      try {
        const afterB64 = canvasToBase64(boardImage);
        const beforeB64 = previousBoardImage
          ? canvasToBase64(previousBoardImage)
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
          body: JSON.stringify({
            model,
            max_tokens: 32,
            temperature: 0.05,
            messages: [{ role: "user", content }],
          }),
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
  for (const candidate of legalMovesSan) {
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
