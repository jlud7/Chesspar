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
