/**
 * Gemini Flash move classifier — primary inference path for v2.
 *
 * Methodology (per the project's redesign decision):
 *   Don't reconstruct the full board state from each image. Instead,
 *   give the VLM a tiny constrained question:
 *
 *     "Given the previous FEN and this list of ~20–40 legal moves,
 *      which one was just played?"
 *
 *   That collapses the search space from billions of board positions
 *   to a few dozen, and turns a hard reconstruction problem into a
 *   small classification one.
 *
 * Operating discipline (also project-decided):
 *   - Minimal prompts beat verbose ones (better speed, accuracy, fewer
 *     hallucinations on this task).
 *   - temperature: 0 — we want deterministic picks, not creative ones.
 *   - thinkingBudget: 0 — instant response. The constrained problem
 *     does not need chain-of-thought; budget can be raised if accuracy
 *     drops on tricky positions.
 *   - chess.js is the source of truth for legality; we only ask Gemini
 *     to pick from the list we pre-validated.
 */

import type { Square } from "./types";

export type FlashCandidate = {
  san: string;
  /** UCI string for chess.js. */
  uci: string;
  fromSquare: Square;
  toSquare: Square;
};

export type FlashClassifierResult =
  | {
      kind: "matched";
      san: string;
      confidence: number;
      latencyMs: number;
      raw: string;
    }
  | {
      kind: "abstain";
      reason: string;
      latencyMs: number;
      raw: string;
    }
  | {
      kind: "error";
      reason: string;
      latencyMs: number;
    };

/**
 * Default model id. The user can override via the `model` option on
 * each call (or by setting NEXT_PUBLIC_GEMINI_FLASH_MODEL at build
 * time). Bump this if Google retires the preview alias.
 */
const DEFAULT_MODEL =
  process.env.NEXT_PUBLIC_GEMINI_FLASH_MODEL || "gemini-3-flash-preview";

export async function classifyMoveWithFlash(opts: {
  proxyUrl: string;
  previousFen: string;
  candidates: FlashCandidate[];
  /** Rectified, top-down view of the board AFTER the move. */
  postImage: HTMLCanvasElement;
  /** Optional override; defaults to gemini-3-flash-preview. */
  model?: string;
  /** Thinking budget for Gemini 2.5+/3 Flash models. 0 = no thinking,
   *  fastest response. Raise if you need more accuracy on close calls. */
  thinkingBudget?: number;
}): Promise<FlashClassifierResult> {
  const t0 = performance.now();
  if (!opts.proxyUrl) {
    return { kind: "error", reason: "no proxyUrl", latencyMs: 0 };
  }
  if (opts.candidates.length === 0) {
    return {
      kind: "error",
      reason: "no legal candidates supplied",
      latencyMs: 0,
    };
  }
  // Single-candidate shortcut: don't burn a model call when there is
  // only one legal move (forced response, only-king-move, etc.).
  if (opts.candidates.length === 1) {
    return {
      kind: "matched",
      san: opts.candidates[0].san,
      confidence: 1,
      latencyMs: performance.now() - t0,
      raw: "(single legal move — no VLM call needed)",
    };
  }

  const model = opts.model ?? DEFAULT_MODEL;
  const sanList = opts.candidates.map((c) => c.san);
  const dataUrl = opts.postImage.toDataURL("image/jpeg", 0.9);

  // The smallest prompt that still constrains the answer. Every word
  // here was kept because removing it changed observed behaviour on
  // the test set.
  const prompt = `Previous FEN: ${opts.previousFen}
One legal move occurred. Pick which from this list:
${sanList.join(", ")}
Return JSON only.`;

  const body = {
    contents: [
      {
        role: "user",
        parts: [
          { text: prompt },
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: dataUrlToBase64(dataUrl),
            },
          },
        ],
      },
    ],
    generationConfig: {
      temperature: 0,
      responseMimeType: "application/json",
      responseSchema: {
        type: "OBJECT",
        properties: {
          san: { type: "STRING", enum: [...sanList, "ABSTAIN"] },
          confidence: { type: "NUMBER" },
        },
        required: ["san", "confidence"],
      },
      // thinkingConfig is the Gemini 2.5+/3 control for chain-of-thought
      // depth. 0 = instant response. Most chess move-id queries don't
      // need any thinking; we keep it at 0 for the latency win.
      thinkingConfig: {
        thinkingBudget: opts.thinkingBudget ?? 0,
      },
    },
  };

  let resp: Response;
  try {
    resp = await fetch(
      `${opts.proxyUrl.replace(/\/$/, "")}/gemini?model=${encodeURIComponent(model)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
  } catch (e) {
    return {
      kind: "error",
      reason: e instanceof Error ? e.message : String(e),
      latencyMs: performance.now() - t0,
    };
  }

  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    return {
      kind: "error",
      reason: `proxy HTTP ${resp.status}: ${text.slice(0, 200)}`,
      latencyMs: performance.now() - t0,
    };
  }

  let data: GeminiResponse;
  try {
    data = (await resp.json()) as GeminiResponse;
  } catch (e) {
    return {
      kind: "error",
      reason: `bad JSON from proxy: ${e instanceof Error ? e.message : e}`,
      latencyMs: performance.now() - t0,
    };
  }

  const raw = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  const latencyMs = performance.now() - t0;
  if (!raw) {
    return { kind: "error", reason: "Gemini returned no text", latencyMs };
  }
  let parsed: { san?: string; confidence?: number };
  try {
    parsed = JSON.parse(raw);
  } catch {
    return {
      kind: "error",
      reason: `unparseable JSON: ${raw.slice(0, 200)}`,
      latencyMs,
    };
  }
  const conf =
    typeof parsed.confidence === "number" ? parsed.confidence : 0.5;
  if (!parsed.san || parsed.san === "ABSTAIN") {
    return {
      kind: "abstain",
      reason: "model abstained",
      latencyMs,
      raw,
    };
  }
  // The structured-output enum should make out-of-list answers
  // impossible, but defend anyway.
  if (!sanList.includes(parsed.san)) {
    return {
      kind: "error",
      reason: `model returned out-of-enum value: ${parsed.san}`,
      latencyMs,
    };
  }
  return {
    kind: "matched",
    san: parsed.san,
    confidence: conf,
    latencyMs,
    raw,
  };
}

function dataUrlToBase64(d: string): string {
  const i = d.indexOf(",");
  return i >= 0 ? d.slice(i + 1) : d;
}

type GeminiResponse = {
  candidates?: Array<{
    content?: { parts?: Array<{ text?: string }> };
  }>;
};
