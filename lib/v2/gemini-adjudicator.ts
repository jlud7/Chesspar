/**
 * VLM adjudicator — Gemini 2.5 Pro on the disputed-tile path only.
 *
 * Per the PDFs, this fires on ~2% of moves where the diff-first decoder
 * is uncertain. We give Gemini:
 *   - the previous FEN (one short string)
 *   - a list of ≤5 candidate legal moves in SAN
 *   - the warped 512×512 post-move board image
 * Gemini returns one of the candidates or `null` (abstain). The
 * structured-output schema forbids freeform answers — the model must
 * pick from the supplied list or refuse.
 *
 * Cost: ~$0.002/escalation. At 2% escalation rate, amortized ~$0.00004
 * per move. Comfortably under the $0.02/move budget in the PDF.
 */

import type { MoveCandidate } from "./types";

export type AdjudicationResult =
  | { kind: "matched"; san: string; confidence: number; raw: string }
  | { kind: "abstain"; reason: string; raw: string }
  | { kind: "error"; reason: string };

export async function adjudicateMove(opts: {
  /** Cloudflare worker root, e.g. https://chesspar-vlm.<acct>.workers.dev */
  proxyUrl: string;
  /** Position before the move. */
  previousFen: string;
  /** Legal candidates, ranked best-first. Cap at 5 — Gemini is sharper
   *  on tight lists than on the full ~35 legal moves. */
  candidates: MoveCandidate[];
  /** Warped, white-at-bottom 512×512 post-move board. */
  postFrame: HTMLCanvasElement;
  /** Optional pre-move frame; meaningfully improves accuracy. */
  preFrame?: HTMLCanvasElement;
  /** Gemini model id. Default gemini-2.5-pro. */
  model?: string;
}): Promise<AdjudicationResult> {
  if (!opts.proxyUrl) return { kind: "error", reason: "no proxyUrl" };
  if (opts.candidates.length === 0) {
    return { kind: "error", reason: "no candidates supplied" };
  }
  if (opts.candidates.length === 1) {
    return {
      kind: "matched",
      san: opts.candidates[0].san,
      confidence: 1,
      raw: "single candidate — no VLM call needed",
    };
  }

  const candidateList = opts.candidates.slice(0, 5);
  const sanList = candidateList.map((c) => c.san);
  const model = opts.model ?? "gemini-2.5-pro";

  const parts: GeminiPart[] = [
    { text: buildPrompt(opts.previousFen, sanList) },
  ];
  if (opts.preFrame) {
    parts.push({
      inlineData: {
        mimeType: "image/jpeg",
        data: dataUrlToBase64(opts.preFrame.toDataURL("image/jpeg", 0.9)),
      },
    });
    parts.push({ text: "(image above: the board BEFORE the move)" });
  }
  parts.push({
    inlineData: {
      mimeType: "image/jpeg",
      data: dataUrlToBase64(opts.postFrame.toDataURL("image/jpeg", 0.9)),
    },
  });
  parts.push({ text: "(image above: the board AFTER the move)" });

  // Gemini structured-output schema: the model can only emit one of
  // {chosen_san: <enum>, confidence: number, evidence: string} or
  // abstain with chosen_san = "ABSTAIN".
  const responseSchema = {
    type: "OBJECT",
    properties: {
      chosen_san: {
        type: "STRING",
        enum: [...sanList, "ABSTAIN"],
      },
      confidence: { type: "NUMBER" },
      evidence: { type: "STRING" },
    },
    required: ["chosen_san", "confidence", "evidence"],
  };

  const body = {
    contents: [{ role: "user", parts }],
    generationConfig: {
      temperature: 0.0,
      responseMimeType: "application/json",
      responseSchema,
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
    };
  }
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    return {
      kind: "error",
      reason: `proxy HTTP ${resp.status}: ${text.slice(0, 240)}`,
    };
  }

  let data: GeminiResponse;
  try {
    data = (await resp.json()) as GeminiResponse;
  } catch (e) {
    return {
      kind: "error",
      reason: `bad JSON from Gemini proxy: ${e instanceof Error ? e.message : e}`,
    };
  }

  const raw = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  if (!raw) {
    return { kind: "error", reason: "Gemini returned no text" };
  }
  let parsed: { chosen_san?: string; confidence?: number; evidence?: string };
  try {
    parsed = JSON.parse(raw);
  } catch {
    return {
      kind: "error",
      reason: `Gemini JSON parse failed: ${raw.slice(0, 240)}`,
    };
  }
  const chosen = parsed.chosen_san;
  if (!chosen || chosen === "ABSTAIN") {
    return {
      kind: "abstain",
      reason: parsed.evidence ?? "model abstained",
      raw,
    };
  }
  if (!sanList.includes(chosen)) {
    return {
      kind: "error",
      reason: `model returned out-of-enum value: ${chosen}`,
    };
  }
  return {
    kind: "matched",
    san: chosen,
    confidence: typeof parsed.confidence === "number" ? parsed.confidence : 0.5,
    raw,
  };
}

function buildPrompt(previousFen: string, candidates: string[]): string {
  return `You are adjudicating exactly one chess move from photos of the board before and after the move.

Previous position (piece-placement FEN): ${previousFen}

Legal candidate moves — exactly ONE was played:
${candidates.map((s, i) => `${i + 1}. ${s}`).join("\n")}

Rules:
- You may ONLY choose from the candidate list, or output "ABSTAIN".
- Do NOT analyze the chess position strategically. Pick the move whose RESULTING POSITION matches what you see in the AFTER image.
- The board is shown standard-orientation: white at bottom (ranks 1–2), black at top (ranks 7–8).
- If you cannot tell which candidate matches the after-image, output "ABSTAIN" with confidence 0.

Return JSON with fields: chosen_san (from the list or "ABSTAIN"), confidence (0–1), evidence (one short sentence describing the visual change you saw).`;
}

function dataUrlToBase64(dataUrl: string): string {
  const i = dataUrl.indexOf(",");
  return i >= 0 ? dataUrl.slice(i + 1) : dataUrl;
}

type GeminiPart =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } };

type GeminiResponse = {
  candidates?: Array<{
    content?: { parts?: Array<{ text?: string }> };
  }>;
};
