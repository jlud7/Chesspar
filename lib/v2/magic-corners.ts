/**
 * Magic corner detector — VLM returns the 4 corners of the actual chess
 * playing surface directly, bypassing CV approaches that can include
 * the user's leg / table / fallen pieces and corrupt the warp.
 *
 * Cascade (tries each in order, returns the first successful result):
 *   1. Gemini 2.5 Flash via /gemini   — fast path for live phone UX
 *   2. Claude Sonnet 4.6 via /verify  — fallback when Gemini abstains
 *
 * Each call returns 4 named corners in normalized [0,1] coords. We map
 * them to the standard [a8, h8, h1, a1] clockwise order that
 * `warpBoard` expects.
 */

import type { Corners, Point } from "./types";

export type MagicCornerResult =
  | {
      kind: "detected";
      corners: Corners;
      detector: "claude" | "gemini";
      raw: string;
    }
  | { kind: "error"; reason: string; tried: Array<{ detector: string; reason: string }> };

const PROMPT = `Identify the EXACT 4 OUTER CORNERS of the chessboard's playing surface (the 8×8 grid of squares).

The four corners are where the OUTER edge of the corner squares meets the wood/cardboard border or rim. Be precise — your corners should be RIGHT AT the outer edge of the grid line, not inside a square, not on the wood border beyond the grid, not on a piece.

For each corner, return its (x, y) position as fractions from 0.0 to 1.0:
- x = 0.0 is the left edge of the image, x = 1.0 is the right edge
- y = 0.0 is the top edge, y = 1.0 is the bottom

Label them by their algebraic name:
- a1 = white's queenside corner (near the white queen's rook)
- h1 = white's kingside corner
- a8 = black's queenside corner
- h8 = black's kingside corner

If the board is set up in starting position, white pieces are on ranks 1-2 and black on ranks 7-8. If the board is at any other position, look at the rank/file labels printed on the board if visible, or infer orientation from piece colours.

If you cannot see all four corners or are uncertain, return {"error": "<short reason>"}.

Return ONLY this JSON (no markdown, no preamble):
{"a1":{"x":0.000,"y":0.000},"h1":{"x":0.000,"y":0.000},"a8":{"x":0.000,"y":0.000},"h8":{"x":0.000,"y":0.000}}

Round each coordinate to 4 decimals.`;

export async function detectCornersMagic(opts: {
  proxyUrl: string;
  image: HTMLCanvasElement;
}): Promise<MagicCornerResult> {
  if (!opts.proxyUrl) {
    return { kind: "error", reason: "no proxyUrl", tried: [] };
  }
  const w = opts.image.width;
  const h = opts.image.height;
  const scaled = downscale(opts.image, 1568);
  const dataUrl = scaled.toDataURL("image/jpeg", 0.9);
  const tried: Array<{ detector: string; reason: string }> = [];

  // ---- 1) Gemini 2.5 Flash via /gemini ----
  const gemini = await detectViaGemini(opts.proxyUrl, dataUrl);
  if (gemini.kind === "parsed") {
    const corners = mapCorners(gemini.parsed, w, h);
    if (corners) return { kind: "detected", corners, detector: "gemini", raw: gemini.raw };
    tried.push({ detector: "gemini", reason: "corners failed sanity check" });
  } else {
    tried.push({ detector: "gemini", reason: gemini.reason });
  }

  // ---- 2) Claude Sonnet 4.6 via /verify ----
  const claude = await detectViaClaude(opts.proxyUrl, dataUrl);
  if (claude.kind === "parsed") {
    const corners = mapCorners(claude.parsed, w, h);
    if (corners) return { kind: "detected", corners, detector: "claude", raw: claude.raw };
    tried.push({ detector: "claude", reason: "corners failed sanity check" });
  } else {
    tried.push({ detector: "claude", reason: claude.reason });
  }

  return {
    kind: "error",
    reason: tried.map((t) => `${t.detector}: ${t.reason}`).join(" · "),
    tried,
  };
}

// ---------- Claude via /verify ----------

type DetectInner =
  | { kind: "parsed"; parsed: ParsedCorners; raw: string }
  | { kind: "error"; reason: string };

async function detectViaClaude(
  proxyUrl: string,
  dataUrl: string,
): Promise<DetectInner> {
  const endpoint = proxyUrl.replace(/\/$/, "") + "/verify";
  const body = {
    model: "claude-sonnet-4-6",
    max_tokens: 600,
    temperature: 0,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/jpeg",
              data: dataUrlToBase64(dataUrl),
            },
          },
          { type: "text", text: PROMPT },
        ],
      },
    ],
  };
  let resp: Response;
  try {
    resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    return { kind: "error", reason: e instanceof Error ? e.message : String(e) };
  }
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    return { kind: "error", reason: `HTTP ${resp.status}: ${text.slice(0, 160)}` };
  }
  const data = (await resp.json()) as {
    content?: { type: string; text?: string }[];
  };
  const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
  if (!raw) return { kind: "error", reason: "Claude returned empty content" };
  const parsed = parseJsonLoose(raw);
  if (!parsed) return { kind: "error", reason: `unparseable: ${raw.slice(0, 160)}` };
  if (parsed.error) {
    return { kind: "error", reason: `Claude abstained: ${parsed.error}` };
  }
  return { kind: "parsed", parsed, raw };
}

// ---------- Gemini via /gemini ----------

async function detectViaGemini(
  proxyUrl: string,
  dataUrl: string,
): Promise<DetectInner> {
  const endpoint =
    proxyUrl.replace(/\/$/, "") + "/gemini?model=gemini-2.5-flash";
  const body = {
    contents: [
      {
        role: "user",
        parts: [
          { text: PROMPT },
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
      temperature: 0.0,
      thinkingConfig: {
        thinkingBudget: 0,
      },
      responseMimeType: "application/json",
      responseSchema: {
        type: "OBJECT",
        properties: {
          a1: cornerSchema(),
          h1: cornerSchema(),
          a8: cornerSchema(),
          h8: cornerSchema(),
          error: { type: "STRING" },
        },
      },
    },
  };
  let resp: Response;
  try {
    resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    return { kind: "error", reason: e instanceof Error ? e.message : String(e) };
  }
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    return { kind: "error", reason: `HTTP ${resp.status}: ${text.slice(0, 160)}` };
  }
  const data = (await resp.json()) as {
    candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
  };
  const raw = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  if (!raw) return { kind: "error", reason: "Gemini empty response" };
  const parsed = parseJsonLoose(raw);
  if (!parsed) return { kind: "error", reason: `unparseable: ${raw.slice(0, 160)}` };
  if (parsed.error) {
    return { kind: "error", reason: `Gemini abstained: ${parsed.error}` };
  }
  return { kind: "parsed", parsed, raw };
}

// ---------- helpers ----------

function mapCorners(
  parsed: ParsedCorners,
  w: number,
  h: number,
): Corners | null {
  if (!parsed.a1 || !parsed.h1 || !parsed.a8 || !parsed.h8) return null;
  const toPx = (c: { x: number; y: number }): Point => ({
    x: clamp(c.x * w, 0, w),
    y: clamp(c.y * h, 0, h),
  });
  const corners: Corners = [
    toPx(parsed.a8),
    toPx(parsed.h8),
    toPx(parsed.h1),
    toPx(parsed.a1),
  ];
  if (!cornersLookValid(corners, w, h)) return null;
  return corners;
}

function cornersLookValid(corners: Corners, w: number, h: number): boolean {
  const minSide = Math.min(w, h) * 0.15;
  for (let i = 0; i < 4; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % 4];
    if (Math.hypot(a.x - b.x, a.y - b.y) < minSide) return false;
  }
  if (polygonArea(corners) < 0.05 * w * h) return false;
  return true;
}

function polygonArea(pts: Corners): number {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return Math.abs(a) / 2;
}

function downscale(canvas: HTMLCanvasElement, maxDim: number): HTMLCanvasElement {
  const scale = Math.min(1, maxDim / Math.max(canvas.width, canvas.height));
  if (scale >= 1) return canvas;
  const c = document.createElement("canvas");
  c.width = Math.round(canvas.width * scale);
  c.height = Math.round(canvas.height * scale);
  const ctx = c.getContext("2d");
  if (!ctx) return canvas;
  ctx.drawImage(canvas, 0, 0, c.width, c.height);
  return c;
}

function dataUrlToBase64(d: string): string {
  const i = d.indexOf(",");
  return i >= 0 ? d.slice(i + 1) : d;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Tolerant JSON parse: tries JSON.parse, then strips markdown fences,
 * then extracts the first {...} substring. Claude sometimes wraps JSON
 * in markdown despite the prompt; Gemini's structured-output usually
 * returns clean JSON but we use the same parser for both.
 */
function parseJsonLoose(raw: string): ParsedCorners | null {
  const trimmed = raw.trim();
  try {
    return JSON.parse(trimmed) as ParsedCorners;
  } catch {
    /* fall through */
  }
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) {
    try {
      return JSON.parse(fenced[1].trim()) as ParsedCorners;
    } catch {
      /* fall through */
    }
  }
  const inner = trimmed.match(/\{[\s\S]*\}/);
  if (inner) {
    try {
      return JSON.parse(inner[0]) as ParsedCorners;
    } catch {
      /* fall through */
    }
  }
  return null;
}

function cornerSchema() {
  return {
    type: "OBJECT",
    properties: {
      x: { type: "NUMBER" },
      y: { type: "NUMBER" },
    },
    required: ["x", "y"],
  };
}

type ParsedCorners = {
  a1?: { x: number; y: number };
  h1?: { x: number; y: number };
  a8?: { x: number; y: number };
  h8?: { x: number; y: number };
  error?: string;
};
