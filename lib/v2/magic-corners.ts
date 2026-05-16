/**
 * Magic corner detector — Gemini 2.5 Pro returns the 4 corners of the
 * actual chess playing surface directly, bypassing the Florence-2 bbox
 * (which can include the user's leg / table / fallen pieces and corrupt
 * the warp).
 *
 * The model returns four named corners (a1, h1, a8, h8) in normalized
 * coordinates. We then map them to the standard `[a8, h8, h1, a1]`
 * clockwise order that `warpBoard` expects.
 *
 * This is the new primary calibration path. Florence-2 + CV stays as
 * fallback when Gemini fails or the user has no proxy URL.
 */

import type { Corners, Point } from "./types";

export type MagicCornerResult =
  | { kind: "detected"; corners: Corners; raw: string }
  | { kind: "error"; reason: string };

const PROMPT = `You are identifying the EXACT 4 CORNERS of the chess board's PLAYING SURFACE in this photo.

The playing surface is the 8×8 squared region only — ignore the border, the wood frame, the table, any captured pieces sitting next to the board, hands, legs, scoresheets, anything outside the actual squares.

Return JSON with the four corners by their algebraic label (a1, h1, a8, h8). Each corner is the POINT WHERE THE OUTER CORNER OF THE CORNER SQUARE MEETS the playing area — at the very corner of the 8×8 grid, NOT at any piece, NOT beyond the grid into the wood border.

- a1 = the corner square nearest white's queenside rook
- h1 = the corner nearest white's kingside rook
- a8 = the corner nearest black's queenside rook
- h8 = the corner nearest black's kingside rook

Coordinates are normalized [0,1]: x=0 left edge of image, x=1 right edge; y=0 top, y=1 bottom. Round to 4 decimals.

If you can not see all four corners or cannot tell white from black, output {"error": "<short reason>"} instead.`;

export async function detectCornersMagic(opts: {
  proxyUrl: string;
  image: HTMLCanvasElement;
  model?: string;
}): Promise<MagicCornerResult> {
  if (!opts.proxyUrl) return { kind: "error", reason: "no proxyUrl" };

  // Downscale to ~1280 px longest edge — enough spatial precision for
  // sub-square corner placement, ~5× faster than full-res.
  const scaled = downscale(opts.image, 1280);
  const dataUrl = scaled.toDataURL("image/jpeg", 0.9);
  const model = opts.model ?? "gemini-2.5-pro";

  const responseSchema = {
    type: "OBJECT",
    properties: {
      a1: cornerSchema(),
      h1: cornerSchema(),
      a8: cornerSchema(),
      h8: cornerSchema(),
      error: { type: "STRING" },
    },
  };
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
      reason: `Gemini proxy HTTP ${resp.status}: ${text.slice(0, 200)}`,
    };
  }
  let data: GeminiResponse;
  try {
    data = (await resp.json()) as GeminiResponse;
  } catch (e) {
    return {
      kind: "error",
      reason: `Bad Gemini JSON: ${e instanceof Error ? e.message : e}`,
    };
  }
  const raw = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  if (!raw) return { kind: "error", reason: "Gemini empty response" };

  let parsed: ParsedCorners;
  try {
    parsed = JSON.parse(raw) as ParsedCorners;
  } catch {
    return { kind: "error", reason: `Could not parse Gemini JSON: ${raw.slice(0, 200)}` };
  }
  if (parsed.error) {
    return { kind: "error", reason: `Gemini abstained: ${parsed.error}` };
  }
  const w = opts.image.width;
  const h = opts.image.height;
  const toPx = (c: { x: number; y: number }): Point => ({
    x: clamp(c.x * w, 0, w),
    y: clamp(c.y * h, 0, h),
  });
  if (!parsed.a1 || !parsed.h1 || !parsed.a8 || !parsed.h8) {
    return { kind: "error", reason: "Gemini missing one or more corners" };
  }
  // warpBoard expects [a8, h8, h1, a1] clockwise.
  const corners: Corners = [
    toPx(parsed.a8),
    toPx(parsed.h8),
    toPx(parsed.h1),
    toPx(parsed.a1),
  ];
  if (!cornersLookValid(corners, w, h)) {
    return {
      kind: "error",
      reason: "Returned corners look degenerate (too small or collinear)",
    };
  }
  return { kind: "detected", corners, raw };
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

function cornersLookValid(corners: Corners, w: number, h: number): boolean {
  // Min side length: 15% of image width — otherwise we're warping a
  // dot, not a board.
  const minSide = Math.min(w, h) * 0.15;
  for (let i = 0; i < 4; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % 4];
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    if (Math.hypot(dx, dy) < minSide) return false;
  }
  // Reject if the quad's area is < 5% of the image — also too small.
  const area = polygonArea(corners);
  if (area < 0.05 * w * h) return false;
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

type ParsedCorners = {
  a1?: { x: number; y: number };
  h1?: { x: number; y: number };
  a8?: { x: number; y: number };
  h8?: { x: number; y: number };
  error?: string;
};

type GeminiResponse = {
  candidates?: Array<{
    content?: { parts?: Array<{ text?: string }> };
  }>;
};
