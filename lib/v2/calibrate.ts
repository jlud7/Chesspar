/**
 * One-call calibration. Send the camera frame to gemini-3-flash and ask
 * for the four named corners (a1, h1, a8, h8) plus a yes/no on whether
 * this is a standard starting position. The four named corners encode
 * rotation implicitly — no separate rotation tiebreaker needed.
 *
 * Returns a BoardLock + the rectified canvas, OR a specific user-facing
 * error so the UI can show the user exactly what to fix.
 */

import { callVlm, parseJsonLoose } from "./vlm.ts";
import { warpBoardWithMargin } from "../board-image.ts";
import type { BoardLock, Corners, Point } from "./types.ts";

const STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
const BOARD_CONTEXT_MARGIN = 0.14;

const PROMPT = `You are looking at a photo of a physical chessboard.

Identify the EXACT 4 OUTER CORNERS of the chessboard's 8x8 playing surface. The corner is where the outer edge of the corner square meets the board's wood/cardboard rim, NOT inside a square or out on the rim.

Label each corner by its algebraic name based on standard chess setup:
- a1 = white's queenside rook corner (lower-left from white's perspective)
- h1 = white's kingside rook corner (lower-right from white's perspective)
- a8 = black's queenside rook corner (upper-left from white's perspective)
- h8 = black's kingside rook corner (upper-right from white's perspective)

Use these cues to assign labels:
1. The white pieces are lighter coloured and sit on ranks 1-2.
2. The black pieces are darker coloured and sit on ranks 7-8.
3. The rank/file letters printed on the board's rim if visible.
4. If the king is to the right of the queen for that side, that's the kingside (h-file).

Coordinates are fractions of the image dimensions, 0.0 to 1.0:
- x = 0.0 at the left edge of the image, x = 1.0 at the right edge
- y = 0.0 at the top edge, y = 1.0 at the bottom

Also report:
- is_starting_position: true ONLY if every piece is in the standard chess starting position (32 pieces, ranks 1-2 white, 7-8 black, middle four ranks empty)
- error: short reason if you cannot see all four corners or are uncertain; empty string otherwise

Return ONLY this JSON (no preamble, no markdown):
{"a1":{"x":0.0,"y":0.0},"h1":{"x":0.0,"y":0.0},"a8":{"x":0.0,"y":0.0},"h8":{"x":0.0,"y":0.0},"is_starting_position":true,"error":""}

Round each coordinate to 4 decimals.`;

export type CalibrationResult =
  | {
      kind: "locked";
      lock: BoardLock;
      rectified: HTMLCanvasElement;
      isStartingPosition: boolean;
      durationMs: number;
    }
  | {
      kind: "failed";
      reason: string;
      durationMs: number;
    };

export async function calibrateBoard(opts: {
  proxyUrl: string;
  /** The camera frame (HTMLCanvasElement in-browser; canvas-compatible in Node). */
  image: HTMLCanvasElement;
  /** Output rectified size in pixels (default 512). */
  size?: number;
  /** Optional Origin override (Node-side tests only — browser sets this automatically). */
  origin?: string;
}): Promise<CalibrationResult> {
  const size = opts.size ?? 512;
  const w = opts.image.width;
  const h = opts.image.height;

  // Downscale to 1280px max dimension before sending. The full-resolution
  // phone frame is overkill and bloats the data URL.
  const scaled = downscale(opts.image, 1280);
  const dataUrl = scaled.toDataURL("image/jpeg", 0.88);

  const call = await callVlm({
    proxyUrl: opts.proxyUrl,
    callName: "calibrate",
    prompt: PROMPT,
    images: [dataUrl],
    origin: opts.origin,
  });

  if (call.kind === "error") {
    return { kind: "failed", reason: call.reason, durationMs: call.durationMs };
  }

  const parsed = parseJsonLoose<ParsedCalibration>(call.text);
  if (!parsed) {
    return {
      kind: "failed",
      reason: `Couldn't parse model JSON. Raw: ${call.text.slice(0, 200)}`,
      durationMs: call.durationMs,
    };
  }

  if (parsed.error && parsed.error.trim().length > 0) {
    return {
      kind: "failed",
      reason: parsed.error,
      durationMs: call.durationMs,
    };
  }

  const corners = mapCorners(parsed, w, h);
  if (!corners) {
    return {
      kind: "failed",
      reason: "Got corners back but they failed sanity check. Try again with a clearer view of all 4 corners.",
      durationMs: call.durationMs,
    };
  }

  const rectified = warpBoardWithMargin(
    opts.image,
    corners as [Point, Point, Point, Point],
    size,
    BOARD_CONTEXT_MARGIN,
  );

  const lock: BoardLock = {
    corners,
    whiteAtBottom: true,
    startingFen: STARTING_FEN,
  };

  return {
    kind: "locked",
    lock,
    rectified,
    isStartingPosition: parsed.is_starting_position === true,
    durationMs: call.durationMs,
  };
}

type ParsedCalibration = {
  a1?: { x: number; y: number };
  h1?: { x: number; y: number };
  a8?: { x: number; y: number };
  h8?: { x: number; y: number };
  is_starting_position?: boolean;
  error?: string;
};

function mapCorners(parsed: ParsedCalibration, w: number, h: number): Corners | null {
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
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % pts.length];
    area += p.x * q.y - q.x * p.y;
  }
  return Math.abs(area) / 2;
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

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
