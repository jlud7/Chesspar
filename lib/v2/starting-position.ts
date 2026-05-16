/**
 * Starting-position validator + bulletproof auto-rotation.
 *
 * Real-world failure mode: the user calibrates with the camera anywhere
 * around the board (left/right/black side), not just behind white. The
 * lock returns "some" 4 corners but the rectified board may be rotated
 * 90°/180°/270° from canonical (white-at-bottom). The user must NEVER
 * see a manual rotate button — auto-detection has to feel like magic.
 *
 * Strategy: for each of the 4 rotations,
 *   (a) check that ranks 1–2 and 7–8 are occupied, ranks 3–6 are empty
 *   (b) check that the bottom-two ranks have lighter pieces (white)
 *       than the top-two ranks
 *   (c) check piece-count: exactly 32 occupied squares
 * Combine into a single score in [0, 1]. The rotation with the highest
 * score wins. If the top score is <0.85 OR the top-2 margin is <0.1,
 * the orchestrator may call Gemini as a magic tiebreaker.
 */

import type { Corners } from "./types";
import { warpBoard } from "../board-image";
import type { Point } from "../homography";
import { rotateCorners } from "../board-detection";

export type StartingCheck = {
  /** Combined match score in [0, 1]. */
  score: number;
  /** Number of squares (out of 64) classified consistently with the start. */
  matchCount: number;
  /** Total non-empty squares detected. Should be 32. */
  pieceCount: number;
  /** Per-row piece counts; back two rows should be 8, middle 4 should be 0. */
  pieceCountPerRow: [number, number, number, number, number, number, number, number];
  /** Mean luminance of pieces on bottom-two ranks. Higher = lighter (likely white). */
  bottomPieceLuma: number;
  /** Mean luminance of pieces on top-two ranks. Lower = darker (likely black). */
  topPieceLuma: number;
  /** True if bottom is brighter than top — i.e., white-at-bottom. */
  whiteAtBottom: boolean;
};

/**
 * Score a rectified-board canvas against the standard starting position.
 * Multi-cue: occupancy pattern + piece-count + white-at-bottom luminance.
 */
export function validateStartingPosition(
  rectified: HTMLCanvasElement,
): StartingCheck {
  const size = rectified.width;
  if (rectified.height !== size) {
    throw new Error("validateStartingPosition: canvas must be square");
  }
  const ctx = rectified.getContext("2d");
  if (!ctx) throw new Error("2d ctx unavailable");
  const square = size / 8;
  const inset = Math.round(square * 0.2);
  const inner = Math.max(2, Math.round(square - 2 * inset));

  const pieceCountPerRow: [number, number, number, number, number, number, number, number] = [
    0, 0, 0, 0, 0, 0, 0, 0,
  ];

  // Per-square mean luminance + std.
  const stats: Array<{ y: number; std: number }> = new Array(64);
  for (let i = 0; i < 64; i++) {
    const r = Math.floor(i / 8);
    const c = i % 8;
    const x0 = Math.round(c * square + inset);
    const y0 = Math.round(r * square + inset);
    const img = ctx.getImageData(x0, y0, inner, inner).data;
    let sum = 0;
    let n = 0;
    for (let p = 0; p < img.length; p += 4) {
      sum += 0.299 * img[p] + 0.587 * img[p + 1] + 0.114 * img[p + 2];
      n++;
    }
    const meanY = sum / n;
    let varSum = 0;
    for (let p = 0; p < img.length; p += 4) {
      const y = 0.299 * img[p] + 0.587 * img[p + 1] + 0.114 * img[p + 2];
      varSum += (y - meanY) ** 2;
    }
    stats[i] = { y: meanY, std: Math.sqrt(varSum / n) };
  }

  // Empty baselines from rows 2–5 (middle ranks, expected empty in start).
  let darkEmpty = 0;
  let darkN = 0;
  let lightEmpty = 0;
  let lightN = 0;
  for (let r = 2; r <= 5; r++) {
    for (let c = 0; c < 8; c++) {
      const idx = r * 8 + c;
      const isLight = (r + c) % 2 === 0;
      if (isLight) {
        lightEmpty += stats[idx].y;
        lightN++;
      } else {
        darkEmpty += stats[idx].y;
        darkN++;
      }
    }
  }
  const darkBaseline = darkN ? darkEmpty / darkN : 90;
  const lightBaseline = lightN ? lightEmpty / lightN : 200;

  let matchCount = 0;
  let pieceCount = 0;
  // Accumulate piece luminance for the bottom/top discriminator.
  let bottomPieceLumaSum = 0;
  let bottomPieceN = 0;
  let topPieceLumaSum = 0;
  let topPieceN = 0;
  for (let i = 0; i < 64; i++) {
    const r = Math.floor(i / 8);
    const c = i % 8;
    const isLight = (r + c) % 2 === 0;
    const expectedFilled = r === 0 || r === 1 || r === 6 || r === 7;
    const baseline = isLight ? lightBaseline : darkBaseline;
    const lumaDiff = Math.abs(stats[i].y - baseline);
    const occupied = stats[i].std > 14 || lumaDiff > 50;
    if (occupied === expectedFilled) matchCount++;
    if (occupied) {
      pieceCount++;
      pieceCountPerRow[r]++;
      if (r === 6 || r === 7) {
        bottomPieceLumaSum += stats[i].y;
        bottomPieceN++;
      } else if (r === 0 || r === 1) {
        topPieceLumaSum += stats[i].y;
        topPieceN++;
      }
    }
  }

  const bottomPieceLuma = bottomPieceN ? bottomPieceLumaSum / bottomPieceN : 0;
  const topPieceLuma = topPieceN ? topPieceLumaSum / topPieceN : 0;
  const whiteAtBottom = bottomPieceLuma > topPieceLuma;

  // Combined score: occupancy match (heavily weighted), piece-count
  // proximity to 32, white-at-bottom contrast.
  const occupancyScore = matchCount / 64;
  // Piece-count score: peak at 32, drop linearly outside.
  const pieceCountScore = 1 - Math.min(1, Math.abs(pieceCount - 32) / 16);
  // White-at-bottom score: how much brighter the bottom pieces are vs top.
  // 30 luma units of separation is a clean signal; 0 means we cannot tell.
  const orientationContrast = bottomPieceLuma - topPieceLuma;
  const orientationScore = Math.max(0, Math.min(1, orientationContrast / 30));

  const score =
    0.6 * occupancyScore + 0.15 * pieceCountScore + 0.25 * orientationScore;

  return {
    score,
    matchCount,
    pieceCount,
    pieceCountPerRow,
    bottomPieceLuma,
    topPieceLuma,
    whiteAtBottom,
  };
}

export type RotationPick = {
  corners: Corners;
  rotation: 0 | 1 | 2 | 3;
  rectified: HTMLCanvasElement;
  startingScore: number;
  whiteAtBottom: boolean;
  /** All 4 rotations' rectified canvases, ranked by score descending. */
  alternatives: Array<{
    rotation: 0 | 1 | 2 | 3;
    score: number;
    rectified: HTMLCanvasElement;
  }>;
  /** True if the top-1 was unambiguous (score ≥ 0.85 AND margin ≥ 0.1). */
  decisive: boolean;
};

/**
 * Try all 4 rotations, pick the one whose starting-position score is
 * highest. The caller may want to escalate to Gemini if `decisive` is
 * false — but on a well-lit starting position this should never happen.
 */
export function pickBestRotation(
  source: HTMLCanvasElement,
  corners: Corners,
  size: number,
): RotationPick {
  type Cand = {
    rotation: 0 | 1 | 2 | 3;
    corners: Corners;
    rectified: HTMLCanvasElement;
    score: number;
    whiteAtBottom: boolean;
  };
  const cands: Cand[] = [];
  const mutableCorners = [...corners] as [Point, Point, Point, Point];
  for (let r = 0; r < 4; r++) {
    const rotated = rotateCorners(mutableCorners, r);
    const rect = warpBoard(source, rotated, size);
    const v = validateStartingPosition(rect);
    cands.push({
      rotation: r as 0 | 1 | 2 | 3,
      corners: rotated,
      rectified: rect,
      score: v.score,
      whiteAtBottom: v.whiteAtBottom,
    });
  }
  cands.sort((a, b) => b.score - a.score);
  const top = cands[0];
  const margin = cands.length > 1 ? cands[0].score - cands[1].score : 1;
  return {
    corners: top.corners,
    rotation: top.rotation,
    rectified: top.rectified,
    startingScore: top.score,
    whiteAtBottom: top.whiteAtBottom,
    alternatives: cands.map((c) => ({
      rotation: c.rotation,
      score: c.score,
      rectified: c.rectified,
    })),
    decisive: top.score >= 0.85 && margin >= 0.1,
  };
}

/**
 * Last-resort magic: when the geometry-based rotation picker isn't
 * decisive (e.g., user calibrated mid-game, board partially out of
 * frame), ask Gemini directly "which of these four rectified images
 * is the standard chess starting position with white at the bottom?"
 *
 * Gemini sees a 2×2 mosaic of the four rotations with a letter label
 * on each. It returns one letter. The orchestrator maps it back to
 * a rotation index.
 *
 * This costs ~$0.001 and fires only when the geometric picker isn't
 * decisive — rare in normal play.
 */
export async function magicRotationFallback(opts: {
  proxyUrl: string;
  alternatives: Array<{
    rotation: 0 | 1 | 2 | 3;
    score: number;
    rectified: HTMLCanvasElement;
  }>;
}): Promise<{ rotation: 0 | 1 | 2 | 3; reason: string } | null> {
  if (!opts.proxyUrl) return null;
  const labels: Array<"A" | "B" | "C" | "D"> = ["A", "B", "C", "D"];
  const mosaic = buildMosaic(opts.alternatives, labels);

  const body = {
    contents: [
      {
        role: "user",
        parts: [
          {
            text: `Four candidate orientations of a chess board (top-down rectified). Which one shows the STANDARD STARTING POSITION with WHITE pieces at the BOTTOM (ranks 1–2)?\n\nReply with one letter A/B/C/D, then a short reason.`,
          },
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: dataUrlToBase64(mosaic.toDataURL("image/jpeg", 0.9)),
            },
          },
        ],
      },
    ],
    generationConfig: {
      temperature: 0.0,
      responseMimeType: "application/json",
      responseSchema: {
        type: "OBJECT",
        properties: {
          choice: { type: "STRING", enum: ["A", "B", "C", "D"] },
          reason: { type: "STRING" },
        },
        required: ["choice", "reason"],
      },
    },
  };
  try {
    const resp = await fetch(
      `${opts.proxyUrl.replace(/\/$/, "")}/gemini?model=gemini-2.5-pro`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    if (!resp.ok) return null;
    const data = (await resp.json()) as {
      candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
    };
    const raw = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
    const parsed = JSON.parse(raw) as { choice?: string; reason?: string };
    const idx = labels.indexOf(parsed.choice as "A");
    if (idx < 0 || idx >= opts.alternatives.length) return null;
    return {
      rotation: opts.alternatives[idx].rotation,
      reason: parsed.reason ?? "",
    };
  } catch {
    return null;
  }
}

function buildMosaic(
  alternatives: Array<{ rectified: HTMLCanvasElement }>,
  labels: string[],
): HTMLCanvasElement {
  const tileSize = 256;
  const c = document.createElement("canvas");
  c.width = tileSize * 2;
  c.height = tileSize * 2;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("2d ctx unavailable for mosaic");
  ctx.fillStyle = "#0c0c0c";
  ctx.fillRect(0, 0, c.width, c.height);
  for (let i = 0; i < 4; i++) {
    const x = (i % 2) * tileSize;
    const y = Math.floor(i / 2) * tileSize;
    if (alternatives[i]) {
      ctx.drawImage(alternatives[i].rectified, x, y, tileSize, tileSize);
    }
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(x + 6, y + 6, 32, 32);
    ctx.fillStyle = "white";
    ctx.font = "bold 22px ui-monospace, monospace";
    ctx.fillText(labels[i] ?? "?", x + 14, y + 30);
  }
  return c;
}

function dataUrlToBase64(d: string): string {
  const i = d.indexOf(",");
  return i >= 0 ? d.slice(i + 1) : d;
}
