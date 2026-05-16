/**
 * Board lock — one-shot session calibration that produces a cached
 * homography from image-space to canonical board space, with
 * bulletproof orientation auto-detection.
 *
 * Per the PDFs, the single biggest CV failure mode in the previous
 * Chesspar build was re-fitting the board geometry every frame and
 * drifting by a few pixels each move. v2 solves the geometry exactly
 * once per session (Florence-2 + CV refinement) and reuses it.
 *
 * Orientation is solved with the multi-cue starting-position scorer
 * over all 4 rotations, then — on the rare not-decisive case — a
 * Gemini magic fallback. The user never sees a rotate button.
 */

import { warpBoard } from "../board-image";
import {
  autoDetectBoardCorners,
  detectBoardCornersViaFlorence,
  rotateCorners,
  type ChessboardBboxFetcher,
} from "../board-detection";
import { getChessboardBbox } from "../florence";
import {
  magicRotationFallback,
  pickBestRotation,
  validateStartingPosition,
  type RotationPick,
  type StartingCheck,
} from "./starting-position";
import type { BoardLock, Corners, Point } from "./types";

export type LockAttempt =
  | {
      kind: "locked";
      lock: BoardLock;
      rectified: HTMLCanvasElement;
      /** Starting-position validation on the chosen rotation. */
      startingCheck: StartingCheck;
      /** True if Gemini was consulted to decide orientation. */
      magicEscalation: boolean;
      /** All 4 rotations + scores, kept for debug. */
      rotationDebug: RotationPick;
    }
  | {
      kind: "failed";
      reason: string;
    };

/**
 * Cold-start calibration. Locates the board in `source`, picks the
 * rotation that matches the starting position, returns the locked
 * homography ready for use by the rest of the pipeline.
 */
export async function lockBoardFromImage(
  source: HTMLCanvasElement,
  opts: {
    proxyUrl: string;
    /** Canonical rectified board size. Default 512. */
    size?: number;
    /** Starting FEN (piece placement) seeded into the lock. */
    startingFen?: string;
  },
): Promise<LockAttempt> {
  const size = opts.size ?? 512;
  const startingFen =
    opts.startingFen ?? "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";

  const fetcher: ChessboardBboxFetcher | null = opts.proxyUrl
    ? async (canvas) => {
        const r = await getChessboardBbox(canvas, opts.proxyUrl);
        return r.kind === "detected" ? r.bbox : null;
      }
    : null;

  let detection;
  try {
    detection = fetcher
      ? await detectBoardCornersViaFlorence(source, fetcher)
      : autoDetectBoardCorners(source);
  } catch (e) {
    return {
      kind: "failed",
      reason: `Corner detection failed: ${e instanceof Error ? e.message : e}. Try better lighting or a less oblique angle.`,
    };
  }
  if (!detection) {
    return {
      kind: "failed",
      reason:
        "Could not find the board. Make sure the whole board is in frame, well-lit, and at a moderate angle (not flat).",
    };
  }

  // Multi-cue rotation pick over all 4 rotations.
  const rotationPick = pickBestRotation(
    source,
    detection.corners as Corners,
    size,
  );

  let chosen: RotationPick = rotationPick;
  let magicEscalation = false;

  // Escalate to Gemini if the geometric pick is ambiguous. Per the PDFs,
  // a VLM lateral-thinks "which of these is the start position?" very
  // reliably; the cost is ~$0.001 and only fires on edge cases.
  if (!rotationPick.decisive && opts.proxyUrl) {
    const magic = await magicRotationFallback({
      proxyUrl: opts.proxyUrl,
      alternatives: rotationPick.alternatives,
    });
    if (magic) {
      magicEscalation = true;
      const alt = rotationPick.alternatives.find(
        (a) => a.rotation === magic.rotation,
      );
      if (alt) {
        chosen = {
          corners: rotateCorners(
            detection.corners,
            magic.rotation,
          ) as unknown as Corners,
          rotation: magic.rotation,
          rectified: alt.rectified,
          startingScore: alt.score,
          whiteAtBottom: true,
          alternatives: rotationPick.alternatives,
          decisive: true,
        };
      }
    }
  }

  const startingCheck = validateStartingPosition(chosen.rectified);

  const lock: BoardLock = {
    corners: chosen.corners,
    whiteAtBottom: true,
    startingFen,
  };
  return {
    kind: "locked",
    lock,
    rectified: chosen.rectified,
    startingCheck,
    magicEscalation,
    rotationDebug: chosen,
  };
}

/**
 * Warp the current frame to a canonical `size × size` board image using
 * the cached corners. Hot path — runs every move, takes a few ms.
 */
export function rectifyWithLock(
  source: HTMLCanvasElement,
  lock: BoardLock,
  size: number,
): HTMLCanvasElement {
  return warpBoard(
    source,
    lock.corners as unknown as [Point, Point, Point, Point],
    size,
  );
}

/**
 * Cheap board-lock health check. Sample a thin band around each corner
 * in the rectified frame; on a healthy lock those bands show "board
 * edge" contrast. Returns [0, 1].
 */
export function boardLockHealth(rectified: HTMLCanvasElement): number {
  const w = rectified.width;
  const h = rectified.height;
  const ctx = rectified.getContext("2d");
  if (!ctx) return 0;
  const band = Math.max(4, Math.round(w * 0.02));
  const bands: Array<{ inner: number; outer: number }> = [];
  bands.push({
    inner: meanLuminance(ctx, 0, band, w, band),
    outer: meanLuminance(ctx, 0, 0, w, band),
  });
  bands.push({
    inner: meanLuminance(ctx, 0, h - 2 * band, w, band),
    outer: meanLuminance(ctx, 0, h - band, w, band),
  });
  bands.push({
    inner: meanLuminance(ctx, band, 0, band, h),
    outer: meanLuminance(ctx, 0, 0, band, h),
  });
  bands.push({
    inner: meanLuminance(ctx, w - 2 * band, 0, band, h),
    outer: meanLuminance(ctx, w - band, 0, band, h),
  });
  const diffs = bands.map((b) => Math.abs(b.inner - b.outer));
  const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  return Math.max(0, Math.min(1, avgDiff / 30));
}

function meanLuminance(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
): number {
  if (w <= 0 || h <= 0) return 0;
  const img = ctx.getImageData(x, y, w, h).data;
  let sum = 0;
  let n = 0;
  for (let i = 0; i < img.length; i += 4) {
    sum += 0.299 * img[i] + 0.587 * img[i + 1] + 0.114 * img[i + 2];
    n++;
  }
  return n === 0 ? 0 : sum / n;
}
