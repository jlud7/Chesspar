/**
 * Board lock — one-shot session calibration with bulletproof corner +
 * orientation detection.
 *
 * Strategy (in order of preference):
 *   1. Gemini 2.5 Pro magic corner detector — asks the model for the
 *      4 corners of the playing surface directly. Beats Florence-2 +
 *      CV refinement on cluttered scenes (board edges next to user's
 *      leg, captured pieces, etc.).
 *   2. Florence-2 polygon + CV grid-fit — fallback when Gemini fails
 *      or there is no proxy URL.
 *   3. CV-only via dark-square material detection — last resort.
 *
 * After corners are found, we run all 4 rotations through the starting-
 * position scorer. Best score wins. If still ambiguous, Gemini picks
 * via a 2×2 mosaic.
 *
 * Locks below 0.6 starting-position score get REJECTED — the user is
 * told to re-aim with a specific reason, rather than locking on broken
 * corners and silently producing wrong PGNs.
 */

import { warpBoard } from "../board-image";
import {
  autoDetectBoardCorners,
  detectBoardCornersViaFlorence,
  type ChessboardBboxFetcher,
} from "../board-detection";
import { getChessboardBbox } from "../florence";
import { detectCornersMagic } from "./magic-corners";
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
      startingCheck: StartingCheck;
      magicCornerEscalation: boolean;
      magicRotationEscalation: boolean;
      rotationDebug: RotationPick;
      /** Diagnostic message for the UI (e.g. which detector succeeded). */
      detector: "claude" | "gemini" | "florence-cv" | "cv-only";
    }
  | {
      kind: "failed";
      reason: string;
    };

/** Minimum starting-position score required to accept a lock. Below this
 *  we reject and ask the user to retake — better than locking on bad
 *  corners and emitting wrong moves for the rest of the game. */
const MIN_LOCK_SCORE = 0.6;

export async function lockBoardFromImage(
  source: HTMLCanvasElement,
  opts: {
    proxyUrl: string;
    size?: number;
    startingFen?: string;
  },
): Promise<LockAttempt> {
  const size = opts.size ?? 512;
  const startingFen =
    opts.startingFen ?? "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";

  // ---- 1) Try the VLM cascade (Claude → Gemini) first ----
  let corners: Corners | null = null;
  let detector: "claude" | "gemini" | "florence-cv" | "cv-only" = "cv-only";

  if (opts.proxyUrl) {
    const magic = await detectCornersMagic({
      proxyUrl: opts.proxyUrl,
      image: source,
    });
    if (magic.kind === "detected") {
      corners = magic.corners;
      detector = magic.detector;
    }
  }

  // ---- 2) Fall back to Florence-2 + CV ----
  if (!corners) {
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
        reason: `Corner detection threw: ${e instanceof Error ? e.message : e}`,
      };
    }
    if (!detection) {
      return {
        kind: "failed",
        reason:
          "Could not find the board. Try a less-cluttered background, all 4 corners visible, and the board occupying most of the frame.",
      };
    }
    corners = detection.corners as Corners;
    detector = opts.proxyUrl ? "florence-cv" : "cv-only";
  }

  // ---- 3) Pick best rotation across all 4 ----
  const rotationPick = pickBestRotation(source, corners, size);
  let chosen: RotationPick = rotationPick;
  let magicRotationEscalation = false;
  if (!rotationPick.decisive && opts.proxyUrl) {
    const magic = await magicRotationFallback({
      proxyUrl: opts.proxyUrl,
      alternatives: rotationPick.alternatives,
    });
    if (magic) {
      magicRotationEscalation = true;
      const alt = rotationPick.alternatives.find(
        (a) => a.rotation === magic.rotation,
      );
      if (alt) {
        chosen = {
          corners: cyclicRotate(corners, magic.rotation),
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

  // ---- 4) Reject low-quality locks rather than ship bad corners ----
  if (startingCheck.score < MIN_LOCK_SCORE) {
    return {
      kind: "failed",
      reason: `Board found, but starting position match is only ${Math.round(startingCheck.score * 100)}% (need ≥ ${Math.round(MIN_LOCK_SCORE * 100)}%). Make sure: (1) standard starting position is set up, (2) all 4 corners are visible, (3) lighting is even, (4) phone is held steady. Then tap Retake.`,
    };
  }

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
    magicCornerEscalation: detector === "claude" || detector === "gemini",
    magicRotationEscalation,
    rotationDebug: chosen,
    detector,
  };
}

/**
 * Per-capture corner refresh — re-detect the board in a new frame so
 * the cached homography survives the user shifting their phone between
 * moves. Returns a fresh `BoardLock` if detection succeeded, or the
 * existing one untouched if it didn't (so we degrade gracefully).
 *
 * Called by `runMovePipeline` before warping each post-move frame.
 */
export async function refreshBoardLock(
  source: HTMLCanvasElement,
  current: BoardLock,
  opts: { proxyUrl: string },
): Promise<{ lock: BoardLock; detector: "claude" | "gemini" | "kept" }> {
  if (!opts.proxyUrl) return { lock: current, detector: "kept" };
  const magic = await detectCornersMagic({ proxyUrl: opts.proxyUrl, image: source });
  if (magic.kind !== "detected") return { lock: current, detector: "kept" };
  return {
    lock: {
      ...current,
      corners: magic.corners,
    },
    detector: magic.detector,
  };
}

/** Rotate corners by `k` quarters in cyclic order. Same semantics as
 *  the existing `rotateCorners` from board-detection but accepts a
 *  readonly `Corners`. */
function cyclicRotate(corners: Corners, k: number): Corners {
  const arr = [...corners];
  const shift = ((k % 4) + 4) % 4;
  for (let i = 0; i < shift; i++) arr.push(arr.shift()!);
  return [arr[0], arr[1], arr[2], arr[3]] as Corners;
}

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
