/**
 * Chesspar v2 — Gemini Flash classifier as the primary move detector.
 *
 * Strategy: each captured frame, we rectify the board, enumerate every
 * legal move from chess.js, and ask Gemini 3 Flash which legal move was
 * just played. The model returns its own confidence; we emit if it's
 * above the threshold and abstain otherwise so the user can tap-to-
 * confirm. No diff, no beam search, no per-square classifier — the VLM
 * is doing the whole inference job on a constrained set.
 */

export type Point = { x: number; y: number };

/** Image-space chess-board corners, ordered [a8, h8, h1, a1]. */
export type Corners = readonly [Point, Point, Point, Point];

export type Side = "white" | "black";

export type Square =
  | `${"a" | "b" | "c" | "d" | "e" | "f" | "g" | "h"}${
      | "1"
      | "2"
      | "3"
      | "4"
      | "5"
      | "6"
      | "7"
      | "8"}`;

/**
 * One captured burst, post-Laplacian selection. The chosen `frame` is the
 * sharpest of the 5; `variance` is its Laplacian variance (used by smoke-
 * test rejection of motion-blurred bursts).
 */
export type CapturedBurst = {
  frame: HTMLCanvasElement;
  variance: number;
  /** Wall-clock ms when the burst was triggered. */
  capturedAt: number;
  /** All 5 frames, ranked sharpest-first. Kept for retry/diagnostics. */
  rankedFrames: HTMLCanvasElement[];
};

/** A frozen homography for the current session. */
export type BoardLock = {
  corners: Corners;
  /** Side at the bottom of the rectified image (default white). */
  whiteAtBottom: boolean;
  /**
   * If the user calibrated against a starting position, this is the
   * confirmed FEN piece-placement. Used as the seed for the move history.
   */
  startingFen: string;
  /**
   * Optional VLM-derived board polygon (Florence-2) kept so we can
   * recover board-lock cheaply if the camera shifts mid-game.
   */
  florenceBbox?: { x1: number; y1: number; x2: number; y2: number };
};

/** Minimal legal-move record the UI needs to render an abstention prompt
 *  and apply the chosen move via chess.js. */
export type MoveCandidate = {
  san: string;
  uci: string;
  fromSquare: Square;
  toSquare: Square;
  /** Resulting FEN if this move is played. */
  resultingFen: string;
};

/**
 * Final outcome of one move-detection attempt. The orchestrator emits one
 * of these per user tap; the UI renders accordingly.
 */
export type MoveDecision =
  | {
      kind: "matched";
      pick: MoveCandidate;
      /** Other legal moves, surfaced to the abstention UI on later disagreement. */
      alternates: MoveCandidate[];
      pConfident: number;
      latencyMs: number;
    }
  | {
      kind: "abstain";
      candidates: MoveCandidate[];
      reason: string;
      pConfident: number;
      latencyMs: number;
    }
  | {
      kind: "error";
      reason: string;
      latencyMs: number;
    };

export type SessionConfig = {
  /** Cloudflare worker root, e.g. https://chesspar-vlm.<acct>.workers.dev */
  proxyUrl: string;
  /**
   * Confidence threshold to emit without abstaining. Flash returns a
   * direct per-call probability (not a noisy aggregator like the old
   * diff-first stack), so this can be much looser than the prior 0.97.
   */
  emitThreshold: number;
  /** Burst size in frames. PDFs recommend 5–10. */
  burstSize: number;
  /** Burst interval in ms between frames. */
  burstIntervalMs: number;
  /** Canonical rectified board size in pixels. PDFs use 512×512. */
  canonicalSize: number;
};

export const DEFAULT_CONFIG: SessionConfig = {
  proxyUrl: "",
  emitThreshold: 0.85,
  burstSize: 5,
  burstIntervalMs: 40,
  canonicalSize: 512,
};

export function squareToIndex(sq: Square): number {
  const file = sq.charCodeAt(0) - "a".charCodeAt(0);
  const rank = 8 - parseInt(sq[1], 10);
  return rank * 8 + file;
}

export function indexToSquare(idx: number): Square {
  const file = String.fromCharCode("a".charCodeAt(0) + (idx % 8));
  const rank = 8 - Math.floor(idx / 8);
  return `${file}${rank}` as Square;
}
