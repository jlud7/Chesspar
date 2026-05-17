/**
 * Chesspar v2 — shared types for the single-VLM-call pipeline.
 *
 * Calibration: one call to gemini-3-flash returns 4 named corners + a
 * confirmation that the board is in the starting position.
 * Per-move: one call returns the played SAN given the pre/post images
 * and the legal-move list. No cascade, no abstention dance.
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
  /** Confirmed starting FEN piece-placement, seeded into chess.js. */
  startingFen: string;
};

export type SessionConfig = {
  /** Cloudflare worker root, e.g. https://chesspar-vlm.<acct>.workers.dev */
  proxyUrl: string;
  /** Burst size in frames. */
  burstSize: number;
  /** Burst interval in ms between frames. */
  burstIntervalMs: number;
  /** Canonical rectified board size in pixels. */
  canonicalSize: number;
  /** Margin around the playing surface in the rectified image. */
  rectifyMargin: number;
};

export const DEFAULT_CONFIG: SessionConfig = {
  proxyUrl: "",
  burstSize: 5,
  burstIntervalMs: 40,
  canonicalSize: 768,
  rectifyMargin: 0.14,
};
