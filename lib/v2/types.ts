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

// ============================================================
// Game-mode + chess clock
// ============================================================

export type GameMode =
  | { kind: "untimed" }
  | { kind: "timed"; baseMs: number; incrementMs: number };

export type ClockState = {
  mode: GameMode;
  /** ms remaining for white (only meaningful in timed mode). */
  whiteMs: number;
  /** ms remaining for black (only meaningful in timed mode). */
  blackMs: number;
  /** Which side's clock is currently counting down. null = paused / not started. */
  runningSide: Side | null;
  /** First side to time out, if any. */
  flagged: Side | null;
};

/** Convenience labels for the chess.com-style preset grid. */
export type TimeControlPreset = {
  id: string;
  label: string;
  /** Group heading: Bullet | Blitz | Rapid | Classical. */
  group: "Bullet" | "Blitz" | "Rapid" | "Classical";
  baseMs: number;
  incrementMs: number;
};

export const TIME_CONTROL_PRESETS: TimeControlPreset[] = [
  { id: "1+0", label: "1 + 0", group: "Bullet", baseMs: 60_000, incrementMs: 0 },
  { id: "1+1", label: "1 + 1", group: "Bullet", baseMs: 60_000, incrementMs: 1000 },
  { id: "2+1", label: "2 + 1", group: "Bullet", baseMs: 120_000, incrementMs: 1000 },
  { id: "3+0", label: "3 + 0", group: "Blitz", baseMs: 180_000, incrementMs: 0 },
  { id: "3+2", label: "3 + 2", group: "Blitz", baseMs: 180_000, incrementMs: 2000 },
  { id: "5+0", label: "5 + 0", group: "Blitz", baseMs: 300_000, incrementMs: 0 },
  { id: "5+3", label: "5 + 3", group: "Blitz", baseMs: 300_000, incrementMs: 3000 },
  { id: "10+0", label: "10 + 0", group: "Rapid", baseMs: 600_000, incrementMs: 0 },
  { id: "10+5", label: "10 + 5", group: "Rapid", baseMs: 600_000, incrementMs: 5000 },
  { id: "15+10", label: "15 + 10", group: "Rapid", baseMs: 900_000, incrementMs: 10_000 },
  { id: "30+0", label: "30 + 0", group: "Classical", baseMs: 1_800_000, incrementMs: 0 },
  { id: "30+20", label: "30 + 20", group: "Classical", baseMs: 1_800_000, incrementMs: 20_000 },
];

// ============================================================
// Move queue (rapid-move pipeline)
// ============================================================

/** A single capture awaiting (or undergoing) classification. */
export type PendingCapture = {
  id: number;
  /** Wall-clock ms when the burst started. */
  capturedAt: number;
  /** Raw camera frame (before rectification). Sent as rawPostImage. */
  rawFrame: HTMLCanvasElement;
  /** Rectified post-move board, used as postImage for identifyMove. */
  rectified: HTMLCanvasElement;
  /** Which side tapped the clock — used only for UI feedback / debugging. */
  byClockSide: Side;
};

/** One resolved move in the score sheet. */
export type MoveEntry = {
  san: string;
  side: Side;
  capturedAt: number;
  resolvedAt: number;
  /** Wall-clock ms between this move's capture and the previous move's capture
   *  (or the lock, for white's first move). Drives the duration column. */
  thinkDurationMs: number;
};
