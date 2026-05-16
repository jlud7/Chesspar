/**
 * Chesspar v2 — diff-first hybrid move detection.
 *
 * Architecture: Option C from the research PDFs at the repo root.
 * Capture a sharp post-move frame, warp to canonical board, diff against
 * the cached pre-move frame, enumerate legal moves whose touched-square
 * set explains the diff, escalate ambiguities to a VLM, abstain rather
 * than emit silent errors.
 */

export type Point = { x: number; y: number };

/** Image-space chess-board corners, ordered [a8, h8, h1, a1]. */
export type Corners = readonly [Point, Point, Point, Point];

export type Side = "white" | "black";

/** Per-square label space for the diff/classifier path. */
export type SquareState = "empty" | "white" | "black";

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
 * sharpest of the 5; `variance` is its Laplacian variance (used by the
 * confidence model + smoke-test rejection of motion-blurred bursts).
 */
export type CapturedBurst = {
  frame: HTMLCanvasElement;
  variance: number;
  /** Wall-clock ms when the burst was triggered. */
  capturedAt: number;
  /** All 5 frames, ranked sharpest-first. Kept for VLM escalation. */
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

/** What the diff detector emits — 0–6 candidate changed squares. */
export type DiffResult = {
  /** Algebraic squares where the HSV-V delta exceeded the threshold. */
  changedSquares: Square[];
  /** Per-square normalized delta in [0, 1]. Length 64, row-major from a8. */
  perSquareDelta: number[];
  /** Sharpest-frame Laplacian variance, propagated for confidence scoring. */
  frameSharpness: number;
};

/**
 * A scored legal-move candidate from the beam search.
 *
 * `score` is in nats and aggregates: log P(piece on destination | classifier),
 * occupancy-delta alignment with the move's touched-square template, and
 * a small Stockfish prior bonus (capped per the PDFs to avoid pushing
 * sub-1400 players' unnatural moves out of the candidate set).
 */
export type MoveCandidate = {
  san: string;
  uci: string;
  fromSquare: Square;
  toSquare: Square;
  /** Resulting FEN if this move is played. */
  resultingFen: string;
  /** Touched-square template (length 2–4 depending on castle/EP). */
  touchedSquares: Square[];
  score: number;
  /** Subscores kept for the confidence model. */
  subscores: {
    diffAlignment: number;
    classifierLogP: number;
    enginePrior: number;
  };
};

/**
 * Final outcome of one move-detection attempt. The orchestrator emits one
 * of these per user tap; the UI renders accordingly.
 */
export type MoveDecision =
  | {
      kind: "matched";
      pick: MoveCandidate;
      /** Top alternates kept for the abstention UI on later disagreement. */
      alternates: MoveCandidate[];
      pConfident: number;
      latencyMs: number;
      escalation: "none" | "vlm";
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
  /** Whether to call Gemini 2.5 Pro on disputed tiles. */
  enableVlmEscalation: boolean;
  /** Confidence threshold to emit without abstaining. PDFs recommend 0.99. */
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
  enableVlmEscalation: true,
  emitThreshold: 0.97,
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
