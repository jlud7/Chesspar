/**
 * Move pipeline — orchestrator the UI calls.
 *
 * One pass per user tap:
 *   1. Refresh the board lock against the new frame (so a phone shift
 *      between moves doesn't poison the warp).
 *   2. Warp the sharpest burst frame to canonical board space.
 *   3. Enumerate every legal move from chess.js.
 *   4. Hand the legal list + post-move image to Gemini Flash and ask
 *      which move was played.
 *   5. Emit if confidence ≥ threshold, otherwise abstain with the top
 *      candidates for tap-to-confirm.
 */

import { Chess, type Move } from "chess.js";
import {
  BOARD_CONTEXT_MARGIN,
  rectifyWithContext,
  refreshBoardLock,
} from "./board-lock";
import {
  classifyMoveWithFlash,
  type FlashCandidate,
} from "./gemini-flash-classifier";
import type {
  BoardLock,
  CapturedBurst,
  MoveCandidate,
  MoveDecision,
  SessionConfig,
  Square,
} from "./types";

export type PipelineInput = {
  burst: CapturedBurst;
  lock: BoardLock;
  previousRectified?: HTMLCanvasElement | null;
  previousFen: string;
  config: SessionConfig;
};

export type PipelineOutput = {
  decision: MoveDecision;
  /** Rectified post-move canvas — caller should keep this around for the
   *  abstention UI / next-move baseline display. */
  rectified: HTMLCanvasElement;
  /** Possibly-refreshed lock — caller should adopt this so future
   *  captures use the latest corners. */
  lock: BoardLock;
  /** Trace data for diagnostics / debug overlay. */
  trace: {
    legalCount: number;
    flashSan?: string;
    flashConfidence?: number;
    flashLatencyMs?: number;
    flashRaw?: string;
    visualSquares?: Square[];
    cornerRefresh: "cv" | "claude" | "gemini" | "kept";
  };
};

export async function runMovePipeline(
  input: PipelineInput,
): Promise<PipelineOutput> {
  const t0 = performance.now();
  const size = input.config.canonicalSize;

  // Per-capture geometry refresh, but orientation remains locked to the
  // calibrated board. This prevents mid-game VLM corner relabeling from
  // rotating or mirroring the board underneath the legal-move classifier.
  const refreshed = await refreshBoardLock(input.burst.frame, input.lock, {
    proxyUrl: input.config.proxyUrl,
  });
  const liveLock = refreshed.lock;
  const rectified = rectifyWithContext(input.burst.frame, liveLock, size);
  const visualSquares = input.previousRectified
    ? findChangedSquares(input.previousRectified, rectified)
    : [];

  const game = new Chess(input.previousFen);
  const legal = game.moves({ verbose: true });

  if (legal.length === 0) {
    return {
      decision: {
        kind: "error",
        reason:
          "No legal moves from the previous position — the game may already be over.",
        latencyMs: performance.now() - t0,
      },
      rectified,
      lock: liveLock,
      trace: { legalCount: 0, cornerRefresh: refreshed.detector },
    };
  }

  const candidates: MoveCandidate[] = legal.map((m) =>
    buildCandidate(input.previousFen, m),
  );
  const flashCandidates: FlashCandidate[] = candidates.map((c) => ({
    san: c.san,
    uci: c.uci,
    fromSquare: c.fromSquare,
    toSquare: c.toSquare,
  }));

  const flash = await classifyMoveWithFlash({
    proxyUrl: input.config.proxyUrl,
    previousFen: input.previousFen,
    candidates: flashCandidates,
    preImage: input.previousRectified,
    postImage: rectified,
  });

  const trace = {
    legalCount: legal.length,
    flashSan: flash.kind === "matched" ? flash.san : undefined,
    flashConfidence: flash.kind === "matched" ? flash.confidence : undefined,
    flashLatencyMs: flash.latencyMs,
    flashRaw: flash.kind === "error" ? undefined : flash.raw,
    visualSquares,
    cornerRefresh: refreshed.detector,
  };
  const latencyMs = performance.now() - t0;

  if (flash.kind === "error") {
    return {
      decision: {
        kind: "error",
        reason: `Move classifier failed: ${flash.reason}`,
        latencyMs,
      },
      rectified,
      lock: liveLock,
      trace,
    };
  }

  if (flash.kind === "abstain") {
    return {
      decision: {
        kind: "abstain",
        candidates: candidates.slice(0, 4),
        reason: "The model couldn't tell which legal move was played.",
        pConfident: 0,
        latencyMs,
      },
      rectified,
      lock: liveLock,
      trace,
    };
  }

  // matched
  const pickIdx = candidates.findIndex((c) => c.uci === flash.uci);
  if (pickIdx < 0) {
    return {
      decision: {
        kind: "error",
        reason: `Move classifier returned ${flash.uci}, which is not legal in the previous position.`,
        latencyMs,
      },
      rectified,
      lock: liveLock,
      trace,
    };
  }
  const pick = candidates[pickIdx];
  const visuallyRanked = rankCandidatesByVisualDiff(
    input.previousFen,
    candidates,
    visualSquares,
  );
  const visualAgreement = visualSquares.length === 0
    ? false
    : candidateMatchesVisual(input.previousFen, pick, visualSquares);
  const geminiAgreement = flash.changedSquares.length === 0
    ? true
    : candidateMatchesVisual(input.previousFen, pick, flash.changedSquares);
  const alternates = visuallyRanked
    .filter((c) => c.uci !== pick.uci)
    .slice(0, 3);

  if (!visualAgreement || !geminiAgreement) {
    const candidatesForUi = [pick, ...alternates].slice(0, 4);
    return {
      decision: {
        kind: "abstain",
        candidates: candidatesForUi,
        reason:
          visualSquares.length === 0
            ? "I couldn't see a clear before/after change. Capture again after the move is made."
            : `Model picked ${flash.san}, but the changed squares look like ${visualSquares.join(", ")}. Please confirm.`,
        pConfident: Math.min(flash.confidence, 0.6),
        latencyMs,
      },
      rectified,
      lock: liveLock,
      trace,
    };
  }

  if (flash.confidence >= input.config.emitThreshold) {
    return {
      decision: {
        kind: "matched",
        pick,
        alternates,
        pConfident: Math.min(flash.confidence, 0.99),
        latencyMs,
      },
      rectified,
      lock: liveLock,
      trace,
    };
  }

  return {
    decision: {
      kind: "abstain",
      candidates: [pick, ...alternates].slice(0, 4),
      reason: `Model picked ${flash.san} (${Math.round(flash.confidence * 100)}%) — please confirm.`,
      pConfident: flash.confidence,
      latencyMs,
    },
    rectified,
    lock: liveLock,
    trace,
  };
}

/** Apply a chosen SAN to the previous FEN and return the new FEN. Used
 *  after a successful emit (or after the user resolves an abstention). */
export function applyMove(previousFen: string, san: string): string | null {
  try {
    const game = new Chess(previousFen);
    const move = game.move(san);
    return move ? game.fen() : null;
  } catch {
    return null;
  }
}

function buildCandidate(previousFen: string, move: Move): MoveCandidate {
  const sim = new Chess(previousFen);
  sim.move({ from: move.from, to: move.to, promotion: move.promotion });
  return {
    san: move.san,
    uci: move.from + move.to + (move.promotion ?? ""),
    fromSquare: move.from as Square,
    toSquare: move.to as Square,
    resultingFen: sim.fen(),
  };
}

function findChangedSquares(
  before: HTMLCanvasElement,
  after: HTMLCanvasElement,
): Square[] {
  if (before.width !== after.width || before.height !== after.height) return [];
  const size = after.width;
  if (after.height !== size) return [];
  const bctx = before.getContext("2d", { willReadFrequently: true });
  const actx = after.getContext("2d", { willReadFrequently: true });
  if (!bctx || !actx) return [];

  const pad = size * BOARD_CONTEXT_MARGIN;
  const grid = size - 2 * pad;
  const cell = grid / 8;
  const inset = cell * 0.22;
  const sampleSize = Math.max(4, Math.round(cell - 2 * inset));

  const deltas: Array<{ square: Square; delta: number }> = [];
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const x = Math.round(pad + col * cell + inset);
      const y = Math.round(pad + row * cell + inset);
      const square = indexToSquare(row * 8 + col);
      deltas.push({
        square,
        delta: meanAbsRgbDiff(bctx, actx, x, y, sampleSize, sampleSize),
      });
    }
  }

  const values = deltas.map((d) => d.delta).sort((a, b) => a - b);
  const median = values[Math.floor(values.length / 2)] ?? 0;
  const top = [...deltas].sort((a, b) => b.delta - a.delta);
  const threshold = Math.max(10, median + 12);
  return top
    .filter((d) => d.delta >= threshold)
    .slice(0, 6)
    .map((d) => d.square);
}

function meanAbsRgbDiff(
  before: CanvasRenderingContext2D,
  after: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
): number {
  const a = before.getImageData(x, y, w, h).data;
  const b = after.getImageData(x, y, w, h).data;
  let total = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    total +=
      Math.abs(a[i] - b[i]) +
      Math.abs(a[i + 1] - b[i + 1]) +
      Math.abs(a[i + 2] - b[i + 2]);
    n += 3;
  }
  return n ? total / n : 0;
}

function rankCandidatesByVisualDiff(
  previousFen: string,
  candidates: MoveCandidate[],
  visualSquares: Square[],
): MoveCandidate[] {
  return [...candidates].sort((a, b) => {
    const aScore = visualMatchScore(previousFen, a, visualSquares);
    const bScore = visualMatchScore(previousFen, b, visualSquares);
    return bScore - aScore;
  });
}

function candidateMatchesVisual(
  previousFen: string,
  candidate: MoveCandidate,
  visualSquares: Square[],
): boolean {
  if (visualSquares.length === 0) return false;
  return visualMatchScore(previousFen, candidate, visualSquares) >= 2;
}

function visualMatchScore(
  previousFen: string,
  candidate: MoveCandidate,
  visualSquares: Square[],
): number {
  const expected = expectedTouchedSquares(previousFen, candidate);
  let score = 0;
  for (const sq of expected) {
    if (visualSquares.includes(sq)) score++;
  }
  return score;
}

function expectedTouchedSquares(
  previousFen: string,
  candidate: MoveCandidate,
): Square[] {
  const game = new Chess(previousFen);
  const move = game
    .moves({ verbose: true })
    .find((m) => m.from + m.to + (m.promotion ?? "") === candidate.uci);
  const touched = new Set<Square>([candidate.fromSquare, candidate.toSquare]);
  if (move?.san === "O-O") {
    touched.add((move.color === "w" ? "h1" : "h8") as Square);
    touched.add((move.color === "w" ? "f1" : "f8") as Square);
  } else if (move?.san === "O-O-O") {
    touched.add((move.color === "w" ? "a1" : "a8") as Square);
    touched.add((move.color === "w" ? "d1" : "d8") as Square);
  }
  if (move?.flags.includes("e")) {
    const file = candidate.toSquare[0];
    const rank = candidate.fromSquare[1];
    touched.add(`${file}${rank}` as Square);
  }
  return [...touched];
}

function indexToSquare(idx: number): Square {
  const file = String.fromCharCode("a".charCodeAt(0) + (idx % 8));
  const rank = 8 - Math.floor(idx / 8);
  return `${file}${rank}` as Square;
}
