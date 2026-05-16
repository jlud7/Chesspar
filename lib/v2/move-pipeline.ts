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
import { rectifyWithContext, refreshBoardLock } from "./board-lock";
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
    postImage: rectified,
  });

  const trace = {
    legalCount: legal.length,
    flashSan: flash.kind === "matched" ? flash.san : undefined,
    flashConfidence: flash.kind === "matched" ? flash.confidence : undefined,
    flashLatencyMs: flash.latencyMs,
    flashRaw: flash.kind === "error" ? undefined : flash.raw,
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
  const alternates = candidates.filter((_, i) => i !== pickIdx).slice(0, 3);

  if (flash.confidence >= input.config.emitThreshold) {
    return {
      decision: {
        kind: "matched",
        pick,
        alternates,
        pConfident: flash.confidence,
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
