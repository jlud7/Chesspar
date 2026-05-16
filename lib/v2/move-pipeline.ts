/**
 * Move pipeline — the orchestrator the UI calls.
 *
 * One pass per user tap:
 *   1. Warp the sharpest burst frame to canonical board space.
 *   2. Diff against the cached pre-move frame to find changed squares.
 *   3. Beam-search legal moves whose template matches.
 *   4. Score confidence; emit if ≥ threshold.
 *   5. Otherwise escalate to Gemini 2.5 Pro on the disputed candidates.
 *   6. If still unsure, abstain with the top-2 for tap-to-confirm.
 *
 * State the caller is expected to keep between moves:
 *   - the BoardLock from session start
 *   - the previous FEN (chess.js position before this move)
 *   - the previous rectified canvas (becomes the pre-frame for diff)
 */

import { Chess } from "chess.js";
import { rectifyWithLock } from "./board-lock";
import { computeOccupancyDelta } from "./occupancy-delta";
import { searchLegalMoves } from "./move-search";
import { pCorrect, buildFeatures } from "./confidence";
import { adjudicateMove } from "./gemini-adjudicator";
import type {
  BoardLock,
  CapturedBurst,
  MoveDecision,
  SessionConfig,
} from "./types";

export type PipelineInput = {
  burst: CapturedBurst;
  lock: BoardLock;
  previousFen: string;
  /** Rectified canvas from the previous emit (or the calibration frame
   *  for move #1). The pre side of the diff. */
  previousRectified: HTMLCanvasElement;
  config: SessionConfig;
};

export type PipelineOutput = {
  decision: MoveDecision;
  /** Rectified post-move canvas — caller should keep this as the next
   *  move's pre-frame on a successful emit. */
  rectified: HTMLCanvasElement;
  /** Trace data for diagnostics / debug overlay. */
  trace: {
    changedSquares: string[];
    perSquareDelta: number[];
    candidateSummary: Array<{ san: string; score: number }>;
    pConfident: number;
    laplacian: number;
    escalation: "none" | "vlm";
  };
};

export async function runMovePipeline(
  input: PipelineInput,
): Promise<PipelineOutput> {
  const t0 = performance.now();
  const size = input.config.canonicalSize;
  const rectified = rectifyWithLock(input.burst.frame, input.lock, size);

  const diff = computeOccupancyDelta(input.previousRectified, rectified, {
    size,
  });
  diff.frameSharpness = input.burst.variance;

  // Run beam search even when changedSquares < 2 — chess.js will return
  // an empty list and we surface a meaningful "no change detected" error
  // rather than crashing.
  let candidates = searchLegalMoves(input.previousFen, diff);

  // Edge case: zero candidates because the diff missed a square that the
  // legality filter requires. Relax the slack to 2 and try again — beats
  // an immediate VLM call.
  if (candidates.length === 0 && diff.changedSquares.length > 0) {
    candidates = searchLegalMoves(input.previousFen, diff, undefined, {
      slack: 2,
    });
  }

  if (candidates.length === 0) {
    return {
      decision: {
        kind: "error",
        reason:
          diff.changedSquares.length === 0
            ? "No change detected — make sure your move is complete and the board is fully in frame"
            : "No legal move matches the observed change — re-check the previous position",
        latencyMs: performance.now() - t0,
      },
      rectified,
      trace: {
        changedSquares: diff.changedSquares,
        perSquareDelta: diff.perSquareDelta,
        candidateSummary: [],
        pConfident: 0,
        laplacian: input.burst.variance,
        escalation: "none",
      },
    };
  }

  const features = buildFeatures(candidates, diff);
  const pConf0 = pCorrect(features);
  let escalation: "none" | "vlm" = "none";
  let chosenSan = candidates[0].san;
  let pConfident = pConf0;

  // Escalate to VLM if confidence is below threshold AND we have at
  // least two plausible candidates AND the user enabled the proxy.
  // Single-candidate moves go through directly because there is nothing
  // to adjudicate.
  if (
    pConf0 < input.config.emitThreshold &&
    candidates.length >= 2 &&
    input.config.enableVlmEscalation &&
    input.config.proxyUrl
  ) {
    const adj = await adjudicateMove({
      proxyUrl: input.config.proxyUrl,
      previousFen: input.previousFen,
      candidates: candidates.slice(0, 5),
      postFrame: rectified,
      preFrame: input.previousRectified,
    });
    if (adj.kind === "matched") {
      chosenSan = adj.san;
      // Re-rank: put the chosen candidate first; keep others as alternates.
      const idx = candidates.findIndex((c) => c.san === adj.san);
      if (idx > 0) {
        const [pick] = candidates.splice(idx, 1);
        candidates.unshift(pick);
      }
      escalation = "vlm";
      // VLM raises confidence proportional to its own self-reported value,
      // but capped so a deeply-uncertain VLM can't push us past the
      // abstain threshold.
      pConfident = Math.min(0.998, pConf0 + 0.5 * adj.confidence);
    } else if (adj.kind === "abstain") {
      escalation = "vlm";
      // Abstain at the orchestrator level too — the VLM saw the same
      // ambiguity we did.
      pConfident = Math.min(pConf0, 0.6);
    }
    // On VLM error, fall back to the pre-VLM confidence — the orchestrator
    // will abstain if below threshold.
  }

  const trace = {
    changedSquares: diff.changedSquares,
    perSquareDelta: diff.perSquareDelta,
    candidateSummary: candidates
      .slice(0, 5)
      .map((c) => ({ san: c.san, score: c.score })),
    pConfident,
    laplacian: input.burst.variance,
    escalation,
  };
  const latencyMs = performance.now() - t0;

  if (pConfident >= input.config.emitThreshold) {
    // Promote chosenSan to the front if VLM swapped order.
    const pickIdx = candidates.findIndex((c) => c.san === chosenSan);
    const pick = candidates[Math.max(0, pickIdx)];
    return {
      decision: {
        kind: "matched",
        pick,
        alternates: candidates.filter((_, i) => i !== pickIdx).slice(0, 3),
        pConfident,
        latencyMs,
        escalation,
      },
      rectified,
      trace,
    };
  }

  return {
    decision: {
      kind: "abstain",
      candidates: candidates.slice(0, 3),
      reason:
        escalation === "vlm"
          ? "VLM was uncertain — please confirm the move"
          : "Multiple moves match the change — please confirm",
      pConfident,
      latencyMs,
    },
    rectified,
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
