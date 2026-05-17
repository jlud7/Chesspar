/**
 * Capture/classify queue for rapid play.
 *
 * The user can tap their clock as fast as they want. Each tap captures
 * a frame, rectifies it, and pushes a PendingCapture onto the FIFO. A
 * background drainer effect runs identifyMove on items in order — the
 * model still has to classify serially because each call needs the
 * previous move's POST image as its PRE image. But snap-and-clock-tap
 * UX is instant: we don't block on the model.
 *
 * The committed FEN advances only when a queue item resolves. The
 * DISPLAY side-to-move advances on every tap — derived as
 *   sideAfter(committedFen, pendingCount)
 * with pendingCount = queue.length + (inflight ? 1 : 0). Since chess.js
 * only emits moves legal for the side to move at committedFen, the
 * model can't return a move for the wrong side, so the committed and
 * display sides stay consistent unless we hit a true classification
 * failure.
 *
 * Failure handling: if identifyMove returns abstain or error, we set
 * `failedAt` and halt the drainer. The UI surfaces a banner with the
 * model's reason. The user can apply a best guess (manual SAN entry)
 * or stop and re-snap. Pending captures stay buffered.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import { identifyMove, type IdentifyResult } from "./identify-move.ts";
import type { BoardLock, MoveEntry, PendingCapture, Side } from "./types.ts";

const STARTING_FEN_FULL =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export type QueueFailure = {
  capture: PendingCapture;
  reason: string;
};

export type MoveQueueState = {
  /** Captures awaiting classification. */
  queue: PendingCapture[];
  /** Capture currently in the model. */
  inflight: PendingCapture | null;
  /** Resolved moves, in order. */
  history: MoveEntry[];
  /** Game FEN after the last RESOLVED move. */
  committedFen: string;
  /** Pre-image for the next identifyMove call. Starts as the calibration's
   *  rectified starting position, then advances to each resolved post-image. */
  lastResolvedRectified: HTMLCanvasElement | null;
  /** Raw camera frame paired with `lastResolvedRectified`, sent as
   *  rawPreImage to identifyMove. */
  lastResolvedRaw: HTMLCanvasElement | null;
  /** Set when classification fails. Halts the drainer until cleared. */
  failedAt: QueueFailure | null;
};

export type MoveQueueApi = {
  state: MoveQueueState;
  /** Optimistic side-to-move (derived from queue depth). */
  displaySideToMove: Side;
  /** Side derived from committedFen alone. */
  committedSideToMove: Side;
  /** Pending count = queue + inflight. */
  pendingCount: number;
  /** Push a fresh capture onto the queue. */
  enqueue: (capture: Omit<PendingCapture, "id">) => void;
  /** Initialize with the calibration's starting-position rectified canvas
   *  + raw frame. Must be called once before any enqueue. CLEARS history. */
  initialize: (opts: {
    rectified: HTMLCanvasElement;
    raw: HTMLCanvasElement;
    fen?: string;
  }) => void;
  /** After a mid-game re-lock: swap in the new rectified/raw pair (the new
   *  geometry) WITHOUT clearing history or committedFen. The new pair
   *  becomes the pre-image for the next classification. */
  updateAfterRelock: (opts: {
    rectified: HTMLCanvasElement;
    raw: HTMLCanvasElement;
  }) => void;
  /** Clear the queue + history (e.g. after End Game). */
  reset: () => void;
  /** Acknowledge a failure: caller has dealt with it, drain may resume.
   *  If `applySan` is provided, commit it for the failed capture and pop. */
  resolveFailure: (applySan?: string) => void;
  /** Drop the failed capture (and everything behind it) so the user can
   *  re-snap manually from a clean state. */
  dropFailureAndAfter: () => void;
};

const EMPTY_STATE: MoveQueueState = {
  queue: [],
  inflight: null,
  history: [],
  committedFen: STARTING_FEN_FULL,
  lastResolvedRectified: null,
  lastResolvedRaw: null,
  failedAt: null,
};

export function useMoveQueue(opts: {
  lock: BoardLock | null;
  proxyUrl: string;
  /** Notify caller when a queued move successfully commits — used by the
   *  clock to call switchTo() with the next side and apply increment. */
  onMoveCommitted?: (entry: MoveEntry) => void;
}): MoveQueueApi {
  const [state, setState] = useState<MoveQueueState>(EMPTY_STATE);
  const nextIdRef = useRef(1);
  // Track the last-committed wall-clock for the next thinkDurationMs calc.
  const lastCapturedAtRef = useRef<number | null>(null);

  const enqueue = useCallback((capture: Omit<PendingCapture, "id">) => {
    setState((s) => {
      const id = nextIdRef.current++;
      return { ...s, queue: [...s.queue, { ...capture, id }] };
    });
  }, []);

  const initialize = useCallback(
    (opts: { rectified: HTMLCanvasElement; raw: HTMLCanvasElement; fen?: string }) => {
      setState({
        ...EMPTY_STATE,
        committedFen: opts.fen ?? STARTING_FEN_FULL,
        lastResolvedRectified: opts.rectified,
        lastResolvedRaw: opts.raw,
      });
      lastCapturedAtRef.current = Date.now();
    },
    [],
  );

  const updateAfterRelock = useCallback(
    (opts: { rectified: HTMLCanvasElement; raw: HTMLCanvasElement }) => {
      setState((s) => ({
        ...s,
        lastResolvedRectified: opts.rectified,
        lastResolvedRaw: opts.raw,
      }));
    },
    [],
  );

  const reset = useCallback(() => {
    setState(EMPTY_STATE);
    nextIdRef.current = 1;
    lastCapturedAtRef.current = null;
  }, []);

  const resolveFailure = useCallback((applySan?: string) => {
    setState((s) => {
      if (!s.failedAt) return s;
      if (!applySan) {
        // Just clear the failure; drainer will retry the same capture.
        return { ...s, failedAt: null };
      }
      // Commit `applySan` for the failed capture, pop it from queue/inflight.
      try {
        const g = new Chess(s.committedFen);
        const mv = g.move(applySan);
        if (!mv) return s;
        const cap = s.failedAt.capture;
        const resolvedAt = Date.now();
        const entry: MoveEntry = {
          san: mv.san,
          side: cap.byClockSide,
          capturedAt: cap.capturedAt,
          resolvedAt,
          thinkDurationMs:
            lastCapturedAtRef.current != null
              ? cap.capturedAt - lastCapturedAtRef.current
              : 0,
        };
        lastCapturedAtRef.current = cap.capturedAt;
        // Remove the failed capture from queue/inflight.
        const isInflight = s.inflight?.id === cap.id;
        return {
          ...s,
          queue: isInflight ? s.queue : s.queue.filter((c) => c.id !== cap.id),
          inflight: isInflight ? null : s.inflight,
          history: [...s.history, entry],
          committedFen: g.fen(),
          lastResolvedRectified: cap.rectified,
          lastResolvedRaw: cap.rawFrame,
          failedAt: null,
        };
      } catch {
        return s;
      }
    });
  }, []);

  const dropFailureAndAfter = useCallback(() => {
    setState((s) => {
      if (!s.failedAt) return s;
      const failedId = s.failedAt.capture.id;
      // Find index of the failed capture; if it's inflight, drop ONLY it.
      // If it's in the queue, drop it + everything after.
      const isInflight = s.inflight?.id === failedId;
      if (isInflight) {
        return { ...s, inflight: null, queue: [], failedAt: null };
      }
      const idx = s.queue.findIndex((c) => c.id === failedId);
      if (idx < 0) return { ...s, failedAt: null };
      return {
        ...s,
        queue: s.queue.slice(0, idx),
        failedAt: null,
      };
    });
  }, []);

  // ----- Drainer -----
  // Keep the latest state in a ref so the drainer effect doesn't capture
  // a stale closure when it re-runs.
  const stateRef = useRef(state);
  stateRef.current = state;
  const lockRef = useRef(opts.lock);
  lockRef.current = opts.lock;
  const proxyRef = useRef(opts.proxyUrl);
  proxyRef.current = opts.proxyUrl;
  const onMoveCommittedRef = useRef(opts.onMoveCommitted);
  onMoveCommittedRef.current = opts.onMoveCommitted;

  useEffect(() => {
    const s = stateRef.current;
    if (s.failedAt) return;
    if (s.inflight) return;
    if (s.queue.length === 0) return;
    if (!s.lastResolvedRectified) return;
    const lock = lockRef.current;
    if (!lock) return;

    const head = s.queue[0];
    // Move head to inflight + remove from queue atomically.
    setState((cur) => ({
      ...cur,
      inflight: head,
      queue: cur.queue.slice(1),
    }));

    let cancelled = false;
    const run = async () => {
      let result: IdentifyResult;
      try {
        result = await identifyMove({
          proxyUrl: proxyRef.current,
          previousFen: stateRef.current.committedFen,
          preImage: stateRef.current.lastResolvedRectified!,
          postImage: head.rectified,
          rawPreImage: stateRef.current.lastResolvedRaw ?? undefined,
          rawPostImage: head.rawFrame,
        });
      } catch (e) {
        result = {
          kind: "error",
          reason: e instanceof Error ? e.message : String(e),
          durationMs: 0,
        };
      }
      if (cancelled) return;
      if (result.kind === "matched") {
        // Commit + advance.
        setState((cur) => {
          try {
            const g = new Chess(cur.committedFen);
            const mv = g.move(result.san);
            if (!mv) {
              return {
                ...cur,
                inflight: null,
                failedAt: {
                  capture: head,
                  reason: `Inferred ${result.san} but couldn't apply to current FEN.`,
                },
              };
            }
            const resolvedAt = Date.now();
            const entry: MoveEntry = {
              san: mv.san,
              side: head.byClockSide,
              capturedAt: head.capturedAt,
              resolvedAt,
              thinkDurationMs:
                lastCapturedAtRef.current != null
                  ? head.capturedAt - lastCapturedAtRef.current
                  : 0,
            };
            lastCapturedAtRef.current = head.capturedAt;
            onMoveCommittedRef.current?.(entry);
            return {
              ...cur,
              inflight: null,
              history: [...cur.history, entry],
              committedFen: g.fen(),
              lastResolvedRectified: head.rectified,
              lastResolvedRaw: head.rawFrame,
            };
          } catch (e) {
            return {
              ...cur,
              inflight: null,
              failedAt: {
                capture: head,
                reason: e instanceof Error ? e.message : String(e),
              },
            };
          }
        });
      } else {
        setState((cur) => ({
          ...cur,
          inflight: null,
          failedAt: {
            capture: head,
            reason: result.reason,
          },
        }));
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [state.queue.length, state.inflight, state.failedAt, state.lastResolvedRectified]);

  // Derived values
  const pendingCount = state.queue.length + (state.inflight ? 1 : 0);
  const committedSideToMove: Side =
    state.committedFen.split(/\s+/)[1] === "b" ? "black" : "white";
  const displaySideToMove: Side =
    pendingCount % 2 === 0
      ? committedSideToMove
      : committedSideToMove === "white"
        ? "black"
        : "white";

  return {
    state,
    displaySideToMove,
    committedSideToMove,
    pendingCount,
    enqueue,
    initialize,
    updateAfterRelock,
    reset,
    resolveFailure,
    dropFailureAndAfter,
  };
}
