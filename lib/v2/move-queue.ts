/**
 * Capture/classify queue for rapid play.
 *
 * The user can tap their clock as fast as they want. Each tap captures
 * a frame, rectifies it, and pushes a PendingCapture onto the FIFO. A
 * background drainer runs identifyMove on items in order — the model
 * still has to classify serially because each call needs the previous
 * move's POST image as its PRE image. But snap-and-clock-tap UX is
 * instant: we don't block on the model.
 *
 * **Drainer design (the source of multiple past bugs).** Earlier
 * implementations triggered the drainer from a useEffect with the
 * queue state in the dep array. That coupled the drainer's
 * cancellation lifecycle to React's effect scheduling, which is racy:
 *   - useEffect cleanup ran when our own setState changed the deps,
 *     setting a `cancelled` flag that discarded the result of an
 *     in-flight identifyMove call.
 *   - Even after fixing that, the effect sometimes failed to re-fire
 *     after a commit because dep equality looked unchanged for one
 *     frame.
 *
 * This implementation removes the useEffect entirely. The drainer is
 * a plain function (`drainOne`) that:
 *   - guards against re-entry with `isDrainingRef`,
 *   - reads the latest state via `stateRef.current`,
 *   - kicks off identifyMove,
 *   - commits or fails via setState,
 *   - **always self-schedules another drainOne via setTimeout(0)**.
 *
 * Every operation that could enable draining (`enqueue`, `initialize`,
 * `resolveFailure`, `dropFailureAndAfter`, `updateAfterRelock`) ends
 * with `setTimeout(drainOne, 0)`. The setTimeout gives React time to
 * commit the preceding setState before drainOne reads stateRef.current.
 *
 * The committed FEN advances only when a queue item resolves. The
 * DISPLAY side-to-move advances on every tap — derived as
 *   sideAfter(committedFen, pendingCount)
 * with pendingCount = queue.length + (inflight ? 1 : 0). Since chess.js
 * only emits moves legal for the side to move at committedFen, the
 * model can't return a wrong-side move, so committed and display sides
 * stay consistent unless we hit a true classification failure.
 *
 * Failure handling: identifyMove returns abstain/error → set `failedAt`
 * and halt the drainer. UI surfaces a banner. The user can apply a
 * best guess (manual SAN entry) or drop and re-snap.
 */

import { useCallback, useRef, useState } from "react";
import { Chess } from "chess.js";
import { identifyMove, type IdentifyResult } from "./identify-move.ts";
import { pushApiLog } from "./api-log.ts";
import type { BoardLock, MoveEntry, PendingCapture, Side } from "./types.ts";

const STARTING_FEN_FULL =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export type QueueFailure = {
  capture: PendingCapture;
  reason: string;
};

export type MoveQueueState = {
  queue: PendingCapture[];
  inflight: PendingCapture | null;
  history: MoveEntry[];
  committedFen: string;
  lastResolvedRectified: HTMLCanvasElement | null;
  lastResolvedRaw: HTMLCanvasElement | null;
  failedAt: QueueFailure | null;
};

export type MoveQueueApi = {
  state: MoveQueueState;
  displaySideToMove: Side;
  committedSideToMove: Side;
  pendingCount: number;
  enqueue: (capture: Omit<PendingCapture, "id">) => void;
  initialize: (opts: {
    rectified: HTMLCanvasElement;
    raw: HTMLCanvasElement;
    fen?: string;
  }) => void;
  updateAfterRelock: (opts: {
    rectified: HTMLCanvasElement;
    raw: HTMLCanvasElement;
  }) => void;
  reset: () => void;
  resolveFailure: (applySan?: string) => void;
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
  onMoveCommitted?: (entry: MoveEntry) => void;
}): MoveQueueApi {
  const [state, setState] = useState<MoveQueueState>(EMPTY_STATE);
  const nextIdRef = useRef(1);
  const lastCapturedAtRef = useRef<number | null>(null);
  const isDrainingRef = useRef(false);

  // Refs that the drainer reads without re-rendering.
  const stateRef = useRef(state);
  stateRef.current = state;
  const lockRef = useRef(opts.lock);
  lockRef.current = opts.lock;
  const proxyRef = useRef(opts.proxyUrl);
  proxyRef.current = opts.proxyUrl;
  const onMoveCommittedRef = useRef(opts.onMoveCommitted);
  onMoveCommittedRef.current = opts.onMoveCommitted;

  // Forward-declare drainOne so the action callbacks can reference it.
  const drainOneRef = useRef<() => void>(() => {});

  const scheduleDrain = useCallback(() => {
    // setTimeout(0) gives React a chance to commit any preceding
    // setState before drainOne reads stateRef.current.
    setTimeout(() => drainOneRef.current(), 0);
  }, []);

  const enqueue = useCallback(
    (capture: Omit<PendingCapture, "id">) => {
      setState((s) => {
        const id = nextIdRef.current++;
        return { ...s, queue: [...s.queue, { ...capture, id }] };
      });
      scheduleDrain();
    },
    [scheduleDrain],
  );

  const initialize = useCallback(
    (opts: { rectified: HTMLCanvasElement; raw: HTMLCanvasElement; fen?: string }) => {
      setState({
        ...EMPTY_STATE,
        committedFen: opts.fen ?? STARTING_FEN_FULL,
        lastResolvedRectified: opts.rectified,
        lastResolvedRaw: opts.raw,
      });
      lastCapturedAtRef.current = Date.now();
      isDrainingRef.current = false;
      scheduleDrain();
    },
    [scheduleDrain],
  );

  const updateAfterRelock = useCallback(
    (opts: { rectified: HTMLCanvasElement; raw: HTMLCanvasElement }) => {
      setState((s) => ({
        ...s,
        lastResolvedRectified: opts.rectified,
        lastResolvedRaw: opts.raw,
      }));
      scheduleDrain();
    },
    [scheduleDrain],
  );

  const reset = useCallback(() => {
    setState(EMPTY_STATE);
    nextIdRef.current = 1;
    lastCapturedAtRef.current = null;
    isDrainingRef.current = false;
  }, []);

  const resolveFailure = useCallback(
    (applySan?: string) => {
      setState((s) => {
        if (!s.failedAt) return s;
        if (!applySan) {
          // Clear failure; next drain retries the same capture (which
          // remains in `inflight` slot or as queue head — but the
          // drainer treats inflight=null as "ready to dequeue", so
          // re-attempting means we need to push the failed capture back
          // to the head of the queue).
          const failedCap = s.failedAt.capture;
          const inflightIsFailed = s.inflight?.id === failedCap.id;
          return {
            ...s,
            failedAt: null,
            inflight: inflightIsFailed ? null : s.inflight,
            queue: inflightIsFailed ? [failedCap, ...s.queue] : s.queue,
          };
        }
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
          onMoveCommittedRef.current?.(entry);
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
      scheduleDrain();
    },
    [scheduleDrain],
  );

  const dropFailureAndAfter = useCallback(() => {
    setState((s) => {
      if (!s.failedAt) return s;
      const failed = s.failedAt.capture;
      const isInflight = s.inflight?.id === failed.id;
      // Advance the pre-image to the failed capture's view so subsequent
      // classifications don't keep comparing against the stale baseline.
      // Even though we couldn't identify the move, the rectified image
      // shows the board AT THAT POINT — closest known approximation to
      // what's on the board now.
      const baselineUpdate = {
        lastResolvedRectified: failed.rectified,
        lastResolvedRaw: failed.rawFrame,
      };
      if (isInflight) {
        return {
          ...s,
          ...baselineUpdate,
          inflight: null,
          queue: [],
          failedAt: null,
        };
      }
      const idx = s.queue.findIndex((c) => c.id === failed.id);
      if (idx < 0) return { ...s, failedAt: null };
      return {
        ...s,
        ...baselineUpdate,
        queue: s.queue.slice(0, idx),
        failedAt: null,
      };
    });
    scheduleDrain();
  }, [scheduleDrain]);

  // ----- The drainer -----
  drainOneRef.current = () => {
    if (isDrainingRef.current) return;
    const s = stateRef.current;
    if (s.failedAt) return;
    if (s.inflight) return; // a previous drain hasn't finished its setState round trip yet
    if (s.queue.length === 0) return;
    if (!s.lastResolvedRectified) return;
    const lock = lockRef.current;
    if (!lock) return;

    isDrainingRef.current = true;
    const head = s.queue[0];
    setState((cur) => ({
      ...cur,
      inflight: head,
      queue: cur.queue.slice(1),
    }));

    (async () => {
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
      if (result.kind === "matched") {
        setState((cur) => {
          try {
            const g = new Chess(cur.committedFen);
            const mv = g.move(result.san);
            if (!mv) {
              pushApiLog({
                callName: "queue-drain",
                model: "internal",
                startedAt: Date.now(),
                finishedAt: Date.now(),
                durationMs: 0,
                ok: false,
                promptPreview: `Couldn't apply ${result.san} to ${cur.committedFen}`,
                outputPreview: "",
                errorMessage: `Illegal in current FEN`,
              });
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
            try {
              onMoveCommittedRef.current?.(entry);
            } catch {
              /* don't let a callback error block the commit */
            }
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
      isDrainingRef.current = false;
      // Always self-schedule: if there's another queued capture this
      // will drain it; otherwise drainOne's early-return guards exit.
      setTimeout(() => drainOneRef.current(), 0);
    })();
  };

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
