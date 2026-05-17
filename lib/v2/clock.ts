/**
 * Chess clock + time formatting for v2 capture.
 *
 * Two game modes:
 *   - "untimed" — no tick loop, sides just show YOUR TURN / WAITING.
 *   - "timed"   — base + Fischer-style increment, decrements the running
 *                 side via requestAnimationFrame, flags at zero.
 *
 * The clock is driven by `displaySideToMove` (the optimistic side derived
 * from queued captures), not the committed FEN side. When the player
 * taps their clock, the opponent's clock starts ticking IMMEDIATELY,
 * even if classification is still in flight — this is what makes rapid
 * play feel responsive.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { ClockState, GameMode, Side } from "./types.ts";

/**
 * Format a duration in milliseconds for the clock face.
 * - ≤0          → "0:00"
 * - <10 seconds → one decimal place ("0.5", "9.2")
 * - <1 hour     → "m:ss"
 * - ≥1 hour     → "h:mm:ss"
 */
export function formatTime(ms: number): string {
  if (ms <= 0) return "0:00";
  if (ms < 10_000) {
    return (ms / 1000).toFixed(1);
  }
  const totalSec = Math.ceil(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  if (m >= 60) {
    const h = Math.floor(m / 60);
    return `${h}:${String(m % 60).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Format a small Δ duration shown in the score sheet, e.g. "12s" or "1:23". */
export function formatDelta(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  if (ms < 60_000) return `${Math.round(ms / 1000)}s`;
  const totalSec = Math.round(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Short human label for a GameMode — used in the score sheet header. */
export function describeGameMode(mode: GameMode): string {
  if (mode.kind === "untimed") return "Untimed";
  const baseMin = mode.baseMs / 60_000;
  const baseStr = Number.isInteger(baseMin) ? `${baseMin}` : baseMin.toFixed(1);
  return `${baseStr} + ${Math.round(mode.incrementMs / 1000)}`;
}

export type ChessClockApi = {
  state: ClockState;
  /** Begin the game with white on move. No-op if already running. */
  start: () => void;
  /** Switch the running side (called by the queue when the active player taps). */
  switchTo: (next: Side) => void;
  /** Pause the clock (e.g. when the game ends or the user opens settings). */
  pause: () => void;
  /** Reset to the initial state. */
  reset: (mode?: GameMode) => void;
};

/**
 * Chess-clock state machine. The hook owns wall-clock arithmetic via a
 * single requestAnimationFrame loop so React renders only ~10× per
 * second (we round display values to whole seconds outside the final
 * 10s; the underlying state stays precise).
 */
export function useChessClock(initialMode: GameMode): ChessClockApi {
  const initial: ClockState = {
    mode: initialMode,
    whiteMs: initialMode.kind === "timed" ? initialMode.baseMs : 0,
    blackMs: initialMode.kind === "timed" ? initialMode.baseMs : 0,
    runningSide: null,
    flagged: null,
  };
  const [state, setState] = useState<ClockState>(initial);
  // Wall-clock instant at which the current `runningSide` started ticking.
  // Kept in a ref because the rAF loop reads it without re-rendering.
  const tickStartRef = useRef<number | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  // Tick loop — only runs when there's a running side AND we're timed.
  useEffect(() => {
    if (state.mode.kind !== "timed") return;
    if (!state.runningSide) return;
    if (state.flagged) return;

    let raf = 0;
    const tick = () => {
      const cur = stateRef.current;
      if (cur.mode.kind !== "timed" || !cur.runningSide || cur.flagged) return;
      const now = performance.now();
      const start = tickStartRef.current;
      if (start == null) {
        tickStartRef.current = now;
        raf = requestAnimationFrame(tick);
        return;
      }
      const elapsed = now - start;
      tickStartRef.current = now;
      const next: ClockState = {
        ...cur,
        whiteMs: cur.runningSide === "white" ? cur.whiteMs - elapsed : cur.whiteMs,
        blackMs: cur.runningSide === "black" ? cur.blackMs - elapsed : cur.blackMs,
      };
      if (next.whiteMs <= 0 && next.runningSide === "white") {
        next.whiteMs = 0;
        next.flagged = "white";
        next.runningSide = null;
      } else if (next.blackMs <= 0 && next.runningSide === "black") {
        next.blackMs = 0;
        next.flagged = "black";
        next.runningSide = null;
      }
      setState(next);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      tickStartRef.current = null;
    };
  }, [state.mode.kind, state.runningSide, state.flagged]);

  const start = useCallback(() => {
    setState((s) => (s.runningSide ? s : { ...s, runningSide: "white" }));
  }, []);

  const switchTo = useCallback((next: Side) => {
    setState((s) => {
      if (s.flagged) return s;
      if (s.runningSide === next) return s; // already that side
      // Apply Fischer-style increment to the side that just stopped.
      let whiteMs = s.whiteMs;
      let blackMs = s.blackMs;
      if (s.mode.kind === "timed" && s.runningSide) {
        const inc = s.mode.incrementMs;
        if (s.runningSide === "white") whiteMs += inc;
        else blackMs += inc;
      }
      tickStartRef.current = null;
      return { ...s, whiteMs, blackMs, runningSide: next };
    });
  }, []);

  const pause = useCallback(() => {
    setState((s) => ({ ...s, runningSide: null }));
    tickStartRef.current = null;
  }, []);

  const reset = useCallback((mode?: GameMode) => {
    setState((s) => {
      const m = mode ?? s.mode;
      return {
        mode: m,
        whiteMs: m.kind === "timed" ? m.baseMs : 0,
        blackMs: m.kind === "timed" ? m.baseMs : 0,
        runningSide: null,
        flagged: null,
      };
    });
    tickStartRef.current = null;
  }, []);

  return { state, start, switchTo, pause, reset };
}
