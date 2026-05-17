"use client";

/**
 * Landscape chess clock — the primary in-game view. The phone is propped
 * upright (portrait) between the two players, so each side's content is
 * rotated 90° to face its player when they look down at the screen.
 *
 * WHITE is on the RIGHT (positive rotation, -90° to face right player).
 * BLACK is on the LEFT (positive rotation, +90° to face left player).
 *
 * The persistent <video> element lives in Capture.tsx and is positioned
 * absolutely at the top-center of THIS view — we don't render the video
 * here; we just leave the space and let Capture position it. The mini
 * camera slot acts as a button to jump to the Camera tab.
 *
 * Visual reference: components/capture-game.tsx SidePanel (lines 2469-2709).
 */

import { useEffect, useRef, useState } from "react";
import type { ClockState, GameMode, Side } from "@/lib/v2/types";
import type { MoveEntry } from "@/lib/v2/types";
import { formatTime } from "@/lib/v2/clock";

export function ClockView({
  clock,
  mode,
  displaySideToMove,
  pendingCount,
  inflight,
  recentMove,
  onTapSide,
  busy,
  canTap,
}: {
  clock: ClockState;
  mode: GameMode;
  displaySideToMove: Side;
  pendingCount: number;
  inflight: boolean;
  /** Most recent committed move — used to flash a "recorded X" chip. */
  recentMove: MoveEntry | null;
  onTapSide: (side: Side) => void;
  /** Show "Capturing…" on the tapped side. */
  busy: boolean;
  /** Whether taps are accepted (e.g. false during calibration / game over). */
  canTap: boolean;
}) {
  return (
    <div className="absolute inset-0 flex">
      {/* LEFT = black */}
      <SidePanel
        side="black"
        rotation={90}
        clock={clock}
        mode={mode}
        active={displaySideToMove === "black"}
        pendingCount={pendingCount}
        inflight={inflight}
        recentMove={recentMove}
        onTap={() => onTapSide("black")}
        busy={busy && displaySideToMove === "black"}
        canTap={canTap}
      />
      {/* RIGHT = white */}
      <SidePanel
        side="white"
        rotation={-90}
        clock={clock}
        mode={mode}
        active={displaySideToMove === "white"}
        pendingCount={pendingCount}
        inflight={inflight}
        recentMove={recentMove}
        onTap={() => onTapSide("white")}
        busy={busy && displaySideToMove === "white"}
        canTap={canTap}
      />
    </div>
  );
}

function SidePanel({
  side,
  rotation,
  clock,
  mode,
  active,
  pendingCount,
  inflight,
  recentMove,
  onTap,
  busy,
  canTap,
}: {
  side: Side;
  rotation: 90 | -90;
  clock: ClockState;
  mode: GameMode;
  active: boolean;
  pendingCount: number;
  inflight: boolean;
  recentMove: MoveEntry | null;
  onTap: () => void;
  busy: boolean;
  canTap: boolean;
}) {
  // Hold the recent SAN visible for ~1.8s after it arrives for the inactive
  // (opposite) side — so the player who just moved sees the result.
  const [showRecent, setShowRecent] = useState(false);
  const lastSeenIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (!recentMove) return;
    const key = `${recentMove.resolvedAt}-${recentMove.san}`;
    if (key === lastSeenIdRef.current) return;
    lastSeenIdRef.current = key;
    setShowRecent(true);
    const t = window.setTimeout(() => setShowRecent(false), 1800);
    return () => window.clearTimeout(t);
  }, [recentMove]);

  const ms = side === "white" ? clock.whiteMs : clock.blackMs;
  const flagged = clock.flagged === side;
  const isTimed = mode.kind === "timed";
  const lowTime = isTimed && ms <= 10_000 && active;
  const showInflightForThisSide = inflight && !active; // model is working on this side's move
  const showsRecent =
    !active && showRecent && recentMove?.side === side;

  const bg = active ? "var(--cp-accent, #5fc99a)" : "transparent";
  const eyebrowTone = active
    ? "rgba(10,36,24,0.7)"
    : "rgba(245,242,235,0.42)";
  const dotColor = active ? "var(--cp-accent-ink, #0a2418)" : "rgba(245,242,235,0.25)";
  const clockColor = active
    ? lowTime
      ? "#3a0a14"
      : "var(--cp-accent-ink, #0a2418)"
    : "rgba(250,247,240,0.55)";
  const turnTextColor = active
    ? "rgba(10,36,24,0.85)"
    : "rgba(245,242,235,0.5)";

  return (
    <button
      type="button"
      onClick={onTap}
      disabled={!canTap || !active || flagged}
      className="relative flex-1 cursor-pointer overflow-hidden text-left transition-[background,transform] duration-300 active:scale-[0.985] disabled:cursor-default"
      style={{ background: bg }}
    >
      <div
        className="absolute left-1/2 top-1/2 flex flex-col items-center justify-center gap-3 px-8"
        style={{
          width: "100vh",
          height: "50vw",
          minWidth: "844px",
          minHeight: "173px",
          maxWidth: "min(100vh, 1100px)",
          maxHeight: "260px",
          transform: `translate(-50%, -50%) rotate(${rotation}deg)`,
          transformOrigin: "center",
        }}
      >
        {/* Eyebrow */}
        <div
          className="flex items-center gap-2 text-[10px] uppercase"
          style={{
            color: eyebrowTone,
            letterSpacing: "0.3em",
            fontFamily: "var(--font-ui, ui-sans-serif)",
          }}
        >
          <span
            className="block h-1.5 w-1.5 rounded-full"
            style={{ background: dotColor }}
          />
          <span>{side === "white" ? "White" : "Black"}</span>
          {pendingCount > 0 && !active && (
            <span
              className="ml-2 inline-flex h-4 items-center rounded-full px-2 text-[9px]"
              style={{
                background: "rgba(95,201,154,0.18)",
                color: "var(--cp-accent, #5fc99a)",
                letterSpacing: "0.2em",
              }}
            >
              {pendingCount} pending
            </span>
          )}
        </div>

        {/* Clock face or YOUR TURN/WAITING */}
        <div
          className="relative tabular-nums"
          style={{
            fontFamily: "var(--font-mono, ui-monospace)",
            fontSize: isTimed
              ? "clamp(64px, 18vh, 120px)"
              : "clamp(38px, 9vh, 64px)",
            lineHeight: 0.95,
            fontWeight: 300,
            letterSpacing: "-0.04em",
            color: clockColor,
            transition: "color 200ms ease",
          }}
        >
          {isTimed
            ? formatTime(ms)
            : active
              ? "YOUR TURN"
              : "WAITING"}
          {showInflightForThisSide && (
            <div className="absolute -bottom-3 left-0 right-0 h-[1.5px] overflow-hidden">
              <div
                className="absolute inset-0"
                style={{ background: "rgba(245,242,235,0.15)" }}
              />
              <div
                className="absolute bottom-0 top-0 w-[38%]"
                style={{
                  background:
                    "linear-gradient(90deg, transparent, rgba(95,201,154,0.85) 50%, transparent)",
                  animation: "chesspar-sweep 1.4s ease-in-out infinite",
                }}
              />
            </div>
          )}
        </div>

        {/* Status line */}
        {flagged && (
          <div
            className="mt-1 text-[14px]"
            style={{
              fontFamily: "var(--font-serif, ui-serif)",
              fontStyle: "italic",
              color: "rgba(250,210,210,0.9)",
            }}
          >
            Flag fell — out of time
          </div>
        )}
        {!flagged && active && busy && (
          <div
            className="mt-1 text-[15px]"
            style={{
              fontFamily: "var(--font-serif, ui-serif)",
              fontStyle: "italic",
              color: turnTextColor,
            }}
          >
            capturing…
          </div>
        )}
        {!flagged && active && !busy && isTimed && (
          <div
            className="mt-1 flex items-center gap-2.5"
            style={{
              fontFamily: "var(--font-serif, ui-serif)",
              fontStyle: "italic",
              fontSize: 16,
              color: turnTextColor,
            }}
          >
            <span className="block h-px w-4" style={{ background: turnTextColor, opacity: 0.5 }} />
            your turn
            <span className="block h-px w-4" style={{ background: turnTextColor, opacity: 0.5 }} />
          </div>
        )}

        {/* Recorded SAN toast (shown on the side that just moved) */}
        {showsRecent && (
          <div
            className="mt-2 inline-flex items-center gap-3 rounded-2xl px-4 py-2"
            style={{
              border: "0.5px solid rgba(95,201,154,0.4)",
              background: "rgba(95,201,154,0.10)",
            }}
          >
            <span
              className="text-[10px] uppercase tracking-[0.3em]"
              style={{ color: "var(--cp-accent, #5fc99a)" }}
            >
              recorded
            </span>
            <span
              className="font-mono text-[20px]"
              style={{ color: "rgba(250,247,240,0.96)" }}
            >
              {recentMove?.san}
            </span>
          </div>
        )}
      </div>
    </button>
  );
}
