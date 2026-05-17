"use client";

/**
 * Chesspar v2 capture — orchestration shell.
 *
 * Phases: needCamera → framing → locking → playing → ended.
 *
 * In the setup phases the screen is dominated by a full live preview and
 * a single primary action button, with the GameModePicker shown once
 * the camera has attached.
 *
 * Once `playing` starts, the screen splits into three tabs (Clock /
 * Score / Camera) and the persistent <video> element is repositioned
 * via inline style depending on the active tab — never remounted, so
 * the BurstCamera stream stays alive across tab switches.
 *
 * Move capture uses a queue (lib/v2/move-queue) so the user can tap
 * rapidly; classification drains in the background while the clock
 * UI advances optimistically.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Chess } from "chess.js";
import { LivePreview } from "./LivePreview";
import { CameraSwitcher } from "./CameraSwitcher";
import { ApiLogPanel } from "./ApiLogPanel";
import { TabBar, type Tab } from "./TabBar";
import { GameModePicker, loadStoredGameMode, saveGameMode } from "./GameModePicker";
import { ClockView } from "./ClockView";
import { ScoreView } from "./ScoreView";
import { CameraView } from "./CameraView";
import { BurstCamera } from "@/lib/v2/burst-capture";
import { calibrateBoard } from "@/lib/v2/calibrate";
import { warpBoardWithMargin } from "@/lib/board-image";
import { buildPgn, type AnnotatedMove } from "@/lib/v2/pgn";
import { useChessClock } from "@/lib/v2/clock";
import { useMoveQueue } from "@/lib/v2/move-queue";
import {
  DEFAULT_CONFIG,
  type BoardLock,
  type GameMode,
  type MoveEntry,
  type SessionConfig,
  type Side,
} from "@/lib/v2/types";

type Phase = "needCamera" | "framing" | "locking" | "playing" | "ended";

const PROXY_URL = process.env.NEXT_PUBLIC_VLM_PROXY_URL || "";
const STARTING_FEN_FULL =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export function Capture() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraRef = useRef<BurstCamera | null>(null);
  const lockRef = useRef<BoardLock | null>(null);

  const [phase, setPhase] = useState<Phase>("needCamera");
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [currentDeviceId, setCurrentDeviceId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("clock");
  const [gameMode, setGameMode] = useState<GameMode>(() => ({ kind: "untimed" }));
  const [showPicker, setShowPicker] = useState(false);
  const [pickerInitialMode, setPickerInitialMode] = useState<GameMode>({ kind: "untimed" });

  const config: SessionConfig = useMemo(
    () => ({ ...DEFAULT_CONFIG, proxyUrl: PROXY_URL }),
    [],
  );

  const clock = useChessClock(gameMode);
  const queue = useMoveQueue({
    lock: lockRef.current,
    proxyUrl: config.proxyUrl,
    onMoveCommitted: (entry) => {
      // Side that just moved gets the increment; opponent's clock starts.
      const next: Side = entry.side === "white" ? "black" : "white";
      clock.switchTo(next);
      // Did this move end the game?
      try {
        const g = new Chess(queue.state.committedFen);
        if (g.isGameOver()) setPhase("ended");
      } catch {
        /* ignore */
      }
    },
  });

  // Reset the clock when the game mode changes (or game restarts).
  useEffect(() => {
    clock.reset(gameMode);
  }, [gameMode]); // eslint-disable-line react-hooks/exhaustive-deps

  // Flag → end game.
  useEffect(() => {
    if (clock.state.flagged) setPhase("ended");
  }, [clock.state.flagged]);

  // Pause the clock while there's an unresolved classification failure
  // — no fair ticking time off either player while the user resolves a
  // stuck queue item.
  const failureRunningBeforeRef = useRef<Side | null>(null);
  useEffect(() => {
    if (queue.state.failedAt) {
      if (clock.state.runningSide) {
        failureRunningBeforeRef.current = clock.state.runningSide;
        clock.pause();
      }
    } else if (failureRunningBeforeRef.current) {
      const resume = failureRunningBeforeRef.current;
      failureRunningBeforeRef.current = null;
      clock.switchTo(resume);
    }
  }, [queue.state.failedAt]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load saved game mode on mount.
  useEffect(() => {
    const stored = loadStoredGameMode();
    setGameMode(stored);
    setPickerInitialMode(stored);
  }, []);

  // Tear down the camera on unmount.
  useEffect(() => {
    return () => {
      cameraRef.current?.detach();
      cameraRef.current = null;
    };
  }, []);

  // --------------- lifecycle handlers ---------------

  const startCamera = useCallback(async () => {
    if (!videoRef.current) return;
    setBusy(true);
    setStatusMsg(null);
    try {
      const cam = new BurstCamera();
      await cam.attach(videoRef.current);
      cameraRef.current = cam;
      setCurrentDeviceId(cam.deviceId);
      setPhase("framing");
      setShowPicker(true);
    } catch (e) {
      setStatusMsg(
        e instanceof Error ? `Camera access failed: ${e.message}` : "Camera access failed.",
      );
    } finally {
      setBusy(false);
    }
  }, []);

  const switchCamera = useCallback(async (deviceId: string) => {
    const cam = cameraRef.current;
    if (!cam) return;
    setBusy(true);
    try {
      await cam.switchTo(deviceId);
      setCurrentDeviceId(cam.deviceId);
    } catch (e) {
      setStatusMsg(
        e instanceof Error ? `Camera switch failed: ${e.message}` : "Switch failed",
      );
    } finally {
      setBusy(false);
    }
  }, []);

  const confirmGameMode = useCallback((mode: GameMode) => {
    setGameMode(mode);
    saveGameMode(mode);
    setShowPicker(false);
  }, []);

  const captureForLock = useCallback(async () => {
    const cam = cameraRef.current;
    if (!cam) return;
    setBusy(true);
    setStatusMsg("Calling Gemini for board corners…");
    setPhase("locking");
    try {
      const burst = await cam.capture({
        count: config.burstSize,
        intervalMs: config.burstIntervalMs,
        maxDim: 1280,
      });
      if (!burst) {
        setStatusMsg("Couldn't get a sharp frame — hold steady and try again.");
        setPhase("framing");
        setBusy(false);
        return;
      }
      const cal = await calibrateBoard({
        proxyUrl: config.proxyUrl,
        image: burst.frame,
        size: config.canonicalSize,
      });
      if (cal.kind !== "locked") {
        setStatusMsg(cal.reason);
        setPhase("framing");
        setBusy(false);
        return;
      }
      lockRef.current = cal.lock;
      queue.initialize({
        rectified: cal.rectified,
        raw: burst.frame,
        fen: `${cal.lock.startingFen} w KQkq - 0 1`,
      });
      clock.reset(gameMode);
      clock.start();
      setStatusMsg(null);
      setPhase("playing");
      setActiveTab("clock");
    } catch (e) {
      setStatusMsg(e instanceof Error ? `Lock failed: ${e.message}` : "Lock failed.");
      setPhase("framing");
    } finally {
      setBusy(false);
    }
  }, [config, queue, clock, gameMode]);

  const relock = useCallback(async () => {
    const cam = cameraRef.current;
    if (!cam) return;
    setBusy(true);
    setStatusMsg("Re-calibrating…");
    try {
      const burst = await cam.capture({
        count: config.burstSize,
        intervalMs: config.burstIntervalMs,
        maxDim: 1280,
      });
      if (!burst) {
        setStatusMsg("Couldn't get a sharp frame — hold steady and try again.");
        setBusy(false);
        return;
      }
      const cal = await calibrateBoard({
        proxyUrl: config.proxyUrl,
        image: burst.frame,
        size: config.canonicalSize,
      });
      if (cal.kind !== "locked") {
        setStatusMsg(cal.reason);
        setBusy(false);
        return;
      }
      lockRef.current = cal.lock;
      // Re-lock keeps the game history intact; we only swap in the new
      // rectified pre-image so the next classification has a clean
      // baseline against the new geometry.
      queue.updateAfterRelock({
        rectified: cal.rectified,
        raw: burst.frame,
      });
      setStatusMsg("Re-locked.");
      setTimeout(() => setStatusMsg(null), 1500);
    } catch (e) {
      setStatusMsg(e instanceof Error ? `Re-lock failed: ${e.message}` : "Re-lock failed.");
    } finally {
      setBusy(false);
    }
  }, [config, queue]);

  // The clock-side tap path: snap, rectify, enqueue, swap clock immediately.
  const onTapSide = useCallback(
    async (side: Side) => {
      const cam = cameraRef.current;
      const lock = lockRef.current;
      if (!cam || !lock) return;
      if (side !== queue.displaySideToMove) return;
      if (queue.state.failedAt) return;
      setBusy(true);
      try {
        const burst = await cam.capture({
          count: config.burstSize,
          intervalMs: config.burstIntervalMs,
          maxDim: 1280,
        });
        if (!burst) {
          setStatusMsg("Missed the snap — hold steady and tap again.");
          return;
        }
        const rectified = warpBoardWithMargin(
          burst.frame,
          lock.corners as Parameters<typeof warpBoardWithMargin>[1],
          config.canonicalSize,
          config.rectifyMargin,
        );
        queue.enqueue({
          capturedAt: burst.capturedAt,
          rawFrame: burst.frame,
          rectified,
          byClockSide: side,
        });
        // Optimistically swap the clock immediately so the opponent
        // starts ticking right away.
        clock.switchTo(side === "white" ? "black" : "white");
      } catch (e) {
        setStatusMsg(e instanceof Error ? `Capture failed: ${e.message}` : "Capture failed.");
      } finally {
        setBusy(false);
      }
    },
    [config, queue, clock],
  );

  const endGame = useCallback(() => {
    if (busy) return;
    clock.pause();
    setPhase("ended");
  }, [busy, clock]);

  const downloadPgn = useCallback(() => {
    const moves: AnnotatedMove[] = queue.state.history.map((e, i) => {
      const remainingMs = clockReadingAfterMove(queue.state.history, i, gameMode);
      return { san: e.san, ...(remainingMs != null ? { remainingMs } : {}) };
    });
    const pgn = buildPgn(moves, {
      result: clock.state.flagged
        ? clock.state.flagged === "white"
          ? "0-1"
          : "1-0"
        : "*",
      gameMode,
    });
    const blob = new Blob([pgn], { type: "application/x-chess-pgn" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chesspar-${Date.now()}.pgn`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, [queue.state.history, gameMode, clock.state.flagged]);

  const recentMove =
    queue.state.history.length > 0
      ? queue.state.history[queue.state.history.length - 1]
      : null;

  // --------------- video positioning ---------------
  // Persistent <video>: rendered once, positioned absolutely. Style
  // changes per active tab during the in-game phases. During setup, it
  // fills the screen behind the setup chrome.
  const videoStyle = useMemo<React.CSSProperties>(() => {
    if (phase === "needCamera") {
      return { display: "none" };
    }
    if (phase === "framing" || phase === "locking") {
      return {
        position: "absolute",
        inset: 0,
        zIndex: 5,
      };
    }
    // playing | ended
    if (activeTab === "camera") {
      return {
        position: "absolute",
        inset: 0,
        zIndex: 5,
      };
    }
    if (activeTab === "clock") {
      return {
        position: "absolute",
        top: 10,
        left: "50%",
        transform: "translateX(-50%)",
        width: 116,
        height: 152,
        borderRadius: 14,
        border: "1px solid rgba(255,255,255,0.12)",
        boxShadow: "0 4px 24px rgba(0,0,0,0.6)",
        zIndex: 25,
        cursor: "pointer",
      };
    }
    // score
    return {
      position: "absolute",
      top: 10,
      right: 12,
      width: 64,
      height: 80,
      borderRadius: 10,
      border: "1px solid rgba(255,255,255,0.1)",
      zIndex: 25,
      cursor: "pointer",
    };
  }, [phase, activeTab]);

  const onVideoClick = useCallback(() => {
    if (phase === "playing" || phase === "ended") {
      if (activeTab !== "camera") setActiveTab("camera");
    }
  }, [phase, activeTab]);

  // --------------- render ---------------

  return (
    <main className="relative h-dvh overflow-hidden bg-zinc-950 text-zinc-100">
      <Header
        phase={phase}
        onResign={endGame}
        onDownload={downloadPgn}
        hasMoves={queue.state.history.length > 0}
        busy={busy}
      />

      {/* Persistent video element */}
      <div style={videoStyle} onClick={onVideoClick}>
        <LivePreview ref={videoRef} style={{ width: "100%", height: "100%" }} />
      </div>

      {/* Setup phases */}
      {phase === "needCamera" && (
        <SetupSplash onStart={startCamera} busy={busy} />
      )}
      {(phase === "framing" || phase === "locking") && (
        <SetupOverlay
          phase={phase}
          statusMsg={statusMsg}
          busy={busy}
          gameMode={gameMode}
          onChangeMode={() => {
            setPickerInitialMode(gameMode);
            setShowPicker(true);
          }}
          onCapture={captureForLock}
          currentDeviceId={currentDeviceId}
          onPickCamera={switchCamera}
        />
      )}

      {/* In-game tabs */}
      {(phase === "playing" || phase === "ended") && (
        <>
          <div
            style={{ display: activeTab === "clock" ? "block" : "none" }}
            className="absolute inset-0"
          >
            <ClockView
              clock={clock.state}
              mode={gameMode}
              displaySideToMove={queue.displaySideToMove}
              pendingCount={queue.pendingCount}
              inflight={!!queue.state.inflight}
              recentMove={recentMove}
              onTapSide={onTapSide}
              busy={busy}
              canTap={phase === "playing" && !clock.state.flagged}
            />
          </div>

          <div
            style={{ display: activeTab === "score" ? "block" : "none" }}
            className="absolute inset-0"
          >
            <ScoreView
              history={queue.state.history}
              pending={queue.state.queue}
              inflight={queue.state.inflight}
              failure={
                queue.state.failedAt
                  ? {
                      reason: queue.state.failedAt.reason,
                      previousFen: queue.state.committedFen,
                    }
                  : null
              }
              mode={gameMode}
              onApplyFailureGuess={queue.resolveFailure}
              onDropFailure={queue.dropFailureAndAfter}
              onRetryFailure={() => queue.resolveFailure(undefined)}
            />
          </div>

          <div
            style={{ display: activeTab === "camera" ? "block" : "none" }}
            className="absolute inset-0"
          >
            <CameraView
              currentDeviceId={currentDeviceId}
              onPickCamera={switchCamera}
              onRelock={relock}
              relockBusy={busy}
            />
          </div>

          <TabBar
            active={activeTab}
            onPick={setActiveTab}
            pendingCount={queue.pendingCount}
            failurePending={!!queue.state.failedAt}
          />

          {queue.state.failedAt && activeTab !== "score" && (
            <button
              onClick={() => setActiveTab("score")}
              className="pointer-events-auto absolute inset-x-3 bottom-[88px] z-40 flex items-center justify-between gap-3 rounded-2xl border border-amber-300/40 bg-amber-300/15 px-3 py-2 text-left text-[12px] text-amber-100 shadow-2xl backdrop-blur"
            >
              <span>
                <span className="font-semibold uppercase tracking-[0.18em]">
                  Queue paused
                </span>
                <span className="ml-2 text-amber-100/85">Tap to resolve</span>
              </span>
              <span className="text-amber-200">→</span>
            </button>
          )}

          {statusMsg && (
            <div className="pointer-events-none absolute inset-x-0 top-[44px] z-40 flex justify-center px-4">
              <div className="pointer-events-auto rounded-full bg-zinc-900/90 px-3 py-1.5 text-[11px] text-zinc-200 shadow-lg">
                {statusMsg}
              </div>
            </div>
          )}
        </>
      )}

      {showPicker && (
        <GameModePicker initial={pickerInitialMode} onConfirm={confirmGameMode} />
      )}
    </main>
  );
}

// ============================================================
// Header + setup chrome
// ============================================================

function Header({
  phase,
  onResign,
  onDownload,
  hasMoves,
  busy,
}: {
  phase: Phase;
  onResign: () => void;
  onDownload: () => void;
  hasMoves: boolean;
  busy: boolean;
}) {
  return (
    <header className="absolute inset-x-0 top-0 z-40 flex h-11 items-center justify-between border-b border-white/5 bg-zinc-950/60 px-4 backdrop-blur">
      <Link
        href="/"
        className="text-[10px] font-semibold uppercase tracking-[0.3em] text-zinc-400 transition hover:text-zinc-100"
      >
        ← Chesspar
      </Link>
      <div className="flex items-center gap-2">
        {hasMoves && (
          <button
            onClick={onDownload}
            className="rounded-full bg-white/5 px-3 py-1 text-[10px] uppercase tracking-widest text-zinc-200 transition hover:bg-white/10"
          >
            Export PGN
          </button>
        )}
        {phase === "playing" && (
          <button
            onClick={onResign}
            disabled={busy}
            className="rounded-full bg-white/5 px-3 py-1 text-[10px] uppercase tracking-widest text-zinc-300 transition hover:bg-white/10 disabled:opacity-40"
          >
            End game
          </button>
        )}
      </div>
    </header>
  );
}

function SetupSplash({ onStart, busy }: { onStart: () => void; busy: boolean }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950 px-6 text-center">
      <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-emerald-300">
        Setup
      </p>
      <h2 className="mt-3 text-2xl font-semibold leading-tight">
        Point your phone at the board.
      </h2>
      <p className="mt-3 max-w-md text-sm text-zinc-400">
        Position the camera so all four corners are visible. Tap to start.
      </p>
      <button
        onClick={onStart}
        disabled={busy}
        className="mt-8 flex h-14 w-full max-w-md items-center justify-center rounded-full bg-emerald-500 text-base font-semibold text-emerald-950 shadow-xl transition hover:bg-emerald-400 disabled:opacity-50"
      >
        {busy ? "Working…" : "Start camera"}
      </button>
    </div>
  );
}

function SetupOverlay({
  phase,
  statusMsg,
  busy,
  gameMode,
  onChangeMode,
  onCapture,
  currentDeviceId,
  onPickCamera,
}: {
  phase: Phase;
  statusMsg: string | null;
  busy: boolean;
  gameMode: GameMode;
  onChangeMode: () => void;
  onCapture: () => void;
  currentDeviceId: string | null;
  onPickCamera: (deviceId: string) => void;
}) {
  return (
    <>
      <div className="pointer-events-none absolute inset-x-0 top-11 z-20 flex justify-end p-3">
        <div className="pointer-events-auto">
          <CameraSwitcher currentDeviceId={currentDeviceId} onPick={onPickCamera} />
        </div>
      </div>

      <div className="absolute inset-x-0 bottom-0 z-20 flex flex-col gap-3 bg-gradient-to-t from-zinc-950 via-zinc-950/85 to-transparent px-4 pb-[calc(env(safe-area-inset-bottom)+0.75rem)] pt-6 sm:px-6">
        <button
          onClick={onChangeMode}
          disabled={busy}
          className="self-center rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-[11px] uppercase tracking-widest text-zinc-300 transition hover:bg-white/10 disabled:opacity-50"
        >
          Mode: {gameMode.kind === "untimed" ? "untimed" : `${Math.round(gameMode.baseMs / 60_000)} + ${Math.round(gameMode.incrementMs / 1000)}`}
        </button>
        {statusMsg && (
          <div className="self-center rounded-full bg-zinc-900/90 px-3 py-1.5 text-[12px] text-zinc-200 shadow-lg">
            {statusMsg}
          </div>
        )}
        <button
          onClick={onCapture}
          disabled={busy || phase === "locking"}
          className="flex h-14 w-full max-w-md items-center justify-center self-center rounded-full bg-emerald-500 text-base font-semibold text-emerald-950 shadow-xl transition hover:bg-emerald-400 disabled:opacity-50"
        >
          {phase === "locking" ? "Locking…" : busy ? "Working…" : "Capture starting position"}
        </button>
      </div>
    </>
  );
}

// ============================================================
// Helpers
// ============================================================

/**
 * Estimate the player's remaining clock time AFTER the i-th move. We
 * don't store per-move remaining times directly, so we reconstruct from
 * `thinkDurationMs` and the time control. Untimed → null. Best-effort.
 */
function clockReadingAfterMove(
  history: MoveEntry[],
  index: number,
  mode: GameMode,
): number | undefined {
  if (mode.kind !== "timed") return undefined;
  let white = mode.baseMs;
  let black = mode.baseMs;
  for (let i = 0; i <= index; i++) {
    const e = history[i];
    if (e.side === "white") {
      white = Math.max(0, white - e.thinkDurationMs) + mode.incrementMs;
    } else {
      black = Math.max(0, black - e.thinkDurationMs) + mode.incrementMs;
    }
  }
  return history[index].side === "white" ? white : black;
}
