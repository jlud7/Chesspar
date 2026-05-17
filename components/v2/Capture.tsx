"use client";

/**
 * Chesspar v2 capture — rebuilt around a single VLM call per question.
 *
 * Phases: framing → playing → ended. No confirm overlay, no abstention
 * modal, no per-move CV diff dance. Calibration is one Replicate call;
 * each move is one Replicate call. The full request/response history is
 * always visible at the bottom via ApiLogPanel.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Chess } from "chess.js";
import { LivePreview } from "./LivePreview";
import { MoveList } from "./MoveList";
import { CameraSwitcher } from "./CameraSwitcher";
import { ApiLogPanel } from "./ApiLogPanel";
import { BurstCamera } from "@/lib/v2/burst-capture";
import { calibrateBoard } from "@/lib/v2/calibrate";
import { identifyMove } from "@/lib/v2/identify-move";
import { warpBoardWithMargin } from "@/lib/board-image";
import { buildPgn } from "@/lib/v2/pgn";
import {
  DEFAULT_CONFIG,
  type BoardLock,
  type SessionConfig,
} from "@/lib/v2/types";

type Phase = "needCamera" | "framing" | "locking" | "playing" | "ended";

const PROXY_URL = process.env.NEXT_PUBLIC_VLM_PROXY_URL || "";

const STARTING_FEN_FULL =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export function Capture() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraRef = useRef<BurstCamera | null>(null);
  const lockRef = useRef<BoardLock | null>(null);
  const previousRectifiedRef = useRef<HTMLCanvasElement | null>(null);
  const previousRawRef = useRef<HTMLCanvasElement | null>(null);
  const fenRef = useRef<string>(STARTING_FEN_FULL);

  const [phase, setPhase] = useState<Phase>("needCamera");
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [moves, setMoves] = useState<string[]>([]);
  const [currentDeviceId, setCurrentDeviceId] = useState<string | null>(null);
  const [lastSan, setLastSan] = useState<string | null>(null);

  const config: SessionConfig = useMemo(
    () => ({ ...DEFAULT_CONFIG, proxyUrl: PROXY_URL }),
    [],
  );

  useEffect(() => {
    return () => {
      cameraRef.current?.detach();
      cameraRef.current = null;
    };
  }, []);

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
      previousRectifiedRef.current = cal.rectified;
      previousRawRef.current = burst.frame;
      fenRef.current = `${cal.lock.startingFen} w KQkq - 0 1`;
      setMoves([]);
      setLastSan(null);
      setStatusMsg(
        cal.isStartingPosition
          ? "Locked. White moves first."
          : "Locked, but the board doesn't look like the standard start. Continuing anyway.",
      );
      setPhase("playing");
    } catch (e) {
      setStatusMsg(e instanceof Error ? `Lock failed: ${e.message}` : "Lock failed.");
      setPhase("framing");
    } finally {
      setBusy(false);
    }
  }, [config]);

  const captureMove = useCallback(async () => {
    const cam = cameraRef.current;
    const lock = lockRef.current;
    const pre = previousRectifiedRef.current;
    if (!cam || !lock || !pre) return;
    setBusy(true);
    setStatusMsg("Calling Gemini for the move…");
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
      const postRect = warpBoardWithMargin(
        burst.frame,
        lock.corners as Parameters<typeof warpBoardWithMargin>[1],
        config.canonicalSize,
        config.rectifyMargin,
      );
      const res = await identifyMove({
        proxyUrl: config.proxyUrl,
        previousFen: fenRef.current,
        preImage: pre,
        postImage: postRect,
        rawPreImage: previousRawRef.current ?? undefined,
        rawPostImage: burst.frame,
      });
      if (res.kind === "matched") {
        const newFen = applyMove(fenRef.current, res.san);
        if (!newFen) {
          setStatusMsg(`Inferred ${res.san} but couldn't apply it. Re-snap.`);
          setBusy(false);
          return;
        }
        fenRef.current = newFen;
        previousRectifiedRef.current = postRect;
        previousRawRef.current = burst.frame;
        setMoves((m) => [...m, res.san]);
        setLastSan(res.san);
        setStatusMsg(null);
        if (gameIsOver(newFen)) setPhase("ended");
      } else if (res.kind === "abstain") {
        setStatusMsg(`Gemini abstained: ${res.reason}. Re-snap.`);
      } else {
        setStatusMsg(`Move classifier failed: ${res.reason}. Re-snap.`);
      }
    } catch (e) {
      setStatusMsg(e instanceof Error ? `Capture failed: ${e.message}` : "Capture failed.");
    } finally {
      setBusy(false);
    }
  }, [config]);

  const resign = useCallback(() => {
    if (busy) return;
    setStatusMsg(null);
    setPhase("ended");
  }, [busy]);

  const downloadPgn = useCallback(() => {
    const pgn = buildPgn(moves, { result: "*" });
    const blob = new Blob([pgn], { type: "application/x-chess-pgn" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chesspar-${Date.now()}.pgn`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, [moves]);

  const sideToMove = getSideToMove(fenRef.current);

  return (
    <main className="flex h-dvh flex-col bg-zinc-950 text-zinc-100">
      <Header
        phase={phase}
        onResign={resign}
        onDownload={downloadPgn}
        hasMoves={moves.length > 0}
        busy={busy}
      />

      {/* Preview dominates the top — explicit aspect-tuned, taller than before. */}
      <div className="relative flex-1 min-h-0">
        <LivePreview ref={videoRef}>
          <CameraSwitcher currentDeviceId={currentDeviceId} onPick={switchCamera} />
        </LivePreview>
        {phase === "needCamera" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/85 p-6 text-center">
            <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-emerald-300">
              Setup
            </p>
            <h2 className="mt-3 text-2xl font-semibold leading-tight">
              Point your phone at the board.
            </h2>
            <p className="mt-3 max-w-md text-sm text-zinc-400">
              Position the camera so all four corners are visible. Tap to start.
            </p>
          </div>
        )}
        {phase === "locking" && (
          <div className="absolute inset-x-0 top-3 mx-auto w-fit rounded-full bg-emerald-500/15 px-3 py-1.5 text-[12px] text-emerald-200 backdrop-blur">
            Locking…
          </div>
        )}
      </div>

      {/* Bottom stack: status + move list + clocks + api log. Always visible, no fold. */}
      <div className="flex flex-col gap-2 border-t border-white/5 bg-zinc-950/95 px-3 pb-[calc(env(safe-area-inset-bottom)+0.5rem)] pt-2 sm:px-4">
        <StatusStrip
          phase={phase}
          statusMsg={statusMsg}
          lastSan={lastSan}
          sideToMove={sideToMove}
          proxyConfigured={!!PROXY_URL}
        />
        {(phase === "playing" || phase === "ended") && (
          <MoveList moves={moves} />
        )}
        <ActionBar
          phase={phase}
          busy={busy}
          onStart={startCamera}
          onCapture={captureForLock}
          onMove={captureMove}
          sideToMove={sideToMove}
        />
        <ApiLogPanel />
      </div>
    </main>
  );
}

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
    <header className="flex shrink-0 items-center justify-between border-b border-white/5 px-4 py-2.5">
      <Link
        href="/"
        className="text-[11px] font-semibold uppercase tracking-[0.3em] text-zinc-400 transition hover:text-zinc-100"
      >
        ← Chesspar
      </Link>
      <div className="flex items-center gap-2">
        {hasMoves && (
          <button
            onClick={onDownload}
            className="rounded-full bg-white/5 px-3 py-1.5 text-[11px] uppercase tracking-widest text-zinc-200 transition hover:bg-white/10"
          >
            Export PGN
          </button>
        )}
        {phase === "playing" && (
          <button
            onClick={onResign}
            disabled={busy}
            className="rounded-full bg-white/5 px-3 py-1.5 text-[11px] uppercase tracking-widest text-zinc-300 transition hover:bg-white/10 disabled:opacity-40"
          >
            End game
          </button>
        )}
      </div>
    </header>
  );
}

function StatusStrip({
  phase,
  statusMsg,
  lastSan,
  sideToMove,
  proxyConfigured,
}: {
  phase: Phase;
  statusMsg: string | null;
  lastSan: string | null;
  sideToMove: "white" | "black";
  proxyConfigured: boolean;
}) {
  if (statusMsg) {
    const tone = /failed|abstained|couldn|isn't|doesn|error|re-snap/i.test(statusMsg)
      ? "border-amber-300/25 bg-amber-300/10 text-amber-100"
      : /calling|locking|capturing|working/i.test(statusMsg)
        ? "border-white/10 bg-white/5 text-zinc-200"
        : "border-emerald-400/25 bg-emerald-400/10 text-emerald-100";
    return (
      <div className={`rounded-xl border px-3 py-2 text-[12.5px] ${tone}`}>
        {statusMsg}
      </div>
    );
  }
  if (phase === "playing" && lastSan) {
    return (
      <div className="rounded-xl border border-emerald-400/20 bg-emerald-400/5 px-3 py-2 text-[12.5px] text-emerald-100">
        Recorded{" "}
        <span className="font-mono font-semibold">{lastSan}</span>
        {" · "}
        <span className="text-emerald-200/70">
          {sideToMove === "white" ? "White to move next" : "Black to move next"}
        </span>
      </div>
    );
  }
  if (phase === "playing" && !proxyConfigured) {
    return (
      <div className="rounded-xl border border-amber-300/20 bg-amber-300/5 px-3 py-2 text-[12px] text-amber-200">
        Set <code className="font-mono">NEXT_PUBLIC_VLM_PROXY_URL</code> to enable Chesspar.
      </div>
    );
  }
  if (phase === "playing") {
    return (
      <div className="rounded-xl border border-white/5 bg-white/5 px-3 py-2 text-[12px] text-zinc-400">
        After the move is played, tap the{" "}
        <span className="font-medium text-zinc-200">
          {sideToMove === "white" ? "White moved" : "Black moved"}
        </span>{" "}
        clock.
      </div>
    );
  }
  if (phase === "framing") {
    return (
      <div className="rounded-xl border border-white/5 bg-white/5 px-3 py-2 text-[12px] text-zinc-400">
        Set up the starting position, then tap to capture.
      </div>
    );
  }
  return null;
}

function ActionBar({
  phase,
  busy,
  onStart,
  onCapture,
  onMove,
  sideToMove,
}: {
  phase: Phase;
  busy: boolean;
  onStart: () => void;
  onCapture: () => void;
  onMove: () => void;
  sideToMove: "white" | "black";
}) {
  if (phase === "ended") {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4 text-center text-sm text-zinc-300">
        Game ended. Export your PGN from the top bar.
      </div>
    );
  }
  if (phase === "playing") {
    return (
      <div className="grid h-28 grid-cols-2 overflow-hidden rounded-2xl border border-white/10 bg-white/5 shadow-xl">
        <ClockButton
          side="white"
          active={sideToMove === "white"}
          busy={busy && sideToMove === "white"}
          onClick={onMove}
        />
        <ClockButton
          side="black"
          active={sideToMove === "black"}
          busy={busy && sideToMove === "black"}
          onClick={onMove}
        />
      </div>
    );
  }
  if (phase === "locking") {
    return (
      <div className="flex h-14 items-center justify-center rounded-full bg-white/5 text-sm font-medium text-zinc-300">
        Working…
      </div>
    );
  }
  const label = phase === "needCamera" ? "Start camera" : "Capture starting position";
  const handler = phase === "needCamera" ? onStart : phase === "framing" ? onCapture : undefined;
  if (!handler) return null;
  return (
    <button
      onClick={handler}
      disabled={busy}
      className="flex h-14 w-full items-center justify-center gap-2 rounded-full bg-emerald-500 text-base font-semibold text-emerald-950 shadow-xl transition hover:bg-emerald-400 disabled:opacity-50"
    >
      {busy ? "Working…" : label}
    </button>
  );
}

function ClockButton({
  side,
  active,
  busy,
  onClick,
}: {
  side: "white" | "black";
  active: boolean;
  busy: boolean;
  onClick: () => void;
}) {
  const isWhite = side === "white";
  return (
    <button
      onClick={onClick}
      disabled={!active || busy}
      className={[
        "flex flex-col items-center justify-center border-white/10 px-3 text-center transition",
        isWhite ? "border-r" : "",
        active
          ? isWhite
            ? "bg-zinc-100 text-zinc-950"
            : "bg-zinc-800 text-zinc-50"
          : "bg-zinc-900/70 text-zinc-500",
        !busy && active ? "active:scale-[0.99]" : "",
      ].join(" ")}
    >
      <span className="text-[10.5px] font-semibold uppercase tracking-[0.28em]">
        {side}
      </span>
      <span className="mt-1.5 text-lg font-semibold">
        {busy ? "Capturing…" : active ? `${capitalize(side)} moved` : "Waiting"}
      </span>
    </button>
  );
}

function applyMove(previousFen: string, san: string): string | null {
  try {
    const game = new Chess(previousFen);
    const move = game.move(san);
    return move ? game.fen() : null;
  } catch {
    return null;
  }
}

function getSideToMove(fen: string): "white" | "black" {
  return fen.split(/\s+/)[1] === "b" ? "black" : "white";
}

function capitalize(side: "white" | "black"): string {
  return side[0].toUpperCase() + side.slice(1);
}

function gameIsOver(fen: string): boolean {
  try {
    const g = new Chess(fen);
    return g.isGameOver();
  } catch {
    return false;
  }
}
