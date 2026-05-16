"use client";

/**
 * Chesspar v2 capture — diff-first hybrid + bulletproof calibration.
 *
 * Key UX invariants enforced here (each fixed a real-world failure):
 *   1. The <video> element NEVER unmounts during a session. We overlay
 *      confirmation and abstention chrome on top of it instead of
 *      replacing the preview. Otherwise the camera stream attaches to
 *      a stale node and the next phase shows a black box.
 *   2. The live preview uses object-contain so the user sees their
 *      WHOLE camera frame — no cropped-right-side surprises.
 *   3. On iPhones with multiple back cameras we default to ultra-wide
 *      (0.5x). The camera chip in the top-right lets the user switch.
 *   4. Status text is never duplicated. The confirm overlay has its
 *      own message strip; the body StatusBar is suppressed in that
 *      phase.
 *   5. Locks below the starting-position threshold are REJECTED with a
 *      specific reason — we never lock on bad corners and then quietly
 *      emit wrong moves the rest of the game.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Chess } from "chess.js";
import { LivePreview } from "./LivePreview";
import { MoveList } from "./MoveList";
import { AbstentionPrompt } from "./AbstentionPrompt";
import { CameraSwitcher } from "./CameraSwitcher";
import { BurstCamera } from "@/lib/v2/burst-capture";
import { lockBoardFromImage } from "@/lib/v2/board-lock";
import { runMovePipeline, applyMove } from "@/lib/v2/move-pipeline";
import { buildPgn } from "@/lib/v2/pgn";
import {
  DEFAULT_CONFIG,
  type BoardLock,
  type MoveCandidate,
  type SessionConfig,
} from "@/lib/v2/types";

type Phase =
  | "needCamera"
  | "framing"
  | "locking"
  | "confirm"
  | "playing"
  | "abstain"
  | "ended";

type AbstentionState = {
  candidates: MoveCandidate[];
  rectifiedDataUrl: string;
  postRectified: HTMLCanvasElement;
};

const PROXY_URL = process.env.NEXT_PUBLIC_VLM_PROXY_URL || "";

export function Capture() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraRef = useRef<BurstCamera | null>(null);
  const lockRef = useRef<BoardLock | null>(null);
  const prevRectifiedRef = useRef<HTMLCanvasElement | null>(null);
  const fenRef = useRef<string>(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  );

  const [phase, setPhase] = useState<Phase>("needCamera");
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmPreview, setConfirmPreview] = useState<string | null>(null);
  const [confirmDetail, setConfirmDetail] = useState<string | null>(null);
  const [lockedFen, setLockedFen] = useState<string>(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  );
  const [moves, setMoves] = useState<string[]>([]);
  const [abstention, setAbstention] = useState<AbstentionState | null>(null);
  const [currentDeviceId, setCurrentDeviceId] = useState<string | null>(null);
  const [lastDecision, setLastDecision] = useState<{
    san?: string;
    pConf?: number;
    latencyMs?: number;
  } | null>(null);

  const config: SessionConfig = useMemo(
    () => ({
      ...DEFAULT_CONFIG,
      proxyUrl: PROXY_URL,
    }),
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
        e instanceof Error
          ? `Camera access failed: ${e.message}`
          : "Camera access failed.",
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

  // ----- calibration -----
  const captureForLock = useCallback(async () => {
    const cam = cameraRef.current;
    if (!cam) return;
    setBusy(true);
    setStatusMsg("Capturing starting position…");
    setPhase("locking");
    try {
      const burst = await cam.capture({
        count: config.burstSize,
        intervalMs: config.burstIntervalMs,
        maxDim: 480,
      });
      if (!burst) {
        setStatusMsg("Couldn't get a sharp frame — hold steady and try again.");
        setPhase("framing");
        setBusy(false);
        return;
      }
      setStatusMsg("Locking board…");
      const result = await lockBoardFromImage(burst.frame, {
        proxyUrl: config.proxyUrl,
        size: config.canonicalSize,
      });
      if (result.kind !== "locked") {
        setStatusMsg(result.reason);
        setPhase("framing");
        setBusy(false);
        return;
      }
      lockRef.current = result.lock;
      prevRectifiedRef.current = result.rectified;
      setConfirmPreview(result.rectified.toDataURL("image/jpeg", 0.9));
      const startingFen = `${result.lock.startingFen} w KQkq - 0 1`;
      fenRef.current = startingFen;
      setLockedFen(startingFen);
      const detectorLabel =
        result.detector === "gemini"
          ? "Gemini corners"
          : result.detector === "florence-cv"
            ? "Florence + CV"
            : "CV";
      setConfirmDetail(
        `${detectorLabel} · ${Math.round(result.startingCheck.score * 100)}% match${result.magicRotationEscalation ? " · VLM-oriented" : ""}`,
      );
      setStatusMsg(null);
      setPhase("confirm");
    } catch (e) {
      setStatusMsg(
        e instanceof Error
          ? `Lock failed: ${e.message}`
          : "Lock failed unexpectedly.",
      );
      setPhase("framing");
    } finally {
      setBusy(false);
    }
  }, [config]);

  const retake = useCallback(() => {
    setConfirmPreview(null);
    setConfirmDetail(null);
    lockRef.current = null;
    prevRectifiedRef.current = null;
    setStatusMsg(null);
    setPhase("framing");
  }, []);

  const confirmLock = useCallback(() => {
    setStatusMsg(null);
    setConfirmDetail(null);
    setMoves([]);
    setLastDecision(null);
    setPhase("playing");
  }, []);

  // ----- gameplay -----
  const captureMove = useCallback(async () => {
    const cam = cameraRef.current;
    const lock = lockRef.current;
    if (!cam || !lock) return;
    setBusy(true);
    setStatusMsg("Capturing move…");
    try {
      const burst = await cam.capture({
        count: config.burstSize,
        intervalMs: config.burstIntervalMs,
        maxDim: 480,
      });
      if (!burst) {
        setStatusMsg("Couldn't get a sharp frame — hold steady and try again.");
        setBusy(false);
        return;
      }
      const result = await runMovePipeline({
        burst,
        lock,
        previousFen: fenRef.current,
        config,
      });
      // Adopt the (possibly-refreshed) lock for the next capture so
      // future moves benefit from fresh corner detection.
      lockRef.current = result.lock;
      if (result.decision.kind === "matched") {
        const san = result.decision.pick.san;
        const newFen = applyMove(fenRef.current, san);
        if (!newFen) {
          setStatusMsg(`Inferred ${san} but couldn't apply it. Try again.`);
          setBusy(false);
          return;
        }
        fenRef.current = newFen;
        prevRectifiedRef.current = result.rectified;
        setMoves((m) => [...m, san]);
        setLastDecision({
          san,
          pConf: result.decision.pConfident,
          latencyMs: result.decision.latencyMs,
        });
        setStatusMsg(null);
        if (gameIsOver(newFen)) setPhase("ended");
      } else if (result.decision.kind === "abstain") {
        setAbstention({
          candidates: result.decision.candidates,
          rectifiedDataUrl: result.rectified.toDataURL("image/jpeg", 0.9),
          postRectified: result.rectified,
        });
        setLastDecision({
          pConf: result.decision.pConfident,
          latencyMs: result.decision.latencyMs,
        });
        setPhase("abstain");
      } else {
        // Even on error, adopt the new rectified frame as the next
        // baseline if corners were refreshed — the user just took a
        // photo so their idea of "current board state" is *this* frame,
        // not the stale calibration one.
        if (result.trace.cornerRefresh !== "kept") {
          prevRectifiedRef.current = result.rectified;
        }
        setStatusMsg(result.decision.reason);
      }
    } catch (e) {
      setStatusMsg(
        e instanceof Error ? `Capture failed: ${e.message}` : "Capture failed.",
      );
    } finally {
      setBusy(false);
    }
  }, [config]);

  const handleAbstentionPick = useCallback(
    (san: string) => {
      const newFen = applyMove(fenRef.current, san);
      if (!newFen) {
        setStatusMsg(`Couldn't apply ${san} — try capturing again.`);
        setAbstention(null);
        setPhase("playing");
        return;
      }
      fenRef.current = newFen;
      if (abstention?.postRectified) {
        prevRectifiedRef.current = abstention.postRectified;
      }
      setMoves((m) => [...m, san]);
      setAbstention(null);
      setStatusMsg(null);
      setPhase(gameIsOver(newFen) ? "ended" : "playing");
    },
    [abstention],
  );

  const cancelAbstention = useCallback(() => {
    setAbstention(null);
    setPhase("playing");
  }, []);

  const resign = useCallback(() => setPhase("ended"), []);

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

  const showConfirmOverlay = phase === "confirm" && !!confirmPreview;

  return (
    <main className="relative flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <Header
        phase={phase}
        onResign={resign}
        onDownload={downloadPgn}
        hasMoves={moves.length > 0}
      />

      <div className="relative flex flex-1 flex-col">
        {/*
          The video element is rendered exactly once and stays mounted
          for the whole session. Overlays slot on top of it; we never
          replace it. This was the root cause of the "black preview"
          regression after the confirm step.
        */}
        <div className="relative h-[55vh] w-full sm:h-[60vh]">
          <LivePreview ref={videoRef}>
            <CameraSwitcher
              currentDeviceId={currentDeviceId}
              onPick={switchCamera}
            />
            <FramingOverlay phase={phase} />
            {showConfirmOverlay && (
              <ConfirmOverlay
                previewUrl={confirmPreview!}
                detail={confirmDetail}
                onConfirm={confirmLock}
                onRetake={retake}
                busy={busy}
              />
            )}
          </LivePreview>
        </div>

        <div className="flex-1 px-4 pb-32 pt-4 sm:px-8">
          {!showConfirmOverlay && (
            <StatusBar
              phase={phase}
              statusMsg={statusMsg}
              lastDecision={lastDecision}
              busy={busy}
              proxyConfigured={!!PROXY_URL}
            />
          )}
          {(phase === "playing" || phase === "abstain" || phase === "ended") && (
            <div className="mt-4">
              <MoveList moves={moves} abstainingOn={phase === "abstain"} />
              <FenBox fen={lockedFen} currentFen={fenRef.current} />
            </div>
          )}
        </div>

        <ActionBar
          phase={phase}
          busy={busy}
          onStart={startCamera}
          onCapture={captureForLock}
          onMove={captureMove}
        />
      </div>

      {abstention && (
        <AbstentionPrompt
          candidates={abstention.candidates}
          rectifiedDataUrl={abstention.rectifiedDataUrl}
          onPick={handleAbstentionPick}
          onCancel={cancelAbstention}
        />
      )}
    </main>
  );
}

function Header({
  phase,
  onResign,
  onDownload,
  hasMoves,
}: {
  phase: Phase;
  onResign: () => void;
  onDownload: () => void;
  hasMoves: boolean;
}) {
  return (
    <header className="flex items-center justify-between border-b border-white/5 px-4 py-3 sm:px-6">
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
            className="rounded-full bg-white/5 px-3 py-1.5 text-[11px] uppercase tracking-widest text-zinc-300 transition hover:bg-white/10"
          >
            End game
          </button>
        )}
      </div>
    </header>
  );
}

function FramingOverlay({ phase }: { phase: Phase }) {
  if (phase === "needCamera") {
    return (
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-zinc-950/70 p-6 text-center">
        <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-emerald-300">
          Setup
        </p>
        <h2 className="mt-3 text-2xl font-semibold leading-tight">
          Point your phone at the board.
        </h2>
        <p className="mt-3 max-w-md text-sm text-zinc-400">
          Set the phone above the board so all four corners are visible.
          Tap start to enable the camera.
        </p>
      </div>
    );
  }
  if (phase === "framing") {
    return (
      <>
        <div className="pointer-events-none absolute inset-6 z-10 rounded-lg border-2 border-dashed border-emerald-300/50" />
        <div className="pointer-events-none absolute left-1/2 top-3 z-10 -translate-x-1/2 rounded-full bg-zinc-950/60 px-3 py-1 text-[11px] font-medium uppercase tracking-widest text-emerald-300">
          Frame the board
        </div>
      </>
    );
  }
  if (phase === "locking") {
    return (
      <div className="absolute inset-0 z-10 flex items-center justify-center bg-zinc-950/60">
        <div className="rounded-full bg-zinc-900/80 px-4 py-2 text-sm font-medium text-emerald-300">
          Locking board…
        </div>
      </div>
    );
  }
  return null;
}

function ConfirmOverlay({
  previewUrl,
  detail,
  onConfirm,
  onRetake,
  busy,
}: {
  previewUrl: string;
  detail: string | null;
  onConfirm: () => void;
  onRetake: () => void;
  busy: boolean;
}) {
  return (
    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-zinc-950/85 p-4 backdrop-blur-sm">
      <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-emerald-300">
        Confirm starting position
      </p>
      <div className="overflow-hidden rounded-xl border border-emerald-400/30 shadow-2xl">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={previewUrl}
          alt="Rectified starting position"
          className="block max-h-[42vh] w-auto"
        />
      </div>
      {detail && (
        <p className="text-[11px] uppercase tracking-widest text-zinc-500">
          {detail}
        </p>
      )}
      <div className="flex gap-2">
        <button
          onClick={onRetake}
          disabled={busy}
          className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-zinc-200 transition hover:bg-white/10 disabled:opacity-50"
        >
          Retake
        </button>
        <button
          onClick={onConfirm}
          disabled={busy}
          className="rounded-full bg-emerald-500 px-5 py-2 text-sm font-semibold text-emerald-950 transition hover:bg-emerald-400 disabled:opacity-50"
        >
          Looks good — start game
        </button>
      </div>
    </div>
  );
}

function StatusBar({
  phase,
  statusMsg,
  lastDecision,
  busy,
  proxyConfigured,
}: {
  phase: Phase;
  statusMsg: string | null;
  lastDecision: {
    san?: string;
    pConf?: number;
    latencyMs?: number;
  } | null;
  busy: boolean;
  proxyConfigured: boolean;
}) {
  if (statusMsg) {
    return (
      <div className="rounded-xl border border-amber-300/20 bg-amber-300/5 px-4 py-3 text-[13px] text-amber-200">
        {statusMsg}
      </div>
    );
  }
  if (phase === "playing" && lastDecision?.san) {
    return (
      <div className="flex flex-wrap items-center gap-2 rounded-xl border border-emerald-400/20 bg-emerald-400/5 px-4 py-3 text-[13px] text-emerald-100">
        <span className="font-mono font-semibold">{lastDecision.san}</span>
        <span className="text-emerald-300/60">·</span>
        <span className="text-emerald-200/80">
          {Math.round((lastDecision.pConf ?? 0) * 100)}% confidence
        </span>
        {typeof lastDecision.latencyMs === "number" && (
          <span className="ml-auto text-[11px] text-emerald-300/60">
            {Math.round(lastDecision.latencyMs)} ms
          </span>
        )}
      </div>
    );
  }
  if (phase === "playing" && !proxyConfigured) {
    return (
      <div className="rounded-xl border border-amber-300/20 bg-amber-300/5 px-4 py-3 text-[12px] text-amber-200">
        Set <code className="font-mono">NEXT_PUBLIC_VLM_PROXY_URL</code> to enable the Gemini Flash move classifier. Without it Chesspar can&apos;t infer moves.
      </div>
    );
  }
  if (phase === "playing") {
    return (
      <div className="rounded-xl border border-white/5 bg-white/5 px-4 py-3 text-[12px] text-zinc-400">
        Make your move, then tap{" "}
        <span className="font-medium text-zinc-200">Capture move</span> below.
      </div>
    );
  }
  if (busy) {
    return (
      <div className="rounded-xl border border-white/5 bg-white/5 px-4 py-3 text-[12px] text-zinc-400">
        Working…
      </div>
    );
  }
  return null;
}

function FenBox({ fen, currentFen }: { fen: string; currentFen: string }) {
  return (
    <details className="mt-3 rounded-xl border border-white/5 bg-white/5 px-3 py-2 text-[11px] text-zinc-400">
      <summary className="cursor-pointer select-none uppercase tracking-widest">
        FEN
      </summary>
      <div className="mt-2 font-mono">
        <div>
          <span className="text-zinc-500">Start:</span> {fen}
        </div>
        <div className="mt-1">
          <span className="text-zinc-500">Now:</span> {currentFen}
        </div>
      </div>
    </details>
  );
}

function ActionBar({
  phase,
  busy,
  onStart,
  onCapture,
  onMove,
}: {
  phase: Phase;
  busy: boolean;
  onStart: () => void;
  onCapture: () => void;
  onMove: () => void;
}) {
  if (phase === "ended") {
    return (
      <div className="fixed inset-x-0 bottom-0 z-30 border-t border-white/5 bg-zinc-950/95 px-4 py-5 text-center backdrop-blur">
        <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-emerald-300">
          Game ended
        </p>
        <p className="mt-1 text-sm text-zinc-300">
          Export your PGN from the top bar.
        </p>
      </div>
    );
  }
  if (phase === "confirm" || phase === "locking") return null;
  const label =
    phase === "needCamera"
      ? "Start camera"
      : phase === "framing"
        ? "Capture starting position"
        : "Capture move";
  const handler =
    phase === "needCamera"
      ? onStart
      : phase === "framing"
        ? onCapture
        : phase === "playing"
          ? onMove
          : undefined;
  if (!handler) return null;
  return (
    <div className="fixed inset-x-0 bottom-0 z-30 flex justify-center border-t border-white/5 bg-zinc-950/95 px-4 py-5 backdrop-blur">
      <button
        onClick={handler}
        disabled={busy}
        className="flex h-14 w-full max-w-md items-center justify-center gap-2 rounded-full bg-emerald-500 text-base font-semibold text-emerald-950 shadow-2xl transition hover:bg-emerald-400 disabled:opacity-50 sm:h-16"
      >
        {busy ? "Working…" : label}
      </button>
    </div>
  );
}

function gameIsOver(fen: string): boolean {
  try {
    const g = new Chess(fen);
    return g.isGameOver();
  } catch {
    return false;
  }
}
