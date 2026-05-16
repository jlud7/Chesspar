"use client";

/**
 * Chesspar v2 — main capture component.
 *
 * Architecture (per the research PDFs at the repo root):
 *   1. Calibrate: snap one photo of the starting position. Florence-2
 *      localizes the playing surface; CV refines to 4 corners; we test
 *      all 4 rotations against a starting-position scorer; on the rare
 *      ambiguous case Gemini picks the right rotation. No manual rotate.
 *   2. Play: each move, capture a 5-frame burst, pick the sharpest,
 *      warp with the cached homography, HSV-V diff against the previous
 *      frame, beam-search legal moves whose template matches, score
 *      confidence, emit if ≥ threshold, else escalate to Gemini, else
 *      tap-to-confirm.
 *
 * State machine (`phase`):
 *   "needCamera"  → user must grant camera access
 *   "framing"     → live preview while user aims at the board
 *   "locking"     → calibration in flight (Florence-2 + rotation)
 *   "confirm"     → show rectified starting position, ask user to confirm
 *   "playing"     → main game loop
 *   "abstain"     → modal showing top-2 candidates for tap-to-confirm
 *   "ended"       → game over, PGN ready for export
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Chess } from "chess.js";
import { LivePreview } from "./LivePreview";
import { MoveList } from "./MoveList";
import { AbstentionPrompt } from "./AbstentionPrompt";
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
  const [lockedFen, setLockedFen] = useState<string>(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  );
  const [moves, setMoves] = useState<string[]>([]);
  const [abstention, setAbstention] = useState<AbstentionState | null>(null);
  const [lastDecision, setLastDecision] = useState<{
    san?: string;
    pConf?: number;
    latencyMs?: number;
    escalation?: "none" | "vlm";
  } | null>(null);

  const config: SessionConfig = useMemo(
    () => ({
      ...DEFAULT_CONFIG,
      proxyUrl: PROXY_URL,
      enableVlmEscalation: !!PROXY_URL,
    }),
    [],
  );

  // ----- camera lifecycle -----
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

  // ----- calibration: capture starting position + lock -----
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
      // If the starting check score is low, surface a warning but still
      // let the user confirm — sometimes mid-game calibration is desired.
      if (result.startingCheck.score < 0.7) {
        setStatusMsg(
          `Looks unusual (${Math.round(result.startingCheck.score * 100)}% match). If this is the starting position, recapture in better light or angle.`,
        );
      } else {
        setStatusMsg(
          result.magicEscalation
            ? "Locked (orientation confirmed by VLM)."
            : "Locked. Confirm starting position.",
        );
      }
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
    lockRef.current = null;
    prevRectifiedRef.current = null;
    setStatusMsg(null);
    setPhase("framing");
  }, []);

  const confirmLock = useCallback(() => {
    setStatusMsg(null);
    setMoves([]);
    setLastDecision(null);
    setPhase("playing");
  }, []);

  // ----- gameplay: capture a move -----
  const captureMove = useCallback(async () => {
    const cam = cameraRef.current;
    const lock = lockRef.current;
    const prev = prevRectifiedRef.current;
    if (!cam || !lock || !prev) return;
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
        previousRectified: prev,
        config,
      });
      if (result.decision.kind === "matched") {
        const newFen = applyMove(fenRef.current, result.decision.pick.san);
        if (!newFen) {
          setStatusMsg(
            `Inferred ${result.decision.pick.san} but couldn't apply it. Try again.`,
          );
          setBusy(false);
          return;
        }
        fenRef.current = newFen;
        prevRectifiedRef.current = result.rectified;
        setMoves((m) => [...m, result.decision.kind === "matched" ? result.decision.pick.san : ""].filter(Boolean));
        setLastDecision({
          san: result.decision.pick.san,
          pConf: result.decision.pConfident,
          latencyMs: result.decision.latencyMs,
          escalation: result.decision.escalation,
        });
        setStatusMsg(null);
        // Auto-end on game-over (checkmate / stalemate / threefold etc.).
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

  const handleAbstentionPick = useCallback((san: string) => {
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
  }, [abstention]);

  const cancelAbstention = useCallback(() => {
    setAbstention(null);
    setPhase("playing");
  }, []);

  const resign = useCallback(() => {
    setPhase("ended");
  }, []);

  const downloadPgn = useCallback(() => {
    const pgn = buildPgn(moves, { result: phase === "ended" ? "*" : "*" });
    const blob = new Blob([pgn], { type: "application/x-chess-pgn" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chesspar-${Date.now()}.pgn`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, [moves, phase]);

  // ----- layout -----
  return (
    <main className="relative flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <Header phase={phase} onResign={resign} onDownload={downloadPgn} hasMoves={moves.length > 0} />

      <div className="relative flex flex-1 flex-col">
        <div className="relative h-[55vh] w-full sm:h-[60vh]">
          {phase === "confirm" && confirmPreview ? (
            <ConfirmView
              previewUrl={confirmPreview}
              onConfirm={confirmLock}
              onRetake={retake}
              statusMsg={statusMsg}
              busy={busy}
            />
          ) : (
            <LivePreview ref={videoRef}>
              <FramingOverlay phase={phase} />
            </LivePreview>
          )}
        </div>

        <div className="flex-1 px-4 pb-32 pt-4 sm:px-8">
          <StatusBar
            phase={phase}
            statusMsg={statusMsg}
            lastDecision={lastDecision}
            busy={busy}
            proxyConfigured={!!PROXY_URL}
          />
          {phase === "playing" || phase === "abstain" || phase === "ended" ? (
            <div className="mt-4">
              <MoveList moves={moves} abstainingOn={phase === "abstain"} />
              <FenBox fen={lockedFen} currentFen={fenRef.current} />
            </div>
          ) : null}
        </div>

        {/* Floating bottom action bar */}
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
      <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/70 p-6 text-center">
        <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-emerald-300">
          Setup
        </p>
        <h2 className="mt-3 text-2xl font-semibold leading-tight">
          Point your phone at the board.
        </h2>
        <p className="mt-3 max-w-md text-sm text-zinc-400">
          Set the phone above the board with all four corners visible. Tap
          start to enable the camera.
        </p>
      </div>
    );
  }
  if (phase === "framing") {
    return (
      <>
        {/* Trapezoid framing guide — visual hint, NOT a constraint. The
            board doesn't have to fit inside this rectangle; corner
            detection works on any quad in the frame. */}
        <div className="pointer-events-none absolute inset-6 rounded-lg border-2 border-dashed border-emerald-300/50" />
        <div className="pointer-events-none absolute left-1/2 top-3 -translate-x-1/2 rounded-full bg-zinc-950/60 px-3 py-1 text-[11px] font-medium uppercase tracking-widest text-emerald-300">
          Frame the board
        </div>
      </>
    );
  }
  if (phase === "locking") {
    return (
      <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/60">
        <div className="rounded-full bg-zinc-900/80 px-4 py-2 text-sm font-medium text-emerald-300">
          Locking board…
        </div>
      </div>
    );
  }
  return null;
}

function ConfirmView({
  previewUrl,
  onConfirm,
  onRetake,
  statusMsg,
  busy,
}: {
  previewUrl: string;
  onConfirm: () => void;
  onRetake: () => void;
  statusMsg: string | null;
  busy: boolean;
}) {
  return (
    <div className="relative flex h-full items-center justify-center bg-zinc-950 p-4">
      <div className="flex flex-col items-center gap-3">
        <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-emerald-300">
          Confirm starting position
        </p>
        <div className="overflow-hidden rounded-xl border border-emerald-400/30 shadow-2xl">
          <img
            src={previewUrl}
            alt="Rectified starting position"
            className="block max-h-[44vh] w-auto"
          />
        </div>
        {statusMsg && (
          <p className="max-w-sm text-center text-[12px] leading-snug text-zinc-400">
            {statusMsg}
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
    escalation?: "none" | "vlm";
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
        <span className="font-mono font-semibold">
          {lastDecision.san}
        </span>
        <span className="text-emerald-300/60">·</span>
        <span className="text-emerald-200/80">
          {Math.round((lastDecision.pConf ?? 0) * 100)}% confidence
        </span>
        {lastDecision.escalation === "vlm" && (
          <span className="rounded-full bg-violet-400/20 px-2 py-0.5 text-[10px] uppercase tracking-widest text-violet-200">
            VLM
          </span>
        )}
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
      <div className="rounded-xl border border-white/5 bg-white/5 px-4 py-3 text-[12px] text-zinc-400">
        Running without VLM escalation. Set `NEXT_PUBLIC_VLM_PROXY_URL` to
        enable Gemini tie-breaks on ambiguous moves.
      </div>
    );
  }
  if (phase === "playing") {
    return (
      <div className="rounded-xl border border-white/5 bg-white/5 px-4 py-3 text-[12px] text-zinc-400">
        Make your move, then tap <span className="font-medium text-zinc-200">Capture move</span> below.
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
  const label =
    phase === "needCamera"
      ? "Start camera"
      : phase === "framing"
        ? "Capture starting position"
        : phase === "confirm" || phase === "locking"
          ? null
          : "Capture move";
  const handler =
    phase === "needCamera"
      ? onStart
      : phase === "framing"
        ? onCapture
        : phase === "playing"
          ? onMove
          : undefined;
  if (!label || !handler) return null;
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
