"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import Link from "next/link";
import { Chess } from "chess.js";
import clsx from "clsx";
import { extractSquareCrops, warpBoard } from "@/lib/board-image";
import type { Point } from "@/lib/homography";
import { classifyBoard } from "@/lib/occupancy";
import { inferMove } from "@/lib/move-inference";

type Side = "white" | "black";
type Phase = "settings" | "calibrating" | "playing" | "paused" | "ended";

type TimeControl = { baseSeconds: number; incrementSeconds: number };

type CaptureInference =
  | { kind: "matched"; san: string; from: string; to: string; fen: string }
  | { kind: "ambiguous"; sans: string[] }
  | { kind: "unmatched"; diff: string[] }
  | { kind: "skipped"; reason: string };

type Capture = {
  side: Side;
  moveNumber: number;
  url: string;
  timestamp: number;
  inference: CaptureInference;
};

type WakeLock = { release: () => Promise<void> };

type VideoDims = { w: number; h: number };

const PRESETS: { label: string; tc: TimeControl }[] = [
  { label: "1 min · Bullet", tc: { baseSeconds: 60, incrementSeconds: 0 } },
  { label: "3 min · Blitz", tc: { baseSeconds: 180, incrementSeconds: 0 } },
  { label: "5 min · Blitz", tc: { baseSeconds: 300, incrementSeconds: 0 } },
  { label: "10 min · Rapid", tc: { baseSeconds: 600, incrementSeconds: 0 } },
  { label: "3 min + 2", tc: { baseSeconds: 180, incrementSeconds: 2 } },
  { label: "5 min + 3", tc: { baseSeconds: 300, incrementSeconds: 3 } },
  { label: "15 min + 10", tc: { baseSeconds: 900, incrementSeconds: 10 } },
];

const DEFAULT_TC = PRESETS[2].tc;
const TC_STORAGE = "chesspar:capture-tc-v1";
const CORNERS_STORAGE = "chesspar:capture-corners-v1";
const RECTIFIED_SIZE = 384;
const CORNER_LABELS = ["a8", "h8", "h1", "a1"] as const;
const CORNER_HINTS = [
  "Tap the a8 corner — Black's queenside-rook corner",
  "Tap the h8 corner — Black's kingside-rook corner",
  "Tap the h1 corner — White's kingside-rook corner",
  "Tap the a1 corner — White's queenside-rook corner",
];

export function CaptureGame() {
  const [phase, setPhase] = useState<Phase>("settings");
  const [tc, setTc] = useState<TimeControl>(DEFAULT_TC);
  const [whiteMs, setWhiteMs] = useState(DEFAULT_TC.baseSeconds * 1000);
  const [blackMs, setBlackMs] = useState(DEFAULT_TC.baseSeconds * 1000);
  const [active, setActive] = useState<Side>("white");
  const [moves, setMoves] = useState<{ white: number; black: number }>({
    white: 0,
    black: 0,
  });
  const [captures, setCaptures] = useState<Capture[]>([]);
  const [showCaptures, setShowCaptures] = useState(false);
  const [soundOn, setSoundOn] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [winner, setWinner] = useState<Side | null>(null);
  const [hydrated, setHydrated] = useState(false);

  const [corners, setCorners] = useState<Point[]>([]);
  const cornersRef = useRef<Point[]>([]);
  const [videoDims, setVideoDims] = useState<VideoDims | null>(null);

  const [lastMove, setLastMove] = useState<
    { san: string; side: Side } | null
  >(null);
  const [inferring, setInferring] = useState(false);

  const chessRef = useRef<Chess>(new Chess());
  const [pgn, setPgn] = useState<string>("");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wakeLockRef = useRef<WakeLock | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const flashTimeoutRef = useRef<number | null>(null);
  const [flashing, setFlashing] = useState<Side | null>(null);

  useEffect(() => {
    cornersRef.current = corners;
  }, [corners]);

  useEffect(() => {
    if (typeof window === "undefined") {
      setHydrated(true);
      return;
    }
    try {
      const rawTc = window.localStorage.getItem(TC_STORAGE);
      if (rawTc) {
        const parsed = JSON.parse(rawTc) as Partial<TimeControl>;
        if (
          typeof parsed.baseSeconds === "number" &&
          typeof parsed.incrementSeconds === "number" &&
          parsed.baseSeconds > 0
        ) {
          setTc({
            baseSeconds: parsed.baseSeconds,
            incrementSeconds: parsed.incrementSeconds,
          });
        }
      }
      const rawCorners = window.localStorage.getItem(CORNERS_STORAGE);
      if (rawCorners) {
        const parsed = JSON.parse(rawCorners) as Point[];
        if (Array.isArray(parsed) && parsed.length === 4) {
          setCorners(parsed);
        }
      }
    } catch {
      /* ignore */
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated || typeof window === "undefined") return;
    try {
      window.localStorage.setItem(TC_STORAGE, JSON.stringify(tc));
    } catch {
      /* ignore */
    }
  }, [tc, hydrated]);

  useEffect(() => {
    if (!hydrated || typeof window === "undefined") return;
    try {
      if (corners.length === 4) {
        window.localStorage.setItem(CORNERS_STORAGE, JSON.stringify(corners));
      }
    } catch {
      /* ignore */
    }
  }, [corners, hydrated]);

  useEffect(() => {
    if (phase === "settings") {
      setWhiteMs(tc.baseSeconds * 1000);
      setBlackMs(tc.baseSeconds * 1000);
    }
  }, [tc, phase]);

  const stopCamera = useCallback(() => {
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const startCamera = useCallback(async () => {
    if (streamRef.current) return;
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera API not available in this browser");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (v) {
        v.srcObject = stream;
        await v.play().catch(() => {});
      }
      setCameraError(null);
    } catch (e) {
      setCameraError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    if (phase === "settings" || phase === "ended") {
      stopCamera();
      return;
    }
    if (!streamRef.current) {
      void startCamera();
    }
  }, [phase, startCamera, stopCamera]);

  useEffect(() => {
    if (phase !== "playing") return;
    let cancelled = false;
    (async () => {
      try {
        const nav = navigator as Navigator & {
          wakeLock?: { request: (kind: "screen") => Promise<WakeLock> };
        };
        const lock = await nav.wakeLock?.request("screen");
        if (!lock) return;
        if (cancelled) {
          await lock.release().catch(() => {});
        } else {
          wakeLockRef.current = lock;
        }
      } catch {
        /* ignore */
      }
    })();
    return () => {
      cancelled = true;
      const lock = wakeLockRef.current;
      if (lock) {
        lock.release().catch(() => {});
        wakeLockRef.current = null;
      }
    };
  }, [phase]);

  useEffect(() => {
    if (phase !== "playing") return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = now - last;
      last = now;
      if (active === "white") {
        setWhiteMs((ms) => Math.max(0, ms - dt));
      } else {
        setBlackMs((ms) => Math.max(0, ms - dt));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [phase, active]);

  useEffect(() => {
    if (phase !== "playing") return;
    if (whiteMs <= 0) {
      setPhase("ended");
      setWinner("black");
      setPgn(chessRef.current.pgn());
    } else if (blackMs <= 0) {
      setPhase("ended");
      setWinner("white");
      setPgn(chessRef.current.pgn());
    }
  }, [phase, whiteMs, blackMs]);

  const capturedUrlsRef = useRef<string[]>([]);
  useEffect(() => {
    capturedUrlsRef.current = captures.map((c) => c.url);
  }, [captures]);

  useEffect(() => {
    return () => {
      capturedUrlsRef.current.forEach((u) => URL.revokeObjectURL(u));
      stopCamera();
      if (flashTimeoutRef.current) {
        window.clearTimeout(flashTimeoutRef.current);
      }
    };
  }, [stopCamera]);

  const playTickSound = useCallback(() => {
    if (!soundOn) return;
    try {
      const ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      if (!ctor) return;
      const ctx = audioCtxRef.current ?? new ctor();
      audioCtxRef.current = ctx;
      if (ctx.state === "suspended") void ctx.resume();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.18, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.08);
      osc.start();
      osc.stop(ctx.currentTime + 0.09);
    } catch {
      /* ignore */
    }
  }, [soundOn]);

  function grabVideoFrame(): HTMLCanvasElement | null {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return null;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(v, 0, 0);
    return c;
  }

  function canvasToBlobUrl(c: HTMLCanvasElement): Promise<string | null> {
    return new Promise((resolve) => {
      c.toBlob(
        (blob) => {
          if (!blob) return resolve(null);
          resolve(URL.createObjectURL(blob));
        },
        "image/jpeg",
        0.85,
      );
    });
  }

  function startSession() {
    chessRef.current = new Chess();
    captures.forEach((c) => URL.revokeObjectURL(c.url));
    setCaptures([]);
    setLastMove(null);
    setMoves({ white: 0, black: 0 });
    setActive("white");
    setWhiteMs(tc.baseSeconds * 1000);
    setBlackMs(tc.baseSeconds * 1000);
    setWinner(null);
    setPgn("");
    if (corners.length === 4) {
      setPhase("playing");
    } else {
      setPhase("calibrating");
    }
  }

  function startPlayingFromCalibration() {
    if (cornersRef.current.length !== 4) return;
    setPhase("playing");
  }

  function recalibrate() {
    setCorners([]);
    setPhase("calibrating");
  }

  function backToSettings() {
    setPhase("settings");
    stopCamera();
  }

  function togglePause() {
    setPhase((p) =>
      p === "playing" ? "paused" : p === "paused" ? "playing" : p,
    );
  }

  const onVideoMeta = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.videoWidth > 0 && v.videoHeight > 0) {
      setVideoDims({ w: v.videoWidth, h: v.videoHeight });
    }
  }, []);

  function onCalibrationTap(e: React.PointerEvent<HTMLDivElement>) {
    if (corners.length >= 4 || !videoDims) return;
    const target = e.currentTarget;
    const rect = target.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * videoDims.w;
    const y = ((e.clientY - rect.top) / rect.height) * videoDims.h;
    setCorners((c) => [...c, { x, y }]);
  }

  function undoCorner() {
    setCorners((c) => c.slice(0, -1));
  }

  function resetCorners() {
    setCorners([]);
  }

  async function runInferencePipeline(side: Side, moveNumber: number) {
    const cs = cornersRef.current;
    if (cs.length !== 4) {
      const frame = grabVideoFrame();
      const url = frame ? await canvasToBlobUrl(frame) : null;
      if (url) {
        setCaptures((prev) => [
          ...prev,
          {
            side,
            moveNumber,
            url,
            timestamp: Date.now(),
            inference: { kind: "skipped", reason: "no calibration" },
          },
        ]);
      }
      return;
    }

    const frame = grabVideoFrame();
    if (!frame) return;
    const url = await canvasToBlobUrl(frame);
    if (!url) return;

    let inference: CaptureInference;
    try {
      const warped = warpBoard(
        frame,
        cs as [Point, Point, Point, Point],
        RECTIFIED_SIZE,
      );
      const crops = extractSquareCrops(warped);
      const occupancy = classifyBoard(crops).map((c) => c.state);
      const result = inferMove(chessRef.current.fen(), occupancy);
      if (result.kind === "matched") {
        const move = result.move;
        chessRef.current.move({
          from: move.from,
          to: move.to,
          promotion: move.promotion ?? "q",
        });
        inference = {
          kind: "matched",
          san: move.san,
          from: move.from,
          to: move.to,
          fen: result.updatedFen,
        };
        setLastMove({ san: move.san, side });
      } else if (result.kind === "ambiguous") {
        // Default to queen-promotion; user can adjust later.
        const queen = result.candidates.find((m) => m.promotion === "q");
        const pick = queen ?? result.candidates[0];
        chessRef.current.move({
          from: pick.from,
          to: pick.to,
          promotion: pick.promotion ?? "q",
        });
        inference = {
          kind: "ambiguous",
          sans: result.candidates.map((m) => m.san),
        };
        setLastMove({ san: pick.san, side });
      } else {
        inference = {
          kind: "unmatched",
          diff: result.diff.map(
            (d) => `${d.square}:${d.before[0]}→${d.after[0]}`,
          ),
        };
      }
    } catch (e) {
      inference = {
        kind: "unmatched",
        diff: [e instanceof Error ? e.message : String(e)],
      };
    }

    setCaptures((prev) => [
      ...prev,
      { side, moveNumber, url, timestamp: Date.now(), inference },
    ]);
  }

  function endTurn(side: Side) {
    if (phase !== "playing" || active !== side) return;
    const isWhite = side === "white";
    if (tc.incrementSeconds) {
      if (isWhite) setWhiteMs((ms) => ms + tc.incrementSeconds * 1000);
      else setBlackMs((ms) => ms + tc.incrementSeconds * 1000);
    }
    const nextMoves = { ...moves, [side]: moves[side] + 1 };
    setMoves(nextMoves);
    setActive(isWhite ? "black" : "white");
    playTickSound();
    setFlashing(side);
    if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
    flashTimeoutRef.current = window.setTimeout(() => setFlashing(null), 180);

    const totalMoves = nextMoves.white + nextMoves.black;
    setInferring(true);
    void runInferencePipeline(side, totalMoves).finally(() =>
      setInferring(false),
    );
  }

  function endGame() {
    setPgn(chessRef.current.pgn());
    setPhase("ended");
  }

  const tcLabel = useMemo(() => describeTc(tc), [tc]);

  const calibrationHint =
    corners.length < 4
      ? CORNER_HINTS[corners.length]
      : "All four corners set. Confirm to start the clock.";

  return (
    <div className="fixed inset-0 flex flex-col overflow-hidden bg-zinc-950 text-zinc-100 select-none">
      {/*
        The video element stays mounted across phases so the MediaStream keeps
        flowing into a DOM-attached video (some browsers pause display:none
        videos). During calibration the wrapper expands to fill the screen;
        otherwise it collapses to a tiny invisible corner.
      */}
      <div
        className={clsx(
          "flex flex-col",
          phase === "calibrating"
            ? "flex-1"
            : "pointer-events-none absolute h-1 w-1 overflow-hidden opacity-0",
        )}
      >
        {phase === "calibrating" && (
          <div className="flex shrink-0 items-start gap-2 px-3 py-2 text-xs">
            <button
              onClick={backToSettings}
              className="shrink-0 rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-1 text-zinc-200 hover:bg-zinc-800"
            >
              ← Back
            </button>
            <div className="flex-1 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-emerald-100">
              {calibrationHint}
            </div>
          </div>
        )}
        <div
          className={clsx(
            "flex flex-1 items-center justify-center bg-black",
            phase === "calibrating" ? "p-2" : "",
          )}
        >
          <div
            onPointerDown={
              phase === "calibrating" ? onCalibrationTap : undefined
            }
            className="relative overflow-hidden rounded-md border border-zinc-800 bg-zinc-950"
            style={
              videoDims
                ? {
                    aspectRatio: `${videoDims.w}/${videoDims.h}`,
                    maxWidth: "100%",
                    maxHeight: "100%",
                  }
                : { aspectRatio: "16/9", width: 1, height: 1 }
            }
          >
            <video
              ref={videoRef}
              muted
              playsInline
              autoPlay
              onLoadedMetadata={onVideoMeta}
              className="block h-full w-full bg-black"
            />
            {phase === "calibrating" && videoDims && (
              <CalibrationOverlay corners={corners} videoDims={videoDims} />
            )}
          </div>
        </div>
        {phase === "calibrating" && (
          <div className="flex shrink-0 items-center justify-center gap-2 bg-black/95 px-3 py-3">
            <button
              onClick={undoCorner}
              disabled={corners.length === 0}
              className="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Undo
            </button>
            <button
              onClick={resetCorners}
              disabled={corners.length === 0}
              className="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Reset
            </button>
            <button
              onClick={startPlayingFromCalibration}
              disabled={corners.length !== 4}
              className="rounded-md border border-emerald-500/50 bg-emerald-500/20 px-4 py-2 text-sm font-medium text-emerald-100 hover:bg-emerald-500/30 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Start clock
            </button>
          </div>
        )}
        {phase === "calibrating" && cameraError && (
          <div className="mx-3 mb-3 shrink-0 rounded-md border border-amber-500/40 bg-amber-500/15 px-3 py-2 text-xs text-amber-100">
            Camera unavailable: {cameraError}
          </div>
        )}
      </div>

      {phase === "settings" && (
        <SettingsScreen
          tc={tc}
          onChangeTc={setTc}
          onStart={startSession}
          hasSavedCorners={corners.length === 4}
          onClearCorners={() => {
            setCorners([]);
            try {
              window.localStorage.removeItem(CORNERS_STORAGE);
            } catch {
              /* ignore */
            }
          }}
          cameraError={cameraError}
        />
      )}

      {(phase === "playing" || phase === "paused") && (
        <>
          <PlayerPanel
            side="black"
            rotated
            isActive={phase === "playing" && active === "black"}
            ms={blackMs}
            moves={moves.black}
            tcLabel={tcLabel}
            flash={flashing === "black"}
            onTap={() => endTurn("black")}
            disabled={phase !== "playing" || active !== "black"}
          />
          <CenterBar
            phase={phase}
            soundOn={soundOn}
            captureCount={captures.length}
            lastMove={lastMove}
            inferring={inferring}
            onTogglePause={togglePause}
            onReset={backToSettings}
            onToggleSound={() => setSoundOn((s) => !s)}
            onShowCaptures={() => setShowCaptures(true)}
            onEndGame={endGame}
          />
          <PlayerPanel
            side="white"
            rotated={false}
            isActive={phase === "playing" && active === "white"}
            ms={whiteMs}
            moves={moves.white}
            tcLabel={tcLabel}
            flash={flashing === "white"}
            onTap={() => endTurn("white")}
            disabled={phase !== "playing" || active !== "white"}
          />
          {phase === "paused" && <PausedOverlay onResume={togglePause} />}
        </>
      )}

      {phase === "ended" && (
        <EndScreen
          winner={winner}
          captureCount={captures.length}
          moves={moves}
          pgn={pgn}
          onNewGame={backToSettings}
          onRecalibrate={recalibrate}
          onViewCaptures={() => setShowCaptures(true)}
        />
      )}

      {showCaptures && (
        <CapturesDrawer
          captures={captures}
          onClose={() => setShowCaptures(false)}
        />
      )}
    </div>
  );
}

function PlayerPanel({
  side,
  rotated,
  isActive,
  ms,
  moves,
  tcLabel,
  flash,
  onTap,
  disabled,
}: {
  side: Side;
  rotated: boolean;
  isActive: boolean;
  ms: number;
  moves: number;
  tcLabel: string;
  flash: boolean;
  onTap: () => void;
  disabled: boolean;
}) {
  const low = ms <= 10_000;
  return (
    <button
      onClick={onTap}
      disabled={disabled}
      className={clsx(
        "flex flex-1 select-none flex-col items-center justify-center transition-colors duration-150",
        isActive
          ? "bg-zinc-100 text-zinc-900"
          : "bg-zinc-800 text-zinc-400",
        flash && "ring-4 ring-inset ring-emerald-400/70",
        rotated && "rotate-180",
      )}
    >
      <div className="text-[10px] font-medium uppercase tracking-[0.2em] opacity-60">
        {side === "white" ? "White" : "Black"}
      </div>
      <div
        className={clsx(
          "mt-2 text-7xl font-light tabular-nums tracking-tight md:text-8xl",
          low && isActive && "text-rose-600",
        )}
      >
        {formatTime(ms)}
      </div>
      <div className="mt-3 flex items-center gap-3 text-[11px] uppercase tracking-[0.18em] opacity-60">
        <span>Moves · {moves}</span>
        <span aria-hidden>•</span>
        <span>{tcLabel}</span>
      </div>
    </button>
  );
}

function CenterBar({
  phase,
  soundOn,
  captureCount,
  lastMove,
  inferring,
  onTogglePause,
  onReset,
  onToggleSound,
  onShowCaptures,
  onEndGame,
}: {
  phase: Phase;
  soundOn: boolean;
  captureCount: number;
  lastMove: { san: string; side: Side } | null;
  inferring: boolean;
  onTogglePause: () => void;
  onReset: () => void;
  onToggleSound: () => void;
  onShowCaptures: () => void;
  onEndGame: () => void;
}) {
  return (
    <div className="flex shrink-0 flex-col bg-black/95">
      <div className="flex h-7 items-center justify-center px-3 text-[11px] text-zinc-500">
        {inferring ? (
          <span className="text-emerald-300">Inferring…</span>
        ) : lastMove ? (
          <span>
            Last move ·{" "}
            <span className="font-mono text-zinc-200">{lastMove.san}</span>{" "}
            <span className="text-zinc-500">
              ({lastMove.side === "white" ? "W" : "B"})
            </span>
          </span>
        ) : (
          <span className="text-zinc-600">No moves yet</span>
        )}
      </div>
      <div className="flex h-14 items-center justify-around px-2">
        <IconBtn onClick={onReset} label="Back to settings">
          <ResetIcon />
        </IconBtn>
        <IconBtn
          onClick={onTogglePause}
          label={phase === "paused" ? "Resume" : "Pause"}
        >
          {phase === "paused" ? <PlayIcon /> : <PauseIcon />}
        </IconBtn>
        <IconBtn onClick={onShowCaptures} label="Captures">
          <CamIcon />
          {captureCount > 0 && (
            <span className="absolute -right-1 -top-1 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-emerald-500 px-1 text-[10px] font-bold text-emerald-950">
              {captureCount}
            </span>
          )}
        </IconBtn>
        <IconBtn onClick={onEndGame} label="End game">
          <FlagIcon />
        </IconBtn>
        <IconBtn onClick={onToggleSound} label={soundOn ? "Mute" : "Unmute"}>
          {soundOn ? <SoundOnIcon /> : <SoundOffIcon />}
        </IconBtn>
      </div>
    </div>
  );
}

function PausedOverlay({ onResume }: { onResume: () => void }) {
  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center bg-black/50">
      <button
        onClick={onResume}
        className="pointer-events-auto rounded-full border border-emerald-400/40 bg-emerald-500/15 px-6 py-3 text-sm font-medium uppercase tracking-widest text-emerald-100 hover:bg-emerald-500/25"
      >
        Paused — tap to resume
      </button>
    </div>
  );
}

function CalibrationOverlay({
  corners,
  videoDims,
}: {
  corners: Point[];
  videoDims: VideoDims;
}) {
  const stroke = Math.max(2, videoDims.w / 400);
  const dotR = Math.max(8, videoDims.w / 100);
  const fontSize = Math.max(16, videoDims.w / 35);
  const labelOffset = dotR * 2.2;
  return (
    <svg
      viewBox={`0 0 ${videoDims.w} ${videoDims.h}`}
      className="pointer-events-none absolute inset-0 h-full w-full"
    >
      {corners.length >= 2 && corners.length < 4 && (
        <polyline
          points={corners.map((p) => `${p.x},${p.y}`).join(" ")}
          fill="none"
          stroke="rgba(74,222,128,0.9)"
          strokeWidth={stroke}
        />
      )}
      {corners.length === 4 && (
        <polygon
          points={corners.map((p) => `${p.x},${p.y}`).join(" ")}
          fill="rgba(74,222,128,0.18)"
          stroke="rgba(74,222,128,0.95)"
          strokeWidth={stroke}
        />
      )}
      {corners.map((p, i) => (
        <g key={i}>
          <circle
            cx={p.x}
            cy={p.y}
            r={dotR}
            fill="rgba(16,185,129,0.95)"
            stroke="white"
            strokeWidth={stroke * 0.6}
          />
          <text
            x={p.x}
            y={p.y - labelOffset}
            fill="white"
            stroke="rgba(0,0,0,0.7)"
            strokeWidth={stroke * 0.6}
            paintOrder="stroke"
            fontSize={fontSize}
            fontWeight="bold"
            textAnchor="middle"
          >
            {CORNER_LABELS[i]}
          </text>
        </g>
      ))}
    </svg>
  );
}

function SettingsScreen({
  tc,
  onChangeTc,
  onStart,
  hasSavedCorners,
  onClearCorners,
  cameraError,
}: {
  tc: TimeControl;
  onChangeTc: (tc: TimeControl) => void;
  onStart: () => void;
  hasSavedCorners: boolean;
  onClearCorners: () => void;
  cameraError: string | null;
}) {
  return (
    <div className="relative flex flex-1 flex-col overflow-y-auto px-6 py-10">
      <Link
        href="/"
        className="absolute left-4 top-4 text-xs uppercase tracking-wider text-zinc-400 hover:text-zinc-200"
      >
        ← Home
      </Link>
      <div className="mx-auto w-full max-w-md">
        <h1 className="mb-2 mt-6 text-2xl font-semibold">Live game</h1>
        <p className="mb-6 text-sm text-zinc-400">
          Pick a time control, calibrate the board corners once, then each
          tap captures + infers the move that just happened.
        </p>

        <div className="mb-3 text-xs uppercase tracking-wider text-zinc-500">
          Time control
        </div>
        <div className="mb-6 grid grid-cols-2 gap-2">
          {PRESETS.map((p) => {
            const selected =
              p.tc.baseSeconds === tc.baseSeconds &&
              p.tc.incrementSeconds === tc.incrementSeconds;
            return (
              <button
                key={p.label}
                onClick={() => onChangeTc(p.tc)}
                className={clsx(
                  "rounded-md border px-3 py-2 text-sm transition",
                  selected
                    ? "border-emerald-500/60 bg-emerald-500/15 text-emerald-100"
                    : "border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800",
                )}
              >
                {p.label}
              </button>
            );
          })}
        </div>

        <CustomTcEditor tc={tc} onChange={onChangeTc} />

        {hasSavedCorners && (
          <div className="mt-4 flex items-center justify-between rounded-md border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-xs text-zinc-300">
            <span>Board corners saved from last session.</span>
            <button
              onClick={onClearCorners}
              className="rounded border border-zinc-700 px-2 py-0.5 text-zinc-300 hover:bg-zinc-800"
            >
              Recalibrate
            </button>
          </div>
        )}

        <button
          onClick={onStart}
          className="mt-6 w-full rounded-md border border-emerald-500/50 bg-emerald-500/20 px-4 py-3 text-base font-medium text-emerald-100 hover:bg-emerald-500/30"
        >
          {hasSavedCorners ? "Start game" : "Calibrate & start"}
        </button>

        <p className="mt-3 text-center text-[11px] text-zinc-500">
          You&apos;ll be asked for camera permission so we can capture each
          position.
        </p>

        {cameraError && (
          <div className="mt-4 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
            Camera unavailable: {cameraError}. The clock will still work; no
            photos will be saved.
          </div>
        )}

        <div className="mt-8 rounded-lg border border-zinc-800 bg-zinc-900/40 p-4 text-xs text-zinc-400">
          <div className="mb-1 font-medium uppercase tracking-wider text-zinc-300">
            Tip
          </div>
          Visit{" "}
          <Link href="/detect" className="text-emerald-300 underline">
            /detect
          </Link>{" "}
          to test the rectifier + classifier on a still photo first.
        </div>
      </div>
    </div>
  );
}

function CustomTcEditor({
  tc,
  onChange,
}: {
  tc: TimeControl;
  onChange: (tc: TimeControl) => void;
}) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
      <div className="mb-2 text-xs uppercase tracking-wider text-zinc-500">
        Custom
      </div>
      <div className="flex items-center gap-3">
        <label className="flex flex-1 flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-zinc-500">
            Minutes
          </span>
          <input
            type="number"
            min={1}
            max={180}
            value={Math.max(1, Math.floor(tc.baseSeconds / 60))}
            onChange={(e) =>
              onChange({
                ...tc,
                baseSeconds: Math.max(1, Number(e.target.value) || 1) * 60,
              })
            }
            className="rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm tabular-nums text-zinc-100"
          />
        </label>
        <label className="flex flex-1 flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-zinc-500">
            + Increment (s)
          </span>
          <input
            type="number"
            min={0}
            max={60}
            value={tc.incrementSeconds}
            onChange={(e) =>
              onChange({
                ...tc,
                incrementSeconds: Math.max(0, Number(e.target.value) || 0),
              })
            }
            className="rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm tabular-nums text-zinc-100"
          />
        </label>
      </div>
    </div>
  );
}

function EndScreen({
  winner,
  captureCount,
  moves,
  pgn,
  onNewGame,
  onRecalibrate,
  onViewCaptures,
}: {
  winner: Side | null;
  captureCount: number;
  moves: { white: number; black: number };
  pgn: string;
  onNewGame: () => void;
  onRecalibrate: () => void;
  onViewCaptures: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const title = winner
    ? `${winner === "white" ? "White" : "Black"} wins on time`
    : "Game ended";

  async function copyPgn() {
    try {
      await navigator.clipboard.writeText(pgn || "");
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  }

  return (
    <div className="flex flex-1 flex-col overflow-y-auto px-6 py-8">
      <div className="mx-auto w-full max-w-md">
        <div className="mb-2 text-xs uppercase tracking-widest text-zinc-500">
          Game over
        </div>
        <div className="mb-6 text-2xl font-semibold">{title}</div>
        <div className="mb-6 grid grid-cols-2 gap-2 text-sm">
          <div className="rounded-md border border-zinc-800 bg-zinc-900/60 px-3 py-2">
            <div className="text-[10px] uppercase tracking-widest text-zinc-500">
              White moves
            </div>
            <div className="text-xl tabular-nums text-zinc-100">{moves.white}</div>
          </div>
          <div className="rounded-md border border-zinc-800 bg-zinc-900/60 px-3 py-2">
            <div className="text-[10px] uppercase tracking-widest text-zinc-500">
              Black moves
            </div>
            <div className="text-xl tabular-nums text-zinc-100">{moves.black}</div>
          </div>
        </div>
        <div className="mb-6 rounded-md border border-zinc-800 bg-zinc-900/60 px-4 py-3 text-sm text-zinc-300">
          {captureCount} position{captureCount === 1 ? "" : "s"} captured.
        </div>

        {pgn && (
          <div className="mb-6 rounded-md border border-zinc-800 bg-zinc-950 p-3">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-xs uppercase tracking-widest text-zinc-500">
                PGN
              </div>
              <button
                onClick={copyPgn}
                className="rounded border border-zinc-700 bg-zinc-800 px-2 py-0.5 text-xs text-zinc-200 hover:bg-zinc-700"
              >
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] text-zinc-300">
              {pgn}
            </pre>
          </div>
        )}

        <div className="flex flex-col gap-2">
          <button
            onClick={onViewCaptures}
            disabled={captureCount === 0}
            className="rounded-md border border-zinc-700 bg-zinc-800 px-4 py-2 text-sm text-zinc-100 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            View captures
          </button>
          <button
            onClick={onNewGame}
            className="rounded-md border border-emerald-500/40 bg-emerald-500/15 px-4 py-2 text-sm text-emerald-100 hover:bg-emerald-500/25"
          >
            New game
          </button>
          <button
            onClick={onRecalibrate}
            className="rounded-md border border-zinc-800 bg-zinc-900 px-4 py-2 text-sm text-zinc-300 hover:bg-zinc-800"
          >
            Recalibrate corners
          </button>
          <Link
            href="/"
            className="mt-2 text-center text-xs uppercase tracking-wider text-zinc-500 hover:text-zinc-300"
          >
            Back to home
          </Link>
        </div>
      </div>
    </div>
  );
}

function CapturesDrawer({
  captures,
  onClose,
}: {
  captures: Capture[];
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-black/95 px-4 py-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="text-sm font-medium text-zinc-100">
          Captures · {captures.length}
        </div>
        <button
          onClick={onClose}
          className="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1 text-xs text-zinc-200 hover:bg-zinc-700"
        >
          Close
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {captures.length === 0 ? (
          <div className="mt-12 text-center text-sm text-zinc-500">
            No captures yet.
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2 pb-6 sm:grid-cols-3 md:grid-cols-4">
            {captures.map((c, i) => (
              <CaptureCard key={i} capture={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function CaptureCard({ capture }: { capture: Capture }) {
  const inf = capture.inference;
  const badgeText =
    inf.kind === "matched"
      ? inf.san
      : inf.kind === "ambiguous"
        ? "?"
        : inf.kind === "unmatched"
          ? "—"
          : "·";
  const badgeTone =
    inf.kind === "matched"
      ? "bg-emerald-500/25 text-emerald-100"
      : inf.kind === "ambiguous"
        ? "bg-amber-500/25 text-amber-100"
        : inf.kind === "unmatched"
          ? "bg-rose-500/25 text-rose-100"
          : "bg-zinc-700/50 text-zinc-300";
  return (
    <div className="overflow-hidden rounded-md border border-zinc-800 bg-zinc-900">
      <div className="relative">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={capture.url}
          alt={`Move ${capture.moveNumber}`}
          className="block w-full"
        />
        <div
          className={clsx(
            "absolute left-1 top-1 rounded px-1.5 py-0.5 text-[10px] font-mono font-semibold",
            badgeTone,
          )}
        >
          {badgeText}
        </div>
      </div>
      <div className="flex items-center justify-between px-2 py-1 text-[10px] uppercase tracking-wider text-zinc-400">
        <span>#{capture.moveNumber}</span>
        <span>after {capture.side === "white" ? "W" : "B"}</span>
      </div>
    </div>
  );
}

function IconBtn({
  onClick,
  label,
  children,
}: {
  onClick: () => void;
  label: string;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className="relative flex h-11 w-11 items-center justify-center rounded-full text-zinc-300 transition-colors hover:bg-zinc-800 hover:text-white"
    >
      {children}
    </button>
  );
}

function formatTime(ms: number): string {
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

function describeTc(tc: TimeControl): string {
  const m = Math.floor(tc.baseSeconds / 60);
  const s = tc.baseSeconds % 60;
  const base = s ? `${m}:${String(s).padStart(2, "0")}` : `${m} min`;
  return tc.incrementSeconds ? `${base} + ${tc.incrementSeconds}` : base;
}

function ResetIcon() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <polyline points="3 4 3 10 9 10" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <rect x="6" y="5" width="4" height="14" rx="1" />
      <rect x="14" y="5" width="4" height="14" rx="1" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M7 5l12 7-12 7z" />
    </svg>
  );
}

function CamIcon() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M3 7h3l2-3h8l2 3h3v13H3z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  );
}

function FlagIcon() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M5 21V4" />
      <path d="M5 4h11l-1.5 3.5L16 11H5" />
    </svg>
  );
}

function SoundOnIcon() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
      <path d="M18.36 5.64a9 9 0 0 1 0 12.72" />
    </svg>
  );
}

function SoundOffIcon() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <line x1="23" y1="9" x2="17" y2="15" />
      <line x1="17" y1="9" x2="23" y2="15" />
    </svg>
  );
}
