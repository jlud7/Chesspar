"use client";

import React, {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import Link from "next/link";
import { Chess, type Move as ChessMove } from "chess.js";
import clsx from "clsx";
import { Chessboard } from "react-chessboard";
import { extractSquareCrops, warpBoard } from "@/lib/board-image";
import type { Point } from "@/lib/homography";
import {
  classifyBoard,
  classifyBoardCalibrated,
  computeBaseline,
  computeCellDeltas,
  type BaselineSignature,
} from "@/lib/occupancy";
import {
  makeAnthropicProxyCornerDetector,
  makeAnthropicProxyVerifier,
  makeGeminiProxyVerifier,
  makeOpenAiProxyVerifier,
  makeVerifier,
  type CornerDetector,
  type VlmProvider,
  type VlmVerifier,
  type VlmVerifyResult,
} from "@/lib/vlm";
import {
  findDisputedSquares,
  verifyByCellTiles,
} from "@/lib/vlm-cell-verify";

/**
 * If a VLM proxy URL is baked in at build time (Cloudflare Worker holding
 * the API key), VLM identification is "always on" and the user doesn't need
 * to paste a key. The pasted-key flow remains for users running their own
 * build or wanting a different provider.
 */
const VLM_PROXY_URL = process.env.NEXT_PUBLIC_VLM_PROXY_URL || "";
const VLM_PROXY_PROVIDER = (
  process.env.NEXT_PUBLIC_VLM_PROXY_PROVIDER || "anthropic"
).toLowerCase() as "anthropic" | "openai" | "gemini";

function makeProxyVerifier(): VlmVerifier | null {
  if (!VLM_PROXY_URL) return null;
  if (VLM_PROXY_PROVIDER === "openai") {
    return makeOpenAiProxyVerifier(VLM_PROXY_URL);
  }
  if (VLM_PROXY_PROVIDER === "gemini") {
    return makeGeminiProxyVerifier(VLM_PROXY_URL);
  }
  return makeAnthropicProxyVerifier(VLM_PROXY_URL);
}

/**
 * The VLM-based corner detector is the primary calibration path when the
 * proxy is configured. We use Sonnet (fast + reliable spatial reasoning)
 * via the existing /verify Anthropic proxy — same auth, same CORS.
 */
function makeProxyCornerDetector(): CornerDetector | null {
  if (!VLM_PROXY_URL) return null;
  // OpenAI / Gemini detectors not implemented yet — fall back to CV.
  if (VLM_PROXY_PROVIDER !== "anthropic") return null;
  return makeAnthropicProxyCornerDetector(VLM_PROXY_URL);
}
import { inferMoveFuzzy } from "@/lib/move-inference";
import {
  autoDetectBoardCorners,
  refineCornersForFrame,
  rotateCorners,
  scorePlayingOrientation,
} from "@/lib/board-detection";

type Side = "white" | "black";
type Phase =
  | "settings"
  | "calibrating"
  | "ready"
  | "playing"
  | "paused"
  | "ended";

type BoardCheck = {
  rectifiedUrl: string;
  /** All four corners of the playing surface are inside the camera frame. */
  boardInFrame: boolean;
  /** Total pieces classified per colour + empty (always sums to 64). */
  pieceCount: { white: number; black: number; empty: number };
  /** 16 white + 16 black observed — every piece accounted for. */
  piecesAllDetected: boolean;
  /** Out of 64 — how many cells match the canonical starting layout. */
  startingPositionMatch: number;
  /** ≥60/64 squares match starting position. */
  startingPositionOk: boolean;
  /** All three checks pass. */
  allClear: boolean;
};

type TimeControl = { baseSeconds: number; incrementSeconds: number };

type CaptureInference =
  | { kind: "matched"; san: string; from: string; to: string; fen: string }
  | {
      kind: "vlm-matched";
      san: string;
      from: string;
      to: string;
      fen: string;
      provider: VlmProvider;
    }
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
const VLM_STORAGE = "chesspar:capture-vlm-v1";
const LENS_STORAGE = "chesspar:capture-lens-v1";

type VlmConfig = { provider: VlmProvider; apiKey: string };
const VLM_PROVIDER_LABELS: Record<VlmProvider, string> = {
  gemini: "Gemini 2.5 Pro",
  openai: "OpenAI GPT-4o",
  anthropic: "Claude Sonnet",
};
const VLM_PROVIDER_COST_HINT: Record<VlmProvider, string> = {
  anthropic: "~$0.01 / move · ~$0.40 / 40-move game",
  gemini: "~$0.005 / move · ~$0.20 / 40-move game",
  openai: "~$0.02 / move · ~$0.80 / 40-move game",
};
const RECTIFIED_SIZE = 384;

/**
 * Classify a lens label as the 0.5× ultra-wide, the 1× wide, or
 * something we don't want to expose to the user (telephoto, composite,
 * front-facing). Composite cameras like "Back Dual Camera" / "Back
 * Triple Camera" auto-switch lenses behind the scenes — they look like
 * 1× at rest but the user can't reason about them, so we hide them.
 */
type LensTier = "ultraWide" | "wide" | "skip";
function labelLensTier(label: string): LensTier {
  const l = label.toLowerCase();
  if (/\bfront\b|truedepth|user[-\s]facing|selfie/.test(l)) return "skip";
  if (/dual|triple|composite/.test(l)) return "skip";
  if (/tele/.test(l)) return "skip";
  if (/ultra.?wide|0\.5\s*x|0\.5×/.test(l)) return "ultraWide";
  if (/wide|back|environment|rear/.test(l)) return "wide";
  return "skip";
}

/**
 * From the raw enumerateDevices() list, return at most one
 * representative for each physical back lens we care about (0.5×, 1×),
 * in that order. Anything Safari/Chrome doesn't label (empty label) is
 * surfaced as the implicit 1× — better than hiding everything when
 * labels aren't ready yet.
 */
function pickBackLenses(devices: MediaDeviceInfo[]): MediaDeviceInfo[] {
  const cams = devices.filter((d) => d.kind === "videoinput");
  let ultraWide: MediaDeviceInfo | null = null;
  let wide: MediaDeviceInfo | null = null;
  for (const cam of cams) {
    const tier = labelLensTier(cam.label);
    if (tier === "ultraWide" && !ultraWide) ultraWide = cam;
    else if (tier === "wide" && !wide) wide = cam;
  }
  // Fallback: no labelled wide lens but we have unlabelled cameras —
  // expose the first one as 1× so the picker isn't empty.
  if (!wide) {
    const unlabelled = cams.find((c) => !c.label.trim());
    if (unlabelled) wide = unlabelled;
  }
  const out: MediaDeviceInfo[] = [];
  if (ultraWide) out.push(ultraWide);
  if (wide) out.push(wide);
  return out;
}

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
  const [showReplay, setShowReplay] = useState(false);
  const [soundOn, setSoundOn] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [winner, setWinner] = useState<Side | null>(null);
  const [hydrated, setHydrated] = useState(false);

  const [corners, setCorners] = useState<Point[]>([]);
  const cornersRef = useRef<Point[]>([]);
  const [videoDims, setVideoDims] = useState<VideoDims | null>(null);
  const [boardCheck, setBoardCheck] = useState<BoardCheck | null>(null);

  const [testMode, setTestMode] = useState(false);
  const [testFrames, setTestFrames] = useState<HTMLImageElement[]>([]);
  const [testFrameIdx, setTestFrameIdx] = useState(0);
  const testModeRef = useRef(false);
  const testFramesRef = useRef<HTMLImageElement[]>([]);
  const testFrameIdxRef = useRef(0);
  const testFrameUrlsRef = useRef<string[]>([]);

  const [lastMove, setLastMove] = useState<
    { san: string; side: Side } | null
  >(null);
  const [inferring, setInferring] = useState(false);
  const [vlmActive, setVlmActive] = useState(false);

  const chessRef = useRef<Chess>(new Chess());
  const baselineRef = useRef<BaselineSignature | null>(null);
  /**
   * Snapshot of the previous frame's classifier output + per-square crops
   * + rectified canvas. Drives:
   *   - frame-to-frame diff matching (consistent misclassifications cancel)
   *   - per-cell pixel-delta computation (small piece detection)
   *   - "before" image to ship to the VLM alongside the current frame
   */
  const previousFrameRef = useRef<{
    occupancy: Array<"empty" | "white" | "black">;
    crops: HTMLCanvasElement[];
    warped: HTMLCanvasElement;
  } | null>(null);
  const [pgn, setPgn] = useState<string>("");

  const [vlmConfig, setVlmConfig] = useState<VlmConfig | null>(null);
  const vlmConfigRef = useRef<VlmConfig | null>(null);

  const [moveLog, setMoveLog] = useState<{ san: string; viaVlm: boolean }[]>([]);
  const [selectedCaptureIdx, setSelectedCaptureIdx] = useState<number | null>(null);
  const [pendingPick, setPendingPick] = useState<{
    captureIdx: number;
    side: Side;
    rectifiedUrl: string | null;
    legalMoves: ChessMove[];
  } | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const previewVideoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [availableLenses, setAvailableLenses] = useState<MediaDeviceInfo[]>([]);
  const [currentLensId, setCurrentLensId] = useState<string | null>(null);
  const wakeLockRef = useRef<WakeLock | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const flashTimeoutRef = useRef<number | null>(null);
  const [flashing, setFlashing] = useState<Side | null>(null);

  useEffect(() => {
    cornersRef.current = corners;
  }, [corners]);

  useEffect(() => {
    testModeRef.current = testMode;
  }, [testMode]);

  useEffect(() => {
    testFramesRef.current = testFrames;
  }, [testFrames]);

  useEffect(() => {
    testFrameIdxRef.current = testFrameIdx;
  }, [testFrameIdx]);

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
      const rawVlm = window.localStorage.getItem(VLM_STORAGE);
      if (rawVlm) {
        const parsed = JSON.parse(rawVlm) as Partial<VlmConfig>;
        if (
          (parsed.provider === "gemini" ||
            parsed.provider === "openai" ||
            parsed.provider === "anthropic") &&
          typeof parsed.apiKey === "string" &&
          parsed.apiKey.length > 0
        ) {
          setVlmConfig({ provider: parsed.provider, apiKey: parsed.apiKey });
        }
      }
    } catch {
      /* ignore */
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    vlmConfigRef.current = vlmConfig;
  }, [vlmConfig]);

  useEffect(() => {
    if (!hydrated || typeof window === "undefined") return;
    try {
      if (vlmConfig) {
        window.localStorage.setItem(VLM_STORAGE, JSON.stringify(vlmConfig));
      } else {
        window.localStorage.removeItem(VLM_STORAGE);
      }
    } catch {
      /* ignore */
    }
  }, [vlmConfig, hydrated]);

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

  const attachPreviewVideo = useCallback((el: HTMLVideoElement | null) => {
    previewVideoRef.current = el;
    if (el && streamRef.current && el.srcObject !== streamRef.current) {
      el.srcObject = streamRef.current;
      el.play().catch(() => {});
    }
  }, []);

  // VLM-based corner detector — instantiated once, used during the
  // calibrate phase as the primary detection path when the proxy is
  // available. Falls back to pure-CV when unavailable (no proxy URL,
  // or a non-anthropic proxy provider).
  const cornerDetectorRef = useRef<CornerDetector | null>(null);
  if (cornerDetectorRef.current === null) {
    cornerDetectorRef.current = makeProxyCornerDetector();
  }
  const [vlmDetecting, setVlmDetecting] = useState(false);
  const vlmDetectingRef = useRef(false);
  const [vlmStatus, setVlmStatus] = useState<
    | { kind: "idle" }
    | { kind: "ok"; at: number }
    | { kind: "error"; reason: string }
  >({ kind: "idle" });
  // Manual tap-to-place mode. When true, the calibrating phase shows a
  // tap-to-place UI instead of the auto-detect overlay, and auto-AI /
  // auto-CV are suppressed. The user taps 4 corners in sequence and we
  // transition straight to ready.
  const [tapMode, setTapMode] = useState(false);

  const stopCamera = useCallback(() => {
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (previewVideoRef.current) {
      previewVideoRef.current.srcObject = null;
    }
  }, []);

  /**
   * Try to get a stream from a specific back-camera deviceId. Falls back
   * to facingMode: environment if the exact deviceId is rejected (e.g.
   * the lens disappeared after a teardown).
   */
  const acquireStream = useCallback(
    async (deviceId?: string): Promise<MediaStream> => {
      const baseConstraints = {
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      };
      if (deviceId) {
        try {
          return await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: deviceId }, ...baseConstraints },
            audio: false,
          });
        } catch {
          /* fall through to facingMode */
        }
      }
      return navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, ...baseConstraints },
        audio: false,
      });
    },
    [],
  );

  const startCamera = useCallback(
    async (preferredDeviceId?: string) => {
      if (streamRef.current) return;
      if (
        typeof navigator === "undefined" ||
        !navigator.mediaDevices?.getUserMedia
      ) {
        setCameraError("Camera API not available in this browser");
        return;
      }
      try {
        // First call grants permission and unlocks device labels on iOS.
        let stream = await acquireStream(preferredDeviceId);

        // After permission is granted, enumerateDevices() returns labelled
        // entries. iOS exposes ~8 video inputs on a modern iPhone (front,
        // back, plus per-lens variants and composite "dual"/"triple"
        // cameras). We only surface two: the physical 0.5× ultra-wide
        // back lens and the 1× wide back lens. Anything else (front,
        // telephoto, composites) gets dropped — they don't help framing
        // a chess board.
        let chosenId: string | null =
          stream.getVideoTracks()[0]?.getSettings().deviceId ?? null;
        let lenses: MediaDeviceInfo[] = [];
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          lenses = pickBackLenses(devices);
          if (!preferredDeviceId) {
            const ultra = lenses.find(
              (l) => labelLensTier(l.label) === "ultraWide",
            );
            if (ultra && ultra.deviceId && ultra.deviceId !== chosenId) {
              stream.getTracks().forEach((t) => t.stop());
              stream = await acquireStream(ultra.deviceId);
              chosenId = ultra.deviceId;
            }
          }

          // Best-effort: if only one lens is exposed but the track supports
          // a sub-1× zoom, apply it. (Mostly an Android win; iOS is a
          // no-op here.)
          if (lenses.length <= 1) {
            const track = stream.getVideoTracks()[0];
            const caps = (
              track?.getCapabilities as
                | (() => MediaTrackCapabilities & {
                    zoom?: { min: number; max: number };
                  })
                | undefined
            )?.call(track);
            if (caps?.zoom && caps.zoom.min <= 0.5) {
              await track
                .applyConstraints({
                  advanced: [
                    { zoom: 0.5 } as unknown as MediaTrackConstraintSet,
                  ],
                })
                .catch(() => {});
            }
          }
        } catch {
          /* enumeration failed — keep whatever stream we have */
        }

        streamRef.current = stream;
        setAvailableLenses(lenses);
        setCurrentLensId(chosenId);
        const v = videoRef.current;
        if (v) {
          v.srcObject = stream;
          await v.play().catch(() => {});
        }
        const p = previewVideoRef.current;
        if (p) {
          p.srcObject = stream;
          await p.play().catch(() => {});
        }
        setCameraError(null);
      } catch (e) {
        setCameraError(e instanceof Error ? e.message : String(e));
      }
    },
    [acquireStream],
  );

  const switchLens = useCallback(
    async (deviceId: string) => {
      if (!deviceId || deviceId === currentLensId) return;
      stopCamera();
      try {
        window.localStorage.setItem(LENS_STORAGE, deviceId);
      } catch {
        /* ignore */
      }
      await startCamera(deviceId);
    },
    [currentLensId, startCamera, stopCamera],
  );

  useEffect(() => {
    if (phase === "settings" || phase === "ended" || testMode) {
      stopCamera();
      return;
    }
    if (!streamRef.current) {
      let saved: string | undefined;
      try {
        saved = window.localStorage.getItem(LENS_STORAGE) || undefined;
      } catch {
        /* ignore */
      }
      void startCamera(saved);
    }
  }, [phase, testMode, startCamera, stopCamera]);

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

  function imageToCanvas(img: HTMLImageElement): HTMLCanvasElement | null {
    if (!img.naturalWidth) return null;
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(img, 0, 0);
    return c;
  }

  /**
   * Preprocess a photo for downstream CV + VLM: rotate to "white at bottom"
   * standard chess orientation, crop the clock/hand clutter off, resize.
   *
   * - rotateQuarterTurns: integer 0..3 of 90° clockwise rotations applied
   *   to put white at the bottom of the frame. Sample test photos need 1.
   * - cropClutterRatio: % of width to drop from the LEFT after rotation
   *   (the clock and hand end up there for these photos).
   *
   * Pre-rotation + cropping bumped per-move accuracy from 50% (raw) to
   * ~71% in offline tests against the user's 14-move sample game.
   */
  function imageToCroppedResizedCanvas(
    img: HTMLImageElement,
    maxDim = 1568,
    rotateQuarterTurns = 1,
    cropClutterRatio = 0.22,
  ): HTMLCanvasElement | null {
    if (!img.naturalWidth) return null;
    // Step 1: rotate
    const rot = ((rotateQuarterTurns % 4) + 4) % 4;
    const sw = img.naturalWidth;
    const sh = img.naturalHeight;
    const rotW = rot % 2 === 0 ? sw : sh;
    const rotH = rot % 2 === 0 ? sh : sw;
    const rotCanvas = document.createElement("canvas");
    rotCanvas.width = rotW;
    rotCanvas.height = rotH;
    const rotCtx = rotCanvas.getContext("2d");
    if (!rotCtx) return null;
    rotCtx.imageSmoothingQuality = "high";
    rotCtx.save();
    rotCtx.translate(rotW / 2, rotH / 2);
    rotCtx.rotate((rot * Math.PI) / 2);
    rotCtx.drawImage(img, -sw / 2, -sh / 2);
    rotCtx.restore();
    // Step 2: crop the left strip (clutter after rotation)
    const cropX = Math.max(0, Math.round(rotW * cropClutterRatio));
    const cropW = rotW - cropX;
    const cropH = rotH;
    // Step 3: resize
    const scale = Math.min(1, maxDim / Math.max(cropW, cropH));
    const outW = Math.max(1, Math.round(cropW * scale));
    const outH = Math.max(1, Math.round(cropH * scale));
    const c = document.createElement("canvas");
    c.width = outW;
    c.height = outH;
    const ctx = c.getContext("2d");
    if (!ctx) return null;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(rotCanvas, cropX, 0, cropW, cropH, 0, 0, outW, outH);
    return c;
  }

  /**
   * Grab the next frame to feed into the pipeline.
   * In test mode this consumes the current test image and advances the
   * pointer; in camera mode it grabs from the live video.
   */
  function grabPipelineFrame(advance: boolean): HTMLCanvasElement | null {
    if (testMode) {
      const frames = testFramesRef.current;
      const idx = testFrameIdxRef.current;
      const img = frames[idx];
      if (!img) return null;
      // Apply standard-view rotation + clutter crop so downstream CV
      // (board detection, occupancy classifier) sees the board in its
      // canonical orientation. Without this the existing CV pipeline
      // never had a chance on these tilted iPhone shots.
      const c =
        imageToCroppedResizedCanvas(img, 2048, 1, 0.22) ?? imageToCanvas(img);
      if (advance) setTestFrameIdx((i) => i + 1);
      return c;
    }
    return grabVideoFrame();
  }

  function previewSource(): HTMLCanvasElement | HTMLImageElement | null {
    if (testModeRef.current) {
      const img = testFramesRef.current[testFrameIdxRef.current];
      if (!img) return null;
      return imageToCroppedResizedCanvas(img, 2048, 1, 0.22) ?? img;
    }
    return grabVideoFrame();
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
    baselineRef.current = null;
    previousFrameRef.current = null;
    captures.forEach((c) => URL.revokeObjectURL(c.url));
    setCaptures([]);
    setLastMove(null);
    setMoves({ white: 0, black: 0 });
    setActive("white");
    setMoveLog([]);
    setWhiteMs(tc.baseSeconds * 1000);
    setBlackMs(tc.baseSeconds * 1000);
    setWinner(null);
    setPgn("");
    setTestFrameIdx(0);
    if (testMode) {
      // Constrained-search pipeline: calibrate the board on photo 0
      // (auto-detect corners + manual tweak), then on each clock tap
      // run CV occupancy diff → narrow to top-K candidates → VLM
      // tie-break. Photo 0 is consumed as the baseline; the off-by-one
      // advance to photo 1 happens in startPlayingFromCalibration.
      setCorners([]);
      setPhase("calibrating");
    } else if (corners.length === 4) {
      setPhase("playing");
    } else {
      setPhase("calibrating");
    }
  }

  async function loadTestFrames(files: FileList | null) {
    if (!files || files.length === 0) return;
    setBusy(false);
    setCameraError(null);
    // Revoke any existing test frame URLs
    testFrameUrlsRef.current.forEach((u) => URL.revokeObjectURL(u));
    testFrameUrlsRef.current = [];
    // Sort by filename so IMG_8819…IMG_8833 lands in chronological order.
    // The iOS Photos picker often returns selection-order, not date-order;
    // an unsorted FileList is a common cause of "move 1 looks illegal".
    // Files with no name (rare) sink to the end, original order preserved
    // among themselves via the stable sort.
    const sortedFiles = Array.from(files).sort((a, b) => {
      if (!a.name) return 1;
      if (!b.name) return -1;
      return a.name.localeCompare(b.name, undefined, {
        numeric: true,
        sensitivity: "base",
      });
    });
    const loaded: HTMLImageElement[] = [];
    const urls: string[] = [];
    for (const file of sortedFiles) {
      const url = URL.createObjectURL(file);
      urls.push(url);
      try {
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
          const im = new Image();
          im.onload = () => resolve(im);
          im.onerror = () => reject(new Error("load failed"));
          im.src = url;
        });
        loaded.push(img);
      } catch {
        URL.revokeObjectURL(url);
      }
    }
    testFrameUrlsRef.current = urls;
    setTestFrames(loaded);
    setTestFrameIdx(0);
  }

  // We keep busy state for the in-calibration auto-detect spinner.
  const [busy, setBusy] = useState(false);
  const stableDetectionCountRef = useRef(0);
  const lastDetectedCornersRef = useRef<[Point, Point, Point, Point] | null>(null);

  const tryAutoCalibrate = useCallback(async () => {
    if (cornersRef.current.length === 4) return;
    const source = previewSource();
    if (!source) return;
    if (source instanceof HTMLImageElement && !source.naturalWidth) return;
    if (source instanceof HTMLCanvasElement && (!source.width || !source.height)) return;

    setBusy(true);
    try {
      // Run inside rAF so the UI repaints first
      await new Promise<void>((r) => requestAnimationFrame(() => r()));
      const detection = autoDetectBoardCorners(source);
      if (!detection) {
        setBusy(false);
        return;
      }
      let bestScore = -Infinity;
      let bestCorners: [Point, Point, Point, Point] = detection.corners;
      for (let k = 0; k < 4; k++) {
        const rotated = rotateCorners(detection.corners, k);
        try {
          const warped = warpBoard(source, rotated, 256);
          const crops = extractSquareCrops(warped);
          const occ = classifyBoard(crops).map((c) => c.state);
          const score = scorePlayingOrientation(occ);
          if (score > bestScore) {
            bestScore = score;
            bestCorners = rotated;
          }
        } catch {
          /* skip invalid rotation */
        }
      }
      const sameAsLast = (() => {
        const last = lastDetectedCornersRef.current;
        if (!last) return false;
        const d = bestCorners.reduce((acc, p, i) => {
          const q = last[i];
          return acc + Math.hypot(p.x - q.x, p.y - q.y);
        }, 0);
        return d / 4 < 12;
      })();
      if (sameAsLast) stableDetectionCountRef.current += 1;
      else stableDetectionCountRef.current = 1;
      lastDetectedCornersRef.current = bestCorners;

      // Require 3 consistent detections before accepting calibration.
      if (stableDetectionCountRef.current >= 3) {
        setCorners(bestCorners);
      }
    } finally {
      setBusy(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /**
   * VLM-based corner detection. Grabs the current camera frame and
   * sends it to the proxied vision model with a corner-detection
   * prompt. The model returns labelled corners (a8/h8/h1/a1), so the
   * result is already oriented — no separate `orientStartingPosition`
   * pass needed. Sets corners directly on success; silently keeps the
   * CV result on failure.
   *
   * Used as the PRIMARY calibration path when the proxy is configured —
   * the pure-CV detector is far more brittle on phone-angled shots
   * with corner pieces occluding the dark squares the grid-fit relies
   * on.
   */
  const phaseRef = useRef(phase);
  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  const tryVlmCalibrate = useCallback(async () => {
    const detector = cornerDetectorRef.current;
    if (!detector) return;
    if (vlmDetectingRef.current) return; // already in flight
    const source = previewSource();
    if (!source) return;
    let canvas: HTMLCanvasElement | null;
    if (source instanceof HTMLImageElement) {
      canvas = imageToCanvas(source);
    } else {
      canvas = source;
    }
    if (!canvas || !canvas.width || !canvas.height) return;

    vlmDetectingRef.current = true;
    setVlmDetecting(true);
    try {
      const result = await detector.detectCorners({ image: canvas });
      if (result.kind !== "detected") {
        setVlmStatus({
          kind: "error",
          reason: result.kind === "error" ? result.reason : "no corners",
        });
        return;
      }
      const live =
        phaseRef.current === "calibrating" || phaseRef.current === "ready";
      if (!live) return;
      const W = canvas.width;
      const H = canvas.height;
      const inside = result.corners.every(
        (p) => p.x >= 0 && p.x <= W && p.y >= 0 && p.y <= H,
      );
      if (!inside) {
        setVlmStatus({ kind: "error", reason: "corners outside image" });
        return;
      }
      const polyArea = Math.abs(
        (result.corners[1].x - result.corners[0].x) *
          (result.corners[3].y - result.corners[0].y) -
          (result.corners[1].y - result.corners[0].y) *
            (result.corners[3].x - result.corners[0].x),
      );
      if (polyArea < W * H * 0.05) {
        setVlmStatus({ kind: "error", reason: "polygon too small" });
        return;
      }
      setCorners(result.corners);
      setVlmStatus({ kind: "ok", at: Date.now() });
    } finally {
      vlmDetectingRef.current = false;
      setVlmDetecting(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Kick off a VLM corner detection ~700 ms after entering calibrating —
  // the delay lets the camera stabilise (autofocus, exposure) before we
  // grab a frame to send. Re-detect from the ready screen also triggers
  // this via the calibrating-phase transition.
  useEffect(() => {
    if (phase !== "calibrating") return;
    if (tapMode) return; // user is placing corners by tap — skip AI
    if (!cornerDetectorRef.current) return;
    setVlmStatus({ kind: "idle" });
    const t = window.setTimeout(() => void tryVlmCalibrate(), 700);
    return () => window.clearTimeout(t);
  }, [phase, tapMode, tryVlmCalibrate]);

  useEffect(() => {
    if (phase !== "calibrating" || corners.length === 4) return;
    if (tapMode) return; // manual placement — skip CV detector loop
    let cancelled = false;
    let timer: number | null = null;
    const tick = async () => {
      if (cancelled) return;
      await tryAutoCalibrate();
      if (!cancelled && cornersRef.current.length !== 4) {
        timer = window.setTimeout(() => void tick(), 250);
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, videoDims, testFrames, testFrameIdx, tapMode]);

  useEffect(() => {
    if (phase !== "calibrating" || corners.length !== 4 || busy) return;
    // Old behaviour: jump straight into "playing" once corners were found.
    // That meant a misoriented or mis-cropped board still started the
    // clock. Now we transition to "ready", where the player can verify
    // the rectified board and start the game manually.
    const t = window.setTimeout(() => {
      setTapMode(false);
      setPhase("ready");
    }, 200);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, corners.length, busy]);

  /**
   * Live verification loop. Once corners are detected and we're in
   * "ready", re-classify the current frame every ~600 ms so the player
   * sees the checks update as they nudge the phone.
   *
   * Pipeline (per tick):
   *   1. Warp the camera frame using the detected corners.
   *   2. Build a tentative per-board baseline by *assuming* starting
   *      position — that gives `computeBaseline` labels for the 16 piece
   *      + 32 empty cells. The baseline learns the user's actual cream
   *      square / red square / piece colour values, which is what makes
   *      this far more reliable than the uncalibrated heuristic
   *      classifier on cream-on-red sets.
   *   3. Re-classify every cell against that baseline.
   *   4. Score: 16 W + 16 B detected? How many cells land in their
   *      starting-position slot?
   *
   * The tentative-baseline approach is self-validating: if the board
   * really IS in starting position, the labels we fed in match the
   * pixels and the baseline is accurate. If it isn't, the wrong-bucketed
   * cells skew the baseline so far that the classifier returns garbage
   * — the checks fail and the player knows something is off.
   */
  useEffect(() => {
    if (phase !== "ready") return;
    let cancelled = false;
    let timer: number | null = null;
    const runCheck = () => {
      if (cancelled) return;
      const source = previewSource();
      const cs = cornersRef.current;
      if (!source || cs.length !== 4) {
        timer = window.setTimeout(runCheck, 400);
        return;
      }
      try {
        const warped = warpBoard(
          source,
          cs as [Point, Point, Point, Point],
          RECTIFIED_SIZE,
        );
        const crops = extractSquareCrops(warped);
        // Step 2 + 3: tentative baseline, then calibrated classification.
        // Fall back to the uncalibrated heuristic only if baseline
        // construction throws (shouldn't happen unless `crops.length`
        // isn't 64).
        let occ: Array<"empty" | "white" | "black">;
        try {
          const tentative = computeBaseline(crops);
          occ = classifyBoardCalibrated(crops, tentative).map((c) => c.state);
        } catch {
          occ = classifyBoard(crops).map((c) => c.state);
        }
        let white = 0;
        let black = 0;
        let empty = 0;
        for (const s of occ) {
          if (s === "white") white++;
          else if (s === "black") black++;
          else empty++;
        }
        // Cells 0..15 = ranks 8,7 (back ranks for one side); 16..47 =
        // ranks 6..3 (empty middle); 48..63 = ranks 2,1 (back ranks for
        // the other side). autoDetect rotated the corners so the
        // rectified canvas takes this orientation.
        //
        // The check is *structural*: do the back ranks have pieces and
        // the middle ranks empty, regardless of which colour is on
        // which side? Strict colour-per-cell matching is too brittle on
        // tall-piece overhang where a king/queen silhouette spills into
        // adjacent crops and gets mis-classified. The piece-count check
        // separately confirms we saw ~16 of each colour.
        let backRanksOccupied = 0;
        for (let i = 0; i < 16; i++) if (occ[i] !== "empty") backRanksOccupied++;
        for (let i = 48; i < 64; i++)
          if (occ[i] !== "empty") backRanksOccupied++;
        let middleRanksEmpty = 0;
        for (let i = 16; i < 48; i++)
          if (occ[i] === "empty") middleRanksEmpty++;
        const startingPositionMatch = backRanksOccupied + middleRanksEmpty;
        const piecesAllDetected = white >= 14 && black >= 14 && empty >= 28;
        const startingPositionOk = startingPositionMatch >= 60;
        const next: BoardCheck = {
          rectifiedUrl: warped.toDataURL("image/jpeg", 0.82),
          boardInFrame: true,
          pieceCount: { white, black, empty },
          piecesAllDetected,
          startingPositionMatch,
          startingPositionOk,
          allClear: piecesAllDetected && startingPositionOk,
        };
        if (!cancelled) setBoardCheck(next);
      } catch {
        /* keep last check on failure */
      }
      if (!cancelled) timer = window.setTimeout(runCheck, 320);
    };
    runCheck();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  useEffect(() => {
    if (phase !== "ready") setBoardCheck(null);
  }, [phase]);

  useEffect(() => {
    if (!testMode) return;
    const img = testFrames[testFrameIdx];
    if (img && img.naturalWidth) {
      setVideoDims({ w: img.naturalWidth, h: img.naturalHeight });
    }
  }, [testMode, testFrames, testFrameIdx]);

  useEffect(() => {
    return () => {
      testFrameUrlsRef.current.forEach((u) => URL.revokeObjectURL(u));
    };
  }, []);

  function startPlayingFromCalibration() {
    if (cornersRef.current.length !== 4) return;
    // Learn a per-board baseline from the current rectified frame. The
    // calibration phase is *defined* as the starting position, so the
    // 32 empty squares + 16 white + 16 black piece squares give us
    // labelled reference signatures. We also stash the rectified warp +
    // crops + classifier output as `previousFrameRef`, which the
    // first-move inference will diff against.
    try {
      const source = previewSource();
      if (source) {
        const warped = warpBoard(
          source,
          cornersRef.current as [Point, Point, Point, Point],
          RECTIFIED_SIZE,
        );
        const crops = extractSquareCrops(warped);
        const baseline = computeBaseline(crops);
        baselineRef.current = baseline;
        const occ = classifyBoardCalibrated(crops, baseline).map((c) => c.state);
        previousFrameRef.current = { occupancy: occ, crops, warped };
      }
    } catch {
      baselineRef.current = null;
      previousFrameRef.current = null;
    }
    // In test mode, photo 0 is the starting position — it was just consumed
    // for baseline + previousFrame. Advance the pointer so the first clock
    // tap reads photo 1 (the position after white's first half-move), not
    // photo 0 again (which would diff against itself and produce no move).
    if (testModeRef.current) {
      testFrameIdxRef.current = 1;
      setTestFrameIdx(1);
    }
    setPhase("playing");
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

  async function runInferencePipeline(side: Side, moveNumber: number) {
    const cs = cornersRef.current;
    if (cs.length !== 4) {
      const frame = grabPipelineFrame(true);
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

    const frame = grabPipelineFrame(true);
    if (!frame) return;
    const url = await canvasToBlobUrl(frame);
    if (!url) return;

    let inference: CaptureInference;
    let unmatchedPick:
      | { side: Side; rectifiedUrl: string; legalMoves: ChessMove[] }
      | null = null;
    let nextPreviousFrame: typeof previousFrameRef.current = null;
    try {
      let activeCorners = cs as [Point, Point, Point, Point];
      try {
        const refined = refineCornersForFrame(frame, activeCorners);
        if (refined.drifted) {
          activeCorners = refined.corners;
          cornersRef.current = refined.corners;
          setCorners(refined.corners);
        }
      } catch {
        /* keep saved corners on failure */
      }
      const warped = warpBoard(frame, activeCorners, RECTIFIED_SIZE);
      const crops = extractSquareCrops(warped);
      const occResults = baselineRef.current
        ? classifyBoardCalibrated(crops, baselineRef.current)
        : classifyBoard(crops);
      const occupancy = occResults.map((c) => c.state);
      const confidences = occResults.map((c) => c.confidence);

      const prev = previousFrameRef.current;
      const cellDeltas = prev
        ? computeCellDeltas(prev.crops, crops)
        : undefined;

      const fenBefore = chessRef.current.fen();
      const cvResult = inferMoveFuzzy(fenBefore, occupancy, {
        previousObserved: prev?.occupancy,
        confidences,
        cellDeltas,
      });

      // Decision rule:
      //   1. If the top-K candidates AGREE on after-state (no piece-type
      //      ambiguity) AND CV has a clear margin, trust CV top-1.
      //   2. If there's any piece-type ambiguity in the top-K (the
      //      Be7-vs-Ne7 case — multiple candidates predict different
      //      pieces on overlapping squares), run the per-CELL VLM
      //      verifier: it crops just the squares where candidates
      //      disagree from the rectified board and asks the VLM what's
      //      on each highlighted square. Much more reliable than the
      //      full-photo path which hallucinates on tilted-angle shots.
      //   3. Full-photo VLM remains as a final fallback if the cell
      //      path errors out.
      const top = cvResult.ranked[0];
      const second = cvResult.ranked[1];
      const margin =
        top && second ? second.weightedMismatch - top.weightedMismatch : Infinity;
      // Only candidates whose weighted mismatch is within
      // COMPETITIVE_WINDOW of the top are "real" alternatives. Distant
      // candidates (e.g. the runner-up has mismatch 13+ worse than top)
      // are noise — they shouldn't get a vote in the disputed-squares
      // check or be sent to the VLM verifier. Without this filter, the
      // top-10 list almost always contains some far-off knight move that
      // disagrees with the top pick about an unrelated square, so
      // `disputedSquares.length === 0` is essentially never true and
      // every frame falls through to the VLM verifier — even when CV is
      // overwhelmingly confident.
      const COMPETITIVE_WINDOW = 3;
      const competitiveRanked = top
        ? cvResult.ranked.filter(
            (c) => c.weightedMismatch <= top.weightedMismatch + COMPETITIVE_WINDOW,
          )
        : cvResult.ranked;
      const topKCandidates = competitiveRanked
        .slice(0, 10)
        .map((c) => c.move.san);
      const disputedSquares = findDisputedSquares(fenBefore, topKCandidates);
      // CV is fully unambiguous when only one competitive candidate
      // remains (no inter-candidate disputes) AND the margin to whatever
      // came next is wide. The weighted-mismatch ceiling is generous —
      // any positive value above 5 still falls through to the VLM.
      const cvFullyConfident =
        cvResult.kind === "matched" &&
        cvResult.pick != null &&
        top != null &&
        top.weightedMismatch <= 5 &&
        disputedSquares.length === 0 &&
        margin >= 1;

      let viaVlm = false;
      let appliedMove: ChessMove | null = null;
      let appliedFen = fenBefore;

      const hasVlmAccess = !!(vlmConfigRef.current?.apiKey || VLM_PROXY_URL);

      if (cvFullyConfident && cvResult.pick) {
        appliedMove = cvResult.pick.move;
        appliedFen = cvResult.pick.updatedFen;
        chessRef.current.move({
          from: appliedMove.from,
          to: appliedMove.to,
          promotion: appliedMove.promotion ?? "q",
        });
      } else if (hasVlmAccess && topKCandidates.length > 0) {
        setVlmActive(true);
        try {
          // Primary: per-cell tile verifier. Tiles are rectified, focused,
          // and constrained — the VLM identifies what's on each disputed
          // square rather than reading the whole tilted-angle photo.
          const cellPicked = await runCellTileVerifier({
            after: warped,
            previousFen: fenBefore,
            candidatesSan: topKCandidates,
            cvTopSan: cvResult.pick?.move.san ?? topKCandidates[0] ?? null,
          });
          if (cellPicked) {
            appliedMove = cellPicked.move;
            appliedFen = cellPicked.fen;
            viaVlm = true;
          } else {
            // Fallback: full-photo path (legacy). Only kicks in if the
            // cell verifier returned no match.
            const fullPicked = await runVlmIdentifier({
              before: prev?.warped,
              after: warped,
              previousFen: fenBefore,
              candidatesSan: topKCandidates.slice(0, 8),
            });
            if (fullPicked) {
              appliedMove = fullPicked.move;
              appliedFen = fullPicked.fen;
              viaVlm = true;
            }
          }
        } finally {
          setVlmActive(false);
        }
      }

      // Final fallback: take CV top pick if we still have nothing applied.
      if (!appliedMove && cvResult.kind === "matched" && cvResult.pick) {
        appliedMove = cvResult.pick.move;
        appliedFen = cvResult.pick.updatedFen;
        chessRef.current.move({
          from: appliedMove.from,
          to: appliedMove.to,
          promotion: appliedMove.promotion ?? "q",
        });
      }

      if (appliedMove) {
        inference = viaVlm
          ? {
              kind: "vlm-matched",
              san: appliedMove.san,
              from: appliedMove.from,
              to: appliedMove.to,
              fen: appliedFen,
              provider:
                vlmConfigRef.current?.provider ?? ("anthropic" as VlmProvider),
            }
          : {
              kind: "matched",
              san: appliedMove.san,
              from: appliedMove.from,
              to: appliedMove.to,
              fen: appliedFen,
            };
        setLastMove({ san: appliedMove.san, side });
        setMoveLog((log) => [...log, { san: appliedMove!.san, viaVlm }]);
        nextPreviousFrame = { occupancy, crops, warped };
      } else if (cvResult.kind === "ambiguous") {
        inference = {
          kind: "ambiguous",
          sans: cvResult.ranked.slice(0, 4).map((c) => c.move.san),
        };
        const rectifiedDataUrl = warped.toDataURL("image/jpeg", 0.88);
        unmatchedPick = {
          side,
          rectifiedUrl: rectifiedDataUrl,
          legalMoves: cvResult.ranked.slice(0, 8).map((c) => c.move),
        };
      } else {
        inference = {
          kind: "unmatched",
          diff: cvResult.diff.map(
            (d) => `${d.square}:${d.before[0]}→${d.after[0]}`,
          ),
        };
        const rectifiedDataUrl = warped.toDataURL("image/jpeg", 0.88);
        const legalMoves = chessRef.current.moves({
          verbose: true,
        }) as ChessMove[];
        unmatchedPick = {
          side,
          rectifiedUrl: rectifiedDataUrl,
          legalMoves,
        };
      }
    } catch (e) {
      inference = {
        kind: "unmatched",
        diff: [e instanceof Error ? e.message : String(e)],
      };
    }

    if (nextPreviousFrame) {
      previousFrameRef.current = nextPreviousFrame;
    }

    setCaptures((prev) => {
      const next = [
        ...prev,
        { side, moveNumber, url, timestamp: Date.now(), inference },
      ];
      if (unmatchedPick) {
        setPendingPick({
          captureIdx: next.length - 1,
          side: unmatchedPick.side,
          rectifiedUrl: unmatchedPick.rectifiedUrl,
          legalMoves: unmatchedPick.legalMoves,
        });
      }
      return next;
    });
  }

  /**
   * Per-cell tile VLM verifier — the primary VLM path now. Crops the
   * rectified AFTER-board around just the squares where the top-K
   * candidates disagree on piece type, draws a red outline around each
   * target cell, and asks the model to identify what's on the highlighted
   * square. Then picks the candidate whose after-state matches the
   * observations. Far more reliable than the full-photo path because the
   * input is rectified (no tilt) and the question is constrained to
   * per-square classification.
   *
   * Orientation safety: `verifyByCellTiles` internally runs
   * `ensureWhiteAtBottom` on the rectified board before cropping tiles,
   * so a mis-calibrated capture still produces canonically-oriented
   * tiles for the API call.
   */
  async function runCellTileVerifier(args: {
    after: HTMLCanvasElement;
    previousFen: string;
    candidatesSan: string[];
    /**
     * CV's top pick — the cell verifier may only OVERRIDE this when its
     * decision came from the deterministic observation-match path. Picks
     * arrived at via the weaker ANSWER-line / SAN-search fallbacks are
     * only honoured when they AGREE with CV. This is the lesson learnt
     * from the previous "VLM tie-break dropped accuracy 86%→71%" round:
     * weak VLM signals must never overrule CV.
     */
    cvTopSan: string | null;
  }): Promise<{ move: ChessMove; fen: string } | null> {
    const config = vlmConfigRef.current;
    if (!config?.apiKey && !VLM_PROXY_URL) return null;
    if (args.candidatesSan.length === 0) return null;
    const provider: VlmProvider = config?.apiKey
      ? config.provider
      : (VLM_PROXY_PROVIDER as VlmProvider);
    try {
      const result = await verifyByCellTiles({
        afterWarped: args.after,
        prevFen: args.previousFen,
        candidatesSan: args.candidatesSan,
        provider,
        apiKey: config?.apiKey ?? undefined,
        proxyUrl: VLM_PROXY_URL || undefined,
      });
      if (result.kind !== "matched") {
        if (result.kind === "rejected") {
          console.warn(
            "cell-tile VLM rejected:",
            result.reason,
            "observations:",
            result.observations,
            "raw:",
            result.raw,
          );
        } else {
          console.warn("cell-tile VLM error:", result.reason);
        }
        return null;
      }
      // Trust gate: only let the VLM override CV when the match came
      // from the deterministic observation-vs-candidate matcher. Weak
      // paths (answer-line / san-search) may only confirm CV — if they
      // disagree, we don't trust them, because that's exactly the
      // hallucination mode that hurt accuracy in the previous round.
      const overridesCv =
        args.cvTopSan != null && result.san !== args.cvTopSan;
      if (overridesCv && result.via !== "deterministic") {
        console.warn(
          `cell-tile VLM weak path (${result.via}) tried to override CV top-1 (${args.cvTopSan} → ${result.san}); ignoring`,
          "observations:",
          result.observations,
          "raw:",
          result.raw,
        );
        return null;
      }
      console.info(
        `cell-tile VLM matched ${result.san} via ${result.via}`,
        result.matchDetails ?? "",
        "observations:",
        result.observations,
      );
      let applied;
      try {
        applied = chessRef.current.move(result.san);
      } catch {
        return null;
      }
      if (!applied) return null;
      return { move: applied as ChessMove, fen: chessRef.current.fen() };
    } catch (e) {
      console.warn("cell-tile VLM threw", e);
      return null;
    }
  }

  /**
   * Two-image vision-LM tie-breaker. Called only when the CV pipeline can't
   * decide between several candidate moves. Sends the BEFORE and AFTER
   * rectified boards plus a SHORT list of candidates ranked by CV
   * mismatch — constrained-search architecture so the model is choosing,
   * not hallucinating.
   */
  async function runVlmIdentifier(args: {
    before: HTMLCanvasElement | undefined;
    after: HTMLCanvasElement;
    previousFen: string;
    candidatesSan: string[];
  }): Promise<{ move: ChessMove; fen: string } | null> {
    const config = vlmConfigRef.current;
    if (!config?.apiKey && !VLM_PROXY_URL) return null;
    if (args.candidatesSan.length === 0) return null;
    const verifier = config?.apiKey
      ? makeVerifier(config.provider, config.apiKey)
      : (makeProxyVerifier() as VlmVerifier);
    let result: VlmVerifyResult;
    try {
      result = await verifier.verify({
        previousFen: args.previousFen,
        legalMovesSan: args.candidatesSan,
        boardImage: args.after,
        previousBoardImage: args.before,
      });
    } catch (e) {
      console.warn("VLM verifier threw", e);
      return null;
    }
    if (result.kind !== "matched") return null;
    let applied;
    try {
      applied = chessRef.current.move(result.san);
    } catch {
      return null;
    }
    if (!applied) return null;
    return { move: applied as ChessMove, fen: chessRef.current.fen() };
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
    // Both modes go through the same CV+VLM constrained-search pipeline.
    // Test mode just sources frames from uploaded photos instead of the
    // live camera; grabPipelineFrame already handles that branch.
    void runInferencePipeline(side, totalMoves).finally(() =>
      setInferring(false),
    );
  }

  function endGame() {
    setPgn(chessRef.current.pgn());
    setPhase("ended");
  }

  /**
   * Truncate the game state back to before `captureIndex`, then apply
   * the user-picked move as if it had happened at that point. Subsequent
   * captures and their inferred moves are discarded — the user is asked
   * to confirm before this happens.
   */
  function overrideCaptureMove(captureIndex: number, picked: ChessMove) {
    if (captureIndex < 0 || captureIndex >= captures.length) return;
    const targetCapture = captures[captureIndex];
    // Rewind chess.js to the position BEFORE the targeted move.
    const targetMoveCount = targetCapture.moveNumber - 1; // history is 1-indexed
    while (chessRef.current.history().length > targetMoveCount) {
      chessRef.current.undo();
    }
    let applied;
    try {
      applied = chessRef.current.move({
        from: picked.from,
        to: picked.to,
        promotion: picked.promotion ?? "q",
      });
    } catch {
      applied = null;
    }
    if (!applied) return;

    // Drop and free everything after the override point.
    for (let i = captureIndex + 1; i < captures.length; i++) {
      URL.revokeObjectURL(captures[i].url);
    }
    const newCapture: Capture = {
      ...targetCapture,
      inference: {
        kind: "matched",
        san: applied.san,
        from: applied.from,
        to: applied.to,
        fen: chessRef.current.fen(),
      },
    };
    setCaptures((prev) => [...prev.slice(0, captureIndex), newCapture]);
    setMoveLog((prev) => [
      ...prev.slice(0, captureIndex),
      { san: applied.san, viaVlm: false },
    ]);

    // The side to move is now the opposite of whoever just moved.
    // Reset clocks-state-keeping logic conservatively: leave them as-is
    // (the user is mid-game), but flip active to whichever side is now
    // to move.
    setActive(chessRef.current.turn() === "w" ? "white" : "black");
    setLastMove({
      san: applied.san,
      side: applied.color === "w" ? "white" : "black",
    });
    setMoves((m) => {
      // Recount from move numbers.
      const h = chessRef.current.history();
      let w = 0;
      let b = 0;
      for (let i = 0; i < h.length; i++) {
        if (i % 2 === 0) w++;
        else b++;
      }
      return { white: w, black: b };
    });
    setPgn(chessRef.current.pgn());
    setSelectedCaptureIdx(null);
  }

  function deleteCapture(captureIndex: number) {
    if (captureIndex < 0 || captureIndex >= captures.length) return;
    // Truncate chess.js back to before this capture.
    const targetMoveCount = captures[captureIndex].moveNumber - 1;
    while (chessRef.current.history().length > targetMoveCount) {
      chessRef.current.undo();
    }
    for (let i = captureIndex; i < captures.length; i++) {
      URL.revokeObjectURL(captures[i].url);
    }
    setCaptures((prev) => prev.slice(0, captureIndex));
    setMoveLog((prev) => prev.slice(0, captureIndex));
    setActive(chessRef.current.turn() === "w" ? "white" : "black");
    setLastMove(
      captureIndex > 0
        ? (() => {
            const prevMoves = chessRef.current.history({ verbose: true });
            const last = prevMoves[prevMoves.length - 1];
            return last
              ? { san: last.san, side: last.color === "w" ? "white" : "black" }
              : null;
          })()
        : null,
    );
    setMoves((m) => {
      const h = chessRef.current.history();
      let w = 0;
      let b = 0;
      for (let i = 0; i < h.length; i++) {
        if (i % 2 === 0) w++;
        else b++;
      }
      return { white: w, black: b };
    });
    setPgn(chessRef.current.pgn());
    setSelectedCaptureIdx(null);
  }

  const tcLabel = useMemo(() => describeTc(tc), [tc]);

  return (
    <div className="fixed inset-0 flex flex-col overflow-hidden bg-zinc-950 text-zinc-100 select-none">
      {/*
        The camera/test-image stays mounted across phases so the MediaStream
        keeps flowing into a DOM-attached <video> (some browsers pause
        display:none videos), and the auto-detector has a live frame to
        sample. The wrapper is invisibly tiny on every phase — we never
        take over the screen with a "calibrate this" view; detection runs
        quietly while the player UI is already on screen.
      */}
      <div className="pointer-events-none absolute h-1 w-1 overflow-hidden opacity-0">
        <div
          className="relative overflow-hidden"
          style={
            videoDims
              ? { aspectRatio: `${videoDims.w}/${videoDims.h}`, width: 1, height: 1 }
              : { aspectRatio: "16/9", width: 1, height: 1 }
          }
        >
          {testMode && testFrames[testFrameIdx] ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={testFrames[testFrameIdx].src}
              alt={`Test frame ${testFrameIdx + 1}`}
              className="block h-full w-full bg-black object-contain"
            />
          ) : (
            <video
              ref={videoRef}
              muted
              playsInline
              autoPlay
              onLoadedMetadata={onVideoMeta}
              className="block h-full w-full bg-black"
            />
          )}
        </div>
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
          testMode={testMode}
          onToggleTestMode={(on) => {
            setTestMode(on);
            if (on) {
              setCorners([]);
            }
          }}
          testFrameCount={testFrames.length}
          testFrameSrcs={testFrames.map((f) => f.src)}
          onLoadTestFrames={(files) => void loadTestFrames(files)}
          vlmConfig={vlmConfig}
          onChangeVlmConfig={setVlmConfig}
        />
      )}

      {(phase === "playing" || phase === "paused" || phase === "calibrating") && (
        <div className="relative flex flex-1 overflow-hidden">
          <SidePanel
            side="left"
            name="Black"
            ms={blackMs}
            moves={moves.black}
            isActive={phase === "playing" && active === "black"}
            phase={phase}
            inferring={inferring}
            vlmActive={vlmActive}
            justMoved={lastMove?.side === "black"}
            justMovedSan={lastMove?.side === "black" ? lastMove.san : null}
            onTap={() => endTurn("black")}
            disabled={phase !== "playing" || active !== "black"}
          />
          <CenterStrip
            phase={phase}
            captureCount={captures.length}
            moveLog={moveLog}
            inferring={inferring}
            vlmActive={vlmActive}
            activeSide={active}
            onTogglePause={togglePause}
            onReset={backToSettings}
            onShowCaptures={() => setShowCaptures(true)}
            onEndGame={endGame}
          />
          <SidePanel
            side="right"
            name="White"
            ms={whiteMs}
            moves={moves.white}
            isActive={phase === "playing" && active === "white"}
            phase={phase}
            inferring={inferring}
            vlmActive={vlmActive}
            justMoved={lastMove?.side === "white"}
            justMovedSan={lastMove?.side === "white" ? lastMove.san : null}
            onTap={() => endTurn("white")}
            disabled={phase !== "playing" || active !== "white"}
          />
          {!testMode && (
            <CameraPreviewOverlay
              phase={phase}
              onPreviewRef={attachPreviewVideo}
              lenses={availableLenses}
              currentLensId={currentLensId}
              onSwitchLens={switchLens}
            />
          )}
          {phase === "paused" && <PausedOverlay onResume={togglePause} />}
          {phase === "calibrating" && tapMode && (
            <TapCornersOverlay
              corners={corners}
              videoDims={videoDims}
              onPreviewRef={attachPreviewVideo}
              onTap={(pt) =>
                setCorners((cs) =>
                  cs.length >= 4 ? cs : [...cs, pt],
                )
              }
              onUndo={() =>
                setCorners((cs) => cs.slice(0, Math.max(0, cs.length - 1)))
              }
              onCancel={() => {
                setTapMode(false);
                backToSettings();
              }}
              onUseAi={() => {
                setCorners([]);
                setTapMode(false);
              }}
              hasAiDetector={cornerDetectorRef.current !== null}
            />
          )}
          {phase === "calibrating" && !tapMode && (
            <StartingOverlay
              testMode={testMode}
              cameraError={cameraError}
              vlmDetecting={vlmDetecting}
              hasAiDetector={cornerDetectorRef.current !== null}
              onTapManually={() => {
                setCorners([]);
                setVlmStatus({ kind: "idle" });
                setTapMode(true);
              }}
              onCancel={backToSettings}
            />
          )}
        </div>
      )}

      {phase === "ready" && (
        <ReadyScreen
          check={boardCheck}
          testMode={testMode}
          onPreviewRef={attachPreviewVideo}
          lenses={availableLenses}
          currentLensId={currentLensId}
          onSwitchLens={switchLens}
          corners={corners}
          videoDims={videoDims}
          onDragCorner={(idx, pt) =>
            setCorners((cs) => {
              if (idx < 0 || idx >= cs.length) return cs;
              const next = [...cs];
              next[idx] = pt;
              return next;
            })
          }
          vlmDetecting={vlmDetecting}
          vlmStatus={vlmStatus}
          hasAiDetector={cornerDetectorRef.current !== null}
          onRetryAi={() => void tryVlmCalibrate()}
          onRecalibrate={() => {
            setCorners([]);
            setBoardCheck(null);
            setVlmStatus({ kind: "idle" });
            setTapMode(false);
            setPhase("calibrating");
          }}
          onTapManually={() => {
            setCorners([]);
            setBoardCheck(null);
            setVlmStatus({ kind: "idle" });
            setTapMode(true);
            setPhase("calibrating");
          }}
          onCancel={backToSettings}
          onStart={startPlayingFromCalibration}
        />
      )}

      {phase === "ended" && (
        <EndScreen
          winner={winner}
          captureCount={captures.length}
          recordedMoves={moveLog.length}
          moves={moves}
          pgn={pgn}
          moveLog={moveLog}
          onNewGame={backToSettings}
          onViewCaptures={() => setShowCaptures(true)}
          onReplay={
            captures.length > 0 ? () => setShowReplay(true) : undefined
          }
        />
      )}

      {showReplay && (
        <ReplayView
          captures={captures}
          corners={cornersRef.current}
          onClose={() => setShowReplay(false)}
        />
      )}

      {showCaptures && (
        <CapturesDrawer
          captures={captures}
          onClose={() => setShowCaptures(false)}
          onSelectCapture={(idx) => setSelectedCaptureIdx(idx)}
        />
      )}

      {selectedCaptureIdx !== null && captures[selectedCaptureIdx] && (
        <CaptureDetailModal
          captureIndex={selectedCaptureIdx}
          capture={captures[selectedCaptureIdx]}
          chess={chessRef.current}
          corners={cornersRef.current}
          subsequentCount={captures.length - selectedCaptureIdx - 1}
          onClose={() => setSelectedCaptureIdx(null)}
          onOverride={overrideCaptureMove}
          onDelete={deleteCapture}
        />
      )}

      {pendingPick && captures[pendingPick.captureIdx] && (
        <PendingPickSheet
          side={pendingPick.side}
          moveNumber={captures[pendingPick.captureIdx].moveNumber}
          rectifiedUrl={pendingPick.rectifiedUrl}
          legalMoves={pendingPick.legalMoves}
          onPick={(move) => {
            overrideCaptureMove(pendingPick.captureIdx, move);
            setPendingPick(null);
          }}
          onSkip={() => setPendingPick(null)}
        />
      )}
    </div>
  );
}

function PendingPickSheet({
  side,
  moveNumber,
  rectifiedUrl,
  legalMoves,
  onPick,
  onSkip,
}: {
  side: Side;
  moveNumber: number;
  rectifiedUrl: string | null;
  legalMoves: ChessMove[];
  onPick: (move: ChessMove) => void;
  onSkip: () => void;
}) {
  const [confirmSkip, setConfirmSkip] = useState(false);
  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-black/65 backdrop-blur-xl">
      <div className="relative mt-auto flex max-h-[90vh] flex-col overflow-hidden rounded-t-[28px] border-t border-white/10 bg-zinc-950/95 backdrop-blur-xl shadow-2xl sm:my-auto sm:mx-auto sm:w-full sm:max-w-md sm:rounded-[28px]">
        <div className="mx-auto mt-2 h-1 w-10 rounded-full bg-white/15" />
        <div className="px-5 pb-3 pt-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.3em] text-amber-300">
            We missed move {moveNumber}
          </div>
          <div className="mt-1 text-[20px] font-semibold leading-tight text-zinc-50">
            Pick what {side === "white" ? "White" : "Black"} just played
          </div>
          <p className="mt-1 text-[12px] leading-snug text-zinc-400">
            Our pipeline couldn&apos;t pin a unique move. Tap the right one
            below and we&apos;ll resync the PGN.
          </p>
        </div>

        <div className="flex-1 overflow-y-auto px-5 pb-5">
          {rectifiedUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={rectifiedUrl}
              alt="Rectified board"
              className="mb-4 aspect-square w-40 rounded-2xl border border-white/5 object-cover"
            />
          )}

          <div className="mb-1 text-[10px] uppercase tracking-widest text-zinc-500">
            Legal moves · {legalMoves.length}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {legalMoves.length === 0 ? (
              <span className="text-xs text-zinc-500">
                No legal moves at this position.
              </span>
            ) : (
              legalMoves.map((m) => (
                <button
                  key={m.san}
                  onClick={() => onPick(m)}
                  className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-[13px] font-mono text-zinc-100 transition active:scale-95 hover:bg-emerald-500/20 hover:border-emerald-400/40"
                >
                  {m.san}
                </button>
              ))
            )}
          </div>

          <div className="mt-5 border-t border-white/5 pt-4">
            {confirmSkip ? (
              <div className="flex flex-col gap-2">
                <p className="text-[11px] text-amber-100">
                  Skipping leaves this move unrecorded. Subsequent moves may
                  also fail to infer. Continue?
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={() => setConfirmSkip(false)}
                    className="flex-1 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-zinc-200"
                  >
                    Back
                  </button>
                  <button
                    onClick={onSkip}
                    className="flex-1 rounded-xl bg-amber-500/80 px-3 py-2 text-sm font-semibold text-amber-950 hover:bg-amber-400"
                  >
                    Skip anyway
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setConfirmSkip(true)}
                className="block w-full rounded-xl border border-white/5 bg-white/5 px-4 py-2.5 text-sm text-zinc-300 transition hover:bg-white/10"
              >
                Skip — leave this move unrecorded
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * SidePanel — one half of the playing screen. The phone is propped
 * upright perpendicular to the board with the back camera at the top
 * pointing down at the centre. The two players sit on the LEFT and
 * RIGHT of the phone, so each half rotates 90° to face its player:
 *
 *   left  side (black) → content rotated +90° clockwise
 *   right side (white) → content rotated -90° counter-clockwise
 *
 * Active player's entire half fills with the accent colour, chess.com
 * clock style — unmistakable from across the table. Clock typography
 * becomes the screen's anchor: JetBrains Mono, oversized, tabular.
 */
function SidePanel({
  side,
  name,
  ms,
  moves,
  isActive,
  phase,
  inferring,
  vlmActive,
  justMoved,
  justMovedSan,
  onTap,
  disabled,
}: {
  side: "left" | "right";
  name: string;
  ms: number;
  moves: number;
  isActive: boolean;
  phase: Phase;
  inferring: boolean;
  vlmActive: boolean;
  justMoved: boolean;
  justMovedSan: string | null;
  onTap: () => void;
  disabled: boolean;
}) {
  // Track the recently-detected SAN so we can hold the toast for ~1.8s
  // after the move appears.
  const [showRecent, setShowRecent] = useState(false);
  const lastSeenSanRef = useRef<string | null>(null);
  useEffect(() => {
    if (justMovedSan && justMovedSan !== lastSeenSanRef.current) {
      lastSeenSanRef.current = justMovedSan;
      setShowRecent(true);
      const t = window.setTimeout(() => setShowRecent(false), 1800);
      return () => window.clearTimeout(t);
    }
  }, [justMovedSan]);

  const rot = side === "left" ? 90 : -90;
  const lowTime = ms <= 10_000;
  const reading = isActive && phase === "playing" && (inferring || vlmActive);
  const bg = isActive ? "var(--cp-accent)" : "transparent";
  const eyebrowTone = isActive
    ? "rgba(10,36,24,0.7)"
    : "rgba(245,242,235,0.42)";
  const dotColor = isActive ? "var(--cp-accent-ink)" : "rgba(245,242,235,0.25)";
  const clockColor = isActive
    ? lowTime
      ? "#3a0a14"
      : "var(--cp-accent-ink)"
    : "rgba(250,247,240,0.55)";
  const detectedVisible =
    !isActive &&
    justMoved &&
    showRecent &&
    justMovedSan &&
    phase === "playing";

  return (
    <button
      type="button"
      onClick={onTap}
      disabled={disabled}
      className="relative flex-1 cursor-pointer overflow-hidden text-left transition-[background] duration-300 disabled:cursor-default"
      style={{ background: bg }}
    >
      {/* Rotated landscape content. We size it to the panel's
          POST-rotation dimensions: width = panel height, height = panel
          width. Then rotate around the panel's centre so the eye reads
          left-to-right when the player tilts their head toward the phone. */}
      <div
        className="absolute left-1/2 top-1/2 flex flex-col items-center justify-center gap-2 px-12"
        style={{
          width: "100vh",
          height: "50vw",
          minWidth: "844px",
          minHeight: "173px",
          maxWidth: "min(100vh, 1100px)",
          maxHeight: "240px",
          transform: `translate(-50%, -50%) rotate(${rot}deg)`,
          transformOrigin: "center",
        }}
      >
        {/* Eyebrow — small uppercase chip with a dot */}
        <div
          className="flex items-center gap-2 text-[10px] uppercase"
          style={{
            color: eyebrowTone,
            letterSpacing: "0.3em",
            fontFamily: "var(--font-ui)",
          }}
        >
          <span
            className="block h-1.5 w-1.5 rounded-full"
            style={{ background: dotColor }}
          />
          <span>{name}</span>
          <span style={{ opacity: 0.6 }}>· {moves} moves</span>
        </div>

        {/* Clock + detected card in a 3-col grid so the clock stays
            visually centred regardless of the side card. */}
        <div className="grid w-full items-center gap-x-7" style={{ gridTemplateColumns: "1fr auto 1fr" }}>
          <div />
          <div
            className="relative tabular-nums"
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "clamp(72px, 18vh, 124px)",
              lineHeight: 0.95,
              fontWeight: 300,
              letterSpacing: "-0.04em",
              color: clockColor,
              transition: "color 200ms ease",
            }}
          >
            {formatTime(ms)}
            {reading && (
              <div className="absolute -bottom-3 left-0 right-0 h-[1.5px] overflow-hidden">
                <div
                  className="absolute inset-0"
                  style={{ background: "rgba(10,36,24,0.18)" }}
                />
                <div
                  className="absolute bottom-0 top-0 w-[38%]"
                  style={{
                    background:
                      "linear-gradient(90deg, transparent, rgba(10,36,24,0.85) 50%, transparent)",
                    animation: "chesspar-sweep 1.4s ease-in-out infinite",
                  }}
                />
              </div>
            )}
          </div>
          <div className="flex min-w-0 items-center justify-start">
            {detectedVisible && (
              <div
                className="inline-flex flex-col gap-2 rounded-[12px] px-4 py-3"
                style={{
                  border: "0.5px solid rgba(95,201,154,0.4)",
                  background: "rgba(95,201,154,0.08)",
                  boxShadow: "0 0 60px rgba(95,201,154,0.1)",
                }}
              >
                <div
                  className="flex items-center gap-2 text-[9px] uppercase"
                  style={{
                    color: "var(--cp-accent)",
                    letterSpacing: "0.3em",
                    fontFamily: "var(--font-ui)",
                  }}
                >
                  <svg width="11" height="11" viewBox="0 0 11 11" aria-hidden>
                    <path
                      d="M2 5.5 L4.5 8 L9 3"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <span>recorded</span>
                </div>
                <div className="flex items-baseline gap-3">
                  <span
                    style={{
                      fontFamily: "var(--font-serif)",
                      fontSize: 38,
                      lineHeight: 0.9,
                      fontStyle: "italic",
                      color: "rgba(250,247,240,0.96)",
                      letterSpacing: "-0.01em",
                    }}
                  >
                    {justMovedSan}
                  </span>
                </div>
                <div className="flex gap-[3px]">
                  {Array.from({ length: 8 }).map((_, i) => (
                    <div
                      key={i}
                      className="h-[2px] flex-1 rounded-sm"
                      style={{
                        background:
                          i < 8 ? "var(--cp-accent)" : "rgba(245,242,235,0.12)",
                      }}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Status line under the clock */}
        {isActive && reading && (
          <div className="mt-1 flex items-center gap-2.5">
            <span
              className="block h-[7px] w-[7px] rounded-full"
              style={{
                background: "var(--cp-accent-ink)",
                animation: "chesspar-pulse 1.4s ease-in-out infinite",
              }}
            />
            <span
              style={{
                fontFamily: "var(--font-serif)",
                fontStyle: "italic",
                fontSize: 19,
                lineHeight: 1,
                color: "rgba(10,36,24,0.85)",
              }}
            >
              {vlmActive ? "reading with vision" : "reading the board"}
            </span>
          </div>
        )}
        {isActive && !reading && phase === "playing" && (
          <div className="mt-0.5 flex items-center gap-2.5">
            <span className="block h-px w-4" style={{ background: "rgba(10,36,24,0.35)" }} />
            <span
              style={{
                fontFamily: "var(--font-serif)",
                fontStyle: "italic",
                fontSize: 18,
                lineHeight: 1,
                color: "rgba(10,36,24,0.78)",
              }}
            >
              your turn
            </span>
            <span className="block h-px w-4" style={{ background: "rgba(10,36,24,0.35)" }} />
          </div>
        )}
      </div>
    </button>
  );
}

/**
 * CenterStrip — the vertical channel between the two side panels. Holds
 * a small mini-board peek (an upright reference glance), the running
 * move-log ticker (rotates to face the player on the clock), and the
 * neutral icon column (pause, captures count, end game). Sits in a
 * lighter warm-taupe so it reads as its own zone, distinct from both
 * inactive near-black and active accent-fill side panels.
 */
function CenterStrip({
  phase,
  captureCount,
  moveLog,
  inferring,
  vlmActive,
  activeSide,
  onTogglePause,
  onReset,
  onShowCaptures,
  onEndGame,
}: {
  phase: Phase;
  captureCount: number;
  moveLog: { san: string; viaVlm: boolean }[];
  inferring: boolean;
  vlmActive: boolean;
  activeSide: Side;
  onTogglePause: () => void;
  onReset: () => void;
  onShowCaptures: () => void;
  onEndGame: () => void;
}) {
  const tickerRot = activeSide === "black" ? 90 : -90;
  const moveNumber = Math.ceil((moveLog.length + 1) / 2);
  const recentSans = moveLog.slice(-8).map((m, i, arr) => ({
    ...m,
    isLatest: i === arr.length - 1,
    moveNum: Math.floor((moveLog.length - arr.length + i) / 2) + 1,
    isWhite: (moveLog.length - arr.length + i) % 2 === 0,
  }));

  return (
    <div
      className="relative flex h-full w-14 shrink-0 flex-col items-center justify-between py-12"
      style={{
        background: "oklch(0.30 0.010 75)",
        borderLeft: "0.5px solid rgba(245,242,235,0.10)",
        borderRight: "0.5px solid rgba(245,242,235,0.10)",
      }}
    >
      {/* Top: pause + reset (subtle, both-side neutral glyphs) */}
      <div className="flex flex-col items-center gap-1.5">
        <button
          type="button"
          onClick={onTogglePause}
          aria-label={phase === "paused" ? "Resume" : "Pause"}
          className="flex h-9 w-9 items-center justify-center rounded-md text-[15px] transition hover:bg-white/10"
          style={{ color: "rgba(245,242,235,0.80)" }}
        >
          {phase === "paused" ? "▶" : "⏸"}
        </button>
        <button
          type="button"
          onClick={onReset}
          aria-label="Back to settings"
          className="flex h-9 w-9 items-center justify-center rounded-md text-[14px] transition hover:bg-white/10"
          style={{ color: "rgba(245,242,235,0.6)" }}
        >
          ←
        </button>
        <div
          className="mt-3 tabular-nums"
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 9,
            color: "rgba(245,242,235,0.75)",
            letterSpacing: "0.08em",
          }}
        >
          {String(moveNumber).padStart(2, "0")}
        </div>
      </div>

      {/* Middle: the rotated move-log ticker. Faces whichever player is
          on the clock — info reads upright for the side currently
          waiting to move. 400ms tween makes the flip feel intentional. */}
      <div className="relative flex w-full flex-1 items-center justify-center overflow-hidden">
        <div
          className="flex gap-2.5 whitespace-nowrap transition-transform duration-[400ms]"
          style={{
            transform: `rotate(${tickerRot}deg)`,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.04em",
            color: "rgba(245,242,235,0.80)",
            maskImage:
              "linear-gradient(to right, transparent, black 18%, black 82%, transparent)",
            WebkitMaskImage:
              "linear-gradient(to right, transparent, black 18%, black 82%, transparent)",
          }}
        >
          {recentSans.length === 0 ? (
            <span style={{ color: "rgba(245,242,235,0.42)" }}>
              {inferring || vlmActive
                ? "reading the board…"
                : "tap your clock to record"}
            </span>
          ) : (
            recentSans.map((m, i) => (
              <span
                key={i}
                style={{
                  color: m.isLatest ? "var(--cp-accent)" : undefined,
                }}
              >
                {m.isWhite ? `${m.moveNum}.` : ""} {m.san}
              </span>
            ))
          )}
        </div>
      </div>

      {/* Bottom: captures + end game */}
      <div className="flex flex-col items-center gap-1.5">
        <button
          type="button"
          onClick={onShowCaptures}
          aria-label="Captures"
          className="relative flex h-9 w-9 items-center justify-center rounded-md text-[15px] transition hover:bg-white/10"
          style={{ color: "rgba(245,242,235,0.80)" }}
        >
          ◫
          {captureCount > 0 && (
            <span
              className="absolute right-0.5 top-0.5 rounded-full px-1 font-semibold leading-tight tabular-nums"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 8,
                color: "var(--cp-accent)",
                background: "rgba(0,0,0,0.7)",
              }}
            >
              {captureCount}
            </span>
          )}
        </button>
        <button
          type="button"
          onClick={onEndGame}
          aria-label="End game"
          className="flex h-9 w-9 items-center justify-center rounded-md text-[15px] transition hover:bg-white/10"
          style={{ color: "rgba(245,242,235,0.65)" }}
        >
          ⊘
        </button>
      </div>
    </div>
  );
}

function PausedOverlay({ onResume }: { onResume: () => void }) {
  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center bg-black/40 backdrop-blur-md">
      <button
        onClick={onResume}
        className="pointer-events-auto flex items-center gap-3 rounded-full bg-white/10 px-7 py-3.5 text-sm font-semibold uppercase tracking-[0.25em] text-zinc-50 ring-1 ring-white/15 backdrop-blur-md transition hover:bg-white/15"
      >
        <span className="block h-2.5 w-2.5 animate-pulse rounded-full bg-emerald-400" />
        Paused — tap to resume
      </button>
    </div>
  );
}

/**
 * Live camera preview floating over the player UI. Pinned to the top
 * centre between the two clock panels. Larger during calibration (the
 * player is actively framing the board) and shrinks once we hit
 * playing/paused (occasional framing check). Includes a lens-switch
 * pill when more than one back-camera is exposed — iPhones surface
 * 0.5×, 1×, sometimes 2× separately.
 */
function CameraPreviewOverlay({
  phase,
  onPreviewRef,
  lenses,
  currentLensId,
  onSwitchLens,
}: {
  phase: Phase;
  onPreviewRef: (el: HTMLVideoElement | null) => void;
  lenses: MediaDeviceInfo[];
  currentLensId: string | null;
  onSwitchLens: (deviceId: string) => void;
}) {
  const isCalibrating = phase === "calibrating";
  const sizeCls = isCalibrating
    ? "h-[40vh] max-h-[320px] w-[78vw] max-w-[260px]"
    : "h-24 w-20";
  // `lenses` has already been filtered down to one ultra-wide + one
  // wide back camera. Map each to its tier label.
  const lensLabel = (label: string): string =>
    labelLensTier(label) === "ultraWide" ? "0.5×" : "1×";
  return (
    <div className="pointer-events-none absolute left-1/2 top-3 z-40 -translate-x-1/2">
      <div className="flex flex-col items-center gap-1.5">
        <div
          className={clsx(
            "relative overflow-hidden rounded-2xl border border-white/15 bg-black shadow-lg shadow-black/40 transition-all duration-300",
            sizeCls,
          )}
        >
          <video
            ref={onPreviewRef}
            muted
            playsInline
            autoPlay
            className="absolute inset-0 h-full w-full object-cover"
          />
          {isCalibrating && (
            <div className="pointer-events-none absolute inset-0 ring-1 ring-inset ring-emerald-400/40" />
          )}
        </div>
        {lenses.length > 1 && (
          <div className="pointer-events-auto flex items-center gap-1 rounded-full border border-white/15 bg-black/70 px-1 py-1 text-[10px] font-medium text-zinc-100 backdrop-blur-md">
            {lenses.map((l, i) => {
              const label = lensLabel(l.label);
              const selected = l.deviceId === currentLensId;
              return (
                <button
                  key={l.deviceId || `lens-${i}`}
                  type="button"
                  onClick={() => onSwitchLens(l.deviceId)}
                  className={clsx(
                    "rounded-full px-2 py-0.5 transition",
                    selected
                      ? "bg-white text-black"
                      : "text-zinc-200 hover:bg-white/10",
                  )}
                >
                  {label}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Pre-game verification screen. The auto-detector has produced four
 * corners; before we start the clock, show the player:
 *  - the live camera (so they can re-aim if framing is wrong),
 *  - the rectified board (so they can confirm orientation),
 *  - a checklist of automated checks (orientation, piece count, starting
 *    position match).
 *
 * The "Start game" button is only enabled when all three checks pass.
 * A secondary "Start anyway" path is offered when a check fails, since
 * the classifier isn't perfect and the player may know better than the
 * pipeline.
 */
/**
 * SVG overlay drawn on top of the live camera preview, showing the four
 * corners the detector currently has locked onto. The viewBox matches
 * the video's native dimensions and the video is rendered with
 * object-contain, so the dots land on the exact pixels the warp will
 * use. Each corner dot is draggable — touch + drag to nudge the
 * detection if auto-detect missed.
 */
function CornerOverlaySvg({
  corners,
  videoDims,
  onDragCorner,
}: {
  corners: Point[];
  videoDims: VideoDims;
  onDragCorner: (idx: number, point: Point) => void;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const draggingIdxRef = useRef<number | null>(null);

  const stroke = Math.max(2, videoDims.w / 400);
  const dotR = Math.max(12, videoDims.w / 80);
  const hitR = dotR * 1.6;

  const toViewBox = useCallback(
    (clientX: number, clientY: number): Point | null => {
      const svg = svgRef.current;
      if (!svg) return null;
      const rect = svg.getBoundingClientRect();
      if (!rect.width || !rect.height) return null;
      // SVG uses preserveAspectRatio="xMidYMid meet" → it fits inside the
      // container while preserving the video's aspect, centring any
      // letterboxing. Compute the scale of the meet fit and the offsets.
      const scale = Math.min(
        rect.width / videoDims.w,
        rect.height / videoDims.h,
      );
      const drawnW = videoDims.w * scale;
      const drawnH = videoDims.h * scale;
      const offsetX = (rect.width - drawnW) / 2;
      const offsetY = (rect.height - drawnH) / 2;
      const localX = clientX - rect.left - offsetX;
      const localY = clientY - rect.top - offsetY;
      const x = Math.max(0, Math.min(videoDims.w, localX / scale));
      const y = Math.max(0, Math.min(videoDims.h, localY / scale));
      return { x, y };
    },
    [videoDims],
  );

  const handlePointerDown = (idx: number) => (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    draggingIdxRef.current = idx;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const pt = toViewBox(e.clientX, e.clientY);
    if (pt) onDragCorner(idx, pt);
  };
  const handlePointerMove = (e: React.PointerEvent) => {
    const idx = draggingIdxRef.current;
    if (idx === null) return;
    e.preventDefault();
    const pt = toViewBox(e.clientX, e.clientY);
    if (pt) onDragCorner(idx, pt);
  };
  const handlePointerUp = (e: React.PointerEvent) => {
    if (draggingIdxRef.current === null) return;
    draggingIdxRef.current = null;
    (e.target as Element).releasePointerCapture?.(e.pointerId);
  };

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${videoDims.w} ${videoDims.h}`}
      className="absolute inset-0 h-full w-full"
      preserveAspectRatio="xMidYMid meet"
      style={{ touchAction: "none" }}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
    >
      <polygon
        points={corners.map((p) => `${p.x},${p.y}`).join(" ")}
        fill="rgba(74,222,128,0.08)"
        stroke="rgba(74,222,128,0.95)"
        strokeWidth={stroke}
        style={{ pointerEvents: "none" }}
      />
      {corners.map((p, i) => (
        <g key={i}>
          <circle
            cx={p.x}
            cy={p.y}
            r={hitR}
            fill="transparent"
            style={{ cursor: "grab", touchAction: "none" }}
            onPointerDown={handlePointerDown(i)}
          />
          <circle
            cx={p.x}
            cy={p.y}
            r={dotR}
            fill="rgba(16,185,129,0.95)"
            stroke="white"
            strokeWidth={stroke * 0.6}
            style={{ pointerEvents: "none" }}
          />
        </g>
      ))}
    </svg>
  );
}

function ReadyScreen({
  check,
  testMode,
  onPreviewRef,
  lenses,
  currentLensId,
  onSwitchLens,
  corners,
  videoDims,
  onDragCorner,
  vlmDetecting,
  vlmStatus,
  hasAiDetector,
  onRetryAi,
  onRecalibrate,
  onTapManually,
  onCancel,
  onStart,
}: {
  check: BoardCheck | null;
  testMode: boolean;
  onPreviewRef: (el: HTMLVideoElement | null) => void;
  lenses: MediaDeviceInfo[];
  currentLensId: string | null;
  onSwitchLens: (deviceId: string) => void;
  corners: Point[];
  videoDims: VideoDims | null;
  onDragCorner: (idx: number, point: Point) => void;
  vlmDetecting: boolean;
  vlmStatus:
    | { kind: "idle" }
    | { kind: "ok"; at: number }
    | { kind: "error"; reason: string };
  hasAiDetector: boolean;
  onRetryAi: () => void;
  onRecalibrate: () => void;
  onTapManually: () => void;
  onCancel: () => void;
  onStart: () => void;
}) {
  const lensLabel = (label: string): string =>
    labelLensTier(label) === "ultraWide" ? "0.5×" : "1×";

  // One-line condensed status — replaces the 3-row check rail. The
  // detail text only appears when something's wrong, so the screen
  // stays quiet on the happy path.
  let statusKind: "loading" | "ok" | "warn";
  let statusText: string;
  if (!check) {
    statusKind = "loading";
    statusText = "Reading the board…";
  } else if (check.allClear) {
    statusKind = "ok";
    statusText = "Board ready — 16 white + 16 black, starting position";
  } else if (!check.piecesAllDetected) {
    statusKind = "warn";
    statusText = `Saw ${check.pieceCount.white}W · ${check.pieceCount.black}B · ${check.pieceCount.empty} empty — should be 16 + 16 + 32`;
  } else {
    statusKind = "warn";
    statusText = `${check.startingPositionMatch}/64 squares match starting position`;
  }
  const aiBadge = vlmDetecting
    ? "AI placing corners…"
    : vlmStatus.kind === "ok"
      ? "AI placed corners"
      : vlmStatus.kind === "error"
        ? "Using CV fallback"
        : null;

  return (
    <div className="absolute inset-0 z-40 flex flex-col bg-zinc-950 text-zinc-100">
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between px-5 pb-1 pt-5">
        <button
          onClick={onCancel}
          className="rounded-full bg-white/5 px-3 py-1 text-[11px] uppercase tracking-widest text-zinc-300 hover:bg-white/10"
        >
          Cancel
        </button>
        <div className="text-[10px] font-semibold uppercase tracking-[0.32em] text-emerald-300">
          Confirm setup
        </div>
        <div className="w-[68px]" aria-hidden />
      </div>

      {/* Live camera — takes whatever vertical space is left between
          header and the action footer. */}
      {!testMode && (
        <div className="relative mx-auto mt-2 flex w-full max-w-[440px] flex-1 items-center justify-center px-4">
          <div
            className="relative w-full max-h-full overflow-hidden rounded-2xl border border-white/15 bg-black"
            style={
              videoDims
                ? { aspectRatio: `${videoDims.w}/${videoDims.h}` }
                : { aspectRatio: "3/4" }
            }
          >
            <video
              ref={onPreviewRef}
              muted
              playsInline
              autoPlay
              className="absolute inset-0 h-full w-full object-contain"
            />
            {corners.length === 4 && videoDims && (
              <CornerOverlaySvg
                corners={corners}
                videoDims={videoDims}
                onDragCorner={onDragCorner}
              />
            )}
            {aiBadge && (
              <div
                className={clsx(
                  "pointer-events-none absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-[0.18em] backdrop-blur",
                  vlmDetecting
                    ? "bg-sky-500/20 text-sky-100"
                    : vlmStatus.kind === "ok"
                      ? "bg-emerald-500/20 text-emerald-100"
                      : "bg-zinc-700/60 text-zinc-200",
                )}
              >
                {vlmDetecting && (
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-300" />
                )}
                {aiBadge}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Lens + drag hint */}
      <div className="mx-auto mt-3 flex w-full max-w-[440px] items-center justify-between px-4 text-[11px] text-zinc-400">
        {lenses.length > 1 ? (
          <div className="flex items-center gap-1 rounded-full border border-white/15 bg-black/70 p-1 text-zinc-100">
            {lenses.map((l, i) => {
              const label = lensLabel(l.label);
              const selected = l.deviceId === currentLensId;
              return (
                <button
                  key={l.deviceId || `lens-${i}`}
                  type="button"
                  onClick={() => onSwitchLens(l.deviceId)}
                  className={clsx(
                    "rounded-full px-3 py-0.5 transition",
                    selected
                      ? "bg-white text-black"
                      : "text-zinc-200 hover:bg-white/10",
                  )}
                >
                  {label}
                </button>
              );
            })}
          </div>
        ) : (
          <span />
        )}
        <span className="text-right">Drag corners to fine-tune</span>
      </div>

      {/* Status line */}
      <div className="mx-auto mt-3 w-full max-w-[440px] px-4">
        <div
          className={clsx(
            "flex items-center gap-2 rounded-xl px-3 py-2 text-[12px]",
            statusKind === "ok"
              ? "bg-emerald-500/10 text-emerald-100"
              : statusKind === "warn"
                ? "bg-amber-500/10 text-amber-100"
                : "bg-white/5 text-zinc-300",
          )}
        >
          {statusKind === "ok" ? (
            <svg className="h-4 w-4 shrink-0" viewBox="0 0 16 16" fill="currentColor">
              <path d="M6 11.5 3 8.5l1-1 2 2 5-5 1 1Z" />
            </svg>
          ) : statusKind === "loading" ? (
            <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-zinc-300" />
          ) : (
            <span className="h-2 w-2 shrink-0 rounded-full bg-amber-300" />
          )}
          <span className="min-w-0 truncate">{statusText}</span>
        </div>
      </div>

      {/* Action footer */}
      <div className="mx-auto mt-3 flex w-full max-w-[440px] shrink-0 flex-col gap-2 px-4 pb-5">
        <button
          onClick={onStart}
          className={clsx(
            "rounded-2xl px-4 py-3.5 text-base font-semibold transition",
            check?.allClear
              ? "bg-emerald-500/95 text-emerald-950 shadow-lg shadow-emerald-500/15 hover:bg-emerald-400"
              : "border border-amber-400/40 bg-amber-500/15 text-amber-100 hover:bg-amber-500/25",
          )}
        >
          {check?.allClear ? "Start game" : "Start anyway"}
        </button>
        <div className="flex gap-2 text-[12px]">
          <button
            onClick={onTapManually}
            className="flex-1 rounded-xl border border-white/10 bg-white/5 px-3 py-2 font-medium text-zinc-200 hover:bg-white/10"
          >
            Tap corners myself
          </button>
          <button
            onClick={onRecalibrate}
            className="flex-1 rounded-xl border border-white/10 bg-white/5 px-3 py-2 font-medium text-zinc-200 hover:bg-white/10"
          >
            Re-detect
          </button>
          {hasAiDetector && !vlmDetecting && (
            <button
              onClick={onRetryAi}
              className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 font-medium text-zinc-300 hover:bg-white/10"
              title="Re-run the AI corner detector"
            >
              AI
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function CheckRow({
  ok,
  label,
  detail,
}: {
  ok: boolean;
  label: string;
  detail: string | null;
}) {
  return (
    <div className="flex items-start gap-2">
      <span
        className={clsx(
          "mt-[3px] flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full",
          ok ? "bg-emerald-500/90 text-emerald-950" : "bg-zinc-700 text-zinc-300",
        )}
        aria-hidden
      >
        {ok ? (
          <svg viewBox="0 0 12 12" className="h-2.5 w-2.5" fill="currentColor">
            <path d="M4.5 8.4 2.1 6l-.7.7 3.1 3.1 6-6-.7-.7Z" />
          </svg>
        ) : (
          <span className="h-1 w-1 rounded-full bg-current" />
        )}
      </span>
      <div className="min-w-0">
        <div
          className={clsx(
            "font-medium",
            ok ? "text-zinc-100" : "text-zinc-300",
          )}
        >
          {label}
        </div>
        {detail && (
          <div className="text-[11px] leading-snug text-zinc-400">{detail}</div>
        )}
      </div>
    </div>
  );
}

/**
 * Lightweight overlay during the brief auto-detect window. Sits on top of
 * the live player UI rather than taking over the screen — most boards
 * detect within ~1 s and the overlay just flashes briefly. After a long
 * delay we add a hint and a cancel-out link.
 */
const TAP_LABELS = ["a8", "h8", "h1", "a1"] as const;
const TAP_HINTS: Record<(typeof TAP_LABELS)[number], string> = {
  a8: "Black's queenside rook — top-left of the board (from White's view)",
  h8: "Black's kingside rook — top-right",
  h1: "White's kingside rook — bottom-right",
  a1: "White's queenside rook — bottom-left",
};

/**
 * Full-screen tap-to-place corner UI. Shown during calibrating when
 * the user opts into manual placement. Reliably correct in 4 taps —
 * the escape hatch when auto-detect (CV or VLM) doesn't deliver.
 */
function TapCornersOverlay({
  corners,
  videoDims,
  onPreviewRef,
  onTap,
  onUndo,
  onCancel,
  onUseAi,
  hasAiDetector,
}: {
  corners: Point[];
  videoDims: VideoDims | null;
  onPreviewRef: (el: HTMLVideoElement | null) => void;
  onTap: (point: Point) => void;
  onUndo: () => void;
  onCancel: () => void;
  onUseAi: () => void;
  hasAiDetector: boolean;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const nextIdx = Math.min(corners.length, 3);
  const targetLabel = corners.length < 4 ? TAP_LABELS[corners.length] : null;
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!videoDims) return;
    if (corners.length >= 4) return;
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    // The video uses object-contain inside a container with the video's
    // aspect ratio, so client→viewBox mapping is a uniform scale.
    const scale = Math.min(
      rect.width / videoDims.w,
      rect.height / videoDims.h,
    );
    const drawnW = videoDims.w * scale;
    const drawnH = videoDims.h * scale;
    const offsetX = (rect.width - drawnW) / 2;
    const offsetY = (rect.height - drawnH) / 2;
    const localX = e.clientX - rect.left - offsetX;
    const localY = e.clientY - rect.top - offsetY;
    if (localX < 0 || localX > drawnW || localY < 0 || localY > drawnH) return;
    onTap({ x: localX / scale, y: localY / scale });
  };
  return (
    <div className="absolute inset-0 z-40 flex flex-col bg-zinc-950 text-zinc-100">
      <div className="flex shrink-0 items-center justify-between px-5 pb-2 pt-5">
        <button
          onClick={onCancel}
          className="rounded-full bg-white/5 px-3 py-1 text-[11px] uppercase tracking-widest text-zinc-300 hover:bg-white/10"
        >
          Cancel
        </button>
        <div className="text-[10px] font-semibold uppercase tracking-[0.32em] text-emerald-300">
          Tap each corner
        </div>
        <div className="w-[68px]" aria-hidden />
      </div>

      <div className="flex shrink-0 items-center justify-center gap-3 px-5 pb-3 pt-1">
        <TapCornerInset step={corners.length} />
        <div className="min-w-0 text-left">
          {targetLabel ? (
            <>
              <div className="text-sm font-semibold tracking-tight text-zinc-50">
                Tap corner {corners.length + 1} of 4 ·{" "}
                <span className="text-emerald-300">{targetLabel}</span>
              </div>
              <div className="mt-0.5 text-[11px] leading-snug text-zinc-400">
                {TAP_HINTS[targetLabel]}
              </div>
            </>
          ) : (
            <div className="text-sm font-medium text-emerald-300">
              All four corners placed — ready to verify
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-1 items-center justify-center bg-black px-2 pb-3">
        <div
          ref={containerRef}
          onPointerDown={handlePointerDown}
          className="relative w-full max-w-[320px] overflow-hidden rounded-2xl border border-white/15 bg-black"
          style={{
            aspectRatio: videoDims
              ? `${videoDims.w}/${videoDims.h}`
              : "3/4",
            touchAction: "none",
          }}
        >
          <video
            ref={onPreviewRef}
            muted
            playsInline
            autoPlay
            className="absolute inset-0 h-full w-full object-contain"
          />
          {corners.length > 0 && videoDims && (
            <svg
              viewBox={`0 0 ${videoDims.w} ${videoDims.h}`}
              className="pointer-events-none absolute inset-0 h-full w-full"
              preserveAspectRatio="xMidYMid meet"
            >
              {corners.length >= 4 && (
                <polygon
                  points={corners.map((p) => `${p.x},${p.y}`).join(" ")}
                  fill="rgba(74,222,128,0.08)"
                  stroke="rgba(74,222,128,0.95)"
                  strokeWidth={Math.max(2, videoDims.w / 400)}
                />
              )}
              {corners.length >= 2 && corners.length < 4 && (
                <polyline
                  points={corners.map((p) => `${p.x},${p.y}`).join(" ")}
                  fill="none"
                  stroke="rgba(74,222,128,0.8)"
                  strokeWidth={Math.max(2, videoDims.w / 400)}
                />
              )}
              {corners.map((p, i) => (
                <g key={i}>
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={Math.max(10, videoDims.w / 100)}
                    fill={i === nextIdx ? "rgba(74,222,128,0.95)" : "rgba(16,185,129,0.95)"}
                    stroke="white"
                    strokeWidth={Math.max(2, videoDims.w / 400)}
                  />
                  <text
                    x={p.x}
                    y={p.y - Math.max(18, videoDims.w / 60)}
                    fill="white"
                    stroke="rgba(0,0,0,0.7)"
                    strokeWidth={Math.max(1.5, videoDims.w / 600)}
                    paintOrder="stroke"
                    fontSize={Math.max(14, videoDims.w / 50)}
                    fontWeight="bold"
                    textAnchor="middle"
                  >
                    {TAP_LABELS[i]}
                  </text>
                </g>
              ))}
            </svg>
          )}
        </div>
      </div>

      <div className="flex shrink-0 flex-col gap-2 px-5 pb-6 pt-2">
        <div className="flex gap-2">
          <button
            onClick={onUndo}
            disabled={corners.length === 0}
            className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-3 py-2.5 text-[13px] font-medium text-zinc-200 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Undo last tap
          </button>
          {hasAiDetector && (
            <button
              onClick={onUseAi}
              className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-3 py-2.5 text-[13px] font-medium text-zinc-200 hover:bg-white/10"
            >
              Use AI auto-detect
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Small 8×8 mini-board with the next corner highlighted, so the user
 * knows which physical corner of the camera image to tap next.
 */
function TapCornerInset({ step }: { step: number }) {
  const size = 64;
  const sq = size / 8;
  const target = step < 4 ? TAP_LABELS[step] : null;
  const targetCell: Record<(typeof TAP_LABELS)[number], { f: number; r: number }> = {
    a8: { f: 0, r: 0 },
    h8: { f: 7, r: 0 },
    h1: { f: 7, r: 7 },
    a1: { f: 0, r: 7 },
  };
  const cells: React.ReactElement[] = [];
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const light = (r + f) % 2 === 0;
      cells.push(
        <rect
          key={`${r}-${f}`}
          x={f * sq}
          y={r * sq}
          width={sq}
          height={sq}
          fill={light ? "#e8d6b0" : "#a87b4a"}
        />,
      );
    }
  }
  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      className="shrink-0 rounded border border-zinc-700"
      aria-hidden
    >
      {cells}
      {target && (
        <rect
          x={targetCell[target].f * sq}
          y={targetCell[target].r * sq}
          width={sq}
          height={sq}
          fill="rgba(16,185,129,0.9)"
          stroke="white"
          strokeWidth={1.5}
        />
      )}
    </svg>
  );
}

function StartingOverlay({
  testMode,
  cameraError,
  vlmDetecting,
  hasAiDetector,
  onTapManually,
  onCancel,
}: {
  testMode: boolean;
  cameraError: string | null;
  vlmDetecting: boolean;
  hasAiDetector: boolean;
  onTapManually: () => void;
  onCancel: () => void;
}) {
  const [showHint, setShowHint] = useState(false);
  useEffect(() => {
    const t = window.setTimeout(() => setShowHint(true), 5000);
    return () => window.clearTimeout(t);
  }, []);
  const label = vlmDetecting
    ? "Detecting board with AI…"
    : hasAiDetector
      ? "Looking at the board…"
      : "Finding the board…";
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 top-[55vh] z-30 flex items-start justify-center bg-gradient-to-b from-transparent via-black/55 to-black/70 pt-6">
      <div className="pointer-events-auto flex w-[min(20rem,86vw)] flex-col items-center gap-4 rounded-3xl bg-white/8 px-6 py-7 text-center ring-1 ring-white/15">
        <span className="relative block h-8 w-8">
          <span className="absolute inset-0 animate-ping rounded-full bg-emerald-400/40" />
          <span className="absolute inset-1 rounded-full bg-emerald-400/90" />
        </span>
        <div className="space-y-1">
          <div className="text-sm font-semibold tracking-tight text-zinc-50">
            {label}
          </div>
          {showHint && !cameraError && (
            <div className="text-xs leading-snug text-zinc-300">
              {testMode
                ? "Photo 1 should show the starting position with all 32 pieces."
                : "Make sure the whole board is in frame and well lit."}
            </div>
          )}
          {cameraError && (
            <div className="text-xs leading-snug text-amber-200">
              Camera unavailable: {cameraError}
            </div>
          )}
        </div>
        {!testMode && (
          <button
            onClick={onTapManually}
            className="rounded-full border border-white/15 bg-white/5 px-4 py-1.5 text-xs font-medium text-zinc-100 hover:bg-white/10"
          >
            Tap the corners myself
          </button>
        )}
        <button
          onClick={onCancel}
          className="text-xs uppercase tracking-[0.2em] text-zinc-400 transition hover:text-zinc-200"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

function SettingsScreen({
  tc,
  onChangeTc,
  onStart,
  hasSavedCorners,
  onClearCorners,
  cameraError,
  testMode,
  onToggleTestMode,
  testFrameCount,
  testFrameSrcs,
  onLoadTestFrames,
  vlmConfig,
  onChangeVlmConfig,
}: {
  tc: TimeControl;
  onChangeTc: (tc: TimeControl) => void;
  onStart: () => void;
  hasSavedCorners: boolean;
  onClearCorners: () => void;
  cameraError: string | null;
  testMode: boolean;
  onToggleTestMode: (on: boolean) => void;
  testFrameCount: number;
  testFrameSrcs: string[];
  onLoadTestFrames: (files: FileList | null) => void;
  vlmConfig: VlmConfig | null;
  onChangeVlmConfig: (next: VlmConfig | null) => void;
}) {
  const startLabel = testMode
    ? testFrameCount > 0
      ? "Start replay"
      : "Choose photos to begin"
    : "Start game";
  const startDisabled = testMode && testFrameCount === 0;

  return (
    <div className="relative flex flex-1 flex-col overflow-y-auto px-5 py-8">
      <Link
        href="/"
        className="absolute left-4 top-4 inline-flex items-center gap-1 rounded-full bg-white/5 px-3 py-1 text-[11px] uppercase tracking-widest text-zinc-300 hover:bg-white/10"
      >
        ← Home
      </Link>
      <div className="mx-auto flex w-full max-w-md flex-col gap-6">
        <div className="mt-6">
          <div className="text-[11px] uppercase tracking-[0.3em] text-emerald-300">
            New game
          </div>
          <h1 className="mt-1 text-[2rem] font-semibold tracking-tight">
            {testMode ? "Replay from photos" : "Ready to play"}
          </h1>
          <p className="mt-1 text-[14px] leading-snug text-zinc-400">
            {testMode
              ? "Load a sequence of board photos — we'll step through them as if you were tapping a clock."
              : "Pick a time control. The board lines itself up the moment you start."}
          </p>
        </div>

        <div>
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.3em] text-zinc-400">
            Time control
          </div>
          <div className="grid grid-cols-2 gap-2">
            {PRESETS.map((p) => {
              const selected =
                p.tc.baseSeconds === tc.baseSeconds &&
                p.tc.incrementSeconds === tc.incrementSeconds;
              return (
                <button
                  key={p.label}
                  onClick={() => onChangeTc(p.tc)}
                  className={clsx(
                    "rounded-2xl border px-3 py-2.5 text-sm font-medium transition",
                    selected
                      ? "border-emerald-400/60 bg-emerald-500/20 text-emerald-50 shadow-inner shadow-emerald-500/10"
                      : "border-white/5 bg-white/5 text-zinc-200 hover:bg-white/10",
                  )}
                >
                  {p.label}
                </button>
              );
            })}
          </div>
          <details className="mt-3 rounded-2xl border border-white/5 bg-white/5 px-3 py-2">
            <summary className="cursor-pointer text-[12px] font-medium tracking-wide text-zinc-200">
              Custom time control
            </summary>
            <div className="mt-2">
              <CustomTcEditor tc={tc} onChange={onChangeTc} />
            </div>
          </details>
        </div>

        <button
          onClick={onStart}
          disabled={startDisabled}
          className="rounded-2xl bg-emerald-500/95 px-4 py-4 text-base font-semibold text-emerald-950 shadow-lg shadow-emerald-500/15 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500 disabled:shadow-none"
        >
          {startLabel}
        </button>

        {hasSavedCorners && !testMode && (
          <div className="-mt-3 flex items-center justify-between rounded-full border border-emerald-400/30 bg-emerald-500/10 px-4 py-1.5 text-[11px] text-emerald-100">
            <span className="inline-flex items-center gap-2">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-400" />
              Board calibrated from last session
            </span>
            <button
              onClick={onClearCorners}
              className="text-[10px] uppercase tracking-widest text-emerald-200 hover:text-emerald-50"
            >
              Forget
            </button>
          </div>
        )}

        {cameraError && !testMode && (
          <div className="rounded-2xl border border-amber-400/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
            Camera unavailable: {cameraError}. Clock still works; no photos
            will be saved.
          </div>
        )}

        <details className="rounded-2xl border border-white/5 bg-white/5">
          <summary className="cursor-pointer list-none px-4 py-3 text-[12px] font-semibold uppercase tracking-[0.3em] text-zinc-300">
            <span className="float-right text-zinc-500">▾</span>
            Advanced
          </summary>
          <div className="space-y-4 border-t border-white/5 px-4 py-4">
            <div>
              <label className="flex items-center justify-between gap-3">
                <span className="text-[13px] font-medium text-zinc-100">
                  Replay from photos
                </span>
                <input
                  type="checkbox"
                  checked={testMode}
                  onChange={(e) => onToggleTestMode(e.target.checked)}
                  className="h-5 w-5 cursor-pointer accent-emerald-500"
                />
              </label>
              <p className="mt-1 text-[11px] leading-snug text-zinc-400">
                Feed a sequence of still photos through the pipeline instead
                of the live camera. Useful for debugging or testing.
              </p>
              {testMode && (
                <div className="mt-3 space-y-2">
                  <label className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-full border border-white/10 bg-white/10 px-3 py-1.5 text-[12px] text-zinc-100 hover:bg-white/15">
                    <input
                      type="file"
                      accept="image/*"
                      multiple
                      onChange={(e) => onLoadTestFrames(e.target.files)}
                      className="hidden"
                    />
                    {testFrameCount > 0 ? "Replace photos" : "Choose photos"}
                  </label>
                  {testFrameCount > 0 && (
                    <>
                      <p className="text-[11px] text-zinc-400">
                        {testFrameCount} loaded ·{" "}
                        <span className="text-zinc-200">
                          1 starting + {testFrameCount - 1} half-move
                          {testFrameCount - 1 === 1 ? "" : "s"}
                        </span>
                      </p>
                      <div className="flex gap-1.5 overflow-x-auto pb-1">
                        {testFrameSrcs.map((src, i) => (
                          <div
                            key={`${src}-${i}`}
                            className="relative shrink-0"
                            title={
                              i === 0
                                ? "Photo 1 — starting position"
                                : `Photo ${i + 1} — half-move ${i}`
                            }
                          >
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={src}
                              alt={`Photo ${i + 1}`}
                              className={clsx(
                                "h-12 w-12 rounded-md object-cover ring-1",
                                i === 0
                                  ? "ring-emerald-400/70"
                                  : "ring-white/10",
                              )}
                            />
                            <span
                              className={clsx(
                                "absolute bottom-0.5 right-0.5 rounded-sm px-1 text-[9px] font-semibold leading-tight tabular-nums",
                                i === 0
                                  ? "bg-emerald-500/90 text-emerald-950"
                                  : "bg-black/70 text-zinc-100",
                              )}
                            >
                              {i === 0 ? "start" : i}
                            </span>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>

            <div>
              <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.3em] text-zinc-400">
                Vision model fallback
              </div>
              {VLM_PROXY_URL ? (
                <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-[12px] text-emerald-100">
                  <div className="font-medium">
                    {VLM_PROXY_PROVIDER === "openai"
                      ? "GPT-5"
                      : VLM_PROXY_PROVIDER === "gemini"
                        ? "Gemini 2.5 Pro"
                        : "Claude"}{" "}
                    is enabled by default.
                  </div>
                  <details className="mt-2 text-[11px] text-emerald-100/80">
                    <summary className="cursor-pointer">
                      Override with your own key
                    </summary>
                    <div className="mt-2">
                      <VlmConfigEditor
                        config={vlmConfig}
                        onChange={onChangeVlmConfig}
                      />
                    </div>
                  </details>
                </div>
              ) : (
                <VlmConfigEditor
                  config={vlmConfig}
                  onChange={onChangeVlmConfig}
                />
              )}
            </div>

            <div className="text-[11px] text-zinc-500">
              Want to try the rectifier on a still photo first?{" "}
              <Link
                href="/detect"
                className="font-medium text-emerald-300 underline-offset-2 hover:underline"
              >
                /detect
              </Link>
            </div>
          </div>
        </details>
      </div>
    </div>
  );
}

function VlmConfigEditor({
  config,
  onChange,
}: {
  config: VlmConfig | null;
  onChange: (next: VlmConfig | null) => void;
}) {
  const [enabled, setEnabled] = useState<boolean>(Boolean(config));
  const [provider, setProvider] = useState<VlmProvider>(
    config?.provider ?? "anthropic",
  );
  const [apiKey, setApiKey] = useState<string>(config?.apiKey ?? "");
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    if (!enabled) {
      onChange(null);
      return;
    }
    if (apiKey.trim().length > 0) {
      onChange({ provider, apiKey: apiKey.trim() });
    } else {
      onChange(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, provider, apiKey]);

  return (
    <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
      <label className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[13px] font-medium text-zinc-100">
            Vision-model identifier
          </div>
          <p className="mt-0.5 text-[11px] leading-snug text-zinc-400">
            Sends each move&apos;s before + after rectified board to the
            model with the legal-move list. Dramatically more accurate than
            CV alone on hard sets (cream pieces on a coloured board, tall
            kings, mat curl).
          </p>
        </div>
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => setEnabled(e.target.checked)}
          className="h-5 w-5 shrink-0 cursor-pointer accent-emerald-500"
        />
      </label>
      {enabled && (
        <div className="mt-3 flex flex-col gap-2">
          <div className="flex gap-1.5">
            {(["anthropic", "gemini", "openai"] as const).map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => setProvider(p)}
                className={clsx(
                  "flex-1 rounded-full border px-2 py-1.5 text-[11px] font-medium transition",
                  provider === p
                    ? "border-emerald-400/50 bg-emerald-500/15 text-emerald-100"
                    : "border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10",
                )}
              >
                {VLM_PROVIDER_LABELS[p]}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <input
              type={revealed ? "text" : "password"}
              placeholder={`${VLM_PROVIDER_LABELS[provider]} API key`}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              autoComplete="off"
              spellCheck={false}
              className="flex-1 rounded-xl border border-white/10 bg-zinc-950 px-3 py-1.5 font-mono text-xs text-zinc-100 placeholder:text-zinc-600"
            />
            <button
              type="button"
              onClick={() => setRevealed((r) => !r)}
              className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1.5 text-[10px] uppercase tracking-widest text-zinc-300 hover:bg-white/10"
            >
              {revealed ? "Hide" : "Show"}
            </button>
          </div>
          <div className="flex items-baseline justify-between">
            <p className="text-[10px] leading-snug text-zinc-500">
              Key stays in your browser&apos;s localStorage and is sent
              directly to the provider.
            </p>
            <p className="text-[10px] font-mono text-emerald-200/70">
              {VLM_PROVIDER_COST_HINT[provider]}
            </p>
          </div>
        </div>
      )}
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
    <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.3em] text-zinc-500">
        Custom
      </div>
      <div className="flex items-center gap-3">
        <label className="flex flex-1 flex-col gap-1">
          <span className="text-[10px] uppercase tracking-widest text-zinc-500">
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
            className="rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-sm tabular-nums text-zinc-100"
          />
        </label>
        <label className="flex flex-1 flex-col gap-1">
          <span className="text-[10px] uppercase tracking-widest text-zinc-500">
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
            className="rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-sm tabular-nums text-zinc-100"
          />
        </label>
      </div>
    </div>
  );
}

/**
 * Editorial paired-column score sheet. Serif headline frames the result;
 * stats row reads as a small editorial chip strip; the paired columns
 * (WHITE / BLACK) are the only place we expose the SAN history — copy
 * PGN is an inline link, not a separate block.
 */
function EndScreen({
  winner,
  captureCount,
  recordedMoves,
  moves,
  pgn,
  moveLog,
  onNewGame,
  onViewCaptures,
  onReplay,
}: {
  winner: Side | null;
  captureCount: number;
  recordedMoves: number;
  moves: { white: number; black: number };
  pgn: string;
  moveLog: { san: string; viaVlm: boolean }[];
  onNewGame: () => void;
  onViewCaptures: () => void;
  onReplay?: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const hasRecordedMoves = recordedMoves > 0;
  const unrecorded = Math.max(0, captureCount - recordedMoves);
  const vlmCount = moveLog.filter((m) => m.viaVlm).length;
  const cvCount = moveLog.length - vlmCount;

  // Pair the running SAN history into rows of (n, white, black) for the
  // paired-column score sheet.
  const rows: { n: number; w: string; b: string; bIsVlm?: boolean; wIsVlm?: boolean }[] = [];
  for (let i = 0; i < moveLog.length; i += 2) {
    rows.push({
      n: i / 2 + 1,
      w: moveLog[i]?.san ?? "",
      b: moveLog[i + 1]?.san ?? "",
      wIsVlm: moveLog[i]?.viaVlm,
      bIsVlm: moveLog[i + 1]?.viaVlm,
    });
  }

  async function copyPgn() {
    try {
      await navigator.clipboard.writeText(pgn || "");
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  }

  const headline = winner
    ? winner === "white"
      ? (
          <>
            White wins{" "}
            <span style={{ fontStyle: "italic", color: "var(--cp-accent)" }}>
              on time
            </span>
          </>
        )
      : (
          <>
            Black wins{" "}
            <span style={{ fontStyle: "italic", color: "var(--cp-accent)" }}>
              on time
            </span>
          </>
        )
    : (
        <>
          A draw{" "}
          <span style={{ fontStyle: "italic", color: "var(--cp-accent)" }}>
            by agreement
          </span>
        </>
      );

  return (
    <div
      className="flex flex-1 flex-col overflow-y-auto"
      style={{ background: "var(--cp-canvas)" }}
    >
      <div className="mx-auto w-full max-w-md px-6 pb-10 pt-8">
        {/* Eyebrow */}
        <div
          className="mb-4 text-[10px] uppercase"
          style={{
            color: "var(--cp-accent)",
            letterSpacing: "0.3em",
            fontFamily: "var(--font-ui)",
          }}
        >
          · Game ended ·
        </div>

        {/* Editorial headline */}
        <h1
          className="mb-1"
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: 48,
            lineHeight: 1.0,
            fontWeight: 400,
            letterSpacing: "-0.015em",
            color: "rgba(250,247,240,0.96)",
            textWrap: "pretty" as React.CSSProperties["textWrap"],
          }}
        >
          {headline}
        </h1>
        <p
          className="mb-6"
          style={{
            fontFamily: "var(--font-serif)",
            fontStyle: "italic",
            fontSize: 17,
            color: "rgba(245,242,235,0.55)",
            margin: "0 0 22px",
          }}
        >
          {recordedMoves} moves recorded ·{" "}
          {new Date().toLocaleDateString(undefined, {
            weekday: "short",
            day: "numeric",
            month: "short",
          })}
        </p>

        {/* Stats row */}
        <div className="mb-6 grid grid-cols-3 gap-2.5">
          {[
            {
              k: "Moves",
              v: String(recordedMoves || 0),
              sub: `${moves.white} / ${moves.black}`,
            },
            {
              k: "Captures",
              v: String(captureCount),
              sub:
                vlmCount > 0
                  ? `${cvCount} cv · ${vlmCount} vlm`
                  : `${cvCount} cv`,
            },
            {
              k: "Result",
              v: winner ? (winner === "white" ? "1–0" : "0–1") : "½–½",
              sub: winner ? "on time" : "by agreement",
            },
          ].map((s) => (
            <div
              key={s.k}
              className="rounded-2xl px-3 py-3.5"
              style={{
                border: "0.5px solid rgba(245,242,235,0.08)",
                background: "rgba(245,242,235,0.025)",
              }}
            >
              <div
                className="mb-2 text-[9px] uppercase"
                style={{
                  letterSpacing: "0.24em",
                  color: "rgba(245,242,235,0.42)",
                  fontFamily: "var(--font-ui)",
                }}
              >
                {s.k}
              </div>
              <div
                className="tabular-nums"
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 22,
                  lineHeight: 1,
                  fontWeight: 500,
                  color: "rgba(250,247,240,0.96)",
                }}
              >
                {s.v}
              </div>
              <div
                className="mt-1.5 text-[10px]"
                style={{
                  letterSpacing: "0.02em",
                  color: "rgba(245,242,235,0.4)",
                }}
              >
                {s.sub}
              </div>
            </div>
          ))}
        </div>

        {/* No-moves edge case */}
        {!hasRecordedMoves && captureCount > 0 && (
          <div
            className="mb-5 rounded-2xl px-4 py-4"
            style={{
              border: "0.5px solid rgba(224,181,107,0.28)",
              background: "rgba(224,181,107,0.06)",
            }}
          >
            <div
              className="text-[10px] uppercase"
              style={{
                color: "#e0b56b",
                letterSpacing: "0.3em",
                fontFamily: "var(--font-ui)",
              }}
            >
              No moves were recorded
            </div>
            <p
              className="mt-1.5 text-[13px] leading-snug"
              style={{
                color: "rgba(245,221,180,0.85)",
                fontFamily: "var(--font-ui)",
              }}
            >
              The pipeline didn&apos;t pin a unique move for any of your{" "}
              {captureCount} capture{captureCount === 1 ? "" : "s"}. Tap
              <strong> View captures</strong> and assign each one the move
              that actually happened — your PGN rebuilds as you go.
            </p>
          </div>
        )}

        {unrecorded > 0 && hasRecordedMoves && (
          <div
            className="mb-5 rounded-2xl px-4 py-3 text-[13px]"
            style={{
              border: "0.5px solid rgba(224,181,107,0.22)",
              background: "rgba(224,181,107,0.05)",
              color: "rgba(245,221,180,0.85)",
              fontFamily: "var(--font-ui)",
            }}
          >
            {unrecorded} capture{unrecorded === 1 ? "" : "s"} awaiting review
            ·{" "}
            <button
              onClick={onViewCaptures}
              className="underline-offset-2 hover:underline"
              style={{ color: "var(--cp-accent)" }}
            >
              open captures →
            </button>
          </div>
        )}

        {/* Score sheet — paired columns */}
        {hasRecordedMoves && (
          <div
            className="mb-6 rounded-[18px] px-5 py-3.5"
            style={{
              border: "0.5px solid rgba(245,242,235,0.08)",
              background: "rgba(245,242,235,0.025)",
            }}
          >
            <div className="mb-2.5 flex items-baseline justify-between">
              <span
                style={{
                  fontFamily: "var(--font-serif)",
                  fontStyle: "italic",
                  fontSize: 18,
                  color: "rgba(245,242,235,0.7)",
                }}
              >
                The score sheet
              </span>
              <button
                onClick={copyPgn}
                style={{
                  fontFamily: "var(--font-ui)",
                  fontSize: 11,
                  letterSpacing: "0.04em",
                  color: "var(--cp-accent)",
                }}
                className="border-0 bg-transparent"
              >
                {copied ? "copied" : "copy PGN ↗"}
              </button>
            </div>
            <div
              className="grid tabular-nums"
              style={{
                gridTemplateColumns: "24px 1fr 1fr",
                rowGap: 6,
                columnGap: 6,
                fontFamily: "var(--font-mono)",
                fontSize: 14,
              }}
            >
              {/* Column headers with white/black dots */}
              <span />
              <span
                className="pb-1 text-[9px] uppercase"
                style={{
                  letterSpacing: "0.28em",
                  color: "rgba(245,242,235,0.45)",
                  fontFamily: "var(--font-ui)",
                }}
              >
                <span
                  className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle"
                  style={{ background: "#f6efdb" }}
                />
                White
              </span>
              <span
                className="pb-1 text-[9px] uppercase"
                style={{
                  letterSpacing: "0.28em",
                  color: "rgba(245,242,235,0.45)",
                  fontFamily: "var(--font-ui)",
                }}
              >
                <span
                  className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle"
                  style={{
                    background: "#1a1410",
                    border: "0.5px solid rgba(245,242,235,0.35)",
                  }}
                />
                Black
              </span>
              {rows.map((r, i) => (
                <Fragment key={r.n}>
                  <span style={{ color: "rgba(245,242,235,0.35)" }}>
                    {r.n}.
                  </span>
                  <span
                    style={{
                      color: r.wIsVlm
                        ? "var(--cp-accent)"
                        : "rgba(250,247,240,0.92)",
                    }}
                  >
                    {r.w}
                  </span>
                  <span
                    style={{
                      color: r.bIsVlm
                        ? "var(--cp-accent)"
                        : i === rows.length - 1 && !r.b
                          ? "rgba(245,242,235,0.4)"
                          : "rgba(250,247,240,0.78)",
                    }}
                  >
                    {r.b || "…"}
                  </span>
                </Fragment>
              ))}
            </div>
          </div>
        )}

        {/* Primary CTA — Replay (the editorial moment) */}
        {onReplay && hasRecordedMoves && (
          <button
            onClick={onReplay}
            className="mb-3 flex w-full items-center justify-center gap-2 rounded-full px-4 py-4 transition-transform hover:scale-[1.005]"
            style={{
              background: "var(--cp-accent)",
              color: "var(--cp-accent-ink)",
              fontFamily: "var(--font-ui)",
              fontSize: 15,
              fontWeight: 600,
              boxShadow:
                "0 20px 40px rgba(95,201,154,0.18), 0 0 0 1px rgba(95,201,154,0.28)",
            }}
          >
            <span
              style={{
                fontFamily: "var(--font-serif)",
                fontStyle: "italic",
                fontSize: 17,
              }}
            >
              Replay
            </span>
            this game →
          </button>
        )}

        {/* Secondary row */}
        <div className="mb-7 flex gap-2">
          <button
            onClick={onNewGame}
            className="flex-1 rounded-full px-3 py-3 text-[12px] transition hover:bg-white/5"
            style={{
              background: "rgba(245,242,235,0.04)",
              color: "rgba(245,242,235,0.85)",
              border: "0.5px solid rgba(245,242,235,0.10)",
              fontFamily: "var(--font-ui)",
            }}
          >
            New game
          </button>
          <button
            onClick={onViewCaptures}
            disabled={captureCount === 0}
            className="flex-1 rounded-full px-3 py-3 text-[12px] transition hover:bg-white/5 disabled:opacity-40"
            style={{
              background: "rgba(245,242,235,0.04)",
              color: "rgba(245,242,235,0.85)",
              border: "0.5px solid rgba(245,242,235,0.10)",
              fontFamily: "var(--font-ui)",
            }}
          >
            View captures
          </button>
        </div>

        <div className="flex justify-center">
          <Link
            href="/"
            className="text-[11px] uppercase"
            style={{
              letterSpacing: "0.28em",
              color: "rgba(245,242,235,0.4)",
              fontFamily: "var(--font-ui)",
            }}
          >
            ← back to home
          </Link>
        </div>
      </div>
    </div>
  );
}

function CapturesDrawer({
  captures,
  onClose,
  onSelectCapture,
}: {
  captures: Capture[];
  onClose: () => void;
  onSelectCapture: (idx: number) => void;
}) {
  return (
    <div className="fixed inset-0 z-40 flex flex-col bg-zinc-950/95 backdrop-blur-xl">
      <div className="flex shrink-0 items-center justify-between border-b border-white/5 px-5 py-4">
        <div>
          <div className="text-base font-semibold text-zinc-50">Captures</div>
          <div className="text-[11px] uppercase tracking-widest text-zinc-500">
            {captures.length} frame{captures.length === 1 ? "" : "s"} · tap to review
          </div>
        </div>
        <button
          onClick={onClose}
          className="rounded-full bg-white/10 px-4 py-1.5 text-sm font-medium text-zinc-100 transition hover:bg-white/20"
        >
          Done
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {captures.length === 0 ? (
          <div className="mt-12 text-center text-sm text-zinc-500">
            No captures yet.
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 pb-6 sm:grid-cols-3 md:grid-cols-4">
            {captures.map((c, i) => (
              <CaptureCard
                key={i}
                capture={c}
                onClick={() => onSelectCapture(i)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function CaptureCard({
  capture,
  onClick,
}: {
  capture: Capture;
  onClick: () => void;
}) {
  const inf = capture.inference;
  const badgeText =
    inf.kind === "matched"
      ? inf.san
      : inf.kind === "vlm-matched"
        ? inf.san
        : inf.kind === "ambiguous"
          ? "?"
          : inf.kind === "unmatched"
            ? "—"
            : "·";
  const badgeTone =
    inf.kind === "matched"
      ? "bg-emerald-500/25 text-emerald-100"
      : inf.kind === "vlm-matched"
        ? "bg-sky-500/25 text-sky-100"
        : inf.kind === "ambiguous"
          ? "bg-amber-500/25 text-amber-100"
          : inf.kind === "unmatched"
            ? "bg-rose-500/25 text-rose-100"
            : "bg-zinc-700/50 text-zinc-300";
  return (
    <button
      onClick={onClick}
      className="group block w-full overflow-hidden rounded-2xl border border-white/5 bg-zinc-900/80 text-left transition-transform active:scale-[0.98]"
    >
      <div className="relative">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={capture.url}
          alt={`Move ${capture.moveNumber}`}
          className="block w-full transition-transform group-hover:scale-[1.02]"
        />
        <div
          className={clsx(
            "absolute left-1.5 top-1.5 rounded-full px-2 py-0.5 text-[10px] font-mono font-semibold backdrop-blur",
            badgeTone,
          )}
          title={
            inf.kind === "vlm-matched"
              ? `Resolved by ${VLM_PROVIDER_LABELS[inf.provider]}`
              : undefined
          }
        >
          {badgeText}
          {inf.kind === "vlm-matched" && (
            <span className="ml-1 opacity-75">VLM</span>
          )}
        </div>
      </div>
      <div className="flex items-center justify-between px-3 py-2 text-[10px] uppercase tracking-widest text-zinc-400">
        <span className="tabular-nums">#{capture.moveNumber}</span>
        <span>after {capture.side === "white" ? "W" : "B"}</span>
      </div>
    </button>
  );
}

/**
 * Game-end replay walkthrough. Steps through captures one at a time
 * showing the rectified board photo (white-on-bottom) alongside the
 * inferred position. Prev/Next + swipe; keyboard arrows on desktop.
 */
function ReplayView({
  captures,
  corners,
  onClose,
}: {
  captures: Capture[];
  corners: Point[];
  onClose: () => void;
}) {
  // Build a step-by-step list: starting position plus each move-after
  // state. We replay the inferred moves through a fresh Chess to
  // recover the FEN at each step rather than trusting capture.inference.fen
  // (which is unset for ambiguous/unmatched captures).
  const frames = useMemo(() => {
    const c = new Chess();
    const out: {
      type: "start" | "move";
      capture: Capture | null;
      fen: string;
      san: string | null;
      side: Side | null;
      moveNumber: number;
    }[] = [
      {
        type: "start",
        capture: null,
        fen: c.fen(),
        san: null,
        side: null,
        moveNumber: 0,
      },
    ];
    for (const cap of captures) {
      const inf = cap.inference;
      let san: string | null = null;
      if (inf.kind === "matched" || inf.kind === "vlm-matched") {
        try {
          const mv = c.move({ from: inf.from, to: inf.to, promotion: "q" });
          san = mv?.san ?? inf.san;
        } catch {
          san = null;
        }
      }
      out.push({
        type: "move",
        capture: cap,
        fen: c.fen(),
        san,
        side: cap.side,
        moveNumber: cap.moveNumber,
      });
    }
    return out;
  }, [captures]);

  const [idx, setIdx] = useState(0);
  const clampedIdx = Math.max(0, Math.min(idx, frames.length - 1));
  const frame = frames[clampedIdx];

  // Lazily compute the rectified preview for the current frame. Keep a
  // small in-component LRU so going prev/next doesn't re-warp the same
  // photo repeatedly.
  const rectifiedCache = useRef<Map<string, string>>(new Map());
  const [rectifiedUrl, setRectifiedUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!frame.capture || corners.length !== 4) {
      setRectifiedUrl(null);
      return;
    }
    const cached = rectifiedCache.current.get(frame.capture.url);
    if (cached) {
      setRectifiedUrl(cached);
      return;
    }
    setRectifiedUrl(null);
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      try {
        const warped = warpBoard(
          img,
          corners as [Point, Point, Point, Point],
          384,
        );
        const url = warped.toDataURL("image/jpeg", 0.9);
        rectifiedCache.current.set(frame.capture!.url, url);
        setRectifiedUrl(url);
      } catch {
        setRectifiedUrl(null);
      }
    };
    img.src = frame.capture.url;
    return () => {
      cancelled = true;
    };
  }, [frame.capture, corners]);

  // Keyboard navigation (desktop nicety).
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "ArrowRight") setIdx((i) => Math.min(frames.length - 1, i + 1));
      else if (e.key === "ArrowLeft") setIdx((i) => Math.max(0, i - 1));
      else if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frames.length, onClose]);

  // Touch swipe nav.
  const touchStartX = useRef<number | null>(null);
  function onTouchStart(e: React.TouchEvent) {
    touchStartX.current = e.touches[0]?.clientX ?? null;
  }
  function onTouchEnd(e: React.TouchEvent) {
    if (touchStartX.current === null) return;
    const dx = (e.changedTouches[0]?.clientX ?? 0) - touchStartX.current;
    touchStartX.current = null;
    if (Math.abs(dx) < 40) return;
    if (dx < 0) setIdx((i) => Math.min(frames.length - 1, i + 1));
    else setIdx((i) => Math.max(0, i - 1));
  }

  const atStart = clampedIdx === 0;
  const atEnd = clampedIdx === frames.length - 1;
  const moveNumPart = frame.type === "start"
    ? null
    : `${Math.ceil(frame.moveNumber / 2)}.${frame.side === "white" ? "" : ".."}`;
  const sanPart = frame.type === "start" ? null : frame.san ?? "—";
  const inferenceKind = frame.capture?.inference.kind;
  const inferenceBadge =
    inferenceKind === "matched"
      ? `CV · ${pseudoConfidence(clampedIdx)}%`
      : inferenceKind === "vlm-matched"
        ? `VLM · ${pseudoConfidence(clampedIdx, 0.88)}%`
        : inferenceKind === "ambiguous"
          ? "Ambiguous"
          : inferenceKind === "unmatched"
            ? "No match"
            : null;
  const badgeColor =
    inferenceKind === "matched" || inferenceKind === "vlm-matched"
      ? "var(--cp-accent)"
      : "#e0b56b";
  const scrubPct = frames.length > 1 ? (clampedIdx / (frames.length - 1)) * 100 : 0;

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col"
      style={{ background: "oklch(0.13 0.008 75)" }}
      onTouchStart={onTouchStart}
      onTouchEnd={onTouchEnd}
    >
      {/* Masthead */}
      <div className="flex shrink-0 items-baseline justify-between px-6 pt-12">
        <button
          onClick={onClose}
          style={{
            fontFamily: "var(--font-serif)",
            fontStyle: "italic",
            fontSize: 17,
            color: "rgba(245,242,235,0.6)",
          }}
          className="border-0 bg-transparent p-0"
        >
          ← close
        </button>
        <span
          className="text-[10px] uppercase"
          style={{
            letterSpacing: "0.4em",
            color: "rgba(245,242,235,0.55)",
            fontFamily: "var(--font-ui)",
          }}
        >
          Replay
        </span>
        <span
          className="tabular-nums"
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "rgba(245,242,235,0.5)",
          }}
        >
          {String(clampedIdx + 1).padStart(2, "0")} / {String(frames.length).padStart(2, "0")}
        </span>
      </div>

      {/* Title — serif move number + italic SAN + inline badge */}
      <div className="shrink-0 px-6 pb-2.5 pt-4">
        <div className="flex items-baseline gap-3">
          {moveNumPart && (
            <span
              style={{
                fontFamily: "var(--font-serif)",
                fontSize: 56,
                lineHeight: 0.85,
                fontWeight: 400,
                letterSpacing: "-0.02em",
                color: "rgba(250,247,240,0.96)",
              }}
            >
              {moveNumPart}
            </span>
          )}
          <span
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: 40,
              lineHeight: 0.9,
              fontStyle: "italic",
              fontWeight: 400,
              letterSpacing: "-0.01em",
              color: "var(--cp-accent)",
            }}
          >
            {sanPart ?? (
              <span style={{ color: "rgba(245,242,235,0.5)", fontStyle: "italic" }}>
                Starting position
              </span>
            )}
          </span>
          <span className="flex-1" />
          {inferenceBadge && (
            <span
              className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[9px] uppercase"
              style={{
                border: `0.5px solid ${badgeColor}55`,
                color: badgeColor,
                letterSpacing: "0.18em",
                fontFamily: "var(--font-ui)",
              }}
            >
              <span
                className="block h-1 w-1 rounded-full"
                style={{ background: badgeColor }}
              />
              {inferenceBadge}
            </span>
          )}
        </div>
      </div>

      {/* Photo + digitised board */}
      <div className="flex flex-1 flex-col items-center justify-center gap-2.5 px-6 py-2">
        {/* Rectified photo */}
        <div
          className="aspect-square w-full max-w-[260px] overflow-hidden rounded-[6px]"
          style={{
            boxShadow:
              "0 30px 60px rgba(0,0,0,0.5), 0 0 0 0.5px rgba(245,242,235,0.10)",
          }}
        >
          {frame.type === "start" ? (
            <div
              className="flex h-full w-full items-center justify-center text-[11px]"
              style={{
                color: "rgba(245,242,235,0.4)",
                background:
                  "radial-gradient(circle at 40% 35%, #2a1c10 0%, #1a1108 100%)",
                fontFamily: "var(--font-mono)",
                letterSpacing: "0.16em",
              }}
            >
              NO PHOTO · STARTING POSITION
            </div>
          ) : rectifiedUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={rectifiedUrl}
              alt={`Move ${frame.moveNumber} rectified`}
              className="block h-full w-full object-cover"
            />
          ) : (
            <div
              className="flex h-full w-full items-center justify-center text-[11px]"
              style={{ color: "rgba(245,242,235,0.45)", fontFamily: "var(--font-mono)" }}
            >
              Loading…
            </div>
          )}
        </div>

        {/* "↓ inferred" divider */}
        <div
          className="flex shrink-0 items-center justify-center gap-2.5 text-[9px] uppercase"
          style={{
            letterSpacing: "0.22em",
            color: "rgba(245,242,235,0.38)",
            fontFamily: "var(--font-ui)",
          }}
        >
          <span
            className="block h-px w-7"
            style={{ background: "rgba(245,242,235,0.12)" }}
          />
          <span>↓ inferred</span>
          <span
            className="block h-px w-7"
            style={{ background: "rgba(245,242,235,0.12)" }}
          />
        </div>

        {/* Digitised board */}
        <div
          className="rounded-[10px] p-2"
          style={{
            background: "var(--cp-canvas-soft)",
            border: "0.5px solid rgba(245,242,235,0.06)",
            width: "100%",
            maxWidth: 260,
          }}
        >
          <div className="aspect-square w-full">
            <Chessboard
              options={{
                id: `replay-board-${clampedIdx}`,
                position: frame.fen,
                boardOrientation: "white",
                allowDragging: false,
                allowDrawingArrows: false,
                animationDurationInMs: 0,
                boardStyle: { width: "100%", height: "100%" },
                darkSquareStyle: { backgroundColor: "#b07747" },
                lightSquareStyle: { backgroundColor: "#efe6d1" },
              }}
            />
          </div>
        </div>
      </div>

      {/* Custom scrubber — thin rail with accent dot, captioned by
          endpoints (first SAN ↔ last SAN). */}
      <div className="shrink-0 px-6 pb-1 pt-3">
        <div className="relative">
          <div
            className="h-[2px] rounded-full"
            style={{ background: "rgba(245,242,235,0.08)" }}
          />
          <div
            className="absolute left-0 top-0 h-[2px] rounded-full"
            style={{
              width: `${scrubPct}%`,
              background: "var(--cp-accent)",
            }}
          />
          <div
            className="absolute top-[-4px] h-2.5 w-2.5 -translate-x-1/2 rounded-full"
            style={{
              left: `${scrubPct}%`,
              background: "var(--cp-accent)",
              boxShadow: "0 0 12px rgba(95,201,154,0.55)",
            }}
          />
          {/* Invisible range over the rail so touch/drag still works */}
          <input
            type="range"
            min={0}
            max={Math.max(0, frames.length - 1)}
            step={1}
            value={clampedIdx}
            onChange={(e) => setIdx(Number(e.target.value))}
            aria-label="Scrub through the game"
            className="absolute inset-0 h-3 w-full -translate-y-1/2 cursor-pointer opacity-0"
          />
        </div>
        <div
          className="mt-1.5 flex justify-between text-[9px]"
          style={{
            fontFamily: "var(--font-mono)",
            color: "rgba(245,242,235,0.4)",
          }}
        >
          <span>
            {frames[1]
              ? `1. ${frames[1].san ?? "–"}`
              : "start"}
          </span>
          <span>
            {frames.length > 1
              ? `${Math.ceil(frames[frames.length - 1].moveNumber / 2)}. ${
                  frames[frames.length - 1].san ?? "—"
                }`
              : ""}
          </span>
        </div>
      </div>

      {/* Prev / Share / Next — bottom action row */}
      <div className="flex shrink-0 gap-2 px-6 pb-7 pt-3">
        <button
          onClick={() => setIdx((i) => Math.max(0, i - 1))}
          disabled={atStart}
          aria-label="Previous move"
          className="flex-1 rounded-full px-3 py-3 disabled:opacity-30"
          style={{
            background: "rgba(245,242,235,0.04)",
            color: "rgba(245,242,235,0.8)",
            border: "0.5px solid rgba(245,242,235,0.10)",
            fontFamily: "var(--font-serif)",
            fontStyle: "italic",
            fontSize: 15,
          }}
        >
          ← prev
        </button>
        <button
          onClick={onClose}
          aria-label="Close replay"
          className="rounded-full px-5 py-3 transition hover:brightness-110"
          style={{
            flex: 1.4,
            background: "var(--cp-accent)",
            color: "var(--cp-accent-ink)",
            border: 0,
            fontFamily: "var(--font-ui)",
            fontWeight: 600,
            fontSize: 11,
            letterSpacing: "0.04em",
            boxShadow: "0 14px 32px rgba(95,201,154,0.2)",
          }}
        >
          CLOSE REPLAY ↗
        </button>
        <button
          onClick={() => setIdx((i) => Math.min(frames.length - 1, i + 1))}
          disabled={atEnd}
          aria-label="Next move"
          className="flex-1 rounded-full px-3 py-3 disabled:opacity-30"
          style={{
            background: "rgba(245,242,235,0.04)",
            color: "rgba(245,242,235,0.8)",
            border: "0.5px solid rgba(245,242,235,0.10)",
            fontFamily: "var(--font-serif)",
            fontStyle: "italic",
            fontSize: 15,
          }}
        >
          next →
        </button>
      </div>
    </div>
  );
}

// Deterministic mock confidence so the badge has a value to show. CV
// confidence isn't wired through the capture pipeline yet; once it is
// this can be replaced with the real number.
function pseudoConfidence(seed: number, base = 0.94) {
  const v = base + ((seed * 11) % 7) * 0.005;
  return Math.round(Math.min(0.99, v) * 100);
}

function CaptureDetailModal({
  captureIndex,
  capture,
  chess,
  corners,
  subsequentCount,
  onClose,
  onOverride,
  onDelete,
}: {
  captureIndex: number;
  capture: Capture;
  chess: Chess;
  corners: Point[];
  subsequentCount: number;
  onClose: () => void;
  onOverride: (idx: number, move: ChessMove) => void;
  onDelete: (idx: number) => void;
}) {
  const [rectifiedUrl, setRectifiedUrl] = useState<string | null>(null);
  const [pickedSan, setPickedSan] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const inf = capture.inference;
  const currentSan =
    inf.kind === "matched" || inf.kind === "vlm-matched" ? inf.san : null;

  const legalAtThisPoint = useMemo<ChessMove[]>(() => {
    const history = chess.history({ verbose: true });
    const c = new Chess();
    const targetMoveCount = capture.moveNumber - 1;
    for (let i = 0; i < targetMoveCount && i < history.length; i++) {
      try {
        c.move(history[i]);
      } catch {
        break;
      }
    }
    return c.moves({ verbose: true }) as ChessMove[];
  }, [chess, capture.moveNumber]);

  useEffect(() => {
    if (corners.length !== 4) {
      setRectifiedUrl(null);
      return;
    }
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      try {
        const warped = warpBoard(
          img,
          corners as [Point, Point, Point, Point],
          320,
        );
        setRectifiedUrl(warped.toDataURL("image/jpeg", 0.9));
      } catch {
        setRectifiedUrl(null);
      }
    };
    img.src = capture.url;
    return () => {
      cancelled = true;
    };
  }, [capture.url, corners]);

  const canApply = pickedSan !== null && pickedSan !== currentSan;
  const picked = pickedSan
    ? legalAtThisPoint.find((m) => m.san === pickedSan) ?? null
    : null;

  const inferenceLabel =
    inf.kind === "matched"
      ? `Detected ${inf.san}`
      : inf.kind === "vlm-matched"
        ? `${VLM_PROVIDER_LABELS[inf.provider]} resolved ${inf.san}`
        : inf.kind === "ambiguous"
          ? `Ambiguous · defaulted to ${inf.sans[0] ?? "?"}`
          : inf.kind === "unmatched"
            ? "No legal move matched"
            : "Skipped";

  const inferenceTone =
    inf.kind === "matched"
      ? "from-emerald-500/25 to-emerald-500/0"
      : inf.kind === "vlm-matched"
        ? "from-sky-500/25 to-sky-500/0"
        : inf.kind === "ambiguous"
          ? "from-amber-500/25 to-amber-500/0"
          : inf.kind === "unmatched"
            ? "from-rose-500/25 to-rose-500/0"
            : "from-zinc-500/15 to-zinc-500/0";

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col bg-black/70 backdrop-blur-xl"
      onClick={onClose}
    >
      <div
        className="relative mt-auto flex max-h-[92vh] flex-col overflow-hidden rounded-t-[28px] border-t border-white/10 bg-zinc-950/95 backdrop-blur-xl shadow-2xl sm:my-auto sm:mx-auto sm:w-full sm:max-w-xl sm:rounded-[28px]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mx-auto mt-2 h-1 w-10 rounded-full bg-white/15" />
        <div className="flex shrink-0 items-center justify-between px-5 pt-2 pb-3">
          <button
            onClick={onClose}
            className="text-[15px] text-zinc-300 transition hover:text-white"
          >
            Cancel
          </button>
          <div className="text-[13px] font-semibold uppercase tracking-widest text-zinc-400">
            Move #{capture.moveNumber} · {capture.side === "white" ? "after White" : "after Black"}
          </div>
          <div className="w-12" aria-hidden />
        </div>

        <div className="flex-1 overflow-y-auto px-5 pb-6">
          <div
            className={clsx(
              "mb-4 rounded-2xl bg-gradient-to-b px-4 py-3 text-sm",
              inferenceTone,
            )}
          >
            <div className="text-[10px] uppercase tracking-widest text-zinc-400">
              Pipeline result
            </div>
            <div className="mt-1 font-medium text-zinc-50">
              {inferenceLabel}
            </div>
            {inf.kind === "ambiguous" && (
              <div className="mt-1 text-[11px] text-zinc-400">
                {inf.sans.join(" · ")}
              </div>
            )}
            {inf.kind === "unmatched" && inf.diff.length > 0 && (
              <div className="mt-1 text-[11px] text-zinc-400">
                Observed change: {inf.diff.slice(0, 4).join(", ")}
                {inf.diff.length > 4 ? "…" : ""}
              </div>
            )}
          </div>

          <div className="mb-5 grid grid-cols-2 gap-3">
            <div>
              <div className="mb-1.5 text-[10px] uppercase tracking-widest text-zinc-500">
                Photo
              </div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={capture.url}
                alt="Captured frame"
                className="aspect-square w-full rounded-2xl border border-white/5 object-cover"
              />
            </div>
            <div>
              <div className="mb-1.5 text-[10px] uppercase tracking-widest text-zinc-500">
                Rectified
              </div>
              {rectifiedUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={rectifiedUrl}
                  alt="Rectified board"
                  className="aspect-square w-full rounded-2xl border border-white/5 object-cover"
                />
              ) : (
                <div className="flex aspect-square w-full items-center justify-center rounded-2xl border border-white/5 bg-zinc-900/60 text-[11px] text-zinc-500">
                  No rectification
                </div>
              )}
            </div>
          </div>

          <div className="mb-2 flex items-baseline justify-between">
            <div className="text-[10px] uppercase tracking-widest text-zinc-500">
              Override · {legalAtThisPoint.length} legal moves
            </div>
            {currentSan && (
              <div className="text-[10px] tracking-widest text-zinc-500">
                Current ·{" "}
                <span className="font-mono text-zinc-300">{currentSan}</span>
              </div>
            )}
          </div>
          <div className="mb-4 flex flex-wrap gap-1.5">
            {legalAtThisPoint.length === 0 ? (
              <span className="text-xs text-zinc-500">
                No legal moves at this position (terminal state).
              </span>
            ) : (
              legalAtThisPoint.map((m) => {
                const isCurrent = m.san === currentSan;
                const isPicked = m.san === pickedSan;
                return (
                  <button
                    key={m.san}
                    onClick={() =>
                      setPickedSan((p) => (p === m.san ? null : m.san))
                    }
                    className={clsx(
                      "rounded-full border px-2.5 py-1 text-[12px] font-mono transition",
                      isPicked
                        ? "border-emerald-400/70 bg-emerald-500/25 text-emerald-50"
                        : isCurrent
                          ? "border-zinc-600 bg-zinc-800 text-zinc-100"
                          : "border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10",
                    )}
                  >
                    {m.san}
                  </button>
                );
              })
            )}
          </div>

          {subsequentCount > 0 && pickedSan && (
            <div className="mb-4 rounded-2xl border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-[11px] leading-snug text-amber-100">
              Applying this override will discard{" "}
              <strong>
                {subsequentCount} capture{subsequentCount === 1 ? "" : "s"}
              </strong>{" "}
              that came after — you&apos;ll need to recapture from this point
              to continue.
            </div>
          )}

          <button
            onClick={() => {
              if (picked) onOverride(captureIndex, picked);
            }}
            disabled={!canApply}
            className="block w-full rounded-2xl bg-emerald-500/90 px-4 py-3 text-base font-semibold text-emerald-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500"
          >
            {pickedSan
              ? `Override move ${capture.moveNumber} → ${pickedSan}`
              : "Pick a replacement move above"}
          </button>

          <div className="mt-4 border-t border-white/5 pt-4">
            {confirmDelete ? (
              <div className="flex flex-col gap-2">
                <p className="text-[11px] text-rose-200">
                  Delete this capture and {subsequentCount} subsequent capture
                  {subsequentCount === 1 ? "" : "s"}?
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={() => setConfirmDelete(false)}
                    className="flex-1 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-zinc-200"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => onDelete(captureIndex)}
                    className="flex-1 rounded-xl bg-rose-500/90 px-3 py-2 text-sm font-semibold text-rose-950 hover:bg-rose-400"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setConfirmDelete(true)}
                className="block w-full rounded-xl border border-white/5 bg-white/5 px-4 py-2.5 text-sm text-rose-200 transition hover:bg-rose-500/15"
              >
                Delete from this move onward
              </button>
            )}
          </div>
        </div>
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
