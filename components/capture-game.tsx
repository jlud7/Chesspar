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
import { Chess, type Move as ChessMove } from "chess.js";
import clsx from "clsx";
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
  makeAnthropicProxyVerifier,
  makeOpenAiProxyVerifier,
  makeVerifier,
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
).toLowerCase() as "anthropic" | "openai";

function makeProxyVerifier(): VlmVerifier | null {
  if (!VLM_PROXY_URL) return null;
  if (VLM_PROXY_PROVIDER === "openai") {
    return makeOpenAiProxyVerifier(VLM_PROXY_URL);
  }
  return makeAnthropicProxyVerifier(VLM_PROXY_URL);
}
import { inferMoveFuzzy } from "@/lib/move-inference";
import {
  autoDetectBoardCorners,
  refineCornersForFrame,
  rotateCorners,
  scorePlayingOrientation,
} from "@/lib/board-detection";

type Side = "white" | "black";
type Phase = "settings" | "calibrating" | "playing" | "paused" | "ended";

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
  const streamRef = useRef<MediaStream | null>(null);
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
    if (phase === "settings" || phase === "ended" || testMode) {
      stopCamera();
      return;
    }
    if (!streamRef.current) {
      void startCamera();
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

  useEffect(() => {
    if (phase !== "calibrating" || corners.length === 4) return;
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
  }, [phase, videoDims, testFrames, testFrameIdx]);

  useEffect(() => {
    if (phase !== "calibrating" || corners.length !== 4 || busy) return;
    const t = window.setTimeout(() => startPlayingFromCalibration(), 300);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, corners.length, busy]);

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

  function recalibrate() {
    setCorners([]);
    stableDetectionCountRef.current = 0;
    lastDetectedCornersRef.current = null;
    baselineRef.current = null;
    previousFrameRef.current = null;
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
            moveLog={moveLog}
            inferring={inferring}
            vlmActive={vlmActive}
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
          {phase === "calibrating" && (
            <StartingOverlay
              testMode={testMode}
              cameraError={cameraError}
              onCancel={backToSettings}
            />
          )}
          <CaptureToast
            phase={phase}
            inferring={inferring}
            vlmActive={vlmActive}
            lastMove={lastMove}
            moveLogLength={moveLog.length}
          />
        </>
      )}

      {phase === "ended" && (
        <EndScreen
          winner={winner}
          captureCount={captures.length}
          recordedMoves={moveLog.length}
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
        "relative flex flex-1 select-none flex-col items-center justify-center overflow-hidden transition-all duration-200",
        isActive
          ? "bg-gradient-to-b from-zinc-50 to-zinc-200 text-zinc-900 shadow-[inset_0_-12px_30px_rgba(0,0,0,0.05)]"
          : "bg-zinc-900 text-zinc-500",
        flash && "ring-4 ring-inset ring-emerald-400/70",
        rotated && "rotate-180",
        "active:scale-[0.995]",
      )}
    >
      <span
        className={clsx(
          "absolute left-1/2 top-7 -translate-x-1/2 text-[10px] font-semibold uppercase tracking-[0.32em]",
          isActive ? "text-zinc-500" : "text-zinc-600",
        )}
      >
        {side === "white" ? "White" : "Black"}
      </span>
      <span
        className={clsx(
          "select-none tabular-nums leading-none",
          "text-[clamp(4rem,18vw,8.5rem)] font-extralight tracking-tighter",
          low && isActive
            ? "text-rose-600"
            : isActive
              ? "text-zinc-900"
              : "text-zinc-500",
        )}
      >
        {formatTime(ms)}
      </span>
      <span
        className={clsx(
          "absolute inset-x-0 bottom-6 flex items-center justify-center gap-3 text-[11px] tracking-[0.22em] uppercase",
          isActive ? "text-zinc-500" : "text-zinc-600",
        )}
      >
        <span>Moves · {moves}</span>
        <span aria-hidden>·</span>
        <span>{tcLabel}</span>
      </span>
    </button>
  );
}

function CenterBar({
  phase,
  soundOn,
  captureCount,
  moveLog,
  inferring,
  vlmActive,
  onTogglePause,
  onReset,
  onToggleSound,
  onShowCaptures,
  onEndGame,
}: {
  phase: Phase;
  soundOn: boolean;
  captureCount: number;
  moveLog: { san: string; viaVlm: boolean }[];
  inferring: boolean;
  vlmActive: boolean;
  onTogglePause: () => void;
  onReset: () => void;
  onToggleSound: () => void;
  onShowCaptures: () => void;
  onEndGame: () => void;
}) {
  return (
    <div className="relative flex shrink-0 flex-col border-y border-white/5 bg-zinc-950/70 backdrop-blur-xl">
      <MoveLogStrip
        moveLog={moveLog}
        inferring={inferring}
        vlmActive={vlmActive}
      />
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
            <span className="absolute -right-0.5 -top-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-emerald-500 px-1 text-[10px] font-bold text-emerald-950 shadow-md shadow-emerald-500/30">
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

function MoveLogStrip({
  moveLog,
  inferring,
  vlmActive,
}: {
  moveLog: { san: string; viaVlm: boolean }[];
  inferring: boolean;
  vlmActive: boolean;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [moveLog.length]);
  const emptyLabel = vlmActive
    ? "Reading the board with vision model…"
    : inferring
      ? "Inferring first move…"
      : "Tap your clock to record the first move";
  return (
    <div className="flex h-9 shrink-0 items-center gap-2 px-3">
      <div
        ref={ref}
        className="flex flex-1 items-center gap-1.5 overflow-x-auto whitespace-nowrap scroll-smooth py-1"
        style={{ scrollbarWidth: "none" }}
      >
        {moveLog.length === 0 ? (
          <span className="text-[11px] tracking-wide text-zinc-500">
            {emptyLabel}
          </span>
        ) : (
          moveLog.map((m, i) => {
            const moveNum = Math.floor(i / 2) + 1;
            const isWhite = i % 2 === 0;
            const isLast = i === moveLog.length - 1;
            return (
              <span
                key={i}
                className="inline-flex items-baseline gap-1 text-[11px] tabular-nums"
              >
                {isWhite && (
                  <span className="text-zinc-600">{moveNum}.</span>
                )}
                <span
                  className={clsx(
                    "rounded-full px-2 py-0.5 font-mono transition",
                    isLast
                      ? m.viaVlm
                        ? "bg-sky-500/25 text-sky-100 ring-1 ring-sky-400/30"
                        : "bg-emerald-500/25 text-emerald-100 ring-1 ring-emerald-400/30"
                      : m.viaVlm
                        ? "text-sky-300"
                        : "text-zinc-200",
                  )}
                  title={m.viaVlm ? "Resolved by VLM" : undefined}
                >
                  {m.san}
                  {m.viaVlm && (
                    <span className="ml-1 text-[9px] uppercase tracking-wider opacity-70">
                      vlm
                    </span>
                  )}
                </span>
              </span>
            );
          })
        )}
      </div>
      {(inferring || vlmActive) && moveLog.length > 0 && (
        <span
          className={clsx(
            "shrink-0 rounded-full px-2 py-0.5 text-[10px] uppercase tracking-widest",
            vlmActive
              ? "bg-sky-500/20 text-sky-200"
              : "bg-emerald-500/15 text-emerald-200",
          )}
        >
          {vlmActive ? "Vision · reading" : "Inferring"}
        </span>
      )}
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
 * Floating capture-status pill. Two states:
 *   - inferring/vlmActive → "Reading the board…" (with spinner)
 *   - just-detected (move appended within the last ~1.8 s) → "✓ <SAN>"
 * Sits as a thin pill in the upper third of the screen so it's visible
 * above the back-rank player panel without covering the center-bar or
 * the white pieces below. The MoveLogStrip in the center bar still
 * shows the running history; this pill is a *moment-of-confirmation*
 * cue rather than a log.
 */
function CaptureToast({
  phase,
  inferring,
  vlmActive,
  lastMove,
  moveLogLength,
}: {
  phase: Phase;
  inferring: boolean;
  vlmActive: boolean;
  lastMove: { san: string; side: Side } | null;
  moveLogLength: number;
}) {
  const [showRecent, setShowRecent] = useState(false);
  const lastSeenLenRef = useRef(0);
  useEffect(() => {
    if (moveLogLength > lastSeenLenRef.current) {
      lastSeenLenRef.current = moveLogLength;
      setShowRecent(true);
      const t = window.setTimeout(() => setShowRecent(false), 1800);
      return () => window.clearTimeout(t);
    }
  }, [moveLogLength]);
  if (phase !== "playing") return null;
  const busy = inferring || vlmActive;
  const showing = busy ? "busy" : showRecent && lastMove ? "recent" : null;
  // Anchor the toast to the side that just moved so the relevant player
  // sees it right-side-up — black on top (rotated), white on bottom.
  const anchor =
    showing === "recent" && lastMove
      ? lastMove.side === "white"
        ? "bottom"
        : "top"
      : "center";
  return (
    <div
      className={clsx(
        "pointer-events-none absolute left-1/2 z-30 -translate-x-1/2 transition-opacity duration-300",
        anchor === "top" && "top-[18%] rotate-180",
        anchor === "bottom" && "bottom-[18%]",
        anchor === "center" && "top-[42%]",
        showing ? "opacity-100" : "opacity-0",
      )}
    >
      {showing === "busy" && (
        <div className="flex items-center gap-2.5 rounded-full bg-zinc-950/85 px-4 py-2 text-sm font-medium tracking-tight text-zinc-50 shadow-lg shadow-black/40 ring-1 ring-white/15 backdrop-blur-xl">
          <span className="relative inline-block h-2.5 w-2.5">
            <span
              className={clsx(
                "absolute inset-0 animate-ping rounded-full",
                vlmActive ? "bg-sky-400/60" : "bg-emerald-400/60",
              )}
            />
            <span
              className={clsx(
                "absolute inset-0 rounded-full",
                vlmActive ? "bg-sky-400" : "bg-emerald-400",
              )}
            />
          </span>
          {vlmActive ? "Reading with vision model…" : "Reading the board…"}
        </div>
      )}
      {showing === "recent" && lastMove && (
        <div className="flex items-center gap-2.5 rounded-full bg-emerald-500/95 px-4 py-2 text-sm font-semibold tracking-tight text-emerald-950 shadow-lg shadow-emerald-500/30 ring-1 ring-emerald-300/40 backdrop-blur-xl">
          <svg
            viewBox="0 0 24 24"
            className="h-4 w-4"
            fill="none"
            stroke="currentColor"
            strokeWidth={3}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden
          >
            <path d="M5 12l4 4L19 7" />
          </svg>
          <span className="font-mono text-base">{lastMove.san}</span>
          <span className="text-[10px] font-bold uppercase tracking-[0.18em] opacity-80">
            {lastMove.side === "white" ? "white" : "black"}
          </span>
        </div>
      )}
    </div>
  );
}

/**
 * Lightweight overlay during the brief auto-detect window. Sits on top of
 * the live player UI rather than taking over the screen — most boards
 * detect within ~1 s and the overlay just flashes briefly. After a long
 * delay we add a hint and a cancel-out link.
 */
function StartingOverlay({
  testMode,
  cameraError,
  onCancel,
}: {
  testMode: boolean;
  cameraError: string | null;
  onCancel: () => void;
}) {
  const [showHint, setShowHint] = useState(false);
  useEffect(() => {
    const t = window.setTimeout(() => setShowHint(true), 5000);
    return () => window.clearTimeout(t);
  }, []);
  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center bg-black/55 backdrop-blur-md">
      <div className="pointer-events-auto flex w-[min(20rem,86vw)] flex-col items-center gap-4 rounded-3xl bg-white/8 px-6 py-7 text-center ring-1 ring-white/15">
        <span className="relative block h-8 w-8">
          <span className="absolute inset-0 animate-ping rounded-full bg-emerald-400/40" />
          <span className="absolute inset-1 rounded-full bg-emerald-400/90" />
        </span>
        <div className="space-y-1">
          <div className="text-sm font-semibold tracking-tight text-zinc-50">
            Finding the board…
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
  return (
    <div className="relative flex flex-1 flex-col overflow-y-auto px-5 py-8">
      <Link
        href="/"
        className="absolute left-4 top-4 inline-flex items-center gap-1 rounded-full bg-white/5 px-3 py-1 text-[11px] uppercase tracking-widest text-zinc-300 hover:bg-white/10"
      >
        ← Home
      </Link>
      <div className="mx-auto w-full max-w-md">
        <div className="mb-1 mt-6 text-[11px] uppercase tracking-[0.3em] text-emerald-300">
          New game
        </div>
        <h1 className="mb-2 text-[2rem] font-semibold tracking-tight">
          Set up the board
        </h1>
        <p className="mb-7 text-[15px] leading-snug text-zinc-400">
          Pick a time control. We&apos;ll calibrate the camera once and then
          watch the board between each tap of your clock.
        </p>

        <SettingsSection title="Time control">
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
          <div className="mt-3">
            <CustomTcEditor tc={tc} onChange={onChangeTc} />
          </div>
        </SettingsSection>

        <SettingsSection title="Source">
          {hasSavedCorners && !testMode && (
            <div className="mb-3 flex items-center justify-between rounded-2xl border border-white/5 bg-white/5 px-3 py-2 text-[13px] text-zinc-300">
              <span>Board corners saved from last session.</span>
              <button
                onClick={onClearCorners}
                className="rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 text-[11px] text-zinc-200 hover:bg-white/10"
              >
                Recalibrate
              </button>
            </div>
          )}
          <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
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
              Skip the live camera and feed the pipeline a sequence of
              still photos instead.
            </p>
            {testMode && (
              <div className="mt-3 space-y-2">
                <ol className="space-y-1 rounded-xl bg-black/30 px-3 py-2 text-[11px] leading-snug text-zinc-300">
                  <li>
                    <span className="font-medium text-emerald-300">
                      Photo 1
                    </span>{" "}
                    — the starting position, before any moves.
                  </li>
                  <li>
                    <span className="font-medium text-emerald-300">
                      Photos 2…N
                    </span>{" "}
                    — one per half-move, in order (white, black, white…).
                    Each clock tap advances to the next photo.
                  </li>
                </ol>
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
                      {testFrameCount >= 3 && (
                        <>
                          {" "}
                          ({Math.floor((testFrameCount - 1) / 2)} full turn
                          {Math.floor((testFrameCount - 1) / 2) === 1
                            ? ""
                            : "s"}
                          {(testFrameCount - 1) % 2 === 1
                            ? " + white's move"
                            : ""}
                          )
                        </>
                      )}
                    </p>
                    <div className="flex gap-1.5 overflow-x-auto pb-1">
                      {testFrameSrcs.map((src, i) => (
                        <div
                          key={`${src}-${i}`}
                          className="relative shrink-0"
                          title={
                            i === 0
                              ? "Photo 1 — starting position"
                              : `Photo ${i + 1} — half-move ${i} (${
                                  i % 2 === 1 ? "white" : "black"
                                })`
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
                    <p className="text-[10px] text-zinc-500">
                      Confirm the order looks right — first should be the
                      starting position, then each half-move in sequence.
                    </p>
                  </>
                )}
              </div>
            )}
          </div>
        </SettingsSection>

        <SettingsSection title="Vision-LM fallback">
          {VLM_PROXY_URL ? (
            <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-[12px] text-emerald-100">
              <div className="font-medium">
                {VLM_PROXY_PROVIDER === "openai" ? "GPT-5" : "Claude"} is
                enabled by default.
              </div>
              <div className="mt-1 text-[11px] text-emerald-200/80">
                Photo identification runs through the bundled proxy — no
                key needed. Paste your own key below to override.
              </div>
              <details className="mt-2 text-[11px] text-emerald-100/80">
                <summary className="cursor-pointer">Override with your own key</summary>
                <div className="mt-2">
                  <VlmConfigEditor
                    config={vlmConfig}
                    onChange={onChangeVlmConfig}
                  />
                </div>
              </details>
            </div>
          ) : (
            <VlmConfigEditor config={vlmConfig} onChange={onChangeVlmConfig} />
          )}
        </SettingsSection>

        <button
          onClick={onStart}
          disabled={testMode && testFrameCount === 0}
          className="mt-2 block w-full rounded-2xl bg-emerald-500/90 px-4 py-3.5 text-base font-semibold text-emerald-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500"
        >
          {testMode
            ? "Start with test photos"
            : hasSavedCorners
              ? "Start game"
              : "Calibrate & start"}
        </button>

        {!testMode && (
          <p className="mt-3 text-center text-[11px] tracking-wide text-zinc-500">
            We&apos;ll ask for camera permission once.
          </p>
        )}

        {cameraError && (
          <div className="mt-4 rounded-2xl border border-amber-400/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
            Camera unavailable: {cameraError}. The clock will still work; no
            photos will be saved.
          </div>
        )}

        <div className="mt-8 rounded-2xl border border-white/5 bg-white/5 px-4 py-3 text-[12px] text-zinc-400">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.3em] text-zinc-300">
            Tip
          </div>
          Visit{" "}
          <Link
            href="/detect"
            className="font-medium text-emerald-300 underline-offset-2 hover:underline"
          >
            /detect
          </Link>{" "}
          to test the rectifier + classifier on a still photo first.
        </div>
      </div>
    </div>
  );
}

function SettingsSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="mb-5">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.3em] text-zinc-500">
        {title}
      </div>
      {children}
    </section>
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

function EndScreen({
  winner,
  captureCount,
  recordedMoves,
  moves,
  pgn,
  onNewGame,
  onRecalibrate,
  onViewCaptures,
}: {
  winner: Side | null;
  captureCount: number;
  recordedMoves: number;
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
  const hasRecordedMoves = recordedMoves > 0;
  const unrecorded = Math.max(0, captureCount - recordedMoves);

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
    <div className="flex flex-1 flex-col overflow-y-auto px-5 py-10">
      <div className="mx-auto w-full max-w-md">
        <div className="mb-1 text-[11px] uppercase tracking-[0.3em] text-zinc-500">
          Game over
        </div>
        <div className="mb-6 text-[2rem] font-semibold tracking-tight text-zinc-50">
          {title}
        </div>
        <div className="mb-5 grid grid-cols-2 gap-2 text-sm">
          <div className="rounded-2xl border border-white/5 bg-white/5 px-4 py-3">
            <div className="text-[10px] uppercase tracking-widest text-zinc-500">
              White moves
            </div>
            <div className="text-2xl tabular-nums text-zinc-50">
              {moves.white}
            </div>
          </div>
          <div className="rounded-2xl border border-white/5 bg-white/5 px-4 py-3">
            <div className="text-[10px] uppercase tracking-widest text-zinc-500">
              Black moves
            </div>
            <div className="text-2xl tabular-nums text-zinc-50">
              {moves.black}
            </div>
          </div>
        </div>

        {!hasRecordedMoves && captureCount > 0 ? (
          <div className="mb-6 rounded-2xl border border-amber-400/30 bg-amber-500/10 px-4 py-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.3em] text-amber-300">
              No moves were recorded
            </div>
            <p className="mt-1 text-[13px] leading-snug text-amber-100">
              The pipeline didn&apos;t pin a unique move for any of your{" "}
              {captureCount} capture{captureCount === 1 ? "" : "s"}. Tap
              <strong> View captures</strong> and assign each one the move
              that actually happened — your PGN rebuilds as you go.
            </p>
            <button
              onClick={onViewCaptures}
              className="mt-3 rounded-full bg-amber-500/80 px-4 py-1.5 text-[12px] font-semibold text-amber-950 hover:bg-amber-400"
            >
              Review captures →
            </button>
          </div>
        ) : (
          <div className="mb-6 rounded-2xl border border-white/5 bg-white/5 px-4 py-3 text-sm text-zinc-300">
            {recordedMoves} move{recordedMoves === 1 ? "" : "s"} recorded
            {unrecorded > 0 && (
              <span className="text-amber-200">
                {" "}
                · {unrecorded} awaiting review
              </span>
            )}
            <span className="text-zinc-500">
              {" "}
              from {captureCount} capture{captureCount === 1 ? "" : "s"}.
            </span>
          </div>
        )}

        {hasRecordedMoves && pgn && (
          <div className="mb-6 rounded-2xl border border-white/5 bg-zinc-950/60 p-3">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-[10px] uppercase tracking-widest text-zinc-500">
                PGN
              </div>
              <button
                onClick={copyPgn}
                className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] uppercase tracking-widest text-zinc-200 hover:bg-white/10"
              >
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed text-zinc-300">
              {pgn}
            </pre>
          </div>
        )}

        <div className="flex flex-col gap-2">
          <button
            onClick={onNewGame}
            className="rounded-2xl bg-emerald-500/90 px-4 py-3 text-base font-semibold text-emerald-950 hover:bg-emerald-400"
          >
            New game
          </button>
          <button
            onClick={onViewCaptures}
            disabled={captureCount === 0}
            className="rounded-2xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-zinc-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
          >
            View captures
          </button>
          <button
            onClick={onRecalibrate}
            className="rounded-2xl border border-white/5 bg-white/5 px-4 py-2.5 text-sm text-zinc-300 hover:bg-white/10"
          >
            Recalibrate corners
          </button>
          <Link
            href="/"
            className="mt-2 text-center text-[11px] uppercase tracking-widest text-zinc-500 hover:text-zinc-300"
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
