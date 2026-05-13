"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import clsx from "clsx";
import { extractSquareCrops, warpBoard } from "@/lib/board-image";
import type { Point } from "@/lib/homography";
import { classifyBoard, type ClassifyResult, type Occupancy } from "@/lib/occupancy";
import { inferMove, type InferResult } from "@/lib/move-inference";

const STARTING_FEN = new Chess().fen();

const CORNER_LABELS = ["a8", "h8", "h1", "a1"] as const;
const CORNER_HINTS = [
  "Tap the a8 corner — next to Black's queenside rook (left of Black's queen).",
  "Tap the h8 corner — next to Black's kingside rook (right of Black's king).",
  "Tap the h1 corner — next to White's kingside rook (right of White's king).",
  "Tap the a1 corner — next to White's queenside rook (left of White's queen).",
];

const RECTIFIED_SIZE = 512;

type ImageDims = { w: number; h: number };

export function BoardRectifier() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageDims, setImageDims] = useState<ImageDims | null>(null);
  const [corners, setCorners] = useState<Point[]>([]);
  const [warpedUrl, setWarpedUrl] = useState<string | null>(null);
  const [squareUrls, setSquareUrls] = useState<string[]>([]);
  const [occupancy, setOccupancy] = useState<ClassifyResult[]>([]);
  const [prevFen, setPrevFen] = useState<string>(STARTING_FEN);
  const [inferResult, setInferResult] = useState<InferResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const imageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  function clearResults() {
    setWarpedUrl(null);
    setSquareUrls([]);
    setOccupancy([]);
    setInferResult(null);
    setError(null);
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
    setImageDims(null);
    setCorners([]);
    clearResults();
  }

  function onImageLoad() {
    const img = imageRef.current;
    if (!img) return;
    setImageDims({ w: img.naturalWidth, h: img.naturalHeight });
  }

  function onImageClick(e: React.MouseEvent<HTMLImageElement>) {
    if (corners.length >= 4) return;
    const img = imageRef.current;
    if (!img) return;
    const rect = img.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * img.naturalWidth;
    const y = ((e.clientY - rect.top) / rect.height) * img.naturalHeight;
    setCorners((c) => [...c, { x, y }]);
    clearResults();
  }

  function undoCorner() {
    setCorners((c) => c.slice(0, -1));
    clearResults();
  }

  function resetCorners() {
    setCorners([]);
    clearResults();
  }

  const compute = useCallback(async () => {
    const img = imageRef.current;
    if (!img || corners.length !== 4) return;
    setBusy(true);
    setError(null);
    setInferResult(null);
    try {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const warped = warpBoard(
        img,
        corners as [Point, Point, Point, Point],
        RECTIFIED_SIZE,
      );
      const crops = extractSquareCrops(warped);
      const classes = classifyBoard(crops);
      setWarpedUrl(warped.toDataURL("image/png"));
      setSquareUrls(crops.map((c) => c.toDataURL("image/png")));
      setOccupancy(classes);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [corners]);

  function runInference() {
    if (occupancy.length !== 64) return;
    try {
      const result = inferMove(
        prevFen,
        occupancy.map((c) => c.state),
      );
      setInferResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function applyInferenceResult() {
    if (inferResult?.kind === "matched") {
      setPrevFen(inferResult.updatedFen);
      setInferResult(null);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <UploadCard onFileChange={onFileChange} hasImage={Boolean(imageUrl)} />

      {imageUrl && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
          <div
            className={clsx(
              "mb-3 flex items-start gap-3 rounded-md border px-3 py-2 text-sm",
              corners.length < 4
                ? "border-emerald-500/30 bg-emerald-500/5 text-emerald-200"
                : "border-zinc-700 bg-zinc-950/60 text-zinc-200",
            )}
          >
            <CornerHintInset step={corners.length} />
            <div className="min-w-0 flex-1">
              <span className="mr-2 inline-block rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs uppercase tracking-wider text-emerald-200">
                Step 2
              </span>
              {corners.length < 4
                ? CORNER_HINTS[corners.length]
                : "All four corners set. Click Compute to rectify."}
              {corners.length < 4 && (
                <div className="mt-1 text-xs text-zinc-400">
                  Photo orientation doesn&apos;t matter — corner names define
                  the rectified output.
                </div>
              )}
            </div>
          </div>

          <div className="relative inline-block w-full max-w-3xl overflow-hidden rounded-lg border border-zinc-800">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={imageRef}
              src={imageUrl}
              alt="Board photo"
              draggable={false}
              onLoad={onImageLoad}
              onClick={onImageClick}
              className="block h-auto w-full cursor-crosshair select-none"
            />
            {imageDims && (
              <CornerOverlay corners={corners} imageDims={imageDims} />
            )}
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            <button
              onClick={undoCorner}
              disabled={corners.length === 0}
              className="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Undo last tap
            </button>
            <button
              onClick={resetCorners}
              disabled={corners.length === 0}
              className="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Reset corners
            </button>
            <button
              onClick={compute}
              disabled={corners.length !== 4 || busy}
              className="rounded-md border border-emerald-500/40 bg-emerald-500/15 px-3 py-1.5 text-sm font-medium text-emerald-200 hover:bg-emerald-500/25 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy ? "Warping…" : "Compute rectified board"}
            </button>
          </div>

          {error && (
            <div className="mt-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
              {error}
            </div>
          )}
        </div>
      )}

      {warpedUrl && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
          <div className="mb-2 text-xs uppercase tracking-wider text-zinc-400">
            Rectified board · always oriented with White at the bottom
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={warpedUrl}
            alt="Rectified board"
            className="block w-full max-w-md rounded-md border border-zinc-800"
          />
        </div>
      )}

      {squareUrls.length === 64 && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
          <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
            <div className="text-xs uppercase tracking-wider text-zinc-400">
              Per-square crops + predicted occupancy
            </div>
            <OccupancyLegend />
          </div>
          <div className="grid w-full max-w-md grid-cols-8 gap-0.5">
            {squareUrls.map((src, i) => {
              const file = "abcdefgh"[i % 8];
              const rank = 8 - Math.floor(i / 8);
              const cls = occupancy[i];
              return (
                <div
                  key={i}
                  className={clsx(
                    "relative aspect-square overflow-hidden bg-zinc-950",
                    cls && occupancyRingClass(cls.state),
                  )}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={src}
                    alt={`${file}${rank}`}
                    className="block h-full w-full object-cover"
                  />
                  {cls && (
                    <span
                      className={clsx(
                        "pointer-events-none absolute left-0 top-0 inline-block h-2.5 w-full",
                        occupancyDotClass(cls.state),
                      )}
                      style={{ opacity: 0.4 + 0.6 * cls.confidence }}
                    />
                  )}
                  <div className="pointer-events-none absolute bottom-0 right-0 rounded-tl bg-black/70 px-1 text-[9px] font-medium leading-tight text-zinc-200">
                    {file}
                    {rank}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {occupancy.length === 64 && (
        <InferencePanel
          prevFen={prevFen}
          onChangePrevFen={(f) => {
            setPrevFen(f);
            setInferResult(null);
          }}
          inferResult={inferResult}
          onInfer={runInference}
          onAccept={applyInferenceResult}
        />
      )}
    </div>
  );
}

function occupancyRingClass(state: Occupancy): string {
  if (state === "white") return "ring-1 ring-inset ring-zinc-100/70";
  if (state === "black") return "ring-1 ring-inset ring-zinc-900";
  return "";
}

function occupancyDotClass(state: Occupancy): string {
  if (state === "white") return "bg-zinc-100";
  if (state === "black") return "bg-zinc-900";
  return "bg-emerald-500/0";
}

function OccupancyLegend() {
  return (
    <div className="flex items-center gap-3 text-[10px] uppercase tracking-wider text-zinc-500">
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-2 w-3 bg-zinc-100" /> White
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-2 w-3 bg-zinc-900 ring-1 ring-zinc-700" />
        Black
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-2 w-3 bg-transparent ring-1 ring-zinc-700" />
        Empty
      </span>
    </div>
  );
}

function InferencePanel({
  prevFen,
  onChangePrevFen,
  inferResult,
  onInfer,
  onAccept,
}: {
  prevFen: string;
  onChangePrevFen: (fen: string) => void;
  inferResult: InferResult | null;
  onInfer: () => void;
  onAccept: () => void;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-zinc-400">
          Move inference
        </div>
        <button
          onClick={() => onChangePrevFen(STARTING_FEN)}
          className="text-[10px] uppercase tracking-wider text-zinc-500 hover:text-zinc-300"
        >
          Reset to start position
        </button>
      </div>
      <p className="mb-3 text-xs text-zinc-400">
        Diff the predicted occupancy against the previous FEN, then pick the
        unique legal move that matches.
      </p>
      <label className="mb-3 block">
        <span className="mb-1 block text-[10px] uppercase tracking-wider text-zinc-500">
          Previous FEN
        </span>
        <input
          type="text"
          value={prevFen}
          onChange={(e) => onChangePrevFen(e.target.value)}
          spellCheck={false}
          className="w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 font-mono text-xs text-zinc-100"
        />
      </label>
      <button
        onClick={onInfer}
        className="rounded-md border border-emerald-500/40 bg-emerald-500/15 px-3 py-1.5 text-sm font-medium text-emerald-200 hover:bg-emerald-500/25"
      >
        Infer move
      </button>

      {inferResult && (
        <div className="mt-4 rounded-md border border-zinc-800 bg-zinc-950/60 p-3 text-sm">
          {inferResult.kind === "matched" && (
            <>
              <div className="mb-1 text-xs uppercase tracking-wider text-emerald-300">
                Matched
              </div>
              <div className="mb-2 text-base">
                <span className="font-mono text-emerald-200">
                  {inferResult.move.san}
                </span>{" "}
                <span className="text-zinc-500">
                  ({inferResult.move.from} → {inferResult.move.to})
                </span>
              </div>
              <div className="mb-2 font-mono text-[11px] text-zinc-400 break-all">
                {inferResult.updatedFen}
              </div>
              <button
                onClick={onAccept}
                className="rounded-md border border-emerald-500/40 bg-emerald-500/15 px-2.5 py-1 text-xs text-emerald-200 hover:bg-emerald-500/25"
              >
                Accept · use as next previous FEN
              </button>
            </>
          )}

          {inferResult.kind === "ambiguous" && (
            <>
              <div className="mb-1 text-xs uppercase tracking-wider text-amber-300">
                Ambiguous · {inferResult.candidates.length} candidates
              </div>
              <div className="text-zinc-300">
                {inferResult.candidates.map((m) => m.san).join(" · ")}
              </div>
              <div className="mt-2 text-xs text-zinc-500">
                Likely a promotion — occupancy alone can&apos;t tell Q/R/B/N
                apart.
              </div>
            </>
          )}

          {inferResult.kind === "none" && (
            <>
              <div className="mb-1 text-xs uppercase tracking-wider text-rose-300">
                No legal move matches the observed diff
              </div>
              <div className="mb-2 text-xs text-zinc-400">
                {inferResult.diff.length === 0
                  ? "Predicted occupancy is identical to the previous position."
                  : `Squares that changed: ${inferResult.diff
                      .map((d) => `${d.square} (${d.before}→${d.after})`)
                      .join(", ")}`}
              </div>
              <div className="text-[11px] text-zinc-500">
                Likely cause: a misclassified square. Inspect the per-square
                crops above — pieces with shadows or unusual lighting are
                the most common offenders.
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function UploadCard({
  onFileChange,
  hasImage,
}: {
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  hasImage: boolean;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-zinc-300">
          <span className="mr-2 inline-block rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs uppercase tracking-wider text-emerald-200">
            Step 1
          </span>
          Upload or capture a photo of the board.
        </div>
        <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/15 px-3 py-1.5 text-sm font-medium text-emerald-200 hover:bg-emerald-500/25">
          <input
            type="file"
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={onFileChange}
          />
          {hasImage ? "Choose another photo" : "Choose photo"}
        </label>
      </div>
    </div>
  );
}

function CornerOverlay({
  corners,
  imageDims,
}: {
  corners: Point[];
  imageDims: ImageDims;
}) {
  const stroke = Math.max(2, imageDims.w / 400);
  const dotR = Math.max(8, imageDims.w / 120);
  const fontSize = Math.max(16, imageDims.w / 40);
  const labelOffset = dotR * 2.2;
  return (
    <svg
      viewBox={`0 0 ${imageDims.w} ${imageDims.h}`}
      className="pointer-events-none absolute inset-0 h-full w-full"
    >
      {corners.length >= 2 && corners.length < 4 && (
        <polyline
          points={corners.map((p) => `${p.x},${p.y}`).join(" ")}
          fill="none"
          stroke="rgba(74,222,128,0.8)"
          strokeWidth={stroke}
        />
      )}
      {corners.length === 4 && (
        <polygon
          points={corners.map((p) => `${p.x},${p.y}`).join(" ")}
          fill="rgba(74,222,128,0.12)"
          stroke="rgba(74,222,128,0.9)"
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

/**
 * Tiny chess-board diagram with the currently-prompted corner highlighted.
 * Helps the user identify which physical corner of the photo to tap, no
 * matter how the photo is rotated.
 */
function CornerHintInset({ step }: { step: number }) {
  const size = 72;
  const sq = size / 8;
  const target = step < 4 ? CORNER_LABELS[step] : null;
  const targetCell: Record<(typeof CORNER_LABELS)[number], { f: number; r: number }> = {
    a8: { f: 0, r: 0 },
    h8: { f: 7, r: 0 },
    h1: { f: 7, r: 7 },
    a1: { f: 0, r: 7 },
  };
  const cells = [];
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
