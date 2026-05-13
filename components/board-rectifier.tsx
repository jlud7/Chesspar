"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { extractSquareCrops, warpBoard } from "@/lib/board-image";
import type { Point } from "@/lib/homography";

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
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const imageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

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
    setWarpedUrl(null);
    setSquareUrls([]);
    setError(null);
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
    setWarpedUrl(null);
    setSquareUrls([]);
    setError(null);
  }

  function undoCorner() {
    setCorners((c) => c.slice(0, -1));
    setWarpedUrl(null);
    setSquareUrls([]);
    setError(null);
  }

  function resetCorners() {
    setCorners([]);
    setWarpedUrl(null);
    setSquareUrls([]);
    setError(null);
  }

  const compute = useCallback(async () => {
    const img = imageRef.current;
    if (!img || corners.length !== 4) return;
    setBusy(true);
    setError(null);
    try {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const warped = warpBoard(
        img,
        corners as [Point, Point, Point, Point],
        RECTIFIED_SIZE,
      );
      const crops = extractSquareCrops(warped);
      setWarpedUrl(warped.toDataURL("image/png"));
      setSquareUrls(crops.map((c) => c.toDataURL("image/png")));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [corners]);

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
          <div className="mb-2 text-xs uppercase tracking-wider text-zinc-400">
            Per-square crops · a8 top-left, h1 bottom-right
          </div>
          <div className="grid w-full max-w-md grid-cols-8 gap-0.5">
            {squareUrls.map((src, i) => {
              const file = "abcdefgh"[i % 8];
              const rank = 8 - Math.floor(i / 8);
              return (
                <div
                  key={i}
                  className="relative aspect-square overflow-hidden bg-zinc-950"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={src}
                    alt={`${file}${rank}`}
                    className="block h-full w-full object-cover"
                  />
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
