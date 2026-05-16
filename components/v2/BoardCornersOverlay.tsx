"use client";

import { useEffect, useRef } from "react";
import type { Corners } from "@/lib/v2/types";

/**
 * Overlay that draws the detected four corners on top of a still image
 * preview. We render to a `<canvas>` sized to match the displayed image
 * area so the dots line up exactly regardless of devicePixelRatio.
 *
 * On the live-preview phase we don't draw anything (would need to
 * project the cached corners through the current camera matrix). On
 * the verification phase we draw on the still calibration photo.
 */
export function BoardCornersOverlay({
  corners,
  imageW,
  imageH,
}: {
  corners: Corners | null;
  imageW: number;
  imageH: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !corners) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = imageW;
    canvas.height = imageH;
    ctx.clearRect(0, 0, imageW, imageH);

    // Draw the polygon outline first (subtle), then the four corner dots
    // (bright). Cyan = "locked," matches the in-app accent.
    ctx.lineWidth = Math.max(2, imageW / 320);
    ctx.strokeStyle = "rgba(94, 234, 212, 0.85)"; // teal-300 @ 85%
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(corners[i].x, corners[i].y);
    ctx.closePath();
    ctx.stroke();

    ctx.fillStyle = "rgba(94, 234, 212, 1)";
    const dotR = Math.max(4, imageW / 80);
    for (const c of corners) {
      ctx.beginPath();
      ctx.arc(c.x, c.y, dotR, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [corners, imageW, imageH]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 h-full w-full"
      style={{ objectFit: "contain" }}
    />
  );
}
