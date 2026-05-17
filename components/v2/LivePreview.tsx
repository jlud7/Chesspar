"use client";

import { forwardRef, type CSSProperties, type ReactNode } from "react";

/**
 * Live camera preview. The <video> element is mounted exactly once by
 * the parent Capture component and stays mounted for the entire session
 * — its size and position are controlled by the parent via `style` so
 * we can resize/reposition between tabs without remounting and losing
 * the stream.
 *
 * `object-contain` (NOT `object-cover`) so the FULL camera frame is
 * visible. With cover the right side of landscape frames gets cropped.
 */
export const LivePreview = forwardRef<
  HTMLVideoElement,
  {
    /** Optional overlay children rendered inside the preview frame. */
    children?: ReactNode;
    /** Parent-controlled CSS — size, position, border radius, z-index. */
    style?: CSSProperties;
    className?: string;
  }
>(function LivePreview({ children, style, className }, ref) {
  return (
    <div
      className={`relative isolate flex items-center justify-center overflow-hidden bg-zinc-950 ${className ?? ""}`}
      style={style}
    >
      <video
        ref={ref}
        autoPlay
        playsInline
        muted
        className="h-full w-full object-contain"
      />
      {children}
    </div>
  );
});
