"use client";

import { forwardRef } from "react";

/**
 * Live camera preview. Critical UX point from real-world testing:
 * `object-contain` (NOT `object-cover`) so the FULL frame is visible
 * to the user. With `object-cover` the right side of landscape photos
 * gets cropped off; the user then thinks the board isn't fully captured
 * when in fact it just isn't being shown.
 *
 * The video element is the source of frames for the burst capture. We
 * don't render it muted in a way that disables audio rendering — chrome
 * needs muted=true for autoplay, but the BurstCamera.attach() handler
 * sets that programmatically too.
 */
export const LivePreview = forwardRef<
  HTMLVideoElement,
  {
    /** Optional overlay children (corner dots, framing guide, etc). */
    children?: React.ReactNode;
    className?: string;
  }
>(function LivePreview({ children, className }, ref) {
  return (
    <div
      className={`relative isolate flex h-full w-full items-center justify-center overflow-hidden bg-zinc-950 ${className ?? ""}`}
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
