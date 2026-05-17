"use client";

/**
 * Full-screen camera tab. The actual <video> element lives in Capture.tsx
 * and is positioned absolutely to fill this view — we just render the
 * overlays (camera switcher, re-lock button, ApiLogPanel) above it.
 */

import { CameraSwitcher } from "./CameraSwitcher";
import { ApiLogPanel } from "./ApiLogPanel";

export function CameraView({
  currentDeviceId,
  onPickCamera,
  onRelock,
  relockBusy,
}: {
  currentDeviceId: string | null;
  onPickCamera: (deviceId: string) => void;
  onRelock: () => void;
  relockBusy: boolean;
}) {
  return (
    <div className="absolute inset-0 flex flex-col pb-16">
      {/* Top-right camera switcher overlay (above the video) */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex justify-end p-3">
        <div className="pointer-events-auto">
          <CameraSwitcher
            currentDeviceId={currentDeviceId}
            onPick={onPickCamera}
          />
        </div>
      </div>

      {/* Bottom controls overlay */}
      <div className="pointer-events-none absolute inset-x-0 bottom-16 z-10 flex flex-col gap-3 p-3 sm:p-4">
        <div className="pointer-events-auto">
          <ApiLogPanel />
        </div>
        <div className="pointer-events-auto">
          <button
            onClick={onRelock}
            disabled={relockBusy}
            className="flex h-12 w-full items-center justify-center rounded-full bg-emerald-500 text-sm font-semibold text-emerald-950 shadow-xl transition hover:bg-emerald-400 disabled:opacity-50"
          >
            {relockBusy ? "Re-locking…" : "Re-lock the board"}
          </button>
        </div>
      </div>
    </div>
  );
}
