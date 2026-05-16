"use client";

import { useEffect, useState } from "react";
import { BurstCamera, type CameraDevice } from "@/lib/v2/burst-capture";

/**
 * Floating chip in the top-right of the live preview. On iPhones with
 * multiple back cameras, lets the user pick ultra-wide / wide /
 * telephoto. Hides itself on devices with only one camera.
 *
 * `currentDeviceId` is the camera id BurstCamera ended up on after
 * `attach()`. We poll once on mount + once a second later (iOS only
 * surfaces device labels after permission is granted, which can lag).
 */
export function CameraSwitcher({
  currentDeviceId,
  onPick,
}: {
  currentDeviceId: string | null;
  onPick: (deviceId: string) => void;
}) {
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      const list = await BurstCamera.listBackCameras();
      if (!cancelled) setDevices(list);
    };
    refresh();
    const t = setTimeout(refresh, 1500);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, []);

  if (devices.length <= 1) return null;
  const current = devices.find((d) => d.deviceId === currentDeviceId);
  const label = current
    ? roleToLabel(current.role)
    : "Camera";

  return (
    <div className="absolute right-3 top-3 z-20 flex flex-col items-end gap-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="rounded-full bg-zinc-950/70 px-3 py-1.5 text-[11px] font-medium uppercase tracking-widest text-zinc-100 shadow-lg ring-1 ring-white/10 backdrop-blur transition hover:bg-zinc-900"
      >
        {label}
      </button>
      {open && (
        <div className="rounded-2xl border border-white/10 bg-zinc-900/95 p-1 shadow-2xl backdrop-blur">
          {devices.map((d) => (
            <button
              key={d.deviceId}
              onClick={() => {
                onPick(d.deviceId);
                setOpen(false);
              }}
              className={`block w-full rounded-xl px-3 py-2 text-left text-[12px] transition hover:bg-white/10 ${
                d.deviceId === currentDeviceId
                  ? "bg-white/10 text-emerald-300"
                  : "text-zinc-200"
              }`}
            >
              {roleToLabel(d.role)}
              <span className="ml-2 text-[10px] uppercase tracking-widest text-zinc-500">
                {d.role === "ultrawide"
                  ? "0.5x"
                  : d.role === "telephoto"
                    ? "2x+"
                    : "1x"}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function roleToLabel(role: CameraDevice["role"]): string {
  if (role === "ultrawide") return "Ultra-wide";
  if (role === "telephoto") return "Telephoto";
  if (role === "wide") return "Wide";
  return "Camera";
}
