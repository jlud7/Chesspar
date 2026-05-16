/**
 * Burst capture + camera control.
 *
 * On iPhones there are 2–3 back cameras (ultra-wide 0.5x, main 1x, and
 * sometimes telephoto 2x/3x). For OTB chess the phone needs to be just
 * above the board — main camera (1x) is usually too zoomed for that
 * distance. We enumerate `videoinput` devices, label them by their
 * focal range, and default to the widest back camera so the user can
 * stand the phone close to the board and still capture the whole thing.
 *
 * Public surface:
 *   BurstCamera.listBackCameras()      — array of {deviceId, label, role}
 *   BurstCamera.attach(video, opts)    — bind to a <video>, start stream
 *   BurstCamera.switchTo(deviceId)     — hot-swap to a different camera
 *   BurstCamera.setZoom(z)             — try `applyConstraints({zoom})` on
 *                                        supported devices; safe no-op
 *                                        otherwise
 *   BurstCamera.capture({count, ...})  — N-frame burst, returns sharpest
 */

import type { CapturedBurst } from "./types";

export type CameraRole = "ultrawide" | "wide" | "telephoto" | "unknown";

export type CameraDevice = {
  deviceId: string;
  label: string;
  role: CameraRole;
};

export class BurstCamera {
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private currentDeviceId: string | null = null;

  /**
   * Enumerate back-facing cameras. iOS exposes them with labels like
   * "Back Ultra Wide Camera" / "Back Camera" / "Back Telephoto Camera"
   * AFTER the user has granted permission once. Before permission the
   * labels are blank — caller should retry after `attach()` resolves.
   */
  static async listBackCameras(): Promise<CameraDevice[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const all = await navigator.mediaDevices.enumerateDevices();
    return all
      .filter((d) => d.kind === "videoinput")
      .filter((d) => {
        const l = (d.label || "").toLowerCase();
        // No label yet (pre-permission) — include everything; we'll
        // re-fetch after grant.
        if (!l) return true;
        // Exclude front cameras when we can.
        return !/front|facetime|user/.test(l);
      })
      .map((d) => ({
        deviceId: d.deviceId,
        label: d.label || "Camera",
        role: roleFromLabel(d.label || ""),
      }));
  }

  /**
   * Bind to a <video> element. If `opts.deviceId` is provided we use
   * that specific camera; otherwise we ask for the back camera with a
   * preference for ultra-wide (so the user can fit a full board into
   * the frame from a short distance above it).
   */
  async attach(
    video: HTMLVideoElement,
    opts: { deviceId?: string } = {},
  ): Promise<void> {
    this.video = video;
    await this.openStream(opts.deviceId);
  }

  private async openStream(deviceId?: string): Promise<void> {
    if (this.stream) {
      for (const t of this.stream.getTracks()) t.stop();
      this.stream = null;
    }
    let constraints: MediaStreamConstraints;
    if (deviceId) {
      constraints = {
        audio: false,
        video: {
          deviceId: { exact: deviceId },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 },
        },
      };
    } else {
      constraints = {
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 },
        },
      };
    }
    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.currentDeviceId = this.stream.getVideoTracks()[0]?.getSettings()?.deviceId ?? null;
    if (this.video) {
      this.video.srcObject = this.stream;
      this.video.playsInline = true;
      this.video.muted = true;
      await this.video.play();
      await waitForPlaying(this.video);
    }
    // If we used `facingMode` and the resulting device isn't the
    // widest back camera, try to switch over. iOS often returns the
    // main 1x camera by default even when ultra-wide exists.
    if (!deviceId) {
      try {
        await this.preferUltraWide();
      } catch {
        /* non-fatal */
      }
    }
  }

  private async preferUltraWide(): Promise<void> {
    const cams = await BurstCamera.listBackCameras();
    if (cams.length <= 1) return;
    const ultra = cams.find((c) => c.role === "ultrawide");
    if (!ultra) return;
    if (ultra.deviceId === this.currentDeviceId) return;
    // Open the ultra-wide stream and check the resulting track is
    // actually distinct (iOS sometimes returns the same physical
    // device under multiple deviceIds).
    try {
      await this.openStream(ultra.deviceId);
    } catch {
      /* keep current stream */
    }
  }

  /** Hot-swap to a specific camera. */
  async switchTo(deviceId: string): Promise<void> {
    await this.openStream(deviceId);
  }

  /** Current camera's deviceId, if known. */
  get deviceId(): string | null {
    return this.currentDeviceId;
  }

  /**
   * Best-effort zoom control. iOS Safari 17+ supports
   * `applyConstraints({ advanced: [{ zoom }] })` on the main back
   * camera. Returns the actual zoom value applied (or null if
   * unsupported).
   */
  async setZoom(zoom: number): Promise<number | null> {
    const track = this.stream?.getVideoTracks()?.[0];
    if (!track) return null;
    type ZoomCapabilities = MediaTrackCapabilities & {
      zoom?: { min: number; max: number; step: number };
    };
    const caps = (track.getCapabilities?.() ?? {}) as ZoomCapabilities;
    if (!caps.zoom) return null;
    const z = clamp(zoom, caps.zoom.min, caps.zoom.max);
    try {
      await track.applyConstraints({
        advanced: [{ zoom: z } as MediaTrackConstraintSet],
      });
      return z;
    } catch {
      return null;
    }
  }

  detach(): void {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
      this.video = null;
    }
    this.currentDeviceId = null;
  }

  async capture(opts: {
    count: number;
    intervalMs: number;
    minVariance?: number;
    maxDim?: number;
  }): Promise<CapturedBurst | null> {
    if (!this.video) throw new Error("BurstCamera not attached");
    const v = this.video;
    const w = v.videoWidth;
    const h = v.videoHeight;
    if (!w || !h) return null;
    const capturedAt = Date.now();
    const frames: HTMLCanvasElement[] = [];
    const variances: number[] = [];
    for (let i = 0; i < opts.count; i++) {
      const c = grabFrame(v, w, h);
      frames.push(c);
      variances.push(
        laplacianVariance(c, Math.min(opts.maxDim ?? 480, w, h)),
      );
      if (i < opts.count - 1) await delay(opts.intervalMs);
    }
    const ranked = frames
      .map((c, i) => ({ c, v: variances[i] }))
      .sort((a, b) => b.v - a.v);
    const best = ranked[0];
    if (opts.minVariance && best.v < opts.minVariance) return null;
    return {
      frame: best.c,
      variance: best.v,
      capturedAt,
      rankedFrames: ranked.map((r) => r.c),
    };
  }
}

/**
 * Heuristic mapping from device.label to a camera role. Apple's labels
 * are stable: "Back Ultra Wide Camera", "Back Camera", "Back Telephoto
 * Camera", "Back Dual Camera", "Back Triple Camera". Android labels
 * vary; we keep a wider net.
 */
function roleFromLabel(label: string): CameraRole {
  const l = label.toLowerCase();
  if (/ultra[- ]?wide|0\.5|wide-angle/.test(l)) return "ultrawide";
  if (/telephoto|tele|zoom|2x|3x|5x/.test(l)) return "telephoto";
  if (/back|rear|environment|main/.test(l)) return "wide";
  return "unknown";
}

function waitForPlaying(v: HTMLVideoElement): Promise<void> {
  return new Promise((resolve) => {
    if (v.readyState >= 2 && !v.paused) {
      resolve();
      return;
    }
    const handler = () => {
      v.removeEventListener("playing", handler);
      resolve();
    };
    v.addEventListener("playing", handler);
  });
}

function grabFrame(
  v: HTMLVideoElement,
  w: number,
  h: number,
): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(v, 0, 0, w, h);
  return c;
}

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function laplacianVariance(
  canvas: HTMLCanvasElement,
  maxDim: number,
): number {
  const scale = Math.min(1, maxDim / Math.max(canvas.width, canvas.height));
  const w = Math.max(2, Math.round(canvas.width * scale));
  const h = Math.max(2, Math.round(canvas.height * scale));
  const tmp = document.createElement("canvas");
  tmp.width = w;
  tmp.height = h;
  const ctx = tmp.getContext("2d");
  if (!ctx) return 0;
  ctx.drawImage(canvas, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    g[i] =
      0.299 * data[i * 4] +
      0.587 * data[i * 4 + 1] +
      0.114 * data[i * 4 + 2];
  }
  let sum = 0;
  let sumSq = 0;
  let n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap =
        g[i - w] + g[i + w] + g[i - 1] + g[i + 1] - 4 * g[i];
      sum += lap;
      sumSq += lap * lap;
      n++;
    }
  }
  if (n === 0) return 0;
  const mean = sum / n;
  return sumSq / n - mean * mean;
}
