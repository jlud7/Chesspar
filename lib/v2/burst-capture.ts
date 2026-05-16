/**
 * Burst capture — getUserMedia → 5 frames in ~200 ms → Laplacian argmax.
 *
 * The +1.5 pp gain from "multi-frame Laplacian-sharpest" capture in the
 * research PDFs comes from rejecting motion-blurred frames. We don't need
 * a continuous video stream for that — a short burst on the user's tap
 * is enough and uses ~0 battery between moves.
 *
 * Public surface: `BurstCamera.attach(video)` then `capture()` returns a
 * `CapturedBurst` with the sharpest frame already selected.
 */

import type { CapturedBurst } from "./types";

export class BurstCamera {
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;

  /**
   * Bind to a `<video>` element and start the camera stream. Asks for the
   * back camera ({facingMode: "environment"}) which is what users will
   * point at the board on a phone; on desktop the browser falls back to
   * whichever camera the user grants.
   *
   * Resolves once the first frame has played so callers can capture
   * immediately without seeing a black frame.
   */
  async attach(video: HTMLVideoElement): Promise<void> {
    this.video = video;
    const constraints: MediaStreamConstraints = {
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 30 },
      },
    };
    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = this.stream;
    video.playsInline = true;
    video.muted = true;
    await video.play();
    await waitForPlaying(video);
  }

  /**
   * Detach and release the camera. Call this on unmount or when the user
   * exits the capture flow — leaving the stream alive drains battery and
   * keeps the camera indicator on.
   */
  detach(): void {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
      this.video = null;
    }
  }

  /**
   * Capture a burst of `count` frames spaced `intervalMs` apart, then
   * pick the one with the highest Laplacian variance. Reject if the best
   * variance is below `minVariance` (caller decides what to do).
   *
   * Frames are downscaled to `maxDim` for the variance calculation but
   * the returned canvas is full resolution.
   */
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

function grabFrame(v: HTMLVideoElement, w: number, h: number): HTMLCanvasElement {
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

/**
 * Cheap blur detector: variance of a 3×3 Laplacian applied to the
 * grayscale image. Higher = sharper. Threshold ~100 on a 480×480 crop
 * separates "phone-still" from "moving-while-shutter" on typical OTB
 * captures.
 *
 * Implemented inline rather than pulled from OpenCV.js because it runs
 * once per burst frame (5×) on ~480² pixels — ~1 ms per call. Loading
 * OpenCV (10+ MB) for this is overkill.
 */
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
  // Grayscale buffer.
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    g[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }
  // Laplacian kernel: [0, 1, 0 / 1, -4, 1 / 0, 1, 0]. Skip 1-px border.
  let sum = 0;
  let sumSq = 0;
  let n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = g[i - w] + g[i + w] + g[i - 1] + g[i + 1] - 4 * g[i];
      sum += lap;
      sumSq += lap * lap;
      n++;
    }
  }
  if (n === 0) return 0;
  const mean = sum / n;
  return sumSq / n - mean * mean;
}
