#!/usr/bin/env tsx
/**
 * SAM-2 speed + mask probe.
 *
 * Posts a Test_Photos chess board image to Replicate (synchronous mode) and
 * saves the resulting combined_mask + individual_masks alongside timing
 * metrics. The point is to see how much of the wall time is real inference
 * (`metrics.predict_time`) vs. cold-start + network — that determines
 * whether SAM-2 is fast enough for live use or only for a ground-truth
 * pipeline.
 *
 * Usage:
 *   REPLICATE_API_TOKEN=<token> npx tsx scripts/test-sam2.ts [imagePath]
 *
 * Defaults to the first image in Test_Photos/ if no path is supplied.
 */

import {
  readFileSync,
  writeFileSync,
  readdirSync,
  mkdirSync,
} from "node:fs";
import { basename, extname, join } from "node:path";

const SAM2_VERSION =
  "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83";
const REPLICATE_URL = "https://api.replicate.com/v1/predictions";

type Sam2Output = {
  combined_mask: string;
  individual_masks?: string[];
};

type Prediction = {
  id: string;
  status: "starting" | "processing" | "succeeded" | "failed" | "canceled";
  output?: Sam2Output;
  error?: string | null;
  metrics?: { predict_time?: number };
};

async function main(): Promise<void> {
  const token = process.env.REPLICATE_API_TOKEN;
  if (!token) {
    console.error("REPLICATE_API_TOKEN env var is required");
    process.exit(1);
  }

  const imagePath = process.argv[2] ?? pickFirstTestPhoto();
  console.log(`> Image:        ${imagePath}`);
  const imgBytes = readFileSync(imagePath).byteLength;
  console.log(`> Size:         ${(imgBytes / 1024).toFixed(0)} KB`);
  const dataUri = toDataUri(imagePath);

  const t0 = performance.now();
  const initial = await postPrediction(token, dataUri);
  const final = await ensureCompleted(initial, token);
  const wallMs = performance.now() - t0;

  if (final.status !== "succeeded" || !final.output) {
    console.error(
      `Prediction failed: status=${final.status} error=${final.error ?? "n/a"}`,
    );
    process.exit(1);
  }

  const predictMs = (final.metrics?.predict_time ?? 0) * 1000;
  const masks = final.output.individual_masks ?? [];
  console.log("");
  console.log(`> Wall time:    ${wallMs.toFixed(0)} ms`);
  console.log(`> predict_time: ${predictMs.toFixed(0)} ms (pure inference)`);
  console.log(
    `> overhead:     ${(wallMs - predictMs).toFixed(0)} ms (cold start + network + upload)`,
  );
  console.log(`> Mask count:   ${masks.length}`);

  const outDir = join(
    "scripts",
    "detection-test",
    "sam2",
    basename(imagePath, extname(imagePath)),
  );
  mkdirSync(outDir, { recursive: true });
  await downloadTo(final.output.combined_mask, join(outDir, "combined.png"));
  for (let i = 0; i < masks.length; i++) {
    await downloadTo(
      masks[i],
      join(outDir, `mask_${String(i).padStart(2, "0")}.png`),
    );
  }
  console.log(`> Saved →       ${outDir}`);
}

function pickFirstTestPhoto(): string {
  const dir = "Test_Photos";
  const entries = readdirSync(dir)
    .filter((f) => /\.(jpe?g|png)$/i.test(f))
    .sort();
  if (entries.length === 0) {
    throw new Error(`No images found in ${dir}/`);
  }
  return join(dir, entries[0]);
}

function toDataUri(path: string): string {
  const buf = readFileSync(path);
  const ext = extname(path).toLowerCase();
  const mime =
    ext === ".png" ? "image/png" : ext === ".webp" ? "image/webp" : "image/jpeg";
  return `data:${mime};base64,${buf.toString("base64")}`;
}

async function postPrediction(
  token: string,
  imageDataUri: string,
): Promise<Prediction> {
  const res = await fetch(REPLICATE_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      // Block on the server side up to ~60s instead of polling immediately.
      Prefer: "wait",
    },
    body: JSON.stringify({
      version: SAM2_VERSION,
      input: { image: imageDataUri },
    }),
  });
  if (!res.ok) {
    throw new Error(`Replicate HTTP ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as Prediction;
}

/**
 * Prefer: wait times out at ~60s. If the prediction is still running when
 * control returns, fall back to polling until it terminates.
 */
async function ensureCompleted(
  initial: Prediction,
  token: string,
): Promise<Prediction> {
  let pred = initial;
  while (
    pred.status !== "succeeded" &&
    pred.status !== "failed" &&
    pred.status !== "canceled"
  ) {
    await new Promise((r) => setTimeout(r, 1000));
    const r = await fetch(
      `https://api.replicate.com/v1/predictions/${pred.id}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    if (!r.ok) {
      throw new Error(`Poll HTTP ${r.status}: ${await r.text()}`);
    }
    pred = (await r.json()) as Prediction;
  }
  return pred;
}

async function downloadTo(url: string, dest: string): Promise<void> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`download ${url} → HTTP ${r.status}`);
  const buf = Buffer.from(await r.arrayBuffer());
  writeFileSync(dest, buf);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
