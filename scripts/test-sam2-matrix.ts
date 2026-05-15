#!/usr/bin/env tsx
/**
 * SAM-2 parameter sweep. Runs the same image through several configurations
 * to find where the speed/quality knee is. Saves each run's combined mask
 * under scripts/detection-test/sam2/<image>/<config>/combined.png for
 * side-by-side visual inspection, then prints a summary table.
 *
 * Usage:
 *   REPLICATE_API_TOKEN=<token> npx tsx scripts/test-sam2-matrix.ts
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { basename, extname, join } from "node:path";

const SAM2_VERSION =
  "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83";
const REPLICATE_URL = "https://api.replicate.com/v1/predictions";

type Config = { label: string; input: Record<string, unknown> };

const CONFIGS: Config[] = [
  // points_per_side controls the density of "click-anywhere" prompts SAM-2
  // generates in automatic mode. Inference scales ~linearly with this.
  { label: "fast-16", input: { points_per_side: 16 } },
  { label: "ultra-8", input: { points_per_side: 8 } },
  // Stricter thresholds: keep 32 sample points but only return masks with
  // very high predicted IoU and stability. Fewer junk masks, same speed.
  {
    label: "strict-32",
    input: {
      points_per_side: 32,
      pred_iou_thresh: 0.92,
      stability_score_thresh: 0.97,
    },
  },
];

const IMAGES = [
  { path: "Test_Photos/IMG_8819.jpeg", label: "start" },
  { path: "Test_Photos/IMG_8830.jpeg", label: "midgame-Nd5" },
];

type Prediction = {
  id: string;
  status: "starting" | "processing" | "succeeded" | "failed" | "canceled";
  output?: { combined_mask: string; individual_masks?: string[] };
  error?: string | null;
  metrics?: { predict_time?: number };
};

type Row = {
  image: string;
  config: string;
  wallMs: number;
  predictMs: number;
  maskCount: number;
};

async function runOne(
  imagePath: string,
  config: Config,
  token: string,
): Promise<Row> {
  const dataUri = toDataUri(imagePath);
  const t0 = performance.now();
  const res = await fetch(REPLICATE_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      Prefer: "wait",
    },
    body: JSON.stringify({
      version: SAM2_VERSION,
      input: { image: dataUri, ...config.input },
    }),
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  }
  let pred = (await res.json()) as Prediction;
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
    pred = (await r.json()) as Prediction;
  }
  const wallMs = performance.now() - t0;
  if (pred.status !== "succeeded" || !pred.output) {
    throw new Error(`Prediction failed: ${pred.error ?? pred.status}`);
  }
  const masks = pred.output.individual_masks ?? [];
  const outDir = join(
    "scripts",
    "detection-test",
    "sam2",
    basename(imagePath, extname(imagePath)),
    config.label,
  );
  mkdirSync(outDir, { recursive: true });
  await downloadTo(pred.output.combined_mask, join(outDir, "combined.png"));
  return {
    image: basename(imagePath, extname(imagePath)),
    config: config.label,
    wallMs,
    predictMs: (pred.metrics?.predict_time ?? 0) * 1000,
    maskCount: masks.length,
  };
}

function toDataUri(path: string): string {
  const buf = readFileSync(path);
  const ext = extname(path).toLowerCase();
  const mime = ext === ".png" ? "image/png" : "image/jpeg";
  return `data:${mime};base64,${buf.toString("base64")}`;
}

async function downloadTo(url: string, dest: string): Promise<void> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`download ${url} → HTTP ${r.status}`);
  const buf = Buffer.from(await r.arrayBuffer());
  writeFileSync(dest, buf);
}

async function main(): Promise<void> {
  const token = process.env.REPLICATE_API_TOKEN;
  if (!token) {
    console.error("REPLICATE_API_TOKEN required");
    process.exit(1);
  }
  const rows: Row[] = [];
  for (const img of IMAGES) {
    for (const cfg of CONFIGS) {
      process.stdout.write(
        `> ${img.label.padEnd(14)} ${cfg.label.padEnd(11)} … `,
      );
      try {
        const row = await runOne(img.path, cfg, token);
        rows.push(row);
        console.log(
          `${row.predictMs.toFixed(0).padStart(6)}ms pred, ${row.maskCount.toString().padStart(3)} masks`,
        );
      } catch (e) {
        console.log(`FAILED ${e instanceof Error ? e.message : e}`);
      }
    }
  }
  console.log("");
  console.log(
    "┌──────────────┬───────────┬───────────┬───────────┬───────┐",
  );
  console.log(
    "│ image        │ config    │ wall (ms) │ pred (ms) │ masks │",
  );
  console.log(
    "├──────────────┼───────────┼───────────┼───────────┼───────┤",
  );
  for (const r of rows) {
    console.log(
      `│ ${r.image.padEnd(12)} │ ${r.config.padEnd(9)} │ ${r.wallMs.toFixed(0).padStart(9)} │ ${r.predictMs.toFixed(0).padStart(9)} │ ${r.maskCount.toString().padStart(5)} │`,
    );
  }
  console.log(
    "└──────────────┴───────────┴───────────┴───────────┴───────┘",
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
