#!/usr/bin/env tsx
/**
 * Validate Florence-2 as the calibration step across all 15 Test_Photos.
 *
 * For each photo: get the "chessboard" bbox from Florence-2, save the
 * cropped board, and save an annotated original. The point is to confirm
 * visually that Florence consistently localises the playing surface tightly
 * enough that downstream per-cell analysis becomes geometrically simple.
 *
 * Outputs land in scripts/detection-test/florence2-all/:
 *   summary.json     — per-photo bbox + timing
 *   <IMG>_anno.png   — original with bbox overlaid
 *   <IMG>_crop.jpg   — bbox-cropped board
 *
 * Usage:
 *   REPLICATE_API_TOKEN=<token> npx tsx scripts/test-florence-crops-all.ts
 */

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { basename, extname, join } from "node:path";
import { createCanvas, loadImage } from "canvas";

const REPLICATE_PREDICTIONS = "https://api.replicate.com/v1/predictions";
const FLORENCE_VERSION =
  "da53547e17d45b9cfb48174b2f18af8b83ca020fa76db62136bf9c6616762595";
const PROMPT = "a chessboard with chess pieces on it";

const TEST_DIR = "Test_Photos";
const OUT_DIR = join("scripts", "detection-test", "florence2-all");

type Prediction = {
  id: string;
  status: "starting" | "processing" | "succeeded" | "failed" | "canceled";
  output?: { img?: string; text?: string } | unknown;
  error?: string | null;
  metrics?: { predict_time?: number };
};

type Result = {
  image: string;
  bbox: [number, number, number, number] | null;
  label: string | null;
  predictMs: number;
  wallMs: number;
  cropAspect: number | null;
  imageWidth: number;
  imageHeight: number;
};

async function main(): Promise<void> {
  const token = process.env.REPLICATE_API_TOKEN;
  if (!token) {
    console.error("REPLICATE_API_TOKEN required");
    process.exit(1);
  }

  mkdirSync(OUT_DIR, { recursive: true });
  const photos = readdirSync(TEST_DIR)
    .filter((f) => /\.(jpe?g|png)$/i.test(f))
    .sort();
  console.log(`> ${photos.length} photos found.`);

  const results: Result[] = [];
  for (const photo of photos) {
    const path = join(TEST_DIR, photo);
    process.stdout.write(`> ${photo.padEnd(20)} … `);
    try {
      const res = await processOne(path, token);
      results.push(res);
      const bb = res.bbox;
      if (bb) {
        console.log(
          `bbox [${bb.map((n) => n.toFixed(0).padStart(4)).join(",")}], ` +
            `${res.predictMs.toFixed(0)}ms, ` +
            `aspect ${res.cropAspect?.toFixed(2)}`,
        );
      } else {
        console.log(`NO BOX RETURNED`);
      }
    } catch (e) {
      console.log(`FAILED ${e instanceof Error ? e.message : e}`);
    }
  }

  writeFileSync(
    join(OUT_DIR, "summary.json"),
    JSON.stringify(results, null, 2),
  );

  // Quick consistency check: every photo in this sequence was shot with
  // the same camera position, so the bbox dimensions should be near
  // identical. Big variance = Florence is unstable on this scene.
  const widths = results.filter((r) => r.bbox).map((r) => r.bbox![2] - r.bbox![0]);
  const heights = results.filter((r) => r.bbox).map((r) => r.bbox![3] - r.bbox![1]);
  if (widths.length > 0) {
    console.log("");
    console.log(
      `> bbox width  : min=${Math.min(...widths).toFixed(0)} max=${Math.max(...widths).toFixed(0)} (variance ${(Math.max(...widths) - Math.min(...widths)).toFixed(0)}px)`,
    );
    console.log(
      `> bbox height : min=${Math.min(...heights).toFixed(0)} max=${Math.max(...heights).toFixed(0)} (variance ${(Math.max(...heights) - Math.min(...heights)).toFixed(0)}px)`,
    );
  }
  console.log(`> Saved → ${OUT_DIR}`);
}

async function processOne(imagePath: string, token: string): Promise<Result> {
  const dataUri = toDataUri(imagePath);
  const t0 = performance.now();
  const res = await fetch(REPLICATE_PREDICTIONS, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      Prefer: "wait",
    },
    body: JSON.stringify({
      version: FLORENCE_VERSION,
      input: {
        image: dataUri,
        task_input: "Caption to Phrase Grounding",
        text_input: PROMPT,
      },
    }),
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  let pred = (await res.json()) as Prediction;
  while (
    pred.status !== "succeeded" &&
    pred.status !== "failed" &&
    pred.status !== "canceled"
  ) {
    await new Promise((r) => setTimeout(r, 500));
    const r = await fetch(
      `https://api.replicate.com/v1/predictions/${pred.id}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    pred = (await r.json()) as Prediction;
  }
  const wallMs = performance.now() - t0;
  if (pred.status !== "succeeded") {
    throw new Error(`status=${pred.status} err=${pred.error ?? "n/a"}`);
  }

  const { bbox, label } = parseFirstBox(pred.output);
  const img = await loadImage(imagePath);

  // 1. Write the bbox-annotated original.
  const stem = basename(imagePath, extname(imagePath));
  if (bbox) {
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    ctx.strokeStyle = "lime";
    ctx.lineWidth = Math.max(4, Math.round(img.width / 350));
    ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
    writeFileSync(
      join(OUT_DIR, `${stem}_anno.png`),
      canvas.toBuffer("image/png"),
    );

    // 2. Write the actual cropped image — this is what downstream CV /
    //    rectification would see.
    const cw = Math.round(bbox[2] - bbox[0]);
    const ch = Math.round(bbox[3] - bbox[1]);
    const cropCanvas = createCanvas(cw, ch);
    const cctx = cropCanvas.getContext("2d");
    cctx.drawImage(img, bbox[0], bbox[1], cw, ch, 0, 0, cw, ch);
    writeFileSync(
      join(OUT_DIR, `${stem}_crop.jpg`),
      cropCanvas.toBuffer("image/jpeg", { quality: 0.9 }),
    );
  }

  return {
    image: stem,
    bbox,
    label,
    predictMs: (pred.metrics?.predict_time ?? 0) * 1000,
    wallMs,
    cropAspect: bbox ? (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) : null,
    imageWidth: img.width,
    imageHeight: img.height,
  };
}

function parseFirstBox(
  raw: unknown,
): { bbox: [number, number, number, number] | null; label: string | null } {
  // Replicate wraps as { img, text } where text is a Python-dict-style string.
  let inner: unknown = raw;
  if (inner && typeof inner === "object" && "text" in inner) {
    inner = (inner as { text: unknown }).text;
  }
  if (typeof inner === "string") {
    try {
      inner = JSON.parse(inner.replace(/'/g, '"'));
    } catch {
      return { bbox: null, label: null };
    }
  }
  if (!inner || typeof inner !== "object") return { bbox: null, label: null };
  const root = inner as Record<string, unknown>;
  const grounding = (root["<CAPTION_TO_PHRASE_GROUNDING>"] ?? root) as Record<
    string,
    unknown
  >;
  const bboxes = (grounding.bboxes ?? []) as number[][];
  const labels = (grounding.labels ?? []) as string[];
  if (bboxes.length === 0) return { bbox: null, label: null };
  return {
    bbox: [bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3]],
    label: labels[0] ?? null,
  };
}

function toDataUri(path: string): string {
  const buf = readFileSync(path);
  const ext = extname(path).toLowerCase();
  const mime = ext === ".png" ? "image/png" : "image/jpeg";
  return `data:${mime};base64,${buf.toString("base64")}`;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
