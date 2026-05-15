#!/usr/bin/env tsx
/**
 * Florence-2 open-vocabulary detection probe.
 *
 * For each (image, prompt) pair, posts to lucataco/florence-2-large on
 * Replicate, parses the returned bboxes, overlays them on the original
 * photo, and saves an annotated PNG. Goal: see whether a fast open-vocab
 * detector can answer the two questions SAM-2 couldn't —
 *   (a) "where is the playing surface?" (board localization)
 *   (b) "where is each piece?" (per-cell occupancy)
 * — at ~$0.0003 and ~1s per call instead of $0.005 and ~17s.
 *
 * Usage:
 *   REPLICATE_API_TOKEN=<token> npx tsx scripts/test-florence2.ts
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { basename, extname, join } from "node:path";
import { createCanvas, loadImage } from "canvas";

const REPLICATE_PREDICTIONS = "https://api.replicate.com/v1/predictions";
// Pinned latest version of lucataco/florence-2-large at the time of writing.
// We use "large" (770M) rather than "base" (230M) because the former is
// kept warm — base hit a 40s cold start on the first call. Bump via:
//   curl -s https://api.replicate.com/v1/models/lucataco/florence-2-large \
//     -H "Authorization: Bearer $TOKEN" | jq -r .latest_version.id
const FLORENCE_VERSION =
  "da53547e17d45b9cfb48174b2f18af8b83ca020fa76db62136bf9c6616762595";

const IMAGES = [
  { path: "Test_Photos/IMG_8819.jpeg", label: "start" },
  { path: "Test_Photos/IMG_8830.jpeg", label: "midgame-Nd5" },
];

/**
 * The Replicate wrapper doesn't expose <OPEN_VOCABULARY_DETECTION>, so we
 * use two adjacent tasks that get us the same information:
 *   - "Caption to Phrase Grounding": pass a sentence; Florence returns a
 *     bbox per noun phrase it can ground. This is how we ask "find the
 *     chessboard AND the chess pieces" in a single call.
 *   - "Object Detection": no text input; returns every COCO-ish object
 *     Florence recognises. Useful as a sanity check — does it call the
 *     board a "chessboard" on its own?
 * The internal output key in both cases is the matching task token:
 *   <CAPTION_TO_PHRASE_GROUNDING> or <OD>.
 */
const PROMPTS = [
  // Baseline that worked in the first probe — loose but reliable.
  {
    label: "board-loose",
    task: "Caption to Phrase Grounding",
    outputKey: "<CAPTION_TO_PHRASE_GROUNDING>",
    text: "a chessboard with chess pieces on it",
  },
  // Tighter phrasing: does Florence draw a smaller box if we ask for the
  // playing surface specifically (excluding the clock + table)?
  {
    label: "board-tight",
    task: "Caption to Phrase Grounding",
    outputKey: "<CAPTION_TO_PHRASE_GROUNDING>",
    text: "the 8x8 grid of red and white squares",
  },
  // Per-piece grounding. Phrase grounding returns one box per noun phrase
  // — if Florence can ground each colour separately, we may not need a
  // separate piece detector at all.
  {
    label: "pieces",
    task: "Caption to Phrase Grounding",
    outputKey: "<CAPTION_TO_PHRASE_GROUNDING>",
    text: "white pawns, black pawns, white pieces, black pieces",
  },
  // Auto-discover what Florence considers a distinct region — no prompt.
  {
    label: "dense",
    task: "Dense Region Caption",
    outputKey: "<DENSE_REGION_CAPTION>",
    text: "",
  },
  // Class-agnostic region proposals. Tells us what Florence "sees" as
  // foreground without imposing a label.
  {
    label: "regions",
    task: "Region Proposal",
    outputKey: "<REGION_PROPOSAL>",
    text: "",
  },
];

type Prediction = {
  id: string;
  status: "starting" | "processing" | "succeeded" | "failed" | "canceled";
  output?: unknown;
  error?: string | null;
  metrics?: { predict_time?: number };
};

type Box = { x1: number; y1: number; x2: number; y2: number; label: string };

type Row = {
  image: string;
  prompt: string;
  wallMs: number;
  predictMs: number;
  boxes: number;
};

async function main(): Promise<void> {
  const token = process.env.REPLICATE_API_TOKEN;
  if (!token) {
    console.error("REPLICATE_API_TOKEN required");
    process.exit(1);
  }

  const rows: Row[] = [];
  for (const img of IMAGES) {
    for (const pr of PROMPTS) {
      process.stdout.write(
        `> ${img.label.padEnd(14)} ${pr.label.padEnd(7)} … `,
      );
      try {
        const row = await runOne(img.path, pr, token);
        rows.push(row);
        console.log(
          `${row.predictMs.toFixed(0).padStart(5)}ms pred, ${row.boxes} boxes`,
        );
      } catch (e) {
        console.log(`FAILED ${e instanceof Error ? e.message : e}`);
      }
    }
  }

  console.log("");
  console.log(
    "┌──────────────┬─────────┬───────────┬───────────┬───────┐",
  );
  console.log(
    "│ image        │ prompt  │ wall (ms) │ pred (ms) │ boxes │",
  );
  console.log(
    "├──────────────┼─────────┼───────────┼───────────┼───────┤",
  );
  for (const r of rows) {
    console.log(
      `│ ${r.image.padEnd(12)} │ ${r.prompt.padEnd(7)} │ ${r.wallMs.toFixed(0).padStart(9)} │ ${r.predictMs.toFixed(0).padStart(9)} │ ${r.boxes.toString().padStart(5)} │`,
    );
  }
  console.log(
    "└──────────────┴─────────┴───────────┴───────────┴───────┘",
  );
}

async function runOne(
  imagePath: string,
  prompt: { label: string; task: string; outputKey: string; text: string },
  token: string,
): Promise<Row> {
  const dataUri = toDataUri(imagePath);
  const t0 = performance.now();
  // Object Detection takes no text. Send only what the task needs so the
  // wrapper doesn't reject the call with a validation error.
  const input: Record<string, unknown> = {
    image: dataUri,
    task_input: prompt.task,
  };
  if (prompt.text) input.text_input = prompt.text;
  const res = await fetch(REPLICATE_PREDICTIONS, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      Prefer: "wait",
    },
    body: JSON.stringify({ version: FLORENCE_VERSION, input }),
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
    await new Promise((r) => setTimeout(r, 750));
    const r = await fetch(
      `https://api.replicate.com/v1/predictions/${pred.id}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    pred = (await r.json()) as Prediction;
  }
  const wallMs = performance.now() - t0;
  if (pred.status !== "succeeded") {
    throw new Error(`status=${pred.status} error=${pred.error ?? "n/a"}`);
  }

  const boxes = parseBoxes(pred.output, prompt.outputKey);
  const outDir = join(
    "scripts",
    "detection-test",
    "florence2",
    basename(imagePath, extname(imagePath)),
  );
  mkdirSync(outDir, { recursive: true });
  // Also dump the raw output JSON so we can debug parsing later without
  // re-spending API calls.
  writeFileSync(
    join(outDir, `${prompt.label}.json`),
    JSON.stringify(pred.output, null, 2),
  );
  await drawAnnotated(imagePath, boxes, join(outDir, `${prompt.label}.png`));
  return {
    image: basename(imagePath, extname(imagePath)),
    prompt: prompt.label,
    wallMs,
    predictMs: (pred.metrics?.predict_time ?? 0) * 1000,
    boxes: boxes.length,
  };
}

/**
 * Florence-2 wrappers on Replicate aren't perfectly consistent about the
 * output shape. We try a few of the common forms before giving up.
 * Possible shapes:
 *   1) The raw Florence dict:
 *      { "<OPEN_VOCABULARY_DETECTION>": { bboxes, bboxes_labels, ... } }
 *   2) A JSON string of the above
 *   3) A wrapper object like { text: "{\"<OPEN…>\": …}" }
 *   4) Already-flattened arrays at the top level.
 */
function parseBoxes(raw: unknown, task: string): Box[] {
  let obj: unknown = raw;
  if (typeof obj === "string") {
    const sentinel = obj;
    try {
      obj = JSON.parse(sentinel);
    } catch {
      // Florence sometimes returns Python-dict-style with single quotes;
      // a last-ditch normalize lets us still recover the bboxes.
      try {
        obj = JSON.parse(sentinel.replace(/'/g, '"'));
      } catch {
        return [];
      }
    }
  }
  if (obj && typeof obj === "object" && "text" in (obj as Record<string, unknown>)) {
    return parseBoxes((obj as { text: unknown }).text, task);
  }
  if (!obj || typeof obj !== "object") return [];

  const root = obj as Record<string, unknown>;
  const inner =
    (root[task] as Record<string, unknown> | undefined) ??
    (root["bboxes"] !== undefined ? root : undefined);
  if (!inner) return [];

  const bboxes = (inner.bboxes ?? []) as number[][];
  const labels = (inner.bboxes_labels ??
    inner.labels ??
    []) as string[];
  return bboxes.map((b, i) => ({
    x1: b[0],
    y1: b[1],
    x2: b[2],
    y2: b[3],
    label: labels[i] ?? "",
  }));
}

async function drawAnnotated(
  srcPath: string,
  boxes: Box[],
  outPath: string,
): Promise<void> {
  const img = await loadImage(srcPath);
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  ctx.lineWidth = Math.max(3, Math.round(img.width / 400));
  ctx.font = `${Math.max(16, Math.round(img.width / 60))}px sans-serif`;
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    // Distinct colours per box so overlapping detections are still legible.
    const hue = (i * 47) % 360;
    ctx.strokeStyle = `hsl(${hue}, 100%, 55%)`;
    ctx.fillStyle = `hsl(${hue}, 100%, 55%)`;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    const tag = b.label ? `${b.label} #${i}` : `#${i}`;
    const m = ctx.measureText(tag);
    ctx.fillRect(b.x1, b.y1 - 26, m.width + 12, 26);
    ctx.fillStyle = "black";
    ctx.fillText(tag, b.x1 + 6, b.y1 - 7);
  }
  writeFileSync(outPath, canvas.toBuffer("image/png"));
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
