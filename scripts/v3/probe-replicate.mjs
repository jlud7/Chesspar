#!/usr/bin/env node
/**
 * Probe the Replicate google/gemini-3-flash schema via the Chesspar worker.
 * Outputs the raw response so we know the exact input/output shape before
 * pinning it in lib/v2/vlm.ts.
 *
 * Run: WORKER_URL=https://... node scripts/v3/probe-replicate.mjs
 */

import sharp from "sharp";
import { fileURLToPath } from "node:url";
import path from "node:path";

const WORKER_URL =
  process.env.WORKER_URL ?? "https://chesspar-vlm.jamesleoluddy.workers.dev";
const ORIGIN = process.env.ORIGIN ?? "https://jlud7.github.io";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const PHOTO = path.join(REPO_ROOT, "Test_Photos", "IMG_8819.jpeg");

async function imageDataUrl(file) {
  const buf = await sharp(file)
    .rotate()
    .resize(1024, 1024, { fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 85 })
    .toBuffer();
  return `data:image/jpeg;base64,${buf.toString("base64")}`;
}

async function tryShape(label, input) {
  process.stdout.write(`\n=== ${label} ===\n`);
  process.stdout.write(`INPUT FIELDS: ${Object.keys(input).join(", ")}\n`);
  const t0 = Date.now();
  let resp;
  try {
    resp = await fetch(WORKER_URL.replace(/\/$/, "") + "/replicate/vlm", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: ORIGIN },
      body: JSON.stringify({
        model: "google/gemini-3-flash",
        input,
      }),
    });
  } catch (e) {
    console.log(`network error: ${e.message}`);
    return;
  }
  const dt = Date.now() - t0;
  const text = await resp.text();
  console.log(`status=${resp.status} time=${dt}ms`);
  console.log(`body (first 1500 chars):`);
  console.log(text.slice(0, 1500));
  if (text.length > 1500) console.log(`... [${text.length - 1500} more chars]`);
}

const PROMPT = "Reply with the literal text PROBE_OK then a one-sentence description of what you see.";

(async () => {
  const dataUrl = await imageDataUrl(PHOTO);
  console.log(`Photo: ${PHOTO}`);
  console.log(`Worker: ${WORKER_URL}`);
  console.log(`Data URL bytes: ${dataUrl.length}`);

  // Shape A: { prompt, image } — most common Replicate VLM shape
  await tryShape("A: prompt + image", { prompt: PROMPT, image: dataUrl });

  // Shape B: { prompt, images: [image] } — some wrappers
  await tryShape("B: prompt + images[]", { prompt: PROMPT, images: [dataUrl] });

  // Shape C: { messages: [...] } — chat-style wrappers
  await tryShape("C: messages chat-style", {
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: PROMPT },
          { type: "image_url", image_url: { url: dataUrl } },
        ],
      },
    ],
  });
})();
