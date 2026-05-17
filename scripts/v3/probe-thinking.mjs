#!/usr/bin/env node
/**
 * Probe whether Replicate's google/gemini-3-flash accepts thinking-budget
 * controls. We try a few candidate input field names and observe (a) whether
 * the API accepts them, and (b) whether latency drops + the "thought_signature"
 * warning disappears from the logs.
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
    .resize(768, 768, { fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 85 })
    .toBuffer();
  return `data:image/jpeg;base64,${buf.toString("base64")}`;
}

const PROMPT =
  "Describe the chessboard in one sentence. Mention orientation only.";

async function tryShape(label, extraInput) {
  process.stdout.write(`\n=== ${label} ===\n`);
  process.stdout.write(`EXTRA FIELDS: ${JSON.stringify(extraInput)}\n`);
  const dataUrl = await imageDataUrl(PHOTO);
  const t0 = Date.now();
  const resp = await fetch(WORKER_URL.replace(/\/$/, "") + "/replicate/vlm", {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: ORIGIN },
    body: JSON.stringify({
      model: "google/gemini-3-flash",
      input: { prompt: PROMPT, images: [dataUrl], ...extraInput },
    }),
  });
  const text = await resp.text();
  const dt = Date.now() - t0;
  console.log(`status=${resp.status} wallclock=${dt}ms`);
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    console.log(text.slice(0, 400));
    return;
  }
  const replicateStatus = json.status;
  const runTimeMs = json.metrics?.predict_time
    ? Math.round(json.metrics.predict_time * 1000)
    : "?";
  console.log(
    `replicate status=${replicateStatus} model predict_time=${runTimeMs}ms`,
  );
  const logs = (json.logs ?? "").toString();
  const hasThinkWarning = /thought_signature/i.test(logs);
  console.log(`thought_signature in logs: ${hasThinkWarning ? "YES (thinking active)" : "no"}`);
  console.log(`logs (last 240): ${logs.slice(-240)}`);
  console.log(
    `output: ${Array.isArray(json.output) ? json.output.join("").slice(0, 180) : String(json.output).slice(0, 180)}`,
  );
}

(async () => {
  console.log(`Worker: ${WORKER_URL}`);
  await tryShape("baseline: no extras", {});
  await tryShape("thinking_budget: 0", { thinking_budget: 0 });
  await tryShape("thinking: false", { thinking: false });
  await tryShape("reasoning_effort: low", { reasoning_effort: "low" });
  await tryShape("max_thinking_tokens: 0", { max_thinking_tokens: 0 });
  await tryShape("thinking_config / thinkingBudget: 0", {
    thinking_config: { thinking_budget: 0 },
  });
})();
