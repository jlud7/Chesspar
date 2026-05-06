import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const src = path.join(root, "node_modules", "stockfish", "bin");
const dest = path.join(root, "public", "stockfish");

const files = ["stockfish-18-lite-single.js", "stockfish-18-lite-single.wasm"];

if (!fs.existsSync(src)) {
  console.warn(`[copy-stockfish] ${src} missing — skipping (run after npm install).`);
  process.exit(0);
}

fs.mkdirSync(dest, { recursive: true });
for (const name of files) {
  const from = path.join(src, name);
  const to = path.join(dest, name);
  if (!fs.existsSync(from)) {
    console.error(`[copy-stockfish] missing source: ${from}`);
    process.exit(1);
  }
  fs.copyFileSync(from, to);
  const bytes = fs.statSync(to).size;
  console.log(`[copy-stockfish] ${name} (${(bytes / 1024 / 1024).toFixed(1)} MB)`);
}
