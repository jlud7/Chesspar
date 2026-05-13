/**
 * Quick debug script: ask Claude to describe what it sees in one photo,
 * with no UNREADABLE escape hatch — to understand why it's bailing.
 */
import sharp from "sharp";
import path from "path";

const WORKER_URL =
  process.env.WORKER_URL ?? "https://chesspar-vlm.jamesleoluddy.workers.dev";
const ORIGIN = process.env.ORIGIN ?? "https://jlud7.github.io";
const MODEL = process.env.MODEL ?? "claude-sonnet-4-6";

const PROMPT =
  process.argv[2] ||
  `Describe what you see in this image. In particular: (1) Is there a chessboard? (2) What is the orientation of the board — which side is white, which side is black? (3) Can you make out the position of the pieces? Be brief.`;

const PHOTO = process.argv[3] || "Test_Photos/IMG_8819.jpeg";

async function main() {
  const buf = await sharp(path.resolve(PHOTO))
    .rotate()
    .resize(1024, 1024, { fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 85 })
    .toBuffer();
  const b64 = buf.toString("base64");
  const resp = await fetch(WORKER_URL.replace(/\/$/, "") + "/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: ORIGIN },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: Number(process.env.MAX_TOKENS ?? "4000"),
      ...(MODEL.includes("opus") ? {} : { temperature: 0.05 }),
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: "image/jpeg",
                data: b64,
              },
            },
            { type: "text", text: PROMPT },
          ],
        },
      ],
    }),
  });
  if (!resp.ok) {
    const t = await resp.text();
    console.error(`HTTP ${resp.status}: ${t}`);
    process.exit(1);
  }
  const data = (await resp.json()) as {
    content?: { type: string; text?: string }[];
  };
  const raw = data.content?.find((c) => c.type === "text")?.text ?? "";
  console.log("--- prompt ---");
  console.log(PROMPT);
  console.log("\n--- photo ---");
  console.log(PHOTO);
  console.log("\n--- response ---");
  console.log(raw);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
