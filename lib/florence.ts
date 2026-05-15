/**
 * Florence-2 client — one-shot chessboard localisation.
 *
 * We use Microsoft's Florence-2 (hosted on Replicate via the Cloudflare
 * worker proxy at /replicate/florence2) as the calibration-time
 * playing-surface localiser. In ~300 ms it returns a tight bbox of the
 * board, which gives downstream corner-detection a clean crop with no
 * table / lamp / hand to confuse it.
 *
 * The model is invoked through the "Caption to Phrase Grounding" task:
 * we pass a sentence; Florence grounds each noun phrase to a region and
 * returns one bbox per phrase. The prompt is tuned to elicit a single
 * "chessboard" bbox — across 15 test photos this returns a bbox with
 * 0–2 px variance.
 */
export type FlorenceBbox = {
  /** Image-pixel coordinates of the bbox in the ORIGINAL canvas. */
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** The Florence-supplied phrase that this bbox was grounded against. */
  label: string;
};

export type FlorenceBboxResult =
  | { kind: "detected"; bbox: FlorenceBbox; raw: string }
  | { kind: "error"; reason: string };

const PROMPT = "a chessboard with chess pieces on it";

/**
 * Post a canvas to the worker's /replicate/florence2 proxy and return the
 * first bbox in the response (parsed back into the original canvas's pixel
 * space — the lucataco wrapper returns coords already scaled to the input
 * image).
 *
 * `proxyUrl` is the same Cloudflare Worker root the rest of the app uses
 * (e.g. `https://chesspar-vlm.<account>.workers.dev`). We append
 * `/replicate/florence2`. The browser never sees the Replicate token —
 * the worker injects it server-side.
 */
export async function getChessboardBbox(
  image: HTMLCanvasElement,
  proxyUrl: string,
): Promise<FlorenceBboxResult> {
  try {
    const endpoint = proxyUrl.replace(/\/$/, "") + "/replicate/florence2";
    const dataUrl = image.toDataURL("image/jpeg", 0.9);
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input: {
          image: dataUrl,
          task_input: "Caption to Phrase Grounding",
          text_input: PROMPT,
        },
      }),
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      return {
        kind: "error",
        reason: `Proxy HTTP ${response.status}: ${text.slice(0, 160)}`,
      };
    }
    const data = (await response.json()) as {
      output?: { img?: string; text?: string } | unknown;
      error?: string | null;
      status?: string;
    };
    if (data.status === "failed") {
      return { kind: "error", reason: data.error ?? "Replicate failed" };
    }
    return parseFlorenceOutput(data.output);
  } catch (e) {
    return {
      kind: "error",
      reason: e instanceof Error ? e.message : String(e),
    };
  }
}

/**
 * The lucataco/florence-2-large wrapper returns
 *   { img: "<rendered overlay URL>", text: "<Python-dict-style JSON string>" }
 * The `text` field is a Python repr (single quotes), so we normalise to
 * JSON before parsing. We only care about the first bbox under the
 * <CAPTION_TO_PHRASE_GROUNDING> key — that's the "chessboard" phrase.
 */
function parseFlorenceOutput(raw: unknown): FlorenceBboxResult {
  let obj: unknown = raw;
  if (obj && typeof obj === "object" && "text" in (obj as Record<string, unknown>)) {
    obj = (obj as { text: unknown }).text;
  }
  if (typeof obj === "string") {
    const sentinel = obj;
    try {
      obj = JSON.parse(sentinel);
    } catch {
      try {
        obj = JSON.parse(sentinel.replace(/'/g, '"'));
      } catch {
        return {
          kind: "error",
          reason: `Could not parse Florence text: ${sentinel.slice(0, 160)}`,
        };
      }
    }
  }
  if (!obj || typeof obj !== "object") {
    return { kind: "error", reason: "Florence returned non-object output" };
  }
  const root = obj as Record<string, unknown>;
  const grounding =
    (root["<CAPTION_TO_PHRASE_GROUNDING>"] as
      | Record<string, unknown>
      | undefined) ?? root;
  const bboxes = (grounding.bboxes ?? []) as unknown;
  const labels = (grounding.labels ?? []) as unknown;
  if (
    !Array.isArray(bboxes) ||
    bboxes.length === 0 ||
    !Array.isArray((bboxes as number[][])[0]) ||
    (bboxes as number[][])[0].length !== 4
  ) {
    return { kind: "error", reason: "No bbox in Florence output" };
  }
  const [x1, y1, x2, y2] = (bboxes as number[][])[0];
  const label = Array.isArray(labels) ? String(labels[0] ?? "") : "";
  return {
    kind: "detected",
    bbox: { x1, y1, x2, y2, label },
    raw: JSON.stringify(obj),
  };
}
