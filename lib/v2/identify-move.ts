/**
 * One-call per-move classifier. Send the rectified pre-move board and the
 * rectified post-move board plus the list of legal SAN moves; ask
 * gemini-3-flash which legal move was played. No CV diff. No abstention
 * dance. Trust the VLM with a constrained, top-down view.
 */

import { Chess, type Move } from "chess.js";
import { callVlm, parseJsonLoose } from "./vlm.ts";

export type IdentifyResult =
  | {
      kind: "matched";
      san: string;
      uci: string;
      confidence: number;
      reason: string;
      durationMs: number;
    }
  | {
      kind: "abstain";
      reason: string;
      durationMs: number;
    }
  | {
      kind: "error";
      reason: string;
      durationMs: number;
    };

export async function identifyMove(opts: {
  proxyUrl: string;
  previousFen: string;
  /** Rectified canvas BEFORE the move (top-down view, white at bottom). */
  preImage: HTMLCanvasElement;
  /** Rectified canvas AFTER the move. */
  postImage: HTMLCanvasElement;
  /** Optional: the original camera frame BEFORE the move. */
  rawPreImage?: HTMLCanvasElement;
  /** Optional: the original camera frame AFTER the move. Sent alongside the
   *  rectified pair so the model can see actual 3D piece silhouettes —
   *  bishops vs knights are clearer at the camera angle than top-down. */
  rawPostImage?: HTMLCanvasElement;
  /** Optional Origin override (Node-side tests only). */
  origin?: string;
}): Promise<IdentifyResult> {
  const game = new Chess(opts.previousFen);
  const legal = game.moves({ verbose: true });
  if (legal.length === 0) {
    return {
      kind: "error",
      reason: "No legal moves from the previous position — game is already over.",
      durationMs: 0,
    };
  }

  if (legal.length === 1) {
    const only = legal[0];
    return {
      kind: "matched",
      san: only.san,
      uci: uciOf(only),
      confidence: 1,
      reason: "Only legal move",
      durationMs: 0,
    };
  }

  const sideToMove = game.turn() === "w" ? "White" : "Black";
  const legalLines = legal
    .map((m) => `${uciOf(m)} (${m.san})`)
    .join("\n");

  const rawPrePart = opts.rawPreImage
    ? `\n- Image 3 is the original camera photo BEFORE the move (not rectified). Use it to see actual 3D piece silhouettes: bishops have a slanted/mitred top, knights have a horse-head profile, pawns are shortest with a round head.`
    : "";
  const rawPostPart = opts.rawPostImage
    ? `\n- Image ${opts.rawPreImage ? "4" : "3"} is the original camera photo AFTER the move (not rectified). Cross-check the moved piece's type and final square against this view.`
    : "";
  const imageCount = 2 + (opts.rawPreImage ? 1 : 0) + (opts.rawPostImage ? 1 : 0);
  const imageWord =
    imageCount === 2 ? "Two" : imageCount === 3 ? "Three" : "Four";

  const prompt = `${imageWord} photos of the same physical chessboard.

- Image 1: BEFORE position, top-down rectified, white at the bottom (ranks 1-2) and black at the top (ranks 7-8). Files a-h run left-to-right.
- Image 2: AFTER position, same view.${rawPrePart}${rawPostPart}

${sideToMove} just made one legal move. Identify it.

Previous position FEN:
${opts.previousFen}

Legal moves (UCI and SAN):
${legalLines}

Procedure (do these IN ORDER and report each in the JSON below):

1. **Enumerate the after-image, rank by rank.** For each of the 8 ranks (8, 7, 6, 5, 4, 3, 2, 1 from the top), list what is on each of the 8 files (a, b, c, d, e, f, g, h from the left) in Image 2. Use one-character notation: K Q R B N P (white) or k q r b n p (black) or . (empty). This forces you to look at every square instead of anchoring on a guess. Cross-reference Image ${opts.rawPostImage ? (opts.rawPreImage ? 4 : 3) : "2"} for piece-type confirmation — bishop has a slanted/mitred top with a slit, knight is horse-shaped and asymmetric, pawn is shortest with a round head.

2. **Find the changed squares** by comparing your enumeration to the previous FEN's piece placement. Expect exactly: one FROM-square (was occupied in BEFORE, empty in AFTER), one TO-square (now holds the moved piece). Plus the rook's two squares for castling, or the captured pawn's square for en-passant.

3. **Pick the legal move** whose from/to/piece-type match. The UCI you return MUST appear verbatim in the legal moves list above.

4. If your enumeration shows no change (images identical) or no single legal move fits, return ABSTAIN.

Return ONLY this JSON (no preamble, no markdown, no extra keys):
{"rank_8":"<8 chars>","rank_7":"<8 chars>","rank_6":"<8 chars>","rank_5":"<8 chars>","rank_4":"<8 chars>","rank_3":"<8 chars>","rank_2":"<8 chars>","rank_1":"<8 chars>","changed_squares":["<square1>","<square2>"],"uci":"<uci from list, or ABSTAIN>","san":"<matching san, or ABSTAIN>","confidence":0.0,"reason":"<one short sentence>"}`;

  const images: string[] = [
    opts.preImage.toDataURL("image/jpeg", 0.92),
    opts.postImage.toDataURL("image/jpeg", 0.92),
  ];
  if (opts.rawPreImage) {
    images.push(opts.rawPreImage.toDataURL("image/jpeg", 0.88));
  }
  if (opts.rawPostImage) {
    images.push(opts.rawPostImage.toDataURL("image/jpeg", 0.88));
  }

  const call = await callVlm({
    proxyUrl: opts.proxyUrl,
    callName: "identify-move",
    prompt,
    images,
    origin: opts.origin,
  });

  if (call.kind === "error") {
    return { kind: "error", reason: call.reason, durationMs: call.durationMs };
  }

  const parsed = parseJsonLoose<ParsedMove>(call.text);
  if (!parsed) {
    return {
      kind: "error",
      reason: `Couldn't parse model JSON. Raw: ${call.text.slice(0, 200)}`,
      durationMs: call.durationMs,
    };
  }

  const uci = (parsed.uci ?? "").trim();
  const sanFromModel = (parsed.san ?? "").trim();
  const confidence = clamp01(typeof parsed.confidence === "number" ? parsed.confidence : 0);
  const reason = (parsed.reason ?? "").trim();

  if (uci === "ABSTAIN" || sanFromModel === "ABSTAIN") {
    return {
      kind: "abstain",
      reason: reason || "Model couldn't determine the move.",
      durationMs: call.durationMs,
    };
  }

  // Verify the returned UCI is actually legal in this position.
  let picked: Move | undefined = legal.find((m) => uciOf(m) === uci);
  if (!picked) {
    // Try matching by SAN as a fallback — some replies use SAN for the
    // "uci" field by mistake.
    picked = legal.find((m) => m.san === uci || m.san === sanFromModel);
    if (!picked) {
      return {
        kind: "error",
        reason: `Model returned "${uci}" which isn't legal. ${reason}`,
        durationMs: call.durationMs,
      };
    }
  }

  // ---- Self-consistency check ----
  // The model's own reported `changed_squares` should mention both the
  // from-square and the to-square of the move it picked. When they
  // don't, the model has been internally inconsistent — typically a
  // visual reasoning step disagreeing with the move-selection step.
  // Abstain rather than commit a contradictory pick.
  const reportedSquares = normalizeSquares(parsed.changed_squares);
  if (reportedSquares.length > 0) {
    const expected = expectedTouchedSquares(picked);
    const missing = expected.filter((sq) => !reportedSquares.includes(sq));
    if (missing.length > 0) {
      return {
        kind: "abstain",
        reason: `Model picked ${picked.san} but its reported changes ${formatSquares(reportedSquares)} don't include ${formatSquares(missing)}. Re-snap.`,
        durationMs: call.durationMs,
      };
    }
  }

  // ---- CV pixel-diff guard + retry ----
  // Independent check: compute the mean absolute RGB delta between pre
  // and post images on each of the 64 squares. If the model's picked
  // move's from/to squares show LOW pixel change vs the rest of the
  // board, the model has hallucinated (e.g. picking Qf7# in a
  // mate-in-1 position when the played move was actually a pawn push).
  // Don't just abstain — retry with explicit feedback so we can self-
  // correct without bothering the user.
  const cv = checkPixelDiff(opts.preImage, opts.postImage, picked);
  if (cv && !cv.ok) {
    const retry = await retryWithCvFeedback({
      opts,
      images,
      legal,
      picked,
      cvSummary: cv.summary,
      lowDeltaSquares: cv.lowDeltaSquares,
    });
    return {
      ...retry,
      durationMs: call.durationMs + retry.durationMs,
    };
  }

  return {
    kind: "matched",
    san: picked.san,
    uci: uciOf(picked),
    confidence,
    reason,
    durationMs: call.durationMs,
  };
}

/**
 * Second attempt after CV catches the first pick as a hallucination.
 * Tell the model which squares ACTUALLY changed (per CV) and which
 * squares it claimed changed but didn't. Restrict the candidate list
 * to legal moves whose from/to are in the CV-confirmed changed set,
 * with the original pick filtered out so the model can't just repeat
 * its mistake.
 */
async function retryWithCvFeedback(args: {
  opts: Parameters<typeof identifyMove>[0];
  images: string[];
  legal: Move[];
  picked: Move;
  cvSummary: string;
  lowDeltaSquares: string[];
}): Promise<IdentifyResult> {
  const { opts, images, legal, picked, cvSummary, lowDeltaSquares } = args;
  const cvDeltas = computeSquareDeltas(opts.preImage, opts.postImage);
  if (!cvDeltas) {
    return {
      kind: "abstain",
      reason: `Model picked ${picked.san} but pixels at ${formatSquares(lowDeltaSquares)} didn't change. Re-snap.`,
      durationMs: 0,
    };
  }
  const sorted = [...cvDeltas.entries()].sort((a, b) => b[1] - a[1]);
  const topChanged = new Set(sorted.slice(0, 6).map(([sq]) => sq));
  const constrained = legal.filter((m) => {
    if (uciOf(m) === uciOf(picked)) return false; // exclude the rejected pick
    const touched = expectedTouchedSquares(m);
    return touched.every((sq) => topChanged.has(sq));
  });
  if (constrained.length === 0) {
    return {
      kind: "abstain",
      reason: `Model picked ${picked.san} but pixels at ${formatSquares(lowDeltaSquares)} didn't change, and no other legal move fits the visual change. Re-snap.`,
      durationMs: 0,
    };
  }
  const legalLines = constrained
    .map((m) => `${uciOf(m)} (${m.san})`)
    .join("\n");
  const topSquaresList = sorted
    .slice(0, 6)
    .map(([sq, d]) => `  ${sq}: delta=${Math.round(d)}`)
    .join("\n");
  const prompt = `Re-evaluation request. Your previous answer was ${picked.san}, but a pixel-level diff shows the board pixels at ${formatSquares(lowDeltaSquares)} barely changed (${cvSummary}). That means ${picked.san} can't be the move that was actually played.

The squares with the highest visible pixel change between Image 1 and Image 2 are:
${topSquaresList}

Pick one of these legal moves whose from-square AND to-square are both in the high-change list above:
${legalLines}

Look at the images again carefully — confirm the piece type by silhouette in the raw camera images (bishop=slanted top, knight=horse head, pawn=short with round head, rook=castellated, queen=tallest with crown, king=tallest with cross).

Return ONLY this JSON (no preamble, no markdown):
{"rank_8":"<8 chars>","rank_7":"<8 chars>","rank_6":"<8 chars>","rank_5":"<8 chars>","rank_4":"<8 chars>","rank_3":"<8 chars>","rank_2":"<8 chars>","rank_1":"<8 chars>","changed_squares":["<square1>","<square2>"],"uci":"<uci from list above, or ABSTAIN>","san":"<matching san, or ABSTAIN>","confidence":0.0,"reason":"<one short sentence>"}`;

  const call2 = await callVlm({
    proxyUrl: opts.proxyUrl,
    callName: "identify-move (retry)",
    prompt,
    images,
    origin: opts.origin,
  });
  if (call2.kind === "error") {
    return { kind: "error", reason: call2.reason, durationMs: call2.durationMs };
  }
  const parsed = parseJsonLoose<ParsedMove>(call2.text);
  if (!parsed) {
    return {
      kind: "error",
      reason: `Retry parse failed: ${call2.text.slice(0, 160)}`,
      durationMs: call2.durationMs,
    };
  }
  const uci = (parsed.uci ?? "").trim();
  const sanFromModel = (parsed.san ?? "").trim();
  if (uci === "ABSTAIN" || sanFromModel === "ABSTAIN") {
    return {
      kind: "abstain",
      reason: parsed.reason?.trim() || "Model abstained on retry.",
      durationMs: call2.durationMs,
    };
  }
  const picked2 =
    constrained.find((m) => uciOf(m) === uci) ??
    constrained.find((m) => m.san === uci || m.san === sanFromModel);
  if (!picked2) {
    return {
      kind: "abstain",
      reason: `Retry returned "${uci}" not in the constrained list. Re-snap.`,
      durationMs: call2.durationMs,
    };
  }
  const confidence = clamp01(
    typeof parsed.confidence === "number" ? parsed.confidence : 0,
  );
  // Re-verify the retry pick against CV. If it ALSO fails, abstain
  // rather than loop forever.
  const cv2 = checkPixelDiff(opts.preImage, opts.postImage, picked2);
  if (cv2 && !cv2.ok) {
    return {
      kind: "abstain",
      reason: `Retry picked ${picked2.san} which still doesn't match the visual change (${cv2.summary}). Re-snap.`,
      durationMs: call2.durationMs,
    };
  }
  return {
    kind: "matched",
    san: picked2.san,
    uci: uciOf(picked2),
    confidence,
    reason: (parsed.reason ?? "").trim() || "Re-evaluated with CV feedback.",
    durationMs: call2.durationMs,
  };
}

type ParsedMove = {
  uci?: string;
  san?: string;
  changed_squares?: string[];
  confidence?: number;
  reason?: string;
};

function uciOf(m: Move): string {
  return m.from + m.to + (m.promotion ?? "");
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

/**
 * Every square that should have visibly changed for the given move:
 * from + to, plus rook squares for castling and the captured pawn for
 * en-passant. The model's reported `changed_squares` should mention all
 * of these for the move to be self-consistent.
 */
function expectedTouchedSquares(m: Move): string[] {
  const sq = new Set<string>([m.from, m.to]);
  if (m.san === "O-O") {
    if (m.color === "w") {
      sq.add("h1");
      sq.add("f1");
    } else {
      sq.add("h8");
      sq.add("f8");
    }
  } else if (m.san === "O-O-O") {
    if (m.color === "w") {
      sq.add("a1");
      sq.add("d1");
    } else {
      sq.add("a8");
      sq.add("d8");
    }
  } else if (m.flags.includes("e")) {
    // En-passant: captured pawn is on the same file as `to`, same rank as `from`.
    const file = m.to[0];
    const rank = m.from[1];
    sq.add(`${file}${rank}`);
  }
  return [...sq];
}

function normalizeSquares(input: unknown): string[] {
  if (!Array.isArray(input)) return [];
  const out: string[] = [];
  for (const v of input) {
    if (typeof v !== "string") continue;
    const s = v.trim().toLowerCase();
    if (/^[a-h][1-8]$/.test(s)) out.push(s);
  }
  return out;
}

function formatSquares(sqs: string[]): string {
  if (sqs.length === 0) return "(none)";
  return sqs.join(", ");
}

/**
 * Compute per-square mean absolute RGB difference between the pre and
 * post rectified canvases. If the squares the picked move SHOULD have
 * touched have visibly less change than the rest of the board, the
 * model has hallucinated a move that didn't physically happen. Returns
 * null when canvases are incomparable (different sizes) and the guard
 * is skipped.
 *
 * Threshold: an expected square is "suspiciously low" if its delta is
 * less than `max(MIN_EXPECTED_DELTA, baseline + DELTA_FLOOR_DELTA)`,
 * where baseline is the median delta across the 56 non-expected
 * squares. Both bars must be met for an abstain — this avoids false
 * abstains on noisy frames where the whole board has high background
 * delta.
 */
/**
 * Per-square mean absolute RGB delta between pre and post rectified
 * images. Returns a Map<square, delta> for all 64 squares, or null if
 * the canvases are incomparable. Used both as a model HINT (best signal —
 * prevents the model anchoring on strategically-expected moves) and as
 * a post-hoc abstention guard.
 */
function computeSquareDeltas(
  pre: HTMLCanvasElement,
  post: HTMLCanvasElement,
): Map<string, number> | null {
  if (pre.width !== post.width || pre.height !== post.height) return null;
  const size = post.width;
  if (post.height !== size) return null;
  const pctx = pre.getContext("2d", { willReadFrequently: true });
  const actx = post.getContext("2d", { willReadFrequently: true });
  if (!pctx || !actx) return null;

  const margin = 0.14;
  const pad = size * margin;
  const grid = size - 2 * pad;
  const cell = grid / 8;
  const inset = cell * 0.2;
  const sampleSize = Math.max(4, Math.round(cell - 2 * inset));

  const deltas = new Map<string, number>();
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const x = Math.round(pad + col * cell + inset);
      const y = Math.round(pad + row * cell + inset);
      const square = indexToSquare(row * 8 + col);
      deltas.set(square, meanAbsRgbDiff(pctx, actx, x, y, sampleSize, sampleSize));
    }
  }
  return deltas;
}

function checkPixelDiff(
  pre: HTMLCanvasElement,
  post: HTMLCanvasElement,
  picked: Move,
): { ok: boolean; lowDeltaSquares: string[]; summary: string } | null {
  const deltas = computeSquareDeltas(pre, post);
  if (!deltas) return null;

  const expected = expectedTouchedSquares(picked);
  const baselineValues: number[] = [];
  for (const [sq, d] of deltas) {
    if (!expected.includes(sq)) baselineValues.push(d);
  }
  baselineValues.sort((a, b) => a - b);
  const baseline = baselineValues[Math.floor(baselineValues.length / 2)] ?? 0;

  const MIN_EXPECTED_DELTA = 8; // absolute floor regardless of baseline
  const DELTA_FLOOR_OVER_BASELINE = 5;
  const threshold = Math.max(MIN_EXPECTED_DELTA, baseline + DELTA_FLOOR_OVER_BASELINE);

  const lowDeltaSquares: string[] = [];
  for (const sq of expected) {
    const d = deltas.get(sq) ?? 0;
    if (d < threshold) lowDeltaSquares.push(sq);
  }
  const expectedDeltas = expected
    .map((sq) => `${sq}=${Math.round(deltas.get(sq) ?? 0)}`)
    .join(" ");
  return {
    ok: lowDeltaSquares.length === 0,
    lowDeltaSquares,
    summary: `baseline=${Math.round(baseline)} threshold=${Math.round(threshold)} ${expectedDeltas}`,
  };
}

function meanAbsRgbDiff(
  pre: CanvasRenderingContext2D,
  post: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
): number {
  const a = pre.getImageData(x, y, w, h).data;
  const b = post.getImageData(x, y, w, h).data;
  let total = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    total +=
      Math.abs(a[i] - b[i]) +
      Math.abs(a[i + 1] - b[i + 1]) +
      Math.abs(a[i + 2] - b[i + 2]);
    n += 3;
  }
  return n > 0 ? total / n : 0;
}

function indexToSquare(idx: number): string {
  const file = String.fromCharCode("a".charCodeAt(0) + (idx % 8));
  const rank = 8 - Math.floor(idx / 8);
  return `${file}${rank}`;
}
