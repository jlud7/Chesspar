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
  const match = legal.find((m) => uciOf(m) === uci);
  if (!match) {
    // Try matching by SAN as a fallback — some replies use SAN for the
    // "uci" field by mistake.
    const sanMatch = legal.find((m) => m.san === uci || m.san === sanFromModel);
    if (!sanMatch) {
      return {
        kind: "error",
        reason: `Model returned "${uci}" which isn't legal. ${reason}`,
        durationMs: call.durationMs,
      };
    }
    return {
      kind: "matched",
      san: sanMatch.san,
      uci: uciOf(sanMatch),
      confidence,
      reason,
      durationMs: call.durationMs,
    };
  }

  return {
    kind: "matched",
    san: match.san,
    uci: uciOf(match),
    confidence,
    reason,
    durationMs: call.durationMs,
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
