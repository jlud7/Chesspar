import { Chess, type Move } from "chess.js";
import type { Occupancy } from "./occupancy";

export type SquareDiff = {
  square: string;
  before: Occupancy;
  after: Occupancy;
};

export type InferResult =
  | { kind: "matched"; move: Move; updatedFen: string; diff: SquareDiff[] }
  | { kind: "ambiguous"; candidates: Move[]; diff: SquareDiff[] }
  | { kind: "none"; diff: SquareDiff[]; legalMoveCount: number };

/**
 * Render a FEN as a length-64 array of {empty,white,black}, indexed
 * row-major from a8 (idx 0) to h1 (idx 63). This matches the layout that
 * `extractSquareCrops` produces from a rectified board.
 */
export function fenToOccupancy(fen: string): Occupancy[] {
  const game = new Chess(fen);
  const board = game.board();
  const out: Occupancy[] = [];
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const cell = board[r][f];
      if (!cell) out.push("empty");
      else out.push(cell.color === "w" ? "white" : "black");
    }
  }
  return out;
}

export function diffOccupancy(
  before: Occupancy[],
  after: Occupancy[],
): SquareDiff[] {
  const out: SquareDiff[] = [];
  for (let i = 0; i < 64; i++) {
    if (before[i] !== after[i]) {
      const file = "abcdefgh"[i % 8];
      const rank = 8 - Math.floor(i / 8);
      out.push({ square: `${file}${rank}`, before: before[i], after: after[i] });
    }
  }
  return out;
}

/**
 * Given the FEN before the move and the observed post-move occupancy,
 * return the unique legal move whose result matches the observed
 * occupancy. For promotions, four candidates (Q/R/B/N) produce identical
 * occupancy — caller must pick the piece.
 */
export function inferMove(prevFen: string, observed: Occupancy[]): InferResult {
  if (observed.length !== 64) {
    throw new Error(`inferMove expects 64 occupancy entries, got ${observed.length}`);
  }
  const game = new Chess(prevFen);
  const prevOcc = fenToOccupancy(prevFen);
  const diff = diffOccupancy(prevOcc, observed);
  const legal = game.moves({ verbose: true });
  const matches: { move: Move; updatedFen: string }[] = [];

  for (const move of legal) {
    const sim = new Chess(prevFen);
    try {
      sim.move({ from: move.from, to: move.to, promotion: move.promotion });
    } catch {
      continue;
    }
    const expected = fenToOccupancy(sim.fen());
    if (occupancyEquals(observed, expected)) {
      matches.push({ move, updatedFen: sim.fen() });
    }
  }

  if (matches.length === 1) {
    return {
      kind: "matched",
      move: matches[0].move,
      updatedFen: matches[0].updatedFen,
      diff,
    };
  }
  if (matches.length > 1) {
    return { kind: "ambiguous", candidates: matches.map((m) => m.move), diff };
  }
  return { kind: "none", diff, legalMoveCount: legal.length };
}

function occupancyEquals(a: Occupancy[], b: Occupancy[]): boolean {
  for (let i = 0; i < 64; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}
