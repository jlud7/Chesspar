/**
 * PGN export — accumulates SAN moves into a compliant PGN string.
 *
 * Kept tiny and dependency-free: chess.js already builds PGNs internally
 * but bundles a lot of move-history machinery we don't need. For the
 * "watch a board and write moves" loop, the move list is just a flat
 * array of SAN strings (optionally with clock annotations).
 */

import type { GameMode } from "./types.ts";

export type GameMeta = {
  white?: string;
  black?: string;
  event?: string;
  site?: string;
  date?: string;
  /** Result: "1-0" | "0-1" | "1/2-1/2" | "*" (in-progress). */
  result?: string;
  /** Optional game mode. Emitted as the standard [TimeControl ...] header
   *  when timed; omitted otherwise. */
  gameMode?: GameMode;
};

/** A move plus the player's clock reading AFTER they played it (ms). */
export type AnnotatedMove = {
  san: string;
  /** Remaining ms for the player who just moved. Used to emit
   *  `{[%clk H:MM:SS]}` per PGN convention. Omit for untimed games. */
  remainingMs?: number;
};

export function buildPgn(
  moves: Array<string | AnnotatedMove>,
  meta: GameMeta = {},
): string {
  const tags: Array<[string, string]> = [
    ["Event", meta.event ?? "Chesspar OTB"],
    ["Site", meta.site ?? "?"],
    ["Date", meta.date ?? new Date().toISOString().slice(0, 10).replace(/-/g, ".")],
    ["Round", "?"],
    ["White", meta.white ?? "White"],
    ["Black", meta.black ?? "Black"],
    ["Result", meta.result ?? "*"],
  ];
  if (meta.gameMode?.kind === "timed") {
    const base = Math.round(meta.gameMode.baseMs / 1000);
    const inc = Math.round(meta.gameMode.incrementMs / 1000);
    tags.push(["TimeControl", `${base}+${inc}`]);
  }
  const header = tags.map(([k, v]) => `[${k} "${v}"]`).join("\n");
  const body: string[] = [];
  const normalize = (m: string | AnnotatedMove): AnnotatedMove =>
    typeof m === "string" ? { san: m } : m;
  for (let i = 0; i < moves.length; i += 2) {
    const num = i / 2 + 1;
    const white = normalize(moves[i]);
    const black = i + 1 < moves.length ? normalize(moves[i + 1]) : null;
    const whiteStr = formatMove(white);
    const blackStr = black ? formatMove(black) : null;
    body.push(blackStr ? `${num}. ${whiteStr} ${blackStr}` : `${num}. ${whiteStr}`);
  }
  body.push(meta.result ?? "*");
  return `${header}\n\n${body.join(" ")}`;
}

function formatMove(m: AnnotatedMove): string {
  if (m.remainingMs == null) return m.san;
  return `${m.san} {[%clk ${formatClk(m.remainingMs)}]}`;
}

function formatClk(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
