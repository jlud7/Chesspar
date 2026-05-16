/**
 * PGN export — accumulates SAN moves into a compliant PGN string.
 *
 * Kept tiny and dependency-free: chess.js already builds PGNs internally
 * but bundles a lot of move-history machinery we don't need. For the
 * "watch a board and write moves" loop, the move list is just a flat
 * array of SAN strings.
 */

export type GameMeta = {
  white?: string;
  black?: string;
  event?: string;
  site?: string;
  date?: string;
  /** Result: "1-0" | "0-1" | "1/2-1/2" | "*" (in-progress). */
  result?: string;
};

export function buildPgn(moves: string[], meta: GameMeta = {}): string {
  const tags: Array<[string, string]> = [
    ["Event", meta.event ?? "Chesspar OTB"],
    ["Site", meta.site ?? "?"],
    ["Date", meta.date ?? new Date().toISOString().slice(0, 10).replace(/-/g, ".")],
    ["Round", "?"],
    ["White", meta.white ?? "White"],
    ["Black", meta.black ?? "Black"],
    ["Result", meta.result ?? "*"],
  ];
  const header = tags.map(([k, v]) => `[${k} "${v}"]`).join("\n");
  const body: string[] = [];
  for (let i = 0; i < moves.length; i += 2) {
    const num = i / 2 + 1;
    const white = moves[i];
    const black = moves[i + 1];
    body.push(black ? `${num}. ${white} ${black}` : `${num}. ${white}`);
  }
  body.push(meta.result ?? "*");
  return `${header}\n\n${body.join(" ")}`;
}
