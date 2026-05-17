"use client";

/**
 * Bottom-sheet picker shown when the queue is paused after a
 * classification failure. Lists every legal move from the FEN
 * preceding the failed capture; tapping one commits the user's pick.
 *
 * Replaces the ugly window.prompt fallback so the user doesn't need
 * to know exact SAN syntax — just see the list and pick.
 */

import { useMemo } from "react";
import { Chess, type Move } from "chess.js";

export function ManualMovePicker({
  previousFen,
  onPick,
  onCancel,
}: {
  previousFen: string;
  onPick: (san: string) => void;
  onCancel: () => void;
}) {
  const grouped = useMemo(() => groupByPiece(previousFen), [previousFen]);
  const sideToMove = previousFen.split(/\s+/)[1] === "b" ? "Black" : "White";

  return (
    <div className="absolute inset-0 z-50 flex flex-col bg-zinc-950/95 backdrop-blur">
      <div className="flex shrink-0 items-center justify-between border-b border-white/5 px-4 py-3 sm:px-6">
        <span className="text-[11px] font-semibold uppercase tracking-[0.3em] text-amber-300">
          Pick {sideToMove}&apos;s move
        </span>
        <button
          onClick={onCancel}
          className="rounded-full bg-white/5 px-3 py-1 text-[10px] uppercase tracking-widest text-zinc-300 transition hover:bg-white/10"
        >
          Cancel
        </button>
      </div>
      <p className="px-4 pt-3 text-[12px] text-zinc-400 sm:px-6">
        The model couldn&apos;t identify this move. Tap the SAN below for the
        move you actually played, or cancel to drop/retry from the score
        sheet.
      </p>
      <div className="flex-1 overflow-y-auto px-3 pb-32 pt-3 sm:px-5">
        {grouped.length === 0 && (
          <div className="rounded-xl border border-white/5 bg-white/5 px-4 py-6 text-center text-sm text-zinc-500">
            No legal moves — game is over.
          </div>
        )}
        <div className="space-y-4">
          {grouped.map(({ label, moves }) => (
            <div key={label}>
              <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.28em] text-zinc-500">
                {label}
              </h3>
              <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
                {moves.map((m) => (
                  <button
                    key={`${m.from}${m.to}${m.promotion ?? ""}`}
                    onClick={() => onPick(m.san)}
                    className="rounded-xl border border-white/10 bg-white/5 px-3 py-2.5 font-mono text-sm text-zinc-100 transition hover:border-emerald-400/40 hover:bg-emerald-400/10"
                  >
                    {m.san}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const PIECE_ORDER = ["P", "N", "B", "R", "Q", "K"] as const;
const PIECE_LABELS: Record<string, string> = {
  P: "Pawn",
  N: "Knight",
  B: "Bishop",
  R: "Rook",
  Q: "Queen",
  K: "King",
};

function groupByPiece(
  previousFen: string,
): Array<{ label: string; moves: Move[] }> {
  let legal: Move[];
  try {
    legal = new Chess(previousFen).moves({ verbose: true });
  } catch {
    return [];
  }
  const buckets = new Map<string, Move[]>();
  for (const m of legal) {
    const key = m.piece.toUpperCase(); // p/n/b/r/q/k → P/N/B/R/Q/K
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(m);
  }
  const out: Array<{ label: string; moves: Move[] }> = [];
  for (const k of PIECE_ORDER) {
    const moves = buckets.get(k);
    if (!moves || moves.length === 0) continue;
    moves.sort((a, b) => a.san.localeCompare(b.san));
    out.push({ label: PIECE_LABELS[k], moves });
  }
  return out;
}
