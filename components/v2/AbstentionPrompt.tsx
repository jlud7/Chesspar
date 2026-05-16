"use client";

import type { MoveCandidate } from "@/lib/v2/types";

/**
 * Two-tap correction UI. Per the PDFs, when the pipeline abstains we
 * present the top-2 candidates (with their corresponding rectified
 * board previews) and let the user tap one. This is the production
 * fix for ~1% of moves and beats a wrong silent PGN every time.
 */
export function AbstentionPrompt({
  candidates,
  onPick,
  onCancel,
  rectifiedDataUrl,
}: {
  candidates: MoveCandidate[];
  onPick: (san: string) => void;
  onCancel: () => void;
  /** The post-move rectified board, shown to help the user resolve the move. */
  rectifiedDataUrl?: string;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-zinc-950/70 backdrop-blur-sm sm:items-center">
      <div className="m-3 w-full max-w-md rounded-2xl border border-white/10 bg-zinc-900 p-5 shadow-2xl">
        <header className="mb-3 flex items-center justify-between">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-amber-300">
              Confirm
            </p>
            <h3 className="mt-1 text-lg font-semibold text-zinc-50">
              Which move was played?
            </h3>
          </div>
          <button
            onClick={onCancel}
            className="rounded-full bg-white/5 px-3 py-1 text-[11px] uppercase tracking-widest text-zinc-400 transition hover:bg-white/10"
          >
            Cancel
          </button>
        </header>

        {rectifiedDataUrl && (
          <div className="mb-4 overflow-hidden rounded-xl border border-white/5">
            <img
              src={rectifiedDataUrl}
              alt="Rectified board after the move"
              className="block w-full"
            />
          </div>
        )}

        <ul className="grid gap-2">
          {candidates.slice(0, 3).map((c) => (
            <li key={c.san}>
              <button
                onClick={() => onPick(c.san)}
                className="flex w-full items-center justify-between rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-left transition hover:border-emerald-400/40 hover:bg-emerald-400/10"
              >
                <span className="font-mono text-base text-zinc-100">{c.san}</span>
                <span className="font-mono text-xs text-zinc-500">
                  {c.fromSquare}→{c.toSquare}
                </span>
              </button>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-[12px] leading-snug text-zinc-500">
          Picked the wrong one earlier? Capture again — Chesspar always
          shows you the move it inferred before adding to the score.
        </p>
      </div>
    </div>
  );
}
