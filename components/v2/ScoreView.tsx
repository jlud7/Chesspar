"use client";

/**
 * Running scoresheet. Columns: # | White | Black | White Δ | Black Δ.
 *
 * Committed moves render normally; queued/inflight captures render as
 * italic "…" rows with a spinner so the user sees the queue depth even
 * here. A failure banner appears at the top if the queue is paused;
 * tapping "Pick move manually" surfaces the ManualMovePicker overlay.
 */

import { useEffect, useRef, useState } from "react";
import type { MoveEntry, PendingCapture, Side } from "@/lib/v2/types";
import { describeGameMode, formatDelta } from "@/lib/v2/clock";
import type { GameMode } from "@/lib/v2/types";
import { ManualMovePicker } from "./ManualMovePicker";

export type ScoreFailure = {
  reason: string;
  /** FEN BEFORE the failed move — feeds the legal-moves picker. */
  previousFen: string;
};

export function ScoreView({
  history,
  pending,
  inflight,
  failure,
  mode,
  onApplyFailureGuess,
  onDropFailure,
  onRetryFailure,
}: {
  history: MoveEntry[];
  pending: PendingCapture[];
  inflight: PendingCapture | null;
  failure: ScoreFailure | null;
  mode: GameMode;
  onApplyFailureGuess?: (san: string) => void;
  onDropFailure?: () => void;
  onRetryFailure?: () => void;
}) {
  const [pickerOpen, setPickerOpen] = useState(false);
  useEffect(() => {
    if (!failure) setPickerOpen(false);
  }, [failure]);

  const rows = buildRows(history, inflight, pending);
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [rows.length, history.length, pending.length, inflight?.id]);

  return (
    <div className="absolute inset-0 flex flex-col bg-zinc-950 pt-3">
      <div className="flex shrink-0 items-baseline justify-between px-4 sm:px-6">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.3em] text-zinc-400">
          Score
        </h2>
        <span className="text-[11px] text-zinc-500">{describeGameMode(mode)}</span>
      </div>

      {failure && (
        <FailureBanner
          reason={failure.reason}
          onPickManually={() => setPickerOpen(true)}
          onRetry={onRetryFailure}
          onDrop={onDropFailure}
        />
      )}

      {failure && pickerOpen && onApplyFailureGuess && (
        <ManualMovePicker
          previousFen={failure.previousFen}
          onPick={(san) => {
            setPickerOpen(false);
            onApplyFailureGuess(san);
          }}
          onCancel={() => setPickerOpen(false)}
        />
      )}

      <div
        ref={scrollRef}
        className="mt-3 flex-1 overflow-y-auto px-2 pb-32 sm:px-4"
      >
        <table className="w-full table-fixed text-sm">
          <thead className="sticky top-0 bg-zinc-950 text-[10px] uppercase tracking-[0.22em] text-zinc-500">
            <tr>
              <th className="w-10 px-2 py-2 text-right font-normal">#</th>
              <th className="px-2 py-2 text-left font-normal">White</th>
              <th className="px-2 py-2 text-left font-normal">Black</th>
              <th className="w-14 px-2 py-2 text-right font-normal">
                W Δ
              </th>
              <th className="w-14 px-2 py-2 text-right font-normal">
                B Δ
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={5} className="px-2 py-6 text-center text-zinc-500">
                  No moves yet — make your first move and tap your clock.
                </td>
              </tr>
            )}
            {rows.map((r) => (
              <tr
                key={r.n}
                className="border-t border-white/5 align-baseline"
              >
                <td className="px-2 py-1.5 text-right font-mono text-zinc-500">
                  {r.n}.
                </td>
                <Cell entry={r.white} />
                <Cell entry={r.black} />
                <DeltaCell entry={r.white} />
                <DeltaCell entry={r.black} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

type RowEntry =
  | { kind: "committed"; move: MoveEntry }
  | { kind: "pending"; side: Side; spinning: boolean }
  | null;
type Row = { n: number; white: RowEntry; black: RowEntry };

function buildRows(
  history: MoveEntry[],
  inflight: PendingCapture | null,
  pending: PendingCapture[],
): Row[] {
  // Interleave committed moves first (in order), then append pending
  // captures: inflight first, then queued. Pending rows occupy slots
  // strictly after the committed history, alternating sides starting
  // with whichever side is up next.
  const rows: Row[] = [];
  const moves = history.slice();
  for (let i = 0; i < moves.length; i += 2) {
    rows.push({
      n: Math.floor(i / 2) + 1,
      white: { kind: "committed", move: moves[i] },
      black: i + 1 < moves.length ? { kind: "committed", move: moves[i + 1] } : null,
    });
  }
  // Determine next side based on history length.
  let nextSide: Side = history.length % 2 === 0 ? "white" : "black";
  let n = Math.floor(history.length / 2) + 1;
  const allPending: Array<{ inflight: boolean; cap: PendingCapture }> = [];
  if (inflight) allPending.push({ inflight: true, cap: inflight });
  for (const c of pending) allPending.push({ inflight: false, cap: c });
  for (const p of allPending) {
    if (nextSide === "white") {
      if (rows[rows.length - 1]?.black === null || rows.length === 0 || rows[rows.length - 1]?.white) {
        rows.push({ n, white: null, black: null });
      }
      const row = rows[rows.length - 1];
      row.white = { kind: "pending", side: "white", spinning: p.inflight };
      nextSide = "black";
    } else {
      let row = rows[rows.length - 1];
      if (!row || (row.white?.kind === "committed" && row.black?.kind === "committed")) {
        rows.push({ n, white: null, black: null });
        row = rows[rows.length - 1];
      }
      row.black = { kind: "pending", side: "black", spinning: p.inflight };
      nextSide = "white";
      n++;
    }
  }
  return rows;
}

function Cell({ entry }: { entry: RowEntry }) {
  if (!entry) {
    return <td className="px-2 py-1.5 text-zinc-700">—</td>;
  }
  if (entry.kind === "committed") {
    return (
      <td className="px-2 py-1.5 font-mono text-zinc-100">{entry.move.san}</td>
    );
  }
  return (
    <td className="px-2 py-1.5 font-mono italic text-zinc-500">
      <span className="inline-flex items-center gap-2">
        {entry.spinning && (
          <span
            className="inline-block h-2 w-2 rounded-full bg-emerald-400"
            style={{ animation: "chesspar-pulse 1.4s ease-in-out infinite" }}
          />
        )}
        …
      </span>
    </td>
  );
}

function DeltaCell({ entry }: { entry: RowEntry }) {
  if (!entry || entry.kind !== "committed") {
    return <td className="px-2 py-1.5 text-right text-[11px] text-zinc-700">—</td>;
  }
  return (
    <td className="px-2 py-1.5 text-right font-mono text-[11px] text-zinc-500">
      {formatDelta(entry.move.thinkDurationMs)}
    </td>
  );
}

function FailureBanner({
  reason,
  onPickManually,
  onRetry,
  onDrop,
}: {
  reason: string;
  onPickManually: () => void;
  onRetry?: () => void;
  onDrop?: () => void;
}) {
  return (
    <div className="mx-4 mt-3 rounded-2xl border border-amber-300/30 bg-amber-300/10 p-3 text-[12px] text-amber-100 sm:mx-6">
      <div className="mb-2">
        <span className="text-[10px] font-semibold uppercase tracking-[0.22em] text-amber-200/80">
          Queue paused
        </span>
      </div>
      <div className="text-amber-100/90">{reason}</div>
      <div className="mt-3 flex flex-wrap gap-2">
        <button
          onClick={onPickManually}
          className="rounded-full border border-emerald-400/40 bg-emerald-400/10 px-3 py-1.5 text-[11px] uppercase tracking-widest text-emerald-100 transition hover:bg-emerald-400/20"
        >
          Pick the move
        </button>
        {onRetry && (
          <button
            onClick={onRetry}
            className="rounded-full border border-amber-300/40 bg-amber-300/10 px-3 py-1.5 text-[11px] uppercase tracking-widest text-amber-100 transition hover:bg-amber-300/20"
          >
            Retry classification
          </button>
        )}
        <button
          onClick={onDrop}
          className="rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-[11px] uppercase tracking-widest text-zinc-200 transition hover:bg-white/10"
        >
          Drop & re-snap
        </button>
      </div>
    </div>
  );
}
