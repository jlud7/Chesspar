"use client";

import { useEffect, useRef } from "react";

export function MoveList({
  moves,
  abstainingOn,
}: {
  moves: string[];
  abstainingOn?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  // Auto-scroll to the latest move so the move log always shows the
  // current state without the user having to scroll. Smooth on
  // increment, instant on full replace.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [moves.length]);

  if (moves.length === 0 && !abstainingOn) {
    return (
      <div className="rounded-2xl border border-white/5 bg-white/5 px-4 py-6 text-center text-sm text-zinc-500">
        No moves yet. Capture your first move to begin.
      </div>
    );
  }
  const pairs: Array<{ n: number; w: string; b?: string }> = [];
  for (let i = 0; i < moves.length; i += 2) {
    pairs.push({ n: i / 2 + 1, w: moves[i], b: moves[i + 1] });
  }
  return (
    <div
      ref={ref}
      className="max-h-[40vh] overflow-y-auto rounded-2xl border border-white/5 bg-white/5 px-3 py-3 text-sm font-mono"
    >
      {pairs.map((p) => (
        <div key={p.n} className="flex gap-3 px-1 py-0.5">
          <span className="w-8 text-right text-zinc-500">{p.n}.</span>
          <span className="w-16 text-zinc-100">{p.w}</span>
          <span className="w-16 text-zinc-100">{p.b ?? ""}</span>
        </div>
      ))}
      {abstainingOn && (
        <div className="mt-2 rounded-md bg-amber-400/10 px-3 py-2 text-[12px] text-amber-200">
          Awaiting confirmation…
        </div>
      )}
    </div>
  );
}
