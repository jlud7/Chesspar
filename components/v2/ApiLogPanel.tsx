"use client";

import { useEffect, useState } from "react";
import { getApiLog, subscribeApiLog, type ApiLogEntry } from "@/lib/v2/api-log";

/**
 * Always-on, always-visible status of the most recent VLM call, with an
 * expandable history. The user explicitly wants to see API pushes — this
 * is the single place that surfaces them in-app.
 */
export function ApiLogPanel() {
  const [entries, setEntries] = useState<ApiLogEntry[]>(() => getApiLog());
  const [open, setOpen] = useState(false);

  useEffect(() => {
    return subscribeApiLog(() => setEntries(getApiLog()));
  }, []);

  const latest = entries[0];

  return (
    <div className="rounded-2xl border border-white/5 bg-white/5 px-3 py-2 text-[11px] text-zinc-400">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 text-left"
      >
        <span className="flex items-center gap-2">
          <span className="font-mono uppercase tracking-widest text-zinc-500">
            API
          </span>
          {latest ? (
            <Pill entry={latest} />
          ) : (
            <span className="text-zinc-500">No calls yet</span>
          )}
        </span>
        <span className="text-zinc-500">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="mt-2 max-h-[40vh] space-y-1 overflow-y-auto border-t border-white/5 pt-2 font-mono text-[10.5px]">
          {entries.length === 0 && (
            <div className="px-1 py-2 text-zinc-500">Empty.</div>
          )}
          {entries.map((e) => (
            <Row key={e.id} entry={e} />
          ))}
        </div>
      )}
    </div>
  );
}

function Pill({ entry }: { entry: ApiLogEntry }) {
  const ok = entry.ok;
  const tone = ok ? "bg-emerald-400/15 text-emerald-200" : "bg-rose-400/15 text-rose-200";
  return (
    <span className={`rounded-full px-2 py-0.5 font-mono uppercase tracking-wider ${tone}`}>
      {entry.callName} · {Math.round(entry.durationMs)}ms
    </span>
  );
}

function Row({ entry }: { entry: ApiLogEntry }) {
  return (
    <div className="rounded-md px-2 py-1 hover:bg-white/5">
      <div className="flex justify-between gap-2 text-zinc-200">
        <span>{entry.callName}</span>
        <span className={entry.ok ? "text-emerald-300" : "text-rose-300"}>
          {entry.ok ? "OK" : "ERR"} · {Math.round(entry.durationMs)}ms
        </span>
      </div>
      {entry.errorMessage && (
        <div className="text-rose-300/80">{entry.errorMessage}</div>
      )}
      {entry.outputPreview && (
        <div className="truncate text-zinc-500">{entry.outputPreview}</div>
      )}
    </div>
  );
}
