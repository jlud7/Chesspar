/**
 * In-memory ring buffer of recent VLM calls. The ApiLogPanel reads from
 * this so the user can see exactly which calls fired, how long they took,
 * and what came back — no Replicate dashboard needed.
 *
 * Lives at module scope (singleton). Subscribers are notified on every
 * push so the React panel re-renders without manual plumbing.
 */

export type ApiLogEntry = {
  id: number;
  callName: string;
  model: string;
  startedAt: number;
  finishedAt: number;
  durationMs: number;
  ok: boolean;
  promptPreview: string;
  outputPreview: string;
  errorMessage?: string;
};

const MAX_ENTRIES = 50;
const entries: ApiLogEntry[] = [];
const listeners = new Set<() => void>();
let nextId = 1;

export function pushApiLog(entry: Omit<ApiLogEntry, "id">): ApiLogEntry {
  const stored: ApiLogEntry = { ...entry, id: nextId++ };
  entries.unshift(stored);
  if (entries.length > MAX_ENTRIES) entries.length = MAX_ENTRIES;
  for (const l of listeners) l();
  return stored;
}

export function getApiLog(): ApiLogEntry[] {
  return entries.slice();
}

export function subscribeApiLog(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function clearApiLog(): void {
  entries.length = 0;
  for (const l of listeners) l();
}
