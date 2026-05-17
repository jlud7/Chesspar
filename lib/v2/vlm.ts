/**
 * Single thin wrapper around the worker's /replicate/vlm route for
 * google/gemini-3-flash. One function, one model, one I/O shape.
 *
 * The worker forwards to Replicate's predictions API synchronously and
 * returns the prediction JSON verbatim. Replicate returns
 *   { status: "succeeded" | "failed" | ..., output: string[] }
 * for this model (output is a stream of token chunks; we join them).
 *
 * Every call is recorded in the api-log ring buffer so the in-app
 * ApiLogPanel can show what fired without a Replicate dashboard.
 */

import { pushApiLog } from "./api-log.ts";

const DEFAULT_MODEL = "google/gemini-3-flash";

export type VlmCallInput = {
  proxyUrl: string;
  /** Human-readable label for the api log (e.g., "calibrate", "identify-move"). */
  callName: string;
  prompt: string;
  /** One or more data URLs (or remote URLs) — sent as `images: []`. */
  images: string[];
  /** Optional override; otherwise google/gemini-3-flash. */
  model?: string;
  /** Optional max output tokens. */
  maxTokens?: number;
  /** Override the Origin header. Needed from Node where the runtime doesn't
   *  set one; in the browser the platform sets it automatically and this is
   *  ignored. */
  origin?: string;
};

export type VlmCallResult =
  | { kind: "ok"; text: string; raw: unknown; durationMs: number }
  | { kind: "error"; reason: string; durationMs: number };

export async function callVlm(input: VlmCallInput): Promise<VlmCallResult> {
  const model = input.model ?? DEFAULT_MODEL;
  const startedAt = Date.now();
  const promptPreview = input.prompt.slice(0, 240);
  const body = {
    model,
    input: {
      prompt: input.prompt,
      images: input.images,
      ...(input.maxTokens ? { max_tokens: input.maxTokens } : {}),
    },
  };

  let resp: Response;
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (input.origin) headers["Origin"] = input.origin;
    resp = await fetch(
      input.proxyUrl.replace(/\/$/, "") + "/replicate/vlm",
      {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      },
    );
  } catch (e) {
    const durationMs = Date.now() - startedAt;
    const reason = e instanceof Error ? e.message : String(e);
    pushApiLog({
      callName: input.callName,
      model,
      startedAt,
      finishedAt: Date.now(),
      durationMs,
      ok: false,
      promptPreview,
      outputPreview: "",
      errorMessage: reason,
    });
    return { kind: "error", reason, durationMs };
  }

  const durationMs = Date.now() - startedAt;
  let json: unknown;
  try {
    json = await resp.json();
  } catch {
    pushApiLog({
      callName: input.callName,
      model,
      startedAt,
      finishedAt: Date.now(),
      durationMs,
      ok: false,
      promptPreview,
      outputPreview: "",
      errorMessage: `Non-JSON HTTP ${resp.status}`,
    });
    return {
      kind: "error",
      reason: `Non-JSON response (HTTP ${resp.status})`,
      durationMs,
    };
  }

  if (!resp.ok) {
    const reason = `HTTP ${resp.status}: ${shortify(JSON.stringify(json), 200)}`;
    pushApiLog({
      callName: input.callName,
      model,
      startedAt,
      finishedAt: Date.now(),
      durationMs,
      ok: false,
      promptPreview,
      outputPreview: "",
      errorMessage: reason,
    });
    return { kind: "error", reason, durationMs };
  }

  let j = json as PredictionPayload;

  // Replicate `Prefer: wait` caps at ~60s. If the prediction is still
  // running after that window, the worker returns the in-flight payload
  // with status "starting" or "processing" and a polling URL. Poll until
  // the prediction reaches a terminal state or we hit a hard ceiling.
  if (isInFlight(j.status) && j.id) {
    const pollResult = await pollUntilDone(input.proxyUrl, j.id, input.origin);
    if (pollResult.kind === "timeout") {
      const finishedAt = Date.now();
      pushApiLog({
        callName: input.callName,
        model,
        startedAt,
        finishedAt,
        durationMs: finishedAt - startedAt,
        ok: false,
        promptPreview,
        outputPreview: "",
        errorMessage: pollResult.reason,
      });
      return {
        kind: "error",
        reason: pollResult.reason,
        durationMs: finishedAt - startedAt,
      };
    }
    j = pollResult.payload;
  }

  const finalDurationMs = Date.now() - startedAt;

  if (j.status && j.status !== "succeeded") {
    const reason = `Replicate status=${j.status}${j.error ? `: ${String(j.error)}` : ""}`;
    pushApiLog({
      callName: input.callName,
      model,
      startedAt,
      finishedAt: Date.now(),
      durationMs: finalDurationMs,
      ok: false,
      promptPreview,
      outputPreview: "",
      errorMessage: reason,
    });
    return { kind: "error", reason, durationMs: finalDurationMs };
  }

  const text = stringifyOutput(j.output);
  pushApiLog({
    callName: input.callName,
    model,
    startedAt,
    finishedAt: Date.now(),
    durationMs: finalDurationMs,
    ok: true,
    promptPreview,
    outputPreview: shortify(text, 240),
  });
  return { kind: "ok", text, raw: j, durationMs: finalDurationMs };
}

type PredictionPayload = {
  id?: string;
  status?: string;
  output?: unknown;
  error?: unknown;
};

const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 180_000;

function isInFlight(status: string | undefined): boolean {
  return status === "starting" || status === "processing";
}

async function pollUntilDone(
  proxyUrl: string,
  id: string,
  origin?: string,
): Promise<
  | { kind: "done"; payload: PredictionPayload }
  | { kind: "timeout"; reason: string }
> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    const headers: Record<string, string> = {};
    if (origin) headers["Origin"] = origin;
    let resp: Response;
    try {
      resp = await fetch(
        `${proxyUrl.replace(/\/$/, "")}/replicate/prediction?id=${encodeURIComponent(id)}`,
        { method: "GET", headers },
      );
    } catch (e) {
      // Network blips: keep polling rather than fail immediately.
      continue;
    }
    if (!resp.ok) continue;
    let payload: PredictionPayload;
    try {
      payload = (await resp.json()) as PredictionPayload;
    } catch {
      continue;
    }
    if (!isInFlight(payload.status)) {
      return { kind: "done", payload };
    }
  }
  return {
    kind: "timeout",
    reason: `Prediction still running after ${Math.round(POLL_TIMEOUT_MS / 1000)}s — gave up.`,
  };
}

/**
 * Tolerant JSON extraction: tries direct parse, then strips ```json fences,
 * then grabs the first {...} block. Gemini-via-Replicate sometimes prepends
 * a line of prose before the JSON; this handles that.
 */
export function parseJsonLoose<T = unknown>(raw: string): T | null {
  const trimmed = raw.trim();
  try {
    return JSON.parse(trimmed) as T;
  } catch {
    /* fall through */
  }
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) {
    try {
      return JSON.parse(fenced[1].trim()) as T;
    } catch {
      /* fall through */
    }
  }
  const objMatch = trimmed.match(/\{[\s\S]*\}/);
  if (objMatch) {
    try {
      return JSON.parse(objMatch[0]) as T;
    } catch {
      /* fall through */
    }
  }
  return null;
}

function stringifyOutput(output: unknown): string {
  if (output == null) return "";
  if (typeof output === "string") return output;
  if (Array.isArray(output)) return output.map(String).join("");
  return String(output);
}

function shortify(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max) + `… (${s.length - max} more)`;
}
