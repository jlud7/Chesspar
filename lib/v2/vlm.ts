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

  const j = json as {
    status?: string;
    output?: unknown;
    error?: unknown;
  };
  if (j.status && j.status !== "succeeded") {
    const reason = `Replicate status=${j.status}${j.error ? `: ${String(j.error)}` : ""}`;
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

  const text = stringifyOutput(j.output);
  pushApiLog({
    callName: input.callName,
    model,
    startedAt,
    finishedAt: Date.now(),
    durationMs,
    ok: true,
    promptPreview,
    outputPreview: shortify(text, 240),
  });
  return { kind: "ok", text, raw: json, durationMs };
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
