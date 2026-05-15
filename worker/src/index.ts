/**
 * Chesspar VLM proxy — Cloudflare Worker.
 *
 * Holds the Anthropic API key as a Worker secret so the static site
 * (GitHub Pages) can call the VLM without ever shipping the key to the
 * browser. The worker accepts the same shape as the in-app verifier:
 *
 *   POST /verify
 *   { model, max_tokens, temperature, messages }
 *
 * and forwards to https://api.anthropic.com/v1/messages with the key
 * attached, returning the upstream response verbatim.
 *
 * Access control is intentionally lightweight:
 *   - CORS allowlist of Origin headers (configure ALLOWED_ORIGINS).
 *   - Optional shared-secret header (X-Chesspar-Token) when CHESSPAR_TOKEN
 *     is set as a Worker variable.
 * For tighter protection layer Cloudflare WAF rate-limiting on top.
 */

export interface Env {
  ANTHROPIC_API_KEY: string;
  /** Set to enable the /openai endpoint. Leave unset if you only want Claude. */
  OPENAI_API_KEY?: string;
  /** Set to enable the /gemini endpoint (Google AI Studio key). */
  GEMINI_API_KEY?: string;
  /** Comma-separated list of allowed origins. */
  ALLOWED_ORIGINS?: string;
  /** Optional shared secret. If set, requests must include
   *  X-Chesspar-Token: <value>. Leave unset to disable. */
  CHESSPAR_TOKEN?: string;
}

const ANTHROPIC_URL = "https://api.anthropic.com/v1/messages";
const OPENAI_URL = "https://api.openai.com/v1/chat/completions";
// Gemini path is per-model; we let the client send the body intact and
// just attach ?key=… server-side. The frontend POSTs to /gemini?model=…
const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models";

function parseAllowedOrigins(env: Env): string[] {
  if (!env.ALLOWED_ORIGINS) return [];
  return env.ALLOWED_ORIGINS.split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function corsHeaders(origin: string | null, allowed: string[]): HeadersInit {
  const allow =
    origin && (allowed.length === 0 || allowed.includes(origin))
      ? origin
      : allowed[0] ?? "";
  return {
    "Access-Control-Allow-Origin": allow,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers":
      "Content-Type, X-Chesspar-Token, anthropic-version",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    const origin = req.headers.get("Origin");
    const allowed = parseAllowedOrigins(env);
    const baseCors = corsHeaders(origin, allowed);

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: baseCors });
    }

    const isVerify = url.pathname === "/verify" && req.method === "POST";
    const isOpenAi = url.pathname === "/openai" && req.method === "POST";
    const isGemini = url.pathname === "/gemini" && req.method === "POST";
    if (!isVerify && !isOpenAi && !isGemini) {
      return new Response("Not found", { status: 404, headers: baseCors });
    }
    if (isOpenAi && !env.OPENAI_API_KEY) {
      return new Response("OpenAI not configured on this worker", {
        status: 501,
        headers: baseCors,
      });
    }
    if (isGemini && !env.GEMINI_API_KEY) {
      return new Response("Gemini not configured on this worker", {
        status: 501,
        headers: baseCors,
      });
    }

    if (allowed.length > 0 && (!origin || !allowed.includes(origin))) {
      return new Response("Forbidden origin", {
        status: 403,
        headers: baseCors,
      });
    }

    if (env.CHESSPAR_TOKEN) {
      const supplied = req.headers.get("X-Chesspar-Token");
      if (supplied !== env.CHESSPAR_TOKEN) {
        return new Response("Unauthorized", {
          status: 401,
          headers: baseCors,
        });
      }
    }

    let body: string;
    try {
      const json = await req.json();
      body = JSON.stringify(json);
    } catch {
      return new Response("Invalid JSON", { status: 400, headers: baseCors });
    }

    let upstream: Response;
    if (isVerify) {
      upstream = await fetch(ANTHROPIC_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": env.ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
        },
        body,
      });
    } else if (isOpenAi) {
      upstream = await fetch(OPENAI_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${env.OPENAI_API_KEY}`,
        },
        body,
      });
    } else {
      // Gemini: model is selected via ?model=... query param so we don't
      // need to parse the body to route. Default to gemini-2.5-pro.
      const model = url.searchParams.get("model") ?? "gemini-2.5-pro";
      const target = `${GEMINI_BASE}/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(env.GEMINI_API_KEY!)}`;
      upstream = await fetch(target, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
    }

    const resHeaders = new Headers(baseCors);
    resHeaders.set(
      "Content-Type",
      upstream.headers.get("Content-Type") ?? "application/json",
    );
    return new Response(upstream.body, {
      status: upstream.status,
      headers: resHeaders,
    });
  },
};
