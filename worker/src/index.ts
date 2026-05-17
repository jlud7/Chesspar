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
  /** Set to enable the /replicate/sam2 endpoint. */
  REPLICATE_API_TOKEN?: string;
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
// Replicate's "run latest version of a model" endpoint. We use synchronous
// mode (Prefer: wait) so the worker returns the final prediction in one
// hop — fine for SAM-2 image runs that finish in seconds.
const REPLICATE_PREDICTIONS_URL =
  "https://api.replicate.com/v1/predictions";
// Version hash pinned so the worker's output shape is stable. Bump this
// when Meta ships a new SAM-2 image build and we want to opt in.
const SAM2_VERSION =
  "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83";
// Florence-2 large is kept warm on Replicate (~300ms inference). Used for
// one-shot chessboard localisation at calibration time. Bump after
// re-fetching via:
//   curl -s https://api.replicate.com/v1/models/lucataco/florence-2-large \
//     -H "Authorization: Bearer $TOKEN" | jq -r .latest_version.id
const FLORENCE2_VERSION =
  "da53547e17d45b9cfb48174b2f18af8b83ca020fa76db62136bf9c6616762595";

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
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
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
    const isSam2 = url.pathname === "/replicate/sam2" && req.method === "POST";
    const isFlorence2 =
      url.pathname === "/replicate/florence2" && req.method === "POST";
    // /replicate/vlm — generic forwarder to any Replicate model's prediction
    // endpoint. Client supplies { model: "owner/name", input: {...} }; the
    // worker injects the Replicate token and waits synchronously for the
    // prediction. Used by the per-move VLM identifier (anthropic/claude-
    // sonnet-4-6) so the entire path stays inside Replicate.
    const isReplicateVlm =
      url.pathname === "/replicate/vlm" && req.method === "POST";
    // /replicate/prediction?id=xxx — GET-poll for an in-flight prediction.
    // Replicate caps `Prefer: wait` at ~60s; longer predictions (gemini-3-flash
    // with heavy prompts can hit 60–90s) come back with status="starting" and
    // must be polled. This route is the polling target.
    const isReplicatePoll =
      url.pathname === "/replicate/prediction" && req.method === "GET";
    if (
      !isVerify &&
      !isOpenAi &&
      !isGemini &&
      !isSam2 &&
      !isFlorence2 &&
      !isReplicateVlm &&
      !isReplicatePoll
    ) {
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
    if (
      (isSam2 || isFlorence2 || isReplicateVlm || isReplicatePoll) &&
      !env.REPLICATE_API_TOKEN
    ) {
      return new Response("Replicate not configured on this worker", {
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

    // Polling route is GET-shaped — no JSON body. Handle it before the
    // body parser would otherwise reject the request.
    if (isReplicatePoll) {
      const id = url.searchParams.get("id");
      if (!id || !/^[A-Za-z0-9]+$/.test(id)) {
        return new Response('Missing or invalid "id" query param', {
          status: 400,
          headers: baseCors,
        });
      }
      const upstream = await fetch(
        `https://api.replicate.com/v1/predictions/${encodeURIComponent(id)}`,
        {
          method: "GET",
          headers: {
            Authorization: `Bearer ${env.REPLICATE_API_TOKEN}`,
          },
        },
      );
      const pollHeaders = new Headers(baseCors);
      pollHeaders.set(
        "Content-Type",
        upstream.headers.get("Content-Type") ?? "application/json",
      );
      return new Response(upstream.body, {
        status: upstream.status,
        headers: pollHeaders,
      });
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
    } else if (isGemini) {
      // Gemini: model is selected via ?model=... query param so we don't
      // need to parse the body to route. Default to the fast live-capture model.
      const model = url.searchParams.get("model") ?? "gemini-2.5-flash";
      const target = `${GEMINI_BASE}/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(env.GEMINI_API_KEY!)}`;
      upstream = await fetch(target, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
    } else if (isReplicateVlm) {
      // /replicate/vlm — generic forwarder. Client sends
      //   { model: "owner/name", input: {...} }
      // and we POST to the per-model predictions endpoint. Unlike SAM2 /
      // Florence-2 (where we pin a version hash so the output shape is
      // stable), VLM models on Replicate use the latest version path so
      // we don't have to bump version hashes every time the upstream
      // model is updated. Synchronous via Prefer: wait — Sonnet finishes
      // in ~1-2 s for a single image.
      const clientBody = JSON.parse(body) as {
        model?: unknown;
        input?: unknown;
      };
      if (
        typeof clientBody.model !== "string" ||
        !/^[a-z0-9_.-]+\/[a-z0-9_.-]+$/i.test(clientBody.model)
      ) {
        return new Response('Missing or invalid "model" field', {
          status: 400,
          headers: baseCors,
        });
      }
      if (!clientBody.input) {
        return new Response('Missing "input" field', {
          status: 400,
          headers: baseCors,
        });
      }
      const target = `https://api.replicate.com/v1/models/${clientBody.model}/predictions`;
      upstream = await fetch(target, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${env.REPLICATE_API_TOKEN}`,
          Prefer: "wait",
        },
        body: JSON.stringify({ input: clientBody.input }),
      });
    } else {
      // Both pinned-version Replicate routes share the same shape:
      // client sends { input: { image, ... } }; we attach a pinned
      // version hash for the chosen model. Synchronous via Prefer: wait.
      const clientBody = JSON.parse(body) as { input?: unknown };
      if (!clientBody.input) {
        return new Response('Missing "input" field', {
          status: 400,
          headers: baseCors,
        });
      }
      const version = isSam2 ? SAM2_VERSION : FLORENCE2_VERSION;
      upstream = await fetch(REPLICATE_PREDICTIONS_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${env.REPLICATE_API_TOKEN}`,
          Prefer: "wait",
        },
        body: JSON.stringify({ version, input: clientBody.input }),
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
