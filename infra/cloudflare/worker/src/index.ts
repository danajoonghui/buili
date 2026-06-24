export interface Env {
  API_BASE: string;
  WEB_ORIGIN?: string;
  INFERENCE_QUEUE?: Queue;
  BUILI_R2?: R2Bucket;
}

function corsHeaders(env: Env): HeadersInit {
  return {
    "Access-Control-Allow-Origin": env.WEB_ORIGIN || "*",
    "Access-Control-Allow-Methods": "GET,POST,PATCH,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Max-Age": "86400"
  };
}

async function proxy(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const upstream = new URL(url.pathname + url.search, env.API_BASE);
  const headers = new Headers(request.headers);
  headers.set("x-buili-gateway", "cloudflare-worker");

  const response = await fetch(upstream, {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    redirect: "manual"
  });
  const nextHeaders = new Headers(response.headers);
  for (const [key, value] of Object.entries(corsHeaders(env))) {
    nextHeaders.set(key, value);
  }
  return new Response(response.body, { status: response.status, headers: nextHeaders });
}

async function enqueueAnalyze(request: Request, env: Env): Promise<void> {
  if (!env.INFERENCE_QUEUE || request.method !== "POST") return;
  const url = new URL(request.url);
  const match = url.pathname.match(/^\/v1\/projects\/([^/]+)\/analyze$/);
  if (!match) return;
  await env.INFERENCE_QUEUE.send({
    kind: "analyze_project",
    project_id: match[1],
    requested_at: new Date().toISOString()
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(env) });
    }
    await enqueueAnalyze(request, env);
    return proxy(request, env);
  }
};
