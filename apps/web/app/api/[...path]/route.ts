export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

const DEFAULT_INTERNAL_API = "http://localhost:8200";

function internalApiBase() {
  const raw = process.env.BUILI_INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL || DEFAULT_INTERNAL_API;
  if (raw.startsWith("/")) return DEFAULT_INTERNAL_API;
  if (/^https?:\/\//.test(raw)) return raw.replace(/\/$/, "");
  return `http://${raw.replace(/\/$/, "")}`;
}

function copyHeaders(request: Request) {
  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.delete("connection");
  headers.delete("content-length");
  // Identity is established by the API from the signed HttpOnly session. Never
  // let a browser manufacture the legacy actor/role headers at this trust edge.
  for (const name of Array.from(headers.keys())) {
    if (name.toLowerCase().startsWith("x-buili-")) headers.delete(name);
  }
  return headers;
}

function responseHeaders(upstream: Response) {
  const headers = new Headers(upstream.headers);
  headers.delete("content-encoding");
  headers.delete("content-length");
  headers.delete("transfer-encoding");
  return headers;
}

async function proxy(request: Request, context: RouteContext) {
  const { path } = await context.params;
  const upstreamUrl = new URL(`${internalApiBase()}/${path.join("/")}`);
  upstreamUrl.search = new URL(request.url).search;

  const init: RequestInit & { duplex?: "half" } = {
    method: request.method,
    headers: copyHeaders(request),
    cache: "no-store"
  };

  if (!["GET", "HEAD"].includes(request.method) && request.body) {
    // Small JSON mutations are buffered to avoid reusing the framework-owned
    // request stream after an upstream fast-fail (for example a 401 login).
    // Multipart evidence uploads remain streamed end-to-end.
    if (request.headers.get("content-type")?.includes("application/json")) {
      init.body = await request.arrayBuffer();
    } else {
      init.body = request.body;
      init.duplex = "half";
    }
  }

  const upstream = await fetch(upstreamUrl, init);
  return new Response([204, 205, 304].includes(upstream.status) ? null : upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders(upstream)
  });
}

export async function GET(request: Request, context: RouteContext) {
  return proxy(request, context);
}

export async function POST(request: Request, context: RouteContext) {
  return proxy(request, context);
}

export async function PATCH(request: Request, context: RouteContext) {
  return proxy(request, context);
}

export async function PUT(request: Request, context: RouteContext) {
  return proxy(request, context);
}

export async function DELETE(request: Request, context: RouteContext) {
  return proxy(request, context);
}
