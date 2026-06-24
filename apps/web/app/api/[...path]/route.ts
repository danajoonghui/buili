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

  if (!["GET", "HEAD"].includes(request.method)) {
    init.body = request.body;
    init.duplex = "half";
  }

  const upstream = await fetch(upstreamUrl, init);
  return new Response(upstream.body, {
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
