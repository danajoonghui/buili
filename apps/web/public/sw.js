const CACHE_NAME = "buili-shell-v3";
const SHELL_ASSETS = [
  "/",
  "/projects",
  "/manifest.webmanifest",
  "/icon.svg",
  "/buili_favicon_transparent.png",
  "/brand/buili-mark.png"
];

const PRIVATE_PREFIXES = [
  "/api/",
  "/v1/",
  "/reports/",
  "/media/",
  "/site-media/",
  "/uploads/",
  "/objects/",
  "/storage/"
];

function isPrivate(url) {
  return PRIVATE_PREFIXES.some((prefix) => url.pathname.startsWith(prefix));
}

function isStaticShellAsset(request, url) {
  if (url.origin !== self.location.origin || isPrivate(url)) return false;
  if (url.pathname.startsWith("/_next/static/")) return true;
  return SHELL_ASSETS.includes(url.pathname) || ["style", "script", "font"].includes(request.destination);
}

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "GET" || isPrivate(url)) return;

  if (request.mode === "navigate") {
    event.respondWith(fetch(request).catch(() => caches.match("/").then((cached) => cached || Response.error())));
    return;
  }

  if (!isStaticShellAsset(request, url)) return;
  event.respondWith(
    caches.match(request).then((cached) => cached || fetch(request).then((response) => {
      if (response.ok && response.type === "basic") {
        const copy = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
      }
      return response;
    }))
  );
});

self.addEventListener("message", (event) => {
  if (event.data === "CLEAR_PRIVATE_CACHES") {
    event.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))));
  }
});
