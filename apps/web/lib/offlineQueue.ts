export type CaptureIntent = "observation" | "issue";

export type CaptureMetadata = {
  floor: string;
  room: string;
  trade: string;
  note: string;
  mediaType: "photo" | "video" | "voice" | "measurement";
  measurement?: string;
  source?: string;
  intent: CaptureIntent;
  locationMethod: "confirmed" | "recent" | "qr" | "unlinked";
};

export type QueuedCapture = {
  id: string;
  projectId: string;
  filename: string;
  mime: string;
  size: number;
  blob: Blob;
  metadata: CaptureMetadata;
  createdAt: string;
  attempts: number;
  state: "queued" | "syncing" | "failed";
  error?: string;
};

const DB_NAME = "buili-field-capture";
const DB_VERSION = 1;
const STORE = "captures";

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("Offline storage is not available in this browser."));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error ?? new Error("Could not open offline storage."));
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE)) {
        const store = database.createObjectStore(STORE, { keyPath: "id" });
        store.createIndex("projectId", "projectId", { unique: false });
        store.createIndex("createdAt", "createdAt", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
  });
}

function transact<T>(
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore, resolve: (value: T) => void, reject: (reason?: unknown) => void) => void
): Promise<T> {
  return openDatabase().then(
    (database) =>
      new Promise<T>((resolve, reject) => {
        const transaction = database.transaction(STORE, mode);
        const store = transaction.objectStore(STORE);
        transaction.oncomplete = () => database.close();
        transaction.onabort = () => {
          database.close();
          reject(transaction.error ?? new Error("Offline storage transaction was aborted."));
        };
        transaction.onerror = () => reject(transaction.error ?? new Error("Offline storage transaction failed."));
        operation(store, resolve, reject);
      })
  );
}

export function makeCaptureId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return `capture_${crypto.randomUUID()}`;
  return `capture_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

export async function saveQueuedCapture(capture: QueuedCapture): Promise<QueuedCapture> {
  return transact<QueuedCapture>("readwrite", (store, resolve, reject) => {
    const request = store.put(capture);
    request.onsuccess = () => resolve(capture);
    request.onerror = () => reject(request.error ?? new Error("Could not save capture locally."));
  });
}

export async function listQueuedCaptures(projectId?: string): Promise<QueuedCapture[]> {
  return transact<QueuedCapture[]>("readonly", (store, resolve, reject) => {
    const request = projectId ? store.index("projectId").getAll(projectId) : store.getAll();
    request.onsuccess = () => {
      const captures = (request.result as QueuedCapture[]).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
      resolve(captures);
    };
    request.onerror = () => reject(request.error ?? new Error("Could not read offline captures."));
  });
}

export async function removeQueuedCapture(id: string): Promise<void> {
  return transact<void>("readwrite", (store, resolve, reject) => {
    const request = store.delete(id);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error ?? new Error("Could not remove offline capture."));
  });
}

export async function updateQueuedCapture(
  capture: QueuedCapture,
  patch: Partial<Pick<QueuedCapture, "state" | "attempts" | "error">>
): Promise<QueuedCapture> {
  return saveQueuedCapture({ ...capture, ...patch });
}
