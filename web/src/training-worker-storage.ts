import { decodeCompactModel } from "./training-gpu.js";
import { TRAINING_IO_TIMEOUT_MS } from "./training-gpu-constants.js";
import { BUFFER_KEY } from "./training-worker-types.js";
import type { CpuParameters } from "./training-worker-types.js";
import type { CompactValueModel, TrainingSample } from "./training-gpu.js";

interface ReplayDb extends IDBDatabase {}

export async function fetchActiveModel(): Promise<CompactValueModel | null> {
  try {
    const response = await withTimeout(
      fetch("/api/training/model"),
      TRAINING_IO_TIMEOUT_MS,
      "Timed out loading active model."
    );
    if (!response.ok) {
      return null;
    }
    const buffer = await response.arrayBuffer();
    const model = decodeCompactModel(buffer);
    if (model) {
      model.bytes = new Uint8Array(buffer);
    }
    return model;
  } catch {
    return null;
  }
}

export async function fetchCpuParameters(): Promise<CpuParameters> {
  const response = await withTimeout(
    fetch("/api/training/cpu-parameters"),
    TRAINING_IO_TIMEOUT_MS,
    "Timed out loading CPU parameters."
  );
  if (!response.ok) {
    throw new Error("No active CPU parameters are available.");
  }
  const value = await response.json() as Record<string, unknown>;
  const parameters: CpuParameters = {};
  for (const [key, raw] of Object.entries(value)) {
    if (typeof raw === "number" && Number.isFinite(raw)) {
      parameters[key] = raw;
    }
  }
  return parameters;
}

export async function loadReplayBuffer(): Promise<TrainingSample[]> {
  let db: ReplayDb | null = null;
  try {
    db = await withTimeout(openReplayDb(), TRAINING_IO_TIMEOUT_MS, "Timed out opening replay buffer.");
    return (await withTimeout(idbGet(db, BUFFER_KEY), TRAINING_IO_TIMEOUT_MS, "Timed out reading replay buffer.")) ?? [];
  } catch {
    return [];
  } finally {
    db?.close();
  }
}

export async function saveReplayBuffer(samples: TrainingSample[]): Promise<void> {
  let db: ReplayDb | null = null;
  try {
    db = await withTimeout(openReplayDb(), TRAINING_IO_TIMEOUT_MS, "Timed out opening replay buffer.");
    await withTimeout(idbPut(db, BUFFER_KEY, samples), TRAINING_IO_TIMEOUT_MS, "Timed out saving replay buffer.");
  } catch {
    // IndexedDB is an optimization; an in-memory run still works without it.
  } finally {
    db?.close();
  }
}

function openReplayDb(): Promise<ReplayDb> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("chronofish-training", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("buffers");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function idbGet(db: ReplayDb, key: string): Promise<TrainingSample[] | undefined> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("buffers", "readonly");
    const request = tx.objectStore("buffers").get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function idbPut(db: ReplayDb, key: string, value: TrainingSample[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("buffers", "readwrite");
    tx.objectStore("buffers").put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      (value) => {
        clearTimeout(timeout);
        resolve(value);
      },
      (error) => {
        clearTimeout(timeout);
        reject(error);
      }
    );
  });
}
