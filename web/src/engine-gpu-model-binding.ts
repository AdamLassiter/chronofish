import { readWasmBytes, readWasmString, writeWasmBytes, writeWasmString } from "./engine-io.js";
import type { ChronofishEngine } from "./types.js";

export function supportsCompactModelEncoding(engine: ChronofishEngine): boolean {
  return typeof engine.chronofish_compact_value_model_bytes_json === "function";
}

export function compactModelBytesAreFinite(engine: ChronofishEngine, bytes: ArrayBuffer | Uint8Array): boolean {
  const input = writeWasmBytes(engine, asBytes(bytes));
  try {
    return Boolean(engine.chronofish_compact_value_model_is_finite_bytes(input.ptr, input.len));
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

export function encodeCompactModel(engine: ChronofishEngine, model: unknown): Uint8Array | null {
  const input = writeWasmString(engine, JSON.stringify(model));
  try {
    const output = engine.chronofish_compact_value_model_bytes_json(input.ptr, input.len);
    return output ? readWasmBytes(engine, output) : null;
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

export function decodeCompactModel<T>(engine: ChronofishEngine, bytes: ArrayBuffer | Uint8Array): T | null {
  const input = writeWasmBytes(engine, asBytes(bytes));
  try {
    const output = engine.chronofish_compact_value_model_json(input.ptr, input.len);
    return output ? JSON.parse(readWasmString(engine, output)) as T : null;
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

export function decodeCompactFrontierModelLayout<T>(engine: ChronofishEngine, bytes: ArrayBuffer | Uint8Array): T | null {
  const input = writeWasmBytes(engine, asBytes(bytes));
  try {
    const output = engine.chronofish_compact_value_model_frontier_layout_json(input.ptr, input.len);
    return output ? JSON.parse(readWasmString(engine, output)) as T : null;
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

function asBytes(buffer: ArrayBuffer | Uint8Array): Uint8Array {
  return buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
}
