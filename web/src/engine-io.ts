import type { ChronofishEngine, WasmString } from "./types.js";

export function readWasmString(engine: ChronofishEngine, ptr: number): string {
  // Rust string exports share one output buffer. Copy the bytes immediately after
  // receiving a pointer because the next engine string call overwrites it.
  const bytes = new Uint8Array(engine.memory.buffer, ptr, engine.chronofish_output_len());
  return new TextDecoder("utf-8").decode(bytes);
}

export function readWasmBytes(engine: ChronofishEngine, ptr: number): Uint8Array {
  return new Uint8Array(engine.memory.buffer, ptr, engine.chronofish_output_len()).slice();
}

export function writeWasmString(engine: ChronofishEngine, value: string): WasmString {
  const bytes = new TextEncoder().encode(value);
  const ptr = engine.chronofish_alloc(bytes.length);
  new Uint8Array(engine.memory.buffer, ptr, bytes.length).set(bytes);
  return { ptr, len: bytes.length };
}
