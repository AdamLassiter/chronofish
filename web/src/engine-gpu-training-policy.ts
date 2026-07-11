import { readWasmBytes, readWasmString, writeWasmBytes, writeWasmString } from "./engine-io.js";
import type { ChronofishEngine } from "./types.js";

export function numeric(engine: ChronofishEngine, operation: string, ...args: number[]): number {
  return call(engine, operation, ...args);
}

export function jsonBoolean(engine: ChronofishEngine, operation: string, input: unknown, ...args: number[]): boolean {
  return jsonNumeric(engine, operation, input, ...args) !== 0;
}

export function jsonNumeric(engine: ChronofishEngine, operation: string, input: unknown, ...args: number[]): number {
  const text = writeWasmString(engine, JSON.stringify(input));
  try {
    return call(engine, operation, text.ptr, text.len, ...args);
  } finally {
    engine.chronofish_dealloc(text.ptr, text.len);
  }
}

export function jsonValue<T>(engine: ChronofishEngine, operation: string, input: unknown, ...args: number[]): T {
  const text = writeWasmString(engine, JSON.stringify(input));
  try {
    const output = call(engine, operation, text.ptr, text.len, ...args);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as T;
  } finally {
    engine.chronofish_dealloc(text.ptr, text.len);
  }
}

export function jsonBytes(engine: ChronofishEngine, operation: string, input: unknown, ...args: number[]): Uint8Array {
  const text = writeWasmString(engine, JSON.stringify(input));
  try {
    const output = call(engine, operation, text.ptr, text.len, ...args);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmBytes(engine, output);
  } finally {
    engine.chronofish_dealloc(text.ptr, text.len);
  }
}

export function textValue(engine: ChronofishEngine, operation: string, input: string, ...args: number[]): string {
  const text = writeWasmString(engine, input);
  try {
    const output = call(engine, operation, text.ptr, text.len, ...args);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmString(engine, output);
  } finally {
    engine.chronofish_dealloc(text.ptr, text.len);
  }
}

export function textNumeric(engine: ChronofishEngine, operation: string, input: string, ...args: number[]): number {
  const text = writeWasmString(engine, input);
  try {
    return call(engine, operation, text.ptr, text.len, ...args);
  } finally {
    engine.chronofish_dealloc(text.ptr, text.len);
  }
}

export function bytesResult(engine: ChronofishEngine, operation: string, input: ArrayBuffer | Uint8Array, ...args: number[]): Uint8Array | null {
  const bytes = writeWasmBytes(engine, asBytes(input));
  try {
    const output = call(engine, operation, bytes.ptr, bytes.len, ...args);
    return output ? readWasmBytes(engine, output) : null;
  } finally {
    engine.chronofish_dealloc(bytes.ptr, bytes.len);
  }
}

export function bytesNumeric(engine: ChronofishEngine, operation: string, input: ArrayBuffer | Uint8Array, ...args: number[]): number {
  const bytes = writeWasmBytes(engine, asBytes(input));
  try {
    return call(engine, operation, bytes.ptr, bytes.len, ...args);
  } finally {
    engine.chronofish_dealloc(bytes.ptr, bytes.len);
  }
}

export function bytesRequired(engine: ChronofishEngine, operation: string, input: ArrayBuffer | Uint8Array, ...args: number[]): Uint8Array {
  const output = bytesResult(engine, operation, input, ...args);
  if (!output) {
    throw new Error(readWasmString(engine, engine.chronofish_last_message()));
  }
  return output;
}

export function bytesBuffer(engine: ChronofishEngine, operation: string, input: ArrayBuffer | Uint8Array, ...args: number[]): ArrayBuffer {
  const output = bytesRequired(engine, operation, input, ...args);
  return new Uint8Array(output).buffer;
}

export function bytesJsonValue<T>(engine: ChronofishEngine, operation: string, bytesInput: ArrayBuffer | Uint8Array, jsonInput: unknown): T {
  const bytes = writeWasmBytes(engine, asBytes(bytesInput));
  const text = writeWasmString(engine, JSON.stringify(jsonInput));
  try {
    const output = call(engine, operation, bytes.ptr, bytes.len, text.ptr, text.len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as T;
  } finally {
    engine.chronofish_dealloc(bytes.ptr, bytes.len);
    engine.chronofish_dealloc(text.ptr, text.len);
  }
}

export function byteBuffer(engine: ChronofishEngine, operation: string, ...args: number[]): ArrayBuffer {
  const output = call(engine, operation, ...args);
  if (!output) {
    throw new Error(readWasmString(engine, engine.chronofish_last_message()));
  }
  return new Uint8Array(readWasmBytes(engine, output)).buffer;
}

function asBytes(value: ArrayBuffer | Uint8Array): Uint8Array {
  return value instanceof Uint8Array ? value : new Uint8Array(value);
}

function call(engine: ChronofishEngine, operation: string, ...args: number[]): number {
  const callback = (engine as unknown as Record<string, (...values: number[]) => number>)[operation];
  if (!callback) {
    throw new Error(`Training engine does not export ${operation}.`);
  }
  return callback(...args);
}
