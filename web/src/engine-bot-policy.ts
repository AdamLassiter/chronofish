import { readWasmString, writeWasmString } from "./engine-io.js";
import type { ChronofishEngine } from "./types.js";

export function numeric(engine: ChronofishEngine, operation: string, ...args: number[]): number {
  return call(engine, operation, ...args);
}

export function jsonValue<T>(engine: ChronofishEngine, operation: string, input: unknown, ...args: number[]): T {
  const { ptr, len } = writeWasmString(engine, JSON.stringify(input));
  try {
    const output = call(engine, operation, ptr, len, ...args);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as T;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export function jsonBoolean(engine: ChronofishEngine, operation: string, input: unknown): boolean {
  const { ptr, len } = writeWasmString(engine, JSON.stringify(input));
  try {
    return call(engine, operation, ptr, len) !== 0;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export function resultValue<T>(engine: ChronofishEngine, operation: string, ...args: number[]): T {
  const output = call(engine, operation, ...args);
  if (!output) {
    throw new Error(readWasmString(engine, engine.chronofish_last_message()));
  }
  return JSON.parse(readWasmString(engine, output)) as T;
}

function call(engine: ChronofishEngine, operation: string, ...args: number[]): number {
  const callback = (engine as unknown as Record<string, (...values: number[]) => number>)[operation];
  if (!callback) {
    throw new Error(`Bot engine does not export ${operation}.`);
  }
  return callback(...args);
}
