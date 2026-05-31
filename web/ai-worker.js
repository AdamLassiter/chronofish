import { instantiateChronofishWasm } from "./wasm-loader.js";

let engine = null;
let activeModelLoaded = false;
let activeModelLoad = null;

function readWasmString(ptr) {
  // Same shared-output convention as the main thread: copy before another export
  // overwrites the buffer.
  const bytes = new Uint8Array(engine.memory.buffer, ptr, engine.chronofish_output_len());
  return new TextDecoder("utf-8").decode(bytes);
}

function writeWasmString(value) {
  const bytes = new TextEncoder().encode(value ?? "");
  const ptr = engine.chronofish_alloc(bytes.length);
  new Uint8Array(engine.memory.buffer, ptr, bytes.length).set(bytes);
  return { ptr, len: bytes.length };
}

async function loadEngine() {
  // The worker owns a separate WASM instance so AI search cannot block the UI
  // thread or mutate the visible game state.
  if (engine) {
    return;
  }

  const instance = await instantiateChronofishWasm("./chronofish_engine.wasm");
  engine = instance.exports;
}

async function loadActiveModel() {
  if (activeModelLoaded) {
    return;
  }
  if (activeModelLoad) {
    return activeModelLoad;
  }
  activeModelLoad = loadActiveModelOnce().finally(() => {
    activeModelLoaded = true;
    activeModelLoad = null;
  });
  return activeModelLoad;
}

async function loadActiveModelOnce() {
  if (!engine?.chronofish_set_neural_model_bytes) {
    return;
  }
  try {
    const response = await fetch("/api/training/model");
    if (!response.ok) {
      engine.chronofish_clear_neural_model?.();
      return;
    }
    const model = new Uint8Array(await response.arrayBuffer());
    const { ptr, len } = writeWasmBytes(model);
    try {
      engine.chronofish_set_neural_model_bytes(ptr, len);
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  } catch {
    engine.chronofish_clear_neural_model?.();
  }
}

function writeWasmBytes(bytes) {
  const ptr = engine.chronofish_alloc(bytes.length);
  new Uint8Array(engine.memory.buffer, ptr, bytes.length).set(bytes);
  return { ptr, len: bytes.length };
}

function replayTurns(turns) {
  // Rebuild state from submitted turns so the worker evaluates the same game the
  // main thread would reconstruct locally.
  engine.chronofish_reset();

  for (const turn of turns) {
    for (const move of turn) {
      engine.chronofish_apply_move(
        move.from.timelineId,
        move.from.time,
        move.from.x,
        move.from.y,
        move.to.timelineId,
        move.to.time,
        move.to.x,
        move.to.y
      );
    }
    engine.chronofish_submit_turn();
  }
}

function replayNotation(notation) {
  const { ptr, len } = writeWasmString(notation);
  try {
    if (!engine.chronofish_load_notation(ptr, len)) {
      throw new Error(readWasmString(engine.chronofish_last_message()));
    }
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

self.addEventListener("message", async (event) => {
  // id is echoed back so the main thread can discard stale search results.
  const { id, notation, turns, depth, nodes, timeMs } = event.data;

  try {
    await loadEngine();
    await loadActiveModel();
    if (notation) {
      replayNotation(notation);
    } else {
      replayTurns(turns ?? []);
    }
    const fn = engine.chronofish_ai_turn_timed_json ?? engine.chronofish_ai_turn_json;
    const pointer = engine.chronofish_ai_turn_timed_json
      ? fn(depth, nodes, timeMs ?? 10_000)
      : fn(depth, nodes);
    const result = JSON.parse(readWasmString(pointer));
    self.postMessage({ id, ok: true, result });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error.message });
  }
});
