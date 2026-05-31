import { instantiateChronofishWasm } from "./wasm-loader.js";

let engine = null;

self.addEventListener("message", async (event) => {
  const { id, notation, depth, nodes, encodeOnly, seed, plies } = event.data;
  try {
    await loadEngine();
    replayNotation(notation ?? "");
    const sample = encodeOnly
      ? neuralPosition(seed, plies)
      : neuralSample(depth, nodes, seed, plies);
    self.postMessage({ id, ok: true, sample });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error.message });
  }
});

async function loadEngine() {
  if (engine) {
    return;
  }
  const instance = await instantiateChronofishWasm("./chronofish_engine.wasm");
  engine = instance.exports;
}

function readWasmString(ptr) {
  const bytes = new Uint8Array(engine.memory.buffer, ptr, engine.chronofish_output_len());
  return new TextDecoder("utf-8").decode(bytes);
}

function writeWasmString(value) {
  const bytes = new TextEncoder().encode(value ?? "");
  const ptr = engine.chronofish_alloc(bytes.length);
  new Uint8Array(engine.memory.buffer, ptr, bytes.length).set(bytes);
  return { ptr, len: bytes.length };
}

function replayNotation(notation) {
  engine.chronofish_reset();
  if (!notation) {
    return;
  }
  const { ptr, len } = writeWasmString(notation);
  try {
    if (!engine.chronofish_load_notation(ptr, len)) {
      throw new Error(readWasmString(engine.chronofish_last_message()));
    }
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function neuralSample(depth, nodes, seed, plies) {
  const fn = engine.chronofish_training_sample_json ?? engine.chronofish_neural_sample_json;
  const pointer = engine.chronofish_training_sample_json
    ? fn(depth, nodes, seed ?? 0, plies ?? 0)
    : fn(depth, nodes);
  return JSON.parse(readWasmString(pointer));
}

function neuralPosition(seed, plies) {
  const fn = engine.chronofish_training_position_json ?? engine.chronofish_neural_position_json;
  const pointer = engine.chronofish_training_position_json
    ? fn(seed ?? 0, plies ?? 0)
    : fn();
  return JSON.parse(readWasmString(pointer));
}
