import { instantiateChronofishWasm } from "./wasm-loader.js";

let engine = null;
let activeModelLoaded = false;

self.addEventListener("message", async (event) => {
  const { id, type, notation, depth, nodes, encodeOnly, seed, plies, maxTurns, outcomeScale } = event.data;
  try {
    await loadEngine();
    if (type === "selfPlay") {
      await loadActiveModel();
    }
    replayNotation(notation ?? "");
    const sample = type === "selfPlay"
      ? selfPlayGame(depth, nodes, maxTurns, outcomeScale, seed)
      : encodeOnly
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

async function loadActiveModel() {
  if (activeModelLoaded || !engine?.chronofish_set_neural_model_bytes) {
    activeModelLoaded = true;
    return;
  }
  activeModelLoaded = true;
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

function writeWasmBytes(bytes) {
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

function selfPlayGame(depth, nodes, maxTurns, outcomeScale, seed) {
  const positions = [];
  const turnLimit = Math.max(1, maxTurns ?? 48);
  const scale = Math.max(1, outcomeScale ?? 90000);
  const jitter = seededOffset(seed ?? 0);
  for (let turn = 0; turn < turnLimit; turn += 1) {
    const position = neuralPosition((seed ?? 0) + turn, 0);
    const result = aiTurn(depth, nodes + jitter + turn);
    if (result.status !== "ok" || !result.moves?.length) {
      return labelOutcomeSamples(positions, oppositeColor(position.sideToMove), scale);
    }
    positions.push({
      ...position,
      policy: policyBucket(result.moves[0]),
      turn
    });
    for (const move of result.moves) {
      if (!engine.chronofish_apply_move(
        move.from.timelineId,
        move.from.time,
        move.from.x,
        move.from.y,
        move.to.timelineId,
        move.to.time,
        move.to.x,
        move.to.y
      )) {
        return [];
      }
    }
    if (!engine.chronofish_submit_turn()) {
      return [];
    }
    const winner = winnerFromMessage(readWasmString(engine.chronofish_last_message()));
    if (winner) {
      return labelOutcomeSamples(positions, winner, scale);
    }
  }
  return [];
}

function aiTurn(depth, nodes) {
  const pointer = engine.chronofish_ai_turn_json(Math.max(1, depth ?? 1), Math.max(1, nodes ?? 1000));
  return JSON.parse(readWasmString(pointer));
}

function labelOutcomeSamples(positions, winner, scale) {
  if (!winner || positions.length === 0) {
    return [];
  }
  const last = positions.length - 1;
  return positions.map((position, index) => {
    const distance = last - index;
    const discount = Math.pow(0.985, distance);
    return {
      label: (position.sideToMove === winner ? scale : -scale) * discount,
      policy: position.policy ?? 0,
      sideToMove: position.sideToMove,
      boardCount: position.boardCount,
      features: position.features,
      selfPlay: true,
      plyDistance: distance,
      result: winner
    };
  });
}

function winnerFromMessage(message) {
  if (/\bWhite wins\b/i.test(message)) {
    return "white";
  }
  if (/\bBlack wins\b/i.test(message)) {
    return "black";
  }
  return null;
}

function oppositeColor(color) {
  return color === "white" ? "black" : "white";
}

function policyBucket(move) {
  if (!move) {
    return 0;
  }
  const from = ((clampSquare(move.from.y) << 3) | clampSquare(move.from.x));
  const to = ((clampSquare(move.to.y) << 3) | clampSquare(move.to.x));
  return 1 + ((from * 64 + to) % 256);
}

function clampSquare(value) {
  return Math.min(7, Math.max(0, Number(value) || 0));
}

function seededOffset(seed) {
  let value = seed >>> 0;
  value ^= value >>> 16;
  value = Math.imul(value, 2246822507) >>> 0;
  value ^= value >>> 13;
  return value % 997;
}
