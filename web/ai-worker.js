let engine = null;

function readWasmString(ptr) {
  // Same shared-output convention as the main thread: copy before another export
  // overwrites the buffer.
  const bytes = new Uint8Array(engine.memory.buffer, ptr, engine.chronofish_output_len());
  return new TextDecoder("utf-8").decode(bytes);
}

async function loadEngine() {
  // The worker owns a separate WASM instance so AI search cannot block the UI
  // thread or mutate the visible game state.
  if (engine) {
    return;
  }

  const { instance } = await WebAssembly.instantiateStreaming(fetch("./chronofish_engine.wasm"), {});
  engine = instance.exports;
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

self.addEventListener("message", async (event) => {
  // id is echoed back so the main thread can discard stale search results.
  const { id, turns, depth, nodes } = event.data;

  try {
    await loadEngine();
    replayTurns(turns);
    const result = JSON.parse(readWasmString(engine.chronofish_ai_turn_json(depth, nodes)));
    self.postMessage({ id, ok: true, result });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error.message });
  }
});
