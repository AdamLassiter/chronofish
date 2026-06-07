import { readWasmString, writeWasmString } from "./engine-io.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import type { ChronofishEngine, GameSnapshot, Move } from "./types.js";

interface CpuAiRequest {
  id: number | string;
  game?: GameSnapshot;
  depth?: number;
  nodes?: number;
  timeMs?: number;
  partitionIndex?: number;
}

interface CpuAiResult {
  status: string;
  moves: Move[];
  score?: number;
  depth?: number;
  nodes?: number;
  cpuSearch?: string;
}

let enginePromise: Promise<ChronofishEngine> | null = null;
let parametersPromise: Promise<void> | null = null;

self.addEventListener("message", async (event: MessageEvent<CpuAiRequest>) => {
  const { id, game, depth = 1, nodes = 64, timeMs = 10_000, partitionIndex = 0 } = event.data;
  try {
    if (!game) {
      throw new Error("CPU AI search requires a game snapshot.");
    }
    const engine = await cpuEngine();
    await loadCpuParameters(engine);
    loadSnapshot(engine, game);
    const ptr = engine.chronofish_ai_turn_timed_json(
      Math.max(1, depth),
      Math.max(1, nodes),
      Math.max(1, Math.floor(timeMs))
    );
    const result = JSON.parse(readWasmString(engine, ptr)) as CpuAiResult;
    result.cpuSearch = "heuristic";
    self.postMessage({ id, ok: true, result, partitionIndex });
  } catch (error) {
    self.postMessage({ id, ok: false, error: errorMessage(error), partitionIndex });
  }
});

async function cpuEngine(): Promise<ChronofishEngine> {
  if (!enginePromise) {
    enginePromise = instantiateChronofishWasm("./chronofish_engine.wasm")
      .then((instance) => instance.exports as unknown as ChronofishEngine);
  }
  return enginePromise;
}

function loadSnapshot(engine: ChronofishEngine, game: GameSnapshot): void {
  const { ptr, len } = writeWasmString(engine, JSON.stringify(game));
  try {
    if (!engine.chronofish_load_snapshot_json(ptr, len)) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function loadCpuParameters(engine: ChronofishEngine): Promise<void> {
  parametersPromise ??= fetchCpuParameters()
    .then((json) => {
      const { ptr, len } = writeWasmString(engine, json);
      try {
        if (!engine.chronofish_load_ai_parameters_json(ptr, len)) {
          throw new Error(readWasmString(engine, engine.chronofish_last_message()));
        }
      } finally {
        engine.chronofish_dealloc(ptr, len);
      }
    });
  return parametersPromise;
}

async function fetchCpuParameters(): Promise<string> {
  for (const path of ["/api/training/cpu-parameters", "/ai/parameters.json"]) {
    try {
      const response = await fetch(path, { cache: "no-store" });
      if (response.ok) {
        return await response.text();
      }
    } catch {
      // Try the next source.
    }
  }
  throw new Error("No CPU parameters JSON is available.");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
