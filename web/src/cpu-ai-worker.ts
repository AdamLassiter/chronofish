import { readWasmString, writeWasmString } from "./engine-io.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import type { ChronofishEngine, GameSnapshot, Move } from "./types.js";

interface CpuAiRequest {
  id: number | string;
  type?: "search" | "applyTurn";
  game?: GameSnapshot;
  moves?: Move[];
  depth?: number;
  minDepth?: number;
  nodes?: number;
  timeMs?: number;
  partitionIndex?: number;
  parametersJson?: string;
}

interface CpuAiResult {
  status: string;
  moves: Move[];
  score?: number;
  depth?: number;
  nodes?: number;
  terminal?: boolean;
  resultReason?: "royal-capture" | "threefold-repetition" | "stalemate" | null;
  principalVariation?: Move[][];
  cpuSearch?: string;
}

interface CpuWorkerSearchConfig {
  depth: number;
  minDepth?: number | null;
  nodes: number;
  timeMs: number;
}

let enginePromise: Promise<ChronofishEngine> | null = null;
let parametersPromise: Promise<void> | null = null;

self.addEventListener("message", async (event: MessageEvent<CpuAiRequest>) => {
  const { id, type = "search", game, moves, depth = 1, minDepth, nodes = 64, timeMs = 10_000, partitionIndex = 0, parametersJson } = event.data;
  try {
    if (!game) {
      throw new Error("CPU AI search requires a game snapshot.");
    }
    const engine = await cpuEngine();
    if (type === "applyTurn") {
      const applied = cpuApplyTurn(engine, game, moves ?? []);
      self.postMessage({
        id,
        ok: true,
        game: applied.game,
        status: applied.status,
        partitionIndex
      });
      return;
    }
    if (parametersJson) {
      loadCpuParametersJson(engine, parametersJson);
    } else {
      await loadCpuParameters(engine);
    }
    loadSnapshot(engine, game);
    const searchConfig = cpuWorkerSearchConfig(engine, { depth, minDepth, nodes, timeMs });
    const ptr = searchConfig.minDepth == null
      ? engine.chronofish_ai_turn_timed_json(
        searchConfig.depth,
        searchConfig.nodes,
        searchConfig.timeMs
      )
      : engine.chronofish_ai_turn_timed_min_depth_json(
        searchConfig.depth,
        searchConfig.minDepth,
        searchConfig.nodes,
        searchConfig.timeMs
      );
    const result = JSON.parse(readWasmString(engine, ptr)) as CpuAiResult;
    result.principalVariation ??= result.moves.length ? [result.moves] : [];
    result.cpuSearch = "heuristic";
    self.postMessage({ id, ok: true, result, partitionIndex });
  } catch (error) {
    self.postMessage({ id, ok: false, error: errorMessage(error), partitionIndex });
  }
});

function cpuApplyTurn(
  engine: ChronofishEngine,
  game: GameSnapshot,
  moves: Move[]
): { game: GameSnapshot; status: { complete: boolean; terminal: boolean; winner?: string; nextTurn: string } } {
  const { ptr, len } = writeWasmString(engine, JSON.stringify({ game, moves }));
  try {
    const output = engine.chronofish_cpu_apply_turn_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output));
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function cpuWorkerSearchConfig(
  engine: ChronofishEngine,
  config: { depth: number; minDepth?: number; nodes: number; timeMs: number }
): CpuWorkerSearchConfig {
  const { ptr, len } = writeWasmString(engine, JSON.stringify(config));
  try {
    const output = engine.chronofish_cpu_worker_search_config_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as CpuWorkerSearchConfig;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

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
      loadCpuParametersJson(engine, json);
    });
  return parametersPromise;
}

function loadCpuParametersJson(engine: ChronofishEngine, json: string): void {
  const { ptr, len } = writeWasmString(engine, json);
  try {
    if (!engine.chronofish_load_ai_parameters_json(ptr, len)) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
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
