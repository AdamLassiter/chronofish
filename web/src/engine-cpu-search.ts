import { readWasmString, writeWasmString } from "./engine-io.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import type { ChronofishEngine, GameSnapshot, Move } from "./types.js";

export interface CpuAiResult {
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

export interface CpuSearchRequest {
  game: GameSnapshot;
  depth: number;
  minDepth?: number | undefined;
  nodes: number;
  timeMs: number;
  parametersJson?: string | undefined;
}

export interface CpuApplyTurnResult {
  game: GameSnapshot;
  status: { complete: boolean; terminal: boolean; winner?: string; nextTurn: string };
}

export class CpuSearchBinding {
  private enginePromise: Promise<ChronofishEngine> | null = null;
  private parametersPromise: Promise<void> | null = null;

  async search(request: CpuSearchRequest): Promise<CpuAiResult> {
    const engine = await this.engine();
    if (request.parametersJson) {
      this.loadParametersJson(engine, request.parametersJson);
    } else {
      await this.loadParameters(engine);
    }
    this.loadSnapshot(engine, request.game);
    const config = this.searchConfig(engine, request);
    const output = config.minDepth == null
      ? engine.chronofish_ai_turn_timed_json(config.depth, config.nodes, config.timeMs)
      : engine.chronofish_ai_turn_timed_min_depth_json(
        config.depth,
        config.minDepth,
        config.nodes,
        config.timeMs
      );
    return this.searchResult(engine, JSON.parse(readWasmString(engine, output)) as CpuAiResult);
  }

  async applyTurn(game: GameSnapshot, moves?: Move[]): Promise<CpuApplyTurnResult> {
    const engine = await this.engine();
    const { ptr, len } = writeWasmString(engine, JSON.stringify({ game, moves }));
    try {
      const output = engine.chronofish_cpu_apply_turn_json(ptr, len);
      if (!output) {
        throw new Error(readWasmString(engine, engine.chronofish_last_message()));
      }
      return JSON.parse(readWasmString(engine, output)) as CpuApplyTurnResult;
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  }

  private async engine(): Promise<ChronofishEngine> {
    this.enginePromise ??= instantiateChronofishWasm("./chronofish_engine.wasm")
      .then((instance) => instance.exports as unknown as ChronofishEngine);
    return this.enginePromise;
  }

  private searchConfig(engine: ChronofishEngine, request: CpuSearchRequest): CpuWorkerSearchConfig {
    const { game: _game, parametersJson: _parametersJson, ...config } = request;
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

  private searchResult(engine: ChronofishEngine, result: CpuAiResult): CpuAiResult {
    const { ptr, len } = writeWasmString(engine, JSON.stringify(result));
    try {
      const output = engine.chronofish_cpu_worker_search_result_json(ptr, len);
      if (!output) {
        throw new Error(readWasmString(engine, engine.chronofish_last_message()));
      }
      return JSON.parse(readWasmString(engine, output)) as CpuAiResult;
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  }

  private loadSnapshot(engine: ChronofishEngine, game: GameSnapshot): void {
    const { ptr, len } = writeWasmString(engine, JSON.stringify(game));
    try {
      if (!engine.chronofish_load_snapshot_json(ptr, len)) {
        throw new Error(readWasmString(engine, engine.chronofish_last_message()));
      }
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  }

  private async loadParameters(engine: ChronofishEngine): Promise<void> {
    this.parametersPromise ??= this.fetchParameters().then((json) => {
      this.loadParametersJson(engine, json);
    });
    return this.parametersPromise;
  }

  private loadParametersJson(engine: ChronofishEngine, json: string): void {
    const { ptr, len } = writeWasmString(engine, json);
    try {
      if (!engine.chronofish_load_ai_parameters_json(ptr, len)) {
        throw new Error(readWasmString(engine, engine.chronofish_last_message()));
      }
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  }

  private async fetchParameters(): Promise<string> {
    for (const path of ["/api/training/cpu-parameters", "/ai/parameters.json"]) {
      try {
        const response = await fetch(path, { cache: "no-store" });
        if (response.ok) {
          return response.text();
        }
      } catch {
        // Try the next source.
      }
    }
    throw new Error("No CPU parameters JSON is available.");
  }
}
