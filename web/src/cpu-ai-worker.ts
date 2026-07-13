import { CpuSearchBinding } from "./engine-cpu-search.js";
import type { CpuAiResult } from "./engine-cpu-search.js";
import type { GameSnapshot, Move } from "./types.js";

interface CpuAiRequest {
  id: number | string;
  type?: "search" | "applyTurn";
  game?: GameSnapshot;
  moves?: Move[];
  depth?: number;
  minDepth?: number;
  nodes?: number;
  timeMs?: number;
  searchStrategy?: "alpha-beta" | "beam";
  partitionIndex?: number;
  parametersJson?: string;
}

const binding = new CpuSearchBinding();

self.addEventListener("message", async (event: MessageEvent<CpuAiRequest>) => {
  const {
    id,
    type = "search",
    game,
    moves,
    depth = 1,
    minDepth,
    nodes = 64,
    timeMs = 10_000,
    searchStrategy,
    partitionIndex = 0,
    parametersJson
  } = event.data;
  try {
    if (!game) {
      throw new Error("CPU AI search requires a game snapshot.");
    }
    if (type === "applyTurn") {
      const applied = await binding.applyTurn(game, moves);
      self.postMessage({ id, ok: true, ...applied, partitionIndex });
      return;
    }
    const result: CpuAiResult = await binding.search({
      game,
      depth,
      ...(minDepth == null ? {} : { minDepth }),
      nodes,
      timeMs,
      ...(searchStrategy == null ? {} : { searchStrategy }),
      ...(parametersJson == null ? {} : { parametersJson })
    });
    self.postMessage({ id, ok: true, result, partitionIndex });
  } catch (error) {
    self.postMessage({ id, ok: false, error: errorMessage(error), partitionIndex });
  }
});

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
