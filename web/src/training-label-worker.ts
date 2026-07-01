import { readWasmString, writeWasmString } from "./engine-io.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import type { ChronofishEngine, Color, GameSnapshot } from "./types.js";

interface WorkerScope {
  addEventListener(type: "message", listener: (event: MessageEvent<TrainingLabelRequest>) => void | Promise<void>): void;
  postMessage(message: TrainingLabelResponse, transfer?: Transferable[]): void;
}

interface TrainingLabelRequest {
  id: number;
  type?: "batchSample" | "selfPlay" | string;
  game?: GameSnapshot;
  games?: GameSnapshot[];
  encodeOnly?: boolean;
}

type TrainingLabelResponse =
  | { id: number; ok: true; sample: NeuralSample }
  | { id: number; ok: true; samples: NeuralSample[] }
  | { id: number; ok: false; error: string };

interface NeuralSample {
  sideToMove: Color;
  boardCount: number;
  positionKey: string;
  features: number[] | Float32Array;
}

const workerSelf = self as unknown as WorkerScope;
let enginePromise: Promise<ChronofishEngine> | null = null;

workerSelf.addEventListener("message", async (event) => {
  const { id, type, game, games, encodeOnly } = event.data;
  try {
    if (type === "batchSample") {
      if (!Array.isArray(games)) {
        throw new Error("Batch training position encoding requires game snapshots.");
      }
      for (const snapshot of games) {
        if (!snapshot.timelines.length) {
          throw new Error("Training position encoding requires a client game snapshot.");
        }
      }
      const samples = await neuralPositions(games);
      workerSelf.postMessage({ id, ok: true, samples });
      return;
    }
    if (type === "selfPlay" || !encodeOnly) {
      throw new Error("CPU search labels are disabled; training labels must come from GPU/model prediction.");
    }
    if (!game?.timelines.length) {
      throw new Error("Training position encoding requires a client game snapshot.");
    }
    const sample = await neuralPosition(game);
    workerSelf.postMessage({ id, ok: true, sample });
  } catch (error: unknown) {
    workerSelf.postMessage({ id, ok: false, error: errorMessage(error) });
  }
});

async function neuralPosition(game: GameSnapshot): Promise<NeuralSample> {
  return (await neuralPositions([game]))[0]!;
}

async function neuralPositions(games: GameSnapshot[]): Promise<NeuralSample[]> {
  const engine = await engineInstance();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(games));
  try {
    const output = engine.chronofish_training_samples_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as NeuralSample[];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineInstance() {
  enginePromise ??= instantiateChronofishWasm("./chronofish_engine.wasm")
    .then((instance) => instance.exports as unknown as ChronofishEngine);
  return enginePromise;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
