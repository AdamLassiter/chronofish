import { GpuTrainingBinding } from "./engine-gpu-training.js";
import type { Color, GameSnapshot } from "./types.js";

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
const trainingBinding = new GpuTrainingBinding();

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
  return trainingBinding.trainingSamples<NeuralSample[]>(games);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
