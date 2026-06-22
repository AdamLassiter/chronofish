import { encodeNeuralPositionFeatures } from "./training-encoding.js";
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
  features: Float32Array;
}

const workerSelf = self as unknown as WorkerScope;

workerSelf.addEventListener("message", async (event) => {
  const { id, type, game, games, encodeOnly } = event.data;
  try {
    if (type === "batchSample") {
      if (!Array.isArray(games)) {
        throw new Error("Batch training position encoding requires game snapshots.");
      }
      const samples: NeuralSample[] = [];
      for (const snapshot of games) {
        if (!snapshot.timelines.length) {
          throw new Error("Training position encoding requires a client game snapshot.");
        }
        samples.push(neuralPosition(snapshot));
      }
      workerSelf.postMessage(
        { id, ok: true, samples },
        samples.map((sample) => sample.features.buffer)
      );
      return;
    }
    if (type === "selfPlay" || !encodeOnly) {
      throw new Error("CPU search labels are disabled; training labels must come from GPU/model prediction.");
    }
    if (!game?.timelines.length) {
      throw new Error("Training position encoding requires a client game snapshot.");
    }
    const sample = neuralPosition(game);
    workerSelf.postMessage({ id, ok: true, sample }, [sample.features.buffer]);
  } catch (error: unknown) {
    workerSelf.postMessage({ id, ok: false, error: errorMessage(error) });
  }
});

function neuralPosition(game: GameSnapshot): NeuralSample {
  const encoded = encodeNeuralPositionFeatures(game, game.turn);
  return {
    sideToMove: game.turn,
    boardCount: encoded.boardCount,
    positionKey: positionKey(game),
    features: encoded.values
  };
}

function positionKey(game: GameSnapshot): string {
  const text = JSON.stringify(game);
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
