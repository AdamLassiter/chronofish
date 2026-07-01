import type { Color } from "./types.js";

export type TrainingLabelKind = "search" | "outcome" | "distilled" | "unknown" | string;

export interface TrainingSample {
  sideToMove?: Color;
  boardCount?: number;
  positionKey?: string;
  features: number[] | Float32Array;
  label: number;
  labelKind?: TrainingLabelKind;
  labelWeight?: number;
  baseLabelWeight?: number;
  labelMass?: number;
  observationCount?: number;
  policy?: number | null;
  pseudo?: boolean;
}

export interface TrainingMetrics {
  phases?: Record<string, number>;
  [key: string]: unknown;
}

export interface TrainingConfig {
  learningRate: number;
  epochs: number;
  batchSize: number;
  validationSplit?: number;
  validationInterval?: number;
  patience: number;
  weightDecay: number;
  labelCounts?: Record<string, number>;
  metrics?: TrainingMetrics | null;
}

export interface SparseProjectionFeatures {
  offsets: Uint32Array;
  indices: Uint32Array;
  values: Float32Array;
  byteLength: number;
}
