import type { Color, GameSnapshot, Move, BoardSnapshot, Timeline } from "./types.js";
import type { CompactValueModel, EncodedCompactModel, TrainingConfig as GpuTrainingConfig, TrainingMetrics, TrainingSample } from "./training-gpu.js";

export const BUFFER_KEY = "value-policy-buffer";

export type TrainingLabelKind = "search" | "outcome" | "distilled" | string;
export type TrainingSubject = "gpu" | "cpu";
export type TrainingMode = "vsGpu" | "vsCpu" | "self" | "distill" | "curriculum" | "tactical";
export type CpuParameters = Record<string, number>;

export interface WorkerScope {
  addEventListener(type: "message", listener: (event: MessageEvent<TrainingWorkerRequest>) => void | Promise<void>): void;
  postMessage(message: TrainingWorkerResponse): void;
}

export interface TrainingWorkerRequest {
  id: number;
  type?: "train" | "validateLossLogs";
  game?: GameSnapshot;
  config?: Partial<NormalizedTrainingConfig>;
  candidateModel?: ArrayBuffer;
}

export type TrainingWorkerResponse = Record<string, unknown> & {
  id: number;
  ok: boolean;
};

export type ProgressMessage = Record<string, unknown>;
export type ProgressCallback = (message: ProgressMessage) => void;

export interface NormalizedTrainingConfig extends GpuTrainingConfig {
  trainingSubject: TrainingSubject;
  trainingModes: TrainingMode[];
  runSeed: number;
  samples: number;
  selfPlayWorkers: number;
  searchWorkers: number;
  explorationTemperature: number;
  depth: number;
  nodes: number;
  maxBuffer: number;
  lossLogReplay: number;
  cpuDepth: number;
  cpuNodes: number;
  cpuTrainingTimeMs: number;
  cpuCandidates: number;
  cpuFinalists: number;
  cpuPairBatch: number;
  cpuOpponentVariants: number;
  cpuScreeningOpponentVariants: number;
  cpuRoundsPerVariant: number;
  cpuHallOfFameEntries: number;
  cpuLeagueContenders: number;
  cpuLeagueHallOfFameEntries: number;
  cpuMinPairs: number;
  cpuMaxPairs: number;
  cpuDrawWindow: number;
  cpuDrawRateLimit: number;
  cpuMaxMatchPlies: number;
  cpuMaxMatchTimeMs: number;
  cpuMaxGenerationsWithoutCandidate: number;
  cpuWorkers: number;
  cpuTrainSeconds: number;
  labelWorkers?: number;
  metrics?: TrainingRunMetrics | null;
}

export interface TrainingRunMetrics extends TrainingMetrics {
  startedAt: number;
  phases: Record<string, number>;
  sampleCounts?: Record<string, number>;
  searchPositionCount?: number;
  searchLabelCount?: number;
  lossLogValidation?: LossLogValidation | null;
}

export interface MetricsSummary {
  totalMs: number;
  phases: Record<string, number>;
  sampleRates: Record<string, number>;
  lossLogValidation: LossLogValidation | null;
}

export interface CpuTrainingResult {
  parametersJson: string;
  score: number;
}

export interface CpuReferenceScore {
  baselineScore?: number;
  baselineMoves?: Move[];
  gpuScore?: number;
  gpuMoves?: Move[];
}

export interface EncodedPosition {
  game: GameSnapshot;
  sample: TrainingSample;
}

export interface LabelWorkerSample extends TrainingSample {
  outcomeTurn?: Color;
  ply?: number;
}

export interface AiSearchResult {
  moves?: Move[];
  score?: number;
  cpuSearch?: string | null;
  gpuSearch?: string | null;
}

export interface AiWorkerStatus {
  complete?: boolean;
  terminal?: boolean;
  winner?: Color;
  nextTurn?: Color;
}

export interface AppliedWorkerTurn {
  game: GameSnapshot;
  status: AiWorkerStatus;
  winner: Color | null;
}

export interface AiWorkerResponse {
  ok: boolean;
  result?: AiSearchResult;
  game?: GameSnapshot;
  status?: AiWorkerStatus;
  sample?: TrainingSample;
  samples?: TrainingSample[];
  error?: string;
}

export interface LossLogDecision {
  game?: GameSnapshot;
  selectedMoves?: Move[];
  ply?: number;
  botColor?: Color;
  selectedScore?: number;
}

export interface LossLog {
  logPath?: string;
  decisions?: LossLogDecision[];
}

export interface LossLogValidation {
  checked: number;
  changed: number;
  unchanged: number;
  skipped: number;
  failed: boolean;
  examples: LossLogValidationExample[];
}

export interface LossLogValidationExample {
  logPath: string | null;
  ply: number | null;
  botColor: Color | null;
  previous: string;
  current: string;
  previousScore: number | null;
  currentScore: number | null;
}

export interface LabelJob {
  game: GameSnapshot;
  index: number;
  seed: number;
  plies: number;
}

export interface WorkerRequestPayload extends Record<string, unknown> {
  type?: string;
  game?: GameSnapshot;
  games?: GameSnapshot[];
  move?: Move | null;
  nodes?: number;
  depth?: number;
  timeMs?: number;
  parametersJson?: string;
}

export interface ReplayDb extends IDBDatabase {}

export type TrainingWorkerModel = CompactValueModel;
export type TrainingWorkerEncodedModel = EncodedCompactModel;
export type TrainingWorkerCpuParameters = CpuParameters;
export type TrainingWorkerBoard = BoardSnapshot;
export type TrainingWorkerTimeline = Timeline;
