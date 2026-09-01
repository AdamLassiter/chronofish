import type { ChronofishEngine, Color, GameSnapshot, Move, Piece, Position, Timeline } from "./types.js";
import type { GpuSnapshot, GpuTimeline } from "./ai-snapshot.js";

interface GpuBufferUsageConstants {
  MAP_READ: number;
  COPY_SRC: number;
  COPY_DST: number;
  UNIFORM: number;
  STORAGE: number;
}

interface GpuMapModeConstants {
  READ: number;
}

export const GPUBufferUsage: GpuBufferUsageConstants = (globalThis as unknown as { GPUBufferUsage?: GpuBufferUsageConstants }).GPUBufferUsage ?? {
  MAP_READ: 1,
  COPY_SRC: 4,
  COPY_DST: 8,
  UNIFORM: 64,
  STORAGE: 128
};

export const GPUMapMode: GpuMapModeConstants = (globalThis as unknown as { GPUMapMode?: GpuMapModeConstants }).GPUMapMode ?? {
  READ: 1
};

export type GpuMode = "full" | "hybrid";

export interface GpuSearchOptions {
  depth?: number | undefined;
  nodes?: number | undefined;
  timeMs?: number | undefined;
  gpuMode?: GpuMode | undefined;
  forceFullGpu?: boolean | undefined;
  disableNeural?: boolean | undefined;
  snapshotOverride?: GpuSnapshot | null | undefined;
  sourceGame?: GameSnapshot | undefined;
  temperature?: number | undefined;
  randomSeed?: number | undefined;
}

export interface TurnStatus {
  complete: boolean;
  terminal?: boolean;
  winner?: Color;
  nextTurn: Color;
  presentTime: number;
  pendingPresentBoardCount: number;
  message?: string;
}

export interface RankedCandidate {
  move: Move;
  index: number;
  score: number;
}

export interface MutatedCandidate extends RankedCandidate {
  mutationStatus: number;
  childBoards: Int32Array | null;
}

export interface ScoredCandidates {
  records: Int32Array;
  scores: Int32Array;
}

export interface SearchChoice {
  rank?: number;
  score?: number | undefined;
  moves?: Move[] | undefined;
  move?: Move | undefined;
  principalVariation?: Move[][] | undefined;
  depth?: number | undefined;
  nodes?: number | undefined;
  gpuSearch?: string | undefined;
  gpuTerminal?: boolean | undefined;
  tactical?: boolean | undefined;
}

export type SearchResultReason = "royal-capture" | "threefold-repetition" | "stalemate";

export interface SearchResult {
  status: string;
  moves: Move[];
  score?: number | undefined;
  choices?: SearchChoice[] | undefined;
  principalVariation?: Move[][] | undefined;
  depth?: number | undefined;
  nodes?: number | undefined;
  terminal?: boolean | undefined;
  winner?: Color | undefined;
  resultReason?: SearchResultReason | undefined;
  gpu?: boolean | undefined;
  gpuMode?: GpuMode | undefined;
  gpuTerminal?: boolean | undefined;
  gpuSnapshot?: string | undefined;
  gpuSearch?: string | undefined;
  tactical?: boolean | undefined;
  gpuDiagnostics?: GpuSearchDiagnostics | undefined;
  authoritativeReplay?: boolean | undefined;
  incompleteMoves?: Move[] | undefined;
  pendingPresentBoardCount?: number | undefined;
}

export interface GpuSearchDiagnostics {
  frontierWidth?: number;
  candidateCapacity?: number;
  selectedCount?: number;
  maxBoards?: number;
  dispatchCandidateLimit?: number;
  cycles?: number;
  completedDepth?: number;
  nodes?: number;
  readbacks?: number;
  candidateOverflow?: number;
  tacticalCandidates?: number;
  selectedTacticalCandidates?: number;
  candidateSelectionRate?: number;
  tacticalSelectionRate?: number;
  effectiveBranchingFactor?: number;
  searchController?: "puct-frontier-graph";
  progressiveWideningLimit?: number;
  graphDeduplication?: number;
  nnCacheHits?: number;
  nnCacheMisses?: number;
  nnCacheStores?: number;
  nnCacheHitRate?: number;
  inferencePrecision?: string | undefined;
  fastNetPolicyFormat?: string | undefined;
  fastNetPolicyScale?: number | undefined;
  fastNetPolicyMaxAbsError?: number | undefined;
  fastNet?: string;
  bigNet?: string;
  legalChoiceCount?: number;
  legalTacticalChoiceCount?: number;
  topPolicyChoiceAgreement?: number;
  top5PolicyChoiceAgreement?: number;
  top20PolicyChoiceAgreement?: number;
  selectedMovePrunedRisk?: number;
  selectedMoveTactical?: number;
  model?: "neural" | "heuristic";
  latencyMs?: number;
  nodesPerSecond?: number;
  candidateWorkgroupSize?: number;
  mutationTileSize?: number;
  neuralBatchSize?: number;
}

export interface ReplySearchResult {
  score: number;
  move?: Move | undefined;
}

export interface LegalTargetSelection {
  source: { piece: Piece; position: Position } | null;
  targets: Position[];
}

export interface WorkerRequest {
  id: number | string;
  type?: "search" | "legalTargets" | "applyMove" | "submitTurn" | "debugLoseDevice" | "setModel";
  modelBytes?: ArrayBuffer;
  game?: GameSnapshot;
  position?: Position;
  move?: Move;
  depth?: number;
  minDepth?: number;
  nodes?: number;
  timeMs?: number;
  partitionIndex?: number;
  partitionCount?: number;
  temperature?: number;
  randomSeed?: number;
  gpuMode?: GpuMode;
  forceFullGpu?: boolean;
  disableNeural?: boolean;
  notation?: string;
  turns?: Move[][];
  stagedMoves?: Move[];
}

export type PendingPresentBoard = { timeline: GpuTimeline | Timeline; board: { time: number; sideToMove: Color } };

export type FrontierRuntime = {
  device: GPUDevice;
  pipeline: import("./ai-frontier.js").FrontierGpuPipeline;
  neural: import("./ai-frontier-neural.js").FrontierNeuralEvaluator;
};

export type ValidationEngine = ChronofishEngine;
