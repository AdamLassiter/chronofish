import { train, predictValues, decodeCompactModel } from "./training-gpu.js";
import type { Color, GameSnapshot, Move, BoardSnapshot, Timeline } from "./types.js";
import type { CompactValueModel, EncodedCompactModel, TrainingConfig as GpuTrainingConfig, TrainingMetrics, TrainingSample } from "./training-gpu.js";
const POLICY_BUCKETS = 257;
const BUFFER_KEY = "value-policy-buffer";
const PROJECTION_SIZE = 2048;
const PROJECTION_SEED = 2166136261;
const MAX_PLAYOUT_PLIES = 10;
const HIDDEN_LAYERS = [1024, 512, 256];
const VALUE_EPOCHS_PER_SUBMIT = 64;
const POLICY_STEPS_PER_SUBMIT = 64;
const DEFAULT_BATCH_SIZE = 1024;
const DEFAULT_VALIDATION_SPLIT = 0.1;
const DEFAULT_PATIENCE = 12;
const DEFAULT_WEIGHT_DECAY = 0.00001;
const PROJECTION_CHUNK_SIZE = 256;
const LABEL_REQUEST_MIN_TIMEOUT_MS = 30000;
const LABEL_REQUEST_MAX_TIMEOUT_MS = 120000;
const LABEL_REQUEST_NODE_TIMEOUT_FACTOR_MS = 3;
const TRAINING_IO_TIMEOUT_MS = 15000;
let cachedGpuAdapter: GPUAdapter | null = null;
let cachedGpuDevice: GPUDevice | null = null;
const pipelineCache = new Map<string, GPUComputePipeline>();

type TrainingLabelMode = "mixed" | "search" | "selfPlay" | "distill";
type TrainingLabelKind = "search" | "outcome" | "distilled" | string;
type TrainingSubject = "gpu" | "cpu";
type TrainingTarget = "trainGpu" | "trainCpu" | "trainBoth";
type CpuTrainingTarget = "vsCpu" | "vsGpu" | "vsBoth";

interface WorkerScope {
  addEventListener(type: "message", listener: (event: MessageEvent<TrainingWorkerRequest>) => void | Promise<void>): void;
  postMessage(message: TrainingWorkerResponse): void;
}

interface TrainingWorkerRequest {
  id: number;
  type?: "train" | "validateLossLogs";
  game?: GameSnapshot;
  config?: Partial<NormalizedTrainingConfig>;
}

type TrainingWorkerResponse = Record<string, unknown> & {
  id: number;
  ok: boolean;
};

type ProgressMessage = Record<string, unknown>;
type ProgressCallback = (message: ProgressMessage) => void;

interface NormalizedTrainingConfig extends GpuTrainingConfig {
  trainingSubject: TrainingSubject;
  trainingTarget: TrainingTarget;
  cpuTrainingTarget: CpuTrainingTarget;
  labelMode: TrainingLabelMode;
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
  cpuWorkers: number;
  cpuTrainSeconds: number;
  labelWorkers?: number;
  metrics?: TrainingRunMetrics | null;
}

interface TrainingRunMetrics extends TrainingMetrics {
  startedAt: number;
  phases: Record<string, number>;
  sampleCounts?: Record<string, number>;
  searchPositionCount?: number;
  searchLabelCount?: number;
  lossLogValidation?: LossLogValidation | null;
}

interface MetricsSummary {
  totalMs: number;
  phases: Record<string, number>;
  sampleRates: Record<string, number>;
  lossLogValidation: LossLogValidation | null;
}

interface CpuTrainingResult {
  parametersJson: string;
  score: number;
}

interface EncodedPosition {
  game: GameSnapshot;
  sample: TrainingSample;
}

interface LabelWorkerSample extends TrainingSample {
  outcomeTurn?: Color;
  ply?: number;
}

interface AiSearchResult {
  moves?: Move[];
  score?: number;
  cpuSearch?: string | null;
  gpuSearch?: string | null;
}

interface AiWorkerStatus {
  complete?: boolean;
  terminal?: boolean;
  winner?: Color;
  nextTurn?: Color;
}

interface AiWorkerResponse {
  ok: boolean;
  result?: AiSearchResult;
  game?: GameSnapshot;
  status?: AiWorkerStatus;
  sample?: TrainingSample;
  samples?: TrainingSample[];
  error?: string;
}

interface LossLogDecision {
  game?: GameSnapshot;
  selectedMoves?: Move[];
  ply?: number;
  botColor?: Color;
  selectedScore?: number;
}

interface LossLog {
  logPath?: string;
  decisions?: LossLogDecision[];
}

interface LossLogValidation {
  checked: number;
  changed: number;
  unchanged: number;
  skipped: number;
  failed: boolean;
  examples: LossLogValidationExample[];
}

interface LossLogValidationExample {
  logPath: string | null;
  ply: number | null;
  botColor: Color | null;
  previous: string;
  current: string;
  previousScore: number | null;
  currentScore: number | null;
}

interface LabelJob {
  game: GameSnapshot;
  index: number;
  seed: number;
  plies: number;
}

type CpuParameters = Record<string, number>;

interface WorkerRequestPayload extends Record<string, unknown> {
  type?: string;
  game?: GameSnapshot;
  games?: GameSnapshot[];
  move?: Move | null;
  nodes?: number;
  depth?: number;
  timeMs?: number;
  parametersJson?: string;
}

interface ReplayDb extends IDBDatabase {}

const workerSelf = self as unknown as WorkerScope;

workerSelf.addEventListener("message", async (event) => {
  const { id, type = "train", game, config } = event.data;
  try {
    const metrics = createTrainingMetrics();
    const normalizedConfig = normalizeTrainingConfig(config);
    normalizedConfig.metrics = metrics;
    if (type === "validateLossLogs") {
      const validation = await timed(metrics, "lossLogValidation", () =>
        validateLossLogs(normalizedConfig, (message) => {
          workerSelf.postMessage({ id, ok: true, ...message });
        })
      );
      metrics.lossLogValidation = validation;
      workerSelf.postMessage({
        id,
        ok: true,
        type: "lossLogValidation",
        validation,
        metrics: metricsSummary(metrics)
      });
      return;
    }
    if (normalizedConfig.trainingSubject === "cpu") {
      const cpuResult = await timed(metrics, "cpuTrain", () => trainCpuParameters(game, normalizedConfig, (message) => {
        workerSelf.postMessage({ id, ok: true, ...message });
      }));
      workerSelf.postMessage({
        id,
        ok: true,
        cpuParameters: cpuResult.parametersJson,
        cpuScore: cpuResult.score,
        metrics: metricsSummary(metrics)
      });
      return;
    }
    const [activeModel, loadedBuffer] = await timed(metrics, "load", () => Promise.all([
      fetchActiveModel(),
      loadReplayBuffer()
    ]));
    let buffer = loadedBuffer;
    const samples = await timed(metrics, "collect", () => collectTrainingSamples(game, normalizedConfig, activeModel, (message) => {
      workerSelf.postMessage({ id, ok: true, ...message });
    }, metrics));
    metrics.sampleCounts = labelSourceCounts(samples);
    buffer = appendReplaySamples(buffer, samples, normalizedConfig.maxBuffer);
    await timed(metrics, "saveReplay", () => saveReplayBuffer(buffer));
    const labelCounts = labelSourceCounts(buffer);
    workerSelf.postMessage({
      id,
      ok: true,
      gpuPhase: true,
      bufferSize: buffer.length,
      labelCounts,
      batchSize: normalizedConfig.batchSize,
      selfPlayWorkers: normalizedConfig.selfPlayWorkers,
      searchWorkers: normalizedConfig.searchWorkers,
      metrics: metricsSummary(metrics)
    });
    const model = await timed(metrics, "train", () => train(buffer, normalizedConfig, activeModel, (progressMetrics) => {
      workerSelf.postMessage({ id, ok: true, ...progressMetrics, metrics: metricsSummary(metrics) });
    }));
    model.metrics = metricsSummary(metrics);
    workerSelf.postMessage({
      id,
      ok: true,
      model,
      loss: model.trainingLoss,
      validationLoss: model.validationLoss,
      bestValidationLoss: model.bestValidationLoss,
      earlyStopReason: model.earlyStopReason,
      labelCounts: model.labelCounts,
      replaySize: model.replayBufferSize,
      nonZeroWeights: model.nonZeroWeights,
      metrics: model.metrics
    });
  } catch (error) {
    workerSelf.postMessage({ id, ok: false, error: errorMessage(error) });
  }
});

function normalizeTrainingConfig(config: Partial<NormalizedTrainingConfig> = {}): NormalizedTrainingConfig {
  return {
    ...config,
    trainingSubject: isTrainingSubject(config.trainingSubject) ? config.trainingSubject : "gpu",
    trainingTarget: isTrainingTarget(config.trainingTarget) ? config.trainingTarget : "trainGpu",
    cpuTrainingTarget: isCpuTrainingTarget(config.cpuTrainingTarget) ? config.cpuTrainingTarget : "vsCpu",
    labelMode: isTrainingLabelMode(config.labelMode) ? config.labelMode : "mixed",
    runSeed: randomRunSeed(),
    learningRate: clampNumber(config.learningRate, 0.0001, 0.1, 0.01),
    samples: clampInteger(config.samples, 1, 1024, 64),
    selfPlayWorkers: clampInteger(config.selfPlayWorkers, 1, 16, 2),
    searchWorkers: clampInteger(config.searchWorkers, 1, 16, 2),
    explorationTemperature: clampNumber(config.explorationTemperature, 0, 2, 0.25),
    depth: clampInteger(config.depth, 1, 8, 5),
    nodes: clampInteger(config.nodes, 1, 131072, 16384),
    epochs: clampInteger(config.epochs, 1, 65536, 8192),
    maxBuffer: clampInteger(config.maxBuffer, 16, 16384, 4096),
    batchSize: clampInteger(config.batchSize, 16, 8192, DEFAULT_BATCH_SIZE),
    validationSplit: clampNumber(config.validationSplit, 0, 0.3, DEFAULT_VALIDATION_SPLIT),
    validationInterval: clampInteger(config.validationInterval, 16, 4096, 256),
    patience: clampInteger(config.patience, 1, 64, DEFAULT_PATIENCE),
    weightDecay: clampNumber(config.weightDecay, 0, 0.01, DEFAULT_WEIGHT_DECAY),
    lossLogReplay: clampInteger(config.lossLogReplay, 0, 32, 4),
    cpuDepth: clampInteger(config.cpuDepth, 1, 16, 4),
    cpuNodes: clampInteger(config.cpuNodes, 1, 131072, 8192),
    cpuWorkers: clampInteger(config.cpuWorkers, 1, 32, 16),
    cpuTrainSeconds: clampInteger(config.cpuTrainSeconds, 1, 86400, 3600)
  };
}

function createTrainingMetrics(): TrainingRunMetrics {
  return {
    startedAt: performance.now(),
    phases: Object.create(null)
  };
}

function isTrainingLabelMode(value: unknown): value is TrainingLabelMode {
  return value === "mixed" || value === "search" || value === "selfPlay" || value === "distill";
}

function isTrainingSubject(value: unknown): value is TrainingSubject {
  return value === "gpu" || value === "cpu";
}

function isTrainingTarget(value: unknown): value is TrainingTarget {
  return value === "trainGpu" || value === "trainCpu" || value === "trainBoth";
}

function isCpuTrainingTarget(value: unknown): value is CpuTrainingTarget {
  return value === "vsCpu" || value === "vsGpu" || value === "vsBoth";
}

async function timed<T>(metrics: TrainingRunMetrics | null | undefined, name: string, fn: () => Promise<T> | T): Promise<T> {
  if (!metrics) {
    return fn();
  }
  const startedAt = performance.now();
  try {
    return await fn();
  } finally {
    const elapsed = performance.now() - startedAt;
    metrics.phases[name] = (metrics.phases[name] ?? 0) + elapsed;
  }
}

function metricsSummary(metrics: TrainingRunMetrics | null | undefined): MetricsSummary | null {
  if (!metrics) {
    return null;
  }
  const phases: Record<string, number> = {};
  for (const [name, ms] of Object.entries(metrics.phases)) {
    phases[name] = Math.round(ms);
  }
  const sampleRates: Record<string, number> = {};
  for (const [kind, count] of Object.entries(metrics.sampleCounts ?? {})) {
    const phaseName = `${kind}Labels`;
    const phaseMs = metrics.phases[phaseName] ?? metrics.phases.collect ?? 0;
    if (phaseMs > 0) {
      sampleRates[kind] = Number((count / (phaseMs / 1000)).toFixed(2));
    }
  }
  const searchPositionsMs = metrics.phases.searchPositions ?? 0;
  if (metrics.searchPositionCount && searchPositionsMs > 0) {
    sampleRates.searchPositions = Number((metrics.searchPositionCount / (searchPositionsMs / 1000)).toFixed(2));
  }
  const searchLabelsMs = metrics.phases.searchLabels ?? 0;
  if (metrics.searchLabelCount && searchLabelsMs > 0) {
    sampleRates.searchLabels = Number((metrics.searchLabelCount / (searchLabelsMs / 1000)).toFixed(2));
  }
  return {
    totalMs: Math.round(performance.now() - metrics.startedAt),
    phases,
    sampleRates,
    lossLogValidation: metrics.lossLogValidation ?? null
  };
}

async function collectTrainingSamples(
  game: GameSnapshot | undefined,
  config: NormalizedTrainingConfig,
  activeModel: CompactValueModel | null,
  progress: ProgressCallback,
  metrics: TrainingRunMetrics | null = null
): Promise<TrainingSample[]> {
  if (!game) {
    throw new Error("Training requires a game snapshot.");
  }
  const collectors: Array<() => Promise<TrainingSample[]>> = [];
  if (config.trainingTarget === "trainCpu") {
    collectors.push(() => timed(metrics, "cpuLabels", () => collectCpuSearchSamples(game, config, progress)));
  } else if (config.trainingTarget === "trainBoth") {
    collectors.push(() => collectSearchSamples(game, config, progress));
    collectors.push(() => timed(metrics, "cpuLabels", () => collectCpuSearchSamples(game, config, progress)));
  } else if (config.labelMode === "mixed" || config.labelMode === "search") {
    collectors.push(() => collectSearchSamples(game, config, progress));
  }
  if (config.trainingTarget !== "trainCpu" && (config.labelMode === "mixed" || config.labelMode === "selfPlay")) {
    collectors.push(() => timed(metrics, "outcomeLabels", () => collectOutcomeSamples(game, config, progress)));
  }
  if (config.trainingTarget !== "trainCpu" && (config.labelMode === "mixed" || config.labelMode === "distill")) {
    collectors.push(() => timed(metrics, "distillLabels", () => collectDistilledSamples(game, config, activeModel, progress)));
  }

  const collected = await Promise.allSettled(collectors.map((collector) => collector()));
  const results = collected
    .filter((result): result is PromiseFulfilledResult<TrainingSample[]> => result.status === "fulfilled")
    .flatMap((result) => result.value);
  if (config.trainingTarget === "trainBoth") {
    const duel = await timed(metrics, "duelLabels", () => collectCpuGpuDuelSamples(game, config, progress));
    const combined = [...results, ...duel];
    if (combined.length > 0) {
      return combined;
    }
  }
  if (results.length > 0) {
    return results;
  }
  if (activeModel?.outputWeights?.length && config.labelMode !== "distill") {
    return collectDistilledSamples(game, config, activeModel, progress);
  }
  throw new Error("No training labels were collected.");
}

async function trainCpuParameters(
  game: GameSnapshot | undefined,
  config: NormalizedTrainingConfig,
  progress: ProgressCallback
): Promise<CpuTrainingResult> {
  if (!game) {
    throw new Error("CPU training requires a game snapshot.");
  }
  const baseline = await fetchCpuParameters();
  const target = mixedLabelTarget(config, 16);
  const positions = await collectGpuPositions(game, config, target, progress, "cpu");
  const sampleGames = positions.map((position) => position.game);
  if (!sampleGames.length) {
    throw new Error("CPU training could not sample positions.");
  }
  const candidateCount = Math.max(4, Math.min(64, config.cpuWorkers * 4));
  const candidates = [baseline];
  for (let index = 1; index < candidateCount; index += 1) {
    candidates.push(mutateCpuParameters(baseline, config.runSeed ^ index));
  }
  const workerCount = Math.min(candidates.length, Math.max(1, config.cpuWorkers));
  let nextCandidate = 0;
  let collected = 0;
  let bestParameters = baseline;
  let bestScore = Number.NEGATIVE_INFINITY;
  progress({ sampleCount: candidates.length, labelWorkers: workerCount, labelKind: "cpu-train" });
  await Promise.all(Array.from({ length: workerCount }, () => runCandidateWorker()));
  return {
    parametersJson: JSON.stringify(bestParameters, null, 2),
    score: bestScore
  };

  async function runCandidateWorker(): Promise<void> {
    const candidateWorker = new Worker("./cpu-ai-worker.js", { type: "module" });
    const baselineWorker = new Worker("./cpu-ai-worker.js", { type: "module" });
    const gpuWorker = new Worker("./ai-worker.js", { type: "module" });
    try {
      while (nextCandidate < candidates.length) {
        const index = nextCandidate;
        nextCandidate += 1;
        const candidate = candidates[index];
        if (!candidate) {
          continue;
        }
        const score = await scoreCpuCandidate(candidate, baseline, sampleGames, config, candidateWorker, baselineWorker, gpuWorker);
        if (score > bestScore) {
          bestScore = score;
          bestParameters = candidate;
        }
        collected += 1;
        progress({ collected, sampleCount: candidates.length, labelWorkers: workerCount, labelKind: "cpu-train" });
      }
    } finally {
      candidateWorker.terminate();
      baselineWorker.terminate();
      gpuWorker.terminate();
    }
  }
}

async function scoreCpuCandidate(
  candidate: CpuParameters,
  baseline: CpuParameters,
  games: GameSnapshot[],
  config: NormalizedTrainingConfig,
  candidateWorker: Worker,
  baselineWorker: Worker,
  gpuWorker: Worker
): Promise<number> {
  let score = 0;
  const candidateJson = JSON.stringify(candidate);
  const baselineJson = JSON.stringify(baseline);
  for (let index = 0; index < games.length; index += 1) {
    const game = games[index];
    if (!game) {
      continue;
    }
    const candidateResult = await requestWorker(candidateWorker, {
      type: "search",
      game,
      depth: config.cpuDepth,
      nodes: config.cpuNodes,
      timeMs: workerSearchTimeMs({ nodes: config.cpuNodes }),
      parametersJson: candidateJson
    }, workerRequestTimeout({ nodes: config.cpuNodes }));
    const candidateScore = candidateResult.result?.score ?? 0;
    if (config.cpuTrainingTarget === "vsCpu" || config.cpuTrainingTarget === "vsBoth") {
      const baselineResult = await requestWorker(baselineWorker, {
        type: "search",
        game,
        depth: config.cpuDepth,
        nodes: config.cpuNodes,
        timeMs: workerSearchTimeMs({ nodes: config.cpuNodes }),
        parametersJson: baselineJson
      }, workerRequestTimeout({ nodes: config.cpuNodes }));
      score += candidateScore - (baselineResult.result?.score ?? 0);
      score += moveAgreementBonus(candidateResult.result?.moves, baselineResult.result?.moves);
    }
    if (config.cpuTrainingTarget === "vsGpu" || config.cpuTrainingTarget === "vsBoth") {
      const gpuResult = await requestWorker(gpuWorker, {
        type: "search",
        game,
        depth: config.depth,
        nodes: config.nodes,
        timeMs: workerSearchTimeMs(config),
        gpuMode: "full"
      }, workerRequestTimeout(config));
      score += candidateScore - (gpuResult.result?.score ?? 0);
      score += moveAgreementBonus(candidateResult.result?.moves, gpuResult.result?.moves);
    }
  }
  return score / Math.max(1, games.length);
}

function moveAgreementBonus(left: Move[] | undefined, right: Move[] | undefined): number {
  const leftKey = botTrainingMovesKey(left ?? []);
  const rightKey = botTrainingMovesKey(right ?? []);
  return leftKey && rightKey && leftKey === rightKey ? 25 : 0;
}

function botTrainingMovesKey(moves: Move[]): string {
  return moves.map((move) => [
    move.from?.timelineId,
    move.from?.time,
    move.from?.x,
    move.from?.y,
    move.to?.timelineId,
    move.to?.time,
    move.to?.x,
    move.to?.y
  ].join(",")).join("|");
}

function mutateCpuParameters(base: CpuParameters, seed: number): CpuParameters {
  let state = seed >>> 0 || 1;
  const nextRandom = (): number => {
    state = Math.imul(state, 1664525) + 1013904223 >>> 0;
    return state / 0xffffffff;
  };
  const next = { ...base };
  for (const [key, value] of Object.entries(base)) {
    if (!Number.isFinite(value) || key === "king" || key === "royal_queen") {
      continue;
    }
    if (nextRandom() > 0.28) {
      continue;
    }
    const spread = Math.max(1, Math.round(Math.abs(value) * 0.08));
    const delta = Math.round((nextRandom() * 2 - 1) * spread);
    next[key] = Math.max(-10_000, Math.min(10_000, Math.round(value + delta)));
  }
  return next;
}

async function collectSearchSamples(game: GameSnapshot, config: NormalizedTrainingConfig, progress: ProgressCallback): Promise<TrainingSample[]> {
  const target = mixedLabelTarget(config, 16);
  const positions = await timed(config.metrics, "searchPositions", () => collectGpuPositions(game, config, target, progress, "search"));
  if (config.metrics) {
    config.metrics.searchPositionCount = positions.length;
  }
  const workerCount = Math.min(positions.length, config.searchWorkers ?? 1);
  const samples: Array<TrainingSample | null> = new Array(positions.length).fill(null);
  let nextPosition = 0;
  let collected = 0;
  progress({ sampleCount: positions.length, labelWorkers: workerCount, labelKind: "search", labelPhase: "labels" });
  await timed(config.metrics, "searchLabels", () =>
    Promise.all(Array.from({ length: workerCount }, (_, workerIndex) => runSearchWorker(workerIndex)))
  );
  const filtered = samples.filter((sample): sample is TrainingSample => Boolean(sample));
  if (config.metrics) {
    config.metrics.searchLabelCount = filtered.length;
  }
  return filtered;

  async function runSearchWorker(workerIndex: number): Promise<void> {
    const ai = new Worker("./ai-worker.js", { type: "module" });
    try {
      while (nextPosition < positions.length) {
        const index = nextPosition;
        nextPosition += 1;
        const position = positions[index];
        if (!position) {
          continue;
        }
        samples[index] = await searchLabelSample(ai, position, index, workerIndex);
        collected += samples[index] ? 1 : 0;
        progress({ collected, sampleCount: positions.length, labelWorkers: workerCount, labelKind: "search", labelPhase: "labels" });
      }
    } finally {
      ai.terminate();
    }
  }

  async function searchLabelSample(ai: Worker, position: EncodedPosition, index: number, workerIndex: number): Promise<TrainingSample | null> {
    try {
      const response = await requestWorker(ai, {
        type: "search",
        game: position.game,
        depth: config.depth,
        nodes: config.nodes,
        timeMs: workerSearchTimeMs(config),
        gpuMode: "full",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("search", index, searchSeed(position.sample, config.runSeed ^ workerIndex ^ 0x51a7_0001))
      }, workerRequestTimeout(config));
      const result = response.result;
      if (!result?.moves?.length) {
        return null;
      }
      return {
        ...position.sample,
        label: normalizeSearchScore(result.score ?? 0),
        policy: policyBucket(result.moves[0]),
        labelKind: "search",
        labelWeight: 1.0,
        pseudo: false
      };
    } catch {
      return null;
    }
  }
}

async function collectCpuSearchSamples(game: GameSnapshot, config: NormalizedTrainingConfig, progress: ProgressCallback): Promise<TrainingSample[]> {
  const target = mixedLabelTarget(config, config.trainingTarget === "trainBoth" ? 16 : 1);
  const positions = await timed(config.metrics, "cpuPositions", () => collectGpuPositions(game, config, target, progress, "cpu"));
  const workerCount = Math.min(positions.length, Math.max(1, config.cpuWorkers ?? 1));
  const samples: Array<TrainingSample | null> = new Array(positions.length).fill(null);
  let nextPosition = 0;
  let collected = 0;
  progress({ sampleCount: positions.length, labelWorkers: workerCount, labelKind: "cpu", labelPhase: "labels" });
  await Promise.all(Array.from({ length: workerCount }, (_, workerIndex) => runCpuWorker(workerIndex)));
  return samples.filter((sample): sample is TrainingSample => Boolean(sample));

  async function runCpuWorker(workerIndex: number): Promise<void> {
    const ai = new Worker("./cpu-ai-worker.js", { type: "module" });
    try {
      while (nextPosition < positions.length) {
        const index = nextPosition;
        nextPosition += 1;
        const position = positions[index];
        if (!position) {
          continue;
        }
        samples[index] = await cpuSearchLabelSample(ai, position, index, workerIndex);
        collected += samples[index] ? 1 : 0;
        progress({ collected, sampleCount: positions.length, labelWorkers: workerCount, labelKind: "cpu", labelPhase: "labels" });
      }
    } finally {
      ai.terminate();
    }
  }

  async function cpuSearchLabelSample(ai: Worker, position: EncodedPosition, index: number, workerIndex: number): Promise<TrainingSample | null> {
    try {
      const response = await requestWorker(ai, {
        type: "search",
        game: position.game,
        depth: config.cpuDepth,
        nodes: config.cpuNodes,
        timeMs: workerSearchTimeMs({ nodes: config.cpuNodes }),
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("cpu", index, searchSeed(position.sample, config.runSeed ^ workerIndex ^ 0xc911_0001))
      }, workerRequestTimeout({ nodes: config.cpuNodes }));
      const result = response.result;
      if (!result?.moves?.length) {
        return null;
      }
      return {
        ...position.sample,
        label: normalizeSearchScore(result.score ?? 0),
        policy: policyBucket(result.moves[0]),
        labelKind: "cpu",
        labelWeight: config.trainingTarget === "trainBoth" ? 1.1 : 1.0,
        pseudo: false
      };
    } catch {
      return null;
    }
  }
}

async function collectCpuGpuDuelSamples(game: GameSnapshot, config: NormalizedTrainingConfig, progress: ProgressCallback): Promise<TrainingSample[]> {
  const target = mixedLabelTarget(config, 8);
  if (target <= 0) {
    return [];
  }
  const workerCount = Math.min(target, Math.max(1, Math.min(config.cpuWorkers, config.searchWorkers)));
  const targets = splitWork(target, workerCount);
  let collected = 0;
  progress({ sampleCount: target, labelWorkers: workerCount, labelKind: "duel" });
  const results = await Promise.all(targets.map((count, workerIndex) =>
    collectCpuGpuDuelRollout(game, config, count, workerIndex, (count) => {
      collected += count;
      progress({ collected, sampleCount: target, labelWorkers: workerCount, labelKind: "duel" });
    })
  ));
  return results.flat().slice(0, target);
}

async function collectCpuGpuDuelRollout(
  game: GameSnapshot,
  config: NormalizedTrainingConfig,
  target: number,
  workerIndex: number,
  progress: (count: number) => void
): Promise<TrainingSample[]> {
  const gpu = new Worker("./ai-worker.js", { type: "module" });
  const cpu = new Worker("./cpu-ai-worker.js", { type: "module" });
  const encoder = new Worker("./training-label-worker.js", { type: "module" });
  const samples: LabelWorkerSample[] = [];
  try {
    let current = cloneGame(game);
    current = await warmupSelfPlayPosition(gpu, current, config, workerIndex);
    const cpuColor = workerIndex % 2 === 0 ? current.turn : oppositeColor(current.turn);
    const maxPlies = Math.max(MAX_PLAYOUT_PLIES, target + workerIndex);
    for (let ply = 0; ply < maxPlies && samples.length < target; ply += 1) {
      const beforeTurn = current.turn;
      const encoded = await encodePosition(encoder, current);
      const useCpu = beforeTurn === cpuColor;
      const response = await requestWorker(useCpu ? cpu : gpu, {
        type: "search",
        game: current,
        depth: useCpu ? config.cpuDepth : config.depth,
        nodes: useCpu ? config.cpuNodes : config.nodes,
        timeMs: workerSearchTimeMs({ nodes: useCpu ? config.cpuNodes : config.nodes }),
        gpuMode: "full",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("duel", ply, searchSeed(encoded, config.runSeed ^ workerIndex ^ 0xd0e1_0001))
      }, workerRequestTimeout({ nodes: useCpu ? config.cpuNodes : config.nodes }));
      const result = response.result;
      const move = result?.moves?.[0];
      if (!move) {
        break;
      }
      samples.push({
        ...encoded,
        label: normalizeSearchScore(result.score ?? 0),
        policy: policyBucket(move),
        labelKind: "duel",
        labelWeight: 1.35,
        outcomeTurn: beforeTurn,
        ply: ply + workerIndex * MAX_PLAYOUT_PLIES
      });
      progress(1);
      const previous = current;
      const applied = await requestWorker(gpu, { type: "applyMove", game: current, move }, workerRequestTimeout(config));
      if (!applied.game) {
        break;
      }
      current = applied.game;
      const winner = royalCaptureWinner(previous, current, beforeTurn);
      if (winner) {
        return backfillOutcomeLabels(samples, winner).map((sample) => ({
          ...sample,
          labelKind: "duel",
          labelWeight: 1.35
        }));
      }
      const status = await requestWorker(gpu, { type: "submitTurn", game: current }, workerRequestTimeout(config));
      if (status.status?.terminal && status.status.winner) {
        return backfillOutcomeLabels(samples, status.status.winner).map((sample) => ({
          ...sample,
          labelKind: "duel",
          labelWeight: 1.35
        }));
      }
      if (status.status?.complete && status.status.nextTurn) {
        current = { ...current, turn: status.status.nextTurn };
      }
    }
    return samples.map(({ outcomeTurn, ply, ...sample }) => sample);
  } catch {
    return samples.map(({ outcomeTurn, ply, ...sample }) => sample);
  } finally {
    gpu.terminate();
    cpu.terminate();
    encoder.terminate();
  }
}

async function collectOutcomeSamples(game: GameSnapshot, config: NormalizedTrainingConfig, progress: ProgressCallback): Promise<TrainingSample[]> {
  const target = mixedLabelTarget(config, 16);
  if (target <= 0) {
    return [];
  }
  const workerCount = Math.min(target, config.selfPlayWorkers ?? 1);
  const targets = splitWork(target, workerCount);
  let collected = 0;
  const report = (count: number): void => {
    collected += count;
    progress({
      collected,
      sampleCount: target,
      labelWorkers: workerCount,
      labelKind: "outcome"
    });
  };
  progress({
    sampleCount: target,
    labelWorkers: workerCount,
    labelKind: "outcome"
  });
  const results = await Promise.all(targets.map((count, workerIndex) =>
    collectOutcomeRollout(game, config, count, workerIndex, report)
  ));
  return results.flat().slice(0, target);
}

async function collectOutcomeRollout(
  game: GameSnapshot,
  config: NormalizedTrainingConfig,
  target: number,
  workerIndex: number,
  progress: (count: number) => void
): Promise<TrainingSample[]> {
  const ai = new Worker("./ai-worker.js", { type: "module" });
  const encoder = new Worker("./training-label-worker.js", { type: "module" });
  const samples: LabelWorkerSample[] = [];
  try {
    let current = cloneGame(game);
    current = await warmupSelfPlayPosition(ai, current, config, workerIndex);
    const maxPlies = Math.max(MAX_PLAYOUT_PLIES, target + workerIndex);
    for (let ply = 0; ply < maxPlies && samples.length < target; ply += 1) {
      const beforeTurn = current.turn;
      const encoded = await encodePosition(encoder, current);
      const response = await requestWorker(ai, {
        type: "search",
        game: current,
        depth: config.depth,
        nodes: config.nodes,
        timeMs: workerSearchTimeMs(config),
        gpuMode: "full",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("outcome", ply, searchSeed(encoded, config.runSeed ^ workerIndex ^ 0x0c70_0001))
      }, workerRequestTimeout(config));
      const result = response.result;
      const move = result?.moves?.[0];
      if (!move) {
        break;
      }
      samples.push({
        ...encoded,
        label: normalizeSearchScore(result.score ?? 0),
        policy: policyBucket(move),
        labelKind: "outcome",
        labelWeight: 1.25,
        outcomeTurn: beforeTurn,
        ply: ply + workerIndex * MAX_PLAYOUT_PLIES
      });
      progress(1);
      const previous = current;
      const applied = await requestWorker(ai, {
        type: "applyMove",
        game: current,
        move
      }, workerRequestTimeout(config));
      if (!applied.game) {
        break;
      }
      current = applied.game;
      const winner = royalCaptureWinner(previous, current, beforeTurn);
      if (winner) {
        return backfillOutcomeLabels(samples, winner);
      }
      const status = await requestWorker(ai, {
        type: "submitTurn",
        game: current
      }, workerRequestTimeout(config));
      if (status.status?.terminal && status.status.winner) {
        return backfillOutcomeLabels(samples, status.status.winner);
      }
      if (status.status?.complete && status.status.nextTurn) {
        current = { ...current, turn: status.status.nextTurn };
      }
    }
    return samples.map(({ outcomeTurn, ply, ...sample }) => sample);
  } catch {
    return samplesFromPartialOutcome(samples);
  } finally {
    ai.terminate();
    encoder.terminate();
  }
}

function splitWork(total: number, workers: number): number[] {
  return Array.from({ length: workers }, (_, index) =>
    Math.floor(total / workers) + (index < total % workers ? 1 : 0)
  ).filter((count) => count > 0);
}

async function warmupSelfPlayPosition(ai: Worker, game: GameSnapshot, config: NormalizedTrainingConfig, workerIndex: number): Promise<GameSnapshot> {
  let current = cloneGame(game);
  const warmupPlies = workerIndex === 0 ? 0 : 1 + (workerIndex % Math.max(1, MAX_PLAYOUT_PLIES - 1));
  for (let ply = 0; ply < warmupPlies; ply += 1) {
    try {
      const response = await requestWorker(ai, {
        type: "search",
        game: current,
        depth: Math.max(1, Math.min(2, config.depth)),
        nodes: Math.max(1, Math.min(1024, config.nodes)),
        timeMs: Math.min(5000, workerSearchTimeMs(config)),
        gpuMode: "full",
        partitionIndex: workerIndex,
        partitionCount: config.selfPlayWorkers ?? 1,
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("warmup", ply, config.runSeed ^ workerIndex ^ 0x0aa5_0001)
      }, workerRequestTimeout({ ...config, nodes: 1024 }));
      const move = response.result?.moves?.[0];
      if (!move) {
        break;
      }
      const applied = await requestWorker(ai, {
        type: "applyMove",
        game: current,
        move
      }, workerRequestTimeout(config));
      if (!applied.game) {
        break;
      }
      current = applied.game;
      const status = await requestWorker(ai, {
        type: "submitTurn",
        game: current
      }, workerRequestTimeout(config));
      if (status.status?.terminal) {
        break;
      }
      if (status.status?.complete && status.status.nextTurn) {
        current = { ...current, turn: status.status.nextTurn };
      }
    } catch {
      break;
    }
  }
  return current;
}

async function collectDistilledSamples(
  game: GameSnapshot,
  config: NormalizedTrainingConfig,
  activeModel: CompactValueModel | null,
  progress: ProgressCallback
): Promise<TrainingSample[]> {
  if (!activeModel?.outputWeights?.length) {
    return [];
  }
  const positions = await collectSamples(game, config, true, (collected, sampleCount, labelWorkers) => {
    progress({ collected, sampleCount, labelWorkers, labelKind: "distilled" });
  });
  const labels = await predictValues(positions, activeModel);
  return positions.map((sample, index) => ({
    ...sample,
    label: labels[index] ?? 0,
    policy: null,
    labelKind: "distilled",
    labelWeight: 0.25,
    pseudo: true
  }));
}

async function collectGpuPositions(
  game: GameSnapshot,
  config: NormalizedTrainingConfig,
  target: number,
  progress: ProgressCallback,
  labelKind: TrainingLabelKind
): Promise<EncodedPosition[]> {
  if (target <= 0) {
    return [];
  }
  const workerCount = Math.min(target, Math.max(1, config.searchWorkers ?? 1));
  const positions: Array<EncodedPosition | null> = new Array(target).fill(null);
  let nextJob = 0;
  let generated = 0;
  progress({ sampleCount: target, labelWorkers: workerCount, labelKind, labelPhase: "positions" });
  await Promise.all(Array.from({ length: workerCount }, (_, workerIndex) => runPositionWorker(workerIndex)));
  return positions.filter((position): position is EncodedPosition => Boolean(position?.sample?.features?.length));

  async function runPositionWorker(workerIndex: number): Promise<void> {
    const ai = new Worker("./ai-worker.js", { type: "module" });
    const encoder = new Worker("./training-label-worker.js", { type: "module" });
    const local: Array<{ index: number; game: GameSnapshot }> = [];
    try {
      while (nextJob < target) {
        const index = nextJob;
        nextJob += 1;
        const positionGame = await generatePositionGame(ai, game, config, index, workerIndex);
        local.push({ index, game: positionGame });
        generated += 1;
        progress({ collected: generated, sampleCount: target, labelWorkers: workerCount, labelKind, labelPhase: "positions" });
      }
      let samples: TrainingSample[] = [];
      try {
        samples = await encodePositions(encoder, local.map((entry) => entry.game));
      } catch {
        samples = [];
      }
      for (let index = 0; index < local.length; index += 1) {
        const entry = local[index];
        if (!entry) {
          continue;
        }
        if (!samples[index]?.features?.length) {
          continue;
        }
        const sample = samples[index];
        if (!sample) {
          continue;
        }
        positions[entry.index] = {
          game: entry.game,
          sample
        };
      }
    } finally {
      ai.terminate();
      encoder.terminate();
    }
  }
}

async function generatePositionGame(ai: Worker, game: GameSnapshot, config: NormalizedTrainingConfig, index: number, workerIndex: number): Promise<GameSnapshot> {
  let current = cloneGame(game);
  const plies = samplePlies(index, false);
  for (let ply = 0; ply < plies; ply += 1) {
    try {
      const shallowConfig = { ...config, nodes: Math.max(1, Math.min(512, config.nodes)) };
      const response = await requestWorker(ai, {
        type: "search",
        game: current,
        depth: Math.max(1, Math.min(2, config.depth)),
        nodes: shallowConfig.nodes,
        timeMs: 3000,
        gpuMode: "full",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("position", index * MAX_PLAYOUT_PLIES + ply, config.runSeed ^ workerIndex ^ 0x9051_0001)
      }, workerRequestTimeout(shallowConfig));
      const move = response.result?.moves?.[0];
      if (!move) {
        break;
      }
      const applied = await requestWorker(ai, { type: "applyMove", game: current, move }, workerRequestTimeout(config));
      if (!applied.game) {
        break;
      }
      current = applied.game;
      const status = await requestWorker(ai, { type: "submitTurn", game: current }, workerRequestTimeout(config));
      if (status.status?.terminal) {
        break;
      }
      if (status.status?.complete && status.status.nextTurn) {
        current = { ...current, turn: status.status.nextTurn };
      }
    } catch {
      break;
    }
  }
  return current;
}

function samplesFromPartialOutcome(samples: LabelWorkerSample[]): TrainingSample[] {
  return samples.map(({ outcomeTurn, ply, ...sample }) => sample);
}

function mixedLabelTarget(config: NormalizedTrainingConfig, divisor: number): number {
  if (config.labelMode !== "mixed") {
    return config.samples;
  }
  return Math.max(1, Math.ceil(config.samples / divisor));
}

function backfillOutcomeLabels(samples: LabelWorkerSample[], winner: Color): TrainingSample[] {
  const maxPly = samples.at(-1)?.ply ?? 0;
  return samples.map(({ outcomeTurn, ply, ...sample }) => ({
    ...sample,
    label: (outcomeTurn === winner ? 1 : -1) * Math.pow(0.96, maxPly - (ply ?? 0)),
    labelKind: "outcome",
    labelWeight: 1.25
  }));
}

function royalCaptureWinner(before: GameSnapshot, after: GameSnapshot, mover: Color): Color | null {
  const opponent = mover === "white" ? "black" : "white";
  return royalCount(after, opponent) < royalCount(before, opponent) ? mover : null;
}

function oppositeColor(color: Color): Color {
  return color === "white" ? "black" : "white";
}

function royalCount(game: GameSnapshot, color: Color): number {
  let count = 0;
  for (const timeline of game.timelines ?? []) {
    const board = latestBoard(timeline);
    for (const row of board?.board ?? []) {
      for (const piece of row ?? []) {
        if (piece?.color === color && ["king", "royalQueen"].includes(piece.type)) {
          count += 1;
        }
      }
    }
  }
  return count;
}

function latestBoard(timeline: Timeline): BoardSnapshot | null {
  const first = timeline.boards[0];
  return first ? timeline.boards.reduce((latest, board) => board.time > latest.time ? board : latest, first) : null;
}

async function encodePosition(worker: Worker, game: GameSnapshot): Promise<TrainingSample> {
  const response = await requestWorker(worker, {
    type: "sample",
    game,
    encodeOnly: true
  }, workerRequestTimeout({ nodes: 1 }));
  if (!response.sample) {
    throw new Error("Position encoder returned no sample.");
  }
  return response.sample;
}

async function encodePositions(worker: Worker, games: GameSnapshot[]): Promise<TrainingSample[]> {
  if (!games.length) {
    return [];
  }
  const response = await requestWorker(worker, {
    type: "batchSample",
    games
  }, workerRequestTimeout({ nodes: games.length }));
  return response.samples ?? [];
}

async function collectSamples(
  game: GameSnapshot,
  config: NormalizedTrainingConfig,
  encodeOnly: boolean,
  progress: (collected: number, sampleCount: number, labelWorkers: number) => void
): Promise<TrainingSample[]> {
  const jobs = Array.from({ length: config.samples }, (_, index) => ({
    game,
    index,
    seed: sampleSeed(JSON.stringify(game), index, encodeOnly ? 0xa11c_e000 : 0x5eed_1000),
    plies: samplePlies(index, encodeOnly)
  }));
  const workerCount = Math.min(
    jobs.length,
    Math.max(1, Math.min(config.labelWorkers ?? autoLabelWorkers(), 8))
  );
  progress(0, jobs.length, workerCount);

  const samples: Array<TrainingSample | null> = new Array(jobs.length).fill(null);
  let nextJob = 0;
  let collected = 0;

  await Promise.all(Array.from({ length: workerCount }, () => runLabelWorker()));
  return samples.filter((sample): sample is TrainingSample => Boolean(sample));

  async function runLabelWorker() {
    const worker = new Worker("./training-label-worker.js", { type: "module" });
    try {
      while (nextJob < jobs.length) {
        const job = jobs[nextJob];
        if (!job) {
          continue;
        }
        nextJob += 1;
        samples[job.index] = await labelSample(worker, job, config, encodeOnly);
        collected += 1;
        progress(collected, jobs.length, workerCount);
      }
    } finally {
      worker.terminate();
    }
  }
}

function labelSample(worker: Worker, job: LabelJob, config: NormalizedTrainingConfig, encodeOnly: boolean): Promise<TrainingSample> {
  const payload = {
    type: "sample",
    game: job.game,
    depth: config.depth,
    nodes: config.nodes,
    encodeOnly,
    seed: job.seed,
    plies: job.plies
  };
  return requestWorker(worker, {
    ...payload,
    timeMs: workerSearchTimeMs(payload)
  }).then((response) => {
    if (!response.sample) {
      throw new Error("Label worker returned no sample.");
    }
    return response.sample;
  });
}

function requestWorker(worker: Worker, payload: WorkerRequestPayload, timeoutMs = workerRequestTimeout(payload)): Promise<AiWorkerResponse> {
  return new Promise((resolve, reject) => {
    const messageId = crypto.randomUUID();
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error("Position worker timed out."));
    }, timeoutMs);
    const handleMessage = (event: MessageEvent<AiWorkerResponse & { id?: string }>) => {
      if (event.data.id !== messageId) {
        return;
      }
      cleanup();
      if (event.data.ok) {
        resolve(event.data);
      } else {
        reject(new Error(event.data.error));
      }
    };
    const handleError = (event: ErrorEvent | MessageEvent) => {
      cleanup();
      const message = event instanceof ErrorEvent ? event.message : "Label worker failed.";
      reject(new Error(message || "Label worker failed."));
    };
    const cleanup = () => {
      clearTimeout(timeout);
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
      worker.removeEventListener("messageerror", handleError);
    };
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.addEventListener("messageerror", handleError);
    worker.postMessage({
      id: messageId,
      ...payload
    });
  });
}

function normalizeSearchScore(score: number): number {
  return Math.max(-1, Math.min(1, score / 20000));
}

function policyBucket(move: Move | null | undefined): number | null {
  if (!move) {
    return null;
  }
  const values = [
    move.from?.timelineId ?? 0,
    move.from?.time ?? 0,
    move.from?.x ?? 0,
    move.from?.y ?? 0,
    move.to?.timelineId ?? 0,
    move.to?.time ?? 0,
    move.to?.x ?? 0,
    move.to?.y ?? 0
  ];
  let hash = 2166136261;
  for (const value of values) {
    hash ^= value & 0xff;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash % POLICY_BUCKETS;
}

function cloneGame(game: GameSnapshot): GameSnapshot {
  return JSON.parse(JSON.stringify(game));
}

function labelSourceCounts(samples: TrainingSample[]): Record<string, number> {
  return samples.reduce<Record<string, number>>((counts, sample) => {
    const key = sample.labelKind ?? (sample.pseudo ? "distilled" : "unknown");
    counts[key] = (counts[key] ?? 0) + 1;
    return counts;
  }, {});
}

function clampInteger(value: unknown, min: number, max: number, fallback: number): number {
  return Math.round(clampNumber(value, min, max, fallback));
}

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, number));
}

function randomRunSeed(): number {
  const values = new Uint32Array(1);
  crypto.getRandomValues(values);
  return values[0]! >>> 0;
}

function workerRequestTimeout(payload: Pick<NormalizedTrainingConfig, "nodes"> | WorkerRequestPayload | { nodes?: number }): number {
  const nodes = Math.max(1, Number(payload.nodes) || 1);
  return Math.min(
    LABEL_REQUEST_MAX_TIMEOUT_MS,
    Math.max(
      LABEL_REQUEST_MIN_TIMEOUT_MS,
      nodes * LABEL_REQUEST_NODE_TIMEOUT_FACTOR_MS
    )
  );
}

function workerSearchTimeMs(payload: Pick<NormalizedTrainingConfig, "nodes"> | WorkerRequestPayload | { nodes?: number }): number {
  const timeout = workerRequestTimeout(payload);
  return Math.max(1000, timeout - 1000);
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      (value) => {
        clearTimeout(timeout);
        resolve(value);
      },
      (error) => {
        clearTimeout(timeout);
        reject(error);
      }
    );
  });
}

function samplePlies(index: number, encodeOnly: boolean): number {
  const stride = encodeOnly ? 2 : 1;
  return 1 + ((index * stride) % MAX_PLAYOUT_PLIES);
}

function sampleSeed(prefix: string, index: number, salt: number): number {
  let hash = salt >>> 0;
  for (let offset = 0; offset < prefix.length; offset += 1) {
    hash ^= prefix.charCodeAt(offset);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  hash ^= index;
  hash = Math.imul(hash, 16777619) >>> 0;
  return hash >>> 0;
}

function searchSeed(value: unknown, salt: number): number {
  let hash = salt >>> 0;
  const text = JSON.stringify(value ?? null);
  for (let offset = 0; offset < text.length; offset += 1) {
    hash ^= text.charCodeAt(offset);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function appendReplaySamples(buffer: TrainingSample[], samples: TrainingSample[], maxBuffer: number): TrainingSample[] {
  const merged = buffer.concat(samples).filter((sample) => Array.isArray(sample.features));
  return merged.slice(Math.max(0, merged.length - maxBuffer));
}

async function validateLossLogs(config: NormalizedTrainingConfig, progress?: ProgressCallback): Promise<LossLogValidation> {
  const logs = await fetchLossLogs(config.lossLogReplay);
  const validation: LossLogValidation = {
    checked: 0,
    changed: 0,
    unchanged: 0,
    skipped: 0,
    failed: false,
    examples: []
  };
  if (!logs.length || config.lossLogReplay <= 0) {
    return validation;
  }

  const ai = new Worker("./ai-worker.js", { type: "module" });
  try {
    for (const log of logs) {
      const decisions = Array.isArray(log.decisions) ? log.decisions : [];
      let logChanged = false;
      for (const decision of decisions) {
        const previousKey = movesKey(decision.selectedMoves);
        if (!decision.game || !previousKey) {
          validation.skipped += 1;
          continue;
        }
        validation.checked += 1;
        progress?.({
          lossLogValidation: {
            checked: validation.checked,
            changed: validation.changed,
            logPath: log.logPath ?? null
          }
        });
        try {
          const response = await requestWorker(ai, {
            type: "search",
            game: decision.game,
            depth: config.depth,
            nodes: config.nodes,
            timeMs: workerSearchTimeMs(config),
            gpuMode: "full",
            temperature: config.explorationTemperature,
            randomSeed: sampleSeed("loss-log", validation.checked, config.runSeed ^ 0x1055_1000)
          }, workerRequestTimeout(config));
          const currentMoves = response.result?.moves ?? [];
          const currentKey = movesKey(currentMoves);
          if (!currentKey) {
            validation.skipped += 1;
            continue;
          }
          if (currentKey !== previousKey) {
            validation.changed += 1;
            logChanged = true;
            validation.examples.push({
              logPath: log.logPath ?? null,
              ply: decision.ply ?? null,
              botColor: decision.botColor ?? null,
              previous: previousKey,
              current: currentKey,
              previousScore: decision.selectedScore ?? null,
              currentScore: response.result?.score ?? null
            });
            break;
          }
          validation.unchanged += 1;
        } catch {
          validation.skipped += 1;
        }
      }
      if (logChanged) {
        continue;
      }
    }
  } finally {
    ai.terminate();
  }
  validation.failed = validation.checked > 0 && validation.changed === 0;
  return validation;
}

async function fetchLossLogs(limit: number): Promise<LossLog[]> {
  if (limit <= 0) {
    return [];
  }
  try {
    const response = await withTimeout(
      fetch("/api/training/loss-logs"),
      TRAINING_IO_TIMEOUT_MS,
      "Timed out loading loss logs."
    );
    if (!response.ok) {
      return [];
    }
    const payload = await response.json() as { logs?: LossLog[] };
    return (payload.logs ?? [])
      .filter((log: LossLog) => Array.isArray(log.decisions) && log.decisions.length > 0)
      .slice(0, limit);
  } catch {
    return [];
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function movesKey(moves: Move[] | undefined): string {
  return (moves ?? []).map((move) =>
    `${move?.from?.timelineId}:${move?.from?.time}:${move?.from?.x}:${move?.from?.y}->${move?.to?.timelineId}:${move?.to?.time}:${move?.to?.x}:${move?.to?.y}`
  ).join("|");
}

async function fetchActiveModel(): Promise<CompactValueModel | null> {
  try {
    const response = await withTimeout(
      fetch("/api/training/model"),
      TRAINING_IO_TIMEOUT_MS,
      "Timed out loading active model."
    );
    if (!response.ok) {
      return null;
    }
    const buffer = await response.arrayBuffer();
    const model = decodeCompactModel(buffer);
    if (model) {
      model.bytes = new Uint8Array(buffer);
    }
    return model;
  } catch {
    return null;
  }
}

async function fetchCpuParameters(): Promise<CpuParameters> {
  const response = await withTimeout(
    fetch("/api/training/cpu-parameters"),
    TRAINING_IO_TIMEOUT_MS,
    "Timed out loading CPU parameters."
  );
  if (!response.ok) {
    throw new Error("No active CPU parameters are available.");
  }
  const value = await response.json() as Record<string, unknown>;
  const parameters: CpuParameters = {};
  for (const [key, raw] of Object.entries(value)) {
    if (typeof raw === "number" && Number.isFinite(raw)) {
      parameters[key] = raw;
    }
  }
  return parameters;
}

async function loadReplayBuffer(): Promise<TrainingSample[]> {
  try {
    const db = await withTimeout(openReplayDb(), TRAINING_IO_TIMEOUT_MS, "Timed out opening replay buffer.");
    return (await withTimeout(idbGet(db, BUFFER_KEY), TRAINING_IO_TIMEOUT_MS, "Timed out reading replay buffer.")) ?? [];
  } catch {
    return [];
  }
}

async function saveReplayBuffer(samples: TrainingSample[]): Promise<void> {
  try {
    const db = await withTimeout(openReplayDb(), TRAINING_IO_TIMEOUT_MS, "Timed out opening replay buffer.");
    await withTimeout(idbPut(db, BUFFER_KEY, samples), TRAINING_IO_TIMEOUT_MS, "Timed out saving replay buffer.");
  } catch {
    // IndexedDB is an optimization; an in-memory run still works without it.
  }
}

function openReplayDb(): Promise<ReplayDb> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("chronofish-training", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("buffers");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function idbGet(db: ReplayDb, key: string): Promise<TrainingSample[] | undefined> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("buffers", "readonly");
    const request = tx.objectStore("buffers").get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function idbPut(db: ReplayDb, key: string, value: TrainingSample[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("buffers", "readwrite");
    tx.objectStore("buffers").put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
function autoLabelWorkers(): number {
  const cores = navigator.hardwareConcurrency ?? 4;
  return Math.max(1, Math.min(cores - 1, 16));
}
