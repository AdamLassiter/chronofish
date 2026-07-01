import { train, predictValues } from "./training-gpu.js";
import { readWasmString, writeWasmString } from "./engine-io.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import { fetchActiveModel, fetchCpuParameters, loadReplayBuffer, saveReplayBuffer } from "./training-worker-storage.js";
import type { ChronofishEngine, Color, GameSnapshot, Move } from "./types.js";
import type { CompactValueModel, EncodedCompactModel, TrainingConfig as GpuTrainingConfig, TrainingMetrics, TrainingSample } from "./training-gpu.js";
import { HIDDEN_LAYERS, MAX_PLAYOUT_PLIES, POLICY_STEPS_PER_SUBMIT, PROJECTION_CHUNK_SIZE, PROJECTION_SEED, PROJECTION_SIZE, TRAINING_IO_TIMEOUT_MS, VALUE_EPOCHS_PER_SUBMIT } from "./training-gpu-constants.js";
import type { AiWorkerResponse, AppliedWorkerTurn, CpuParameters, CpuReferenceScore, CpuTrainingResult, EncodedPosition, LabelJob, LabelWorkerSample, LossLog, LossLogValidation, LossLogValidationExample, MetricsSummary, NormalizedTrainingConfig, ProgressCallback, TrainingLabelKind, TrainingMode, TrainingRunMetrics, TrainingSubject, TrainingWorkerRequest, TrainingWorkerResponse, WorkerRequestPayload, WorkerScope } from "./training-worker-types.js";
let cachedGpuAdapter: GPUAdapter | null = null;
let cachedGpuDevice: GPUDevice | null = null;
let enginePromise: Promise<ChronofishEngine> | null = null;
let trainingEngine: ChronofishEngine | null = null;
const pipelineCache = new Map<string, GPUComputePipeline>();
const trainingModePolicyCache = new Map<string, TrainingModePolicy>();
let trainingLabelPolicyCache: TrainingLabelPolicy | null = null;

const workerSelf = self as unknown as WorkerScope;

workerSelf.addEventListener("message", async (event) => {
  const { id, type = "train", game, config, candidateModel } = event.data;
  try {
    const metrics = createTrainingMetrics();
    const normalizedConfig = await normalizeTrainingConfig(config);
    normalizedConfig.metrics = metrics;
    if (type === "validateLossLogs") {
      const validation = await timed(metrics, "lossLogValidation", () =>
        validateLossLogs(normalizedConfig, (message) => {
          workerSelf.postMessage({ id, ok: true, ...message });
        }, candidateModel)
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
    const collectedSamples = await timed(metrics, "collect", () => collectTrainingSamples(game, normalizedConfig, activeModel, (message) => {
      workerSelf.postMessage({ id, ok: true, ...message });
    }, metrics));
    const samples = await dedupeTrainingSamplesWithEngine(collectedSamples);
    metrics.sampleCounts = await labelSourceCountsWithEngine(samples);
    buffer = await appendReplaySamplesWithEngine(buffer, samples, normalizedConfig.maxBuffer);
    await timed(metrics, "saveReplay", () => saveReplayBuffer(buffer));
    const labelCounts = await labelSourceCountsWithEngine(buffer);
    normalizedConfig.labelCounts = labelCounts;
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
      initialValidationLoss: model.initialValidationLoss,
      validationLoss: model.validationLoss,
      bestValidationLoss: model.bestValidationLoss,
      initialPolicyValidationLoss: model.initialPolicyValidationLoss,
      policyValidationLoss: model.policyValidationLoss,
      bestPolicyValidationLoss: model.bestPolicyValidationLoss,
      valueCheckpointImproved: model.valueCheckpointImproved,
      policyCheckpointImproved: model.policyCheckpointImproved,
      modelChanged: model.modelChanged,
      earlyStopReason: model.earlyStopReason,
      labelCounts: model.labelCounts,
      replaySize: model.replayBufferSize,
      trainingSampleCount: model.trainingSampleCount,
      policyTrainingSampleCount: model.policyTrainingSampleCount,
      nonZeroWeights: model.nonZeroWeights,
      metrics: model.metrics
    });
  } catch (error) {
    workerSelf.postMessage({ id, ok: false, error: errorMessage(error) });
  }
});

async function normalizeTrainingConfig(config: Partial<NormalizedTrainingConfig> = {}): Promise<NormalizedTrainingConfig> {
  return {
    ...config,
    ...await normalizeTrainingConfigWithEngine(config),
    runSeed: randomRunSeed()
  };
}

type EngineNormalizedTrainingConfig = Omit<NormalizedTrainingConfig, "runSeed" | "metrics">;

async function normalizeTrainingConfigWithEngine(
  config: Partial<NormalizedTrainingConfig>
): Promise<EngineNormalizedTrainingConfig> {
  const engine = await engineInstance();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(config));
  try {
    const output = engine.chronofish_normalize_training_config_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as EngineNormalizedTrainingConfig;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function createTrainingMetrics(): TrainingRunMetrics {
  return {
    startedAt: performance.now(),
    phases: Object.create(null)
  };
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

async function dedupeTrainingSamplesWithEngine(samples: TrainingSample[]): Promise<TrainingSample[]> {
  const engine = await engineInstance();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(samplesForEngine(samples)));
  try {
    const output = engine.chronofish_dedupe_training_samples_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as TrainingSample[];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function appendReplaySamplesWithEngine(
  buffer: TrainingSample[],
  samples: TrainingSample[],
  maxBuffer: number
): Promise<TrainingSample[]> {
  const engine = await engineInstance();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    buffer: samplesForEngine(buffer),
    samples: samplesForEngine(samples)
  }));
  try {
    const output = engine.chronofish_append_replay_samples_json(ptr, len, maxBuffer);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as TrainingSample[];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function samplesForEngine(samples: TrainingSample[]): TrainingSample[] {
  return samples.map((sample) => ({
    ...sample,
    features: Array.from(sample.features ?? [])
  }));
}

async function labelSourceCountsWithEngine(samples: TrainingSample[]): Promise<Record<string, number>> {
  const engine = await engineInstance();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(samplesForEngine(samples)));
  try {
    const output = engine.chronofish_label_source_counts_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as Record<string, number>;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineInstance(): Promise<ChronofishEngine> {
  enginePromise ??= instantiateChronofishWasm("./chronofish_engine.wasm")
    .then((instance) => {
      trainingEngine = instance.exports as unknown as ChronofishEngine;
      return trainingEngine;
    });
  return enginePromise;
}

function loadedEngine(): ChronofishEngine {
  if (!trainingEngine) {
    throw new Error("Training engine is not initialized.");
  }
  return trainingEngine;
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
  const vsGpu = trainingModeEnabled(config, "vsGpu");
  const vsCpu = trainingModeEnabled(config, "vsCpu");
  const self = trainingModeEnabled(config, "self");
  const distill = config.trainingSubject === "gpu" && trainingModeEnabled(config, "distill");
  const curriculum = config.trainingSubject === "gpu" && trainingModeEnabled(config, "curriculum");
  const tactical = config.trainingSubject === "gpu" && trainingModeEnabled(config, "tactical");
  if (config.trainingSubject === "cpu") {
    collectors.push(() => timed(metrics, "cpuLabels", () => collectCpuSearchSamples(game, config, progress)));
  } else {
    if (curriculum) {
      collectors.push(() => timed(metrics, "curriculumLabels", () => collectCurriculumSamples(game, config, progress)));
    }
    if (vsGpu) {
      collectors.push(() => collectSearchSamples(game, config, progress));
    }
    if (vsCpu) {
      collectors.push(() => timed(metrics, "cpuLabels", () => collectCpuSearchSamples(game, config, progress)));
    }
    if (self) {
      collectors.push(() => timed(metrics, "outcomeLabels", () => collectOutcomeSamples(game, config, progress)));
    }
    if (distill) {
      collectors.push(() => timed(metrics, "distillLabels", () => collectDistilledSamples(game, config, activeModel, progress)));
    }
    if (tactical) {
      collectors.push(() => timed(metrics, "tacticalLabels", () => collectTacticalSamples(game, config, progress)));
    }
  }
  if (config.trainingSubject === "gpu" && vsGpu && vsCpu) {
    collectors.push(() => timed(metrics, "duelLabels", () => collectCpuGpuDuelSamples(game, config, progress)));
  }

  const collected = await Promise.allSettled(collectors.map((collector) => collector()));
  const results = collected
    .filter((result): result is PromiseFulfilledResult<TrainingSample[]> => result.status === "fulfilled")
    .flatMap((result) => result.value);
  if (results.length > 0) {
    return results;
  }
  if (activeModel?.outputWeights?.length && !distill && config.trainingSubject === "gpu") {
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
  const target = cpuTrainingPositionTarget(config);
  const sampleGames = await timed(config.metrics, "cpuPositions", () =>
    collectCpuTrainingGames(game, baseline, config, target, progress)
  );
  if (!sampleGames.length) {
    throw new Error("CPU training could not sample positions.");
  }
  const candidateCount = cpuTrainingCandidateCount(config);
  const deadlineAt = cpuTrainingDeadlineAt(config);
  const screeningGames = sampleGames.slice(0, Math.max(1, Math.min(sampleGames.length, config.cpuScreeningOpponentVariants)));
  const screeningConfig = cpuScreeningTrainingConfig(config);
  const screeningReferences = await timed(config.metrics, "cpuScreeningReferences", () =>
    precomputeCpuReferenceScores(baseline, screeningGames, screeningConfig, deadlineAt, progress)
  );
  const screeningFitness = new Map<string, number>();
  const finalistFitness = new Map<string, number>();
  let population = breedCpuPopulation(baseline, [], candidateCount, config.runSeed, 0, 0);
  let bestCandidate: { parameters: CpuParameters; score: number } | null = null;
  let baselineScore = Number.NEGATIVE_INFINITY;
  let generationsWithoutCandidate = 0;
  let generation = 0;
  while (
    performance.now() < deadlineAt &&
    generationsWithoutCandidate < config.cpuMaxGenerationsWithoutCandidate
  ) {
    const screened = await timed(config.metrics, "cpuScreening", () =>
      scoreCpuCandidates(
        population,
        screeningGames,
        screeningConfig,
        deadlineAt,
        `cpu-screen-${generation + 1}`,
        screeningReferences,
        screeningFitness,
        false
      )
    );
    const finalistTarget = cpuTrainingFinalistTarget(config, population.length, screened.length);
    const finalistCandidates = uniqueCpuParameters([
      baseline,
      ...screened.slice(0, finalistTarget).map((entry) => entry.parameters)
    ]);
    const finalists = await timed(config.metrics, "cpuFinalists", () =>
      scoreCpuCandidates(
        finalistCandidates,
        sampleGames,
        config,
        deadlineAt,
        `cpu-train-${generation + 1}`,
        [],
        finalistFitness,
        true
      )
    );
    const baselineResult = finalists.find((entry) => cpuParametersKey(entry.parameters) === cpuParametersKey(baseline));
    baselineScore = baselineResult?.score ?? baselineScore;
    const winner = finalists.find((entry) => cpuParametersKey(entry.parameters) !== cpuParametersKey(baseline));
    const improved = Boolean(
      winner &&
      winner.score > baselineScore &&
      (!bestCandidate || winner.score > bestCandidate.score)
    );
    if (improved && winner) {
      bestCandidate = winner;
      generationsWithoutCandidate = 0;
    } else {
      generationsWithoutCandidate += 1;
    }
    progress({
      labelKind: "cpu-generation",
      generation: generation + 1,
      generationsWithoutCandidate,
      candidateScore: winner?.score ?? null,
      baselineScore,
      screeningCacheSize: screeningFitness.size,
      finalistCacheSize: finalistFitness.size
    });
    generation += 1;
    const elites = finalists
      .filter((entry) => cpuParametersKey(entry.parameters) !== cpuParametersKey(baseline))
      .slice(0, cpuTrainingEliteCount(config))
      .map((entry) => entry.parameters);
    population = breedCpuPopulation(
      baseline,
      bestCandidate ? [bestCandidate.parameters, ...elites] : elites,
      candidateCount,
      config.runSeed,
      generation,
      generationsWithoutCandidate
    );
  }
  if (bestCandidate) {
    return {
      parametersJson: JSON.stringify(bestCandidate.parameters, null, 2),
      score: bestCandidate.score
    };
  }
  return {
    parametersJson: JSON.stringify(baseline, null, 2),
    score: baselineScore
  };

  async function scoreCpuCandidates(
    stageCandidates: CpuParameters[],
    stageGames: GameSnapshot[],
    stageConfig: NormalizedTrainingConfig,
    stageDeadlineAt: number,
    labelKind: string,
    references: CpuReferenceScore[],
    fitnessCache: Map<string, number>,
    pairedMatches: boolean
  ): Promise<Array<{ parameters: CpuParameters; score: number }>> {
    const uniqueCandidates = uniqueCpuParameters(stageCandidates);
    const workerCount = cpuCandidateWorkerCount(uniqueCandidates.length, stageConfig.cpuWorkers, stageConfig.cpuPairBatch);
    let nextCandidate = 0;
    let collected = 0;
    const scored: Array<{ parameters: CpuParameters; score: number }> = uniqueCandidates
      .filter((candidate) => fitnessCache.has(cpuParametersKey(candidate)))
      .map((parameters) => ({ parameters, score: fitnessCache.get(cpuParametersKey(parameters))! }));
    const uncachedCandidates = uniqueCandidates.filter((candidate) => !fitnessCache.has(cpuParametersKey(candidate)));
    const cacheHits = scored.length;
    progress({ sampleCount: uniqueCandidates.length, labelWorkers: workerCount, labelKind });
    await Promise.all(Array.from({ length: workerCount }, () => runCandidateWorker()));
    progress({ collected: scored.length, sampleCount: uniqueCandidates.length, labelWorkers: workerCount, labelKind, cacheHits });
    return scored.sort((left, right) => right.score - left.score);

    async function runCandidateWorker(): Promise<void> {
      const candidateWorker = new Worker("./cpu-ai-worker.js", { type: "module" });
      const baselineWorker = pairedMatches ? new Worker("./cpu-ai-worker.js", { type: "module" }) : null;
      try {
        while (nextCandidate < uncachedCandidates.length) {
          if (performance.now() >= stageDeadlineAt) {
            break;
          }
          const index = nextCandidate;
          nextCandidate += 1;
          const candidate = uncachedCandidates[index];
          if (!candidate) {
            continue;
          }
          const score = pairedMatches && baselineWorker
            ? await scoreCpuCandidateByPairedMatches(
              candidate,
              baseline,
              stageGames,
              stageConfig,
              candidateWorker,
              baselineWorker,
              stageDeadlineAt
            )
            : await scoreCpuCandidate(candidate, stageGames, references, stageConfig, candidateWorker, stageDeadlineAt);
          if (score === null) {
            continue;
          }
          fitnessCache.set(cpuParametersKey(candidate), score);
          scored.push({ parameters: candidate, score });
          collected += 1;
          progress({ collected: cacheHits + collected, sampleCount: uniqueCandidates.length, labelWorkers: workerCount, labelKind, cacheHits });
        }
      } finally {
        candidateWorker.terminate();
        baselineWorker?.terminate();
      }
    }
  }
}

async function collectCpuTrainingGames(
  game: GameSnapshot,
  baseline: CpuParameters,
  config: NormalizedTrainingConfig,
  target: number,
  progress: ProgressCallback
): Promise<GameSnapshot[]> {
  const games: Array<GameSnapshot | null> = new Array(target).fill(null);
  const workerCount = cpuTrainingPositionWorkerCount(target, config.cpuWorkers);
  const parametersJson = JSON.stringify(baseline);
  let nextIndex = 0;
  let collected = 0;
  progress({ collected, sampleCount: target, labelWorkers: workerCount, labelKind: "cpu-positions" });
  await Promise.all(Array.from({ length: workerCount }, (_, workerIndex) => runWorker(workerIndex)));
  return games.filter((entry): entry is GameSnapshot => Boolean(entry));

  async function runWorker(workerIndex: number): Promise<void> {
    const cpu = new Worker("./cpu-ai-worker.js", { type: "module" });
    try {
      while (nextIndex < target) {
        const index = nextIndex;
        nextIndex += 1;
        let current = cloneGame(game);
        for (let ply = 0; ply < samplePlies(index, false); ply += 1) {
          const searchConfig = cpuTrainingPositionSearchConfig(config);
          const response = await requestWorker(cpu, {
            type: "search",
            game: current,
            depth: searchConfig.depth,
            nodes: searchConfig.nodes,
            timeMs: config.cpuTrainingTimeMs,
            parametersJson
          }, workerRequestTimeout({ nodes: config.cpuNodes, timeMs: config.cpuTrainingTimeMs }));
          const moves = response.result?.moves ?? [];
          if (!moves.length) {
            break;
          }
          const applied = await applyCpuWorkerTurn(cpu, current, moves, config);
          if (!applied) {
            break;
          }
          current = applied.game;
          if (applied.status.terminal) {
            break;
          }
        }
        games[index] = current;
        collected += 1;
        progress({ collected, sampleCount: target, labelWorkers: workerCount, labelKind: "cpu-positions", workerIndex });
      }
    } finally {
      cpu.terminate();
    }
  }
}

function cpuScreeningTrainingConfig(config: NormalizedTrainingConfig): NormalizedTrainingConfig {
  const screening = cpuScreeningTrainingConfigWithEngine(config);
  return {
    ...config,
    cpuDepth: screening.cpuDepth,
    depth: screening.depth,
    cpuNodes: screening.cpuNodes,
    nodes: screening.nodes,
    cpuTrainingTimeMs: screening.cpuTrainingTimeMs
  };
}

interface CpuTrainingPositionSearchConfig {
  depth: number;
  nodes: number;
}

interface CpuScreeningTrainingConfig {
  cpuDepth: number;
  depth: number;
  cpuNodes: number;
  nodes: number;
  cpuTrainingTimeMs: number;
}

function cpuTrainingPositionSearchConfig(config: NormalizedTrainingConfig): CpuTrainingPositionSearchConfig {
  const engine = loadedEngine();
  const output = engine.chronofish_cpu_training_position_search_config_json(
    config.cpuDepth,
    config.cpuNodes
  );
  return JSON.parse(readWasmString(engine, output)) as CpuTrainingPositionSearchConfig;
}

function cpuScreeningTrainingConfigWithEngine(config: NormalizedTrainingConfig): CpuScreeningTrainingConfig {
  const engine = loadedEngine();
  const output = engine.chronofish_cpu_screening_training_config_json(
    config.cpuDepth,
    config.depth,
    config.cpuNodes,
    config.nodes,
    config.cpuTrainingTimeMs
  );
  return JSON.parse(readWasmString(engine, output)) as CpuScreeningTrainingConfig;
}

function cpuTrainingPositionWorkerCount(target: number, cpuWorkers: number): number {
  return loadedEngine().chronofish_cpu_training_position_worker_count(target, cpuWorkers);
}

function cpuReferenceWorkerCount(gameCount: number, requestedWorkers: number, pairBatch: number): number {
  return loadedEngine().chronofish_cpu_reference_worker_count(gameCount, requestedWorkers, pairBatch);
}

function cpuCandidateWorkerCount(candidateCount: number, cpuWorkers: number, pairBatch: number): number {
  return loadedEngine().chronofish_cpu_candidate_worker_count(candidateCount, cpuWorkers, pairBatch);
}

function cpuLabelWorkerCount(positionCount: number, cpuWorkers: number): number {
  return loadedEngine().chronofish_cpu_label_worker_count(positionCount, cpuWorkers);
}

function cpuSearchLabelWeight(config: NormalizedTrainingConfig): number {
  return loadedEngine().chronofish_cpu_search_label_weight(trainingModeCount(config));
}

async function precomputeCpuReferenceScores(
  baseline: CpuParameters,
  games: GameSnapshot[],
  config: NormalizedTrainingConfig,
  deadlineAt: number,
  progress: ProgressCallback
): Promise<CpuReferenceScore[]> {
  const references: CpuReferenceScore[] = Array.from({ length: games.length }, () => ({}));
  const baselineJson = JSON.stringify(baseline);
  const workerCount = cpuReferenceWorkerCount(games.length, config.cpuWorkers, config.cpuPairBatch);
  let nextGame = 0;
  let collected = 0;
  progress({ collected: 0, sampleCount: games.length, labelWorkers: workerCount, labelKind: "cpu-reference" });
  await Promise.all(Array.from({ length: workerCount }, () => runReferenceWorker()));
  return references;

  async function runReferenceWorker(): Promise<void> {
    const baselineWorker = new Worker("./cpu-ai-worker.js", { type: "module" });
    const gpuWorker = new Worker("./ai-worker.js", { type: "module" });
    try {
      while (nextGame < games.length && performance.now() < deadlineAt) {
        const index = nextGame;
        nextGame += 1;
        const game = games[index];
        if (!game) {
          continue;
        }
        const reference: CpuReferenceScore = {};
        if (cpuBaselineModeEnabled(config)) {
          const baselineResult = await requestWorker(baselineWorker, {
            type: "search",
            game,
            depth: config.cpuDepth,
            nodes: config.cpuNodes,
            timeMs: config.cpuTrainingTimeMs,
            parametersJson: baselineJson
          }, workerRequestTimeout({ nodes: config.cpuNodes, timeMs: config.cpuTrainingTimeMs }));
          reference.baselineScore = baselineResult.result?.score ?? 0;
          if (baselineResult.result?.moves) {
            reference.baselineMoves = baselineResult.result.moves;
          }
        }
        if (trainingModeEnabled(config, "vsGpu")) {
          const gpuResult = await requestWorker(gpuWorker, {
            type: "search",
            game,
            depth: config.depth,
            nodes: config.nodes,
            timeMs: workerSearchTimeMs(config),
            gpuMode: "full"
          }, workerRequestTimeout(config));
          reference.gpuScore = gpuResult.result?.score ?? 0;
          if (gpuResult.result?.moves) {
            reference.gpuMoves = gpuResult.result.moves;
          }
        }
        references[index] = reference;
        collected += 1;
        progress({ collected, sampleCount: games.length, labelWorkers: workerCount, labelKind: "cpu-reference" });
      }
    } finally {
      baselineWorker.terminate();
      gpuWorker.terminate();
    }
  }
}

async function scoreCpuCandidate(
  candidate: CpuParameters,
  games: GameSnapshot[],
  references: CpuReferenceScore[],
  config: NormalizedTrainingConfig,
  candidateWorker: Worker,
  deadlineAt: number
): Promise<number> {
  let score = 0;
  let compared = 0;
  let nearDraws = 0;
  const candidateJson = JSON.stringify(candidate);
  const comparisonCount = Math.min(games.length, references.length || games.length);
  for (let index = 0; index < comparisonCount; index += 1) {
    if (performance.now() >= deadlineAt || compared >= config.cpuMaxMatchPlies) {
      break;
    }
    const game = games[index];
    if (!game) {
      continue;
    }
    const candidateResult = await requestWorker(candidateWorker, {
      type: "search",
      game,
      depth: config.cpuDepth,
      nodes: config.cpuNodes,
      timeMs: config.cpuTrainingTimeMs,
      parametersJson: candidateJson
    }, workerRequestTimeout({ nodes: config.cpuNodes, timeMs: config.cpuTrainingTimeMs }));
    const candidateScore = candidateResult.result?.score ?? 0;
    const reference = references[index] ?? {};
    if (cpuBaselineModeEnabled(config) && reference.baselineScore !== undefined) {
      const delta = cpuReferenceScoreDelta(
        candidateScore,
        reference.baselineScore,
        candidateResult.result?.moves,
        reference.baselineMoves,
        config.cpuDrawWindow
      );
      score += delta.score;
      if (delta.nearDraw) {
        nearDraws += 1;
      }
    }
    if (trainingModeEnabled(config, "vsGpu") && reference.gpuScore !== undefined) {
      score += cpuReferenceScoreDelta(
        candidateScore,
        reference.gpuScore,
        candidateResult.result?.moves,
        reference.gpuMoves,
        config.cpuDrawWindow
      ).score;
    }
    compared += 1;
  }
  return cpuReferenceCandidateAverage(score, compared, nearDraws, config.cpuDrawRateLimit);
}

async function scoreCpuCandidateByPairedMatches(
  candidate: CpuParameters,
  baseline: CpuParameters,
  games: GameSnapshot[],
  config: NormalizedTrainingConfig,
  candidateWorker: Worker,
  baselineWorker: Worker,
  deadlineAt: number
): Promise<number | null> {
  const candidateJson = JSON.stringify(candidate);
  const baselineJson = JSON.stringify(baseline);
  let score = 0;
  let completed = 0;
  const totalMatches = games.length * 2;
  for (let gameIndex = 0; gameIndex < games.length; gameIndex += 1) {
    const game = games[gameIndex]!;
    for (const candidateColor of [game.turn, oppositeColor(game.turn)]) {
      if (performance.now() >= deadlineAt) {
        return null;
      }
      const remainingMatches = Math.max(1, totalMatches - completed);
      const matchDeadlineAt = Math.min(
        deadlineAt,
        performance.now() + Math.max(1, (deadlineAt - performance.now()) / remainingMatches)
      );
      const result = await playCpuTrainingMatch(
        game,
        candidateColor,
        candidateJson,
        baselineJson,
        config,
        candidateWorker,
        baselineWorker,
        matchDeadlineAt
      );
      if (result === null) {
        return null;
      }
      score += result;
      completed += 1;
    }
  }
  return completed > 0 ? score / completed : null;
}

async function playCpuTrainingMatch(
  start: GameSnapshot,
  candidateColor: Color,
  candidateJson: string,
  baselineJson: string,
  config: NormalizedTrainingConfig,
  candidateWorker: Worker,
  baselineWorker: Worker,
  matchDeadlineAt: number
): Promise<number | null> {
  let current = cloneGame(start);
  for (let ply = 0; ply < config.cpuMaxMatchPlies; ply += 1) {
    if (performance.now() >= matchDeadlineAt) {
      return null;
    }
    const turnTimeMs = cpuMatchTurnTimeMs(config, matchDeadlineAt, config.cpuMaxMatchPlies - ply + 1);
    const candidateTurn = current.turn === candidateColor;
    const response = await requestWorker(candidateTurn ? candidateWorker : baselineWorker, {
      type: "search",
      game: current,
      depth: config.cpuDepth,
      nodes: config.cpuNodes,
      timeMs: turnTimeMs,
      parametersJson: candidateTurn ? candidateJson : baselineJson
    }, workerRequestTimeout({
      nodes: config.cpuNodes,
      timeMs: turnTimeMs
    }));
    const moves = response.result?.moves ?? [];
    if (!moves.length) {
      return cpuTrainingNoMoveScore(candidateTurn);
    }
    const applied = await applyCpuWorkerTurn(baselineWorker, current, moves, config);
    if (!applied) {
      return null;
    }
    current = applied.game;
    if (applied.winner) {
      return cpuTrainingWinnerScore(applied.winner, candidateColor);
    }
    if (applied.status.terminal) {
      return cpuTrainingWinnerScore(applied.status.winner ?? null, candidateColor);
    }
  }

  if (performance.now() >= matchDeadlineAt) {
    return null;
  }
  const adjudicationTimeMs = cpuMatchTurnTimeMs(config, matchDeadlineAt, 1);
  const adjudication = await requestWorker(baselineWorker, {
    type: "search",
    game: current,
    depth: config.cpuDepth,
    nodes: config.cpuNodes,
    timeMs: adjudicationTimeMs,
    parametersJson: baselineJson
  }, workerRequestTimeout({
    nodes: config.cpuNodes,
    timeMs: adjudicationTimeMs
  }));
  const baselineScore = adjudication.result?.score ?? 0;
  return cpuTrainingAdjudicationScore(current.turn, candidateColor, baselineScore);
}

function cpuMatchTurnTimeMs(config: NormalizedTrainingConfig, deadlineAt: number, remainingSearches: number): number {
  return loadedEngine().chronofish_cpu_match_turn_time_ms(
    config.cpuTrainingTimeMs,
    performance.now(),
    deadlineAt,
    remainingSearches
  );
}

function cpuTrainingPositionTarget(config: NormalizedTrainingConfig): number {
  return loadedEngine().chronofish_cpu_training_position_target(
    config.samples,
    trainingModeCount(config),
    config.cpuOpponentVariants,
    config.cpuScreeningOpponentVariants,
    config.cpuRoundsPerVariant,
    config.cpuLeagueContenders,
    config.cpuLeagueHallOfFameEntries,
    config.cpuHallOfFameEntries,
    config.cpuMinPairs,
    config.cpuMaxPairs,
    config.cpuMaxMatchPlies
  );
}

function cpuTrainingCandidateCount(config: NormalizedTrainingConfig): number {
  return loadedEngine().chronofish_cpu_training_candidate_count(config.cpuCandidates);
}

function cpuTrainingFinalistTarget(config: NormalizedTrainingConfig, populationLength: number, screenedLength: number): number {
  return loadedEngine().chronofish_cpu_training_finalist_target(
    populationLength,
    config.cpuFinalists,
    config.cpuPairBatch,
    screenedLength
  );
}

function cpuTrainingEliteCount(config: NormalizedTrainingConfig): number {
  return loadedEngine().chronofish_cpu_training_elite_count(config.cpuFinalists);
}

function cpuTrainingDeadlineAt(config: NormalizedTrainingConfig): number {
  const budgetMs = loadedEngine().chronofish_cpu_training_budget_ms(
    config.cpuTrainSeconds,
    config.cpuTrainingTimeMs,
    config.cpuMaxMatchPlies,
    config.cpuMaxMatchTimeMs
  );
  return performance.now() + budgetMs;
}

interface CpuReferenceScoreDelta {
  score: number;
  nearDraw: boolean;
}

interface CpuTrainingMove {
  fromTimelineId: number;
  fromTime: number;
  fromX: number;
  fromY: number;
  toTimelineId: number;
  toTime: number;
  toX: number;
  toY: number;
}

function cpuReferenceScoreDelta(
  candidateScore: number,
  referenceScore: number,
  candidateMoves: Move[] | undefined,
  referenceMoves: Move[] | undefined,
  drawWindow: number
): CpuReferenceScoreDelta {
  const engine = loadedEngine();
  const request = {
    candidateScore,
    referenceScore,
    candidateMoves: cpuTrainingMoves(candidateMoves ?? []),
    referenceMoves: cpuTrainingMoves(referenceMoves ?? []),
    drawWindow
  };
  const input = JSON.stringify(request);
  const [ptr, len] = writeWasmString(engine, input);
  try {
    const output = engine.chronofish_cpu_reference_score_delta_json(ptr, len);
    return JSON.parse(readWasmString(engine, output)) as CpuReferenceScoreDelta;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function cpuReferenceCandidateAverage(score: number, compared: number, nearDraws: number, drawRateLimit: number): number {
  return loadedEngine().chronofish_cpu_reference_candidate_average(score, compared, nearDraws, drawRateLimit);
}

function cpuTrainingNoMoveScore(candidateTurn: boolean): number {
  return loadedEngine().chronofish_cpu_training_no_move_score(candidateTurn ? 1 : 0);
}

function cpuTrainingWinnerScore(winner: Color | null, candidateColor: Color): number {
  const engine = loadedEngine();
  const [ptr, len] = writeWasmString(engine, JSON.stringify({
    winner,
    candidateColor
  }));
  try {
    return engine.chronofish_cpu_training_winner_score_json(ptr, len);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function cpuTrainingAdjudicationScore(currentTurn: Color, candidateColor: Color, baselineScore: number): number {
  const engine = loadedEngine();
  const [ptr, len] = writeWasmString(engine, JSON.stringify({
    currentTurn,
    candidateColor,
    baselineScore
  }));
  try {
    return engine.chronofish_cpu_training_adjudication_score_json(ptr, len);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function cpuTrainingMoves(moves: Move[]): CpuTrainingMove[] {
  return moves.map((move) => ({
    fromTimelineId: move.from.timelineId,
    fromTime: move.from.time,
    fromX: move.from.x,
    fromY: move.from.y,
    toTimelineId: move.to.timelineId,
    toTime: move.to.time,
    toX: move.to.x,
    toY: move.to.y
  }));
}

async function collectSearchSamples(game: GameSnapshot, config: NormalizedTrainingConfig, progress: ProgressCallback): Promise<TrainingSample[]> {
  const target = modeLabelTarget(config, 16);
  const positions = await timed(config.metrics, "searchPositions", () => collectGpuPositions(game, config, target, progress, "search", config.searchWorkers));
  if (config.metrics) {
    config.metrics.searchPositionCount = positions.length;
  }
  const workerCount = gpuTrainingWorkerCount(positions.length, config.searchWorkers);
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
  const target = modeLabelTarget(config, trainingModeCount(config) > 1 ? 16 : 1);
  const positions = await timed(config.metrics, "cpuPositions", () => collectGpuPositions(game, config, target, progress, "cpu", config.searchWorkers));
  const workerCount = cpuLabelWorkerCount(positions.length, config.cpuWorkers);
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
        labelWeight: cpuSearchLabelWeight(config),
        pseudo: false
      };
    } catch {
      return null;
    }
  }
}

async function collectCurriculumSamples(game: GameSnapshot, config: NormalizedTrainingConfig, progress: ProgressCallback): Promise<TrainingSample[]> {
  const target = modeLabelTarget(config, 8);
  const positions = await timed(config.metrics, "curriculumPositions", () =>
    collectGpuPositions(game, config, target, progress, "curriculum", config.searchWorkers, generateCurriculumPositionGame)
  );
  return collectGpuSearchLabels(positions, config, progress, "curriculum", 1.05, 0xc374_0001);
}

async function collectTacticalSamples(game: GameSnapshot, config: NormalizedTrainingConfig, progress: ProgressCallback): Promise<TrainingSample[]> {
  const target = modeLabelTarget(config, 8);
  const positions = await timed(config.metrics, "tacticalPositions", () =>
    collectGpuPositions(game, config, target, progress, "tactical", config.searchWorkers, generateTacticalPositionGame)
  );
  return collectGpuSearchLabels(
    positions,
    config,
    progress,
    "tactical",
    1.6,
    0x7ac7_0001,
    (position) => 1 + tacticalPositionPriority(position.game) * 0.2
  );
}

async function collectGpuSearchLabels(
  positions: EncodedPosition[],
  config: NormalizedTrainingConfig,
  progress: ProgressCallback,
  labelKind: TrainingLabelKind,
  baseLabelWeight: number,
  seedSalt: number,
  labelWeightMultiplier: (position: EncodedPosition) => number = () => 1
): Promise<TrainingSample[]> {
  const workerCount = gpuTrainingWorkerCount(positions.length, config.searchWorkers);
  const samples: Array<TrainingSample | null> = new Array(positions.length).fill(null);
  let nextPosition = 0;
  let collected = 0;
  progress({ sampleCount: positions.length, labelWorkers: workerCount, labelKind, labelPhase: "labels" });
  await Promise.all(Array.from({ length: workerCount }, (_, workerIndex) => runWorker(workerIndex)));
  return samples.filter((sample): sample is TrainingSample => Boolean(sample));

  async function runWorker(workerIndex: number): Promise<void> {
    const ai = new Worker("./ai-worker.js", { type: "module" });
    try {
      while (nextPosition < positions.length) {
        const index = nextPosition;
        nextPosition += 1;
        const position = positions[index];
        if (!position) {
          continue;
        }
        samples[index] = await labelPosition(ai, position, index, workerIndex);
        collected += samples[index] ? 1 : 0;
        progress({ collected, sampleCount: positions.length, labelWorkers: workerCount, labelKind, labelPhase: "labels" });
      }
    } finally {
      ai.terminate();
    }
  }

  async function labelPosition(ai: Worker, position: EncodedPosition, index: number, workerIndex: number): Promise<TrainingSample | null> {
    try {
      const response = await requestWorker(ai, {
        type: "search",
        game: position.game,
        depth: config.depth,
        nodes: config.nodes,
        timeMs: workerSearchTimeMs(config),
        gpuMode: "full",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed(String(labelKind), index, searchSeed(position.sample, config.runSeed ^ workerIndex ^ seedSalt))
      }, workerRequestTimeout(config));
      const result = response.result;
      if (!result?.moves?.length) {
        return null;
      }
      return {
        ...position.sample,
        label: normalizeSearchScore(result.score ?? 0),
        policy: policyBucket(result.moves[0]),
        labelKind,
        labelWeight: baseLabelWeight * labelWeightMultiplier(position),
        pseudo: false
      };
    } catch {
      return null;
    }
  }
}

async function collectCpuGpuDuelSamples(game: GameSnapshot, config: NormalizedTrainingConfig, progress: ProgressCallback): Promise<TrainingSample[]> {
  const target = modeLabelTarget(config, 8);
  if (target <= 0) {
    return [];
  }
  const workerCount = gpuTrainingWorkerCount(target, Math.min(config.searchWorkers, config.selfPlayWorkers));
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
  const labelPolicy = trainingLabelPolicy();
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
      const moves = result?.moves ?? [];
      if (!moves.length) {
        break;
      }
      samples.push({
        ...encoded,
        label: normalizeSearchScore(result?.score ?? 0),
        policy: policyBucket(moves[0]),
        labelKind: "duel",
        labelWeight: labelPolicy.duelLabelWeight,
        outcomeTurn: beforeTurn,
        ply: ply + workerIndex * MAX_PLAYOUT_PLIES
      });
      progress(1);
      const applied = await applyWorkerTurn(gpu, current, moves, config, beforeTurn);
      if (!applied) {
        break;
      }
      current = applied.game;
      if (applied.winner) {
        return relabelOutcomeSamplesWithEngine(samples, {
          kind: "outcome",
          winner: applied.winner,
          labelKind: "duel",
          labelWeight: labelPolicy.duelLabelWeight
        });
      }
      if (applied.status.terminal && applied.status.winner) {
        return relabelOutcomeSamplesWithEngine(samples, {
          kind: "outcome",
          winner: applied.status.winner,
          labelKind: "duel",
          labelWeight: labelPolicy.duelLabelWeight
        });
      }
      if (applied.status.terminal) {
        return relabelOutcomeSamplesWithEngine(samples, {
          kind: "draw",
          labelKind: "duel",
          labelWeight: labelPolicy.duelDrawLabelWeight
        });
      }
    }
    return relabelOutcomeSamplesWithEngine(samples, {
      kind: "partial",
      labelKind: "duel-search",
      labelWeight: 1.0
    });
  } catch {
    return relabelOutcomeSamplesWithEngine(samples, {
      kind: "partial",
      labelKind: "duel-search",
      labelWeight: 1.0
    });
  } finally {
    gpu.terminate();
    cpu.terminate();
    encoder.terminate();
  }
}

async function collectOutcomeSamples(game: GameSnapshot, config: NormalizedTrainingConfig, progress: ProgressCallback): Promise<TrainingSample[]> {
  const target = modeLabelTarget(config, 16);
  if (target <= 0) {
    return [];
  }
  const workerCount = gpuTrainingWorkerCount(target, config.selfPlayWorkers);
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
  const labelPolicy = trainingLabelPolicy();
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
      const moves = result?.moves ?? [];
      if (!moves.length) {
        break;
      }
      samples.push({
        ...encoded,
        label: normalizeSearchScore(result?.score ?? 0),
        policy: policyBucket(moves[0]),
        labelKind: "outcome",
        labelWeight: labelPolicy.outcomeLabelWeight,
        outcomeTurn: beforeTurn,
        ply: ply + workerIndex * MAX_PLAYOUT_PLIES
      });
      progress(1);
      const applied = await applyWorkerTurn(ai, current, moves, config, beforeTurn);
      if (!applied) {
        break;
      }
      current = applied.game;
      if (applied.winner) {
        return relabelOutcomeSamplesWithEngine(samples, { kind: "outcome", winner: applied.winner });
      }
      if (applied.status.terminal && applied.status.winner) {
        return relabelOutcomeSamplesWithEngine(samples, { kind: "outcome", winner: applied.status.winner });
      }
      if (applied.status.terminal) {
        return relabelOutcomeSamplesWithEngine(samples, {
          kind: "draw",
          labelKind: "outcome",
          labelWeight: 1.0
        });
      }
    }
    return relabelOutcomeSamplesWithEngine(samples, { kind: "partial" });
  } catch {
    return relabelOutcomeSamplesWithEngine(samples, { kind: "partial" });
  } finally {
    ai.terminate();
    encoder.terminate();
  }
}

function splitWork(total: number, workers: number): number[] {
  const engine = loadedEngine();
  const output = engine.chronofish_training_split_work_json(total, workers);
  return JSON.parse(readWasmString(engine, output)) as number[];
}

function gpuTrainingWorkerCount(total: number, requestedWorkers: number): number {
  return loadedEngine().chronofish_gpu_training_worker_count(total, requestedWorkers);
}

async function warmupSelfPlayPosition(ai: Worker, game: GameSnapshot, config: NormalizedTrainingConfig, workerIndex: number): Promise<GameSnapshot> {
  let current = cloneGame(game);
  const warmupPlies = gpuWarmupPlies(workerIndex);
  for (let ply = 0; ply < warmupPlies; ply += 1) {
    try {
      const warmupConfig = gpuWarmupSearchConfig(config);
      const response = await requestWorker(ai, {
        type: "search",
        game: current,
        depth: warmupConfig.depth,
        nodes: warmupConfig.nodes,
        timeMs: warmupConfig.timeMs,
        gpuMode: "full",
        partitionIndex: workerIndex,
        partitionCount: config.selfPlayWorkers ?? 1,
        temperature: warmupConfig.explorationTemperature,
        randomSeed: sampleSeed("warmup", ply, config.runSeed ^ workerIndex ^ 0x0aa5_0001)
      }, workerRequestTimeout({ ...config, nodes: 1024 }));
      const moves = response.result?.moves ?? [];
      if (!moves.length) {
        break;
      }
      const applied = await applyWorkerTurn(ai, current, moves, config, current.turn);
      if (!applied) {
        break;
      }
      current = applied.game;
      if (applied.status.terminal) {
        break;
      }
    } catch {
      break;
    }
  }
  return current;
}

interface TrainingWorkerSearchConfig {
  depth: number;
  nodes: number;
  timeMs: number;
  explorationTemperature: number;
}

interface TrainingSearchConfig {
  depth: number;
  nodes: number;
  explorationTemperature: number;
}

function gpuWarmupPlies(workerIndex: number): number {
  return loadedEngine().chronofish_gpu_warmup_plies(workerIndex);
}

function gpuWarmupSearchConfig(config: NormalizedTrainingConfig): TrainingWorkerSearchConfig {
  const engine = loadedEngine();
  const output = engine.chronofish_gpu_warmup_search_config_json(
    config.depth,
    config.nodes,
    workerSearchTimeMs(config),
    config.explorationTemperature
  );
  return JSON.parse(readWasmString(engine, output)) as TrainingWorkerSearchConfig;
}

function gpuPositionGenerationSearchConfig(config: NormalizedTrainingConfig): TrainingWorkerSearchConfig {
  const engine = loadedEngine();
  const output = engine.chronofish_gpu_position_generation_search_config_json(
    config.depth,
    config.nodes,
    config.explorationTemperature
  );
  return JSON.parse(readWasmString(engine, output)) as TrainingWorkerSearchConfig;
}

function curriculumSearchConfig(config: NormalizedTrainingConfig, index: number): NormalizedTrainingConfig {
  const engine = loadedEngine();
  const output = engine.chronofish_curriculum_search_config_json(
    config.depth,
    config.nodes,
    config.explorationTemperature,
    index
  );
  return {
    ...config,
    ...(JSON.parse(readWasmString(engine, output)) as TrainingSearchConfig)
  };
}

function tacticalSearchConfig(config: NormalizedTrainingConfig, attempt: number): NormalizedTrainingConfig {
  const engine = loadedEngine();
  const output = engine.chronofish_tactical_search_config_json(
    config.depth,
    config.nodes,
    config.explorationTemperature,
    attempt
  );
  return {
    ...config,
    ...(JSON.parse(readWasmString(engine, output)) as TrainingSearchConfig)
  };
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
  const positions = await collectGpuPositions(
    game,
    config,
    modeLabelTarget(config, 1),
    progress,
    "distilled",
    config.searchWorkers
  );
  const samples = positions.map((position) => position.sample);
  const labels = await predictValues(samples, activeModel);
  return samples.map((sample, index) => ({
    ...sample,
    label: labels[index] ?? 0,
    policy: null,
    labelKind: "distilled",
    labelWeight: trainingLabelPolicy().distilledLabelWeight,
    pseudo: true
  }));
}

async function collectGpuPositions(
  game: GameSnapshot,
  config: NormalizedTrainingConfig,
  target: number,
  progress: ProgressCallback,
  labelKind: TrainingLabelKind,
  requestedWorkers: number = config.searchWorkers,
  positionGenerator: (
    ai: Worker,
    game: GameSnapshot,
    config: NormalizedTrainingConfig,
    index: number,
    workerIndex: number
  ) => Promise<GameSnapshot> = generatePositionGame
): Promise<EncodedPosition[]> {
  if (target <= 0) {
    return [];
  }
  const workerCount = gpuTrainingWorkerCount(target, requestedWorkers);
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
        const positionGame = await positionGenerator(ai, game, config, index, workerIndex);
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
      const searchConfig = gpuPositionGenerationSearchConfig(config);
      const response = await requestWorker(ai, {
        type: "search",
        game: current,
        depth: searchConfig.depth,
        nodes: searchConfig.nodes,
        timeMs: searchConfig.timeMs,
        gpuMode: "full",
        temperature: searchConfig.explorationTemperature,
        randomSeed: sampleSeed("position", index * MAX_PLAYOUT_PLIES + ply, config.runSeed ^ workerIndex ^ 0x9051_0001)
      }, workerRequestTimeout(searchConfig));
      const moves = response.result?.moves ?? [];
      if (!moves.length) {
        break;
      }
      const applied = await applyWorkerTurn(ai, current, moves, config, current.turn);
      if (!applied) {
        break;
      }
      current = applied.game;
      if (applied.status.terminal) {
        break;
      }
    } catch {
      break;
    }
  }
  return current;
}

async function generateCurriculumPositionGame(
  ai: Worker,
  game: GameSnapshot,
  config: NormalizedTrainingConfig,
  index: number,
  workerIndex: number
): Promise<GameSnapshot> {
  const generated = await generatePositionGame(ai, game, curriculumSearchConfig(config, index), index, workerIndex);
  return curriculumGame(generated, index);
}

async function generateTacticalPositionGame(
  ai: Worker,
  game: GameSnapshot,
  config: NormalizedTrainingConfig,
  index: number,
  workerIndex: number
): Promise<GameSnapshot> {
  let best = cloneGame(game);
  let bestPriority = tacticalPositionPriority(best);
  const attempts = 1 + (index % 4);
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const generated = await generatePositionGame(
      ai,
      bestPriority > 0 ? best : game,
      tacticalSearchConfig(config, attempt),
      index + attempt * MAX_PLAYOUT_PLIES,
      workerIndex
    );
    const priority = tacticalPositionPriority(generated);
    if (priority > bestPriority) {
      best = generated;
      bestPriority = priority;
    }
    if (priority >= 4) {
      break;
    }
  }
  return best;
}

function curriculumGame(game: GameSnapshot, index: number): GameSnapshot {
  const engine = loadedEngine();
  const [ptr, len] = writeWasmString(engine, JSON.stringify(game));
  try {
    const output = engine.chronofish_curriculum_game_snapshot_json(ptr, len, index);
    return JSON.parse(readWasmString(engine, output)) as GameSnapshot;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function tacticalPositionPriority(game: GameSnapshot): number {
  const engine = loadedEngine();
  const [ptr, len] = writeWasmString(engine, JSON.stringify(game));
  try {
    return engine.chronofish_tactical_position_priority_snapshot_json(ptr, len);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function applyWorkerTurn(
  worker: Worker,
  game: GameSnapshot,
  moves: Move[],
  config: NormalizedTrainingConfig,
  mover: Color
): Promise<AppliedWorkerTurn | null> {
  let current = game;
  for (const move of moves) {
    const beforeMove = current;
    const applied = await requestWorker(worker, {
      type: "applyMove",
      game: current,
      move
    }, workerRequestTimeout(config));
    if (!applied.game) {
      return null;
    }
    current = applied.game;
    const winner = royalCaptureWinner(beforeMove, current, mover);
    if (winner) {
      return {
        game: current,
        status: { terminal: true, winner },
        winner
      };
    }
  }

  const submitted = await requestWorker(worker, {
    type: "submitTurn",
    game: current
  }, workerRequestTimeout(config));
  const status = submitted.status ?? {};
  if (status.complete && status.nextTurn) {
    current = { ...current, turn: status.nextTurn };
  }
  return {
    game: current,
    status,
    winner: status.winner ?? null
  };
}

async function applyCpuWorkerTurn(
  worker: Worker,
  game: GameSnapshot,
  moves: Move[],
  config: NormalizedTrainingConfig
): Promise<AppliedWorkerTurn | null> {
  const response = await requestWorker(worker, {
    type: "applyTurn",
    game,
    moves
  }, workerRequestTimeout(config));
  if (!response.game) {
    return null;
  }
  const status = response.status ?? {};
  return {
    game: response.game,
    status,
    winner: status.winner ?? null
  };
}

interface OutcomeRelabelRequest {
  kind: "outcome" | "draw" | "partial";
  winner?: Color;
  labelKind?: TrainingLabelKind;
  labelWeight?: number;
}

async function relabelOutcomeSamplesWithEngine(
  samples: LabelWorkerSample[],
  request: OutcomeRelabelRequest
): Promise<TrainingSample[]> {
  const engine = await engineInstance();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    ...request,
    samples: samplesForEngine(samples)
  }));
  try {
    const output = engine.chronofish_relabel_outcome_samples_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as TrainingSample[];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function cpuParametersKey(parameters: CpuParameters): string {
  const engine = loadedEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(parameters));
  try {
    const output = engine.chronofish_cpu_parameters_key_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmString(engine, output);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function uniqueCpuParameters(values: CpuParameters[]): CpuParameters[] {
  const engine = loadedEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(values));
  try {
    const output = engine.chronofish_unique_cpu_parameters_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as CpuParameters[];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function breedCpuPopulation(
  baseline: CpuParameters,
  elites: CpuParameters[],
  target: number,
  seed: number,
  generation: number,
  stagnation: number
): CpuParameters[] {
  const engine = loadedEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    baseline,
    elites,
    target,
    seed,
    generation,
    stagnation
  }));
  try {
    const output = engine.chronofish_breed_cpu_population_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as CpuParameters[];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function modeLabelTarget(config: NormalizedTrainingConfig, divisor: number): number {
  return loadedEngine().chronofish_mode_label_target(
    config.samples,
    trainingModeCount(config),
    divisor
  );
}

function trainingModeEnabled(config: NormalizedTrainingConfig, mode: TrainingMode): boolean {
  return trainingModePolicy(config, mode).modeEnabled === true;
}

function cpuBaselineModeEnabled(config: NormalizedTrainingConfig): boolean {
  return trainingModePolicy(config).cpuBaselineModeEnabled;
}

function trainingModeCount(config: NormalizedTrainingConfig): number {
  return trainingModePolicy(config).trainingModeCount;
}

interface TrainingModePolicy {
  trainingModeCount: number;
  cpuBaselineModeEnabled: boolean;
  modeEnabled?: boolean;
}

function trainingModePolicy(config: NormalizedTrainingConfig, mode?: TrainingMode): TrainingModePolicy {
  const key = JSON.stringify({
    trainingSubject: config.trainingSubject,
    trainingModes: config.trainingModes,
    mode: mode ?? null
  });
  const cached = trainingModePolicyCache.get(key);
  if (cached) {
    return cached;
  }
  const engine = loadedEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    trainingSubject: config.trainingSubject,
    trainingModes: config.trainingModes,
    mode: mode ?? null
  }));
  try {
    const output = engine.chronofish_training_mode_policy_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const policy = JSON.parse(readWasmString(engine, output)) as TrainingModePolicy;
    trainingModePolicyCache.set(key, policy);
    return policy;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function royalCaptureWinner(before: GameSnapshot, after: GameSnapshot, mover: Color): Color | null {
  const engine = loadedEngine();
  const [ptr, len] = writeWasmString(engine, JSON.stringify({
    before: JSON.stringify(before),
    after: JSON.stringify(after),
    mover
  }));
  try {
    const output = engine.chronofish_royal_capture_winner_snapshot_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as Color | null;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function oppositeColor(color: Color): Color {
  return color === "white" ? "black" : "white";
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

function requestWorker(
  worker: Worker,
  payload: WorkerRequestPayload,
  timeoutMs = workerRequestTimeout(payload),
  transfer: Transferable[] = []
): Promise<AiWorkerResponse> {
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
    }, transfer);
  });
}

function normalizeSearchScore(score: number): number {
  return loadedEngine().chronofish_normalized_search_score(score);
}

function policyBucket(move: Move | null | undefined, intent = 0): number | null {
  if (!move) {
    return null;
  }
  return loadedEngine().chronofish_policy_bucket_from_move_values(
    move.from.timelineId,
    move.from.time,
    move.from.x,
    move.from.y,
    move.to.timelineId,
    move.to.time,
    move.to.x,
    move.to.y,
    intent
  );
}

interface TrainingLabelPolicy {
  outcomeLabelWeight: number;
  duelLabelWeight: number;
  duelDrawLabelWeight: number;
  distilledLabelWeight: number;
  defaultPartialOutcomeLabelKind: string;
  defaultPartialOutcomeLabelWeight: number;
}

function trainingLabelPolicy(): TrainingLabelPolicy {
  if (trainingLabelPolicyCache) {
    return trainingLabelPolicyCache;
  }
  const engine = loadedEngine();
  const output = engine.chronofish_training_label_policy_json();
  trainingLabelPolicyCache = JSON.parse(readWasmString(engine, output)) as TrainingLabelPolicy;
  return trainingLabelPolicyCache;
}

function cloneGame(game: GameSnapshot): GameSnapshot {
  return JSON.parse(JSON.stringify(game));
}

function randomRunSeed(): number {
  const values = new Uint32Array(1);
  crypto.getRandomValues(values);
  return values[0]! >>> 0;
}

function workerRequestTimeout(payload: { nodes?: unknown; timeMs?: unknown }): number {
  return loadedEngine().chronofish_training_worker_request_timeout_ms(
    Number(payload.nodes) || 1,
    Number(payload.timeMs) || 0
  );
}

function workerSearchTimeMs(payload: { nodes?: unknown; timeMs?: unknown }): number {
  return loadedEngine().chronofish_training_worker_search_time_ms(
    Number(payload.nodes) || 1,
    Number(payload.timeMs) || 0
  );
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
  return loadedEngine().chronofish_training_sample_plies(index, encodeOnly ? 1 : 0);
}

function sampleSeed(prefix: string, index: number, salt: number): number {
  const engine = loadedEngine();
  const { ptr, len } = writeWasmString(engine, prefix);
  try {
    return engine.chronofish_training_sample_seed(ptr, len, index, salt) >>> 0;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function searchSeed(value: unknown, salt: number): number {
  const engine = loadedEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(value ?? null));
  try {
    return engine.chronofish_training_search_seed_json(ptr, len, salt) >>> 0;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function validateLossLogs(
  config: NormalizedTrainingConfig,
  progress?: ProgressCallback,
  candidateModel?: ArrayBuffer
): Promise<LossLogValidation> {
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
    if (candidateModel) {
      await requestWorker(ai, {
        type: "setModel",
        modelBytes: candidateModel
      }, TRAINING_IO_TIMEOUT_MS, [candidateModel]);
    }
    for (const log of logs) {
      const decisions = Array.isArray(log.decisions) ? log.decisions : [];
      let logChanged = false;
      for (const decision of decisions) {
        const previousKey = movesKey(decision.selectedMoves);
        if (!decision.game || !previousKey) {
          validation.skipped += 1;
          continue;
        }
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
            temperature: 0,
            randomSeed: sampleSeed("loss-log", validation.checked, config.runSeed ^ 0x1055_1000)
          }, workerRequestTimeout(config));
          const currentMoves = response.result?.moves ?? [];
          const currentKey = movesKey(currentMoves);
          if (!currentKey) {
            validation.skipped += 1;
            continue;
          }
          validation.checked += 1;
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

function autoLabelWorkers(): number {
  const cores = navigator.hardwareConcurrency ?? 4;
  return Math.max(1, Math.min(cores - 1, 16));
}
