import { train, predictValues, normalizedSearchScore } from "./training-gpu.js";
import { policyBucket } from "./training-policy.js";
import { breedCpuPopulation, cpuParametersKey, cpuReferenceWorkerCount, uniqueCpuParameters } from "./training-cpu.js";
import type { CpuParameters } from "./training-cpu.js";
import { appendReplaySamples, dedupeTrainingSamples } from "./training-replay.js";
import { fetchActiveModel, fetchCpuParameters, loadReplayBuffer, saveReplayBuffer } from "./training-worker-storage.js";
import type { Color, GameSnapshot, Move, BoardSnapshot, Timeline } from "./types.js";
import type { CompactValueModel, EncodedCompactModel, TrainingConfig as GpuTrainingConfig, TrainingMetrics, TrainingSample } from "./training-gpu.js";
import { DEFAULT_BATCH_SIZE, DEFAULT_PATIENCE, DEFAULT_VALIDATION_SPLIT, DEFAULT_WEIGHT_DECAY, HIDDEN_LAYERS, LABEL_REQUEST_MAX_TIMEOUT_MS, LABEL_REQUEST_MIN_TIMEOUT_MS, LABEL_REQUEST_NODE_TIMEOUT_FACTOR_MS, MAX_GPU_TRAINING_BATCH, MAX_GPU_TRAINING_SAMPLES, MAX_GPU_VALIDATION_INTERVAL, MAX_PARALLEL_GPU_TRAINING_WORKERS, MAX_PLAYOUT_PLIES, POLICY_STEPS_PER_SUBMIT, PROJECTION_CHUNK_SIZE, PROJECTION_SEED, PROJECTION_SIZE, TRAINING_IO_TIMEOUT_MS, VALUE_EPOCHS_PER_SUBMIT } from "./training-worker-types.js";
import type { AiWorkerResponse, AppliedWorkerTurn, CpuReferenceScore, CpuTrainingResult, EncodedPosition, LabelJob, LabelWorkerSample, LossLog, LossLogValidation, LossLogValidationExample, MetricsSummary, NormalizedTrainingConfig, ProgressCallback, TrainingLabelKind, TrainingMode, TrainingRunMetrics, TrainingSubject, TrainingWorkerRequest, TrainingWorkerResponse, WorkerRequestPayload, WorkerScope } from "./training-worker-types.js";
let cachedGpuAdapter: GPUAdapter | null = null;
let cachedGpuDevice: GPUDevice | null = null;
const pipelineCache = new Map<string, GPUComputePipeline>();

const workerSelf = self as unknown as WorkerScope;

workerSelf.addEventListener("message", async (event) => {
  const { id, type = "train", game, config, candidateModel } = event.data;
  try {
    const metrics = createTrainingMetrics();
    const normalizedConfig = normalizeTrainingConfig(config);
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
    const samples = dedupeTrainingSamples(collectedSamples);
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

function normalizeTrainingConfig(config: Partial<NormalizedTrainingConfig> = {}): NormalizedTrainingConfig {
  const trainingSubject = isTrainingSubject(config.trainingSubject) ? config.trainingSubject : legacyTrainingSubject(config);
  return {
    ...config,
    trainingSubject,
    trainingModes: normalizeTrainingModes(config, trainingSubject),
    runSeed: randomRunSeed(),
    learningRate: clampNumber(config.learningRate, 0.0001, 0.1, 0.01),
    samples: clampInteger(config.samples, 1, MAX_GPU_TRAINING_SAMPLES, 64),
    selfPlayWorkers: clampInteger(config.selfPlayWorkers, 1, 16, 2),
    searchWorkers: clampInteger(config.searchWorkers, 1, 16, 2),
    explorationTemperature: clampNumber(config.explorationTemperature, 0, 2, 0.25),
    depth: clampInteger(config.depth, 1, 8, 5),
    nodes: clampInteger(config.nodes, 1, 131072, 16384),
    epochs: clampInteger(config.epochs, 1, 65536, 8192),
    maxBuffer: clampInteger(config.maxBuffer, 16, 16384, 4096),
    batchSize: clampInteger(config.batchSize, 16, MAX_GPU_TRAINING_BATCH, DEFAULT_BATCH_SIZE),
    validationSplit: clampNumber(config.validationSplit, 0, 0.3, DEFAULT_VALIDATION_SPLIT),
    validationInterval: clampInteger(config.validationInterval, 16, MAX_GPU_VALIDATION_INTERVAL, 256),
    patience: clampInteger(config.patience, 1, 64, DEFAULT_PATIENCE),
    weightDecay: clampNumber(config.weightDecay, 0, 0.01, DEFAULT_WEIGHT_DECAY),
    lossLogReplay: clampInteger(config.lossLogReplay, 0, 32, 4),
    cpuDepth: clampInteger(config.cpuDepth, 1, 16, 4),
    cpuNodes: clampInteger(config.cpuNodes, 1, 131072, 8192),
    cpuTrainingTimeMs: clampInteger(config.cpuTrainingTimeMs, 1, 600000, 10000),
    cpuCandidates: clampInteger(config.cpuCandidates, 1, 256, 8),
    cpuFinalists: clampInteger(config.cpuFinalists, 1, 64, 1),
    cpuPairBatch: clampInteger(config.cpuPairBatch, 1, 64, 4),
    cpuOpponentVariants: clampInteger(config.cpuOpponentVariants, 1, 128, 8),
    cpuScreeningOpponentVariants: clampInteger(config.cpuScreeningOpponentVariants, 1, 128, 2),
    cpuRoundsPerVariant: clampInteger(config.cpuRoundsPerVariant, 1, 64, 1),
    cpuHallOfFameEntries: clampInteger(config.cpuHallOfFameEntries, 0, 64, 1),
    cpuLeagueContenders: clampInteger(config.cpuLeagueContenders, 1, 64, 2),
    cpuLeagueHallOfFameEntries: clampInteger(config.cpuLeagueHallOfFameEntries, 0, 64, 2),
    cpuMinPairs: clampInteger(config.cpuMinPairs, 1, 256, 2),
    cpuMaxPairs: clampInteger(config.cpuMaxPairs, 1, 512, 8),
    cpuDrawWindow: clampInteger(config.cpuDrawWindow, 1, 128, 4),
    cpuDrawRateLimit: clampNumber(config.cpuDrawRateLimit, 0, 1, 0.8),
    cpuMaxMatchPlies: clampInteger(config.cpuMaxMatchPlies, 1, 512, 40),
    cpuMaxMatchTimeMs: clampInteger(config.cpuMaxMatchTimeMs, 0, 3600000, 0),
    cpuMaxGenerationsWithoutCandidate: clampInteger(config.cpuMaxGenerationsWithoutCandidate, 1, 256, 2),
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

function isTrainingSubject(value: unknown): value is TrainingSubject {
  return value === "gpu" || value === "cpu";
}

function isTrainingMode(value: unknown): value is TrainingMode {
  return value === "vsGpu"
    || value === "vsCpu"
    || value === "self"
    || value === "distill"
    || value === "curriculum"
    || value === "tactical";
}

function legacyTrainingSubject(config: Partial<NormalizedTrainingConfig>): TrainingSubject {
  return (config as { trainingTarget?: unknown }).trainingTarget === "trainCpu" ? "cpu" : "gpu";
}

function normalizeTrainingModes(config: Partial<NormalizedTrainingConfig>, subject: TrainingSubject): TrainingMode[] {
  const explicit = Array.isArray((config as { trainingModes?: unknown }).trainingModes)
    ? (config as { trainingModes: unknown[] }).trainingModes.filter(isTrainingMode)
    : [];
  const legacy = explicit.length > 0 ? explicit : legacyTrainingModes(config, subject);
  const filtered = subject === "cpu" ? legacy.filter((mode) => mode !== "distill") : legacy;
  const deduped = Array.from(new Set(filtered));
  return deduped.length > 0 ? deduped : subject === "cpu" ? ["vsCpu"] : ["vsGpu", "self"];
}

function legacyTrainingModes(config: Partial<NormalizedTrainingConfig>, subject: TrainingSubject): TrainingMode[] {
  if (subject === "cpu") {
    const target = (config as { cpuTrainingTarget?: unknown }).cpuTrainingTarget;
    if (target === "vsGpu") return ["vsGpu"];
    if (target === "vsBoth") return ["vsCpu", "vsGpu"];
    return ["vsCpu"];
  }
  const target = (config as { trainingTarget?: unknown }).trainingTarget;
  const labelMode = (config as { labelMode?: unknown }).labelMode;
  const modes: TrainingMode[] = [];
  if (target === "trainCpu" || target === "trainBoth") {
    modes.push("vsCpu");
  }
  if (target !== "trainCpu" && (target === "trainBoth" || labelMode === "mixed" || labelMode === "search" || labelMode === undefined)) {
    modes.push("vsGpu");
  }
  if (target !== "trainCpu" && (labelMode === "mixed" || labelMode === "selfPlay" || labelMode === undefined)) {
    modes.push("self");
  }
  if (target !== "trainCpu" && (labelMode === "mixed" || labelMode === "distill" || labelMode === undefined)) {
    modes.push("distill");
  }
  if (target !== "trainCpu" && labelMode === "mixed") {
    modes.push("curriculum", "tactical");
  }
  return modes;
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
  const candidateCount = Math.max(1, Math.min(256, config.cpuCandidates));
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
    const finalistTarget = Math.min(
      population.length,
      Math.max(config.cpuFinalists, Math.min(config.cpuPairBatch, screened.length || population.length))
    );
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
      .slice(0, Math.max(1, Math.min(4, config.cpuFinalists)))
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
    const workerCount = Math.min(uniqueCandidates.length, Math.max(1, stageConfig.cpuWorkers), Math.max(1, stageConfig.cpuPairBatch));
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
  const workerCount = Math.min(target, Math.max(1, config.cpuWorkers));
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
          const response = await requestWorker(cpu, {
            type: "search",
            game: current,
            depth: Math.max(1, Math.min(2, config.cpuDepth)),
            nodes: Math.max(1, Math.min(512, config.cpuNodes)),
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
  return {
    ...config,
    cpuDepth: Math.max(1, Math.min(config.cpuDepth, 2)),
    depth: Math.max(1, Math.min(config.depth, 2)),
    cpuNodes: Math.max(1, Math.min(config.cpuNodes, Math.ceil(config.cpuNodes / 4))),
    nodes: Math.max(1, Math.min(config.nodes, Math.ceil(config.nodes / 4))),
    cpuTrainingTimeMs: Math.max(1, Math.min(config.cpuTrainingTimeMs, Math.ceil(config.cpuTrainingTimeMs / 4)))
  };
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
      const delta = candidateScore - reference.baselineScore;
      score += delta;
      if (Math.abs(delta) <= config.cpuDrawWindow) {
        nearDraws += 1;
      }
      score += moveAgreementBonus(candidateResult.result?.moves, reference.baselineMoves);
    }
    if (trainingModeEnabled(config, "vsGpu") && reference.gpuScore !== undefined) {
      score += candidateScore - reference.gpuScore;
      score += moveAgreementBonus(candidateResult.result?.moves, reference.gpuMoves);
    }
    compared += 1;
  }
  const average = score / Math.max(1, compared);
  const nearDrawRate = nearDraws / Math.max(1, compared);
  return nearDrawRate > config.cpuDrawRateLimit ? average * 0.5 : average;
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
      return candidateTurn ? -100_000 : 100_000;
    }
    const applied = await applyCpuWorkerTurn(baselineWorker, current, moves, config);
    if (!applied) {
      return null;
    }
    current = applied.game;
    if (applied.winner) {
      return applied.winner === candidateColor ? 100_000 : -100_000;
    }
    if (applied.status.terminal) {
      return applied.status.winner === candidateColor
        ? 100_000
        : applied.status.winner
          ? -100_000
          : 0;
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
  return current.turn === candidateColor ? baselineScore : -baselineScore;
}

function cpuMatchTurnTimeMs(config: NormalizedTrainingConfig, deadlineAt: number, remainingSearches: number): number {
  return Math.max(
    1,
    Math.min(
      config.cpuTrainingTimeMs,
      Math.floor((deadlineAt - performance.now()) / Math.max(1, remainingSearches))
    )
  );
}

function cpuTrainingPositionTarget(config: NormalizedTrainingConfig): number {
  const variantPairs = (config.cpuOpponentVariants + config.cpuScreeningOpponentVariants) * config.cpuRoundsPerVariant;
  const leaguePairs = config.cpuLeagueContenders * Math.max(1, config.cpuLeagueHallOfFameEntries);
  const hallPairs = Math.max(0, config.cpuHallOfFameEntries);
  const requested = Math.max(config.cpuMinPairs, variantPairs + leaguePairs + hallPairs);
  const cappedPairs = Math.min(Math.max(config.cpuMinPairs, config.cpuMaxPairs), requested, config.cpuMaxMatchPlies);
  return modeLabelTarget(config, Math.max(1, cappedPairs));
}

function cpuTrainingDeadlineAt(config: NormalizedTrainingConfig): number {
  const fallbackMs = Math.min(
    config.cpuTrainSeconds * 1000,
    config.cpuTrainingTimeMs * Math.max(1, config.cpuMaxMatchPlies) * 60
  );
  const budgetMs = config.cpuMaxMatchTimeMs > 0 ? Math.min(config.cpuMaxMatchTimeMs, fallbackMs) : fallbackMs;
  return performance.now() + Math.max(1000, budgetMs);
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
        labelWeight: trainingModeCount(config) > 1 ? 1.1 : 1.0,
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
        labelWeight: 1.35,
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
        return backfillOutcomeLabels(samples, applied.winner).map((sample) => ({
          ...sample,
          labelKind: "duel",
          labelWeight: 1.35
        }));
      }
      if (applied.status.terminal && applied.status.winner) {
        return backfillOutcomeLabels(samples, applied.status.winner).map((sample) => ({
          ...sample,
          labelKind: "duel",
          labelWeight: 1.35
        }));
      }
      if (applied.status.terminal) {
        return backfillDrawLabels(samples, "duel", 1.1);
      }
    }
    return samplesFromPartialOutcome(samples, "duel-search", 1.0);
  } catch {
    return samplesFromPartialOutcome(samples, "duel-search", 1.0);
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
        labelWeight: 1.25,
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
        return backfillOutcomeLabels(samples, applied.winner);
      }
      if (applied.status.terminal && applied.status.winner) {
        return backfillOutcomeLabels(samples, applied.status.winner);
      }
      if (applied.status.terminal) {
        return backfillDrawLabels(samples, "outcome", 1.0);
      }
    }
    return samplesFromPartialOutcome(samples);
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

function gpuTrainingWorkerCount(total: number, requestedWorkers: number): number {
  return Math.min(
    Math.max(0, total),
    Math.max(1, Math.min(MAX_PARALLEL_GPU_TRAINING_WORKERS, Math.floor(requestedWorkers) || 1))
  );
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
    labelWeight: 0.25,
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

function curriculumSearchConfig(config: NormalizedTrainingConfig, index: number): NormalizedTrainingConfig {
  const stage = index % 6;
  return {
    ...config,
    depth: Math.max(1, Math.min(config.depth, 1 + Math.floor(stage / 2))),
    nodes: Math.max(1, Math.min(config.nodes, 512 * (stage + 1))),
    explorationTemperature: Math.min(0.6, Math.max(config.explorationTemperature, stage >= 3 ? 0.35 : 0.15))
  };
}

function tacticalSearchConfig(config: NormalizedTrainingConfig, attempt: number): NormalizedTrainingConfig {
  return {
    ...config,
    depth: Math.max(2, Math.min(config.depth, 3 + attempt)),
    nodes: Math.max(1024, Math.min(config.nodes, 2048 * (attempt + 1))),
    explorationTemperature: Math.min(0.8, Math.max(config.explorationTemperature, 0.4 + attempt * 0.1))
  };
}

function curriculumGame(game: GameSnapshot, index: number): GameSnapshot {
  const stage = index % 6;
  const cloned = cloneGame(game);
  const presentTime = activePresentTime(cloned);
  let timelines = cloned.timelines
    .map((timeline) => ({
      ...timeline,
      boards: curriculumBoards(timeline.boards, presentTime, stage)
    }))
    .filter((timeline) => timeline.boards.length > 0)
    .sort((left, right) => curriculumTimelinePriority(right, presentTime) - curriculumTimelinePriority(left, presentTime));
  const timelineLimit = stage <= 1 ? 1 : stage <= 3 ? 2 : Math.max(2, Math.min(timelines.length, 4));
  timelines = timelines.slice(0, timelineLimit).map((timeline, row) => {
    const next = { ...timeline, row };
    if (stage < 4) {
      next.active = row === 0;
    }
    return next;
  });
  const timelineIds = new Set(timelines.map((timeline) => timeline.id));
  const boardTimes = new Map(timelines.map((timeline) => [
    timeline.id,
    new Set(timeline.boards.map((board) => board.time))
  ]));
  return {
    ...cloned,
    presentTime,
    timelines,
    checkedRoyals: cloned.checkedRoyals.filter((position) =>
      timelineIds.has(position.timelineId) && boardTimes.get(position.timelineId)?.has(position.time)
    )
  };
}

function curriculumBoards(boards: BoardSnapshot[], presentTime: number, stage: number): BoardSnapshot[] {
  if (!boards.length) {
    return [];
  }
  const latest = latestBoard({ id: 0, row: 0, label: "", owner: "neutral", boards });
  const presentBoards = boards.filter((board) => board.time === presentTime);
  const candidates = stage <= 1
    ? [presentBoards.at(-1) ?? latest]
    : stage <= 3
      ? [...boards.slice(-2), presentBoards.at(-1)]
      : boards.slice(Math.max(0, boards.length - (stage + 1)));
  const unique = new Map<number, BoardSnapshot>();
  for (const board of candidates) {
    if (board) {
      unique.set(board.time, curriculumBoard(board, stage));
    }
  }
  return Array.from(unique.values()).sort((left, right) => left.time - right.time);
}

function curriculumBoard(board: BoardSnapshot, stage: number): BoardSnapshot {
  if (stage !== 0) {
    return board;
  }
  const classic = new Set(["king", "queen", "rook", "bishop", "knight", "pawn"]);
  return {
    ...board,
    board: board.board.map((row) => row.map((piece) => {
      if (!piece) {
        return null;
      }
      if (classic.has(piece.type)) {
        return piece;
      }
      return piece.type === "royalQueen"
        ? { ...piece, type: "queen" }
        : null;
    }))
  };
}

function curriculumTimelinePriority(timeline: Timeline, presentTime: number): number {
  const hasPresent = timeline.boards.some((board) => board.time === presentTime) ? 4 : 0;
  const active = timeline.active ? 2 : 0;
  const latest = latestBoard(timeline)?.time ?? Number.NEGATIVE_INFINITY;
  return hasPresent + active + latest / 1000;
}

function tacticalPositionPriority(game: GameSnapshot): number {
  let priority = 0;
  const activeTimelines = game.timelines.filter((timeline) => timeline.active !== false);
  priority += Math.min(3, game.checkedRoyals.length * 2);
  priority += Math.max(0, activeTimelines.length - 1);
  priority += Math.max(0, game.timelines.length - 2);
  priority += royalExposure(game);
  priority += temporalPowerPieceCount(game) > 1 ? 1 : 0;
  return priority;
}

function royalExposure(game: GameSnapshot): number {
  let exposed = 0;
  for (const timeline of game.timelines) {
    const board = latestBoard(timeline);
    for (const row of board?.board ?? []) {
      for (const piece of row) {
        if (piece && ["king", "royalQueen"].includes(piece.type)) {
          exposed += timeline.active === false ? 0 : 1;
        }
      }
    }
  }
  return Math.min(2, exposed);
}

function temporalPowerPieceCount(game: GameSnapshot): number {
  let count = 0;
  for (const timeline of game.timelines) {
    const board = latestBoard(timeline);
    for (const row of board?.board ?? []) {
      for (const piece of row) {
        if (piece && ["queen", "royalQueen", "unicorn", "dragon"].includes(piece.type)) {
          count += 1;
        }
      }
    }
  }
  return count;
}

function activePresentTime(game: GameSnapshot): number {
  if (Number.isFinite(game.presentTime)) {
    return game.presentTime ?? 0;
  }
  const times = game.timelines
    .filter((timeline) => timeline.active !== false)
    .map((timeline) => latestBoard(timeline)?.time)
    .filter((time): time is number => typeof time === "number");
  return times.length ? Math.min(...times) : 0;
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

function samplesFromPartialOutcome(
  samples: LabelWorkerSample[],
  labelKind: TrainingLabelKind = "search-bootstrap",
  labelWeight = 0.5
): TrainingSample[] {
  return samples.map(({ outcomeTurn, ply, ...sample }) => ({
    ...sample,
    labelKind,
    labelWeight
  }));
}

function modeLabelTarget(config: NormalizedTrainingConfig, divisor: number): number {
  if (trainingModeCount(config) <= 1) {
    return config.samples;
  }
  return Math.max(1, Math.ceil(config.samples / divisor));
}

function trainingModeEnabled(config: NormalizedTrainingConfig, mode: TrainingMode): boolean {
  return config.trainingModes.includes(mode);
}

function cpuBaselineModeEnabled(config: NormalizedTrainingConfig): boolean {
  return trainingModeEnabled(config, "vsCpu") || trainingModeEnabled(config, "self");
}

function trainingModeCount(config: NormalizedTrainingConfig): number {
  return config.trainingSubject === "cpu"
    ? config.trainingModes.filter((mode) => mode !== "distill").length
    : config.trainingModes.length;
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

function backfillDrawLabels(samples: LabelWorkerSample[], labelKind: TrainingLabelKind, labelWeight: number): TrainingSample[] {
  return samples.map(({ outcomeTurn, ply, ...sample }) => ({
    ...sample,
    label: 0,
    labelKind,
    labelWeight
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
  return normalizedSearchScore(score);
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

function workerRequestTimeout(payload: { nodes?: unknown; timeMs?: unknown }): number {
  const nodes = Math.max(1, Number(payload.nodes) || 1);
  const timeMs = Math.max(0, Number(payload.timeMs) || 0);
  return Math.min(
    Math.max(LABEL_REQUEST_MAX_TIMEOUT_MS, timeMs + 5000),
    Math.max(
      LABEL_REQUEST_MIN_TIMEOUT_MS,
      timeMs + 1000,
      nodes * LABEL_REQUEST_NODE_TIMEOUT_FACTOR_MS
    )
  );
}

function workerSearchTimeMs(payload: { nodes?: unknown; timeMs?: unknown }): number {
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
