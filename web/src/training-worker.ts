import { train, predictValues } from "./training-gpu.js";
import { GpuTrainingBinding } from "./engine-gpu-training.js";
import { fetchActiveModel, fetchCpuParameters, loadReplayBuffer, saveReplayBuffer } from "./training-worker-storage.js";
import type { ChronofishEngine, Color, GameSnapshot, Move } from "./types.js";
import type { CompactValueModel, EncodedCompactModel, TrainingConfig as GpuTrainingConfig, TrainingMetrics, TrainingSample } from "./training-gpu.js";
import type { AiSearchResult, AiWorkerResponse, AppliedWorkerTurn, CpuParameters, CpuReferenceScore, CpuTrainingResult, EncodedPosition, LabelJob, LabelWorkerSample, LossLog, LossLogValidation, LossLogValidationExample, MetricsSummary, NormalizedTrainingConfig, ProgressCallback, TrainingLabelKind, TrainingMode, TrainingRunMetrics, TrainingSubject, TrainingWorkerRequest, TrainingWorkerResponse, WorkerRequestPayload, WorkerScope } from "./training-worker-types.js";
let cachedGpuAdapter: GPUAdapter | null = null;
let cachedGpuDevice: GPUDevice | null = null;
const trainingBinding = new GpuTrainingBinding();
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
    const [engine, loadedBuffer] = await timed(metrics, "load", () => Promise.all([
      engineInstance(),
      loadReplayBuffer()
    ]));
    const activeModel = await fetchActiveModel(engine);
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
    }, engine));
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
  return trainingBinding.normalizeConfig<EngineNormalizedTrainingConfig>(config);
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
  return trainingBinding.metricsSummary<MetricsSummary>({
    startedAt: metrics.startedAt,
    nowMs: performance.now(),
    phases: metrics.phases,
    sampleCounts: metrics.sampleCounts ?? {},
    searchPositionCount: metrics.searchPositionCount ?? null,
    searchLabelCount: metrics.searchLabelCount ?? null,
    lossLogValidation: metrics.lossLogValidation ?? null
  });
}

async function dedupeTrainingSamplesWithEngine(samples: TrainingSample[]): Promise<TrainingSample[]> {
  return trainingBinding.dedupeSamples(samples);
}

async function appendReplaySamplesWithEngine(
  buffer: TrainingSample[],
  samples: TrainingSample[],
  maxBuffer: number
): Promise<TrainingSample[]> {
  return trainingBinding.appendReplaySamples(buffer, samples, maxBuffer);
}

async function labelSourceCountsWithEngine(samples: TrainingSample[]): Promise<Record<string, number>> {
  return trainingBinding.labelSourceCounts(samples);
}

function samplesForEngine(samples: TrainingSample[]): TrainingSample[] {
  return samples.map((sample) => ({
    ...sample,
    features: Array.from(sample.features ?? [])
  }));
}

async function engineInstance(): Promise<ChronofishEngine> {
  return trainingBinding.engine();
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
  const failures = collected
    .filter((result): result is PromiseRejectedResult => result.status === "rejected")
    .map((result) => errorMessage(result.reason))
    .filter((message, index, messages) => message && messages.indexOf(message) === index);
  if (failures.length > 0) {
    throw new Error(`Training label collection failed: ${failures.join("; ")}`);
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
  const screeningGames = sampleGames.slice(0, cpuScreeningGameCount(sampleGames.length, config.cpuScreeningOpponentVariants));
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
  while (cpuTrainingShouldContinue(deadlineAt, generationsWithoutCandidate, config)) {
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
    const finalistCandidates = cpuTrainingFinalistCandidates(baseline, screened, finalistTarget);
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
    const outcome = cpuTrainingGenerationOutcome(baseline, finalists, baselineScore, bestCandidate?.score);
    baselineScore = outcome.baselineScore;
    const winner = outcome.winner;
    const improved = outcome.improved;
    if (improved && winner) {
      bestCandidate = winner;
    }
    generationsWithoutCandidate = cpuTrainingNextStagnation(generationsWithoutCandidate, improved);
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
    const elites = cpuTrainingElites(finalists, baseline, config);
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
    const plan = cpuCandidateScoringPlan(stageCandidates, fitnessCache);
    const uniqueCandidates = plan.uniqueCandidates;
    const workerCount = cpuCandidateWorkerCount(uniqueCandidates.length, stageConfig.cpuWorkers, stageConfig.cpuPairBatch);
    let nextCandidate = 0;
    let collected = 0;
    const scored = plan.cachedScores;
    const uncachedCandidates = plan.uncachedCandidates;
    const cacheHits = plan.cacheHits;
    progress({ sampleCount: uniqueCandidates.length, labelWorkers: workerCount, labelKind });
    await Promise.all(Array.from({ length: workerCount }, () => runCandidateWorker()));
    progress({ collected: scored.length, sampleCount: uniqueCandidates.length, labelWorkers: workerCount, labelKind, cacheHits });
    return rankCpuScoredCandidates(scored);

    async function runCandidateWorker(): Promise<void> {
      const candidateWorker = new Worker("./cpu-ai-worker.js", { type: "module" });
      const baselineWorker = pairedMatches ? new Worker("./cpu-ai-worker.js", { type: "module" }) : null;
      try {
        while (cpuCandidateScoringShouldContinue(stageDeadlineAt, nextCandidate, uncachedCandidates.length)) {
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
          const fitness = cpuFitnessEntryForCandidate(candidate, score);
          fitnessCache.set(fitness.key, fitness.score);
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
          const moves = searchResultTurn(response.result).moves;
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
  return trainingBinding.resultValue<CpuTrainingPositionSearchConfig>(
    "chronofish_cpu_training_position_search_config_json",
    config.cpuDepth,
    config.cpuNodes
  );
}

function cpuScreeningTrainingConfigWithEngine(config: NormalizedTrainingConfig): CpuScreeningTrainingConfig {
  return trainingBinding.resultValue<CpuScreeningTrainingConfig>(
    "chronofish_cpu_screening_training_config_json",
    config.cpuDepth,
    config.depth,
    config.cpuNodes,
    config.nodes,
    config.cpuTrainingTimeMs
  );
}

function cpuTrainingPositionWorkerCount(target: number, cpuWorkers: number): number {
  return trainingBinding.numericValue("chronofish_cpu_training_position_worker_count", target, cpuWorkers);
}

function cpuReferenceWorkerCount(gameCount: number, requestedWorkers: number, pairBatch: number): number {
  return trainingBinding.numericValue("chronofish_cpu_reference_worker_count", gameCount, requestedWorkers, pairBatch);
}

function cpuCandidateWorkerCount(candidateCount: number, cpuWorkers: number, pairBatch: number): number {
  return trainingBinding.numericValue("chronofish_cpu_candidate_worker_count", candidateCount, cpuWorkers, pairBatch);
}

function cpuLabelWorkerCount(positionCount: number, cpuWorkers: number): number {
  return trainingBinding.numericValue("chronofish_cpu_label_worker_count", positionCount, cpuWorkers);
}

function cpuSearchLabelWeight(config: NormalizedTrainingConfig): number {
  return trainingBinding.numericValue("chronofish_cpu_search_label_weight", trainingModeCount(config));
}

function cpuReferenceComparisonCount(gameCount: number, referenceCount: number): number {
  return trainingBinding.numericValue("chronofish_cpu_reference_comparison_count", gameCount, referenceCount);
}

function cpuReferenceShouldContinue(nowMs: number, deadlineAtMs: number, compared: number, maxMatchPlies: number): boolean {
  return trainingBinding.numericValue(
    "chronofish_cpu_reference_should_continue",
    nowMs,
    deadlineAtMs,
    compared,
    maxMatchPlies
  ) !== 0;
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
      while (cpuReferenceCollectionShouldContinue(deadlineAt, nextGame, games.length)) {
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
          const baselineReference = cpuReferenceScoreFromResult(baselineResult.result);
          reference.baselineScore = baselineReference.score;
          if (baselineReference.moves) {
            reference.baselineMoves = baselineReference.moves;
          }
        }
        if (trainingModeEnabled(config, "vsGpu")) {
          const gpuReferenceTimeMs = cpuGpuReferenceTimeMs(config);
          const gpuResult = await requestWorker(gpuWorker, {
            type: "search",
            game,
            depth: config.depth,
            nodes: config.nodes,
            timeMs: gpuReferenceTimeMs,
            gpuMode: "hybrid"
          }, workerRequestTimeout({ nodes: config.nodes, timeMs: gpuReferenceTimeMs }));
          const gpuReference = cpuReferenceScoreFromResult(gpuResult.result);
          reference.gpuScore = gpuReference.score;
          if (gpuReference.moves) {
            reference.gpuMoves = gpuReference.moves;
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

function cpuGpuReferenceTimeMs(config: NormalizedTrainingConfig): number {
  // High CPU training evaluates a small number of expensive GPU references.
  // Give the full/hybrid fallback enough time to initialize and finish rather
  // than deriving a short budget solely from the GPU node count.
  return Math.max(config.cpuTrainingTimeMs, 90_000);
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
  const comparisonCount = cpuReferenceComparisonCount(games.length, references.length);
  for (let index = 0; index < comparisonCount; index += 1) {
    if (!cpuReferenceShouldContinue(performance.now(), deadlineAt, compared, config.cpuMaxMatchPlies)) {
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
    const reference = references[index] ?? {};
    if (cpuBaselineModeEnabled(config) && reference.baselineScore !== undefined) {
      const delta = cpuReferenceScoreDeltaFromResult(
        candidateResult.result,
        reference.baselineScore,
        reference.baselineMoves,
        config.cpuDrawWindow
      );
      score += delta.score;
      if (delta.nearDraw) {
        nearDraws += 1;
      }
    }
    if (trainingModeEnabled(config, "vsGpu") && reference.gpuScore !== undefined) {
      score += cpuReferenceScoreDeltaFromResult(
        candidateResult.result,
        reference.gpuScore,
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
  const totalMatches = cpuPairedMatchTotalMatches(games.length);
  for (let gameIndex = 0; gameIndex < games.length; gameIndex += 1) {
    const game = games[gameIndex]!;
    for (const candidateColor of cpuPairedMatchCandidateColors(game.turn)) {
      if (!cpuMatchShouldContinue(deadlineAt)) {
        return null;
      }
      const matchDeadlineAt = cpuPairedMatchDeadlineAt(deadlineAt, totalMatches, completed);
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
  return cpuPairedMatchAverageScore(score, completed);
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
    if (!cpuMatchShouldContinue(matchDeadlineAt)) {
      return null;
    }
    const turnTimeMs = cpuMatchTurnTimeMs(config, matchDeadlineAt, cpuMatchRemainingSearches(config.cpuMaxMatchPlies, ply));
    const candidateTurn = cpuTrainingCandidateTurn(current.turn, candidateColor);
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
    const moves = searchResultTurn(response.result).moves;
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

  if (!cpuMatchShouldContinue(matchDeadlineAt)) {
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
  return cpuTrainingAdjudicationScoreFromResult(current.turn, candidateColor, adjudication.result);
}

function cpuMatchTurnTimeMs(config: NormalizedTrainingConfig, deadlineAt: number, remainingSearches: number): number {
  return trainingBinding.numericValue(
    "chronofish_cpu_match_turn_time_ms",
    config.cpuTrainingTimeMs,
    performance.now(),
    deadlineAt,
    remainingSearches
  );
}

function cpuMatchRemainingSearches(maxMatchPlies: number, ply: number): number {
  return trainingBinding.numericValue("chronofish_cpu_match_remaining_searches", maxMatchPlies, ply);
}

function cpuMatchShouldContinue(deadlineAt: number): boolean {
  return trainingBinding.numericValue("chronofish_cpu_match_should_continue", performance.now(), deadlineAt) !== 0;
}

function cpuPairedMatchDeadlineAt(deadlineAt: number, totalMatches: number, completedMatches: number): number {
  return trainingBinding.numericValue(
    "chronofish_cpu_paired_match_deadline_ms",
    performance.now(),
    deadlineAt,
    totalMatches,
    completedMatches
  );
}

function cpuPairedMatchTotalMatches(gameCount: number): number {
  return trainingBinding.numericValue("chronofish_cpu_paired_match_total_matches", gameCount);
}

function cpuPairedMatchCandidateColors(turn: Color): Color[] {
  return trainingBinding.jsonValue<Color[]>("chronofish_cpu_paired_match_candidate_colors_json", turn);
}

function cpuPairedMatchAverageScore(score: number, completedMatches: number): number | null {
  const average = trainingBinding.numericValue("chronofish_cpu_paired_match_average_score", score, completedMatches);
  return Number.isFinite(average) ? average : null;
}

function cpuTrainingPositionTarget(config: NormalizedTrainingConfig): number {
  return trainingBinding.numericValue(
    "chronofish_cpu_training_position_target",
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
  return trainingBinding.numericValue("chronofish_cpu_training_candidate_count", config.cpuCandidates);
}

function cpuScreeningGameCount(sampleGameCount: number, cpuScreeningOpponentVariants: number): number {
  return trainingBinding.numericValue(
    "chronofish_cpu_screening_game_count",
    sampleGameCount,
    cpuScreeningOpponentVariants
  );
}

function cpuTrainingFinalistTarget(config: NormalizedTrainingConfig, populationLength: number, screenedLength: number): number {
  return trainingBinding.numericValue(
    "chronofish_cpu_training_finalist_target",
    populationLength,
    config.cpuFinalists,
    config.cpuPairBatch,
    screenedLength
  );
}

function cpuTrainingEliteCount(config: NormalizedTrainingConfig): number {
  return trainingBinding.numericValue("chronofish_cpu_training_elite_count", config.cpuFinalists);
}

function cpuTrainingCandidateImproved(candidateScore: number | null | undefined, baselineScore: number, bestCandidateScore: number | null | undefined): boolean {
  return trainingBinding.numericValue(
    "chronofish_cpu_training_candidate_improved",
    candidateScore ?? Number.NaN,
    baselineScore,
    bestCandidateScore ?? Number.NEGATIVE_INFINITY
  ) !== 0;
}

function cpuTrainingNextStagnation(generationsWithoutCandidate: number, improved: boolean): number {
  return trainingBinding.numericValue(
    "chronofish_cpu_training_next_stagnation",
    generationsWithoutCandidate,
    improved ? 1 : 0
  );
}

function cpuTrainingShouldContinue(deadlineAt: number, generationsWithoutCandidate: number, config: NormalizedTrainingConfig): boolean {
  return trainingBinding.numericValue(
    "chronofish_cpu_training_should_continue",
    performance.now(),
    deadlineAt,
    generationsWithoutCandidate,
    config.cpuMaxGenerationsWithoutCandidate
  ) !== 0;
}

function cpuCandidateScoringShouldContinue(deadlineAt: number, nextCandidate: number, uncachedCandidateCount: number): boolean {
  return trainingBinding.numericValue(
    "chronofish_cpu_candidate_scoring_should_continue",
    performance.now(),
    deadlineAt,
    nextCandidate,
    uncachedCandidateCount
  ) !== 0;
}

function cpuReferenceCollectionShouldContinue(deadlineAt: number, nextGame: number, gameCount: number): boolean {
  return trainingBinding.numericValue(
    "chronofish_cpu_reference_collection_should_continue",
    performance.now(),
    deadlineAt,
    nextGame,
    gameCount
  ) !== 0;
}

function cpuTrainingDeadlineAt(config: NormalizedTrainingConfig): number {
  const budgetMs = trainingBinding.numericValue(
    "chronofish_cpu_training_budget_ms",
    config.cpuTrainSeconds,
    config.cpuTrainingTimeMs,
    config.cpuMaxMatchPlies,
    config.cpuMaxMatchTimeMs
  );
  return performance.now() + budgetMs;
}

interface EngineCpuReferenceScore {
  score: number;
  moves?: Move[];
}

function cpuReferenceScoreFromResult(result: AiSearchResult | null | undefined): EngineCpuReferenceScore {
  return trainingBinding.jsonValue<EngineCpuReferenceScore>(
    "chronofish_cpu_reference_score_from_result_json",
    { result: result ?? null }
  );
}

interface CpuReferenceScoreDelta {
  score: number;
  nearDraw: boolean;
}

function cpuReferenceScoreDeltaFromResult(
  candidateResult: AiSearchResult | null | undefined,
  referenceScore: number,
  referenceMoves: Move[] | undefined,
  drawWindow: number
): CpuReferenceScoreDelta {
  return trainingBinding.jsonValue<CpuReferenceScoreDelta>(
    "chronofish_cpu_reference_score_delta_from_result_json",
    {
      candidateResult: candidateResult ?? null,
      referenceScore,
      referenceMoves: referenceMoves ?? [],
      drawWindow
    }
  );
}

function cpuReferenceCandidateAverage(score: number, compared: number, nearDraws: number, drawRateLimit: number): number {
  return trainingBinding.numericValue(
    "chronofish_cpu_reference_candidate_average",
    score,
    compared,
    nearDraws,
    drawRateLimit
  );
}

function cpuTrainingNoMoveScore(candidateTurn: boolean): number {
  return trainingBinding.numericValue("chronofish_cpu_training_no_move_score", candidateTurn ? 1 : 0);
}

function cpuTrainingCandidateTurn(currentTurn: Color, candidateColor: Color): boolean {
  return trainingBinding.jsonBooleanValue("chronofish_cpu_training_candidate_turn_json", {
    currentTurn,
    candidateColor
  });
}

function cpuTrainingWinnerScore(winner: Color | null, candidateColor: Color): number {
  return trainingBinding.jsonNumericValue("chronofish_cpu_training_winner_score_json", {
    winner,
    candidateColor
  });
}

function cpuTrainingAdjudicationScore(currentTurn: Color, candidateColor: Color, baselineScore: number): number {
  return trainingBinding.jsonNumericValue("chronofish_cpu_training_adjudication_score_json", {
    currentTurn,
    candidateColor,
    baselineScore
  });
}

function cpuTrainingAdjudicationScoreFromResult(
  currentTurn: Color,
  candidateColor: Color,
  result: AiSearchResult | null | undefined
): number {
  return trainingBinding.jsonNumericValue("chronofish_cpu_training_adjudication_score_from_result_json", {
    currentTurn,
    candidateColor,
    result: result ?? null
  });
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
  const filtered = compactTrainingSamples(samples);
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
        gpuMode: "hybrid",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("search", index, searchSeed(position.sample, config.runSeed ^ workerIndex ^ 0x51a7_0001))
      }, workerRequestTimeout(config));
      return searchResultLabelSampleFromResult(position.sample, response.result, "search", 1.0);
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
  return compactTrainingSamples(samples);

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
      return searchResultLabelSampleFromResult(position.sample, response.result, "cpu", cpuSearchLabelWeight(config));
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
  return compactTrainingSamples(samples);

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
        gpuMode: "hybrid",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed(String(labelKind), index, searchSeed(position.sample, config.runSeed ^ workerIndex ^ seedSalt))
      }, workerRequestTimeout(config));
      return searchResultLabelSampleFromResult(
        position.sample,
        response.result,
        labelKind,
        baseLabelWeight * labelWeightMultiplier(position)
      );
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
  const workerCount = gpuDuelTrainingWorkerCount(target, config.searchWorkers, config.selfPlayWorkers);
  const targets = splitWork(target, workerCount);
  let collected = 0;
  progress({ sampleCount: target, labelWorkers: workerCount, labelKind: "duel" });
  const results = await Promise.all(targets.map((count, workerIndex) =>
    collectCpuGpuDuelRollout(game, config, count, workerIndex, (count) => {
      collected += count;
      progress({ collected, sampleCount: target, labelWorkers: workerCount, labelKind: "duel" });
    })
  ));
  return takeTrainingSampleBatches(results, target);
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
    const cpuColor = workerIndex % 2 === 0 ? current.turn : engineOppositeColor(current.turn);
    const maxPlies = gpuRolloutMaxPlies(target, workerIndex);
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
        gpuMode: "hybrid",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("duel", ply, searchSeed(encoded, config.runSeed ^ workerIndex ^ 0xd0e1_0001))
      }, workerRequestTimeout({ nodes: useCpu ? config.cpuNodes : config.nodes }));
      const result = response.result;
      const moves = searchResultTurn(result).moves;
      if (!moves.length) {
        break;
      }
      const sample = searchResultLabelSampleFromResult(encoded, result, "duel", labelPolicy.duelLabelWeight);
      if (!sample) {
        break;
      }
      samples.push({
        ...sample,
        outcomeTurn: beforeTurn,
        ply: rolloutPlyOffset(ply, workerIndex)
      });
      progress(1);
      const applied = await applyWorkerTurn(current, moves, beforeTurn);
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
  return takeTrainingSampleBatches(results, target);
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
    const maxPlies = gpuRolloutMaxPlies(target, workerIndex);
    for (let ply = 0; ply < maxPlies && samples.length < target; ply += 1) {
      const beforeTurn = current.turn;
      const encoded = await encodePosition(encoder, current);
      const response = await requestWorker(ai, {
        type: "search",
        game: current,
        depth: config.depth,
        nodes: config.nodes,
        timeMs: workerSearchTimeMs(config),
        gpuMode: "hybrid",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("outcome", ply, searchSeed(encoded, config.runSeed ^ workerIndex ^ 0x0c70_0001))
      }, workerRequestTimeout(config));
      const result = response.result;
      const moves = searchResultTurn(result).moves;
      if (!moves.length) {
        break;
      }
      const sample = searchResultLabelSampleFromResult(encoded, result, "outcome", labelPolicy.outcomeLabelWeight);
      if (!sample) {
        break;
      }
      samples.push({
        ...sample,
        outcomeTurn: beforeTurn,
        ply: rolloutPlyOffset(ply, workerIndex)
      });
      progress(1);
      const applied = await applyWorkerTurn(current, moves, beforeTurn);
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
  return trainingBinding.splitWork(total, workers);
}

function takeTrainingSampleBatches(batches: TrainingSample[][], target: number): TrainingSample[] {
  return trainingBinding.takeSampleBatches<TrainingSample[]>(batches, target);
}

function compactTrainingSamples(samples: Array<TrainingSample | null>): TrainingSample[] {
  return trainingBinding.compactSamples<TrainingSample[]>(samples);
}

function gpuTrainingWorkerCount(total: number, requestedWorkers: number): number {
  return trainingBinding.gpuTrainingWorkerCount(total, requestedWorkers);
}

function gpuDuelTrainingWorkerCount(total: number, searchWorkers: number, selfPlayWorkers: number): number {
  return trainingBinding.gpuDuelTrainingWorkerCount(total, searchWorkers, selfPlayWorkers);
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
        gpuMode: "hybrid",
        partitionIndex: workerIndex,
        partitionCount: config.selfPlayWorkers ?? 1,
        temperature: warmupConfig.explorationTemperature,
        randomSeed: sampleSeed("warmup", ply, config.runSeed ^ workerIndex ^ 0x0aa5_0001)
      }, workerRequestTimeout({ ...config, nodes: 1024 }));
      const moves = searchResultTurn(response.result).moves;
      if (!moves.length) {
        break;
      }
      const applied = await applyWorkerTurn(current, moves, current.turn);
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
  return trainingBinding.gpuWarmupPlies(workerIndex);
}

function gpuRolloutMaxPlies(target: number, workerIndex: number): number {
  return trainingBinding.gpuRolloutMaxPlies(target, workerIndex);
}

function rolloutPlyOffset(ply: number, workerIndex: number): number {
  return trainingBinding.rolloutPlyOffset(ply, workerIndex);
}

function gpuWarmupSearchConfig(config: NormalizedTrainingConfig): TrainingWorkerSearchConfig {
  return trainingBinding.gpuWarmupSearchConfig<TrainingWorkerSearchConfig>(
    config.depth,
    config.nodes,
    workerSearchTimeMs(config),
    config.explorationTemperature
  );
}

function gpuPositionGenerationSearchConfig(config: NormalizedTrainingConfig): TrainingWorkerSearchConfig {
  return trainingBinding.gpuPositionGenerationSearchConfig<TrainingWorkerSearchConfig>(
    config.depth,
    config.nodes,
    config.explorationTemperature
  );
}

function curriculumSearchConfig(config: NormalizedTrainingConfig, index: number): NormalizedTrainingConfig {
  return {
    ...config,
    ...trainingBinding.curriculumSearchConfig<TrainingSearchConfig>(
      config.depth,
      config.nodes,
      config.explorationTemperature,
      index
    )
  };
}

function tacticalSearchConfig(config: NormalizedTrainingConfig, attempt: number): NormalizedTrainingConfig {
  return {
    ...config,
    ...trainingBinding.tacticalSearchConfig<TrainingSearchConfig>(
      config.depth,
      config.nodes,
      config.explorationTemperature,
      attempt
    )
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
  const labels = await predictValues(samples, activeModel, await engineInstance());
  return distillSamplesWithEngine(samples, labels);
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
      const samples = await trainingBinding.trainingSamples<TrainingSample[]>(
        local.map((entry) => entry.game)
      );
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
        gpuMode: "hybrid",
        temperature: searchConfig.explorationTemperature,
        randomSeed: sampleSeed("position", rolloutPlyOffset(ply, index), config.runSeed ^ workerIndex ^ 0x9051_0001)
      }, workerRequestTimeout(searchConfig));
      const moves = searchResultTurn(response.result).moves;
      if (!moves.length) {
        break;
      }
      const applied = await applyWorkerTurn(current, moves, current.turn);
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
  const attempts = tacticalPositionAttemptCount(index);
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const generated = await generatePositionGame(
      ai,
      tacticalPositionUseBestSource(bestPriority) ? best : game,
      tacticalSearchConfig(config, attempt),
      rolloutPlyOffset(index, attempt),
      workerIndex
    );
    const priority = tacticalPositionPriority(generated);
    const selection = tacticalPositionSelection(bestPriority, priority);
    if (selection.useGenerated) {
      best = generated;
      bestPriority = selection.nextPriority;
    }
    if (selection.complete) {
      break;
    }
  }
  return best;
}

function curriculumGame(game: GameSnapshot, index: number): GameSnapshot {
  return trainingBinding.jsonValue<GameSnapshot>("chronofish_curriculum_game_snapshot_json", game, index);
}

function tacticalPositionPriority(game: GameSnapshot): number {
  return trainingBinding.jsonNumericValue("chronofish_tactical_position_priority_snapshot_json", game);
}

function tacticalPositionAttemptCount(index: number): number {
  return trainingBinding.numericValue("chronofish_tactical_position_attempt_count", index);
}

function tacticalPositionUseBestSource(bestPriority: number): boolean {
  return trainingBinding.numericValue("chronofish_tactical_position_use_best_source", bestPriority) !== 0;
}

function tacticalPositionSelection(
  bestPriority: number,
  generatedPriority: number
): { useGenerated: boolean; nextPriority: number; complete: boolean } {
  return trainingBinding.resultValue("chronofish_tactical_position_selection_json", bestPriority, generatedPriority);
}

async function applyWorkerTurn(game: GameSnapshot, moves: Move[], _mover: Color): Promise<AppliedWorkerTurn | null> {
  try {
    const applied = trainingBinding.jsonValue<AppliedWorkerTurn>("chronofish_cpu_apply_turn_json", { game, moves });
    return {
      game: applied.game,
      status: applied.status,
      winner: applied.status.winner ?? null
    };
  } catch {
    return null;
  }
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
  return trainingBinding.asyncJsonValue<TrainingSample[]>("chronofish_relabel_outcome_samples_json", {
    ...request,
    samples: samplesForEngine(samples)
  });
}

async function distillSamplesWithEngine(
  samples: TrainingSample[],
  labels: number[]
): Promise<TrainingSample[]> {
  return trainingBinding.asyncJsonValue<TrainingSample[]>(
    "chronofish_distill_training_samples_with_labels_json",
    { samples, labels }
  );
}

function searchResultLabelSampleFromResult(
  sample: TrainingSample,
  result: AiSearchResult | null | undefined,
  labelKind: TrainingLabelKind,
  labelWeight: number
): TrainingSample | null {
  return trainingBinding.jsonValue<TrainingSample | null>("chronofish_search_result_label_sample_from_result_json", {
    sample,
    result: result ?? null,
    labelKind,
    labelWeight
  });
}

interface EngineSearchResultTurn {
  moves: Move[];
  score: number | null;
}

function searchResultTurn(result: AiSearchResult | null | undefined): EngineSearchResultTurn {
  return trainingBinding.jsonValue<EngineSearchResultTurn>(
    "chronofish_search_result_turn_json",
    { result: result ?? null }
  );
}

function cpuParametersKey(parameters: CpuParameters): string {
  return trainingBinding.jsonTextValue("chronofish_cpu_parameters_key_json", parameters);
}

function uniqueCpuParameters(values: CpuParameters[]): CpuParameters[] {
  return trainingBinding.jsonValue<CpuParameters[]>("chronofish_unique_cpu_parameters_json", values);
}

function breedCpuPopulation(
  baseline: CpuParameters,
  elites: CpuParameters[],
  target: number,
  seed: number,
  generation: number,
  stagnation: number
): CpuParameters[] {
  return trainingBinding.jsonValue<CpuParameters[]>("chronofish_breed_cpu_population_json", {
    baseline,
    elites,
    target,
    seed,
    generation,
    stagnation
  });
}

function rankCpuScoredCandidates(scored: Array<{ parameters: CpuParameters; score: number }>): Array<{ parameters: CpuParameters; score: number }> {
  return trainingBinding.jsonValue<Array<{ parameters: CpuParameters; score: number }>>(
    "chronofish_rank_cpu_scored_candidates_json",
    scored
  );
}

function cpuTrainingElites(
  candidates: Array<{ parameters: CpuParameters; score: number }>,
  baseline: CpuParameters,
  config: NormalizedTrainingConfig
): CpuParameters[] {
  return trainingBinding.jsonValue<CpuParameters[]>(
    "chronofish_cpu_training_elites_json",
    { baseline, candidates },
    config.cpuFinalists
  );
}

function cpuTrainingFinalistCandidates(
  baseline: CpuParameters,
  screened: Array<{ parameters: CpuParameters; score: number }>,
  target: number
): CpuParameters[] {
  return trainingBinding.jsonValue<CpuParameters[]>(
    "chronofish_cpu_training_finalist_candidates_json",
    { baseline, screened },
    target
  );
}

interface CpuTrainingGenerationOutcome {
  baselineScore: number;
  winner: { parameters: CpuParameters; score: number } | null;
  improved: boolean;
}

function cpuTrainingGenerationOutcome(
  baseline: CpuParameters,
  finalists: Array<{ parameters: CpuParameters; score: number }>,
  previousBaselineScore: number,
  bestCandidateScore: number | null | undefined
): CpuTrainingGenerationOutcome {
  return trainingBinding.jsonValue<CpuTrainingGenerationOutcome>("chronofish_cpu_training_generation_outcome_json", {
    baseline,
    finalists,
    previousBaselineScore,
    bestCandidateScore: bestCandidateScore ?? Number.NEGATIVE_INFINITY
  });
}

interface CpuCandidateScoringPlan {
  uniqueCandidates: CpuParameters[];
  cachedScores: Array<{ parameters: CpuParameters; score: number }>;
  uncachedCandidates: CpuParameters[];
  cacheHits: number;
}

interface CpuFitnessEntry {
  key: string;
  score: number;
}

function cpuCandidateScoringPlan(
  candidates: CpuParameters[],
  fitnessCache: Map<string, number>
): CpuCandidateScoringPlan {
  const fitness = Array.from(fitnessCache, ([key, score]) => ({ key, score }));
  return trainingBinding.jsonValue<CpuCandidateScoringPlan>(
    "chronofish_cpu_candidate_scoring_plan_json",
    { candidates, fitness }
  );
}

function cpuFitnessEntryForCandidate(parameters: CpuParameters, score: number): CpuFitnessEntry {
  return trainingBinding.jsonValue<CpuFitnessEntry>(
    "chronofish_cpu_fitness_entry_for_candidate_json",
    { parameters, score }
  );
}

function modeLabelTarget(config: NormalizedTrainingConfig, divisor: number): number {
  return trainingBinding.numericValue(
    "chronofish_mode_label_target",
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
  const policy = trainingBinding.jsonValue<TrainingModePolicy>("chronofish_training_mode_policy_json", {
    trainingSubject: config.trainingSubject,
    trainingModes: config.trainingModes,
    mode: mode ?? null
  });
  trainingModePolicyCache.set(key, policy);
  return policy;
}

function engineOppositeColor(color: Color): Color {
  return trainingBinding.textResultValue<Color>("chronofish_opposite_color_json", color);
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
  const workerCount = trainingLabelWorkerCount(jobs.length, config.labelWorkers);
  progress(0, jobs.length, workerCount);

  const samples: Array<TrainingSample | null> = new Array(jobs.length).fill(null);
  let nextJob = 0;
  let collected = 0;

  await Promise.all(Array.from({ length: workerCount }, () => runLabelWorker()));
  return compactTrainingSamples(samples);

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
  trainingLabelPolicyCache = trainingBinding.trainingLabelPolicy<TrainingLabelPolicy>();
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
  return trainingBinding.workerRequestTimeout(payload ?? {});
}

function workerSearchTimeMs(payload: { nodes?: unknown; timeMs?: unknown }): number {
  return trainingBinding.workerSearchTime(payload ?? {});
}

function trainingIoTimeoutMs(): number {
  return workerRequestTimeout({ nodes: 0, timeMs: 0 });
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
  return trainingBinding.samplePlies(index, encodeOnly);
}

function sampleSeed(prefix: string, index: number, salt: number): number {
  return trainingBinding.sampleSeed(prefix, index, salt);
}

function searchSeed(value: unknown, salt: number): number {
  return trainingBinding.searchSeed(value ?? null, salt);
}

function trainingLabelWorkerCount(jobCount: number, requestedWorkers?: number): number {
  return trainingBinding.labelWorkerCount(jobCount, requestedWorkers ?? -1, navigator.hardwareConcurrency ?? 4);
}

async function validateLossLogs(
  config: NormalizedTrainingConfig,
  progress?: ProgressCallback,
  candidateModel?: ArrayBuffer
): Promise<LossLogValidation> {
  const logs = await fetchLossLogs(config.lossLogReplay);
  let validation: LossLogValidation = {
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
      }, trainingIoTimeoutMs(), [candidateModel]);
    }
    for (const log of logs) {
      const decisions = Array.isArray(log.decisions) ? log.decisions : [];
      let logChanged = false;
      for (const decision of decisions) {
        const previousKey = engineMovePlanKey(decision.selectedMoves);
        if (!decision.game || !previousKey) {
          validation = lossLogValidationUpdate(validation, "skip");
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
            gpuMode: "hybrid",
            temperature: 0,
            randomSeed: sampleSeed("loss-log", validation.checked, config.runSeed ^ 0x1055_1000)
          }, workerRequestTimeout(config));
          const current = searchResultTurn(response.result);
          const currentMoves = current.moves;
          const currentKey = engineMovePlanKey(currentMoves);
          if (!currentKey) {
            validation = lossLogValidationUpdate(validation, "skip");
            continue;
          }
          if (currentKey !== previousKey) {
            logChanged = true;
            validation = lossLogValidationUpdate(validation, "changed", {
              logPath: log.logPath ?? null,
              ply: decision.ply ?? null,
              botColor: decision.botColor ?? null,
              previous: previousKey,
              current: currentKey,
              previousScore: decision.selectedScore ?? null,
              currentScore: current.score
            });
            break;
          }
          validation = lossLogValidationUpdate(validation, "unchanged");
        } catch {
          validation = lossLogValidationUpdate(validation, "skip");
        }
      }
      if (logChanged) {
        continue;
      }
    }
  } finally {
    ai.terminate();
  }
  return lossLogValidationUpdate(validation, "finalize");
}

async function fetchLossLogs(limit: number): Promise<LossLog[]> {
  if (limit <= 0) {
    return [];
  }
  try {
    const response = await withTimeout(
      fetch("/api/training/loss-logs"),
      trainingIoTimeoutMs(),
      "Timed out loading loss logs."
    );
    if (!response.ok) {
      return [];
    }
    const payload = await response.json() as { logs?: LossLog[] };
    return lossLogReplayLogs(payload.logs ?? [], limit);
  } catch {
    return [];
  }
}

function lossLogValidationUpdate(
  validation: LossLogValidation,
  event: "skip" | "unchanged" | "changed" | "finalize",
  example?: LossLogValidationExample
): LossLogValidation {
  return trainingBinding.lossLogValidationUpdate<LossLogValidation>(validation, event, example);
}

function lossLogReplayLogs(logs: LossLog[], limit: number): LossLog[] {
  return trainingBinding.lossLogReplayLogs<LossLog[]>(logs, limit);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function engineMovePlanKey(moves: Move[] | undefined): string {
  return trainingBinding.movePlanKey(moves);
}
