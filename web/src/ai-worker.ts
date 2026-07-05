import { GPU_CANDIDATE_STRIDE, GPU_SOURCE_STRIDE, GPU_TARGET_STRIDE, GPU_BOARD_STRIDE, GPU_MUTATION_BOARD_STRIDE, GPU_MUTATION_CHILD_STRIDE, GPU_MUTATION_STATUS_ROYAL_CAPTURE, GPU_MUTATION_STATUS_BRANCH_OK, GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE, GPU_TURN_STATUS_RECORD_STRIDE, GPU_FRONTIER_BOARD_OFFSET, GPU_FRONTIER_HEADER_BOARD_COUNT, GPU_FRONTIER_HEADER_HASH_HIGH, GPU_FRONTIER_HEADER_HASH_LOW } from "./ai-layout.js";
import { colorCode } from "./ai-snapshot.js";
import { GPU_TURN_STATUS_SHADER, GPU_MOVEGEN_SHADER, GPU_REPLY_SHADER, GPU_MUTATE_SHADER } from "./ai-shaders.js";
import { autotuneFrontier, frontierStateBytes, frontierStateStride, FrontierGpuPipeline } from "./ai-frontier.js";
import type { EncodedFrontierRoot, FrontierBufferSet, FrontierTuning } from "./ai-frontier.js";
import { FrontierNeuralEvaluator } from "./ai-frontier-neural.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import { readWasmBytes, readWasmString, writeWasmBytes, writeWasmString } from "./engine-io.js";
import { GPUBufferUsage, GPUMapMode } from "./ai-worker-types.js";
import { align4, clearComputePipelineCache, createComputePipelineChecked, requestHighLimitDevice, storageBuffer } from "./ai-gpu-device.js";
import type { ChronofishEngine, Color, GameSnapshot, Move, Position } from "./types.js";
import type { GpuCandidateInputs, GpuSnapshot } from "./ai-snapshot.js";
import type { GpuMode, GpuSearchOptions, LegalTargetSelection, MutatedCandidate, RankedCandidate, ReplySearchResult, ScoredCandidates, SearchChoice, SearchResult, TurnStatus, WorkerRequest } from "./ai-worker-types.js";

let cachedGpuAdapter: GPUAdapter | null = null;
let cachedGpuDevice: GPUDevice | null = null;
let frontierRuntime: { device: GPUDevice; pipeline: FrontierGpuPipeline; neural: FrontierNeuralEvaluator } | null = null;
let validationEnginePromise: Promise<ChronofishEngine> | null = null;
let activeSearchGeneration = 0;
let frontierModelOverride: ArrayBuffer | null = null;
const GPU_CANDIDATE_INPUT_HEADER_I32S = 7;

interface PendingBoardRef {
  timelineId: number;
  time: number;
}

interface GpuWorkerSearchConfig {
  requestedDepth: number;
  minimumDepth: number;
  searchTimeMs: number;
  deadlineDelayMs: number | null;
}

async function tryGpuSearch({
  depth,
  nodes,
  timeMs,
  gpuMode = "hybrid",
  disableNeural = false,
  snapshotOverride = null,
  sourceGame,
  temperature = 0,
  randomSeed = 0
}: GpuSearchOptions): Promise<SearchResult | null> {
  if (!navigator.gpu) {
    return null;
  }
  const requestedDepth = await engineSearchDepthAtLeastOne(depth);
  const snapshot = snapshotOverride;
  if (!snapshot) {
    return null;
  }

  const device = await getGpuDevice();
  if (!device) {
    return null;
  }
  const searchNodes = await engineGpuSearchNodes(nodes);
  const turnStatus = await turnStatusOnGpu(device, snapshot);
  const pendingBoards = await enginePendingPresentBoards(snapshot, snapshot.turn);
  if (gpuMode === "full") {
    try {
      return await tryGpuResidentFrontierSearch(device, snapshot, {
        requestedDepth,
        nodes: searchNodes,
        temperature,
        randomSeed,
        disableNeural,
        sourceGame
      });
    } catch (error) {
      console.warn("Full GPU search failed; falling back to hybrid GPU search.", error);
    }
  }
  const candidates = sourceGame
    ? await engineGpuCandidateInputs(sourceGame)
    : await engineGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (candidates.sourceCount === 0 || candidates.targetCount === 0) {
    return null;
  }
  const scored = await scoreCandidatesOnGpu(device, candidates, snapshot.turn);
  let ranked = await engineRankedCandidates(scored, {
    pendingBoards,
    requirePending: true,
    limit: await engineGpuSearchRankingLimit(nodes)
  });
  if (ranked.length === 0) {
    throw new Error(`GPU scoring produced no pending legal candidates (${await engineGpuScoringSummary(scored, pendingBoards)})`);
  }

  if (requestedDepth > 1) {
    const result = await searchSingleMoveRepliesOnGpu(device, snapshot, candidates, scored.records, ranked, {
      requestedDepth,
      nodes: searchNodes,
      temperature,
      randomSeed
    });
    return completeGpuResultTurn(device, snapshot, result, { nodes: searchNodes, temperature, randomSeed });
  }

  if (pendingBoards.length >= 1 && ranked.length > 0) {
    const mutated = await mutateRankedCandidatesOnGpu(device, candidates, scored.records, ranked);
    const supported = await supportedMutatedCandidates(mutated, { requireChildBoards: false });
    if (supported.length === 0) {
      throw new Error(`GPU mutation rejected ranked candidates (${await engineGpuMutationSummary(mutated)})`);
    }
    const selected = await selectSearchCandidate(
      supported,
      temperature,
      randomSeed
    );
    if (!selected) {
      throw new Error(`GPU mutation produced no selectable candidate (${await engineGpuMutationSummary(mutated)})`);
    }
    if (selected) {
      const result: SearchResult = {
        moves: [selected.move],
        score: selected.score,
        choices: selected.choices,
        principalVariation: [[selected.move]],
        depth: requestedDepth,
        nodes: ranked.length,
        status: "ok",
        gpu: true,
        gpuTerminal: selected.mutationStatus === GPU_MUTATION_STATUS_ROYAL_CAPTURE || selected.mutationStatus === GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE,
        gpuSnapshot: snapshot.format,
        gpuSearch: "single-present-gpu-mutated"
      };
      return completeGpuResultTurn(device, snapshot, result, { nodes: searchNodes, temperature, randomSeed });
    }
  }

  return null;
}

async function tryGpuResidentFrontierSearch(
  device: GPUDevice,
  snapshot: GpuSnapshot,
  { requestedDepth, nodes, temperature, randomSeed, disableNeural, sourceGame }: {
    requestedDepth: number;
    nodes: number;
    temperature: number;
    randomSeed: number;
    disableNeural: boolean;
    sourceGame?: GameSnapshot;
  }
): Promise<SearchResult> {
  const boardCount = snapshot.timelines.reduce((sum, timeline) => sum + timeline.boards.length, 0);
  const adapter = cachedGpuAdapter;
  if (!adapter) {
    throw new Error("GPU frontier search has no adapter for tuning.");
  }
  const maxCycles = await engineFrontierMaxCycles(requestedDepth, snapshot.timelines.length);
  const tuning = await autotuneFrontier(
    adapter,
    device,
    nodes,
    boardCount,
    "gpu-v1-cfnn-v3-policy-head",
    maxCycles * 2
  );
  const runtime = await frontierRuntimeFor(device, tuning);
  runtime.neural.beginSearch();
  const buffers = runtime.pipeline.pool.createSearchBuffers();
  const root = sourceGame
    ? await engineFrontierRoot(sourceGame, tuning.maxBoards)
    : await engineFrontierRootFromSnapshot(snapshot, tuning.maxBoards);
  const startedAt = performance.now();
  const initialize = device.createCommandEncoder();
  initialize.clearBuffer(buffers.states);
  initialize.clearBuffer(buffers.nextStates);
  initialize.clearBuffer(buffers.counters);
  device.queue.submit([initialize.finish()]);
  runtime.pipeline.uploadRoot(buffers, root);

  const rootColor = colorCode(snapshot.turn);
  const perParentLimit = await engineFrontierPerParentLimit(tuning.frontierWidth);
  let modelUsed = false;
  let cyclesCompleted = 0;
  let activeStateLimit = 1;
  try {
    for (let cycle = 0; cycle < maxCycles; cycle += 1) {
      if (cycle > 0 && cyclesCompleted >= requestedDepth && Date.now() >= gpuDeadlineAt) {
        break;
      }
      const encoder = device.createCommandEncoder();
      const validationScope = pushGpuValidationScope(device);
      await runtime.pipeline.encodeExpansionCycle(
        encoder,
        buffers,
        {
          rootColor,
          targetDepth: requestedDepth,
          cycleIndex: cycle,
          stateCount: activeStateLimit,
          perParentLimit
        },
        async (policyEncoder, policyBuffers, candidateCapacity) => {
          modelUsed = await runtime.neural.encodePolicyPrior(
            policyEncoder,
            policyBuffers.states,
            policyBuffers.candidates,
            candidateCapacity,
            GPU_CANDIDATE_STRIDE,
            activeStateLimit,
            frontierStateStride(tuning.maxBoards, runtime.pipeline.engine),
            GPU_FRONTIER_BOARD_OFFSET,
            tuning.maxBoards,
            tuning.neuralBatchSize,
            requestedDepth
          ) || modelUsed;
        }
      );
      activeStateLimit = await engineFrontierNextActiveStateLimit(tuning.frontierWidth, activeStateLimit, perParentLimit);
      if (!disableNeural) {
        modelUsed = await runtime.neural.encode(
          encoder,
          buffers.nextStates,
          buffers.summaries,
          activeStateLimit,
          frontierStateStride(tuning.maxBoards, runtime.pipeline.engine),
          GPU_FRONTIER_BOARD_OFFSET,
          tuning.maxBoards,
          rootColor,
          tuning.neuralBatchSize,
          requestedDepth
        ) || modelUsed;
      }
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await popGpuValidationScope(device, validationScope, `GPU frontier cycle ${cycle}`);
      runtime.pipeline.releaseCycleTemporaries();
      runtime.neural.releaseTemporaries();
      runtime.pipeline.swapFrontiers(buffers);
      runtime.neural.advancePolicyFeatures();
      cyclesCompleted += 1;
    }

    const reduction = device.createCommandEncoder();
    const reductionValidationScope = pushGpuValidationScope(device);
    await runtime.pipeline.encodeMinimax(reduction, buffers, requestedDepth);
    device.queue.submit([reduction.finish()]);
    await device.queue.onSubmittedWorkDone();
    await popGpuValidationScope(device, reductionValidationScope, "GPU frontier minimax reduction");
    runtime.pipeline.releaseCycleTemporaries();

    const readback = await readFrontierOnce(device, buffers, tuning);
    if (readback.candidateOverflow && readback.selectedCount === 0) {
      throw new Error("GPU frontier candidate capacity overflowed before completing search.");
    }
    const gpuSearch = modelUsed ? "neural-frontier" : "heuristic-frontier";
    const choices = await validatedFrontierChoices(snapshot, readback.states, tuning, requestedDepth, gpuSearch, sourceGame);
    const selected = await selectSearchCandidate(choices, temperature, randomSeed);
    if (!selected) {
      throw new Error("GPU frontier produced no authoritative legal turn.");
    }
    const latencyMs = performance.now() - startedAt;
    const neuralCache = runtime.neural.cacheStats();
    const networkRoles = runtime.neural.networkRoles();
    const quantization = await runtime.neural.quantizationStats();
    const policyChoiceAgreement = await engineChoiceAgreement(selected, selected.choices, [1, 5, 20]);
    return {
      ...selected,
      status: "ok",
      gpu: true,
      gpuMode: "full",
      gpuSnapshot: snapshot.format,
      gpuSearch,
      nodes: readback.nodes,
      gpuDiagnostics: {
        frontierWidth: tuning.frontierWidth,
        candidateCapacity: tuning.candidateCapacity,
        selectedCount: readback.selectedCount,
        maxBoards: tuning.maxBoards,
        dispatchCandidateLimit: tuning.dispatchCandidateLimit,
        cycles: cyclesCompleted,
        completedDepth: selected.depth ?? 0,
        nodes: readback.nodes,
        readbacks: 1,
        candidateOverflow: readback.candidateOverflow ? 1 : 0,
        tacticalCandidates: readback.tacticalCandidates,
        selectedTacticalCandidates: readback.selectedTacticalCandidates,
        candidateSelectionRate: gpuDiagnosticRate(readback.selectedCount, readback.nodes, runtime.pipeline.engine),
        tacticalSelectionRate: gpuDiagnosticRate(readback.selectedTacticalCandidates, readback.tacticalCandidates, runtime.pipeline.engine),
        effectiveBranchingFactor: gpuEffectiveBranchingFactor(readback.selectedCount, cyclesCompleted, runtime.pipeline.engine),
        searchController: "puct-frontier-graph",
        progressiveWideningLimit: perParentLimit,
        graphDeduplication: 1,
        nnCacheHits: neuralCache.hits,
        nnCacheMisses: neuralCache.misses,
        nnCacheStores: neuralCache.stores,
        nnCacheHitRate: neuralCache.hitRate,
        inferencePrecision: quantization.inferencePrecision ?? undefined,
        fastNetPolicyFormat: quantization.fastNetPolicy?.format,
        fastNetPolicyScale: quantization.fastNetPolicy?.scale,
        fastNetPolicyMaxAbsError: quantization.fastNetPolicy?.maxAbsError,
        fastNet: networkRoles.fastNet,
        bigNet: networkRoles.bigNet,
        legalChoiceCount: selected.choices.length,
        legalTacticalChoiceCount: selected.choices.filter((choice) => choice.tactical).length,
        topPolicyChoiceAgreement: policyChoiceAgreement[0] ?? 0,
        top5PolicyChoiceAgreement: policyChoiceAgreement[1] ?? 0,
        top20PolicyChoiceAgreement: policyChoiceAgreement[2] ?? 0,
        selectedMovePrunedRisk: selected.tactical ? 0 : 1,
        selectedMoveTactical: selected.tactical ? 1 : 0,
        model: modelUsed ? "neural" : "heuristic",
        latencyMs: gpuReportedLatencyMs(latencyMs, runtime.pipeline.engine),
        nodesPerSecond: gpuNodesPerSecond(readback.nodes, latencyMs, runtime.pipeline.engine),
        candidateWorkgroupSize: tuning.candidateWorkgroupSize,
        mutationTileSize: tuning.mutationTileSize,
        neuralBatchSize: tuning.neuralBatchSize
      },
      choices: selected.choices
    };
  } finally {
    runtime.pipeline.pool.releaseSearchBuffers(buffers);
    runtime.pipeline.releaseCycleTemporaries();
    runtime.neural.releaseTemporaries();
  }
}

async function frontierRuntimeFor(device: GPUDevice, tuning: FrontierTuning): Promise<{ device: GPUDevice; pipeline: FrontierGpuPipeline; neural: FrontierNeuralEvaluator }> {
  if (frontierRuntime?.device === device
    && frontierRuntime.pipeline.tuning.maxBoards === tuning.maxBoards
    && frontierRuntime.pipeline.tuning.frontierWidth === tuning.frontierWidth
    && frontierRuntime.pipeline.tuning.candidateCapacity === tuning.candidateCapacity) {
    return frontierRuntime;
  }
  frontierRuntime?.pipeline.destroy();
  frontierRuntime?.neural.destroy();
  const engine = await validationEngine();
  frontierRuntime = {
    device,
    pipeline: new FrontierGpuPipeline(device, tuning, undefined, engine),
    neural: new FrontierNeuralEvaluator(device, frontierModelOverride)
  };
  return frontierRuntime;
}

function pushGpuValidationScope(device: GPUDevice): boolean {
  try {
    device.pushErrorScope?.("validation");
    return true;
  } catch {
    return false;
  }
}

async function popGpuValidationScope(device: GPUDevice, scoped: boolean, label: string): Promise<void> {
  if (!scoped) {
    return;
  }
  const error = await device.popErrorScope?.();
  if (error) {
    throw new Error(`${label} validation failed: ${gpuErrorMessage(error)}`);
  }
}

function gpuErrorMessage(error: GPUError): string {
  return error.message || String(error);
}

async function readFrontierOnce(
  device: GPUDevice,
  buffers: FrontierBufferSet,
  tuning: FrontierTuning
): Promise<{
  states: Int32Array;
  nodes: number;
  selectedCount: number;
  candidateOverflow: boolean;
  tacticalCandidates: number;
  selectedTacticalCandidates: number;
}> {
  const stateByteLength = frontierStateBytes(tuning.maxBoards, frontierRuntime?.pipeline.engine) * tuning.frontierWidth;
  const counterByteLength = 24;
  const staging = device.createBuffer({
    size: align4(stateByteLength + counterByteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffers.states, 0, staging, 0, stateByteLength);
  encoder.copyBufferToBuffer(buffers.counters, 0, staging, stateByteLength, counterByteLength);
  device.queue.submit([encoder.finish()]);
  try {
    await staging.mapAsync(GPUMapMode.READ);
    const bytes = staging.getMappedRange();
    const statesCopy = bytes.slice(0, stateByteLength);
    const countersCopy = bytes.slice(stateByteLength, stateByteLength + counterByteLength);
    staging.unmap();
    staging.destroy();
    const states = new Int32Array(statesCopy);
    const counters = new Uint32Array(countersCopy);
    return {
      states,
      nodes: counters[3] ?? 0,
      selectedCount: counters[1] ?? 0,
      candidateOverflow: (counters[2] ?? 0) !== 0,
      tacticalCandidates: counters[4] ?? 0,
      selectedTacticalCandidates: counters[5] ?? 0
    };
  } catch (error) {
    staging.destroy();
    clearCachedGpuState();
    throw error;
  }
}

async function validatedFrontierChoices(
  snapshot: GpuSnapshot,
  states: Int32Array,
  tuning: FrontierTuning,
  requestedDepth: number,
  gpuSearch: string,
  sourceGame?: GameSnapshot
): Promise<SearchResult[]> {
  const choices: SearchResult[] = [];
  const seen = new Set<string>();
  const engine = await validationEngine();
  const candidates = engineFrontierChoicesFromWords(engine, states, tuning, requestedDepth, gpuSearch);
  for (const candidate of candidates) {
    const plan = candidate.moves ?? [];
    const moves = await validateFirstFrontierTurn(snapshot, plan, sourceGame);
    const key = await engineMovePlanKey(moves);
    if (!moves.length || seen.has(key)) {
      continue;
    }
    seen.add(key);
    choices.push({
      ...candidate,
      status: "ok",
      moves,
      principalVariation: [moves],
      gpu: true,
      gpuMode: "full",
      gpuSearch
    });
    if (choices.length >= 12) {
      break;
    }
  }
  return choices;
}

function engineFrontierChoicesFromWords(
  engine: ChronofishEngine,
  words: Int32Array,
  tuning: FrontierTuning,
  requestedDepth: number,
  gpuSearch: string
): SearchResult[] {
  const input = writeWasmBytes(engine, new Uint8Array(words.buffer, words.byteOffset, words.byteLength));
  const label = writeWasmString(engine, gpuSearch);
  try {
    const output = engine.chronofish_gpu_frontier_choices_json_bytes(
      input.ptr,
      input.len,
      tuning.maxBoards,
      tuning.frontierWidth,
      requestedDepth,
      label.ptr,
      label.len,
      12
    );
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as SearchResult[];
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
    engine.chronofish_dealloc(label.ptr, label.len);
  }
}

async function validateFirstFrontierTurn(snapshot: GpuSnapshot, plan: Move[], sourceGame?: GameSnapshot): Promise<Move[]> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, await validationGameSnapshot(snapshot, sourceGame));
  const moves: Move[] = [];
  for (const move of plan) {
    if (!engine.chronofish_apply_move(
      move.from.timelineId, move.from.time, move.from.x, move.from.y,
      move.to.timelineId, move.to.time, move.to.x, move.to.y
    )) {
      return [];
    }
    moves.push(move);
    if (engine.chronofish_submit_turn()) {
      return moves;
    }
  }
  return [];
}

async function validateSearchResultBeforePost(snapshot: GpuSnapshot, result: SearchResult, sourceGame?: GameSnapshot): Promise<SearchResult | null> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, await validationGameSnapshot(snapshot, sourceGame));
  for (let index = 0; index < result.moves.length; index += 1) {
    const move = result.moves[index]!;
    if (!engine.chronofish_apply_move(
      move.from.timelineId, move.from.time, move.from.x, move.from.y,
      move.to.timelineId, move.to.time, move.to.x, move.to.y
    )) {
      return null;
    }
    if (engine.chronofish_submit_turn()) {
      if (index !== result.moves.length - 1) {
        return null;
      }
      const replayed = JSON.parse(readWasmString(engine, engine.chronofish_snapshot_json())) as GameSnapshot;
      return {
        ...result,
        authoritativeReplay: true,
        terminal: replayed.result?.terminal === true,
        winner: replayed.result?.winner ?? undefined,
        resultReason: replayed.result?.reason,
        gpuTerminal: result.gpuTerminal === true || replayed.result?.reason === "royal-capture"
      };
    }
  }
  return null;
}

async function validationGameSnapshot(snapshot: GpuSnapshot, sourceGame?: GameSnapshot): Promise<GameSnapshot> {
  return sourceGame ?? await engineGpuSnapshotGame(snapshot);
}

async function validationEngine(): Promise<ChronofishEngine> {
  validationEnginePromise ??= instantiateChronofishWasm("./chronofish_engine.wasm")
    .then((instance) => instance.exports as unknown as ChronofishEngine);
  return validationEnginePromise;
}

function loadValidationSnapshot(engine: ChronofishEngine, game: GameSnapshot): void {
  const { ptr, len } = writeWasmString(engine, JSON.stringify(game));
  try {
    if (!engine.chronofish_load_snapshot_json(ptr, len)) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineGpuSnapshot(game: GameSnapshot): Promise<GpuSnapshot> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  return JSON.parse(readWasmString(engine, engine.chronofish_gpu_snapshot_json())) as GpuSnapshot;
}

async function engineGpuSnapshotGame(snapshot: GpuSnapshot): Promise<GameSnapshot> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(snapshot));
  try {
    const output = engine.chronofish_gpu_snapshot_game_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as GameSnapshot;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineSnapshotWithGpuChildBoards(
  snapshot: GpuSnapshot,
  childBoardRecords: Int32Array,
  mutationStatus: number,
  { move = null, advanceTurn = true }: { move?: Move | null; advanceTurn?: boolean } = {}
): Promise<GpuSnapshot> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    snapshot,
    childBoardRecords: Array.from(childBoardRecords),
    mutationStatus,
    move,
    advanceTurn
  }));
  try {
    const output = engine.chronofish_gpu_snapshot_child_boards_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as GpuSnapshot;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineFrontierMaxCycles(requestedDepth: number, timelineCount: number): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_frontier_max_cycles(requestedDepth, timelineCount);
}

async function engineFrontierPerParentLimit(frontierWidth: number): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_frontier_per_parent_limit(frontierWidth);
}

async function engineFrontierNextActiveStateLimit(frontierWidth: number, activeStateLimit: number, perParentLimit: number): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_frontier_next_active_state_limit(frontierWidth, activeStateLimit, perParentLimit);
}

async function engineSearchDepthAtLeastOne(depth: unknown): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_bot_search_depth_at_least_one(typeof depth === "number" ? depth : Number.NaN);
}

async function engineGpuWorkerSearchConfig(depth: unknown, minDepth: unknown, timeMs: unknown): Promise<GpuWorkerSearchConfig> {
  const engine = await validationEngine();
  const output = engine.chronofish_gpu_worker_search_config_json(
    typeof depth === "number" ? depth : Number.NaN,
    typeof minDepth === "number" ? minDepth : Number.NaN,
    typeof timeMs === "number" ? timeMs : Number.NaN
  );
  if (!output) {
    throw new Error(readWasmString(engine, engine.chronofish_last_message()));
  }
  return JSON.parse(readWasmString(engine, output)) as GpuWorkerSearchConfig;
}

async function engineGpuSearchRankingLimit(nodes: unknown): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_search_ranking_limit(typeof nodes === "number" ? nodes : Number.NaN);
}

async function engineGpuSearchReplyLimit(nodes: unknown): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_search_reply_limit(typeof nodes === "number" ? nodes : Number.NaN);
}

async function engineGpuSearchValidationLimit(nodes: unknown): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_search_validation_limit(typeof nodes === "number" ? nodes : Number.NaN);
}

async function engineGpuFullSearchReportedDepth(requestedDepth: number): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_full_search_reported_depth(requestedDepth);
}

function gpuDiagnosticRate(numerator: number, denominator: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_gpu_diagnostic_rate(numerator, denominator);
  }
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator <= 0) {
    return 0;
  }
  return Math.round((numerator / denominator) * 1000) / 1000;
}

function gpuEffectiveBranchingFactor(selectedCount: number, cyclesCompleted: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_gpu_effective_branching_factor(selectedCount, cyclesCompleted);
  }
  return cyclesCompleted > 0
    ? Math.round((selectedCount / cyclesCompleted) * 100) / 100
    : selectedCount;
}

function gpuReportedLatencyMs(latencyMs: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_gpu_reported_latency_ms(latencyMs);
  }
  return Number.isFinite(latencyMs) ? Math.max(0, Math.round(latencyMs)) : 0;
}

function gpuNodesPerSecond(nodes: number, latencyMs: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_gpu_nodes_per_second(nodes, latencyMs);
  }
  const boundedLatency = Number.isFinite(latencyMs) ? Math.max(0, latencyMs) : 0;
  return boundedLatency > 0 ? Math.round((nodes * 1000) / boundedLatency) : nodes;
}

async function engineGpuSearchNodes(nodes: unknown): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_search_nodes(typeof nodes === "number" ? nodes : Number.NaN);
}

function engineGpuMutationCandidateLimit(engine: ChronofishEngine, candidateCount: number): number {
  return engine.chronofish_gpu_mutation_candidate_limit(candidateCount);
}

function engineGpuMutationCandidateWorkgroups(engine: ChronofishEngine, candidateLimit: number): number {
  return engine.chronofish_gpu_mutation_candidate_workgroups(candidateLimit);
}

async function engineGpuTurnCompletionMaxMoves(existingMoves: number, timelineCount: number): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_turn_completion_max_moves(existingMoves, timelineCount);
}

function engineGpuCandidateMaxCandidatesPerBatch(engine: ChronofishEngine, maxBindingSize: number): number {
  return engine.chronofish_gpu_candidate_max_candidates_per_batch(maxBindingSize);
}

function engineGpuCandidateSourceBatchSize(engine: ChronofishEngine, maxCandidatesPerBatch: number, targetCount: number): number {
  return engine.chronofish_gpu_candidate_source_batch_size(maxCandidatesPerBatch, targetCount);
}

function engineGpuCandidateBatchSourceCount(engine: ChronofishEngine, sourceCount: number, sourceStart: number, sourceBatchSize: number): number {
  return engine.chronofish_gpu_candidate_batch_source_count(sourceCount, sourceStart, sourceBatchSize);
}

function engineGpuCandidateBatchCandidateCount(engine: ChronofishEngine, sourceCount: number, targetCount: number): number {
  return engine.chronofish_gpu_candidate_batch_candidate_count(sourceCount, targetCount);
}

function engineGpuCandidateScoreWorkgroups(engine: ChronofishEngine, batchCandidateCount: number): number {
  return engine.chronofish_gpu_candidate_score_workgroups(batchCandidateCount);
}

function engineGpuReplyScoreWorkgroupsX(engine: ChronofishEngine, rootCount: number): number {
  return engine.chronofish_gpu_reply_score_workgroups_x(rootCount);
}

function engineGpuReplyScoreWorkgroupsY(engine: ChronofishEngine, replyCount: number): number {
  return engine.chronofish_gpu_reply_score_workgroups_y(replyCount);
}

async function engineGpuCandidateInputs(game: GameSnapshot): Promise<GpuCandidateInputs> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  const bytes = readWasmBytes(engine, engine.chronofish_gpu_candidate_inputs_bytes());
  if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Engine GPU candidate input byte length is not i32-aligned.");
  }
  return candidateInputsFromWords(new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Int32Array.BYTES_PER_ELEMENT), engine);
}

async function engineGpuCandidateInputsFromSnapshot(snapshot: GpuSnapshot, color: Color): Promise<GpuCandidateInputs> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({ ...snapshot, turn: color }));
  try {
    const output = engine.chronofish_gpu_candidate_inputs_snapshot_bytes(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const bytes = readWasmBytes(engine, output);
    if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error("Engine GPU candidate input snapshot byte length is not i32-aligned.");
    }
    return candidateInputsFromWords(new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Int32Array.BYTES_PER_ELEMENT), engine);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function candidateInputsFromWords(words: Int32Array, engine?: ChronofishEngine): GpuCandidateInputs {
  if (words.length < GPU_CANDIDATE_INPUT_HEADER_I32S) {
    throw new Error("Engine GPU candidate inputs are truncated.");
  }
  const sourceCount = words[0] ?? 0;
  const targetCount = words[1] ?? 0;
  const boardCount = words[2] ?? 0;
  const sourceLength = words[3] ?? -1;
  const targetLength = words[4] ?? -1;
  const boardLength = words[5] ?? -1;
  const mutationBoardLength = words[6] ?? -1;
  const totalLength = GPU_CANDIDATE_INPUT_HEADER_I32S + sourceLength + targetLength + boardLength + mutationBoardLength;
  if (
    sourceCount < 0 || targetCount < 0 || boardCount < 0
    || sourceLength < 0 || targetLength < 0 || boardLength < 0 || mutationBoardLength < 0
    || totalLength !== words.length
  ) {
    throw new Error("Engine GPU candidate input header is invalid.");
  }
  let offset = GPU_CANDIDATE_INPUT_HEADER_I32S;
  const sources = words.slice(offset, offset + sourceLength);
  offset += sourceLength;
  const targets = words.slice(offset, offset + targetLength);
  offset += targetLength;
  const boards = words.slice(offset, offset + boardLength);
  offset += boardLength;
  const mutationBoards = words.slice(offset, offset + mutationBoardLength);
  const meta = engine ? candidateInputMetaFromEngine(words, engine) : {
    sourceMeta: candidateMetaFromRecords(sources, GPU_SOURCE_STRIDE),
    targetMeta: candidateMetaFromRecords(targets, GPU_TARGET_STRIDE)
  };
  return {
    sourceMeta: meta.sourceMeta,
    targetMeta: meta.targetMeta,
    sourceCount,
    targetCount,
    boardCount,
    sources,
    targets,
    boards,
    mutationBoards
  };
}

function candidateInputMetaFromEngine(words: Int32Array, engine: ChronofishEngine): { sourceMeta: Position[]; targetMeta: Position[] } {
  const input = writeWasmBytes(engine, new Uint8Array(words.buffer, words.byteOffset, words.byteLength));
  try {
    const output = engine.chronofish_gpu_candidate_input_meta_json_bytes(input.ptr, input.len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as { sourceMeta: Position[]; targetMeta: Position[] };
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

function candidateMetaFromRecords(records: Int32Array, stride: number): Position[] {
  const meta: Position[] = [];
  for (let offset = 0; offset + stride <= records.length; offset += stride) {
    meta.push({
      timelineId: records[offset + 2] ?? 0,
      time: records[offset + 3] ?? 0,
      x: records[offset + 4] ?? 0,
      y: records[offset + 5] ?? 0
    });
  }
  return meta;
}

async function engineFrontierRoot(game: GameSnapshot, maxBoards: number): Promise<EncodedFrontierRoot> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  const ptr = engine.chronofish_frontier_root_bytes(maxBoards);
  return readEngineFrontierRoot(engine, ptr);
}

async function engineFrontierRootFromSnapshot(snapshot: GpuSnapshot, maxBoards: number): Promise<EncodedFrontierRoot> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(snapshot));
  try {
    return readEngineFrontierRoot(engine, engine.chronofish_frontier_root_snapshot_bytes(ptr, len, maxBoards));
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function enginePendingPresentBoards(snapshot: GpuSnapshot, color: Color): Promise<PendingBoardRef[]> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({ ...snapshot, turn: color }));
  try {
    const output = engine.chronofish_gpu_pending_present_boards_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as PendingBoardRef[];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineTurnStatusRecords(snapshot: GpuSnapshot): Promise<Int32Array> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(snapshot));
  try {
    const output = engine.chronofish_gpu_turn_status_records_snapshot_bytes(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const bytes = readWasmBytes(engine, output);
    if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error("Engine turn-status record byte length is not i32-aligned.");
    }
    return new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Int32Array.BYTES_PER_ELEMENT).slice();
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineTurnStatusFromWords(words: Int32Array): Promise<TurnStatus> {
  const engine = await validationEngine();
  const byteLength = words.byteLength;
  const ptr = engine.chronofish_alloc(byteLength);
  new Uint8Array(engine.memory.buffer, ptr, byteLength).set(new Uint8Array(words.buffer, words.byteOffset, byteLength));
  try {
    const output = engine.chronofish_gpu_turn_status_json_bytes(ptr, byteLength);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as TurnStatus;
  } finally {
    engine.chronofish_dealloc(ptr, byteLength);
  }
}

async function engineRankedCandidates(
  scored: ScoredCandidates,
  { pendingBoards = [], requirePending = false, limit }: {
    pendingBoards?: PendingBoardRef[];
    requirePending?: boolean;
    limit: number;
  }
): Promise<RankedCandidate[]> {
  const candidateCount = scored.scores.length;
  const request = new Int32Array(
    4
    + candidateCount
    + candidateCount * GPU_CANDIDATE_STRIDE
    + pendingBoards.length * 2
  );
  request[0] = candidateCount;
  request[1] = Math.max(0, Math.floor(limit));
  request[2] = pendingBoards.length;
  request[3] = requirePending ? 1 : 0;
  request.set(scored.scores, 4);
  request.set(scored.records.subarray(0, candidateCount * GPU_CANDIDATE_STRIDE), 4 + candidateCount);
  let offset = 4 + candidateCount + candidateCount * GPU_CANDIDATE_STRIDE;
  for (const board of pendingBoards) {
    request[offset] = board.timelineId;
    request[offset + 1] = board.time;
    offset += 2;
  }
  const engine = await validationEngine();
  const byteLength = request.byteLength;
  const ptr = engine.chronofish_alloc(byteLength);
  new Uint8Array(engine.memory.buffer, ptr, byteLength).set(new Uint8Array(request.buffer, request.byteOffset, byteLength));
  try {
    const output = engine.chronofish_gpu_ranked_candidates_json_bytes(ptr, byteLength);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const ranked = JSON.parse(readWasmString(engine, output)) as RankedCandidate[];
    return Array.isArray(ranked) ? ranked : [];
  } finally {
    engine.chronofish_dealloc(ptr, byteLength);
  }
}

async function engineGpuScoringSummary(scored: ScoredCandidates, pendingBoards: PendingBoardRef[]): Promise<string> {
  const candidateCount = scored.scores.length;
  const request = new Int32Array(
    2
    + candidateCount
    + candidateCount * GPU_CANDIDATE_STRIDE
    + pendingBoards.length * 2
  );
  request[0] = candidateCount;
  request[1] = pendingBoards.length;
  request.set(scored.scores, 2);
  request.set(scored.records.subarray(0, candidateCount * GPU_CANDIDATE_STRIDE), 2 + candidateCount);
  let offset = 2 + candidateCount + candidateCount * GPU_CANDIDATE_STRIDE;
  for (const board of pendingBoards) {
    request[offset] = board.timelineId;
    request[offset + 1] = board.time;
    offset += 2;
  }
  const engine = await validationEngine();
  const byteLength = request.byteLength;
  const ptr = engine.chronofish_alloc(byteLength);
  new Uint8Array(engine.memory.buffer, ptr, byteLength).set(new Uint8Array(request.buffer, request.byteOffset, byteLength));
  try {
    const output = engine.chronofish_gpu_scoring_summary_bytes(ptr, byteLength);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmString(engine, output);
  } finally {
    engine.chronofish_dealloc(ptr, byteLength);
  }
}

async function engineGpuMutationSummary(mutated: MutatedCandidate[]): Promise<string> {
  const statuses = new Int32Array(mutated.map((entry) => entry.mutationStatus));
  const engine = await validationEngine();
  const byteLength = statuses.byteLength;
  const ptr = engine.chronofish_alloc(byteLength);
  new Uint8Array(engine.memory.buffer, ptr, byteLength).set(new Uint8Array(statuses.buffer, statuses.byteOffset, byteLength));
  try {
    const output = engine.chronofish_gpu_mutation_summary_bytes(ptr, byteLength);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmString(engine, output);
  } finally {
    engine.chronofish_dealloc(ptr, byteLength);
  }
}

async function engineGpuTurnCompletionKey(pendingBoards: PendingBoardRef[]): Promise<string> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(pendingBoards));
  try {
    const output = engine.chronofish_gpu_turn_completion_key_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmString(engine, output);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineChoiceAgreement(selected: SearchChoice, choices: SearchChoice[], limits: number[]): Promise<number[]> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    selected,
    choices,
    limits
  }));
  try {
    const output = engine.chronofish_gpu_choice_agreement_choices_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const response = JSON.parse(readWasmString(engine, output)) as { agreements?: number[] };
    return Array.isArray(response.agreements) ? response.agreements : [];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineSelectedSearchChoice<T extends SearchChoice>(request: unknown): Promise<(T & { choices: SearchChoice[] }) | null> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(request));
  try {
    const output = engine.chronofish_gpu_selected_choice_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as (T & { choices: SearchChoice[] }) | null;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineMovePlanKey(moves: Move[]): Promise<string> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(moves));
  try {
    const output = engine.chronofish_gpu_move_plan_key_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmString(engine, output);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineNonPostableResultSummary(result: unknown): Promise<string> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(result ?? null));
  try {
    const output = engine.chronofish_gpu_non_postable_result_summary_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmString(engine, output);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function enginePostableSearchResult(result: unknown): Promise<boolean> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(result ?? null));
  try {
    const output = engine.chronofish_gpu_postable_search_result_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmString(engine, output) === "true";
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineGpuSearchFailureSummary(snapshot: GpuSnapshot): Promise<string> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(snapshot));
  try {
    const output = engine.chronofish_gpu_search_failure_summary_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmString(engine, output);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function engineWithCompletedTurnChoice(
  result: SearchResult,
  moves: Move[],
  gpuSearch = result.gpuSearch,
  principalVariation?: Move[][]
): Promise<SearchResult> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    result,
    moves,
    gpuSearch,
    principalVariation
  }));
  try {
    const output = engine.chronofish_gpu_completed_turn_choice_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as SearchResult;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

async function enginePickCandidateRecords(records: Int32Array, indices: number[]): Promise<Int32Array> {
  const recordCount = Math.floor(records.length / GPU_CANDIDATE_STRIDE);
  const request = new Int32Array(2 + recordCount * GPU_CANDIDATE_STRIDE + indices.length);
  request[0] = recordCount;
  request[1] = indices.length;
  request.set(records.subarray(0, recordCount * GPU_CANDIDATE_STRIDE), 2);
  request.set(indices.map((index) => Math.trunc(index)), 2 + recordCount * GPU_CANDIDATE_STRIDE);
  const engine = await validationEngine();
  const byteLength = request.byteLength;
  const ptr = engine.chronofish_alloc(byteLength);
  new Uint8Array(engine.memory.buffer, ptr, byteLength).set(new Uint8Array(request.buffer, request.byteOffset, byteLength));
  try {
    const output = engine.chronofish_gpu_pick_candidate_records_bytes(ptr, byteLength);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const bytes = readWasmBytes(engine, output);
    if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error("Engine picked candidate record byte length is not i32-aligned.");
    }
    return new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Int32Array.BYTES_PER_ELEMENT).slice();
  } finally {
    engine.chronofish_dealloc(ptr, byteLength);
  }
}

async function engineCandidateIndex(scored: ScoredCandidates, move: Move): Promise<number> {
  const recordCount = Math.floor(scored.records.length / GPU_CANDIDATE_STRIDE);
  const request = new Int32Array(9 + recordCount * GPU_CANDIDATE_STRIDE);
  request[0] = recordCount;
  request.set([
    move.from.timelineId,
    move.from.time,
    move.from.x,
    move.from.y,
    move.to.timelineId,
    move.to.time,
    move.to.x,
    move.to.y
  ], 1);
  request.set(scored.records.subarray(0, recordCount * GPU_CANDIDATE_STRIDE), 9);
  const engine = await validationEngine();
  const byteLength = request.byteLength;
  const ptr = engine.chronofish_alloc(byteLength);
  new Uint8Array(engine.memory.buffer, ptr, byteLength).set(new Uint8Array(request.buffer, request.byteOffset, byteLength));
  try {
    const index = engine.chronofish_gpu_candidate_index_bytes(ptr, byteLength);
    if (index < -1) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return index;
  } finally {
    engine.chronofish_dealloc(ptr, byteLength);
  }
}

async function engineReplyPressureRankedRoots(
  rankedRoots: RankedCandidate[],
  pairScores: Int32Array,
  replyCount: number
): Promise<RankedCandidate[]> {
  const request = new Int32Array(2 + rankedRoots.length * 2 + rankedRoots.length * replyCount);
  request[0] = rankedRoots.length;
  request[1] = replyCount;
  let offset = 2;
  for (const root of rankedRoots) {
    request[offset] = root.index;
    offset += 1;
  }
  for (const root of rankedRoots) {
    request[offset] = root.score;
    offset += 1;
  }
  request.set(pairScores.subarray(0, rankedRoots.length * replyCount), offset);
  const engine = await validationEngine();
  const byteLength = request.byteLength;
  const ptr = engine.chronofish_alloc(byteLength);
  new Uint8Array(engine.memory.buffer, ptr, byteLength).set(new Uint8Array(request.buffer, request.byteOffset, byteLength));
  try {
    const output = engine.chronofish_gpu_reply_pressure_ranked_roots_bytes(ptr, byteLength);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const bytes = readWasmBytes(engine, output);
    if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error("Engine reply-pressure result byte length is not i32-aligned.");
    }
    const words = new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Int32Array.BYTES_PER_ELEMENT);
    const byIndex = new Map(rankedRoots.map((root) => [root.index, root]));
    const ranked: RankedCandidate[] = [];
    for (let index = 0; index + 1 < words.length; index += 2) {
      const root = byIndex.get(words[index] ?? -1);
      if (!root) {
        continue;
      }
      ranked.push({ ...root, score: words[index + 1] ?? root.score });
    }
    return ranked;
  } finally {
    engine.chronofish_dealloc(ptr, byteLength);
  }
}

function readEngineFrontierRoot(engine: ChronofishEngine, ptr: number): EncodedFrontierRoot {
  if (!ptr) {
    throw new Error(readWasmString(engine, engine.chronofish_last_message()));
  }
  const bytes = readWasmBytes(engine, ptr);
  if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Engine frontier root byte length is not i32-aligned.");
  }
  const words = new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Int32Array.BYTES_PER_ELEMENT).slice();
  return {
    words,
    boardCount: words[GPU_FRONTIER_HEADER_BOARD_COUNT] ?? 0,
    hashLow: words[GPU_FRONTIER_HEADER_HASH_LOW] ?? 0,
    hashHigh: words[GPU_FRONTIER_HEADER_HASH_HIGH] ?? 0
  };
}

async function engineLegalTargets(game: GameSnapshot, position: Position): Promise<LegalTargetSelection> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  return JSON.parse(readWasmString(engine, engine.chronofish_legal_selection_json(
    position.timelineId,
    position.time,
    position.x,
    position.y
  ))) as LegalTargetSelection;
}

async function engineApplyMove(game: GameSnapshot, move: Move): Promise<GameSnapshot> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  if (!engine.chronofish_apply_move(
    move.from.timelineId,
    move.from.time,
    move.from.x,
    move.from.y,
    move.to.timelineId,
    move.to.time,
    move.to.x,
    move.to.y
  )) {
    throw new Error(readWasmString(engine, engine.chronofish_last_message()));
  }
  return JSON.parse(readWasmString(engine, engine.chronofish_snapshot_json())) as GameSnapshot;
}

async function engineSubmitTurn(game: GameSnapshot): Promise<TurnStatus> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  return JSON.parse(readWasmString(engine, engine.chronofish_submit_turn_status_json())) as TurnStatus;
}

async function searchSingleMoveRepliesOnGpu(
  device: GPUDevice,
  snapshot: GpuSnapshot,
  inputs: GpuCandidateInputs,
  allCandidateRecords: Int32Array,
  ranked: RankedCandidate[],
  { requestedDepth, nodes, temperature = 0, randomSeed = 0 }: { requestedDepth: number; nodes: number; temperature?: number; randomSeed?: number }
): Promise<SearchResult | null> {
  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, allCandidateRecords, ranked, { readChildren: true });
  const candidates: SearchResult[] = [];
  for (const entry of await supportedMutatedCandidates(mutated)) {
    let score = entry.score;
    let principalVariation: Move[][] = [[entry.move]];
    if (entry.mutationStatus !== GPU_MUTATION_STATUS_ROYAL_CAPTURE && entry.mutationStatus !== GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
      const childSnapshot = await engineSnapshotWithGpuChildBoards(snapshot, entry.childBoards, entry.mutationStatus, { move: entry.move, advanceTurn: true });
      const reply = await bestReplyOnGpu(device, childSnapshot, nodes);
      if (reply.move) {
        score -= reply.score;
        principalVariation = [[entry.move], [reply.move]];
      }
    }
    const candidate: SearchResult = {
      moves: [entry.move],
      score,
      principalVariation,
      depth: await engineGpuFullSearchReportedDepth(requestedDepth),
      nodes: mutated.length,
      status: "ok",
      gpu: true,
      gpuSnapshot: snapshot.format,
      gpuSearch: "single-move-replies"
    };
    candidates.push(candidate);
  }
  return selectSearchCandidate(candidates, temperature, randomSeed);
}

async function completeGpuResultTurn(
  device: GPUDevice,
  snapshot: GpuSnapshot,
  result: SearchResult | null,
  { nodes, temperature = 0, randomSeed = 0 }: {
    nodes?: number | undefined;
    temperature?: number | undefined;
    randomSeed?: number | undefined;
  } = {}
): Promise<SearchResult | null> {
  if (!result?.moves?.length || result.gpuTerminal) {
    return result?.moves?.length ? await engineWithCompletedTurnChoice(result, result.moves, result.gpuSearch) : result;
  }
  const rootTurn = snapshot.turn;
  let current = snapshot;
  const moves: Move[] = [];
  let extraNodes = 0;
  const searchNodes = await engineGpuSearchNodes(nodes);
  for (const move of result.moves) {
    current = await applyGpuMoveToSnapshot(device, { ...current, turn: rootTurn }, move, { advanceTurn: true });
    moves.push(move);
    if (current.royalCaptureBy) {
      return {
        ...(await engineWithCompletedTurnChoice(result, moves, `${result.gpuSearch ?? "gpu"}-turn-complete`)),
        gpuTerminal: true
      };
    }
  }

  const maxMoves = await engineGpuTurnCompletionMaxMoves(moves.length, snapshot.timelines.length);
  while (moves.length < maxMoves) {
    const status = await turnStatusOnGpu(device, { ...current, turn: rootTurn });
    const pendingBoards = await enginePendingPresentBoards(current, rootTurn);
    if ((status.complete || status.pendingPresentBoardCount === 0) && pendingBoards.length === 0) {
      break;
    }
    const stepSnapshot = { ...current, turn: rootTurn };
    const inputs = await engineGpuCandidateInputsFromSnapshot(stepSnapshot, rootTurn);
    if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
      break;
    }
    const scored = await scoreCandidatesOnGpu(device, inputs, rootTurn);
    const ranked = await engineRankedCandidates(scored, {
      pendingBoards,
      requirePending: true,
      limit: await engineGpuSearchRankingLimit(nodes)
    });
    if (ranked.length === 0) {
      break;
    }
    const mutated = await mutateRankedCandidatesOnGpu(device, inputs, scored.records, ranked, { readChildren: true });
    extraNodes += mutated.length;
    const selected = await selectSearchCandidate(
      await supportedMutatedCandidates(mutated),
      temperature,
      randomSeed + moves.length
    );
    if (!selected) {
      break;
    }
    current = await engineSnapshotWithGpuChildBoards(stepSnapshot, selected.childBoards, selected.mutationStatus, {
      move: selected.move,
      advanceTurn: true
    });
    moves.push(selected.move);
    if (selected.mutationStatus === GPU_MUTATION_STATUS_ROYAL_CAPTURE || selected.mutationStatus === GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
      return {
        ...(await engineWithCompletedTurnChoice(result, moves, `${result.gpuSearch ?? "gpu"}-turn-complete`)),
        nodes: (result.nodes ?? 0) + extraNodes,
        gpuTerminal: true
      };
    }
  }

  const finalStatus = await turnStatusOnGpu(device, { ...current, turn: rootTurn });
  const finalPendingBoards = await enginePendingPresentBoards(current, rootTurn);
  if (finalPendingBoards.length > 0 || (!finalStatus.complete && finalStatus.pendingPresentBoardCount > 0)) {
    const fallback = await findCompleteGpuTurn(device, snapshot, rootTurn, searchNodes);
    if (fallback) {
      return engineWithCompletedTurnChoice({
        ...result,
        nodes: (result.nodes ?? 0) + extraNodes + fallback.nodes
      }, fallback.moves, `${result.gpuSearch ?? "gpu"}-turn-fallback`);
    }
    return {
      status: "incompleteTurn",
      moves: [],
      score: result.score,
      depth: result.depth,
      nodes: (result.nodes ?? 0) + extraNodes,
      gpu: true,
      gpuSnapshot: result.gpuSnapshot,
      gpuSearch: `${result.gpuSearch ?? "gpu"}-turn-incomplete`,
      incompleteMoves: moves,
      pendingPresentBoardCount: Math.max(finalStatus.pendingPresentBoardCount ?? 0, finalPendingBoards.length),
      choices: []
    };
  }

  const principalVariation = [moves];
  if ((result.depth ?? 1) > 1) {
    const reply = await completedGpuReplyTurn(device, current, {
      nodes,
      temperature,
      randomSeed: randomSeed + moves.length
    });
    if (reply.length > 0) {
      principalVariation.push(reply);
    }
  }

  return engineWithCompletedTurnChoice({
    ...result,
    nodes: (result.nodes ?? 0) + extraNodes
  }, moves, moves.length > result.moves.length ? `${result.gpuSearch ?? "gpu"}-turn-complete` : result.gpuSearch, principalVariation);
}

async function completedGpuReplyTurn(
  device: GPUDevice,
  snapshot: GpuSnapshot,
  { nodes, temperature = 0, randomSeed = 0 }: {
    nodes?: number | undefined;
    temperature?: number | undefined;
    randomSeed?: number | undefined;
  }
): Promise<Move[]> {
  if (snapshot.royalCaptureBy || Date.now() >= gpuDeadlineAt) {
    return [];
  }
  const searchNodes = await engineGpuSearchNodes(nodes);
  const reply = await bestReplyOnGpu(device, snapshot, searchNodes);
  if (!reply.move) {
    return [];
  }
  const completed = await completeGpuResultTurn(device, snapshot, {
    status: "ok",
    moves: [reply.move],
    score: reply.score,
    depth: 1,
    gpu: true,
    gpuSnapshot: snapshot.format,
    gpuSearch: "projected-reply"
  }, { nodes, temperature, randomSeed });
  return completed?.status === "ok" ? completed.moves : [];
}

async function findCompleteGpuTurn(
  device: GPUDevice,
  snapshot: GpuSnapshot,
  rootTurn: Color,
  nodes: number,
  moves: Move[] = [],
  visited = new Set<string>()
): Promise<{ moves: Move[]; nodes: number } | null> {
  if (snapshot.royalCaptureBy) {
    return { moves, nodes: 0 };
  }
  const pendingBoards = await enginePendingPresentBoards(snapshot, rootTurn);
  const status = await turnStatusOnGpu(device, { ...snapshot, turn: rootTurn });
  if (pendingBoards.length === 0 && (status.complete || status.pendingPresentBoardCount === 0)) {
    return { moves, nodes: 0 };
  }
  const maxMoves = await engineGpuTurnCompletionMaxMoves(1, snapshot.timelines.length);
  if (moves.length >= maxMoves) {
    return null;
  }

  const stateKey = await engineGpuTurnCompletionKey(pendingBoards);
  if (visited.has(stateKey)) {
    return null;
  }
  const nextVisited = new Set(visited);
  nextVisited.add(stateKey);

  const stepSnapshot = { ...snapshot, turn: rootTurn };
  const inputs = await engineGpuCandidateInputsFromSnapshot(stepSnapshot, rootTurn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    return null;
  }
  const scored = await scoreCandidatesOnGpu(device, inputs, rootTurn);
  const ranked = await engineRankedCandidates(scored, {
    pendingBoards,
    requirePending: true,
    limit: await engineGpuSearchReplyLimit(nodes)
  });
  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, scored.records, ranked, { readChildren: true });
  const supported = await supportedMutatedCandidates(mutated);
  for (const candidate of supported) {
    const child = await engineSnapshotWithGpuChildBoards(stepSnapshot, candidate.childBoards, candidate.mutationStatus, {
      move: candidate.move,
      advanceTurn: true
    });
    const completed = await findCompleteGpuTurn(
      device,
      child,
      rootTurn,
      nodes,
      [...moves, candidate.move],
      nextVisited
    );
    if (completed) {
      return { moves: completed.moves, nodes: supported.length + completed.nodes };
    }
  }
  return null;
}

async function applyGpuMoveToSnapshot(
  device: GPUDevice,
  snapshot: GpuSnapshot,
  move: Move,
  { advanceTurn = false }: { advanceTurn?: boolean } = {}
): Promise<GpuSnapshot> {
  const inputs = await engineGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    throw new Error("No GPU move candidates are available.");
  }
  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  const index = await engineCandidateIndex(scored, move);
  if (index < 0 || (scored.scores[index] ?? -2147483647) <= -2147480000) {
    throw new Error("GPU rejected that move.");
  }
  const candidateRecords = await enginePickCandidateRecords(scored.records, [index]);
  const ranked = [{ move, index: 0, score: scored.scores[index] ?? 0 }];
  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, candidateRecords, ranked, { readChildren: true });
  const [selected] = await supportedMutatedCandidates(mutated, { limit: 1 });
  if (!selected) {
    throw new Error("GPU move mutation is unsupported for that move.");
  }
  return await engineSnapshotWithGpuChildBoards(snapshot, selected.childBoards, selected.mutationStatus, { move, advanceTurn });
}

async function turnStatusOnGpu(device: GPUDevice, snapshot: GpuSnapshot): Promise<TurnStatus> {
  const boardRecords = await engineTurnStatusRecords(snapshot);
  const boardBuffer = storageBuffer(device, boardRecords, GPUBufferUsage.STORAGE);
  const resultBuffer = device.createBuffer({
    size: align4(4 * Int32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, boardRecords.length / GPU_TURN_STATUS_RECORD_STRIDE, true);
  view.setInt32(4, colorCode(snapshot.turn), true);
  const paramsBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
  const pipeline = await createComputePipelineChecked(device, "turn_status", GPU_TURN_STATUS_SHADER, "turn_status");
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [boardBuffer, resultBuffer, paramsBuffer]
      .map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const result = await readInts(device, resultBuffer, 4 * Int32Array.BYTES_PER_ELEMENT);
  return engineTurnStatusFromWords(result);
}

async function tryFullGpuSearch(
  device: GPUDevice,
  snapshot: GpuSnapshot,
  inputs: GpuCandidateInputs,
  { requestedDepth, nodes, turnStatus, temperature = 0, randomSeed = 0 }: { requestedDepth: number; nodes: number; turnStatus: TurnStatus; temperature?: number; randomSeed?: number }
): Promise<SearchResult> {
  if (turnStatus.pendingPresentBoardCount !== 1) {
    throw new Error("Full GPU search currently requires one pending present board.");
  }

  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  const ranked = await engineRankedCandidates(scored, {
    requirePending: false,
    limit: await engineGpuSearchRankingLimit(nodes)
  });
  if (ranked.length === 0) {
    throw new Error("Full GPU search found no candidate moves.");
  }

  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, scored.records, ranked, { readChildren: true });
  const supported = await supportedMutatedCandidates(mutated);
  if (supported.length === 0) {
    throw new Error("Full GPU mutation produced no supported child states.");
  }

  const candidates: SearchResult[] = [];
  for (const entry of await supportedMutatedCandidates(mutated, { limit: await engineGpuSearchValidationLimit(nodes) })) {
    let score = entry.score;
    let principalVariation: Move[][] = [[entry.move]];
    if (requestedDepth > 1 && entry.mutationStatus !== GPU_MUTATION_STATUS_ROYAL_CAPTURE && entry.mutationStatus !== GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
      const childSnapshot = await engineSnapshotWithGpuChildBoards(snapshot, entry.childBoards, entry.mutationStatus, { move: entry.move, advanceTurn: true });
      const reply = await bestReplyOnGpu(device, childSnapshot, nodes);
      score -= reply.score;
      if (reply.move) {
        principalVariation = [[entry.move], [reply.move]];
      }
    }
    const candidate: SearchResult = {
      moves: [entry.move],
      score,
      principalVariation,
      depth: await engineGpuFullSearchReportedDepth(requestedDepth),
      nodes: supported.length,
      status: "ok",
      gpu: true,
      gpuMode: "full",
      gpuTerminal: entry.mutationStatus === GPU_MUTATION_STATUS_ROYAL_CAPTURE || entry.mutationStatus === GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE,
      gpuSnapshot: snapshot.format,
      gpuSearch: "full-single-present"
    };
    candidates.push(candidate);
  }
  const selected = await selectSearchCandidate(candidates, temperature, randomSeed);
  if (!selected) {
    throw new Error("Full GPU search produced no legal result.");
  }
  return selected;
}

async function bestReplyOnGpu(device: GPUDevice, snapshot: GpuSnapshot, nodes: number): Promise<ReplySearchResult> {
  const inputs = await engineGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    return { score: 0 };
  }
  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  const pendingBoards = await enginePendingPresentBoards(snapshot, snapshot.turn);
  const [best] = await engineRankedCandidates(scored, {
    pendingBoards,
    requirePending: true,
    limit: 1
  });
  return best ? { score: best.score, move: best.move } : { score: 0 };
}

async function selectSearchCandidate<T extends SearchChoice>(candidates: T[], temperature = 0, randomSeed = 0): Promise<(T & { choices: SearchChoice[] }) | null> {
  return engineSelectedSearchChoice<T>({
    temperature: Number.isFinite(Number(temperature)) ? Number(temperature) : 0,
    randomSeed: Math.trunc(Number(randomSeed) || 0),
    candidates
  });
}

async function supportedMutatedCandidates(
  candidates: MutatedCandidate[],
  options?: { limit?: number; requireChildBoards?: true }
): Promise<Array<MutatedCandidate & { childBoards: Int32Array }>>;
async function supportedMutatedCandidates(
  candidates: MutatedCandidate[],
  options: { limit?: number; requireChildBoards: false }
): Promise<MutatedCandidate[]>;
async function supportedMutatedCandidates(
  candidates: MutatedCandidate[],
  options: { limit?: number; requireChildBoards?: boolean } = {}
): Promise<MutatedCandidate[]> {
  if (!candidates.length) {
    return [];
  }
  const requireChildBoards = options.requireChildBoards !== false;
  const request = new Int32Array(3 + candidates.length * 2);
  request[0] = candidates.length;
  request[1] = Math.max(0, Math.floor(options.limit ?? 0));
  request[2] = requireChildBoards ? 1 : 0;
  candidates.forEach((candidate, index) => {
    const offset = 3 + index * 2;
    request[offset] = candidate.mutationStatus;
    request[offset + 1] = candidate.childBoards ? 1 : 0;
  });
  const engine = await validationEngine();
  const byteLength = request.byteLength;
  const ptr = engine.chronofish_alloc(byteLength);
  new Uint8Array(engine.memory.buffer, ptr, byteLength).set(new Uint8Array(request.buffer, request.byteOffset, byteLength));
  try {
    const output = engine.chronofish_gpu_supported_mutation_candidate_indexes_bytes(ptr, byteLength);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const bytes = readWasmBytes(engine, output);
    if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error("Engine supported mutation index byte length is not i32-aligned.");
    }
    const indexes = new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Int32Array.BYTES_PER_ELEMENT);
    const supported: MutatedCandidate[] = [];
    for (const index of indexes) {
      const candidate = candidates[index];
      if (candidate && (!requireChildBoards || candidate.childBoards)) {
        supported.push(candidate);
      }
    }
    return supported;
  } finally {
    engine.chronofish_dealloc(ptr, byteLength);
  }
}

let gpuDeadlineAt = 0;

async function scoreCandidatesOnGpu(device: GPUDevice, inputs: GpuCandidateInputs, turn: Color): Promise<ScoredCandidates> {
  const engine = await validationEngine();
  const candidateCount = inputs.sourceCount * inputs.targetCount;
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const maxCandidatesPerBatch = engineGpuCandidateMaxCandidatesPerBatch(engine, maxBindingSize);
  if (inputs.targetCount > maxCandidatesPerBatch) {
    throw new Error(`GPU move generation target set is too large for this device (${inputs.targetCount} targets).`);
  }
  const targetBuffer = storageBuffer(device, inputs.targets, GPUBufferUsage.STORAGE);
  const boardBuffer = storageBuffer(device, inputs.boards ?? new Int32Array(GPU_BOARD_STRIDE), GPUBufferUsage.STORAGE);
  const pipeline = await createComputePipelineChecked(device, "score_candidates", GPU_MOVEGEN_SHADER, "score_candidates");
  const records = new Int32Array(candidateCount * GPU_CANDIDATE_STRIDE);
  const scores = new Int32Array(candidateCount);
  const sourceBatchSize = engineGpuCandidateSourceBatchSize(engine, maxCandidatesPerBatch, inputs.targetCount);

  for (let sourceStart = 0; sourceStart < inputs.sourceCount; sourceStart += sourceBatchSize) {
    const sourceCount = engineGpuCandidateBatchSourceCount(engine, inputs.sourceCount, sourceStart, sourceBatchSize);
    const batchCandidateCount = engineGpuCandidateBatchCandidateCount(engine, sourceCount, inputs.targetCount);
    const sourceBuffer = storageBuffer(
      device,
      inputs.sources.subarray(sourceStart * GPU_SOURCE_STRIDE, (sourceStart + sourceCount) * GPU_SOURCE_STRIDE),
      GPUBufferUsage.STORAGE
    );
    const candidateBuffer = device.createBuffer({
      size: align4(batchCandidateCount * GPU_CANDIDATE_STRIDE * Int32Array.BYTES_PER_ELEMENT),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const scoreBuffer = device.createBuffer({
      size: align4(batchCandidateCount * Int32Array.BYTES_PER_ELEMENT),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const params = new ArrayBuffer(16);
    const view = new DataView(params);
    view.setUint32(0, sourceCount, true);
    view.setUint32(4, inputs.targetCount, true);
    view.setUint32(8, colorCode(turn), true);
    view.setUint32(12, inputs.boardCount ?? 0, true);
    const paramsBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
    const encoder = device.createCommandEncoder();
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [sourceBuffer, targetBuffer, candidateBuffer, scoreBuffer, paramsBuffer, boardBuffer]
        .map((buffer, binding) => ({ binding, resource: { buffer } }))
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(engineGpuCandidateScoreWorkgroups(engine, batchCandidateCount));
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const batchRecords = await readInts(device, candidateBuffer, batchCandidateCount * GPU_CANDIDATE_STRIDE * Int32Array.BYTES_PER_ELEMENT);
    const batchScores = await readInts(device, scoreBuffer, batchCandidateCount * Int32Array.BYTES_PER_ELEMENT);
    const candidateOffset = sourceStart * inputs.targetCount;
    records.set(batchRecords, candidateOffset * GPU_CANDIDATE_STRIDE);
    scores.set(batchScores, candidateOffset);
    sourceBuffer.destroy();
    candidateBuffer.destroy();
    scoreBuffer.destroy();
    paramsBuffer.destroy();
  }

  targetBuffer.destroy();
  boardBuffer.destroy();
  return { records, scores };
}

async function mutateRankedCandidatesOnGpu(
  device: GPUDevice,
  inputs: GpuCandidateInputs,
  allCandidateRecords: Int32Array,
  ranked: RankedCandidate[],
  { readChildren = false }: { readChildren?: boolean } = {}
): Promise<MutatedCandidate[]> {
  const engine = await validationEngine();
  const limit = engineGpuMutationCandidateLimit(engine, ranked.length);
  if (limit === 0 || !inputs.mutationBoards || inputs.boardCount === 0) {
    return [];
  }
  const selected = ranked.slice(0, limit);
  const candidateRecords = await enginePickCandidateRecords(allCandidateRecords, selected.map((entry) => entry.index));
  const candidateBuffer = storageBuffer(device, candidateRecords, GPUBufferUsage.STORAGE);
  const boardBuffer = storageBuffer(device, inputs.mutationBoards, GPUBufferUsage.STORAGE);
  const childBoardBuffer = device.createBuffer({
    size: align4(limit * GPU_MUTATION_CHILD_STRIDE * Int32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const statusBuffer = device.createBuffer({
    size: align4(limit * Int32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, limit, true);
  view.setUint32(4, inputs.boardCount, true);
  view.setUint32(8, candidateRecords[1] ?? 0, true);
  const paramsBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
  const pipeline = await createComputePipelineChecked(device, "mutate_candidates", GPU_MUTATE_SHADER, "mutate_candidates");
  const encoder = device.createCommandEncoder();
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [candidateBuffer, boardBuffer, childBoardBuffer, statusBuffer, paramsBuffer]
      .map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(engineGpuMutationCandidateWorkgroups(engine, limit));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const statuses = await readInts(device, statusBuffer, limit * Int32Array.BYTES_PER_ELEMENT);
  const childBoards = readChildren
    ? await readInts(device, childBoardBuffer, limit * GPU_MUTATION_CHILD_STRIDE * Int32Array.BYTES_PER_ELEMENT)
    : null;
  candidateBuffer.destroy();
  boardBuffer.destroy();
  childBoardBuffer.destroy();
  statusBuffer.destroy();
  paramsBuffer.destroy();
  return selected.map((entry, index) => ({
    ...entry,
    mutationStatus: statuses[index] ?? 0,
    childBoards: childBoards?.subarray(index * GPU_MUTATION_CHILD_STRIDE, (index + 1) * GPU_MUTATION_CHILD_STRIDE) ?? null
  }));
}

async function scoreRootCandidatesWithReplies(
  device: GPUDevice,
  allRootRecords: Int32Array,
  rankedRoots: RankedCandidate[],
  allRootScores: Int32Array,
  allReplyRecords: Int32Array,
  allReplyScores: Int32Array
): Promise<RankedCandidate[]> {
  const engine = await validationEngine();
  const replyLimit = 512;
  const rankedReplies = await engineRankedCandidates(
    { records: allReplyRecords, scores: allReplyScores },
    { requirePending: false, limit: replyLimit }
  );
  if (rankedReplies.length === 0) {
    return rankedRoots;
  }

  const rootRecords = await enginePickCandidateRecords(allRootRecords, rankedRoots.map((entry) => entry.index));
  const replyRecords = await enginePickCandidateRecords(allReplyRecords, rankedReplies.map((entry) => entry.index));
  const rootScores = new Int32Array(rankedRoots.map((entry) => allRootScores[entry.index] ?? -2147483647));
  const replyScores = new Int32Array(rankedReplies.map((entry) => allReplyScores[entry.index] ?? -2147483647));
  const pairCount = rankedRoots.length * rankedReplies.length;
  const rootBuffer = storageBuffer(device, rootRecords, GPUBufferUsage.STORAGE);
  const replyBuffer = storageBuffer(device, replyRecords, GPUBufferUsage.STORAGE);
  const rootScoreBuffer = storageBuffer(device, rootScores, GPUBufferUsage.STORAGE);
  const replyScoreBuffer = storageBuffer(device, replyScores, GPUBufferUsage.STORAGE);
  const pairBuffer = device.createBuffer({
    size: align4(pairCount * Int32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, rankedRoots.length, true);
  view.setUint32(4, rankedReplies.length, true);
  const paramsBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
  const pipeline = await createComputePipelineChecked(device, "score_replies", GPU_REPLY_SHADER, "score_replies");
  const encoder = device.createCommandEncoder();
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [rootBuffer, replyBuffer, rootScoreBuffer, replyScoreBuffer, pairBuffer, paramsBuffer]
      .map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    engineGpuReplyScoreWorkgroupsX(engine, rankedRoots.length),
    engineGpuReplyScoreWorkgroupsY(engine, rankedReplies.length)
  );
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const pairScores = await readInts(device, pairBuffer, pairCount * Int32Array.BYTES_PER_ELEMENT);

  return engineReplyPressureRankedRoots(rankedRoots, pairScores, rankedReplies.length);
}

async function getGpuDevice(): Promise<GPUDevice | null> {
  if (!navigator.gpu) {
    return null;
  }
  if (cachedGpuDevice) {
    return cachedGpuDevice;
  }
  cachedGpuAdapter = cachedGpuAdapter ?? await navigator.gpu.requestAdapter();
  if (!cachedGpuAdapter) {
    return null;
  }
  cachedGpuDevice = await requestHighLimitDevice(cachedGpuAdapter);
  cachedGpuDevice.lost?.then(() => {
    clearCachedGpuState();
  });
  return cachedGpuDevice;
}

function clearCachedGpuState(): void {
  frontierRuntime?.pipeline.destroy();
  frontierRuntime?.neural.destroy();
  frontierRuntime = null;
  cachedGpuDevice = null;
  cachedGpuAdapter = null;
  clearComputePipelineCache();
}

async function destroyCachedGpuDeviceForSmoke(): Promise<boolean> {
  const device = cachedGpuDevice;
  if (!device) {
    return false;
  }
  device.destroy();
  try {
    await device.lost;
  } catch {
    // Browser implementations should resolve GPUDevice.lost, but the smoke
    // path still needs to force the same cleanup if an implementation differs.
  }
  clearCachedGpuState();
  return true;
}

async function readInts(device: GPUDevice, buffer: GPUBuffer, byteLength: number): Promise<Int32Array> {
  const readBuffer = device.createBuffer({
    size: align4(byteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  try {
    await readBuffer.mapAsync(GPUMapMode.READ);
    const bytes = readBuffer.getMappedRange().slice(0);
    readBuffer.unmap();
    readBuffer.destroy();
    return new Int32Array(bytes);
  } catch (error) {
    readBuffer.destroy();
    clearCachedGpuState();
    throw error;
  }
}

self.addEventListener("message", async (event: MessageEvent<WorkerRequest>) => {
  // id is echoed back so the main thread can discard stale search results.
  const {
    id,
    type = "search",
    notation,
    turns,
    stagedMoves,
    game: clientGame,
    position,
    move,
    depth,
    minDepth,
    nodes,
    timeMs,
    partitionIndex,
    partitionCount,
    temperature = 0,
    randomSeed = 0,
    gpuMode = "hybrid",
    disableNeural = false,
    modelBytes
  } = event.data;
  const searchGeneration = type === "search" ? ++activeSearchGeneration : activeSearchGeneration;

  try {
    if (type === "setModel") {
      if (!modelBytes) {
        throw new Error("GPU model override request is missing model bytes.");
      }
      frontierModelOverride = modelBytes;
      frontierRuntime?.pipeline.destroy();
      frontierRuntime?.neural.destroy();
      frontierRuntime = null;
      self.postMessage({ id, ok: true, modelConfigured: true });
      return;
    }
    if (type === "debugLoseDevice") {
      const hadDevice = await destroyCachedGpuDeviceForSmoke();
      self.postMessage({ id, ok: true, lostDevice: hadDevice });
      return;
    }

    if (clientGame && type === "legalTargets") {
      if (!position) {
        throw new Error("GPU legal target request is missing a source position.");
      }
      const selection = await engineLegalTargets(clientGame, position);
      self.postMessage({ id, ok: true, selection });
      return;
    }

    if (clientGame && type === "applyMove") {
      if (!move) {
        throw new Error("GPU move request is missing a move.");
      }
      const game = await engineApplyMove(clientGame, move);
      self.postMessage({ id, ok: true, game });
      return;
    }

    if (clientGame && type === "submitTurn") {
      const status = await engineSubmitTurn(clientGame);
      self.postMessage({ id, ok: true, status });
      return;
    }

    if (type === "legalTargets" || type === "applyMove" || type === "submitTurn") {
      throw new Error("GPU worker rules commands require a client game snapshot.");
    }

    if (!clientGame) {
      throw new Error("GPU worker calculations require a client game snapshot.");
    }

    const snapshotOverride = await engineGpuSnapshot(clientGame);
    if (!snapshotOverride) {
      throw new Error("GPU worker calculations require a client game snapshot.");
    }

    const searchConfig = await engineGpuWorkerSearchConfig(depth, minDepth, timeMs);
    gpuDeadlineAt = searchConfig.deadlineDelayMs == null
      ? Number.POSITIVE_INFINITY
      : Date.now() + searchConfig.deadlineDelayMs;
    try {
      const gpuResult = await tryGpuSearch({
        depth: searchConfig.requestedDepth,
        nodes,
        timeMs: searchConfig.searchTimeMs,
        gpuMode,
        disableNeural,
        snapshotOverride,
        sourceGame: clientGame,
        temperature,
        randomSeed
      });
      if (gpuResult && await enginePostableSearchResult(gpuResult)) {
        const validatedResult = await validateSearchResultBeforePost(snapshotOverride, gpuResult, clientGame);
        if (!validatedResult) {
          throw new Error("GPU search produced a turn that failed authoritative WASM replay.");
        }
        if (searchGeneration !== activeSearchGeneration) {
          return;
        }
        self.postMessage({ id, ok: true, result: validatedResult, partitionIndex: partitionIndex ?? 0 });
        return;
      }
      if (gpuResult) {
        throw new Error(`GPU search produced a non-postable result (${await engineNonPostableResultSummary(gpuResult)})`);
      }
    } catch (gpuError) {
      console.debug?.("GPU search failed", gpuError);
      if (gpuMode === "full") {
        try {
          const hybridResult = await tryGpuSearch({
            depth: requestedDepth,
            nodes,
            timeMs: searchTimeMs,
            gpuMode: "hybrid",
            disableNeural,
            snapshotOverride,
            sourceGame: clientGame,
            temperature,
            randomSeed
          });
          if (hybridResult && await enginePostableSearchResult(hybridResult)) {
            const validatedResult = await validateSearchResultBeforePost(snapshotOverride, hybridResult, clientGame);
            if (!validatedResult) {
              throw new Error("Hybrid GPU search produced a turn that failed authoritative WASM replay.", { cause: gpuError });
            }
            if (searchGeneration !== activeSearchGeneration) {
              return;
            }
            self.postMessage({ id, ok: true, result: validatedResult, partitionIndex: partitionIndex ?? 0 });
            return;
          }
        } catch (hybridError) {
          console.debug?.("Hybrid GPU search failed", hybridError);
        }
      }
      throw gpuError;
    }

    throw new Error(`GPU search did not produce a legal turn (${await gpuSearchFailureSummary(snapshotOverride)})`);
  } catch (error) {
    if (type === "search" && searchGeneration !== activeSearchGeneration) {
      return;
    }
    self.postMessage({ id, ok: false, error: errorMessage(error), partitionIndex: partitionIndex ?? 0 });
  }
});

async function gpuSearchFailureSummary(snapshot: GpuSnapshot): Promise<string> {
  try {
    return await engineGpuSearchFailureSummary(snapshot);
  } catch (error) {
    return `summary failed: ${errorMessage(error)}`;
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
