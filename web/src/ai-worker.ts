import { GPU_CANDIDATE_STRIDE, GPU_SOURCE_STRIDE, GPU_TARGET_STRIDE, GPU_BOARD_STRIDE, GPU_MUTATION_BOARD_STRIDE, GPU_MUTATION_CHILD_STRIDE, GPU_MUTATION_STATUS_OK, GPU_MUTATION_STATUS_ROYAL_CAPTURE, GPU_MUTATION_STATUS_BRANCH_OK, GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE, GPU_TURN_STATUS_RECORD_STRIDE, GPU_FRONTIER_BOARD_OFFSET, GPU_FRONTIER_BOARD_PENDING, GPU_FRONTIER_BOARD_STRIDE, GPU_FRONTIER_BOARD_TIME, GPU_FRONTIER_BOARD_TIMELINE_ID, GPU_FRONTIER_HEADER_BOARD_COUNT, GPU_FRONTIER_HEADER_DEPTH, GPU_FRONTIER_HEADER_HASH_HIGH, GPU_FRONTIER_HEADER_HASH_LOW, GPU_FRONTIER_HEADER_PLAN_LENGTH, GPU_FRONTIER_HEADER_SCORE, GPU_FRONTIER_HEADER_TERMINAL, GPU_FRONTIER_MAX_PLAN_MOVES, GPU_FRONTIER_MOVE_STRIDE, GPU_FRONTIER_PLAN_OFFSET } from "./ai-layout.js";
import { buildGpuCandidateInputsFromSnapshot, parseGpuSnapshotBytes, snapshotWithGpuChildBoards, gpuSnapshotToGame, colorFromCode, moveFromCandidateRecord, oppositeColor, presentTimeForSnapshot, colorCode } from "./ai-snapshot.js";
import { GPU_TURN_STATUS_SHADER, GPU_MOVEGEN_SHADER, GPU_REPLY_SHADER, GPU_MUTATE_SHADER } from "./ai-shaders.js";
import { autotuneFrontier, frontierStateBytes, frontierStateStride, FrontierGpuPipeline } from "./ai-frontier.js";
import type { EncodedFrontierRoot, FrontierBufferSet, FrontierTuning } from "./ai-frontier.js";
import { FrontierNeuralEvaluator } from "./ai-frontier-neural.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import { readWasmBytes, readWasmString, writeWasmString } from "./engine-io.js";
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

interface SearchCandidateSelectionPlan {
  selectedIndex: number | null;
  rankedIndexes: number[];
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
  const requestedDepth = Math.max(1, depth ?? 1);
  const snapshot = snapshotOverride;
  if (!snapshot) {
    return null;
  }

  const device = await getGpuDevice();
  if (!device) {
    return null;
  }
  const turnStatus = await turnStatusOnGpu(device, snapshot);
  const pendingBoards = await enginePendingPresentBoards(snapshot, snapshot.turn);
  if (gpuMode === "full") {
    try {
      return await tryGpuResidentFrontierSearch(device, snapshot, {
        requestedDepth,
        nodes: nodes ?? 64,
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
    : buildGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (candidates.sourceCount === 0 || candidates.targetCount === 0) {
    return null;
  }
  const scored = await scoreCandidatesOnGpu(device, candidates, snapshot.turn);
  let ranked = await engineRankedCandidates(scored, {
    pendingBoards,
    requirePending: true,
    limit: Math.min(128, Math.max(16, nodes ?? 64))
  });
  if (ranked.length === 0) {
    throw new Error(`GPU scoring produced no pending legal candidates (${await engineGpuScoringSummary(scored, pendingBoards)})`);
  }

  if (requestedDepth > 1) {
    const result = await searchSingleMoveRepliesOnGpu(device, snapshot, candidates, scored.records, ranked, {
      requestedDepth,
      nodes: nodes ?? 64,
      temperature,
      randomSeed
    });
    return completeGpuResultTurn(device, snapshot, result, { nodes: nodes ?? 64, temperature, randomSeed });
  }

  if (pendingBoards.length >= 1 && ranked.length > 0) {
    const mutated = await mutateRankedCandidatesOnGpu(device, candidates, scored.records, ranked);
    if (!mutated.some((entry) => entry.mutationStatus >= GPU_MUTATION_STATUS_OK)) {
      throw new Error(`GPU mutation rejected ranked candidates (${await engineGpuMutationSummary(mutated)})`);
    }
    const selected = await selectSearchCandidate(
      mutated.filter((entry) => entry.mutationStatus >= GPU_MUTATION_STATUS_OK),
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
      return completeGpuResultTurn(device, snapshot, result, { nodes: nodes ?? 64, temperature, randomSeed });
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
  const runtime = frontierRuntimeFor(device, tuning);
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
            frontierStateStride(tuning.maxBoards),
            GPU_FRONTIER_BOARD_OFFSET,
            tuning.maxBoards,
            tuning.neuralBatchSize,
            requestedDepth
          ) || modelUsed;
        }
      );
      activeStateLimit = Math.min(tuning.frontierWidth, activeStateLimit * perParentLimit);
      if (!disableNeural) {
        modelUsed = await runtime.neural.encode(
          encoder,
          buffers.nextStates,
          buffers.summaries,
          activeStateLimit,
          frontierStateStride(tuning.maxBoards),
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
    const latencyMs = Math.max(0, performance.now() - startedAt);
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
        candidateSelectionRate: ratio(readback.selectedCount, readback.nodes),
        tacticalSelectionRate: ratio(readback.selectedTacticalCandidates, readback.tacticalCandidates),
        effectiveBranchingFactor: cyclesCompleted > 0
          ? Math.round((readback.selectedCount / cyclesCompleted) * 100) / 100
          : readback.selectedCount,
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
        latencyMs: Math.round(latencyMs),
        nodesPerSecond: latencyMs > 0 ? Math.round((readback.nodes * 1000) / latencyMs) : readback.nodes,
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

function frontierRuntimeFor(device: GPUDevice, tuning: FrontierTuning): { device: GPUDevice; pipeline: FrontierGpuPipeline; neural: FrontierNeuralEvaluator } {
  if (frontierRuntime?.device === device
    && frontierRuntime.pipeline.tuning.maxBoards === tuning.maxBoards
    && frontierRuntime.pipeline.tuning.frontierWidth === tuning.frontierWidth
    && frontierRuntime.pipeline.tuning.candidateCapacity === tuning.candidateCapacity) {
    return frontierRuntime;
  }
  frontierRuntime?.pipeline.destroy();
  frontierRuntime?.neural.destroy();
  frontierRuntime = {
    device,
    pipeline: new FrontierGpuPipeline(device, tuning),
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
  const stateByteLength = frontierStateBytes(tuning.maxBoards) * tuning.frontierWidth;
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

function ratio(numerator: number, denominator: number): number {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator <= 0) {
    return 0;
  }
  return Math.round((numerator / denominator) * 1000) / 1000;
}

async function validatedFrontierChoices(
  snapshot: GpuSnapshot,
  states: Int32Array,
  tuning: FrontierTuning,
  requestedDepth: number,
  gpuSearch: string,
  sourceGame?: GameSnapshot
): Promise<SearchResult[]> {
  const stride = frontierStateStride(tuning.maxBoards);
  const ranked = Array.from({ length: tuning.frontierWidth }, (_, index) => {
    const base = index * stride;
    return {
      index,
      depth: states[base + GPU_FRONTIER_HEADER_DEPTH] ?? 0,
      score: states[base + GPU_FRONTIER_HEADER_SCORE] ?? -2147483647,
      terminal: (states[base + GPU_FRONTIER_HEADER_TERMINAL] ?? 0) !== 0,
      planLength: Math.min(GPU_FRONTIER_MAX_PLAN_MOVES, Math.max(0, states[base + GPU_FRONTIER_HEADER_PLAN_LENGTH] ?? 0))
    };
  })
    .filter((entry) => entry.planLength > 0 && entry.depth > 0)
    .sort((left, right) => right.depth - left.depth || right.score - left.score || left.index - right.index);

  const choices: SearchResult[] = [];
  const seen = new Set<string>();
  for (const entry of ranked) {
    const base = entry.index * stride + GPU_FRONTIER_PLAN_OFFSET;
    const plan: Move[] = [];
    for (let moveIndex = 0; moveIndex < entry.planLength; moveIndex += 1) {
      const offset = base + moveIndex * GPU_FRONTIER_MOVE_STRIDE;
      plan.push(moveFromFrontierWords(states, offset));
    }
    const moves = await validateFirstFrontierTurn(snapshot, plan, sourceGame);
    const key = turnPlanKey(moves);
    if (!moves.length || seen.has(key)) {
      continue;
    }
    seen.add(key);
    choices.push({
      status: "ok",
      moves,
      score: entry.score,
      depth: Math.min(requestedDepth, entry.depth),
      principalVariation: [moves],
      gpu: true,
      gpuMode: "full",
      gpuTerminal: entry.terminal,
      gpuSearch,
      tactical: entry.terminal
    });
    if (choices.length >= 12) {
      break;
    }
  }
  return choices;
}

function moveFromFrontierWords(words: Int32Array, offset: number): Move {
  return {
    from: { timelineId: words[offset] ?? 0, time: words[offset + 1] ?? 0, x: words[offset + 2] ?? 0, y: words[offset + 3] ?? 0 },
    to: { timelineId: words[offset + 4] ?? 0, time: words[offset + 5] ?? 0, x: words[offset + 6] ?? 0, y: words[offset + 7] ?? 0 }
  };
}

async function validateFirstFrontierTurn(snapshot: GpuSnapshot, plan: Move[], sourceGame?: GameSnapshot): Promise<Move[]> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, validationGameSnapshot(snapshot, sourceGame));
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
  loadValidationSnapshot(engine, validationGameSnapshot(snapshot, sourceGame));
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

function validationGameSnapshot(snapshot: GpuSnapshot, sourceGame?: GameSnapshot): GameSnapshot {
  return sourceGame ?? gpuSnapshotToGame(snapshot);
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
  return parseGpuSnapshotBytes(readWasmBytes(engine, engine.chronofish_gpu_snapshot_bytes()));
}

async function engineFrontierMaxCycles(requestedDepth: number, timelineCount: number): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_frontier_max_cycles(requestedDepth, timelineCount);
}

async function engineFrontierPerParentLimit(frontierWidth: number): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_frontier_per_parent_limit(frontierWidth);
}

async function engineGpuCandidateInputs(game: GameSnapshot): Promise<GpuCandidateInputs> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  const bytes = readWasmBytes(engine, engine.chronofish_gpu_candidate_inputs_bytes());
  if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Engine GPU candidate input byte length is not i32-aligned.");
  }
  return candidateInputsFromWords(new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Int32Array.BYTES_PER_ELEMENT));
}

function candidateInputsFromWords(words: Int32Array): GpuCandidateInputs {
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
  return {
    sourceMeta: candidateMetaFromRecords(sources, GPU_SOURCE_STRIDE),
    targetMeta: candidateMetaFromRecords(targets, GPU_TARGET_STRIDE),
    sourceCount,
    targetCount,
    boardCount,
    sources,
    targets,
    boards,
    mutationBoards
  };
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
  const boardCount = Math.max(1, snapshot.timelines.reduce((sum, timeline) => sum + timeline.boards.length, 0));
  const root = await engineFrontierRootFromSnapshot({ ...snapshot, turn: color }, boardCount);
  const pending: PendingBoardRef[] = [];
  const encodedBoardCount = Math.max(0, root.words[GPU_FRONTIER_HEADER_BOARD_COUNT] ?? 0);
  for (let index = 0; index < encodedBoardCount; index += 1) {
    const base = GPU_FRONTIER_BOARD_OFFSET + index * GPU_FRONTIER_BOARD_STRIDE;
    if ((root.words[base + GPU_FRONTIER_BOARD_PENDING] ?? 0) === 0) {
      continue;
    }
    pending.push({
      timelineId: root.words[base + GPU_FRONTIER_BOARD_TIMELINE_ID] ?? 0,
      time: root.words[base + GPU_FRONTIER_BOARD_TIME] ?? 0
    });
  }
  return pending;
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
    const output = engine.chronofish_gpu_ranked_candidate_indexes_bytes(ptr, byteLength);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const bytes = readWasmBytes(engine, output);
    if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error("Engine ranked candidate index byte length is not i32-aligned.");
    }
    return Array.from(new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Int32Array.BYTES_PER_ELEMENT), (index) => ({
      move: moveFromCandidateRecord(scored.records, index),
      index,
      score: scored.scores[index] ?? -2147483647
    }));
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
    selected: choiceMoves(selected),
    choices: choices.map(choiceMoves),
    limits
  }));
  try {
    const output = engine.chronofish_gpu_choice_agreement_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const response = JSON.parse(readWasmString(engine, output)) as { agreements?: number[] };
    return Array.isArray(response.agreements) ? response.agreements : [];
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

async function engineSearchCandidateSelection(request: unknown): Promise<SearchCandidateSelectionPlan> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(request));
  try {
    const output = engine.chronofish_gpu_select_candidate_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as SearchCandidateSelectionPlan;
  } finally {
    engine.chronofish_dealloc(ptr, len);
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
  const complete = Boolean(engine.chronofish_submit_turn());
  const snapshot = JSON.parse(readWasmString(engine, engine.chronofish_snapshot_json())) as GameSnapshot;
  const message = readWasmString(engine, engine.chronofish_last_message());
  return {
    complete,
    terminal: snapshot.result?.terminal === true,
    winner: snapshot.result?.winner ?? undefined,
    nextTurn: snapshot.turn,
    presentTime: snapshot.presentTime ?? presentTimeForSnapshot({ ...snapshot, format: "json" }) ?? 0,
    pendingPresentBoardCount: complete ? 0 : 1,
    message
  };
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
  for (const entry of mutated.filter(hasSupportedChildBoards)) {
    let score = entry.score;
    let principalVariation: Move[][] = [[entry.move]];
    if (entry.mutationStatus !== GPU_MUTATION_STATUS_ROYAL_CAPTURE && entry.mutationStatus !== GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
      const childSnapshot = snapshotWithGpuChildBoards(snapshot, entry.childBoards, entry.mutationStatus, { move: entry.move, advanceTurn: true });
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
      depth: Math.min(requestedDepth, 2),
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
    return result?.moves?.length ? withCompletedTurnChoice(result, result.moves, result.gpuSearch) : result;
  }
  const rootTurn = snapshot.turn;
  let current = snapshot;
  const moves: Move[] = [];
  let extraNodes = 0;
  for (const move of result.moves) {
    current = await applyGpuMoveToSnapshot(device, { ...current, turn: rootTurn }, move, { advanceTurn: true });
    moves.push(move);
    if (current.royalCaptureBy) {
      return {
        ...withCompletedTurnChoice(result, moves, `${result.gpuSearch ?? "gpu"}-turn-complete`),
        gpuTerminal: true
      };
    }
  }

  const maxMoves = Math.max(moves.length, snapshot.timelines.length + 4);
  while (moves.length < maxMoves) {
    const status = await turnStatusOnGpu(device, { ...current, turn: rootTurn });
    const pendingBoards = await enginePendingPresentBoards(current, rootTurn);
    if ((status.complete || status.pendingPresentBoardCount === 0) && pendingBoards.length === 0) {
      break;
    }
    const stepSnapshot = { ...current, turn: rootTurn };
    const inputs = buildGpuCandidateInputsFromSnapshot(stepSnapshot, rootTurn);
    if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
      break;
    }
    const scored = await scoreCandidatesOnGpu(device, inputs, rootTurn);
    const ranked = await engineRankedCandidates(scored, {
      pendingBoards,
      requirePending: true,
      limit: Math.min(128, Math.max(16, nodes ?? 64))
    });
    if (ranked.length === 0) {
      break;
    }
    const mutated = await mutateRankedCandidatesOnGpu(device, inputs, scored.records, ranked, { readChildren: true });
    extraNodes += mutated.length;
    const selected = await selectSearchCandidate(
      mutated.filter(hasSupportedChildBoards),
      temperature,
      randomSeed + moves.length
    );
    if (!selected) {
      break;
    }
    current = snapshotWithGpuChildBoards(stepSnapshot, selected.childBoards, selected.mutationStatus, {
      move: selected.move,
      advanceTurn: true
    });
    moves.push(selected.move);
    if (selected.mutationStatus === GPU_MUTATION_STATUS_ROYAL_CAPTURE || selected.mutationStatus === GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
      return {
        ...withCompletedTurnChoice(result, moves, `${result.gpuSearch ?? "gpu"}-turn-complete`),
        nodes: (result.nodes ?? 0) + extraNodes,
        gpuTerminal: true
      };
    }
  }

  const finalStatus = await turnStatusOnGpu(device, { ...current, turn: rootTurn });
  const finalPendingBoards = await enginePendingPresentBoards(current, rootTurn);
  if (finalPendingBoards.length > 0 || (!finalStatus.complete && finalStatus.pendingPresentBoardCount > 0)) {
    const fallback = await findCompleteGpuTurn(device, snapshot, rootTurn, nodes ?? 64);
    if (fallback) {
      return withCompletedTurnChoice({
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

  return withCompletedTurnChoice({
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
  const reply = await bestReplyOnGpu(device, snapshot, nodes ?? 64);
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

function withCompletedTurnChoice(
  result: SearchResult,
  moves: Move[],
  gpuSearch = result.gpuSearch,
  principalVariation: Move[][] = [moves, ...(result.principalVariation ?? []).slice(1)]
): SearchResult {
  const completedChoice = {
    rank: 1,
    score: result.score,
    moves,
    principalVariation,
    depth: result.depth,
    nodes: result.nodes,
    gpuSearch
  };
  const existingChoices = Array.isArray(result.choices) ? result.choices : [];
  return {
    ...result,
    moves,
    gpuSearch,
    principalVariation,
    choices: [
      completedChoice,
      ...existingChoices
        .filter((choice) => !sameMoveSequence(choice.moves ?? [], moves))
        .slice(0, 11)
    ]
  };
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
  const maxMoves = Math.max(1, snapshot.timelines.length + 4);
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
  const inputs = buildGpuCandidateInputsFromSnapshot(stepSnapshot, rootTurn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    return null;
  }
  const scored = await scoreCandidatesOnGpu(device, inputs, rootTurn);
  const ranked = await engineRankedCandidates(scored, {
    pendingBoards,
    requirePending: true,
    limit: Math.min(12, Math.max(4, nodes))
  });
  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, scored.records, ranked, { readChildren: true });
  const supported = mutated.filter(hasSupportedChildBoards);
  for (const candidate of supported) {
    const child = snapshotWithGpuChildBoards(stepSnapshot, candidate.childBoards, candidate.mutationStatus, {
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

function sameMoveSequence(left: Move[], right: Move[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  return left.every((move, index) => sameMove(move, right[index]));
}

function sameMove(left: Move | undefined, right: Move | undefined): boolean {
  if (!left || !right) {
    return false;
  }
  return true
    && left.from.timelineId === right.from.timelineId
    && left.from.time === right.from.time
    && left.from.x === right.from.x
    && left.from.y === right.from.y
    && left.to.timelineId === right.to.timelineId
    && left.to.time === right.to.time
    && left.to.x === right.to.x
    && left.to.y === right.to.y;
}

async function applyGpuMoveToSnapshot(
  device: GPUDevice,
  snapshot: GpuSnapshot,
  move: Move,
  { advanceTurn = false }: { advanceTurn?: boolean } = {}
): Promise<GpuSnapshot> {
  const inputs = buildGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
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
  const selected = mutated[0];
  if (!selected || selected.mutationStatus < GPU_MUTATION_STATUS_OK || !selected.childBoards) {
    throw new Error("GPU move mutation is unsupported for that move.");
  }
  return snapshotWithGpuChildBoards(snapshot, selected.childBoards, selected.mutationStatus, { move, advanceTurn });
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
  return {
    complete: result[0] === 0,
    nextTurn: colorFromCode(result[1] ?? 0),
    presentTime: result[2] ?? 0,
    pendingPresentBoardCount: result[3] ?? 0
  };
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
    limit: Math.min(128, Math.max(16, nodes ?? 64))
  });
  if (ranked.length === 0) {
    throw new Error("Full GPU search found no candidate moves.");
  }

  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, scored.records, ranked, { readChildren: true });
  const supported = mutated.filter(hasSupportedChildBoards);
  if (supported.length === 0) {
    throw new Error("Full GPU mutation produced no supported child states.");
  }

  const candidates: SearchResult[] = [];
  for (const entry of supported.slice(0, Math.min(32, Math.max(8, nodes ?? 64)))) {
    let score = entry.score;
    let principalVariation: Move[][] = [[entry.move]];
    if (requestedDepth > 1 && entry.mutationStatus !== GPU_MUTATION_STATUS_ROYAL_CAPTURE && entry.mutationStatus !== GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
      const childSnapshot = snapshotWithGpuChildBoards(snapshot, entry.childBoards, entry.mutationStatus, { move: entry.move, advanceTurn: true });
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
      depth: Math.min(requestedDepth, 2),
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
  const inputs = buildGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
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
  const plan = await engineSearchCandidateSelection({
    temperature: Number.isFinite(Number(temperature)) ? Number(temperature) : 0,
    randomSeed: Math.trunc(Number(randomSeed) || 0),
    candidates: candidates.map((candidate, index) => {
      const moves = choiceMoves(candidate);
      const score = Number(candidate.score);
      return {
        index,
        score: Number.isFinite(score) ? score : 0,
        key: turnPlanKey(moves),
        moveCount: moves.length
      };
    })
  });
  const supported = plan.rankedIndexes
    .map((index) => candidates[index])
    .filter((candidate): candidate is T => Boolean(candidate));
  const selected = plan.selectedIndex === null ? null : candidates[plan.selectedIndex];
  return selected ? withSearchChoices(selected, supported) as T & { choices: SearchChoice[] } : null;
}

function hasSupportedChildBoards(entry: MutatedCandidate): entry is MutatedCandidate & { childBoards: Int32Array } {
  return entry.mutationStatus >= GPU_MUTATION_STATUS_OK && Boolean(entry.childBoards);
}

function withSearchChoices<T extends SearchChoice>(selected: T, candidates: SearchChoice[]): T & { choices: SearchChoice[] } {
  return {
    ...selected,
    choices: summarizeSearchChoices(candidates)
  };
}

function summarizeSearchChoices(candidates: SearchChoice[]): SearchChoice[] {
  return candidates
    .slice(0, 12)
    .map((candidate, index) => ({
      rank: index + 1,
      score: candidate.score,
      moves: choiceMoves(candidate),
      principalVariation: candidate.principalVariation,
      depth: candidate.depth,
      nodes: candidate.nodes,
      gpuSearch: candidate.gpuSearch,
      gpuTerminal: candidate.gpuTerminal,
      tactical: candidate.tactical
    }));
}

function choiceMoves(candidate: SearchChoice): Move[] {
  return candidate.moves ?? (candidate.move ? [candidate.move] : []);
}

function turnPlanKey(moves: Move[]): string {
  return moves.map((move) => [
    move.from.timelineId,
    move.from.time,
    move.from.x,
    move.from.y,
    move.to.timelineId,
    move.to.time,
    move.to.x,
    move.to.y
  ].join(":")).join("/");
}

let gpuDeadlineAt = 0;

async function scoreCandidatesOnGpu(device: GPUDevice, inputs: GpuCandidateInputs, turn: Color): Promise<ScoredCandidates> {
  const candidateCount = inputs.sourceCount * inputs.targetCount;
  const maxDispatchWorkgroups = 65_535;
  const maxCandidatesPerDispatch = maxDispatchWorkgroups * 64;
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const maxCandidatesPerBatch = Math.max(
    1,
    Math.min(
      maxCandidatesPerDispatch,
      Math.floor(maxBindingSize / (GPU_CANDIDATE_STRIDE * Int32Array.BYTES_PER_ELEMENT))
    )
  );
  if (inputs.targetCount > maxCandidatesPerBatch) {
    throw new Error(`GPU move generation target set is too large for this device (${inputs.targetCount} targets).`);
  }
  const targetBuffer = storageBuffer(device, inputs.targets, GPUBufferUsage.STORAGE);
  const boardBuffer = storageBuffer(device, inputs.boards ?? new Int32Array(GPU_BOARD_STRIDE), GPUBufferUsage.STORAGE);
  const pipeline = await createComputePipelineChecked(device, "score_candidates", GPU_MOVEGEN_SHADER, "score_candidates");
  const records = new Int32Array(candidateCount * GPU_CANDIDATE_STRIDE);
  const scores = new Int32Array(candidateCount);
  const sourceBatchSize = Math.max(1, Math.floor(maxCandidatesPerBatch / inputs.targetCount));

  for (let sourceStart = 0; sourceStart < inputs.sourceCount; sourceStart += sourceBatchSize) {
    const sourceCount = Math.min(sourceBatchSize, inputs.sourceCount - sourceStart);
    const batchCandidateCount = sourceCount * inputs.targetCount;
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
    pass.dispatchWorkgroups(Math.min(maxDispatchWorkgroups, Math.ceil(batchCandidateCount / 64)));
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
  const limit = Math.min(ranked.length, 64);
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
  pass.dispatchWorkgroups(Math.ceil(limit / 64));
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
  pass.dispatchWorkgroups(Math.ceil(rankedRoots.length / 16), Math.ceil(rankedReplies.length / 16));
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

    const requestedDepth = Math.max(1, depth ?? 1);
    const minimumDepth = Math.min(requestedDepth, Math.max(1, Math.floor(minDepth ?? 1)));
    const searchTimeMs = Math.max(1, timeMs ?? 10_000);
    gpuDeadlineAt = minimumDepth >= requestedDepth
      ? Number.POSITIVE_INFINITY
      : Date.now() + Math.max(1, Math.floor(searchTimeMs * 0.8));
    try {
      const gpuResult = await tryGpuSearch({
        depth: requestedDepth,
        nodes,
        timeMs: searchTimeMs,
        gpuMode,
        disableNeural,
        snapshotOverride,
        sourceGame: clientGame,
        temperature,
        randomSeed
      });
      if (isPostableSearchResult(gpuResult)) {
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
        throw new Error(`GPU search produced a non-postable result (${nonPostableResultSummary(gpuResult)})`);
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
          if (isPostableSearchResult(hybridResult)) {
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

function isPostableSearchResult(result: SearchResult | null): result is SearchResult {
  return Boolean(result?.status === "ok" && result.moves?.length);
}

function nonPostableResultSummary(result: unknown): string {
  const candidate = result as Partial<SearchResult> | null | undefined;
  return `status=${candidate?.status ?? "unknown"}, moves=${candidate?.moves?.length ?? 0}, incomplete=${candidate?.incompleteMoves?.length ?? 0}, pending=${candidate?.pendingPresentBoardCount ?? "unknown"}`;
}

async function gpuSearchFailureSummary(snapshot: GpuSnapshot): Promise<string> {
  try {
    const inputs = buildGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
    const pendingBoards = await enginePendingPresentBoards(snapshot, snapshot.turn);
    return `sources=${inputs.sourceCount}, targets=${inputs.targetCount}, pending=${pendingBoards.length}, timelines=${snapshot.timelines.length}`;
  } catch (error) {
    return `summary failed: ${errorMessage(error)}`;
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
