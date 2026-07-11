import { GPU_CANDIDATE_STRIDE, GPU_SOURCE_STRIDE, GPU_TARGET_STRIDE, GPU_BOARD_STRIDE, GPU_MUTATION_BOARD_STRIDE, GPU_MUTATION_CHILD_STRIDE, GPU_MUTATION_STATUS_BRANCH_OK, GPU_TURN_STATUS_RECORD_STRIDE, GPU_FRONTIER_BOARD_OFFSET } from "./ai-layout.js";
import { colorCode } from "./ai-snapshot.js";
import { GPU_TURN_STATUS_SHADER, GPU_MOVEGEN_SHADER, GPU_REPLY_SHADER, GPU_MUTATE_SHADER } from "./ai-shaders.js";
import { autotuneFrontier, frontierStateBytes, frontierStateStride, FrontierGpuPipeline } from "./ai-frontier.js";
import type { EncodedFrontierRoot, FrontierBufferSet, FrontierTuning } from "./ai-frontier.js";
import { FrontierNeuralEvaluator } from "./ai-frontier-neural.js";
import * as engineGpuSearch from "./engine-gpu-search.js";
import { GPUBufferUsage, GPUMapMode } from "./ai-worker-types.js";
import { align4, clearComputePipelineCache, createComputePipelineChecked, requestHighLimitDevice, storageBuffer } from "./ai-gpu-device.js";
import type { Color, GameSnapshot, Move } from "./types.js";
import type { GpuCandidateInputs, GpuSnapshot } from "./ai-snapshot.js";
import type { GpuMode, GpuSearchOptions, MutatedCandidate, RankedCandidate, ReplySearchResult, ScoredCandidates, SearchChoice, SearchResult, TurnStatus, WorkerRequest } from "./ai-worker-types.js";

let cachedGpuAdapter: GPUAdapter | null = null;
let cachedGpuDevice: GPUDevice | null = null;
let frontierRuntime: { device: GPUDevice; pipeline: FrontierGpuPipeline; neural: FrontierNeuralEvaluator } | null = null;
let activeSearchGeneration = 0;
let frontierModelOverride: ArrayBuffer | null = null;

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
  const requestedDepth = await engineGpuSearch.engineSearchDepthAtLeastOne(depth);
  const snapshot = snapshotOverride;
  if (!snapshot) {
    return null;
  }

  const device = await getGpuDevice();
  if (!device) {
    return null;
  }
  const searchNodes = await engineGpuSearch.engineGpuSearchNodes(nodes);
  const turnStatus = await turnStatusOnGpu(device, snapshot);
  const pendingBoards = await engineGpuSearch.enginePendingPresentBoards(snapshot, snapshot.turn);
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
    ? await engineGpuSearch.engineGpuCandidateInputs(sourceGame)
    : await engineGpuSearch.engineGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (candidates.sourceCount === 0 || candidates.targetCount === 0) {
    return null;
  }
  const scored = await scoreCandidatesOnGpu(device, candidates, snapshot.turn);
  let ranked = await engineGpuSearch.engineRankedCandidates(scored, {
    pendingBoards,
    requirePending: true,
    limit: await engineGpuSearch.engineGpuSearchRankingLimit(nodes)
  });
  if (ranked.length === 0) {
    throw new Error(`GPU scoring produced no pending legal candidates (${await engineGpuSearch.engineGpuScoringSummary(scored, pendingBoards)})`);
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
      throw new Error(`GPU mutation rejected ranked candidates (${await engineGpuSearch.engineGpuMutationSummary(mutated)})`);
    }
    const selected = await selectSearchCandidate(
      supported,
      temperature,
      randomSeed
    );
    if (!selected) {
      throw new Error(`GPU mutation produced no selectable candidate (${await engineGpuSearch.engineGpuMutationSummary(mutated)})`);
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
        gpuTerminal: await gpuMutationStatusIsTerminal(selected.mutationStatus),
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
    sourceGame?: GameSnapshot | undefined;
  }
): Promise<SearchResult> {
  const searchSize = await engineGpuSearch.engineGpuSnapshotSearchSize(snapshot);
  const adapter = cachedGpuAdapter;
  if (!adapter) {
    throw new Error("GPU frontier search has no adapter for tuning.");
  }
  const orchestration = await engineFrontierOrchestrationPlan(
    requestedDepth,
    searchSize.timelineCount,
    nodes,
    searchSize.boardCount,
    adapter,
    device
  );
  const { maxCycles, tuning, perParentLimit, stateLimits } = orchestration;
  const runtime = await frontierRuntimeFor(device, tuning);
  const frontierEngine = runtime.pipeline.engine;
  if (!frontierEngine) {
    throw new Error("GPU frontier search has no WASM engine binding.");
  }
  runtime.neural.beginSearch();
  const buffers = runtime.pipeline.pool.createSearchBuffers();
  const root = sourceGame
    ? await engineGpuSearch.engineFrontierRoot(sourceGame, tuning.maxBoards)
    : await engineGpuSearch.engineFrontierRootFromSnapshot(snapshot, tuning.maxBoards);
  const startedAt = performance.now();
  const initialize = device.createCommandEncoder();
  initialize.clearBuffer(buffers.states);
  initialize.clearBuffer(buffers.nextStates);
  initialize.clearBuffer(buffers.counters);
  device.queue.submit([initialize.finish()]);
  runtime.pipeline.uploadRoot(buffers, root);

  const rootColor = colorCode(snapshot.turn);
  let modelUsed = false;
  let cyclesCompleted = 0;
  try {
    for (let cycle = 0; cycle < maxCycles; cycle += 1) {
      if (engineGpuSearch.engineGpuFrontierCycleShouldStop(frontierEngine, cycle, cyclesCompleted, requestedDepth)) {
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
          stateCount: stateLimits[cycle] ?? 1,
          perParentLimit
        },
        async (policyEncoder, policyBuffers, candidateCapacity) => {
          modelUsed = await runtime.neural.encodePolicyPrior(
            policyEncoder,
            policyBuffers.states,
            policyBuffers.candidates,
            candidateCapacity,
            GPU_CANDIDATE_STRIDE,
            stateLimits[cycle] ?? 1,
            frontierStateStride(tuning.maxBoards, runtime.pipeline.engine),
            GPU_FRONTIER_BOARD_OFFSET,
            tuning.maxBoards,
            tuning.neuralBatchSize,
            requestedDepth
          ) || modelUsed;
        }
      );
      const nextStateLimit = stateLimits[cycle + 1] ?? 1;
      if (!disableNeural) {
        modelUsed = await runtime.neural.encode(
          encoder,
          buffers.nextStates,
          buffers.summaries,
          nextStateLimit,
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
    const policyChoiceAgreement = await engineGpuSearch.enginePolicyChoiceAgreementDiagnostics(selected, selected.choices);
    const choiceDiagnostics = await engineGpuSearch.engineFrontierChoiceDiagnostics(selected, selected.choices);
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
        candidateSelectionRate: engineGpuSearch.gpuDiagnosticRate(readback.selectedCount, readback.nodes, frontierEngine),
        tacticalSelectionRate: engineGpuSearch.gpuDiagnosticRate(readback.selectedTacticalCandidates, readback.tacticalCandidates, frontierEngine),
        effectiveBranchingFactor: engineGpuSearch.gpuEffectiveBranchingFactor(readback.selectedCount, cyclesCompleted, frontierEngine),
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
        legalChoiceCount: choiceDiagnostics.legalChoiceCount,
        legalTacticalChoiceCount: choiceDiagnostics.legalTacticalChoiceCount,
        topPolicyChoiceAgreement: policyChoiceAgreement.topPolicyChoiceAgreement,
        top5PolicyChoiceAgreement: policyChoiceAgreement.top5PolicyChoiceAgreement,
        top20PolicyChoiceAgreement: policyChoiceAgreement.top20PolicyChoiceAgreement,
        selectedMovePrunedRisk: choiceDiagnostics.selectedMovePrunedRisk,
        selectedMoveTactical: choiceDiagnostics.selectedMoveTactical,
        model: modelUsed ? "neural" : "heuristic",
        latencyMs: engineGpuSearch.gpuReportedLatencyMs(latencyMs, frontierEngine),
        nodesPerSecond: engineGpuSearch.gpuNodesPerSecond(readback.nodes, latencyMs, frontierEngine),
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
  const engine = await engineGpuSearch.validationEngine();
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
} & engineGpuSearch.GpuFrontierReadbackSummary> {
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
    return { states, ...await engineGpuSearch.engineGpuFrontierReadbackSummary(counters) };
  } catch (error) {
    staging.destroy();
    clearCachedGpuState();
    throw error;
  }
}

interface FrontierOrchestrationPlan {
  maxCycles: number;
  perParentLimit: number;
  stateLimits: number[];
}

async function engineFrontierOrchestrationPlan(
  requestedDepth: number,
  timelineCount: number,
  nodes: number,
  boardCount: number,
  adapter: GPUAdapter,
  device: GPUDevice
): Promise<FrontierOrchestrationPlan & { tuning: FrontierTuning }> {
  const provisionalCycles = await engineGpuSearch.engineFrontierMaxCycles(requestedDepth, timelineCount);
  const tuning = await autotuneFrontier(
    adapter,
    device,
    nodes,
    boardCount,
    "gpu-v1-cfnn-v3-policy-head",
    provisionalCycles * 2
  );
  return {
    ...await engineGpuSearch.engineFrontierOrchestrationPlanForWidth(
      requestedDepth,
      timelineCount,
      tuning.frontierWidth
    ),
    tuning
  };
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
  const seenKeys: string[] = [];
  const engine = await engineGpuSearch.validationEngine();
  const candidates = engineGpuSearch.engineFrontierChoicesFromWords(engine, states, tuning, requestedDepth, gpuSearch);
  for (const candidate of candidates) {
    const moves = await engineGpuSearch.validateFirstFrontierTurn(snapshot, candidate.moves, sourceGame);
    const accepted = await engineGpuSearch.engineValidatedFrontierChoice(candidate, moves, seenKeys, choices.length, 12, gpuSearch);
    if (!accepted.accepted || !accepted.choice) {
      continue;
    }
    if (accepted.key) {
      seenKeys.push(accepted.key);
    }
    choices.push(accepted.choice);
    if (choices.length >= 12) {
      break;
    }
  }
  return choices;
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
    if (!await gpuMutationStatusIsTerminal(entry.mutationStatus)) {
      const childSnapshot = await engineGpuSearch.engineSnapshotWithGpuChildBoards(snapshot, entry.childBoards, entry.mutationStatus, { move: entry.move, advanceTurn: true });
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
      depth: await engineGpuSearch.engineGpuFullSearchReportedDepth(requestedDepth),
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
    return result?.moves?.length ? await engineGpuSearch.engineWithCompletedTurnChoice(result, result.moves, result.gpuSearch) : result;
  }
  const rootTurn = snapshot.turn;
  let current = snapshot;
  const moves: Move[] = [];
  let extraNodes = 0;
  const searchNodes = await engineGpuSearch.engineGpuSearchNodes(nodes);
  for (const move of result.moves) {
    current = await applyGpuMoveToSnapshot(device, { ...current, turn: rootTurn }, move, { advanceTurn: true });
    moves.push(move);
    if (current.royalCaptureBy) {
      return {
        ...(await engineGpuSearch.engineWithCompletedTurnChoice(result, moves, `${result.gpuSearch ?? "gpu"}-turn-complete`)),
        gpuTerminal: true
      };
    }
  }

  while (true) {
    const status = await turnStatusOnGpu(device, { ...current, turn: rootTurn });
    const pendingBoards = await engineGpuSearch.enginePendingPresentBoards(current, rootTurn);
    const completion = await engineGpuSearch.engineGpuTurnCompletionStep(current, moves.length, pendingBoards, status);
    if (completion.action === "terminal") {
      return {
        ...(await engineGpuSearch.engineWithCompletedTurnChoice(result, moves, `${result.gpuSearch ?? "gpu"}-turn-complete`)),
        gpuTerminal: true
      };
    }
    if (completion.action === "complete") {
      break;
    }
    if (completion.action !== "search") {
      break;
    }
    const stepSnapshot = { ...current, turn: rootTurn };
    const inputs = await engineGpuSearch.engineGpuCandidateInputsFromSnapshot(stepSnapshot, rootTurn);
    if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
      break;
    }
    const scored = await scoreCandidatesOnGpu(device, inputs, rootTurn);
    const ranked = await engineGpuSearch.engineRankedCandidates(scored, {
      pendingBoards,
      requirePending: true,
      limit: await engineGpuSearch.engineGpuSearchRankingLimit(nodes)
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
    current = await engineGpuSearch.engineSnapshotWithGpuChildBoards(stepSnapshot, selected.childBoards, selected.mutationStatus, {
      move: selected.move,
      advanceTurn: true
    });
    moves.push(selected.move);
    if (await gpuMutationStatusIsTerminal(selected.mutationStatus)) {
      return {
        ...(await engineGpuSearch.engineWithCompletedTurnChoice(result, moves, `${result.gpuSearch ?? "gpu"}-turn-complete`)),
        nodes: await engineGpuSearch.engineGpuAccumulatedSearchNodes(result.nodes, extraNodes),
        gpuTerminal: true
      };
    }
  }

  const finalStatus = await turnStatusOnGpu(device, { ...current, turn: rootTurn });
  const finalPendingBoards = await engineGpuSearch.enginePendingPresentBoards(current, rootTurn);
  const finalCompletion = await engineGpuSearch.engineGpuTurnCompletionStep(current, moves.length, finalPendingBoards, finalStatus);
  if (finalCompletion.action !== "complete" && finalCompletion.action !== "terminal") {
    const fallback = await findCompleteGpuTurn(device, snapshot, rootTurn, searchNodes);
    if (fallback) {
      return engineGpuSearch.engineWithCompletedTurnChoice({
        ...result,
        nodes: await engineGpuSearch.engineGpuAccumulatedSearchNodes(result.nodes, extraNodes, fallback.nodes)
      }, fallback.moves, `${result.gpuSearch ?? "gpu"}-turn-fallback`);
    }
    return {
      status: "incompleteTurn",
      moves: [],
      score: result.score,
      depth: result.depth,
      nodes: await engineGpuSearch.engineGpuAccumulatedSearchNodes(result.nodes, extraNodes),
      gpu: true,
      gpuSnapshot: result.gpuSnapshot,
      gpuSearch: `${result.gpuSearch ?? "gpu"}-turn-incomplete`,
      incompleteMoves: moves,
      pendingPresentBoardCount: await engineGpuSearch.engineIncompleteTurnPendingPresentBoardCount(finalStatus, finalPendingBoards),
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

  return engineGpuSearch.engineWithCompletedTurnChoice({
    ...result,
    nodes: await engineGpuSearch.engineGpuAccumulatedSearchNodes(result.nodes, extraNodes)
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
  if (!(await engineGpuSearch.engineGpuCompletedReplyShouldSearch(snapshot))) {
    return [];
  }
  const searchNodes = await engineGpuSearch.engineGpuSearchNodes(nodes);
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
  const pendingBoards = await engineGpuSearch.enginePendingPresentBoards(snapshot, rootTurn);
  const status = await turnStatusOnGpu(device, { ...snapshot, turn: rootTurn });
  const completion = await engineGpuSearch.engineGpuTurnCompletionStep(snapshot, moves.length, pendingBoards, status, visited);
  if (completion.action === "terminal" || completion.action === "complete") {
    return { moves, nodes: 0 };
  }
  if (completion.action !== "search" || !completion.stateKey) {
    return null;
  }

  const nextVisited = new Set(visited);
  nextVisited.add(completion.stateKey);

  const stepSnapshot = { ...snapshot, turn: rootTurn };
  const inputs = await engineGpuSearch.engineGpuCandidateInputsFromSnapshot(stepSnapshot, rootTurn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    return null;
  }
  const scored = await scoreCandidatesOnGpu(device, inputs, rootTurn);
  const ranked = await engineGpuSearch.engineRankedCandidates(scored, {
    pendingBoards,
    requirePending: true,
    limit: await engineGpuSearch.engineGpuSearchReplyLimit(nodes)
  });
  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, scored.records, ranked, { readChildren: true });
  const supported = await supportedMutatedCandidates(mutated);
  for (const candidate of supported) {
    const child = await engineGpuSearch.engineSnapshotWithGpuChildBoards(stepSnapshot, candidate.childBoards, candidate.mutationStatus, {
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
  const inputs = await engineGpuSearch.engineGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    throw new Error("No GPU move candidates are available.");
  }
  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  const index = await engineGpuSearch.engineCandidateIndex(scored, move);
  if (index < 0) {
    throw new Error("GPU rejected that move.");
  }
  const score = await engineGpuSearch.engineCandidateScore(scored.scores, index, -2147483647);
  if (await engineGpuSearch.engineCandidateScoreIsRejected(score)) {
    throw new Error("GPU rejected that move.");
  }
  const candidateRecords = await engineGpuSearch.enginePickCandidateRecords(scored.records, [index]);
  const ranked = [{ move, index: 0, score }];
  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, candidateRecords, ranked, { readChildren: true });
  const [selected] = await supportedMutatedCandidates(mutated, { limit: 1 });
  if (!selected) {
    throw new Error("GPU move mutation is unsupported for that move.");
  }
  return await engineGpuSearch.engineSnapshotWithGpuChildBoards(snapshot, selected.childBoards, selected.mutationStatus, { move, advanceTurn });
}

async function turnStatusOnGpu(device: GPUDevice, snapshot: GpuSnapshot): Promise<TurnStatus> {
  const boardRecords = await engineGpuSearch.engineTurnStatusRecords(snapshot);
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
  return engineGpuSearch.engineTurnStatusFromWords(result);
}

async function tryFullGpuSearch(
  device: GPUDevice,
  snapshot: GpuSnapshot,
  inputs: GpuCandidateInputs,
  { requestedDepth, nodes, turnStatus, temperature = 0, randomSeed = 0 }: { requestedDepth: number; nodes: number; turnStatus: TurnStatus; temperature?: number; randomSeed?: number }
): Promise<SearchResult> {
  const precondition = await engineGpuSearch.engineFullSearchPrecondition(turnStatus);
  if (!precondition.supported) {
    throw new Error(precondition.error ?? "Full GPU search is not supported for this turn status.");
  }

  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  const ranked = await engineGpuSearch.engineRankedCandidates(scored, {
    requirePending: false,
    limit: await engineGpuSearch.engineGpuSearchRankingLimit(nodes)
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
  for (const entry of await supportedMutatedCandidates(mutated, { limit: await engineGpuSearch.engineGpuSearchValidationLimit(nodes) })) {
    let score = entry.score;
    let principalVariation: Move[][] = [[entry.move]];
    if (requestedDepth > 1 && !await gpuMutationStatusIsTerminal(entry.mutationStatus)) {
      const childSnapshot = await engineGpuSearch.engineSnapshotWithGpuChildBoards(snapshot, entry.childBoards, entry.mutationStatus, { move: entry.move, advanceTurn: true });
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
      depth: await engineGpuSearch.engineGpuFullSearchReportedDepth(requestedDepth),
      nodes: supported.length,
      status: "ok",
      gpu: true,
      gpuMode: "full",
      gpuTerminal: await gpuMutationStatusIsTerminal(entry.mutationStatus),
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
  const inputs = await engineGpuSearch.engineGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    return { score: 0 };
  }
  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  const pendingBoards = await engineGpuSearch.enginePendingPresentBoards(snapshot, snapshot.turn);
  const [best] = await engineGpuSearch.engineRankedCandidates(scored, {
    pendingBoards,
    requirePending: true,
    limit: 1
  });
  return best ? { score: best.score, move: best.move } : { score: 0 };
}

async function selectSearchCandidate<T extends SearchChoice>(candidates: T[], temperature = 0, randomSeed = 0): Promise<(T & { choices: SearchChoice[] }) | null> {
  return engineGpuSearch.engineSelectedSearchChoice<T>({
    temperature,
    randomSeed,
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
  const indexes = await engineGpuSearch.engineSupportedMutationCandidateIndexes(
    candidates,
    options.limit,
    requireChildBoards
  );
  const supported: MutatedCandidate[] = [];
  for (const index of indexes) {
    const candidate = candidates[index];
    if (candidate) {
      supported.push(candidate);
    }
  }
  return supported;
}

async function gpuMutationStatusIsTerminal(status: number): Promise<boolean> {
  return engineGpuSearch.engineGpuMutationStatusIsTerminal(status);
}

async function scoreCandidatesOnGpu(device: GPUDevice, inputs: GpuCandidateInputs, turn: Color): Promise<ScoredCandidates> {
  const engine = await engineGpuSearch.validationEngine();
  const candidateCount = inputs.sourceCount * inputs.targetCount;
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const maxCandidatesPerBatch = engineGpuSearch.engineGpuCandidateMaxCandidatesPerBatch(engine, maxBindingSize);
  if (inputs.targetCount > maxCandidatesPerBatch) {
    throw new Error(`GPU move generation target set is too large for this device (${inputs.targetCount} targets).`);
  }
  const targetBuffer = storageBuffer(device, inputs.targets, GPUBufferUsage.STORAGE);
  const boardBuffer = storageBuffer(device, inputs.boards ?? new Int32Array(GPU_BOARD_STRIDE), GPUBufferUsage.STORAGE);
  const pipeline = await createComputePipelineChecked(device, "score_candidates", GPU_MOVEGEN_SHADER, "score_candidates");
  const records = new Int32Array(candidateCount * GPU_CANDIDATE_STRIDE);
  const scores = new Int32Array(candidateCount);
  const sourceBatchSize = engineGpuSearch.engineGpuCandidateSourceBatchSize(engine, maxCandidatesPerBatch, inputs.targetCount);

  for (let sourceStart = 0; sourceStart < inputs.sourceCount; sourceStart += sourceBatchSize) {
    const sourceCount = engineGpuSearch.engineGpuCandidateBatchSourceCount(engine, inputs.sourceCount, sourceStart, sourceBatchSize);
    const batchCandidateCount = engineGpuSearch.engineGpuCandidateBatchCandidateCount(engine, sourceCount, inputs.targetCount);
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
    pass.dispatchWorkgroups(engineGpuSearch.engineGpuCandidateScoreWorkgroups(engine, batchCandidateCount));
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
  const engine = await engineGpuSearch.validationEngine();
  const limit = engineGpuSearch.engineGpuMutationCandidateLimit(engine, ranked.length);
  if (limit === 0 || !inputs.mutationBoards || inputs.boardCount === 0) {
    return [];
  }
  const selected = await engineGpuSearch.engineMutationSelectedCandidates(ranked, limit);
  const candidateRecords = await engineGpuSearch.enginePickCandidateRecords(allCandidateRecords, await engineGpuSearch.engineCandidateIndexes(selected));
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
  view.setUint32(8, await engineGpuSearch.engineGpuMutationTurnCode(candidateRecords), true);
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
  pass.dispatchWorkgroups(engineGpuSearch.engineGpuMutationCandidateWorkgroups(engine, limit));
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
  const mutationStatuses = await engineGpuSearch.engineGpuMutationStatuses(statuses, selected.length);
  return selected.map((entry, index) => ({
    ...entry,
    mutationStatus: mutationStatuses[index]!,
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
  const engine = await engineGpuSearch.validationEngine();
  const replyLimit = engineGpuSearch.engineGpuReplyPressureReplyLimit(engine);
  const rankedReplies = await engineGpuSearch.engineRankedCandidates(
    { records: allReplyRecords, scores: allReplyScores },
    { requirePending: false, limit: replyLimit }
  );
  if (rankedReplies.length === 0) {
    return rankedRoots;
  }

  const rootRecords = await engineGpuSearch.enginePickCandidateRecords(allRootRecords, await engineGpuSearch.engineCandidateIndexes(rankedRoots));
  const replyRecords = await engineGpuSearch.enginePickCandidateRecords(allReplyRecords, await engineGpuSearch.engineCandidateIndexes(rankedReplies));
  const rootScores = await engineGpuSearch.engineCandidateScores(allRootScores, rankedRoots, -2147483647);
  const replyScores = await engineGpuSearch.engineCandidateScores(allReplyScores, rankedReplies, -2147483647);
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
    engineGpuSearch.engineGpuReplyScoreWorkgroupsX(engine, rankedRoots.length),
    engineGpuSearch.engineGpuReplyScoreWorkgroupsY(engine, rankedReplies.length)
  );
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const pairScores = await readInts(device, pairBuffer, pairCount * Int32Array.BYTES_PER_ELEMENT);

  return engineGpuSearch.engineReplyPressureRankedRoots(rankedRoots, pairScores, rankedReplies.length);
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
      const selection = await engineGpuSearch.engineLegalTargets(clientGame, position);
      self.postMessage({ id, ok: true, selection });
      return;
    }

    if (clientGame && type === "applyMove") {
      if (!move) {
        throw new Error("GPU move request is missing a move.");
      }
      const game = await engineGpuSearch.engineApplyMove(clientGame, move);
      self.postMessage({ id, ok: true, game });
      return;
    }

    if (clientGame && type === "submitTurn") {
      const status = await engineGpuSearch.engineSubmitTurn(clientGame);
      self.postMessage({ id, ok: true, status });
      return;
    }

    if (type === "legalTargets" || type === "applyMove" || type === "submitTurn") {
      throw new Error("GPU worker rules commands require a client game snapshot.");
    }

    if (!clientGame) {
      throw new Error("GPU worker calculations require a client game snapshot.");
    }

    const snapshotOverride = await engineGpuSearch.engineGpuSnapshot(clientGame);
    if (!snapshotOverride) {
      throw new Error("GPU worker calculations require a client game snapshot.");
    }

    const searchConfig = await engineGpuSearch.engineGpuWorkerSearchConfig(depth, minDepth, timeMs);
    engineGpuSearch.setGpuSearchDeadline(searchConfig.deadlineDelayMs == null
      ? Number.POSITIVE_INFINITY
      : Date.now() + searchConfig.deadlineDelayMs);
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
      if (gpuResult && await engineGpuSearch.enginePostableSearchResult(gpuResult)) {
        const validatedResult = await engineGpuSearch.validateSearchResultBeforePost(snapshotOverride, gpuResult, clientGame);
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
        throw new Error(`GPU search produced a non-postable result (${await engineGpuSearch.engineNonPostableResultSummary(gpuResult)})`);
      }
    } catch (gpuError) {
      console.debug?.("GPU search failed", gpuError);
      if (gpuMode === "full") {
        try {
          const hybridResult = await tryGpuSearch({
            depth: searchConfig.requestedDepth,
            nodes,
            timeMs: searchConfig.searchTimeMs,
            gpuMode: "hybrid",
            disableNeural,
            snapshotOverride,
            sourceGame: clientGame,
            temperature,
            randomSeed
          });
          if (hybridResult && await engineGpuSearch.enginePostableSearchResult(hybridResult)) {
            const validatedResult = await engineGpuSearch.validateSearchResultBeforePost(snapshotOverride, hybridResult, clientGame);
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
    return await engineGpuSearch.engineGpuSearchFailureSummary(snapshot);
  } catch (error) {
    return `summary failed: ${errorMessage(error)}`;
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
