import { GPU_CANDIDATE_STRIDE, GPU_SOURCE_STRIDE, GPU_TARGET_STRIDE, GPU_BOARD_STRIDE, GPU_MUTATION_BOARD_STRIDE, GPU_MUTATION_CHILD_STRIDE, GPU_MUTATION_STATUS_OK, GPU_MUTATION_STATUS_ROYAL_CAPTURE, GPU_MUTATION_STATUS_BRANCH_OK, GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE, GPU_TURN_STATUS_RECORD_STRIDE } from "./ai-layout.js";
import { readGpuSnapshot, buildGpuCandidateInputsFromSnapshot, snapshotWithGpuChildBoards, originForGpuChild, gpuMutationBoardRecordToSnapshot, gpuSnapshotToGame, gpuBoardToGameBoard, squaresToGameBoard, pieceFromCode, buildGpuCandidateInputs, squareCodesForBoard, pushGpuBoardRecord, pushGpuMutationBoardRecord, colorFromCode, ownerCode, moveFromCandidateRecord, oppositeColor, sortedTimelines, latestBoard, presentTimeForSnapshot, capitalize, pieceTypeCode, pieceTypeFromCode, colorCode } from "./ai-snapshot.js";
import { GPU_TURN_STATUS_SHADER, GPU_MOVEGEN_SHADER, GPU_REPLY_SHADER, GPU_MUTATE_SHADER } from "./ai-shaders.js";
import type { Color, GameSnapshot, Move, Piece, Position, Timeline } from "./types.js";
import type { GpuCandidateInputs, GpuSnapshot, GpuTimeline } from "./ai-snapshot.js";

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

const GPUBufferUsage: GpuBufferUsageConstants = (globalThis as unknown as { GPUBufferUsage?: GpuBufferUsageConstants }).GPUBufferUsage ?? {
  MAP_READ: 1,
  COPY_SRC: 4,
  COPY_DST: 8,
  UNIFORM: 64,
  STORAGE: 128
};
const GPUMapMode: GpuMapModeConstants = (globalThis as unknown as { GPUMapMode?: GpuMapModeConstants }).GPUMapMode ?? {
  READ: 1
};

type GpuMode = "full" | "hybrid";

interface GpuSearchOptions {
  depth?: number | undefined;
  nodes?: number | undefined;
  timeMs?: number | undefined;
  gpuMode?: GpuMode | undefined;
  snapshotOverride?: GpuSnapshot | null | undefined;
  temperature?: number | undefined;
  randomSeed?: number | undefined;
}

interface TurnStatus {
  complete: boolean;
  terminal?: boolean;
  winner?: Color;
  nextTurn: Color;
  presentTime: number;
  pendingPresentBoardCount: number;
  message?: string;
}

interface RankedCandidate {
  move: Move;
  index: number;
  score: number;
}

interface MutatedCandidate extends RankedCandidate {
  mutationStatus: number;
  childBoards: Int32Array | null;
}

interface ScoredCandidates {
  records: Int32Array;
  scores: Int32Array;
}

interface SearchChoice {
  rank?: number;
  score?: number | undefined;
  moves?: Move[] | undefined;
  move?: Move | undefined;
  depth?: number | undefined;
  nodes?: number | undefined;
  gpuSearch?: string | undefined;
}

interface SearchResult {
  status: string;
  moves: Move[];
  score?: number | undefined;
  choices?: SearchChoice[] | undefined;
  principalVariation?: Move[][] | undefined;
  depth?: number | undefined;
  nodes?: number | undefined;
  gpu?: boolean | undefined;
  gpuMode?: GpuMode | undefined;
  gpuTerminal?: boolean | undefined;
  gpuSnapshot?: string | undefined;
  gpuSearch?: string | undefined;
  incompleteMoves?: Move[] | undefined;
  pendingPresentBoardCount?: number | undefined;
}

interface LegalTargetSelection {
  source: { piece: Piece; position: Position } | null;
  targets: Position[];
}

interface WorkerRequest {
  id: number | string;
  type?: "search" | "legalTargets" | "applyMove" | "submitTurn";
  game?: GameSnapshot;
  position?: Position;
  move?: Move;
  depth?: number;
  nodes?: number;
  timeMs?: number;
  partitionIndex?: number;
  partitionCount?: number;
  temperature?: number;
  randomSeed?: number;
  gpuMode?: GpuMode;
  notation?: string;
  turns?: Move[][];
  stagedMoves?: Move[];
}

let cachedGpuAdapter: GPUAdapter | null = null;
let cachedGpuDevice: GPUDevice | null = null;
const pipelineCache = new Map<string, GPUComputePipeline>();

async function tryGpuSearch({ depth, nodes, timeMs, gpuMode = "hybrid", snapshotOverride = null, temperature = 0, randomSeed = 0 }: GpuSearchOptions): Promise<SearchResult | null> {
  if (!navigator.gpu) {
    return null;
  }
  const requestedDepth = Math.max(1, depth ?? 1);
  const snapshot = snapshotOverride ?? readGpuSnapshot();
  if (!snapshot) {
    return null;
  }
  const candidates = buildGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (candidates.sourceCount === 0 || candidates.targetCount === 0) {
    return null;
  }

  const device = await getGpuDevice();
  if (!device) {
    return null;
  }
  const turnStatus = await turnStatusOnGpu(device, snapshot);
  if (gpuMode === "full" && turnStatus.pendingPresentBoardCount === 1) {
    try {
      const result = await tryFullGpuSearch(device, snapshot, candidates, { requestedDepth, nodes: nodes ?? 64, turnStatus, temperature, randomSeed });
      return completeGpuResultTurn(device, snapshot, result, { nodes: nodes ?? 64, temperature, randomSeed });
    } catch (error) {
      console.warn("Full GPU search failed; falling back to hybrid GPU search.", error);
    }
  }
  const scored = await scoreCandidatesOnGpu(device, candidates, snapshot.turn);
  let ranked = Array.from(scored.scores, (score, index) => ({
    move: moveFromCandidateRecord(scored.records, index),
    index,
    score: score ?? -2147483647
  }))
    .filter((entry) => entry.score > -2147480000)
    .sort((left, right) => right.score - left.score)
    .slice(0, Math.min(128, Math.max(16, nodes ?? 64)));

  if (requestedDepth > 1) {
    const result = await searchSingleMoveRepliesOnGpu(device, snapshot, candidates, scored.records, ranked, {
      requestedDepth,
      nodes: nodes ?? 64,
      temperature,
      randomSeed
    });
    return completeGpuResultTurn(device, snapshot, result, { nodes: nodes ?? 64, temperature, randomSeed });
  }

  if (turnStatus.pendingPresentBoardCount >= 1 && ranked.length > 0) {
    const mutated = await mutateRankedCandidatesOnGpu(device, candidates, scored.records, ranked);
    const selected = selectSearchCandidate(
      mutated.filter((entry) => entry.mutationStatus >= GPU_MUTATION_STATUS_OK),
      temperature,
      randomSeed
    );
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
    if (entry.mutationStatus !== GPU_MUTATION_STATUS_ROYAL_CAPTURE && entry.mutationStatus !== GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
      const childSnapshot = snapshotWithGpuChildBoards(snapshot, entry.childBoards, entry.mutationStatus, { move: entry.move, advanceTurn: true });
      score -= await bestReplyScoreOnGpu(device, childSnapshot, nodes);
    }
    const candidate: SearchResult = {
      moves: [entry.move],
      score,
      principalVariation: [[entry.move]],
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
  { nodes, temperature = 0, randomSeed = 0 }: { nodes?: number; temperature?: number; randomSeed?: number } = {}
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
    const pendingBoards = pendingPresentBoardsForSnapshot(current, rootTurn);
    if ((status.complete || status.pendingPresentBoardCount === 0) && pendingBoards.length === 0) {
      break;
    }
    const stepSnapshot = { ...current, turn: rootTurn };
    const inputs = buildGpuCandidateInputsFromSnapshot(stepSnapshot, rootTurn);
    if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
      break;
    }
    const scored = await scoreCandidatesOnGpu(device, inputs, rootTurn);
    const ranked = Array.from(scored.scores, (score, index) => ({
      move: moveFromCandidateRecord(scored.records, index),
      index,
      score: score ?? -2147483647
    }))
      .filter((entry) => entry.score > -2147480000)
      .sort((left, right) => right.score - left.score)
      .slice(0, Math.min(128, Math.max(16, nodes ?? 64)));
    if (ranked.length === 0) {
      break;
    }
    const mutated = await mutateRankedCandidatesOnGpu(device, inputs, scored.records, ranked, { readChildren: true });
    extraNodes += mutated.length;
    const selected = selectSearchCandidate(
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
  const finalPendingBoards = pendingPresentBoardsForSnapshot(current, rootTurn);
  if (finalPendingBoards.length > 0 || (!finalStatus.complete && finalStatus.pendingPresentBoardCount > 0)) {
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
      choices: summarizeSearchChoices([{
        ...result,
        moves,
        gpuSearch: `${result.gpuSearch ?? "gpu"}-turn-incomplete`
      }])
    };
  }

  return withCompletedTurnChoice({
    ...result,
    nodes: (result.nodes ?? 0) + extraNodes
  }, moves, moves.length > result.moves.length ? `${result.gpuSearch ?? "gpu"}-turn-complete` : result.gpuSearch);
}

function withCompletedTurnChoice(result: SearchResult, moves: Move[], gpuSearch = result.gpuSearch): SearchResult {
  const completedChoice = {
    rank: 1,
    score: result.score,
    moves,
    depth: result.depth,
    nodes: result.nodes,
    gpuSearch
  };
  const existingChoices = Array.isArray(result.choices) ? result.choices : [];
  return {
    ...result,
    moves,
    gpuSearch,
    principalVariation: result.principalVariation ?? [moves],
    choices: [
      completedChoice,
      ...existingChoices
        .filter((choice) => !sameMoveSequence(choice.moves ?? [], moves))
        .slice(0, 11)
    ]
  };
}

function pendingPresentBoardsForSnapshot(snapshot: GpuSnapshot, color: Color): Array<{ timeline: GpuTimeline | Timeline; board: { time: number; sideToMove: Color } }> {
  const present = activePresentTimeForSnapshot(snapshot);
  if (present === null) {
    return [];
  }
  const pending: Array<{ timeline: GpuTimeline | Timeline; board: { time: number; sideToMove: Color } }> = [];
  for (const timeline of sortedTimelines(snapshot)) {
    if (!isActiveSnapshotTimeline(snapshot, timeline)) {
      continue;
    }
    const board = latestBoard(timeline);
    if (board && board.time === present && board.sideToMove === color) {
      pending.push({ timeline, board });
    }
  }
  return pending;
}

function activePresentTimeForSnapshot(snapshot: GpuSnapshot): number | null {
  let present: number | null = null;
  for (const timeline of sortedTimelines(snapshot)) {
    if (!isActiveSnapshotTimeline(snapshot, timeline)) {
      continue;
    }
    const board = latestBoard(timeline);
    if (!board) {
      continue;
    }
    if (present === null || board.time < present) {
      present = board.time;
    }
  }
  return present;
}

function isActiveSnapshotTimeline(snapshot: GpuSnapshot, timeline: GpuTimeline | Timeline): boolean {
  if (timeline.owner === "neutral") {
    return true;
  }
  const ids = snapshot.timelines.map((candidate) => candidate.id);
  const minTimeline = Math.min(...ids, 0);
  const maxTimeline = Math.max(...ids, 0);
  const activeDistance = Math.max(0, Math.min(-minTimeline, maxTimeline)) + 1;
  return Math.abs(timeline.id) <= activeDistance;
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

async function legalTargetsOnGpu(position: Position, snapshotOverride: GpuSnapshot | null = null): Promise<LegalTargetSelection> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable for legal target calculation.");
  }
  const snapshot = snapshotOverride ?? readGpuSnapshot();
  if (!snapshot) {
    throw new Error("GPU snapshot is unavailable.");
  }
  const inputs = buildGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    return { source: null, targets: [] };
  }

  const device = await getGpuDevice();
  if (!device) {
    throw new Error("No WebGPU adapter is available.");
  }
  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  const targets: Position[] = [];
  const seen = new Set<string>();
  let source: LegalTargetSelection["source"] = null;

  for (let index = 0; index < scored.scores.length; index += 1) {
    const score = scored.scores[index] ?? -2147483647;
    if (score <= -2147480000) {
      continue;
    }
    const offset = index * GPU_CANDIDATE_STRIDE;
    if (
      scored.records[offset + 11] !== position.timelineId ||
      scored.records[offset + 12] !== position.time ||
      scored.records[offset + 13] !== position.x ||
      scored.records[offset + 14] !== position.y
    ) {
      continue;
    }
    const sourceType = pieceTypeFromCode(scored.records[offset + 0] ?? 0);
    if (sourceType) {
      source ??= {
        piece: {
          type: sourceType,
          color: colorFromCode(scored.records[offset + 1] ?? 0)
        },
        position: { ...position }
      };
    }
    const target = {
      timelineId: scored.records[offset + 15] ?? 0,
      time: scored.records[offset + 16] ?? 0,
      x: scored.records[offset + 17] ?? 0,
      y: scored.records[offset + 18] ?? 0
    };
    const key = `${target.timelineId}:${target.time}:${target.x}:${target.y}`;
    if (!seen.has(key)) {
      seen.add(key);
      targets.push(target);
    }
  }

  targets.sort((left, right) =>
    left.timelineId - right.timelineId ||
    left.time - right.time ||
    left.y - right.y ||
    left.x - right.x
  );
  return { source, targets };
}

async function applyMoveOnGpu(move: Move, snapshotOverride: GpuSnapshot | null = null): Promise<GameSnapshot> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable for move application.");
  }
  const snapshot = snapshotOverride ?? readGpuSnapshot();
  if (!snapshot) {
    throw new Error("GPU snapshot is unavailable.");
  }
  const device = await getGpuDevice();
  if (!device) {
    throw new Error("No WebGPU adapter is available.");
  }
  const nextSnapshot = await applyGpuMoveToSnapshot(device, snapshot, move, { advanceTurn: false });
  return gpuSnapshotToGame(nextSnapshot);
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
  const index = findCandidateIndex(scored, move);
  if (index < 0 || (scored.scores[index] ?? -2147483647) <= -2147480000) {
    throw new Error("GPU rejected that move.");
  }
  const candidateRecords = pickCandidateRecords(scored.records, [index]);
  const ranked = [{ move, index: 0, score: scored.scores[index] ?? 0 }];
  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, candidateRecords, ranked, { readChildren: true });
  const selected = mutated[0];
  if (!selected || selected.mutationStatus < GPU_MUTATION_STATUS_OK || !selected.childBoards) {
    throw new Error("GPU move mutation is unsupported for that move.");
  }
  return snapshotWithGpuChildBoards(snapshot, selected.childBoards, selected.mutationStatus, { move, advanceTurn });
}

async function submitTurnOnGpu(snapshotOverride: GpuSnapshot | null = null): Promise<TurnStatus> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable for turn submission.");
  }
  const snapshot = snapshotOverride ?? readGpuSnapshot();
  if (!snapshot) {
    throw new Error("GPU snapshot is unavailable.");
  }
  if (snapshot.royalCaptureBy) {
    return {
      complete: true,
      terminal: true,
      winner: snapshot.royalCaptureBy,
      nextTurn: snapshot.turn,
      presentTime: presentTimeForSnapshot(snapshot) ?? 0,
      pendingPresentBoardCount: 0,
      message: `${capitalize(snapshot.royalCaptureBy)} wins by royal capture.`
    };
  }
  const device = await getGpuDevice();
  if (!device) {
    throw new Error("No WebGPU adapter is available.");
  }
  return turnStatusOnGpu(device, snapshot);
}

async function turnStatusOnGpu(device: GPUDevice, snapshot: GpuSnapshot): Promise<TurnStatus> {
  const records: number[] = [];
  for (const timeline of sortedTimelines(snapshot)) {
    const board = latestBoard(timeline);
    if (!board) {
      continue;
    }
    records.push(
      timeline.id,
      ownerCode(timeline.owner),
      board.time,
      colorCode(board.sideToMove)
    );
  }
  const boardRecords = new Int32Array(records.length > 0 ? records : [0, 0, 0, colorCode(snapshot.turn)]);
  const boardBuffer = storageBuffer(device, boardRecords, GPUBufferUsage.STORAGE);
  const resultBuffer = device.createBuffer({
    size: align4(4 * Int32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, records.length / GPU_TURN_STATUS_RECORD_STRIDE, true);
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

function findCandidateIndex(scored: ScoredCandidates, move: Move): number {
  for (let index = 0; index < scored.scores.length; index += 1) {
    const offset = index * GPU_CANDIDATE_STRIDE;
    if (
      scored.records[offset + 11] === move.from.timelineId &&
      scored.records[offset + 12] === move.from.time &&
      scored.records[offset + 13] === move.from.x &&
      scored.records[offset + 14] === move.from.y &&
      scored.records[offset + 15] === move.to.timelineId &&
      scored.records[offset + 16] === move.to.time &&
      scored.records[offset + 17] === move.to.x &&
      scored.records[offset + 18] === move.to.y
    ) {
      return index;
    }
  }
  return -1;
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
  const ranked = Array.from(scored.scores, (score, index) => ({
    move: moveFromCandidateRecord(scored.records, index),
    index,
    score: score ?? -2147483647
  }))
    .filter((entry) => entry.score > -2147480000)
    .sort((left, right) => right.score - left.score)
    .slice(0, Math.min(128, Math.max(16, nodes ?? 64)));
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
    if (requestedDepth > 1 && entry.mutationStatus !== GPU_MUTATION_STATUS_ROYAL_CAPTURE && entry.mutationStatus !== GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
      const childSnapshot = snapshotWithGpuChildBoards(snapshot, entry.childBoards, entry.mutationStatus, { move: entry.move, advanceTurn: true });
      const replyScore = await bestReplyScoreOnGpu(device, childSnapshot, nodes);
      score -= replyScore;
    }
    const candidate: SearchResult = {
      moves: [entry.move],
      score,
      principalVariation: [[entry.move]],
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
  const selected = selectSearchCandidate(candidates, temperature, randomSeed);
  if (!selected) {
    throw new Error("Full GPU search produced no legal result.");
  }
  return selected;
}

async function bestReplyScoreOnGpu(device: GPUDevice, snapshot: GpuSnapshot, nodes: number): Promise<number> {
  const inputs = buildGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    return 0;
  }
  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  let best = 0;
  for (let index = 0; index < scored.scores.length; index += 1) {
    const score = scored.scores[index] ?? -2147483647;
    if (score > best) {
      best = score;
    }
  }
  return best;
}

function selectSearchCandidate<T extends SearchChoice>(candidates: T[], temperature = 0, randomSeed = 0): (T & { choices: SearchChoice[] }) | null {
  const supported = candidates
    .filter(hasMovesAndScore)
    .sort((left, right) => {
      const score = right.score - left.score;
      if (score !== 0) {
        return score;
      }
      const leftKey = turnPlanKey(left.moves);
      const rightKey = turnPlanKey(right.moves);
      if (leftKey === rightKey) {
        return 0;
      }
      return leftKey < rightKey ? -1 : 1;
    });
  if (supported.length === 0) {
    return null;
  }
  const temp = Number(temperature) || 0;
  if (temp <= 0) {
    const first = supported[0];
    return first ? withSearchChoices(first, supported) as T & { choices: SearchChoice[] } : null;
  }

  const candidateLimit = Math.min(32, supported.length);
  const top = supported.slice(0, candidateLimit);
  const maxScore = top[0]?.score ?? 0;
  const scoreScale = Math.max(1, temp * 100);
  const weights = top.map((candidate) => Math.exp(Math.max(-50, Math.min(0, (candidate.score - maxScore) / scoreScale))));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  let pick = seededUnit(randomSeed) * total;
  for (let index = 0; index < top.length; index += 1) {
    pick -= weights[index] ?? 0;
    if (pick <= 0) {
      const selected = top[index];
      return selected ? withSearchChoices(selected, supported) as T & { choices: SearchChoice[] } : null;
    }
  }
  const fallback = top.at(-1);
  return fallback ? withSearchChoices(fallback, supported) as T & { choices: SearchChoice[] } : null;
}

function hasSupportedChildBoards(entry: MutatedCandidate): entry is MutatedCandidate & { childBoards: Int32Array } {
  return entry.mutationStatus >= GPU_MUTATION_STATUS_OK && Boolean(entry.childBoards);
}

function hasMovesAndScore<T extends SearchChoice>(candidate: T): candidate is T & { moves: Move[]; score: number } {
  return Boolean(candidate.moves?.length) && Number.isFinite(candidate.score);
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
      moves: candidate.moves ?? (candidate.move ? [candidate.move] : []),
      depth: candidate.depth,
      nodes: candidate.nodes,
      gpuSearch: candidate.gpuSearch
    }));
}

function seededUnit(seed: number): number {
  let state = (Number(seed) || 0) >>> 0;
  state ^= state << 13;
  state ^= state >>> 17;
  state ^= state << 5;
  return ((state >>> 0) || 1) / 0xffffffff;
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
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const maxCandidatesPerBatch = Math.max(1, Math.floor(maxBindingSize / (GPU_CANDIDATE_STRIDE * Int32Array.BYTES_PER_ELEMENT)));
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
    pass.dispatchWorkgroups(Math.ceil(batchCandidateCount / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const [batchRecords, batchScores] = await Promise.all([
      readInts(device, candidateBuffer, batchCandidateCount * GPU_CANDIDATE_STRIDE * Int32Array.BYTES_PER_ELEMENT),
      readInts(device, scoreBuffer, batchCandidateCount * Int32Array.BYTES_PER_ELEMENT)
    ]);
    const candidateOffset = sourceStart * inputs.targetCount;
    records.set(batchRecords, candidateOffset * GPU_CANDIDATE_STRIDE);
    scores.set(batchScores, candidateOffset);
  }

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
  const candidateRecords = pickCandidateRecords(allCandidateRecords, selected.map((entry) => entry.index));
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
  const [statuses, childBoards] = await Promise.all([
    readInts(device, statusBuffer, limit * Int32Array.BYTES_PER_ELEMENT),
    readChildren
      ? readInts(device, childBoardBuffer, limit * GPU_MUTATION_CHILD_STRIDE * Int32Array.BYTES_PER_ELEMENT)
      : Promise.resolve(null)
  ]);
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
  const rankedReplies = Array.from(allReplyScores, (score, index) => ({ index, score }))
    .filter((entry) => entry.score > -2147480000)
    .sort((left, right) => right.score - left.score)
    .slice(0, replyLimit);
  if (rankedReplies.length === 0) {
    return rankedRoots;
  }

  const rootRecords = pickCandidateRecords(allRootRecords, rankedRoots.map((entry) => entry.index));
  const replyRecords = pickCandidateRecords(allReplyRecords, rankedReplies.map((entry) => entry.index));
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

  return rankedRoots
    .map((entry, rootIndex) => {
      let maxPressure = 0;
      const offset = rootIndex * rankedReplies.length;
      for (let replyIndex = 0; replyIndex < rankedReplies.length; replyIndex += 1) {
        maxPressure = Math.max(maxPressure, pairScores[offset + replyIndex] ?? -2147483647);
      }
      return { ...entry, score: entry.score - maxPressure };
    })
    .sort((left, right) => right.score - left.score);
}

function pickCandidateRecords(records: Int32Array, indices: number[]): Int32Array {
  const picked = new Int32Array(indices.length * GPU_CANDIDATE_STRIDE);
  for (let outputIndex = 0; outputIndex < indices.length; outputIndex += 1) {
    const sourceOffset = (indices[outputIndex] ?? 0) * GPU_CANDIDATE_STRIDE;
    picked.set(
      records.subarray(sourceOffset, sourceOffset + GPU_CANDIDATE_STRIDE),
      outputIndex * GPU_CANDIDATE_STRIDE
    );
  }
  return picked;
}

function storageBuffer(device: GPUDevice, data: ArrayBuffer | ArrayBufferView, usage: number): GPUBuffer {
  const bytes = data instanceof ArrayBuffer
    ? data
    : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const buffer = device.createBuffer({
    size: align4(bytes.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
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
    cachedGpuDevice = null;
    pipelineCache.clear();
  });
  return cachedGpuDevice;
}

async function requestHighLimitDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  const requiredLimits: Record<string, number> = {};
  for (const key of ["maxStorageBufferBindingSize", "maxBufferSize"] as const) {
    const value = adapter.limits[key];
    if (Number.isFinite(value) && value > 0) {
      requiredLimits[key] = value;
    }
  }
  if (Object.keys(requiredLimits).length === 0) {
    return adapter.requestDevice();
  }
  try {
    return await adapter.requestDevice({ requiredLimits });
  } catch {
    return adapter.requestDevice();
  }
}

async function createComputePipelineChecked(device: GPUDevice, label: string, code: string, entryPoint: string): Promise<GPUComputePipeline> {
  const cacheKey = `${label}:${entryPoint}`;
  const cached = pipelineCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const module = device.createShaderModule({ label: `${label}.module`, code });
  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message: GPUCompilationMessage) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(formatShaderErrors(label, errors));
    }
  }
  const pipeline = device.createComputePipeline({
    label,
    layout: "auto",
    compute: { module, entryPoint }
  });
  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

function formatShaderErrors(label: string, errors: GPUCompilationMessage[]): string {
  return `${label} shader compilation failed: ${errors.map((error) =>
    `line ${error.lineNum ?? "?"}, column ${error.linePos ?? "?"}: ${error.message}`
  ).join("; ")}`;
}

async function readInts(device: GPUDevice, buffer: GPUBuffer, byteLength: number): Promise<Int32Array> {
  const readBuffer = device.createBuffer({
    size: align4(byteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const copy = new Int32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  return copy;
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
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
    nodes,
    timeMs,
    partitionIndex,
    partitionCount,
    temperature = 0,
    randomSeed = 0,
    gpuMode = "hybrid"
  } = event.data;

  try {
    const snapshotOverride = clientGame ? { ...clientGame, format: "json" } : null;
    if (!snapshotOverride) {
      throw new Error("GPU worker calculations require a client game snapshot.");
    }

    if (type === "legalTargets") {
      if (!position) {
        throw new Error("GPU legal target request is missing a source position.");
      }
      const selection = await legalTargetsOnGpu(position, snapshotOverride);
      self.postMessage({ id, ok: true, selection });
      return;
    }

    if (type === "applyMove") {
      if (!move) {
        throw new Error("GPU move request is missing a move.");
      }
      const game = await applyMoveOnGpu(move, snapshotOverride);
      self.postMessage({ id, ok: true, game });
      return;
    }

    if (type === "submitTurn") {
      const status = await submitTurnOnGpu(snapshotOverride);
      self.postMessage({ id, ok: true, status });
      return;
    }

    const searchTimeMs = Math.max(1, timeMs ?? 10_000);
    gpuDeadlineAt = Date.now() + Math.max(1, Math.floor(searchTimeMs * 0.8));
    try {
      const gpuResult = await tryGpuSearch({ depth, nodes, timeMs: searchTimeMs, gpuMode, snapshotOverride, temperature, randomSeed });
      if (isPostableSearchResult(gpuResult)) {
        self.postMessage({ id, ok: true, result: gpuResult, partitionIndex: partitionIndex ?? 0 });
        return;
      }
    } catch (gpuError) {
      console.debug?.("GPU search failed", gpuError);
      if (gpuMode === "full") {
        try {
          const hybridResult = await tryGpuSearch({ depth, nodes, timeMs: searchTimeMs, gpuMode: "hybrid", snapshotOverride, temperature, randomSeed });
          if (isPostableSearchResult(hybridResult)) {
            self.postMessage({ id, ok: true, result: hybridResult, partitionIndex: partitionIndex ?? 0 });
            return;
          }
        } catch (hybridError) {
          console.debug?.("Hybrid GPU search failed", hybridError);
        }
      }
      throw gpuError;
    }

    throw new Error("GPU search did not produce a legal turn.");
  } catch (error) {
    self.postMessage({ id, ok: false, error: errorMessage(error), partitionIndex: partitionIndex ?? 0 });
  }
});

function isPostableSearchResult(result: SearchResult | null): result is SearchResult {
  return Boolean(
    (result?.status === "ok" && result.moves?.length)
    || result?.status === "incompleteTurn"
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
