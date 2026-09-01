import { GPU_CANDIDATE_STRIDE, GPU_FRONTIER_HEADER_BOARD_COUNT, GPU_FRONTIER_HEADER_HASH_HIGH, GPU_FRONTIER_HEADER_HASH_LOW, GPU_SOURCE_STRIDE, GPU_TARGET_STRIDE } from "./ai-layout.js";
import type { EncodedFrontierRoot, FrontierTuning } from "./ai-frontier.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import { readWasmBytes, readWasmString, writeWasmBytes, writeWasmString } from "./engine-io.js";
import type { ChronofishEngine, Color, GameSnapshot, Move, Position } from "./types.js";
import type { GpuCandidateInputs, GpuSnapshot } from "./ai-snapshot.js";
import type { LegalTargetSelection, MutatedCandidate, RankedCandidate, ScoredCandidates, SearchChoice, SearchResult, TurnStatus } from "./ai-worker-types.js";

let validationEnginePromise: Promise<ChronofishEngine> | null = null;
let gpuDeadlineAt = 0;
const GPU_CANDIDATE_INPUT_HEADER_I32S = 7;

export interface PendingBoardRef {
  timelineId: number;
  time: number;
}

export interface GpuWorkerSearchConfig {
  requestedDepth: number;
  minimumDepth: number;
  searchTimeMs: number;
  deadlineDelayMs: number | null;
}

export interface GpuSnapshotSearchSize {
  boardCount: number;
  timelineCount: number;
}

export interface GpuFrontierReadbackSummary {
  nodes: number;
  selectedCount: number;
  candidateOverflow: boolean;
  tacticalCandidates: number;
  selectedTacticalCandidates: number;
}

export interface GpuPolicyChoiceAgreementDiagnostics {
  topPolicyChoiceAgreement: number;
  top5PolicyChoiceAgreement: number;
  top20PolicyChoiceAgreement: number;
}

export interface GpuTurnCompletionStep {
  action: "terminal" | "complete" | "maxMoves" | "loop" | "search";
  stateKey?: string | null;
  maxMoves: number;
}

export interface GpuFullSearchPrecondition {
  supported: boolean;
  error?: string | null;
}

export function engineGpuSearchColorCode(engine: ChronofishEngine, color: string): number {
  const { ptr, len } = writeWasmString(engine, color);
  try {
    const output = engine.chronofish_gpu_search_color_code_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return Number.parseInt(readWasmString(engine, output), 10);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineDeriveFrontierTuning<T>(limits: number[], requestedNodes: number, boardCount: number, additionalBoardCapacity: number): Promise<T> {
  const engine = await validationEngine();
  const output = engine.chronofish_derive_frontier_tuning_json(
    limits[0] ?? 0,
    limits[1] ?? 0,
    limits[2] ?? 0,
    requestedNodes,
    boardCount,
    additionalBoardCapacity
  );
  return JSON.parse(readWasmString(engine, output)) as T;
}

export async function engineFrontierSelectionPlan<T>(values: number[]): Promise<T> {
  const engine = await validationEngine();
  const output = engine.chronofish_frontier_selection_plan_json(
    values[0] ?? 0,
    values[1] ?? 0,
    values[2] ?? 0,
    values[3] ?? 0,
    values[4] ?? 0,
    values[5] ?? 0,
    values[6] ?? 0,
    values[7] ?? 0
  );
  return JSON.parse(readWasmString(engine, output)) as T;
}

export function engineFrontierNumber(engine: ChronofishEngine, operation: string, ...args: number[]): number {
  const callback = (engine as unknown as Record<string, (...values: number[]) => number>)[operation];
  if (!callback) {
    throw new Error(`GPU search engine does not export ${operation}.`);
  }
  return callback(...args);
}

export function engineGpuSearchBytesRequired(engine: ChronofishEngine, operation: string, input: Uint8Array): Uint8Array {
  const { ptr, len } = writeWasmBytes(engine, input);
  try {
    const output = engineFrontierNumber(engine, operation, ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmBytes(engine, output);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export function engineGpuSearchOutputBytes(engine: ChronofishEngine, operation: string, ...args: number[]): Uint8Array {
  const output = engineFrontierNumber(engine, operation, ...args);
  if (!output) {
    throw new Error(readWasmString(engine, engine.chronofish_last_message()));
  }
  return readWasmBytes(engine, output);
}

export interface ValidatedFrontierChoiceResponse {
  accepted: boolean;
  key?: string | null;
  choice?: SearchResult | null;
}

export async function engineValidatedFrontierChoice(
  candidate: SearchResult,
  moves: Move[],
  seenKeys: string[],
  choiceCount: number,
  choiceLimit: number,
  gpuSearch: string
): Promise<ValidatedFrontierChoiceResponse> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    candidate,
    moves,
    seenKeys,
    choiceCount,
    choiceLimit,
    gpuSearch
  }));
  try {
    const output = engine.chronofish_gpu_validated_frontier_choice_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as ValidatedFrontierChoiceResponse;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export function engineFrontierChoicesFromWords(
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

export async function validateFirstFrontierTurn(snapshot: GpuSnapshot, plan: Move[], sourceGame?: GameSnapshot | undefined): Promise<Move[]> {
  const engine = await validationEngine();
  const game = await validationGameSnapshot(snapshot, sourceGame);
  const { ptr, len } = writeWasmString(engine, JSON.stringify({ game, moves: plan }));
  try {
    const output = engine.chronofish_gpu_validate_first_frontier_turn_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as Move[];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function validateSearchResultBeforePost(snapshot: GpuSnapshot, result: SearchResult, sourceGame?: GameSnapshot | undefined): Promise<SearchResult | null> {
  const engine = await validationEngine();
  const game = await validationGameSnapshot(snapshot, sourceGame);
  const { ptr, len } = writeWasmString(engine, JSON.stringify({ game, result }));
  try {
    const output = engine.chronofish_gpu_validate_search_result_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as SearchResult | null;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function validationGameSnapshot(snapshot: GpuSnapshot, sourceGame?: GameSnapshot): Promise<GameSnapshot> {
  return sourceGame ?? await engineGpuSnapshotGame(snapshot);
}

export async function validationEngine(): Promise<ChronofishEngine> {
  validationEnginePromise ??= instantiateChronofishWasm("./chronofish_engine.wasm")
    .then((instance) => instance.exports as unknown as ChronofishEngine);
  return validationEnginePromise;
}

export function loadValidationSnapshot(engine: ChronofishEngine, game: GameSnapshot): void {
  const { ptr, len } = writeWasmString(engine, JSON.stringify(game));
  try {
    if (!engine.chronofish_load_snapshot_json(ptr, len)) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineGpuSnapshot(game: GameSnapshot): Promise<GpuSnapshot> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  return JSON.parse(readWasmString(engine, engine.chronofish_gpu_snapshot_json())) as GpuSnapshot;
}

export async function engineGpuSnapshotGame(snapshot: GpuSnapshot): Promise<GameSnapshot> {
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

export async function engineSnapshotWithGpuChildBoards(
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

export async function engineFrontierMaxCycles(requestedDepth: number, timelineCount: number): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_frontier_max_cycles(requestedDepth, timelineCount);
}

export async function engineFrontierOrchestrationPlanForWidth(
  requestedDepth: number,
  timelineCount: number,
  frontierWidth: number
): Promise<FrontierOrchestrationPlan> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    requestedDepth,
    timelineCount,
    frontierWidth
  }));
  try {
    const output = engine.chronofish_frontier_orchestration_plan_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as FrontierOrchestrationPlan;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

interface FrontierOrchestrationPlan {
  maxCycles: number;
  perParentLimit: number;
  stateLimits: number[];
}

export async function engineSearchDepthAtLeastOne(depth: unknown): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_bot_search_depth_at_least_one(typeof depth === "number" ? depth : Number.NaN);
}

export async function engineGpuWorkerSearchConfig(depth: unknown, minDepth: unknown, timeMs: unknown): Promise<GpuWorkerSearchConfig> {
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

export async function engineGpuSearchRankingLimit(nodes: unknown): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_search_ranking_limit(typeof nodes === "number" ? nodes : Number.NaN);
}

export async function engineGpuSearchReplyLimit(nodes: unknown): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_search_reply_limit(typeof nodes === "number" ? nodes : Number.NaN);
}

export function engineGpuReplyPressureReplyLimit(engine: ChronofishEngine): number {
  return engine.chronofish_gpu_reply_pressure_reply_limit();
}

export async function engineGpuSearchValidationLimit(nodes: unknown): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_search_validation_limit(typeof nodes === "number" ? nodes : Number.NaN);
}

export async function engineGpuFullSearchReportedDepth(requestedDepth: number): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_full_search_reported_depth(requestedDepth);
}

export async function engineGpuCompletedReplyShouldSearch(snapshot: GpuSnapshot): Promise<boolean> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_completed_reply_should_search(snapshot.royalCaptureBy ? 1 : 0, Date.now(), gpuDeadlineAt) !== 0;
}

export function setGpuSearchDeadline(deadline: number): void {
  gpuDeadlineAt = deadline;
}

export function engineGpuFrontierCycleShouldStop(
  engine: ChronofishEngine,
  cycle: number,
  cyclesCompleted: number,
  requestedDepth: number
): boolean {
  return engine.chronofish_gpu_frontier_cycle_should_stop(cycle, cyclesCompleted, requestedDepth, Date.now(), gpuDeadlineAt) !== 0;
}

export function gpuDiagnosticRate(numerator: number, denominator: number, engine: ChronofishEngine): number {
  return engine.chronofish_gpu_diagnostic_rate(numerator, denominator);
}

export function gpuEffectiveBranchingFactor(selectedCount: number, cyclesCompleted: number, engine: ChronofishEngine): number {
  return engine.chronofish_gpu_effective_branching_factor(selectedCount, cyclesCompleted);
}

export function gpuReportedLatencyMs(latencyMs: number, engine: ChronofishEngine): number {
  return engine.chronofish_gpu_reported_latency_ms(latencyMs);
}

export function gpuNodesPerSecond(nodes: number, latencyMs: number, engine: ChronofishEngine): number {
  return engine.chronofish_gpu_nodes_per_second(nodes, latencyMs);
}

export async function engineGpuAccumulatedSearchNodes(baseNodes: number | null | undefined, extraNodes: number, fallbackNodes = 0): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_accumulated_search_nodes(baseNodes ?? 0, extraNodes, fallbackNodes);
}

export async function engineGpuSearchNodes(nodes: unknown): Promise<number> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_search_nodes(typeof nodes === "number" ? nodes : Number.NaN);
}

export function engineGpuMutationCandidateLimit(engine: ChronofishEngine, candidateCount: number): number {
  return engine.chronofish_gpu_mutation_candidate_limit(candidateCount);
}

export function engineGpuMutationCandidateWorkgroups(engine: ChronofishEngine, candidateLimit: number): number {
  return engine.chronofish_gpu_mutation_candidate_workgroups(candidateLimit);
}

export function engineGpuCandidateMaxCandidatesPerBatch(engine: ChronofishEngine, maxBindingSize: number): number {
  return engine.chronofish_gpu_candidate_max_candidates_per_batch(maxBindingSize);
}

export function engineGpuCandidateSourceBatchSize(engine: ChronofishEngine, maxCandidatesPerBatch: number, targetCount: number): number {
  return engine.chronofish_gpu_candidate_source_batch_size(maxCandidatesPerBatch, targetCount);
}

export function engineGpuCandidateBatchSourceCount(engine: ChronofishEngine, sourceCount: number, sourceStart: number, sourceBatchSize: number): number {
  return engine.chronofish_gpu_candidate_batch_source_count(sourceCount, sourceStart, sourceBatchSize);
}

export function engineGpuCandidateBatchCandidateCount(engine: ChronofishEngine, sourceCount: number, targetCount: number): number {
  return engine.chronofish_gpu_candidate_batch_candidate_count(sourceCount, targetCount);
}

export function engineGpuCandidateScoreWorkgroups(engine: ChronofishEngine, batchCandidateCount: number): number {
  return engine.chronofish_gpu_candidate_score_workgroups(batchCandidateCount);
}

export function engineGpuReplyScoreWorkgroupsX(engine: ChronofishEngine, rootCount: number): number {
  return engine.chronofish_gpu_reply_score_workgroups_x(rootCount);
}

export function engineGpuReplyScoreWorkgroupsY(engine: ChronofishEngine, replyCount: number): number {
  return engine.chronofish_gpu_reply_score_workgroups_y(replyCount);
}

export async function engineGpuCandidateInputs(game: GameSnapshot): Promise<GpuCandidateInputs> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  const bytes = readWasmBytes(engine, engine.chronofish_gpu_candidate_inputs_bytes());
  if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Engine GPU candidate input byte length is not i32-aligned.");
  }
  return candidateInputsFromWords(new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Int32Array.BYTES_PER_ELEMENT), engine);
}

export async function engineGpuCandidateInputsFromSnapshot(snapshot: GpuSnapshot, color: Color): Promise<GpuCandidateInputs> {
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

function candidateInputsFromWords(words: Int32Array, engine: ChronofishEngine): GpuCandidateInputs {
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
  const meta = candidateInputMetaFromEngine(words, engine);
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

export async function engineFrontierRoot(game: GameSnapshot, maxBoards: number): Promise<EncodedFrontierRoot> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  const ptr = engine.chronofish_frontier_root_bytes(maxBoards);
  return readEngineFrontierRoot(engine, ptr);
}

export async function engineFrontierRootFromSnapshot(snapshot: GpuSnapshot, maxBoards: number): Promise<EncodedFrontierRoot> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(snapshot));
  try {
    return readEngineFrontierRoot(engine, engine.chronofish_frontier_root_snapshot_bytes(ptr, len, maxBoards));
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineGpuSnapshotSearchSize(snapshot: GpuSnapshot): Promise<GpuSnapshotSearchSize> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify(snapshot));
  try {
    const output = engine.chronofish_gpu_snapshot_search_size_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as GpuSnapshotSearchSize;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function enginePendingPresentBoards(snapshot: GpuSnapshot, color: Color): Promise<PendingBoardRef[]> {
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

export async function engineTurnStatusRecords(snapshot: GpuSnapshot): Promise<Int32Array> {
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

export async function engineTurnStatusFromWords(words: Int32Array): Promise<TurnStatus> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    records: Array.from(words)
  }));
  try {
    const output = engine.chronofish_gpu_turn_status_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as TurnStatus;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineRankedCandidates(
  scored: ScoredCandidates,
  { pendingBoards = [], requirePending = false, limit }: {
    pendingBoards?: PendingBoardRef[];
    requirePending?: boolean;
    limit: number;
  }
): Promise<RankedCandidate[]> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    scores: Array.from(scored.scores),
    records: Array.from(scored.records),
    pendingBoards,
    requirePending,
    limit
  }));
  try {
    const output = engine.chronofish_gpu_ranked_candidates_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const ranked = JSON.parse(readWasmString(engine, output)) as RankedCandidate[];
    return Array.isArray(ranked) ? ranked : [];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineMutationSelectedCandidates(ranked: RankedCandidate[], limit: number): Promise<RankedCandidate[]> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    ranked,
    limit
  }));
  try {
    const output = engine.chronofish_gpu_mutation_selected_candidates_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const selected = JSON.parse(readWasmString(engine, output)) as RankedCandidate[];
    return Array.isArray(selected) ? selected : [];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineCandidateIndexes(candidates: RankedCandidate[]): Promise<number[]> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    candidates
  }));
  try {
    const output = engine.chronofish_gpu_candidate_indexes_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const indexes = JSON.parse(readWasmString(engine, output)) as number[];
    return Array.isArray(indexes) ? indexes : [];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineCandidateScores(scores: Int32Array, candidates: Array<Pick<RankedCandidate, "index">>, fallback: number): Promise<Int32Array> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    scores: Array.from(scores),
    candidates,
    fallback
  }));
  try {
    const output = engine.chronofish_gpu_candidate_scores_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const selectedScores = JSON.parse(readWasmString(engine, output)) as number[];
    return new Int32Array(Array.isArray(selectedScores) ? selectedScores : []);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineCandidateScore(scores: Int32Array, index: number, fallback: number): Promise<number> {
  const selected = await engineCandidateScores(scores, [{ index }], fallback);
  if (selected.length !== 1) {
    throw new Error("GPU candidate score response is invalid.");
  }
  return selected[0]!;
}

export async function engineCandidateScoreIsRejected(score: number): Promise<boolean> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_candidate_score_is_rejected(score) !== 0;
}

export async function engineGpuScoringSummary(scored: ScoredCandidates, pendingBoards: PendingBoardRef[]): Promise<string> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    scores: Array.from(scored.scores),
    records: Array.from(scored.records),
    pendingBoards
  }));
  try {
    const output = engine.chronofish_gpu_scoring_summary_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmString(engine, output);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineGpuMutationSummary(mutated: MutatedCandidate[]): Promise<string> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    statuses: mutated.map((entry) => entry.mutationStatus)
  }));
  try {
    const output = engine.chronofish_gpu_mutation_summary_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readWasmString(engine, output);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineGpuMutationStatuses(statuses: Int32Array, candidateCount: number): Promise<Int32Array> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    statuses: Array.from(statuses),
    candidateCount
  }));
  try {
    const output = engine.chronofish_gpu_mutation_statuses_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const normalized = JSON.parse(readWasmString(engine, output)) as number[];
    return new Int32Array(Array.isArray(normalized) ? normalized : []);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineGpuFrontierReadbackSummary(counters: Uint32Array): Promise<GpuFrontierReadbackSummary> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    counters: Array.from(counters)
  }));
  try {
    const output = engine.chronofish_gpu_frontier_readback_summary_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as GpuFrontierReadbackSummary;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineGpuTurnCompletionStep(
  snapshot: GpuSnapshot,
  movesLength: number,
  pendingBoards: PendingBoardRef[],
  status: TurnStatus,
  visitedKeys: Iterable<string> = []
): Promise<GpuTurnCompletionStep> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    snapshot,
    movesLength,
    pendingBoards,
    status,
    visitedKeys: Array.from(visitedKeys)
  }));
  try {
    const output = engine.chronofish_gpu_turn_completion_step_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as GpuTurnCompletionStep;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineIncompleteTurnPendingPresentBoardCount(
  status: TurnStatus,
  pendingBoards: PendingBoardRef[]
): Promise<number> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({ status, pendingBoards }));
  try {
    return engine.chronofish_gpu_incomplete_turn_pending_present_board_count_json(ptr, len);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineFullSearchPrecondition(status: TurnStatus): Promise<GpuFullSearchPrecondition> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({ status }));
  try {
    const output = engine.chronofish_gpu_full_search_precondition_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as GpuFullSearchPrecondition;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function enginePolicyChoiceAgreementDiagnostics(
  selected: SearchChoice,
  choices: SearchChoice[]
): Promise<GpuPolicyChoiceAgreementDiagnostics> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    selected,
    choices
  }));
  try {
    const output = engine.chronofish_gpu_policy_choice_agreement_diagnostics_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as GpuPolicyChoiceAgreementDiagnostics;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineFrontierChoiceDiagnostics(
  selected: SearchChoice,
  choices: SearchChoice[]
): Promise<{
  legalChoiceCount: number;
  legalTacticalChoiceCount: number;
  selectedMovePrunedRisk: number;
  selectedMoveTactical: number;
}> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    selected,
    choices
  }));
  try {
    const output = engine.chronofish_gpu_frontier_choice_diagnostics_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output));
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineSelectedSearchChoice<T extends SearchChoice>(request: unknown): Promise<(T & { choices: SearchChoice[] }) | null> {
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

export async function engineNonPostableResultSummary(result: unknown): Promise<string> {
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

export async function enginePostableSearchResult(result: unknown): Promise<boolean> {
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

export async function engineGpuSearchFailureSummary(snapshot: GpuSnapshot): Promise<string> {
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

export async function engineWithCompletedTurnChoice(
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

export async function enginePickCandidateRecords(records: Int32Array, indices: number[]): Promise<Int32Array> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    records: Array.from(records),
    indices
  }));
  try {
    const output = engine.chronofish_gpu_pick_candidate_records_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const bytes = readWasmBytes(engine, output);
    if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error("Engine picked candidate record byte length is not i32-aligned.");
    }
    return new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Int32Array.BYTES_PER_ELEMENT).slice();
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineGpuMutationTurnCode(records: Int32Array): Promise<number> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({ records: Array.from(records) }));
  try {
    return engine.chronofish_gpu_mutation_turn_code_json(ptr, len);
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineCandidateIndex(scored: ScoredCandidates, move: Move): Promise<number> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    records: Array.from(scored.records),
    move
  }));
  try {
    const index = engine.chronofish_gpu_candidate_index_json(ptr, len);
    if (index < -1) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return index;
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineReplyPressureRankedRoots(
  rankedRoots: RankedCandidate[],
  pairScores: Int32Array,
  replyCount: number
): Promise<RankedCandidate[]> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    rankedRoots,
    pairScores: Array.from(pairScores),
    replyCount
  }));
  try {
    const output = engine.chronofish_gpu_reply_pressure_ranked_roots_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const ranked = JSON.parse(readWasmString(engine, output)) as RankedCandidate[];
    return Array.isArray(ranked) ? ranked : [];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export function readEngineFrontierRoot(engine: ChronofishEngine, ptr: number): EncodedFrontierRoot {
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

export async function engineLegalTargets(game: GameSnapshot, position: Position): Promise<LegalTargetSelection> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  return JSON.parse(readWasmString(engine, engine.chronofish_legal_selection_json(
    position.timelineId,
    position.time,
    position.x,
    position.y
  ))) as LegalTargetSelection;
}

export async function engineApplyMove(game: GameSnapshot, move: Move): Promise<GameSnapshot> {
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

export async function engineSubmitTurn(game: GameSnapshot): Promise<TurnStatus> {
  const engine = await validationEngine();
  loadValidationSnapshot(engine, game);
  return JSON.parse(readWasmString(engine, engine.chronofish_submit_turn_status_json())) as TurnStatus;
}

export async function engineSupportedMutationCandidateIndexes(
  candidates: MutatedCandidate[],
  limit: number | undefined,
  requireChildBoards: boolean
): Promise<number[]> {
  const engine = await validationEngine();
  const { ptr, len } = writeWasmString(engine, JSON.stringify({
    candidates: candidates.map((candidate) => ({
      mutationStatus: candidate.mutationStatus,
      hasChildBoards: Boolean(candidate.childBoards)
    })),
    limit,
    requireChildBoards
  }));
  try {
    const output = engine.chronofish_gpu_supported_mutation_candidate_indexes_json(ptr, len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const indexes = JSON.parse(readWasmString(engine, output)) as number[];
    return Array.isArray(indexes) ? indexes : [];
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

export async function engineGpuMutationStatusIsTerminal(status: number): Promise<boolean> {
  const engine = await validationEngine();
  return engine.chronofish_gpu_mutation_status_is_terminal(status) !== 0;
}
