export type Color = "white" | "black";
export type TimelineOwner = Color | "neutral";

export type PieceType =
  | "king"
  | "commonKing"
  | "queen"
  | "royalQueen"
  | "princess"
  | "rook"
  | "bishop"
  | "unicorn"
  | "dragon"
  | "knight"
  | "pawn"
  | "brawn";

export interface Piece {
  color: Color;
  type: PieceType;
}

export type BoardSquares = Array<Array<Piece | null>>;

export interface EnPassantTarget {
  x: number;
  y: number;
  capturedX: number;
  capturedY: number;
}

export interface MoveOrigin {
  type: string;
  from?: Position;
  to?: Position;
}

export interface BoardSnapshot {
  time: number;
  sideToMove: Color;
  castling: number;
  enPassant: EnPassantTarget | null;
  origin: MoveOrigin | null;
  board: BoardSquares;
}

export interface Timeline {
  id: number;
  row: number;
  label: string;
  owner: TimelineOwner;
  active?: boolean;
  boards: BoardSnapshot[];
}

export interface GameSnapshot {
  turn: Color;
  presentTime?: number;
  nextTimelineId: number;
  nextBlackTimelineId: number;
  checkedRoyals: Position[];
  royalCaptureBy?: Color | null;
  result?: GameResult | null;
  timelines: Timeline[];
}

export interface GameResult {
  terminal: true;
  outcome: "win" | "draw";
  winner: Color | null;
  reason: "royal-capture" | "threefold-repetition" | "stalemate";
}

export interface Position {
  timelineId: number;
  time: number;
  x: number;
  y: number;
}

export interface Move {
  from: Position;
  to: Position;
}

export interface PlannedArrow {
  from: Position;
  to: Position;
  kind: "planned" | "bot-review";
}

export interface GhostBoard {
  nodeId: string;
  timelineId: number;
  board: BoardSnapshot;
  kind: "planned" | "bot-review";
}

export interface WasmString {
  ptr: number;
  len: number;
}

export interface ChronofishEngine {
  memory: WebAssembly.Memory;
  chronofish_output_len(): number;
  chronofish_alloc(length: number): number;
  chronofish_dealloc(ptr: number, length: number): void;
  chronofish_version(): number;
  chronofish_reset(): void;
  chronofish_snapshot_json(): number;
  chronofish_gpu_snapshot_bytes(): number;
  chronofish_gpu_candidate_inputs_json(): number;
  chronofish_gpu_candidate_inputs_bytes(): number;
  chronofish_frontier_root_bytes(maxBoards: number): number;
  chronofish_frontier_root_snapshot_bytes(ptr: number, length: number, maxBoards: number): number;
  chronofish_gpu_select_candidate_json(ptr: number, length: number): number;
  chronofish_gpu_turn_status_records_snapshot_bytes(ptr: number, length: number): number;
  chronofish_gpu_ranked_candidate_indexes_bytes(ptr: number, length: number): number;
  chronofish_gpu_scoring_summary_bytes(ptr: number, length: number): number;
  chronofish_gpu_mutation_summary_bytes(ptr: number, length: number): number;
  chronofish_gpu_turn_completion_key_json(ptr: number, length: number): number;
  chronofish_gpu_choice_agreement_json(ptr: number, length: number): number;
  chronofish_gpu_pick_candidate_records_bytes(ptr: number, length: number): number;
  chronofish_gpu_candidate_index_bytes(ptr: number, length: number): number;
  chronofish_gpu_reply_pressure_ranked_roots_bytes(ptr: number, length: number): number;
  chronofish_staged_turn_notation(): number;
  chronofish_evaluation_json(): number;
  chronofish_last_message(): number;
  chronofish_load_snapshot_json(ptr: number, length: number): number;
  chronofish_load_ai_parameters_json(ptr: number, length: number): number;
  chronofish_training_sample_json(ptr: number, length: number): number;
  chronofish_training_samples_json(ptr: number, length: number): number;
  chronofish_dedupe_training_samples_json(ptr: number, length: number): number;
  chronofish_append_replay_samples_json(ptr: number, length: number, maxBuffer: number): number;
  chronofish_label_source_counts_json(ptr: number, length: number): number;
  chronofish_relabel_outcome_samples_json(ptr: number, length: number): number;
  chronofish_cpu_parameters_key_json(ptr: number, length: number): number;
  chronofish_unique_cpu_parameters_json(ptr: number, length: number): number;
  chronofish_breed_cpu_population_json(ptr: number, length: number): number;
  chronofish_normalize_training_modes_json(ptr: number, length: number): number;
  chronofish_training_mode_policy_json(ptr: number, length: number): number;
  chronofish_normalize_training_config_json(ptr: number, length: number): number;
  chronofish_gpu_training_worker_count(total: number, requestedWorkers: number): number;
  chronofish_training_split_work_json(total: number, workers: number): number;
  chronofish_training_sample_plies(index: number, encodeOnly: number): number;
  chronofish_training_sample_seed(ptr: number, length: number, index: number, salt: number): number;
  chronofish_training_search_seed_json(ptr: number, length: number, salt: number): number;
  chronofish_gpu_warmup_plies(workerIndex: number): number;
  chronofish_gpu_warmup_search_config_json(depth: number, nodes: number, searchTimeMs: number, explorationTemperature: number): number;
  chronofish_gpu_position_generation_search_config_json(depth: number, nodes: number, explorationTemperature: number): number;
  chronofish_curriculum_search_config_json(depth: number, nodes: number, explorationTemperature: number, index: number): number;
  chronofish_curriculum_game_snapshot_json(ptr: number, length: number, index: number): number;
  chronofish_tactical_search_config_json(depth: number, nodes: number, explorationTemperature: number, attempt: number): number;
  chronofish_tactical_position_priority_snapshot_json(ptr: number, length: number): number;
  chronofish_royal_capture_winner_snapshot_json(ptr: number, length: number): number;
  chronofish_training_worker_request_timeout_ms(nodes: number, timeMs: number): number;
  chronofish_training_worker_search_time_ms(nodes: number, timeMs: number): number;
  chronofish_normalized_search_score(score: number): number;
  chronofish_training_label_policy_json(): number;
  chronofish_policy_bucket_from_move_values(fromTimelineId: number, fromTime: number, fromX: number, fromY: number, toTimelineId: number, toTime: number, toX: number, toY: number, intent: number): number;
  chronofish_bot_search_depth_at_least_one(depth: number): number;
  chronofish_bot_next_search_depth(currentDepth: number, targetDepth: number): number;
  chronofish_bot_worker_search_time_ms(timeMs: number): number;
  chronofish_bot_completed_search_depth(resultDepth: number, requestedDepth: number, resultEndsInRoyalCapture: number): number;
  chronofish_frontier_max_cycles(requestedDepth: number, timelineCount: number): number;
  chronofish_frontier_per_parent_limit(frontierWidth: number): number;
  chronofish_derive_frontier_tuning_json(
    maxStorageBufferBindingSize: number,
    maxBufferSize: number,
    maxComputeInvocationsPerWorkgroup: number,
    requestedNodes: number,
    boardCount: number,
    additionalBoardCapacity: number
  ): number;
  chronofish_frontier_selection_plan_json(
    maxBoards: number,
    frontierWidth: number,
    candidateCapacity: number,
    neuralBatchSize: number,
    candidateWorkgroupSize: number,
    mutationTileSize: number,
    dispatchCandidateLimit: number,
    maxSelectionScan: number
  ): number;
  chronofish_cpu_match_turn_time_ms(cpuTrainingTimeMs: number, nowMs: number, deadlineAtMs: number, remainingSearches: number): number;
  chronofish_cpu_training_position_target(
    samples: number,
    trainingModeCount: number,
    cpuOpponentVariants: number,
    cpuScreeningOpponentVariants: number,
    cpuRoundsPerVariant: number,
    cpuLeagueContenders: number,
    cpuLeagueHallOfFameEntries: number,
    cpuHallOfFameEntries: number,
    cpuMinPairs: number,
    cpuMaxPairs: number,
    cpuMaxMatchPlies: number
  ): number;
  chronofish_cpu_training_budget_ms(cpuTrainSeconds: number, cpuTrainingTimeMs: number, cpuMaxMatchPlies: number, cpuMaxMatchTimeMs: number): number;
  chronofish_mode_label_target(samples: number, trainingModeCount: number, divisor: number): number;
  chronofish_cpu_reference_score_delta_json(ptr: number, length: number): number;
  chronofish_cpu_reference_candidate_average(score: number, compared: number, nearDraws: number, drawRateLimit: number): number;
  chronofish_cpu_training_no_move_score(candidateTurn: number): number;
  chronofish_cpu_training_winner_score_json(ptr: number, length: number): number;
  chronofish_cpu_training_adjudication_score_json(ptr: number, length: number): number;
  chronofish_cpu_training_position_worker_count(target: number, cpuWorkers: number): number;
  chronofish_cpu_reference_worker_count(gameCount: number, requestedWorkers: number, pairBatch: number): number;
  chronofish_cpu_candidate_worker_count(candidateCount: number, cpuWorkers: number, pairBatch: number): number;
  chronofish_cpu_label_worker_count(positionCount: number, cpuWorkers: number): number;
  chronofish_cpu_search_label_weight(trainingModeCount: number): number;
  chronofish_cpu_training_candidate_count(cpuCandidates: number): number;
  chronofish_cpu_training_finalist_target(populationLen: number, cpuFinalists: number, cpuPairBatch: number, screenedLen: number): number;
  chronofish_cpu_training_elite_count(cpuFinalists: number): number;
  chronofish_cpu_training_position_search_config_json(cpuDepth: number, cpuNodes: number): number;
  chronofish_cpu_screening_training_config_json(cpuDepth: number, depth: number, cpuNodes: number, nodes: number, cpuTrainingTimeMs: number): number;
  chronofish_apply_move(
    fromTimelineId: number,
    fromTime: number,
    fromX: number,
    fromY: number,
    toTimelineId: number,
    toTime: number,
    toX: number,
    toY: number
  ): number;
  chronofish_legal_targets_json(timelineId: number, time: number, x: number, y: number): number;
  chronofish_legal_selection_json(timelineId: number, time: number, x: number, y: number): number;
  chronofish_submit_turn(): number;
  chronofish_ai_turn_json(maxDepth: number, maxNodes: number): number;
  chronofish_ai_turn_timed_json(maxDepth: number, maxNodes: number, millis: number): number;
  chronofish_ai_turn_timed_min_depth_json(maxDepth: number, minDepth: number, maxNodes: number, millis: number): number;
}
