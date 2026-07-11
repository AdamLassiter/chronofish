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
  chronofish_gpu_snapshot_json(): number;
  chronofish_gpu_candidate_inputs_json(): number;
  chronofish_gpu_candidate_inputs_bytes(): number;
  chronofish_gpu_candidate_inputs_snapshot_bytes(ptr: number, length: number): number;
  chronofish_gpu_candidate_input_meta_json_bytes(ptr: number, length: number): number;
  chronofish_gpu_snapshot_game_json(ptr: number, length: number): number;
  chronofish_gpu_snapshot_search_size_json(ptr: number, length: number): number;
  chronofish_gpu_snapshot_child_boards_json(ptr: number, length: number): number;
  chronofish_compact_value_model_json(ptr: number, length: number): number;
  chronofish_compact_value_model_frontier_layout_json(ptr: number, length: number): number;
  chronofish_compact_value_model_bytes_json(ptr: number, length: number): number;
  chronofish_compact_value_model_is_finite_bytes(ptr: number, length: number): number;
  chronofish_compact_value_model_architecture_matches_bytes(ptr: number, length: number): number;
  chronofish_compact_value_model_policy_weights_bytes(ptr: number, length: number, inputSize: number): number;
  chronofish_quantized_policy_upload_bytes(ptr: number, length: number): number;
  chronofish_f32_to_f16_upload_bytes(ptr: number, length: number): number;
  chronofish_compact_value_model_predict_values_json(modelPtr: number, modelLength: number, samplesPtr: number, samplesLength: number): number;
  chronofish_compact_value_model_training_layout_bytes(modelPtr: number, modelLength: number, averageLabel: number): number;
  chronofish_compact_value_model_hidden_features_json(modelPtr: number, modelLength: number, samplesPtr: number, samplesLength: number): number;
  chronofish_frontier_root_bytes(maxBoards: number): number;
  chronofish_frontier_root_snapshot_bytes(ptr: number, length: number, maxBoards: number): number;
  chronofish_gpu_pending_present_boards_json(ptr: number, length: number): number;
  chronofish_gpu_select_candidate_json(ptr: number, length: number): number;
  chronofish_gpu_turn_status_records_snapshot_bytes(ptr: number, length: number): number;
  chronofish_gpu_turn_status_json_bytes(ptr: number, length: number): number;
  chronofish_gpu_turn_status_json(ptr: number, length: number): number;
  chronofish_gpu_full_search_precondition_json(ptr: number, length: number): number;
  chronofish_gpu_ranked_candidate_indexes_bytes(ptr: number, length: number): number;
  chronofish_gpu_ranked_candidates_json_bytes(ptr: number, length: number): number;
  chronofish_gpu_ranked_candidates_json(ptr: number, length: number): number;
  chronofish_gpu_mutation_selected_candidates_json(ptr: number, length: number): number;
  chronofish_gpu_candidate_indexes_json(ptr: number, length: number): number;
  chronofish_gpu_candidate_scores_json(ptr: number, length: number): number;
  chronofish_gpu_candidate_score_is_rejected(score: number): number;
  chronofish_gpu_frontier_readback_summary_json(ptr: number, length: number): number;
  chronofish_gpu_scoring_summary_bytes(ptr: number, length: number): number;
  chronofish_gpu_scoring_summary_json(ptr: number, length: number): number;
  chronofish_gpu_mutation_summary_bytes(ptr: number, length: number): number;
  chronofish_gpu_mutation_summary_json(ptr: number, length: number): number;
  chronofish_gpu_mutation_statuses_json(ptr: number, length: number): number;
  chronofish_gpu_turn_completion_key_json(ptr: number, length: number): number;
  chronofish_gpu_choice_agreement_json(ptr: number, length: number): number;
  chronofish_gpu_choice_agreement_choices_json(ptr: number, length: number): number;
  chronofish_gpu_policy_choice_agreement_diagnostics_json(ptr: number, length: number): number;
  chronofish_gpu_select_choice_json(ptr: number, length: number): number;
  chronofish_gpu_selected_choice_json(ptr: number, length: number): number;
  chronofish_gpu_move_plan_key_json(ptr: number, length: number): number;
  chronofish_gpu_frontier_plan_json_bytes(ptr: number, length: number, offset: number, planLength: number): number;
  chronofish_gpu_frontier_choices_json_bytes(ptr: number, length: number, maxBoards: number, frontierWidth: number, requestedDepth: number, gpuSearchPtr: number, gpuSearchLength: number, choiceLimit: number): number;
  chronofish_gpu_validated_frontier_choice_json(ptr: number, length: number): number;
  chronofish_gpu_frontier_choice_diagnostics_json(ptr: number, length: number): number;
  chronofish_gpu_non_postable_result_summary_json(ptr: number, length: number): number;
  chronofish_gpu_postable_search_result_json(ptr: number, length: number): number;
  chronofish_gpu_validate_first_frontier_turn_json(ptr: number, length: number): number;
  chronofish_gpu_validate_search_result_json(ptr: number, length: number): number;
  chronofish_gpu_search_failure_summary_json(ptr: number, length: number): number;
  chronofish_gpu_completed_turn_choice_json(ptr: number, length: number): number;
  chronofish_gpu_turn_completion_step_json(ptr: number, length: number): number;
  chronofish_gpu_incomplete_turn_pending_present_board_count_json(ptr: number, length: number): number;
  chronofish_gpu_normalize_principal_variation_json(ptr: number, length: number): number;
  chronofish_gpu_summarize_search_choices_json(ptr: number, length: number): number;
  chronofish_gpu_pick_candidate_records_bytes(ptr: number, length: number): number;
  chronofish_gpu_pick_candidate_records_json(ptr: number, length: number): number;
  chronofish_gpu_mutation_turn_code_json(ptr: number, length: number): number;
  chronofish_gpu_candidate_index_bytes(ptr: number, length: number): number;
  chronofish_gpu_candidate_index_json(ptr: number, length: number): number;
  chronofish_gpu_reply_pressure_ranked_roots_bytes(ptr: number, length: number): number;
  chronofish_gpu_reply_pressure_ranked_roots_json(ptr: number, length: number): number;
  chronofish_staged_turn_notation(): number;
  chronofish_evaluation_json(): number;
  chronofish_last_message(): number;
  chronofish_load_snapshot_json(ptr: number, length: number): number;
  chronofish_load_ai_parameters_json(ptr: number, length: number): number;
  chronofish_training_sample_json(ptr: number, length: number): number;
  chronofish_training_samples_json(ptr: number, length: number): number;
  chronofish_dedupe_training_samples_json(ptr: number, length: number): number;
  chronofish_select_training_working_set_indexes_bytes(ptr: number, length: number, maxProjectedBytes: number): number;
  chronofish_stable_sample_hash_json(ptr: number, length: number, index: number): number;
  chronofish_shuffled_training_indices_bytes(ptr: number, length: number, epoch: number, seed: number): number;
  chronofish_split_validation_samples_json(ptr: number, length: number, validationSplit: number): number;
  chronofish_split_policy_training_indices_json(ptr: number, length: number, validationSplit: number): number;
  chronofish_unique_training_position_count_json(ptr: number, length: number): number;
  chronofish_group_training_indices_by_position_json(ptr: number, length: number): number;
  chronofish_feature_length_json(ptr: number, length: number): number;
  chronofish_sparse_projection_features_bytes(ptr: number, length: number, inputSize: number): number;
  chronofish_fill_grouped_training_batch_indices_bytes(ptr: number, length: number): number;
  chronofish_policy_training_indices_bytes(ptr: number, length: number, requirePositiveWeight: number): number;
  chronofish_has_policy_training_target_json(ptr: number, length: number): number;
  chronofish_auxiliary_value_targets_bytes(ptr: number, length: number): number;
  chronofish_policy_training_steps(valueEpochs: number): number;
  chronofish_policy_training_target(policy: number): number;
  chronofish_training_label_weight(labelWeight: number): number;
  chronofish_training_weighted_average(total: number, totalWeight: number): number;
  chronofish_training_batch_normalization(batchWeight: number): number;
  chronofish_value_training_batch_size(configBatchSize: number, trainingCount: number): number;
  chronofish_policy_training_batch_size(configBatchSize: number, trainingCount: number): number;
  chronofish_value_head_validation_interval(epochs: number, validationInterval: number): number;
  chronofish_value_gpu_batches_per_submit(epochs: number): number;
  chronofish_value_gpu_validation_interval(batchesPerSubmit: number, validationInterval: number): number;
  chronofish_policy_training_steps_per_submit(steps: number): number;
  chronofish_append_replay_samples_json(ptr: number, length: number, maxBuffer: number): number;
  chronofish_label_source_counts_json(ptr: number, length: number): number;
  chronofish_relabel_outcome_samples_json(ptr: number, length: number): number;
  chronofish_distill_training_samples_with_labels_json(ptr: number, length: number): number;
  chronofish_search_result_label_sample_json(ptr: number, length: number): number;
  chronofish_search_result_label_sample_from_result_json(ptr: number, length: number): number;
  chronofish_search_result_turn_json(ptr: number, length: number): number;
  chronofish_cpu_parameters_key_json(ptr: number, length: number): number;
  chronofish_unique_cpu_parameters_json(ptr: number, length: number): number;
  chronofish_breed_cpu_population_json(ptr: number, length: number): number;
  chronofish_rank_cpu_scored_candidates_json(ptr: number, length: number): number;
  chronofish_cpu_training_elites_json(ptr: number, length: number, cpuFinalists: number): number;
  chronofish_cpu_training_finalist_candidates_json(ptr: number, length: number, target: number): number;
  chronofish_cpu_training_generation_outcome_json(ptr: number, length: number): number;
  chronofish_cpu_candidate_scoring_plan_json(ptr: number, length: number): number;
  chronofish_cpu_fitness_entry_for_candidate_json(ptr: number, length: number): number;
  chronofish_cpu_worker_search_config_json(ptr: number, length: number): number;
  chronofish_cpu_worker_search_result_json(ptr: number, length: number): number;
  chronofish_cpu_apply_turn_json(ptr: number, length: number): number;
  chronofish_normalize_training_modes_json(ptr: number, length: number): number;
  chronofish_training_mode_policy_json(ptr: number, length: number): number;
  chronofish_normalize_training_config_json(ptr: number, length: number): number;
  chronofish_gpu_training_worker_count(total: number, requestedWorkers: number): number;
  chronofish_gpu_duel_training_worker_count(total: number, searchWorkers: number, selfPlayWorkers: number): number;
  chronofish_training_label_worker_count(jobCount: number, requestedWorkers: number, hardwareCores: number): number;
  chronofish_training_split_work_json(total: number, workers: number): number;
  chronofish_take_training_sample_batches_json(ptr: number, length: number, target: number): number;
  chronofish_compact_training_samples_json(ptr: number, length: number): number;
  chronofish_training_sample_plies(index: number, encodeOnly: number): number;
  chronofish_training_sample_seed(ptr: number, length: number, index: number, salt: number): number;
  chronofish_training_search_seed_json(ptr: number, length: number, salt: number): number;
  chronofish_gpu_warmup_plies(workerIndex: number): number;
  chronofish_gpu_rollout_max_plies(target: number, workerIndex: number): number;
  chronofish_gpu_rollout_ply_offset(ply: number, workerIndex: number): number;
  chronofish_gpu_warmup_search_config_json(depth: number, nodes: number, searchTimeMs: number, explorationTemperature: number): number;
  chronofish_gpu_position_generation_search_config_json(depth: number, nodes: number, explorationTemperature: number): number;
  chronofish_curriculum_search_config_json(depth: number, nodes: number, explorationTemperature: number, index: number): number;
  chronofish_curriculum_game_snapshot_json(ptr: number, length: number, index: number): number;
  chronofish_tactical_search_config_json(depth: number, nodes: number, explorationTemperature: number, attempt: number): number;
  chronofish_tactical_position_attempt_count(index: number): number;
  chronofish_tactical_position_use_best_source(bestPriority: number): number;
  chronofish_tactical_position_selection_json(bestPriority: number, generatedPriority: number): number;
  chronofish_tactical_position_priority_snapshot_json(ptr: number, length: number): number;
  chronofish_royal_capture_winner_snapshot_json(ptr: number, length: number): number;
  chronofish_training_worker_request_timeout_ms(nodes: number, timeMs: number): number;
  chronofish_training_worker_search_time_ms(nodes: number, timeMs: number): number;
  chronofish_training_worker_request_timeout_ms_json(ptr: number, length: number): number;
  chronofish_training_worker_search_time_ms_json(ptr: number, length: number): number;
  chronofish_loss_log_replay_logs_json(ptr: number, length: number): number;
  chronofish_loss_log_validation_update_json(ptr: number, length: number): number;
  chronofish_training_metrics_summary_json(ptr: number, length: number): number;
  chronofish_normalized_search_score(score: number): number;
  chronofish_denormalized_search_score(value: number): number;
  chronofish_bounded_value(value: number): number;
  chronofish_inverse_tanh(value: number): number;
  chronofish_optimizer_velocity(previous: number, gradient: number, momentum: number): number;
  chronofish_loss_reduction_workgroup_count(sampleCount: number): number;
  chronofish_training_workgroups_16(itemCount: number): number;
  chronofish_training_workgroups_64(itemCount: number): number;
  chronofish_align4(value: number): number;
  chronofish_cpu_prediction_max_batch(): number;
  chronofish_cpu_head_training_max_positions(): number;
  chronofish_min_hidden_training_positions(): number;
  chronofish_projection_chunk_size(): number;
  chronofish_projection_temporary_budget(maxBufferSize: number): number;
  chronofish_dense_kernel_entry_point_bytes(ptr: number, length: number, sampleCount: number): number;
  chronofish_projection_hash(rawIndex: number, projectionIndex: number, seed: number): number;
  chronofish_default_output_layer_size(): number;
  chronofish_default_previous_layer_size(layerIndex: number, inputSize: number): number;
  chronofish_default_initial_hidden_weights_bytes(): number;
  chronofish_initial_hidden_weights_bytes(ptr: number, length: number): number;
  chronofish_split_hidden_weights_bytes(ptr: number, length: number): number;
  chronofish_concat_f32_bytes(ptr: number, length: number): number;
  chronofish_count_non_zero_f32_bytes(ptr: number, length: number): number;
  chronofish_output_delta_params_bytes(sampleCount: number, totalWeight: number): number;
  chronofish_hidden_delta_params_bytes(sampleCount: number, currentSize: number, nextSize: number): number;
  chronofish_policy_params_bytes(batchCount: number, inputSize: number, totalWeight: number, learningRate: number, weightDecay: number, momentum: number): number;
  chronofish_layer_params_bytes(sampleCount: number, inputSize: number, outputSize: number, learningRate: number, weightDecay: number, momentum: number): number;
  chronofish_output_params_bytes(sampleCount: number, inputSize: number, learningRate: number, weightDecay: number, momentum: number): number;
  chronofish_projection_params_bytes(sampleCount: number, inputSize: number, projectionSize: number, seed: number, outputOffset: number): number;
  chronofish_opposite_color_json(ptr: number, length: number): number;
  chronofish_gpu_search_color_code_json(ptr: number, length: number): number;
  chronofish_training_label_policy_json(): number;
  chronofish_training_label_priority(ptr: number, length: number, pseudo: number): number;
  chronofish_policy_bucket_from_move_values(fromTimelineId: number, fromTime: number, fromX: number, fromY: number, toTimelineId: number, toTime: number, toX: number, toY: number, intent: number): number;
  chronofish_bot_search_depth_at_least_one(depth: number): number;
  chronofish_bot_search_config_json(depth: number, minDepth: number, nodes: number, timeMs: number): number;
  chronofish_gpu_worker_search_config_json(depth: number, minDepth: number, timeMs: number): number;
  chronofish_gpu_search_ranking_limit(nodes: number): number;
  chronofish_gpu_search_reply_limit(nodes: number): number;
  chronofish_gpu_reply_pressure_reply_limit(): number;
  chronofish_gpu_search_validation_limit(nodes: number): number;
  chronofish_gpu_supported_mutation_candidate_indexes_bytes(ptr: number, length: number): number;
  chronofish_gpu_supported_mutation_candidate_indexes_json(ptr: number, length: number): number;
  chronofish_gpu_mutation_status_is_terminal(status: number): number;
  chronofish_gpu_full_search_reported_depth(requestedDepth: number): number;
  chronofish_gpu_completed_reply_should_search(royalCapturePresent: number, nowMs: number, deadlineAtMs: number): number;
  chronofish_gpu_frontier_cycle_should_stop(cycle: number, cyclesCompleted: number, requestedDepth: number, nowMs: number, deadlineAtMs: number): number;
  chronofish_gpu_diagnostic_rate(numerator: number, denominator: number): number;
  chronofish_gpu_effective_branching_factor(selectedCount: number, cyclesCompleted: number): number;
  chronofish_gpu_reported_latency_ms(latencyMs: number): number;
  chronofish_gpu_nodes_per_second(nodes: number, latencyMs: number): number;
  chronofish_gpu_search_nodes(nodes: number): number;
  chronofish_gpu_accumulated_search_nodes(baseNodes: number, extraNodes: number, fallbackNodes: number): number;
  chronofish_gpu_mutation_candidate_limit(candidateCount: number): number;
  chronofish_gpu_mutation_candidate_workgroups(candidateLimit: number): number;
  chronofish_gpu_turn_completion_max_moves(existingMoves: number, timelineCount: number): number;
  chronofish_gpu_candidate_max_dispatch_workgroups(): number;
  chronofish_gpu_candidate_max_candidates_per_dispatch(): number;
  chronofish_gpu_candidate_max_candidates_per_batch(maxBindingSize: number): number;
  chronofish_gpu_candidate_source_batch_size(maxCandidatesPerBatch: number, targetCount: number): number;
  chronofish_gpu_candidate_batch_source_count(sourceCount: number, sourceStart: number, sourceBatchSize: number): number;
  chronofish_gpu_candidate_batch_candidate_count(sourceCount: number, targetCount: number): number;
  chronofish_gpu_candidate_score_workgroups(batchCandidateCount: number): number;
  chronofish_gpu_reply_score_workgroups_x(rootCount: number): number;
  chronofish_gpu_reply_score_workgroups_y(replyCount: number): number;
  chronofish_bot_next_search_depth(currentDepth: number, targetDepth: number): number;
  chronofish_bot_worker_search_time_ms(timeMs: number): number;
  chronofish_bot_completed_search_depth(resultDepth: number, requestedDepth: number, resultEndsInRoyalCapture: number): number;
  chronofish_bot_result_ends_in_royal_capture_json(ptr: number, length: number): number;
  chronofish_bot_ranked_choices_json(ptr: number, length: number): number;
  chronofish_bot_select_best_result_json(ptr: number, length: number): number;
  chronofish_frontier_max_cycles(requestedDepth: number, timelineCount: number): number;
  chronofish_frontier_orchestration_plan_json(ptr: number, length: number): number;
  chronofish_frontier_per_parent_limit(frontierWidth: number): number;
  chronofish_frontier_next_active_state_limit(frontierWidth: number, activeStateLimit: number, perParentLimit: number): number;
  chronofish_frontier_state_stride(maxBoards: number): number;
  chronofish_frontier_state_bytes(maxBoards: number): number;
  chronofish_frontier_neural_params_bytes(stateCount: number, stateStride: number, boardOffset: number, maxBoards: number, stateOffset: number, projectionSize: number, projectionSeed: number, targetDepth: number): number;
  chronofish_frontier_neural_apply_params_bytes(stateCount: number, rootColor: number, valueScale: number, valueBias: number, stateOffset: number): number;
  chronofish_frontier_neural_layer_params_bytes(sampleCount: number, inputSize: number, outputSize: number): number;
  chronofish_frontier_neural_effective_batch_size(stateCount: number, requestedBatchSize: number): number;
  chronofish_frontier_neural_batch_count(stateCount: number, stateOffset: number, effectiveBatchSize: number): number;
  chronofish_frontier_neural_cache_hit_rate(hits: number, misses: number): number;
  chronofish_frontier_cycle_state_count(frontierWidth: number, requestedStateCount: number): number;
  chronofish_frontier_expansion_source_scan_limit(candidateWorkgroupSize: number, dispatchCandidateLimit: number): number;
  chronofish_frontier_expansion_source_scan_count(sourceScanLimit: number, sourceScans: number, sourceScanBase: number): number;
  chronofish_frontier_minimax_bounded_depth(targetDepth: number, ancestryStride: number): number;
  chronofish_frontier_neural_select_board_workgroups(batchCount: number): number;
  chronofish_frontier_neural_project_workgroups_x(batchCount: number): number;
  chronofish_frontier_neural_project_workgroups_y(projectionSize: number): number;
  chronofish_frontier_neural_layer_workgroups_x(batchCount: number): number;
  chronofish_frontier_neural_layer_workgroups_y(outputSize: number): number;
  chronofish_frontier_neural_output_workgroups(batchCount: number): number;
  chronofish_frontier_policy_workgroups(candidateCount: number): number;
  chronofish_frontier_expand_workgroups(count: number, candidateWorkgroupSize: number): number;
  chronofish_frontier_selection_workgroups(capacity: number, candidateWorkgroupSize: number): number;
  chronofish_frontier_materialize_workgroups(frontierWidth: number, mutationTileSize: number): number;
  chronofish_frontier_minimax_workgroups(frontierWidth: number): number;
  chronofish_frontier_policy_params_bytes(candidateCount: number, candidateStride: number, inputSize: number, policyScale: number): number;
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
  chronofish_cpu_match_remaining_searches(maxMatchPlies: number, ply: number): number;
  chronofish_cpu_match_should_continue(nowMs: number, deadlineAtMs: number): number;
  chronofish_cpu_paired_match_deadline_ms(nowMs: number, deadlineAtMs: number, totalMatches: number, completedMatches: number): number;
  chronofish_cpu_paired_match_total_matches(gameCount: number): number;
  chronofish_cpu_paired_match_candidate_colors_json(ptr: number, length: number): number;
  chronofish_cpu_paired_match_average_score(score: number, completedMatches: number): number;
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
  chronofish_cpu_reference_score_from_result_json(ptr: number, length: number): number;
  chronofish_cpu_reference_score_delta_from_result_json(ptr: number, length: number): number;
  chronofish_cpu_reference_candidate_average(score: number, compared: number, nearDraws: number, drawRateLimit: number): number;
  chronofish_cpu_training_no_move_score(candidateTurn: number): number;
  chronofish_cpu_training_candidate_turn_json(ptr: number, length: number): number;
  chronofish_cpu_training_winner_score_json(ptr: number, length: number): number;
  chronofish_cpu_training_adjudication_score_json(ptr: number, length: number): number;
  chronofish_cpu_training_adjudication_score_from_result_json(ptr: number, length: number): number;
  chronofish_cpu_training_position_worker_count(target: number, cpuWorkers: number): number;
  chronofish_cpu_reference_worker_count(gameCount: number, requestedWorkers: number, pairBatch: number): number;
  chronofish_cpu_candidate_worker_count(candidateCount: number, cpuWorkers: number, pairBatch: number): number;
  chronofish_cpu_label_worker_count(positionCount: number, cpuWorkers: number): number;
  chronofish_cpu_search_label_weight(trainingModeCount: number): number;
  chronofish_cpu_reference_comparison_count(gameCount: number, referenceCount: number): number;
  chronofish_cpu_reference_should_continue(nowMs: number, deadlineAtMs: number, compared: number, maxMatchPlies: number): number;
  chronofish_cpu_training_candidate_count(cpuCandidates: number): number;
  chronofish_cpu_screening_game_count(sampleGameCount: number, cpuScreeningOpponentVariants: number): number;
  chronofish_cpu_training_finalist_target(populationLen: number, cpuFinalists: number, cpuPairBatch: number, screenedLen: number): number;
  chronofish_cpu_training_elite_count(cpuFinalists: number): number;
  chronofish_cpu_training_candidate_improved(candidateScore: number, baselineScore: number, bestCandidateScore: number): number;
  chronofish_cpu_training_next_stagnation(generationsWithoutCandidate: number, improved: number): number;
  chronofish_cpu_training_should_continue(nowMs: number, deadlineAtMs: number, generationsWithoutCandidate: number, maxGenerationsWithoutCandidate: number): number;
  chronofish_cpu_candidate_scoring_should_continue(nowMs: number, deadlineAtMs: number, nextCandidate: number, uncachedCandidateCount: number): number;
  chronofish_cpu_reference_collection_should_continue(nowMs: number, deadlineAtMs: number, nextGame: number, gameCount: number): number;
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
  chronofish_submit_turn_status_json(): number;
  chronofish_ai_turn_json(maxDepth: number, maxNodes: number): number;
  chronofish_ai_turn_timed_json(maxDepth: number, maxNodes: number, millis: number): number;
  chronofish_ai_turn_timed_min_depth_json(maxDepth: number, minDepth: number, maxNodes: number, millis: number): number;
}
