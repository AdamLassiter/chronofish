use std::path::PathBuf;

use chronofish_engine::gpu::search::{
    bot_completed_search_depth,
    bot_next_search_depth,
    bot_ranked_choices_json,
    bot_result_ends_in_royal_capture,
    bot_result_ends_in_royal_capture_json,
    bot_search_config_json,
    bot_search_depth_at_least_one,
    bot_select_best_result_json,
    bot_worker_search_time_ms,
    derive_frontier_tuning,
    frontier_cycle_state_count,
    frontier_expand_workgroups,
    frontier_expansion_source_scan_count,
    frontier_expansion_source_scan_limit,
    frontier_materialize_workgroups,
    frontier_max_cycles,
    frontier_minimax_bounded_depth,
    frontier_minimax_workgroups,
    frontier_neural_cache_hit_rate,
    frontier_next_active_state_limit,
    frontier_orchestration_plan,
    frontier_per_parent_limit,
    frontier_policy_workgroups,
    frontier_selection_plan,
    frontier_selection_workgroups,
    frontier_state_bytes,
    frontier_state_stride,
    gpu_accumulated_search_nodes,
    gpu_board_record_from_snapshot,
    gpu_candidate_batch_candidate_count,
    gpu_candidate_batch_source_count,
    gpu_candidate_board_records_from_snapshot,
    gpu_candidate_index_from_i32s,
    gpu_candidate_index_json,
    gpu_candidate_indexes_json,
    gpu_candidate_input_meta_json_from_i32s,
    gpu_candidate_inputs_from_gpu_snapshot_json,
    gpu_candidate_inputs_from_snapshot_json,
    gpu_candidate_inputs_from_timelines,
    gpu_candidate_inputs_i32s_from_snapshot_json,
    gpu_candidate_inputs_json_from_snapshot_json,
    gpu_candidate_max_candidates_per_batch,
    gpu_candidate_max_candidates_per_dispatch,
    gpu_candidate_max_dispatch_workgroups,
    gpu_candidate_move_from_record,
    gpu_candidate_score_is_rejected,
    gpu_candidate_score_workgroups,
    gpu_candidate_scores_json,
    gpu_candidate_source_batch_size,
    gpu_child_is_source_advance,
    gpu_choice_agreement_choices_json,
    gpu_choice_agreement_json,
    gpu_completed_reply_should_search,
    gpu_completed_turn_choice_json,
    gpu_diagnostic_rate,
    gpu_effective_branching_factor,
    gpu_frontier_active_timeline_distance,
    gpu_frontier_choice_diagnostics_json,
    gpu_frontier_choices_json_from_i32s,
    gpu_frontier_clamp_usize,
    gpu_frontier_cycle_should_stop,
    gpu_frontier_floor_power_of_two,
    gpu_frontier_hash_words,
    gpu_frontier_next_power_of_two,
    gpu_frontier_origin_code,
    gpu_frontier_pending_board_count,
    gpu_frontier_plan_json_from_i32s,
    gpu_frontier_positive_limit,
    gpu_frontier_present_time,
    gpu_frontier_readback_summary_json,
    gpu_frontier_root_i32s_from_snapshot_json,
    gpu_frontier_timeline_active,
    gpu_frontier_workgroup_size,
    gpu_full_search_precondition_json,
    gpu_full_search_reported_depth,
    gpu_incomplete_turn_pending_present_board_count,
    gpu_incomplete_turn_pending_present_board_count_json,
    gpu_latest_board_index,
    gpu_move_plan_key_json,
    gpu_mutation_board_record_from_snapshot,
    gpu_mutation_board_record_to_snapshot,
    gpu_mutation_candidate_limit,
    gpu_mutation_candidate_workgroups,
    gpu_mutation_selected_candidates_json,
    gpu_mutation_status_is_terminal,
    gpu_mutation_statuses_json,
    gpu_mutation_summary_from_i32s,
    gpu_mutation_summary_json,
    gpu_mutation_turn_code_from_records,
    gpu_mutation_turn_code_json,
    gpu_next_branch_row,
    gpu_nodes_per_second,
    gpu_non_postable_result_summary_json,
    gpu_normalize_principal_variation_json,
    gpu_pending_present_boards_json_from_snapshot_json,
    gpu_pick_candidate_records_from_i32s,
    gpu_pick_candidate_records_json,
    gpu_policy_choice_agreement_diagnostics_json,
    gpu_postable_search_result_json,
    gpu_ranked_candidate_indexes_from_i32s,
    gpu_ranked_candidates_json,
    gpu_ranked_candidates_json_from_i32s,
    gpu_reply_pressure_ranked_roots_from_i32s,
    gpu_reply_pressure_ranked_roots_json,
    gpu_reply_pressure_reply_limit,
    gpu_reply_score_workgroups_x,
    gpu_reply_score_workgroups_y,
    gpu_reported_latency_ms,
    gpu_scoring_summary_from_i32s,
    gpu_scoring_summary_json,
    gpu_search_board_to_square_codes,
    gpu_search_color_code,
    gpu_search_color_from_code,
    gpu_search_failure_summary_json,
    gpu_search_nodes,
    gpu_search_opposite_color,
    gpu_search_owner_code,
    gpu_search_owner_from_code,
    gpu_search_piece_code,
    gpu_search_piece_from_code,
    gpu_search_piece_type_code,
    gpu_search_piece_type_from_code,
    gpu_search_ranking_limit,
    gpu_search_reply_limit,
    gpu_search_select_candidate_json,
    gpu_search_select_choice_json,
    gpu_search_selected_choice_json,
    gpu_search_square_codes_to_board,
    gpu_search_validation_limit,
    gpu_snapshot_game_json,
    gpu_snapshot_search_size_json,
    gpu_snapshot_with_child_boards_json,
    gpu_source_square_records_for_board,
    gpu_square_record_from_code,
    gpu_summarize_search_choices_json,
    gpu_supported_mutation_candidate_indexes_from_i32s,
    gpu_supported_mutation_candidate_indexes_json,
    gpu_target_square_records_for_board,
    gpu_timeline_sort_order,
    gpu_turn_completion_key_json,
    gpu_turn_completion_max_moves,
    gpu_turn_completion_step_json,
    gpu_turn_status_json,
    gpu_turn_status_json_from_i32s,
    gpu_turn_status_records_i32s_from_snapshot_json,
    gpu_validate_first_frontier_turn_json,
    gpu_validate_search_result_json,
    gpu_validated_frontier_choice_json,
    gpu_worker_search_config_json,
    search,
    FrontierOrchestrationPlan,
    FrontierSelectionPlan,
    FrontierTuning,
    FrontierTuningLimits,
    GpuBoardRecordInput,
    GpuCandidateBoardInput,
    GpuCandidateBoardRecords,
    GpuCandidateInputBoard,
    GpuCandidateInputTimeline,
    GpuCandidateInputs,
    GpuCandidateMove,
    GpuCandidatePosition,
    GpuCandidateSquareRecord,
    GpuChildBoardRef,
    GpuDecodedPiece,
    GpuEnPassantRecord,
    GpuFrontierActiveBoard,
    GpuMutationBoardRecordInput,
    GpuMutationBoardSnapshot,
    GpuSearchRequest,
    GpuSquareRecordBoardInput,
    GpuSquareRecordInput,
    GpuTimelineSortKey,
    FRONTIER_BOARD_OFFSET,
    FRONTIER_BOARD_PENDING,
    FRONTIER_HEADER_BOARD_COUNT,
    FRONTIER_HEADER_DEPTH,
    FRONTIER_HEADER_PENDING_BOARDS,
    FRONTIER_HEADER_PLAN_LENGTH,
    FRONTIER_HEADER_PRESENT_TIME,
    FRONTIER_HEADER_SCORE,
    FRONTIER_HEADER_TERMINAL,
    FRONTIER_HEADER_TURN,
    FRONTIER_MOVE_STRIDE,
    FRONTIER_PLAN_OFFSET,
    GPU_BOARD_STRIDE,
    GPU_CANDIDATE_INPUT_HEADER_I32S,
    GPU_CANDIDATE_STRIDE,
    GPU_MUTATION_BOARD_STRIDE,
    GPU_MUTATION_CHILD_STRIDE,
    GPU_MUTATION_STATUS_BRANCH_OK,
    GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE,
    GPU_MUTATION_STATUS_OK,
    GPU_MUTATION_STATUS_ROYAL_CAPTURE,
    GPU_SOURCE_STRIDE,
    GPU_TARGET_STRIDE,
    GPU_TURN_STATUS_RECORD_STRIDE,
    MAX_CANDIDATES,
    MAX_FRONTIER_WIDTH,
    MAX_SELECTION_SCAN,
    MIN_CANDIDATES,
    MIN_FRONTIER_WIDTH,
};

#[test]
fn bot_search_policy_matches_web_timeout_contract() {
    assert_eq!(bot_search_depth_at_least_one(0.0), 1);
    assert_eq!(bot_search_depth_at_least_one(2.9), 2);
    assert_eq!(bot_search_depth_at_least_one(f64::NAN), 1);

    assert_eq!(bot_next_search_depth(0, 5), 2);
    assert_eq!(bot_next_search_depth(2, 5), 4);
    assert_eq!(bot_next_search_depth(4, 5), 5);
    assert_eq!(bot_next_search_depth(0, 1), 1);

    assert_eq!(bot_worker_search_time_ms(10_000), 9_500);
    assert_eq!(bot_worker_search_time_ms(1_000), 900);
    assert_eq!(bot_worker_search_time_ms(20), 1);

    let config: serde_json::Value = serde_json::from_str(
        &bot_search_config_json(f64::NAN, f64::NAN, f64::NAN, f64::NAN).unwrap(),
    )
    .unwrap();
    assert_eq!(config["targetDepth"], 2);
    assert_eq!(config["minDepth"], 2);
    assert_eq!(config["nodes"], 64.0);
    assert_eq!(config["timeMs"], 10_000.0);

    let config: serde_json::Value =
        serde_json::from_str(&bot_search_config_json(1.9, 4.0, 0.0, -5.0).unwrap()).unwrap();
    assert_eq!(config["targetDepth"], 1);
    assert_eq!(config["minDepth"], 1);
    assert_eq!(config["nodes"], 1.0);
    assert_eq!(config["timeMs"], 1.0);

    let worker_config: serde_json::Value =
        serde_json::from_str(&gpu_worker_search_config_json(4.9, 2.1, 10_000.0).unwrap()).unwrap();
    assert_eq!(worker_config["requestedDepth"], 4);
    assert_eq!(worker_config["minimumDepth"], 2);
    assert_eq!(worker_config["searchTimeMs"], 10_000.0);
    assert_eq!(worker_config["deadlineDelayMs"], 8_000.0);

    let worker_config: serde_json::Value =
        serde_json::from_str(&gpu_worker_search_config_json(1.0, 3.0, f64::NAN).unwrap()).unwrap();
    assert_eq!(worker_config["requestedDepth"], 1);
    assert_eq!(worker_config["minimumDepth"], 1);
    assert_eq!(worker_config["searchTimeMs"], 10_000.0);
    assert!(worker_config["deadlineDelayMs"].is_null());

    assert_eq!(gpu_search_ranking_limit(f64::NAN), 64);
    assert_eq!(gpu_search_ranking_limit(4.0), 16);
    assert_eq!(gpu_search_ranking_limit(20.9), 20);
    assert_eq!(gpu_search_ranking_limit(256.0), 128);
    assert_eq!(gpu_search_reply_limit(1.0), 4);
    assert_eq!(gpu_search_reply_limit(8.9), 8);
    assert_eq!(gpu_search_reply_limit(64.0), 12);
    assert_eq!(gpu_reply_pressure_reply_limit(), 512);
    assert_eq!(gpu_search_validation_limit(1.0), 8);
    assert_eq!(gpu_search_validation_limit(20.9), 20);
    assert_eq!(gpu_search_validation_limit(64.0), 32);
    assert_eq!(gpu_search_nodes(f64::NAN), 64.0);
    assert_eq!(gpu_search_nodes(0.0), 0.0);
    assert_eq!(gpu_search_nodes(12.5), 12.5);
    assert_eq!(gpu_mutation_candidate_limit(0), 0);
    assert_eq!(gpu_mutation_candidate_limit(12), 12);
    assert_eq!(gpu_mutation_candidate_limit(99), 64);
    assert_eq!(gpu_mutation_candidate_workgroups(0), 0);
    assert_eq!(gpu_mutation_candidate_workgroups(1), 1);
    assert_eq!(gpu_mutation_candidate_workgroups(64), 1);
    assert_eq!(gpu_mutation_candidate_workgroups(65), 2);
    assert_eq!(gpu_turn_completion_max_moves(0, 0), 4);
    assert_eq!(gpu_turn_completion_max_moves(1, 3), 7);
    assert_eq!(gpu_turn_completion_max_moves(12, 3), 12);
    assert_eq!(gpu_candidate_max_dispatch_workgroups(), 65_535);
    assert_eq!(gpu_candidate_max_candidates_per_dispatch(), 65_535 * 64);
    assert_eq!(
        gpu_candidate_max_candidates_per_batch(128 * 1024 * 1024),
        (128 * 1024 * 1024) / (GPU_CANDIDATE_STRIDE * std::mem::size_of::<i32>())
    );
    assert_eq!(gpu_candidate_max_candidates_per_batch(96), 1);
    assert_eq!(gpu_candidate_source_batch_size(1000, 32), 31);
    assert_eq!(gpu_candidate_source_batch_size(1000, 0), 1000);
    assert_eq!(gpu_candidate_batch_source_count(100, 64, 32), 32);
    assert_eq!(gpu_candidate_batch_source_count(100, 96, 32), 4);
    assert_eq!(gpu_candidate_batch_candidate_count(7, 11), 77);
    assert_eq!(gpu_candidate_score_workgroups(0), 0);
    assert_eq!(gpu_candidate_score_workgroups(65), 2);
    assert_eq!(
        gpu_candidate_score_workgroups(usize::MAX),
        gpu_candidate_max_dispatch_workgroups()
    );
    assert_eq!(gpu_reply_score_workgroups_x(0), 0);
    assert_eq!(gpu_reply_score_workgroups_x(1), 1);
    assert_eq!(gpu_reply_score_workgroups_x(17), 2);
    assert_eq!(gpu_reply_score_workgroups_y(0), 0);
    assert_eq!(gpu_reply_score_workgroups_y(16), 1);
    assert_eq!(gpu_reply_score_workgroups_y(33), 3);

    assert_eq!(bot_completed_search_depth(2.0, 2, false), 2);
    assert_eq!(bot_completed_search_depth(3.0, 3, false), 0);
    assert_eq!(bot_completed_search_depth(3.0, 3, true), 3);
    assert_eq!(bot_completed_search_depth(1.0, 1, false), 0);
    assert_eq!(bot_completed_search_depth(1.0, 1, true), 1);
    assert_eq!(bot_completed_search_depth(1.0, 2, true), 0);
    assert_eq!(bot_completed_search_depth(f64::NAN, 2, true), 0);

    assert!(bot_result_ends_in_royal_capture(
        Some("royal-capture"),
        false,
        false
    ));
    assert!(bot_result_ends_in_royal_capture(None, true, false));
    assert!(!bot_result_ends_in_royal_capture(
        Some("stalemate"),
        false,
        true
    ));
    assert!(
        bot_result_ends_in_royal_capture_json(r#"{"resultReason":"royal-capture"}"#)
            .expect("result reason")
    );
    assert!(
        bot_result_ends_in_royal_capture_json(r#"{"gpuTerminal":true}"#).expect("gpu terminal")
    );
    assert!(!bot_result_ends_in_royal_capture_json(
        r#"{"terminal":true,"resultReason":"stalemate"}"#
    )
    .expect("non royal terminal"));
}

#[test]
fn bot_ranked_choices_match_controller_preference_order() {
    let request = serde_json::json!({
        "selectedMoves": [{
            "from": { "timelineId": 1, "time": 0, "x": 0, "y": 1 },
            "to": { "timelineId": 1, "time": 1, "x": 0, "y": 2 }
        }],
        "results": [
            {
                "partitionIndex": 2,
                "result": {
                    "status": "ok",
                    "moves": [{
                        "from": { "timelineId": 0, "time": 0, "x": 1, "y": 1 },
                        "to": { "timelineId": 0, "time": 1, "x": 1, "y": 2 }
                    }],
                    "score": 90,
                    "depth": 1,
                    "nodes": 8,
                    "gpuSearch": "legacy"
                }
            },
            {
                "partitionIndex": 1,
                "result": {
                    "status": "ok",
                    "moves": [],
                    "choices": [
                        {
                            "moves": [{
                                "from": { "timelineId": 1, "time": 0, "x": 0, "y": 1 },
                                "to": { "timelineId": 1, "time": 1, "x": 0, "y": 2 }
                            }],
                            "score": 5,
                            "depth": 3,
                            "nodes": 4,
                            "cpuSearch": "candidate"
                        },
                        {
                            "moves": [{
                                "from": { "timelineId": 0, "time": 0, "x": 1, "y": 1 },
                                "to": { "timelineId": 0, "time": 1, "x": 1, "y": 2 }
                            }],
                            "score": 1,
                            "depth": 2,
                            "nodes": 16
                        }
                    ]
                }
            }
        ]
    });

    let choices: serde_json::Value = serde_json::from_str(
        &bot_ranked_choices_json(&request.to_string()).expect("ranked choices"),
    )
    .expect("ranked choices parse");
    let choices = choices.as_array().expect("choices array");
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0]["depth"], 3);
    assert_eq!(choices[0]["selected"], true);
    assert_eq!(choices[0]["partitionIndex"], 1);
    assert_eq!(choices[0]["principalVariation"][0], choices[0]["moves"]);
    assert_eq!(choices[1]["score"], 1);
    assert_eq!(choices[1]["depth"], 2);
}

#[test]
fn bot_select_best_result_prefers_depth_score_nodes_and_legal_moves() {
    let results = serde_json::json!([
        {
            "status": "ok",
            "moves": [],
            "score": 1000,
            "depth": 9,
            "nodes": 99
        },
        {
            "status": "noLegalTurn",
            "moves": [{
                "from": { "timelineId": 0, "time": 0, "x": 0, "y": 0 },
                "to": { "timelineId": 0, "time": 1, "x": 0, "y": 1 }
            }],
            "score": 1000,
            "depth": 9,
            "nodes": 99
        },
        {
            "status": "ok",
            "moves": [{
                "from": { "timelineId": 0, "time": 0, "x": 0, "y": 0 },
                "to": { "timelineId": 0, "time": 1, "x": 0, "y": 1 }
            }],
            "score": 50,
            "depth": 2,
            "nodes": 8,
            "label": "score-loses-to-depth"
        },
        {
            "status": "ok",
            "moves": [{
                "from": { "timelineId": 0, "time": 0, "x": 1, "y": 0 },
                "to": { "timelineId": 0, "time": 1, "x": 1, "y": 1 }
            }],
            "score": 3,
            "depth": 3,
            "nodes": 10,
            "label": "nodes-wins"
        },
        {
            "status": "ok",
            "moves": [{
                "from": { "timelineId": 0, "time": 0, "x": 2, "y": 0 },
                "to": { "timelineId": 0, "time": 1, "x": 2, "y": 1 }
            }],
            "score": 3,
            "depth": 3,
            "nodes": 4,
            "label": "same-depth-score"
        }
    ]);

    let selected: serde_json::Value = serde_json::from_str(
        &bot_select_best_result_json(&results.to_string()).expect("selected result"),
    )
    .expect("selected result parse");
    assert_eq!(selected["label"], "nodes-wins");

    let none: serde_json::Value =
        serde_json::from_str(&bot_select_best_result_json("[]").expect("empty selection"))
            .expect("empty selection parse");
    assert!(none.is_null());
}

#[test]
fn native_gpu_model_search_returns_web_worker_compatible_json() {
    let response = search(GpuSearchRequest {
        model_path: Some(committed_model_path()),
        depth: 1,
        min_depth: Some(1),
        nodes: 64,
        time_ms: 1_000,
        ..GpuSearchRequest::default()
    })
    .expect("run native GPU model search");

    #[cfg(feature = "neural-wgpu")]
    {
        assert_eq!(response.gpu_search, "native-wgpu-frontier-depth1");
        assert_eq!(response.backend, "wgpu-frontier");
        assert_eq!(
            response.native_frontier_round.as_deref(),
            Some("wgpu-frontier-search rounds=1 candidates=20 selected=4 states=4 plans=4")
        );
    }
    #[cfg(not(feature = "neural-wgpu"))]
    {
        assert_eq!(response.gpu_search, "cpu-orchestrated-compact-value-model");
        assert_eq!(response.backend, "cpu-search-with-gpu-model");
        assert_eq!(response.native_frontier_round, None);
    }
    let value: serde_json::Value =
        serde_json::from_str(&response.result_json).expect("GPU model search JSON should parse");
    assert_eq!(value["status"], "ok");
    assert!(value["moves"].as_array().is_some());
    assert!(value["principalVariation"].as_array().is_some());
    assert_eq!(value["depth"], 1);
    #[cfg(feature = "neural-wgpu")]
    assert_eq!(value["nodes"], 20);
}

#[test]
fn native_gpu_model_search_rejects_missing_model() {
    let error = search(GpuSearchRequest {
        model_path: Some("missing-value-model.cfnn".to_string()),
        ..GpuSearchRequest::default()
    })
    .expect_err("missing compact model should fail");

    assert!(error.contains("failed to read GPU value model"));
}

#[test]
fn gpu_search_wire_codes_match_web_snapshot_contract() {
    assert_eq!(gpu_search_color_code("white"), Ok(0));
    assert_eq!(gpu_search_color_code("black"), Ok(1));
    assert_eq!(gpu_search_color_from_code(0), "white");
    assert_eq!(gpu_search_color_from_code(1), "black");
    assert_eq!(gpu_search_color_from_code(99), "white");
    assert_eq!(gpu_search_opposite_color("white"), Ok("black"));
    assert_eq!(gpu_search_opposite_color("black"), Ok("white"));
    assert!(gpu_search_opposite_color("neutral").is_err());
    assert!(gpu_search_color_code("green").is_err());

    assert_eq!(gpu_search_owner_code("neutral"), Ok(0));
    assert_eq!(gpu_search_owner_code("white"), Ok(1));
    assert_eq!(gpu_search_owner_code("black"), Ok(2));
    assert_eq!(gpu_search_owner_from_code(0), "neutral");
    assert_eq!(gpu_search_owner_from_code(1), "white");
    assert_eq!(gpu_search_owner_from_code(2), "black");
    assert_eq!(gpu_search_owner_from_code(99), "neutral");
    assert!(gpu_search_owner_code("purple").is_err());

    let piece_codes = [
        ("king", 1),
        ("commonKing", 2),
        ("queen", 3),
        ("royalQueen", 4),
        ("princess", 5),
        ("rook", 6),
        ("bishop", 7),
        ("unicorn", 8),
        ("dragon", 9),
        ("knight", 10),
        ("pawn", 11),
        ("brawn", 12),
    ];
    for (piece_type, code) in piece_codes {
        assert_eq!(gpu_search_piece_type_code(piece_type), Some(code));
        assert_eq!(gpu_search_piece_type_from_code(code), Some(piece_type));
        assert_eq!(gpu_search_piece_code(piece_type, "white"), Ok(code));
        assert_eq!(
            gpu_search_piece_code(piece_type, "black"),
            Ok(code | (1 << 8))
        );
    }
    assert_eq!(gpu_search_piece_type_code("archbishop"), None);
    assert_eq!(gpu_search_piece_type_from_code(99), None);
    assert!(gpu_search_piece_code("archbishop", "white").is_err());

    let encoded_board =
        gpu_search_board_to_square_codes(&[Some(("king", "white")), None, Some(("rook", "black"))])
            .expect("encode board pieces");
    assert_eq!(encoded_board.len(), 64);
    assert_eq!(encoded_board[0], 1);
    assert_eq!(encoded_board[1], 0);
    assert_eq!(encoded_board[2], 6 | (1 << 8));
    assert!(encoded_board[3..].iter().all(|value| *value == 0));
    assert_eq!(
        gpu_search_board_to_square_codes(&vec![Some(("pawn", "black")); 70])
            .expect("truncate long board")
            .len(),
        64
    );
    assert!(gpu_search_board_to_square_codes(&[Some(("archbishop", "white"))]).is_err());
    assert!(gpu_search_board_to_square_codes(&[Some(("king", "green"))]).is_err());

    assert_eq!(gpu_search_piece_from_code(0), None);
    assert_eq!(
        gpu_search_piece_from_code(6),
        Some(GpuDecodedPiece {
            piece_type: "rook",
            color: "white",
        })
    );
    assert_eq!(
        gpu_search_piece_from_code(12 | (1 << 8)),
        Some(GpuDecodedPiece {
            piece_type: "brawn",
            color: "black",
        })
    );

    let mut squares = vec![0; 64];
    squares[0] = 6;
    squares[7] = 1 | (1 << 8);
    squares[63] = 99;
    let board = gpu_search_square_codes_to_board(&squares);
    assert_eq!(board.len(), 8);
    assert_eq!(board[0].len(), 8);
    assert_eq!(
        board[0][0],
        Some(GpuDecodedPiece {
            piece_type: "rook",
            color: "white",
        })
    );
    assert_eq!(
        board[0][7],
        Some(GpuDecodedPiece {
            piece_type: "king",
            color: "black",
        })
    );
    assert_eq!(board[7][7], None);

    let truncated = gpu_search_square_codes_to_board(&[11 | (1 << 8)]);
    assert_eq!(
        truncated[0][0],
        Some(GpuDecodedPiece {
            piece_type: "pawn",
            color: "black",
        })
    );
    assert_eq!(truncated[0][1], None);
}

#[test]
fn gpu_search_selection_policy_matches_web_worker_contract() {
    let response = gpu_search_select_candidate_json(
        r#"{
            "temperature": 0,
            "randomSeed": 7,
            "candidates": [
                { "index": 0, "score": 50, "key": "b", "moveCount": 1 },
                { "index": 1, "score": 75, "key": "z", "moveCount": 1 },
                { "index": 2, "score": 75, "key": "a", "moveCount": 1 },
                { "index": 3, "score": 100, "key": "ignored", "moveCount": 0 }
            ]
        }"#,
    )
    .expect("selection response");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("selection response JSON");

    assert_eq!(value["selectedIndex"], 2);
    assert_eq!(value["rankedIndexes"], serde_json::json!([2, 1, 0]));

    let warm_response = gpu_search_select_candidate_json(
        r#"{
            "temperature": 1,
            "randomSeed": 1,
            "candidates": [
                { "index": 0, "score": 100, "key": "a", "moveCount": 1 },
                { "index": 1, "score": 99, "key": "b", "moveCount": 1 }
            ]
        }"#,
    )
    .expect("warm selection response");
    let warm_value: serde_json::Value =
        serde_json::from_str(&warm_response).expect("warm selection response JSON");
    assert!(warm_value["selectedIndex"].as_u64().is_some());
    assert_eq!(warm_value["rankedIndexes"], serde_json::json!([0, 1]));

    let fractional_seed_response = gpu_search_select_candidate_json(
        r#"{
            "temperature": 1,
            "randomSeed": 1.9,
            "candidates": [
                { "index": 0, "score": 100, "key": "a", "moveCount": 1 },
                { "index": 1, "score": 99, "key": "b", "moveCount": 1 }
            ]
        }"#,
    )
    .expect("fractional seed selection response");
    assert_eq!(fractional_seed_response, warm_response);

    let null_seed_response = gpu_search_select_candidate_json(
        r#"{
            "temperature": 1,
            "randomSeed": null,
            "candidates": [
                { "index": 0, "score": 100, "key": "a", "moveCount": 1 },
                { "index": 1, "score": 99, "key": "b", "moveCount": 1 }
            ]
        }"#,
    )
    .expect("null seed selection response");
    let null_seed_value: serde_json::Value =
        serde_json::from_str(&null_seed_response).expect("null seed selection response JSON");
    assert!(null_seed_value["selectedIndex"].as_u64().is_some());
    assert_eq!(null_seed_value["rankedIndexes"], serde_json::json!([0, 1]));
}

#[test]
fn gpu_search_choice_selection_normalizes_web_search_choices() {
    let response = gpu_search_select_choice_json(
        r#"{
            "temperature": 0,
            "randomSeed": 7,
            "candidates": [
                {
                    "score": 50,
                    "move": {
                        "from": { "timelineId": 1, "time": 0, "x": 1, "y": 1 },
                        "to": { "timelineId": 1, "time": 1, "x": 1, "y": 2 }
                    }
                },
                {
                    "score": 75,
                    "moves": [{
                        "from": { "timelineId": 2, "time": 0, "x": 2, "y": 2 },
                        "to": { "timelineId": 2, "time": 1, "x": 2, "y": 3 }
                    }]
                },
                { "score": 100, "moves": [] }
            ]
        }"#,
    )
    .expect("choice selection response");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("choice selection response JSON");

    assert_eq!(value["selectedIndex"], 1);
    assert_eq!(value["rankedIndexes"], serde_json::json!([1, 0]));
}

#[test]
fn gpu_search_selected_choice_attaches_engine_summarized_choices() {
    let response = gpu_search_selected_choice_json(
        r#"{
            "temperature": 0,
            "randomSeed": 7,
            "candidates": [
                {
                    "score": 50,
                    "depth": 2,
                    "gpuSearch": "first",
                    "move": {
                        "from": { "timelineId": 1, "time": 0, "x": 1, "y": 1 },
                        "to": { "timelineId": 1, "time": 1, "x": 1, "y": 2 }
                    }
                },
                {
                    "score": 75,
                    "depth": 3,
                    "gpuSearch": "selected",
                    "moves": [{
                        "from": { "timelineId": 2, "time": 0, "x": 2, "y": 2 },
                        "to": { "timelineId": 2, "time": 1, "x": 2, "y": 3 }
                    }]
                },
                { "score": 100, "gpuSearch": "unsupported", "moves": [] }
            ]
        }"#,
    )
    .expect("selected choice response");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("selected choice response JSON");

    assert_eq!(value["score"], 75);
    assert_eq!(value["gpuSearch"], "selected");
    assert_eq!(value["choices"].as_array().map(Vec::len), Some(2));
    assert_eq!(value["choices"][0]["rank"], 1);
    assert_eq!(value["choices"][0]["score"], 75);
    assert_eq!(value["choices"][1]["rank"], 2);
    assert_eq!(value["choices"][1]["score"], 50);
    assert_eq!(
        value["choices"][0]["moves"].as_array().map(Vec::len),
        Some(1)
    );

    let empty = gpu_search_selected_choice_json(r#"{ "candidates": [] }"#)
        .expect("empty selected choice response");
    assert_eq!(empty, "null");
}

#[test]
fn gpu_turn_status_records_match_web_worker_contract() {
    let records = gpu_turn_status_records_i32s_from_snapshot_json(
        r#"{
            "format": "json",
            "turn": "black",
            "timelines": [
                {
                    "id": 4,
                    "row": 1,
                    "owner": "black",
                    "boards": [
                        { "time": 2, "sideToMove": "white", "castling": 0, "enPassant": null, "origin": null, "board": [] },
                        { "time": 5, "sideToMove": "black", "castling": 0, "enPassant": null, "origin": null, "board": [] }
                    ]
                },
                {
                    "id": -1,
                    "row": 0,
                    "owner": "white",
                    "boards": [
                        { "time": 3, "sideToMove": "white", "castling": 0, "enPassant": null, "origin": null, "board": [] }
                    ]
                }
            ]
        }"#,
    )
    .expect("turn-status records");

    assert_eq!(records, vec![-1, 1, 3, 0, 4, 2, 5, 1,]);

    let fallback = gpu_turn_status_records_i32s_from_snapshot_json(
        r#"{ "format": "json", "turn": "black", "timelines": [] }"#,
    )
    .expect("fallback turn-status records");
    assert_eq!(fallback, vec![0, 0, 0, 1]);
}

#[test]
fn gpu_turn_status_json_matches_web_worker_contract() {
    let response = gpu_turn_status_json_from_i32s(&[0, 1, 6, 2]).expect("turn-status JSON");
    let value: serde_json::Value = serde_json::from_str(&response).expect("turn-status parses");
    assert_eq!(value["complete"], true);
    assert_eq!(value["nextTurn"], "black");
    assert_eq!(value["presentTime"], 6);
    assert_eq!(value["pendingPresentBoardCount"], 2);

    let fallback = gpu_turn_status_json_from_i32s(&[0, 2, 6, 2]).expect("fallback color");
    let fallback_value: serde_json::Value =
        serde_json::from_str(&fallback).expect("fallback turn-status parses");
    assert_eq!(fallback_value["nextTurn"], "white");

    let json_response = gpu_turn_status_json(r#"{"records":[0,1,6,2]}"#).expect("JSON turn-status");
    assert_eq!(json_response, response);
    let supported = gpu_full_search_precondition_json(
        r#"{"status":{"complete":false,"pendingPresentBoardCount":1}}"#,
    )
    .expect("supported full-search precondition");
    let supported_value: serde_json::Value =
        serde_json::from_str(&supported).expect("supported precondition JSON");
    assert_eq!(supported_value["supported"], true);
    assert!(supported_value["error"].is_null());
    let unsupported = gpu_full_search_precondition_json(
        r#"{"status":{"complete":false,"pendingPresentBoardCount":2}}"#,
    )
    .expect("unsupported full-search precondition");
    let unsupported_value: serde_json::Value =
        serde_json::from_str(&unsupported).expect("unsupported precondition JSON");
    assert_eq!(unsupported_value["supported"], false);
    assert_eq!(
        unsupported_value["error"],
        "Full GPU search currently requires one pending present board."
    );
    let error = gpu_turn_status_json(r#"{"records":[0,1,6]}"#).expect_err("truncated JSON status");
    assert!(error.contains("truncated"));
}

#[test]
fn gpu_pending_present_boards_json_matches_web_worker_contract() {
    let json = gpu_pending_present_boards_json_from_snapshot_json(
        r#"{
            "format": "json",
            "turn": "white",
            "timelines": [
                {
                    "id": -1,
                    "row": 0,
                    "owner": "white",
                    "boards": [
                        { "time": 3, "sideToMove": "white", "castling": 0, "enPassant": null, "origin": null, "board": [] }
                    ]
                },
                {
                    "id": 0,
                    "row": 1,
                    "owner": "white",
                    "boards": [
                        { "time": 3, "sideToMove": "white", "castling": 0, "enPassant": null, "origin": null, "board": [] }
                    ]
                },
                {
                    "id": 1,
                    "row": 2,
                    "owner": "white",
                    "boards": [
                        { "time": 4, "sideToMove": "white", "castling": 0, "enPassant": null, "origin": null, "board": [] }
                    ]
                },
                {
                    "id": 2,
                    "row": 3,
                    "owner": "black",
                    "boards": [
                        { "time": 3, "sideToMove": "black", "castling": 0, "enPassant": null, "origin": null, "board": [] }
                    ]
                }
            ]
        }"#,
    )
    .expect("pending present boards");
    let value: serde_json::Value = serde_json::from_str(&json).expect("pending JSON parses");
    assert_eq!(
        value,
        serde_json::json!([
            { "timelineId": -1, "time": 3 },
            { "timelineId": 0, "time": 3 }
        ])
    );
}

#[test]
fn gpu_ranked_candidate_indexes_match_web_worker_contract() {
    fn candidate_record(
        from_timeline: i32,
        from_time: i32,
        to_timeline: i32,
        to_time: i32,
    ) -> [i32; GPU_CANDIDATE_STRIDE] {
        let mut record = [0; GPU_CANDIDATE_STRIDE];
        record[11] = from_timeline;
        record[12] = from_time;
        record[13] = 1;
        record[14] = 2;
        record[15] = to_timeline;
        record[16] = to_time;
        record[17] = 3;
        record[18] = 4;
        record
    }

    let scores = [10, 30, -2_147_480_000, 20, 30];
    let records = [
        candidate_record(1, 5, 1, 6),
        candidate_record(2, 7, 2, 8),
        candidate_record(1, 5, 1, 7),
        candidate_record(9, 9, 9, 10),
        candidate_record(1, 5, 1, 8),
    ];
    let mut request = vec![scores.len() as i32, 2, 1, 1];
    request.extend(scores);
    request.extend(records.into_iter().flatten());
    request.extend([1, 5]);

    assert_eq!(
        gpu_ranked_candidate_indexes_from_i32s(&request).expect("ranked indexes"),
        vec![4, 0]
    );

    request[2] = 0;
    request[3] = 0;
    request.truncate(4 + scores.len() + scores.len() * GPU_CANDIDATE_STRIDE);
    assert_eq!(
        gpu_ranked_candidate_indexes_from_i32s(&request).expect("unfiltered ranked indexes"),
        vec![1, 4]
    );
    let ranked_json =
        gpu_ranked_candidates_json_from_i32s(&request).expect("ranked candidate JSON");
    let ranked: serde_json::Value =
        serde_json::from_str(&ranked_json).expect("ranked candidate JSON parses");
    assert_eq!(ranked.as_array().map(Vec::len), Some(2));
    assert_eq!(ranked[0]["index"], 1);
    assert_eq!(ranked[0]["score"], 30);
    assert_eq!(ranked[0]["move"]["from"]["timelineId"], 2);
    assert_eq!(ranked[0]["move"]["from"]["time"], 7);
    assert_eq!(ranked[0]["move"]["to"]["timelineId"], 2);
    assert_eq!(ranked[0]["move"]["to"]["time"], 8);

    let json_request = serde_json::json!({
        "scores": scores,
        "records": request[4 + scores.len()..],
        "pendingBoards": [],
        "requirePending": false,
        "limit": 2
    });
    assert_eq!(
        gpu_ranked_candidates_json(&json_request.to_string()).expect("JSON ranked candidates"),
        ranked_json
    );
    let error = gpu_ranked_candidates_json(
        r#"{"scores":[1],"records":[1],"pendingBoards":[],"requirePending":false,"limit":1}"#,
    )
    .expect_err("short JSON records");
    assert!(error.contains("record length mismatch"));
}

#[test]
fn gpu_scoring_summary_matches_web_worker_contract() {
    fn candidate_record(from_timeline: i32, from_time: i32) -> [i32; GPU_CANDIDATE_STRIDE] {
        let mut record = [0; GPU_CANDIDATE_STRIDE];
        record[11] = from_timeline;
        record[12] = from_time;
        record
    }

    let scores = [10, -2_147_480_000, 30];
    let records = [
        candidate_record(1, 5),
        candidate_record(1, 5),
        candidate_record(2, 7),
    ];
    let mut request = vec![scores.len() as i32, 1];
    request.extend(scores);
    request.extend(records.into_iter().flatten());
    request.extend([1, 5]);

    assert_eq!(
        gpu_scoring_summary_from_i32s(&request).expect("scoring summary"),
        "validScores=2, pendingStarts=1, best=30"
    );

    let json_request = serde_json::json!({
        "scores": scores,
        "records": records.into_iter().flatten().collect::<Vec<_>>(),
        "pendingBoards": [{ "timelineId": 1, "time": 5 }]
    });
    assert_eq!(
        gpu_scoring_summary_json(&json_request.to_string()).expect("JSON scoring summary"),
        "validScores=2, pendingStarts=1, best=30"
    );

    let error = gpu_scoring_summary_json(r#"{"scores":[1],"records":[],"pendingBoards":[]}"#)
        .expect_err("short record summary");
    assert!(error.contains("record length mismatch"));
}

#[test]
fn gpu_mutation_summary_matches_web_worker_contract() {
    assert_eq!(gpu_mutation_summary_from_i32s(&[]), "none");
    assert_eq!(
        gpu_mutation_summary_from_i32s(&[3, 1, 3, 4, 1, 1]),
        "1:3,3:2,4:1"
    );
    assert_eq!(
        gpu_mutation_summary_json(r#"{"statuses":[3,1,3,4,1,1]}"#).expect("JSON mutation summary"),
        "1:3,3:2,4:1"
    );
    assert_eq!(
        gpu_mutation_summary_json(r#"{"statuses":[]}"#).expect("empty JSON mutation summary"),
        "none"
    );
    assert_eq!(
        gpu_mutation_statuses_json(r#"{"statuses":[3,1],"candidateCount":4}"#)
            .expect("normalized mutation statuses"),
        "[3,1,0,0]"
    );
    assert_eq!(
        gpu_mutation_statuses_json(r#"{"statuses":[3,1,2],"candidateCount":2}"#)
            .expect("trimmed mutation statuses"),
        "[3,1]"
    );
}

#[test]
fn gpu_supported_mutation_candidates_match_web_worker_contract() {
    let request = [
        5,
        0,
        1,
        0,
        1,
        GPU_MUTATION_STATUS_OK,
        0,
        GPU_MUTATION_STATUS_OK,
        1,
        GPU_MUTATION_STATUS_BRANCH_OK,
        1,
        GPU_MUTATION_STATUS_ROYAL_CAPTURE,
        1,
    ];
    assert_eq!(
        gpu_supported_mutation_candidate_indexes_from_i32s(&request).expect("supported indexes"),
        vec![2, 3, 4]
    );

    let mut limited = request;
    limited[1] = 2;
    assert_eq!(
        gpu_supported_mutation_candidate_indexes_from_i32s(&limited).expect("limited indexes"),
        vec![2, 3]
    );
    let mut status_only = request;
    status_only[2] = 0;
    assert_eq!(
        gpu_supported_mutation_candidate_indexes_from_i32s(&status_only).expect("status indexes"),
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        serde_json::from_str::<Vec<i32>>(
            &gpu_supported_mutation_candidate_indexes_json(
                r#"{
                    "candidates": [
                        { "mutationStatus": 0, "hasChildBoards": true },
                        { "mutationStatus": 1, "hasChildBoards": false },
                        { "mutationStatus": 1, "hasChildBoards": true },
                        { "mutationStatus": 3, "hasChildBoards": true },
                        { "mutationStatus": 2, "hasChildBoards": true }
                    ],
                    "limit": 2,
                    "requireChildBoards": true
                }"#,
            )
            .expect("supported mutation JSON"),
        )
        .expect("supported mutation JSON indexes"),
        vec![2, 3]
    );
    assert_eq!(
        serde_json::from_str::<Vec<i32>>(
            &gpu_supported_mutation_candidate_indexes_json(
                r#"{
                    "candidates": [
                        { "mutationStatus": 0, "hasChildBoards": true },
                        { "mutationStatus": 1, "hasChildBoards": false },
                        { "mutationStatus": 1, "hasChildBoards": true }
                    ],
                    "requireChildBoards": false
                }"#,
            )
            .expect("status-only supported mutation JSON"),
        )
        .expect("status-only supported mutation JSON indexes"),
        vec![1, 2]
    );
    assert_eq!(
        serde_json::from_str::<Vec<i32>>(
            &gpu_supported_mutation_candidate_indexes_json(
                r#"{
                    "candidates": [
                        { "mutationStatus": 1, "hasChildBoards": true },
                        { "mutationStatus": 1, "hasChildBoards": true },
                        { "mutationStatus": 1, "hasChildBoards": true }
                    ],
                    "limit": 2.9
                }"#,
            )
            .expect("fractional limit supported mutation JSON"),
        )
        .expect("fractional limit supported mutation JSON indexes"),
        vec![0, 1]
    );
    assert_eq!(
        serde_json::from_str::<Vec<i32>>(
            &gpu_supported_mutation_candidate_indexes_json(
                r#"{
                    "candidates": [
                        { "mutationStatus": 1, "hasChildBoards": true },
                        { "mutationStatus": 1, "hasChildBoards": true }
                    ],
                    "limit": -1
                }"#,
            )
            .expect("negative limit supported mutation JSON"),
        )
        .expect("negative limit supported mutation JSON indexes"),
        vec![0, 1]
    );
    assert_eq!(
        serde_json::from_str::<Vec<i32>>(
            &gpu_supported_mutation_candidate_indexes_json(
                r#"{
                    "candidates": [
                        { "mutationStatus": 1, "hasChildBoards": true },
                        { "mutationStatus": 1, "hasChildBoards": true }
                    ],
                    "limit": null
                }"#,
            )
            .expect("null limit supported mutation JSON"),
        )
        .expect("null limit supported mutation JSON indexes"),
        vec![0, 1]
    );

    assert!(
        gpu_supported_mutation_candidate_indexes_from_i32s(&[2, 0, GPU_MUTATION_STATUS_OK])
            .is_err()
    );
}

#[test]
fn gpu_mutation_terminal_status_matches_web_worker_contract() {
    assert!(!gpu_mutation_status_is_terminal(0));
    assert!(!gpu_mutation_status_is_terminal(GPU_MUTATION_STATUS_OK));
    assert!(gpu_mutation_status_is_terminal(
        GPU_MUTATION_STATUS_ROYAL_CAPTURE
    ));
    assert!(!gpu_mutation_status_is_terminal(
        GPU_MUTATION_STATUS_BRANCH_OK
    ));
    assert!(gpu_mutation_status_is_terminal(
        GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE
    ));
}

#[test]
fn gpu_turn_completion_key_matches_web_worker_contract() {
    assert_eq!(
        gpu_turn_completion_key_json(
            r#"[
                { "timelineId": 3, "time": 1 },
                { "timelineId": -1, "time": 5 },
                { "timelineId": 3, "time": 0 }
            ]"#,
        )
        .expect("turn completion key"),
        "-1:5|3:0|3:1"
    );
    assert_eq!(
        gpu_turn_completion_key_json("[]").expect("empty turn completion key"),
        ""
    );
}

#[test]
fn gpu_choice_agreement_matches_web_worker_contract() {
    let response = gpu_choice_agreement_json(
        r#"{
            "selected": [
                {
                    "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                    "to": { "timelineId": 5, "time": 6, "x": 7, "y": 8 }
                }
            ],
            "choices": [
                [
                    {
                        "from": { "timelineId": 9, "time": 9, "x": 9, "y": 9 },
                        "to": { "timelineId": 8, "time": 8, "x": 8, "y": 8 }
                    }
                ],
                [
                    {
                        "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                        "to": { "timelineId": 5, "time": 6, "x": 7, "y": 8 }
                    }
                ]
            ],
            "limits": [1, 2, 20]
        }"#,
    )
    .expect("choice agreement response");
    let value: serde_json::Value = serde_json::from_str(&response).expect("choice agreement JSON");
    assert_eq!(value["agreements"], serde_json::json!([0, 1, 1]));
    let diagnostics = gpu_policy_choice_agreement_diagnostics_json(
        r#"{
            "selected": {
                "move": {
                    "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                    "to": { "timelineId": 5, "time": 6, "x": 7, "y": 8 }
                }
            },
            "choices": [
                {
                    "move": {
                        "from": { "timelineId": 9, "time": 9, "x": 9, "y": 9 },
                        "to": { "timelineId": 8, "time": 8, "x": 8, "y": 8 }
                    }
                },
                {
                    "move": {
                        "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                        "to": { "timelineId": 5, "time": 6, "x": 7, "y": 8 }
                    }
                }
            ]
        }"#,
    )
    .expect("policy choice agreement diagnostics");
    let diagnostics: serde_json::Value =
        serde_json::from_str(&diagnostics).expect("policy diagnostics JSON");
    assert_eq!(diagnostics["topPolicyChoiceAgreement"], 0);
    assert_eq!(diagnostics["top5PolicyChoiceAgreement"], 1);
    assert_eq!(diagnostics["top20PolicyChoiceAgreement"], 1);

    let empty = gpu_choice_agreement_json(r#"{ "selected": [], "choices": [[]], "limits": [1] }"#)
        .expect("empty choice agreement response");
    let empty_value: serde_json::Value =
        serde_json::from_str(&empty).expect("empty choice agreement JSON");
    assert_eq!(empty_value["agreements"], serde_json::json!([0]));
}

#[test]
fn gpu_choice_agreement_choices_normalizes_web_search_choices() {
    let response = gpu_choice_agreement_choices_json(
        r#"{
            "selected": {
                "move": {
                    "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                    "to": { "timelineId": 5, "time": 6, "x": 7, "y": 8 }
                }
            },
            "choices": [
                {
                    "move": {
                        "from": { "timelineId": 9, "time": 9, "x": 9, "y": 9 },
                        "to": { "timelineId": 8, "time": 8, "x": 8, "y": 8 }
                    }
                },
                {
                    "moves": [{
                        "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                        "to": { "timelineId": 5, "time": 6, "x": 7, "y": 8 }
                    }]
                }
            ],
            "limits": [1, 2]
        }"#,
    )
    .expect("choice agreement choices response");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("choice agreement choices JSON");
    assert_eq!(value["agreements"], serde_json::json!([0, 1]));
}

#[test]
fn gpu_move_plan_key_matches_web_worker_contract() {
    assert_eq!(
        gpu_move_plan_key_json(
            r#"[
                {
                    "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                    "to": { "timelineId": 5, "time": 6, "x": 7, "y": 8 }
                },
                {
                    "from": { "timelineId": -1, "time": 0, "x": 2, "y": 3 },
                    "to": { "timelineId": -1, "time": 1, "x": 2, "y": 4 }
                }
            ]"#,
        )
        .expect("move plan key"),
        "1:2:3:4:5:6:7:8/-1:0:2:3:-1:1:2:4"
    );
    assert_eq!(
        gpu_move_plan_key_json("[]").expect("empty move plan key"),
        ""
    );
}

#[test]
fn gpu_turn_completion_step_matches_web_worker_contract() {
    let search = gpu_turn_completion_step_json(
        r#"{
            "snapshot": {
                "format": "chronofish-gpu-snapshot-v1",
                "turn": "white",
                "nextTimelineId": 1,
                "nextBlackTimelineId": -1,
                "royalCaptureBy": null,
                "timelines": [
                    { "id": 1, "row": 0, "owner": "white", "boards": [] },
                    { "id": -1, "row": 1, "owner": "black", "boards": [] }
                ]
            },
            "movesLength": 1,
            "pendingBoards": [
                { "timelineId": 5, "time": 4 },
                { "timelineId": 1, "time": 2 }
            ],
            "status": { "complete": false, "pendingPresentBoardCount": 2 },
            "visitedKeys": []
        }"#,
    )
    .expect("search step");
    let value: serde_json::Value = serde_json::from_str(&search).expect("search step JSON");
    assert_eq!(value["action"], "search");
    assert_eq!(value["stateKey"], "1:2|5:4");
    assert_eq!(value["maxMoves"], 6);
    assert_eq!(
        gpu_incomplete_turn_pending_present_board_count(Some(1), 2),
        2
    );
    assert_eq!(
        gpu_incomplete_turn_pending_present_board_count(Some(3), 2),
        3
    );
    assert_eq!(
        gpu_incomplete_turn_pending_present_board_count_json(
            r#"{
                "pendingBoards": [
                    { "timelineId": 5, "time": 4 },
                    { "timelineId": 1, "time": 2 }
                ],
                "status": { "complete": false, "pendingPresentBoardCount": 1 }
            }"#
        )
        .expect("incomplete pending count"),
        2
    );

    let complete = gpu_turn_completion_step_json(
        r#"{
            "snapshot": {
                "turn": "white",
                "royalCaptureBy": null,
                "timelines": []
            },
            "movesLength": 0,
            "pendingBoards": [],
            "status": { "complete": true, "pendingPresentBoardCount": 0 },
            "visitedKeys": []
        }"#,
    )
    .expect("complete step");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&complete).unwrap()["action"],
        "complete"
    );

    let terminal = gpu_turn_completion_step_json(
        r#"{
            "snapshot": {
                "turn": "white",
                "royalCaptureBy": "white",
                "timelines": []
            },
            "movesLength": 0,
            "pendingBoards": [{ "timelineId": 1, "time": 2 }],
            "status": { "complete": false, "pendingPresentBoardCount": 1 },
            "visitedKeys": []
        }"#,
    )
    .expect("terminal step");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&terminal).unwrap()["action"],
        "terminal"
    );

    let looped = gpu_turn_completion_step_json(
        r#"{
            "snapshot": {
                "turn": "white",
                "royalCaptureBy": null,
                "timelines": []
            },
            "movesLength": 0,
            "pendingBoards": [{ "timelineId": 1, "time": 2 }],
            "status": { "complete": false, "pendingPresentBoardCount": 1 },
            "visitedKeys": ["1:2"]
        }"#,
    )
    .expect("loop step");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&looped).unwrap()["action"],
        "loop"
    );

    let capped = gpu_turn_completion_step_json(
        r#"{
            "snapshot": {
                "turn": "white",
                "royalCaptureBy": null,
                "timelines": [
                    { "id": 1, "row": 0, "owner": "white", "boards": [] }
                ]
            },
            "movesLength": 5,
            "pendingBoards": [{ "timelineId": 1, "time": 2 }],
            "status": { "complete": false, "pendingPresentBoardCount": 1 },
            "visitedKeys": []
        }"#,
    )
    .expect("max move step");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&capped).unwrap()["action"],
        "maxMoves"
    );
}

#[test]
fn gpu_principal_variation_normalization_matches_bot_controller_contract() {
    let response = gpu_normalize_principal_variation_json(
        r#"{
            "variation": [
                [
                    null,
                    {
                        "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                        "to": { "timelineId": 5, "time": 6, "x": 7, "y": 8 }
                    }
                ],
                [],
                [
                    {
                        "from": { "timelineId": -1, "time": 0, "x": 2, "y": 3 },
                        "to": { "timelineId": -1, "time": 1, "x": 2, "y": 4 }
                    }
                ]
            ],
            "fallback": [
                {
                    "from": { "timelineId": 9, "time": 9, "x": 0, "y": 0 },
                    "to": { "timelineId": 9, "time": 10, "x": 0, "y": 1 }
                }
            ]
        }"#,
    )
    .expect("principal variation");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("principal variation JSON");
    assert_eq!(value.as_array().map(Vec::len), Some(2));
    assert_eq!(value[0].as_array().map(Vec::len), Some(1));
    assert_eq!(value[0][0]["from"]["timelineId"], 1);
    assert_eq!(value[1][0]["from"]["timelineId"], -1);

    let fallback = gpu_normalize_principal_variation_json(
        r#"{
            "variation": [[null]],
            "fallback": [
                {
                    "from": { "timelineId": 9, "time": 9, "x": 0, "y": 0 },
                    "to": { "timelineId": 9, "time": 10, "x": 0, "y": 1 }
                }
            ]
        }"#,
    )
    .expect("fallback variation");
    let value: serde_json::Value = serde_json::from_str(&fallback).expect("fallback JSON");
    assert_eq!(value.as_array().map(Vec::len), Some(1));
    assert_eq!(value[0][0]["from"]["timelineId"], 9);
}

#[test]
fn gpu_frontier_plan_decoder_matches_web_worker_contract() {
    let mut words = vec![0; FRONTIER_PLAN_OFFSET + FRONTIER_MOVE_STRIDE * 2 + 4];
    let offset = FRONTIER_PLAN_OFFSET + 4;
    words[offset..offset + FRONTIER_MOVE_STRIDE].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    words[offset + FRONTIER_MOVE_STRIDE..offset + FRONTIER_MOVE_STRIDE * 2]
        .copy_from_slice(&[-1, 0, 2, 3, -1, 1, 2, 4]);

    let value: serde_json::Value = serde_json::from_str(
        &gpu_frontier_plan_json_from_i32s(&words, offset, 2).expect("frontier plan JSON"),
    )
    .expect("frontier plan JSON parses");

    assert_eq!(value[0]["from"]["timelineId"], 1);
    assert_eq!(value[0]["to"]["y"], 8);
    assert_eq!(value[1]["from"]["timelineId"], -1);
    assert_eq!(value[1]["to"]["time"], 1);
    assert!(gpu_frontier_plan_json_from_i32s(&words, words.len() - 1, 2)
        .expect_err("truncated plan should fail")
        .contains("truncated"));
}

#[test]
fn gpu_frontier_choices_rank_dedupe_and_decode_plans() {
    let stride = frontier_state_stride(1);
    let mut words = vec![0; stride * 3];
    let first = 0;
    words[first + FRONTIER_HEADER_SCORE] = 10;
    words[first + FRONTIER_HEADER_DEPTH] = 2;
    words[first + FRONTIER_HEADER_PLAN_LENGTH] = 1;
    words[first + FRONTIER_PLAN_OFFSET..first + FRONTIER_PLAN_OFFSET + FRONTIER_MOVE_STRIDE]
        .copy_from_slice(&[1, 0, 0, 1, 1, 1, 0, 2]);

    let second = stride;
    words[second + FRONTIER_HEADER_SCORE] = 40;
    words[second + FRONTIER_HEADER_DEPTH] = 3;
    words[second + FRONTIER_HEADER_TERMINAL] = 1;
    words[second + FRONTIER_HEADER_PLAN_LENGTH] = 1;
    words[second + FRONTIER_PLAN_OFFSET..second + FRONTIER_PLAN_OFFSET + FRONTIER_MOVE_STRIDE]
        .copy_from_slice(&[2, 0, 3, 4, 2, 1, 3, 5]);

    let duplicate = stride * 2;
    words[duplicate + FRONTIER_HEADER_SCORE] = 80;
    words[duplicate + FRONTIER_HEADER_DEPTH] = 3;
    words[duplicate + FRONTIER_HEADER_PLAN_LENGTH] = 1;
    words
        [duplicate + FRONTIER_PLAN_OFFSET..duplicate + FRONTIER_PLAN_OFFSET + FRONTIER_MOVE_STRIDE]
        .copy_from_slice(&[2, 0, 3, 4, 2, 1, 3, 5]);

    let choices: serde_json::Value = serde_json::from_str(
        &gpu_frontier_choices_json_from_i32s(&words, 1, 3, 4, "resident-frontier", 12)
            .expect("frontier choices JSON"),
    )
    .expect("frontier choices parse");

    let choices = choices.as_array().expect("choices array");
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0]["score"], 80);
    assert_eq!(choices[0]["depth"], 3);
    assert_eq!(choices[0]["gpuTerminal"], false);
    assert_eq!(choices[0]["moves"][0]["from"]["timelineId"], 2);
    assert_eq!(choices[0]["gpuSearch"], "resident-frontier");
    assert_eq!(choices[1]["score"], 10);
    assert_eq!(choices[1]["depth"], 2);
}

#[test]
fn gpu_validated_frontier_choice_matches_web_worker_contract() {
    let accepted = gpu_validated_frontier_choice_json(
        r#"{
            "candidate": {
                "score": 42,
                "depth": 3,
                "gpuTerminal": true,
                "tactical": true,
                "gpuSearch": "candidate-frontier"
            },
            "moves": [
                {
                    "from": { "timelineId": 2, "time": 0, "x": 3, "y": 4 },
                    "to": { "timelineId": 2, "time": 1, "x": 3, "y": 5 }
                }
            ],
            "seenKeys": [],
            "choiceCount": 0,
            "choiceLimit": 12,
            "gpuSearch": "validated-frontier"
        }"#,
    )
    .expect("validated frontier choice");
    let value: serde_json::Value =
        serde_json::from_str(&accepted).expect("validated frontier choice JSON");
    assert_eq!(value["accepted"], true);
    assert_eq!(value["key"], "2:0:3:4:2:1:3:5");
    assert_eq!(value["choice"]["status"], "ok");
    assert_eq!(value["choice"]["score"], 42);
    assert_eq!(value["choice"]["moves"][0]["from"]["timelineId"], 2);
    assert_eq!(value["choice"]["principalVariation"][0][0]["to"]["y"], 5);
    assert_eq!(value["choice"]["gpu"], true);
    assert_eq!(value["choice"]["gpuMode"], "full");
    assert_eq!(value["choice"]["gpuSearch"], "validated-frontier");
    assert_eq!(value["choice"]["tactical"], true);

    let duplicate = gpu_validated_frontier_choice_json(
        r#"{
            "candidate": { "score": 42 },
            "moves": [
                {
                    "from": { "timelineId": 2, "time": 0, "x": 3, "y": 4 },
                    "to": { "timelineId": 2, "time": 1, "x": 3, "y": 5 }
                }
            ],
            "seenKeys": ["2:0:3:4:2:1:3:5"],
            "choiceCount": 0,
            "choiceLimit": 12,
            "gpuSearch": "validated-frontier"
        }"#,
    )
    .expect("duplicate frontier choice");
    let value: serde_json::Value =
        serde_json::from_str(&duplicate).expect("duplicate frontier choice JSON");
    assert_eq!(value["accepted"], false);
    assert_eq!(value["key"], "2:0:3:4:2:1:3:5");
    assert!(value["choice"].is_null());

    let capped = gpu_validated_frontier_choice_json(
        r#"{
            "candidate": { "score": 42 },
            "moves": [
                {
                    "from": { "timelineId": 2, "time": 0, "x": 3, "y": 4 },
                    "to": { "timelineId": 2, "time": 1, "x": 3, "y": 5 }
                }
            ],
            "seenKeys": [],
            "choiceCount": 12,
            "choiceLimit": 12,
            "gpuSearch": "validated-frontier"
        }"#,
    )
    .expect("capped frontier choice");
    let value: serde_json::Value =
        serde_json::from_str(&capped).expect("capped frontier choice JSON");
    assert_eq!(value["accepted"], false);
    assert!(value["key"].is_null());
    assert!(value["choice"].is_null());
}

#[test]
fn gpu_frontier_choice_diagnostics_match_web_worker_contract() {
    let response = gpu_frontier_choice_diagnostics_json(
        r#"{
            "selected": {
                "moves": [],
                "tactical": false
            },
            "choices": [
                { "moves": [], "tactical": true },
                { "moves": [], "tactical": false },
                { "moves": [] },
                { "moves": [], "tactical": true }
            ]
        }"#,
    )
    .expect("frontier choice diagnostics");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("frontier choice diagnostics JSON");
    assert_eq!(value["legalChoiceCount"], 4);
    assert_eq!(value["legalTacticalChoiceCount"], 2);
    assert_eq!(value["selectedMovePrunedRisk"], 1);
    assert_eq!(value["selectedMoveTactical"], 0);

    let tactical = gpu_frontier_choice_diagnostics_json(
        r#"{
            "selected": { "moves": [], "tactical": true },
            "choices": []
        }"#,
    )
    .expect("tactical frontier choice diagnostics");
    let value: serde_json::Value =
        serde_json::from_str(&tactical).expect("tactical frontier choice diagnostics JSON");
    assert_eq!(value["legalChoiceCount"], 0);
    assert_eq!(value["legalTacticalChoiceCount"], 0);
    assert_eq!(value["selectedMovePrunedRisk"], 0);
    assert_eq!(value["selectedMoveTactical"], 1);
}

#[test]
fn gpu_non_postable_result_summary_matches_web_worker_contract() {
    assert_eq!(
        gpu_non_postable_result_summary_json(
            r#"{
                "status": "incompleteTurn",
                "moves": [],
                "incompleteMoves": [1, 2],
                "pendingPresentBoardCount": 3
            }"#,
        )
        .expect("non-postable summary"),
        "status=incompleteTurn, moves=0, incomplete=2, pending=3"
    );
    assert_eq!(
        gpu_non_postable_result_summary_json("null").expect("null summary"),
        "status=unknown, moves=0, incomplete=0, pending=unknown"
    );
}

#[test]
fn gpu_postable_search_result_matches_web_worker_contract() {
    assert!(
        gpu_postable_search_result_json(r#"{ "status": "ok", "moves": [1] }"#)
            .expect("postable result")
    );
    assert!(
        !gpu_postable_search_result_json(r#"{ "status": "ok", "moves": [] }"#)
            .expect("empty moves")
    );
    assert!(
        !gpu_postable_search_result_json(r#"{ "status": "incompleteTurn", "moves": [1] }"#)
            .expect("wrong status")
    );
    assert!(!gpu_postable_search_result_json("null").expect("null result"));
}

#[test]
fn gpu_validate_first_frontier_turn_matches_web_replay_contract() {
    let response = gpu_validate_first_frontier_turn_json(
        r#"{
            "game": { "turn": "white", "timelines": [] },
            "moves": [{
                "from": { "timelineId": 0, "time": 0, "x": 0, "y": 0 },
                "to": { "timelineId": 0, "time": 0, "x": 1, "y": 0 }
            }]
        }"#,
    )
    .expect("invalid first frontier turn validation");
    assert_eq!(response, "[]");

    let response = gpu_validate_first_frontier_turn_json(
        r#"{
            "game": { "turn": "white", "timelines": [] },
            "moves": []
        }"#,
    )
    .expect("empty first frontier turn validation");
    assert_eq!(response, "[]");
}

#[test]
fn gpu_validate_search_result_matches_web_replay_contract() {
    let response = gpu_validate_search_result_json(
        r#"{
            "game": { "turn": "white", "timelines": [] },
            "result": { "status": "ok", "moves": [] }
        }"#,
    )
    .expect("empty result validation");
    assert_eq!(response, "null");

    let response = gpu_validate_search_result_json(
        r#"{
            "game": { "turn": "white", "timelines": [] },
            "result": {
                "status": "ok",
                "moves": [{
                    "from": { "timelineId": 0, "time": 0, "x": 0, "y": 0 },
                    "to": { "timelineId": 0, "time": 0, "x": 1, "y": 0 }
                }]
            }
        }"#,
    )
    .expect("invalid replay validation");
    assert_eq!(response, "null");
}

#[test]
fn gpu_completed_turn_choice_matches_web_worker_contract() {
    let response = gpu_completed_turn_choice_json(
        r#"{
            "result": {
                "status": "ok",
                "moves": [],
                "score": 42,
                "depth": 2,
                "nodes": 9,
                "gpuSearch": "old-search",
                "principalVariation": [
                    [],
                    [{
                        "from": { "timelineId": 8, "time": 7, "x": 6, "y": 5 },
                        "to": { "timelineId": 4, "time": 3, "x": 2, "y": 1 }
                    }]
                ],
                "choices": [
                    {
                        "rank": 3,
                        "score": 10,
                        "moves": [{
                            "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                            "to": { "timelineId": 5, "time": 6, "x": 7, "y": 8 }
                        }]
                    },
                    {
                        "rank": 4,
                        "score": 5,
                        "moves": [{
                            "from": { "timelineId": -1, "time": 0, "x": 2, "y": 3 },
                            "to": { "timelineId": -1, "time": 1, "x": 2, "y": 4 }
                        }]
                    }
                ]
            },
            "moves": [{
                "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                "to": { "timelineId": 5, "time": 6, "x": 7, "y": 8 }
            }],
            "gpuSearch": "completed-search"
        }"#,
    )
    .expect("completed-turn choice response");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("completed-turn choice JSON");
    assert_eq!(value["moves"].as_array().map(Vec::len), Some(1));
    assert_eq!(value["gpuSearch"], "completed-search");
    assert_eq!(
        value["principalVariation"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(value["choices"].as_array().map(Vec::len), Some(2));
    assert_eq!(value["choices"][0]["rank"], 1);
    assert_eq!(value["choices"][0]["score"], 42);
    assert_eq!(value["choices"][0]["gpuSearch"], "completed-search");
    assert_eq!(value["choices"][1]["rank"], 4);
}

#[test]
fn gpu_summarize_search_choices_matches_web_worker_contract() {
    let response = gpu_summarize_search_choices_json(
        r#"[
            {
                "score": 12,
                "move": {
                    "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                    "to": { "timelineId": 5, "time": 6, "x": 7, "y": 8 }
                },
                "depth": 2,
                "nodes": 99,
                "gpuSearch": "single",
                "gpuTerminal": true,
                "tactical": true
            },
            {
                "score": 5,
                "moves": [{
                    "from": { "timelineId": -1, "time": 0, "x": 2, "y": 3 },
                    "to": { "timelineId": -1, "time": 1, "x": 2, "y": 4 }
                }],
                "principalVariation": []
            }
        ]"#,
    )
    .expect("search choice summary");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("search choice summary JSON");
    assert_eq!(value.as_array().map(Vec::len), Some(2));
    assert_eq!(value[0]["rank"], 1);
    assert_eq!(value[0]["moves"].as_array().map(Vec::len), Some(1));
    assert_eq!(value[0]["gpuSearch"], "single");
    assert_eq!(value[0]["gpuTerminal"], true);
    assert_eq!(value[0]["tactical"], true);
    assert_eq!(value[1]["rank"], 2);
    assert_eq!(
        value[1]["principalVariation"].as_array().map(Vec::len),
        Some(0)
    );
}

#[test]
fn gpu_search_failure_summary_matches_web_worker_contract() {
    let mut board_a = vec![vec![serde_json::Value::Null; 8]; 8];
    board_a[0][0] = serde_json::json!({"type": "rook", "color": "black"});
    let mut board_b = vec![vec![serde_json::Value::Null; 8]; 8];
    board_b[7][7] = serde_json::json!({"type": "king", "color": "white"});
    let snapshot = serde_json::json!({
        "turn": "white",
        "timelines": [
            {
                "id": 2,
                "row": 2,
                "owner": "white",
                "boards": [{
                    "time": 1,
                    "sideToMove": "white",
                    "castling": 0,
                    "enPassant": null,
                    "board": board_b
                }]
            },
            {
                "id": -1,
                "row": -1,
                "owner": "black",
                "boards": [{
                    "time": 4,
                    "sideToMove": "black",
                    "castling": 9,
                    "enPassant": {"x": 2, "y": 3, "capturedX": 2, "capturedY": 4},
                    "board": board_a
                }]
            }
        ],
        "nextTimelineId": 3,
        "nextBlackTimelineId": -2
    })
    .to_string();

    assert_eq!(
        gpu_search_failure_summary_json(&snapshot).expect("failure summary"),
        "sources=2, targets=128, pending=1, timelines=2"
    );
}

#[test]
fn gpu_pick_candidate_records_matches_web_worker_contract() {
    let record_a = [1; GPU_CANDIDATE_STRIDE];
    let mut record_b = [2; GPU_CANDIDATE_STRIDE];
    record_b[1] = 7;
    let record_c = [3; GPU_CANDIDATE_STRIDE];
    let mut request = vec![3, 2];
    request.extend(record_a);
    request.extend(record_b);
    request.extend(record_c);
    request.extend([2, 0]);

    let picked = gpu_pick_candidate_records_from_i32s(&request).expect("picked candidate records");
    assert_eq!(picked.len(), 2 * GPU_CANDIDATE_STRIDE);
    assert_eq!(&picked[..GPU_CANDIDATE_STRIDE], &[3; GPU_CANDIDATE_STRIDE]);
    assert_eq!(&picked[GPU_CANDIDATE_STRIDE..], &[1; GPU_CANDIDATE_STRIDE]);
    let records = [record_a, record_b, record_c]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let json_request = serde_json::json!({
        "records": records,
        "indices": [2, 0]
    });
    let picked =
        gpu_pick_candidate_records_json(&json_request.to_string()).expect("JSON picked records");
    assert_eq!(picked.len(), 2 * GPU_CANDIDATE_STRIDE);
    assert_eq!(&picked[..GPU_CANDIDATE_STRIDE], &[3; GPU_CANDIDATE_STRIDE]);
    assert_eq!(&picked[GPU_CANDIDATE_STRIDE..], &[1; GPU_CANDIDATE_STRIDE]);
    let fractional_json_request = serde_json::json!({
        "records": records,
        "indices": [2.9, 0.2]
    });
    let picked = gpu_pick_candidate_records_json(&fractional_json_request.to_string())
        .expect("fractional JSON picked records");
    assert_eq!(picked.len(), 2 * GPU_CANDIDATE_STRIDE);
    assert_eq!(&picked[..GPU_CANDIDATE_STRIDE], &[3; GPU_CANDIDATE_STRIDE]);
    assert_eq!(&picked[GPU_CANDIDATE_STRIDE..], &[1; GPU_CANDIDATE_STRIDE]);
    let mut unaligned_records = records.clone();
    unaligned_records.extend([99, 100]);
    let unaligned_json_request = serde_json::json!({
        "records": unaligned_records,
        "indices": [1]
    });
    let picked = gpu_pick_candidate_records_json(&unaligned_json_request.to_string())
        .expect("unaligned JSON picked records");
    assert_eq!(picked.len(), GPU_CANDIDATE_STRIDE);
    assert_eq!(picked[0], 2);
    assert_eq!(picked[1], 7);
    assert_eq!(gpu_mutation_turn_code_from_records(&picked), 7);
    assert_eq!(
        gpu_mutation_turn_code_json(&serde_json::json!({ "records": picked }).to_string())
            .expect("mutation turn code"),
        7
    );
    assert_eq!(
        gpu_mutation_turn_code_json(r#"{"records":[]}"#).expect("empty mutation turn code"),
        0
    );

    let mut invalid = request;
    let index_offset = 2 + 3 * GPU_CANDIDATE_STRIDE;
    invalid[index_offset] = 3;
    let error = gpu_pick_candidate_records_from_i32s(&invalid).expect_err("out of range index");
    assert!(error.contains("out of range"));
    let error = gpu_pick_candidate_records_json(r#"{"records":[1],"indices":[0]}"#)
        .expect_err("partial JSON record only");
    assert!(error.contains("out of range"));
}

#[test]
fn gpu_mutation_selected_candidates_match_web_worker_contract() {
    let response = gpu_mutation_selected_candidates_json(
        &serde_json::json!({
            "ranked": [
                { "index": 3, "score": 30 },
                { "index": 1, "score": 20 },
                { "index": 2, "score": 10 }
            ],
            "limit": 2
        })
        .to_string(),
    )
    .expect("selected mutation candidates");
    let selected: serde_json::Value =
        serde_json::from_str(&response).expect("selected mutation candidates JSON");
    assert_eq!(selected.as_array().map(Vec::len), Some(2));
    assert_eq!(selected[0]["index"], 3);
    assert_eq!(selected[1]["index"], 1);
    let indexes =
        gpu_candidate_indexes_json(&serde_json::json!({ "candidates": selected }).to_string())
            .expect("candidate indexes");
    assert_eq!(indexes, "[3,1]");
    let scores = gpu_candidate_scores_json(
        &serde_json::json!({
            "scores": [10, 20, 30, 40],
            "candidates": selected,
            "fallback": -7
        })
        .to_string(),
    )
    .expect("candidate scores");
    assert_eq!(scores, "[40,20]");

    let empty = gpu_mutation_selected_candidates_json(r#"{"ranked":[{"index":1}],"limit":0}"#)
        .expect("empty selected mutation candidates");
    assert_eq!(empty, "[]");
    let error = gpu_candidate_indexes_json(r#"{"candidates":[{"score": 1}]}"#)
        .expect_err("missing candidate index");
    assert!(error.contains("missing"));
    let fallback = gpu_candidate_scores_json(
        r#"{"scores":[10],"candidates":[{"index":0},{"index":2}],"fallback":-7}"#,
    )
    .expect("fallback candidate score");
    assert_eq!(fallback, "[10,-7]");
    assert!(gpu_candidate_score_is_rejected(-2_147_480_000));
    assert!(!gpu_candidate_score_is_rejected(-2_147_479_999));
}

#[test]
fn gpu_candidate_index_matches_web_worker_contract() {
    let mut record_a = [0; GPU_CANDIDATE_STRIDE];
    record_a[11..19].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut record_b = [0; GPU_CANDIDATE_STRIDE];
    record_b[11..19].copy_from_slice(&[8, 7, 6, 5, 4, 3, 2, 1]);

    let mut request = vec![2, 8, 7, 6, 5, 4, 3, 2, 1];
    request.extend(record_a);
    request.extend(record_b);

    assert_eq!(
        gpu_candidate_index_from_i32s(&request).expect("candidate index"),
        1
    );
    let json_request = serde_json::json!({
        "records": request[9..],
        "move": {
            "from": { "timelineId": 8, "time": 7, "x": 6, "y": 5 },
            "to": { "timelineId": 4, "time": 3, "x": 2, "y": 1 }
        }
    });
    assert_eq!(
        gpu_candidate_index_json(&json_request.to_string()).expect("JSON candidate index"),
        1
    );

    request[1..9].copy_from_slice(&[9, 9, 9, 9, 9, 9, 9, 9]);
    assert_eq!(
        gpu_candidate_index_from_i32s(&request).expect("missing candidate index"),
        -1
    );
    let error = gpu_candidate_index_json(r#"{"records":[1],"move":{"from":{"timelineId":0,"time":0,"x":0,"y":0},"to":{"timelineId":0,"time":0,"x":0,"y":0}}}"#)
        .expect_err("unaligned JSON records");
    assert!(error.contains("stride-aligned"));
}

#[test]
fn gpu_reply_pressure_ranks_roots_like_web_worker_contract() {
    let request = vec![
        3, 2, // root count, reply count
        7, 2, 5, // root candidate indexes
        100, 120, 90, // root scores
        10, 80, // root 7 reply pressure => 20
        30, 40, // root 2 reply pressure => 80
        -5, 5, // root 5 reply pressure => 85
    ];

    assert_eq!(
        gpu_reply_pressure_ranked_roots_from_i32s(&request).expect("reply pressure roots"),
        vec![5, 85, 2, 80, 7, 20]
    );

    let malformed = &request[..request.len() - 1];
    let error = gpu_reply_pressure_ranked_roots_from_i32s(malformed).expect_err("bad length");
    assert!(error.contains("length mismatch"));
}

#[test]
fn gpu_reply_pressure_json_returns_adjusted_root_objects() {
    let response = gpu_reply_pressure_ranked_roots_json(
        r#"{
            "rankedRoots": [
                { "index": 7, "score": 100, "move": { "from": { "timelineId": 0, "time": 1, "x": 2, "y": 3 }, "to": { "timelineId": 0, "time": 1, "x": 2, "y": 4 } } },
                { "index": 2, "score": 120, "move": { "from": { "timelineId": 0, "time": 1, "x": 1, "y": 1 }, "to": { "timelineId": 0, "time": 1, "x": 1, "y": 2 } } },
                { "index": 5, "score": 90, "move": { "from": { "timelineId": 0, "time": 1, "x": 3, "y": 3 }, "to": { "timelineId": 0, "time": 1, "x": 3, "y": 4 } } }
            ],
            "pairScores": [10, 80, 30, 40, -5, 5],
            "replyCount": 2
        }"#,
    )
    .expect("reply pressure JSON");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("reply pressure JSON parses");
    assert_eq!(value[0]["index"], 5);
    assert_eq!(value[0]["score"], 85);
    assert_eq!(value[1]["index"], 2);
    assert_eq!(value[1]["score"], 80);
    assert_eq!(value[2]["index"], 7);
    assert_eq!(value[2]["score"], 20);
    assert_eq!(value[0]["move"]["from"]["x"], 3);

    let error = gpu_reply_pressure_ranked_roots_json(
        r#"{
            "rankedRoots": [{ "index": 1, "score": 2 }],
            "pairScores": [],
            "replyCount": 1
        }"#,
    )
    .expect_err("bad reply pressure JSON");
    assert!(error.contains("length mismatch"));
}

#[test]
fn gpu_frontier_root_encoders_match_web_contract() {
    assert_eq!(gpu_frontier_origin_code(None), 0);
    assert_eq!(gpu_frontier_origin_code(Some("source-advance")), 1);
    assert_eq!(gpu_frontier_origin_code(Some("branch")), 2);
    assert_eq!(gpu_frontier_origin_code(Some("cross-board")), 3);
    assert_eq!(gpu_frontier_origin_code(Some("castle")), 4);

    assert_eq!(gpu_frontier_hash_words(&[]), (-2128831035, -1640531527));
    assert_eq!(
        gpu_frontier_hash_words(&[1, -2, 3, 0x4000_0000, i32::MIN, i32::MAX]),
        (-1_617_142_616, -188_060_083)
    );
}

#[test]
fn gpu_frontier_root_snapshot_json_matches_web_worker_contract() {
    let snapshot = serde_json::json!({
        "turn": "white",
        "nextTimelineId": 1,
        "nextBlackTimelineId": -1,
        "timelines": [{
            "id": 0,
            "row": 0,
            "owner": "neutral",
            "boards": [{
                "time": 0,
                "sideToMove": "white",
                "castling": 15,
                "enPassant": null,
                "origin": null,
                "squares": [1, 0, 0, 0, 0, 0, 0, 257]
            }]
        }]
    })
    .to_string();

    let words = gpu_frontier_root_i32s_from_snapshot_json(&snapshot, 1)
        .expect("encode GPU snapshot frontier root");

    assert_eq!(words[FRONTIER_HEADER_TURN], 0);
    assert_eq!(words[FRONTIER_HEADER_BOARD_COUNT], 1);
    assert_eq!(words[FRONTIER_HEADER_PRESENT_TIME], 0);
    assert_eq!(words[FRONTIER_HEADER_PENDING_BOARDS], 1);
    assert_eq!(words[FRONTIER_BOARD_OFFSET + FRONTIER_BOARD_PENDING], 1);
}

#[test]
fn gpu_frontier_active_timeline_rules_match_web_contract() {
    assert_eq!(gpu_frontier_active_timeline_distance(&[]), 1);
    assert_eq!(gpu_frontier_active_timeline_distance(&[0]), 1);
    assert_eq!(gpu_frontier_active_timeline_distance(&[-1, 0, 1]), 2);
    assert_eq!(gpu_frontier_active_timeline_distance(&[-4, -1, 2]), 3);
    assert_eq!(gpu_frontier_active_timeline_distance(&[-2, 5]), 3);

    assert_eq!(gpu_frontier_timeline_active("neutral", 99, 0), Ok(true));
    assert_eq!(gpu_frontier_timeline_active("white", 2, 2), Ok(true));
    assert_eq!(gpu_frontier_timeline_active("black", -2, 2), Ok(true));
    assert_eq!(gpu_frontier_timeline_active("white", 3, 2), Ok(false));
    assert_eq!(gpu_frontier_timeline_active("black", -3, 2), Ok(false));
    assert!(gpu_frontier_timeline_active("purple", 0, 2).is_err());
}

#[test]
fn gpu_frontier_present_and_pending_counts_match_web_contract() {
    assert_eq!(gpu_frontier_present_time(&[]), 0);
    assert_eq!(gpu_frontier_pending_board_count(&[], 0, 0), 0);

    let active_latest = [
        GpuFrontierActiveBoard {
            time: 4,
            side_to_move: 0,
        },
        GpuFrontierActiveBoard {
            time: 2,
            side_to_move: 1,
        },
        GpuFrontierActiveBoard {
            time: 2,
            side_to_move: 0,
        },
        GpuFrontierActiveBoard {
            time: 5,
            side_to_move: 0,
        },
    ];
    assert_eq!(gpu_frontier_present_time(&active_latest), 2);
    assert_eq!(gpu_frontier_pending_board_count(&active_latest, 2, 0), 1);
    assert_eq!(gpu_frontier_pending_board_count(&active_latest, 2, 1), 1);
    assert_eq!(gpu_frontier_pending_board_count(&active_latest, 4, 0), 1);
    assert_eq!(gpu_frontier_pending_board_count(&active_latest, 5, 1), 0);
}

#[test]
fn gpu_timeline_and_latest_board_selection_match_web_contract() {
    let timelines = [
        GpuTimelineSortKey { row: 2, id: 4 },
        GpuTimelineSortKey { row: -1, id: 3 },
        GpuTimelineSortKey { row: -1, id: -2 },
        GpuTimelineSortKey { row: 0, id: 0 },
        GpuTimelineSortKey { row: 2, id: -1 },
    ];
    assert_eq!(gpu_timeline_sort_order(&timelines), vec![2, 1, 3, 4, 0]);
    assert_eq!(gpu_timeline_sort_order(&[]), Vec::<usize>::new());

    assert_eq!(gpu_latest_board_index(&[]), None);
    assert_eq!(gpu_latest_board_index(&[3]), Some(0));
    assert_eq!(gpu_latest_board_index(&[1, 4, 2, 4]), Some(1));
    assert_eq!(gpu_latest_board_index(&[-2, -1, -3]), Some(1));
}

#[test]
fn gpu_movegen_layout_matches_web_ai_layout_contract() {
    assert_eq!(GPU_CANDIDATE_STRIDE, 24);
    assert_eq!(GPU_SOURCE_STRIDE, 10);
    assert_eq!(GPU_TARGET_STRIDE, 10);
    assert_eq!(GPU_BOARD_STRIDE, 73);
    assert_eq!(GPU_MUTATION_BOARD_STRIDE, 76);
    assert_eq!(GPU_MUTATION_CHILD_STRIDE, 152);
    assert_eq!(GPU_MUTATION_STATUS_OK, 1);
    assert_eq!(GPU_MUTATION_STATUS_ROYAL_CAPTURE, 2);
    assert_eq!(GPU_MUTATION_STATUS_BRANCH_OK, 3);
    assert_eq!(GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE, 4);
    assert_eq!(GPU_TURN_STATUS_RECORD_STRIDE, 4);
    assert_eq!(MIN_FRONTIER_WIDTH, 8);
    assert_eq!(MAX_FRONTIER_WIDTH, 512);
    assert_eq!(MIN_CANDIDATES, 256);
    assert_eq!(MAX_CANDIDATES, 65_536);
    assert_eq!(MAX_SELECTION_SCAN, 2_048);
}

#[test]
fn gpu_square_record_encoder_matches_web_candidate_input_contract() {
    assert_eq!(GPU_SOURCE_STRIDE, GPU_TARGET_STRIDE);
    assert_eq!(
        gpu_square_record_from_code(GpuSquareRecordInput {
            piece_code: 6 | (1 << 8),
            timeline_id: -2,
            time: 5,
            x: 3,
            y: 4,
            timeline_row: 7,
            side_to_move: 1,
            owner: 2,
            latest: true,
        }),
        [6, 1, -2, 5, 3, 4, 7, 1, 2, 1]
    );
    assert_eq!(
        gpu_square_record_from_code(GpuSquareRecordInput {
            piece_code: 0,
            timeline_id: 0,
            time: 1,
            x: 2,
            y: 3,
            timeline_row: 0,
            side_to_move: 0,
            owner: 0,
            latest: false,
        }),
        [0, 0, 0, 1, 2, 3, 0, 0, 0, 0]
    );
    assert_eq!(
        gpu_square_record_from_code(GpuSquareRecordInput {
            piece_code: -1,
            timeline_id: 1,
            time: 2,
            x: 3,
            y: 4,
            timeline_row: 5,
            side_to_move: 0,
            owner: 1,
            latest: true,
        })[..2],
        [255, 255]
    );
}

#[test]
fn gpu_square_record_collections_match_web_candidate_input_contract() {
    let mut squares = vec![0; 64];
    squares[0] = 6 | (1 << 8);
    squares[7] = 11;
    squares[63] = 99;
    let board = GpuSquareRecordBoardInput {
        timeline_id: -2,
        time: 5,
        timeline_row: 7,
        side_to_move: 1,
        owner: 2,
        latest: true,
        squares,
    };

    let targets = gpu_target_square_records_for_board(&board);
    assert_eq!(targets.len(), 64);
    assert_eq!(
        targets[0],
        GpuCandidateSquareRecord {
            meta: GpuCandidatePosition {
                timeline_id: -2,
                time: 5,
                x: 0,
                y: 0,
            },
            words: [6, 1, -2, 5, 0, 0, 7, 1, 2, 1],
        }
    );
    assert_eq!(targets[7].meta.x, 7);
    assert_eq!(targets[7].meta.y, 0);
    assert_eq!(targets[63].meta.x, 7);
    assert_eq!(targets[63].meta.y, 7);
    assert_eq!(targets[63].words[..2], [99, 0]);

    let sources = gpu_source_square_records_for_board(&board);
    assert_eq!(sources.len(), 3);
    assert_eq!(sources[0], targets[0]);
    assert_eq!(sources[1], targets[7]);
    assert_eq!(sources[2], targets[63]);

    let truncated = gpu_target_square_records_for_board(&GpuSquareRecordBoardInput {
        squares: vec![12],
        ..board.clone()
    });
    assert_eq!(truncated.len(), 64);
    assert_eq!(truncated[0].words[..2], [12, 0]);
    assert_eq!(truncated[1].words[..2], [0, 0]);
    assert_eq!(
        gpu_source_square_records_for_board(&GpuSquareRecordBoardInput {
            squares: vec![0, 0, 0],
            ..board
        }),
        Vec::<GpuCandidateSquareRecord>::new()
    );
}

#[test]
fn gpu_candidate_board_bundle_matches_web_candidate_input_contract() {
    let mut squares = vec![0; 64];
    squares[0] = 6 | (1 << 8);
    squares[9] = 11;
    let records = gpu_candidate_board_records_from_snapshot(&GpuCandidateBoardInput {
        timeline_id: -2,
        timeline_row: 7,
        timeline_index: 3,
        time: 5,
        side_to_move: 1,
        owner: 2,
        castling: 9,
        en_passant: Some(GpuEnPassantRecord {
            x: 4,
            y: 5,
            captured_x: 4,
            captured_y: 6,
        }),
        latest: true,
        origin_kind: 3,
        squares,
    });

    assert_eq!(records.board.len(), GPU_BOARD_STRIDE);
    assert_eq!(&records.board[..9], &[-2, 7, 5, 1, 9, 4, 5, 4, 6]);
    assert_eq!(records.board[9], 6 | (1 << 8));
    assert_eq!(records.board[18], 11);
    assert_eq!(records.mutation_board.len(), GPU_MUTATION_BOARD_STRIDE);
    assert_eq!(
        &records.mutation_board[..12],
        &[3, -2, 5, 1, 9, 4, 5, 4, 6, 1, 3, 0]
    );
    assert_eq!(records.mutation_board[12], 6 | (1 << 8));
    assert_eq!(records.mutation_board[21], 11);
    assert_eq!(records.targets.len(), 64);
    assert_eq!(records.sources.len(), 2);
    assert_eq!(records.targets[0].words, [6, 1, -2, 5, 0, 0, 7, 1, 2, 1]);
    assert_eq!(records.targets[9].words, [11, 0, -2, 5, 1, 1, 7, 1, 2, 1]);
    assert_eq!(records.sources[0], records.targets[0]);
    assert_eq!(records.sources[1], records.targets[9]);

    assert_eq!(
        gpu_candidate_board_records_from_snapshot(&GpuCandidateBoardInput {
            timeline_id: 0,
            timeline_row: 0,
            timeline_index: 0,
            time: 0,
            side_to_move: 0,
            owner: 0,
            castling: 0,
            en_passant: None,
            latest: false,
            origin_kind: 0,
            squares: vec![],
        }),
        GpuCandidateBoardRecords {
            board: {
                let mut record = vec![0, 0, 0, 0, 0, -1, -1, -1, -1];
                record.resize(GPU_BOARD_STRIDE, 0);
                record
            },
            mutation_board: {
                let mut record = vec![0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0];
                record.resize(GPU_MUTATION_BOARD_STRIDE, 0);
                record
            },
            sources: vec![],
            targets: gpu_target_square_records_for_board(&GpuSquareRecordBoardInput {
                timeline_id: 0,
                time: 0,
                timeline_row: 0,
                side_to_move: 0,
                owner: 0,
                latest: false,
                squares: vec![],
            }),
        }
    );
}

#[test]
fn gpu_candidate_inputs_from_timelines_match_web_candidate_input_contract() {
    let inputs = gpu_candidate_inputs_from_timelines(&[
        GpuCandidateInputTimeline {
            id: 3,
            row: 2,
            owner: 1,
            boards: vec![GpuCandidateInputBoard {
                time: 1,
                side_to_move: 0,
                castling: 0,
                en_passant: None,
                origin_kind: 0,
                squares: {
                    let mut squares = vec![0; 64];
                    squares[0] = 11;
                    squares
                },
            }],
        },
        GpuCandidateInputTimeline {
            id: -1,
            row: -1,
            owner: 2,
            boards: vec![
                GpuCandidateInputBoard {
                    time: 4,
                    side_to_move: 1,
                    castling: 9,
                    en_passant: Some(GpuEnPassantRecord {
                        x: 2,
                        y: 3,
                        captured_x: 2,
                        captured_y: 4,
                    }),
                    origin_kind: 3,
                    squares: {
                        let mut squares = vec![0; 64];
                        squares[9] = 6 | (1 << 8);
                        squares
                    },
                },
                GpuCandidateInputBoard {
                    time: 2,
                    side_to_move: 0,
                    castling: 0,
                    en_passant: None,
                    origin_kind: 0,
                    squares: {
                        let mut squares = vec![0; 64];
                        squares[63] = 1;
                        squares
                    },
                },
            ],
        },
    ]);

    assert_eq!(inputs.board_count, 3);
    assert_eq!(inputs.target_count, 192);
    assert_eq!(inputs.source_count, 3);
    assert_eq!(inputs.boards.len(), GPU_BOARD_STRIDE * 3);
    assert_eq!(inputs.mutation_boards.len(), GPU_MUTATION_BOARD_STRIDE * 3);
    assert_eq!(inputs.targets.len(), GPU_TARGET_STRIDE * 192);
    assert_eq!(inputs.sources.len(), GPU_SOURCE_STRIDE * 3);

    assert_eq!(inputs.target_meta[0].timeline_id, -1);
    assert_eq!(inputs.target_meta[0].time, 4);
    assert_eq!(
        inputs.source_meta[0],
        GpuCandidatePosition {
            timeline_id: -1,
            time: 4,
            x: 1,
            y: 1,
        }
    );
    assert_eq!(
        &inputs.sources[..GPU_SOURCE_STRIDE],
        &[6, 1, -1, 4, 1, 1, -1, 1, 2, 1]
    );

    let second_board_target = 64 * GPU_TARGET_STRIDE;
    assert_eq!(
        &inputs.targets[second_board_target..second_board_target + GPU_TARGET_STRIDE],
        &[0, 0, -1, 2, 0, 0, -1, 0, 2, 0]
    );
    let third_board = GPU_BOARD_STRIDE * 2;
    assert_eq!(&inputs.boards[third_board..third_board + 3], &[3, 2, 1]);
    let third_source = GPU_SOURCE_STRIDE * 2;
    assert_eq!(
        &inputs.sources[third_source..third_source + GPU_SOURCE_STRIDE],
        &[11, 0, 3, 1, 0, 0, 2, 0, 1, 1]
    );

    assert_eq!(
        gpu_candidate_inputs_from_timelines(&[]),
        GpuCandidateInputs {
            source_meta: vec![],
            target_meta: vec![],
            source_count: 0,
            target_count: 0,
            board_count: 0,
            sources: vec![],
            targets: vec![],
            boards: vec![],
            mutation_boards: vec![],
        }
    );
}

#[test]
fn gpu_snapshot_game_json_matches_web_validation_snapshot_contract() {
    let game = gpu_snapshot_game_json(
        r#"{
            "turn": "white",
            "timelines": [{
                "id": 2,
                "row": 1,
                "label": "Main",
                "owner": "white",
                "boards": [{
                    "time": 3,
                    "sideToMove": "black",
                    "castling": 9,
                    "enPassant": {"x": 2, "y": 3, "capturedX": 2, "capturedY": 4},
                    "origin": {"type": "move"},
                    "squares": [262]
                }]
            }],
            "nextTimelineId": 4,
            "nextBlackTimelineId": -2,
            "royalCaptureBy": null
        }"#,
    )
    .expect("game snapshot JSON");
    let value: serde_json::Value = serde_json::from_str(&game).expect("game snapshot parses");
    assert_eq!(value["turn"], "white");
    assert_eq!(value["nextTimelineId"], 4);
    assert_eq!(value["nextBlackTimelineId"], -2);
    assert_eq!(value["checkedRoyals"], serde_json::json!([]));
    assert_eq!(value["timelines"][0]["label"], "Main");
    assert_eq!(value["timelines"][0]["boards"][0]["time"], 3);
    assert_eq!(value["timelines"][0]["boards"][0]["sideToMove"], "black");
    assert_eq!(
        value["timelines"][0]["boards"][0]["board"][0][0]["type"],
        "rook"
    );
    assert_eq!(
        value["timelines"][0]["boards"][0]["board"][0][0]["color"],
        "black"
    );
    assert_eq!(
        value["timelines"][0]["boards"][0]["board"][0][1],
        serde_json::Value::Null
    );
}

#[test]
fn gpu_snapshot_with_child_boards_json_matches_web_branch_contract() {
    let mut records = vec![0; GPU_MUTATION_CHILD_STRIDE];
    records[1] = 0;
    records[2] = 4;
    records[3] = 1;
    records[5] = -1;
    records[6] = -1;
    records[7] = -1;
    records[8] = -1;
    records[9] = 1;
    let branch = GPU_MUTATION_BOARD_STRIDE;
    records[branch + 1] = 0;
    records[branch + 2] = 3;
    records[branch + 3] = 1;
    records[branch + 5] = -1;
    records[branch + 6] = -1;
    records[branch + 7] = -1;
    records[branch + 8] = -1;
    records[branch + 9] = 1;

    let request = serde_json::json!({
        "snapshot": {
            "format": "engine-gpu-snapshot-v1",
            "turn": "white",
            "nextTimelineId": 1,
            "nextBlackTimelineId": -1,
            "royalCaptureBy": null,
            "timelines": [{
                "id": 0,
                "row": 0,
                "owner": "neutral",
                "boards": [
                    {"timelineIndex": 0, "timelineId": 0, "time": 2, "sideToMove": "white", "castling": 0, "enPassant": null, "latest": false, "originKind": 0, "squares": {"0": 6}},
                    {"timelineIndex": 0, "timelineId": 0, "time": 3, "sideToMove": "white", "castling": 0, "enPassant": null, "latest": true, "originKind": 0, "squares": {"0": 1}}
                ]
            }]
        },
        "childBoardRecords": records,
        "mutationStatus": GPU_MUTATION_STATUS_BRANCH_OK,
        "move": {
            "from": {"timelineId": 0, "time": 3, "x": 3, "y": 7},
            "to": {"timelineId": 0, "time": 2, "x": 3, "y": 5}
        },
        "advanceTurn": true
    });
    let output = gpu_snapshot_with_child_boards_json(&request.to_string()).expect("child snapshot");
    let value: serde_json::Value = serde_json::from_str(&output).expect("child snapshot JSON");

    assert_eq!(value["turn"], "black");
    assert_eq!(value["nextTimelineId"], 2);
    assert_eq!(value["timelines"].as_array().unwrap().len(), 2);
    assert_eq!(value["timelines"][0]["boards"].as_array().unwrap().len(), 3);
    assert_eq!(value["timelines"][0]["boards"][0]["latest"], false);
    assert_eq!(value["timelines"][0]["boards"][0]["squares"][0], 6);
    assert_eq!(value["timelines"][0]["boards"][2]["time"], 4);
    assert_eq!(
        value["timelines"][0]["boards"][2]["origin"]["type"],
        "source-advance"
    );
    assert_eq!(value["timelines"][1]["id"], 1);
    assert_eq!(value["timelines"][1]["row"], 1);
    assert_eq!(value["timelines"][1]["owner"], "white");
    assert_eq!(value["timelines"][1]["boards"][0]["time"], 3);
    assert_eq!(
        value["timelines"][1]["boards"][0]["origin"]["type"],
        "branch"
    );
    assert_eq!(value["boards"].as_array().unwrap().len(), 4);
}

#[test]
fn gpu_candidate_inputs_from_snapshot_json_matches_web_candidate_input_contract() {
    let mut board_a = vec![vec![serde_json::Value::Null; 8]; 8];
    board_a[0][0] = serde_json::json!({"type": "rook", "color": "black"});
    let mut board_b = vec![vec![serde_json::Value::Null; 8]; 8];
    board_b[7][7] = serde_json::json!({"type": "king", "color": "white"});
    let snapshot = serde_json::json!({
        "turn": "white",
        "timelines": [
            {
                "id": 2,
                "row": 2,
                "owner": "white",
                "boards": [{
                    "time": 1,
                    "sideToMove": "white",
                    "castling": 0,
                    "enPassant": null,
                    "board": board_b
                }]
            },
            {
                "id": -1,
                "row": -1,
                "owner": "black",
                "boards": [{
                    "time": 4,
                    "sideToMove": "black",
                    "castling": 9,
                    "enPassant": {"x": 2, "y": 3, "capturedX": 2, "capturedY": 4},
                    "board": board_a
                }]
            }
        ],
        "nextTimelineId": 3,
        "nextBlackTimelineId": -2
    })
    .to_string();

    let inputs =
        gpu_candidate_inputs_from_snapshot_json(&snapshot).expect("snapshot candidate inputs");
    let words =
        gpu_candidate_inputs_i32s_from_snapshot_json(&snapshot).expect("candidate input words");
    let json: serde_json::Value = serde_json::from_str(
        &gpu_candidate_inputs_json_from_snapshot_json(&snapshot).expect("candidate JSON"),
    )
    .expect("candidate JSON parses");
    assert_eq!(inputs.board_count, 2);
    assert_eq!(inputs.source_count, 2);
    assert_eq!(inputs.target_count, 128);
    assert_eq!(json["boardCount"], 2);
    assert_eq!(json["sourceCount"], 2);
    assert_eq!(json["targetCount"], 128);
    assert_eq!(json["sourceMeta"][0]["timelineId"], -1);
    let search_size: serde_json::Value = serde_json::from_str(
        &gpu_snapshot_search_size_json(&snapshot).expect("snapshot search size JSON"),
    )
    .expect("snapshot search size parses");
    assert_eq!(search_size["boardCount"], 2);
    assert_eq!(search_size["timelineCount"], 2);
    let meta_json: serde_json::Value = serde_json::from_str(
        &gpu_candidate_input_meta_json_from_i32s(&words).expect("candidate input metadata JSON"),
    )
    .expect("candidate input metadata JSON parses");
    assert_eq!(meta_json["sourceMeta"], json["sourceMeta"]);
    assert_eq!(meta_json["targetMeta"], json["targetMeta"]);
    assert_eq!(words[0], 2);
    assert_eq!(words[1], 128);
    assert_eq!(words[2], 2);
    assert_eq!(words[3], (GPU_SOURCE_STRIDE * 2) as i32);
    assert_eq!(words[4], (GPU_TARGET_STRIDE * 128) as i32);
    assert_eq!(words[5], (GPU_BOARD_STRIDE * 2) as i32);
    assert_eq!(words[6], (GPU_MUTATION_BOARD_STRIDE * 2) as i32);
    assert_eq!(
        words.len(),
        GPU_CANDIDATE_INPUT_HEADER_I32S
            + GPU_SOURCE_STRIDE * 2
            + GPU_TARGET_STRIDE * 128
            + GPU_BOARD_STRIDE * 2
            + GPU_MUTATION_BOARD_STRIDE * 2
    );
    assert_eq!(
        &words
            [GPU_CANDIDATE_INPUT_HEADER_I32S..GPU_CANDIDATE_INPUT_HEADER_I32S + GPU_SOURCE_STRIDE],
        &[6, 1, -1, 4, 0, 0, -1, 1, 2, 1]
    );
    assert_eq!(
        json["sources"].as_array().map(Vec::len),
        Some(GPU_SOURCE_STRIDE * 2)
    );
    assert_eq!(
        json["targets"].as_array().map(Vec::len),
        Some(GPU_TARGET_STRIDE * 128)
    );
    assert_eq!(
        json["boards"].as_array().map(Vec::len),
        Some(GPU_BOARD_STRIDE * 2)
    );
    assert_eq!(
        json["mutationBoards"].as_array().map(Vec::len),
        Some(GPU_MUTATION_BOARD_STRIDE * 2)
    );
    assert_eq!(inputs.target_meta[0].timeline_id, -1);
    assert_eq!(inputs.target_meta[0].time, 4);
    assert_eq!(
        &inputs.sources[..GPU_SOURCE_STRIDE],
        &[6, 1, -1, 4, 0, 0, -1, 1, 2, 1]
    );
    assert_eq!(&inputs.boards[..9], &[-1, -1, 4, 1, 9, 2, 3, 2, 4]);
    let second_source = GPU_SOURCE_STRIDE;
    assert_eq!(
        &inputs.sources[second_source..second_source + GPU_SOURCE_STRIDE],
        &[1, 0, 2, 1, 7, 7, 2, 0, 1, 1]
    );

    let error = gpu_candidate_inputs_from_snapshot_json("{\"turn\":\"white\",\"timelines\":null}")
        .expect_err("invalid snapshot should fail");
    assert!(error.contains("Snapshot timelines must be an array"));
}

#[test]
fn gpu_candidate_inputs_from_compact_snapshot_accepts_flat_squares() {
    let mut squares = vec![0; 64];
    squares[0] = 6;
    squares[63] = 1;
    let snapshot = serde_json::json!({
        "format": "chronofish-gpu-v1",
        "turn": "white",
        "nextTimelineId": 1,
        "nextBlackTimelineId": -1,
        "timelines": [{
            "id": 0,
            "row": 0,
            "owner": "white",
            "boards": [{
                "time": 0,
                "sideToMove": "white",
                "castling": 0,
                "enPassant": null,
                "squares": squares
            }]
        }]
    })
    .to_string();

    let inputs = gpu_candidate_inputs_from_gpu_snapshot_json(&snapshot)
        .expect("compact snapshot candidate inputs");
    assert_eq!(inputs.board_count, 1);
    assert_eq!(inputs.source_count, 2);
    assert_eq!(inputs.target_count, 64);
}

#[test]
fn frontier_tuning_derivation_matches_browser_gpu_device_math() {
    assert_eq!(gpu_frontier_positive_limit(Some(512), 256), 512);
    assert_eq!(gpu_frontier_positive_limit(Some(0), 256), 256);
    assert_eq!(gpu_frontier_positive_limit(None, 256), 256);
    assert_eq!(gpu_frontier_workgroup_size(1024), 256);
    assert_eq!(gpu_frontier_workgroup_size(256), 256);
    assert_eq!(gpu_frontier_workgroup_size(255), 128);
    assert_eq!(gpu_frontier_workgroup_size(127), 64);
    assert_eq!(gpu_frontier_workgroup_size(63), 32);
    assert_eq!(gpu_frontier_clamp_usize(4, 8, 512), 8);
    assert_eq!(gpu_frontier_clamp_usize(64, 8, 512), 64);
    assert_eq!(gpu_frontier_clamp_usize(1024, 8, 512), 512);
    assert_eq!(gpu_frontier_floor_power_of_two(0), 1);
    assert_eq!(gpu_frontier_floor_power_of_two(1), 1);
    assert_eq!(gpu_frontier_floor_power_of_two(300), 256);
    assert_eq!(gpu_frontier_next_power_of_two(0), 1);
    assert_eq!(gpu_frontier_next_power_of_two(1), 1);
    assert_eq!(gpu_frontier_next_power_of_two(300), 512);
    assert_eq!(frontier_expand_workgroups(0, 128), 0);
    assert_eq!(frontier_expand_workgroups(129, 128), 2);
    assert_eq!(frontier_selection_workgroups(256, 64), 4);
    assert_eq!(frontier_selection_workgroups(7, 0), 7);
    assert_eq!(frontier_materialize_workgroups(512, 128), 4);
    assert_eq!(frontier_materialize_workgroups(9, 0), 9);
    assert_eq!(frontier_minimax_workgroups(0), 0);
    assert_eq!(frontier_minimax_workgroups(65), 2);
    assert_eq!(frontier_policy_workgroups(0), 0);
    assert_eq!(frontier_policy_workgroups(65), 2);

    assert_eq!(frontier_state_bytes(1), 2_488);
    assert_eq!(
        derive_frontier_tuning(FrontierTuningLimits::default(), 1_024, 1, 0),
        FrontierTuning {
            max_boards: 1,
            frontier_width: 512,
            candidate_capacity: 4_096,
            neural_batch_size: 512,
            candidate_workgroup_size: 256,
            mutation_tile_size: 128,
            dispatch_candidate_limit: 4_096,
        }
    );
    assert_eq!(
        derive_frontier_tuning(
            FrontierTuningLimits {
                max_storage_buffer_binding_size: Some(1024 * 1024),
                max_buffer_size: Some(1024 * 1024),
                max_compute_invocations_per_workgroup: Some(64),
            },
            200,
            3,
            5,
        ),
        FrontierTuning {
            max_boards: 8,
            frontier_width: 112,
            candidate_capacity: 800,
            neural_batch_size: 8,
            candidate_workgroup_size: 64,
            mutation_tile_size: 64,
            dispatch_candidate_limit: 800,
        }
    );
}

#[test]
fn frontier_selection_plan_matches_browser_sort_shortlist_math() {
    let tuning = derive_frontier_tuning(FrontierTuningLimits::default(), 1_024, 1, 0);
    assert_eq!(
        frontier_selection_plan(&tuning, None),
        FrontierSelectionPlan {
            candidate_capacity: 4_096,
            selection_capacity: 2_048,
        }
    );
    assert_eq!(
        frontier_selection_plan(&tuning, Some(300)),
        FrontierSelectionPlan {
            candidate_capacity: 4_096,
            selection_capacity: 256,
        }
    );
    assert_eq!(
        frontier_selection_plan(
            &FrontierTuning {
                candidate_capacity: 1_500,
                frontier_width: 80,
                ..tuning
            },
            None,
        ),
        FrontierSelectionPlan {
            candidate_capacity: 1_024,
            selection_capacity: 256,
        }
    );
}

#[test]
fn frontier_cycle_policy_matches_web_worker_orchestration() {
    assert_eq!(frontier_max_cycles(1, 1), 3);
    assert_eq!(frontier_max_cycles(2, 1), 6);
    assert_eq!(frontier_max_cycles(4, 6), 32);
    assert_eq!(frontier_max_cycles(100, 10), 64);
    assert_eq!(frontier_max_cycles(0, 0), 2);

    assert_eq!(frontier_per_parent_limit(1), 2);
    assert_eq!(frontier_per_parent_limit(8), 2);
    assert_eq!(frontier_per_parent_limit(112), 14);
    assert_eq!(frontier_per_parent_limit(512), 16);

    assert_eq!(frontier_next_active_state_limit(512, 1, 2), 2);
    assert_eq!(frontier_next_active_state_limit(512, 32, 16), 512);
    assert_eq!(frontier_next_active_state_limit(128, 128, 16), 128);
    assert_eq!(frontier_next_active_state_limit(128, 32, 0), 32);

    assert_eq!(
        frontier_orchestration_plan(2, 1, 112),
        FrontierOrchestrationPlan {
            max_cycles: 6,
            per_parent_limit: 14,
            state_limits: vec![1, 14, 112, 112, 112, 112, 112],
        }
    );

    assert_eq!(frontier_cycle_state_count(512, 0), 1);
    assert_eq!(frontier_cycle_state_count(512, 32), 32);
    assert_eq!(frontier_cycle_state_count(128, 512), 128);

    assert_eq!(frontier_neural_cache_hit_rate(2.0, 3.0), 0.4);
    assert_eq!(frontier_neural_cache_hit_rate(1.0, 2.0), 0.333);
    assert_eq!(frontier_neural_cache_hit_rate(0.0, 0.0), 0.0);

    assert_eq!(frontier_expansion_source_scan_limit(64, 256), 256);
    assert_eq!(frontier_expansion_source_scan_limit(512, 128), 512);
    assert_eq!(frontier_expansion_source_scan_limit(0, 0), 1);
    assert_eq!(frontier_expansion_source_scan_count(256, 1024, 512), 256);
    assert_eq!(frontier_expansion_source_scan_count(256, 1024, 900), 124);
    assert_eq!(frontier_expansion_source_scan_count(256, 1024, 1200), 0);

    assert_eq!(frontier_minimax_bounded_depth(6, 4), 4);
    assert_eq!(frontier_minimax_bounded_depth(2, 4), 2);

    assert_eq!(gpu_full_search_reported_depth(1), 1);
    assert_eq!(gpu_full_search_reported_depth(2), 2);
    assert_eq!(gpu_full_search_reported_depth(5), 2);
    assert!(gpu_completed_reply_should_search(false, 100.0, 101.0));
    assert!(!gpu_completed_reply_should_search(true, 100.0, 101.0));
    assert!(!gpu_completed_reply_should_search(false, 101.0, 101.0));
    assert!(!gpu_frontier_cycle_should_stop(0, 2, 2, 101.0, 100.0));
    assert!(!gpu_frontier_cycle_should_stop(1, 1, 2, 101.0, 100.0));
    assert!(!gpu_frontier_cycle_should_stop(1, 2, 2, 99.0, 100.0));
    assert!(gpu_frontier_cycle_should_stop(1, 2, 2, 100.0, 100.0));
    let readback: serde_json::Value = serde_json::from_str(
        &gpu_frontier_readback_summary_json(r#"{"counters":[0,7,1,123,11,5]}"#)
            .expect("frontier readback summary"),
    )
    .expect("frontier readback summary parses");
    assert_eq!(readback["nodes"], 123);
    assert_eq!(readback["selectedCount"], 7);
    assert_eq!(readback["candidateOverflow"], true);
    assert_eq!(readback["tacticalCandidates"], 11);
    assert_eq!(readback["selectedTacticalCandidates"], 5);
    let sparse: serde_json::Value =
        serde_json::from_str(&gpu_frontier_readback_summary_json(r#"{"counters":[9]}"#).unwrap())
            .unwrap();
    assert_eq!(sparse["nodes"], 0);
    assert_eq!(sparse["candidateOverflow"], false);

    assert_eq!(gpu_diagnostic_rate(12.0, 100.0), 0.12);
    assert_eq!(gpu_diagnostic_rate(1.0, 3.0), 0.333);
    assert_eq!(gpu_diagnostic_rate(1.0, 0.0), 0.0);
    assert_eq!(gpu_effective_branching_factor(7.0, 3.0), 2.33);
    assert_eq!(gpu_effective_branching_factor(7.0, 0.0), 7.0);
    assert_eq!(gpu_reported_latency_ms(12.4), 12.0);
    assert_eq!(gpu_reported_latency_ms(-5.0), 0.0);
    assert_eq!(gpu_nodes_per_second(1000.0, 250.0), 4000.0);
    assert_eq!(gpu_nodes_per_second(1000.0, 0.0), 1000.0);
    assert_eq!(gpu_accumulated_search_nodes(10.0, 3.0, 0.0), 13.0);
    assert_eq!(gpu_accumulated_search_nodes(10.0, 3.0, 7.0), 20.0);
}

#[test]
fn gpu_candidate_record_move_decoder_matches_web_snapshot_contract() {
    let mut records = vec![0; GPU_CANDIDATE_STRIDE * 2];
    let second = GPU_CANDIDATE_STRIDE;
    records[second + 11] = -2;
    records[second + 12] = 4;
    records[second + 13] = 3;
    records[second + 14] = 6;
    records[second + 15] = 5;
    records[second + 16] = 7;
    records[second + 17] = 1;
    records[second + 18] = 2;

    assert_eq!(
        gpu_candidate_move_from_record(&records, 1),
        GpuCandidateMove {
            from: GpuCandidatePosition {
                timeline_id: -2,
                time: 4,
                x: 3,
                y: 6,
            },
            to: GpuCandidatePosition {
                timeline_id: 5,
                time: 7,
                x: 1,
                y: 2,
            },
        }
    );
    assert_eq!(
        gpu_candidate_move_from_record(&records[..second + 13], 1),
        GpuCandidateMove {
            from: GpuCandidatePosition {
                timeline_id: -2,
                time: 4,
                x: 0,
                y: 0,
            },
            to: GpuCandidatePosition {
                timeline_id: 0,
                time: 0,
                x: 0,
                y: 0,
            },
        }
    );
}

#[test]
fn gpu_branch_child_helpers_match_web_snapshot_contract() {
    let source = GpuCandidatePosition {
        timeline_id: 3,
        time: 4,
        x: 1,
        y: 2,
    };

    assert!(gpu_child_is_source_advance(
        GpuChildBoardRef {
            timeline_id: 3,
            time: 5,
        },
        source,
    ));
    assert!(!gpu_child_is_source_advance(
        GpuChildBoardRef {
            timeline_id: 3,
            time: 6,
        },
        source,
    ));
    assert!(!gpu_child_is_source_advance(
        GpuChildBoardRef {
            timeline_id: 4,
            time: 5,
        },
        source,
    ));

    assert_eq!(gpu_next_branch_row(&[-1, 0, 1, 2], 0, "white"), Ok(3));
    assert_eq!(gpu_next_branch_row(&[-2, -1, 0, 1], 0, "black"), Ok(-3));
    assert!(gpu_next_branch_row(&[], 0, "neutral").is_err());
}

#[test]
fn gpu_board_record_encoder_matches_web_snapshot_contract() {
    let record = gpu_board_record_from_snapshot(&GpuBoardRecordInput {
        timeline_id: -2,
        timeline_row: 7,
        time: 11,
        side_to_move: 1,
        castling: 9,
        en_passant: Some(GpuEnPassantRecord {
            x: 3,
            y: 4,
            captured_x: 3,
            captured_y: 5,
        }),
        squares: vec![0, 11, 6 | (1 << 8)],
    });

    assert_eq!(record.len(), GPU_BOARD_STRIDE);
    assert_eq!(&record[..9], &[-2, 7, 11, 1, 9, 3, 4, 3, 5]);
    assert_eq!(record[9], 0);
    assert_eq!(record[10], 11);
    assert_eq!(record[11], 6 | (1 << 8));
    assert!(record[12..].iter().all(|value| *value == 0));

    let empty_record = gpu_board_record_from_snapshot(&GpuBoardRecordInput {
        timeline_id: 1,
        timeline_row: -1,
        time: 0,
        side_to_move: 0,
        castling: 0,
        en_passant: None,
        squares: vec![],
    });
    assert_eq!(&empty_record[..9], &[1, -1, 0, 0, 0, -1, -1, -1, -1]);
    assert!(empty_record[9..].iter().all(|value| *value == 0));
}

#[test]
fn gpu_mutation_board_record_encoder_matches_web_snapshot_contract() {
    let record = gpu_mutation_board_record_from_snapshot(&GpuMutationBoardRecordInput {
        timeline_index: 2,
        timeline_id: -3,
        time: 8,
        side_to_move: 1,
        castling: 9,
        en_passant: Some(GpuEnPassantRecord {
            x: 4,
            y: 5,
            captured_x: 4,
            captured_y: 6,
        }),
        latest: true,
        origin_kind: 3,
        squares: vec![6, 0, 11 | (1 << 8)],
    });

    assert_eq!(record.len(), GPU_MUTATION_BOARD_STRIDE);
    assert_eq!(&record[..12], &[2, -3, 8, 1, 9, 4, 5, 4, 6, 1, 3, 0]);
    assert_eq!(record[12], 6);
    assert_eq!(record[13], 0);
    assert_eq!(record[14], 11 | (1 << 8));
    assert!(record[15..].iter().all(|value| *value == 0));

    let snapshot = gpu_mutation_board_record_to_snapshot(&record);
    assert_eq!(snapshot.timeline_index, 2);
    assert_eq!(snapshot.timeline_id, -3);
    assert_eq!(snapshot.time, 8);
    assert_eq!(snapshot.side_to_move, "black");
    assert_eq!(snapshot.castling, 9);
    assert_eq!(
        snapshot.en_passant,
        Some(GpuEnPassantRecord {
            x: 4,
            y: 5,
            captured_x: 4,
            captured_y: 6,
        })
    );
    assert!(snapshot.latest);
    assert_eq!(snapshot.origin_kind, 3);
    assert_eq!(snapshot.squares.len(), 64);
    assert_eq!(&snapshot.squares[..3], &[6, 0, 11 | (1 << 8)]);

    let empty_record = gpu_mutation_board_record_from_snapshot(&GpuMutationBoardRecordInput {
        timeline_index: 0,
        timeline_id: 1,
        time: 0,
        side_to_move: 0,
        castling: 0,
        en_passant: None,
        latest: false,
        origin_kind: 0,
        squares: vec![],
    });
    assert_eq!(
        &empty_record[..12],
        &[0, 1, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0]
    );
    assert!(empty_record[12..].iter().all(|value| *value == 0));
}

#[test]
fn gpu_mutation_board_record_decoder_matches_web_snapshot_contract() {
    let mut record = vec![0; GPU_MUTATION_BOARD_STRIDE];
    record[0] = 2;
    record[1] = -3;
    record[2] = 8;
    record[3] = 1;
    record[4] = 9;
    record[5] = 4;
    record[6] = 5;
    record[7] = 4;
    record[8] = 6;
    record[10] = 3;
    record[12] = 6;
    record[75] = 1 | (1 << 8);

    let snapshot = gpu_mutation_board_record_to_snapshot(&record);

    assert_eq!(
        snapshot,
        GpuMutationBoardSnapshot {
            timeline_index: 2,
            timeline_id: -3,
            time: 8,
            side_to_move: "black",
            castling: 9,
            en_passant: Some(GpuEnPassantRecord {
                x: 4,
                y: 5,
                captured_x: 4,
                captured_y: 6,
            }),
            latest: true,
            origin_kind: 3,
            squares: record[12..76].to_vec(),
        }
    );
    assert_eq!(
        gpu_mutation_board_record_to_snapshot(&record[..6]).en_passant,
        Some(GpuEnPassantRecord {
            x: 4,
            y: -1,
            captured_x: -1,
            captured_y: -1,
        })
    );
    assert_eq!(
        gpu_mutation_board_record_to_snapshot(&[]),
        GpuMutationBoardSnapshot {
            timeline_index: 0,
            timeline_id: 0,
            time: 0,
            side_to_move: "white",
            castling: 0,
            en_passant: None,
            latest: true,
            origin_kind: 0,
            squares: vec![],
        }
    );
}

#[test]
fn native_gpu_model_search_returns_native_depth2_when_frontier_expands() {
    let response = search(GpuSearchRequest {
        model_path: Some(committed_model_path()),
        depth: 2,
        min_depth: Some(2),
        nodes: 64,
        time_ms: 1_000,
        ..GpuSearchRequest::default()
    })
    .expect("run native GPU model search");

    #[cfg(feature = "neural-wgpu")]
    {
        assert_eq!(response.gpu_search, "native-wgpu-frontier-depth2-minimax");
        assert_eq!(response.backend, "wgpu-frontier");
        assert_eq!(
            response.native_frontier_round.as_deref(),
            Some("wgpu-frontier-search rounds=2 candidates=100 selected=8 states=8 plans=8 minimax_roots=2")
        );
        let value: serde_json::Value =
            serde_json::from_str(&response.result_json).expect("native depth2 JSON should parse");
        assert_eq!(value["status"], "ok");
        assert_eq!(value["depth"], 2);
        assert_eq!(value["nodes"], 100);
    }
    #[cfg(not(feature = "neural-wgpu"))]
    {
        assert_eq!(response.gpu_search, "cpu-orchestrated-compact-value-model");
        assert_eq!(response.backend, "cpu-search-with-gpu-model");
    }
}

fn committed_model_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/gpu-v1/value-model.cfnn")
        .to_string_lossy()
        .to_string()
}
