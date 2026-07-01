use std::path::PathBuf;

use chronofish_engine::gpu::search::{
    bot_completed_search_depth,
    bot_next_search_depth,
    bot_search_depth_at_least_one,
    bot_worker_search_time_ms,
    derive_frontier_tuning,
    frontier_max_cycles,
    frontier_per_parent_limit,
    frontier_selection_plan,
    frontier_state_bytes,
    gpu_board_record_from_snapshot,
    gpu_candidate_board_records_from_snapshot,
    gpu_candidate_index_from_i32s,
    gpu_candidate_inputs_from_snapshot_json,
    gpu_candidate_inputs_from_timelines,
    gpu_candidate_inputs_i32s_from_snapshot_json,
    gpu_candidate_inputs_json_from_snapshot_json,
    gpu_candidate_move_from_record,
    gpu_child_is_source_advance,
    gpu_choice_agreement_json,
    gpu_frontier_active_timeline_distance,
    gpu_frontier_clamp_usize,
    gpu_frontier_floor_power_of_two,
    gpu_frontier_hash_words,
    gpu_frontier_next_power_of_two,
    gpu_frontier_origin_code,
    gpu_frontier_pending_board_count,
    gpu_frontier_positive_limit,
    gpu_frontier_present_time,
    gpu_frontier_root_i32s_from_snapshot_json,
    gpu_frontier_timeline_active,
    gpu_frontier_workgroup_size,
    gpu_latest_board_index,
    gpu_mutation_board_record_from_snapshot,
    gpu_mutation_board_record_to_snapshot,
    gpu_mutation_summary_from_i32s,
    gpu_next_branch_row,
    gpu_pick_candidate_records_from_i32s,
    gpu_ranked_candidate_indexes_from_i32s,
    gpu_reply_pressure_ranked_roots_from_i32s,
    gpu_scoring_summary_from_i32s,
    gpu_search_board_to_square_codes,
    gpu_search_color_code,
    gpu_search_color_from_code,
    gpu_search_opposite_color,
    gpu_search_owner_code,
    gpu_search_owner_from_code,
    gpu_search_piece_code,
    gpu_search_piece_from_code,
    gpu_search_piece_type_code,
    gpu_search_piece_type_from_code,
    gpu_search_select_candidate_json,
    gpu_search_square_codes_to_board,
    gpu_source_square_records_for_board,
    gpu_square_record_from_code,
    gpu_target_square_records_for_board,
    gpu_timeline_sort_order,
    gpu_turn_completion_key_json,
    gpu_turn_status_records_i32s_from_snapshot_json,
    search,
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
    FRONTIER_HEADER_PENDING_BOARDS,
    FRONTIER_HEADER_PRESENT_TIME,
    FRONTIER_HEADER_TURN,
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

    assert_eq!(bot_completed_search_depth(2.0, 2, false), 2);
    assert_eq!(bot_completed_search_depth(3.0, 3, false), 0);
    assert_eq!(bot_completed_search_depth(3.0, 3, true), 3);
    assert_eq!(bot_completed_search_depth(1.0, 1, false), 0);
    assert_eq!(bot_completed_search_depth(1.0, 1, true), 1);
    assert_eq!(bot_completed_search_depth(1.0, 2, true), 0);
    assert_eq!(bot_completed_search_depth(f64::NAN, 2, true), 0);
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
}

#[test]
fn gpu_mutation_summary_matches_web_worker_contract() {
    assert_eq!(gpu_mutation_summary_from_i32s(&[]), "none");
    assert_eq!(
        gpu_mutation_summary_from_i32s(&[3, 1, 3, 4, 1, 1]),
        "1:3,3:2,4:1"
    );
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

    let empty = gpu_choice_agreement_json(r#"{ "selected": [], "choices": [[]], "limits": [1] }"#)
        .expect("empty choice agreement response");
    let empty_value: serde_json::Value =
        serde_json::from_str(&empty).expect("empty choice agreement JSON");
    assert_eq!(empty_value["agreements"], serde_json::json!([0]));
}

#[test]
fn gpu_pick_candidate_records_matches_web_worker_contract() {
    let record_a = [1; GPU_CANDIDATE_STRIDE];
    let record_b = [2; GPU_CANDIDATE_STRIDE];
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

    let mut invalid = request;
    let index_offset = 2 + 3 * GPU_CANDIDATE_STRIDE;
    invalid[index_offset] = 3;
    let error = gpu_pick_candidate_records_from_i32s(&invalid).expect_err("out of range index");
    assert!(error.contains("out of range"));
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

    request[1..9].copy_from_slice(&[9, 9, 9, 9, 9, 9, 9, 9]);
    assert_eq!(
        gpu_candidate_index_from_i32s(&request).expect("missing candidate index"),
        -1
    );
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
