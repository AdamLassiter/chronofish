use super::*;
use crate::{cpu::*, gpu_snapshot::*};

#[test]
fn neural_encoder_shape_mask_and_order_are_stable() {
    let game = Game::new();
    let encoded = game.encode_neural_position(Color::White);

    assert_eq!(encoded.values.len(), NEURAL_INPUT_SIZE);
    assert_eq!(encoded.board_count, 1);
    for square in 0..NEURAL_BOARD_SQUARES {
        assert_eq!(encoded.values[neural_feature_index(0, 30, square)], 1.0);
        assert_eq!(encoded.values[neural_feature_index(1, 30, square)], 0.0);
    }
    assert_eq!(
        game.neural_board_selection(),
        game.neural_board_selection(),
        "board ordering must be deterministic"
    );
}

#[test]
fn neural_evaluator_missing_model_falls_back_to_heuristic() {
    let game = Game::new();
    let weights = EvalWeights::default_tuned();
    let evaluator = NeuralEvaluator::missing_model(Some("missing/value-model.json".to_string()));

    assert!(!evaluator.is_available());
    assert_eq!(evaluator.model_path(), Some("missing/value-model.json"));
    assert_eq!(
        evaluator.evaluate(&game, Color::White, &weights),
        game.evaluate(Color::White, &weights)
    );
}

#[test]
fn neural_model_inference_is_deterministic_for_fixed_model() {
    let path = std::env::temp_dir().join(format!(
        "chronofish-value-model-{}.json",
        std::process::id()
    ));
    let model = NeuralLinearModel {
        bias: 42.0,
        scale: 1.0,
        feature_weights: vec![0.0; NEURAL_INPUT_SIZE],
        projection_size: 0,
        projection_seed: default_projection_seed(),
        hidden_layers: Vec::new(),
        hidden_weights: Vec::new(),
    };
    std::fs::write(&path, serde_json::to_string(&model).unwrap()).unwrap();

    let game = Game::new();
    let evaluator = NeuralEvaluator::from_path(path.to_string_lossy().to_string());
    assert!(evaluator.is_available());
    assert_eq!(evaluator.predict(&game, Color::White), Some(42));
    assert_eq!(evaluator.predict(&game, Color::White), Some(42));

    let _ = std::fs::remove_file(path);
}

#[test]
fn hybrid_evaluator_completes_legal_bot_move_with_missing_model() {
    let game = Game::new();
    let evaluator = ValueEvaluator::Hybrid(HybridEvaluator {
        heuristic_weight: 3,
        neural_weight: 1,
        neural: NeuralEvaluator::missing_model(None),
    });
    let (result, _) = game.best_ai_turn_with_value_evaluator(
        1,
        20,
        None,
        SearchOptions::optimized(),
        evaluator,
        None,
    );

    assert_eq!(result.status, "ok");
    assert!(!result.moves.is_empty());
}

#[test]
fn projected_neural_features_are_dense_and_deterministic() {
    let game = Game::new();
    let encoded = game.encode_neural_position(Color::White);
    let projected = project_neural_features(&encoded.values, 64, default_projection_seed());

    assert_eq!(projected.len(), 64);
    assert!(projected.iter().filter(|value| **value != 0.0).count() > 48);
    assert_eq!(
        projected,
        project_neural_features(&encoded.values, 64, default_projection_seed())
    );
}

#[test]
fn gpu_snapshot_binary_header_and_initial_board_are_stable() {
    let game = Game::new();
    let bytes = game.gpu_snapshot_bytes();
    let words = bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();

    assert_eq!(words[0], GPU_SNAPSHOT_MAGIC);
    assert_eq!(words[1], GPU_SNAPSHOT_VERSION);
    assert_eq!(words[2], 0);
    assert_eq!(words[3], 1);
    assert_eq!(words[4], 1);
    assert_eq!(words[9], GPU_TIMELINE_RECORD_I32S);
    assert_eq!(words[10], GPU_BOARD_RECORD_I32S);
    assert_eq!(words[11], GPU_BOARD_SQUARE_I32S);

    let timeline_offset = 16;
    assert_eq!(words[timeline_offset], 0);
    assert_eq!(words[timeline_offset + 1], 0);
    assert_eq!(words[timeline_offset + 2], 0);
    assert_eq!(words[timeline_offset + 5], 1);

    let board_offset = timeline_offset + GPU_TIMELINE_RECORD_I32S as usize;
    assert_eq!(words[board_offset], 0);
    assert_eq!(words[board_offset + 2], 0);
    assert_eq!(words[board_offset + 3], 0);
    assert_eq!(words[board_offset + 9], 1);

    let squares_offset = board_offset + GPU_BOARD_RECORD_I32S as usize;
    assert_eq!(words[squares_offset], piece_type_code(PieceType::Rook));
    assert_eq!(words[squares_offset + 4], piece_type_code(PieceType::King));
    assert_eq!(
        words[squares_offset + 7 * 8],
        piece_type_code(PieceType::Rook) | (1 << 8)
    );
    assert_eq!(
        words[squares_offset + 7 * 8 + 4],
        piece_type_code(PieceType::King) | (1 << 8)
    );
}

#[test]
fn branch_present_line_blocks_future_timeline_moves() {
    let mut game = Game::new();
    game.load_notation(
        "1. T0L0e2Pe4\n\
         2. T1L0g7pg6\n\
         3. T2L0d1Qg4\n\
         4. T3L0f7pf5\n\
         5. T4L0g4QT0L0e4>L1",
    )
    .unwrap();

    assert_eq!(game.turn, Color::Black);
    assert_eq!(game.present_time(), Some(1));
    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 5,
                x: 5,
                y: 4,
            },
            Position {
                timeline_id: 0,
                time: 5,
                x: 4,
                y: 3,
            },
        ),
        0,
        "future L0 move must wait while L1 is the present line"
    );
    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 1,
                time: 1,
                x: 4,
                y: 6,
            },
            Position {
                timeline_id: 1,
                time: 1,
                x: 4,
                y: 4,
            },
        ),
        1
    );
    assert_eq!(game.submit_turn(), 1);
    assert_eq!(game.turn, Color::White);
    assert_eq!(game.present_time(), Some(2));
}

#[test]
fn detects_threefold_repetition_on_same_timeline() {
    let board = empty_board_with_kings();
    let mut game = Game::new();
    game.timelines = vec![Timeline {
        id: 0,
        row: 0,
        label: "Sacred T0".to_string(),
        owner: TimelineOwner::Neutral,
        boards: vec![
            snapshot(0, Color::White, board),
            snapshot(1, Color::Black, board),
            snapshot(2, Color::White, board),
            snapshot(3, Color::White, board),
        ],
    }];

    assert!(game.has_threefold_repetition());
    assert_eq!(game.terminal_score(Color::White), Some(0));
    assert_eq!(game.terminal_score(Color::Black), Some(0));
}

#[test]
fn repetition_array_key_matches_vector_key() {
    let mut game = Game::new();
    game.load_notation(
        "1. T0L0g1Nf3\n\
         2. T1L0g8nf6\n\
         3. T2L0f3Ng1\n\
         4. T3L0f6ng8",
    )
    .expect("notation should produce reversible knight history");
    let timeline = game.timeline(0).expect("default timeline exists");

    for board in &timeline.boards {
        assert_eq!(
            Game::board_repetition_key(board),
            Game::board_repetition_key_array(board).to_vec()
        );
    }
}

#[test]
fn submit_marks_threefold_repetition_as_stalemate() {
    let mut game = Game::new();
    game.load_notation(
        "1. T0L0g1Nf3\n\
         2. T1L0g8nf6\n\
         3. T2L0f3Ng1\n\
         4. T3L0f6ng8\n\
         5. T4L0g1Nf3\n\
         6. T5L0g8nf6\n\
         7. T6L0f3Ng1",
    )
    .unwrap();

    assert!(!game.has_threefold_repetition());
    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 7,
                x: 5,
                y: 5,
            },
            Position {
                timeline_id: 0,
                time: 7,
                x: 6,
                y: 7,
            },
        ),
        1
    );
    assert_eq!(game.submit_turn(), 1);
    assert_eq!(game.last_message, "Stalemate by threefold repetition.");
    assert_eq!(
        game.result,
        Some(GameResult {
            winner: None,
            reason: GameResultReason::ThreefoldRepetition,
        })
    );
    assert_eq!(game.terminal_score(Color::White), Some(0));
    assert_eq!(game.terminal_score(Color::Black), Some(0));
}

#[test]
fn detects_classic_stalemate_on_present_board() {
    let mut board = [[None; 8]; 8];
    board[7][7] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    board[6][5] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    board[5][6] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Queen,
    });
    let mut game = Game::new();
    game.turn = Color::Black;
    game.timelines = vec![Timeline {
        id: 0,
        row: 0,
        label: "Sacred T0".to_string(),
        owner: TimelineOwner::Neutral,
        boards: vec![snapshot(0, Color::Black, board)],
    }];

    assert!(!game.is_in_check(Color::Black));
    assert!(game.is_classic_stalemate(Color::Black));
    assert_eq!(game.terminal_score(Color::White), Some(0));
    assert_eq!(game.terminal_score(Color::Black), Some(0));
}

#[test]
fn check_detection_uses_latest_board_fronts_only() {
    let mut board = empty_board_with_kings();
    board[7][4] = None;
    board[7][7] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    let mut checked_board = board;
    checked_board[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Rook,
    });
    let mut safe_board = board;
    safe_board[7][0] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Rook,
    });

    let mut game = Game::new();
    game.timelines = vec![Timeline {
        id: 0,
        row: 0,
        label: "Sacred T0".to_string(),
        owner: TimelineOwner::Neutral,
        boards: (0..40)
            .map(|time| {
                if time < 39 {
                    snapshot(time, Color::White, checked_board)
                } else {
                    snapshot(time, Color::White, safe_board)
                }
            })
            .collect(),
    }];

    assert!(!game.is_in_check(Color::White));
}

#[test]
fn submit_marks_classic_stalemate() {
    let mut board = [[None; 8]; 8];
    board[7][7] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    board[6][5] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    board[4][6] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Queen,
    });
    let mut game = Game::new();
    game.timelines = vec![Timeline {
        id: 0,
        row: 0,
        label: "Sacred T0".to_string(),
        owner: TimelineOwner::Neutral,
        boards: vec![snapshot(0, Color::White, board)],
    }];

    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 0,
                x: 6,
                y: 4,
            },
            Position {
                timeline_id: 0,
                time: 0,
                x: 6,
                y: 5,
            },
        ),
        1
    );
    assert_eq!(game.submit_turn(), 1);
    assert_eq!(game.turn, Color::Black);
    assert_eq!(game.last_message, "Stalemate.");
    assert_eq!(
        game.result,
        Some(GameResult {
            winner: None,
            reason: GameResultReason::Stalemate,
        })
    );
    assert_eq!(game.terminal_score(Color::White), Some(0));
    assert_eq!(game.terminal_score(Color::Black), Some(0));
}
