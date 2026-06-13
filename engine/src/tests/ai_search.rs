use super::*;
use crate::{
    ai::effort::ai_effort_config,
    training::{train_weights, TrainerConfig, TrainingSearchStrategy},
};

#[test]
fn bounded_evaluation_is_deterministic_and_respects_fast_limits() {
    let game = Game::new();
    let weights = EvalWeights::default_tuned();
    let limits = EvaluationLimits::for_nodes(2_000);
    let mut first_stats = EvaluationStats::default();
    let mut second_stats = EvaluationStats::default();

    let first =
        game.evaluate_heuristic_with_limits(Color::White, &weights, limits, &mut first_stats);
    let second =
        game.evaluate_heuristic_with_limits(Color::White, &weights, limits, &mut second_stats);

    assert_eq!(first, second);
    assert_eq!(first_stats.turn_moves, second_stats.turn_moves);
    assert_eq!(first_stats.setup_probes, second_stats.setup_probes);
    assert!(first_stats.turn_moves <= 12);
    assert!(first_stats.setup_probes <= 2_000);
    assert!(first_stats.clones <= 2);
}

#[test]
fn full_evaluation_limits_preserve_direct_evaluation_score() {
    let game = Game::new();
    let weights = EvalWeights::default_tuned();
    let mut stats = EvaluationStats::default();

    assert_eq!(
        game.evaluate_heuristic(Color::White, &weights),
        game.evaluate_heuristic_with_limits(
            Color::White,
            &weights,
            EvaluationLimits::FULL,
            &mut stats,
        )
    );
}

#[test]
fn direct_candidate_generation_keeps_all_default_legal_moves() {
    let game = Game::new();

    for y in 0..8 {
        for x in 0..8 {
            let from = Position {
                timeline_id: 0,
                time: 0,
                x,
                y,
            };
            let Some(piece) = game.piece_at(from) else {
                continue;
            };
            let candidates = game.piece_candidate_destinations(from, piece);
            for target_y in 0..8 {
                for target_x in 0..8 {
                    let to = Position {
                        timeline_id: 0,
                        time: 0,
                        x: target_x,
                        y: target_y,
                    };
                    if game.legal_move_kind(from, to).is_some() {
                        assert!(
                            candidates.contains(&to),
                            "candidate generation rejected legal move from ({x}, {y}) to ({target_x}, {target_y})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn incremental_hash_matches_recompute_and_search_undo() {
    let mut game = Game::new();
    let initial_json = game.to_json();
    let initial_hash = game.position_hash;
    assert_eq!(initial_hash, game.recompute_position_hash());

    let movement = MoveStep {
        from: Position {
            timeline_id: 0,
            time: 0,
            x: 4,
            y: 1,
        },
        to: Position {
            timeline_id: 0,
            time: 0,
            x: 4,
            y: 3,
        },
    };
    let undo = game
        .make_search_move(movement)
        .expect("search move is legal");
    assert_eq!(game.position_hash, game.recompute_position_hash());

    game.unmake_search_move(undo);
    assert_eq!(game.position_hash, initial_hash);
    assert_eq!(game.position_hash, game.recompute_position_hash());
    assert_eq!(game.to_json(), initial_json);
}

#[test]
fn incremental_hash_tracks_branch_search_moves() {
    let mut game = Game::new();
    game.load_notation(
        "1. T0L0e2Pe4\n\
         2. T1L0g7pg6\n\
         3. T2L0d1Qg4\n\
         4. T3L0g8nf6\n\
         5. T4L0f1Bc4\n\
         6. T5L0d7pd5\n\
         7. T6L0c4Bd5xp\n\
         8. T7L0d8qd6\n\
         9. T8L0g4Qc8xb+\n\
         10. T9L0d6qd8",
    )
    .expect("notation should create a historical target");
    assert_eq!(game.position_hash, game.recompute_position_hash());
    let initial_json = game.to_json();
    let initial_hash = game.position_hash;
    let movement = MoveStep {
        from: Position {
            timeline_id: 0,
            time: 10,
            x: 2,
            y: 7,
        },
        to: Position {
            timeline_id: 0,
            time: 6,
            x: 4,
            y: 7,
        },
    };

    let undo = game
        .make_search_move(movement)
        .expect("branch move is legal");
    assert_eq!(game.position_hash, game.recompute_position_hash());

    game.unmake_search_move(undo);
    assert_eq!(game.position_hash, initial_hash);
    assert_eq!(game.position_hash, game.recompute_position_hash());
    assert_eq!(game.to_json(), initial_json);
}

#[test]
fn verbose_notation_writes_and_replays_turns() {
    let mut game = Game::new();
    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 0,
                x: 4,
                y: 1,
            },
            Position {
                timeline_id: 0,
                time: 0,
                x: 4,
                y: 3,
            },
        ),
        1
    );
    assert_eq!(game.staged_turn_notation(), "T0L0e2Pe4");
    assert_eq!(game.submit_turn(), 1);

    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 1,
                x: 4,
                y: 6,
            },
            Position {
                timeline_id: 0,
                time: 1,
                x: 4,
                y: 4,
            },
        ),
        1
    );
    assert_eq!(game.staged_turn_notation(), "T1L0e7pe5");

    let mut replay = Game::new();
    replay
        .load_notation(
            "1. T0L0e2Pe4 [White pawn e4]\n\
             2. T1L0e7pe5 [Black pawn e5]",
        )
        .expect("notation should replay");

    assert_eq!(game.submit_turn(), 1);
    assert_eq!(replay.to_json(), game.to_json());
}

#[test]
fn royal_threat_highlights_but_only_capture_ends_game() {
    let mut game = Game::new();
    game.load_notation(
        "1. T0L0e2Pe4\n\
         2. T1L0g7pg6\n\
         3. T2L0d1Qg4\n\
         4. T3L0g8nf6\n\
         5. T4L0f1Bc4\n\
         6. T5L0d7pd5\n\
         7. T6L0c4Bd5xp\n\
         8. T7L0d8qd6\n\
         9. T8L0g4Qc8xb+",
    )
    .expect("notation should replay to black in check");

    assert_eq!(game.turn, Color::Black);
    assert!(game.is_in_check(Color::Black));
    assert!(game
        .to_json()
        .contains("\"checkedRoyals\":[{\"timelineId\":0,\"time\":9,\"x\":4,\"y\":7}]"));
    assert!(game.can_move_to(
        Position {
            timeline_id: 0,
            time: 9,
            x: 3,
            y: 5,
        },
        Position {
            timeline_id: 0,
            time: 9,
            x: 3,
            y: 7,
        },
    ));
    let mut blocked = game.clone_for_search();
    assert_eq!(
        blocked.apply_move(
            Position {
                timeline_id: 0,
                time: 9,
                x: 3,
                y: 5,
            },
            Position {
                timeline_id: 0,
                time: 9,
                x: 3,
                y: 7,
            },
        ),
        1
    );
    assert_eq!(blocked.submit_turn(), 1);
    assert_ne!(blocked.last_message, "White wins by royal capture.");

    assert_ne!(game.last_message, "White wins by royal capture.");
}

#[test]
fn ai_uses_immediate_king_capture_to_escape_check() {
    let mut game = Game::new();
    let mut board = [[None; 8]; 8];
    board[0][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    board[6][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Rook,
    });
    board[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.turn = Color::Black;
    game.timelines[0].boards = vec![snapshot(0, Color::Black, board)];

    let expected = MoveStep {
        from: Position {
            timeline_id: 0,
            time: 0,
            x: 4,
            y: 7,
        },
        to: Position {
            timeline_id: 0,
            time: 0,
            x: 4,
            y: 6,
        },
    };

    assert_eq!(game.turn, Color::Black);
    assert!(game.is_in_check(Color::Black));
    assert!(game.can_move_to(expected.from, expected.to));

    let result = game.best_ai_turn(2, 80_000, None);
    assert_eq!(result.status, "ok");
    assert_eq!(result.moves.first().copied(), Some(expected));

    let mut replay = game.clone_for_search();
    for movement in result.moves {
        assert_eq!(replay.apply_move(movement.from, movement.to), 1);
    }
    assert_eq!(replay.submit_turn(), 1);
    assert_ne!(replay.last_message, "White wins by checkmate.");
}

#[test]
fn submit_detects_committed_branch_royal_capture() {
    let mut game = Game::new();
    game.load_notation(
        "1. T0L0e2Pe4\n\
         2. T1L0g7pg6\n\
         3. T2L0d1Qg4\n\
         4. T3L0g8nf6\n\
         5. T4L0f1Bc4\n\
         6. T5L0d7pd5\n\
         7. T6L0c4Bd5xp\n\
         8. T7L0d8qd6\n\
         9. T8L0g4Qc8xb+\n\
         10. T9L0d6qd8",
    )
    .expect("notation should replay before royal capture");

    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 10,
                x: 2,
                y: 7,
            },
            Position {
                timeline_id: 0,
                time: 6,
                x: 4,
                y: 7,
            },
        ),
        1
    );
    assert_eq!(game.submit_turn(), 1);
    assert_eq!(game.last_message, "White wins by royal capture.");
    assert_eq!(
        game.result,
        Some(GameResult {
            winner: Some(Color::White),
            reason: GameResultReason::RoyalCapture,
        })
    );
}

#[test]
fn royal_capture_submit_wins_even_with_other_present_boards_pending() {
    let mut game = Game::new();
    let mut main_board = [[None; 8]; 8];
    main_board[0][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    main_board[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Rook,
    });
    main_board[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.timelines[0].boards = vec![snapshot(0, Color::White, main_board)];

    let mut pending_board = [[None; 8]; 8];
    pending_board[0][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    pending_board[7][7] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.timelines.push(Timeline {
        id: 1,
        row: 1,
        label: "L1".to_string(),
        owner: TimelineOwner::White,
        boards: vec![snapshot(0, Color::White, pending_board)],
    });

    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 0,
                x: 4,
                y: 0,
            },
            Position {
                timeline_id: 0,
                time: 0,
                x: 4,
                y: 7,
            },
        ),
        1
    );
    assert!(game.has_pending_present_board(Color::White));
    assert_eq!(game.submit_turn(), 1);
    assert_eq!(game.last_message, "White wins by royal capture.");
}

#[test]
fn evaluator_sees_one_move_temporal_royal_capture_setup() {
    let mut game = Game::new();
    game.load_notation(
        "1. T0L0e2Pe4\n\
         2. T1L0g7pg6\n\
         3. T2L0d1Qg4\n\
         4. T3L0g8nf6\n\
         5. T4L0f1Bc4\n\
         6. T5L0d7pd5\n\
         7. T6L0c4Bd5xp\n\
         8. T7L0d8qd6",
    )
    .expect("notation should replay before recurring queen tactic");

    let weights = EvalWeights::default_tuned();
    let setup_pressure = game.royal_capture_setup_pressure_for(Color::White, &weights);
    assert!(
        setup_pressure >= weights.royal_capture_setup,
        "expected Qc8 to register as a one-move temporal royal-capture setup, got {setup_pressure}"
    );
}

#[test]
fn ai_timed_json_has_expected_shape() {
    let game = Game::new();
    let json = game.ai_turn_timed_json(3, 10_000, 250);

    assert!(json.contains("\"moves\""));
    assert!(json.contains("\"nodes\""));
    assert!(json.contains("\"status\":\"ok\""));
}

#[test]
fn ai_timed_json_reports_principal_variation_beyond_selected_turn() {
    let mut game = Game::new();
    game.timelines[0].boards = vec![snapshot(0, Color::White, empty_board_with_kings())];
    let json = game.ai_turn_timed_json(2, 20_000, 500);
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid AI JSON");
    let principal_variation = value["principalVariation"]
        .as_array()
        .expect("principalVariation should be an array");

    assert_eq!(
        value["depth"].as_i64(),
        Some(2),
        "depth-2 search should complete: {json}"
    );
    assert!(
        principal_variation.len() >= 2,
        "depth-2 search should include the selected turn and the searched reply: {json}"
    );
}

#[test]
fn timed_ai_search_completes_default_minimum_depth_before_timing_out() {
    let game = Game::new();
    let json = game.ai_turn_timed_json(3, 20_000, 1);
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid AI JSON");

    assert_eq!(
        value["depth"].as_i64(),
        Some(Game::DEFAULT_MIN_AI_SEARCH_DEPTH as i64),
        "timed search should complete the default minimum depth before returning: {json}"
    );
}

#[test]
fn timed_ai_search_respects_lower_requested_depth() {
    let game = Game::new();
    let json = game.ai_turn_timed_json(1, 20_000, 1);
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid AI JSON");

    assert_eq!(
        value["depth"].as_i64(),
        Some(1),
        "requested depth 1 should remain a valid explicit cap: {json}"
    );
}

#[test]
fn effort_config_parses_configurable_min_depth() {
    let effort = ai_effort_config("fast").expect("fast effort config should parse");

    assert_eq!(effort.min_depth, Game::DEFAULT_MIN_AI_SEARCH_DEPTH);
    assert!(effort.min_depth <= effort.depth);
}

#[test]
fn ai_prefers_immediate_high_value_capture() {
    let mut game = Game::new();
    let mut board = empty_board_with_kings();
    board[0][3] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Queen,
    });
    board[3][3] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Queen,
    });
    game.timelines[0].boards = vec![snapshot(0, Color::White, board)];

    let result = game.best_ai_turn(1, 2_000, None);

    assert_eq!(result.status, "ok");
    assert!(result.moves.iter().any(|movement| {
        movement.to.timeline_id == 0
            && movement.to.time == 0
            && movement.to.x == 3
            && movement.to.y == 3
    }));
}

#[test]
fn submit_ignores_stale_historical_royal_capture() {
    let mut game = Game::new();
    let mut past = empty_board_with_kings();
    past[4][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    let middle = empty_board_with_kings();
    let mut latest = empty_board_with_kings();
    latest[4][3] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Bishop,
    });
    latest[7][7] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });

    game.turn = Color::Black;
    game.timelines[0].boards = vec![
        snapshot(0, Color::White, past),
        snapshot(1, Color::Black, middle),
        snapshot(2, Color::White, latest),
    ];
    game.staged_turn.push(game.checkpoint());

    assert_eq!(game.submit_turn(), 1);
    assert_eq!(game.last_message, "White to move.");
}

#[test]
fn evaluation_rewards_temporal_royal_capture_trajectory() {
    let mut game = Game::new();
    let mut past = empty_board_with_kings();
    past[4][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    let middle = empty_board_with_kings();
    let mut latest = empty_board_with_kings();
    latest[4][3] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Bishop,
    });
    latest[7][7] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.timelines[0].boards = vec![
        snapshot(0, Color::White, past),
        snapshot(1, Color::Black, middle),
        snapshot(2, Color::White, latest),
    ];

    assert!(game.royal_capture_pressure_for(Color::White, &EvalWeights::default_tuned()) > 0);
}

fn temporal_trade_game(attacker_type: PieceType) -> (Game, MoveStep) {
    let mut game = Game::new();
    game.turn = Color::White;
    let mut source = empty_board_with_kings();
    source[3][3] = Some(Piece {
        color: Color::White,
        piece_type: attacker_type,
    });
    let mut target = empty_board_with_kings();
    target[3][3] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Pawn,
    });
    game.timelines = vec![
        Timeline {
            id: 0,
            row: 0,
            label: "L0".to_string(),
            owner: TimelineOwner::Neutral,
            boards: vec![snapshot(0, Color::White, source)],
        },
        Timeline {
            id: 1,
            row: 1,
            label: "L1".to_string(),
            owner: TimelineOwner::White,
            boards: vec![snapshot(0, Color::White, target)],
        },
    ];
    let movement = MoveStep {
        from: Position {
            timeline_id: 0,
            time: 0,
            x: 3,
            y: 3,
        },
        to: Position {
            timeline_id: 1,
            time: 0,
            x: 3,
            y: 3,
        },
    };
    assert!(
        game.legal_move_kind(movement.from, movement.to).is_some(),
        "temporal trade fixture must be legal"
    );
    (game, movement)
}

#[test]
fn evaluation_charges_temporal_source_material_abandonment() {
    let (game, movement) = temporal_trade_game(PieceType::Queen);
    let weights = EvalWeights::default_tuned();
    let summary = game.turn_feature_summary(Color::White, &weights);
    let piece = game
        .piece_at(movement.from)
        .expect("fixture should have attacker");

    assert!(summary.source_abandonment > 0);
    assert!(
        summary.source_abandonment
            >= game.source_material_abandonment_cost(movement.from, piece, &weights)
    );
}

#[test]
fn source_abandonment_scales_with_temporal_mover_value() {
    let (queen_game, queen_capture) = temporal_trade_game(PieceType::Queen);
    let (rook_game, rook_capture) = temporal_trade_game(PieceType::Rook);
    let weights = EvalWeights::default_tuned();
    let queen = queen_game
        .piece_at(queen_capture.from)
        .expect("fixture should have queen");
    let rook = rook_game
        .piece_at(rook_capture.from)
        .expect("fixture should have rook");

    assert!(
        queen_game.source_material_abandonment_cost(queen_capture.from, queen, &weights)
            > rook_game.source_material_abandonment_cost(rook_capture.from, rook, &weights)
    );
}

#[test]
fn temporal_capture_ordering_accounts_for_attacker_value() {
    let (queen_game, queen_capture) = temporal_trade_game(PieceType::Queen);
    let (rook_game, rook_capture) = temporal_trade_game(PieceType::Rook);
    let weights = EvalWeights::default_tuned();

    assert!(
        queen_game.cheap_move_order_score(&queen_capture, &weights)
            < rook_game.cheap_move_order_score(&rook_capture, &weights)
    );
}

#[test]
fn evaluation_penalizes_exposed_royal_piece() {
    let mut exposed = Game::new();
    let mut board = empty_board_with_kings();
    board[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    board[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Rook,
    });
    board[7][7] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    exposed.timelines[0].boards = vec![snapshot(0, Color::White, board)];

    let safe = Game::new();
    let weights = EvalWeights::default_tuned();

    assert!(
        exposed.royal_safety_balance(Color::White, &weights)
            < safe.royal_safety_balance(Color::White, &weights)
    );
}

#[test]
fn evaluation_rewards_material_forks() {
    let mut game = Game::new();
    let mut board = empty_board_with_kings();
    board[3][3] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Knight,
    });
    board[4][5] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Queen,
    });
    board[2][5] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Rook,
    });
    game.timelines[0].boards = vec![snapshot(0, Color::White, board)];

    assert!(game.fork_pressure_for(Color::White, &EvalWeights::default_tuned()) > 0);
}

#[test]
fn training_json_round_trips_tactical_weights() {
    let weights = EvalWeights::default_tuned();
    let json = weights.to_json();
    let parsed = EvalWeights::from_json(&json).expect("weights should round-trip");

    assert_eq!(parsed.royal_capture_threat, weights.royal_capture_threat);
    assert_eq!(parsed.own_royal_exposure, weights.own_royal_exposure);
    assert_eq!(parsed.fork_pressure, weights.fork_pressure);
    assert_eq!(parsed.board_control, weights.board_control);
    assert_eq!(parsed.timeline_economy, weights.timeline_economy);
    assert_eq!(parsed.royal_shelter, weights.royal_shelter);
    assert_eq!(parsed.space_advantage, weights.space_advantage);
    assert_eq!(parsed.mandatory_move_burden, weights.mandatory_move_burden);
    assert_eq!(
        parsed.turn_completion_safety,
        weights.turn_completion_safety
    );
    assert_eq!(parsed.present_zugzwang, weights.present_zugzwang);
    assert_eq!(parsed.weakest_royal_safety, weights.weakest_royal_safety);
    assert_eq!(parsed.branch_payload, weights.branch_payload);
    assert_eq!(parsed.temporal_pin, weights.temporal_pin);
    assert_eq!(parsed.mate_net_depth_1_2, weights.mate_net_depth_1_2);
    assert_eq!(
        parsed.board_importance_weight,
        weights.board_importance_weight
    );
}

#[test]
fn evaluation_rewards_board_control_and_activity() {
    let weights = EvalWeights::default_tuned();
    let mut center = Game::new();
    let mut center_board = empty_board_with_kings();
    center_board[3][3] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Bishop,
    });
    center.timelines[0].boards = vec![snapshot(0, Color::White, center_board)];

    let mut edge = Game::new();
    let mut edge_board = empty_board_with_kings();
    edge_board[0][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Bishop,
    });
    edge.timelines[0].boards = vec![snapshot(0, Color::White, edge_board)];

    assert!(
        center.board_control_for(Color::White, &weights)
            > edge.board_control_for(Color::White, &weights)
    );
    assert!(
        center.piece_activity_for(Color::White, &weights)
            > edge.piece_activity_for(Color::White, &weights)
    );
}

#[test]
fn evaluation_rewards_healthier_pawn_structure() {
    let weights = EvalWeights::default_tuned();
    let mut healthy = Game::new();
    let mut healthy_board = empty_board_with_kings();
    healthy_board[4][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Pawn,
    });
    healthy_board[3][3] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Pawn,
    });
    healthy.timelines[0].boards = vec![snapshot(0, Color::White, healthy_board)];

    let mut weak = Game::new();
    let mut weak_board = empty_board_with_kings();
    weak_board[1][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Pawn,
    });
    weak_board[2][0] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Pawn,
    });
    weak.timelines[0].boards = vec![snapshot(0, Color::White, weak_board)];

    assert!(
        healthy.pawn_structure_for(Color::White, &weights)
            > weak.pawn_structure_for(Color::White, &weights)
    );
}

#[test]
fn evaluation_penalizes_inactive_owned_timelines() {
    let weights = EvalWeights::default_tuned();
    let mut inactive = Game::new();
    inactive.timelines.push(Timeline {
        id: 1,
        row: 1,
        label: "White T1".to_string(),
        owner: TimelineOwner::White,
        boards: vec![snapshot(0, Color::White, empty_board_with_kings())],
    });
    inactive.timelines.push(Timeline {
        id: 2,
        row: 2,
        label: "White T2".to_string(),
        owner: TimelineOwner::White,
        boards: vec![snapshot(0, Color::White, empty_board_with_kings())],
    });

    let mut active = inactive.clone();
    active.timelines.push(Timeline {
        id: -1,
        row: -1,
        label: "Black T1".to_string(),
        owner: TimelineOwner::Black,
        boards: vec![snapshot(0, Color::Black, empty_board_with_kings())],
    });

    assert!(
        active.timeline_economy_for(Color::White, &weights)
            > inactive.timeline_economy_for(Color::White, &weights)
    );
}

#[test]
fn evaluator_prunes_inactive_timelines() {
    let weights = EvalWeights::default_tuned();
    let board = empty_board_with_kings();
    let mut active_only = Game::new();
    active_only.timelines = vec![
        Timeline {
            id: 0,
            row: 0,
            label: "Sacred T0".to_string(),
            owner: TimelineOwner::Neutral,
            boards: vec![snapshot(0, Color::White, board)],
        },
        Timeline {
            id: -1,
            row: -1,
            label: "Black T-1".to_string(),
            owner: TimelineOwner::Black,
            boards: vec![snapshot(0, Color::White, board)],
        },
    ];

    let mut with_inactive = active_only.clone();
    let mut inactive_board = board;
    inactive_board[3][3] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Queen,
    });
    with_inactive.timelines.push(Timeline {
        id: -2,
        row: -2,
        label: "Black T-2".to_string(),
        owner: TimelineOwner::Black,
        boards: vec![snapshot(0, Color::White, inactive_board)],
    });

    assert!(!with_inactive.is_active_timeline(-2));
    assert_eq!(
        active_only.evaluate_heuristic(Color::White, &weights),
        with_inactive.evaluate_heuristic(Color::White, &weights)
    );
    assert!(with_inactive
        .pruned_for_evaluation()
        .timelines
        .iter()
        .all(|timeline| timeline.id != -2));
}

#[test]
fn evaluation_rewards_royal_shelter() {
    let weights = EvalWeights::default_tuned();
    let mut sheltered = Game::new();
    let mut sheltered_board = empty_board_with_kings();
    for square in &mut sheltered_board[1][3..=5] {
        *square = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Pawn,
        });
    }
    sheltered.timelines[0].boards = vec![snapshot(0, Color::White, sheltered_board)];

    let mut exposed = Game::new();
    exposed.timelines[0].boards = vec![snapshot(0, Color::White, empty_board_with_kings())];

    assert!(
        sheltered.royal_shelter_for(Color::White, &weights)
            > exposed.royal_shelter_for(Color::White, &weights)
    );
}

#[test]
fn training_mutation_is_seeded() {
    let config = TrainerConfig {
        effort: "expert".to_string(),
        generations: 1,
        population: 4,
        training_time_ms: 10,
        nodes: 5,
        seed: 7,
        max_seconds: Some(1),
        out: None,
        score: None,
        score_default: false,
        train_cycle: false,
        compare_seeds: vec![101, 202],
        min_wins: 1,
        min_total_delta: 1,
        verify: "cargo test -q".to_string(),
        ai_src: "engine/models/cpu-v1/parameters.json".to_string(),
        hall_of_fame: "engine/src/ai/hall_of_fame.jsonl".to_string(),
        min_pairs: 3,
        pair_batch: 1,
        max_pairs: 3,
        draw_window: 3,
        draw_rate_limit: 0.8,
        max_match_plies: 12,
        max_match_time_ms: 0,
        max_generations_without_candidate: 1,
        finalist_count: 2,
        search_strategy: TrainingSearchStrategy::AlphaBeta,
    };

    assert_eq!(
        train_weights(&config).to_json(),
        train_weights(&config).to_json()
    );
}
