use super::*;
use crate::{gpu_snapshot::*, training::*};

fn trainer_test_config() -> TrainerConfig {
    TrainerConfig {
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
        hall_of_fame: default_hall_of_fame_path(),
        opponent_variants: 4,
        screening_opponent_variants: 2,
        rounds_per_variant: 1,
        hall_of_fame_entries: 4,
        league_contenders: 3,
        league_hall_of_fame_entries: 2,
        min_pairs: 3,
        pair_batch: 1,
        max_pairs: 8,
        draw_window: 4,
        draw_rate_limit: 0.75,
        max_match_plies: 10,
        max_match_time_ms: 0,
        max_generations_without_candidate: 1,
        finalist_count: 2,
        search_strategy: TrainingSearchStrategy::AlphaBeta,
    }
}

#[test]
fn trainer_loads_global_parameters_and_allows_cli_overrides() {
    let config = TrainerConfig::from_env(vec![
        "--effort".to_string(),
        "expert".to_string(),
        "--rounds-per-variant".to_string(),
        "3".to_string(),
        "--opponent-variants".to_string(),
        "5".to_string(),
    ]);

    assert_eq!(config.training_time_ms, 10_000);
    assert_eq!(config.nodes, 20_000);
    assert_eq!(config.population, 8);
    assert_eq!(config.min_pairs, 2);
    assert_eq!(config.max_pairs, 8);
    assert_eq!(config.draw_window, 4);
    assert_eq!(config.max_generations_without_candidate, 2);
    assert_eq!(config.rounds_per_variant, 3);
    assert_eq!(config.opponent_variants, 5);
    assert_eq!(config.screening_opponent_variants, 2);
    assert!(config
        .hall_of_fame
        .ends_with("models/cpu-v1/hall_of_fame.jsonl"));
}

#[test]
fn promotion_writes_weight_parameters_only() {
    let path = std::env::temp_dir().join(format!(
        "chronofish-parameters-{}-{}.json",
        std::process::id(),
        random_seed()
    ));
    std::fs::write(&path, r#"{"king":1}"#).expect("test parameters should be written");

    promote_weights(
        EvalWeights::default_tuned(),
        path.to_str().expect("UTF-8 path"),
    );

    let value: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path).expect("promoted parameters should be readable"),
    )
    .expect("promoted parameters should be JSON");
    assert!(value.get("training").is_none());
    assert_eq!(value["king"], EvalWeights::default_tuned().king);
    let _ = std::fs::remove_file(path);
}

#[test]
fn training_json_contains_the_global_training_config() {
    let training = load_training_parameters();
    assert_eq!(training.time_ms, 10_000);
    assert_eq!(training.nodes, 20_000);
    assert_eq!(training.candidates, Some(8));
    assert_eq!(training.opponent_variants, 8);
    assert_eq!(training.rounds_per_variant, 1);
    assert_eq!(training.min_pairs, 2);
    assert_eq!(training.max_pairs, 8);
}

#[test]
fn initialized_hall_of_fame_contains_valid_weights() {
    let entries = load_hall_of_fame(&default_hall_of_fame_path(), 4);
    assert!(!entries.is_empty());
    assert!(entries[0] == EvalWeights::default_tuned());
}

#[test]
fn full_match_scoring_stops_at_match_ply_cap() {
    let mut config = trainer_test_config();
    config.max_match_plies = 1;
    config.nodes = 50;
    config.training_time_ms = 20;
    let weights = EvalWeights::default_tuned();
    let report = play_match_until(
        Game::new(),
        weights,
        weights,
        Color::White,
        "candidate",
        "baseline",
        "ply cap smoke",
        &config,
        None,
    );

    assert!(!report.blunder);
}

#[test]
fn full_match_scoring_stops_at_match_time_cap() {
    let mut config = trainer_test_config();
    config.max_match_plies = 80;
    config.max_match_time_ms = 1;
    config.nodes = 200;
    config.training_time_ms = 20;
    let weights = EvalWeights::default_tuned();
    let report = play_match_until(
        Game::new(),
        weights,
        weights,
        Color::White,
        "candidate",
        "baseline",
        "time cap smoke",
        &config,
        None,
    );

    assert!(!report.blunder);
}

#[test]
fn alpha_beta_training_strategy_is_always_available() {
    assert_eq!(
        TrainingSearchStrategy::parse("alpha-beta"),
        Ok(TrainingSearchStrategy::AlphaBeta)
    );
}

#[test]
fn time_bounded_training_search_returns_only_applicable_turns() {
    let mut config = trainer_test_config();
    config.nodes = 200;
    config.training_time_ms = 20;
    let weights = EvalWeights::default_tuned();
    let seeds: Vec<u64> = (0..8).chain([3_471_131_662_115_554_319]).collect();

    for seed in seeds {
        let mut game = seeded_start_position(seed, &config, None);
        for turn in 1..=12 {
            let Some(plan) = training_turn_plan(&game, weights, &config, None) else {
                break;
            };
            game = game.apply_turn_plan_for_search(&plan).unwrap_or_else(|| {
                panic!("seed {seed} turn {turn} produced an inapplicable training plan")
            });
        }
    }
}

#[test]
fn bounded_evaluation_honors_attack_budget() {
    let game = multi_present_training_game(3);
    let weights = EvalWeights::default_tuned();
    let mut limits = EvaluationLimits::training_fast_late_game(10);
    limits.attack_checks = 1;
    let mut stats = EvaluationStats::default();

    let _ = game.evaluate_heuristic_with_limits(Color::White, &weights, limits, &mut stats);

    assert!(stats.attack_checks <= 1);
    assert!(
        stats.attack_caps > 0,
        "tiny attack budget should cap at least one evaluation attack probe"
    );
}

fn multi_present_training_game(count: i32) -> Game {
    let mut game = Game::new();
    game.timelines = (0..count)
        .map(|id| Timeline {
            id,
            row: id,
            label: format!("L{id}"),
            owner: TimelineOwner::Neutral,
            boards: vec![snapshot(0, Color::White, empty_board_with_kings())],
        })
        .collect();
    game
}

#[test]
fn training_search_profile_caps_late_game_branching() {
    let weights = EvalWeights::default_tuned();
    let normal = multi_present_training_game(1);
    let mut normal_context = SearchContext::new(weights, normal.turn, 20_000, None);
    let normal_profile = apply_training_search_profile(&normal, &mut normal_context, 0);
    assert_eq!(normal_profile.obligations, 1);
    assert_eq!(normal_context.root_plan_limit(), MAX_ROOT_TURN_PLANS);
    assert_eq!(normal_context.child_plan_limit(), MAX_CHILD_TURN_PLANS);

    let busy = multi_present_training_game(3);
    let mut busy_context = SearchContext::new(weights, busy.turn, 20_000, None);
    let busy_profile = apply_training_search_profile(&busy, &mut busy_context, 0);
    assert_eq!(busy_profile.obligations, 3);
    assert_eq!(busy_context.root_plan_limit(), 4);
    assert_eq!(busy_context.child_plan_limit(), 2);

    let late = multi_present_training_game(4);
    let mut late_context = SearchContext::new(weights, late.turn, 20_000, None);
    let late_profile = apply_training_search_profile(&late, &mut late_context, 24);
    assert_eq!(late_profile.obligations, 4);
    assert_eq!(late_context.root_plan_limit(), 2);
    assert_eq!(late_context.child_plan_limit(), 1);
    assert!(late_context.evaluation_limits.is_some());
}

#[test]
#[ignore = "reported overnight seed throughput smoke test; run in release mode with --ignored --nocapture"]
fn reported_seed_reaches_late_training_turns_with_bounded_search() {
    let mut config = trainer_test_config();
    config.nodes = 2_000;
    config.training_time_ms = 1_000;
    let weights = EvalWeights::default_tuned();
    let mut game = seeded_start_position(10_848_506_003_217_676_803, &config, None);

    let mut completed_turns = 0;
    for turn in 1..=30 {
        if game.terminal_score(game.turn).is_some() {
            break;
        }
        let started = SearchInstant::now();
        let Some(outcome) = training_turn_search(&game, weights, &config, None, turn - 1) else {
            panic!("training search should keep producing full turns through turn {turn}");
        };
        let elapsed = SearchInstant::now().duration_since(started).as_millis();
        assert!(
            elapsed < 5_000,
            "reported seed turn {turn} took {elapsed}ms with bounded training search"
        );
        game = game
            .apply_turn_plan_for_search(&outcome.plan)
            .expect("training plan should apply and submit");
        completed_turns = turn;
    }
    assert!(
        completed_turns >= 15 || game.terminal_score(game.turn).is_some(),
        "reported seed should either reach late training turns or finish naturally"
    );
}

#[test]
#[ignore = "release-mode training throughput smoke test; run with --ignored --nocapture"]
fn fast_training_search_reaches_turn_fifteen() {
    let mut game = Game::new();
    let mut config = trainer_test_config();
    config.training_time_ms = 1_000;
    config.nodes = 2_000;
    let weights = EvalWeights::default_tuned();

    for turn in 1..=15 {
        let started = SearchInstant::now();
        let plan = training_turn_plan(&game, weights, &config, None)
            .expect("training search should find a turn");
        let elapsed = SearchInstant::now().duration_since(started).as_millis();
        eprintln!(
            "training smoke turn {turn}: {elapsed}ms notation={}",
            turn_plan_notation(&game, &plan),
        );
        game = game
            .apply_turn_plan_for_search(&plan)
            .expect("training plan should apply and submit");
    }
}

#[test]
fn beam_training_strategy_returns_submit_valid_turn() {
    assert_eq!(
        TrainingSearchStrategy::parse("beam"),
        Ok(TrainingSearchStrategy::Beam)
    );

    let game = Game::new();
    let mut config = trainer_test_config();
    config.nodes = 200;
    config.search_strategy = TrainingSearchStrategy::Beam;

    let plan = training_turn_plan(&game, EvalWeights::default_tuned(), &config, None)
        .expect("beam search should find a turn");
    let mut replay = game;
    for movement in plan.moves {
        assert_eq!(replay.apply_move(movement.from, movement.to), 1);
    }
    assert_eq!(replay.submit_turn(), 1);
}

#[test]
fn statistical_decision_promotes_significant_winner() {
    let config = trainer_test_config();
    let mut stats = ComparisonStats::default();
    let mut deltas = Vec::new();
    for delta in [100, 120, 140, 160, 180, 200, 220, 240] {
        stats.record(delta);
        deltas.push(delta);
    }

    assert_eq!(
        statistical_decision(stats, &deltas, significance(&deltas), &config),
        StatisticalDecision::Promote
    );
}

#[test]
fn statistical_decision_rejects_significant_loser() {
    let config = trainer_test_config();
    let mut stats = ComparisonStats::default();
    let mut deltas = Vec::new();
    for delta in [-100, -120, -140, -160, -180, -200, -220, -240] {
        stats.record(delta);
        deltas.push(delta);
    }

    assert_eq!(
        statistical_decision(stats, &deltas, significance(&deltas), &config),
        StatisticalDecision::Reject
    );
}

#[test]
fn statistical_decision_marks_draw_stagnation_inconclusive() {
    let config = trainer_test_config();
    let mut stats = ComparisonStats::default();
    let mut deltas = Vec::new();
    for delta in [0, 0, 0, 0] {
        stats.record(delta);
        deltas.push(delta);
    }

    assert_eq!(
        statistical_decision(stats, &deltas, significance(&deltas), &config),
        StatisticalDecision::Inconclusive
    );
}

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
