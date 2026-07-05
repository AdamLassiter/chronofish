use super::*;
use crate::{cpu::*, wasm_api::parse_game_snapshot};

#[test]
fn browser_snapshot_round_trip_preserves_castling_and_pawn_targets() {
    let mut game = Game::new();
    let mut board = [[None; 8]; 8];
    board[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    board[0][7] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Rook,
    });
    board[1][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Pawn,
    });
    board[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.timelines[0].boards = vec![snapshot(0, Color::White, board)];

    let parsed = parse_game_snapshot(&game.to_json()).expect("browser snapshot parses");

    assert!(parsed.can_move_to(
        Position {
            timeline_id: 0,
            time: 0,
            x: 4,
            y: 0,
        },
        Position {
            timeline_id: 0,
            time: 0,
            x: 6,
            y: 0,
        },
    ));
    assert!(parsed.can_move_to(
        Position {
            timeline_id: 0,
            time: 0,
            x: 0,
            y: 1,
        },
        Position {
            timeline_id: 0,
            time: 0,
            x: 0,
            y: 3,
        },
    ));
}

#[test]
fn present_line_keeps_turn_until_leftmost_active_board_advances() {
    let mut game = Game::new();
    let mut board_a = empty_board_with_kings();
    board_a[1][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Pawn,
    });
    let mut board_b = empty_board_with_kings();
    board_b[1][1] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Pawn,
    });
    game.timelines.push(Timeline {
        id: 1,
        row: 1,
        label: "White T1".to_string(),
        owner: TimelineOwner::White,
        boards: vec![snapshot(0, Color::White, board_b)],
    });
    game.next_timeline_id = 2;
    game.timelines[0].boards = vec![snapshot(0, Color::White, board_a)];

    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 0,
                x: 0,
                y: 1,
            },
            Position {
                timeline_id: 0,
                time: 0,
                x: 0,
                y: 2,
            },
        ),
        1
    );
    assert_eq!(game.turn, Color::White);

    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 1,
                time: 0,
                x: 1,
                y: 1,
            },
            Position {
                timeline_id: 1,
                time: 0,
                x: 1,
                y: 2,
            },
        ),
        1
    );
    assert_eq!(game.turn, Color::White);
    assert_eq!(game.submit_turn(), 1);
    assert_eq!(game.turn, Color::Black);
}

#[test]
fn royal_capture_threat_highlights_without_ending_game() {
    let mut game = Game::new();
    let mut board = [[None; 8]; 8];
    board[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    board[1][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Rook,
    });
    board[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Rook,
    });
    board[7][0] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.timelines[0].boards = vec![snapshot(0, Color::White, board)];

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
                x: 5,
                y: 1,
            },
        ),
        1
    );
    assert_eq!(game.submit_turn(), 1);
    assert_eq!(game.last_message, "Black to move.");
    assert!(game
        .to_json()
        .contains("\"checkedRoyals\":[{\"timelineId\":0,\"time\":1,\"x\":4,\"y\":0}]"));
}

#[test]
fn checkmate_allows_capturing_adjacent_attacker() {
    let mut game = Game::new();
    let mut board = [[None; 8]; 8];
    board[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    board[1][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Rook,
    });
    board[7][0] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.timelines[0].boards = vec![snapshot(0, Color::White, board)];

    assert!(game.is_in_check(Color::White));
    assert!(!game.is_checkmate(Color::White));
}

#[test]
fn checkmate_allows_capturing_parallel_timeline_attacker() {
    let mut game = Game::new();
    let mut royal_board = [[None; 8]; 8];
    royal_board[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    royal_board[7][0] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });

    let mut attacker_board = [[None; 8]; 8];
    attacker_board[0][3] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Rook,
    });
    attacker_board[0][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Rook,
    });
    attacker_board[7][0] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });

    game.timelines = vec![
        Timeline {
            id: 0,
            row: 0,
            label: "Sacred T0".to_string(),
            owner: TimelineOwner::Neutral,
            boards: vec![snapshot(0, Color::Black, royal_board)],
        },
        Timeline {
            id: 1,
            row: 1,
            label: "White T1".to_string(),
            owner: TimelineOwner::White,
            boards: vec![snapshot(0, Color::White, attacker_board)],
        },
    ];

    assert!(game.is_in_check(Color::White));
    assert!(!game.is_checkmate(Color::White));
}

#[test]
fn unicorn_moves_on_three_dimensions() {
    let mut game = Game::new();
    let mut source = empty_board_with_kings();
    source[1][1] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Unicorn,
    });
    let mut target = empty_board_with_kings();
    target[2][2] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Knight,
    });
    game.timelines[0].boards = vec![snapshot(0, Color::White, source)];
    game.timelines.push(Timeline {
        id: 1,
        row: 1,
        label: "White T1".to_string(),
        owner: TimelineOwner::White,
        boards: vec![snapshot(0, Color::White, target)],
    });
    game.next_timeline_id = 2;

    assert!(game.can_move_to(
        Position {
            timeline_id: 0,
            time: 0,
            x: 1,
            y: 1,
        },
        Position {
            timeline_id: 1,
            time: 0,
            x: 2,
            y: 2,
        },
    ));
}

#[test]
fn dragon_moves_on_four_dimensions() {
    let mut game = Game::new();
    let mut source = empty_board_with_kings();
    source[1][1] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Dragon,
    });
    let target = empty_board_with_kings();
    game.timelines[0].boards = vec![snapshot(0, Color::White, source)];
    game.timelines.push(Timeline {
        id: 1,
        row: 1,
        label: "White T1".to_string(),
        owner: TimelineOwner::White,
        boards: vec![
            snapshot(0, Color::White, target),
            snapshot(1, Color::Black, target),
            snapshot(2, Color::White, target),
        ],
    });
    game.next_timeline_id = 2;

    assert!(game.can_move_to(
        Position {
            timeline_id: 0,
            time: 0,
            x: 1,
            y: 1,
        },
        Position {
            timeline_id: 1,
            time: 2,
            x: 2,
            y: 2,
        },
    ));
}

#[test]
fn princess_does_not_move_triagonally() {
    let mut game = Game::new();
    let mut source = empty_board_with_kings();
    source[1][1] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Princess,
    });
    let target = empty_board_with_kings();
    game.timelines[0].boards = vec![snapshot(0, Color::White, source)];
    game.timelines.push(Timeline {
        id: 1,
        row: 1,
        label: "White T1".to_string(),
        owner: TimelineOwner::White,
        boards: vec![snapshot(0, Color::White, target)],
    });
    game.next_timeline_id = 2;

    assert!(!game.can_move_to(
        Position {
            timeline_id: 0,
            time: 0,
            x: 1,
            y: 1,
        },
        Position {
            timeline_id: 1,
            time: 0,
            x: 2,
            y: 2,
        },
    ));
}

#[test]
fn common_king_is_not_royal() {
    let mut game = Game::new();
    let mut board = [[None; 8]; 8];
    board[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::CommonKing,
    });
    board[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Rook,
    });
    board[7][0] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.timelines[0].boards = vec![snapshot(0, Color::White, board)];

    assert!(!game.is_in_check(Color::White));
}

#[test]
fn royal_queen_is_royal() {
    let mut game = Game::new();
    let mut board = [[None; 8]; 8];
    board[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::RoyalQueen,
    });
    board[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Rook,
    });
    board[7][0] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.timelines[0].boards = vec![snapshot(0, Color::White, board)];

    assert!(game.is_in_check(Color::White));
}

#[test]
fn brawn_can_capture_on_mixed_forward_diagonal() {
    let mut game = Game::new();
    let mut source = empty_board_with_kings();
    source[3][3] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Brawn,
    });
    let mut target = empty_board_with_kings();
    target[4][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Knight,
    });
    game.timelines[0].boards = vec![snapshot(0, Color::White, source)];
    game.timelines.push(Timeline {
        id: 1,
        row: 1,
        label: "White T1".to_string(),
        owner: TimelineOwner::White,
        boards: vec![snapshot(0, Color::White, target)],
    });
    game.next_timeline_id = 2;

    assert!(game.can_move_to(
        Position {
            timeline_id: 0,
            time: 0,
            x: 3,
            y: 3,
        },
        Position {
            timeline_id: 1,
            time: 0,
            x: 4,
            y: 4,
        },
    ));
}

#[test]
fn ai_returns_submit_valid_turn() {
    let game = Game::new();
    let result = game.best_ai_turn(1, 1_000, None);

    assert_eq!(result.status, "ok");
    assert!(!result.moves.is_empty());
    assert!(result.nodes <= 1_000);

    let mut replay = Game::new();
    for movement in result.moves {
        assert_eq!(replay.apply_move(movement.from, movement.to), 1);
    }
    assert_eq!(replay.submit_turn(), 1);
}

#[test]
fn ai_can_complete_turn_across_more_than_four_active_boards() {
    let mut game = Game::new();
    game.timelines.clear();

    for id in 0..5 {
        let mut board = empty_board_with_kings();
        board[1][id as usize] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Pawn,
        });
        game.timelines.push(Timeline {
            id,
            row: id,
            label: format!("Neutral T{id}"),
            owner: TimelineOwner::Neutral,
            boards: vec![snapshot(0, Color::White, board)],
        });
    }

    let result = game.best_ai_turn(1, 50_000, None);

    assert_eq!(result.status, "ok");
    assert!(result.moves.len() >= 5);

    let mut replay = game;
    for movement in result.moves {
        assert_eq!(replay.apply_move(movement.from, movement.to), 1);
    }
    assert_eq!(replay.submit_turn(), 1);
}

#[test]
#[ignore = "wall-clock performance check; run with --ignored --nocapture"]
fn ai_search_perf_stockfish_style_steps_do_not_regress() {
    let game = five_board_perf_position();
    let stages = [
        ("compact-hash + cheap-order", SearchOptions::baseline()),
        (
            "+ tt best move",
            SearchOptions {
                tt_best_move: true,
                ..SearchOptions::baseline()
            },
        ),
        (
            "+ killer moves",
            SearchOptions {
                tt_best_move: true,
                killer_moves: true,
                ..SearchOptions::baseline()
            },
        ),
        (
            "+ history",
            SearchOptions {
                tt_best_move: true,
                killer_moves: true,
                history_heuristic: true,
                ..SearchOptions::baseline()
            },
        ),
        (
            "+ direct quiescence",
            SearchOptions {
                tt_best_move: true,
                killer_moves: true,
                history_heuristic: true,
                direct_quiescence: true,
                ..SearchOptions::baseline()
            },
        ),
        (
            "+ late reductions",
            SearchOptions {
                tt_best_move: true,
                killer_moves: true,
                history_heuristic: true,
                direct_quiescence: true,
                late_move_reduction: true,
                ..SearchOptions::baseline()
            },
        ),
        (
            "+ aspiration windows",
            SearchOptions {
                tt_best_move: true,
                killer_moves: true,
                history_heuristic: true,
                direct_quiescence: true,
                late_move_reduction: true,
                aspiration_windows: true,
                ..SearchOptions::baseline()
            },
        ),
        (
            "+ capture sanity",
            SearchOptions {
                tt_best_move: true,
                killer_moves: true,
                history_heuristic: true,
                direct_quiescence: true,
                late_move_reduction: true,
                aspiration_windows: true,
                capture_sanity: true,
                ..SearchOptions::baseline()
            },
        ),
        ("+ turn-plan cache", SearchOptions::optimized()),
    ];

    let mut previous_effort = u128::MAX;
    for (label, options) in stages {
        let (result, sample) = game.best_ai_turn_with_options(2, 2_500, None, options, Some(label));
        let sample = sample.expect("perf label should request sample");
        let effort = sample.elapsed_micros + sample.nodes as u128 * 10;
        eprintln!(
            "{label}: elapsed={}us nodes={} tt={} plan_cache={} cutoffs={} reductions={} probes={}",
            sample.elapsed_micros,
            sample.nodes,
            sample.stats.tt_hits,
            sample.stats.turn_plan_cache_hits,
            sample.stats.beta_cutoffs,
            sample.stats.reduced_searches,
            sample.stats.expensive_order_probes
        );

        assert_eq!(result.status, "ok");
        assert!(
            effort <= previous_effort.saturating_mul(13) / 10,
            "{label} regressed effort too far: {effort} after {previous_effort}"
        );
        previous_effort = previous_effort.min(effort);
    }
}

#[test]
#[ignore = "wall-clock late-game performance check; run with --ignored --nocapture"]
fn late_history_search_cost_stays_bounded() {
    let early = five_board_perf_position();
    let late = with_repeated_history(&early, 12);
    let weights = EvalWeights::default_tuned();

    let full_started = SearchInstant::now();
    let mut full_score = 0;
    for _ in 0..3 {
        full_score = late.evaluate_heuristic(Color::White, &weights);
    }
    let full_elapsed = SearchInstant::now()
        .duration_since(full_started)
        .as_micros();

    let bounded_started = SearchInstant::now();
    let mut bounded_score = 0;
    for _ in 0..3 {
        bounded_score = late.evaluate_heuristic_for_nodes(Color::White, &weights, 2_000);
    }
    let bounded_elapsed = SearchInstant::now()
        .duration_since(bounded_started)
        .as_micros();

    let (_, early_sample) = early.best_ai_turn_with_options(
        3,
        2_000,
        None,
        SearchOptions::optimized(),
        Some("early-history"),
    );
    let (late_result, late_sample) = late.best_ai_turn_with_options(
        3,
        2_000,
        None,
        SearchOptions::optimized(),
        Some("late-history"),
    );
    let early_sample = early_sample.expect("perf sample");
    let late_sample = late_sample.expect("perf sample");

    eprintln!(
        "full_eval={}us bounded_eval={}us scores={full_score}/{bounded_score}; early={}us late={}us generated={} candidates={} legal_attempts={} attack_queries={} attack_hits={} eval_calls={} cache_hits={} eval_moves={} setup_probes={} attack_checks={} attack_caps={} eval_clones={}",
        full_elapsed,
        bounded_elapsed,
        early_sample.elapsed_micros,
        late_sample.elapsed_micros,
        late_sample.stats.generated_moves,
        late_sample.stats.candidate_destinations,
        late_sample.stats.legal_move_attempts,
        late_sample.stats.attack_queries,
        late_sample.stats.attack_cache_hits,
        late_sample.stats.evaluation_calls,
        late_sample.stats.evaluation_cache_hits,
        late_sample.stats.evaluated_turn_moves,
        late_sample.stats.evaluation_setup_probes,
        late_sample.stats.evaluation_attack_checks,
        late_sample.stats.evaluation_attack_caps,
        late_sample.stats.evaluation_clones,
    );

    assert!(
        bounded_elapsed.saturating_mul(4) <= full_elapsed.saturating_mul(3),
        "bounded evaluation was not at least 25% faster: {bounded_elapsed}us vs {full_elapsed}us"
    );
    assert!(
        late_sample.elapsed_micros <= early_sample.elapsed_micros.saturating_mul(3),
        "late search regressed beyond 3x: {}us vs {}us",
        late_sample.elapsed_micros,
        early_sample.elapsed_micros,
    );
    assert_eq!(late_result.status, "ok");
    let mut replay = late;
    for movement in late_result.moves {
        assert!(replay.make_search_move(movement).is_some());
    }
    assert!(replay.submit_turn_for_search());
}

#[test]
fn late_history_evaluation_restores_search_state() {
    let early = five_board_perf_position();
    let late = with_repeated_history(&early, 12);
    let initial_json = late.to_json();
    let initial_hash = late.position_hash;
    let weights = EvalWeights::default_tuned();

    for _ in 0..3 {
        let _ = late.evaluate_heuristic(Color::White, &weights);
        assert_eq!(late.position_hash, initial_hash);
        assert_eq!(late.position_hash, late.recompute_position_hash());
        assert_eq!(late.to_json(), initial_json);
    }
}

#[test]
fn compact_search_undo_matches_direct_search_application_on_perf_position() {
    let game = five_board_perf_position();
    let weights = EvalWeights::default_tuned();
    let moves = game.legal_single_moves(&weights);
    assert!(
        !moves.is_empty(),
        "perf fixture should expose candidate search moves"
    );

    for movement in moves {
        let mut direct = game.clone_for_search();
        let mut undoable = game.clone_for_search();

        assert!(direct.apply_move_for_search(movement.from, movement.to));
        let undo = undoable
            .make_search_move(movement)
            .expect("directly applicable move should be undoable");
        assert_eq!(undoable.position_hash, direct.position_hash);
        assert_eq!(
            undoable.position_hash,
            undoable.recompute_position_hash(),
            "incremental hash drift after applying {movement:?}"
        );
        assert_eq!(undoable.to_json(), direct.to_json());

        undoable.unmake_search_move(undo);
        assert_eq!(undoable.position_hash, game.position_hash);
        assert_eq!(undoable.position_hash, undoable.recompute_position_hash());
        assert_eq!(undoable.to_json(), game.to_json());
    }
}

fn five_board_perf_position() -> Game {
    let mut game = Game::new();
    game.timelines.clear();

    for id in 0..5 {
        let mut board = empty_board_with_kings();
        board[1][id as usize] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Pawn,
        });
        board[0][(id as usize + 1).min(7)] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Knight,
        });
        board[6][7 - id as usize] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::Pawn,
        });
        game.timelines.push(Timeline {
            id,
            row: id,
            label: format!("Neutral T{id}"),
            owner: TimelineOwner::Neutral,
            boards: vec![snapshot(0, Color::White, board)],
        });
    }

    game.position_hash = game.recompute_position_hash();
    game
}

fn with_repeated_history(game: &Game, latest_time: i32) -> Game {
    let mut game = game.clone_for_search();
    for timeline in &mut game.timelines {
        let board = timeline.boards[0].clone();
        for time in 1..=latest_time {
            let mut historical = board.clone();
            historical.time = time;
            historical.side_to_move = if time == latest_time || time % 2 == 0 {
                Color::White
            } else {
                Color::Black
            };
            if time < latest_time {
                for row in &mut historical.board {
                    for square in row {
                        if square.is_some_and(|piece| {
                            piece.color == Color::White && piece.piece_type == PieceType::Pawn
                        }) {
                            *square = None;
                        }
                    }
                }
                historical.board[2 + time as usize / 8][time as usize % 8] = Some(Piece {
                    color: Color::White,
                    piece_type: PieceType::Pawn,
                });
            }
            timeline.boards.push(historical);
        }
    }
    game.position_hash = game.recompute_position_hash();
    game
}

#[test]
fn ai_json_has_expected_shape() {
    let game = Game::new();
    let json = game.ai_turn_json(1, 1_000);

    assert!(json.contains("\"moves\""));
    assert!(json.contains("\"score\""));
    assert!(json.contains("\"status\":\"ok\""));
}
