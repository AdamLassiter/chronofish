use super::*;
use crate::wasm_api::parse_game_snapshot;

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
#[ignore = "wall-clock performance check; run with `cargo test ai_search_perf -- --ignored --nocapture`"]
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
