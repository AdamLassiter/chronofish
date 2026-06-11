use super::*;
use crate::wasm_api::parse_game_snapshot;

fn wasm_output(pointer: *const u8) -> String {
    let bytes =
        unsafe { std::slice::from_raw_parts(pointer, crate::wasm_api::chronofish_output_len()) };
    std::str::from_utf8(bytes)
        .expect("WASM output is UTF-8")
        .to_string()
}

#[test]
fn standard_move_advances_main_timeline() {
    let mut game = Game::new();
    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 0,
                x: 4,
                y: 1
            },
            Position {
                timeline_id: 0,
                time: 0,
                x: 4,
                y: 3
            },
        ),
        1
    );

    assert_eq!(game.latest_time(0), Some(1));
    assert_eq!(game.turn.as_str(), "white");
    assert_eq!(game.submit_turn(), 1);
    assert_eq!(game.turn.as_str(), "black");
    assert!(game.board(0, 1).expect("new board").board[1][4].is_none());
}

#[test]
fn browser_wasm_contract_round_trips_engine_behavior() {
    crate::wasm_api::chronofish_reset();

    let snapshot: serde_json::Value =
        serde_json::from_str(&wasm_output(crate::wasm_api::chronofish_snapshot_json()))
            .expect("snapshot JSON");
    assert_eq!(snapshot["turn"], "white");
    assert_eq!(snapshot["presentTime"], 0);
    assert_eq!(snapshot["timelines"][0]["active"], true);

    let selection: serde_json::Value = serde_json::from_str(&wasm_output(
        crate::wasm_api::chronofish_legal_selection_json(0, 0, 4, 1),
    ))
    .expect("selection JSON");
    assert_eq!(selection["source"]["piece"]["type"], "pawn");
    assert_eq!(selection["targets"].as_array().map(Vec::len), Some(2));

    assert_eq!(
        crate::wasm_api::chronofish_apply_move(0, 0, 4, 1, 0, 0, 4, 3),
        1
    );
    assert_eq!(
        wasm_output(crate::wasm_api::chronofish_staged_turn_notation()),
        "T0L0e2Pe4"
    );

    let evaluation: serde_json::Value =
        serde_json::from_str(&wasm_output(crate::wasm_api::chronofish_evaluation_json()))
            .expect("evaluation JSON");
    assert!(evaluation["score"].is_i64());
    assert_eq!(evaluation["source"], "engine heuristic");

    assert_eq!(crate::wasm_api::chronofish_submit_turn(), 1);
    let submitted: serde_json::Value =
        serde_json::from_str(&wasm_output(crate::wasm_api::chronofish_snapshot_json()))
            .expect("submitted snapshot JSON");
    assert_eq!(submitted["turn"], "black");
    assert_eq!(submitted["timelines"][0]["boards"][1]["time"], 1);
}

#[test]
fn active_timelines_are_nearest_to_zero() {
    let mut game = Game::new();
    let board = empty_board_with_kings();
    game.timelines = vec![
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
        Timeline {
            id: -2,
            row: -2,
            label: "Black T-2".to_string(),
            owner: TimelineOwner::Black,
            boards: vec![snapshot(0, Color::White, board)],
        },
    ];

    assert!(game.is_active_timeline(0));
    assert!(game.is_active_timeline(-1));
    assert!(!game.is_active_timeline(-2));
}

#[test]
fn browser_snapshot_exposes_rust_derived_timeline_state() {
    let mut game = Game::new();
    let board = empty_board_with_kings();
    game.timelines = vec![
        Timeline {
            id: 0,
            row: 0,
            label: "Sacred T0".to_string(),
            owner: TimelineOwner::Neutral,
            boards: vec![snapshot(3, Color::White, board)],
        },
        Timeline {
            id: -1,
            row: -1,
            label: "Black T-1".to_string(),
            owner: TimelineOwner::Black,
            boards: vec![snapshot(2, Color::White, board)],
        },
        Timeline {
            id: -2,
            row: -2,
            label: "Black T-2".to_string(),
            owner: TimelineOwner::Black,
            boards: vec![snapshot(1, Color::White, board)],
        },
    ];

    let json: serde_json::Value = serde_json::from_str(&game.to_json()).expect("valid snapshot");
    assert_eq!(json["presentTime"], 2);
    assert_eq!(json["timelines"][0]["active"], false);
    assert_eq!(json["timelines"][1]["active"], true);
    assert_eq!(json["timelines"][2]["active"], true);
}

#[test]
fn browser_evaluation_uses_engine_heuristics() {
    let game = Game::new();
    let evaluation: serde_json::Value =
        serde_json::from_str(&game.evaluation_json()).expect("valid evaluation");

    assert!(evaluation["score"].is_i64());
    assert_eq!(evaluation["source"], "engine heuristic");
    assert_eq!(
        evaluation["score"].as_i64(),
        Some(game.evaluate_heuristic(Color::White, &EvalWeights::active_tuned()) as i64)
    );
}

#[test]
fn browser_legal_selection_is_decided_by_the_engine() {
    let game = Game::new();
    let white_pawn: serde_json::Value =
        serde_json::from_str(&game.legal_selection_json(Position {
            timeline_id: 0,
            time: 0,
            x: 4,
            y: 1,
        }))
        .expect("valid legal selection");
    assert_eq!(white_pawn["source"]["piece"]["color"], "white");
    assert_eq!(white_pawn["source"]["piece"]["type"], "pawn");
    assert_eq!(white_pawn["targets"].as_array().map(Vec::len), Some(2));

    let black_pawn: serde_json::Value =
        serde_json::from_str(&game.legal_selection_json(Position {
            timeline_id: 0,
            time: 0,
            x: 4,
            y: 6,
        }))
        .expect("valid rejected selection");
    assert!(black_pawn["source"].is_null());
    assert_eq!(black_pawn["targets"].as_array().map(Vec::len), Some(0));
}

#[test]
fn terminal_result_round_trips_through_browser_snapshot() {
    let mut game = Game::new();
    game.result = Some(GameResult {
        winner: Some(Color::Black),
        reason: GameResultReason::RoyalCapture,
    });
    game.last_message = game.result.expect("result").message();

    let json = game.to_json();
    let restored = parse_game_snapshot(&json).expect("terminal snapshot parses");
    assert_eq!(restored.result, game.result);
    assert_eq!(restored.last_message, "Black wins by royal capture.");
    assert_eq!(
        restored.terminal_score(Color::White),
        Some(-CHECKMATE_SCORE)
    );
    assert_eq!(restored.terminal_score(Color::Black), Some(CHECKMATE_SCORE));

    let value: serde_json::Value = serde_json::from_str(&json).expect("valid snapshot JSON");
    assert_eq!(value["result"]["terminal"], true);
    assert_eq!(value["result"]["outcome"], "win");
    assert_eq!(value["result"]["winner"], "black");
    assert_eq!(value["result"]["reason"], "royal-capture");
}

#[test]
fn browser_snapshot_rejects_inconsistent_terminal_result() {
    let snapshot = Game::new().to_json().replace(
        "\"result\":null",
        "\"result\":{\"terminal\":true,\"outcome\":\"draw\",\"winner\":null,\"reason\":\"royal-capture\"}",
    );

    assert_eq!(
        parse_game_snapshot(&snapshot)
            .err()
            .expect("invalid result rejected"),
        "Snapshot result winner does not match its reason."
    );
}

#[test]
fn search_ignores_inactive_timeline_sources() {
    let mut game = Game::new();
    let mut board = empty_board_with_kings();
    board[0][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Rook,
    });
    game.timelines = vec![
        Timeline {
            id: 0,
            row: 0,
            label: "Sacred T0".to_string(),
            owner: TimelineOwner::Neutral,
            boards: vec![snapshot(0, Color::White, empty_board_with_kings())],
        },
        Timeline {
            id: -1,
            row: -1,
            label: "Black T-1".to_string(),
            owner: TimelineOwner::Black,
            boards: vec![snapshot(0, Color::White, empty_board_with_kings())],
        },
        Timeline {
            id: -2,
            row: -2,
            label: "Black T-2".to_string(),
            owner: TimelineOwner::Black,
            boards: vec![snapshot(0, Color::White, board)],
        },
    ];

    let weights = EvalWeights::default_tuned();
    let moves = game.legal_single_moves(&weights);

    assert!(game.can_move_to(
        Position {
            timeline_id: -2,
            time: 0,
            x: 0,
            y: 0,
        },
        Position {
            timeline_id: -2,
            time: 0,
            x: 0,
            y: 1,
        },
    ));
    assert!(moves.iter().all(|movement| movement.from.timeline_id != -2));
}

#[test]
fn search_does_not_create_inactive_timelines() {
    let mut game = Game::new();
    let mut latest = empty_board_with_kings();
    latest[0][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Rook,
    });
    game.next_timeline_id = 2;
    game.timelines = vec![
        Timeline {
            id: 0,
            row: 0,
            label: "Sacred T0".to_string(),
            owner: TimelineOwner::Neutral,
            boards: vec![
                snapshot(0, Color::White, empty_board_with_kings()),
                snapshot(1, Color::White, latest),
            ],
        },
        Timeline {
            id: 1,
            row: 1,
            label: "White T1".to_string(),
            owner: TimelineOwner::White,
            boards: vec![snapshot(1, Color::Black, empty_board_with_kings())],
        },
    ];

    let from = Position {
        timeline_id: 0,
        time: 1,
        x: 0,
        y: 0,
    };
    let to = Position {
        timeline_id: 0,
        time: 0,
        x: 0,
        y: 0,
    };

    assert!(game.can_move_to(from, to));
    assert!(!game.apply_move_for_search(from, to));
}

#[test]
fn search_allows_royal_pieces_to_create_inactive_timelines() {
    let mut game = Game::new();
    let mut latest = empty_board_with_kings();
    latest[0][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::RoyalQueen,
    });
    game.next_timeline_id = 2;
    game.timelines = vec![
        Timeline {
            id: 0,
            row: 0,
            label: "Sacred T0".to_string(),
            owner: TimelineOwner::Neutral,
            boards: vec![
                snapshot(0, Color::White, empty_board_with_kings()),
                snapshot(1, Color::White, latest),
            ],
        },
        Timeline {
            id: 1,
            row: 1,
            label: "White T1".to_string(),
            owner: TimelineOwner::White,
            boards: vec![snapshot(1, Color::Black, empty_board_with_kings())],
        },
    ];

    let from = Position {
        timeline_id: 0,
        time: 1,
        x: 0,
        y: 0,
    };
    let to = Position {
        timeline_id: 0,
        time: 0,
        x: 0,
        y: 0,
    };

    assert!(game.can_move_to(from, to));
    assert!(game.apply_move_for_search(from, to));
    assert!(game.timeline(2).is_some());
    assert!(!game.is_active_timeline(2));
}

#[test]
fn branch_move_advances_source_and_destination() {
    let mut game = Game::new();
    let empty = [[None; 8]; 8];
    let mut latest = empty;
    latest[0][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Rook,
    });
    latest[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    latest[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.timelines[0].boards = vec![
        BoardSnapshot {
            time: 0,
            side_to_move: Color::White,
            board: empty,
            castling: CastlingRights::new(),
            en_passant: None,
            origin: Origin::None,
        },
        BoardSnapshot {
            time: 1,
            side_to_move: Color::Black,
            board: empty,
            castling: CastlingRights::new(),
            en_passant: None,
            origin: Origin::None,
        },
        BoardSnapshot {
            time: 2,
            side_to_move: Color::White,
            board: latest,
            castling: CastlingRights::new(),
            en_passant: None,
            origin: Origin::None,
        },
    ];

    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 2,
                x: 0,
                y: 0,
            },
            Position {
                timeline_id: 0,
                time: 0,
                x: 0,
                y: 0,
            },
        ),
        1
    );

    assert_eq!(game.timelines.len(), 2);
    assert!(game.board(0, 3).is_some());
    assert!(game.board(1, 1).is_some());
}

#[test]
fn cross_board_move_to_playable_board_does_not_create_timeline() {
    let mut game = Game::new();
    let mut source = empty_board_with_kings();
    source[1][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Pawn,
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

    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 0,
                x: 4,
                y: 1,
            },
            Position {
                timeline_id: 1,
                time: 0,
                x: 4,
                y: 1,
            },
        ),
        1
    );

    assert_eq!(game.timelines.len(), 2);
    assert!(game.board(0, 1).is_some());
    assert!(game.board(1, 1).is_some());
}

#[test]
fn black_created_timelines_use_negative_ids() {
    let mut game = Game::new();
    let empty = [[None; 8]; 8];
    let mut latest = empty;
    latest[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    latest[7][0] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Rook,
    });
    latest[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.turn = Color::Black;
    game.timelines[0].boards = vec![
        snapshot(0, Color::Black, empty),
        snapshot(1, Color::White, empty),
        snapshot(2, Color::Black, latest),
    ];

    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 2,
                x: 0,
                y: 7,
            },
            Position {
                timeline_id: 0,
                time: 0,
                x: 0,
                y: 7,
            },
        ),
        1
    );

    assert!(game.timeline(-1).is_some());
    assert_eq!(game.next_black_timeline_id, -2);
}

#[test]
fn time_travel_only_targets_boards_where_mover_is_to_play() {
    let mut game = Game::new();
    let empty = [[None; 8]; 8];
    let mut latest = empty;
    latest[0][0] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Rook,
    });
    latest[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    latest[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.timelines[0].boards = vec![
        BoardSnapshot {
            time: 0,
            side_to_move: Color::White,
            board: empty,
            castling: CastlingRights::new(),
            en_passant: None,
            origin: Origin::None,
        },
        BoardSnapshot {
            time: 1,
            side_to_move: Color::Black,
            board: empty,
            castling: CastlingRights::new(),
            en_passant: None,
            origin: Origin::None,
        },
        BoardSnapshot {
            time: 2,
            side_to_move: Color::White,
            board: latest,
            castling: CastlingRights::new(),
            en_passant: None,
            origin: Origin::None,
        },
    ];

    assert!(!game.can_move_to(
        Position {
            timeline_id: 0,
            time: 2,
            x: 0,
            y: 0,
        },
        Position {
            timeline_id: 0,
            time: 1,
            x: 0,
            y: 0,
        },
    ));
}

#[test]
fn time_travel_distance_counts_same_color_boards() {
    let mut game = Game::new();
    let empty = [[None; 8]; 8];
    let mut latest = empty;
    latest[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    latest[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    game.timelines[0].boards = vec![
        snapshot(0, Color::White, empty),
        snapshot(1, Color::Black, empty),
        snapshot(2, Color::White, latest),
    ];

    assert!(game.can_move_to(
        Position {
            timeline_id: 0,
            time: 2,
            x: 4,
            y: 0,
        },
        Position {
            timeline_id: 0,
            time: 0,
            x: 3,
            y: 0,
        },
    ));
}

#[test]
fn en_passant_is_available_only_on_the_immediate_reply_board() {
    let mut game = Game::new();
    let mut board = empty_board_with_kings();
    board[4][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::Pawn,
    });
    board[6][3] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::Pawn,
    });
    game.turn = Color::Black;
    game.timelines[0].boards = vec![snapshot(0, Color::Black, board)];

    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 0,
                x: 3,
                y: 6,
            },
            Position {
                timeline_id: 0,
                time: 0,
                x: 3,
                y: 4,
            },
        ),
        1
    );
    assert_eq!(game.submit_turn(), 1);
    assert!(game.can_move_to(
        Position {
            timeline_id: 0,
            time: 1,
            x: 4,
            y: 4,
        },
        Position {
            timeline_id: 0,
            time: 1,
            x: 3,
            y: 5,
        },
    ));

    assert_eq!(
        game.apply_move(
            Position {
                timeline_id: 0,
                time: 1,
                x: 4,
                y: 4,
            },
            Position {
                timeline_id: 0,
                time: 1,
                x: 3,
                y: 5,
            },
        ),
        1
    );
    assert_eq!(game.submit_turn(), 1);
    let board_after = game.board(0, 2).expect("en passant result");
    assert!(board_after.board[4][3].is_none());
    assert_eq!(
        board_after.board[5][3],
        Some(Piece {
            color: Color::White,
            piece_type: PieceType::Pawn,
        })
    );
    assert!(board_after.en_passant.is_none());
}

#[test]
fn staged_move_can_be_undone() {
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
    assert_eq!(game.latest_time(0), Some(1));
    assert_eq!(game.undo_staged_move(), 1);
    assert_eq!(game.latest_time(0), Some(0));
    assert!(game.staged_turn.is_empty());
}

#[test]
fn castling_moves_the_rook_and_expires_rights() {
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
    board[7][4] = Some(Piece {
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
                y: 0,
            },
            Position {
                timeline_id: 0,
                time: 0,
                x: 6,
                y: 0,
            },
        ),
        1
    );

    let board_after = game.board(0, 1).expect("castled board");
    assert_eq!(
        board_after.board[0][6],
        Some(Piece {
            color: Color::White,
            piece_type: PieceType::King,
        })
    );
    assert_eq!(
        board_after.board[0][5],
        Some(Piece {
            color: Color::White,
            piece_type: PieceType::Rook,
        })
    );
    assert!(!board_after.castling.white_kingside);
    assert!(!board_after.castling.white_queenside);
}
