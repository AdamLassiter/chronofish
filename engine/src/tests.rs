#[cfg(test)]
mod tests {
    use super::*;

    fn empty_board_with_kings() -> [[Option<Piece>; 8]; 8] {
        let mut board = [[None; 8]; 8];
        board[0][4] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::King,
        });
        board[7][4] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::King,
        });
        board
    }

    fn snapshot(time: i32, side_to_move: Color, board: [[Option<Piece>; 8]; 8]) -> BoardSnapshot {
        BoardSnapshot {
            time,
            side_to_move,
            board,
            castling: CastlingRights::new(),
            en_passant: None,
            origin: Origin::None,
        }
    }

    #[test]
    fn starts_with_white_to_move() {
        let game = Game::new();
        assert_eq!(game.turn.as_str(), "white");
        assert_eq!(game.timelines.len(), 1);
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
    fn submitting_a_committed_royal_capture_ends_the_game() {
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
        assert_eq!(game.last_message, "Black wins by checkmate.");
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

        let mut replay = Game::new();
        for movement in result.moves {
            assert_eq!(replay.apply_move(movement.from, movement.to), 1);
        }
        assert_eq!(replay.submit_turn(), 1);
    }

    #[test]
    fn ai_json_has_expected_shape() {
        let game = Game::new();
        let json = game.ai_turn_json(1, 1_000);

        assert!(json.contains("\"moves\""));
        assert!(json.contains("\"score\""));
        assert!(json.contains("\"status\":\"ok\""));
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
    fn submit_detects_historical_royal_capture() {
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
        assert_eq!(game.last_message, "White wins by checkmate.");
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

        assert!(exposed.royal_safety_balance(Color::White, &weights) < safe.royal_safety_balance(Color::White, &weights));
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

        assert!(center.board_control_for(Color::White, &weights) > edge.board_control_for(Color::White, &weights));
        assert!(center.piece_activity_for(Color::White, &weights) > edge.piece_activity_for(Color::White, &weights));
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

        assert!(healthy.pawn_structure_for(Color::White, &weights) > weak.pawn_structure_for(Color::White, &weights));
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
    fn evaluation_rewards_royal_shelter() {
        let weights = EvalWeights::default_tuned();
        let mut sheltered = Game::new();
        let mut sheltered_board = empty_board_with_kings();
        for x in 3..=5 {
            sheltered_board[1][x] = Some(Piece {
                color: Color::White,
                piece_type: PieceType::Pawn,
            });
        }
        sheltered.timelines[0].boards = vec![snapshot(0, Color::White, sheltered_board)];

        let mut exposed = Game::new();
        exposed.timelines[0].boards = vec![snapshot(0, Color::White, empty_board_with_kings())];

        assert!(sheltered.royal_shelter_for(Color::White, &weights) > exposed.royal_shelter_for(Color::White, &weights));
    }

    #[test]
    fn training_mutation_is_seeded() {
        let config = TrainerConfig {
            generations: 1,
            population: 4,
            depth: 1,
            nodes: 50,
            plies: 1,
            seed: 7,
            time_budget_secs: 30,
            out: None,
            score: None,
            score_default: false,
            train_cycle: false,
            compare_seeds: vec![101, 202],
            min_wins: 1,
            min_total_delta: 1,
            verify: "cargo test -q".to_string(),
            ai_src: "engine/src/ai/parameters.rs".to_string(),
        };

        assert_eq!(train_weights(&config).to_json(), train_weights(&config).to_json());
    }
}
