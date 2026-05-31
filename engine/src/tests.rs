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
    fn quiet_development_orders_non_pawn_moves_before_pawn_pushes() {
        let game = Game::new();
        let weights = EvalWeights::default_tuned();
        let moves = game.legal_single_moves(&weights);
        let first = moves.first().expect("opening position has legal moves");

        assert!(matches!(
            game.piece_at(first.from).map(|piece| piece.piece_type),
            Some(PieceType::Knight)
        ));
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
            let (result, sample) =
                game.best_ai_turn_with_options(2, 2_500, None, options, Some(label));
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
    fn time_travel_source_board_still_counts_for_checkmate() {
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
        assert_ne!(blocked.last_message, "White wins by checkmate.");

        let result = game.best_ai_turn(2, 80_000, None);
        assert_eq!(result.status, "ok");
        let mut replay = game.clone_for_search();
        for movement in &result.moves {
            assert_eq!(replay.apply_move(movement.from, movement.to), 1);
        }
        assert_eq!(replay.submit_turn(), 1);
        assert_ne!(
            replay.last_message,
            "White wins by checkmate.",
            "{}",
            result.to_json()
        );

        assert_eq!(
            game.apply_move(
                Position {
                    timeline_id: 0,
                    time: 9,
                    x: 3,
                    y: 5,
                },
                Position {
                    timeline_id: 0,
                    time: 1,
                    x: 7,
                    y: 1,
                },
            ),
            1
        );
        assert_eq!(game.submit_turn(), 1);
        assert_eq!(game.last_message, "White wins by checkmate.");
        assert!(game.royal_capture_available(Color::White));
    }

    #[test]
    fn ai_uses_immediate_king_capture_to_escape_check() {
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
             10. T9L0d6qd8\n\
             11. T10L0c8QT6L0e8xk>L1\n\
             12. T7L1d8qe8xQ\n\
             13. T8L1c4Bd5xp\n\
             14. T9L1c8bg4xQ\n\
             15. T10L1g1Nf3\n\
             16. T11L0d8qT9L0c8xQ>L-1\n\
             17. T10L-1d5Bf7xp+\n\
             18. T11L-1e8kf7xB/T11L1g4bf3xN\n\
             19. T12L0d5BT6L0d8xq>L2\n\
             20. T7L2c8bg4xQ\n\
             21. T8L2d8BT8L1e8xq>L3\n\
             22. T9L3c8bg4xQ/T9L2d5pc4xB\n\
             23. T10L3e8BT8L2e8xk>L4/T12L1g2Pf3xb/T10L2b1NT10L3b3>L5\n\
             24. T13L0b8nT9L0c8xQ>L-2\n\
             25. T10L-2d5Bf7xp+",
        )
        .expect("notation should replay");

        let expected = MoveStep {
            from: Position {
                timeline_id: -2,
                time: 11,
                x: 4,
                y: 7,
            },
            to: Position {
                timeline_id: -2,
                time: 11,
                x: 5,
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
        assert_eq!(game.last_message, "White wins by checkmate.");
        assert!(game.to_json().contains(
            "\"checkedRoyals\":[{\"timelineId\":0,\"time\":6,\"x\":4,\"y\":7}]"
        ));
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
            effort: "expert".to_string(),
            generations: 1,
            population: 4,
            depth: 1,
            nodes: 5,
            plies: 0,
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
            ai_src: "engine/src/ai/parameters.json".to_string(),
            hall_of_fame: "engine/src/ai/hall_of_fame.jsonl".to_string(),
            min_pairs: 3,
            pair_batch: 1,
            max_pairs: 3,
            draw_window: 3,
            draw_rate_limit: 0.8,
            max_generations_without_candidate: 1,
        };

        assert_eq!(train_weights(&config).to_json(), train_weights(&config).to_json());
    }

    fn trainer_test_config() -> TrainerConfig {
        TrainerConfig {
            effort: "expert".to_string(),
            generations: 1,
            population: 4,
            depth: 1,
            nodes: 5,
            plies: 0,
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
            ai_src: "engine/src/ai/parameters.json".to_string(),
            hall_of_fame: "engine/src/ai/hall_of_fame.jsonl".to_string(),
            min_pairs: 3,
            pair_batch: 1,
            max_pairs: 8,
            draw_window: 4,
            draw_rate_limit: 0.75,
            max_generations_without_candidate: 1,
        }
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
}
