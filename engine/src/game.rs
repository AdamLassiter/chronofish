impl Game {
    fn new() -> Self {
        let mut board = [[None; 8]; 8];
        let back_rank = [
            PieceType::Rook,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Queen,
            PieceType::King,
            PieceType::Bishop,
            PieceType::Knight,
            PieceType::Rook,
        ];

        for x in 0..8 {
            board[0][x] = Some(Piece {
                color: Color::White,
                piece_type: back_rank[x],
            });
            board[1][x] = Some(Piece {
                color: Color::White,
                piece_type: PieceType::Pawn,
            });
            board[6][x] = Some(Piece {
                color: Color::Black,
                piece_type: PieceType::Pawn,
            });
            board[7][x] = Some(Piece {
                color: Color::Black,
                piece_type: back_rank[x],
            });
        }

        Self {
            turn: Color::White,
            timelines: vec![Timeline {
                id: 0,
                row: 0,
                label: "Sacred T0".to_string(),
                owner: TimelineOwner::Neutral,
                boards: vec![BoardSnapshot {
                    time: 0,
                    side_to_move: Color::White,
                    board,
                    castling: CastlingRights::new(),
                    en_passant: None,
                    origin: Origin::None,
                }],
            }],
            next_timeline_id: 1,
            next_black_timeline_id: -1,
            staged_turn: Vec::new(),
            last_message: "Select a white piece on a latest board.".to_string(),
        }
    }

    fn timeline(&self, timeline_id: i32) -> Option<&Timeline> {
        self.timelines
            .iter()
            .find(|timeline| timeline.id == timeline_id)
    }

    fn timeline_mut(&mut self, timeline_id: i32) -> Option<&mut Timeline> {
        self.timelines
            .iter_mut()
            .find(|timeline| timeline.id == timeline_id)
    }

    fn board(&self, timeline_id: i32, time: i32) -> Option<&BoardSnapshot> {
        self.timeline(timeline_id)?
            .boards
            .iter()
            .find(|board| board.time == time)
    }

    fn latest_time(&self, timeline_id: i32) -> Option<i32> {
        self.timeline(timeline_id)?
            .boards
            .iter()
            .map(|board| board.time)
            .max()
    }

    fn is_latest_board(&self, timeline_id: i32, time: i32) -> bool {
        self.latest_time(timeline_id) == Some(time)
    }

    fn piece_at(&self, position: Position) -> Option<Piece> {
        if !Self::in_bounds(position.x, position.y) {
            return None;
        }

        self.board(position.timeline_id, position.time)?.board[position.y as usize]
            [position.x as usize]
    }

    fn can_move_to(&self, from: Position, to: Position) -> bool {
        self.legal_move_kind(from, to).is_some()
    }

    fn legal_move_kind(&self, from: Position, to: Position) -> Option<(Piece, MoveKind)> {
        if !Self::in_bounds(from.x, from.y) || !Self::in_bounds(to.x, to.y) {
            return None;
        }

        let source_board = self.board(from.timeline_id, from.time)?;
        let target_board = self.board(to.timeline_id, to.time)?;
        let piece = self.piece_at(from)?;
        let same_board = from.timeline_id == to.timeline_id && from.time == to.time;

        if !self.is_latest_board(from.timeline_id, from.time)
            || source_board.side_to_move != self.turn
            || piece.color != self.turn
        {
            return None;
        }

        if !same_board && target_board.side_to_move != piece.color {
            return None;
        }

        if self
            .piece_at(to)
            .is_some_and(|target| target.color == piece.color)
        {
            return None;
        }

        self.move_kind_for(piece, from, to)
            .map(|kind| (piece, kind))
    }

    fn legal_targets_json(&self, from: Position) -> String {
        let mut targets = Vec::new();

        for timeline in &self.timelines {
            for board in &timeline.boards {
                for y in 0..8 {
                    for x in 0..8 {
                        let to = Position {
                            timeline_id: timeline.id,
                            time: board.time,
                            x,
                            y,
                        };

                        if self.can_move_to(from, to) {
                            targets.push(position_json(to));
                        }
                    }
                }
            }
        }

        format!("[{}]", targets.join(","))
    }

    fn apply_move(&mut self, from: Position, to: Position) -> i32 {
        let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
            self.last_message = "Illegal move.".to_string();
            return 0;
        };

        self.staged_turn.push(self.checkpoint());
        self.apply_move_unchecked(from, to, piece, move_kind);
        self.last_message = self.move_message(
            piece,
            from,
            to,
            matches!(
                move_kind,
                MoveKind::Standard | MoveKind::Castle { .. } | MoveKind::EnPassant { .. }
            ),
        );
        1
    }

    fn clone_for_search(&self) -> Self {
        self.clone()
    }

    fn checkpoint(&self) -> GameCheckpoint {
        GameCheckpoint {
            turn: self.turn,
            timelines: self.timelines.clone(),
            next_timeline_id: self.next_timeline_id,
            next_black_timeline_id: self.next_black_timeline_id,
            last_message: self.last_message.clone(),
        }
    }

    fn restore(&mut self, checkpoint: GameCheckpoint) {
        self.turn = checkpoint.turn;
        self.timelines = checkpoint.timelines;
        self.next_timeline_id = checkpoint.next_timeline_id;
        self.next_black_timeline_id = checkpoint.next_black_timeline_id;
        self.last_message = checkpoint.last_message;
    }

    fn undo_staged_move(&mut self) -> i32 {
        let Some(checkpoint) = self.staged_turn.pop() else {
            self.last_message = "No staged move to undo.".to_string();
            return 0;
        };

        self.restore(checkpoint);
        self.last_message = if self.staged_turn.is_empty() {
            format!("{} to move.", self.turn.capitalized())
        } else {
            "Undid staged move.".to_string()
        };
        1
    }

    fn submit_turn(&mut self) -> i32 {
        if self.staged_turn.is_empty() {
            self.last_message = "Make at least one move before submitting.".to_string();
            return 0;
        }

        let Some(present_side) = self.present_board().map(|board| board.side_to_move) else {
            self.last_message = "No active present board.".to_string();
            return 0;
        };

        if present_side == self.turn {
            self.last_message =
                "Make moves until the present line reaches the opponent's turn.".to_string();
            return 0;
        }

        if self.is_in_check(self.turn) {
            self.last_message = "Cannot submit while a royal piece is in check.".to_string();
            return 0;
        }

        self.turn = present_side;
        self.staged_turn.clear();

        let suffix = if self.is_checkmate(self.turn) {
            " Checkmate."
        } else if self.is_in_check(self.turn) {
            " Check."
        } else {
            ""
        };
        self.last_message = format!("{} to move.{}", self.turn.capitalized(), suffix);
        1
    }

    fn apply_move_unchecked(
        &mut self,
        from: Position,
        to: Position,
        piece: Piece,
        move_kind: MoveKind,
    ) {
        let source_board = self
            .board(from.timeline_id, from.time)
            .expect("legal move has source board")
            .clone();
        let target_board = self
            .board(to.timeline_id, to.time)
            .expect("legal move has target board")
            .clone();
        let source_row = self
            .timeline(from.timeline_id)
            .expect("legal move has source timeline")
            .row;
        let next_turn = self.turn.opposite();
        let is_branch = matches!(move_kind, MoveKind::Branch);

        if !is_branch {
            let mut next_board = source_board.board;
            next_board[from.y as usize][from.x as usize] = None;
            if let MoveKind::EnPassant {
                captured_x,
                captured_y,
            } = move_kind
            {
                next_board[captured_y as usize][captured_x as usize] = None;
            }
            next_board[to.y as usize][to.x as usize] = Some(promote_if_needed(piece, to.y));
            if let MoveKind::Castle {
                rook_from_x,
                rook_to_x,
            } = move_kind
            {
                let rook = next_board[from.y as usize][rook_from_x as usize];
                next_board[from.y as usize][rook_from_x as usize] = None;
                next_board[from.y as usize][rook_to_x as usize] = rook;
            }

            let mut castling = source_board.castling;
            update_castling_rights(&mut castling, piece, from, to, source_board.board);

            self.timeline_mut(from.timeline_id)
                .expect("source timeline exists")
                .boards
                .push(BoardSnapshot {
                    time: from.time + 1,
                    side_to_move: next_turn,
                    board: next_board,
                    castling,
                    en_passant: en_passant_after_move(piece, from, to, move_kind),
                    origin: Origin::Move {
                        from,
                        to,
                        move_type: move_kind.name(),
                    },
                });
        } else {
            let target_is_historical = !self.is_latest_board(to.timeline_id, to.time);
            let destination_timeline_id = if target_is_historical {
                self.place_timeline(piece.color, source_row)
            } else {
                to.timeline_id
            };

            let mut advanced_source = source_board.board;
            advanced_source[from.y as usize][from.x as usize] = None;
            let mut source_castling = source_board.castling;
            update_castling_rights(&mut source_castling, piece, from, to, source_board.board);
            self.timeline_mut(from.timeline_id)
                .expect("source timeline exists")
                .boards
                .push(BoardSnapshot {
                    time: from.time + 1,
                    side_to_move: next_turn,
                    board: advanced_source,
                    castling: source_castling,
                    en_passant: None,
                    origin: Origin::Move {
                        from,
                        to,
                        move_type: "source-advance",
                    },
                });

            let mut branch_board = target_board.board;
            branch_board[to.y as usize][to.x as usize] = Some(promote_if_needed(piece, to.y));
            let mut target_castling = target_board.castling;
            update_castling_rights(&mut target_castling, piece, from, to, target_board.board);
            self.timeline_mut(destination_timeline_id)
                .expect("destination timeline exists")
                .boards
                .push(BoardSnapshot {
                    time: to.time + 1,
                    side_to_move: next_turn,
                    board: branch_board,
                    castling: target_castling,
                    en_passant: None,
                    origin: Origin::Move {
                        from,
                        to,
                        move_type: if target_is_historical {
                            "branch"
                        } else {
                            "cross-board"
                        },
                    },
                });
        }
    }

    fn place_timeline(&mut self, owner: Color, source_row: i32) -> i32 {
        let direction = if owner == Color::White { 1 } else { -1 };
        let mut row = source_row + direction;

        while self.timelines.iter().any(|timeline| timeline.row == row) {
            row += direction;
        }

        let id = if owner == Color::White {
            let id = self.next_timeline_id;
            self.next_timeline_id += 1;
            id
        } else {
            let id = self.next_black_timeline_id;
            self.next_black_timeline_id -= 1;
            id
        };
        self.timelines.push(Timeline {
            id,
            row,
            label: format!(
                "{} T{}",
                if owner == Color::White {
                    "White"
                } else {
                    "Black"
                },
                id
            ),
            owner: TimelineOwner::from_color(owner),
            boards: Vec::new(),
        });
        id
    }

    fn present_board(&self) -> Option<&BoardSnapshot> {
        self.timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| timeline.boards.iter().max_by_key(|board| board.time))
            .min_by_key(|board| board.time)
    }

    fn is_active_timeline(&self, timeline_id: i32) -> bool {
        let Some(timeline) = self.timeline(timeline_id) else {
            return false;
        };
        match timeline.owner {
            TimelineOwner::Neutral => true,
            TimelineOwner::White | TimelineOwner::Black => {
                let same_owner = self
                    .timelines
                    .iter()
                    .filter(|candidate| {
                        candidate.owner == timeline.owner && candidate.id.abs() <= timeline.id.abs()
                    })
                    .count();
                let opponent_owner = match timeline.owner {
                    TimelineOwner::White => TimelineOwner::Black,
                    TimelineOwner::Black => TimelineOwner::White,
                    TimelineOwner::Neutral => TimelineOwner::Neutral,
                };
                let opponent_count = self
                    .timelines
                    .iter()
                    .filter(|candidate| candidate.owner == opponent_owner)
                    .count();
                same_owner <= opponent_count + 1
            }
        }
    }
}
