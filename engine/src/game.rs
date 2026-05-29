impl Game {
    // Create the default orthodox board on neutral T0. The expanded piece model
    // is implemented but not used by the initial setup.
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
            staged_notation: Vec::new(),
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

    // Shared legality gate for UI highlighting, user moves, and AI search. It
    // checks turn ownership, latest-board source rules, destination turn rules
    // for time travel, and friendly occupancy before piece geometry runs.
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

        // Moves are staged until submit_turn accepts the whole turn. This allows
        // a side to make the multiple moves needed to advance the present line.
        let notation = self.move_notation(piece, from, to, move_kind);
        self.staged_turn.push(self.checkpoint());
        self.apply_move_unchecked(from, to, piece, move_kind);
        self.staged_notation.push(self.finish_move_notation(notation, piece.color));
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

    // Capture enough state to undo one staged move.
    fn checkpoint(&self) -> GameCheckpoint {
        GameCheckpoint {
            turn: self.turn,
            timelines: self.timelines.clone(),
            next_timeline_id: self.next_timeline_id,
            next_black_timeline_id: self.next_black_timeline_id,
            staged_notation: self.staged_notation.clone(),
            last_message: self.last_message.clone(),
        }
    }

    fn restore(&mut self, checkpoint: GameCheckpoint) {
        self.turn = checkpoint.turn;
        self.timelines = checkpoint.timelines;
        self.next_timeline_id = checkpoint.next_timeline_id;
        self.next_black_timeline_id = checkpoint.next_black_timeline_id;
        self.staged_notation = checkpoint.staged_notation;
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

        // Turn passing follows the present line, not a simple alternating
        // single-move clock. The player keeps moving until the earliest active
        // latest board is now waiting for the opponent.
        let Some(present_side) = self.present_board().map(|board| board.side_to_move) else {
            self.last_message = "No active present board.".to_string();
            return 0;
        };

        if self.has_pending_present_board(self.turn) {
            self.last_message =
                "Make moves until the present line reaches the opponent's turn.".to_string();
            return 0;
        }

        self.turn = present_side;
        self.staged_turn.clear();
        self.staged_notation.clear();

        if self.royal_capture_available(self.turn) {
            self.last_message = format!(
                "{} wins by checkmate.",
                self.turn.capitalized()
            );
            return 1;
        }

        let suffix = if self.is_checkmate(self.turn) {
            " Checkmate."
        } else if self.is_in_check(self.turn) {
            " Check."
        } else {
            ""
        };
        self.last_message = format!(
            "{} to move.{}",
            self.turn.capitalized(),
            suffix
        );
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
            // Same-board moves append one successor snapshot and carry forward
            // transient state such as castling rights and en-passant eligibility.
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
            // Time/timeline moves advance the source line without the piece, then
            // place it on the destination. Targeting a historical board creates a
            // new branch owned by the moving color.
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
        // White-created ids are positive and black-created ids are negative,
        // matching RULES.md notation while row supplies the visual L-axis.
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
        // The present line is the earliest latest-board among active timelines.
        // Inactive timelines do not hold the turn hostage.
        self.timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| timeline.boards.iter().max_by_key(|board| board.time))
            .min_by_key(|board| board.time)
    }

    fn present_time(&self) -> Option<i32> {
        self.present_board().map(|board| board.time)
    }

    fn has_pending_present_board(&self, color: Color) -> bool {
        let Some(present_time) = self.present_time() else {
            return false;
        };
        self.timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| timeline.boards.iter().max_by_key(|board| board.time))
            .any(|board| board.time == present_time && board.side_to_move == color)
    }

    fn is_active_timeline(&self, timeline_id: i32) -> bool {
        let Some(timeline) = self.timeline(timeline_id) else {
            return false;
        };
        if timeline.owner == TimelineOwner::Neutral {
            return true;
        }

        // Active timelines are balanced by distance from T0, not by owner rank.
        // If one side has branched farther than the other, only the outermost
        // timelines on that side go inactive.
        let min_timeline = self
            .timelines
            .iter()
            .map(|timeline| timeline.id)
            .min()
            .unwrap_or(0);
        let max_timeline = self
            .timelines
            .iter()
            .map(|timeline| timeline.id)
            .max()
            .unwrap_or(0);
        let active_distance = (-min_timeline).min(max_timeline).max(0) + 1;

        timeline.id.abs() <= active_distance
    }
}
