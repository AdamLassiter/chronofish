use crate::{
    cpu::{deadline_expired, MoveStep, SearchInstant},
    *,
};

impl Game {
    // Check is evaluated over the latest board of every timeline because royal
    // pieces may exist on multiple active branch fronts.
    pub(crate) fn is_in_check(&self, color: Color) -> bool {
        for timeline in &self.timelines {
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            for y in 0..8 {
                for x in 0..8 {
                    let Some(piece) = board.board[y][x] else {
                        continue;
                    };
                    if piece.color == color
                        && Self::is_royal_piece(piece.piece_type)
                        && self.is_square_attacked(
                            Position {
                                timeline_id: timeline.id,
                                time: board.time,
                                x: x as i32,
                                y: y as i32,
                            },
                            color.opposite(),
                        )
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub(crate) fn has_latest_royal_piece(&self, color: Color) -> bool {
        self.timelines.iter().any(|timeline| {
            timeline.boards.last().is_some_and(|board| {
                board.board.iter().any(|rank| {
                    rank.iter().any(|piece| {
                        piece.is_some_and(|piece| {
                            piece.color == color && Self::is_royal_piece(piece.piece_type)
                        })
                    })
                })
            })
        })
    }

    #[allow(dead_code)]
    pub(crate) fn checked_royal_positions(&self) -> Vec<Position> {
        let Some(present_time) = self.present_time() else {
            return Vec::new();
        };

        [Color::White, Color::Black]
            .into_iter()
            .flat_map(|color| {
                self.royal_piece_positions(color)
                    .into_iter()
                    .filter(move |position| position.time == present_time)
                    .filter(move |position| self.is_square_attacked(*position, color.opposite()))
            })
            .collect::<Vec<_>>()
    }

    #[allow(dead_code)]
    pub(crate) fn is_checkmate(&self, color: Color) -> bool {
        if !self.is_in_check(color) {
            return false;
        }

        let mut search = self.clone_for_search();
        search.turn = color;
        !search.has_legal_turn_completion(color)
    }

    pub(crate) fn is_classic_stalemate(&self, color: Color) -> bool {
        self.is_classic_stalemate_until(color, None)
    }

    pub(crate) fn is_classic_stalemate_until(
        &self,
        color: Color,
        deadline: Option<SearchInstant>,
    ) -> bool {
        if !self.has_latest_royal_piece(color) || self.is_in_check(color) {
            return false;
        }

        let mut search = self.clone_for_search();
        search.turn = color;
        !search.has_legal_turn_completion_until(color, deadline)
    }

    #[allow(dead_code)]
    pub(crate) fn royal_capture_available(&self, color: Color) -> bool {
        let Some(present_time) = self.present_time() else {
            return false;
        };

        for target_timeline in &self.timelines {
            let Some(target_board) = target_timeline.boards.last() else {
                continue;
            };
            for target_y in 0..8 {
                for target_x in 0..8 {
                    let Some(target_piece) = target_board.board[target_y][target_x] else {
                        continue;
                    };
                    if target_piece.color != color.opposite()
                        || !Self::is_royal_piece(target_piece.piece_type)
                    {
                        continue;
                    }
                    let target = Position {
                        timeline_id: target_timeline.id,
                        time: target_board.time,
                        x: target_x as i32,
                        y: target_y as i32,
                    };
                    if self.royal_capture_target_is_reachable(
                        color,
                        present_time,
                        target,
                        target_board.side_to_move,
                    ) {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn royal_capture_target_is_reachable(
        &self,
        color: Color,
        present_time: i32,
        target: Position,
        target_side_to_move: Color,
    ) -> bool {
        for timeline in &self.timelines {
            if !self.is_active_timeline(timeline.id) {
                continue;
            }
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            if board.time != present_time || board.side_to_move != color {
                continue;
            }

            for y in 0..8 {
                for x in 0..8 {
                    let Some(piece) = board.board[y][x] else {
                        continue;
                    };
                    if piece.color != color {
                        continue;
                    }
                    let from = Position {
                        timeline_id: timeline.id,
                        time: board.time,
                        x: x as i32,
                        y: y as i32,
                    };
                    let same_board =
                        from.timeline_id == target.timeline_id && from.time == target.time;
                    if !same_board && target_side_to_move != color {
                        continue;
                    }
                    if self.move_kind_for(piece, from, target).is_some() {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[cfg(test)]
    pub(crate) fn royal_capture_available_via_legal_moves(&self, color: Color) -> bool {
        let mut search = self.clone_for_search();
        search.turn = color;

        for target in search.royal_piece_positions(color.opposite()) {
            for timeline in &search.timelines {
                if !search.is_active_timeline(timeline.id) {
                    continue;
                }
                let Some(board) = timeline.boards.last() else {
                    continue;
                };
                if board.side_to_move != color {
                    continue;
                }

                for y in 0..8 {
                    for x in 0..8 {
                        if !board.board[y][x].is_some_and(|piece| piece.color == color) {
                            continue;
                        }
                        let from = Position {
                            timeline_id: timeline.id,
                            time: board.time,
                            x: x as i32,
                            y: y as i32,
                        };
                        if search.legal_move_kind(from, target).is_some() {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    #[allow(dead_code)]
    pub(crate) fn has_legal_turn_completion(&self, color: Color) -> bool {
        self.has_legal_turn_completion_until(color, None)
    }

    pub(crate) fn has_legal_turn_completion_until(
        &self,
        color: Color,
        deadline: Option<SearchInstant>,
    ) -> bool {
        // Escaping check may require a whole-turn sequence, not just one move, so
        // mate search follows staged moves until the present line changes color.
        let max_depth = self
            .timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .count()
            + 4;
        let mut search = self.clone_for_search();
        search.has_legal_turn_completion_in_place(color, 0, max_depth, deadline)
    }

    #[allow(dead_code)]
    pub(crate) fn has_legal_turn_completion_at_depth(
        &self,
        color: Color,
        depth: usize,
        max_depth: usize,
    ) -> bool {
        let mut search = self.clone_for_search();
        search.has_legal_turn_completion_in_place(color, depth, max_depth, None)
    }

    fn has_legal_turn_completion_in_place(
        &mut self,
        color: Color,
        depth: usize,
        max_depth: usize,
        deadline: Option<SearchInstant>,
    ) -> bool {
        if !self.has_pending_present_board(color) {
            return !self.is_in_check(color);
        }

        if depth >= max_depth || deadline_expired(deadline) {
            return false;
        }

        let mut moves = Vec::new();
        for timeline in &self.timelines {
            if !self.is_active_timeline(timeline.id) {
                continue;
            }
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            if board.side_to_move != color {
                continue;
            }

            for y in 0..8 {
                for x in 0..8 {
                    let from = Position {
                        timeline_id: timeline.id,
                        time: board.time,
                        x,
                        y,
                    };
                    if !self
                        .piece_at(from)
                        .is_some_and(|piece| piece.color == color)
                    {
                        continue;
                    }

                    let piece = self.piece_at(from).expect("source piece was checked");
                    self.for_each_piece_candidate_destination(from, piece, |to| {
                        if deadline_expired(deadline) {
                            return false;
                        }
                        let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
                            return true;
                        };
                        if self.allows_search_move(from, to, piece, move_kind) {
                            let movement = MoveStep { from, to };
                            if !moves.contains(&movement) {
                                moves.push(movement);
                            }
                        }
                        true
                    });
                    if deadline_expired(deadline) {
                        return false;
                    }
                }
            }
        }

        for movement in moves {
            if deadline_expired(deadline) {
                return false;
            }
            let Some(undo) = self.make_search_move(movement) else {
                continue;
            };
            let completes =
                self.has_legal_turn_completion_in_place(color, depth + 1, max_depth, deadline);
            self.unmake_search_move(undo);
            if completes {
                return true;
            }
        }

        false
    }

    pub(crate) fn royal_piece_positions(&self, color: Color) -> Vec<Position> {
        let mut positions = Vec::new();
        for timeline in &self.timelines {
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            for y in 0..8 {
                for x in 0..8 {
                    let Some(piece) = board.board[y][x] else {
                        continue;
                    };
                    if piece.color == color && Self::is_royal_piece(piece.piece_type) {
                        positions.push(Position {
                            timeline_id: timeline.id,
                            time: board.time,
                            x: x as i32,
                            y: y as i32,
                        });
                    }
                }
            }
        }
        positions
    }

    pub(crate) fn latest_royal_pieces(&self, color: Color) -> Vec<(Position, Piece)> {
        let mut positions = Vec::new();
        for timeline in &self.timelines {
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            for y in 0..8 {
                for x in 0..8 {
                    let Some(piece) = board.board[y][x] else {
                        continue;
                    };
                    if piece.color == color && Self::is_royal_piece(piece.piece_type) {
                        positions.push((
                            Position {
                                timeline_id: timeline.id,
                                time: board.time,
                                x: x as i32,
                                y: y as i32,
                            },
                            piece,
                        ));
                    }
                }
            }
        }
        positions
    }

    pub(crate) fn royal_pieces(&self, color: Color) -> Vec<(Position, Piece)> {
        let mut positions = Vec::new();
        for timeline in &self.timelines {
            for board in &timeline.boards {
                for y in 0..8 {
                    for x in 0..8 {
                        let Some(piece) = board.board[y][x] else {
                            continue;
                        };
                        if piece.color != color || !Self::is_royal_piece(piece.piece_type) {
                            continue;
                        }
                        positions.push((
                            Position {
                                timeline_id: timeline.id,
                                time: board.time,
                                x: x as i32,
                                y: y as i32,
                            },
                            piece,
                        ));
                    }
                }
            }
        }
        positions
    }
}
