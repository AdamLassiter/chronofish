impl Game {
    fn mobility_balance(&self, color: Color) -> i32 {
        self.legal_single_move_count_for(color) - self.legal_single_move_count_for(color.opposite())
    }

    fn legal_single_move_count_for(&self, color: Color) -> i32 {
        let mut count = 0;
        for timeline in &self.timelines {
            if !self.is_active_timeline(timeline.id) {
                continue;
            }
            for board in &timeline.boards {
                if !self.is_latest_board(timeline.id, board.time) || board.side_to_move != color {
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
                        for target_timeline in &self.timelines {
                            for target_board in &target_timeline.boards {
                                for target_y in 0..8 {
                                    for target_x in 0..8 {
                                        let to = Position {
                                            timeline_id: target_timeline.id,
                                            time: target_board.time,
                                            x: target_x,
                                            y: target_y,
                                        };
                                        let Some((piece, move_kind)) =
                                            self.legal_move_kind_for_turn(color, from, to)
                                        else {
                                            continue;
                                        };
                                        if self.allows_search_move(from, to, piece, move_kind) {
                                            count += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        count
    }

    fn legal_move_kind_for_turn(
        &self,
        turn: Color,
        from: Position,
        to: Position,
    ) -> Option<(Piece, MoveKind)> {
        if !Self::in_bounds(from.x, from.y) || !Self::in_bounds(to.x, to.y) {
            return None;
        }
        let source_board = self.board(from.timeline_id, from.time)?;
        let target_board = self.board(to.timeline_id, to.time)?;
        let piece = source_board.board[from.y as usize][from.x as usize]?;
        let same_board = from.timeline_id == to.timeline_id && from.time == to.time;

        if !self.is_present_source_board(from)
            || !self.is_latest_board(from.timeline_id, from.time)
            || source_board.side_to_move != turn
            || piece.color != turn
        {
            return None;
        }
        if !same_board && target_board.side_to_move != piece.color {
            return None;
        }
        if target_board.board[to.y as usize][to.x as usize]
            .is_some_and(|target| target.color == piece.color)
        {
            return None;
        }

        self.move_kind_for(piece, from, to).map(|kind| (piece, kind))
    }
}
