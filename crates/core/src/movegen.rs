use crate::*;

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

    pub(crate) fn is_standard_stalemate(&self, color: Color) -> bool {
        self.is_standard_stalemate_until(color, None)
    }

    pub(crate) fn is_standard_stalemate_until(
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

impl Game {
    pub(crate) fn for_each_piece_candidate_destination(
        &self,
        from: Position,
        piece: Piece,
        mut visit: impl FnMut(Position) -> bool,
    ) {
        match piece.piece_type {
            PieceType::Pawn => {
                self.visit_pawn_candidates(from, piece.color, false, &mut visit);
            }
            PieceType::Brawn => {
                self.visit_pawn_candidates(from, piece.color, true, &mut visit);
            }
            PieceType::Knight => {
                for long_axis in 0..4 {
                    for short_axis in 0..4 {
                        if long_axis == short_axis {
                            continue;
                        }
                        for long_sign in [-1, 1] {
                            for short_sign in [-1, 1] {
                                let mut offset = [0; 4];
                                offset[long_axis] = long_sign * 2;
                                offset[short_axis] = short_sign;
                                if let Some(target) = self.offset_target(from, offset) {
                                    if !visit(target) {
                                        return;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            PieceType::King | PieceType::CommonKing => {
                if !self.visit_direction_targets(from, 1, 4, 1, &mut visit) {
                    return;
                }
                for offset in [[2, 0, 0, 0], [-2, 0, 0, 0]] {
                    if let Some(target) = self.offset_target(from, offset) {
                        if !visit(target) {
                            return;
                        }
                    }
                }
            }
            PieceType::Rook => {
                self.visit_slider_targets(from, 1, 1, &mut visit);
            }
            PieceType::Bishop => {
                self.visit_slider_targets(from, 2, 2, &mut visit);
            }
            PieceType::Unicorn => {
                self.visit_slider_targets(from, 3, 3, &mut visit);
            }
            PieceType::Dragon => {
                self.visit_slider_targets(from, 4, 4, &mut visit);
            }
            PieceType::Princess => {
                self.visit_slider_targets(from, 1, 2, &mut visit);
            }
            PieceType::Queen | PieceType::RoyalQueen => {
                self.visit_slider_targets(from, 1, 4, &mut visit);
            }
        }
    }

    fn visit_pawn_candidates(
        &self,
        from: Position,
        color: Color,
        brawn: bool,
        visit: &mut impl FnMut(Position) -> bool,
    ) -> bool {
        let forward = if color == Color::White { 1 } else { -1 };
        for offset in [
            [0, forward, 0, 0],
            [0, forward * 2, 0, 0],
            [-1, forward, 0, 0],
            [1, forward, 0, 0],
            [0, 0, 0, forward],
            [0, 0, 0, forward * 2],
            [0, 0, -1, forward],
            [0, 0, 1, forward],
        ] {
            if let Some(target) = self.offset_target(from, offset) {
                if !visit(target) {
                    return false;
                }
            }
        }
        if brawn {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dt in -1..=1 {
                        for dl in -1..=1 {
                            let offset = [dx, dy, dt, dl];
                            let changed = offset.iter().filter(|value| **value != 0).count();
                            if changed >= 2
                                && (dy == forward || dl == forward)
                                && dy != -forward
                                && dl != -forward
                            {
                                if let Some(target) = self.offset_target(from, offset) {
                                    if !visit(target) {
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        true
    }

    fn visit_slider_targets(
        &self,
        from: Position,
        min_axes: usize,
        max_axes: usize,
        visit: &mut impl FnMut(Position) -> bool,
    ) -> bool {
        self.visit_direction_targets(from, min_axes, max_axes, self.max_ray_distance(from), visit)
    }

    fn visit_direction_targets(
        &self,
        from: Position,
        min_axes: usize,
        max_axes: usize,
        max_distance: i32,
        visit: &mut impl FnMut(Position) -> bool,
    ) -> bool {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dt in -1..=1 {
                    for dl in -1..=1 {
                        let direction = [dx, dy, dt, dl];
                        let axes = direction.iter().filter(|value| **value != 0).count();
                        if axes < min_axes || axes > max_axes {
                            continue;
                        }
                        for distance in 1..=max_distance {
                            let Some(target) =
                                self.offset_target(from, direction.map(|v| v * distance))
                            else {
                                break;
                            };
                            if !visit(target) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }

    fn offset_target(&self, from: Position, [dx, dy, dt, dl]: [i32; 4]) -> Option<Position> {
        let x = from.x + dx;
        let y = from.y + dy;
        if !Self::in_bounds(x, y) {
            return None;
        }
        if self.ruleset == Ruleset::Standard && (dt != 0 || dl != 0) {
            return None;
        }
        let from_row = self
            .timeline(from.timeline_id)
            .map_or(0, |timeline| timeline.row);
        let timeline = self
            .timelines
            .iter()
            .find(|timeline| timeline.row == from_row + dl)?;
        let time = from.time + dt * 2;
        self.board(timeline.id, time).is_some().then_some(Position {
            timeline_id: timeline.id,
            time,
            x,
            y,
        })
    }

    fn max_ray_distance(&self, from: Position) -> i32 {
        if self.ruleset == Ruleset::Standard {
            return 7;
        }
        let from_row = self
            .timeline(from.timeline_id)
            .map_or(0, |timeline| timeline.row);
        self.timelines
            .iter()
            .flat_map(|timeline| {
                timeline.boards.iter().map(move |board| {
                    (timeline.row - from_row)
                        .abs()
                        .max((board.time - from.time).abs() / 2)
                })
            })
            .max()
            .unwrap_or(0)
            .max(7)
    }

    fn apply_legal_move_for_search(
        &mut self,
        from: Position,
        to: Position,
        piece: Piece,
        kind: MoveKind,
    ) {
        let captured = self.captured_piece(to, kind);
        self.record_staged_capture(piece.color, captured);
        self.apply_move_unchecked(from, to, piece, kind);
    }

    pub(crate) fn make_search_move(&mut self, movement: MoveStep) -> Option<SearchUndo> {
        let (piece, move_kind) = self.legal_move_kind(movement.from, movement.to)?;
        if !self.allows_search_move(movement.from, movement.to, piece, move_kind) {
            return None;
        }
        let source_board_len = (
            movement.from.timeline_id,
            self.timeline(movement.from.timeline_id)?.boards.len(),
        );
        let target_board_len = if matches!(move_kind, MoveKind::Branch)
            && self.is_latest_board(movement.to.timeline_id, movement.to.time)
            && movement.to.timeline_id != movement.from.timeline_id
        {
            Some((
                movement.to.timeline_id,
                self.timeline(movement.to.timeline_id)?.boards.len(),
            ))
        } else {
            None
        };
        let undo = SearchUndo {
            timeline_count: self.timelines.len(),
            source_board_len,
            target_board_len,
            next_timeline_id: self.next_timeline_id,
            next_black_timeline_id: self.next_black_timeline_id,
            staged_royal_capture_by: self.staged_royal_capture_by,
            position_hash: self.position_hash,
        };
        self.apply_legal_move_for_search(movement.from, movement.to, piece, move_kind);
        Some(undo)
    }

    pub(crate) fn unmake_search_move(&mut self, undo: SearchUndo) {
        self.timelines.truncate(undo.timeline_count);
        if let Some(timeline) = self.timeline_mut(undo.source_board_len.0) {
            timeline.boards.truncate(undo.source_board_len.1);
        }
        if let Some((id, len)) = undo.target_board_len {
            if let Some(timeline) = self.timeline_mut(id) {
                timeline.boards.truncate(len);
            }
        }
        self.next_timeline_id = undo.next_timeline_id;
        self.next_black_timeline_id = undo.next_black_timeline_id;
        self.staged_royal_capture_by = undo.staged_royal_capture_by;
        self.position_hash = undo.position_hash;
    }
}
