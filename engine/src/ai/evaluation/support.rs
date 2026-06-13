use super::*;

impl EvaluationStats {
    pub(crate) fn allow_attack_check(&mut self, limits: EvaluationLimits) -> bool {
        if deadline_expired(limits.deadline) || self.attack_checks >= limits.attack_checks {
            self.attack_caps += 1;
            return false;
        }
        self.attack_checks += 1;
        true
    }
}

#[allow(dead_code)]
impl Game {
    pub(crate) fn latest_position_view(&self) -> LatestPositionView {
        let mut view = LatestPositionView {
            pieces: Vec::new(),
            board_positions: Vec::new(),
            white_royals: Vec::new(),
            black_royals: Vec::new(),
        };
        for timeline in &self.timelines {
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            for y in 0..8 {
                for x in 0..8 {
                    let position = Position {
                        timeline_id: timeline.id,
                        time: board.time,
                        x,
                        y,
                    };
                    view.board_positions.push(position);
                    let Some(piece) = board.board[y as usize][x as usize] else {
                        continue;
                    };
                    view.pieces.push((position, piece));
                    if Self::is_royal_piece(piece.piece_type) {
                        match piece.color {
                            Color::White => view.white_royals.push((position, piece)),
                            Color::Black => view.black_royals.push((position, piece)),
                        }
                    }
                }
            }
        }
        view
    }

    pub(crate) fn individual_royal_safety_scores(&self, color: Color) -> Vec<i32> {
        let mut stats = EvaluationStats::default();
        self.individual_royal_safety_scores_with_limits(color, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn individual_royal_safety_scores_with_limits(
        &self,
        color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> Vec<i32> {
        self.latest_royal_pieces(color)
            .into_iter()
            .map(|(position, _)| {
                self.raw_royal_safety_score_with_limits(position, color, limits, stats)
            })
            .collect()
    }

    pub(crate) fn individual_royal_safety(
        &self,
        position: Position,
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
        let mut stats = EvaluationStats::default();
        self.individual_royal_safety_with_limits(
            position,
            color,
            weights,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn individual_royal_safety_with_limits(
        &self,
        position: Position,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let attackers = self.attack_summary_with_limits(position, color.opposite(), limits, stats);
        let defenders = self.attack_summary_with_limits(position, color, limits, stats);
        let escapes = self.royal_escape_count(position, color);
        let mut score = 0;
        score -= attackers.count * weights.own_royal_exposure;
        score -= attackers.temporal_count * weights.royal_capture_threat;
        score += defenders.count * weights.defended_piece;
        score += escapes * weights.royal_escape_pressure;
        if attackers.count > 0 && escapes == 0 {
            score -= weights.check_penalty / 2;
        }
        score
    }

    pub(crate) fn raw_royal_safety_score(&self, position: Position, color: Color) -> i32 {
        let mut stats = EvaluationStats::default();
        self.raw_royal_safety_score_with_limits(position, color, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn raw_royal_safety_score_with_limits(
        &self,
        position: Position,
        color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let attackers = self.attack_summary_with_limits(position, color.opposite(), limits, stats);
        let defenders = self.attack_summary_with_limits(position, color, limits, stats);
        let escapes = self.royal_escape_count(position, color);
        let shield = self.royal_shield_count(position, color);
        let mut score = defenders.count + shield * 2 + escapes * 3;
        score -= attackers.count * 4 + attackers.temporal_count * 3;
        score -= (attackers.timeline_count - 1).max(0) * 2;
        score -= (attackers.time_count - 1).max(0) * 2;
        if attackers.count > 0 && escapes == 0 {
            score -= 4;
        }
        score
    }

    pub(crate) fn royal_shield_count(&self, position: Position, color: Color) -> i32 {
        let forward = if color == Color::White { 1 } else { -1 };
        let mut shields = 0;
        for dx in -1..=1 {
            let shield = Position {
                timeline_id: position.timeline_id,
                time: position.time,
                x: position.x + dx,
                y: position.y + forward,
            };
            if Self::in_bounds(shield.x, shield.y)
                && self.piece_at(shield).is_some_and(|piece| {
                    piece.color == color
                        && matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
                })
            {
                shields += 1;
            }
        }
        shields
    }

    pub(crate) fn latest_arrival_position(&self, color: Color, x: i32, y: i32) -> Option<Position> {
        self.latest_pieces()
            .into_iter()
            .filter(|(position, piece)| piece.color == color && position.x == x && position.y == y)
            .max_by_key(|(position, _)| position.time)
            .map(|(position, _)| position)
    }

    pub(crate) fn active_timeline_count(&self) -> i32 {
        self.timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .count() as i32
    }

    pub(crate) fn clear_piece_at(&mut self, position: Position) {
        let Some(timeline) = self.timeline_mut(position.timeline_id) else {
            return;
        };
        let Some(board) = timeline
            .boards
            .iter_mut()
            .find(|board| board.time == position.time)
        else {
            return;
        };
        board.board[position.y as usize][position.x as usize] = None;
    }

    pub(crate) fn latest_pieces(&self) -> Vec<(Position, Piece)> {
        let mut pieces = Vec::new();
        for timeline in &self.timelines {
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            for y in 0..8 {
                for x in 0..8 {
                    if let Some(piece) = board.board[y][x] {
                        pieces.push((
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
        pieces
    }

    pub(crate) fn near_enemy_royal_in_view(
        &self,
        target: Position,
        color: Color,
        view: &LatestPositionView,
    ) -> bool {
        view.royals(color.opposite()).iter().any(|(position, _)| {
            let delta = self.movement_delta(target, *position);
            delta
                .x
                .abs()
                .max(delta.y.abs())
                .max(delta.t.abs())
                .max(delta.l.abs())
                <= 2
        })
    }

    pub(crate) fn pseudo_attack_count_in_view(
        &self,
        position: Position,
        piece: Piece,
        view: &LatestPositionView,
    ) -> i32 {
        view.board_positions
            .iter()
            .filter(|target| self.attacks_square(piece, position, **target))
            .count() as i32
    }

    pub(crate) fn pseudo_attack_count_in_view_with_limits(
        &self,
        position: Position,
        piece: Piece,
        view: &LatestPositionView,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        view.board_positions
            .iter()
            .filter(|target| {
                self.attacks_square_with_limits(piece, position, **target, limits, stats)
            })
            .count() as i32
    }

    pub(crate) fn open_line_count(&self, position: Position, piece: Piece) -> i32 {
        let directions: &[(i32, i32, i32, i32)] = &[
            (1, 0, 0, 0),
            (-1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, -1, 0, 0),
            (1, 1, 0, 0),
            (1, -1, 0, 0),
            (-1, 1, 0, 0),
            (-1, -1, 0, 0),
            (0, 0, 2, 0),
            (0, 0, -2, 0),
            (0, 0, 0, 1),
            (0, 0, 0, -1),
        ];
        directions
            .iter()
            .filter(|(dx, dy, dt, dl)| {
                self.first_step_on_line(position, *dx, *dy, *dt, *dl)
                    .is_some_and(|target| {
                        self.piece_at(target).is_none()
                            && self.attacks_square(piece, position, target)
                    })
            })
            .count() as i32
    }

    pub(crate) fn open_line_count_with_limits(
        &self,
        position: Position,
        piece: Piece,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let directions: &[(i32, i32, i32, i32)] = &[
            (1, 0, 0, 0),
            (-1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, -1, 0, 0),
            (1, 1, 0, 0),
            (1, -1, 0, 0),
            (-1, 1, 0, 0),
            (-1, -1, 0, 0),
            (0, 0, 2, 0),
            (0, 0, -2, 0),
            (0, 0, 0, 1),
            (0, 0, 0, -1),
        ];
        directions
            .iter()
            .filter(|(dx, dy, dt, dl)| {
                self.first_step_on_line(position, *dx, *dy, *dt, *dl)
                    .is_some_and(|target| {
                        self.piece_at(target).is_none()
                            && self
                                .attacks_square_with_limits(piece, position, target, limits, stats)
                    })
            })
            .count() as i32
    }

    pub(crate) fn first_step_on_line(
        &self,
        position: Position,
        dx: i32,
        dy: i32,
        dt: i32,
        dl: i32,
    ) -> Option<Position> {
        let from_row = self.timeline(position.timeline_id)?.row;
        let target_row = from_row + dl;
        let target_timeline_id = if dl == 0 {
            position.timeline_id
        } else {
            self.timelines
                .iter()
                .find(|timeline| timeline.row == target_row)?
                .id
        };
        let target = Position {
            timeline_id: target_timeline_id,
            time: position.time + dt,
            x: position.x + dx,
            y: position.y + dy,
        };
        (Self::in_bounds(target.x, target.y)
            && self.board(target.timeline_id, target.time).is_some())
        .then_some(target)
    }

    pub(crate) fn is_passed_pawn(&self, position: Position, color: Color) -> bool {
        let direction = if color == Color::White { 1 } else { -1 };
        let Some(board) = self.board(position.timeline_id, position.time) else {
            return false;
        };
        let mut y = position.y + direction;
        while Self::in_bounds(position.x, y) {
            for x in position.x - 1..=position.x + 1 {
                if Self::in_bounds(x, y)
                    && board.board[y as usize][x as usize].is_some_and(|piece| {
                        piece.color == color.opposite()
                            && matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
                    })
                {
                    return false;
                }
            }
            y += direction;
        }
        true
    }

    pub(crate) fn is_supported_pawn(&self, position: Position, color: Color) -> bool {
        let behind = if color == Color::White { -1 } else { 1 };
        [-1, 1].into_iter().any(|dx| {
            let supporter = Position {
                timeline_id: position.timeline_id,
                time: position.time,
                x: position.x + dx,
                y: position.y + behind,
            };
            Self::in_bounds(supporter.x, supporter.y)
                && self.piece_at(supporter).is_some_and(|piece| {
                    piece.color == color
                        && matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
                })
        })
    }

    pub(crate) fn is_isolated_pawn(&self, position: Position, color: Color) -> bool {
        let Some(board) = self.board(position.timeline_id, position.time) else {
            return false;
        };
        ![-1, 1].into_iter().any(|dx| {
            let file = position.x + dx;
            Self::in_bounds(file, position.y)
                && (0..8).any(|y| {
                    board.board[y][file as usize].is_some_and(|piece| {
                        piece.color == color
                            && matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
                    })
                })
        })
    }

    pub(crate) fn timeline_time_spread(&self) -> i32 {
        let mut latest_times = self
            .timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| self.latest_time(timeline.id));
        let Some(first) = latest_times.next() else {
            return 0;
        };
        let (min, max) = latest_times.fold((first, first), |(min, max), time| {
            (min.min(time), max.max(time))
        });
        max - min
    }

    pub(crate) fn attack_summary(&self, target: Position, by_color: Color) -> AttackSummary {
        let mut summary = AttackSummary::default();
        let mut timelines = Vec::new();
        let mut times = Vec::new();

        for (from, piece) in self.latest_pieces() {
            if piece.color != by_color
                || from.timeline_id == target.timeline_id
                    && from.time == target.time
                    && from.x == target.x
                    && from.y == target.y
                || !self.attacks_square(piece, from, target)
            {
                continue;
            }

            summary.count += 1;
            if from.timeline_id != target.timeline_id || from.time != target.time {
                summary.temporal_count += 1;
            }
            if !timelines.contains(&from.timeline_id) {
                timelines.push(from.timeline_id);
            }
            if !times.contains(&from.time) {
                times.push(from.time);
            }
        }

        summary.timeline_count = timelines.len() as i32;
        summary.time_count = times.len() as i32;
        summary
    }

    pub(crate) fn attacks_square_with_limits(
        &self,
        piece: Piece,
        from: Position,
        target: Position,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> bool {
        stats.allow_attack_check(limits) && self.attacks_square(piece, from, target)
    }

    pub(crate) fn attack_summary_with_limits(
        &self,
        target: Position,
        by_color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> AttackSummary {
        let mut summary = AttackSummary::default();
        let mut timelines = Vec::new();
        let mut times = Vec::new();

        for (from, piece) in self.latest_pieces() {
            if piece.color != by_color
                || from.timeline_id == target.timeline_id
                    && from.time == target.time
                    && from.x == target.x
                    && from.y == target.y
                || !self.attacks_square_with_limits(piece, from, target, limits, stats)
            {
                continue;
            }

            summary.count += 1;
            if from.timeline_id != target.timeline_id || from.time != target.time {
                summary.temporal_count += 1;
            }
            if !timelines.contains(&from.timeline_id) {
                timelines.push(from.timeline_id);
            }
            if !times.contains(&from.time) {
                times.push(from.time);
            }
        }

        summary.timeline_count = timelines.len() as i32;
        summary.time_count = times.len() as i32;
        summary
    }
}

pub(crate) fn weights_for_tropism(piece_type: PieceType) -> i32 {
    match piece_type {
        PieceType::King | PieceType::CommonKing => 1,
        PieceType::Pawn | PieceType::Brawn => 1,
        PieceType::Knight | PieceType::Bishop => 2,
        PieceType::Rook | PieceType::Princess | PieceType::Unicorn => 3,
        PieceType::Dragon | PieceType::Queen | PieceType::RoyalQueen => 4,
    }
}
