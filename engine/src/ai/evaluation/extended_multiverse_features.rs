#![allow(dead_code)]

use super::*;

impl Game {
    pub(crate) fn latest_material_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.latest_piece_score_sum(|_, piece| {
            if piece.color == color {
                weights.piece_value(piece.piece_type)
            } else {
                0
            }
        })
    }

    pub(crate) fn source_material_abandonment_cost(
        &self,
        from: Position,
        piece: Piece,
        weights: &EvalWeights,
    ) -> i32 {
        self.board(from.timeline_id, from.time)
            .map(|board| {
                weights.piece_value(piece.piece_type)
                    * self.board_importance(from.timeline_id, board)
                    / 200
            })
            .unwrap_or(0)
    }

    pub(crate) fn opponent_temporal_tactic_pressure(
        &self,
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
        let mut stats = EvaluationStats::default();
        self.opponent_temporal_tactic_pressure_with_limits(
            color,
            weights,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn opponent_temporal_tactic_pressure_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        self.royal_capture_pressure_for_with_limits(color.opposite(), weights, limits, stats)
            + self.temporal_royal_corridor_pressure_for_with_limits(
                color.opposite(),
                weights,
                limits,
                stats,
            )
    }

    pub(crate) fn royal_liability_score_for(&self, color: Color) -> i32 {
        let mut stats = EvaluationStats::default();
        self.royal_liability_score_for_with_limits(color, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn royal_liability_score_for_with_limits(
        &self,
        color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let royal_scores = self.individual_royal_safety_scores_with_limits(color, limits, stats);
        royal_scores.iter().filter(|score| **score < 0).count() as i32
            + royal_scores.len().saturating_sub(1) as i32
    }

    pub(crate) fn multi_royal_attack_score_for(&self, color: Color) -> i32 {
        let mut stats = EvaluationStats::default();
        self.multi_royal_attack_score_for_with_limits(color, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn multi_royal_attack_score_for_with_limits(
        &self,
        color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let enemy_royals = self.latest_royal_pieces(color.opposite());
        let mut score = 0;
        for (from, piece) in self.latest_pieces() {
            if stats.attack_budget_exhausted(limits) {
                break;
            }
            if piece.color != color {
                continue;
            }
            let mut count = 0;
            for (target, _) in &enemy_royals {
                if stats.attack_budget_exhausted(limits) {
                    break;
                }
                if self.attacks_square_with_limits(piece, from, *target, limits, stats) {
                    count += 1;
                }
            }
            if count >= 2 {
                score += count - 1;
            }
        }
        score
    }

    pub(crate) fn urgent_threat_count_for(&self, color: Color) -> i32 {
        let mut stats = EvaluationStats::default();
        self.urgent_threat_count_for_with_limits(color, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn urgent_threat_count_for_with_limits(
        &self,
        color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let enemy_royals = self.latest_piece_score_sum(|_, piece| {
            (piece.color == color.opposite() && Self::is_royal_piece(piece.piece_type)) as i32
        });
        let mut enemy_hanging = 0;
        for (position, piece) in self.latest_pieces() {
            if stats.attack_budget_exhausted(limits) {
                break;
            }
            if piece.color != color.opposite() {
                continue;
            }
            let attackers = self
                .attack_summary_with_limits(position, color, limits, stats)
                .count;
            if attackers == 0 || stats.attack_budget_exhausted(limits) {
                continue;
            }
            let defenders = self
                .attack_summary_with_limits(position, color.opposite(), limits, stats)
                .count;
            if defenders == 0 {
                enemy_hanging += 1;
            }
        }
        enemy_royals + enemy_hanging
    }

    pub(crate) fn active_branch_capacity_score_for(&self, color: Color) -> i32 {
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
        let frontier = match color {
            Color::White => max_timeline.max(0),
            Color::Black => (-min_timeline).max(0),
        };
        (active_distance - frontier).max(0)
    }

    pub(crate) fn latent_timeline_reactivation_score_for(
        &self,
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
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
        let owner = TimelineOwner::from_color(color);
        self.timelines
            .iter()
            .filter(|timeline| timeline.owner == owner && !self.is_active_timeline(timeline.id))
            .map(|timeline| {
                let distance = (timeline.id.abs() - active_distance).max(1);
                self.timeline_material(timeline.id, weights) / distance
            })
            .sum()
    }

    pub(crate) fn inactive_material_score_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let owner = TimelineOwner::from_color(color);
        self.timelines
            .iter()
            .filter(|timeline| timeline.owner == owner && !self.is_active_timeline(timeline.id))
            .map(|timeline| self.timeline_material(timeline.id, weights) / 100)
            .sum()
    }

    pub(crate) fn timeline_material(&self, timeline_id: i32, weights: &EvalWeights) -> i32 {
        let Some(timeline) = self.timeline(timeline_id) else {
            return 0;
        };
        timeline
            .boards
            .iter()
            .flat_map(|board| board.board.iter().flatten())
            .filter_map(|piece| *piece)
            .map(|piece| weights.piece_value(piece.piece_type))
            .sum()
    }

    pub(crate) fn timeline_compaction_score_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let active_material = self.latest_piece_score_sum(|position, piece| {
            if piece.color == color && self.is_active_timeline(position.timeline_id) {
                weights.piece_value(piece.piece_type) / 100
            } else {
                0
            }
        });
        let inactive_material = self.inactive_material_score_for(color, weights);
        let present_material = self.present_board_material(color, weights).max(0);
        active_material + present_material - inactive_material
    }

    pub(crate) fn present_board_material(&self, color: Color, weights: &EvalWeights) -> i32 {
        let Some(present_time) = self.present_time() else {
            return 0;
        };
        self.timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| timeline.boards.last().map(|board| (timeline, board)))
            .filter(|(_, board)| board.time == present_time)
            .map(|(_, board)| {
                board
                    .board
                    .iter()
                    .flatten()
                    .filter_map(|piece| *piece)
                    .map(|piece| {
                        if piece.color == color {
                            weights.piece_value(piece.piece_type) / 100
                        } else {
                            -weights.piece_value(piece.piece_type) / 100
                        }
                    })
                    .sum::<i32>()
            })
            .sum()
    }

    pub(crate) fn historical_access_score_for(&self, color: Color) -> i32 {
        let mut stats = EvaluationStats::default();
        self.historical_access_score_for_with_limits(color, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn historical_access_score_for_with_limits(
        &self,
        color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let mut score = 0;
        for (from, piece) in self.latest_pieces() {
            if stats.attack_budget_exhausted(limits) {
                break;
            }
            if piece.color != color {
                continue;
            }
            for timeline in &self.timelines {
                if stats.attack_budget_exhausted(limits) {
                    break;
                }
                for board in &timeline.boards {
                    if stats.attack_budget_exhausted(limits) {
                        break;
                    }
                    if self.is_latest_board(timeline.id, board.time) {
                        continue;
                    }
                    for y in 0..8 {
                        if stats.attack_budget_exhausted(limits) {
                            break;
                        }
                        for x in 0..8 {
                            if stats.attack_budget_exhausted(limits) {
                                break;
                            }
                            let target = Position {
                                timeline_id: timeline.id,
                                time: board.time,
                                x: x as i32,
                                y: y as i32,
                            };
                            if self.attacks_square_with_limits(piece, from, target, limits, stats) {
                                score += 1;
                                if board.board[y][x].is_some_and(|target_piece| {
                                    Self::is_royal_piece(target_piece.piece_type)
                                }) {
                                    score += 2;
                                }
                            }
                        }
                    }
                }
            }
        }
        score
    }

    pub(crate) fn temporal_lane_control_score_for(&self, color: Color) -> i32 {
        let mut stats = EvaluationStats::default();
        self.temporal_lane_control_score_for_with_limits(color, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn temporal_lane_control_score_for_with_limits(
        &self,
        color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let mut score = 0;
        for (position, piece) in self.latest_pieces() {
            if stats.attack_budget_exhausted(limits) {
                break;
            }
            if piece.color == color {
                score += self.temporal_open_line_count_with_limits(position, piece, limits, stats);
            }
        }
        score
    }

    pub(crate) fn temporal_open_line_count(&self, position: Position, piece: Piece) -> i32 {
        let mut stats = EvaluationStats::default();
        self.temporal_open_line_count_with_limits(
            position,
            piece,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn temporal_open_line_count_with_limits(
        &self,
        position: Position,
        piece: Piece,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        if !self.is_temporal_slider(piece.piece_type) {
            return 0;
        }
        let directions: &[(i32, i32, i32, i32)] = &[
            (0, 0, 2, 0),
            (0, 0, -2, 0),
            (0, 0, 0, 1),
            (0, 0, 0, -1),
            (1, 0, 2, 0),
            (-1, 0, 2, 0),
            (0, 1, 2, 0),
            (0, -1, 2, 0),
            (1, 0, 0, 1),
            (-1, 0, 0, -1),
        ];
        let mut count = 0;
        for (dx, dy, dt, dl) in directions {
            if stats.attack_budget_exhausted(limits) {
                break;
            }
            if self
                .first_step_on_line(position, *dx, *dy, *dt, *dl)
                .is_some_and(|target| {
                    self.piece_at(target).is_none()
                        && self.attacks_square_with_limits(piece, position, target, limits, stats)
                })
            {
                count += 1;
            }
        }
        count
    }

    pub(crate) fn temporal_pin_score_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.temporal_pin_score_for_with_limits(color, weights, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn temporal_pin_score_for_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        self.temporal_xray_score_for_with_limits(color, weights, true, limits, stats)
    }

    pub(crate) fn temporal_skewer_score_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.temporal_skewer_score_for_with_limits(
            color,
            weights,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn temporal_skewer_score_for_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        self.temporal_xray_score_for_with_limits(color, weights, false, limits, stats)
    }

    pub(crate) fn temporal_xray_score_for(
        &self,
        color: Color,
        weights: &EvalWeights,
        pin_mode: bool,
    ) -> i32 {
        let mut stats = EvaluationStats::default();
        self.temporal_xray_score_for_with_limits(
            color,
            weights,
            pin_mode,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn temporal_xray_score_for_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        pin_mode: bool,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let enemy_royals = self.latest_royal_pieces(color.opposite());
        let mut score = 0;
        for (from, piece) in self.latest_pieces() {
            if stats.attack_budget_exhausted(limits) {
                break;
            }
            if piece.color != color || !self.is_temporal_slider(piece.piece_type) {
                continue;
            }
            for (target, victim) in self.latest_pieces() {
                if stats.attack_budget_exhausted(limits) {
                    break;
                }
                if victim.color != color.opposite() || Self::is_royal_piece(victim.piece_type) {
                    continue;
                }
                if !self.attacks_square_with_limits(piece, from, target, limits, stats) {
                    continue;
                }
                let delta = self.movement_delta(from, target);
                if delta.t == 0 && delta.l == 0 {
                    continue;
                }
                let mut cleared = self.clone_for_search();
                cleared.clear_piece_at(target);
                for (royal_position, royal_piece) in &enemy_royals {
                    if stats.attack_budget_exhausted(limits) {
                        break;
                    }
                    if !cleared.attacks_square_with_limits(
                        piece,
                        from,
                        *royal_position,
                        limits,
                        stats,
                    ) {
                        continue;
                    }
                    if pin_mode
                        || weights.piece_value(royal_piece.piece_type)
                            > weights.piece_value(victim.piece_type)
                    {
                        score += 1;
                    }
                }
            }
        }
        score
    }

    pub(crate) fn causal_battery_score_for(&self, color: Color) -> i32 {
        let mut stats = EvaluationStats::default();
        self.causal_battery_score_for_with_limits(color, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn causal_battery_score_for_with_limits(
        &self,
        color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let enemy_royals = self.latest_royal_pieces(color.opposite());
        let own_pieces = self.latest_pieces();
        let mut score = 0;
        for (front_pos, front_piece) in &own_pieces {
            if stats.attack_budget_exhausted(limits) {
                break;
            }
            if front_piece.color != color || !self.is_temporal_slider(front_piece.piece_type) {
                continue;
            }
            if !enemy_royals.iter().any(|(royal_pos, _)| {
                self.attacks_square_with_limits(*front_piece, *front_pos, *royal_pos, limits, stats)
            }) {
                continue;
            }
            for (rear_pos, rear_piece) in &own_pieces {
                if stats.attack_budget_exhausted(limits) {
                    break;
                }
                if rear_piece.color != color || !self.is_temporal_slider(rear_piece.piece_type) {
                    continue;
                }
                let delta = self.movement_delta(*rear_pos, *front_pos);
                if (delta.t != 0 || delta.l != 0)
                    && self.attacks_square_with_limits(
                        *rear_piece,
                        *rear_pos,
                        *front_pos,
                        limits,
                        stats,
                    )
                {
                    score += 1;
                }
            }
        }
        score
    }

    pub(crate) fn is_temporal_slider(&self, piece_type: PieceType) -> bool {
        matches!(
            piece_type,
            PieceType::Rook
                | PieceType::Bishop
                | PieceType::Unicorn
                | PieceType::Dragon
                | PieceType::Queen
                | PieceType::RoyalQueen
                | PieceType::Princess
        )
    }

    pub(crate) fn piece_temporal_flexibility_score_for(&self, color: Color) -> i32 {
        let mut stats = EvaluationStats::default();
        self.piece_temporal_flexibility_score_for_with_limits(
            color,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn piece_temporal_flexibility_score_for_with_limits(
        &self,
        color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let view = self.latest_position_view();
        let mut score = 0;
        for (position, piece) in &view.pieces {
            if stats.attack_budget_exhausted(limits) {
                break;
            }
            if piece.color != color {
                continue;
            }
            let mut spatial = false;
            let mut temporal = false;
            for target in &view.board_positions {
                if stats.attack_budget_exhausted(limits) {
                    break;
                }
                if !self.attacks_square_with_limits(*piece, *position, *target, limits, stats) {
                    continue;
                }
                let delta = self.movement_delta(*position, *target);
                spatial |= delta.x != 0 || delta.y != 0;
                temporal |= delta.t != 0 || delta.l != 0;
            }
            score += (spatial && temporal) as i32;
        }
        score
    }

    pub(crate) fn dimension_coverage_score_for(&self, color: Color) -> i32 {
        let mut stats = EvaluationStats::default();
        self.dimension_coverage_score_for_with_limits(color, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn dimension_coverage_score_for_with_limits(
        &self,
        color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let view = self.latest_position_view();
        let mut x = 0;
        let mut y = 0;
        let mut t = 0;
        let mut l = 0;
        for (position, piece) in &view.pieces {
            if stats.attack_budget_exhausted(limits) {
                break;
            }
            if piece.color != color {
                continue;
            }
            for target in &view.board_positions {
                if stats.attack_budget_exhausted(limits) {
                    break;
                }
                if !self.attacks_square_with_limits(*piece, *position, *target, limits, stats) {
                    continue;
                }
                let delta = self.movement_delta(*position, *target);
                x += (delta.x != 0) as i32;
                y += (delta.y != 0) as i32;
                t += (delta.t != 0) as i32;
                l += (delta.l != 0) as i32;
            }
        }
        [x, y, t, l].into_iter().min().unwrap_or(0)
    }

    pub(crate) fn past_royal_vulnerability_score_for(&self, color: Color) -> i32 {
        let mut stats = EvaluationStats::default();
        self.past_royal_vulnerability_score_for_with_limits(
            color,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn past_royal_vulnerability_score_for_with_limits(
        &self,
        color: Color,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let mut score = 0;
        for (position, _) in self.royal_pieces(color) {
            if stats.attack_budget_exhausted(limits) {
                break;
            }
            if !self.is_latest_board(position.timeline_id, position.time) {
                score += self
                    .attack_summary_with_limits(position, color.opposite(), limits, stats)
                    .count;
            }
        }
        score
    }

    pub(crate) fn safe_haven_board_score_for(&self, color: Color) -> i32 {
        self.latest_piece_score_sum(|position, piece| {
            if piece.color == color && Self::is_royal_piece(piece.piece_type) {
                let shield = self.royal_shield_count(position, color);
                let escapes = self.royal_escape_count(position, color);
                let active = self.is_active_timeline(position.timeline_id) as i32;
                (shield + escapes + active).saturating_sub(1)
            } else {
                0
            }
        })
    }

    pub(crate) fn royal_distance_score_for(&self, color: Color) -> i32 {
        let enemy_royals = self.latest_royal_pieces(color.opposite());
        self.latest_piece_score_sum(|from, piece| {
            if piece.color == color {
                enemy_royals
                    .iter()
                    .map(|(target, _)| {
                        let distance = tactical_distance(self.movement_delta(from, *target)).max(1);
                        weights_for_tropism(piece.piece_type) / distance
                    })
                    .max()
                    .unwrap_or(0)
            } else {
                0
            }
        })
    }

    pub(crate) fn board_importance_material_score_for(
        &self,
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
        self.timelines
            .iter()
            .flat_map(|timeline| {
                timeline
                    .boards
                    .iter()
                    .map(move |board| (timeline.id, board))
            })
            .map(|(timeline_id, board)| {
                self.board_importance(timeline_id, board)
                    * board
                        .board
                        .iter()
                        .flatten()
                        .filter_map(|piece| *piece)
                        .map(|piece| {
                            if piece.color == color {
                                weights.piece_value(piece.piece_type) / 200
                            } else {
                                -weights.piece_value(piece.piece_type) / 200
                            }
                        })
                        .sum::<i32>()
            })
            .sum()
    }

    pub(crate) fn board_importance(&self, timeline_id: i32, board: &BoardSnapshot) -> i32 {
        let latest = self.is_latest_board(timeline_id, board.time) as i32;
        let active = self.is_active_timeline(timeline_id) as i32;
        let present_distance = self
            .present_time()
            .map(|present| 4 - (board.time - present).abs().min(3))
            .unwrap_or(1);
        let royal_count = board
            .board
            .iter()
            .flatten()
            .filter(|piece| piece.is_some_and(|piece| Self::is_royal_piece(piece.piece_type)))
            .count() as i32;
        latest * 3 + active * 2 + present_distance + royal_count.max(1)
    }

    pub(crate) fn timeline_repetition_risk_score_for(&self, color: Color) -> i32 {
        let owner = TimelineOwner::from_color(color);
        let inactive = self
            .timelines
            .iter()
            .filter(|timeline| timeline.owner == owner && !self.is_active_timeline(timeline.id))
            .count() as i32;
        let mut counts = std::collections::HashMap::new();
        let repeated = self
            .timelines
            .iter()
            .filter(|timeline| timeline.owner == owner)
            .filter_map(|timeline| timeline.boards.last())
            .map(|board| {
                let key = Self::board_repetition_key(board);
                let count = counts.entry(key).or_insert(0);
                *count += 1;
                (*count > 1) as i32
            })
            .sum::<i32>();
        inactive + repeated
    }

    pub(crate) fn development_count_for(&self, color: Color) -> i32 {
        self.latest_piece_score_sum(|position, piece| {
            (piece.color == color && development(color, piece.piece_type, position.y) > 0) as i32
        })
    }
}
