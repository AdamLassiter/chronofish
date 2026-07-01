use super::*;

#[derive(Default)]
pub(crate) struct TurnFeatureSummary {
    pub(crate) obligations: i32,
    pub(crate) completion_count: i32,
    pub(crate) safe_move_count: i32,
    pub(crate) zugzwang_boards: i32,
    pub(crate) branch_payload: i32,
    pub(crate) branch_waste: i32,
    pub(crate) safe_arrivals: i32,
    pub(crate) source_abandonment: i32,
    pub(crate) escape_branches: i32,
    pub(crate) mate_nets: i32,
    pub(crate) anti_mate_resources: i32,
    pub(crate) check_quality: i32,
    pub(crate) volatility: i32,
    pub(crate) promotion_choices: i32,
    pub(crate) promotion_checks: i32,
    pub(crate) branch_moves: i32,
    pub(crate) checking_moves: i32,
}

#[allow(dead_code)]
impl Game {
    pub(crate) fn evaluate(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.evaluate_heuristic(color, weights)
    }

    pub(crate) fn evaluate_heuristic(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.evaluate_heuristic_with_limits(color, weights, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn evaluate_heuristic_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        stats.calls += 1;
        if self
            .timelines
            .iter()
            .all(|timeline| self.is_active_timeline(timeline.id))
        {
            self.evaluate_heuristic_without_pruning_with_limits(color, weights, limits, stats)
        } else {
            stats.clones += 1;
            self.pruned_for_evaluation()
                .evaluate_heuristic_without_pruning_with_limits(color, weights, limits, stats)
        }
    }

    pub(crate) fn evaluate_heuristic_for_nodes(
        &self,
        color: Color,
        weights: &EvalWeights,
        max_nodes: usize,
    ) -> i32 {
        self.evaluate_heuristic_for_nodes_until(color, weights, max_nodes, None)
    }

    pub(crate) fn evaluate_heuristic_for_nodes_until(
        &self,
        color: Color,
        weights: &EvalWeights,
        max_nodes: usize,
        deadline: Option<SearchInstant>,
    ) -> i32 {
        let mut stats = EvaluationStats::default();
        self.evaluate_heuristic_with_limits(
            color,
            weights,
            EvaluationLimits::for_nodes(max_nodes).with_deadline(deadline),
            &mut stats,
        )
    }

    pub(crate) fn evaluate_heuristic_without_pruning(
        &self,
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
        let mut stats = EvaluationStats::default();
        self.evaluate_heuristic_without_pruning_with_limits(
            color,
            weights,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn evaluate_heuristic_without_pruning_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        if let Some(score) = self.terminal_score_until(color, limits.deadline) {
            return score;
        }

        let mut score = 0;
        for timeline in &self.timelines {
            let active = self.is_active_timeline(timeline.id);
            score += if active {
                weights.active_timeline
            } else {
                weights.inactive_timeline
            } * owner_factor(timeline.owner, color);

            let Some(board) = timeline.boards.last() else {
                continue;
            };
            for (y, rank) in board.board.iter().enumerate() {
                for (x, piece) in rank.iter().enumerate() {
                    let Some(piece) = piece else {
                        continue;
                    };
                    let value = weights.piece_value(piece.piece_type);
                    let positional = weights.advancement * advancement(piece.color, y as i32)
                        + weights.centrality * centrality(x as i32, y as i32);
                    let development =
                        weights.development * development(piece.color, piece.piece_type, y as i32);
                    score += if piece.color == color {
                        value + positional + development
                    } else {
                        -value - positional - development
                    };
                }
            }
        }

        if self.is_in_check(color) {
            score -= weights.check_penalty;
        }
        if self.is_in_check(color.opposite()) {
            score += weights.check_penalty;
        }
        score
            + self.extended_multiverse_balance_with_limits(color, weights, limits, stats)
            + self.present_progress(color) * weights.present_progress
            + self.strategic_balance_with_limits(color, weights, limits, stats)
            + self.timeline_coordination(color, weights)
            + self.royal_capture_pressure_with_limits(color, weights, limits, stats)
            + self.temporal_royal_corridor_balance_with_limits(color, weights, limits, stats)
            + self.royal_capture_setup_balance_with_limits(color, weights, limits, stats)
            + self.royal_safety_balance_with_limits(color, weights, limits, stats)
            + self.fork_pressure_balance_with_limits(color, weights, limits, stats)
            + self.forcing_pressure_balance_with_limits(color, weights, limits, stats)
            + self.board_control_balance_with_limits(color, weights, limits, stats)
            + self.piece_activity_balance_with_limits(color, weights, limits, stats)
            + self.pawn_structure_balance(color, weights)
            + self.timeline_economy_balance(color, weights)
            + self.present_tempo_balance(color, weights)
            + self.royal_shelter_balance(color, weights)
            + self.space_advantage_balance(color, weights)
            + if weights.mobility == 0 {
                0
            } else {
                self.mobility_balance(color) * weights.mobility
            }
    }

    pub(crate) fn terminal_score(&self, color: Color) -> Option<i32> {
        self.terminal_score_until(color, None)
    }

    pub(crate) fn terminal_score_until(
        &self,
        color: Color,
        deadline: Option<SearchInstant>,
    ) -> Option<i32> {
        if let Some(result) = self.result {
            return match result.winner {
                Some(winner) if winner == color => Some(CHECKMATE_SCORE),
                Some(_) => Some(-CHECKMATE_SCORE),
                None => Some(0),
            };
        }
        if self.staged_royal_capture_by == Some(color) {
            Some(CHECKMATE_SCORE)
        } else if self.staged_royal_capture_by == Some(color.opposite()) {
            Some(-CHECKMATE_SCORE)
        } else if self.has_threefold_repetition()
            || (!deadline_expired(deadline) && self.is_classic_stalemate_until(self.turn, deadline))
        {
            Some(0)
        } else {
            None
        }
    }
}
