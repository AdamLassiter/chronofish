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
        self.pruned_for_evaluation()
            .evaluate_heuristic_without_pruning(color, weights)
    }

    pub(crate) fn evaluate_heuristic_without_pruning(
        &self,
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
        if let Some(score) = self.terminal_score(color) {
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

            let Some(board) = timeline.boards.iter().max_by_key(|board| board.time) else {
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
            + self.extended_multiverse_balance(color, weights)
            + self.present_progress(color) * weights.present_progress
            + self.strategic_balance(color, weights)
            + self.timeline_coordination(color, weights)
            + self.royal_capture_pressure(color, weights)
            + self.temporal_royal_corridor_balance(color, weights)
            + self.royal_capture_setup_balance(color, weights)
            + self.royal_safety_balance(color, weights)
            + self.fork_pressure_balance(color, weights)
            + self.forcing_pressure_balance(color, weights)
            + self.board_control_balance(color, weights)
            + self.piece_activity_balance(color, weights)
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
        if self.staged_royal_capture_by == Some(color) {
            Some(CHECKMATE_SCORE)
        } else if self.staged_royal_capture_by == Some(color.opposite()) {
            Some(-CHECKMATE_SCORE)
        } else if self.has_threefold_repetition() || self.is_classic_stalemate(self.turn) {
            Some(0)
        } else {
            None
        }
    }
}
