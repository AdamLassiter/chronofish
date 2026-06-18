use super::*;

#[allow(dead_code)]
impl Game {
    pub(crate) fn royal_capture_pressure(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.royal_capture_pressure_with_limits(color, weights, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn royal_capture_pressure_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        self.royal_capture_pressure_for_with_limits(color, weights, limits, stats)
            - self.royal_capture_pressure_for_with_limits(color.opposite(), weights, limits, stats)
    }

    pub(crate) fn royal_capture_pressure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.royal_capture_pressure_for_with_limits(
            color,
            weights,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn royal_capture_pressure_for_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let royal_targets = self.royal_pieces(color.opposite());
        self.latest_piece_score_sum_with_attack_budget(limits, stats, |from, piece, stats| {
            if piece.color != color {
                return 0;
            }
            let mut score = 0;
            for (target, _) in &royal_targets {
                if stats.attack_budget_exhausted(limits) {
                    break;
                }
                if self.attacks_square_with_limits(piece, from, *target, limits, stats) {
                    let distance = tactical_distance(self.movement_delta(from, *target));
                    let urgency = 6_i32.saturating_sub(distance.min(6)).max(1);
                    score += weights.royal_capture_threat * urgency;
                    if from.timeline_id != target.timeline_id || from.time != target.time {
                        score += weights.temporal_threat * urgency;
                    }
                }
            }
            score
        })
    }
}
