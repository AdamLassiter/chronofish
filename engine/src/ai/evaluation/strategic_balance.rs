use super::*;

#[allow(dead_code)]
impl Game {
    pub(crate) fn strategic_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.strategic_balance_with_limits(color, weights, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn strategic_balance_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let mut score = 0;
        for (position, piece) in self.latest_pieces() {
            let attackers =
                self.attack_summary_with_limits(position, piece.color.opposite(), limits, stats);
            let defenders = self.attack_summary_with_limits(position, piece.color, limits, stats);
            let value = weights.piece_value(piece.piece_type);
            let mut piece_score = 0;

            if defenders.count > 0 {
                piece_score += weights.defended_piece;
            }
            if attackers.count > 0 {
                piece_score -= weights.attacked_piece + value / 32;
            }
            if attackers.count > 0 && defenders.count == 0 {
                piece_score -= weights.hanging_piece + value / 16;
            }
            if attackers.count > 0 && Self::is_royal_piece(piece.piece_type) {
                piece_score -= weights.royal_threat;
            }
            if attackers.temporal_count > 0 {
                piece_score -= weights.temporal_threat * attackers.temporal_count;
            }
            if attackers.count >= 2 {
                piece_score -= weights.pincer_threat * (attackers.count - 1);
            }
            if attackers.timeline_count >= 2 {
                piece_score -= weights.timeline_pincer * (attackers.timeline_count - 1);
            }
            if attackers.time_count >= 2 {
                piece_score -= weights.historical_pincer * (attackers.time_count - 1);
            }

            score += if piece.color == color {
                piece_score
            } else {
                -piece_score
            };
        }
        score
    }
}
