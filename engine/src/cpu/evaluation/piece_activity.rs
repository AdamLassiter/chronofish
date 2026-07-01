use super::*;

#[allow(dead_code)]
impl Game {
    pub(crate) fn piece_activity_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.piece_activity_balance_with_limits(color, weights, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn piece_activity_balance_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let view = self.latest_position_view();
        self.piece_activity_for_in_view_with_limits(color, weights, &view, limits, stats)
            - self.piece_activity_for_in_view_with_limits(
                color.opposite(),
                weights,
                &view,
                limits,
                stats,
            )
    }

    #[cfg(test)]
    pub(crate) fn piece_activity_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let view = self.latest_position_view();
        self.piece_activity_for_in_view(color, weights, &view)
    }

    pub(crate) fn piece_activity_for_in_view(
        &self,
        color: Color,
        weights: &EvalWeights,
        view: &LatestPositionView,
    ) -> i32 {
        let mut stats = EvaluationStats::default();
        self.piece_activity_for_in_view_with_limits(
            color,
            weights,
            view,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn piece_activity_for_in_view_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        view: &LatestPositionView,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let mut score = 0;
        for (position, piece) in &view.pieces {
            if piece.color != color || Self::is_royal_piece(piece.piece_type) {
                continue;
            }
            let mobility = self
                .pseudo_attack_count_in_view_with_limits(*position, *piece, view, limits, stats)
                .min(24);
            let activity = match piece.piece_type {
                PieceType::Pawn | PieceType::Brawn => mobility / 2,
                PieceType::Knight => mobility,
                PieceType::Bishop | PieceType::Rook | PieceType::Unicorn | PieceType::Dragon => {
                    mobility
                        + self.open_line_count_with_limits(*position, *piece, limits, stats) * 2
                }
                PieceType::Queen | PieceType::Princess | PieceType::RoyalQueen => mobility + 2,
                PieceType::King | PieceType::CommonKing => 0,
            };
            score += activity * weights.piece_activity;
            if mobility <= 1 {
                score -= weights.piece_activity * 3;
            }
        }
        score
    }
}
