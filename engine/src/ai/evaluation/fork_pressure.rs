use super::*;

#[allow(dead_code)]
impl Game {
    pub(crate) fn fork_pressure_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.fork_pressure_balance_with_limits(color, weights, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn fork_pressure_balance_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let view = self.latest_position_view();
        self.fork_pressure_for_in_view_with_limits(color, weights, &view, limits, stats)
            - self.fork_pressure_for_in_view_with_limits(
                color.opposite(),
                weights,
                &view,
                limits,
                stats,
            )
    }

    #[cfg(test)]
    pub(crate) fn fork_pressure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let view = self.latest_position_view();
        self.fork_pressure_for_in_view(color, weights, &view)
    }

    pub(crate) fn fork_pressure_for_in_view(
        &self,
        color: Color,
        weights: &EvalWeights,
        view: &LatestPositionView,
    ) -> i32 {
        let mut stats = EvaluationStats::default();
        self.fork_pressure_for_in_view_with_limits(
            color,
            weights,
            view,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn fork_pressure_for_in_view_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        view: &LatestPositionView,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let enemies = view
            .pieces
            .iter()
            .filter(|(_, piece)| piece.color == color.opposite());
        let mut score = 0;
        for (from, piece) in &view.pieces {
            if piece.color != color {
                continue;
            }
            let mut threatened = 0;
            let mut value_sum = 0;
            let mut royal = false;
            for (target, enemy) in enemies.clone() {
                if !self.attacks_square_with_limits(*piece, *from, *target, limits, stats) {
                    continue;
                }
                threatened += 1;
                value_sum += weights.piece_value(enemy.piece_type);
                royal |= Self::is_royal_piece(enemy.piece_type);
            }
            if threatened >= 2 {
                score += weights.fork_pressure * (threatened - 1) + value_sum / 24;
                if royal {
                    score += weights.royal_threat;
                }
            }
        }
        score
    }
}
