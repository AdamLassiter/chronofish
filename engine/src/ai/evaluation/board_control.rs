use super::*;

#[allow(dead_code)]
impl Game {
    pub(crate) fn board_control_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.board_control_balance_with_limits(color, weights, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn board_control_balance_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let view = self.latest_position_view();
        self.board_control_for_in_view_with_limits(color, weights, &view, limits, stats)
            - self.board_control_for_in_view_with_limits(
                color.opposite(),
                weights,
                &view,
                limits,
                stats,
            )
    }

    pub(crate) fn board_control_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.board_control_for_with_limits(color, weights, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn board_control_for_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let view = self.latest_position_view();
        self.board_control_for_in_view_with_limits(color, weights, &view, limits, stats)
    }

    pub(crate) fn board_control_for_in_view_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        view: &LatestPositionView,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let mut score = 0;
        for (from, piece) in &view.pieces {
            if stats.attack_budget_exhausted(limits) {
                break;
            }
            if piece.color != color {
                continue;
            }
            let mut controlled = 0;
            let mut central = 0;
            let mut royal_zone = 0;
            for target in &view.board_positions {
                if stats.attack_budget_exhausted(limits) {
                    break;
                }
                if !self.attacks_square_with_limits(*piece, *from, *target, limits, stats) {
                    continue;
                }
                controlled += 1;
                central += centrality(target.x, target.y).max(0);
                if self.near_enemy_royal_in_view(*target, color, view) {
                    royal_zone += 1;
                }
            }
            score += controlled * weights.board_control
                + central * weights.board_control / 8
                + royal_zone * weights.royal_threat / 4;
        }
        score
    }
}
