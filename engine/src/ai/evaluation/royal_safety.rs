use super::*;

#[allow(dead_code)]
impl Game {
    pub(crate) fn royal_safety_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.royal_safety_balance_with_limits(color, weights, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn royal_safety_balance_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        self.royal_safety_for_with_limits(color, weights, limits, stats)
            - self.royal_safety_for_with_limits(color.opposite(), weights, limits, stats)
    }

    pub(crate) fn royal_safety_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.royal_safety_for_with_limits(color, weights, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn royal_safety_for_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        self.latest_royal_pieces(color)
            .into_iter()
            .map(|(position, _)| {
                self.individual_royal_safety_with_limits(position, color, weights, limits, stats)
            })
            .sum()
    }

    pub(crate) fn royal_escape_count(&self, position: Position, color: Color) -> i32 {
        let mut search = self.clone_for_search();
        search.turn = color;
        let mut escapes = 0;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let to = Position {
                    timeline_id: position.timeline_id,
                    time: position.time,
                    x: position.x + dx,
                    y: position.y + dy,
                };
                if !Self::in_bounds(to.x, to.y)
                    || search
                        .piece_at(to)
                        .is_some_and(|piece| piece.color == color)
                    || search.is_square_attacked(to, color.opposite())
                {
                    continue;
                }
                if search.legal_move_kind(position, to).is_some() {
                    escapes += 1;
                }
            }
        }
        escapes
    }
}
