use super::*;

#[allow(dead_code)]
impl Game {
    pub(crate) fn temporal_royal_corridor_balance(
        &self,
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
        let mut stats = EvaluationStats::default();
        self.temporal_royal_corridor_balance_with_limits(
            color,
            weights,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn temporal_royal_corridor_balance_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        self.temporal_royal_corridor_pressure_for_with_limits(color, weights, limits, stats)
            - self.temporal_royal_corridor_pressure_for_with_limits(
                color.opposite(),
                weights,
                limits,
                stats,
            )
    }

    pub(crate) fn temporal_royal_corridor_pressure_for(
        &self,
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
        let mut stats = EvaluationStats::default();
        self.temporal_royal_corridor_pressure_for_with_limits(
            color,
            weights,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn temporal_royal_corridor_pressure_for_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        if weights.royal_capture_setup == 0 {
            return 0;
        }

        let royal_targets = self.royal_pieces(color.opposite());
        let mut score = 0;
        for (from, piece) in self.latest_pieces() {
            if piece.color != color
                || matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
            {
                continue;
            }
            score += self.temporal_royal_corridor_from_with_targets(
                piece,
                from,
                &royal_targets,
                weights,
                limits,
                stats,
            );
        }
        score
    }

    pub(crate) fn temporal_royal_corridor_from_with_targets(
        &self,
        piece: Piece,
        from: Position,
        royal_targets: &[(Position, Piece)],
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let mut score = 0;
        for (target, _) in royal_targets {
            if from.timeline_id == target.timeline_id
                && from.time == target.time
                && from.x == target.x
                && from.y == target.y
            {
                continue;
            }

            for wait in 1..=4 {
                let future_from = Position {
                    time: from.time + wait,
                    ..from
                };
                if future_from.time <= target.time
                    || !self.attacks_square_with_limits(piece, future_from, *target, limits, stats)
                {
                    continue;
                }

                let urgency = 5 - wait;
                let fixed_target_bonus = if self.is_latest_board(target.timeline_id, target.time) {
                    0
                } else {
                    weights.temporal_threat * 2
                };
                let piece_bonus = match piece.piece_type {
                    PieceType::Queen | PieceType::RoyalQueen => weights.royal_capture_setup / 2,
                    PieceType::Bishop
                    | PieceType::Rook
                    | PieceType::Unicorn
                    | PieceType::Dragon
                    | PieceType::Princess => weights.royal_capture_setup / 3,
                    _ => weights.royal_capture_setup / 6,
                };
                score +=
                    weights.royal_capture_setup * urgency / 2 + piece_bonus + fixed_target_bonus;
                break;
            }
        }
        score
    }
}
