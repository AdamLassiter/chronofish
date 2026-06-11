use super::*;

impl Game {
    #[allow(dead_code)]
    pub(crate) fn royal_capture_setup_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.royal_capture_setup_pressure_for(color, weights)
            - self.royal_capture_setup_pressure_for(color.opposite(), weights)
    }

    pub(crate) fn royal_capture_setup_balance_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        self.royal_capture_setup_pressure_bounded(
            color,
            weights,
            limits.setup_results,
            limits.setup_probes,
            stats,
        ) - self.royal_capture_setup_pressure_bounded(
            color.opposite(),
            weights,
            limits.setup_results,
            limits.setup_probes,
            stats,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn royal_capture_setup_pressure_for(
        &self,
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
        self.royal_capture_setup_pressure_for_limited(color, weights, 48)
    }

    pub(crate) fn royal_capture_setup_pressure_for_limited(
        &self,
        color: Color,
        weights: &EvalWeights,
        limit: usize,
    ) -> i32 {
        let mut stats = EvaluationStats::default();
        self.royal_capture_setup_pressure_bounded(color, weights, limit, usize::MAX, &mut stats)
    }

    pub(crate) fn royal_capture_setup_pressure_bounded(
        &self,
        color: Color,
        weights: &EvalWeights,
        result_limit: usize,
        probe_limit: usize,
        stats: &mut EvaluationStats,
    ) -> i32 {
        if weights.royal_capture_setup == 0 || self.royal_capture_available(color) {
            return 0;
        }

        let royal_targets = self.royal_pieces(color.opposite());
        let mut pieces = self.latest_pieces();
        if probe_limit != usize::MAX {
            pieces.sort_by_key(|(position, piece)| {
                (
                    std::cmp::Reverse(weights.piece_value(piece.piece_type)),
                    position.timeline_id,
                    position.time,
                    position.y,
                    position.x,
                )
            });
        }
        let mut score = 0;
        let mut counted = 0;
        let mut probes = 0;
        'pieces: for (from, piece) in pieces {
            if piece.color != color
                || matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
            {
                continue;
            }
            let major_piece_bonus = match piece.piece_type {
                PieceType::Queen | PieceType::RoyalQueen => weights.royal_capture_setup / 2,
                PieceType::Bishop
                | PieceType::Rook
                | PieceType::Unicorn
                | PieceType::Dragon
                | PieceType::Princess => weights.royal_capture_setup / 3,
                _ => 0,
            };

            for y in 0..8 {
                for x in 0..8 {
                    if counted >= result_limit || probes >= probe_limit {
                        break 'pieces;
                    }
                    probes += 1;
                    stats.setup_probes += 1;
                    let to = Position {
                        timeline_id: from.timeline_id,
                        time: from.time,
                        x,
                        y,
                    };
                    let target = self.piece_at(to);
                    if target.is_some_and(|target| target.color == color)
                        || self.move_kind_for(piece, from, to).is_none()
                    {
                        continue;
                    }

                    let arrival = Position {
                        time: from.time + 1,
                        ..to
                    };
                    let corridor_pressure = self.temporal_royal_corridor_from_with_targets(
                        piece,
                        arrival,
                        &royal_targets,
                        weights,
                    );
                    if corridor_pressure <= 0 {
                        continue;
                    }

                    counted += 1;
                    let capture_bonus = target
                        .map(|target| weights.piece_value(target.piece_type) / 20)
                        .unwrap_or(0);
                    score += weights.royal_capture_setup
                        + major_piece_bonus
                        + capture_bonus
                        + corridor_pressure;
                }
            }
        }

        score
    }
}
