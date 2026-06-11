use super::*;

impl Game {
    pub(crate) fn temporal_royal_corridor_balance(
        &self,
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
        self.temporal_royal_corridor_pressure_for(color, weights)
            - self.temporal_royal_corridor_pressure_for(color.opposite(), weights)
    }

    pub(crate) fn temporal_royal_corridor_pressure_for(
        &self,
        color: Color,
        weights: &EvalWeights,
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
            );
        }
        score
    }

    pub(crate) fn temporal_royal_corridor_from(
        &self,
        piece: Piece,
        from: Position,
        weights: &EvalWeights,
    ) -> i32 {
        let royal_targets = self.royal_pieces(piece.color.opposite());
        self.temporal_royal_corridor_from_with_targets(piece, from, &royal_targets, weights)
    }

    pub(crate) fn temporal_royal_corridor_from_with_targets(
        &self,
        piece: Piece,
        from: Position,
        royal_targets: &[(Position, Piece)],
        weights: &EvalWeights,
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
                    || !self.attacks_square(piece, future_from, *target)
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
