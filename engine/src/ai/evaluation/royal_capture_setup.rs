impl Game {
    fn royal_capture_setup_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.royal_capture_setup_pressure_for(color, weights)
            - self.royal_capture_setup_pressure_for(color.opposite(), weights)
    }

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
        if weights.royal_capture_setup == 0 || self.royal_capture_available(color) {
            return 0;
        }

        let mut score = 0;
        let mut counted = 0;
        for (from, piece) in self.latest_pieces() {
            if piece.color != color || matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn) {
                continue;
            }

            for y in 0..8 {
                for x in 0..8 {
                    if counted >= limit {
                        break;
                    }
                    let to = Position {
                        timeline_id: from.timeline_id,
                        time: from.time,
                        x,
                        y,
                    };
                    if self.piece_at(to).is_some_and(|target| target.color == color)
                        || self.move_kind_for(piece, from, to).is_none()
                    {
                        continue;
                    }

                    let arrival = Position {
                        time: from.time + 1,
                        ..to
                    };
                    let corridor_pressure = self.temporal_royal_corridor_from(piece, arrival, weights);
                    if corridor_pressure <= 0 {
                        continue;
                    }

                    counted += 1;
                    let major_piece_bonus = match piece.piece_type {
                        PieceType::Queen | PieceType::RoyalQueen => weights.royal_capture_setup / 2,
                        PieceType::Bishop
                        | PieceType::Rook
                        | PieceType::Unicorn
                        | PieceType::Dragon
                        | PieceType::Princess => weights.royal_capture_setup / 3,
                        _ => 0,
                    };
                    let capture_bonus = self
                        .piece_at(to)
                        .map(|target| weights.piece_value(target.piece_type) / 20)
                        .unwrap_or(0);
                    score +=
                        weights.royal_capture_setup + major_piece_bonus + capture_bonus + corridor_pressure;
                }
            }
            if counted >= limit {
                break;
            }
        }

        score
    }
}
