use super::*;

impl Game {
    pub(crate) fn fork_pressure_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.fork_pressure_for(color, weights) - self.fork_pressure_for(color.opposite(), weights)
    }

    pub(crate) fn fork_pressure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let enemies: Vec<(Position, Piece)> = self
            .latest_pieces()
            .into_iter()
            .filter(|(_, piece)| piece.color == color.opposite())
            .collect();
        let mut score = 0;
        for (from, piece) in self.latest_pieces() {
            if piece.color != color {
                continue;
            }
            let mut threatened = 0;
            let mut value_sum = 0;
            let mut royal = false;
            for (target, enemy) in &enemies {
                if !self.attacks_square(piece, from, *target) {
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
