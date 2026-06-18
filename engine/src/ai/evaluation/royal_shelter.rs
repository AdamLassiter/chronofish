use super::*;

impl Game {
    pub(crate) fn royal_shelter_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.royal_shelter_for(color, weights) - self.royal_shelter_for(color.opposite(), weights)
    }

    pub(crate) fn royal_shelter_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.latest_piece_score_sum(|position, piece| {
            if piece.color != color || !Self::is_royal_piece(piece.piece_type) {
                return 0;
            }
            let shield_count = self.royal_shield_count(position, color);
            shield_count * weights.royal_shelter - (3 - shield_count) * (weights.royal_shelter / 2)
        })
    }
}
