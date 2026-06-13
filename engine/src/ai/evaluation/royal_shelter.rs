use super::*;

impl Game {
    pub(crate) fn royal_shelter_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.royal_shelter_for(color, weights) - self.royal_shelter_for(color.opposite(), weights)
    }

    pub(crate) fn royal_shelter_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (position, _) in self.latest_royal_pieces(color) {
            let shield_count = self.royal_shield_count(position, color);
            score += shield_count * weights.royal_shelter;
            score -= (3 - shield_count) * (weights.royal_shelter / 2);
        }
        score
    }
}
