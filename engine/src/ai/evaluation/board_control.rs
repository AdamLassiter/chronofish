use super::*;

impl Game {
    pub(crate) fn board_control_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.board_control_for(color, weights) - self.board_control_for(color.opposite(), weights)
    }

    pub(crate) fn board_control_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (from, piece) in self.latest_pieces() {
            if piece.color != color {
                continue;
            }
            let mut controlled = 0;
            let mut central = 0;
            let mut royal_zone = 0;
            for target in self.latest_board_positions() {
                if !self.attacks_square(piece, from, target) {
                    continue;
                }
                controlled += 1;
                central += centrality(target.x, target.y).max(0);
                if self.near_enemy_royal(target, color) {
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
