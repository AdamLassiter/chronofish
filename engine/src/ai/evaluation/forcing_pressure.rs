impl Game {
    fn forcing_pressure_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.forcing_pressure_for(color, weights) - self.forcing_pressure_for(color.opposite(), weights)
    }

    pub(crate) fn forcing_pressure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (position, piece) in self.latest_pieces() {
            if piece.color != color.opposite() {
                continue;
            }
            let attackers = self.attack_summary(position, color);
            if attackers.count == 0 {
                continue;
            }
            let value = weights.piece_value(piece.piece_type);
            score += attackers.count * weights.forcing_move_pressure + value / 48;
            if Self::is_royal_piece(piece.piece_type) {
                score += weights.royal_threat + attackers.temporal_count * weights.temporal_threat;
            }
            if attackers.timeline_count >= 2 || attackers.time_count >= 2 {
                score += weights.pincer_threat
                    + weights.timeline_pincer * (attackers.timeline_count - 1).max(0)
                    + weights.historical_pincer * (attackers.time_count - 1).max(0);
            }
        }
        score
    }
}
