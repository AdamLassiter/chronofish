use super::*;

impl Game {
    pub(crate) fn royal_capture_pressure(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.royal_capture_pressure_for(color, weights)
            - self.royal_capture_pressure_for(color.opposite(), weights)
    }

    pub(crate) fn royal_capture_pressure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        let royal_targets = self.royal_pieces(color.opposite());
        for (from, piece) in self.latest_pieces() {
            if piece.color != color {
                continue;
            }
            for (target, _) in &royal_targets {
                if self.attacks_square(piece, from, *target) {
                    let distance = tactical_distance(self.movement_delta(from, *target));
                    let urgency = 6_i32.saturating_sub(distance.min(6)).max(1);
                    score += weights.royal_capture_threat * urgency;
                    if from.timeline_id != target.timeline_id || from.time != target.time {
                        score += weights.temporal_threat * urgency;
                    }
                }
            }
        }
        score
    }
}
