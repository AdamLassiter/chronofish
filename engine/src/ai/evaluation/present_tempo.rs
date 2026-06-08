impl Game {
    fn present_tempo_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        let Some(present) = self.present_board() else {
            return 0;
        };
        let factor = if present.side_to_move == color { 1 } else { -1 };
        let spread = self.timeline_time_spread();
        factor * (weights.present_tempo * (spread + 1))
    }
}
