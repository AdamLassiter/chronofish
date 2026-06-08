impl Game {
    fn pawn_structure_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.pawn_structure_for(color, weights) - self.pawn_structure_for(color.opposite(), weights)
    }

    pub(crate) fn pawn_structure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (position, piece) in self.latest_pieces() {
            if piece.color != color || !matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
            {
                continue;
            }
            let forward = if color == Color::White { 1 } else { -1 };
            let advance = advancement(color, position.y);
            score += advance * weights.space_advantage;
            if self.is_passed_pawn(position, color) {
                score += weights.pawn_structure * (advance + 1);
            }
            if self.is_supported_pawn(position, color) {
                score += weights.pawn_structure;
            }
            if self.is_isolated_pawn(position, color) {
                score -= weights.pawn_structure;
            }
            let ahead = Position {
                timeline_id: position.timeline_id,
                time: position.time,
                x: position.x,
                y: position.y + forward,
            };
            if Self::in_bounds(ahead.x, ahead.y) && self.piece_at(ahead).is_some() {
                score -= weights.pawn_structure;
            }
        }
        score
    }
}
