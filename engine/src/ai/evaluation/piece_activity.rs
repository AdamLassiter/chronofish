impl Game {
    fn piece_activity_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.piece_activity_for(color, weights) - self.piece_activity_for(color.opposite(), weights)
    }

    pub(crate) fn piece_activity_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (position, piece) in self.latest_pieces() {
            if piece.color != color || Self::is_royal_piece(piece.piece_type) {
                continue;
            }
            let mobility = self.pseudo_attack_count(position, piece).min(24);
            let activity = match piece.piece_type {
                PieceType::Pawn | PieceType::Brawn => mobility / 2,
                PieceType::Knight => mobility,
                PieceType::Bishop | PieceType::Rook | PieceType::Unicorn | PieceType::Dragon => {
                    mobility + self.open_line_count(position, piece) * 2
                }
                PieceType::Queen | PieceType::Princess | PieceType::RoyalQueen => mobility + 2,
                PieceType::King | PieceType::CommonKing => 0,
            };
            score += activity * weights.piece_activity;
            if mobility <= 1 {
                score -= weights.piece_activity * 3;
            }
        }
        score
    }
}
