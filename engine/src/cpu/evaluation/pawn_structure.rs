use super::*;

impl Game {
    pub(crate) fn pawn_structure_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.pawn_structure_for(color, weights) - self.pawn_structure_for(color.opposite(), weights)
    }

    pub(crate) fn pawn_structure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        let forward = if color == Color::White { 1 } else { -1 };
        for timeline in &self.timelines {
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            for y in 0..8 {
                for x in 0..8 {
                    let Some(piece) = board.board[y][x] else {
                        continue;
                    };
                    if piece.color != color
                        || !matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
                    {
                        continue;
                    }
                    let position = Position {
                        timeline_id: timeline.id,
                        time: board.time,
                        x: x as i32,
                        y: y as i32,
                    };
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
                    let ahead_y = position.y + forward;
                    if Self::in_bounds(position.x, ahead_y)
                        && board.board[ahead_y as usize][x].is_some()
                    {
                        score -= weights.pawn_structure;
                    }
                }
            }
        }
        score
    }
}
