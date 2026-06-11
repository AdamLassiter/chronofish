use super::*;

impl Game {
    pub(crate) fn space_advantage_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.space_advantage_for(color, weights)
            - self.space_advantage_for(color.opposite(), weights)
    }

    pub(crate) fn space_advantage_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.latest_pieces()
            .into_iter()
            .filter(|(_, piece)| piece.color == color)
            .map(|(position, piece)| {
                let value_scale = if matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
                {
                    2
                } else {
                    1
                };
                advancement(color, position.y) * weights.space_advantage * value_scale
            })
            .sum()
    }
}
