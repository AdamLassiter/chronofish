#[allow(dead_code)]
impl EvalWeights {
    fn default_tuned() -> Self {
        // Committed training data lives in a dedicated JSON include target so
        // the trainer never edits this type definition.
        serde_json::from_str(include_str!("parameters.json"))
            .expect("committed AI parameters should be valid JSON")
    }

    fn piece_value(self, piece_type: PieceType) -> i32 {
        match piece_type {
            PieceType::King => self.king,
            PieceType::CommonKing => self.common_king,
            PieceType::Queen => self.queen,
            PieceType::RoyalQueen => self.royal_queen,
            PieceType::Princess => self.princess,
            PieceType::Rook => self.rook,
            PieceType::Bishop => self.bishop,
            PieceType::Unicorn => self.unicorn,
            PieceType::Dragon => self.dragon,
            PieceType::Knight => self.knight,
            PieceType::Pawn => self.pawn,
            PieceType::Brawn => self.brawn,
        }
    }
}

#[allow(dead_code)]
fn owner_factor(owner: TimelineOwner, color: Color) -> i32 {
    match owner {
        TimelineOwner::Neutral => 0,
        TimelineOwner::White => {
            if color == Color::White {
                1
            } else {
                -1
            }
        }
        TimelineOwner::Black => {
            if color == Color::Black {
                1
            } else {
                -1
            }
        }
    }
}

#[allow(dead_code)]
fn advancement(color: Color, y: i32) -> i32 {
    match color {
        Color::White => y,
        Color::Black => 7 - y,
    }
}

#[allow(dead_code)]
fn centrality(x: i32, y: i32) -> i32 {
    14 - ((2 * x - 7).abs() + (2 * y - 7).abs())
}

#[allow(dead_code)]
fn tactical_distance(delta: Delta) -> i32 {
    delta.x
        .abs()
        .max(delta.y.abs())
        .max(delta.t.abs())
        .max(delta.l.abs())
}

#[allow(dead_code)]
fn development(color: Color, piece_type: PieceType, y: i32) -> i32 {
    if matches!(
        piece_type,
        PieceType::Pawn | PieceType::Brawn | PieceType::King | PieceType::RoyalQueen
    ) {
        return 0;
    }
    match color {
        Color::White => (y > 0) as i32,
        Color::Black => (y < 7) as i32,
    }
}

#[allow(dead_code)]
fn position_key(position: Position) -> (i32, i32, i32, i32) {
    (position.timeline_id, position.time, position.y, position.x)
}
