#[allow(dead_code)]
impl EvalWeights {
    fn default_tuned() -> Self {
        serde_json::from_str(&active_parameters_json())
            .expect("committed AI parameters should be valid JSON")
    }

    fn active_tuned() -> Self {
        ACTIVE_EVAL_WEIGHTS
            .with(|weights| *weights.borrow())
            .unwrap_or_else(Self::default_tuned)
    }

    fn set_active_from_json(json: &str) -> Result<(), String> {
        let weights: Self = serde_json::from_str(json).map_err(|error| error.to_string())?;
        ACTIVE_EVAL_WEIGHTS.with(|active| {
            *active.borrow_mut() = Some(weights);
        });
        Ok(())
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

thread_local! {
    static ACTIVE_EVAL_WEIGHTS: RefCell<Option<EvalWeights>> = const { RefCell::new(None) };
}

fn active_parameters_json() -> String {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Ok(json) = std::fs::read_to_string("engine/models/cpu-v1/parameters.json") {
            return json;
        }
    }
    include_str!("parameters.json").to_string()
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
