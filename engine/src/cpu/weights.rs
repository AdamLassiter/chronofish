use std::cell::RefCell;

use super::*;

#[allow(dead_code)]
impl EvalWeights {
    pub(crate) fn default_tuned() -> Self {
        serde_json::from_str(&active_parameters_json())
            .map(Self::constrained)
            .expect("runtime AI parameters or built-in defaults should be valid JSON")
    }

    pub(crate) fn active_tuned() -> Self {
        ACTIVE_EVAL_WEIGHTS
            .with(|weights| *weights.borrow())
            .unwrap_or_else(Self::default_tuned)
    }

    pub(crate) fn set_active_from_json(json: &str) -> Result<(), String> {
        let weights = serde_json::from_str::<Self>(json)
            .map(Self::constrained)
            .map_err(|error| error.to_string())?;
        ACTIVE_EVAL_WEIGHTS.with(|active| {
            *active.borrow_mut() = Some(weights);
        });
        Ok(())
    }

    pub(crate) fn piece_value(self, piece_type: PieceType) -> i32 {
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

    /// Keep training from learning obviously losing material arithmetic. The
    /// search may still prefer a sacrifice when tactics justify it, but static
    /// positional terms cannot redefine a pawn as more valuable than a rook.
    pub(crate) fn constrained(mut self) -> Self {
        self.king = 20_000;
        self.royal_queen = 20_500;
        self.pawn = self.pawn.clamp(80, 200);
        self.brawn = self.brawn.clamp(80, 240);

        let minor_floor = self.pawn.saturating_mul(2);
        let minor_ceiling = self.pawn.saturating_mul(5);
        self.knight = self.knight.clamp(minor_floor, minor_ceiling);
        self.bishop = self.bishop.clamp(minor_floor, minor_ceiling);
        self.rook = self
            .rook
            .clamp(self.knight.max(self.bishop) + self.pawn, self.pawn * 8);
        self.queen = self
            .queen
            .clamp(self.rook + self.pawn, self.pawn.saturating_mul(14));

        self.common_king = self.common_king.clamp(minor_floor, self.rook);
        self.princess = self
            .princess
            .clamp(self.knight.max(self.bishop), self.queen);
        self.unicorn = self.unicorn.clamp(minor_floor, self.queen);
        self.dragon = self.dragon.clamp(minor_floor, self.queen);

        self.mobility = self.mobility.clamp(0, (self.pawn / 10).max(1));
        self.centrality = self.centrality.clamp(0, (self.pawn / 10).max(1));
        self.advancement = self.advancement.clamp(0, (self.pawn / 5).max(1));
        self.development = self.development.clamp(0, (self.pawn / 4).max(1));
        self.check_penalty = self.check_penalty.clamp(self.pawn, self.queen * 2);
        self
    }
}

thread_local! {
    static ACTIVE_EVAL_WEIGHTS: RefCell<Option<EvalWeights>> = const { RefCell::new(None) };
}

pub(crate) fn active_parameters_json() -> String {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let path = cpu_model_dir().join("parameters.json");
        if let Ok(json) = std::fs::read_to_string(&path) {
            return json;
        }
    }
    "{}".to_string()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn cpu_model_dir() -> std::path::PathBuf {
    if let Some(path) = std::env::var_os("CHRONOFISH_CPU_MODEL_DIR") {
        return path.into();
    }

    let workspace_path = std::path::PathBuf::from("engine/models/cpu-v1");
    if workspace_path.is_dir() {
        workspace_path
    } else {
        std::path::PathBuf::from("models/cpu-v1")
    }
}

#[allow(dead_code)]
pub(crate) fn owner_factor(owner: TimelineOwner, color: Color) -> i32 {
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
pub(crate) fn advancement(color: Color, y: i32) -> i32 {
    match color {
        Color::White => y,
        Color::Black => 7 - y,
    }
}

#[allow(dead_code)]
pub(crate) fn centrality(x: i32, y: i32) -> i32 {
    14 - ((2 * x - 7).abs() + (2 * y - 7).abs())
}

#[allow(dead_code)]
pub(crate) fn tactical_distance(delta: Delta) -> i32 {
    delta
        .x
        .abs()
        .max(delta.y.abs())
        .max(delta.t.abs())
        .max(delta.l.abs())
}

#[allow(dead_code)]
pub(crate) fn development(color: Color, piece_type: PieceType, y: i32) -> i32 {
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
pub(crate) fn position_key(position: Position) -> (i32, i32, i32, i32) {
    (position.timeline_id, position.time, position.y, position.x)
}
