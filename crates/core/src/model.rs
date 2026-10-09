use std::fmt;

use serde::{Deserialize, Serialize};

pub const BOARD_SIZE: usize = 8;
pub const BOARD_SIZE_I32: i32 = 8;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum Ruleset {
    Standard,
    #[default]
    Multiverse,
    MultiverseVariant,
}

impl fmt::Display for Ruleset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Standard => "Standard Chess",
            Self::Multiverse => "5D Chess",
            Self::MultiverseVariant => "5D Variant Chess",
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Color {
    White,
    Black,
}

pub type Side = Color;

impl Color {
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    #[must_use]
    pub const fn timeline_direction(self) -> i32 {
        match self {
            Self::White => 1,
            Self::Black => -1,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::White => "White",
            Self::Black => "Black",
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub enum PieceType {
    King,
    CommonKing,
    Queen,
    RoyalQueen,
    Princess,
    Rook,
    Bishop,
    Unicorn,
    Dragon,
    Knight,
    Pawn,
    Brawn,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Piece {
    pub color: Color,
    #[serde(rename = "type")]
    pub piece_type: PieceType,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Origin {
    None,
    Move {
        from: Position,
        to: Position,
        move_type: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct BoardSnapshot {
    pub time: i32,
    pub side_to_move: Color,
    pub board: [[Option<Piece>; BOARD_SIZE]; BOARD_SIZE],
    pub castling: CastlingRights,
    pub en_passant: Option<EnPassant>,
    pub origin: Origin,
    #[serde(default)]
    pub halfmove_clock: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Timeline {
    pub id: i32,
    pub row: i32,
    pub label: String,
    pub owner: TimelineOwner,
    pub boards: Vec<BoardSnapshot>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum TimelineOwner {
    Neutral,
    White,
    Black,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub struct Position {
    pub timeline_id: i32,
    pub time: i32,
    pub x: i32,
    pub y: i32,
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = char::from(b'a' + u8::try_from(self.x).unwrap_or_default());
        write!(
            f,
            "T{}L{}:{file}{}",
            self.time,
            self.timeline_id,
            self.y + 1
        )
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct Move {
    pub from: Position,
    pub to: Position,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promotion: Option<PieceType>,
}

impl Move {
    #[must_use]
    pub const fn new(from: Position, to: Position) -> Self {
        Self {
            from,
            to,
            promotion: None,
        }
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new(Ruleset::default())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PlayerAction {
    Move { movement: Move },
    SubmitTurn,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MoveError {
    #[error("the game is already over")]
    GameOver,
    #[error("illegal move")]
    IllegalMove,
    #[error("a promotion choice is required")]
    PromotionRequired,
    #[error("that promotion piece is not allowed")]
    InvalidPromotion,
    #[error("the turn cannot be submitted yet")]
    CannotSubmit,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum OutcomeReason {
    Checkmate,
    Stalemate,
    RoyalCapture,
    ThreefoldRepetition,
    FiftyMoveRule,
    InsufficientMaterial,
    NoLegalTurn,
    Concession,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct GameOutcome {
    pub winner: Option<Color>,
    pub reason: OutcomeReason,
}

pub type GameResult = GameOutcome;
pub type GameResultReason = OutcomeReason;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Game {
    pub(crate) ruleset: Ruleset,
    pub(crate) turn: Color,
    pub(crate) timelines: Vec<Timeline>,
    pub(crate) next_timeline_id: i32,
    pub(crate) next_black_timeline_id: i32,
    pub(crate) staged_turn: Vec<GameCheckpoint>,
    pub(crate) staged_notation: Vec<String>,
    pub(crate) staged_royal_capture_by: Option<Color>,
    pub(crate) result: Option<GameResult>,
    pub(crate) last_message: String,
    pub(crate) position_hash: u64,
    #[serde(default)]
    pub(crate) repetition_history: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct GameCheckpoint {
    pub(crate) turn: Color,
    pub(crate) timelines: Vec<Timeline>,
    pub(crate) next_timeline_id: i32,
    pub(crate) next_black_timeline_id: i32,
    pub(crate) staged_notation: Vec<String>,
    pub(crate) staged_royal_capture_by: Option<Color>,
    pub(crate) last_message: String,
    pub(crate) position_hash: u64,
}

pub(crate) struct SearchUndo {
    pub(crate) timeline_count: usize,
    pub(crate) source_board_len: (i32, usize),
    pub(crate) target_board_len: Option<(i32, usize)>,
    pub(crate) next_timeline_id: i32,
    pub(crate) next_black_timeline_id: i32,
    pub(crate) staged_royal_capture_by: Option<Color>,
    pub(crate) position_hash: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct CastlingRights {
    pub white_kingside: bool,
    pub white_queenside: bool,
    pub black_kingside: bool,
    pub black_queenside: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase")]
pub struct EnPassant {
    pub x: i32,
    pub y: i32,
    pub captured_x: i32,
    pub captured_y: i32,
}

#[derive(Clone, Copy)]
pub(crate) struct Delta {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) t: i32,
    pub(crate) l: i32,
}

#[derive(Clone, Copy)]
pub(crate) enum MoveKind {
    Standard,
    Branch,
    Castle { rook_from_x: i32, rook_to_x: i32 },
    EnPassant { captured_x: i32, captured_y: i32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MoveStep {
    pub(crate) from: Position,
    pub(crate) to: Position,
}

pub(crate) type SearchInstant = std::time::Instant;
pub(crate) fn deadline_expired(deadline: Option<SearchInstant>) -> bool {
    deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline)
}
