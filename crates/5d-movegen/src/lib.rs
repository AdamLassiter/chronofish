//! Authoritative standard and multiverse chess rules.

// The rules kernel intentionally uses compact integer coordinates, fixed-size
// boards, and explicit move-generation loops. Pedantic style lints obscure the
// invariants in that code, so they are scoped off here rather than weakening
// linting for the rest of the workspace.
#![allow(clippy::all, clippy::pedantic, dead_code)]

mod game;
mod hash;
mod model;
mod move_effects;
mod movegen;
mod movegen_piece;
mod notation;
mod reference;
mod view;

pub use view::GameView;

pub use model::{
    BOARD_SIZE, BoardSnapshot, CastlingRights, Color, EnPassant, Game, GameOutcome, Move,
    MoveDescription, MoveError, Origin, OutcomeReason, Piece, PieceType, PlayerAction, Position,
    Ruleset, Side, Timeline, TimelineOwner,
};
pub use reference::AiDecision;

pub(crate) use model::*;

pub use reference::{ReferenceBotTurn, reference_setup};
