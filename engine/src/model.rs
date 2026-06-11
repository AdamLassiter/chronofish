// Core value types for the engine. These stay small and Clone-heavy because the
// rules, AI search, and training harness all explore speculative states by
// copying board snapshots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Color {
    White,
    Black,
}

// Variant pieces are modeled now even though the default setup still places only
// orthodox chess pieces. That keeps notation, legality, AI scoring, and tests on
// one shared representation before variant setup is exposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum PieceType {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Piece {
    pub(crate) color: Color,
    pub(crate) piece_type: PieceType,
}

// Board snapshots are append-only history. Moves create a later snapshot rather
// than mutating the old one, which is what makes historical time-travel targets
// available.
#[derive(Clone)]
pub(crate) struct BoardSnapshot {
    pub(crate) time: i32,
    pub(crate) side_to_move: Color,
    pub(crate) board: [[Option<Piece>; 8]; 8],
    pub(crate) castling: CastlingRights,
    pub(crate) en_passant: Option<EnPassant>,
    pub(crate) origin: Origin,
}

// Render/debug metadata explaining how a snapshot came to exist. The rules never
// depend on Origin.
#[derive(Clone)]
pub(crate) enum Origin {
    None,
    #[allow(dead_code)]
    Move {
        from: Position,
        to: Position,
        move_type: &'static str,
    },
}

// Timeline ids carry notation and ownership meaning: T0 is neutral, white-made
// timelines count upward, and black-made timelines count downward. row is the
// geometric L-axis used by movement and rendering.
#[derive(Clone)]
pub(crate) struct Timeline {
    pub(crate) id: i32,
    pub(crate) row: i32,
    #[allow(dead_code)]
    pub(crate) label: String,
    pub(crate) owner: TimelineOwner,
    pub(crate) boards: Vec<BoardSnapshot>,
}

// Owned timelines can become inactive when one side has branched more than the
// opponent can answer. Neutral T0 always remains active.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimelineOwner {
    Neutral,
    White,
    Black,
}

// A position identifies one square on one board on one timeline. The frontend
// serializes the same logical shape as timelineId/time/x/y.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Position {
    pub(crate) timeline_id: i32,
    pub(crate) time: i32,
    pub(crate) x: i32,
    pub(crate) y: i32,
}

// Game is the authoritative engine state. staged_turn holds undo checkpoints for
// the current unsubmitted turn; the frontend separately tracks submitted turns
// for replay and multiplayer sync.
#[derive(Clone)]
pub(crate) struct Game {
    pub(crate) turn: Color,
    pub(crate) timelines: Vec<Timeline>,
    pub(crate) next_timeline_id: i32,
    pub(crate) next_black_timeline_id: i32,
    pub(crate) staged_turn: Vec<GameCheckpoint>,
    pub(crate) staged_notation: Vec<String>,
    pub(crate) staged_royal_capture_by: Option<Color>,
    pub(crate) last_message: String,
    pub(crate) position_hash: u64,
}

// Whole-state checkpoints are simpler and safer than trying to reverse 5D moves.
// Turns are short enough that copying the visible game state is acceptable.
#[derive(Clone)]
#[allow(dead_code)]
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
    pub(crate) board_lengths: Vec<(i32, usize)>,
    pub(crate) next_timeline_id: i32,
    pub(crate) next_black_timeline_id: i32,
    pub(crate) staged_royal_capture_by: Option<Color>,
    pub(crate) position_hash: u64,
}

// Castling rights belong to a snapshot and are carried forward on each new board.
// Moving a king/rook, or capturing a rook on its home square, clears rights.
#[derive(Clone, Copy)]
pub(crate) struct CastlingRights {
    pub(crate) white_kingside: bool,
    pub(crate) white_queenside: bool,
    pub(crate) black_kingside: bool,
    pub(crate) black_queenside: bool,
}

// En-passant is snapshot-local because the capture is legal only on the next
// viable board for that side.
#[derive(Clone, Copy)]
pub(crate) struct EnPassant {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) captured_x: i32,
    pub(crate) captured_y: i32,
}

// Movement delta across file, rank, time, and timeline row.
#[derive(Clone, Copy)]
pub(crate) struct Delta {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) t: i32,
    pub(crate) l: i32,
}

// Side effects that from/to alone cannot describe.
#[derive(Clone, Copy)]
pub(crate) enum MoveKind {
    Standard,
    Branch,
    Castle { rook_from_x: i32, rook_to_x: i32 },
    EnPassant { captured_x: i32, captured_y: i32 },
}
