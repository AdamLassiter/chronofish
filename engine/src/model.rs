// Core value types for the engine. These stay small and Clone-heavy because the
// rules, AI search, and training harness all explore speculative states by
// copying board snapshots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Color {
    White,
    Black,
}

// Variant pieces are modeled now even though the default setup still places only
// orthodox chess pieces. That keeps notation, legality, AI scoring, and tests on
// one shared representation before variant setup is exposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum PieceType {
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
struct Piece {
    color: Color,
    piece_type: PieceType,
}

// Board snapshots are append-only history. Moves create a later snapshot rather
// than mutating the old one, which is what makes historical time-travel targets
// available.
#[derive(Clone)]
struct BoardSnapshot {
    time: i32,
    side_to_move: Color,
    board: [[Option<Piece>; 8]; 8],
    castling: CastlingRights,
    en_passant: Option<EnPassant>,
    origin: Origin,
}

// Render/debug metadata explaining how a snapshot came to exist. The rules never
// depend on Origin.
#[derive(Clone)]
enum Origin {
    None,
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
struct Timeline {
    id: i32,
    row: i32,
    label: String,
    owner: TimelineOwner,
    boards: Vec<BoardSnapshot>,
}

// Owned timelines can become inactive when one side has branched more than the
// opponent can answer. Neutral T0 always remains active.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TimelineOwner {
    Neutral,
    White,
    Black,
}

// A position identifies one square on one board on one timeline. The frontend
// serializes the same logical shape as timelineId/time/x/y.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Position {
    timeline_id: i32,
    time: i32,
    x: i32,
    y: i32,
}

// Game is the authoritative engine state. staged_turn holds undo checkpoints for
// the current unsubmitted turn; the frontend separately tracks submitted turns
// for replay and multiplayer sync.
#[derive(Clone)]
struct Game {
    turn: Color,
    timelines: Vec<Timeline>,
    next_timeline_id: i32,
    next_black_timeline_id: i32,
    staged_turn: Vec<GameCheckpoint>,
    staged_notation: Vec<String>,
    last_message: String,
}

// Whole-state checkpoints are simpler and safer than trying to reverse 5D moves.
// Turns are short enough that copying the visible game state is acceptable.
#[derive(Clone)]
struct GameCheckpoint {
    turn: Color,
    timelines: Vec<Timeline>,
    next_timeline_id: i32,
    next_black_timeline_id: i32,
    staged_notation: Vec<String>,
    last_message: String,
}

// Castling rights belong to a snapshot and are carried forward on each new board.
// Moving a king/rook, or capturing a rook on its home square, clears rights.
#[derive(Clone, Copy)]
struct CastlingRights {
    white_kingside: bool,
    white_queenside: bool,
    black_kingside: bool,
    black_queenside: bool,
}

// En-passant is snapshot-local because the capture is legal only on the next
// viable board for that side.
#[derive(Clone, Copy)]
struct EnPassant {
    x: i32,
    y: i32,
    captured_x: i32,
    captured_y: i32,
}

// Movement delta across file, rank, time, and timeline row.
#[derive(Clone, Copy)]
struct Delta {
    x: i32,
    y: i32,
    t: i32,
    l: i32,
}

// Side effects that from/to alone cannot describe.
#[derive(Clone, Copy)]
enum MoveKind {
    Standard,
    Branch,
    Castle { rook_from_x: i32, rook_to_x: i32 },
    EnPassant { captured_x: i32, captured_y: i32 },
}
