#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Color {
    White,
    Black,
}

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

#[derive(Clone)]
struct BoardSnapshot {
    time: i32,
    side_to_move: Color,
    board: [[Option<Piece>; 8]; 8],
    castling: CastlingRights,
    en_passant: Option<EnPassant>,
    origin: Origin,
}

#[derive(Clone)]
enum Origin {
    None,
    Move {
        from: Position,
        to: Position,
        move_type: &'static str,
    },
}

#[derive(Clone)]
struct Timeline {
    id: i32,
    row: i32,
    label: String,
    owner: TimelineOwner,
    boards: Vec<BoardSnapshot>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TimelineOwner {
    Neutral,
    White,
    Black,
}

#[derive(Clone, Copy)]
struct Position {
    timeline_id: i32,
    time: i32,
    x: i32,
    y: i32,
}

#[derive(Clone)]
struct Game {
    turn: Color,
    timelines: Vec<Timeline>,
    next_timeline_id: i32,
    next_black_timeline_id: i32,
    staged_turn: Vec<GameCheckpoint>,
    last_message: String,
}

#[derive(Clone)]
struct GameCheckpoint {
    turn: Color,
    timelines: Vec<Timeline>,
    next_timeline_id: i32,
    next_black_timeline_id: i32,
    last_message: String,
}

#[derive(Clone, Copy)]
struct CastlingRights {
    white_kingside: bool,
    white_queenside: bool,
    black_kingside: bool,
    black_queenside: bool,
}

#[derive(Clone, Copy)]
struct EnPassant {
    x: i32,
    y: i32,
    captured_x: i32,
    captured_y: i32,
}

#[derive(Clone, Copy)]
struct Delta {
    x: i32,
    y: i32,
    t: i32,
    l: i32,
}

#[derive(Clone, Copy)]
enum MoveKind {
    Standard,
    Branch,
    Castle { rook_from_x: i32, rook_to_x: i32 },
    EnPassant { captured_x: i32, captured_y: i32 },
}
