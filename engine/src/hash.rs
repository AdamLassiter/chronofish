use crate::*;

const POSITION_HASH_SEED: u64 = 0x8a5c_7d13_9e37_79b9;

impl Game {
    pub(crate) fn recompute_position_hash(&self) -> u64 {
        let mut hash = mix_position_hash(POSITION_HASH_SEED);
        for timeline in &self.timelines {
            hash ^= timeline_position_hash(timeline);
            for board in &timeline.boards {
                hash ^= board_position_hash(timeline.id, board);
            }
        }
        hash
    }
}

pub(crate) fn timeline_position_hash(timeline: &Timeline) -> u64 {
    let owner = match timeline.owner {
        TimelineOwner::Neutral => 0,
        TimelineOwner::White => 1,
        TimelineOwner::Black => 2,
    };
    position_token(1, timeline.id as u64, timeline.row as u64, owner)
}

pub(crate) fn board_position_hash(timeline_id: i32, board: &BoardSnapshot) -> u64 {
    let mut hash = position_token(
        2,
        timeline_id as u64,
        board.time as u64,
        color_position_code(board.side_to_move),
    );
    hash ^= position_token(
        3,
        timeline_id as u64,
        board.time as u64,
        castling_code(board.castling),
    );
    if let Some(en_passant) = board.en_passant {
        hash ^= position_token(
            4,
            ((en_passant.x as u64) << 32) | en_passant.y as u64,
            ((en_passant.captured_x as u64) << 32) | en_passant.captured_y as u64,
            board.time as u64,
        );
    }
    for (y, rank) in board.board.iter().enumerate() {
        for (x, piece) in rank.iter().enumerate() {
            let Some(piece) = piece else {
                continue;
            };
            hash ^= position_token(
                5,
                timeline_id as u64,
                ((board.time as u64) << 8) | ((x as u64) << 4) | y as u64,
                piece_position_code(*piece),
            );
        }
    }
    hash
}

fn position_token(kind: u64, a: u64, b: u64, c: u64) -> u64 {
    mix_position_hash(
        POSITION_HASH_SEED
            ^ kind.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ a.rotate_left(17)
            ^ b.rotate_left(33)
            ^ c.rotate_left(49),
    )
}

fn color_position_code(color: Color) -> u64 {
    match color {
        Color::White => 1,
        Color::Black => 2,
    }
}

fn piece_position_code(piece: Piece) -> u64 {
    let piece_type = match piece.piece_type {
        PieceType::King => 1,
        PieceType::CommonKing => 2,
        PieceType::Queen => 3,
        PieceType::RoyalQueen => 4,
        PieceType::Princess => 5,
        PieceType::Rook => 6,
        PieceType::Bishop => 7,
        PieceType::Unicorn => 8,
        PieceType::Dragon => 9,
        PieceType::Knight => 10,
        PieceType::Pawn => 11,
        PieceType::Brawn => 12,
    };
    (color_position_code(piece.color) << 8) | piece_type
}

fn castling_code(castling: CastlingRights) -> u64 {
    castling.white_kingside as u64
        | ((castling.white_queenside as u64) << 1)
        | ((castling.black_kingside as u64) << 2)
        | ((castling.black_queenside as u64) << 3)
}

fn mix_position_hash(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
