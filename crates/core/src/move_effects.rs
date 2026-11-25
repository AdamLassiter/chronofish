//! Board rights and piece changes caused by a move.

use crate::*;

pub(crate) fn en_passant_after_move(
    piece: Piece,
    from: Position,
    to: Position,
    move_kind: MoveKind,
) -> Option<EnPassant> {
    // Only a same-board two-square pawn/brawn advance creates the immediately
    // capturable en-passant target.
    if !matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
        || !matches!(move_kind, MoveKind::Standard)
    {
        return None;
    }

    let forward = if piece.color == Color::White { 1 } else { -1 };
    (from.x == to.x && to.y - from.y == forward * 2).then_some(EnPassant {
        x: from.x,
        y: from.y + forward,
        captured_x: to.x,
        captured_y: to.y,
    })
}

pub(crate) fn update_castling_rights(
    castling: &mut CastlingRights,
    piece: Piece,
    from: Position,
    to: Position,
    board_before: [[Option<Piece>; 8]; 8],
) {
    // Moving a king, moving a rook from its home square, or capturing a rook on a
    // home square permanently clears the associated right on later snapshots.
    match (piece.color, piece.piece_type) {
        (Color::White, PieceType::King | PieceType::CommonKing) => {
            castling.white_kingside = false;
            castling.white_queenside = false;
        }
        (Color::Black, PieceType::King | PieceType::CommonKing) => {
            castling.black_kingside = false;
            castling.black_queenside = false;
        }
        (Color::White, PieceType::Rook | PieceType::Princess) if from.y == 0 && from.x == 0 => {
            castling.white_queenside = false;
        }
        (Color::White, PieceType::Rook | PieceType::Princess) if from.y == 0 && from.x == 7 => {
            castling.white_kingside = false;
        }
        (Color::Black, PieceType::Rook | PieceType::Princess) if from.y == 7 && from.x == 0 => {
            castling.black_queenside = false;
        }
        (Color::Black, PieceType::Rook | PieceType::Princess) if from.y == 7 && from.x == 7 => {
            castling.black_kingside = false;
        }
        _ => {}
    }

    match board_before[to.y as usize][to.x as usize] {
        Some(Piece {
            color: Color::White,
            piece_type: PieceType::Rook | PieceType::Princess,
        }) if to.y == 0 && to.x == 0 => castling.white_queenside = false,
        Some(Piece {
            color: Color::White,
            piece_type: PieceType::Rook | PieceType::Princess,
        }) if to.y == 0 && to.x == 7 => castling.white_kingside = false,
        Some(Piece {
            color: Color::Black,
            piece_type: PieceType::Rook | PieceType::Princess,
        }) if to.y == 7 && to.x == 0 => castling.black_queenside = false,
        Some(Piece {
            color: Color::Black,
            piece_type: PieceType::Rook | PieceType::Princess,
        }) if to.y == 7 && to.x == 7 => castling.black_kingside = false,
        _ => {}
    }
}

pub(crate) fn promote_if_needed(piece: Piece, y: i32) -> Piece {
    if !matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn) {
        return piece;
    }

    if (piece.color == Color::White && y == 7) || (piece.color == Color::Black && y == 0) {
        Piece {
            color: piece.color,
            piece_type: if piece.piece_type == PieceType::Brawn {
                PieceType::RoyalQueen
            } else {
                PieceType::Queen
            },
        }
    } else {
        piece
    }
}
