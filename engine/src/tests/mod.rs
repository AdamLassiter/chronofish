use crate::*;

fn empty_board_with_kings() -> [[Option<Piece>; 8]; 8] {
    let mut board = [[None; 8]; 8];
    board[0][4] = Some(Piece {
        color: Color::White,
        piece_type: PieceType::King,
    });
    board[7][4] = Some(Piece {
        color: Color::Black,
        piece_type: PieceType::King,
    });
    board
}

fn snapshot(time: i32, side_to_move: Color, board: [[Option<Piece>; 8]; 8]) -> BoardSnapshot {
    BoardSnapshot {
        time,
        side_to_move,
        board,
        castling: CastlingRights::new(),
        en_passant: None,
        origin: Origin::None,
    }
}

mod ai_search;
mod evaluation_training;
mod rules_foundation;
mod rules_special_pieces;
