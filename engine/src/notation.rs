impl Game {
    fn move_message(
        &self,
        piece: Piece,
        from: Position,
        to: Position,
        is_standard_move: bool,
    ) -> String {
        let move_name = format!(
            "{}{} to {}{}",
            file_name(from.x),
            from.y + 1,
            file_name(to.x),
            to.y + 1
        );

        if is_standard_move {
            format!("{} played {}.", piece.color.capitalized(), move_name)
        } else {
            format!("{} branched {}.", piece.color.capitalized(), move_name)
        }
    }

    fn to_json(&self) -> String {
        let mut timelines = self.timelines.clone();
        timelines.sort_by(|left, right| left.row.cmp(&right.row).then(left.id.cmp(&right.id)));

        format!(
            "{{\"turn\":\"{}\",\"timelines\":[{}],\"nextTimelineId\":{}}}",
            self.turn.as_str(),
            timelines
                .iter()
                .map(Timeline::to_json)
                .collect::<Vec<_>>()
                .join(","),
            self.next_timeline_id
        )
    }

    fn in_bounds(x: i32, y: i32) -> bool {
        (0..8).contains(&x) && (0..8).contains(&y)
    }
}

impl Timeline {
    fn to_json(&self) -> String {
        let mut boards = self.boards.clone();
        boards.sort_by_key(|board| board.time);

        format!(
            "{{\"id\":{},\"row\":{},\"label\":\"{}\",\"owner\":\"{}\",\"boards\":[{}]}}",
            self.id,
            self.row,
            escape_json(&self.label),
            self.owner.as_str(),
            boards
                .iter()
                .map(|board| board.to_json(self.id))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

impl BoardSnapshot {
    fn to_json(&self, timeline_id: i32) -> String {
        let ranks = self
            .board
            .iter()
            .map(|rank| {
                format!(
                    "[{}]",
                    rank.iter().map(piece_json).collect::<Vec<_>>().join(",")
                )
            })
            .collect::<Vec<_>>()
            .join(",");

        format!(
            "{{\"id\":\"{}:{}\",\"time\":{},\"sideToMove\":\"{}\",\"board\":[{}],\"origin\":{}}}",
            timeline_id,
            self.time,
            self.time,
            self.side_to_move.as_str(),
            ranks,
            self.origin.to_json()
        )
    }
}

impl Origin {
    fn to_json(&self) -> String {
        match self {
            Origin::None => "null".to_string(),
            Origin::Move {
                from,
                to,
                move_type,
            } => format!(
                "{{\"from\":{},\"to\":{},\"type\":\"{}\"}}",
                position_json(*from),
                position_json(*to),
                move_type
            ),
        }
    }
}

impl Color {
    fn opposite(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Color::White => "white",
            Color::Black => "black",
        }
    }

    fn capitalized(self) -> &'static str {
        match self {
            Color::White => "White",
            Color::Black => "Black",
        }
    }
}

impl PieceType {
    fn as_str(self) -> &'static str {
        match self {
            PieceType::King => "king",
            PieceType::Queen => "queen",
            PieceType::Rook => "rook",
            PieceType::Bishop => "bishop",
            PieceType::Knight => "knight",
            PieceType::Pawn => "pawn",
        }
    }
}

impl TimelineOwner {
    fn from_color(color: Color) -> Self {
        match color {
            Color::White => TimelineOwner::White,
            Color::Black => TimelineOwner::Black,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            TimelineOwner::Neutral => "neutral",
            TimelineOwner::White => "white",
            TimelineOwner::Black => "black",
        }
    }
}

impl CastlingRights {
    fn new() -> Self {
        Self {
            white_kingside: true,
            white_queenside: true,
            black_kingside: true,
            black_queenside: true,
        }
    }
}

impl MoveKind {
    fn name(self) -> &'static str {
        match self {
            MoveKind::Standard => "standard",
            MoveKind::Branch => "branch",
            MoveKind::Castle { .. } => "castle",
            MoveKind::EnPassant { .. } => "en-passant",
        }
    }
}

fn en_passant_after_move(
    piece: Piece,
    from: Position,
    to: Position,
    move_kind: MoveKind,
) -> Option<EnPassant> {
    if piece.piece_type != PieceType::Pawn || !matches!(move_kind, MoveKind::Standard) {
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

fn update_castling_rights(
    castling: &mut CastlingRights,
    piece: Piece,
    from: Position,
    to: Position,
    board_before: [[Option<Piece>; 8]; 8],
) {
    match (piece.color, piece.piece_type) {
        (Color::White, PieceType::King) => {
            castling.white_kingside = false;
            castling.white_queenside = false;
        }
        (Color::Black, PieceType::King) => {
            castling.black_kingside = false;
            castling.black_queenside = false;
        }
        (Color::White, PieceType::Rook) if from.y == 0 && from.x == 0 => {
            castling.white_queenside = false;
        }
        (Color::White, PieceType::Rook) if from.y == 0 && from.x == 7 => {
            castling.white_kingside = false;
        }
        (Color::Black, PieceType::Rook) if from.y == 7 && from.x == 0 => {
            castling.black_queenside = false;
        }
        (Color::Black, PieceType::Rook) if from.y == 7 && from.x == 7 => {
            castling.black_kingside = false;
        }
        _ => {}
    }

    match board_before[to.y as usize][to.x as usize] {
        Some(Piece {
            color: Color::White,
            piece_type: PieceType::Rook,
        }) if to.y == 0 && to.x == 0 => castling.white_queenside = false,
        Some(Piece {
            color: Color::White,
            piece_type: PieceType::Rook,
        }) if to.y == 0 && to.x == 7 => castling.white_kingside = false,
        Some(Piece {
            color: Color::Black,
            piece_type: PieceType::Rook,
        }) if to.y == 7 && to.x == 0 => castling.black_queenside = false,
        Some(Piece {
            color: Color::Black,
            piece_type: PieceType::Rook,
        }) if to.y == 7 && to.x == 7 => castling.black_kingside = false,
        _ => {}
    }
}

fn promote_if_needed(piece: Piece, y: i32) -> Piece {
    if piece.piece_type != PieceType::Pawn {
        return piece;
    }

    if (piece.color == Color::White && y == 7) || (piece.color == Color::Black && y == 0) {
        Piece {
            color: piece.color,
            piece_type: PieceType::Queen,
        }
    } else {
        piece
    }
}

fn piece_json(piece: &Option<Piece>) -> String {
    match piece {
        Some(piece) => format!(
            "{{\"color\":\"{}\",\"type\":\"{}\"}}",
            piece.color.as_str(),
            piece.piece_type.as_str()
        ),
        None => "null".to_string(),
    }
}

fn position_json(position: Position) -> String {
    format!(
        "{{\"timelineId\":{},\"time\":{},\"x\":{},\"y\":{}}}",
        position.timeline_id, position.time, position.x, position.y
    )
}

fn file_name(x: i32) -> &'static str {
    match x {
        0 => "a",
        1 => "b",
        2 => "c",
        3 => "d",
        4 => "e",
        5 => "f",
        6 => "g",
        7 => "h",
        _ => "?",
    }
}

fn escape_json(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
