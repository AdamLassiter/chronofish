use crate::{
    cpu::{MAX_MOVES_PER_NODE, MAX_QUIESCENCE_DEPTH, MAX_TURN_PLANS, REQUIRED_MOVES_PER_BOARD},
    *,
};

pub(crate) const GPU_SNAPSHOT_MAGIC: i32 = 0x4346_4750;
pub(crate) const GPU_SNAPSHOT_VERSION: i32 = 1;
pub(crate) const GPU_TIMELINE_RECORD_I32S: i32 = 8;
pub(crate) const GPU_BOARD_RECORD_I32S: i32 = 12;
pub(crate) const GPU_BOARD_SQUARE_I32S: i32 = 64;

impl Game {
    pub(crate) fn gpu_snapshot_json(&self) -> String {
        let mut timelines = self.timelines.clone();
        timelines.sort_by(|left, right| left.row.cmp(&right.row).then(left.id.cmp(&right.id)));
        let mut timeline_values = Vec::with_capacity(timelines.len());
        let mut board_values = Vec::new();
        for (timeline_index, timeline) in timelines.iter().enumerate() {
            let mut boards = timeline.boards.clone();
            boards.sort_by_key(|board| board.time);
            let latest_time = boards.last().map_or(0, |board| board.time);
            let timeline_boards = boards
                .iter()
                .map(|board| {
                    gpu_snapshot_board_json(
                        timeline_index as i32,
                        timeline.id,
                        board,
                        board.time == latest_time,
                    )
                })
                .collect::<Vec<_>>();
            board_values.extend(timeline_boards.iter().cloned());
            timeline_values.push(serde_json::json!({
                "id": timeline.id,
                "row": timeline.row,
                "label": timeline.label,
                "owner": owner_name(timeline.owner),
                "boardCount": timeline_boards.len(),
                "latestTime": latest_time,
                "boards": timeline_boards,
            }));
        }
        serde_json::json!({
            "format": "engine-gpu-snapshot-v1",
            "turn": color_name(self.turn),
            "nextTimelineId": self.next_timeline_id,
            "nextBlackTimelineId": self.next_black_timeline_id,
            "royalCaptureBy": self.staged_royal_capture_by.map(color_name),
            "timelines": timeline_values,
            "boards": board_values,
        })
        .to_string()
    }

    pub(crate) fn gpu_snapshot_bytes(&self) -> Vec<u8> {
        let mut timelines = self.timelines.clone();
        timelines.sort_by(|left, right| left.row.cmp(&right.row).then(left.id.cmp(&right.id)));
        let board_count = timelines
            .iter()
            .map(|timeline| timeline.boards.len())
            .sum::<usize>();
        let present_time = self.present_time().unwrap_or(0);

        let mut words = Vec::with_capacity(
            16 + timelines.len() * GPU_TIMELINE_RECORD_I32S as usize
                + board_count * (GPU_BOARD_RECORD_I32S + GPU_BOARD_SQUARE_I32S) as usize,
        );
        push_i32(&mut words, GPU_SNAPSHOT_MAGIC);
        push_i32(&mut words, GPU_SNAPSHOT_VERSION);
        push_i32(&mut words, color_code(self.turn));
        push_i32(&mut words, timelines.len() as i32);
        push_i32(&mut words, board_count as i32);
        push_i32(&mut words, self.next_timeline_id);
        push_i32(&mut words, self.next_black_timeline_id);
        push_i32(&mut words, option_color_code(self.staged_royal_capture_by));
        push_i32(&mut words, present_time);
        push_i32(&mut words, GPU_TIMELINE_RECORD_I32S);
        push_i32(&mut words, GPU_BOARD_RECORD_I32S);
        push_i32(&mut words, GPU_BOARD_SQUARE_I32S);
        push_i32(&mut words, MAX_TURN_PLANS as i32);
        push_i32(&mut words, MAX_MOVES_PER_NODE as i32);
        push_i32(&mut words, REQUIRED_MOVES_PER_BOARD as i32);
        push_i32(&mut words, MAX_QUIESCENCE_DEPTH);

        let mut first_board = 0_i32;
        for timeline in &timelines {
            let mut boards = timeline.boards.clone();
            boards.sort_by_key(|board| board.time);
            push_i32(&mut words, timeline.id);
            push_i32(&mut words, timeline.row);
            push_i32(&mut words, owner_code(timeline.owner));
            push_i32(&mut words, first_board);
            push_i32(&mut words, boards.len() as i32);
            push_i32(&mut words, self.is_active_timeline(timeline.id) as i32);
            push_i32(&mut words, boards.last().map_or(0, |board| board.time));
            push_i32(&mut words, 0);
            first_board += boards.len() as i32;
        }

        for (timeline_index, timeline) in timelines.iter().enumerate() {
            let mut boards = timeline.boards.clone();
            boards.sort_by_key(|board| board.time);
            let latest_time = boards.last().map_or(0, |board| board.time);
            for board in &boards {
                push_i32(&mut words, timeline_index as i32);
                push_i32(&mut words, timeline.id);
                push_i32(&mut words, board.time);
                push_i32(&mut words, color_code(board.side_to_move));
                push_i32(&mut words, castling_code(board.castling));
                push_i32(&mut words, board.en_passant.map_or(-1, |target| target.x));
                push_i32(&mut words, board.en_passant.map_or(-1, |target| target.y));
                push_i32(
                    &mut words,
                    board.en_passant.map_or(-1, |target| target.captured_x),
                );
                push_i32(
                    &mut words,
                    board.en_passant.map_or(-1, |target| target.captured_y),
                );
                push_i32(&mut words, (board.time == latest_time) as i32);
                push_i32(&mut words, origin_code(&board.origin));
                push_i32(&mut words, 0);
                for y in 0..8 {
                    for x in 0..8 {
                        push_i32(&mut words, piece_code(board.board[y][x]));
                    }
                }
            }
        }

        words
    }
}

fn gpu_snapshot_board_json(
    timeline_index: i32,
    timeline_id: i32,
    board: &BoardSnapshot,
    latest: bool,
) -> serde_json::Value {
    serde_json::json!({
        "timelineIndex": timeline_index,
        "timelineId": timeline_id,
        "time": board.time,
        "sideToMove": color_name(board.side_to_move),
        "castling": castling_code(board.castling),
        "enPassant": board.en_passant.map(|target| serde_json::json!({
            "x": target.x,
            "y": target.y,
            "capturedX": target.captured_x,
            "capturedY": target.captured_y,
        })),
        "origin": origin_json(&board.origin),
        "latest": latest,
        "originKind": origin_code(&board.origin),
        "squares": board_squares_json(board),
    })
}

fn board_squares_json(board: &BoardSnapshot) -> serde_json::Value {
    serde_json::Value::Array(
        (0..8)
            .flat_map(|y| (0..8).map(move |x| serde_json::json!(piece_code(board.board[y][x]))))
            .collect(),
    )
}

fn origin_json(origin: &Origin) -> serde_json::Value {
    match origin {
        Origin::None => serde_json::Value::Null,
        Origin::Move {
            from,
            to,
            move_type,
        } => serde_json::json!({
            "type": move_type,
            "from": position_json(from),
            "to": position_json(to),
        }),
    }
}

fn position_json(position: &Position) -> serde_json::Value {
    serde_json::json!({
        "timelineId": position.timeline_id,
        "time": position.time,
        "x": position.x,
        "y": position.y,
    })
}

fn push_i32(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn color_code(color: Color) -> i32 {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}

fn option_color_code(color: Option<Color>) -> i32 {
    match color {
        None => -1,
        Some(Color::White) => 0,
        Some(Color::Black) => 1,
    }
}

fn owner_code(owner: TimelineOwner) -> i32 {
    match owner {
        TimelineOwner::Neutral => 0,
        TimelineOwner::White => 1,
        TimelineOwner::Black => 2,
    }
}

fn color_name(color: Color) -> &'static str {
    match color {
        Color::White => "white",
        Color::Black => "black",
    }
}

fn owner_name(owner: TimelineOwner) -> &'static str {
    match owner {
        TimelineOwner::Neutral => "neutral",
        TimelineOwner::White => "white",
        TimelineOwner::Black => "black",
    }
}

pub(crate) fn piece_type_code(piece_type: PieceType) -> i32 {
    match piece_type {
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
    }
}

fn piece_code(piece: Option<Piece>) -> i32 {
    piece.map_or(0, |piece| {
        piece_type_code(piece.piece_type) | (color_code(piece.color) << 8)
    })
}

fn castling_code(castling: CastlingRights) -> i32 {
    castling.white_kingside as i32
        | ((castling.white_queenside as i32) << 1)
        | ((castling.black_kingside as i32) << 2)
        | ((castling.black_queenside as i32) << 3)
}

fn origin_code(origin: &Origin) -> i32 {
    match origin {
        Origin::None => 0,
        Origin::Move { move_type, .. } if *move_type == "source-advance" => 1,
        Origin::Move { move_type, .. } if *move_type == "branch" => 2,
        Origin::Move { move_type, .. } if *move_type == "cross-board" => 3,
        Origin::Move { .. } => 4,
    }
}
