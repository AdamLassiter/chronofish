const GPU_SNAPSHOT_MAGIC: i32 = 0x4346_4750;
const GPU_SNAPSHOT_VERSION: i32 = 1;
const GPU_TIMELINE_RECORD_I32S: i32 = 8;
const GPU_BOARD_RECORD_I32S: i32 = 12;
const GPU_BOARD_SQUARE_I32S: i32 = 64;

impl Game {
    fn gpu_snapshot_bytes(&self) -> Vec<u8> {
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
                push_i32(&mut words, board.en_passant.map_or(-1, |target| target.captured_x));
                push_i32(&mut words, board.en_passant.map_or(-1, |target| target.captured_y));
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

fn piece_type_code(piece_type: PieceType) -> i32 {
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
