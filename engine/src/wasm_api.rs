// The browser talks to the engine through a deliberately small C ABI. A single
// thread-local Game mirrors the current UI state for non-bot rules work.
thread_local! {
    static GAME: RefCell<Option<Game>> = const { RefCell::new(None) };
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[no_mangle]
pub extern "C" fn chronofish_alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::with_capacity(len);
    let pointer = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    pointer
}

/// # Safety
///
/// `ptr` and `len` must be a pointer/length pair previously returned by
/// `chronofish_alloc` and not already freed.
#[no_mangle]
pub unsafe extern "C" fn chronofish_dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

#[no_mangle]
pub extern "C" fn chronofish_version() -> *const u8 {
    set_output(env!("CARGO_PKG_VERSION").into())
}

#[no_mangle]
pub extern "C" fn chronofish_reset() {
    GAME.with(|game| {
        *game.borrow_mut() = Some(Game::new());
    });
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON in this WASM instance
/// for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_load_snapshot_json(ptr: *const u8, len: usize) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let Ok(text) = std::str::from_utf8(bytes) else {
        return set_last_message("Snapshot is not valid UTF-8.");
    };
    match parse_game_snapshot(text) {
        Ok(next) => {
            GAME.with(|game| {
                *game.borrow_mut() = Some(next);
            });
            1
        }
        Err(error) => set_last_message(&error),
    }
}

#[no_mangle]
pub extern "C" fn chronofish_snapshot_json() -> *const u8 {
    let json = with_game(|game| game.to_json());
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_staged_turn_notation() -> *const u8 {
    let notation = with_game(Game::staged_turn_notation);
    set_output(notation)
}

#[no_mangle]
pub extern "C" fn chronofish_legal_targets_json(
    from_timeline_id: i32,
    from_time: i32,
    from_x: i32,
    from_y: i32,
) -> *const u8 {
    let from = Position {
        timeline_id: from_timeline_id,
        time: from_time,
        x: from_x,
        y: from_y,
    };
    let json = with_game(|game| game.legal_targets_json(from));
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_apply_move(
    from_timeline_id: i32,
    from_time: i32,
    from_x: i32,
    from_y: i32,
    to_timeline_id: i32,
    to_time: i32,
    to_x: i32,
    to_y: i32,
) -> i32 {
    let from = Position {
        timeline_id: from_timeline_id,
        time: from_time,
        x: from_x,
        y: from_y,
    };
    let to = Position {
        timeline_id: to_timeline_id,
        time: to_time,
        x: to_x,
        y: to_y,
    };
    with_game_mut(|game| game.apply_move(from, to))
}

#[no_mangle]
pub extern "C" fn chronofish_submit_turn() -> i32 {
    with_game_mut(Game::submit_turn)
}

#[no_mangle]
pub extern "C" fn chronofish_undo_staged_move() -> i32 {
    with_game_mut(Game::undo_staged_move)
}

#[no_mangle]
pub extern "C" fn chronofish_last_message() -> *const u8 {
    let message = with_game(|game| game.last_message.clone());
    set_output(message)
}

#[no_mangle]
pub extern "C" fn chronofish_output_len() -> usize {
    OUTPUT.with(|output| output.borrow().len())
}

fn set_output(value: String) -> *const u8 {
    set_output_bytes(value.into_bytes())
}

fn set_output_bytes(value: Vec<u8>) -> *const u8 {
    OUTPUT.with(|output| {
        let mut output = output.borrow_mut();
        *output = value;
        output.as_ptr()
    })
}

fn with_game<T>(callback: impl FnOnce(&Game) -> T) -> T {
    GAME.with(|game| {
        let mut game = game.borrow_mut();
        if game.is_none() {
            *game = Some(Game::new());
        }
        callback(game.as_ref().expect("game initialized"))
    })
}

fn with_game_mut<T>(callback: impl FnOnce(&mut Game) -> T) -> T {
    GAME.with(|game| {
        let mut game = game.borrow_mut();
        if game.is_none() {
            *game = Some(Game::new());
        }
        callback(game.as_mut().expect("game initialized"))
    })
}

fn set_last_message(message: &str) -> i32 {
    GAME.with(|game| {
        let mut game = game.borrow_mut();
        if game.is_none() {
            *game = Some(Game::new());
        }
        game.as_mut().expect("game initialized").last_message = message.to_string();
    });
    0
}

fn parse_game_snapshot(text: &str) -> Result<Game, String> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|error| format!("Snapshot JSON failed: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Snapshot must be an object.".to_string())?;
    let turn = parse_color(object.get("turn"))?;
    let timelines_value = object
        .get("timelines")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "Snapshot timelines must be an array.".to_string())?;
    let mut timelines = Vec::with_capacity(timelines_value.len());
    for timeline in timelines_value {
        timelines.push(parse_timeline(timeline)?);
    }
    timelines.sort_by(|left, right| left.row.cmp(&right.row).then(left.id.cmp(&right.id)));
    let next_timeline_id = optional_i32(object.get("nextTimelineId"))
        .unwrap_or_else(|| next_timeline_id_for(&timelines, Color::White));
    let next_black_timeline_id = optional_i32(object.get("nextBlackTimelineId"))
        .unwrap_or_else(|| next_timeline_id_for(&timelines, Color::Black));
    Ok(Game {
        turn,
        timelines,
        next_timeline_id,
        next_black_timeline_id,
        staged_turn: Vec::new(),
        staged_notation: Vec::new(),
        staged_royal_capture_by: None,
        last_message: format!("{} to move.", turn.capitalized()),
    })
}

fn parse_timeline(value: &serde_json::Value) -> Result<Timeline, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Timeline must be an object.".to_string())?;
    let id = required_i32(object.get("id"), "timeline id")?;
    let row = optional_i32(object.get("row")).unwrap_or(id);
    let owner = match object.get("owner").and_then(|value| value.as_str()) {
        Some("white") => TimelineOwner::White,
        Some("black") => TimelineOwner::Black,
        _ => TimelineOwner::Neutral,
    };
    let boards_value = object
        .get("boards")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "Timeline boards must be an array.".to_string())?;
    let mut boards = Vec::with_capacity(boards_value.len());
    for board in boards_value {
        boards.push(parse_board(board)?);
    }
    boards.sort_by_key(|board| board.time);
    Ok(Timeline {
        id,
        row,
        label: object
            .get("label")
            .and_then(|value| value.as_str())
            .unwrap_or("Timeline")
            .to_string(),
        owner,
        boards,
    })
}

fn parse_board(value: &serde_json::Value) -> Result<BoardSnapshot, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Board must be an object.".to_string())?;
    let time = required_i32(object.get("time"), "board time")?;
    let side_to_move = parse_color(object.get("sideToMove"))?;
    let board_value = object
        .get("board")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "Board squares must be an array.".to_string())?;
    if board_value.len() != 8 {
        return Err("Board must contain 8 ranks.".to_string());
    }
    let mut board = [[None; 8]; 8];
    for (y, row_value) in board_value.iter().enumerate() {
        let row = row_value
            .as_array()
            .ok_or_else(|| "Board rank must be an array.".to_string())?;
        if row.len() != 8 {
            return Err("Board rank must contain 8 files.".to_string());
        }
        for (x, piece_value) in row.iter().enumerate() {
            board[y][x] = parse_piece(piece_value)?;
        }
    }
    Ok(BoardSnapshot {
        time,
        side_to_move,
        board,
        castling: parse_castling(object.get("castling")),
        en_passant: parse_en_passant(object.get("enPassant")),
        origin: parse_origin(object.get("origin")),
    })
}

fn parse_piece(value: &serde_json::Value) -> Result<Option<Piece>, String> {
    if value.is_null() {
        return Ok(None);
    }
    let object = value
        .as_object()
        .ok_or_else(|| "Piece must be an object or null.".to_string())?;
    let color = parse_color(object.get("color"))?;
    let piece_type = match object.get("type").and_then(|value| value.as_str()) {
        Some("king") => PieceType::King,
        Some("commonKing") => PieceType::CommonKing,
        Some("queen") => PieceType::Queen,
        Some("royalQueen") => PieceType::RoyalQueen,
        Some("princess") => PieceType::Princess,
        Some("rook") => PieceType::Rook,
        Some("bishop") => PieceType::Bishop,
        Some("unicorn") => PieceType::Unicorn,
        Some("dragon") => PieceType::Dragon,
        Some("knight") => PieceType::Knight,
        Some("pawn") => PieceType::Pawn,
        Some("brawn") => PieceType::Brawn,
        Some(other) => return Err(format!("Unknown piece type `{other}`.")),
        None => return Err("Piece type is missing.".to_string()),
    };
    Ok(Some(Piece { color, piece_type }))
}

fn parse_castling(value: Option<&serde_json::Value>) -> CastlingRights {
    let Some(value) = value else {
        return CastlingRights::new();
    };
    if let Some(bits) = value.as_i64() {
        return CastlingRights {
            white_kingside: bits & 1 != 0,
            white_queenside: bits & 2 != 0,
            black_kingside: bits & 4 != 0,
            black_queenside: bits & 8 != 0,
        };
    }
    value.as_object().map_or_else(CastlingRights::new, |object| CastlingRights {
        white_kingside: optional_bool(object.get("whiteKingside")).unwrap_or(true),
        white_queenside: optional_bool(object.get("whiteQueenside")).unwrap_or(true),
        black_kingside: optional_bool(object.get("blackKingside")).unwrap_or(true),
        black_queenside: optional_bool(object.get("blackQueenside")).unwrap_or(true),
    })
}

fn parse_en_passant(value: Option<&serde_json::Value>) -> Option<EnPassant> {
    let object = value?.as_object()?;
    Some(EnPassant {
        x: required_i32(object.get("x"), "enPassant x").ok()?,
        y: required_i32(object.get("y"), "enPassant y").ok()?,
        captured_x: required_i32(object.get("capturedX"), "enPassant capturedX").ok()?,
        captured_y: required_i32(object.get("capturedY"), "enPassant capturedY").ok()?,
    })
}

fn parse_origin(value: Option<&serde_json::Value>) -> Origin {
    let Some(object) = value.and_then(|value| value.as_object()) else {
        return Origin::None;
    };
    let Some(from) = object.get("from").and_then(parse_position_value) else {
        return Origin::None;
    };
    let Some(to) = object.get("to").and_then(parse_position_value) else {
        return Origin::None;
    };
    Origin::Move {
        from,
        to,
        move_type: match object.get("type").and_then(|value| value.as_str()) {
            Some("branch") => "branch",
            Some("castle") => "castle",
            Some("en-passant") => "en-passant",
            Some("source-advance") => "source-advance",
            Some("cross-board") => "cross-board",
            _ => "standard",
        },
    }
}

fn parse_position_value(value: &serde_json::Value) -> Option<Position> {
    let object = value.as_object()?;
    Some(Position {
        timeline_id: required_i32(object.get("timelineId"), "timelineId").ok()?,
        time: required_i32(object.get("time"), "time").ok()?,
        x: required_i32(object.get("x"), "x").ok()?,
        y: required_i32(object.get("y"), "y").ok()?,
    })
}

fn parse_color(value: Option<&serde_json::Value>) -> Result<Color, String> {
    match value.and_then(|value| value.as_str()) {
        Some("white") => Ok(Color::White),
        Some("black") => Ok(Color::Black),
        Some(other) => Err(format!("Unknown color `{other}`.")),
        None => Err("Color is missing.".to_string()),
    }
}

fn required_i32(value: Option<&serde_json::Value>, name: &str) -> Result<i32, String> {
    optional_i32(value).ok_or_else(|| format!("{name} must be an integer."))
}

fn optional_i32(value: Option<&serde_json::Value>) -> Option<i32> {
    value?.as_i64()?.try_into().ok()
}

fn optional_bool(value: Option<&serde_json::Value>) -> Option<bool> {
    value?.as_bool()
}

fn next_timeline_id_for(timelines: &[Timeline], color: Color) -> i32 {
    match color {
        Color::White => timelines
            .iter()
            .map(|timeline| timeline.id)
            .filter(|id| *id > 0)
            .max()
            .unwrap_or(0)
            + 1,
        Color::Black => timelines
            .iter()
            .map(|timeline| timeline.id)
            .filter(|id| *id < 0)
            .min()
            .unwrap_or(0)
            - 1,
    }
}
