// The browser talks to the engine through a deliberately small C ABI. A single
// thread-local Game mirrors the current UI state, and string-returning exports
// write UTF-8 into OUTPUT for JavaScript to copy immediately.
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
    // Compile-time crate version, so the frontend reports the version of the
    // actual WASM artifact it loaded.
    set_output(env!("CARGO_PKG_VERSION").into())
}

#[no_mangle]
pub extern "C" fn chronofish_reset() {
    GAME.with(|game| {
        *game.borrow_mut() = Some(Game::new());
    });
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

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 memory in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_load_notation(ptr: *const u8, len: usize) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let Ok(notation) = std::str::from_utf8(bytes) else {
        return with_game_mut(|game| {
            game.last_message = "Notation is not valid UTF-8.".to_string();
            0
        });
    };
    with_game_mut(|game| match game.load_notation(notation) {
        Ok(()) => {
            game.last_message = "Loaded notation.".to_string();
            1
        }
        Err(error) => {
            game.last_message = error;
            0
        }
    })
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
pub extern "C" fn chronofish_ai_turn_json(max_depth: i32, max_nodes: i32) -> *const u8 {
    let json = with_game(|game| game.ai_turn_json(max_depth, max_nodes));
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_ai_turn_timed_json(
    max_depth: i32,
    max_nodes: i32,
    millis: i32,
) -> *const u8 {
    let json = with_game(|game| game.ai_turn_timed_json(max_depth, max_nodes, millis));
    set_output(json)
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
    // Pointers returned by exports remain valid only until the next exported
    // string call rewrites OUTPUT.
    OUTPUT.with(|output| {
        let mut output = output.borrow_mut();
        *output = value.into_bytes();
        output.as_ptr()
    })
}

fn with_game<T>(callback: impl FnOnce(&Game) -> T) -> T {
    // Lazily initialize so snapshot/version style APIs can be called before an
    // explicit reset.
    GAME.with(|game| {
        let mut game = game.borrow_mut();

        if game.is_none() {
            *game = Some(Game::new());
        }

        callback(game.as_ref().expect("game initialized"))
    })
}

fn with_game_mut<T>(callback: impl FnOnce(&mut Game) -> T) -> T {
    // Mutating exports share the same lazy initialization path as read-only APIs.
    GAME.with(|game| {
        let mut game = game.borrow_mut();

        if game.is_none() {
            *game = Some(Game::new());
        }

        callback(game.as_mut().expect("game initialized"))
    })
}
