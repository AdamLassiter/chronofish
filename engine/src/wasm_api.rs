// The browser talks to the engine through a deliberately small C ABI. A single
// thread-local Game mirrors the current UI state, and string-returning exports
// write UTF-8 into OUTPUT for JavaScript to copy immediately.
thread_local! {
    static GAME: RefCell<Option<Game>> = const { RefCell::new(None) };
    static VALUE_MODEL: RefCell<Option<NeuralLinearModel>> = const { RefCell::new(None) };
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
    let json = with_game(|game| ai_turn_json_with_optional_model(game, max_depth, max_nodes, None));
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_ai_turn_timed_json(
    max_depth: i32,
    max_nodes: i32,
    millis: i32,
) -> *const u8 {
    let json = with_game(|game| {
        ai_turn_json_with_optional_model(game, max_depth, max_nodes, search_deadline(millis))
    });
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_ai_turn_partitioned_timed_json(
    max_depth: i32,
    max_nodes: i32,
    millis: i32,
    partition_index: i32,
    partition_count: i32,
) -> *const u8 {
    let json = with_game(|game| {
        ai_turn_partitioned_json_with_optional_model(
            game,
            max_depth,
            max_nodes,
            search_deadline(millis),
            partition_index.max(0) as usize,
            partition_count.max(1) as usize,
        )
    });
    set_output(json)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 model JSON in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_set_neural_model_json(ptr: *const u8, len: usize) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let Ok(json) = std::str::from_utf8(bytes) else {
        return 0;
    };
    let Ok(model) = serde_json::from_str::<NeuralLinearModel>(json) else {
        return 0;
    };
    VALUE_MODEL.with(|value_model| {
        *value_model.borrow_mut() = Some(model);
    });
    1
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable compact model data in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_set_neural_model_bytes(ptr: *const u8, len: usize) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let Some(model) = parse_compact_neural_model(bytes) else {
        return 0;
    };
    VALUE_MODEL.with(|value_model| {
        *value_model.borrow_mut() = Some(model);
    });
    1
}

#[no_mangle]
pub extern "C" fn chronofish_clear_neural_model() {
    VALUE_MODEL.with(|value_model| {
        *value_model.borrow_mut() = None;
    });
}

#[no_mangle]
pub extern "C" fn chronofish_neural_sample_json(max_depth: i32, max_nodes: i32) -> *const u8 {
    let json = with_game(|game| {
        let result = game.best_ai_turn(max_depth.max(1), max_nodes.max(1), None);
        let encoded = game.encode_neural_position(game.turn);
        serde_json::json!({
            "label": result.score,
            "policy": policy_bucket(result.moves.first()),
            "depth": result.depth,
            "nodes": result.nodes,
            "sideToMove": match game.turn {
                Color::White => "white",
                Color::Black => "black",
            },
            "boardCount": encoded.board_count,
            "features": encoded.values
        })
        .to_string()
    });
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_neural_position_json() -> *const u8 {
    let json = with_game(|game| {
        let encoded = game.encode_neural_position(game.turn);
        serde_json::json!({
            "sideToMove": match game.turn {
                Color::White => "white",
                Color::Black => "black",
            },
            "boardCount": encoded.board_count,
            "features": encoded.values
        })
        .to_string()
    });
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_evaluation_json() -> *const u8 {
    let json = with_game(|game| {
        let weights = EvalWeights::default_tuned();
        let (white, black, source) = VALUE_MODEL.with(|value_model| {
            if let Some(model) = value_model.borrow().clone() {
                let evaluator = ValueEvaluator::hybrid_from_model(model, 3, 1);
                (
                    evaluator.evaluate(game, Color::White, &weights),
                    evaluator.evaluate(game, Color::Black, &weights),
                    "nn",
                )
            } else {
                (
                    game.evaluate_heuristic(Color::White, &weights),
                    game.evaluate_heuristic(Color::Black, &weights),
                    "heuristic",
                )
            }
        });
        let score = (white - black) / 2;
        serde_json::json!({
            "score": score,
            "white": white,
            "black": black,
            "source": source,
        })
        .to_string()
    });
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_training_sample_json(
    max_depth: i32,
    max_nodes: i32,
    seed: u32,
    plies: i32,
) -> *const u8 {
    let json = with_game(|game| {
        let mut sample_game = game.clone_for_search();
        apply_training_playout(&mut sample_game, seed, plies.max(0) as usize);
        let result = sample_game.best_ai_turn(max_depth.max(1), max_nodes.max(1), None);
        let encoded = sample_game.encode_neural_position(sample_game.turn);
        serde_json::json!({
            "label": result.score,
            "policy": policy_bucket(result.moves.first()),
            "depth": result.depth,
            "nodes": result.nodes,
            "sideToMove": match sample_game.turn {
                Color::White => "white",
                Color::Black => "black",
            },
            "boardCount": encoded.board_count,
            "features": encoded.values
        })
        .to_string()
    });
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_training_position_json(seed: u32, plies: i32) -> *const u8 {
    let json = with_game(|game| {
        let mut sample_game = game.clone_for_search();
        apply_training_playout(&mut sample_game, seed, plies.max(0) as usize);
        let encoded = sample_game.encode_neural_position(sample_game.turn);
        serde_json::json!({
            "sideToMove": match sample_game.turn {
                Color::White => "white",
                Color::Black => "black",
            },
            "boardCount": encoded.board_count,
            "features": encoded.values
        })
        .to_string()
    });
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

fn ai_turn_json_with_optional_model(
    game: &Game,
    max_depth: i32,
    max_nodes: i32,
    deadline: Option<SearchInstant>,
) -> String {
    VALUE_MODEL.with(|value_model| {
        let Some(model) = value_model.borrow().clone() else {
            return match deadline {
                Some(deadline) => game.best_ai_turn(max_depth, max_nodes, Some(deadline)).to_json(),
                None => game.ai_turn_json(max_depth, max_nodes),
            };
        };
        game.best_ai_turn_with_value_evaluator(
            max_depth,
            max_nodes,
            deadline,
            SearchOptions::optimized(),
            ValueEvaluator::hybrid_from_model(model, 3, 1),
            None,
        )
        .0
        .to_json()
    })
}

fn ai_turn_partitioned_json_with_optional_model(
    game: &Game,
    max_depth: i32,
    max_nodes: i32,
    deadline: Option<SearchInstant>,
    partition_index: usize,
    partition_count: usize,
) -> String {
    VALUE_MODEL.with(|value_model| {
        let Some(model) = value_model.borrow().clone() else {
            return game
                .best_ai_turn_partitioned(
                    max_depth,
                    max_nodes,
                    deadline,
                    partition_index,
                    partition_count,
                )
                .to_json();
        };
        game.best_ai_turn_partitioned_with_value_evaluator(
            max_depth,
            max_nodes,
            deadline,
            partition_index,
            partition_count,
            ValueEvaluator::hybrid_from_model(model, 3, 1),
        )
        .to_json()
    })
}

fn policy_bucket(movement: Option<&MoveStep>) -> usize {
    let Some(movement) = movement else {
        return 0;
    };
    let from = ((movement.from.y.clamp(0, 7) as usize) << 3) | movement.from.x.clamp(0, 7) as usize;
    let to = ((movement.to.y.clamp(0, 7) as usize) << 3) | movement.to.x.clamp(0, 7) as usize;
    1 + ((from * 64 + to) % 256)
}

fn apply_training_playout(game: &mut Game, seed: u32, plies: usize) {
    let weights = EvalWeights::default_tuned();
    let mut state = seed as u64 ^ 0x9e37_79b9_7f4a_7c15;
    for ply in 0..plies {
        let moves = game.legal_single_moves(&weights);
        if moves.is_empty() {
            break;
        }
        state = training_mix64(state ^ ply as u64);
        let movement = moves[(state as usize) % moves.len()];
        if !game.apply_move_for_search(movement.from, movement.to) {
            break;
        }
        let _ = game.submit_turn_for_search();
    }
}

fn training_mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
