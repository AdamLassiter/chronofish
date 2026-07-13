use std::cell::RefCell;

use crate::{cpu::EvalWeights, *};

// The browser talks to the engine through a deliberately small C ABI. A single
// thread-local Game mirrors the current UI state for non-bot rules work.
thread_local! {
    static GAME: RefCell<Option<Game>> = const { RefCell::new(None) };
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

// Raw WebAssembly i64 exports require JavaScript BigInt values. Browser-facing
// time budgets are bounded well below i32::MAX, so expose them as Numbers.
fn wasm_milliseconds(value: u64) -> i32 {
    value.min(i32::MAX as u64) as i32
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

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON in this WASM instance
/// for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_load_ai_parameters_json(ptr: *const u8, len: usize) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let Ok(text) = std::str::from_utf8(bytes) else {
        return set_last_message("AI parameters are not valid UTF-8.");
    };
    match EvalWeights::set_active_from_json(text) {
        Ok(()) => 1,
        Err(error) => set_last_message(&format!("Invalid AI parameters: {error}")),
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a CPU
/// parameter object in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_parameters_key_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU parameters key request") else {
        return std::ptr::null();
    };
    match parse_cpu_parameters_value_from_text(text)
        .map(|parameters| crate::cpu::search::cpu_parameters_key(&parameters))
    {
        Ok(key) => set_output(key),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing an array of
/// CPU parameter objects in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_unique_cpu_parameters_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Unique CPU parameters request") else {
        return std::ptr::null();
    };
    match parse_cpu_parameter_array_from_text(text)
        .map(|parameters| crate::cpu::search::unique_cpu_parameters(&parameters))
        .and_then(encode_cpu_parameter_array_json)
    {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing baseline,
/// elites, target, seed, generation, and stagnation fields in this WASM instance
/// for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_breed_cpu_population_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU population breeding request") else {
        return std::ptr::null();
    };
    match breed_cpu_population_json(text).and_then(encode_cpu_parameter_array_json) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing an array of
/// `{ parameters, score }` CPU candidate objects in this WASM instance for the
/// duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_rank_cpu_scored_candidates_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Rank CPU scored candidates request") else {
        return std::ptr::null();
    };
    match parse_cpu_scored_candidate_array_from_text(text)
        .map(crate::cpu::search::rank_cpu_scored_candidates)
        .and_then(encode_cpu_scored_candidate_array_json)
    {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing `baseline`
/// and `candidates` fields in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_training_elites_json(
    ptr: *const u8,
    len: usize,
    cpu_finalists: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU training elites request") else {
        return std::ptr::null();
    };
    match cpu_training_elites_json(text, cpu_finalists).and_then(encode_cpu_parameter_array_json) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing `baseline`
/// and `screened` fields in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_training_finalist_candidates_json(
    ptr: *const u8,
    len: usize,
    target: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU training finalist candidates request") else {
        return std::ptr::null();
    };
    match cpu_training_finalist_candidates_json(text, target)
        .and_then(encode_cpu_parameter_array_json)
    {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing
/// `baseline`, `finalists`, `previousBaselineScore`, and `bestCandidateScore`
/// fields in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_training_generation_outcome_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU training generation outcome request") else {
        return std::ptr::null();
    };
    match cpu_training_generation_outcome_json(text)
        .and_then(encode_cpu_training_generation_outcome_json)
    {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing
/// `candidates` and `fitness` fields in this WASM instance for the duration of
/// the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_candidate_scoring_plan_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU candidate scoring plan request") else {
        return std::ptr::null();
    };
    match cpu_candidate_scoring_plan_json(text).and_then(encode_cpu_candidate_scoring_plan_json) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing
/// `parameters` and `score` fields in this WASM instance for the duration of
/// the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_fitness_entry_for_candidate_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU fitness entry request") else {
        return std::ptr::null();
    };
    match cpu_fitness_entry_for_candidate_json(text).and_then(encode_cpu_fitness_entry_json) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 CPU worker search config
/// JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_worker_search_config_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU worker search config request") else {
        return std::ptr::null();
    };
    match crate::cpu::search::cpu_worker_search_config_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 CPU search configuration
/// JSON in this WASM instance for the duration of the call. The current game
/// must already have been loaded through the normal snapshot API.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_search_json(ptr: *const u8, len: usize) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU search request") else {
        return std::ptr::null();
    };
    match crate::cpu::search::cpu_worker_search_config(text) {
        Ok(config) => {
            let json = with_game(|game| {
                crate::cpu::search::search_game_json(
                    game,
                    config.depth,
                    config.min_depth,
                    config.nodes,
                    config.time_ms,
                    config.search_strategy,
                )
            });
            set_output(json)
        }
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 CPU worker search result
/// JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_worker_search_result_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU worker search result request") else {
        return std::ptr::null();
    };
    match crate::cpu::search::cpu_worker_search_result_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 CPU apply-turn JSON in
/// this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_apply_turn_json(ptr: *const u8, len: usize) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU apply-turn request") else {
        return std::ptr::null();
    };
    match crate::cpu::search::cpu_apply_turn_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 snapshot JSON in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_training_sample_json(ptr: *const u8, len: usize) -> *const u8 {
    if ptr.is_null() {
        set_last_message("Training sample snapshot pointer is null.");
        return std::ptr::null();
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let Ok(text) = std::str::from_utf8(bytes) else {
        set_last_message("Training sample snapshot is not valid UTF-8.");
        return std::ptr::null();
    };
    match crate::gpu::training::sample_from_snapshot_label(Some(text), 0.0, 1.0).and_then(
        |sample| {
            serde_json::to_string(&sample)
                .map_err(|error| format!("failed to encode training sample: {error}"))
        },
    ) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing an array of
/// snapshot JSON values in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_training_samples_json(ptr: *const u8, len: usize) -> *const u8 {
    if ptr.is_null() {
        set_last_message("Training sample snapshot batch pointer is null.");
        return std::ptr::null();
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let Ok(text) = std::str::from_utf8(bytes) else {
        set_last_message("Training sample snapshot batch is not valid UTF-8.");
        return std::ptr::null();
    };
    let snapshots = match serde_json::from_str::<Vec<serde_json::Value>>(text) {
        Ok(snapshots) => snapshots,
        Err(error) => {
            set_last_message(&format!(
                "Training sample snapshot batch is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    let samples = snapshots
        .iter()
        .map(|snapshot| {
            let snapshot = serde_json::to_string(snapshot)
                .map_err(|error| format!("failed to encode snapshot: {error}"))?;
            crate::gpu::training::sample_from_snapshot_label(Some(&snapshot), 0.0, 1.0)
        })
        .collect::<Result<Vec<_>, _>>();
    match samples.and_then(|samples| {
        serde_json::to_string(&samples)
            .map_err(|error| format!("failed to encode training samples: {error}"))
    }) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing training
/// samples in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_dedupe_training_samples_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Training sample replay buffer") else {
        return std::ptr::null();
    };
    let samples = match serde_json::from_str::<Vec<crate::gpu::training::TrainingSample>>(text) {
        Ok(samples) => samples,
        Err(error) => {
            set_last_message(&format!("Training samples are not valid JSON: {error}"));
            return std::ptr::null();
        }
    };
    encode_training_samples_json(crate::gpu::training::dedupe_training_samples(&samples))
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing training
/// samples in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_select_training_working_set_indexes_bytes(
    ptr: *const u8,
    len: usize,
    max_projected_bytes: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Training working set samples") else {
        return std::ptr::null();
    };
    let samples = match serde_json::from_str::<Vec<crate::gpu::training::TrainingSample>>(text) {
        Ok(samples) => samples,
        Err(error) => {
            set_last_message(&format!(
                "Training working set samples are not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    let mut indexes = Vec::new();
    for index in
        crate::gpu::training::select_training_working_set_indices(&samples, max_projected_bytes)
    {
        let Ok(index) = i32::try_from(index) else {
            set_last_message("Training working set index exceeds i32 range.");
            return std::ptr::null();
        };
        indexes.push(index);
    }
    set_output_i32s(indexes)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing one
/// training sample in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_stable_sample_hash_json(
    ptr: *const u8,
    len: usize,
    index: usize,
) -> u32 {
    let Some(text) = wasm_input_text(ptr, len, "Stable sample hash request") else {
        return 0;
    };
    let sample = match serde_json::from_str::<crate::gpu::training::TrainingSample>(text) {
        Ok(sample) => sample,
        Err(error) => {
            set_last_message(&format!(
                "Stable sample hash request is not valid JSON: {error}"
            ));
            return 0;
        }
    };
    crate::gpu::training::stable_sample_hash(&sample, index)
}

/// # Safety
///
/// `ptr` must point to `len` bytes containing little-endian u32 training
/// indices in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_shuffled_training_indices_bytes(
    ptr: *const u8,
    len: usize,
    epoch: u32,
    seed: u32,
) -> *const u8 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Shuffled training indices") else {
        return std::ptr::null();
    };
    if bytes.len() % 4 != 0 {
        set_last_message("Shuffled training indices length is not a multiple of u32 size.");
        return std::ptr::null();
    }
    let mut indices = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        indices.push(u32::from_le_bytes(chunk.try_into().unwrap()) as usize);
    }
    match crate::gpu::training::shuffled_indices_bytes(&indices, epoch, seed) {
        Ok(bytes) => set_output_bytes(bytes),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing training
/// samples in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_split_validation_samples_json(
    ptr: *const u8,
    len: usize,
    validation_split: f32,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Training validation split samples") else {
        return std::ptr::null();
    };
    let samples = match serde_json::from_str::<Vec<crate::gpu::training::TrainingSample>>(text) {
        Ok(samples) => samples,
        Err(error) => {
            set_last_message(&format!(
                "Training validation split samples are not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    match serde_json::to_string(&crate::gpu::training::split_validation_samples(
        &samples,
        validation_split,
    )) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&format!(
                "failed to encode training validation split: {error}"
            ));
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing
/// `samples`, `policyIndices`, and `split` fields in this WASM instance for the
/// duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_split_policy_training_indices_json(
    ptr: *const u8,
    len: usize,
    validation_split: f32,
) -> *const u8 {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Request {
        samples: Vec<crate::gpu::training::TrainingSample>,
        policy_indices: Vec<usize>,
        split: crate::gpu::training::ValidationSplit,
    }

    let Some(text) = wasm_input_text(ptr, len, "Policy training split request") else {
        return std::ptr::null();
    };
    let request = match serde_json::from_str::<Request>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "Policy training split request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    match serde_json::to_string(&crate::gpu::training::split_policy_training_indices(
        &request.samples,
        &request.policy_indices,
        &request.split,
        validation_split,
    )) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&format!("failed to encode policy training split: {error}"));
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing `samples`
/// and `indices` fields in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_unique_training_position_count_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Request {
        samples: Vec<crate::gpu::training::TrainingSample>,
        indices: Vec<usize>,
    }

    let Some(text) = wasm_input_text(ptr, len, "Unique training position count request") else {
        return std::ptr::null();
    };
    let request = match serde_json::from_str::<Request>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "Unique training position count request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    set_output(
        crate::gpu::training::unique_training_position_count(&request.samples, &request.indices)
            .to_string(),
    )
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing `samples`
/// and `indices` fields in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_group_training_indices_by_position_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Request {
        samples: Vec<crate::gpu::training::TrainingSample>,
        indices: Vec<usize>,
    }

    let Some(text) = wasm_input_text(ptr, len, "Training position group request") else {
        return std::ptr::null();
    };
    let request = match serde_json::from_str::<Request>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "Training position group request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    match serde_json::to_string(&crate::gpu::training::group_training_indices_by_position(
        &request.samples,
        &request.indices,
    )) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&format!(
                "failed to encode training position groups: {error}"
            ));
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing training
/// samples in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_feature_length_json(ptr: *const u8, len: usize) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Training feature length samples") else {
        return std::ptr::null();
    };
    let samples = match serde_json::from_str::<Vec<crate::gpu::training::TrainingSample>>(text) {
        Ok(samples) => samples,
        Err(error) => {
            set_last_message(&format!(
                "Training feature length samples are not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    match crate::gpu::training::feature_length(&samples) {
        Ok(length) => set_output(length.to_string()),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing training
/// samples in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_sparse_projection_features_bytes(
    ptr: *const u8,
    len: usize,
    input_size: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Sparse projection feature samples") else {
        return std::ptr::null();
    };
    let samples = match serde_json::from_str::<Vec<crate::gpu::training::TrainingSample>>(text) {
        Ok(samples) => samples,
        Err(error) => {
            set_last_message(&format!(
                "Sparse projection feature samples are not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    match crate::gpu::training::sparse_projection_features_bytes(
        &samples,
        if input_size == 0 {
            None
        } else {
            Some(input_size)
        },
    ) {
        Ok(bytes) => set_output_bytes(bytes),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` readable grouped-batch request bytes in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_fill_grouped_training_batch_indices_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Grouped training batch request") else {
        return std::ptr::null();
    };
    match crate::gpu::training::fill_grouped_training_batch_indices_bytes(bytes) {
        Ok(response) => set_output_bytes(response),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing training
/// samples in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_policy_training_indices_bytes(
    ptr: *const u8,
    len: usize,
    require_positive_weight: i32,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Policy training index samples") else {
        return std::ptr::null();
    };
    let samples = match serde_json::from_str::<Vec<crate::gpu::training::TrainingSample>>(text) {
        Ok(samples) => samples,
        Err(error) => {
            set_last_message(&format!(
                "Policy training index samples are not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    let mut indexes = Vec::new();
    for index in
        crate::gpu::training::policy_training_indices(&samples, require_positive_weight != 0)
    {
        let Ok(index) = i32::try_from(index) else {
            set_last_message("Policy training index exceeds i32 range.");
            return std::ptr::null();
        };
        indexes.push(index);
    }
    set_output_i32s(indexes)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a training
/// sample in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_has_policy_training_target_json(
    ptr: *const u8,
    len: usize,
) -> i32 {
    let Some(text) = wasm_input_text(ptr, len, "Policy training target sample") else {
        return 0;
    };
    let sample = match serde_json::from_str::<crate::gpu::training::TrainingSample>(text) {
        Ok(sample) => sample,
        Err(error) => {
            set_last_message(&format!(
                "Policy training target sample is not valid JSON: {error}"
            ));
            return 0;
        }
    };
    i32::from(crate::gpu::training::has_policy_training_target(&sample))
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing training
/// samples in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_auxiliary_value_targets_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Auxiliary value target samples") else {
        return std::ptr::null();
    };
    let samples = match serde_json::from_str::<Vec<crate::gpu::training::TrainingSample>>(text) {
        Ok(samples) => samples,
        Err(error) => {
            set_last_message(&format!(
                "Auxiliary value target samples are not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    set_output_bytes(crate::gpu::training::auxiliary_value_targets_bytes(
        &samples,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_policy_training_steps(value_epochs: usize) -> usize {
    crate::gpu::training::policy_training_steps(value_epochs)
}

#[no_mangle]
pub extern "C" fn chronofish_policy_training_target(policy: u32) -> usize {
    crate::gpu::training::policy_training_target(policy)
}

#[no_mangle]
pub extern "C" fn chronofish_training_label_weight(label_weight: f32) -> f32 {
    crate::gpu::training::training_label_weight(label_weight)
}

#[no_mangle]
pub extern "C" fn chronofish_training_weighted_average(total: f64, total_weight: f64) -> f64 {
    crate::gpu::training::training_weighted_average(total, total_weight)
}

#[no_mangle]
pub extern "C" fn chronofish_training_batch_normalization(batch_weight: f64) -> f64 {
    crate::gpu::training::training_batch_normalization(batch_weight)
}

#[no_mangle]
pub extern "C" fn chronofish_value_training_batch_size(
    config_batch_size: usize,
    training_count: usize,
) -> usize {
    crate::gpu::training::value_training_batch_size(config_batch_size, training_count)
}

#[no_mangle]
pub extern "C" fn chronofish_policy_training_batch_size(
    config_batch_size: usize,
    training_count: usize,
) -> usize {
    crate::gpu::training::policy_training_batch_size(config_batch_size, training_count)
}

#[no_mangle]
pub extern "C" fn chronofish_value_head_validation_interval(
    epochs: usize,
    validation_interval: i32,
) -> usize {
    crate::gpu::training::value_head_validation_interval(
        epochs,
        usize::try_from(validation_interval).ok(),
    )
}

#[no_mangle]
pub extern "C" fn chronofish_value_gpu_batches_per_submit(epochs: usize) -> usize {
    crate::gpu::training::value_gpu_batches_per_submit(epochs)
}

#[no_mangle]
pub extern "C" fn chronofish_value_gpu_validation_interval(
    batches_per_submit: usize,
    validation_interval: i32,
) -> usize {
    crate::gpu::training::value_gpu_validation_interval(
        batches_per_submit,
        usize::try_from(validation_interval).ok(),
    )
}

#[no_mangle]
pub extern "C" fn chronofish_policy_training_steps_per_submit(steps: usize) -> usize {
    crate::gpu::training::policy_training_steps_per_submit(steps)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON with `buffer` and
/// `samples` arrays in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_append_replay_samples_json(
    ptr: *const u8,
    len: usize,
    max_buffer: usize,
) -> *const u8 {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct ReplayAppendRequest {
        buffer: Vec<crate::gpu::training::TrainingSample>,
        samples: Vec<crate::gpu::training::TrainingSample>,
    }

    let Some(text) = wasm_input_text(ptr, len, "Training replay append request") else {
        return std::ptr::null();
    };
    let request = match serde_json::from_str::<ReplayAppendRequest>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "Training replay append request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    encode_training_samples_json(crate::gpu::training::append_replay_samples(
        &request.buffer,
        &request.samples,
        max_buffer.max(1),
    ))
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing training
/// samples in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_label_source_counts_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Training sample label counts") else {
        return std::ptr::null();
    };
    let samples = match serde_json::from_str::<Vec<crate::gpu::training::TrainingSample>>(text) {
        Ok(samples) => samples,
        Err(error) => {
            set_last_message(&format!("Training samples are not valid JSON: {error}"));
            return std::ptr::null();
        }
    };
    match serde_json::to_string(&crate::gpu::training::label_source_counts(&samples)) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&format!("failed to encode label source counts: {error}"));
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing an outcome
/// sample relabel request in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_relabel_outcome_samples_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct RelabelRequest {
        samples: Vec<serde_json::Value>,
        kind: String,
        winner: Option<String>,
        label_kind: Option<String>,
        label_weight: Option<f32>,
    }

    let Some(text) = wasm_input_text(ptr, len, "Training outcome relabel request") else {
        return std::ptr::null();
    };
    let request = match serde_json::from_str::<RelabelRequest>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "Training outcome relabel request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    let mut samples = match request
        .samples
        .iter()
        .map(|value| serde_json::from_value::<crate::gpu::training::TrainingSample>(value.clone()))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(samples) => samples,
        Err(error) => {
            set_last_message(&format!(
                "Training outcome samples are not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    match request.kind.as_str() {
        "outcome" => {
            let Some(winner) = request.winner.as_deref() else {
                set_last_message("Outcome relabel requires a winner.");
                return std::ptr::null();
            };
            let max_ply = request
                .samples
                .iter()
                .filter_map(|value| value.get("ply").and_then(serde_json::Value::as_u64))
                .max()
                .unwrap_or(0) as usize;
            for (sample, source) in samples.iter_mut().zip(request.samples.iter()) {
                let outcome_turn = source
                    .get("outcomeTurn")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default();
                let ply = source
                    .get("ply")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as usize;
                if let Err(error) = crate::gpu::training::apply_outcome_label(
                    sample,
                    winner,
                    outcome_turn,
                    ply,
                    max_ply,
                ) {
                    set_last_message(&error);
                    return std::ptr::null();
                }
                if let Some(label_kind) = request.label_kind.as_ref() {
                    sample.label_kind = Some(label_kind.clone());
                }
                if let Some(label_weight) = request.label_weight {
                    sample.label_weight = label_weight;
                }
            }
            encode_training_samples_json(samples)
        }
        "draw" => {
            let label_kind = request.label_kind.as_deref().unwrap_or("outcome");
            let label_weight = request.label_weight.unwrap_or(1.0);
            for sample in &mut samples {
                crate::gpu::training::apply_draw_label(sample, label_kind, label_weight);
            }
            encode_training_samples_json(samples)
        }
        "partial" => {
            encode_training_samples_json(crate::gpu::training::samples_from_partial_outcome(
                &samples,
                request.label_kind.as_deref(),
                request.label_weight,
            ))
        }
        other => {
            set_last_message(&format!("Unknown outcome relabel kind `{other}`."));
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing distilled
/// training samples and labels in this WASM instance for the duration of the
/// call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_distill_training_samples_with_labels_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Distilled training sample request") else {
        return std::ptr::null();
    };
    match crate::gpu::training::distill_training_samples_with_labels_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a search
/// result label sample request in this WASM instance for the duration of the
/// call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_search_result_label_sample_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Search result label sample request") else {
        return std::ptr::null();
    };
    match crate::gpu::training::search_result_label_sample_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a search
/// result label sample-from-result request in this WASM instance for the
/// duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_search_result_label_sample_from_result_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Search result label sample-from-result request")
    else {
        return std::ptr::null();
    };
    match crate::gpu::training::search_result_label_sample_from_result_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a worker
/// search result in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_search_result_turn_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Search result turn request") else {
        return std::ptr::null();
    };
    match crate::gpu::training::search_result_turn_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing training
/// mode options in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_normalize_training_modes_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct NormalizeTrainingModesRequest {
        training_subject: Option<String>,
        training_modes: Option<Vec<String>>,
        training_target: Option<String>,
        cpu_training_target: Option<String>,
        label_mode: Option<String>,
    }

    #[derive(serde::Serialize)]
    #[serde(rename_all = "camelCase")]
    struct NormalizeTrainingModesResponse {
        training_subject: String,
        training_modes: Vec<String>,
    }

    let Some(text) = wasm_input_text(ptr, len, "Training mode normalization request") else {
        return std::ptr::null();
    };
    let request = match serde_json::from_str::<NormalizeTrainingModesRequest>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "Training mode normalization request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    let training_subject = request
        .training_subject
        .as_deref()
        .filter(|subject| crate::gpu::training::is_training_subject(subject))
        .unwrap_or_else(|| {
            crate::gpu::training::legacy_training_subject(request.training_target.as_deref())
        });
    let explicit_modes = request
        .training_modes
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let response = NormalizeTrainingModesResponse {
        training_subject: training_subject.to_string(),
        training_modes: crate::gpu::training::normalize_training_modes(
            &explicit_modes,
            training_subject,
            request.training_target.as_deref(),
            request.cpu_training_target.as_deref(),
            request.label_mode.as_deref(),
        ),
    };
    match serde_json::to_string(&response) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&format!(
                "failed to encode normalized training modes: {error}"
            ));
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing training
/// mode policy options in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_training_mode_policy_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TrainingModePolicyRequest {
        training_subject: String,
        training_modes: Vec<String>,
        mode: Option<String>,
    }

    let Some(text) = wasm_input_text(ptr, len, "Training mode policy request") else {
        return std::ptr::null();
    };
    let request = match serde_json::from_str::<TrainingModePolicyRequest>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "Training mode policy request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    let response = serde_json::json!({
        "trainingModeCount": crate::gpu::training::training_mode_count(
            &request.training_subject,
            &request.training_modes,
        ),
        "cpuBaselineModeEnabled": crate::gpu::training::cpu_baseline_mode_enabled(
            &request.training_modes,
        ),
        "modeEnabled": request.mode.as_deref().map(|mode| {
            crate::gpu::training::training_mode_enabled(&request.training_modes, mode)
        }),
    });
    set_output(response.to_string())
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing partial
/// training config options in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_normalize_training_config_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Training config normalization request") else {
        return std::ptr::null();
    };
    let request = match serde_json::from_str::<serde_json::Value>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "Training config normalization request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    let training_subject = training_subject_from_config(&request);
    let explicit_modes = string_array_field(&request, "trainingModes");
    let training_modes = crate::gpu::training::normalize_training_modes(
        &explicit_modes
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        &training_subject,
        string_field(&request, "trainingTarget").as_deref(),
        string_field(&request, "cpuTrainingTarget").as_deref(),
        string_field(&request, "labelMode").as_deref(),
    );
    let response = serde_json::json!({
        "trainingSubject": training_subject,
        "trainingModes": training_modes,
        "learningRate": clamp_config_number(&request, "learningRate", 0.0001, 0.1, 0.01),
        "samples": clamp_config_integer(&request, "samples", 1, crate::gpu::training::MAX_GPU_TRAINING_SAMPLES as i64, 64),
        "selfPlayWorkers": clamp_config_integer(&request, "selfPlayWorkers", 1, 16, 2),
        "searchWorkers": clamp_config_integer(&request, "searchWorkers", 1, 16, 2),
        "explorationTemperature": clamp_config_number(&request, "explorationTemperature", 0.0, 2.0, 0.25),
        "depth": clamp_config_integer(&request, "depth", 1, 8, 5),
        "nodes": clamp_config_integer(&request, "nodes", 1, 131_072, 16_384),
        "epochs": clamp_config_integer(&request, "epochs", 1, 65_536, 8_192),
        "maxBuffer": clamp_config_integer(&request, "maxBuffer", 16, 16_384, 4_096),
        "batchSize": clamp_config_integer(&request, "batchSize", 16, crate::gpu::training::MAX_GPU_TRAINING_BATCH as i64, crate::gpu::training::DEFAULT_BATCH_SIZE as i64),
        "validationSplit": clamp_config_number(&request, "validationSplit", 0.0, 0.3, crate::gpu::training::DEFAULT_VALIDATION_SPLIT as f64),
        "validationInterval": clamp_config_integer(&request, "validationInterval", 16, crate::gpu::training::MAX_GPU_VALIDATION_INTERVAL as i64, 256),
        "patience": clamp_config_integer(&request, "patience", 1, 64, crate::gpu::training::DEFAULT_PATIENCE as i64),
        "weightDecay": clamp_config_number(&request, "weightDecay", 0.0, 0.01, crate::gpu::training::DEFAULT_WEIGHT_DECAY as f64),
        "lossLogReplay": clamp_config_integer(&request, "lossLogReplay", 0, 32, 4),
        "cpuDepth": clamp_config_integer(&request, "cpuDepth", 1, 16, 4),
        "cpuNodes": clamp_config_integer(&request, "cpuNodes", 1, 131_072, 8_192),
        "cpuTrainingTimeMs": clamp_config_integer(&request, "cpuTrainingTimeMs", 1, 600_000, 10_000),
        "cpuCandidates": clamp_config_integer(&request, "cpuCandidates", 1, 256, 8),
        "cpuFinalists": clamp_config_integer(&request, "cpuFinalists", 1, 64, 1),
        "cpuPairBatch": clamp_config_integer(&request, "cpuPairBatch", 1, 64, 4),
        "cpuOpponentVariants": clamp_config_integer(&request, "cpuOpponentVariants", 1, 128, 8),
        "cpuScreeningOpponentVariants": clamp_config_integer(&request, "cpuScreeningOpponentVariants", 1, 128, 2),
        "cpuRoundsPerVariant": clamp_config_integer(&request, "cpuRoundsPerVariant", 1, 64, 1),
        "cpuHallOfFameEntries": clamp_config_integer(&request, "cpuHallOfFameEntries", 0, 64, 1),
        "cpuLeagueContenders": clamp_config_integer(&request, "cpuLeagueContenders", 1, 64, 2),
        "cpuLeagueHallOfFameEntries": clamp_config_integer(&request, "cpuLeagueHallOfFameEntries", 0, 64, 2),
        "cpuMinPairs": clamp_config_integer(&request, "cpuMinPairs", 1, 256, 2),
        "cpuMaxPairs": clamp_config_integer(&request, "cpuMaxPairs", 1, 512, 8),
        "cpuDrawWindow": clamp_config_integer(&request, "cpuDrawWindow", 1, 128, 4),
        "cpuDrawRateLimit": clamp_config_number(&request, "cpuDrawRateLimit", 0.0, 1.0, 0.8),
        "cpuMaxMatchPlies": clamp_config_integer(&request, "cpuMaxMatchPlies", 1, 512, 40),
        "cpuMaxMatchTimeMs": clamp_config_integer(&request, "cpuMaxMatchTimeMs", 0, 3_600_000, 0),
        "cpuMaxGenerationsWithoutCandidate": clamp_config_integer(&request, "cpuMaxGenerationsWithoutCandidate", 1, 256, 2),
        "cpuWorkers": clamp_config_integer(&request, "cpuWorkers", 1, 32, 16),
        "cpuTrainSeconds": clamp_config_integer(&request, "cpuTrainSeconds", 1, 86_400, 3_600),
    });
    match serde_json::to_string(&response) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&format!(
                "failed to encode normalized training config: {error}"
            ));
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_training_worker_count(
    total: usize,
    requested_workers: usize,
) -> usize {
    crate::gpu::training::gpu_training_worker_count(total, requested_workers)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_duel_training_worker_count(
    total: usize,
    search_workers: usize,
    self_play_workers: usize,
) -> usize {
    crate::gpu::training::gpu_duel_training_worker_count(total, search_workers, self_play_workers)
}

#[no_mangle]
pub extern "C" fn chronofish_training_label_worker_count(
    job_count: usize,
    requested_workers: i32,
    hardware_cores: usize,
) -> usize {
    let requested_workers = usize::try_from(requested_workers).ok();
    crate::gpu::training::training_label_worker_count(job_count, requested_workers, hardware_cores)
}

#[no_mangle]
pub extern "C" fn chronofish_training_split_work_json(total: usize, workers: usize) -> *const u8 {
    let splits = crate::gpu::training::split_work(total, workers);
    match serde_json::to_string(&splits) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&format!("failed to encode training work split: {error}"));
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing an array of
/// training-sample arrays in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_take_training_sample_batches_json(
    ptr: *const u8,
    len: usize,
    target: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Training sample batches request") else {
        return std::ptr::null();
    };
    let batches = match serde_json::from_str::<Vec<Vec<crate::gpu::training::TrainingSample>>>(text)
    {
        Ok(batches) => batches,
        Err(error) => {
            set_last_message(&format!(
                "Training sample batches request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    encode_training_samples_json(crate::gpu::training::take_training_sample_batches(
        &batches, target,
    ))
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing an array of
/// nullable training samples in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_compact_training_samples_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Compact training samples request") else {
        return std::ptr::null();
    };
    let samples =
        match serde_json::from_str::<Vec<Option<crate::gpu::training::TrainingSample>>>(text) {
            Ok(samples) => samples,
            Err(error) => {
                set_last_message(&format!(
                    "Compact training samples request is not valid JSON: {error}"
                ));
                return std::ptr::null();
            }
        };
    encode_training_samples_json(crate::gpu::training::compact_training_samples(&samples))
}

#[no_mangle]
pub extern "C" fn chronofish_training_sample_plies(index: usize, encode_only: i32) -> usize {
    crate::gpu::training::sample_plies(index, encode_only != 0)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 seed prefix text in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_training_sample_seed(
    ptr: *const u8,
    len: usize,
    index: u32,
    salt: u32,
) -> u32 {
    let Some(prefix) = wasm_input_text(ptr, len, "Training sample seed prefix") else {
        return 0;
    };
    crate::gpu::training::sample_seed(prefix, index, salt)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON text in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_training_search_seed_json(
    ptr: *const u8,
    len: usize,
    salt: u32,
) -> u32 {
    let Some(text) = wasm_input_text(ptr, len, "Training search seed JSON") else {
        return 0;
    };
    crate::gpu::training::search_seed_json(Some(text), salt)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_warmup_plies(worker_index: usize) -> usize {
    crate::gpu::training::gpu_warmup_plies(worker_index)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_rollout_max_plies(target: usize, worker_index: usize) -> usize {
    crate::gpu::training::gpu_rollout_max_plies(target, worker_index)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_rollout_ply_offset(ply: usize, worker_index: usize) -> usize {
    crate::gpu::training::gpu_rollout_ply_offset(ply, worker_index)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_warmup_search_config_json(
    depth: i32,
    nodes: i32,
    search_time_ms: i32,
    exploration_temperature: f32,
) -> *const u8 {
    let config = crate::gpu::training::gpu_warmup_search_config(
        depth,
        nodes,
        search_time_ms.max(0) as u64,
        exploration_temperature,
    );
    let response = serde_json::json!({
        "depth": config.depth,
        "nodes": config.nodes,
        "timeMs": config.time_ms,
        "explorationTemperature": config.exploration_temperature,
    });
    set_output(response.to_string())
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_position_generation_search_config_json(
    depth: i32,
    nodes: i32,
    exploration_temperature: f32,
) -> *const u8 {
    let config = crate::gpu::training::gpu_position_generation_search_config(
        depth,
        nodes,
        exploration_temperature,
    );
    let response = serde_json::json!({
        "depth": config.depth,
        "nodes": config.nodes,
        "timeMs": config.time_ms,
        "explorationTemperature": config.exploration_temperature,
    });
    set_output(response.to_string())
}

#[no_mangle]
pub extern "C" fn chronofish_curriculum_search_config_json(
    depth: i32,
    nodes: i32,
    exploration_temperature: f32,
    index: usize,
) -> *const u8 {
    let config = crate::gpu::training::curriculum_search_config(
        depth,
        nodes,
        exploration_temperature,
        index,
    );
    let response = serde_json::json!({
        "depth": config.depth,
        "nodes": config.nodes,
        "explorationTemperature": config.exploration_temperature,
    });
    set_output(response.to_string())
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 snapshot JSON in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_curriculum_game_snapshot_json(
    ptr: *const u8,
    len: usize,
    index: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Curriculum game snapshot") else {
        return std::ptr::null();
    };
    match crate::gpu::training::curriculum_game_snapshot_json(text, index) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_tactical_search_config_json(
    depth: i32,
    nodes: i32,
    exploration_temperature: f32,
    attempt: usize,
) -> *const u8 {
    let config = crate::gpu::training::tactical_search_config(
        depth,
        nodes,
        exploration_temperature,
        attempt,
    );
    let response = serde_json::json!({
        "depth": config.depth,
        "nodes": config.nodes,
        "explorationTemperature": config.exploration_temperature,
    });
    set_output(response.to_string())
}

#[no_mangle]
pub extern "C" fn chronofish_tactical_position_attempt_count(index: usize) -> usize {
    crate::gpu::training::tactical_position_attempt_count(index)
}

#[no_mangle]
pub extern "C" fn chronofish_tactical_position_use_best_source(best_priority: i32) -> i32 {
    i32::from(crate::gpu::training::tactical_position_use_best_source(
        best_priority,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_tactical_position_selection_json(
    best_priority: i32,
    generated_priority: i32,
) -> *const u8 {
    match crate::gpu::training::tactical_position_selection_json(best_priority, generated_priority)
    {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 snapshot JSON in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_tactical_position_priority_snapshot_json(
    ptr: *const u8,
    len: usize,
) -> i32 {
    let Some(text) = wasm_input_text(ptr, len, "Tactical position priority snapshot") else {
        return 0;
    };
    match crate::gpu::training::tactical_position_priority_snapshot_json(text) {
        Ok(priority) => priority,
        Err(error) => {
            set_last_message(&error);
            0
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing `before`,
/// `after`, and `mover` fields in this WASM instance for the duration of the
/// call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_royal_capture_winner_snapshot_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    #[derive(serde::Deserialize)]
    struct RoyalCaptureWinnerRequest {
        before: String,
        after: String,
        mover: String,
    }

    let Some(text) = wasm_input_text(ptr, len, "Royal capture winner request") else {
        return std::ptr::null();
    };
    let request = match serde_json::from_str::<RoyalCaptureWinnerRequest>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "Royal capture winner request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    match crate::gpu::training::royal_capture_winner_snapshot_json(
        &request.before,
        &request.after,
        &request.mover,
    )
    .and_then(|winner| {
        serde_json::to_string(&winner)
            .map_err(|error| format!("failed to encode royal capture winner: {error}"))
    }) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_training_worker_request_timeout_ms(nodes: i32, time_ms: i32) -> i32 {
    wasm_milliseconds(crate::gpu::training::worker_request_timeout_ms(
        i64::from(nodes),
        i64::from(time_ms),
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_training_worker_search_time_ms(nodes: i32, time_ms: i32) -> i32 {
    wasm_milliseconds(crate::gpu::training::worker_search_time_ms(
        i64::from(nodes),
        i64::from(time_ms),
    ))
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 training worker timeout
/// request JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_training_worker_request_timeout_ms_json(
    ptr: *const u8,
    len: usize,
) -> i32 {
    let Some(text) = wasm_input_text(ptr, len, "training worker timeout request") else {
        return 0;
    };
    match crate::gpu::training::worker_request_timeout_ms_json(text) {
        Ok(value) => wasm_milliseconds(value),
        Err(error) => {
            set_last_message(&error);
            0
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 training worker timeout
/// request JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_training_worker_search_time_ms_json(
    ptr: *const u8,
    len: usize,
) -> i32 {
    let Some(text) = wasm_input_text(ptr, len, "training worker search timeout request") else {
        return 0;
    };
    match crate::gpu::training::worker_search_time_ms_json(text) {
        Ok(value) => wasm_milliseconds(value),
        Err(error) => {
            set_last_message(&error);
            0
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 loss-log replay request
/// JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_loss_log_replay_logs_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "loss-log replay request") else {
        return std::ptr::null();
    };
    match crate::gpu::training::loss_log_replay_logs_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 loss-log validation update
/// request JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_loss_log_validation_update_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "loss-log validation update request") else {
        return std::ptr::null();
    };
    match crate::gpu::training::loss_log_validation_update_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 training metrics summary
/// request JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_training_metrics_summary_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "training metrics summary request") else {
        return std::ptr::null();
    };
    match crate::gpu::training::training_metrics_summary_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_normalized_search_score(score: i32) -> f32 {
    crate::gpu::training::normalized_search_score(score)
}

#[no_mangle]
pub extern "C" fn chronofish_denormalized_search_score(value: f32) -> i32 {
    crate::gpu::training::denormalized_search_score(value)
}

#[no_mangle]
pub extern "C" fn chronofish_bounded_value(value: f32) -> f32 {
    crate::gpu::training::bounded_value(value)
}

#[no_mangle]
pub extern "C" fn chronofish_inverse_tanh(value: f32) -> f32 {
    crate::gpu::training::inverse_tanh(value)
}

#[no_mangle]
pub extern "C" fn chronofish_optimizer_velocity(
    previous: f32,
    gradient: f32,
    momentum: f32,
) -> f32 {
    crate::gpu::training::optimizer_velocity(previous, gradient, momentum)
}

#[no_mangle]
pub extern "C" fn chronofish_loss_reduction_workgroup_count(sample_count: usize) -> usize {
    crate::gpu::training::loss_reduction_workgroup_count(sample_count)
}

#[no_mangle]
pub extern "C" fn chronofish_training_workgroups_16(item_count: usize) -> usize {
    crate::gpu::training::training_workgroups_16(item_count)
}

#[no_mangle]
pub extern "C" fn chronofish_training_workgroups_64(item_count: usize) -> usize {
    crate::gpu::training::training_workgroups_64(item_count)
}

#[no_mangle]
pub extern "C" fn chronofish_align4(value: usize) -> usize {
    crate::gpu::training::align4(value)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_prediction_max_batch() -> usize {
    crate::gpu::training::cpu_prediction_max_batch()
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_head_training_max_positions() -> usize {
    crate::gpu::training::cpu_head_training_max_positions()
}

#[no_mangle]
pub extern "C" fn chronofish_min_hidden_training_positions() -> usize {
    crate::gpu::training::min_hidden_training_positions()
}

#[no_mangle]
pub extern "C" fn chronofish_projection_chunk_size() -> usize {
    crate::gpu::training::projection_chunk_size()
}

#[no_mangle]
pub extern "C" fn chronofish_projection_temporary_budget(max_buffer_size: usize) -> usize {
    crate::gpu::training::projection_temporary_budget(max_buffer_size)
}

#[no_mangle]
pub unsafe extern "C" fn chronofish_dense_kernel_entry_point_bytes(
    ptr: *const u8,
    len: usize,
    sample_count: usize,
) -> *const u8 {
    let Some(entry_point) = wasm_input_text(ptr, len, "Dense kernel entry point") else {
        return std::ptr::null();
    };
    set_output_bytes(
        crate::gpu::training::dense_kernel_entry_point(entry_point, sample_count).into_bytes(),
    )
}

#[no_mangle]
pub extern "C" fn chronofish_projection_hash(
    raw_index: u32,
    projection_index: u32,
    seed: u32,
) -> u32 {
    crate::gpu::training::projection_hash(raw_index, projection_index, seed)
}

#[no_mangle]
pub extern "C" fn chronofish_default_output_layer_size() -> usize {
    crate::gpu::training::output_layer_size(crate::gpu::training::DEFAULT_HIDDEN_LAYERS)
        .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn chronofish_default_previous_layer_size(
    layer_index: usize,
    input_size: usize,
) -> usize {
    crate::gpu::training::previous_layer_size(
        crate::gpu::training::DEFAULT_HIDDEN_LAYERS,
        layer_index,
        input_size,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_default_initial_hidden_weights_bytes() -> *const u8 {
    let weights = crate::gpu::training::default_initial_hidden_weights();
    let mut bytes = Vec::with_capacity(weights.len() * std::mem::size_of::<f32>());
    for value in weights {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    set_output_bytes(bytes)
}

/// # Safety
///
/// `ptr` must point to `len` readable initial hidden-weight request bytes in
/// this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_initial_hidden_weights_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Initial hidden weight request") else {
        return std::ptr::null();
    };
    match crate::gpu::training::initial_hidden_weights_bytes(bytes) {
        Ok(response) => set_output_bytes(response),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` readable hidden-weight split request bytes in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_split_hidden_weights_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Hidden weight split request") else {
        return std::ptr::null();
    };
    match crate::gpu::training::split_hidden_weights_bytes(bytes) {
        Ok(response) => set_output_bytes(response),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` readable Float32 concat request bytes in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_concat_f32_bytes(ptr: *const u8, len: usize) -> *const u8 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Float32 concat request") else {
        return std::ptr::null();
    };
    match crate::gpu::training::concat_f32_bytes(bytes) {
        Ok(response) => set_output_bytes(response),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` readable Float32 bytes in this WASM instance for
/// the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_count_non_zero_f32_bytes(ptr: *const u8, len: usize) -> usize {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Non-zero count request") else {
        return 0;
    };
    match crate::gpu::training::count_non_zero_bytes(bytes) {
        Ok(count) => count,
        Err(error) => {
            set_last_message(&error);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_output_delta_params_bytes(
    sample_count: usize,
    total_weight: f32,
) -> *const u8 {
    set_output_bytes(crate::gpu::training::output_delta_params_bytes(
        sample_count,
        total_weight,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_hidden_delta_params_bytes(
    sample_count: usize,
    current_size: usize,
    next_size: usize,
) -> *const u8 {
    set_output_bytes(crate::gpu::training::hidden_delta_params_bytes(
        sample_count,
        current_size,
        next_size,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_policy_params_bytes(
    batch_count: usize,
    input_size: usize,
    total_weight: f32,
    learning_rate: f32,
    weight_decay: f32,
    momentum: f32,
) -> *const u8 {
    set_output_bytes(crate::gpu::training::policy_params_bytes(
        batch_count,
        input_size,
        total_weight,
        learning_rate,
        weight_decay,
        momentum,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_layer_params_bytes(
    sample_count: usize,
    input_size: usize,
    output_size: usize,
    learning_rate: f32,
    weight_decay: f32,
    momentum: f32,
) -> *const u8 {
    set_output_bytes(crate::gpu::training::layer_params_bytes(
        sample_count,
        input_size,
        output_size,
        learning_rate,
        weight_decay,
        momentum,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_output_params_bytes(
    sample_count: usize,
    input_size: usize,
    learning_rate: f32,
    weight_decay: f32,
    momentum: f32,
) -> *const u8 {
    set_output_bytes(crate::gpu::training::output_params_bytes(
        sample_count,
        input_size,
        learning_rate,
        weight_decay,
        momentum,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_projection_params_bytes(
    sample_count: usize,
    input_size: usize,
    projection_size: usize,
    seed: u32,
    output_offset: usize,
) -> *const u8 {
    set_output_bytes(crate::gpu::training::projection_params_bytes(
        sample_count,
        input_size,
        projection_size,
        seed,
        output_offset,
    ))
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 color text in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_opposite_color_json(ptr: *const u8, len: usize) -> *const u8 {
    let Some(color) = wasm_input_text(ptr, len, "Opposite color request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_search_opposite_color(color) {
        Ok(opposite) => set_output(format!("\"{opposite}\"")),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 color text in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_search_color_code_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(color) = wasm_input_text(ptr, len, "GPU search color code request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_search_color_code(color) {
        Ok(code) => set_output(code.to_string()),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_training_label_policy_json() -> *const u8 {
    let response = serde_json::json!({
        "outcomeLabelWeight": crate::gpu::training::OUTCOME_LABEL_WEIGHT,
        "duelLabelWeight": crate::gpu::training::DUEL_LABEL_WEIGHT,
        "duelDrawLabelWeight": crate::gpu::training::DUEL_DRAW_LABEL_WEIGHT,
        "distilledLabelWeight": crate::gpu::training::DISTILLED_LABEL_WEIGHT,
        "defaultPartialOutcomeLabelKind": crate::gpu::training::DEFAULT_PARTIAL_OUTCOME_LABEL_KIND,
        "defaultPartialOutcomeLabelWeight": crate::gpu::training::DEFAULT_PARTIAL_OUTCOME_LABEL_WEIGHT,
    });
    set_output(response.to_string())
}

/// # Safety
///
/// `ptr` must point to `len` readable UTF-8 label-kind bytes in this WASM
/// instance for the duration of the call. Empty text is treated as no label
/// kind.
#[no_mangle]
pub unsafe extern "C" fn chronofish_training_label_priority(
    ptr: *const u8,
    len: usize,
    pseudo: i32,
) -> f32 {
    let Some(label_kind) = wasm_input_text(ptr, len, "Training label priority kind") else {
        return 0.0;
    };
    crate::gpu::training::training_label_priority(
        (!label_kind.is_empty()).then_some(label_kind),
        pseudo != 0,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_policy_bucket_from_move_values(
    from_timeline_id: i32,
    from_time: i32,
    from_x: i32,
    from_y: i32,
    to_timeline_id: i32,
    to_time: i32,
    to_x: i32,
    to_y: i32,
    intent: i32,
) -> u32 {
    crate::gpu::training::policy_bucket_from_move_values(
        from_timeline_id,
        from_time,
        from_x,
        from_y,
        to_timeline_id,
        to_time,
        to_x,
        to_y,
        intent,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_bot_search_depth_at_least_one(depth: f64) -> i32 {
    crate::gpu::search::bot_search_depth_at_least_one(depth)
}

#[no_mangle]
pub extern "C" fn chronofish_bot_search_config_json(
    depth: f64,
    min_depth: f64,
    nodes: f64,
    time_ms: f64,
) -> *const u8 {
    match crate::gpu::search::bot_search_config_json(depth, min_depth, nodes, time_ms) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_worker_search_config_json(
    depth: f64,
    min_depth: f64,
    time_ms: f64,
) -> *const u8 {
    match crate::gpu::search::gpu_worker_search_config_json(depth, min_depth, time_ms) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_search_ranking_limit(nodes: f64) -> usize {
    crate::gpu::search::gpu_search_ranking_limit(nodes)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_search_reply_limit(nodes: f64) -> usize {
    crate::gpu::search::gpu_search_reply_limit(nodes)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_reply_pressure_reply_limit() -> usize {
    crate::gpu::search::gpu_reply_pressure_reply_limit()
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_search_validation_limit(nodes: f64) -> usize {
    crate::gpu::search::gpu_search_validation_limit(nodes)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 mutation
/// support request data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_supported_mutation_candidate_indexes_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(words) = wasm_input_i32s(ptr, len, "GPU supported mutation candidate request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_supported_mutation_candidate_indexes_from_i32s(&words) {
        Ok(indexes) => set_output_i32s(indexes),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 mutation-support JSON in
/// this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_supported_mutation_candidate_indexes_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU supported mutation candidate JSON request")
    else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_supported_mutation_candidate_indexes_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_mutation_status_is_terminal(status: i32) -> i32 {
    i32::from(crate::gpu::search::gpu_mutation_status_is_terminal(status))
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_full_search_reported_depth(requested_depth: i32) -> i32 {
    crate::gpu::search::gpu_full_search_reported_depth(requested_depth)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_completed_reply_should_search(
    royal_capture_present: i32,
    now_ms: f64,
    deadline_at_ms: f64,
) -> i32 {
    i32::from(crate::gpu::search::gpu_completed_reply_should_search(
        royal_capture_present != 0,
        now_ms,
        deadline_at_ms,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_frontier_cycle_should_stop(
    cycle: usize,
    cycles_completed: usize,
    requested_depth: usize,
    now_ms: f64,
    deadline_at_ms: f64,
) -> i32 {
    i32::from(crate::gpu::search::gpu_frontier_cycle_should_stop(
        cycle,
        cycles_completed,
        requested_depth,
        now_ms,
        deadline_at_ms,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_diagnostic_rate(numerator: f64, denominator: f64) -> f64 {
    crate::gpu::search::gpu_diagnostic_rate(numerator, denominator)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_effective_branching_factor(
    selected_count: f64,
    cycles_completed: f64,
) -> f64 {
    crate::gpu::search::gpu_effective_branching_factor(selected_count, cycles_completed)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_reported_latency_ms(latency_ms: f64) -> f64 {
    crate::gpu::search::gpu_reported_latency_ms(latency_ms)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_nodes_per_second(nodes: f64, latency_ms: f64) -> f64 {
    crate::gpu::search::gpu_nodes_per_second(nodes, latency_ms)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_search_nodes(nodes: f64) -> f64 {
    crate::gpu::search::gpu_search_nodes(nodes)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_accumulated_search_nodes(
    base_nodes: f64,
    extra_nodes: f64,
    fallback_nodes: f64,
) -> f64 {
    crate::gpu::search::gpu_accumulated_search_nodes(base_nodes, extra_nodes, fallback_nodes)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_mutation_candidate_limit(candidate_count: usize) -> usize {
    crate::gpu::search::gpu_mutation_candidate_limit(candidate_count)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_mutation_candidate_workgroups(candidate_limit: usize) -> usize {
    crate::gpu::search::gpu_mutation_candidate_workgroups(candidate_limit)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_turn_completion_max_moves(
    existing_moves: usize,
    timeline_count: usize,
) -> usize {
    crate::gpu::search::gpu_turn_completion_max_moves(existing_moves, timeline_count)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_candidate_max_dispatch_workgroups() -> usize {
    crate::gpu::search::gpu_candidate_max_dispatch_workgroups()
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_candidate_max_candidates_per_dispatch() -> usize {
    crate::gpu::search::gpu_candidate_max_candidates_per_dispatch()
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_candidate_max_candidates_per_batch(
    max_binding_size: usize,
) -> usize {
    crate::gpu::search::gpu_candidate_max_candidates_per_batch(max_binding_size)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_candidate_source_batch_size(
    max_candidates_per_batch: usize,
    target_count: usize,
) -> usize {
    crate::gpu::search::gpu_candidate_source_batch_size(max_candidates_per_batch, target_count)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_candidate_batch_source_count(
    source_count: usize,
    source_start: usize,
    source_batch_size: usize,
) -> usize {
    crate::gpu::search::gpu_candidate_batch_source_count(
        source_count,
        source_start,
        source_batch_size,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_candidate_batch_candidate_count(
    source_count: usize,
    target_count: usize,
) -> usize {
    crate::gpu::search::gpu_candidate_batch_candidate_count(source_count, target_count)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_candidate_score_workgroups(batch_candidate_count: usize) -> usize {
    crate::gpu::search::gpu_candidate_score_workgroups(batch_candidate_count)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_reply_score_workgroups_x(root_count: usize) -> usize {
    crate::gpu::search::gpu_reply_score_workgroups_x(root_count)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_reply_score_workgroups_y(reply_count: usize) -> usize {
    crate::gpu::search::gpu_reply_score_workgroups_y(reply_count)
}

#[no_mangle]
pub extern "C" fn chronofish_bot_next_search_depth(current_depth: i32, target_depth: i32) -> i32 {
    crate::gpu::search::bot_next_search_depth(current_depth, target_depth)
}

#[no_mangle]
pub extern "C" fn chronofish_bot_worker_search_time_ms(time_ms: i32) -> i32 {
    crate::gpu::search::bot_worker_search_time_ms(time_ms)
}

#[no_mangle]
pub extern "C" fn chronofish_bot_completed_search_depth(
    result_depth: f64,
    requested_depth: i32,
    result_ends_in_royal_capture: i32,
) -> i32 {
    crate::gpu::search::bot_completed_search_depth(
        result_depth,
        requested_depth,
        result_ends_in_royal_capture != 0,
    )
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 bot result JSON in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_bot_result_ends_in_royal_capture_json(
    ptr: *const u8,
    len: usize,
) -> i32 {
    let Some(text) = wasm_input_text(ptr, len, "Bot result terminal request") else {
        return 0;
    };
    match crate::gpu::search::bot_result_ends_in_royal_capture_json(text) {
        Ok(value) => i32::from(value),
        Err(error) => {
            set_last_message(&error);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_max_cycles(
    requested_depth: i32,
    timeline_count: usize,
) -> i32 {
    crate::gpu::search::frontier_max_cycles(requested_depth, timeline_count)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU frontier
/// orchestration JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_frontier_orchestration_plan_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU frontier orchestration request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::frontier_orchestration_plan_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_per_parent_limit(frontier_width: usize) -> i32 {
    crate::gpu::search::frontier_per_parent_limit(frontier_width)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_next_active_state_limit(
    frontier_width: usize,
    active_state_limit: usize,
    per_parent_limit: i32,
) -> usize {
    crate::gpu::search::frontier_next_active_state_limit(
        frontier_width,
        active_state_limit,
        per_parent_limit,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_state_stride(max_boards: usize) -> usize {
    crate::gpu::search::frontier_state_stride(max_boards)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_state_bytes(max_boards: usize) -> usize {
    crate::gpu::search::frontier_state_bytes(max_boards)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_params_bytes(
    state_count: usize,
    state_stride: usize,
    board_offset: usize,
    max_boards: usize,
    state_offset: usize,
    projection_size: usize,
    projection_seed: u32,
    target_depth: usize,
) -> *const u8 {
    set_output_bytes(crate::gpu::search::frontier_neural_params_bytes(
        state_count,
        state_stride,
        board_offset,
        max_boards,
        state_offset,
        projection_size,
        projection_seed,
        target_depth,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_apply_params_bytes(
    state_count: usize,
    root_color: i32,
    value_scale: f32,
    value_bias: f32,
    state_offset: usize,
) -> *const u8 {
    set_output_bytes(crate::gpu::search::frontier_neural_apply_params_bytes(
        state_count,
        root_color,
        value_scale,
        value_bias,
        state_offset,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_layer_params_bytes(
    sample_count: usize,
    input_size: usize,
    output_size: usize,
) -> *const u8 {
    set_output_bytes(crate::gpu::search::frontier_neural_layer_params_bytes(
        sample_count,
        input_size,
        output_size,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_effective_batch_size(
    state_count: usize,
    requested_batch_size: f64,
) -> usize {
    crate::gpu::search::frontier_neural_effective_batch_size(state_count, requested_batch_size)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_batch_count(
    state_count: usize,
    state_offset: usize,
    effective_batch_size: usize,
) -> usize {
    crate::gpu::search::frontier_neural_batch_count(state_count, state_offset, effective_batch_size)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_cache_hit_rate(hits: f64, misses: f64) -> f64 {
    crate::gpu::search::frontier_neural_cache_hit_rate(hits, misses)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_cycle_state_count(
    frontier_width: usize,
    requested_state_count: usize,
) -> usize {
    crate::gpu::search::frontier_cycle_state_count(frontier_width, requested_state_count)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_expansion_source_scan_limit(
    candidate_workgroup_size: usize,
    dispatch_candidate_limit: usize,
) -> usize {
    crate::gpu::search::frontier_expansion_source_scan_limit(
        candidate_workgroup_size,
        dispatch_candidate_limit,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_expansion_source_scan_count(
    source_scan_limit: usize,
    source_scans: usize,
    source_scan_base: usize,
) -> usize {
    crate::gpu::search::frontier_expansion_source_scan_count(
        source_scan_limit,
        source_scans,
        source_scan_base,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_minimax_bounded_depth(
    target_depth: i32,
    ancestry_stride: i32,
) -> i32 {
    crate::gpu::search::frontier_minimax_bounded_depth(target_depth, ancestry_stride)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_select_board_workgroups(batch_count: usize) -> usize {
    crate::gpu::search::frontier_neural_select_board_workgroups(batch_count)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_project_workgroups_x(batch_count: usize) -> usize {
    crate::gpu::search::frontier_neural_project_workgroups_x(batch_count)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_project_workgroups_y(projection_size: usize) -> usize {
    crate::gpu::search::frontier_neural_project_workgroups_y(projection_size)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_layer_workgroups_x(batch_count: usize) -> usize {
    crate::gpu::search::frontier_neural_layer_workgroups_x(batch_count)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_layer_workgroups_y(output_size: usize) -> usize {
    crate::gpu::search::frontier_neural_layer_workgroups_y(output_size)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_neural_output_workgroups(batch_count: usize) -> usize {
    crate::gpu::search::frontier_neural_output_workgroups(batch_count)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_policy_workgroups(candidate_count: usize) -> usize {
    crate::gpu::search::frontier_policy_workgroups(candidate_count)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_expand_workgroups(
    count: usize,
    candidate_workgroup_size: usize,
) -> usize {
    crate::gpu::search::frontier_expand_workgroups(count, candidate_workgroup_size)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_selection_workgroups(
    capacity: usize,
    candidate_workgroup_size: usize,
) -> usize {
    crate::gpu::search::frontier_selection_workgroups(capacity, candidate_workgroup_size)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_materialize_workgroups(
    frontier_width: usize,
    mutation_tile_size: usize,
) -> usize {
    crate::gpu::search::frontier_materialize_workgroups(frontier_width, mutation_tile_size)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_minimax_workgroups(frontier_width: usize) -> usize {
    crate::gpu::search::frontier_minimax_workgroups(frontier_width)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_policy_params_bytes(
    candidate_count: usize,
    candidate_stride: usize,
    input_size: usize,
    policy_scale: f32,
) -> *const u8 {
    set_output_bytes(crate::gpu::search::frontier_policy_params_bytes(
        candidate_count,
        candidate_stride,
        input_size,
        policy_scale,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_derive_frontier_tuning_json(
    max_storage_buffer_binding_size: usize,
    max_buffer_size: usize,
    max_compute_invocations_per_workgroup: usize,
    requested_nodes: usize,
    board_count: usize,
    additional_board_capacity: usize,
) -> *const u8 {
    let tuning = crate::gpu::search::derive_frontier_tuning(
        crate::gpu::search::FrontierTuningLimits {
            max_storage_buffer_binding_size: Some(max_storage_buffer_binding_size),
            max_buffer_size: Some(max_buffer_size),
            max_compute_invocations_per_workgroup: Some(max_compute_invocations_per_workgroup),
        },
        requested_nodes,
        board_count,
        additional_board_capacity,
    );
    let response = serde_json::json!({
        "maxBoards": tuning.max_boards,
        "frontierWidth": tuning.frontier_width,
        "candidateCapacity": tuning.candidate_capacity,
        "neuralBatchSize": tuning.neural_batch_size,
        "candidateWorkgroupSize": tuning.candidate_workgroup_size,
        "mutationTileSize": tuning.mutation_tile_size,
        "dispatchCandidateLimit": tuning.dispatch_candidate_limit,
    });
    set_output(response.to_string())
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_selection_plan_json(
    max_boards: usize,
    frontier_width: usize,
    candidate_capacity: usize,
    neural_batch_size: usize,
    candidate_workgroup_size: usize,
    mutation_tile_size: usize,
    dispatch_candidate_limit: usize,
    max_selection_scan: usize,
) -> *const u8 {
    let tuning = crate::gpu::search::FrontierTuning {
        max_boards,
        frontier_width,
        candidate_capacity,
        neural_batch_size,
        candidate_workgroup_size,
        mutation_tile_size,
        dispatch_candidate_limit,
    };
    let plan = crate::gpu::search::frontier_selection_plan(
        &tuning,
        if max_selection_scan > 0 {
            Some(max_selection_scan)
        } else {
            None
        },
    );
    let response = serde_json::json!({
        "candidateCapacity": plan.candidate_capacity,
        "selectionCapacity": plan.selection_capacity,
    });
    set_output(response.to_string())
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_match_turn_time_ms(
    cpu_training_time_ms: i32,
    now_ms: f64,
    deadline_at_ms: f64,
    remaining_searches: usize,
) -> i32 {
    crate::cpu::search::cpu_match_turn_time_ms(
        cpu_training_time_ms,
        now_ms,
        deadline_at_ms,
        remaining_searches,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_match_remaining_searches(
    max_match_plies: usize,
    ply: usize,
) -> usize {
    crate::cpu::search::cpu_match_remaining_searches(max_match_plies, ply)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_match_should_continue(now_ms: f64, deadline_at_ms: f64) -> i32 {
    i32::from(crate::cpu::search::cpu_match_should_continue(
        now_ms,
        deadline_at_ms,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_paired_match_deadline_ms(
    now_ms: f64,
    deadline_at_ms: f64,
    total_matches: usize,
    completed_matches: usize,
) -> f64 {
    crate::cpu::search::cpu_paired_match_deadline_ms(
        now_ms,
        deadline_at_ms,
        total_matches,
        completed_matches,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_paired_match_total_matches(game_count: usize) -> usize {
    crate::cpu::search::cpu_paired_match_total_matches(game_count)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a CPU
/// paired-match turn color string in this WASM instance for the duration of the
/// call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_paired_match_candidate_colors_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU paired-match candidate colors request") else {
        return std::ptr::null();
    };
    let turn = match serde_json::from_str::<String>(text) {
        Ok(turn) => turn,
        Err(error) => {
            set_last_message(&format!(
                "CPU paired-match candidate colors request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    match crate::cpu::search::cpu_paired_match_candidate_colors(&turn).and_then(|colors| {
        serde_json::to_string(&colors).map_err(|error| {
            format!("CPU paired-match candidate colors response failed to encode: {error}")
        })
    }) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_paired_match_average_score(
    score: f64,
    completed_matches: usize,
) -> f64 {
    crate::cpu::search::cpu_paired_match_average_score(score, completed_matches)
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn chronofish_cpu_training_position_target(
    samples: usize,
    training_mode_count: usize,
    cpu_opponent_variants: usize,
    cpu_screening_opponent_variants: usize,
    cpu_rounds_per_variant: usize,
    cpu_league_contenders: usize,
    cpu_league_hall_of_fame_entries: usize,
    cpu_hall_of_fame_entries: i32,
    cpu_min_pairs: usize,
    cpu_max_pairs: usize,
    cpu_max_match_plies: usize,
) -> usize {
    crate::cpu::search::cpu_training_position_target(
        samples,
        training_mode_count,
        cpu_opponent_variants,
        cpu_screening_opponent_variants,
        cpu_rounds_per_variant,
        cpu_league_contenders,
        cpu_league_hall_of_fame_entries,
        cpu_hall_of_fame_entries,
        cpu_min_pairs,
        cpu_max_pairs,
        cpu_max_match_plies,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_training_budget_ms(
    cpu_train_seconds: i32,
    cpu_training_time_ms: i32,
    cpu_max_match_plies: usize,
    cpu_max_match_time_ms: i32,
) -> i32 {
    wasm_milliseconds(crate::cpu::search::cpu_training_budget_ms(
        cpu_train_seconds.max(0) as u64,
        cpu_training_time_ms.max(0) as u64,
        cpu_max_match_plies,
        cpu_max_match_time_ms.max(0) as u64,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_mode_label_target(
    samples: usize,
    training_mode_count: usize,
    divisor: usize,
) -> usize {
    crate::cpu::search::mode_label_target(samples, training_mode_count, divisor)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a CPU
/// reference score delta request in this WASM instance for the duration of the
/// call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_reference_score_delta_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU reference score delta request") else {
        return std::ptr::null();
    };
    match crate::cpu::search::cpu_reference_score_delta_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a CPU
/// reference score-from-result request in this WASM instance for the duration
/// of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_reference_score_from_result_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU reference score-from-result request") else {
        return std::ptr::null();
    };
    match crate::cpu::search::cpu_reference_score_from_result_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a CPU
/// reference score delta-from-result request in this WASM instance for the
/// duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_reference_score_delta_from_result_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "CPU reference score delta-from-result request")
    else {
        return std::ptr::null();
    };
    match crate::cpu::search::cpu_reference_score_delta_from_result_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_reference_candidate_average(
    score: i32,
    compared: usize,
    near_draws: usize,
    draw_rate_limit: f32,
) -> f32 {
    crate::cpu::search::cpu_reference_candidate_average(
        score,
        compared,
        near_draws,
        draw_rate_limit,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_training_no_move_score(candidate_turn: i32) -> i32 {
    crate::cpu::search::cpu_training_no_move_score(candidate_turn != 0)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a CPU
/// training candidate-turn request in this WASM instance for the duration of the
/// call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_training_candidate_turn_json(
    ptr: *const u8,
    len: usize,
) -> i32 {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct CandidateTurnRequest {
        current_turn: String,
        candidate_color: String,
    }

    let Some(text) = wasm_input_text(ptr, len, "CPU training candidate-turn request") else {
        return 0;
    };
    let request = match serde_json::from_str::<CandidateTurnRequest>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "CPU training candidate-turn request is not valid JSON: {error}"
            ));
            return 0;
        }
    };
    i32::from(crate::cpu::search::cpu_training_candidate_turn(
        &request.current_turn,
        &request.candidate_color,
    ))
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a CPU
/// training winner score request in this WASM instance for the duration of the
/// call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_training_winner_score_json(
    ptr: *const u8,
    len: usize,
) -> i32 {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct WinnerScoreRequest {
        winner: Option<String>,
        candidate_color: String,
    }

    let Some(text) = wasm_input_text(ptr, len, "CPU training winner score request") else {
        return 0;
    };
    let request = match serde_json::from_str::<WinnerScoreRequest>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "CPU training winner score request is not valid JSON: {error}"
            ));
            return 0;
        }
    };
    crate::cpu::search::cpu_training_winner_score(
        request.winner.as_deref(),
        &request.candidate_color,
    )
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a CPU
/// training adjudication score request in this WASM instance for the duration of
/// the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_training_adjudication_score_json(
    ptr: *const u8,
    len: usize,
) -> i32 {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct AdjudicationScoreRequest {
        current_turn: String,
        candidate_color: String,
        baseline_score: i32,
    }

    let Some(text) = wasm_input_text(ptr, len, "CPU training adjudication score request") else {
        return 0;
    };
    let request = match serde_json::from_str::<AdjudicationScoreRequest>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "CPU training adjudication score request is not valid JSON: {error}"
            ));
            return 0;
        }
    };
    crate::cpu::search::cpu_training_adjudication_score(
        &request.current_turn,
        &request.candidate_color,
        request.baseline_score,
    )
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 JSON containing a CPU
/// training adjudication score-from-result request in this WASM instance for
/// the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_cpu_training_adjudication_score_from_result_json(
    ptr: *const u8,
    len: usize,
) -> i32 {
    let Some(text) = wasm_input_text(
        ptr,
        len,
        "CPU training adjudication score-from-result request",
    ) else {
        return 0;
    };
    match crate::cpu::search::cpu_training_adjudication_score_from_result_json(text) {
        Ok(score) => score,
        Err(error) => {
            set_last_message(&error);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_training_position_worker_count(
    target: usize,
    cpu_workers: usize,
) -> usize {
    crate::cpu::search::cpu_training_position_worker_count(target, cpu_workers)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_reference_worker_count(
    game_count: usize,
    requested_workers: usize,
    pair_batch: usize,
) -> usize {
    crate::cpu::search::cpu_reference_worker_count(game_count, requested_workers, pair_batch)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_candidate_worker_count(
    candidate_count: usize,
    cpu_workers: usize,
    pair_batch: usize,
) -> usize {
    crate::cpu::search::cpu_candidate_worker_count(candidate_count, cpu_workers, pair_batch)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_label_worker_count(
    position_count: usize,
    cpu_workers: usize,
) -> usize {
    crate::cpu::search::cpu_label_worker_count(position_count, cpu_workers)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_search_label_weight(training_mode_count: usize) -> f32 {
    crate::cpu::search::cpu_search_label_weight(training_mode_count)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_reference_comparison_count(
    game_count: usize,
    reference_count: usize,
) -> usize {
    crate::cpu::search::cpu_reference_comparison_count(game_count, reference_count)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_reference_should_continue(
    now_ms: f64,
    deadline_at_ms: f64,
    compared: usize,
    max_match_plies: usize,
) -> i32 {
    i32::from(crate::cpu::search::cpu_reference_should_continue(
        now_ms,
        deadline_at_ms,
        compared,
        max_match_plies,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_training_candidate_count(cpu_candidates: usize) -> usize {
    crate::cpu::search::cpu_training_candidate_count(cpu_candidates)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_screening_game_count(
    sample_game_count: usize,
    cpu_screening_opponent_variants: usize,
) -> usize {
    crate::cpu::search::cpu_screening_game_count(sample_game_count, cpu_screening_opponent_variants)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_training_finalist_target(
    population_len: usize,
    cpu_finalists: usize,
    cpu_pair_batch: usize,
    screened_len: usize,
) -> usize {
    crate::cpu::search::cpu_training_finalist_target(
        population_len,
        cpu_finalists,
        cpu_pair_batch,
        screened_len,
    )
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_training_elite_count(cpu_finalists: usize) -> usize {
    crate::cpu::search::cpu_training_elite_count(cpu_finalists)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_training_candidate_improved(
    candidate_score: f64,
    baseline_score: f64,
    best_candidate_score: f64,
) -> i32 {
    i32::from(crate::cpu::search::cpu_training_candidate_improved(
        candidate_score,
        baseline_score,
        best_candidate_score,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_training_next_stagnation(
    generations_without_candidate: usize,
    improved: i32,
) -> usize {
    crate::cpu::search::cpu_training_next_stagnation(generations_without_candidate, improved != 0)
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_training_should_continue(
    now_ms: f64,
    deadline_at_ms: f64,
    generations_without_candidate: usize,
    max_generations_without_candidate: usize,
) -> i32 {
    i32::from(crate::cpu::search::cpu_training_should_continue(
        now_ms,
        deadline_at_ms,
        generations_without_candidate,
        max_generations_without_candidate,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_candidate_scoring_should_continue(
    now_ms: f64,
    deadline_at_ms: f64,
    next_candidate: usize,
    uncached_candidate_count: usize,
) -> i32 {
    i32::from(crate::cpu::search::cpu_candidate_scoring_should_continue(
        now_ms,
        deadline_at_ms,
        next_candidate,
        uncached_candidate_count,
    ))
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_reference_collection_should_continue(
    now_ms: f64,
    deadline_at_ms: f64,
    next_game: usize,
    game_count: usize,
) -> i32 {
    i32::from(
        crate::cpu::search::cpu_reference_collection_should_continue(
            now_ms,
            deadline_at_ms,
            next_game,
            game_count,
        ),
    )
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_training_position_search_config_json(
    cpu_depth: i32,
    cpu_nodes: i32,
) -> *const u8 {
    let config = crate::cpu::search::cpu_training_position_search_config(cpu_depth, cpu_nodes);
    let response = serde_json::json!({
        "depth": config.depth,
        "nodes": config.nodes,
    });
    set_output(response.to_string())
}

#[no_mangle]
pub extern "C" fn chronofish_cpu_screening_training_config_json(
    cpu_depth: i32,
    depth: i32,
    cpu_nodes: i32,
    nodes: i32,
    cpu_training_time_ms: i32,
) -> *const u8 {
    let config = crate::cpu::search::cpu_screening_training_config(
        cpu_depth,
        depth,
        cpu_nodes,
        nodes,
        cpu_training_time_ms,
    );
    let response = serde_json::json!({
        "cpuDepth": config.cpu_depth,
        "depth": config.depth,
        "cpuNodes": config.cpu_nodes,
        "nodes": config.nodes,
        "cpuTrainingTimeMs": config.cpu_training_time_ms,
    });
    set_output(response.to_string())
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
pub extern "C" fn chronofish_evaluation_json() -> *const u8 {
    let json = with_game(Game::evaluation_json);
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_snapshot_bytes() -> *const u8 {
    let bytes = with_game(Game::gpu_snapshot_bytes);
    set_output_bytes(bytes)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_snapshot_json() -> *const u8 {
    let json = with_game(Game::gpu_snapshot_json);
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_candidate_inputs_json() -> *const u8 {
    let json = with_game(crate::gpu::search::gpu_candidate_inputs_json_from_game);
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_candidate_inputs_bytes() -> *const u8 {
    let words = with_game(crate::gpu::search::gpu_candidate_inputs_i32s_from_game);
    set_output_i32s(words)
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU snapshot JSON in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_candidate_inputs_snapshot_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU candidate input snapshot request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_candidate_inputs_i32s_from_gpu_snapshot_json(text) {
        Ok(words) => set_output_i32s(words),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 GPU candidate
/// input data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_candidate_input_meta_json_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(words) = wasm_input_i32s(ptr, len, "GPU candidate input metadata") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_candidate_input_meta_json_from_i32s(&words) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU snapshot JSON in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_snapshot_game_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU snapshot game conversion request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_snapshot_game_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU child snapshot request
/// JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_snapshot_child_boards_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU child snapshot request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_snapshot_with_child_boards_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` readable CFNN bytes in this WASM instance for the
/// duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_compact_value_model_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Compact value model") else {
        return std::ptr::null();
    };
    match crate::gpu::training::compact_value_model_json(bytes) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` readable CFNN bytes in this WASM instance for the
/// duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_compact_value_model_frontier_layout_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Compact value model frontier layout") else {
        return std::ptr::null();
    };
    match crate::gpu::training::compact_value_model_frontier_layout_json(bytes) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable compact value model JSON in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_compact_value_model_bytes_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Compact value model JSON") else {
        return std::ptr::null();
    };
    match crate::gpu::training::compact_value_model_bytes_from_json(text) {
        Ok(bytes) => set_output_bytes(bytes),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` readable CFNN bytes in this WASM instance for the
/// duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_compact_value_model_is_finite_bytes(
    ptr: *const u8,
    len: usize,
) -> i32 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Compact value model") else {
        return 0;
    };
    i32::from(crate::gpu::training::compact_value_model_is_finite_bytes(
        bytes,
    ))
}

/// # Safety
///
/// `ptr` must point to `len` readable CFNN bytes in this WASM instance for the
/// duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_compact_value_model_architecture_matches_bytes(
    ptr: *const u8,
    len: usize,
) -> i32 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Compact value model") else {
        return 0;
    };
    i32::from(crate::gpu::training::compact_value_model_architecture_matches_bytes(bytes))
}

/// # Safety
///
/// `ptr` must point to `len` readable CFNN bytes in this WASM instance for the
/// duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_compact_value_model_policy_weights_bytes(
    ptr: *const u8,
    len: usize,
    input_size: usize,
) -> *const u8 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Compact value model") else {
        return std::ptr::null();
    };
    match crate::gpu::training::compact_value_model_policy_weights_bytes(bytes, input_size) {
        Ok(Some(weights)) => set_output_bytes(weights),
        Ok(None) => std::ptr::null(),
        Err(error) => {
            set_last_message(&error.to_string());
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` readable little-endian f32 bytes in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_quantized_policy_upload_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "Policy upload weights") else {
        return std::ptr::null();
    };
    match crate::gpu::training::quantized_policy_upload_bytes(bytes) {
        Ok(output) => set_output_bytes(output),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` readable little-endian f32 bytes in this WASM
/// instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_f32_to_f16_upload_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(bytes) = wasm_input_bytes(ptr, len, "f16 upload weights") else {
        return std::ptr::null();
    };
    match crate::gpu::training::f32_to_f16_upload_bytes(bytes) {
        Ok(output) => set_output_bytes(output),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `model_ptr` must point to `model_len` readable CFNN bytes and `samples_ptr`
/// must point to `samples_len` bytes of readable UTF-8 training sample JSON in
/// this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_compact_value_model_predict_values_json(
    model_ptr: *const u8,
    model_len: usize,
    samples_ptr: *const u8,
    samples_len: usize,
) -> *const u8 {
    let Some(model_bytes) = wasm_input_bytes(model_ptr, model_len, "Compact value model") else {
        return std::ptr::null();
    };
    let Some(samples_text) = wasm_input_text(samples_ptr, samples_len, "Prediction samples") else {
        return std::ptr::null();
    };
    let model = match crate::gpu::training::decode_compact_value_model(model_bytes) {
        Ok(model) => model,
        Err(error) => {
            set_last_message(&error.to_string());
            return std::ptr::null();
        }
    };
    let samples =
        match serde_json::from_str::<Vec<crate::gpu::training::TrainingSample>>(samples_text) {
            Ok(samples) => samples,
            Err(error) => {
                set_last_message(&format!("Prediction samples are not valid JSON: {error}"));
                return std::ptr::null();
            }
        };
    match serde_json::to_string(
        &samples
            .iter()
            .map(|sample| model.predict_value(&sample.features))
            .collect::<Vec<_>>(),
    ) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&format!("failed to encode predictions: {error}"));
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// If `model_len` is non-zero, `model_ptr` must point to readable CFNN bytes in
/// this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_compact_value_model_training_layout_bytes(
    model_ptr: *const u8,
    model_len: usize,
    average_label: f32,
) -> *const u8 {
    let model_bytes = if model_len == 0 {
        None
    } else {
        wasm_input_bytes(model_ptr, model_len, "Compact value model training layout")
    };
    if model_len != 0 && model_bytes.is_none() {
        return std::ptr::null();
    }
    match crate::gpu::training::compact_value_model_training_layout_bytes(
        model_bytes,
        average_label,
    ) {
        Ok(bytes) => set_output_bytes(bytes),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `model_ptr` must point to `model_len` bytes of readable CFNN model data and
/// `samples_ptr` must point to `samples_len` bytes of readable UTF-8 training
/// sample JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_compact_value_model_hidden_features_json(
    model_ptr: *const u8,
    model_len: usize,
    samples_ptr: *const u8,
    samples_len: usize,
) -> *const u8 {
    let Some(model_bytes) =
        wasm_input_bytes(model_ptr, model_len, "Compact value model hidden features")
    else {
        return std::ptr::null();
    };
    let Some(samples_text) = wasm_input_text(
        samples_ptr,
        samples_len,
        "Compact value model hidden feature samples",
    ) else {
        return std::ptr::null();
    };
    match crate::gpu::training::compact_value_model_hidden_features_json(model_bytes, samples_text)
    {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_root_bytes(max_boards: usize) -> *const u8 {
    match with_game(|game| crate::gpu::search::encode_frontier_root(game, max_boards)) {
        Ok(root) => set_output_i32s(root.words),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU snapshot JSON in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_frontier_root_snapshot_bytes(
    ptr: *const u8,
    len: usize,
    max_boards: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU frontier root snapshot") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_frontier_root_i32s_from_snapshot_json(text, max_boards) {
        Ok(words) => set_output_i32s(words),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU snapshot JSON in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_snapshot_search_size_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU snapshot search size") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_snapshot_search_size_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU snapshot JSON in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_pending_present_boards_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU pending present board snapshot") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_pending_present_boards_json_from_snapshot_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU search selection JSON
/// in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_select_candidate_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU search selection") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_search_select_candidate_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU snapshot JSON in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_turn_status_records_snapshot_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU turn-status snapshot") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_turn_status_records_i32s_from_snapshot_json(text) {
        Ok(words) => set_output_i32s(words),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 turn-status
/// response data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_turn_status_json_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    if ptr.is_null() {
        set_last_message("GPU turn-status response pointer is null.");
        return std::ptr::null();
    }
    if !len.is_multiple_of(std::mem::size_of::<i32>()) {
        set_last_message("GPU turn-status response byte length is not i32-aligned.");
        return std::ptr::null();
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let words = bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    match crate::gpu::search::gpu_turn_status_json_from_i32s(&words) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 turn-status JSON request
/// data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_turn_status_json(ptr: *const u8, len: usize) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU turn-status JSON request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_turn_status_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 full-search precondition
/// JSON request data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_full_search_precondition_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU full-search precondition request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_full_search_precondition_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 ranking request
/// data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_ranked_candidate_indexes_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    if ptr.is_null() {
        set_last_message("GPU candidate ranking request pointer is null.");
        return std::ptr::null();
    }
    if !len.is_multiple_of(std::mem::size_of::<i32>()) {
        set_last_message("GPU candidate ranking request byte length is not i32-aligned.");
        return std::ptr::null();
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let words = bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    match crate::gpu::search::gpu_ranked_candidate_indexes_from_i32s(&words) {
        Ok(indexes) => set_output_i32s(indexes),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 ranking request
/// data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_ranked_candidates_json_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    if ptr.is_null() {
        set_last_message("GPU ranked candidates request pointer is null.");
        return std::ptr::null();
    }
    if !len.is_multiple_of(std::mem::size_of::<i32>()) {
        set_last_message("GPU ranked candidates request byte length is not i32-aligned.");
        return std::ptr::null();
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let words = bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    match crate::gpu::search::gpu_ranked_candidates_json_from_i32s(&words) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 ranked-candidates JSON
/// request data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_ranked_candidates_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU ranked candidates JSON request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_ranked_candidates_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 mutation selected-candidates
/// JSON request data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_mutation_selected_candidates_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU mutation selected candidates request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_mutation_selected_candidates_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 candidate-index JSON
/// request data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_candidate_indexes_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU candidate indexes request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_candidate_indexes_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 candidate-scores JSON
/// request data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_candidate_scores_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU candidate scores request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_candidate_scores_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_candidate_score_is_rejected(score: i32) -> i32 {
    crate::gpu::search::gpu_candidate_score_is_rejected(score) as i32
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 frontier readback-summary
/// JSON request data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_frontier_readback_summary_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU frontier readback summary request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_frontier_readback_summary_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 scoring-summary
/// request data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_scoring_summary_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    if ptr.is_null() {
        set_last_message("GPU scoring summary request pointer is null.");
        return std::ptr::null();
    }
    if !len.is_multiple_of(std::mem::size_of::<i32>()) {
        set_last_message("GPU scoring summary request byte length is not i32-aligned.");
        return std::ptr::null();
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let words = bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    match crate::gpu::search::gpu_scoring_summary_from_i32s(&words) {
        Ok(summary) => set_output(summary),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU scoring summary JSON
/// in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_scoring_summary_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU scoring summary JSON request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_scoring_summary_json(text) {
        Ok(summary) => set_output(summary),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 mutation
/// statuses in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_mutation_summary_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    if ptr.is_null() {
        set_last_message("GPU mutation summary request pointer is null.");
        return std::ptr::null();
    }
    if !len.is_multiple_of(std::mem::size_of::<i32>()) {
        set_last_message("GPU mutation summary request byte length is not i32-aligned.");
        return std::ptr::null();
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let statuses = bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    set_output(crate::gpu::search::gpu_mutation_summary_from_i32s(
        &statuses,
    ))
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU mutation summary JSON
/// in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_mutation_summary_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU mutation summary JSON request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_mutation_summary_json(text) {
        Ok(summary) => set_output(summary),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU mutation statuses JSON
/// in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_mutation_statuses_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU mutation statuses JSON request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_mutation_statuses_json(text) {
        Ok(statuses) => set_output(statuses),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 pending-board JSON in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_turn_completion_key_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU turn completion key request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_turn_completion_key_json(text) {
        Ok(key) => set_output(key),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 choice-agreement JSON in
/// this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_choice_agreement_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU choice agreement request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_choice_agreement_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 choice-agreement JSON
/// containing raw GPU search choices in this WASM instance for the duration of
/// the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_choice_agreement_choices_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU choice agreement choices request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_choice_agreement_choices_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU search choice
/// selection JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_select_choice_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU search choice selection request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_search_select_choice_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU search choice
/// selection JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_selected_choice_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU selected search choice request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_search_selected_choice_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 choice-agreement JSON
/// containing raw GPU search choices in this WASM instance for the duration of
/// the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_policy_choice_agreement_diagnostics_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU policy choice agreement diagnostics request")
    else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_policy_choice_agreement_diagnostics_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 move-plan JSON in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_move_plan_key_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU move plan key request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_move_plan_key_json(text) {
        Ok(key) => set_output(key),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 GPU frontier
/// state data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_frontier_plan_json_bytes(
    ptr: *const u8,
    len: usize,
    offset: usize,
    plan_length: usize,
) -> *const u8 {
    let Some(words) = wasm_input_i32s(ptr, len, "GPU frontier plan request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_frontier_plan_json_from_i32s(&words, offset, plan_length) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 GPU frontier
/// state data and `gpu_search_ptr` must point to `gpu_search_len` bytes of
/// readable UTF-8 search-label data in this WASM instance for the duration of
/// the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_frontier_choices_json_bytes(
    ptr: *const u8,
    len: usize,
    max_boards: usize,
    frontier_width: usize,
    requested_depth: i32,
    gpu_search_ptr: *const u8,
    gpu_search_len: usize,
    choice_limit: usize,
) -> *const u8 {
    let Some(words) = wasm_input_i32s(ptr, len, "GPU frontier choices request") else {
        return std::ptr::null();
    };
    let Some(gpu_search) = wasm_input_text(
        gpu_search_ptr,
        gpu_search_len,
        "GPU frontier choices search label",
    ) else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_frontier_choices_json_from_i32s(
        &words,
        max_boards,
        frontier_width,
        requested_depth,
        gpu_search,
        choice_limit,
    ) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 validated frontier choice
/// JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_validated_frontier_choice_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU validated frontier choice request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_validated_frontier_choice_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 frontier choice
/// diagnostics JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_frontier_choice_diagnostics_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU frontier choice diagnostics request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_frontier_choice_diagnostics_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 non-postable result JSON
/// in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_non_postable_result_summary_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU non-postable result summary request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_non_postable_result_summary_json(text) {
        Ok(summary) => set_output(summary),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU search result JSON in
/// this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_postable_search_result_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU postable search result request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_postable_search_result_json(text) {
        Ok(postable) => set_output(if postable { "true" } else { "false" }.to_string()),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 first-frontier-turn
/// validation JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_validate_first_frontier_turn_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU validate first frontier turn request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_validate_first_frontier_turn_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU search result
/// validation request JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_validate_search_result_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU validate search result request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_validate_search_result_json(text) {
        Ok(response) => set_output(response),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU snapshot JSON in this
/// WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_search_failure_summary_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU search failure summary request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_search_failure_summary_json(text) {
        Ok(summary) => set_output(summary),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 completed-turn choice JSON
/// in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_completed_turn_choice_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU completed-turn choice request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_completed_turn_choice_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 completed-turn step JSON
/// in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_turn_completion_step_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU completed-turn step request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_turn_completion_step_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 incomplete-turn pending
/// count JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_incomplete_turn_pending_present_board_count_json(
    ptr: *const u8,
    len: usize,
) -> usize {
    let Some(text) = wasm_input_text(ptr, len, "GPU incomplete-turn pending count request") else {
        return 0;
    };
    match crate::gpu::search::gpu_incomplete_turn_pending_present_board_count_json(text) {
        Ok(count) => count,
        Err(error) => {
            set_last_message(&error);
            0
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 principal variation JSON
/// in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_normalize_principal_variation_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU principal variation request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_normalize_principal_variation_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 GPU search choices JSON in
/// this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_summarize_search_choices_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU search choice summary request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_summarize_search_choices_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 bot ranked-choice request
/// JSON in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_bot_ranked_choices_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Bot ranked choices request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::bot_ranked_choices_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 bot search result JSON in
/// this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_bot_select_best_result_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "Bot search result selection request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::bot_select_best_result_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 candidate
/// record pick request data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_pick_candidate_records_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    if ptr.is_null() {
        set_last_message("GPU candidate record pick request pointer is null.");
        return std::ptr::null();
    }
    if !len.is_multiple_of(std::mem::size_of::<i32>()) {
        set_last_message("GPU candidate record pick request byte length is not i32-aligned.");
        return std::ptr::null();
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let words = bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    match crate::gpu::search::gpu_pick_candidate_records_from_i32s(&words) {
        Ok(records) => set_output_i32s(records),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 candidate-record pick JSON
/// in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_pick_candidate_records_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU candidate record pick JSON request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_pick_candidate_records_json(text) {
        Ok(records) => set_output_i32s(records),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 mutation turn-code JSON in
/// this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_mutation_turn_code_json(ptr: *const u8, len: usize) -> i32 {
    let Some(text) = wasm_input_text(ptr, len, "GPU mutation turn-code request") else {
        return 0;
    };
    match crate::gpu::search::gpu_mutation_turn_code_json(text) {
        Ok(value) => value,
        Err(error) => {
            set_last_message(&error);
            0
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 candidate
/// index request data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_candidate_index_bytes(ptr: *const u8, len: usize) -> i32 {
    if ptr.is_null() {
        set_last_message("GPU candidate index request pointer is null.");
        return -2;
    }
    if !len.is_multiple_of(std::mem::size_of::<i32>()) {
        set_last_message("GPU candidate index request byte length is not i32-aligned.");
        return -2;
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let words = bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    match crate::gpu::search::gpu_candidate_index_from_i32s(&words) {
        Ok(index) => index,
        Err(error) => {
            set_last_message(&error);
            -2
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 candidate-index JSON in
/// this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_candidate_index_json(ptr: *const u8, len: usize) -> i32 {
    let Some(text) = wasm_input_text(ptr, len, "GPU candidate index JSON request") else {
        return -2;
    };
    match crate::gpu::search::gpu_candidate_index_json(text) {
        Ok(index) => index,
        Err(error) => {
            set_last_message(&error);
            -2
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable little-endian i32 reply-pressure
/// request data in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_reply_pressure_ranked_roots_bytes(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    if ptr.is_null() {
        set_last_message("GPU reply pressure request pointer is null.");
        return std::ptr::null();
    }
    if !len.is_multiple_of(std::mem::size_of::<i32>()) {
        set_last_message("GPU reply pressure request byte length is not i32-aligned.");
        return std::ptr::null();
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    let words = bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    match crate::gpu::search::gpu_reply_pressure_ranked_roots_from_i32s(&words) {
        Ok(records) => set_output_i32s(records),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
}

/// # Safety
///
/// `ptr` must point to `len` bytes of readable UTF-8 reply-pressure root JSON
/// in this WASM instance for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn chronofish_gpu_reply_pressure_ranked_roots_json(
    ptr: *const u8,
    len: usize,
) -> *const u8 {
    let Some(text) = wasm_input_text(ptr, len, "GPU reply pressure JSON request") else {
        return std::ptr::null();
    };
    match crate::gpu::search::gpu_reply_pressure_ranked_roots_json(text) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&error);
            std::ptr::null()
        }
    }
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
pub extern "C" fn chronofish_legal_selection_json(
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
    let json = with_game(|game| game.legal_selection_json(from));
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
pub extern "C" fn chronofish_submit_turn_status_json() -> *const u8 {
    let status = with_game_mut(|game| {
        let complete = game.submit_turn() != 0;
        serde_json::json!({
            "complete": complete,
            "terminal": game.result.is_some(),
            "winner": game.result.and_then(|result| result.winner.map(|winner| winner.as_str())),
            "resultReason": game.result.map(|result| result.reason.as_str()),
            "nextTurn": game.turn.as_str(),
            "presentTime": game.present_time().unwrap_or(0),
            "pendingPresentBoardCount": if complete { 0 } else { 1 },
            "message": game.last_message.clone(),
        })
        .to_string()
    });
    set_output(status)
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
pub extern "C" fn chronofish_ai_turn_timed_min_depth_json(
    max_depth: i32,
    min_depth: i32,
    max_nodes: i32,
    millis: i32,
) -> *const u8 {
    let json = with_game(|game| {
        game.ai_turn_timed_min_depth_json(max_depth, min_depth, max_nodes, millis)
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
    set_output_bytes(value.into_bytes())
}

fn set_output_bytes(value: Vec<u8>) -> *const u8 {
    OUTPUT.with(|output| {
        let mut output = output.borrow_mut();
        *output = value;
        output.as_ptr()
    })
}

fn set_output_i32s(values: Vec<i32>) -> *const u8 {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<i32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    set_output_bytes(bytes)
}

unsafe fn wasm_input_text<'a>(ptr: *const u8, len: usize, label: &str) -> Option<&'a str> {
    if ptr.is_null() {
        set_last_message(&format!("{label} pointer is null."));
        return None;
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    match std::str::from_utf8(bytes) {
        Ok(text) => Some(text),
        Err(_) => {
            set_last_message(&format!("{label} is not valid UTF-8."));
            None
        }
    }
}

unsafe fn wasm_input_bytes<'a>(ptr: *const u8, len: usize, label: &str) -> Option<&'a [u8]> {
    if ptr.is_null() {
        set_last_message(&format!("{label} pointer is null."));
        return None;
    }
    Some(std::slice::from_raw_parts(ptr, len))
}

unsafe fn wasm_input_i32s(ptr: *const u8, len: usize, label: &str) -> Option<Vec<i32>> {
    let bytes = wasm_input_bytes(ptr, len, label)?;
    if !len.is_multiple_of(std::mem::size_of::<i32>()) {
        set_last_message(&format!("{label} byte length is not i32-aligned."));
        return None;
    }
    Some(
        bytes
            .chunks_exact(std::mem::size_of::<i32>())
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect(),
    )
}

fn encode_training_samples_json(samples: Vec<crate::gpu::training::TrainingSample>) -> *const u8 {
    match serde_json::to_string(&samples) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&format!("failed to encode training samples: {error}"));
            std::ptr::null()
        }
    }
}

fn training_subject_from_config(config: &serde_json::Value) -> String {
    if let Some(subject) = string_field(config, "trainingSubject")
        .filter(|subject| crate::gpu::training::is_training_subject(subject))
    {
        subject
    } else {
        crate::gpu::training::legacy_training_subject(
            string_field(config, "trainingTarget").as_deref(),
        )
        .to_string()
    }
}

fn string_field(config: &serde_json::Value, key: &str) -> Option<String> {
    config
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn string_array_field(config: &serde_json::Value, key: &str) -> Vec<String> {
    config
        .get(key)
        .and_then(serde_json::Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn number_field(config: &serde_json::Value, key: &str) -> Option<f64> {
    config.get(key).and_then(serde_json::Value::as_f64)
}

fn clamp_config_number(
    config: &serde_json::Value,
    key: &str,
    min: f64,
    max: f64,
    fallback: f64,
) -> f64 {
    crate::gpu::training::clamp_training_number(number_field(config, key), min, max, fallback)
}

fn clamp_config_integer(
    config: &serde_json::Value,
    key: &str,
    min: i64,
    max: i64,
    fallback: i64,
) -> i64 {
    crate::gpu::training::clamp_training_integer(number_field(config, key), min, max, fallback)
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

pub(crate) fn parse_game_snapshot(text: &str) -> Result<Game, String> {
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
    let result = parse_game_result(object.get("result"))?;
    let last_message = result.map_or_else(
        || format!("{} to move.", turn.capitalized()),
        GameResult::message,
    );
    let mut game = Game {
        turn,
        timelines,
        next_timeline_id,
        next_black_timeline_id,
        staged_turn: Vec::new(),
        staged_notation: Vec::new(),
        staged_royal_capture_by: None,
        result,
        last_message,
        position_hash: 0,
    };
    game.position_hash = game.recompute_position_hash();
    Ok(game)
}

fn parse_game_result(value: Option<&serde_json::Value>) -> Result<Option<GameResult>, String> {
    let Some(object) = value.and_then(serde_json::Value::as_object) else {
        return Ok(None);
    };
    if object.get("terminal").and_then(serde_json::Value::as_bool) != Some(true) {
        return Ok(None);
    }
    let winner = match object.get("winner").and_then(serde_json::Value::as_str) {
        Some("white") => Some(Color::White),
        Some("black") => Some(Color::Black),
        Some(_) => return Err("Snapshot result winner must be white, black, or null.".to_string()),
        None => None,
    };
    let reason = match object.get("reason").and_then(serde_json::Value::as_str) {
        Some("royal-capture") => GameResultReason::RoyalCapture,
        Some("threefold-repetition") => GameResultReason::ThreefoldRepetition,
        Some("stalemate") => GameResultReason::Stalemate,
        _ => return Err("Snapshot result has an unsupported reason.".to_string()),
    };
    match (winner, reason) {
        (Some(_), GameResultReason::RoyalCapture)
        | (None, GameResultReason::ThreefoldRepetition | GameResultReason::Stalemate) => {}
        _ => return Err("Snapshot result winner does not match its reason.".to_string()),
    }
    Ok(Some(GameResult { winner, reason }))
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
    value
        .as_object()
        .map_or_else(CastlingRights::new, |object| CastlingRights {
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

fn parse_cpu_parameters_value_from_text(
    text: &str,
) -> Result<crate::cpu::search::CpuParameters, String> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|error| format!("CPU parameters are not valid JSON: {error}"))?;
    parse_cpu_parameters_value(&value)
}

fn parse_cpu_parameter_array_from_text(
    text: &str,
) -> Result<Vec<crate::cpu::search::CpuParameters>, String> {
    let values: Vec<serde_json::Value> = serde_json::from_str(text)
        .map_err(|error| format!("CPU parameter array is not valid JSON: {error}"))?;
    values
        .iter()
        .map(parse_cpu_parameters_value)
        .collect::<Result<Vec<_>, _>>()
}

fn parse_cpu_scored_candidate_array_from_text(
    text: &str,
) -> Result<Vec<crate::cpu::search::CpuScoredCandidate>, String> {
    let values: Vec<serde_json::Value> = serde_json::from_str(text)
        .map_err(|error| format!("CPU scored candidate array is not valid JSON: {error}"))?;
    values
        .iter()
        .map(parse_cpu_scored_candidate_value)
        .collect::<Result<Vec<_>, _>>()
}

fn parse_cpu_parameters_value(
    value: &serde_json::Value,
) -> Result<crate::cpu::search::CpuParameters, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "CPU parameters must be an object.".to_string())?;
    let mut parameters = object
        .iter()
        .map(|(key, value)| {
            let Some(number) = value.as_i64().and_then(|value| i32::try_from(value).ok()) else {
                return Err(format!("CPU parameter `{key}` must be an integer."));
            };
            Ok((key.clone(), number))
        })
        .collect::<Result<Vec<_>, String>>()?;
    parameters.sort_by(|(left, _), (right, _)| left.cmp(right));
    Ok(parameters)
}

fn parse_cpu_scored_candidate_value(
    value: &serde_json::Value,
) -> Result<crate::cpu::search::CpuScoredCandidate, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "CPU scored candidate must be an object.".to_string())?;
    let parameters = object
        .get("parameters")
        .ok_or_else(|| "CPU scored candidate requires parameters.".to_string())
        .and_then(parse_cpu_parameters_value)?;
    let score = object
        .get("score")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| "CPU scored candidate requires a numeric score.".to_string())?;
    Ok(crate::cpu::search::CpuScoredCandidate { parameters, score })
}

fn encode_cpu_parameter_array_json(
    parameters: Vec<crate::cpu::search::CpuParameters>,
) -> Result<String, String> {
    let values = parameters
        .iter()
        .map(|parameters| {
            let mut object = serde_json::Map::new();
            for (key, value) in parameters {
                object.insert(key.clone(), serde_json::json!(value));
            }
            serde_json::Value::Object(object)
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&values)
        .map_err(|error| format!("failed to encode CPU parameters: {error}"))
}

fn encode_cpu_scored_candidate_array_json(
    candidates: Vec<crate::cpu::search::CpuScoredCandidate>,
) -> Result<String, String> {
    let values = candidates
        .into_iter()
        .map(|candidate| {
            let mut parameters = serde_json::Map::new();
            for (key, value) in candidate.parameters {
                parameters.insert(key, serde_json::json!(value));
            }
            serde_json::json!({
                "parameters": parameters,
                "score": candidate.score
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&values)
        .map_err(|error| format!("failed to encode CPU scored candidates: {error}"))
}

fn cpu_parameters_json_value(parameters: crate::cpu::search::CpuParameters) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    for (key, value) in parameters {
        object.insert(key, serde_json::json!(value));
    }
    serde_json::Value::Object(object)
}

fn cpu_scored_candidate_json_value(
    candidate: crate::cpu::search::CpuScoredCandidate,
) -> serde_json::Value {
    serde_json::json!({
        "parameters": cpu_parameters_json_value(candidate.parameters),
        "score": candidate.score
    })
}

fn encode_cpu_candidate_scoring_plan_json(
    plan: crate::cpu::search::CpuCandidateScoringPlan,
) -> Result<String, String> {
    let unique_candidates = plan
        .unique_candidates
        .into_iter()
        .map(cpu_parameters_json_value)
        .collect::<Vec<_>>();
    let cached_scores = plan
        .cached_scores
        .into_iter()
        .map(cpu_scored_candidate_json_value)
        .collect::<Vec<_>>();
    let uncached_candidates = plan
        .uncached_candidates
        .into_iter()
        .map(cpu_parameters_json_value)
        .collect::<Vec<_>>();
    serde_json::to_string(&serde_json::json!({
        "uniqueCandidates": unique_candidates,
        "cachedScores": cached_scores,
        "uncachedCandidates": uncached_candidates,
        "cacheHits": plan.cache_hits
    }))
    .map_err(|error| format!("failed to encode CPU candidate scoring plan: {error}"))
}

fn encode_cpu_fitness_entry_json(
    entry: crate::cpu::search::CpuFitnessEntry,
) -> Result<String, String> {
    serde_json::to_string(&serde_json::json!({
        "key": entry.key,
        "score": entry.score,
    }))
    .map_err(|error| format!("failed to encode CPU fitness entry: {error}"))
}

fn encode_cpu_training_generation_outcome_json(
    outcome: crate::cpu::search::CpuTrainingGenerationOutcome,
) -> Result<String, String> {
    let winner = outcome.winner.map(cpu_scored_candidate_json_value);
    serde_json::to_string(&serde_json::json!({
        "baselineScore": outcome.baseline_score,
        "winner": winner,
        "improved": outcome.improved,
    }))
    .map_err(|error| format!("failed to encode CPU training generation outcome: {error}"))
}

fn breed_cpu_population_json(text: &str) -> Result<Vec<crate::cpu::search::CpuParameters>, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct BreedCpuPopulationRequest {
        baseline: serde_json::Value,
        elites: Vec<serde_json::Value>,
        target: usize,
        seed: u32,
        generation: u32,
        stagnation: u32,
    }

    let request: BreedCpuPopulationRequest = serde_json::from_str(text)
        .map_err(|error| format!("CPU population breeding request is not valid JSON: {error}"))?;
    let baseline = parse_cpu_parameters_value(&request.baseline)?;
    let elites = request
        .elites
        .iter()
        .map(parse_cpu_parameters_value)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(crate::cpu::search::breed_cpu_population(
        &baseline,
        &elites,
        request.target,
        request.seed,
        request.generation,
        request.stagnation,
    ))
}

fn cpu_training_elites_json(
    text: &str,
    cpu_finalists: usize,
) -> Result<Vec<crate::cpu::search::CpuParameters>, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct CpuTrainingElitesRequest {
        baseline: serde_json::Value,
        candidates: Vec<serde_json::Value>,
    }

    let request: CpuTrainingElitesRequest = serde_json::from_str(text)
        .map_err(|error| format!("CPU training elites request is not valid JSON: {error}"))?;
    let baseline = parse_cpu_parameters_value(&request.baseline)?;
    let candidates = request
        .candidates
        .iter()
        .map(parse_cpu_scored_candidate_value)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(crate::cpu::search::cpu_training_elites(
        &candidates,
        &baseline,
        cpu_finalists,
    ))
}

fn cpu_training_finalist_candidates_json(
    text: &str,
    target: usize,
) -> Result<Vec<crate::cpu::search::CpuParameters>, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct CpuTrainingFinalistCandidatesRequest {
        baseline: serde_json::Value,
        screened: Vec<serde_json::Value>,
    }

    let request: CpuTrainingFinalistCandidatesRequest =
        serde_json::from_str(text).map_err(|error| {
            format!("CPU training finalist candidates request is not valid JSON: {error}")
        })?;
    let baseline = parse_cpu_parameters_value(&request.baseline)?;
    let screened = request
        .screened
        .iter()
        .map(parse_cpu_scored_candidate_value)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(crate::cpu::search::cpu_training_finalist_candidates(
        &baseline, &screened, target,
    ))
}

fn cpu_training_generation_outcome_json(
    text: &str,
) -> Result<crate::cpu::search::CpuTrainingGenerationOutcome, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct CpuTrainingGenerationOutcomeRequest {
        baseline: serde_json::Value,
        finalists: Vec<serde_json::Value>,
        previous_baseline_score: Option<f64>,
        best_candidate_score: Option<f64>,
    }

    let request: CpuTrainingGenerationOutcomeRequest =
        serde_json::from_str(text).map_err(|error| {
            format!("CPU training generation outcome request is not valid JSON: {error}")
        })?;
    let baseline = parse_cpu_parameters_value(&request.baseline)?;
    let finalists = request
        .finalists
        .iter()
        .map(parse_cpu_scored_candidate_value)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(crate::cpu::search::cpu_training_generation_outcome(
        &baseline,
        &finalists,
        request.previous_baseline_score.unwrap_or(-1.0e300),
        request.best_candidate_score.unwrap_or(-1.0e300),
    ))
}

fn cpu_candidate_scoring_plan_json(
    text: &str,
) -> Result<crate::cpu::search::CpuCandidateScoringPlan, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct CpuFitnessEntryRequest {
        key: String,
        score: f64,
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct CpuCandidateScoringPlanRequest {
        candidates: Vec<serde_json::Value>,
        fitness: Vec<CpuFitnessEntryRequest>,
    }

    let request: CpuCandidateScoringPlanRequest = serde_json::from_str(text).map_err(|error| {
        format!("CPU candidate scoring plan request is not valid JSON: {error}")
    })?;
    let candidates = request
        .candidates
        .iter()
        .map(parse_cpu_parameters_value)
        .collect::<Result<Vec<_>, _>>()?;
    let fitness = request
        .fitness
        .into_iter()
        .map(|entry| crate::cpu::search::CpuFitnessEntry {
            key: entry.key,
            score: entry.score,
        })
        .collect::<Vec<_>>();
    Ok(crate::cpu::search::cpu_candidate_scoring_plan(
        &candidates,
        &fitness,
    ))
}

fn cpu_fitness_entry_for_candidate_json(
    text: &str,
) -> Result<crate::cpu::search::CpuFitnessEntry, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct CpuFitnessEntryForCandidateRequest {
        parameters: serde_json::Value,
        score: f64,
    }

    let request: CpuFitnessEntryForCandidateRequest = serde_json::from_str(text)
        .map_err(|error| format!("CPU fitness entry request is not valid JSON: {error}"))?;
    let parameters = parse_cpu_parameters_value(&request.parameters)?;
    Ok(crate::cpu::search::cpu_fitness_entry_for_candidate(
        &parameters,
        request.score,
    ))
}

fn next_timeline_id_for(timelines: &[Timeline], color: Color) -> i32 {
    match color {
        Color::White => {
            timelines
                .iter()
                .map(|timeline| timeline.id)
                .filter(|id| *id > 0)
                .max()
                .unwrap_or(0)
                + 1
        }
        Color::Black => {
            timelines
                .iter()
                .map(|timeline| timeline.id)
                .filter(|id| *id < 0)
                .min()
                .unwrap_or(0)
                - 1
        }
    }
}
