use std::cell::RefCell;

use crate::{cpu::EvalWeights, *};

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
pub extern "C" fn chronofish_gpu_warmup_search_config_json(
    depth: i32,
    nodes: i32,
    search_time_ms: u64,
    exploration_temperature: f32,
) -> *const u8 {
    let config = crate::gpu::training::gpu_warmup_search_config(
        depth,
        nodes,
        search_time_ms,
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
pub extern "C" fn chronofish_training_worker_request_timeout_ms(nodes: i64, time_ms: i64) -> u64 {
    crate::gpu::training::worker_request_timeout_ms(nodes, time_ms)
}

#[no_mangle]
pub extern "C" fn chronofish_training_worker_search_time_ms(nodes: i64, time_ms: i64) -> u64 {
    crate::gpu::training::worker_search_time_ms(nodes, time_ms)
}

#[no_mangle]
pub extern "C" fn chronofish_normalized_search_score(score: i32) -> f32 {
    crate::gpu::training::normalized_search_score(score)
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

#[no_mangle]
pub extern "C" fn chronofish_frontier_max_cycles(
    requested_depth: i32,
    timeline_count: usize,
) -> i32 {
    crate::gpu::search::frontier_max_cycles(requested_depth, timeline_count)
}

#[no_mangle]
pub extern "C" fn chronofish_frontier_per_parent_limit(frontier_width: usize) -> i32 {
    crate::gpu::search::frontier_per_parent_limit(frontier_width)
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
    cpu_train_seconds: u64,
    cpu_training_time_ms: u64,
    cpu_max_match_plies: usize,
    cpu_max_match_time_ms: u64,
) -> u64 {
    crate::cpu::search::cpu_training_budget_ms(
        cpu_train_seconds,
        cpu_training_time_ms,
        cpu_max_match_plies,
        cpu_max_match_time_ms,
    )
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
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct ScoreDeltaRequest {
        candidate_score: i32,
        reference_score: i32,
        candidate_moves: Vec<crate::cpu::search::CpuTrainingMove>,
        reference_moves: Vec<crate::cpu::search::CpuTrainingMove>,
        draw_window: i32,
    }

    let Some(text) = wasm_input_text(ptr, len, "CPU reference score delta request") else {
        return std::ptr::null();
    };
    let request = match serde_json::from_str::<ScoreDeltaRequest>(text) {
        Ok(request) => request,
        Err(error) => {
            set_last_message(&format!(
                "CPU reference score delta request is not valid JSON: {error}"
            ));
            return std::ptr::null();
        }
    };
    let delta = crate::cpu::search::cpu_reference_score_delta(
        request.candidate_score,
        request.reference_score,
        &request.candidate_moves,
        &request.reference_moves,
        request.draw_window,
    );
    match serde_json::to_string(&delta) {
        Ok(json) => set_output(json),
        Err(error) => {
            set_last_message(&format!(
                "failed to encode CPU reference score delta: {error}"
            ));
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
pub extern "C" fn chronofish_cpu_training_candidate_count(cpu_candidates: usize) -> usize {
    crate::cpu::search::cpu_training_candidate_count(cpu_candidates)
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
pub extern "C" fn chronofish_gpu_candidate_inputs_json() -> *const u8 {
    let json = with_game(crate::gpu::search::gpu_candidate_inputs_json_from_game);
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_gpu_candidate_inputs_bytes() -> *const u8 {
    let words = with_game(crate::gpu::search::gpu_candidate_inputs_i32s_from_game);
    set_output_i32s(words)
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
