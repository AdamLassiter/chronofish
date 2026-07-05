use std::path::PathBuf;

use chronofish_engine::gpu::training::{
    align4,
    append_replay_samples,
    apply_draw_label,
    apply_outcome_label,
    auxiliary_value_targets,
    auxiliary_value_targets_bytes,
    bounded_value,
    byte_arrays_equal,
    clamp_training_integer,
    clamp_training_number,
    collect_search_label_samples,
    compact_model_is_finite,
    compact_training_samples,
    compact_value_model_architecture_matches_bytes,
    compact_value_model_encoded_len,
    compact_value_model_hidden_features_json,
    compact_value_model_policy_values,
    compact_value_model_policy_weights_bytes,
    concat_f32,
    concat_f32_bytes,
    count_non_zero,
    count_non_zero_bytes,
    cpu_baseline_mode_enabled,
    cpu_head_training_max_positions,
    cpu_prediction_max_batch,
    curriculum_board_times,
    curriculum_game_snapshot_json,
    curriculum_piece_type,
    curriculum_search_config,
    curriculum_stage,
    curriculum_timeline_limit,
    curriculum_timeline_priority,
    decode_compact_value_model,
    dedupe_training_samples,
    default_initial_hidden_weights,
    denormalized_search_score,
    dense_kernel_entry_point,
    distill_training_samples,
    encode_compact_value_model,
    f32_to_f16_bits,
    f32_to_f16_upload_bytes,
    f32_values_are_finite,
    feature_length,
    fill_grouped_training_batch_indices,
    fill_grouped_training_batch_indices_bytes,
    format_bytes,
    gpu_position_generation_search_config,
    gpu_rollout_max_plies,
    gpu_training_worker_count,
    gpu_warmup_plies,
    gpu_warmup_search_config,
    group_training_indices_by_position,
    has_policy_training_target,
    hidden_delta_params_bytes,
    hidden_features_from_projected,
    initial_hidden_weights,
    initial_hidden_weights_bytes,
    inverse_tanh,
    is_training_mode,
    is_training_subject,
    label_source_counts,
    layer_params_bytes,
    legacy_training_modes,
    legacy_training_subject,
    load_compact_value_model,
    loss_reduction_workgroup_count,
    min_hidden_training_positions,
    model_architecture_matches,
    normalize_training_modes,
    normalized_search_score,
    optimizer_velocity,
    outcome_label_for_turns,
    output_delta_params_bytes,
    output_layer_size,
    output_params_bytes,
    pack_sparse_projection_features,
    policy_bucket_from_move_values,
    policy_bucket_from_values,
    policy_logits_array,
    policy_params_bytes,
    policy_training_batch_size,
    policy_training_indices,
    policy_training_steps,
    policy_training_steps_per_submit,
    policy_training_target,
    policy_weights_array,
    previous_layer_size,
    project_features,
    projection_chunk_size,
    projection_hash,
    projection_params_bytes,
    projection_temporary_budget,
    quantized_policy_upload_bytes,
    replay_sample_priority,
    royal_capture_winner_snapshot_json,
    royal_count_snapshot_json,
    sample_from_snapshot_label,
    sample_plies,
    sample_seed,
    samples_from_partial_outcome,
    search_label_sample,
    search_seed_json,
    select_training_working_set_for_projection,
    select_training_working_set_indices_for_projection,
    select_training_working_set_with_capacity,
    shuffled_indices,
    shuffled_indices_bytes,
    sparse_projection_features_bytes,
    split_hidden_weights,
    split_hidden_weights_bytes,
    split_policy_training_indices,
    split_validation_samples,
    split_work,
    stable_sample_hash,
    tactical_position_priority_from_counts,
    tactical_position_priority_snapshot_json,
    tactical_search_config,
    take_training_sample_batches,
    train_policy_head_cpu,
    train_policy_head_from_features_cpu,
    train_value_head_cpu,
    train_value_head_from_features_cpu,
    training_batch_normalization,
    training_label_priority,
    training_label_weight,
    training_label_worker_count,
    training_mode_count,
    training_mode_enabled,
    training_weighted_average,
    training_workgroups_16,
    training_workgroups_64,
    unique_training_position_count,
    value_gpu_batches_per_submit,
    value_gpu_validation_interval,
    value_head_validation_interval,
    value_training_batch_size,
    worker_request_timeout_ms,
    worker_search_time_ms,
    xorshift32,
    CompactValueModelError,
    OutputActivation,
    SearchLabelBatchRequest,
    SearchLabelMode,
    SearchLabelSampleRequest,
    TrainingSample,
    TrainingWorkerSearchConfig,
    ValueHeadTrainingConfig,
    AUXILIARY_VALUE_HEAD_COUNT,
    CPU_HEAD_TRAINING_MAX_POSITIONS,
    CPU_PREDICTION_MAX_BATCH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_PARTIAL_OUTCOME_LABEL_KIND,
    DEFAULT_PARTIAL_OUTCOME_LABEL_WEIGHT,
    DEFAULT_PATIENCE,
    DEFAULT_PROJECTED_WORKING_SET_BYTES,
    DEFAULT_PROJECTION_SEED,
    DEFAULT_PROJECTION_SIZE,
    DEFAULT_VALIDATION_SPLIT,
    DEFAULT_WEIGHT_DECAY,
    GPU_POSITION_GENERATION_TIME_MS,
    GPU_WARMUP_MAX_TIME_MS,
    MAX_GPU_TRAINING_BATCH,
    MAX_GPU_TRAINING_SAMPLES,
    MAX_GPU_VALIDATION_INTERVAL,
    MIN_HIDDEN_TRAINING_POSITIONS,
    OPTIMIZER_MOMENTUM,
    POLICY_STEPS_PER_SUBMIT,
    PROJECTION_CHUNK_SIZE,
    PROJECTION_TEMPORARY_BUDGET,
    TILED_TRAINING_MIN_BATCH,
    VALUE_EPOCHS_PER_SUBMIT,
};

#[test]
fn decodes_compact_value_model_v5() {
    let bytes = encode_test_model(TestModel {
        version: 5,
        projection_size: 8,
        projection_seed: 123,
        hidden_layers: vec![4, 2],
        hidden_weights: vec![0.25, -0.5, 0.75],
        output_weights: vec![1.0, -1.0],
        policy_values: vec![0.1, 0.2, 0.3],
        auxiliary_value_weights: vec![0.4, 0.5],
        scale: 2.0,
        bias: -0.25,
    });

    let model = decode_compact_value_model(&bytes).expect("decode test CFNN model");

    assert_eq!(model.version, 5);
    assert_eq!(model.projection_size, 8);
    assert_eq!(model.projection_seed, 123);
    assert_eq!(model.hidden_layers, vec![4, 2]);
    assert_eq!(model.hidden_weights, vec![0.25, -0.5, 0.75]);
    assert_eq!(model.output_weights, vec![1.0, -1.0]);
    assert_eq!(model.policy_weights, vec![0.1, 0.2, 0.3]);
    assert!(model.policy_logits.is_empty());
    assert_eq!(model.auxiliary_value_weights, vec![0.4, 0.5]);
    assert_eq!(model.output_activation, OutputActivation::Tanh);
}

#[test]
fn compact_value_model_round_trips_through_rust_encoder() {
    let bytes = encode_test_model(TestModel {
        version: 5,
        projection_size: 2,
        projection_seed: 123,
        hidden_layers: vec![1],
        hidden_weights: vec![0.5, 0.25, -0.5],
        output_weights: vec![0.75, 0.1],
        policy_values: vec![0.2, -0.2],
        auxiliary_value_weights: vec![0.3],
        scale: 1.0,
        bias: 0.0,
    });
    let model = decode_compact_value_model(&bytes).expect("decode model");

    let encoded = encode_compact_value_model(&model);

    assert_eq!(encoded, bytes);
    assert_eq!(encoded.len(), compact_value_model_encoded_len(&model));
    assert_eq!(compact_value_model_policy_values(&model), &[0.2, -0.2]);
}

#[test]
fn compact_value_model_architecture_match_decodes_from_bytes() {
    let bytes = encode_test_model(TestModel {
        version: 5,
        projection_size: DEFAULT_PROJECTION_SIZE as u32,
        projection_seed: DEFAULT_PROJECTION_SEED,
        hidden_layers: DEFAULT_HIDDEN_LAYERS.to_vec(),
        hidden_weights: default_initial_hidden_weights(),
        output_weights: vec![0.0; DEFAULT_HIDDEN_LAYERS.last().copied().unwrap() as usize + 1],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    });

    assert!(compact_value_model_architecture_matches_bytes(&bytes));

    let mut wrong_projection = bytes.clone();
    wrong_projection[8..12].copy_from_slice(&4u32.to_le_bytes());

    assert!(!compact_value_model_architecture_matches_bytes(
        &wrong_projection
    ));
    assert!(!compact_value_model_architecture_matches_bytes(
        b"not a cfnn"
    ));
}

#[test]
fn compact_value_model_encoder_respects_format_version_policy_fields() {
    let mut model = decode_compact_value_model(&encode_test_model(TestModel {
        version: 2,
        projection_size: 2,
        projection_seed: 123,
        hidden_layers: vec![1],
        hidden_weights: vec![0.5],
        output_weights: vec![0.75],
        policy_values: vec![0.2, -0.2],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    }))
    .expect("decode v2 model");

    assert_eq!(compact_value_model_policy_values(&model), &[0.2, -0.2]);
    assert_eq!(
        encode_compact_value_model(&model).len(),
        compact_value_model_encoded_len(&model)
    );

    model.version = 1;
    model.policy_logits.clear();
    model.policy_weights = vec![1.0, 2.0, 3.0];
    let encoded = encode_compact_value_model(&model);
    let decoded = decode_compact_value_model(&encoded).expect("decode v1 without policy payload");

    assert!(compact_value_model_policy_values(&model).is_empty());
    assert_eq!(encoded.len(), compact_value_model_encoded_len(&model));
    assert!(decoded.policy_logits.is_empty());
    assert!(decoded.policy_weights.is_empty());
}

#[test]
fn predicts_values_with_webgpu_trainer_cpu_math() {
    let bytes = encode_test_model(TestModel {
        version: 1,
        projection_size: 2,
        projection_seed: 123,
        hidden_layers: vec![],
        hidden_weights: vec![],
        output_weights: vec![0.5, -0.25, 0.1],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    });
    let model = decode_compact_value_model(&bytes).expect("decode linear model");

    assert_eq!(projection_hash(0, 0, 123), 1_127_256_899);
    assert_eq!(projection_hash(1, 1, 123), 506_350_003);
    let projected = project_features(&[1.0, 2.0], 2, 123);
    let expected_first = 1.0 / 2.0_f32.sqrt();
    assert!((projected[0] - expected_first).abs() < 1e-6);
    assert!((projected[1] + expected_first).abs() < 1e-6);

    let prediction = model.predict_value(&[1.0, 2.0]);
    let expected = 0.5 * expected_first + -0.25 * -expected_first + 0.1;
    assert!((prediction - expected).abs() < 1e-6);
}

#[test]
fn distills_training_samples_with_compact_value_model() {
    let bytes = encode_test_model(TestModel {
        version: 1,
        projection_size: 1,
        projection_seed: 123,
        hidden_layers: vec![],
        hidden_weights: vec![],
        output_weights: vec![0.5, 0.1],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    });
    let model = decode_compact_value_model(&bytes).expect("decode distillation model");
    let mut sample = training_sample_with("root", "search", 2.0, 0.75, Some(42), vec![1.0, 2.0]);
    sample.pseudo = Some(false);

    let distilled = distill_training_samples(&[sample.clone()], &model);

    assert_eq!(distilled.len(), 1);
    assert_eq!(distilled[0].position_key, sample.position_key);
    assert_eq!(distilled[0].features, sample.features);
    assert_eq!(distilled[0].label, model.predict_value(&sample.features));
    assert_eq!(distilled[0].policy, None);
    assert_eq!(distilled[0].label_kind.as_deref(), Some("distilled"));
    assert_eq!(distilled[0].label_weight, 0.25);
    assert_eq!(distilled[0].pseudo, Some(true));
}

#[test]
fn value_targets_and_search_scores_use_browser_training_scale() {
    assert_eq!(bounded_value(0.5), 0.5);
    assert_eq!(bounded_value(2.0), 1.0);
    assert_eq!(bounded_value(-2.0), -1.0);
    assert_eq!(bounded_value(f32::NAN), 0.0);
    assert_eq!(normalized_search_score(20_000), 1.0);
    assert_eq!(normalized_search_score(-10_000), -0.5);
    assert_eq!(normalized_search_score(100_000), 1.0);
    assert_eq!(denormalized_search_score(0.75), 15_000);
    assert_eq!(denormalized_search_score(-2.0), -20_000);
    assert!((inverse_tanh(0.75).tanh() - 0.75).abs() < 1e-6);
}

#[test]
fn training_math_helpers_match_browser_layout_policy() {
    assert_eq!(loss_reduction_workgroup_count(1), 1);
    assert_eq!(loss_reduction_workgroup_count(64), 1);
    assert_eq!(loss_reduction_workgroup_count(65), 2);
    assert_eq!(loss_reduction_workgroup_count(4096), 64);
    assert_eq!(training_workgroups_16(0), 0);
    assert_eq!(training_workgroups_16(1), 1);
    assert_eq!(training_workgroups_16(16), 1);
    assert_eq!(training_workgroups_16(17), 2);
    assert_eq!(training_workgroups_64(0), 0);
    assert_eq!(training_workgroups_64(64), 1);
    assert_eq!(training_workgroups_64(65), 2);
    assert_eq!(projection_temporary_budget(0), 1);
    assert_eq!(projection_temporary_budget(64), 32);
    assert_eq!(
        projection_temporary_budget(512 * 1024 * 1024),
        PROJECTION_TEMPORARY_BUDGET
    );

    let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let layers = split_hidden_weights(&hidden, 2, &[2, 1]);
    assert_eq!(layers, vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![7.0]]);
    assert_eq!(concat_f32(&layers), hidden);
    assert_eq!(count_non_zero(&[0.0, -1.0, 0.0, 2.0]), 2);

    let mut split_request = Vec::new();
    for value in [2_u32, 2, hidden.len() as u32, 2, 1] {
        split_request.extend_from_slice(&value.to_le_bytes());
    }
    for value in &hidden {
        split_request.extend_from_slice(&value.to_le_bytes());
    }
    let split_response = split_hidden_weights_bytes(&split_request).unwrap();
    let mut cursor = 0;
    let read_u32 = |bytes: &[u8], cursor: &mut usize| {
        let value = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
        *cursor += 4;
        value
    };
    assert_eq!(read_u32(&split_response, &mut cursor), 2);
    assert_eq!(read_u32(&split_response, &mut cursor), 6);
    assert_eq!(read_u32(&split_response, &mut cursor), 1);
    let split_values = split_response[cursor..]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(split_values, hidden);

    let mut concat_request = Vec::new();
    for value in [2_u32, 2, 1] {
        concat_request.extend_from_slice(&value.to_le_bytes());
    }
    for value in [8.0_f32, 9.0, 10.0] {
        concat_request.extend_from_slice(&value.to_le_bytes());
    }
    let concat_values = concat_f32_bytes(&concat_request)
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(concat_values, vec![8.0, 9.0, 10.0]);

    let mut count_request = Vec::new();
    for value in [0.0_f32, -1.0, 0.0, 2.0] {
        count_request.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(count_non_zero_bytes(&count_request).unwrap(), 2);

    let layer_params = layer_params_bytes(8, 32, 64, 0.25, 0.01, 0.9);
    assert_eq!(
        u32::from_le_bytes(layer_params[0..4].try_into().unwrap()),
        8
    );
    assert_eq!(
        u32::from_le_bytes(layer_params[4..8].try_into().unwrap()),
        32
    );
    assert_eq!(
        u32::from_le_bytes(layer_params[8..12].try_into().unwrap()),
        64
    );
    assert_eq!(
        f32::from_le_bytes(layer_params[12..16].try_into().unwrap()),
        0.25
    );
    assert_eq!(
        f32::from_le_bytes(layer_params[16..20].try_into().unwrap()),
        0.01
    );
    assert_eq!(
        f32::from_le_bytes(layer_params[20..24].try_into().unwrap()),
        0.9
    );

    let output_params = output_params_bytes(9, 128, 0.5, 0.02, 0.75);
    assert_eq!(
        u32::from_le_bytes(output_params[0..4].try_into().unwrap()),
        9
    );
    assert_eq!(
        u32::from_le_bytes(output_params[4..8].try_into().unwrap()),
        128
    );
    assert_eq!(
        f32::from_le_bytes(output_params[12..16].try_into().unwrap()),
        0.5
    );
    assert_eq!(
        f32::from_le_bytes(output_params[16..20].try_into().unwrap()),
        0.02
    );
    assert_eq!(
        f32::from_le_bytes(output_params[20..24].try_into().unwrap()),
        0.75
    );

    let projection_params = projection_params_bytes(4, 128, 2048, u32::MAX, 16);
    assert_eq!(
        u32::from_le_bytes(projection_params[0..4].try_into().unwrap()),
        4
    );
    assert_eq!(
        u32::from_le_bytes(projection_params[4..8].try_into().unwrap()),
        128
    );
    assert_eq!(
        u32::from_le_bytes(projection_params[8..12].try_into().unwrap()),
        2048
    );
    assert_eq!(
        u32::from_le_bytes(projection_params[12..16].try_into().unwrap()),
        u32::MAX
    );
    assert_eq!(
        u32::from_le_bytes(projection_params[16..20].try_into().unwrap()),
        16
    );
}

#[test]
fn initial_hidden_weights_follow_browser_hash_initialization() {
    let weights = initial_hidden_weights(2, &[2, 1]);

    assert_eq!(weights.len(), 9);
    assert_eq!(weights[2], 0.0);
    assert_eq!(weights[5], 0.0);
    assert_eq!(weights[8], 0.0);

    let expected_first = (((projection_hash(0, 0, 2_166_136_261) as f32 / u32::MAX as f32) * 2.0)
        - 1.0)
        * (2.0_f32 / 2.0).sqrt();
    let expected_second_layer =
        (((projection_hash(0, 4099, 2_166_136_261) as f32 / u32::MAX as f32) * 2.0) - 1.0)
            * (2.0_f32 / 2.0).sqrt();

    assert!((weights[0] - expected_first).abs() < 1e-6);
    assert!((weights[6] - expected_second_layer).abs() < 1e-6);

    let mut request = Vec::new();
    for value in [2_u32, 2, 2, 1] {
        request.extend_from_slice(&value.to_le_bytes());
    }
    let decoded = initial_hidden_weights_bytes(&request)
        .expect("initial hidden weights bytes")
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(decoded, weights);
}

#[test]
fn training_worker_scheduling_helpers_match_browser_policy() {
    assert_eq!(split_work(10, 3), vec![4, 3, 3]);
    assert_eq!(split_work(2, 8), vec![1, 1]);
    assert!(split_work(0, 4).is_empty());
    assert!(split_work(5, 0).is_empty());
    let batches = vec![
        vec![
            training_sample("a", "outcome", 1.0, vec![1.0]),
            training_sample("b", "outcome", 1.0, vec![2.0]),
        ],
        vec![training_sample("c", "outcome", 1.0, vec![3.0])],
    ];
    let capped = take_training_sample_batches(&batches, 2);
    assert_eq!(
        capped
            .iter()
            .map(|sample| sample.position_key.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("a"), Some("b")]
    );
    assert!(take_training_sample_batches(&batches, 0).is_empty());
    let compacted = compact_training_samples(&[
        Some(training_sample("present-a", "search", 1.0, vec![1.0])),
        None,
        Some(training_sample("present-b", "search", 1.0, vec![2.0])),
    ]);
    assert_eq!(
        compacted
            .iter()
            .map(|sample| sample.position_key.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("present-a"), Some("present-b")]
    );

    assert_eq!(gpu_training_worker_count(0, 8), 0);
    assert_eq!(gpu_training_worker_count(12, 8), 8);
    assert_eq!(gpu_training_worker_count(32, 99), 16);
    assert_eq!(gpu_training_worker_count(5, 0), 1);

    assert_eq!(training_label_worker_count(0, None, 12), 0);
    assert_eq!(training_label_worker_count(12, Some(99), 12), 8);
    assert_eq!(training_label_worker_count(3, Some(99), 12), 3);
    assert_eq!(training_label_worker_count(12, Some(0), 12), 1);
    assert_eq!(training_label_worker_count(12, None, 16), 8);
    assert_eq!(training_label_worker_count(12, None, 2), 3);

    assert_eq!(sample_plies(0, false), 1);
    assert_eq!(sample_plies(9, false), 10);
    assert_eq!(sample_plies(10, false), 1);
    assert_eq!(sample_plies(4, true), 9);
    assert_eq!(sample_plies(5, true), 1);

    assert_eq!(GPU_WARMUP_MAX_TIME_MS, 5_000);
    assert_eq!(GPU_POSITION_GENERATION_TIME_MS, 3_000);
    assert_eq!(gpu_warmup_plies(0), 0);
    assert_eq!(gpu_warmup_plies(1), 2);
    assert_eq!(gpu_warmup_plies(8), 9);
    assert_eq!(gpu_warmup_plies(9), 1);
    assert_eq!(gpu_rollout_max_plies(0, 0), 10);
    assert_eq!(gpu_rollout_max_plies(8, 1), 10);
    assert_eq!(gpu_rollout_max_plies(12, 3), 15);

    assert_eq!(
        gpu_warmup_search_config(5, 50_000, 29_000, 0.25),
        TrainingWorkerSearchConfig {
            depth: 2,
            nodes: 1024,
            time_ms: 5_000,
            exploration_temperature: 0.25,
        }
    );
    assert_eq!(
        gpu_warmup_search_config(0, -1, 400, 0.5),
        TrainingWorkerSearchConfig {
            depth: 1,
            nodes: 1,
            time_ms: 400,
            exploration_temperature: 0.5,
        }
    );
    assert_eq!(
        gpu_position_generation_search_config(5, 50_000, 0.35),
        TrainingWorkerSearchConfig {
            depth: 2,
            nodes: 512,
            time_ms: 3_000,
            exploration_temperature: 0.35,
        }
    );
    assert_eq!(
        gpu_position_generation_search_config(0, -1, 0.1),
        TrainingWorkerSearchConfig {
            depth: 1,
            nodes: 1,
            time_ms: 3_000,
            exploration_temperature: 0.1,
        }
    );
}

#[test]
fn training_worker_timeout_helpers_match_browser_policy() {
    assert_eq!(worker_request_timeout_ms(1, 0), 30_000);
    assert_eq!(worker_request_timeout_ms(20_000, 0), 60_000);
    assert_eq!(worker_request_timeout_ms(10, 80_000), 81_000);
    assert_eq!(worker_request_timeout_ms(1_000_000, 0), 120_000);
    assert_eq!(worker_request_timeout_ms(-5, -1), 30_000);

    assert_eq!(worker_search_time_ms(1, 0), 29_000);
    assert_eq!(worker_search_time_ms(10, 500), 29_000);
    assert_eq!(worker_search_time_ms(10, 125_000), 125_000);
}

#[test]
fn auxiliary_value_targets_match_browser_head_order() {
    let mut features = vec![0.0; 32 * 64];
    features[25 * 64] = 1.0;
    features[27 * 64] = 1.0;
    features[31 * 64] = 0.25;
    features[0] = 1.0;
    features[13 * 64] = 1.0;
    let sample = training_sample_with("aux", "search", 1.0, 0.2, Some(12), features);

    let targets = auxiliary_value_targets(&sample);

    assert_eq!(targets.len(), AUXILIARY_VALUE_HEAD_COUNT);
    assert_eq!(targets[0], 1.0);
    assert_eq!(targets[1], 0.2);
    assert_eq!(targets[2], 0.25);
    assert_eq!(targets[3], 0.5);
    assert_eq!(targets[4], 1.0);
    assert_eq!(targets[5], 1.0);
    assert_eq!(targets[6], 7.0 / 16.0);
    assert_eq!(targets[7], 0.0);
    assert_eq!(targets[8], 0.0);

    let bytes = auxiliary_value_targets_bytes(&[sample]);
    assert_eq!(
        bytes.len(),
        AUXILIARY_VALUE_HEAD_COUNT * std::mem::size_of::<f32>()
    );
    assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1.0);
}

#[test]
fn training_config_clamps_match_browser_policy() {
    assert_eq!(VALUE_EPOCHS_PER_SUBMIT, 64);
    assert_eq!(POLICY_STEPS_PER_SUBMIT, 64);
    assert_eq!(DEFAULT_BATCH_SIZE, 1024);
    assert_eq!(DEFAULT_VALIDATION_SPLIT, 0.1);
    assert_eq!(DEFAULT_PATIENCE, 12);
    assert_eq!(DEFAULT_WEIGHT_DECAY, 0.00001);
    assert_eq!(MAX_GPU_TRAINING_SAMPLES, 16_384);
    assert_eq!(MAX_GPU_TRAINING_BATCH, 16_384);
    assert_eq!(MAX_GPU_VALIDATION_INTERVAL, 16_384);
    assert_eq!(TILED_TRAINING_MIN_BATCH, 16);
    assert_eq!(CPU_PREDICTION_MAX_BATCH, 4);
    assert_eq!(cpu_prediction_max_batch(), CPU_PREDICTION_MAX_BATCH);
    assert_eq!(MIN_HIDDEN_TRAINING_POSITIONS, 256);
    assert_eq!(
        min_hidden_training_positions(),
        MIN_HIDDEN_TRAINING_POSITIONS
    );
    assert_eq!(CPU_HEAD_TRAINING_MAX_POSITIONS, 32);
    assert_eq!(
        cpu_head_training_max_positions(),
        CPU_HEAD_TRAINING_MAX_POSITIONS
    );
    assert_eq!(OPTIMIZER_MOMENTUM, 0.9);
    assert_eq!(PROJECTION_CHUNK_SIZE, 256);
    assert_eq!(projection_chunk_size(), PROJECTION_CHUNK_SIZE);
    assert_eq!(PROJECTION_TEMPORARY_BUDGET, 128 * 1024 * 1024);
    assert!((optimizer_velocity(0.25, 0.75, OPTIMIZER_MOMENTUM) - 0.3).abs() < 1e-6);
    assert_eq!(
        dense_kernel_entry_point("forward_layer", 15),
        "forward_layer_naive"
    );
    assert_eq!(
        dense_kernel_entry_point("forward_layer", 16),
        "forward_layer"
    );
    assert_eq!(format_bytes(512 * 1024), "0.5 MiB");
    assert_eq!(format_bytes(9 * 1024 * 1024), "9.0 MiB");
    assert_eq!(format_bytes(10 * 1024 * 1024), "10 MiB");
    assert_eq!(align4(0), 0);
    assert_eq!(align4(1), 4);
    assert_eq!(align4(4), 4);
    assert_eq!(align4(5), 8);
    let output_delta = output_delta_params_bytes(32, -4.0);
    assert_eq!(output_delta.len(), 16);
    assert_eq!(
        u32::from_le_bytes(output_delta[0..4].try_into().unwrap()),
        32
    );
    assert_eq!(
        f32::from_le_bytes(output_delta[4..8].try_into().unwrap()),
        0.0
    );
    let hidden_delta = hidden_delta_params_bytes(48, 64, 128);
    assert_eq!(hidden_delta.len(), 16);
    assert_eq!(
        u32::from_le_bytes(hidden_delta[0..4].try_into().unwrap()),
        48
    );
    assert_eq!(
        u32::from_le_bytes(hidden_delta[4..8].try_into().unwrap()),
        64
    );
    assert_eq!(
        u32::from_le_bytes(hidden_delta[8..12].try_into().unwrap()),
        128
    );
    let policy_params = policy_params_bytes(16, 2048, 12.5, 0.01, 0.00001, OPTIMIZER_MOMENTUM);
    assert_eq!(policy_params.len(), 32);
    assert_eq!(
        u32::from_le_bytes(policy_params[0..4].try_into().unwrap()),
        16
    );
    assert_eq!(
        u32::from_le_bytes(policy_params[4..8].try_into().unwrap()),
        2048
    );
    assert_eq!(
        u32::from_le_bytes(policy_params[8..12].try_into().unwrap()),
        257
    );
    assert_eq!(
        f32::from_le_bytes(policy_params[16..20].try_into().unwrap()),
        12.5
    );

    assert_eq!(clamp_training_number(Some(0.05), 0.0001, 0.1, 0.01), 0.05);
    assert_eq!(clamp_training_number(Some(-1.0), 0.0, 2.0, 0.25), 0.0);
    assert_eq!(clamp_training_number(Some(5.0), 0.0, 2.0, 0.25), 2.0);
    assert_eq!(clamp_training_number(None, 0.0, 2.0, 0.25), 0.25);
    assert_eq!(clamp_training_number(Some(f64::NAN), 0.0, 2.0, 0.25), 0.25);
    assert_eq!(
        clamp_training_number(Some(f64::INFINITY), 0.0, 2.0, 0.25),
        0.25
    );

    assert_eq!(clamp_training_integer(Some(63.4), 1, 16_384, 64), 63);
    assert_eq!(clamp_training_integer(Some(63.5), 1, 16_384, 64), 64);
    assert_eq!(clamp_training_integer(Some(0.0), 1, 16_384, 64), 1);
    assert_eq!(
        clamp_training_integer(Some(20_000.0), 1, 16_384, 64),
        16_384
    );
    assert_eq!(clamp_training_integer(None, 1, 16_384, 64), 64);
}

#[test]
fn training_worker_seed_helpers_match_browser_hashing() {
    assert_eq!(sample_seed("loss-log", 3, 1234), 2_200_105_291);
    assert_eq!(sample_seed("gpu-label", 17, 0), 176_132_928);
    assert_eq!(sample_seed("π", 2, 42), 3_283_295_188);

    assert_eq!(search_seed_json(None, 1234), 3_974_300_691);
    assert_eq!(
        search_seed_json(Some("{\"a\":1,\"b\":\"x\"}"), 1234),
        2_107_325_072
    );
    assert_eq!(search_seed_json(Some("[3,2,1]"), 0), 3_534_733_298);
}

#[test]
fn curriculum_training_helpers_match_browser_policy() {
    assert_eq!(curriculum_stage(0), 0);
    assert_eq!(curriculum_stage(5), 5);
    assert_eq!(curriculum_stage(6), 0);

    assert_eq!(curriculum_timeline_limit(0, 4), 1);
    assert_eq!(curriculum_timeline_limit(2, 4), 2);
    assert_eq!(curriculum_timeline_limit(4, 1), 1);
    assert_eq!(curriculum_timeline_limit(4, 6), 4);
    assert_eq!(curriculum_timeline_limit(5, 0), 0);

    assert_eq!(curriculum_board_times(&[1, 3, 5], 3, 0), vec![3]);
    assert_eq!(curriculum_board_times(&[1, 3, 5], 4, 1), vec![5]);
    assert_eq!(curriculum_board_times(&[1, 3, 5], 3, 2), vec![3, 5]);
    assert_eq!(curriculum_board_times(&[1, 3, 5, 8], 99, 3), vec![5, 8]);
    assert_eq!(
        curriculum_board_times(&[1, 3, 5, 8, 13, 21], 3, 4),
        vec![3, 5, 8, 13, 21]
    );

    assert_eq!(curriculum_piece_type("king", 0).as_deref(), Some("king"));
    assert_eq!(
        curriculum_piece_type("royalQueen", 0).as_deref(),
        Some("queen")
    );
    assert_eq!(curriculum_piece_type("dragon", 0), None);
    assert_eq!(
        curriculum_piece_type("dragon", 2).as_deref(),
        Some("dragon")
    );

    assert!((curriculum_timeline_priority(true, true, Some(12)) - 6.012).abs() < 1e-12);
    assert!((curriculum_timeline_priority(false, true, Some(5)) - 2.005).abs() < 1e-12);
    assert!(curriculum_timeline_priority(false, false, None).is_infinite());
}

#[test]
fn curriculum_game_snapshot_json_shapes_snapshot_in_engine() {
    let snapshot = training_snapshot_json(&[
        timeline_json(
            0,
            0,
            "neutral",
            &[
                board_json(0, &[piece_json(4, 0, "white", "king")]),
                board_json(
                    2,
                    &[
                        piece_json(4, 0, "white", "king"),
                        piece_json(4, 7, "black", "king"),
                        piece_json(2, 0, "white", "royalQueen"),
                        piece_json(6, 0, "white", "dragon"),
                    ],
                ),
            ],
        ),
        timeline_json(
            -1,
            -1,
            "black",
            &[board_json(4, &[piece_json(4, 7, "black", "king")])],
        ),
        timeline_json(
            1,
            1,
            "white",
            &[board_json(3, &[piece_json(0, 0, "white", "dragon")])],
        ),
    ]);

    let output = curriculum_game_snapshot_json(&snapshot, 0).expect("curriculum snapshot");
    let json: serde_json::Value = serde_json::from_str(&output).expect("valid output json");

    let timelines = json["timelines"].as_array().expect("timelines array");
    assert_eq!(timelines.len(), 1);
    assert_eq!(timelines[0]["id"], 0);
    assert_eq!(timelines[0]["row"], 0);
    assert_eq!(timelines[0]["active"], true);
    let boards = timelines[0]["boards"].as_array().expect("boards array");
    assert_eq!(boards.len(), 1);
    assert_eq!(boards[0]["time"], 2);
    assert_eq!(boards[0]["board"][0][2]["type"], "queen");
    assert!(boards[0]["board"][0][6].is_null());
}

#[test]
fn curriculum_and_tactical_search_configs_match_browser_policy() {
    let early = curriculum_search_config(5, 10_000, 0.1, 0);
    assert_eq!(early.depth, 1);
    assert_eq!(early.nodes, 512);
    assert!((early.exploration_temperature - 0.15).abs() < 1e-6);

    let late = curriculum_search_config(5, 10_000, 0.9, 5);
    assert_eq!(late.depth, 3);
    assert_eq!(late.nodes, 3_072);
    assert!((late.exploration_temperature - 0.6).abs() < 1e-6);

    let tactical_min = tactical_search_config(1, 500, 0.1, 0);
    assert_eq!(tactical_min.depth, 2);
    assert_eq!(tactical_min.nodes, 1_024);
    assert!((tactical_min.exploration_temperature - 0.4).abs() < 1e-6);

    let tactical_cap = tactical_search_config(10, 99_999, 0.95, 3);
    assert_eq!(tactical_cap.depth, 6);
    assert_eq!(tactical_cap.nodes, 8_192);
    assert!((tactical_cap.exploration_temperature - 0.8).abs() < 1e-6);
}

#[test]
fn training_mode_routing_matches_browser_policy() {
    assert!(is_training_subject("gpu"));
    assert!(is_training_subject("cpu"));
    assert!(!is_training_subject("both"));
    assert!(is_training_mode("tactical"));
    assert!(!is_training_mode("unknown"));
    assert_eq!(legacy_training_subject(Some("trainCpu")), "cpu");
    assert_eq!(legacy_training_subject(Some("trainBoth")), "gpu");

    assert_eq!(
        legacy_training_modes("cpu", None, Some("vsBoth"), None),
        vec!["vsCpu".to_string(), "vsGpu".to_string()]
    );
    assert_eq!(
        legacy_training_modes("gpu", Some("trainBoth"), None, Some("mixed")),
        vec![
            "vsCpu".to_string(),
            "vsGpu".to_string(),
            "self".to_string(),
            "distill".to_string(),
            "curriculum".to_string(),
            "tactical".to_string(),
        ]
    );
    assert_eq!(
        legacy_training_modes("gpu", None, None, Some("selfPlay")),
        vec!["self".to_string()]
    );
    assert_eq!(
        legacy_training_modes("gpu", Some("trainCpu"), None, None),
        vec!["vsCpu".to_string()]
    );

    let explicit = normalize_training_modes(
        &["vsGpu", "distill", "bad", "vsGpu", "self"],
        "cpu",
        None,
        None,
        None,
    );
    assert_eq!(explicit, vec!["vsGpu".to_string(), "self".to_string()]);
    assert_eq!(
        normalize_training_modes(&[], "cpu", None, None, None),
        vec!["vsCpu".to_string()]
    );
    assert_eq!(
        normalize_training_modes(&["bad"], "gpu", None, None, None),
        vec!["vsGpu".to_string(), "self".to_string()]
    );

    assert!(training_mode_enabled(&explicit, "self"));
    assert!(cpu_baseline_mode_enabled(&explicit));
    assert_eq!(training_mode_count("cpu", &explicit), 2);
    assert_eq!(
        training_mode_count(
            "gpu",
            &[
                "vsGpu".to_string(),
                "distill".to_string(),
                "self".to_string()
            ]
        ),
        3
    );
}

#[test]
fn outcome_label_backfill_matches_browser_policy() {
    let near = outcome_label_for_turns("white", "white", 4, 4).expect("outcome label");
    let discounted = outcome_label_for_turns("white", "black", 1, 4).expect("outcome label");

    assert_eq!(near, 1.0);
    assert!((discounted + 0.96_f32.powi(3)).abs() < 1e-6);
    assert!(outcome_label_for_turns("neutral", "white", 0, 1).is_err());

    let mut sample = training_sample_with("outcome", "search", 0.5, 0.25, None, vec![1.0]);
    apply_outcome_label(&mut sample, "black", "white", 2, 5).expect("apply outcome label");
    assert!((sample.label + 0.96_f32.powi(3)).abs() < 1e-6);
    assert_eq!(sample.label_kind.as_deref(), Some("outcome"));
    assert_eq!(sample.label_weight, 1.25);

    apply_draw_label(&mut sample, "duel", 1.1);
    assert_eq!(sample.label, 0.0);
    assert_eq!(sample.label_kind.as_deref(), Some("duel"));
    assert_eq!(sample.label_weight, 1.1);
}

#[test]
fn tactical_position_priority_matches_browser_formula() {
    assert_eq!(tactical_position_priority_from_counts(0, 1, 1, 0, 0), 0);
    assert_eq!(tactical_position_priority_from_counts(1, 3, 4, 2, 2), 9);
    assert_eq!(tactical_position_priority_from_counts(4, 1, 2, 9, 1), 5);

    let snapshot = training_snapshot_json(&[
        timeline_json(
            -1,
            -1,
            "black",
            &[board_json(5, &[piece_json(4, 7, "black", "king")])],
        ),
        timeline_json(
            0,
            0,
            "neutral",
            &[board_json(
                5,
                &[
                    piece_json(4, 0, "white", "king"),
                    piece_json(3, 0, "white", "queen"),
                    piece_json(2, 0, "white", "royalQueen"),
                ],
            )],
        ),
        timeline_json(
            1,
            1,
            "white",
            &[board_json(5, &[piece_json(0, 0, "white", "dragon")])],
        ),
    ]);

    assert_eq!(
        tactical_position_priority_snapshot_json(&snapshot).expect("priority"),
        6
    );
}

#[test]
fn royal_count_and_capture_winner_use_latest_engine_boards() {
    let before = training_snapshot_json(&[timeline_json(
        0,
        0,
        "neutral",
        &[
            board_json(
                0,
                &[
                    piece_json(4, 0, "white", "king"),
                    piece_json(4, 7, "black", "king"),
                    piece_json(3, 7, "black", "royalQueen"),
                ],
            ),
            board_json(
                1,
                &[
                    piece_json(4, 0, "white", "king"),
                    piece_json(4, 7, "black", "king"),
                ],
            ),
        ],
    )]);
    let after = training_snapshot_json(&[timeline_json(
        0,
        0,
        "neutral",
        &[board_json(1, &[piece_json(4, 0, "white", "king")])],
    )]);

    assert_eq!(
        royal_count_snapshot_json(&before, "black").expect("black royal count"),
        1
    );
    assert_eq!(
        royal_count_snapshot_json(&before, "white").expect("white royal count"),
        1
    );
    assert_eq!(
        royal_capture_winner_snapshot_json(&before, &after, "white").expect("capture winner"),
        Some("white")
    );
    assert_eq!(
        royal_capture_winner_snapshot_json(&before, &after, "black").expect("capture winner"),
        None
    );
    assert!(royal_count_snapshot_json(&before, "neutral").is_err());
}

#[test]
fn policy_buckets_match_browser_move_hashing() {
    let first = policy_bucket_from_move_values(1, 4, 2, 2, 1, 4, 4, 3, 0);
    let translated = policy_bucket_from_move_values(9, 40, 2, 2, 9, 40, 4, 3, 0);
    let different_geometry = policy_bucket_from_move_values(1, 4, 2, 2, 1, 4, 3, 4, 0);

    assert_eq!(first, 18);
    assert_eq!(translated, 18);
    assert_eq!(different_geometry, 240);
    assert_eq!(
        policy_bucket_from_move_values(1, 4, 2, 2, 1, 4, 4, 3, 1),
        193
    );
    assert_eq!(
        policy_bucket_from_move_values(1, 4, 2, 2, 1, 4, 4, 3, 6),
        255
    );
    assert_eq!(
        policy_bucket_from_move_values(3, 8, 6, 6, 1, 5, 4, 2, 0),
        197
    );
    assert_eq!(policy_bucket_from_values([0, 0, 2, 1, 2, 2, 0]), first);
}

#[test]
fn applies_hidden_layers_tanh_scale_bias_and_bounds() {
    let bytes = encode_test_model(TestModel {
        version: 4,
        projection_size: 1,
        projection_seed: 123,
        hidden_layers: vec![1],
        hidden_weights: vec![2.0, -0.5],
        output_weights: vec![3.0, 0.25],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 2.0,
        bias: 0.5,
    });
    let model = decode_compact_value_model(&bytes).expect("decode tanh model");

    assert_eq!(model.output_activation, OutputActivation::Tanh);
    assert!((model.predict_value(&[0.0]) - (0.25_f32.tanh() * 2.0 + 0.5)).abs() < 1e-6);
    assert!(model.predict_value(&[1.0]).abs() <= 1.0);
}

#[test]
fn trains_value_head_on_cpu_and_encodes_updated_model() {
    let bytes = encode_test_model(TestModel {
        version: 4,
        projection_size: 1,
        projection_seed: 123,
        hidden_layers: vec![],
        hidden_weights: vec![],
        output_weights: vec![0.0, 0.0],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    });
    let model = decode_compact_value_model(&bytes).expect("decode trainable model");
    let samples = vec![TrainingSample {
        side_to_move: None,
        board_count: None,
        position_key: None,
        features: vec![0.0],
        label: 0.75,
        label_kind: None,
        label_weight: 1.0,
        base_label_weight: None,
        label_mass: None,
        observation_count: None,
        policy: None,
        pseudo: None,
    }];

    let (trained, report) = train_value_head_cpu(
        &model,
        &samples,
        ValueHeadTrainingConfig {
            learning_rate: 0.5,
            epochs: 16,
            weight_decay: 0.0,
            momentum: 0.0,
        },
    )
    .expect("train output head");

    assert!(report.final_loss < report.initial_loss);
    assert!(trained.predict_value(&[0.0]) > model.predict_value(&[0.0]));
    assert!(decode_compact_value_model(&trained.encode()).is_ok());
}

#[test]
fn trains_value_head_from_precomputed_features() {
    let bytes = encode_test_model(TestModel {
        version: 4,
        projection_size: 2,
        projection_seed: 123,
        hidden_layers: vec![],
        hidden_weights: vec![],
        output_weights: vec![0.0, 0.0, 0.0],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    });
    let model = decode_compact_value_model(&bytes).expect("decode trainable model");
    let samples = vec![TrainingSample {
        side_to_move: None,
        board_count: None,
        position_key: None,
        features: vec![1.0, 2.0],
        label: 0.75,
        label_kind: None,
        label_weight: 1.0,
        base_label_weight: None,
        label_mass: None,
        observation_count: None,
        policy: None,
        pseudo: None,
    }];
    let projected = project_features(
        &samples[0].features,
        model.projection_size as usize,
        model.projection_seed,
    );
    let features = vec![hidden_features_from_projected(projected, &model)];

    let (trained, report) = train_value_head_from_features_cpu(
        &model,
        &samples,
        &features,
        ValueHeadTrainingConfig {
            learning_rate: 0.5,
            epochs: 16,
            weight_decay: 0.0,
            momentum: 0.0,
        },
    )
    .expect("train output head from precomputed features");

    assert!(report.final_loss < report.initial_loss);
    assert!(
        trained.predict_value(&samples[0].features) > model.predict_value(&samples[0].features)
    );
}

#[test]
fn compact_model_hidden_features_match_cpu_head_forward_path() {
    let bytes = encode_test_model(TestModel {
        version: 4,
        projection_size: 2,
        projection_seed: 123,
        hidden_layers: vec![2],
        hidden_weights: vec![1.0, 0.0, 0.25, 0.0, 1.0, -0.5],
        output_weights: vec![0.0, 0.0, 0.0],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    });
    let model = decode_compact_value_model(&bytes).expect("decode trainable model");
    let samples = vec![TrainingSample {
        side_to_move: None,
        board_count: None,
        position_key: None,
        features: vec![1.0, 2.0],
        label: 0.75,
        label_kind: None,
        label_weight: 1.0,
        base_label_weight: None,
        label_mass: None,
        observation_count: None,
        policy: None,
        pseudo: None,
    }];
    let projected = project_features(
        &samples[0].features,
        model.projection_size as usize,
        model.projection_seed,
    );
    let expected = hidden_features_from_projected(projected, &model);
    let json = compact_value_model_hidden_features_json(
        &bytes,
        &serde_json::to_string(&samples).expect("samples JSON"),
    )
    .expect("hidden features JSON");
    let rows = serde_json::from_str::<Vec<Vec<f32>>>(&json).expect("hidden features parse");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0], expected);
}

#[test]
fn trains_policy_head_on_cpu_and_encodes_updated_model() {
    let bytes = encode_test_model(TestModel {
        version: 4,
        projection_size: 1,
        projection_seed: 123,
        hidden_layers: vec![],
        hidden_weights: vec![],
        output_weights: vec![0.0, 0.0],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    });
    let model = decode_compact_value_model(&bytes).expect("decode trainable model");
    let samples = vec![TrainingSample {
        side_to_move: Some("white".to_string()),
        board_count: Some(1),
        position_key: Some("policy-test".to_string()),
        features: vec![0.0],
        label: 0.0,
        label_kind: Some("search".to_string()),
        label_weight: 1.0,
        base_label_weight: None,
        label_mass: None,
        observation_count: None,
        policy: Some(7),
        pseudo: Some(false),
    }];

    let (trained, report) = train_policy_head_cpu(
        &model,
        &samples,
        ValueHeadTrainingConfig {
            learning_rate: 0.5,
            epochs: 128,
            weight_decay: 0.0,
            momentum: 0.0,
        },
    )
    .expect("train policy head");

    assert_eq!(report.samples, 1);
    assert!(report.final_loss < report.initial_loss);
    assert_eq!(trained.policy_weights.len(), 257 * 2);
    let target_bias = trained.policy_weights[7 * 2 + 1];
    let other_bias = trained.policy_weights[1];
    assert!(target_bias > other_bias);
    assert!(decode_compact_value_model(&trained.encode()).is_ok());
}

#[test]
fn trains_policy_head_from_precomputed_features() {
    let bytes = encode_test_model(TestModel {
        version: 4,
        projection_size: 1,
        projection_seed: 123,
        hidden_layers: vec![],
        hidden_weights: vec![],
        output_weights: vec![0.0, 0.0],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    });
    let model = decode_compact_value_model(&bytes).expect("decode trainable model");
    let samples = vec![TrainingSample {
        side_to_move: Some("white".to_string()),
        board_count: Some(1),
        position_key: Some("policy-precomputed-test".to_string()),
        features: vec![0.0],
        label: 0.0,
        label_kind: Some("search".to_string()),
        label_weight: 1.0,
        base_label_weight: None,
        label_mass: None,
        observation_count: None,
        policy: Some(11),
        pseudo: Some(false),
    }];
    let projected = project_features(
        &samples[0].features,
        model.projection_size as usize,
        model.projection_seed,
    );
    let features = vec![hidden_features_from_projected(projected, &model)];

    let (trained, report) = train_policy_head_from_features_cpu(
        &model,
        &samples,
        &features,
        ValueHeadTrainingConfig {
            learning_rate: 0.5,
            epochs: 128,
            weight_decay: 0.0,
            momentum: 0.0,
        },
    )
    .expect("train policy head from precomputed features");

    assert_eq!(report.samples, 1);
    assert!(report.final_loss < report.initial_loss);
    let target_bias = trained.policy_weights[11 * 2 + 1];
    let other_bias = trained.policy_weights[1];
    assert!(target_bias > other_bias);
}

#[test]
fn creates_training_samples_from_engine_positions_and_search_labels() {
    let sample = sample_from_snapshot_label(None, 2.0, 0.5).expect("encode default position");

    assert_eq!(sample.features.len(), 16 * 32 * 64);
    assert_eq!(sample.side_to_move.as_deref(), Some("white"));
    assert_eq!(sample.board_count, Some(1));
    assert!(sample
        .position_key
        .as_deref()
        .is_some_and(|key| key.len() == 16));
    assert_eq!(sample.label, 1.0);
    assert_eq!(sample.label_weight, 0.5);
    assert_eq!(sample.label_kind, None);
    assert_eq!(sample.policy, None);
    assert_eq!(sample.pseudo, None);

    let encoded = serde_json::to_string(&vec![sample.clone()]).expect("encode samples");
    let decoded: Vec<TrainingSample> = serde_json::from_str(&encoded).expect("decode samples");
    assert_eq!(decoded, vec![sample]);

    let response = search_label_sample(SearchLabelSampleRequest {
        nodes: 64,
        time_ms: 1_000,
        ..SearchLabelSampleRequest::default()
    })
    .expect("search label sample");

    assert_eq!(response.samples.len(), 1);
    assert_eq!(response.source, "heuristic-search");
    assert_eq!(response.samples[0].features.len(), 16 * 32 * 64);
    assert_eq!(response.samples[0].side_to_move.as_deref(), Some("white"));
    assert_eq!(response.samples[0].board_count, Some(1));
    assert_eq!(response.samples[0].label_kind.as_deref(), Some("search"));
    assert_eq!(response.samples[0].pseudo, Some(false));
    assert!(response.samples[0]
        .policy
        .is_some_and(|policy| policy < 257));
    assert!(response.samples[0].label.is_finite());
    assert!((-1.0..=1.0).contains(&response.samples[0].label));
    assert!(response.depth >= 1);
    assert!(response.nodes > 0);
}

#[test]
fn collects_search_labeled_training_sample_batches_from_engine_playouts() {
    let response = collect_search_label_samples(SearchLabelBatchRequest {
        count: 3,
        max_plies: 2,
        position_nodes: 64,
        position_time_ms: 1_000,
        label_nodes: 64,
        label_time_ms: 1_000,
        ..SearchLabelBatchRequest::default()
    })
    .expect("collect search label samples");

    assert_eq!(response.requested, 3);
    assert_eq!(response.generated_positions, 3);
    assert_eq!(response.labeled_positions, response.samples.len());
    assert!(!response.samples.is_empty());
    for sample in &response.samples {
        assert_eq!(sample.features.len(), 16 * 32 * 64);
        assert!(sample
            .position_key
            .as_deref()
            .is_some_and(|key| key.len() == 16));
        assert_eq!(sample.label_kind.as_deref(), Some("search"));
        assert_eq!(sample.pseudo, Some(false));
        assert!(sample.policy.is_some_and(|policy| policy < 257));
    }

    let cpu = collect_search_label_samples(SearchLabelBatchRequest {
        mode: SearchLabelMode::Cpu,
        count: 2,
        max_plies: 1,
        position_nodes: 32,
        position_time_ms: 1_000,
        label_nodes: 32,
        label_time_ms: 1_000,
        ..SearchLabelBatchRequest::default()
    })
    .expect("collect cpu label samples");
    assert_eq!(cpu.source, "heuristic-cpu-batch");
    assert_eq!(cpu.requested, 2);
    assert_eq!(cpu.generated_positions, 2);
    assert!(!cpu.samples.is_empty());
    assert!(cpu
        .samples
        .iter()
        .all(|sample| sample.label_kind.as_deref() == Some("cpu")));
    assert!(cpu
        .samples
        .iter()
        .all(|sample| sample.policy.is_some_and(|policy| policy < 257)));

    let distill_model = decode_compact_value_model(&encode_test_model(TestModel {
        version: 1,
        projection_size: 1,
        projection_seed: 123,
        hidden_layers: vec![],
        hidden_weights: vec![],
        output_weights: vec![0.1, 0.0],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    }))
    .expect("decode distillation sample model");
    let distilled = collect_search_label_samples(SearchLabelBatchRequest {
        mode: SearchLabelMode::Distilled,
        distill_model: Some(distill_model),
        count: 2,
        max_plies: 1,
        position_nodes: 32,
        position_time_ms: 1_000,
        ..SearchLabelBatchRequest::default()
    })
    .expect("collect distilled label samples");
    assert_eq!(distilled.source, "heuristic-distilled-batch");
    assert_eq!(distilled.requested, 2);
    assert_eq!(distilled.generated_positions, 2);
    assert!(!distilled.samples.is_empty());
    assert!(distilled.samples.iter().all(|sample| {
        sample.label_kind.as_deref() == Some("distilled")
            && sample.policy.is_none()
            && sample.pseudo == Some(true)
            && sample.label_weight == 0.25
    }));

    let curriculum = collect_search_label_samples(SearchLabelBatchRequest {
        mode: SearchLabelMode::Curriculum,
        count: 2,
        max_plies: 1,
        position_nodes: 32,
        position_time_ms: 1_000,
        label_nodes: 32,
        label_time_ms: 1_000,
        ..SearchLabelBatchRequest::default()
    })
    .expect("collect curriculum label samples");
    assert_eq!(curriculum.source, "heuristic-curriculum-batch");
    assert_eq!(curriculum.requested, 2);
    assert_eq!(curriculum.generated_positions, 2);
    assert!(curriculum
        .samples
        .iter()
        .all(|sample| sample.label_kind.as_deref() == Some("curriculum")));
    assert!(curriculum
        .samples
        .iter()
        .all(|sample| sample.label_weight >= 1.05));

    let tactical = collect_search_label_samples(SearchLabelBatchRequest {
        mode: SearchLabelMode::Tactical,
        count: 2,
        max_plies: 1,
        position_nodes: 32,
        position_time_ms: 1_000,
        label_nodes: 32,
        label_time_ms: 1_000,
        ..SearchLabelBatchRequest::default()
    })
    .expect("collect tactical label samples");
    assert_eq!(tactical.source, "heuristic-tactical-batch");
    assert_eq!(tactical.requested, 2);
    assert_eq!(tactical.generated_positions, 2);
    assert!(tactical
        .samples
        .iter()
        .all(|sample| sample.label_kind.as_deref() == Some("tactical")));
    assert!(tactical
        .samples
        .iter()
        .all(|sample| sample.label_weight >= 1.6));

    let outcome = collect_search_label_samples(SearchLabelBatchRequest {
        mode: SearchLabelMode::Outcome,
        count: 2,
        max_plies: 2,
        label_nodes: 32,
        label_time_ms: 1_000,
        ..SearchLabelBatchRequest::default()
    })
    .expect("collect outcome label samples");
    assert_eq!(outcome.source, "heuristic-outcome-batch");
    assert_eq!(outcome.requested, 2);
    assert_eq!(outcome.generated_positions, 2);
    assert!(!outcome.samples.is_empty());
    assert!(outcome.samples.iter().all(|sample| {
        matches!(
            sample.label_kind.as_deref(),
            Some("outcome") | Some("search-bootstrap")
        )
    }));

    let duel = collect_search_label_samples(SearchLabelBatchRequest {
        mode: SearchLabelMode::Duel,
        count: 2,
        max_plies: 2,
        position_nodes: 32,
        position_time_ms: 1_000,
        label_nodes: 32,
        label_time_ms: 1_000,
        ..SearchLabelBatchRequest::default()
    })
    .expect("collect duel label samples");
    assert_eq!(duel.source, "heuristic-duel-batch");
    assert_eq!(duel.requested, 2);
    assert_eq!(duel.generated_positions, 2);
    assert!(!duel.samples.is_empty());
    assert!(duel.samples.iter().all(|sample| {
        matches!(
            sample.label_kind.as_deref(),
            Some("duel") | Some("duel-search")
        )
    }));
}

#[test]
fn training_sample_split_groups_batches_and_hashes_match_browser_policy() {
    let samples = vec![
        training_sample("same", "search", 1.0, vec![1.0, 0.0, 2.0]),
        training_sample("same", "outcome", 10.0, vec![3.0, 0.0, 4.0]),
        training_sample("other", "search", 1.0, vec![5.0, 0.0, 6.0]),
    ];

    let groups = group_training_indices_by_position(&samples, &[0, 1, 2]);
    let mut first_batch = vec![0; 32];
    let mut second_batch = vec![0; 32];
    let uniform_weight =
        fill_grouped_training_batch_indices(&mut first_batch, &groups, 1, 1234, &[1.0, 1.0, 1.0])
            .expect("uniform batch");
    let skewed_weight =
        fill_grouped_training_batch_indices(&mut second_batch, &groups, 1, 1234, &[1.0, 10.0, 1.0])
            .expect("skewed batch");

    assert_eq!(stable_sample_hash(&samples[0], 0), 2_303_567_373);
    assert_eq!(xorshift32(1234), 332_584_831);
    assert_eq!(shuffled_indices(&[0, 1, 2, 3], 1, 1234), vec![0, 3, 1, 2]);
    let shuffled_bytes =
        shuffled_indices_bytes(&[0, 1, 2, 3], 1, 1234).expect("shuffled indices bytes");
    assert_eq!(
        shuffled_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>(),
        vec![0, 3, 1, 2]
    );
    assert_eq!(groups, vec![vec![0, 1], vec![2]]);
    assert_eq!(second_batch, first_batch);
    assert_eq!(uniform_weight, first_batch.len() as f32);
    assert!(skewed_weight > uniform_weight);
    assert!(first_batch.iter().any(|index| *index == 0 || *index == 1));
    assert!(first_batch.iter().any(|index| *index == 2));
    let mut request = Vec::new();
    for value in [32_u32, 2, 3, 3, 1, 1234, 0, 2, 3, 0, 1, 2] {
        request.extend_from_slice(&value.to_le_bytes());
    }
    for value in [1.0_f32, 1.0, 1.0] {
        request.extend_from_slice(&value.to_le_bytes());
    }
    let response =
        fill_grouped_training_batch_indices_bytes(&request).expect("fill grouped batch bytes");
    let batch_weight = f32::from_le_bytes(response[0..4].try_into().unwrap());
    let batch_len = u32::from_le_bytes(response[4..8].try_into().unwrap());
    let batch = response[8..]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(batch_weight, uniform_weight);
    assert_eq!(batch_len, 32);
    assert_eq!(batch, first_batch);
    assert_eq!(unique_training_position_count(&samples, &[0, 1, 2]), 2);
    assert_eq!(unique_training_position_count(&samples, &[0, 1]), 1);
    assert_eq!(feature_length(&samples), Ok(3));
}

#[test]
fn working_set_indices_keep_high_signal_samples_and_policy_targets() {
    let mut samples = (0..6)
        .map(|index| {
            training_sample_with(
                &format!("outcome-{index}"),
                "outcome",
                10.0 - index as f32,
                0.0,
                None,
                vec![1.0],
            )
        })
        .collect::<Vec<_>>();
    samples.push(training_sample_with(
        "policy-search",
        "search",
        1.0,
        0.0,
        Some(42),
        vec![1.0],
    ));
    samples.push(training_sample_with(
        "distilled",
        "distilled",
        1.0,
        0.0,
        None,
        vec![1.0],
    ));
    samples.last_mut().unwrap().pseudo = Some(true);

    let indexes = select_training_working_set_indices_for_projection(
        &samples,
        2_048,
        4 * 2_048 * std::mem::size_of::<f32>(),
    );
    let selected = indexes
        .iter()
        .map(|index| samples[*index].position_key.as_deref().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(indexes.len(), 4);
    assert_eq!(
        selected,
        vec!["outcome-0", "outcome-1", "outcome-2", "policy-search"]
    );
    assert_eq!(
        indexes
            .iter()
            .filter(|index| has_policy_training_target(&samples[**index]))
            .count(),
        1
    );
}

#[test]
fn fallback_validation_split_keeps_position_labels_together() {
    let samples = vec![
        training_sample("same", "search", 1.0, vec![1.0]),
        training_sample("same", "outcome", 1.0, vec![2.0]),
        training_sample("other", "search", 1.0, vec![3.0]),
    ];

    let split = split_validation_samples(&samples, 0.000001);
    let train_keys = split
        .train_indices
        .iter()
        .map(|index| samples[*index].position_key.as_deref())
        .collect::<Vec<_>>();
    let validation_keys = split
        .validation_indices
        .iter()
        .map(|index| samples[*index].position_key.as_deref())
        .collect::<Vec<_>>();

    assert!(!split.validation_indices.is_empty());
    assert!(train_keys.iter().all(|key| !validation_keys.contains(key)));
}

#[test]
fn policy_holdout_reuses_position_split_and_falls_back_for_sparse_labels() {
    let samples = ["a", "b", "c", "d", "e"]
        .into_iter()
        .map(|key| training_sample(key, "search", 1.0, vec![1.0]))
        .collect::<Vec<_>>();
    let split = chronofish_engine::gpu::training::ValidationSplit {
        train_indices: vec![0, 1, 2, 3],
        validation_indices: vec![4],
        seed: 123,
    };

    let direct = split_policy_training_indices(&samples, &[1, 3, 4], &split, 0.2);
    assert_eq!(direct.train_indices, vec![1, 3]);
    assert_eq!(direct.validation_indices, vec![4]);
    assert_eq!(direct.seed, 123);

    let fallback = split_policy_training_indices(&samples, &[1, 3], &split, 0.2);
    assert_eq!(fallback.train_indices.len(), 1);
    assert_eq!(fallback.validation_indices.len(), 1);
    assert_ne!(fallback.train_indices[0], fallback.validation_indices[0]);
}

#[test]
fn policy_target_detection_and_training_step_bounds_match_browser_policy() {
    let policy = training_sample_with("policy", "search", 1.0, 0.0, Some(7), vec![1.0]);
    let distilled = training_sample_with("distilled", "distilled", 1.0, 0.0, Some(7), vec![1.0]);
    let missing = training_sample_with("missing", "search", 1.0, 0.0, None, vec![1.0]);

    assert!(has_policy_training_target(&policy));
    assert!(!has_policy_training_target(&distilled));
    assert!(!has_policy_training_target(&missing));
    let zero_weight = training_sample_with("zero", "search", 0.0, 0.0, Some(5), vec![1.0]);
    let samples = vec![policy, distilled, missing, zero_weight];
    assert_eq!(policy_training_indices(&samples, false), vec![0, 3]);
    assert_eq!(policy_training_indices(&samples, true), vec![0]);
    assert_eq!(policy_training_target(0), 0);
    assert_eq!(policy_training_target(7), 7);
    assert_eq!(policy_training_target(999), 256);
    assert_eq!(training_label_weight(-0.5), 0.0);
    assert_eq!(training_label_weight(0.0), 0.0);
    assert_eq!(training_label_weight(1.5), 1.5);
    assert_eq!(training_weighted_average(10.0, 2.0), 5.0);
    assert_eq!(training_weighted_average(10.0, 0.0), 0.0);
    assert_eq!(training_batch_normalization(2.0), 0.5);
    assert_eq!(training_batch_normalization(0.0), 1_000_000.0);
    assert_eq!(policy_training_steps(0), 16);
    assert_eq!(policy_training_steps(128), 16);
    assert_eq!(policy_training_steps(2048), 32);
    assert_eq!(policy_training_steps(1_000_000), 256);
    assert_eq!(value_training_batch_size(64, 0), 1);
    assert_eq!(value_training_batch_size(64, 12), 12);
    assert_eq!(value_training_batch_size(64, 128), 64);
    assert_eq!(policy_training_batch_size(64, 0), 0);
    assert_eq!(policy_training_batch_size(64, 12), 12);
    assert_eq!(policy_training_batch_size(64, 128), 64);
    assert_eq!(value_head_validation_interval(0, None), 1);
    assert_eq!(value_head_validation_interval(100, Some(16)), 16);
    assert_eq!(value_head_validation_interval(10, Some(256)), 10);
    assert_eq!(value_gpu_batches_per_submit(0), 1);
    assert_eq!(value_gpu_batches_per_submit(12), 12);
    assert_eq!(value_gpu_batches_per_submit(128), 64);
    assert_eq!(value_gpu_validation_interval(12, None), 256);
    assert_eq!(value_gpu_validation_interval(64, Some(16)), 64);
    assert_eq!(value_gpu_validation_interval(64, Some(128)), 128);
    assert_eq!(policy_training_steps_per_submit(0), 0);
    assert_eq!(policy_training_steps_per_submit(12), 12);
    assert_eq!(policy_training_steps_per_submit(128), 64);
}

#[test]
fn sparse_projection_packs_dense_samples_into_stable_rows() {
    let samples = vec![
        training_sample("first", "search", 1.0, vec![0.0, 2.0, 0.0, -1.0]),
        training_sample("empty", "search", 1.0, vec![0.0, 0.0, 0.0, 0.0]),
        training_sample("last", "search", 1.0, vec![3.0, 0.0, 4.0, 0.0]),
    ];

    let packed = pack_sparse_projection_features(&samples, None).expect("pack sparse features");

    assert_eq!(packed.offsets, vec![0, 2, 2, 4]);
    assert_eq!(packed.indices, vec![1, 3, 0, 2]);
    assert_eq!(packed.values, vec![2.0, -1.0, 3.0, 4.0]);
    assert_eq!(packed.byte_length, 16 + 16 + 16);

    let bytes = sparse_projection_features_bytes(&samples, None).expect("pack sparse bytes");
    let words = bytes[..16]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(words, vec![4, 4, 4, 48]);
    assert_eq!(
        bytes[16..32]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>(),
        vec![0, 2, 2, 4]
    );
}

#[test]
fn sparse_projection_allocates_valid_empty_storage_buffers() {
    let samples = vec![training_sample("empty", "search", 1.0, vec![0.0, 0.0, 0.0])];

    let packed =
        pack_sparse_projection_features(&samples, None).expect("pack empty sparse features");

    assert_eq!(packed.offsets, vec![0, 0]);
    assert_eq!(packed.indices.len(), 1);
    assert_eq!(packed.values.len(), 1);
    assert_eq!(packed.byte_length, 8 + 4 + 4);
}

#[test]
fn native_training_rejects_inconsistent_sample_feature_lengths() {
    let samples = vec![
        training_sample("a", "search", 1.0, vec![1.0, 2.0]),
        training_sample("b", "search", 1.0, vec![1.0]),
    ];

    assert_eq!(
        feature_length(&samples),
        Err("Training samples have inconsistent feature lengths.".to_string())
    );
}

#[test]
fn replay_dedupe_averages_labels_and_keeps_strongest_policy_target() {
    let retained = dedupe_training_samples(&[
        training_sample_with("same-position", "search", 1.0, -1.0, Some(1), vec![1.0]),
        training_sample_with("same-position", "search", 5.0, 1.0, Some(2), vec![2.0]),
        training_sample_with("same-position", "outcome", 1.0, 0.25, Some(3), vec![3.0]),
    ]);

    assert_eq!(retained.len(), 2);
    assert_eq!(retained[0].label_kind.as_deref(), Some("search"));
    assert_eq!(retained[0].policy, Some(2));
    assert!((retained[0].label - (4.0 / 6.0)).abs() < 1e-6);
    assert!((retained[0].label_weight - (5.0 * 2.0_f32.sqrt())).abs() < 1e-6);
    assert_eq!(retained[0].base_label_weight, Some(5.0));
    assert_eq!(retained[0].label_mass, Some(6.0));
    assert_eq!(retained[0].observation_count, Some(2));
    assert_eq!(retained[1].label_kind.as_deref(), Some("outcome"));
}

#[test]
fn replay_confidence_is_bounded_across_repeated_observations() {
    let repeated = (0..100)
        .map(|index| {
            training_sample_with(
                "repeated",
                "search",
                2.0,
                if index % 2 == 0 { -1.0 } else { 1.0 },
                Some(7),
                vec![1.0],
            )
        })
        .collect::<Vec<_>>();

    let retained = dedupe_training_samples(&repeated);

    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].observation_count, Some(64));
    assert_eq!(retained[0].label_mass, Some(64.0));
    assert_eq!(retained[0].label_weight, 4.0);
    assert!(retained[0].label.abs() < 0.02);
}

#[test]
fn replay_dedupe_fingerprints_legacy_samples_without_position_keys() {
    let mut first = training_sample_with("", "search", 1.0, 0.0, Some(1), vec![0.0, 2.0, 0.0]);
    let mut second = training_sample_with("", "search", 4.0, 1.0, Some(2), vec![0.0, 2.0, 0.0]);
    let mut third = training_sample_with("", "outcome", 1.0, 0.25, Some(3), vec![0.0, 2.0, 0.0]);
    let mut fourth = training_sample_with("", "search", 1.0, 0.0, Some(4), vec![0.0, 0.0, 0.0]);
    let mut fifth = training_sample_with("", "search", 2.0, 1.0, Some(5), vec![0.0, 0.0, 0.0]);
    for sample in [&mut first, &mut second, &mut third, &mut fourth, &mut fifth] {
        sample.position_key = None;
    }

    let retained = dedupe_training_samples(&[first, second, third, fourth, fifth]);

    assert_eq!(retained.len(), 4);
    assert_eq!(retained[0].label_kind.as_deref(), Some("search"));
    assert_eq!(retained[0].policy, Some(2));
    assert!((retained[0].label_weight - (4.0 * 2.0_f32.sqrt())).abs() < 1e-6);
    let retained_summary = retained
        .iter()
        .skip(1)
        .map(|sample| {
            (
                sample.label_kind.as_deref(),
                sample.policy,
                sample.label_weight,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        retained_summary,
        vec![
            (Some("outcome"), Some(3), 1.0),
            (Some("search"), Some(4), 1.0),
            (Some("search"), Some(5), 2.0),
        ]
    );
}

#[test]
fn replay_retention_keeps_high_signal_samples_and_policy_supervision() {
    let retained = append_replay_samples(
        &[],
        &[
            training_sample_with("distilled", "distilled", 1.0, 0.0, None, vec![1.0]),
            training_sample_with("search", "search", 1.0, 0.0, None, vec![1.0]),
            training_sample_with("outcome", "outcome", 1.0, 0.0, None, vec![1.0]),
            training_sample_with("cpu", "cpu", 2.0, 0.0, None, vec![1.0]),
        ],
        2,
    );

    assert_eq!(
        retained
            .iter()
            .map(|sample| sample.position_key.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("outcome"), Some("cpu")]
    );

    let policy_retained = append_replay_samples(
        &[],
        &[
            training_sample_with("outcome-0", "outcome", 10.0, 0.0, None, vec![1.0]),
            training_sample_with("outcome-1", "outcome", 9.0, 0.0, None, vec![1.0]),
            training_sample_with("outcome-2", "outcome", 8.0, 0.0, None, vec![1.0]),
            training_sample_with("outcome-3", "outcome", 7.0, 0.0, None, vec![1.0]),
            training_sample_with("outcome-4", "outcome", 6.0, 0.0, None, vec![1.0]),
            training_sample_with("outcome-5", "outcome", 5.0, 0.0, None, vec![1.0]),
            training_sample_with("policy-search", "search", 1.0, 0.0, Some(13), vec![1.0]),
        ],
        4,
    );

    assert_eq!(policy_retained.len(), 4);
    assert!(policy_retained
        .iter()
        .any(|sample| sample.position_key.as_deref() == Some("policy-search")));
    assert_eq!(
        replay_sample_priority(&policy_retained[0], 0, policy_retained.len()).is_finite(),
        true
    );
}

#[test]
fn label_source_counts_match_browser_metrics_policy() {
    let mut pseudo = training_sample_with("pseudo", "distilled", 1.0, 0.0, None, vec![1.0]);
    pseudo.label_kind = None;
    pseudo.pseudo = Some(true);
    let mut unknown = training_sample_with("unknown", "search", 1.0, 0.0, None, vec![1.0]);
    unknown.label_kind = None;

    let counts = label_source_counts(&[
        training_sample_with("search-a", "search", 1.0, 0.0, None, vec![1.0]),
        training_sample_with("search-b", "search", 1.0, 0.0, None, vec![1.0]),
        training_sample_with("outcome", "outcome", 1.0, 0.0, None, vec![1.0]),
        pseudo,
        unknown,
    ]);

    assert_eq!(counts.get("search"), Some(&2));
    assert_eq!(counts.get("outcome"), Some(&1));
    assert_eq!(counts.get("distilled"), Some(&1));
    assert_eq!(counts.get("unknown"), Some(&1));
    assert_eq!(counts.len(), 4);
}

#[test]
fn training_label_priority_matches_browser_replay_policy() {
    assert_eq!(training_label_priority(Some("outcome"), false), 4.0);
    assert_eq!(training_label_priority(Some("duel"), true), 4.0);
    assert_eq!(training_label_priority(Some("search"), false), 3.0);
    assert_eq!(training_label_priority(Some("cpu"), true), 3.0);
    assert_eq!(training_label_priority(Some("duel-search"), false), 2.0);
    assert_eq!(training_label_priority(Some("search-bootstrap"), true), 2.0);
    assert_eq!(training_label_priority(Some("distilled"), false), 1.0);
    assert_eq!(training_label_priority(Some("distilled"), true), 1.0);
    assert_eq!(training_label_priority(None, false), 2.0);
    assert_eq!(training_label_priority(Some("unknown"), false), 2.0);
    assert_eq!(training_label_priority(None, true), 1.0);
    assert_eq!(training_label_priority(Some("unknown"), true), 1.0);
}

#[test]
fn partial_outcome_samples_match_browser_bootstrap_policy() {
    assert_eq!(DEFAULT_PARTIAL_OUTCOME_LABEL_KIND, "search-bootstrap");
    assert_eq!(DEFAULT_PARTIAL_OUTCOME_LABEL_WEIGHT, 0.5);

    let samples = vec![
        training_sample_with("a", "search", 3.0, 0.25, Some(1), vec![1.0]),
        training_sample_with("b", "outcome", 4.0, -0.5, None, vec![2.0]),
    ];

    let defaulted = samples_from_partial_outcome(&samples, None, None);
    assert_eq!(
        defaulted
            .iter()
            .map(|sample| (
                sample.label_kind.as_deref(),
                sample.label_weight,
                sample.label
            ))
            .collect::<Vec<_>>(),
        vec![
            (Some("search-bootstrap"), 0.5, 0.25),
            (Some("search-bootstrap"), 0.5, -0.5),
        ]
    );
    assert_eq!(defaulted[0].policy, Some(1));
    assert_eq!(defaulted[0].features, vec![1.0]);

    let duel = samples_from_partial_outcome(&samples, Some("duel-search"), Some(1.0));
    assert_eq!(
        duel.iter()
            .map(|sample| (sample.label_kind.as_deref(), sample.label_weight))
            .collect::<Vec<_>>(),
        vec![(Some("duel-search"), 1.0), (Some("duel-search"), 1.0)]
    );
}

#[test]
fn training_working_set_respects_capacity_and_keeps_stronger_labels() {
    let samples = vec![
        training_sample_with("distilled-old", "distilled", 1.0, 0.0, None, vec![1.0]),
        training_sample_with("search-low", "search", 1.0, 0.0, None, vec![1.0]),
        training_sample_with("outcome", "outcome", 1.0, 0.0, None, vec![1.0]),
        training_sample_with("search-high", "search", 10.0, 0.0, None, vec![1.0]),
        training_sample_with("unknown-recent", "unknown", 1.0, 0.0, None, vec![1.0]),
    ];

    let selected = select_training_working_set_with_capacity(&samples, 3);

    assert_eq!(
        selected
            .iter()
            .map(|sample| sample.position_key.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("search-low"), Some("outcome"), Some("search-high")]
    );
}

#[test]
fn training_working_set_reserves_policy_supervision_when_value_labels_dominate() {
    let mut samples = (0..6)
        .map(|index| {
            training_sample_with(
                &format!("outcome-{index}"),
                "outcome",
                10.0 - index as f32,
                0.0,
                None,
                vec![1.0],
            )
        })
        .collect::<Vec<_>>();
    samples.push(training_sample_with(
        "policy-search",
        "search",
        1.0,
        0.0,
        Some(42),
        vec![1.0],
    ));
    samples.push(training_sample_with(
        "distilled",
        "distilled",
        1.0,
        0.0,
        None,
        vec![1.0],
    ));

    let selected = select_training_working_set_with_capacity(&samples, 4);

    assert_eq!(selected.len(), 4);
    assert_eq!(
        selected
            .iter()
            .filter(|sample| sample.policy.is_some())
            .count(),
        1
    );
    assert!(selected
        .iter()
        .any(|sample| sample.position_key.as_deref() == Some("policy-search")));
    assert_eq!(
        selected
            .iter()
            .filter(|sample| sample.label_kind.as_deref() == Some("outcome"))
            .count(),
        3
    );
}

#[test]
fn projected_training_working_set_uses_model_projection_size() {
    assert_eq!(DEFAULT_PROJECTED_WORKING_SET_BYTES, 128 * 1024 * 1024);

    let samples = vec![
        training_sample_with("old", "distilled", 1.0, 0.0, None, vec![1.0]),
        training_sample_with("search-low", "search", 1.0, 0.0, None, vec![1.0]),
        training_sample_with("outcome", "outcome", 1.0, 0.0, None, vec![1.0]),
        training_sample_with("search-high", "search", 10.0, 0.0, None, vec![1.0]),
    ];

    let selected = select_training_working_set_for_projection(&samples, 8, 8 * 2 * 4);

    assert_eq!(
        selected
            .iter()
            .map(|sample| sample.position_key.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("outcome"), Some("search-high")]
    );
    assert_eq!(
        select_training_working_set_for_projection(&samples, 0, 0).len(),
        samples.len()
    );
}

#[test]
fn rejects_invalid_compact_value_model_data() {
    assert!(matches!(
        decode_compact_value_model(b"NOPE"),
        Err(CompactValueModelError::InvalidMagic(_))
    ));

    let mut bytes = encode_test_model(TestModel {
        version: 4,
        projection_size: 8,
        projection_seed: 123,
        hidden_layers: vec![4],
        hidden_weights: vec![f32::NAN],
        output_weights: vec![1.0],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    });
    assert!(matches!(
        decode_compact_value_model(&bytes),
        Err(CompactValueModelError::NonFinite {
            section: "hidden_weights",
            index: 0,
            ..
        })
    ));

    bytes = encode_test_model(TestModel {
        version: 4,
        projection_size: 8,
        projection_seed: 123,
        hidden_layers: vec![4],
        hidden_weights: vec![0.0],
        output_weights: vec![1.0],
        policy_values: vec![],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    });
    bytes.push(0);
    assert!(matches!(
        decode_compact_value_model(&bytes),
        Err(CompactValueModelError::TrailingBytes { .. })
    ));
}

#[test]
fn policy_head_shape_helpers_match_browser_model_adapters() {
    assert_eq!(output_layer_size(&[1024, 512, 256]), Ok(256));
    assert_eq!(
        output_layer_size(&[]),
        Err("Model must have at least one hidden layer.".to_string())
    );
    assert_eq!(previous_layer_size(&[1024, 512, 256], 0, 2048), 2048);
    assert_eq!(previous_layer_size(&[1024, 512, 256], 2, 2048), 512);
    assert_eq!(previous_layer_size(&[1024], 99, 2048), 0);

    let mut model = decode_compact_value_model(&encode_test_model(TestModel {
        version: 2,
        projection_size: 2,
        projection_seed: 123,
        hidden_layers: vec![3],
        hidden_weights: vec![],
        output_weights: vec![1.0],
        policy_values: vec![0.25, -0.5, 0.75],
        auxiliary_value_weights: vec![],
        scale: 1.0,
        bias: 0.0,
    }))
    .expect("decode policy logits model");
    model.policy_weights.clear();

    assert_eq!(policy_logits_array(None), None);
    assert_eq!(
        policy_logits_array(Some(&model)).expect("logits"),
        vec![0.25, -0.5, 0.75]
    );

    let weights = policy_weights_array(Some(&model), 2).expect("weights from logits");
    assert_eq!(weights.len(), 257 * 3);
    assert_eq!(weights[2], 0.25);
    assert_eq!(weights[5], -0.5);
    assert_eq!(weights[8], 0.75);
    assert_eq!(weights[0], 0.0);

    let encoded = encode_compact_value_model(&model);
    let weight_bytes = compact_value_model_policy_weights_bytes(&encoded, 2)
        .expect("decode policy weights")
        .expect("weights from logits");
    assert_eq!(
        weight_bytes.len(),
        weights.len() * std::mem::size_of::<f32>()
    );
    assert_eq!(
        f32::from_le_bytes(weight_bytes[8..12].try_into().unwrap()),
        0.25
    );
    assert_eq!(
        f32::from_le_bytes(weight_bytes[20..24].try_into().unwrap()),
        -0.5
    );

    let explicit_weights = (0..(257 * 2)).map(|value| value as f32).collect::<Vec<_>>();
    model.policy_weights = explicit_weights.clone();
    assert_eq!(
        policy_weights_array(Some(&model), 1),
        Some(explicit_weights)
    );
}

#[test]
fn neural_frontier_upload_transforms_match_browser_policy() {
    let weights = [-1.0_f32, -0.25, 0.0, 0.5, 1.0];
    let mut request = Vec::new();
    for value in weights {
        request.extend_from_slice(&value.to_le_bytes());
    }

    let quantized = quantized_policy_upload_bytes(&request).expect("quantized policy upload");
    assert_eq!(
        quantized.len(),
        8 + weights.len() * std::mem::size_of::<f32>()
    );
    let scale = f32::from_le_bytes(quantized[0..4].try_into().unwrap());
    let max_abs_error = f32::from_le_bytes(quantized[4..8].try_into().unwrap());
    let dequantized = quantized[8..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!((scale - (1.0 / 127.0)).abs() < 1e-8);
    assert!(max_abs_error <= scale / 2.0);
    assert_eq!(dequantized[0], -1.0);
    assert_eq!(dequantized[2], 0.0);
    assert_eq!(dequantized[4], 1.0);

    let halves = f32_to_f16_upload_bytes(&request).expect("f16 upload");
    let half_values = halves
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(half_values, vec![0xbc00, 0xb400, 0x0000, 0x3800, 0x3c00]);
    assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
    assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
    assert_eq!(f32_to_f16_bits(f32::NAN), 0x7e00);
}

#[test]
fn compact_model_validation_helpers_match_browser_model_adapters() {
    let mut model = decode_compact_value_model(&encode_test_model(TestModel {
        version: 5,
        projection_size: 2,
        projection_seed: 123,
        hidden_layers: vec![3],
        hidden_weights: vec![0.0, 1.0],
        output_weights: vec![0.25],
        policy_values: vec![0.1, 0.2],
        auxiliary_value_weights: vec![0.3],
        scale: 1.0,
        bias: 0.0,
    }))
    .expect("decode finite model");

    assert!(compact_model_is_finite(&model));
    assert!(f32_values_are_finite(&[0.0, 1.0, -1.0]));
    assert!(!f32_values_are_finite(&[0.0, f32::NAN]));

    model.policy_weights[1] = f32::INFINITY;
    assert!(!compact_model_is_finite(&model));

    assert!(byte_arrays_equal(Some(&[1, 2, 3]), Some(&[1, 2, 3])));
    assert!(!byte_arrays_equal(Some(&[1, 2, 3]), Some(&[1, 2, 4])));
    assert!(!byte_arrays_equal(Some(&[1, 2, 3]), Some(&[1, 2])));
    assert!(!byte_arrays_equal(None, Some(&[1, 2, 3])));
    assert!(!byte_arrays_equal(Some(&[1, 2, 3]), None));
    assert!(!byte_arrays_equal(None, None));
}

#[test]
fn loads_committed_gpu_v1_value_model() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/gpu-v1/value-model.cfnn");
    let model = load_compact_value_model(path).expect("load committed gpu-v1 model");
    let summary = model.summary();

    assert_eq!(summary.version, 4);
    assert_eq!(summary.projection_size, 2048);
    assert_eq!(summary.projection_seed, 2166136261);
    assert_eq!(summary.hidden_layers, vec![1024, 512, 256]);
    assert_eq!(summary.output_weight_count, 257);
    assert_eq!(summary.output_activation, OutputActivation::Tanh);
    assert!(summary.hidden_weight_count > 2_000_000);
    assert!(model.predict_value(&[]).is_finite());
    assert!(model_architecture_matches(&model));
    assert_eq!(
        default_initial_hidden_weights().len(),
        summary.hidden_weight_count
    );

    let mut wrong_projection = model.clone();
    wrong_projection.projection_seed = 1;
    assert!(!model_architecture_matches(&wrong_projection));
}

fn training_sample(
    position_key: &str,
    label_kind: &str,
    label_weight: f32,
    features: Vec<f32>,
) -> TrainingSample {
    training_sample_with(
        position_key,
        label_kind,
        label_weight,
        0.25,
        Some(7),
        features,
    )
}

fn training_sample_with(
    position_key: &str,
    label_kind: &str,
    label_weight: f32,
    label: f32,
    policy: Option<u32>,
    features: Vec<f32>,
) -> TrainingSample {
    TrainingSample {
        side_to_move: Some("white".to_string()),
        board_count: Some(1),
        position_key: Some(position_key.to_string()),
        features,
        label,
        label_kind: Some(label_kind.to_string()),
        label_weight,
        base_label_weight: None,
        label_mass: None,
        observation_count: None,
        policy,
        pseudo: Some(false),
    }
}

struct TestModel {
    version: u32,
    projection_size: u32,
    projection_seed: u32,
    hidden_layers: Vec<u32>,
    hidden_weights: Vec<f32>,
    output_weights: Vec<f32>,
    policy_values: Vec<f32>,
    auxiliary_value_weights: Vec<f32>,
    scale: f32,
    bias: f32,
}

fn encode_test_model(model: TestModel) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"CFNN");
    push_u32(&mut bytes, model.version);
    push_u32(&mut bytes, model.projection_size);
    push_u32(&mut bytes, model.projection_seed);
    push_u32(&mut bytes, model.hidden_layers.len() as u32);
    push_u32(&mut bytes, model.output_weights.len() as u32);
    if model.version >= 2 {
        push_u32(&mut bytes, model.policy_values.len() as u32);
    }
    if model.version >= 5 {
        push_u32(&mut bytes, model.auxiliary_value_weights.len() as u32);
    }
    push_f32(&mut bytes, model.scale);
    push_f32(&mut bytes, model.bias);
    for layer in model.hidden_layers {
        push_u32(&mut bytes, layer);
    }
    push_u32(&mut bytes, model.hidden_weights.len() as u32);
    for value in model
        .hidden_weights
        .into_iter()
        .chain(model.output_weights)
        .chain(model.policy_values)
        .chain(model.auxiliary_value_weights)
    {
        push_f32(&mut bytes, value);
    }
    bytes
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn training_snapshot_json(timelines: &[serde_json::Value]) -> String {
    serde_json::json!({
        "turn": "white",
        "nextTimelineId": 1,
        "nextBlackTimelineId": -1,
        "timelines": timelines,
    })
    .to_string()
}

fn timeline_json(
    id: i32,
    row: i32,
    owner: &str,
    boards: &[serde_json::Value],
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "row": row,
        "label": format!("T{id}"),
        "owner": owner,
        "boards": boards,
    })
}

fn board_json(time: i32, pieces: &[serde_json::Value]) -> serde_json::Value {
    let mut board = vec![vec![serde_json::Value::Null; 8]; 8];
    for piece in pieces {
        let x = piece["x"].as_u64().expect("piece x") as usize;
        let y = piece["y"].as_u64().expect("piece y") as usize;
        board[y][x] = serde_json::json!({
            "color": piece["color"],
            "type": piece["type"],
        });
    }
    serde_json::json!({
        "time": time,
        "sideToMove": "white",
        "castling": 0,
        "enPassant": null,
        "origin": null,
        "board": board,
    })
}

fn piece_json(x: usize, y: usize, color: &str, piece_type: &str) -> serde_json::Value {
    serde_json::json!({
        "x": x,
        "y": y,
        "color": color,
        "type": piece_type,
    })
}
