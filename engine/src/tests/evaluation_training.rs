use super::*;
use crate::cpu::{training::*, *};

fn trainer_test_config() -> TrainerConfig {
    TrainerConfig {
        generations: 1,
        population: 4,
        training_time_ms: 10,
        nodes: 5,
        seed: 7,
        max_seconds: Some(1),
        out: None,
        score: None,
        score_default: false,
        gpu_backend_info: false,
        gpu_compile_shaders: false,
        gpu_compile_kernels: false,
        gpu_dispatch_smoke: false,
        gpu_training_dispatch_smoke: false,
        gpu_shader_info: false,
        gpu_value_model_path: crate::gpu::training::DEFAULT_VALUE_MODEL_PATH.to_string(),
        gpu_model_info: None,
        gpu_model_probe_zero: None,
        gpu_project_samples: None,
        gpu_predict_samples: None,
        gpu_distill_samples: None,
        gpu_replay_buffer: None,
        gpu_replay_append: None,
        gpu_replay_max: crate::gpu::training::MAX_GPU_TRAINING_SAMPLES,
        gpu_search_snapshot: None,
        gpu_search_depth: None,
        gpu_search_min_depth: None,
        gpu_sample_search_snapshot: None,
        gpu_sample_mode: crate::gpu::training::SearchLabelMode::Search,
        gpu_sample_count: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_COUNT,
        gpu_sample_max_plies: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_MAX_PLIES,
        gpu_train_search_snapshot: None,
        gpu_train_samples: None,
        gpu_train_projected_samples: None,
        cpu_search_snapshot: None,
        train_cycle: false,
        training_strategy: CpuTrainingStrategy::Sweep,
        compare_seeds: vec![101, 202],
        min_wins: 1,
        min_total_delta: 1,
        verify: "cargo test -q".to_string(),
        ai_src: "engine/models/cpu-v1/parameters.json".to_string(),
        hall_of_fame: default_hall_of_fame_path(),
        opponent_variants: 4,
        screening_opponent_variants: 2,
        rounds_per_variant: 1,
        hall_of_fame_entries: 4,
        league_contenders: 3,
        league_hall_of_fame_entries: 2,
        min_pairs: 3,
        pair_batch: 1,
        max_pairs: 8,
        draw_window: 4,
        draw_rate_limit: 0.75,
        max_match_plies: 10,
        max_match_time_ms: 0,
        max_generations_without_candidate: 1,
        finalist_count: 2,
        search_strategy: TrainingSearchStrategy::AlphaBeta,
        sweep_parameter_groups: vec![SweepParameterGroup::ClassicBasic],
        sweep_points: 5,
        sweep_passes: Some(1),
        sweep_range_low: 1.0 / 3.0,
        sweep_range_high: 5.0 / 3.0,
        sweep_shrink: 0.5,
    }
}

#[test]
fn cpu_weight_mutations_are_usually_sparse_and_preserve_royals() {
    let baseline = EvalWeights::default_tuned();
    let baseline_json = serde_json::to_value(baseline).expect("serialize baseline");
    let mut sparse = 0;
    for seed in 1..=64 {
        let candidate = baseline.mutate(&mut Lcg::new(seed));
        assert_eq!(candidate.king, baseline.king);
        assert_eq!(candidate.royal_queen, baseline.royal_queen);
        let candidate_json = serde_json::to_value(candidate).expect("serialize candidate");
        let changed = baseline_json
            .as_object()
            .expect("baseline object")
            .iter()
            .filter(|(key, value)| candidate_json.get(*key) != Some(*value))
            .count();
        assert!(changed >= 1);
        if changed <= 6 {
            sparse += 1;
        }
    }
    assert!(sparse >= 48);
}

#[test]
fn trainer_loads_global_parameters_and_allows_cli_overrides() {
    let config = TrainerConfig::from_env(vec![
        "--effort".to_string(),
        "expert".to_string(),
        "--rounds-per-variant".to_string(),
        "3".to_string(),
        "--opponent-variants".to_string(),
        "5".to_string(),
    ]);

    assert_eq!(config.training_time_ms, 10_000);
    assert_eq!(config.nodes, 20_000);
    assert_eq!(config.population, 8);
    assert_eq!(config.min_pairs, 2);
    assert_eq!(config.max_pairs, 8);
    assert_eq!(config.draw_window, 4);
    assert_eq!(config.max_generations_without_candidate, 2);
    assert_eq!(config.rounds_per_variant, 3);
    assert_eq!(config.opponent_variants, 5);
    assert_eq!(config.screening_opponent_variants, 2);
    assert!(config
        .hall_of_fame
        .ends_with("models/cpu-v1/hall_of_fame.jsonl"));
    assert_eq!(config.training_strategy, CpuTrainingStrategy::Sweep);
    assert_eq!(
        config.sweep_parameter_groups,
        vec![SweepParameterGroup::ClassicBasic]
    );
    assert_eq!(config.sweep_points, 5);
    assert_eq!(config.sweep_passes, Some(2));
}

#[test]
fn trainer_parses_sweep_and_genetic_strategy_options() {
    let sweep = TrainerConfig::from_env(vec![
        "--parameter-groups".to_string(),
        "classic-basic,advanced".to_string(),
        "--sweep-points".to_string(),
        "7".to_string(),
        "--sweep-passes".to_string(),
        "3".to_string(),
        "--sweep-range".to_string(),
        "0.5:1.5".to_string(),
        "--sweep-shrink".to_string(),
        "0.25".to_string(),
    ]);

    assert_eq!(sweep.training_strategy, CpuTrainingStrategy::Sweep);
    assert_eq!(
        sweep.sweep_parameter_groups,
        vec![
            SweepParameterGroup::ClassicBasic,
            SweepParameterGroup::Advanced
        ]
    );
    assert_eq!(sweep.sweep_points, 7);
    assert_eq!(sweep.sweep_passes, Some(3));
    assert_eq!(sweep.sweep_range_low, 0.5);
    assert_eq!(sweep.sweep_range_high, 1.5);
    assert_eq!(sweep.sweep_shrink, 0.25);

    let genetic = TrainerConfig::from_env(vec!["--strategy".to_string(), "genetic".to_string()]);
    assert_eq!(genetic.training_strategy, CpuTrainingStrategy::Genetic);

    assert!(SweepParameterGroup::parse_list("basic").is_err());

    let split_basic_groups = TrainerConfig::from_env(vec![
        "--parameter-groups".to_string(),
        "classic-basic,alternate-basic".to_string(),
    ]);
    assert_eq!(
        split_basic_groups.sweep_parameter_groups,
        vec![
            SweepParameterGroup::ClassicBasic,
            SweepParameterGroup::AlternateBasic
        ]
    );
}

#[test]
fn compact_value_model_json_boundary_matches_cfnn_codec() {
    let input = serde_json::json!({
        "projectionSize": 4,
        "projectionSeed": 9,
        "hiddenLayers": [2],
        "hiddenWeights": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "outputWeights": [0.25, -0.5, 0.75],
        "policyWeights": [0.125, -0.25, 0.5],
        "auxiliaryValueWeights": [0.0625, -0.125],
        "scale": 1.5,
        "bias": -0.25,
        "outputActivation": "tanh"
    });
    let bytes =
        crate::gpu::training::compact_value_model_bytes_from_json(&input.to_string()).unwrap();
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 5);
    assert!(crate::gpu::training::compact_value_model_is_finite_bytes(
        &bytes
    ));

    let output: serde_json::Value =
        serde_json::from_str(&crate::gpu::training::compact_value_model_json(&bytes).unwrap())
            .unwrap();
    assert_eq!(output["projectionSize"], 4);
    assert_eq!(output["projectionSeed"], 9);
    assert_eq!(output["hiddenLayers"], serde_json::json!([2]));
    assert_eq!(output["outputActivation"], "tanh");
    assert_eq!(
        output["policyWeights"],
        serde_json::json!([0.125, -0.25, 0.5])
    );
    assert_eq!(
        output["auxiliaryValueWeights"],
        serde_json::json!([0.0625, -0.125])
    );

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(crate::gpu::training::compact_value_model_json(&trailing)
        .unwrap_err()
        .contains("trailing"));

    let mut non_finite = bytes;
    non_finite[48..52].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(!crate::gpu::training::compact_value_model_is_finite_bytes(
        &non_finite
    ));
    assert!(crate::gpu::training::compact_value_model_json(&non_finite).is_err());
}

#[test]
fn compact_value_model_frontier_layout_expands_search_model_buffers() {
    let input_size = crate::gpu::training::DEFAULT_PROJECTION_SIZE;
    let hidden_layers = crate::gpu::training::DEFAULT_HIDDEN_LAYERS;
    let hidden_len = hidden_layers
        .iter()
        .copied()
        .fold((0usize, input_size), |(total, previous), layer| {
            (total + layer as usize * (previous + 1), layer as usize)
        })
        .0;
    let output_size = *hidden_layers.last().unwrap() as usize;
    let policy_weight_len = crate::gpu::training::POLICY_BUCKETS as usize * (output_size + 1);
    let input = serde_json::json!({
        "projectionSize": input_size,
        "projectionSeed": crate::gpu::training::DEFAULT_PROJECTION_SEED,
        "hiddenLayers": hidden_layers,
        "hiddenWeights": vec![0.0; hidden_len],
        "outputWeights": vec![0.0; output_size + 1],
        "policyWeights": vec![0.25; policy_weight_len],
        "scale": 1.0,
        "bias": 0.0,
        "outputActivation": "tanh"
    });
    let bytes =
        crate::gpu::training::compact_value_model_bytes_from_json(&input.to_string()).unwrap();
    let layout: serde_json::Value = serde_json::from_str(
        &crate::gpu::training::compact_value_model_frontier_layout_json(&bytes).unwrap(),
    )
    .unwrap();

    assert_eq!(layout["architectureMatches"], true);
    assert_eq!(layout["outputLayerSize"], output_size);
    assert_eq!(
        layout["hiddenLayerWeights"].as_array().unwrap().len(),
        hidden_layers.len()
    );
    assert_eq!(
        layout["hiddenLayerWeights"][0].as_array().unwrap().len(),
        hidden_layers[0] as usize * (input_size + 1)
    );
    let policy_weights = layout["policyWeights"].as_array().unwrap();
    assert_eq!(policy_weights.len(), policy_weight_len);
    assert_eq!(policy_weights[output_size], serde_json::json!(0.25));
    assert_eq!(
        policy_weights
            [(crate::gpu::training::POLICY_BUCKETS as usize - 1) * (output_size + 1) + output_size],
        serde_json::json!(0.25)
    );

    let invalid = serde_json::json!({
        "projectionSize": 4,
        "projectionSeed": crate::gpu::training::DEFAULT_PROJECTION_SEED,
        "hiddenLayers": hidden_layers,
        "hiddenWeights": [0.0],
        "outputWeights": [0.0],
        "outputActivation": "tanh"
    });
    let invalid_bytes =
        crate::gpu::training::compact_value_model_bytes_from_json(&invalid.to_string()).unwrap();
    let invalid_layout: serde_json::Value = serde_json::from_str(
        &crate::gpu::training::compact_value_model_frontier_layout_json(&invalid_bytes).unwrap(),
    )
    .unwrap();
    assert_eq!(invalid_layout["architectureMatches"], false);
}

#[test]
fn compact_value_model_predicts_values_from_samples() {
    let seed = 17;
    let projected = [0, 1].map(|output| {
        if crate::gpu::training::projection_hash(0, output, seed) & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    });
    let hidden = 0.0_f32.max(projected[0] + projected[1] * 2.0 + 0.5);
    let model = crate::gpu::training::CompactValueModel {
        version: 4,
        projection_size: 2,
        projection_seed: seed,
        hidden_layers: vec![1],
        hidden_weights: vec![1.0, 2.0, 0.5],
        output_weights: vec![3.0, 1.0],
        policy_logits: Vec::new(),
        policy_weights: Vec::new(),
        auxiliary_value_weights: Vec::new(),
        scale: 0.1,
        bias: 0.2,
        output_activation: crate::gpu::training::OutputActivation::Tanh,
    };
    let sample = crate::gpu::training::TrainingSample {
        side_to_move: None,
        board_count: None,
        position_key: Some("predict".to_string()),
        features: vec![1.0, 0.0],
        label: 0.0,
        label_kind: None,
        label_weight: 1.0,
        base_label_weight: None,
        label_mass: None,
        observation_count: None,
        policy: None,
        pseudo: None,
    };
    let predictions = model.predict_values([sample.features.as_slice()]);
    let expected = ((hidden * 3.0 + 1.0).tanh() * 0.1 + 0.2).clamp(-1.0, 1.0);
    assert_eq!(predictions.len(), 1);
    assert!((predictions[0] - expected).abs() < 1e-6);
}

#[test]
fn compact_value_model_training_layout_bytes_initializes_defaults() {
    let bytes = crate::gpu::training::compact_value_model_training_layout_bytes(None, 0.5).unwrap();
    let header = bytes[..16]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(header[0], 0);
    assert_eq!(header[1], 256);
    assert_eq!(
        header[2] as usize,
        crate::gpu::training::default_initial_hidden_weights().len()
    );
    assert_eq!(header[3], 257);
    let output_start = 16 + header[2] as usize * 4;
    let output_bias_offset = output_start + header[1] as usize * 4;
    let output_bias = f32::from_le_bytes(
        bytes[output_bias_offset..output_bias_offset + 4]
            .try_into()
            .unwrap(),
    );
    assert!((output_bias - crate::gpu::training::inverse_tanh(0.5)).abs() < 1e-6);

    let model = crate::gpu::training::CompactValueModel {
        version: 4,
        projection_size: crate::gpu::training::DEFAULT_PROJECTION_SIZE as u32,
        projection_seed: crate::gpu::training::DEFAULT_PROJECTION_SEED,
        hidden_layers: crate::gpu::training::DEFAULT_HIDDEN_LAYERS.to_vec(),
        hidden_weights: crate::gpu::training::default_initial_hidden_weights(),
        output_weights: vec![0.25; 257],
        policy_logits: Vec::new(),
        policy_weights: Vec::new(),
        auxiliary_value_weights: Vec::new(),
        scale: 1.0,
        bias: 0.0,
        output_activation: crate::gpu::training::OutputActivation::Tanh,
    };
    let active = crate::gpu::training::compact_value_model_training_layout_bytes(
        Some(&model.encode()),
        -0.5,
    )
    .unwrap();
    let active_header = active[..16]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(active_header[0], 1);
    let active_output_start = 16 + active_header[2] as usize * 4;
    let first_output = f32::from_le_bytes(
        active[active_output_start..active_output_start + 4]
            .try_into()
            .unwrap(),
    );
    assert_eq!(first_output, 0.25);
}

#[test]
fn trainer_parses_gpu_model_info_mode() {
    let default_info = TrainerConfig::from_env(vec!["--gpu-model-info".to_string()]);
    assert_eq!(
        default_info.gpu_model_info.as_deref(),
        Some(crate::gpu::training::DEFAULT_VALUE_MODEL_PATH)
    );

    let backend_info = TrainerConfig::from_env(vec!["--gpu-backend-info".to_string()]);
    assert!(backend_info.gpu_backend_info);

    let compile_shaders = TrainerConfig::from_env(vec!["--gpu-compile-shaders".to_string()]);
    assert!(compile_shaders.gpu_compile_shaders);

    let compile_kernels = TrainerConfig::from_env(vec!["--gpu-compile-kernels".to_string()]);
    assert!(compile_kernels.gpu_compile_kernels);

    let dispatch_smoke = TrainerConfig::from_env(vec!["--gpu-dispatch-smoke".to_string()]);
    assert!(dispatch_smoke.gpu_dispatch_smoke);

    let training_dispatch_smoke =
        TrainerConfig::from_env(vec!["--gpu-training-dispatch-smoke".to_string()]);
    assert!(training_dispatch_smoke.gpu_training_dispatch_smoke);

    let shader_info = TrainerConfig::from_env(vec!["--gpu-shader-info".to_string()]);
    assert!(shader_info.gpu_shader_info);

    let custom_model = TrainerConfig::from_env(vec![
        "--gpu-model".to_string(),
        "custom-value.cfnn".to_string(),
    ]);
    assert_eq!(custom_model.gpu_value_model_path, "custom-value.cfnn");

    let custom_info = TrainerConfig::from_env(vec![
        "--gpu-model-info".to_string(),
        "custom-model.cfnn".to_string(),
    ]);
    assert_eq!(
        custom_info.gpu_model_info.as_deref(),
        Some("custom-model.cfnn")
    );

    let probe_zero = TrainerConfig::from_env(vec!["--gpu-model-probe-zero".to_string()]);
    assert_eq!(
        probe_zero.gpu_model_probe_zero.as_deref(),
        Some(crate::gpu::training::DEFAULT_VALUE_MODEL_PATH)
    );

    let gpu_project_samples = TrainerConfig::from_env(vec![
        "--gpu-project-samples".to_string(),
        "samples.json".to_string(),
    ]);
    assert_eq!(
        gpu_project_samples.gpu_project_samples.as_deref(),
        Some("samples.json")
    );

    let gpu_predict_samples = TrainerConfig::from_env(vec![
        "--gpu-predict-samples".to_string(),
        "samples.json".to_string(),
    ]);
    assert_eq!(
        gpu_predict_samples.gpu_predict_samples.as_deref(),
        Some("samples.json")
    );

    let gpu_distill_samples = TrainerConfig::from_env(vec![
        "--gpu-distill-samples".to_string(),
        "samples.json".to_string(),
    ]);
    assert_eq!(
        gpu_distill_samples.gpu_distill_samples.as_deref(),
        Some("samples.json")
    );

    let gpu_replay_append = TrainerConfig::from_env(vec![
        "--gpu-replay-buffer".to_string(),
        "buffer.json".to_string(),
        "--gpu-replay-append".to_string(),
        "samples.json".to_string(),
        "--gpu-replay-max".to_string(),
        "17".to_string(),
    ]);
    assert_eq!(
        gpu_replay_append.gpu_replay_buffer.as_deref(),
        Some("buffer.json")
    );
    assert_eq!(
        gpu_replay_append.gpu_replay_append.as_deref(),
        Some("samples.json")
    );
    assert_eq!(gpu_replay_append.gpu_replay_max, 17);

    let gpu_search = TrainerConfig::from_env(vec!["--gpu-search".to_string()]);
    assert_eq!(gpu_search.gpu_search_snapshot.as_deref(), Some(""));

    let gpu_search_file = TrainerConfig::from_env(vec![
        "--gpu-search".to_string(),
        "snapshot.json".to_string(),
    ]);
    assert_eq!(
        gpu_search_file.gpu_search_snapshot.as_deref(),
        Some("snapshot.json")
    );

    let gpu_search_depth = TrainerConfig::from_env(vec![
        "--gpu-search".to_string(),
        "--gpu-search-depth".to_string(),
        "1".to_string(),
        "--gpu-search-min-depth".to_string(),
        "1".to_string(),
    ]);
    assert_eq!(gpu_search_depth.gpu_search_snapshot.as_deref(), Some(""));
    assert_eq!(gpu_search_depth.gpu_search_depth, Some(1));
    assert_eq!(gpu_search_depth.gpu_search_min_depth, Some(1));

    let gpu_train_samples = TrainerConfig::from_env(vec![
        "--gpu-train-samples".to_string(),
        "samples.json".to_string(),
    ]);
    assert_eq!(
        gpu_train_samples.gpu_train_samples.as_deref(),
        Some("samples.json")
    );

    let gpu_train_projected_samples = TrainerConfig::from_env(vec![
        "--gpu-train-projected-samples".to_string(),
        "samples.json".to_string(),
    ]);
    assert_eq!(
        gpu_train_projected_samples
            .gpu_train_projected_samples
            .as_deref(),
        Some("samples.json")
    );

    let gpu_train_search = TrainerConfig::from_env(vec!["--gpu-train-search".to_string()]);
    assert_eq!(
        gpu_train_search.gpu_train_search_snapshot.as_deref(),
        Some("")
    );

    let gpu_train_search_file = TrainerConfig::from_env(vec![
        "--gpu-train-search".to_string(),
        "snapshot.json".to_string(),
    ]);
    assert_eq!(
        gpu_train_search_file.gpu_train_search_snapshot.as_deref(),
        Some("snapshot.json")
    );

    let gpu_sample_search = TrainerConfig::from_env(vec!["--gpu-sample-search".to_string()]);
    assert_eq!(
        gpu_sample_search.gpu_sample_search_snapshot.as_deref(),
        Some("")
    );

    let gpu_sample_search_file = TrainerConfig::from_env(vec![
        "--gpu-sample-search".to_string(),
        "snapshot.json".to_string(),
    ]);
    assert_eq!(
        gpu_sample_search_file.gpu_sample_search_snapshot.as_deref(),
        Some("snapshot.json")
    );

    let gpu_sample_batch = TrainerConfig::from_env(vec![
        "--gpu-sample-search".to_string(),
        "--gpu-sample-count".to_string(),
        "7".to_string(),
        "--gpu-sample-mode".to_string(),
        "cpu".to_string(),
        "--gpu-sample-plies".to_string(),
        "3".to_string(),
    ]);
    assert_eq!(gpu_sample_batch.gpu_sample_count, 7);
    assert_eq!(
        gpu_sample_batch.gpu_sample_mode,
        crate::gpu::training::SearchLabelMode::Cpu
    );
    assert_eq!(gpu_sample_batch.gpu_sample_max_plies, 3);
    assert_eq!(
        gpu_sample_search.gpu_sample_mode,
        crate::gpu::training::SearchLabelMode::Search
    );
    assert_eq!(
        crate::gpu::training::SearchLabelMode::parse("duel").expect("duel sample mode"),
        crate::gpu::training::SearchLabelMode::Duel
    );
    assert_eq!(
        crate::gpu::training::SearchLabelMode::parse("distilled").expect("distilled sample mode"),
        crate::gpu::training::SearchLabelMode::Distilled
    );

    let cpu_search = TrainerConfig::from_env(vec!["--cpu-search".to_string()]);
    assert_eq!(cpu_search.cpu_search_snapshot.as_deref(), Some(""));

    let cpu_search_file = TrainerConfig::from_env(vec![
        "--cpu-search".to_string(),
        "snapshot.json".to_string(),
    ]);
    assert_eq!(
        cpu_search_file.cpu_search_snapshot.as_deref(),
        Some("snapshot.json")
    );
}

#[test]
fn weight_parameter_groups_match_readme_boundaries_and_preserve_royals() {
    let parameters = weight_parameters();
    let classic_basic = sweep_weight_parameters(&[SweepParameterGroup::ClassicBasic]);
    let alternate_basic = sweep_weight_parameters(&[SweepParameterGroup::AlternateBasic]);
    let basic = sweep_weight_parameters(&[
        SweepParameterGroup::ClassicBasic,
        SweepParameterGroup::AlternateBasic,
    ]);
    let intermediate = sweep_weight_parameters(&[SweepParameterGroup::Intermediate]);
    let advanced = sweep_weight_parameters(&[SweepParameterGroup::Advanced]);

    assert_eq!(
        classic_basic.first().map(|parameter| parameter.name),
        Some("queen")
    );
    assert!(classic_basic
        .iter()
        .any(|parameter| parameter.name == "rook"));
    assert!(classic_basic
        .iter()
        .any(|parameter| parameter.name == "bishop"));
    assert!(classic_basic
        .iter()
        .any(|parameter| parameter.name == "knight"));
    assert!(classic_basic
        .iter()
        .any(|parameter| parameter.name == "pawn"));
    assert!(!classic_basic
        .iter()
        .any(|parameter| parameter.name == "common_king"));
    assert!(!classic_basic
        .iter()
        .any(|parameter| parameter.name == "king"));

    assert_eq!(
        alternate_basic.first().map(|parameter| parameter.name),
        Some("common_king")
    );
    assert!(alternate_basic
        .iter()
        .any(|parameter| parameter.name == "princess"));
    assert!(alternate_basic
        .iter()
        .any(|parameter| parameter.name == "unicorn"));
    assert!(alternate_basic
        .iter()
        .any(|parameter| parameter.name == "dragon"));
    assert!(alternate_basic
        .iter()
        .any(|parameter| parameter.name == "brawn"));
    assert!(!alternate_basic
        .iter()
        .any(|parameter| parameter.name == "royal_queen"));
    assert!(!alternate_basic
        .iter()
        .any(|parameter| parameter.name == "queen"));

    assert!(classic_basic
        .iter()
        .any(|parameter| parameter.name == "royal_threat"));
    assert!(alternate_basic
        .iter()
        .any(|parameter| parameter.name == "royal_threat"));
    assert!(basic
        .iter()
        .any(|parameter| parameter.name == "royal_threat"));
    assert!(!basic.iter().any(|parameter| parameter.name == "king"));
    assert!(!basic
        .iter()
        .any(|parameter| parameter.name == "royal_queen"));
    assert_eq!(
        intermediate.first().map(|parameter| parameter.name),
        Some("temporal_threat")
    );
    assert_eq!(
        advanced.first().map(|parameter| parameter.name),
        Some("mandatory_move_burden")
    );
    assert!(parameters.iter().any(|parameter| {
        parameter.name == "mate_net_depth_1_2" && parameter.json_name == "mateNetDepth12"
    }));
}

#[test]
fn sweep_value_generation_uses_ranges_bounds_zero_windows_and_deduplication() {
    let queen = weight_parameters()
        .iter()
        .find(|parameter| parameter.name == "queen")
        .copied()
        .expect("queen metadata");
    assert_eq!(
        sweep_values(queen, 900, 5, 1.0 / 3.0, 5.0 / 3.0),
        vec![300, 600, 900, 1200, 1500]
    );

    let mobility = weight_parameters()
        .iter()
        .find(|parameter| parameter.name == "mobility")
        .copied()
        .expect("mobility metadata");
    assert_eq!(
        sweep_values(mobility, 0, 5, 1.0 / 3.0, 5.0 / 3.0),
        vec![0, 1, 2, 3, 4]
    );
    assert_eq!(
        sweep_values(mobility, 80, 5, 1.0 / 3.0, 5.0 / 3.0),
        vec![27, 40, 54, 67, 80]
    );

    let centrality = weight_parameters()
        .iter()
        .find(|parameter| parameter.name == "centrality")
        .copied()
        .expect("centrality metadata");
    assert_eq!(sweep_values(centrality, 1, 5, 0.9, 1.1), vec![1]);
}

#[test]
fn sweep_winner_prefers_score_then_blunders_then_smaller_movement() {
    let weights = EvalWeights::default_tuned();
    let lower_score = SweepScore {
        value: 1,
        weights,
        score: 9,
        blunders: 0,
        movement: 0,
    };
    let more_blunders = SweepScore {
        value: 2,
        weights,
        score: 10,
        blunders: 3,
        movement: 1,
    };
    let larger_movement = SweepScore {
        value: 3,
        weights,
        score: 10,
        blunders: 1,
        movement: 4,
    };
    let winner = SweepScore {
        value: 4,
        weights,
        score: 10,
        blunders: 1,
        movement: 2,
    };

    assert_eq!(
        select_sweep_winner(&[lower_score, more_blunders, larger_movement, winner])
            .expect("winner")
            .value,
        4
    );
}

#[test]
fn promotion_writes_weight_parameters_only() {
    let path = std::env::temp_dir().join(format!(
        "chronofish-parameters-{}-{}.json",
        std::process::id(),
        random_seed()
    ));
    std::fs::write(&path, r#"{"king":1}"#).expect("test parameters should be written");

    promote_weights(
        EvalWeights::default_tuned(),
        path.to_str().expect("UTF-8 path"),
    );

    let value: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path).expect("promoted parameters should be readable"),
    )
    .expect("promoted parameters should be JSON");
    assert!(value.get("training").is_none());
    assert_eq!(value["king"], EvalWeights::default_tuned().king);
    let _ = std::fs::remove_file(path);
}

#[test]
fn training_json_contains_the_global_training_config() {
    let training = load_training_parameters();
    assert_eq!(training.time_ms, 10_000);
    assert_eq!(training.nodes, 20_000);
    assert_eq!(training.candidates, Some(8));
    assert_eq!(training.opponent_variants, 8);
    assert_eq!(training.rounds_per_variant, 1);
    assert_eq!(training.min_pairs, 2);
    assert_eq!(training.max_pairs, 8);
}

#[test]
fn baseline_fitness_cache_key_tracks_scoring_inputs() {
    let config = trainer_test_config();
    let same = config.with_search(config.nodes, config.training_time_ms);
    let deeper = config.with_search(config.nodes + 1, config.training_time_ms);
    let slower = config.with_search(config.nodes, config.training_time_ms + 1);
    let mut more_opponents = trainer_test_config();
    more_opponents.opponent_variants += 1;
    let mut shorter_matches = trainer_test_config();
    shorter_matches.max_match_plies -= 1;
    let mut timed_matches = trainer_test_config();
    timed_matches.max_match_time_ms = 123;
    let mut beam_search = trainer_test_config();
    beam_search.search_strategy = TrainingSearchStrategy::Beam;
    let mut different_hall = trainer_test_config();
    different_hall.hall_of_fame.push_str(".other");

    assert_eq!(baseline_fitness_key(&config), baseline_fitness_key(&same));
    assert_ne!(baseline_fitness_key(&config), baseline_fitness_key(&deeper));
    assert_ne!(baseline_fitness_key(&config), baseline_fitness_key(&slower));
    assert_ne!(
        baseline_fitness_key(&config),
        baseline_fitness_key(&more_opponents)
    );
    assert_ne!(
        baseline_fitness_key(&config),
        baseline_fitness_key(&shorter_matches)
    );
    assert_ne!(
        baseline_fitness_key(&config),
        baseline_fitness_key(&timed_matches)
    );
    assert_ne!(
        baseline_fitness_key(&config),
        baseline_fitness_key(&beam_search)
    );
    assert_ne!(
        baseline_fitness_key(&config),
        baseline_fitness_key(&different_hall)
    );
}

#[test]
fn finalist_scoring_jobs_parallelize_uncached_baseline_without_duplicate_candidates() {
    let committed = EvalWeights::default_tuned();
    let mut rng = Lcg::new(42);
    let candidate = committed.mutate(&mut rng);
    let finalists = [committed, candidate, candidate];

    let uncached = finalist_scoring_jobs(&finalists, committed, false);
    assert_eq!(uncached.len(), 2);
    assert!(uncached[0] == candidate);
    assert!(uncached[1] == committed);

    let cached = finalist_scoring_jobs(&finalists, committed, true);
    assert_eq!(cached.len(), 1);
    assert!(cached[0] == candidate);
}

#[test]
fn candidate_population_deduplication_preserves_first_seen_order() {
    let committed = EvalWeights::default_tuned();
    let mut rng = Lcg::new(99);
    let first = committed.mutate(&mut rng);
    let second = committed.mutate(&mut rng);

    let unique = unique_weights(&[committed, first, committed, second, first]);

    assert_eq!(unique.len(), 3);
    assert!(unique[0] == committed);
    assert!(unique[1] == first);
    assert!(unique[2] == second);
}

#[test]
fn candidate_fitness_cache_separates_search_configs_and_opponent_limits() {
    let config = trainer_test_config();
    let key = baseline_fitness_key(&config);
    let mut deeper = config.clone();
    deeper.nodes += 1;
    let deeper_key = baseline_fitness_key(&deeper);
    let weights = EvalWeights::default_tuned();
    let report = FitnessReport {
        score: 123,
        ..FitnessReport::default()
    };
    let mut cache = Vec::new();

    cache_fitness_report(&mut cache, key.clone(), 2, weights, report);
    cache_fitness_report(&mut cache, key.clone(), 2, weights, report);

    assert_eq!(cache.len(), 1);
    assert_eq!(
        cached_fitness_report(&cache, &key, 2, weights)
            .expect("matching fitness should be cached")
            .score,
        123
    );
    assert!(cached_fitness_report(&cache, &key, 3, weights).is_none());
    assert!(cached_fitness_report(&cache, &deeper_key, 2, weights).is_none());
}

#[test]
fn initialized_hall_of_fame_contains_valid_weights() {
    let entries = load_hall_of_fame(&default_hall_of_fame_path(), 4);
    assert!(!entries.is_empty());
    assert!(entries[0] == EvalWeights::default_tuned());
}

#[test]
fn full_match_scoring_stops_at_match_ply_cap() {
    let mut config = trainer_test_config();
    config.max_match_plies = 1;
    config.nodes = 50;
    config.training_time_ms = 20;
    let weights = EvalWeights::default_tuned();
    let report = play_match_until(
        Game::new(),
        weights,
        weights,
        Color::White,
        "candidate",
        "baseline",
        "ply cap smoke",
        &config,
        None,
    );

    assert!(!report.blunder);
}

#[test]
fn full_match_scoring_stops_at_match_time_cap() {
    let mut config = trainer_test_config();
    config.max_match_plies = 80;
    config.max_match_time_ms = 1;
    config.nodes = 200;
    config.training_time_ms = 20;
    let weights = EvalWeights::default_tuned();
    let report = play_match_until(
        Game::new(),
        weights,
        weights,
        Color::White,
        "candidate",
        "baseline",
        "time cap smoke",
        &config,
        None,
    );

    assert!(!report.blunder);
}

#[test]
fn alpha_beta_training_strategy_is_always_available() {
    assert_eq!(
        TrainingSearchStrategy::parse("alpha-beta"),
        Ok(TrainingSearchStrategy::AlphaBeta)
    );
}

#[test]
fn time_bounded_training_search_returns_only_applicable_turns() {
    let mut config = trainer_test_config();
    config.nodes = 200;
    config.training_time_ms = 20;
    let weights = EvalWeights::default_tuned();
    let seeds: Vec<u64> = (0..8).chain([3_471_131_662_115_554_319]).collect();

    for seed in seeds {
        let mut game = seeded_start_position(seed, &config, None);
        for turn in 1..=12 {
            let Some(plan) = training_turn_plan(&game, weights, &config, None) else {
                break;
            };
            game = game.apply_turn_plan_for_search(&plan).unwrap_or_else(|| {
                panic!("seed {seed} turn {turn} produced an inapplicable training plan")
            });
        }
    }
}

#[test]
fn bounded_evaluation_honors_attack_budget() {
    let game = multi_present_training_game(3);
    let weights = EvalWeights::default_tuned();
    let mut limits = EvaluationLimits::training_fast_late_game(10);
    limits.attack_checks = 1;
    let mut stats = EvaluationStats::default();

    let first = game.evaluate_heuristic_with_limits(Color::White, &weights, limits, &mut stats);
    let mut second_stats = EvaluationStats::default();
    let second =
        game.evaluate_heuristic_with_limits(Color::White, &weights, limits, &mut second_stats);

    assert!(stats.attack_checks <= 1);
    assert_eq!(first, second);
    assert_eq!(stats.attack_checks, second_stats.attack_checks);
    assert_eq!(stats.attack_caps, second_stats.attack_caps);
}

fn multi_present_training_game(count: i32) -> Game {
    let mut game = Game::new();
    game.timelines = (0..count)
        .map(|id| Timeline {
            id,
            row: id,
            label: format!("L{id}"),
            owner: TimelineOwner::Neutral,
            boards: vec![snapshot(0, Color::White, empty_board_with_kings())],
        })
        .collect();
    game
}

#[test]
fn training_search_profile_caps_late_game_branching() {
    let weights = EvalWeights::default_tuned();
    let normal = multi_present_training_game(1);
    let mut normal_context = SearchContext::new(weights, normal.turn, 20_000, None);
    let normal_profile = apply_training_search_profile(&normal, &mut normal_context, 0);
    assert_eq!(normal_profile.obligations, 1);
    assert_eq!(normal_context.root_plan_limit(), MAX_ROOT_TURN_PLANS);
    assert_eq!(normal_context.child_plan_limit(), MAX_CHILD_TURN_PLANS);

    let busy = multi_present_training_game(3);
    let mut busy_context = SearchContext::new(weights, busy.turn, 20_000, None);
    let busy_profile = apply_training_search_profile(&busy, &mut busy_context, 0);
    assert_eq!(busy_profile.obligations, 3);
    assert_eq!(busy_context.root_plan_limit(), 4);
    assert_eq!(busy_context.child_plan_limit(), 2);

    let late = multi_present_training_game(4);
    let mut late_context = SearchContext::new(weights, late.turn, 20_000, None);
    let late_profile = apply_training_search_profile(&late, &mut late_context, 24);
    assert_eq!(late_profile.obligations, 4);
    assert_eq!(late_context.root_plan_limit(), 2);
    assert_eq!(late_context.child_plan_limit(), 1);
    assert!(late_context.evaluation_limits.is_some());
}

#[test]
#[ignore = "reported overnight seed throughput smoke test; run with --ignored --nocapture"]
fn reported_seed_reaches_late_training_turns_with_bounded_search() {
    let mut config = trainer_test_config();
    config.nodes = 2_000;
    config.training_time_ms = 1_000;
    let weights = EvalWeights::default_tuned();
    let mut game = seeded_start_position(10_848_506_003_217_676_803, &config, None);

    let mut completed_turns = 0;
    for turn in 1..=30 {
        if game.terminal_score(game.turn).is_some() {
            break;
        }
        let started = SearchInstant::now();
        let Some(outcome) = training_turn_search(&game, weights, &config, None, turn - 1) else {
            panic!("training search should keep producing full turns through turn {turn}");
        };
        let elapsed = SearchInstant::now().duration_since(started).as_millis();
        assert!(
            elapsed < 5_000,
            "reported seed turn {turn} took {elapsed}ms with bounded training search"
        );
        game = game
            .apply_turn_plan_for_search(&outcome.plan)
            .expect("training plan should apply and submit");
        completed_turns = turn;
    }
    assert!(
        completed_turns >= 15 || game.terminal_score(game.turn).is_some(),
        "reported seed should either reach late training turns or finish naturally"
    );
}

#[test]
#[ignore = "release-mode training throughput smoke test; run with --ignored --nocapture"]
fn fast_training_search_reaches_turn_fifteen() {
    let mut game = Game::new();
    let mut config = trainer_test_config();
    config.training_time_ms = 1_000;
    config.nodes = 2_000;
    let weights = EvalWeights::default_tuned();

    for turn in 1..=15 {
        let started = SearchInstant::now();
        let plan = training_turn_plan(&game, weights, &config, None)
            .expect("training search should find a turn");
        let elapsed = SearchInstant::now().duration_since(started).as_millis();
        eprintln!(
            "training smoke turn {turn}: {elapsed}ms notation={}",
            turn_plan_notation(&game, &plan),
        );
        game = game
            .apply_turn_plan_for_search(&plan)
            .expect("training plan should apply and submit");
    }
}

#[test]
fn beam_training_strategy_returns_submit_valid_turn() {
    assert_eq!(
        TrainingSearchStrategy::parse("beam"),
        Ok(TrainingSearchStrategy::Beam)
    );

    let game = Game::new();
    let mut config = trainer_test_config();
    config.nodes = 200;
    config.search_strategy = TrainingSearchStrategy::Beam;

    let plan = training_turn_plan(&game, EvalWeights::default_tuned(), &config, None)
        .expect("beam search should find a turn");
    let mut replay = game;
    for movement in plan.moves {
        assert_eq!(replay.apply_move(movement.from, movement.to), 1);
    }
    assert_eq!(replay.submit_turn(), 1);
}

#[test]
fn statistical_decision_promotes_significant_winner() {
    let config = trainer_test_config();
    let mut stats = ComparisonStats::default();
    let mut deltas = Vec::new();
    for delta in [100, 120, 140, 160, 180, 200, 220, 240] {
        stats.record(delta);
        deltas.push(delta);
    }

    assert_eq!(
        statistical_decision(stats, &deltas, significance(&deltas), &config),
        StatisticalDecision::Promote
    );
}

#[test]
fn statistical_decision_rejects_significant_loser() {
    let config = trainer_test_config();
    let mut stats = ComparisonStats::default();
    let mut deltas = Vec::new();
    for delta in [-100, -120, -140, -160, -180, -200, -220, -240] {
        stats.record(delta);
        deltas.push(delta);
    }

    assert_eq!(
        statistical_decision(stats, &deltas, significance(&deltas), &config),
        StatisticalDecision::Reject
    );
}

#[test]
fn statistical_decision_marks_draw_stagnation_inconclusive() {
    let config = trainer_test_config();
    let mut stats = ComparisonStats::default();
    let mut deltas = Vec::new();
    for delta in [0, 0, 0, 0] {
        stats.record(delta);
        deltas.push(delta);
    }

    assert_eq!(
        statistical_decision(stats, &deltas, significance(&deltas), &config),
        StatisticalDecision::Inconclusive
    );
}
