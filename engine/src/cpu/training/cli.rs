use super::*;
use crate::cpu::EvalWeights;

pub fn run_training_cli() {
    let config = TrainerConfig::from_env(std::env::args().skip(1).collect());

    if config.train_cycle {
        // The top-level ./train script loops this mode until interrupted.
        match config.training_strategy {
            CpuTrainingStrategy::Sweep => run_sweep_training_cycle(&config),
            CpuTrainingStrategy::Genetic => run_training_cycle(&config),
        }
        return;
    }

    if config.score_default {
        println!(
            "{}",
            fitness(EvalWeights::default_tuned(), &config).summary()
        );
        return;
    }

    if config.gpu_backend_info {
        println!("{}", native_gpu_backend_info());
        return;
    }

    if config.gpu_compile_shaders {
        println!("{}", native_gpu_compile_shaders());
        return;
    }

    if config.gpu_compile_kernels {
        println!("{}", native_gpu_compile_kernels());
        return;
    }

    if config.gpu_dispatch_smoke {
        println!("{}", native_gpu_dispatch_smoke());
        return;
    }

    if config.gpu_training_dispatch_smoke {
        println!("{}", native_gpu_training_dispatch_smoke());
        return;
    }

    if config.gpu_shader_info {
        println!("{}", gpu_shader_info());
        return;
    }

    if let Some(path) = &config.gpu_model_info {
        let model = crate::gpu::training::load_compact_value_model(path)
            .unwrap_or_else(|message| panic!("{message}"));
        println!("{}", model.summary());
        return;
    }

    if let Some(path) = &config.gpu_model_probe_zero {
        let model = crate::gpu::training::load_compact_value_model(path)
            .unwrap_or_else(|message| panic!("{message}"));
        let prediction = model.predict_value(&[]);
        println!("gpu_value_zero={prediction}");
        return;
    }

    if let Some(path) = &config.gpu_project_samples {
        println!("{}", native_gpu_project_samples(path, &config));
        return;
    }

    if let Some(path) = &config.gpu_predict_samples {
        println!("{}", native_gpu_predict_samples(path, &config));
        return;
    }

    if let Some(path) = &config.gpu_distill_samples {
        println!("{}", gpu_distill_samples(path, &config));
        return;
    }

    if config.gpu_replay_append.is_some() {
        println!("{}", gpu_append_replay_samples(&config));
        return;
    }

    if config.gpu_search_snapshot.is_some() {
        let request = gpu_search_request(&config);
        let response =
            crate::gpu::search::search(request).unwrap_or_else(|message| panic!("{message}"));
        println!("{}", response.result_json);
        return;
    }

    if config.gpu_sample_search_snapshot.is_some() {
        let request = gpu_search_label_batch_request(&config);
        let response = crate::gpu::training::collect_search_label_samples(request)
            .unwrap_or_else(|message| panic!("{message}"));
        if let Some(out) = &config.out {
            crate::gpu::training::save_training_samples_json(out, &response.samples)
                .unwrap_or_else(|message| panic!("{message}"));
        }
        println!(
            "gpu_sample_search samples={} source={} requested={} generated={} labeled={} wrote={}",
            response.samples.len(),
            response.source,
            response.requested,
            response.generated_positions,
            response.labeled_positions,
            config.out.as_deref().unwrap_or("")
        );
        if config.out.is_none() {
            let json = serde_json::to_string_pretty(&response.samples)
                .unwrap_or_else(|error| panic!("failed to encode GPU training samples: {error}"));
            println!("{json}");
        }
        return;
    }

    if config.gpu_train_search_snapshot.is_some() {
        let request = gpu_train_search_label_batch_request(&config);
        let response = crate::gpu::training::collect_search_label_samples(request)
            .unwrap_or_else(|message| panic!("{message}"));
        let (value_report, policy_report, wrote) =
            train_gpu_value_model_from_samples(&response.samples, &config);
        println!(
            "gpu_train_search samples={} requested={} generated={} labeled={} value_epochs={} value_initial_loss={} value_final_loss={} policy_samples={} policy_steps={} policy_initial_loss={} policy_final_loss={} wrote={}",
            response.samples.len(),
            response.requested,
            response.generated_positions,
            response.labeled_positions,
            value_report.epochs,
            value_report.initial_loss,
            value_report.final_loss,
            policy_report.samples,
            policy_report.steps,
            policy_report.initial_loss,
            policy_report.final_loss,
            wrote
        );
        return;
    }

    if let Some(path) = &config.gpu_train_samples {
        let samples = crate::gpu::training::load_training_samples_json(path)
            .unwrap_or_else(|message| panic!("{message}"));
        let (value_report, policy_report, wrote) =
            train_gpu_value_model_from_samples(&samples, &config);
        println!(
            "gpu_train_value_head samples={} epochs={} initial_loss={} final_loss={} policy_samples={} policy_steps={} policy_initial_loss={} policy_final_loss={} wrote={}",
            value_report.samples,
            value_report.epochs,
            value_report.initial_loss,
            value_report.final_loss,
            policy_report.samples,
            policy_report.steps,
            policy_report.initial_loss,
            policy_report.final_loss,
            wrote
        );
        return;
    }

    if let Some(path) = &config.gpu_train_projected_samples {
        println!("{}", native_gpu_train_projected_samples(path, &config));
        return;
    }

    if config.cpu_search_snapshot.is_some() {
        let request = cpu_search_request(&config);
        let response =
            crate::cpu::search::search(request).unwrap_or_else(|message| panic!("{message}"));
        println!("{}", response.result_json);
        return;
    }

    if let Some(path) = &config.score {
        let json = std::fs::read_to_string(path).expect("failed to read score weights");
        let weights = EvalWeights::from_json(&json).expect("failed to parse score weights");
        println!("{}", fitness(weights, &config).summary());
        return;
    }

    let weights = match config.training_strategy {
        CpuTrainingStrategy::Sweep => train_weights_sweep(&config),
        CpuTrainingStrategy::Genetic => train_weights(&config),
    };
    let json = weights.to_json();
    if let Some(path) = &config.out {
        std::fs::write(path, &json).expect("failed to write training output");
    }
    println!("{json}");
}

impl TrainerConfig {
    pub(crate) fn from_env(args: Vec<String>) -> Self {
        // The script-facing CLI is intentionally small, so hand parsing keeps the
        // training harness dependency-free.
        let seed = random_seed();
        let training = load_training_parameters();
        let mut config = Self {
            generations: usize::MAX,
            population: training.candidates.unwrap_or_else(auto_population),
            training_time_ms: training.time_ms,
            nodes: training.nodes,
            seed,
            max_seconds: None,
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
            compare_seeds: default_compare_seeds(seed),
            min_wins: 0,
            min_total_delta: 0,
            verify: "cargo test -q".to_string(),
            ai_src: "engine/models/cpu-v1/parameters.json".to_string(),
            hall_of_fame: default_hall_of_fame_path(),
            opponent_variants: training.opponent_variants,
            screening_opponent_variants: training.screening_opponent_variants,
            rounds_per_variant: training.rounds_per_variant,
            hall_of_fame_entries: training.hall_of_fame_entries,
            league_contenders: training.league_contenders,
            league_hall_of_fame_entries: training.league_hall_of_fame_entries,
            min_pairs: training.min_pairs,
            pair_batch: training
                .pair_batch
                .unwrap_or_else(|| host_parallelism().max(1)),
            max_pairs: training.max_pairs,
            draw_window: training.draw_window,
            draw_rate_limit: training.draw_rate_limit,
            max_match_plies: training.max_match_plies,
            max_match_time_ms: training.max_match_time_ms,
            max_generations_without_candidate: training.max_generations_without_candidate,
            finalist_count: training.finalists.unwrap_or_else(auto_finalists),
            search_strategy: TrainingSearchStrategy::AlphaBeta,
            sweep_parameter_groups: vec![SweepParameterGroup::ClassicBasic],
            sweep_points: 5,
            sweep_passes: Some(2),
            sweep_range_low: 1.0 / 3.0,
            sweep_range_high: 5.0 / 3.0,
            sweep_shrink: 0.5,
        };
        let mut index = 0;
        let mut compare_seeds_overridden = false;
        while index < args.len() {
            let value = args.get(index + 1).cloned();
            match args[index].as_str() {
                "--train-cycle" => {
                    config.train_cycle = true;
                    index += 1;
                }
                "--generations" => {
                    config.generations = parse_arg(value, config.generations);
                    index += 2;
                }
                "--population" => {
                    config.population = parse_arg(value, config.population);
                    index += 2;
                }
                "--strategy" | "--training-strategy" => {
                    if let Some(strategy) = value {
                        config.training_strategy = CpuTrainingStrategy::parse(&strategy)
                            .unwrap_or_else(|message| panic!("{message}"));
                    }
                    index += 2;
                }
                "--parameter-groups" | "--sweep-groups" => {
                    if let Some(groups) = value {
                        config.sweep_parameter_groups = SweepParameterGroup::parse_list(&groups)
                            .unwrap_or_else(|message| panic!("{message}"));
                    }
                    index += 2;
                }
                "--sweep-points" => {
                    config.sweep_points = parse_arg(value, config.sweep_points);
                    index += 2;
                }
                "--sweep-passes" => {
                    config.sweep_passes = value.and_then(|raw| raw.parse().ok());
                    index += 2;
                }
                "--sweep-range" => {
                    if let Some(range) = value {
                        let (low, high) =
                            parse_sweep_range(&range).unwrap_or_else(|message| panic!("{message}"));
                        config.sweep_range_low = low;
                        config.sweep_range_high = high;
                    }
                    index += 2;
                }
                "--sweep-shrink" => {
                    config.sweep_shrink = parse_arg(value, config.sweep_shrink);
                    index += 2;
                }
                "--config" | "--effort" => {
                    // Training parameters are global. Consume the legacy effort
                    // selector without changing the loaded training config.
                    index += 2;
                }
                "--depth" => {
                    // Training search is now time bounded. Keep consuming the
                    // retired flag so older scripts do not skew later args.
                    index += 2;
                }
                "--training-time-ms" | "--turn-time-ms" => {
                    config.training_time_ms = parse_arg(value, config.training_time_ms);
                    index += 2;
                }
                "--nodes" => {
                    config.nodes = parse_arg(value, config.nodes);
                    index += 2;
                }
                "--search-strategy" => {
                    if let Some(strategy) = value {
                        config.search_strategy = TrainingSearchStrategy::parse(&strategy)
                            .unwrap_or_else(|message| panic!("{message}"));
                    }
                    index += 2;
                }
                "--plies" => {
                    // Full-match training no longer uses plies, but keep consuming
                    // the flag so older local scripts do not skew later args.
                    index += 2;
                }
                "--seed" => {
                    config.seed = parse_arg(value, config.seed);
                    index += 2;
                }
                "--max-seconds" | "--time-seconds" | "--time-budget" => {
                    config.max_seconds = value.and_then(|raw| raw.parse().ok());
                    index += 2;
                }
                "--out" => {
                    config.out = value;
                    index += 2;
                }
                "--score" => {
                    config.score = value;
                    index += 2;
                }
                "--score-default" => {
                    config.score_default = true;
                    index += 1;
                }
                "--gpu-backend-info" => {
                    config.gpu_backend_info = true;
                    index += 1;
                }
                "--gpu-compile-shaders" => {
                    config.gpu_compile_shaders = true;
                    index += 1;
                }
                "--gpu-compile-kernels" => {
                    config.gpu_compile_kernels = true;
                    index += 1;
                }
                "--gpu-dispatch-smoke" => {
                    config.gpu_dispatch_smoke = true;
                    index += 1;
                }
                "--gpu-training-dispatch-smoke" => {
                    config.gpu_training_dispatch_smoke = true;
                    index += 1;
                }
                "--gpu-shader-info" => {
                    config.gpu_shader_info = true;
                    index += 1;
                }
                "--gpu-model-info" => {
                    if matches!(value.as_deref(), Some(next) if !next.starts_with("--")) {
                        config.gpu_model_info = value;
                        index += 2;
                    } else {
                        config.gpu_model_info =
                            Some(crate::gpu::training::DEFAULT_VALUE_MODEL_PATH.to_string());
                        index += 1;
                    }
                }
                "--gpu-model-probe-zero" => {
                    if matches!(value.as_deref(), Some(next) if !next.starts_with("--")) {
                        config.gpu_model_probe_zero = value;
                        index += 2;
                    } else {
                        config.gpu_model_probe_zero =
                            Some(crate::gpu::training::DEFAULT_VALUE_MODEL_PATH.to_string());
                        index += 1;
                    }
                }
                "--gpu-model" | "--gpu-value-model" => {
                    if let Some(model_path) = value {
                        config.gpu_value_model_path = model_path;
                    }
                    index += 2;
                }
                "--gpu-project-samples" => {
                    if let Some(samples_path) = value {
                        config.gpu_project_samples = Some(samples_path);
                    }
                    index += 2;
                }
                "--gpu-predict-samples" => {
                    if let Some(samples_path) = value {
                        config.gpu_predict_samples = Some(samples_path);
                    }
                    index += 2;
                }
                "--gpu-distill-samples" => {
                    if let Some(samples_path) = value {
                        config.gpu_distill_samples = Some(samples_path);
                    }
                    index += 2;
                }
                "--gpu-replay-buffer" => {
                    if let Some(samples_path) = value {
                        config.gpu_replay_buffer = Some(samples_path);
                    }
                    index += 2;
                }
                "--gpu-replay-append" | "--gpu-append-replay-samples" => {
                    if let Some(samples_path) = value {
                        config.gpu_replay_append = Some(samples_path);
                    }
                    index += 2;
                }
                "--gpu-replay-max" => {
                    config.gpu_replay_max = parse_arg(value, config.gpu_replay_max);
                    index += 2;
                }
                "--gpu-search" => {
                    if matches!(value.as_deref(), Some(next) if !next.starts_with("--")) {
                        config.gpu_search_snapshot = value;
                        index += 2;
                    } else {
                        config.gpu_search_snapshot = Some(String::new());
                        index += 1;
                    }
                }
                "--gpu-search-depth" => {
                    config.gpu_search_depth = value.and_then(|raw| raw.parse().ok());
                    index += 2;
                }
                "--gpu-search-min-depth" => {
                    config.gpu_search_min_depth = value.and_then(|raw| raw.parse().ok());
                    index += 2;
                }
                "--gpu-train-samples" => {
                    if let Some(samples_path) = value {
                        config.gpu_train_samples = Some(samples_path);
                    }
                    index += 2;
                }
                "--gpu-train-projected-samples" => {
                    if let Some(samples_path) = value {
                        config.gpu_train_projected_samples = Some(samples_path);
                    }
                    index += 2;
                }
                "--gpu-train-search" => {
                    if matches!(value.as_deref(), Some(next) if !next.starts_with("--")) {
                        config.gpu_train_search_snapshot = value;
                        index += 2;
                    } else {
                        config.gpu_train_search_snapshot = Some(String::new());
                        index += 1;
                    }
                }
                "--gpu-sample-search" => {
                    if matches!(value.as_deref(), Some(next) if !next.starts_with("--")) {
                        config.gpu_sample_search_snapshot = value;
                        index += 2;
                    } else {
                        config.gpu_sample_search_snapshot = Some(String::new());
                        index += 1;
                    }
                }
                "--gpu-sample-count" | "--sample-count" => {
                    config.gpu_sample_count = parse_arg(value, config.gpu_sample_count);
                    index += 2;
                }
                "--gpu-sample-mode" | "--sample-mode" => {
                    if let Some(mode) = value {
                        config.gpu_sample_mode =
                            crate::gpu::training::SearchLabelMode::parse(&mode)
                                .unwrap_or_else(|message| panic!("{message}"));
                    }
                    index += 2;
                }
                "--gpu-sample-plies" | "--sample-plies" => {
                    config.gpu_sample_max_plies = parse_arg(value, config.gpu_sample_max_plies);
                    index += 2;
                }
                "--cpu-search" => {
                    if matches!(value.as_deref(), Some(next) if !next.starts_with("--")) {
                        config.cpu_search_snapshot = value;
                        index += 2;
                    } else {
                        config.cpu_search_snapshot = Some(String::new());
                        index += 1;
                    }
                }
                "--compare-seeds" => {
                    config.compare_seeds =
                        parse_seed_list(value.as_deref()).unwrap_or(config.compare_seeds);
                    compare_seeds_overridden = true;
                    index += 2;
                }
                "--min-wins" => {
                    config.min_wins = parse_arg(value, config.min_wins);
                    index += 2;
                }
                "--min-total-delta" => {
                    config.min_total_delta = parse_arg(value, config.min_total_delta);
                    index += 2;
                }
                "--verify" => {
                    config.verify = value.unwrap_or(config.verify);
                    index += 2;
                }
                "--ai-src" => {
                    config.ai_src = value.unwrap_or(config.ai_src);
                    index += 2;
                }
                "--hall-of-fame" => {
                    config.hall_of_fame = value.unwrap_or(config.hall_of_fame);
                    index += 2;
                }
                "--opponent-variants" => {
                    config.opponent_variants = parse_arg(value, config.opponent_variants);
                    index += 2;
                }
                "--screening-opponent-variants" => {
                    config.screening_opponent_variants =
                        parse_arg(value, config.screening_opponent_variants);
                    index += 2;
                }
                "--rounds-per-variant" => {
                    config.rounds_per_variant = parse_arg(value, config.rounds_per_variant);
                    index += 2;
                }
                "--hall-of-fame-entries" => {
                    config.hall_of_fame_entries = parse_arg(value, config.hall_of_fame_entries);
                    index += 2;
                }
                "--league-contenders" => {
                    config.league_contenders = parse_arg(value, config.league_contenders);
                    index += 2;
                }
                "--league-hall-of-fame-entries" => {
                    config.league_hall_of_fame_entries =
                        parse_arg(value, config.league_hall_of_fame_entries);
                    index += 2;
                }
                "--min-pairs" => {
                    config.min_pairs = parse_arg(value, config.min_pairs);
                    index += 2;
                }
                "--pair-batch" => {
                    config.pair_batch = parse_arg(value, config.pair_batch);
                    index += 2;
                }
                "--max-pairs" => {
                    config.max_pairs = parse_arg(value, config.max_pairs);
                    index += 2;
                }
                "--draw-window" => {
                    config.draw_window = parse_arg(value, config.draw_window);
                    index += 2;
                }
                "--draw-rate-limit" => {
                    config.draw_rate_limit = parse_arg(value, config.draw_rate_limit);
                    index += 2;
                }
                "--max-match-plies" | "--match-plies" => {
                    config.max_match_plies = parse_arg(value, config.max_match_plies);
                    index += 2;
                }
                "--max-match-ms" | "--match-ms" => {
                    config.max_match_time_ms = parse_arg(value, config.max_match_time_ms);
                    index += 2;
                }
                "--max-generations-without-candidate" => {
                    config.max_generations_without_candidate =
                        parse_arg(value, config.max_generations_without_candidate);
                    index += 2;
                }
                "--finalists" => {
                    config.finalist_count = parse_arg(value, config.finalist_count);
                    index += 2;
                }
                _ => index += 1,
            }
        }
        config.population = config.population.max(4);
        config.training_time_ms = config.training_time_ms.max(1);
        config.nodes = config.nodes.max(1);
        config.gpu_sample_count = config.gpu_sample_count.max(1);
        config.pair_batch = config.pair_batch.max(1);
        config.opponent_variants = config.opponent_variants.max(1);
        config.screening_opponent_variants = config
            .screening_opponent_variants
            .clamp(1, config.opponent_variants);
        config.rounds_per_variant = config.rounds_per_variant.max(1);
        config.hall_of_fame_entries = config.hall_of_fame_entries.max(1);
        config.league_contenders = config.league_contenders.max(1);
        config.league_hall_of_fame_entries = config.league_hall_of_fame_entries.max(1);
        config.min_pairs = config.min_pairs.max(1);
        config.max_pairs = config.max_pairs.max(config.min_pairs);
        config.draw_window = config.draw_window.max(1);
        config.draw_rate_limit = config.draw_rate_limit.clamp(0.0, 1.0);
        config.max_match_plies = config.max_match_plies.max(1);
        config.max_generations_without_candidate = config.max_generations_without_candidate.max(1);
        config.finalist_count = config.finalist_count.clamp(2, config.population);
        config.sweep_points = config.sweep_points.max(3);
        if config.sweep_range_low <= 0.0 || config.sweep_range_high <= config.sweep_range_low {
            config.sweep_range_low = 1.0 / 3.0;
            config.sweep_range_high = 5.0 / 3.0;
        }
        config.sweep_shrink = config.sweep_shrink.clamp(0.01, 0.99);
        if !compare_seeds_overridden {
            config.compare_seeds = default_compare_seeds(config.seed);
        }
        if config.min_wins == 0 {
            config.min_wins = config.compare_seeds.len() * 2 / 3 + 1;
        }
        if config.min_total_delta == 0 {
            config.min_total_delta = (config.compare_seeds.len() as i32) * 50;
        }
        config
    }

    pub(crate) fn with_search(&self, nodes: usize, training_time_ms: u64) -> Self {
        let mut config = self.clone();
        config.nodes = nodes;
        config.training_time_ms = training_time_ms;
        config
    }

    pub(crate) fn screening_search(&self) -> Self {
        let mut config = self.clone();
        config.nodes = (self.nodes / 4).max(20).min(self.nodes);
        config.training_time_ms = (self.training_time_ms / 4)
            .max(1)
            .min(self.training_time_ms);
        config
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_backend_info() -> String {
    crate::gpu::native::backend_info()
        .map(|info| info.to_string())
        .unwrap_or_else(|message| format!("native_gpu error={message}"))
}

#[cfg(any(target_arch = "wasm32", not(feature = "neural-wgpu")))]
fn native_gpu_backend_info() -> String {
    "native_gpu unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_compile_shaders() -> String {
    crate::gpu::native::compile_engine_shaders()
        .map(|report| report.to_string())
        .unwrap_or_else(|message| format!("native_gpu_shader_compile error={message}"))
}

#[cfg(any(target_arch = "wasm32", not(feature = "neural-wgpu")))]
fn native_gpu_compile_shaders() -> String {
    "native_gpu_shader_compile unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_compile_kernels() -> String {
    crate::gpu::native::compile_engine_kernels()
        .map(|report| report.to_string())
        .unwrap_or_else(|message| format!("native_gpu_kernel_compile error={message}"))
}

#[cfg(any(target_arch = "wasm32", not(feature = "neural-wgpu")))]
fn native_gpu_compile_kernels() -> String {
    "native_gpu_kernel_compile unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_dispatch_smoke() -> String {
    crate::gpu::native::dispatch_search_smoke()
        .map(|report| report.to_string())
        .unwrap_or_else(|message| format!("native_gpu_dispatch error={message}"))
}

#[cfg(any(target_arch = "wasm32", not(feature = "neural-wgpu")))]
fn native_gpu_dispatch_smoke() -> String {
    "native_gpu_dispatch unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_training_dispatch_smoke() -> String {
    crate::gpu::native::dispatch_project_features_smoke()
        .map(|report| report.to_string())
        .unwrap_or_else(|message| format!("native_gpu_training_dispatch error={message}"))
}

#[cfg(any(target_arch = "wasm32", not(feature = "neural-wgpu")))]
fn native_gpu_training_dispatch_smoke() -> String {
    "native_gpu_training_dispatch unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_project_samples(path: &str, config: &TrainerConfig) -> String {
    let samples = crate::gpu::training::load_training_samples_json(path)
        .unwrap_or_else(|message| panic!("{message}"));
    let model = load_gpu_value_model(config);
    let projected = crate::gpu::native::project_features_batch(
        crate::gpu::native::NativeProjectFeaturesBatchRequest {
            projection_size: model.projection_size as usize,
            seed: model.projection_seed,
            output_offset: 0,
            features: samples
                .iter()
                .map(|sample| sample.features.clone())
                .collect::<Vec<_>>(),
        },
    )
    .unwrap_or_else(|message| panic!("{message}"));
    let response = ProjectedSamplesJson {
        sample_count: samples.len(),
        projection_size: model.projection_size as usize,
        seed: model.projection_seed,
        features: projected,
    };
    let json = serde_json::to_string_pretty(&response)
        .unwrap_or_else(|error| panic!("failed to encode projected samples: {error}"));
    if let Some(out) = &config.out {
        std::fs::write(out, &json)
            .unwrap_or_else(|error| panic!("failed to write projected samples {out}: {error}"));
    }
    let summary = format!(
        "gpu_project_samples samples={} projection_size={} seed={} values={} wrote={}",
        response.sample_count,
        response.projection_size,
        response.seed,
        response.features.len(),
        config.out.as_deref().unwrap_or("")
    );
    if config.out.is_some() {
        summary
    } else {
        format!("{summary}\n{json}")
    }
}

#[cfg(any(target_arch = "wasm32", not(feature = "neural-wgpu")))]
fn native_gpu_project_samples(_path: &str, _config: &TrainerConfig) -> String {
    "native_gpu_project_samples unavailable=engine built without neural-wgpu feature".to_string()
}

fn gpu_distill_samples(path: &str, config: &TrainerConfig) -> String {
    let samples = crate::gpu::training::load_training_samples_json(path)
        .unwrap_or_else(|message| panic!("{message}"));
    let model = load_gpu_value_model(config);
    let distilled = crate::gpu::training::distill_training_samples(&samples, &model);
    if let Some(out) = &config.out {
        crate::gpu::training::save_training_samples_json(out, &distilled)
            .unwrap_or_else(|message| panic!("{message}"));
    }
    let summary = format!(
        "gpu_distill_samples samples={} distilled={} source_model={} wrote={}",
        samples.len(),
        distilled.len(),
        gpu_value_model_path(config),
        config.out.as_deref().unwrap_or("")
    );
    if config.out.is_some() {
        summary
    } else {
        let json = serde_json::to_string_pretty(&distilled)
            .unwrap_or_else(|error| panic!("failed to encode distilled samples: {error}"));
        format!("{summary}\n{json}")
    }
}

fn gpu_append_replay_samples(config: &TrainerConfig) -> String {
    let append_path = config
        .gpu_replay_append
        .as_deref()
        .expect("GPU replay append path is required");
    let buffer = match config.gpu_replay_buffer.as_deref() {
        Some(path) => crate::gpu::training::load_training_samples_json(path)
            .unwrap_or_else(|message| panic!("{message}")),
        None => Vec::new(),
    };
    let samples = crate::gpu::training::load_training_samples_json(append_path)
        .unwrap_or_else(|message| panic!("{message}"));
    let max_buffer = config.gpu_replay_max.max(1);
    let retained = crate::gpu::training::append_replay_samples(&buffer, &samples, max_buffer);
    if let Some(out) = &config.out {
        crate::gpu::training::save_training_samples_json(out, &retained)
            .unwrap_or_else(|message| panic!("{message}"));
    }
    let summary = format!(
        "gpu_replay_append buffer={} appended={} retained={} max={} wrote={}",
        buffer.len(),
        samples.len(),
        retained.len(),
        max_buffer,
        config.out.as_deref().unwrap_or("")
    );
    if config.out.is_some() {
        summary
    } else {
        let json = serde_json::to_string_pretty(&retained)
            .unwrap_or_else(|error| panic!("failed to encode replay samples: {error}"));
        format!("{summary}\n{json}")
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_predict_samples(path: &str, config: &TrainerConfig) -> String {
    let samples = crate::gpu::training::load_training_samples_json(path)
        .unwrap_or_else(|message| panic!("{message}"));
    let model = load_gpu_value_model(config);
    let predictions =
        crate::gpu::native::predict_values(crate::gpu::native::NativeValuePredictionRequest {
            model,
            features: samples
                .iter()
                .map(|sample| sample.features.clone())
                .collect::<Vec<_>>(),
        })
        .unwrap_or_else(|message| panic!("{message}"));
    let response = PredictedSamplesJson {
        sample_count: samples.len(),
        predictions,
    };
    let json = serde_json::to_string_pretty(&response)
        .unwrap_or_else(|error| panic!("failed to encode predicted samples: {error}"));
    if let Some(out) = &config.out {
        std::fs::write(out, &json)
            .unwrap_or_else(|error| panic!("failed to write predicted samples {out}: {error}"));
    }
    let summary = format!(
        "gpu_predict_samples samples={} predictions={} wrote={}",
        response.sample_count,
        response.predictions.len(),
        config.out.as_deref().unwrap_or("")
    );
    if config.out.is_some() {
        summary
    } else {
        format!("{summary}\n{json}")
    }
}

#[cfg(any(target_arch = "wasm32", not(feature = "neural-wgpu")))]
fn native_gpu_predict_samples(_path: &str, _config: &TrainerConfig) -> String {
    "native_gpu_predict_samples unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_train_projected_samples(path: &str, config: &TrainerConfig) -> String {
    let samples = crate::gpu::training::load_training_samples_json(path)
        .unwrap_or_else(|message| panic!("{message}"));
    let model = load_gpu_value_model(config);
    let projected = crate::gpu::native::project_features_batch(
        crate::gpu::native::NativeProjectFeaturesBatchRequest {
            projection_size: model.projection_size as usize,
            seed: model.projection_seed,
            output_offset: 0,
            features: samples
                .iter()
                .map(|sample| sample.features.clone())
                .collect::<Vec<_>>(),
        },
    )
    .unwrap_or_else(|message| panic!("{message}"));
    let (value_report, policy_report, _wrote) =
        train_native_gpu_value_model_from_projected(&samples, &projected, model, config);
    format!(
        "gpu_train_projected_samples samples={} projected_values={} value_epochs={} value_initial_loss={} value_final_loss={} policy_samples={} policy_steps={} policy_initial_loss={} policy_final_loss={} wrote={}",
        value_report.samples,
        projected.len(),
        value_report.epochs,
        value_report.initial_loss,
        value_report.final_loss,
        policy_report.samples,
        policy_report.steps,
        policy_report.initial_loss,
        policy_report.final_loss,
        config.out.as_deref().unwrap_or("")
    )
}

#[cfg(any(target_arch = "wasm32", not(feature = "neural-wgpu")))]
fn native_gpu_train_projected_samples(_path: &str, _config: &TrainerConfig) -> String {
    "native_gpu_train_projected_samples unavailable=engine built without neural-wgpu feature"
        .to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ProjectedSamplesJson {
    sample_count: usize,
    projection_size: usize,
    seed: u32,
    features: Vec<f32>,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct PredictedSamplesJson {
    sample_count: usize,
    predictions: Vec<f32>,
}

fn gpu_shader_info() -> String {
    crate::gpu::search::KERNELS
        .iter()
        .chain(crate::gpu::training::KERNELS.iter())
        .map(|kernel| {
            let constants = if kernel.constants.is_empty() {
                String::new()
            } else {
                format!(
                    " constants={}",
                    kernel
                        .constants
                        .iter()
                        .map(|(name, value)| format!("{name}:{value}"))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            };
            format!(
                "gpu_kernel set={} label={} shader={} entry={}{}",
                kernel.set.as_str(),
                kernel.label,
                kernel.shader,
                kernel.entry_point,
                constants
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn gpu_search_request(config: &TrainerConfig) -> crate::gpu::search::GpuSearchRequest {
    let snapshot_json = config.gpu_search_snapshot.as_ref().and_then(|path| {
        if path.is_empty() {
            None
        } else {
            Some(std::fs::read_to_string(path).unwrap_or_else(|error| {
                panic!("failed to read GPU search snapshot {path}: {error}")
            }))
        }
    });
    crate::gpu::search::GpuSearchRequest {
        snapshot_json,
        model_path: Some(gpu_value_model_path(config).to_string()),
        depth: config
            .gpu_search_depth
            .unwrap_or(crate::gpu::search::DEFAULT_GPU_SEARCH_DEPTH),
        min_depth: Some(
            config
                .gpu_search_min_depth
                .unwrap_or(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
        ),
        nodes: config.nodes.max(1).min(i32::MAX as usize) as i32,
        time_ms: config.training_time_ms.max(1).min(i32::MAX as u64) as i32,
    }
}

fn gpu_search_label_batch_request(
    config: &TrainerConfig,
) -> crate::gpu::training::SearchLabelBatchRequest {
    let snapshot_json = config.gpu_sample_search_snapshot.as_ref().and_then(|path| {
        if path.is_empty() {
            None
        } else {
            Some(std::fs::read_to_string(path).unwrap_or_else(|error| {
                panic!("failed to read GPU sample search snapshot {path}: {error}")
            }))
        }
    });
    crate::gpu::training::SearchLabelBatchRequest {
        snapshot_json,
        mode: config.gpu_sample_mode,
        distill_model: gpu_sample_distill_model(config),
        count: config.gpu_sample_count,
        max_plies: config.gpu_sample_max_plies,
        position_depth: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_POSITION_DEPTH,
        position_nodes: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_POSITION_NODES,
        position_time_ms: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_POSITION_TIME_MS,
        label_depth: crate::cpu::search::DEFAULT_CPU_SEARCH_DEPTH,
        label_min_depth: Some(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
        label_nodes: config.nodes.max(1).min(i32::MAX as usize) as i32,
        label_time_ms: config.training_time_ms.max(1).min(i32::MAX as u64) as i32,
        label_weight: 1.0,
    }
}

fn gpu_train_search_label_batch_request(
    config: &TrainerConfig,
) -> crate::gpu::training::SearchLabelBatchRequest {
    let snapshot_json = config.gpu_train_search_snapshot.as_ref().and_then(|path| {
        if path.is_empty() {
            None
        } else {
            Some(std::fs::read_to_string(path).unwrap_or_else(|error| {
                panic!("failed to read GPU train search snapshot {path}: {error}")
            }))
        }
    });
    crate::gpu::training::SearchLabelBatchRequest {
        snapshot_json,
        mode: config.gpu_sample_mode,
        distill_model: gpu_sample_distill_model(config),
        count: config.gpu_sample_count,
        max_plies: config.gpu_sample_max_plies,
        position_depth: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_POSITION_DEPTH,
        position_nodes: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_POSITION_NODES,
        position_time_ms: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_POSITION_TIME_MS,
        label_depth: crate::cpu::search::DEFAULT_CPU_SEARCH_DEPTH,
        label_min_depth: Some(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
        label_nodes: config.nodes.max(1).min(i32::MAX as usize) as i32,
        label_time_ms: config.training_time_ms.max(1).min(i32::MAX as u64) as i32,
        label_weight: 1.0,
    }
}

fn gpu_sample_distill_model(
    config: &TrainerConfig,
) -> Option<crate::gpu::training::CompactValueModel> {
    (config.gpu_sample_mode == crate::gpu::training::SearchLabelMode::Distilled)
        .then(|| load_gpu_value_model(config))
}

fn gpu_value_model_path(config: &TrainerConfig) -> &str {
    &config.gpu_value_model_path
}

fn load_gpu_value_model(config: &TrainerConfig) -> crate::gpu::training::CompactValueModel {
    crate::gpu::training::load_compact_value_model(gpu_value_model_path(config))
        .unwrap_or_else(|message| panic!("{message}"))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn train_gpu_value_model_from_samples(
    samples: &[crate::gpu::training::TrainingSample],
    config: &TrainerConfig,
) -> (
    crate::gpu::training::ValueHeadTrainingReport,
    crate::gpu::training::PolicyHeadTrainingReport,
    String,
) {
    let samples = crate::gpu::training::dedupe_training_samples(samples);
    let model = load_gpu_value_model(config);
    let working_set = crate::gpu::training::select_training_working_set_for_projection(
        &samples,
        model.projection_size as usize,
        crate::gpu::training::DEFAULT_PROJECTED_WORKING_SET_BYTES,
    );
    let projected = crate::gpu::native::project_features_batch(
        crate::gpu::native::NativeProjectFeaturesBatchRequest {
            projection_size: model.projection_size as usize,
            seed: model.projection_seed,
            output_offset: 0,
            features: working_set
                .iter()
                .map(|sample| sample.features.clone())
                .collect::<Vec<_>>(),
        },
    )
    .unwrap_or_else(|message| panic!("{message}"));
    train_native_gpu_value_model_from_projected(&working_set, &projected, model, config)
}

#[cfg(any(target_arch = "wasm32", not(feature = "neural-wgpu")))]
fn train_gpu_value_model_from_samples(
    samples: &[crate::gpu::training::TrainingSample],
    config: &TrainerConfig,
) -> (
    crate::gpu::training::ValueHeadTrainingReport,
    crate::gpu::training::PolicyHeadTrainingReport,
    String,
) {
    let samples = crate::gpu::training::dedupe_training_samples(samples);
    let model = load_gpu_value_model(config);
    let (value_trained, value_report) = crate::gpu::training::train_value_head_cpu(
        &model,
        &samples,
        crate::gpu::training::ValueHeadTrainingConfig::default(),
    )
    .unwrap_or_else(|message| panic!("{message}"));
    let (trained, policy_report) = crate::gpu::training::train_policy_head_cpu(
        &value_trained,
        &samples,
        crate::gpu::training::ValueHeadTrainingConfig::default(),
    )
    .unwrap_or_else(|message| panic!("{message}"));
    if let Some(out) = &config.out {
        crate::gpu::training::save_compact_value_model(out, &trained)
            .unwrap_or_else(|message| panic!("{message}"));
    }
    (
        value_report,
        policy_report,
        config.out.as_deref().unwrap_or("").to_string(),
    )
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn train_native_gpu_value_model_from_projected(
    samples: &[crate::gpu::training::TrainingSample],
    projected: &[f32],
    model: crate::gpu::training::CompactValueModel,
    config: &TrainerConfig,
) -> (
    crate::gpu::training::ValueHeadTrainingReport,
    crate::gpu::training::PolicyHeadTrainingReport,
    String,
) {
    let (value_trained, value_report) =
        crate::gpu::native::train_value_head(crate::gpu::native::NativeValueHeadTrainingRequest {
            model,
            samples: samples.to_vec(),
            projected_features: projected.to_vec(),
            config: crate::gpu::training::ValueHeadTrainingConfig::default(),
            train_hidden_layers: true,
        })
        .unwrap_or_else(|message| panic!("{message}"));
    let (trained, policy_report) = crate::gpu::native::train_policy_head(
        crate::gpu::native::NativePolicyHeadTrainingRequest {
            model: value_trained,
            samples: samples.to_vec(),
            projected_features: projected.to_vec(),
            config: crate::gpu::training::ValueHeadTrainingConfig::default(),
        },
    )
    .unwrap_or_else(|message| panic!("{message}"));
    if let Some(out) = &config.out {
        crate::gpu::training::save_compact_value_model(out, &trained)
            .unwrap_or_else(|message| panic!("{message}"));
    }
    (
        value_report,
        policy_report,
        config.out.as_deref().unwrap_or("").to_string(),
    )
}

fn cpu_search_request(config: &TrainerConfig) -> crate::cpu::search::CpuSearchRequest {
    let snapshot_json = config.cpu_search_snapshot.as_ref().and_then(|path| {
        if path.is_empty() {
            None
        } else {
            Some(std::fs::read_to_string(path).unwrap_or_else(|error| {
                panic!("failed to read CPU search snapshot {path}: {error}")
            }))
        }
    });
    crate::cpu::search::CpuSearchRequest {
        snapshot_json,
        parameters_json: std::fs::read_to_string(&config.ai_src).ok(),
        depth: crate::cpu::search::DEFAULT_CPU_SEARCH_DEPTH,
        min_depth: Some(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
        nodes: config.nodes.max(1).min(i32::MAX as usize) as i32,
        time_ms: config.training_time_ms.max(1).min(i32::MAX as u64) as i32,
    }
}

fn parse_sweep_range(value: &str) -> Result<(f64, f64), String> {
    let Some((low, high)) = value.split_once(':') else {
        return Err("sweep range must use LOW:HIGH".to_string());
    };
    let low = low
        .parse::<f64>()
        .map_err(|_| format!("invalid sweep range low value `{low}`"))?;
    let high = high
        .parse::<f64>()
        .map_err(|_| format!("invalid sweep range high value `{high}`"))?;
    if low <= 0.0 || high <= low {
        return Err("sweep range must satisfy 0 < LOW < HIGH".to_string());
    }
    Ok((low, high))
}
