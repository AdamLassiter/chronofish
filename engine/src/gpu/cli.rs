use clap::{Args, Parser};

/// Options shared by native CPU and GPU training commands.
#[derive(Args, Clone, Debug, Default)]
pub(crate) struct CommonCliArgs {
    /// Search node budget.
    #[arg(long)]
    pub(crate) nodes: Option<usize>,
    /// Per-search time budget in milliseconds.
    #[arg(long = "training-time-ms", visible_alias = "turn-time-ms")]
    pub(crate) training_time_ms: Option<u64>,
    /// Maximum iterative-deepening depth used by search and training labels.
    #[arg(
        long = "search-depth",
        visible_aliases = ["training-search-depth", "depth"]
    )]
    pub(crate) search_depth: Option<i32>,
    /// Minimum completed depth accepted from a training search.
    #[arg(long = "search-min-depth", visible_alias = "training-search-min-depth")]
    pub(crate) search_min_depth: Option<i32>,
    /// Write generated samples or the trained model to this path.
    #[arg(long)]
    pub(crate) out: Option<String>,
    /// Progress renderer. `auto` uses the TUI only when stdin and stdout are terminals.
    #[arg(long, default_value = "auto", value_parser = parse_ui_mode)]
    pub(crate) ui: crate::training_runtime::UiMode,
    /// Wall-clock limit for the complete training run. Paused time is included.
    #[arg(long = "max-seconds", visible_aliases = ["time-seconds", "time-budget"])]
    pub(crate) max_seconds: Option<u64>,
    /// Durable artifact written whenever a new best candidate is discovered.
    #[arg(long)]
    pub(crate) candidate_out: Option<String>,
    /// Append-only JSONL audit journal for candidate and promotion events.
    #[arg(long)]
    pub(crate) improvement_log: Option<String>,
}

fn parse_ui_mode(value: &str) -> Result<crate::training_runtime::UiMode, String> {
    crate::training_runtime::UiMode::parse(value)
}

/// GPU-only native search, sample collection, and model-training options.
#[derive(Args, Clone, Debug, Default)]
pub(crate) struct GpuCliArgs {
    #[command(flatten)]
    pub(crate) common: CommonCliArgs,
    /// Print the native WGPU adapter and backend information.
    #[arg(long)]
    pub(crate) backend_info: bool,
    /// Compile all engine GPU shaders and report the result.
    #[arg(long)]
    pub(crate) compile_shaders: bool,
    /// Compile the GPU training kernels and report the result.
    #[arg(long)]
    pub(crate) compile_kernels: bool,
    /// Dispatch a minimal GPU search to validate the native search path.
    #[arg(long)]
    pub(crate) dispatch_smoke: bool,
    /// Dispatch a minimal GPU feature-projection workload.
    #[arg(long)]
    pub(crate) training_dispatch_smoke: bool,
    /// List the GPU shader kernels used by search and training.
    #[arg(long)]
    pub(crate) shader_info: bool,
    /// CFNN model path used for inference or as the training starting point.
    #[arg(long = "gpu-model", visible_alias = "gpu-value-model")]
    pub(crate) value_model_path: Option<String>,
    /// Print a model summary; omit the value to use the default model path.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    pub(crate) model_info: Option<String>,
    /// Evaluate the model with an all-zero feature vector; omit the value for the default model.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    pub(crate) model_probe_zero: Option<String>,
    /// Project features from a training-sample JSON file on the GPU.
    #[arg(long)]
    pub(crate) project_samples: Option<String>,
    /// Predict values for a training-sample JSON file on the GPU.
    #[arg(long)]
    pub(crate) predict_samples: Option<String>,
    /// Distill a training-sample JSON file using the selected model.
    #[arg(long)]
    pub(crate) distill_samples: Option<String>,
    /// Existing replay-buffer training-sample JSON file.
    #[arg(long)]
    pub(crate) replay_buffer: Option<String>,
    /// Training-sample JSON file to append to the replay buffer.
    #[arg(long, visible_alias = "gpu-append-replay-samples")]
    pub(crate) replay_append: Option<String>,
    /// Maximum samples retained when appending a replay buffer.
    #[arg(long)]
    pub(crate) replay_max: Option<usize>,
    /// Search the initial position or the optional snapshot JSON file on the GPU.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    pub(crate) snapshot: Option<String>,
    /// Collect GPU-search labels from the initial position or an optional snapshot JSON file.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    pub(crate) sample_search: Option<String>,
    /// Number of labeled positions to collect.
    #[arg(long = "gpu-sample-count", visible_alias = "sample-count")]
    pub(crate) sample_count: Option<usize>,
    /// Label source: `search`, `cpu`, `duel`, or `distilled`.
    #[arg(long = "gpu-sample-mode", visible_alias = "sample-mode", value_parser = parse_search_label_mode)]
    pub(crate) sample_mode: Option<crate::gpu::training::SearchLabelMode>,
    /// Maximum self-play plies used while collecting labels.
    #[arg(long = "gpu-sample-plies", visible_alias = "sample-plies")]
    pub(crate) sample_max_plies: Option<usize>,
    /// Collect GPU-search labels and train from the initial position or an optional snapshot JSON file.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    pub(crate) train_search: Option<String>,
    /// Train the model from an existing training-sample JSON file.
    #[arg(long)]
    pub(crate) train_samples: Option<String>,
    /// Train from projected samples using native GPU feature projection.
    #[arg(long)]
    pub(crate) train_projected_samples: Option<String>,
    /// Learning rate for value and policy optimization.
    #[arg(long)]
    pub(crate) learning_rate: Option<f32>,
    /// Number of value-training epochs.
    #[arg(long)]
    pub(crate) epochs: Option<usize>,
    /// L2 weight-decay coefficient.
    #[arg(long)]
    pub(crate) weight_decay: Option<f32>,
    /// Momentum coefficient for value and policy optimization.
    #[arg(long)]
    pub(crate) momentum: Option<f32>,
    /// Fraction of stable position groups reserved for deterministic holdout validation.
    #[arg(long, default_value_t = 0.1)]
    pub(crate) validation_split: f32,
}

fn parse_search_label_mode(value: &str) -> Result<crate::gpu::training::SearchLabelMode, String> {
    crate::gpu::training::SearchLabelMode::parse(value)
}

#[derive(Clone)]
pub(crate) struct GpuCliConfig {
    pub(crate) nodes: usize,
    pub(crate) training_time_ms: u64,
    pub(crate) out: Option<String>,
    pub(crate) ui: crate::training_runtime::UiMode,
    pub(crate) max_seconds: Option<u64>,
    pub(crate) candidate_out: String,
    pub(crate) improvement_log: String,
    pub(crate) validation_split: f32,
    pub(crate) backend_info: bool,
    pub(crate) compile_shaders: bool,
    pub(crate) compile_kernels: bool,
    pub(crate) dispatch_smoke: bool,
    pub(crate) training_dispatch_smoke: bool,
    pub(crate) shader_info: bool,
    pub(crate) value_model_path: String,
    pub(crate) model_info: Option<String>,
    pub(crate) model_probe_zero: Option<String>,
    pub(crate) project_samples: Option<String>,
    pub(crate) predict_samples: Option<String>,
    pub(crate) distill_samples: Option<String>,
    pub(crate) replay_buffer: Option<String>,
    pub(crate) replay_append: Option<String>,
    pub(crate) replay_max: usize,
    pub(crate) search_snapshot: Option<String>,
    pub(crate) search_depth: Option<i32>,
    pub(crate) search_min_depth: Option<i32>,
    pub(crate) sample_search_snapshot: Option<String>,
    pub(crate) sample_mode: crate::gpu::training::SearchLabelMode,
    pub(crate) sample_count: usize,
    pub(crate) sample_max_plies: usize,
    pub(crate) train_search_snapshot: Option<String>,
    pub(crate) train_samples: Option<String>,
    pub(crate) train_projected_samples: Option<String>,
    pub(crate) learning_rate: f32,
    pub(crate) epochs: usize,
    pub(crate) weight_decay: f32,
    pub(crate) momentum: f32,
}

impl Default for GpuCliConfig {
    fn default() -> Self {
        let training = crate::gpu::training::ValueHeadTrainingConfig::default();
        Self {
            nodes: crate::gpu::search::DEFAULT_GPU_SEARCH_NODES as usize,
            training_time_ms: crate::gpu::search::DEFAULT_GPU_SEARCH_TIME_MS as u64,
            out: None,
            ui: crate::training_runtime::UiMode::Auto,
            max_seconds: None,
            candidate_out: "engine/models/gpu-v1/value-model.candidate.cfnn".to_string(),
            improvement_log: "engine/models/gpu-v1/value-model.improvements.jsonl".to_string(),
            validation_split: 0.1,
            backend_info: false,
            compile_shaders: false,
            compile_kernels: false,
            dispatch_smoke: false,
            training_dispatch_smoke: false,
            shader_info: false,
            value_model_path: crate::gpu::training::DEFAULT_VALUE_MODEL_PATH.to_string(),
            model_info: None,
            model_probe_zero: None,
            project_samples: None,
            predict_samples: None,
            distill_samples: None,
            replay_buffer: None,
            replay_append: None,
            replay_max: crate::gpu::training::MAX_GPU_TRAINING_SAMPLES,
            search_snapshot: None,
            search_depth: None,
            search_min_depth: None,
            sample_search_snapshot: None,
            sample_mode: crate::gpu::training::SearchLabelMode::Search,
            sample_count: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_COUNT,
            sample_max_plies: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_MAX_PLIES,
            train_search_snapshot: None,
            train_samples: None,
            train_projected_samples: None,
            learning_rate: training.learning_rate,
            epochs: training.epochs,
            weight_decay: training.weight_decay,
            momentum: training.momentum,
        }
    }
}

impl GpuCliConfig {
    pub(crate) fn from_args(args: GpuCliArgs) -> Self {
        let mut config = Self::default();
        if let Some(value) = args.common.nodes {
            config.nodes = value;
        }
        if let Some(value) = args.common.training_time_ms {
            config.training_time_ms = value;
        }
        if let Some(value) = args.common.out {
            config.out = Some(value);
        }
        config.ui = args.common.ui;
        config.max_seconds = args.common.max_seconds;
        if let Some(value) = args.common.candidate_out {
            config.candidate_out = value;
        }
        if let Some(value) = args.common.improvement_log {
            config.improvement_log = value;
        }
        config.backend_info = args.backend_info;
        config.compile_shaders = args.compile_shaders;
        config.compile_kernels = args.compile_kernels;
        config.dispatch_smoke = args.dispatch_smoke;
        config.training_dispatch_smoke = args.training_dispatch_smoke;
        config.shader_info = args.shader_info;
        if let Some(value) = args.value_model_path {
            config.value_model_path = value;
        }
        let default_model = crate::gpu::training::DEFAULT_VALUE_MODEL_PATH.to_string();
        config.model_info = args.model_info.map(|value| {
            if value.is_empty() {
                default_model.clone()
            } else {
                value
            }
        });
        config.model_probe_zero = args.model_probe_zero.map(|value| {
            if value.is_empty() {
                default_model
            } else {
                value
            }
        });
        config.project_samples = args.project_samples;
        config.predict_samples = args.predict_samples;
        config.distill_samples = args.distill_samples;
        config.replay_buffer = args.replay_buffer;
        config.replay_append = args.replay_append;
        if let Some(value) = args.replay_max {
            config.replay_max = value;
        }
        config.search_snapshot = args.snapshot;
        config.search_depth = args.common.search_depth;
        config.search_min_depth = args.common.search_min_depth;
        config.sample_search_snapshot = args.sample_search;
        if let Some(value) = args.sample_count {
            config.sample_count = value;
        }
        if let Some(value) = args.sample_mode {
            config.sample_mode = value;
        }
        if let Some(value) = args.sample_max_plies {
            config.sample_max_plies = value;
        }
        config.train_search_snapshot = args.train_search;
        config.train_samples = args.train_samples;
        config.train_projected_samples = args.train_projected_samples;
        if let Some(value) = args.learning_rate {
            config.learning_rate = value;
        }
        if let Some(value) = args.epochs {
            config.epochs = value;
        }
        if let Some(value) = args.weight_decay {
            config.weight_decay = value;
        }
        if let Some(value) = args.momentum {
            config.momentum = value;
        }
        config.validation_split = args.validation_split.clamp(0.0, 0.9);
        config.normalize();
        config
    }

    pub(crate) fn normalize(&mut self) {
        self.sample_count = self.sample_count.max(1);
        self.search_depth = self.search_depth.map(|depth| depth.max(1));
        let maximum = self
            .search_depth
            .unwrap_or(crate::gpu::search::DEFAULT_GPU_SEARCH_DEPTH);
        self.search_min_depth = self.search_min_depth.map(|depth| depth.clamp(1, maximum));
    }
}

#[derive(Parser)]
#[command(
    name = "train-gpu",
    about = "Run native GPU search, sampling, and value/policy training",
    after_help = "Training command: train --source search|samples|projected [INPUT]. Existing flat flags remain supported."
)]
struct GpuCli {
    #[command(flatten)]
    args: GpuCliArgs,
}

impl GpuCliConfig {
    #[cfg(test)]
    pub(crate) fn from_env(args: Vec<String>) -> Self {
        Self::from_args(
            GpuCli::try_parse_from(
                std::iter::once("train-gpu".to_string()).chain(normalize_gpu_command(args)),
            )
            .unwrap_or_else(|error| error.exit())
            .args,
        )
    }
}

pub fn run_gpu_cli() {
    let args = normalize_gpu_command(std::env::args().skip(1).collect());
    let config = GpuCliConfig::from_args(
        GpuCli::parse_from(std::iter::once("train-gpu".to_string()).chain(args)).args,
    );
    crate::training_runtime::set_global_ui_mode(config.ui);
    let training = config.train_search_snapshot.is_some()
        || config.train_samples.is_some()
        || config.train_projected_samples.is_some();
    if training {
        crate::training_runtime::set_cooperative_deadline(Some(
            std::time::Instant::now()
                + std::time::Duration::from_secs(config.max_seconds.unwrap_or(600).max(1)),
        ));
    }
    let handled = if training && config.ui.resolve() == crate::training_runtime::UiMode::Tui {
        crate::training_runtime::run_interactive(move || run(&config))
            .unwrap_or_else(|message| panic!("interactive GPU training failed: {message}"))
    } else {
        run(&config)
    };
    if !handled {
        eprintln!("No GPU command selected. Try --gpu-search, --gpu-sample-search, or --gpu-train-samples.");
    }
}

fn normalize_gpu_command(mut args: Vec<String>) -> Vec<String> {
    if args.first().is_some_and(|arg| arg == "train") {
        args.remove(0);
        let source_index = args.iter().position(|arg| arg == "--source");
        let source = source_index
            .and_then(|index| args.get(index + 1).cloned())
            .unwrap_or_else(|| "search".into());
        if let Some(index) = source_index {
            args.drain(index..=(index + 1).min(args.len() - 1));
        }
        let flag = match source.as_str() {
            "search" => "--train-search",
            "samples" => "--train-samples",
            "projected" => "--train-projected-samples",
            other => panic!(
                "unknown GPU training source `{other}`; expected search, samples, or projected"
            ),
        };
        args.insert(0, flag.into());
    }
    for arg in &mut args {
        let normalized = match arg.as_str() {
            "--gpu-search" => "--snapshot".to_string(),
            "--gpu-search-depth" => "--search-depth".to_string(),
            "--gpu-search-min-depth" => "--search-min-depth".to_string(),
            "--gpu-sample-search" => "--sample-search".to_string(),
            "--gpu-train-search" => "--train-search".to_string(),
            "--gpu-train-samples" => "--train-samples".to_string(),
            "--gpu-train-projected-samples" => "--train-projected-samples".to_string(),
            value if value.starts_with("--gpu-") && value != "--gpu-model" => {
                format!("--{}", &value[6..])
            }
            value => value.to_string(),
        };
        *arg = normalized;
    }
    args
}

/// Runs a GPU-focused native CLI command and reports whether one was selected.
/// The CPU training CLI can delegate here for backward-compatible mixed-mode
/// invocation, while the dedicated `gpu` binary parses this config directly.
pub(crate) fn run(config: &GpuCliConfig) -> bool {
    if config.backend_info {
        println!("{}", native_gpu_backend_info());
    } else if config.compile_shaders {
        println!("{}", native_gpu_compile_shaders());
    } else if config.compile_kernels {
        println!("{}", native_gpu_compile_kernels());
    } else if config.dispatch_smoke {
        println!("{}", native_gpu_dispatch_smoke());
    } else if config.training_dispatch_smoke {
        println!("{}", native_gpu_training_dispatch_smoke());
    } else if config.shader_info {
        println!("{}", gpu_shader_info());
    } else if let Some(path) = &config.model_info {
        let model = crate::gpu::training::load_compact_value_model(path)
            .unwrap_or_else(|message| panic!("{message}"));
        println!("{}", model.summary());
    } else if let Some(path) = &config.model_probe_zero {
        let model = crate::gpu::training::load_compact_value_model(path)
            .unwrap_or_else(|message| panic!("{message}"));
        println!("gpu_value_zero={}", model.predict_value(&[]));
    } else if let Some(path) = &config.project_samples {
        println!("{}", native_gpu_project_samples(path, config));
    } else if let Some(path) = &config.predict_samples {
        println!("{}", native_gpu_predict_samples(path, config));
    } else if let Some(path) = &config.distill_samples {
        println!("{}", gpu_distill_samples(path, config));
    } else if config.replay_append.is_some() {
        println!("{}", gpu_append_replay_samples(config));
    } else if config.search_snapshot.is_some() {
        let response = crate::gpu::search::search(gpu_search_request(config))
            .unwrap_or_else(|message| panic!("{message}"));
        println!("{}", response.result_json);
    } else if config.sample_search_snapshot.is_some() {
        let response =
            collect_gpu_cli_search_labels(gpu_search_label_batch_request(config), config)
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
            println!(
                "{}",
                serde_json::to_string_pretty(&response.samples).unwrap_or_else(|error| panic!(
                    "failed to encode GPU training samples: {error}"
                ))
            );
        }
    } else if config.train_search_snapshot.is_some() {
        register_gpu_sampling_job(config);
        gpu_stage_state("sampling", crate::training_runtime::JobState::Running, None);
        let response =
            collect_gpu_cli_search_labels(gpu_train_search_label_batch_request(config), config)
                .unwrap_or_else(|message| panic!("{message}"));
        gpu_stage_state(
            "sampling",
            if crate::training_runtime::cooperative_timed_out() {
                crate::training_runtime::JobState::TimedOut
            } else if crate::training_runtime::cooperative_cancelled() {
                crate::training_runtime::JobState::Cancelled
            } else {
                crate::training_runtime::JobState::Completed
            },
            None,
        );
        let (value_report, policy_report, wrote) =
            train_gpu_value_model_from_samples(&response.samples, config);
        crate::training_runtime::log(
            crate::training_runtime::LogLevel::Success,
            "gpu/train",
            format!(
            "search samples={} requested={} generated={} labeled={} value_epochs={} value_initial_loss={} value_final_loss={} policy_samples={} policy_steps={} policy_initial_loss={} policy_final_loss={} wrote={}",
            response.samples.len(), response.requested, response.generated_positions,
            response.labeled_positions, value_report.epochs, value_report.initial_loss,
            value_report.final_loss, policy_report.samples, policy_report.steps,
            policy_report.initial_loss, policy_report.final_loss, wrote
        ));
    } else if let Some(path) = &config.train_samples {
        let samples = crate::gpu::training::load_training_samples_json(path)
            .unwrap_or_else(|message| panic!("{message}"));
        let (value_report, policy_report, wrote) =
            train_gpu_value_model_from_samples(&samples, config);
        crate::training_runtime::log(
            crate::training_runtime::LogLevel::Success,
            "gpu/train",
            format!(
            "samples={} epochs={} initial_loss={} final_loss={} policy_samples={} policy_steps={} policy_initial_loss={} policy_final_loss={} wrote={}",
            value_report.samples, value_report.epochs, value_report.initial_loss,
            value_report.final_loss, policy_report.samples, policy_report.steps,
            policy_report.initial_loss, policy_report.final_loss, wrote
        ));
    } else if let Some(path) = &config.train_projected_samples {
        crate::training_runtime::log(
            crate::training_runtime::LogLevel::Success,
            "gpu/train",
            native_gpu_train_projected_samples(path, config),
        );
    } else {
        return false;
    }
    true
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_backend_info() -> String {
    crate::gpu::native::backend_info()
        .map(|info| info.to_string())
        .unwrap_or_else(|message| format!("native_gpu error={message}"))
}

#[cfg(not(feature = "neural-wgpu"))]
fn native_gpu_backend_info() -> String {
    "native_gpu unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_compile_shaders() -> String {
    crate::gpu::native::compile_engine_shaders()
        .map(|report| report.to_string())
        .unwrap_or_else(|message| format!("native_gpu_shader_compile error={message}"))
}

#[cfg(not(feature = "neural-wgpu"))]
fn native_gpu_compile_shaders() -> String {
    "native_gpu_shader_compile unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_compile_kernels() -> String {
    crate::gpu::native::compile_engine_kernels()
        .map(|report| report.to_string())
        .unwrap_or_else(|message| format!("native_gpu_kernel_compile error={message}"))
}

#[cfg(not(feature = "neural-wgpu"))]
fn native_gpu_compile_kernels() -> String {
    "native_gpu_kernel_compile unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_dispatch_smoke() -> String {
    crate::gpu::native::dispatch_search_smoke()
        .map(|report| report.to_string())
        .unwrap_or_else(|message| format!("native_gpu_dispatch error={message}"))
}

#[cfg(not(feature = "neural-wgpu"))]
fn native_gpu_dispatch_smoke() -> String {
    "native_gpu_dispatch unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_training_dispatch_smoke() -> String {
    crate::gpu::native::dispatch_project_features_smoke()
        .map(|report| report.to_string())
        .unwrap_or_else(|message| format!("native_gpu_training_dispatch error={message}"))
}

#[cfg(not(feature = "neural-wgpu"))]
fn native_gpu_training_dispatch_smoke() -> String {
    "native_gpu_training_dispatch unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_project_samples(path: &str, config: &GpuCliConfig) -> String {
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
                .collect(),
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

#[cfg(not(feature = "neural-wgpu"))]
fn native_gpu_project_samples(_path: &str, _config: &GpuCliConfig) -> String {
    "native_gpu_project_samples unavailable=engine built without neural-wgpu feature".to_string()
}

fn gpu_distill_samples(path: &str, config: &GpuCliConfig) -> String {
    let samples = crate::gpu::training::load_training_samples_json(path)
        .unwrap_or_else(|message| panic!("{message}"));
    let distilled =
        crate::gpu::training::distill_training_samples(&samples, &load_gpu_value_model(config));
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

fn gpu_append_replay_samples(config: &GpuCliConfig) -> String {
    let append_path = config
        .replay_append
        .as_deref()
        .expect("GPU replay append path is required");
    let buffer = config
        .replay_buffer
        .as_deref()
        .map(|path| {
            crate::gpu::training::load_training_samples_json(path)
                .unwrap_or_else(|message| panic!("{message}"))
        })
        .unwrap_or_default();
    let samples = crate::gpu::training::load_training_samples_json(append_path)
        .unwrap_or_else(|message| panic!("{message}"));
    let retained =
        crate::gpu::training::append_replay_samples(&buffer, &samples, config.replay_max.max(1));
    if let Some(out) = &config.out {
        crate::gpu::training::save_training_samples_json(out, &retained)
            .unwrap_or_else(|message| panic!("{message}"));
    }
    let summary = format!(
        "gpu_replay_append buffer={} appended={} retained={} max={} wrote={}",
        buffer.len(),
        samples.len(),
        retained.len(),
        config.replay_max.max(1),
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
fn native_gpu_predict_samples(path: &str, config: &GpuCliConfig) -> String {
    let samples = crate::gpu::training::load_training_samples_json(path)
        .unwrap_or_else(|message| panic!("{message}"));
    let predictions =
        crate::gpu::native::predict_values(crate::gpu::native::NativeValuePredictionRequest {
            model: load_gpu_value_model(config),
            features: samples
                .iter()
                .map(|sample| sample.features.clone())
                .collect(),
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

#[cfg(not(feature = "neural-wgpu"))]
fn native_gpu_predict_samples(_path: &str, _config: &GpuCliConfig) -> String {
    "native_gpu_predict_samples unavailable=engine built without neural-wgpu feature".to_string()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_gpu_train_projected_samples(path: &str, config: &GpuCliConfig) -> String {
    let samples = crate::gpu::training::load_training_samples_json(path)
        .unwrap_or_else(|message| panic!("{message}"));
    let (value_report, policy_report, wrote) = train_gpu_value_model_from_samples(&samples, config);
    format!(
        "gpu_train_projected_samples samples={} value_epochs={} value_initial_loss={} value_final_loss={} policy_samples={} policy_steps={} policy_initial_loss={} policy_final_loss={} wrote={}",
        value_report.samples, value_report.epochs, value_report.initial_loss,
        value_report.final_loss, policy_report.samples, policy_report.steps,
        policy_report.initial_loss, policy_report.final_loss, wrote
    )
}

#[cfg(not(feature = "neural-wgpu"))]
fn native_gpu_train_projected_samples(_path: &str, _config: &GpuCliConfig) -> String {
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

fn gpu_search_request(config: &GpuCliConfig) -> crate::gpu::search::GpuSearchRequest {
    crate::gpu::search::GpuSearchRequest {
        snapshot_json: read_optional_snapshot(&config.search_snapshot, "GPU search"),
        model_path: Some(gpu_value_model_path(config).to_string()),
        depth: config
            .search_depth
            .unwrap_or(crate::gpu::search::DEFAULT_GPU_SEARCH_DEPTH),
        min_depth: Some(
            config
                .search_min_depth
                .unwrap_or(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
        ),
        nodes: config.nodes.max(1).min(i32::MAX as usize) as i32,
        time_ms: config.training_time_ms.max(1).min(i32::MAX as u64) as i32,
    }
}

fn gpu_search_label_batch_request(
    config: &GpuCliConfig,
) -> crate::gpu::training::SearchLabelBatchRequest {
    gpu_label_batch_request(
        read_optional_snapshot(&config.sample_search_snapshot, "GPU sample search"),
        config,
    )
}

fn gpu_train_search_label_batch_request(
    config: &GpuCliConfig,
) -> crate::gpu::training::SearchLabelBatchRequest {
    gpu_label_batch_request(
        read_optional_snapshot(&config.train_search_snapshot, "GPU train search"),
        config,
    )
}

fn gpu_label_batch_request(
    snapshot_json: Option<String>,
    config: &GpuCliConfig,
) -> crate::gpu::training::SearchLabelBatchRequest {
    crate::gpu::training::SearchLabelBatchRequest {
        snapshot_json,
        mode: config.sample_mode,
        distill_model: gpu_sample_distill_model(config),
        count: config.sample_count,
        max_plies: config.sample_max_plies,
        position_depth: config
            .search_depth
            .unwrap_or(crate::gpu::training::DEFAULT_SEARCH_SAMPLE_POSITION_DEPTH),
        position_nodes: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_POSITION_NODES,
        position_time_ms: crate::gpu::training::DEFAULT_SEARCH_SAMPLE_POSITION_TIME_MS,
        label_depth: config
            .search_depth
            .unwrap_or(crate::cpu::search::DEFAULT_CPU_SEARCH_DEPTH),
        label_min_depth: Some(
            config
                .search_min_depth
                .unwrap_or(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
        ),
        label_nodes: config.nodes.max(1).min(i32::MAX as usize) as i32,
        label_time_ms: config.training_time_ms.max(1).min(i32::MAX as u64) as i32,
        label_weight: 1.0,
    }
}

fn read_optional_snapshot(path: &Option<String>, description: &str) -> Option<String> {
    path.as_ref().and_then(|path| {
        (!path.is_empty()).then(|| {
            std::fs::read_to_string(path).unwrap_or_else(|error| {
                panic!("failed to read {description} snapshot {path}: {error}")
            })
        })
    })
}

fn gpu_sample_distill_model(
    config: &GpuCliConfig,
) -> Option<crate::gpu::training::CompactValueModel> {
    (config.sample_mode == crate::gpu::training::SearchLabelMode::Distilled)
        .then(|| load_gpu_value_model(config))
}

fn collect_gpu_cli_search_labels(
    request: crate::gpu::training::SearchLabelBatchRequest,
    config: &GpuCliConfig,
) -> Result<crate::gpu::training::SearchLabelBatchResponse, String> {
    if request.mode == crate::gpu::training::SearchLabelMode::Search {
        crate::gpu::training::collect_native_gpu_search_label_samples(
            request,
            gpu_value_model_path(config),
        )
    } else {
        crate::gpu::training::collect_search_label_samples(request)
    }
}

fn gpu_value_model_path(config: &GpuCliConfig) -> &str {
    &config.value_model_path
}

pub(crate) fn value_head_training_config(
    config: &GpuCliConfig,
) -> crate::gpu::training::ValueHeadTrainingConfig {
    crate::gpu::training::ValueHeadTrainingConfig {
        learning_rate: config.learning_rate.clamp(0.0001, 0.1),
        epochs: config.epochs.clamp(1, 65_536),
        weight_decay: config.weight_decay.clamp(0.0, 0.01),
        momentum: config.momentum.clamp(0.0, 0.999),
    }
}

fn load_gpu_value_model(config: &GpuCliConfig) -> crate::gpu::training::CompactValueModel {
    crate::gpu::training::load_compact_value_model(gpu_value_model_path(config))
        .unwrap_or_else(|message| panic!("{message}"))
}

fn train_gpu_value_model_from_samples(
    samples: &[crate::gpu::training::TrainingSample],
    config: &GpuCliConfig,
) -> (
    crate::gpu::training::ValueHeadTrainingReport,
    crate::gpu::training::PolicyHeadTrainingReport,
    String,
) {
    let started = std::time::Instant::now();
    const DEFAULT_GPU_TRAINING_TIMEOUT_SECONDS: u64 = 600;
    let timeout_seconds = config
        .max_seconds
        .unwrap_or(DEFAULT_GPU_TRAINING_TIMEOUT_SECONDS)
        .max(1);
    crate::training_runtime::log(
        crate::training_runtime::LogLevel::Info,
        "gpu/train",
        format!(
            "starting samples={} epochs={} validation_split={} max_seconds={}",
            samples.len(),
            config.epochs,
            config.validation_split,
            timeout_seconds
        ),
    );
    let training_source = if config.train_search_snapshot.is_some() {
        crate::gpu::training::NativeTrainingSource::Search
    } else {
        crate::gpu::training::NativeTrainingSource::Samples
    };
    for (stage, dependency) in crate::gpu::training::native_training_stage_graph(training_source) {
        let id = format!("gpu-{}", format!("{stage:?}").to_ascii_lowercase());
        let dependency_id =
            dependency.map(|value| format!("gpu-{}", format!("{value:?}").to_ascii_lowercase()));
        let mut detail = std::collections::BTreeMap::from([
            (
                "source".into(),
                format!("{training_source:?}").to_ascii_lowercase(),
            ),
            ("input samples".into(), samples.len().to_string()),
            ("epochs".into(), config.epochs.to_string()),
            (
                "validation split".into(),
                config.validation_split.to_string(),
            ),
            ("learning rate".into(), config.learning_rate.to_string()),
            ("node limit".into(), config.nodes.to_string()),
            (
                "search depth".into(),
                format!(
                    "{}..={}",
                    config
                        .search_min_depth
                        .unwrap_or(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
                    config
                        .search_depth
                        .unwrap_or(crate::gpu::search::DEFAULT_GPU_SEARCH_DEPTH)
                ),
            ),
            ("timeout".into(), format!("{timeout_seconds} s")),
            (
                "safe checkpoints".into(),
                "between epochs or GPU submissions".into(),
            ),
        ]);
        if let Some(dependency) = &dependency_id {
            detail.insert("waits for".into(), dependency.clone());
        }
        crate::training_runtime::render_structured_event(
            &crate::training_runtime::TrainingEvent::Added {
                job: crate::training_runtime::JobSnapshot::new(
                    crate::training_runtime::JobMetadata {
                        id,
                        label: format!("{stage:?}"),
                        kind: "gpu-stage".into(),
                        seed: config.nodes as u64,
                        dependencies: dependency_id.clone().into_iter().collect(),
                        persistence_path: Some(config.candidate_out.clone().into()),
                        detail,
                    },
                ),
            },
        );
        crate::training_runtime::log(
            crate::training_runtime::LogLevel::Debug,
            "gpu/scheduler",
            format!(
                "registered gpu-{stage:?} dependency={}",
                dependency_id.as_deref().unwrap_or("none")
            ),
        );
    }
    gpu_stage_state(
        "projection",
        crate::training_runtime::JobState::Running,
        None,
    );
    let samples = crate::gpu::training::dedupe_training_samples(samples);
    let split = crate::gpu::training::split_validation_samples(&samples, config.validation_split);
    let training_samples = split
        .train_indices
        .iter()
        .map(|&index| samples[index].clone())
        .collect::<Vec<_>>();
    let validation_samples = split
        .validation_indices
        .iter()
        .map(|&index| samples[index].clone())
        .collect::<Vec<_>>();
    crate::training_runtime::log(
        crate::training_runtime::LogLevel::Info,
        "gpu/train",
        format!(
            "deduplicated={} train={} validation={} split_seed={}",
            samples.len(),
            training_samples.len(),
            validation_samples.len(),
            split.seed
        ),
    );
    let baseline = load_gpu_value_model(config);
    if config.out.as_deref() == Some(config.candidate_out.as_str()) {
        panic!("--candidate-out must differ from --out so an unvalidated model cannot replace the active model");
    }
    let mut candidate_config = config.clone();
    candidate_config.out = Some(config.candidate_out.clone());
    let (value_report, policy_report, _) =
        train_gpu_value_model_from_samples_unvalidated(&training_samples, &candidate_config);
    gpu_stage_state(
        "validation",
        crate::training_runtime::JobState::Running,
        None,
    );
    let candidate = crate::gpu::training::load_compact_value_model(&config.candidate_out)
        .unwrap_or_else(|message| panic!("failed to reload GPU candidate: {message}"));
    let baseline_loss = validation_value_loss(&baseline, &validation_samples);
    let candidate_loss = validation_value_loss(&candidate, &validation_samples);
    let run_id = crate::training_runtime::stable_run_id(split.seed as u64);
    let artifact = std::path::PathBuf::from(&config.candidate_out);
    let mut record = crate::training_runtime::ImprovementRecord::now(
        &run_id,
        "gpu-validation",
        split.seed as u64,
        baseline_loss as f64,
        candidate_loss as f64,
        format!(
            "deterministic position-group holdout train={} validation={}",
            training_samples.len(),
            validation_samples.len()
        ),
        artifact.clone(),
    );
    let persistence =
        crate::training_runtime::PersistenceWorker::start(config.improvement_log.clone().into());
    persistence
        .sender()
        .send(crate::training_runtime::PersistenceRequest::Candidate {
            path: artifact,
            bytes: crate::gpu::training::encode_compact_value_model(&candidate),
            record: record.clone(),
        })
        .unwrap_or_else(|error| panic!("GPU persistence worker stopped: {error}"));
    let cancelled = matches!(
        crate::training_runtime::cooperative_checkpoint(),
        crate::training_runtime::Checkpoint::Cancelled
    );
    let timed_out = crate::training_runtime::cooperative_timed_out()
        || started.elapsed().as_secs() >= timeout_seconds;
    let improved = !validation_samples.is_empty()
        && candidate_loss.is_finite()
        && baseline_loss.is_finite()
        && candidate_loss < baseline_loss;
    crate::training_runtime::log(
        if improved {
            crate::training_runtime::LogLevel::Success
        } else {
            crate::training_runtime::LogLevel::Warn
        },
        "gpu/validation",
        format!(
            "baseline_loss={baseline_loss} candidate_loss={candidate_loss} improved={improved}"
        ),
    );
    if improved && !timed_out && !cancelled {
        if let Some(out) = &config.out {
            crate::training_runtime::atomic_replace(
                std::path::Path::new(out),
                &crate::gpu::training::encode_compact_value_model(&candidate),
            )
            .unwrap_or_else(|message| panic!("GPU promotion failed: {message}"));
            record.outcome = "promoted".into();
            record.artifact_path = out.into();
            persistence
                .sender()
                .send(crate::training_runtime::PersistenceRequest::Journal { record })
                .unwrap_or_else(|error| panic!("GPU persistence worker stopped: {error}"));
        }
    } else {
        record.outcome = if cancelled {
            "cancelled".into()
        } else if timed_out {
            "timed-out".into()
        } else {
            "validation-rejected".into()
        };
        persistence
            .sender()
            .send(crate::training_runtime::PersistenceRequest::Journal { record })
            .unwrap_or_else(|error| panic!("GPU persistence worker stopped: {error}"));
    }
    gpu_stage_state(
        "validation",
        if cancelled {
            crate::training_runtime::JobState::Cancelled
        } else if timed_out {
            crate::training_runtime::JobState::TimedOut
        } else if improved {
            crate::training_runtime::JobState::Completed
        } else {
            crate::training_runtime::JobState::Failed
        },
        (!improved && !timed_out && !cancelled)
            .then(|| "candidate did not improve deterministic holdout loss".to_string()),
    );
    persistence
        .shutdown()
        .unwrap_or_else(|message| panic!("GPU candidate persistence failed: {message}"));
    crate::training_runtime::set_cooperative_deadline(None);
    (value_report, policy_report, config.candidate_out.clone())
}

fn register_gpu_sampling_job(config: &GpuCliConfig) {
    crate::training_runtime::render_structured_event(
        &crate::training_runtime::TrainingEvent::Added {
            job: crate::training_runtime::JobSnapshot::new(crate::training_runtime::JobMetadata {
                id: "gpu-sampling".into(),
                label: "Sampling".into(),
                kind: "gpu-stage".into(),
                seed: config.nodes as u64,
                dependencies: Vec::new(),
                persistence_path: config.out.clone().map(Into::into),
                detail: std::collections::BTreeMap::from([
                    ("source".into(), "search".into()),
                    ("requested samples".into(), config.sample_count.to_string()),
                    ("label node limit".into(), config.nodes.to_string()),
                    (
                        "label time".into(),
                        format!("{} ms", config.training_time_ms),
                    ),
                    (
                        "timeout".into(),
                        format!("{} s", config.max_seconds.unwrap_or(600).max(1)),
                    ),
                    (
                        "safe checkpoints".into(),
                        "between completed samples".into(),
                    ),
                ]),
            }),
        },
    );
}

fn gpu_stage_state(stage: &str, state: crate::training_runtime::JobState, error: Option<String>) {
    crate::training_runtime::render_structured_event(
        &crate::training_runtime::TrainingEvent::State {
            job_id: format!("gpu-{stage}"),
            state,
            error,
        },
    );
}

fn gpu_stage_checkpoint_state() -> crate::training_runtime::JobState {
    if crate::training_runtime::cooperative_timed_out() {
        crate::training_runtime::JobState::TimedOut
    } else if crate::training_runtime::cooperative_cancelled() {
        crate::training_runtime::JobState::Cancelled
    } else {
        crate::training_runtime::JobState::Completed
    }
}

fn validation_value_loss(
    model: &crate::gpu::training::CompactValueModel,
    samples: &[crate::gpu::training::TrainingSample],
) -> f32 {
    if samples.is_empty() {
        return f32::NAN;
    }
    samples
        .iter()
        .map(|sample| {
            let error = model.predict_value(&sample.features) - sample.label;
            error * error * sample.label_weight.max(0.0)
        })
        .sum::<f32>()
        / samples
            .iter()
            .map(|sample| sample.label_weight.max(0.0))
            .sum::<f32>()
            .max(f32::EPSILON)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn train_gpu_value_model_from_samples_unvalidated(
    samples: &[crate::gpu::training::TrainingSample],
    config: &GpuCliConfig,
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
                .collect(),
        },
    )
    .unwrap_or_else(|message| panic!("{message}"));
    gpu_stage_state("projection", gpu_stage_checkpoint_state(), None);
    gpu_stage_state(
        "valuehead",
        crate::training_runtime::JobState::Running,
        None,
    );
    train_native_gpu_value_model_from_projected(&working_set, &projected, model, config)
}

#[cfg(not(feature = "neural-wgpu"))]
fn train_gpu_value_model_from_samples_unvalidated(
    samples: &[crate::gpu::training::TrainingSample],
    config: &GpuCliConfig,
) -> (
    crate::gpu::training::ValueHeadTrainingReport,
    crate::gpu::training::PolicyHeadTrainingReport,
    String,
) {
    let samples = crate::gpu::training::dedupe_training_samples(samples);
    let model = load_gpu_value_model(config);
    gpu_stage_state("projection", gpu_stage_checkpoint_state(), None);
    gpu_stage_state(
        "valuehead",
        crate::training_runtime::JobState::Running,
        None,
    );
    let (value_trained, value_report) = crate::gpu::training::train_value_head_cpu(
        &model,
        &samples,
        value_head_training_config(config),
    )
    .unwrap_or_else(|message| panic!("{message}"));
    gpu_stage_state("valuehead", gpu_stage_checkpoint_state(), None);
    gpu_stage_state(
        "policyhead",
        crate::training_runtime::JobState::Running,
        None,
    );
    let (trained, policy_report) = crate::gpu::training::train_policy_head_cpu(
        &value_trained,
        &samples,
        value_head_training_config(config),
    )
    .unwrap_or_else(|message| panic!("{message}"));
    gpu_stage_state("policyhead", gpu_stage_checkpoint_state(), None);
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
    config: &GpuCliConfig,
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
            config: value_head_training_config(config),
            train_hidden_layers: true,
        })
        .unwrap_or_else(|message| panic!("{message}"));
    gpu_stage_state("valuehead", gpu_stage_checkpoint_state(), None);
    gpu_stage_state(
        "policyhead",
        crate::training_runtime::JobState::Running,
        None,
    );
    let (trained, policy_report) = crate::gpu::native::train_policy_head(
        crate::gpu::native::NativePolicyHeadTrainingRequest {
            model: value_trained,
            samples: samples.to_vec(),
            projected_features: projected.to_vec(),
            config: value_head_training_config(config),
        },
    )
    .unwrap_or_else(|message| panic!("{message}"));
    gpu_stage_state("policyhead", gpu_stage_checkpoint_state(), None);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_training_options_are_parsed_and_clamped_for_engine_training() {
        let config = crate::cpu::training::CpuCliConfig::from_env(vec![
            "--gpu-learning-rate".to_string(),
            "0.025".to_string(),
            "--gpu-epochs".to_string(),
            "96".to_string(),
            "--gpu-weight-decay".to_string(),
            "0.0005".to_string(),
            "--gpu-momentum".to_string(),
            "0.85".to_string(),
        ]);
        let training = value_head_training_config(&config.gpu);
        assert_eq!(training.learning_rate, 0.025);
        assert_eq!(training.epochs, 96);
        assert_eq!(training.weight_decay, 0.0005);
        assert_eq!(training.momentum, 0.85);
    }

    #[test]
    fn gpu_training_search_depth_controls_label_generation() {
        let config = GpuCliConfig::from_env(vec![
            "train".into(),
            "--source".into(),
            "search".into(),
            "--search-depth".into(),
            "9".into(),
            "--search-min-depth".into(),
            "4".into(),
        ]);
        let request = gpu_train_search_label_batch_request(&config);
        assert_eq!(request.position_depth, 9);
        assert_eq!(request.label_depth, 9);
        assert_eq!(request.label_min_depth, Some(4));
    }

    #[test]
    fn gpu_train_subcommands_and_flat_aliases_select_the_same_sources() {
        let samples = GpuCliConfig::from_env(vec![
            "train".into(),
            "--source".into(),
            "samples".into(),
            "samples.json".into(),
        ]);
        assert_eq!(samples.train_samples.as_deref(), Some("samples.json"));
        let projected = GpuCliConfig::from_env(vec![
            "--gpu-train-projected-samples".into(),
            "projected.json".into(),
        ]);
        assert_eq!(
            projected.train_projected_samples.as_deref(),
            Some("projected.json")
        );
        let search =
            GpuCliConfig::from_env(vec!["train".into(), "--source".into(), "search".into()]);
        assert_eq!(search.train_search_snapshot.as_deref(), Some(""));
    }
}
