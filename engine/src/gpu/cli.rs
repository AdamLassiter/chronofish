#[derive(Clone)]
pub(crate) struct GpuCliConfig {
    pub(crate) nodes: usize,
    pub(crate) training_time_ms: u64,
    pub(crate) out: Option<String>,
    pub(crate) gpu_backend_info: bool,
    pub(crate) gpu_compile_shaders: bool,
    pub(crate) gpu_compile_kernels: bool,
    pub(crate) gpu_dispatch_smoke: bool,
    pub(crate) gpu_training_dispatch_smoke: bool,
    pub(crate) gpu_shader_info: bool,
    pub(crate) gpu_value_model_path: String,
    pub(crate) gpu_model_info: Option<String>,
    pub(crate) gpu_model_probe_zero: Option<String>,
    pub(crate) gpu_project_samples: Option<String>,
    pub(crate) gpu_predict_samples: Option<String>,
    pub(crate) gpu_distill_samples: Option<String>,
    pub(crate) gpu_replay_buffer: Option<String>,
    pub(crate) gpu_replay_append: Option<String>,
    pub(crate) gpu_replay_max: usize,
    pub(crate) gpu_search_snapshot: Option<String>,
    pub(crate) gpu_search_depth: Option<i32>,
    pub(crate) gpu_search_min_depth: Option<i32>,
    pub(crate) gpu_sample_search_snapshot: Option<String>,
    pub(crate) gpu_sample_mode: crate::gpu::training::SearchLabelMode,
    pub(crate) gpu_sample_count: usize,
    pub(crate) gpu_sample_max_plies: usize,
    pub(crate) gpu_train_search_snapshot: Option<String>,
    pub(crate) gpu_train_samples: Option<String>,
    pub(crate) gpu_train_projected_samples: Option<String>,
    pub(crate) gpu_learning_rate: f32,
    pub(crate) gpu_epochs: usize,
    pub(crate) gpu_weight_decay: f32,
    pub(crate) gpu_momentum: f32,
}

impl Default for GpuCliConfig {
    fn default() -> Self {
        let training = crate::gpu::training::ValueHeadTrainingConfig::default();
        Self {
            nodes: crate::gpu::search::DEFAULT_GPU_SEARCH_NODES as usize,
            training_time_ms: crate::gpu::search::DEFAULT_GPU_SEARCH_TIME_MS as u64,
            out: None,
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
            gpu_learning_rate: training.learning_rate,
            gpu_epochs: training.epochs,
            gpu_weight_decay: training.weight_decay,
            gpu_momentum: training.momentum,
        }
    }
}

impl GpuCliConfig {
    pub(crate) fn with_run_options(
        &self,
        nodes: usize,
        training_time_ms: u64,
        out: Option<String>,
    ) -> Self {
        let mut config = self.clone();
        config.nodes = nodes;
        config.training_time_ms = training_time_ms;
        config.out = out;
        config
    }

    pub(crate) fn consume_option(&mut self, option: &str, value: Option<&str>) -> Option<usize> {
        let optional_path = || value.filter(|next| !next.starts_with("--"));
        let consumed_optional = || if optional_path().is_some() { 2 } else { 1 };
        let parse_usize = || value.and_then(|raw| raw.parse::<usize>().ok());
        let parse_i32 = || value.and_then(|raw| raw.parse::<i32>().ok());
        let parse_f32 = || value.and_then(|raw| raw.parse::<f32>().ok());
        match option {
            "--gpu-backend-info" => self.gpu_backend_info = true,
            "--gpu-compile-shaders" => self.gpu_compile_shaders = true,
            "--gpu-compile-kernels" => self.gpu_compile_kernels = true,
            "--gpu-dispatch-smoke" => self.gpu_dispatch_smoke = true,
            "--gpu-training-dispatch-smoke" => self.gpu_training_dispatch_smoke = true,
            "--gpu-shader-info" => self.gpu_shader_info = true,
            "--gpu-model-info" => {
                self.gpu_model_info = Some(
                    optional_path()
                        .unwrap_or(crate::gpu::training::DEFAULT_VALUE_MODEL_PATH)
                        .to_string(),
                );
                return Some(consumed_optional());
            }
            "--gpu-model-probe-zero" => {
                self.gpu_model_probe_zero = Some(
                    optional_path()
                        .unwrap_or(crate::gpu::training::DEFAULT_VALUE_MODEL_PATH)
                        .to_string(),
                );
                return Some(consumed_optional());
            }
            "--gpu-model" | "--gpu-value-model" => {
                if let Some(value) = value {
                    self.gpu_value_model_path = value.to_string();
                }
            }
            "--gpu-project-samples" => self.gpu_project_samples = value.map(ToOwned::to_owned),
            "--gpu-predict-samples" => self.gpu_predict_samples = value.map(ToOwned::to_owned),
            "--gpu-distill-samples" => self.gpu_distill_samples = value.map(ToOwned::to_owned),
            "--gpu-replay-buffer" => self.gpu_replay_buffer = value.map(ToOwned::to_owned),
            "--gpu-replay-append" | "--gpu-append-replay-samples" => {
                self.gpu_replay_append = value.map(ToOwned::to_owned)
            }
            "--gpu-replay-max" => {
                self.gpu_replay_max = parse_usize().unwrap_or(self.gpu_replay_max)
            }
            "--gpu-search" => {
                self.gpu_search_snapshot = Some(optional_path().unwrap_or("").to_string());
                return Some(consumed_optional());
            }
            "--gpu-search-depth" => self.gpu_search_depth = parse_i32(),
            "--gpu-search-min-depth" => self.gpu_search_min_depth = parse_i32(),
            "--gpu-train-samples" => self.gpu_train_samples = value.map(ToOwned::to_owned),
            "--gpu-train-projected-samples" => {
                self.gpu_train_projected_samples = value.map(ToOwned::to_owned)
            }
            "--gpu-learning-rate" => {
                self.gpu_learning_rate = parse_f32().unwrap_or(self.gpu_learning_rate)
            }
            "--gpu-epochs" => self.gpu_epochs = parse_usize().unwrap_or(self.gpu_epochs),
            "--gpu-weight-decay" => {
                self.gpu_weight_decay = parse_f32().unwrap_or(self.gpu_weight_decay)
            }
            "--gpu-momentum" => self.gpu_momentum = parse_f32().unwrap_or(self.gpu_momentum),
            "--gpu-train-search" => {
                self.gpu_train_search_snapshot = Some(optional_path().unwrap_or("").to_string());
                return Some(consumed_optional());
            }
            "--gpu-sample-search" => {
                self.gpu_sample_search_snapshot = Some(optional_path().unwrap_or("").to_string());
                return Some(consumed_optional());
            }
            "--gpu-sample-count" | "--sample-count" => {
                self.gpu_sample_count = parse_usize().unwrap_or(self.gpu_sample_count)
            }
            "--gpu-sample-mode" | "--sample-mode" => {
                if let Some(value) = value {
                    self.gpu_sample_mode = crate::gpu::training::SearchLabelMode::parse(value)
                        .unwrap_or_else(|message| panic!("{message}"));
                }
            }
            "--gpu-sample-plies" | "--sample-plies" => {
                self.gpu_sample_max_plies = parse_usize().unwrap_or(self.gpu_sample_max_plies)
            }
            _ => return None,
        }
        Some(
            if option.starts_with("--gpu-")
                || matches!(
                    option,
                    "--sample-count" | "--sample-mode" | "--sample-plies"
                )
            {
                2
            } else {
                1
            },
        )
    }

    pub(crate) fn normalize(&mut self) {
        self.gpu_sample_count = self.gpu_sample_count.max(1);
    }
}

pub fn run_gpu_cli() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut config = GpuCliConfig::default();
    let mut index = 0;
    while index < args.len() {
        let option = args[index].as_str();
        let value = args.get(index + 1).map(String::as_str);
        if let Some(consumed) = config.consume_option(option, value) {
            index += consumed;
            continue;
        }
        match option {
            "--nodes" => {
                config.nodes = value
                    .and_then(|raw| raw.parse().ok())
                    .unwrap_or(config.nodes);
                index += 2;
            }
            "--training-time-ms" | "--turn-time-ms" => {
                config.training_time_ms = value
                    .and_then(|raw| raw.parse().ok())
                    .unwrap_or(config.training_time_ms);
                index += 2;
            }
            "--out" => {
                config.out = value.map(ToOwned::to_owned);
                index += 2;
            }
            _ => index += 1,
        }
    }
    config.normalize();
    if !run(&config) {
        eprintln!("No GPU command selected. Try --gpu-search, --gpu-sample-search, or --gpu-train-samples.");
    }
}

/// Runs a GPU-focused native CLI command and reports whether one was selected.
/// The CPU training CLI can delegate here for backward-compatible mixed-mode
/// invocation, while the dedicated `gpu` binary parses this config directly.
pub(crate) fn run(config: &GpuCliConfig) -> bool {
    if config.gpu_backend_info {
        println!("{}", native_gpu_backend_info());
    } else if config.gpu_compile_shaders {
        println!("{}", native_gpu_compile_shaders());
    } else if config.gpu_compile_kernels {
        println!("{}", native_gpu_compile_kernels());
    } else if config.gpu_dispatch_smoke {
        println!("{}", native_gpu_dispatch_smoke());
    } else if config.gpu_training_dispatch_smoke {
        println!("{}", native_gpu_training_dispatch_smoke());
    } else if config.gpu_shader_info {
        println!("{}", gpu_shader_info());
    } else if let Some(path) = &config.gpu_model_info {
        let model = crate::gpu::training::load_compact_value_model(path)
            .unwrap_or_else(|message| panic!("{message}"));
        println!("{}", model.summary());
    } else if let Some(path) = &config.gpu_model_probe_zero {
        let model = crate::gpu::training::load_compact_value_model(path)
            .unwrap_or_else(|message| panic!("{message}"));
        println!("gpu_value_zero={}", model.predict_value(&[]));
    } else if let Some(path) = &config.gpu_project_samples {
        println!("{}", native_gpu_project_samples(path, config));
    } else if let Some(path) = &config.gpu_predict_samples {
        println!("{}", native_gpu_predict_samples(path, config));
    } else if let Some(path) = &config.gpu_distill_samples {
        println!("{}", gpu_distill_samples(path, config));
    } else if config.gpu_replay_append.is_some() {
        println!("{}", gpu_append_replay_samples(config));
    } else if config.gpu_search_snapshot.is_some() {
        let response = crate::gpu::search::search(gpu_search_request(config))
            .unwrap_or_else(|message| panic!("{message}"));
        println!("{}", response.result_json);
    } else if config.gpu_sample_search_snapshot.is_some() {
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
    } else if config.gpu_train_search_snapshot.is_some() {
        let response =
            collect_gpu_cli_search_labels(gpu_train_search_label_batch_request(config), config)
                .unwrap_or_else(|message| panic!("{message}"));
        let (value_report, policy_report, wrote) =
            train_gpu_value_model_from_samples(&response.samples, config);
        println!(
            "gpu_train_search samples={} requested={} generated={} labeled={} value_epochs={} value_initial_loss={} value_final_loss={} policy_samples={} policy_steps={} policy_initial_loss={} policy_final_loss={} wrote={}",
            response.samples.len(), response.requested, response.generated_positions,
            response.labeled_positions, value_report.epochs, value_report.initial_loss,
            value_report.final_loss, policy_report.samples, policy_report.steps,
            policy_report.initial_loss, policy_report.final_loss, wrote
        );
    } else if let Some(path) = &config.gpu_train_samples {
        let samples = crate::gpu::training::load_training_samples_json(path)
            .unwrap_or_else(|message| panic!("{message}"));
        let (value_report, policy_report, wrote) =
            train_gpu_value_model_from_samples(&samples, config);
        println!(
            "gpu_train_value_head samples={} epochs={} initial_loss={} final_loss={} policy_samples={} policy_steps={} policy_initial_loss={} policy_final_loss={} wrote={}",
            value_report.samples, value_report.epochs, value_report.initial_loss,
            value_report.final_loss, policy_report.samples, policy_report.steps,
            policy_report.initial_loss, policy_report.final_loss, wrote
        );
    } else if let Some(path) = &config.gpu_train_projected_samples {
        println!("{}", native_gpu_train_projected_samples(path, config));
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
        .gpu_replay_append
        .as_deref()
        .expect("GPU replay append path is required");
    let buffer = config
        .gpu_replay_buffer
        .as_deref()
        .map(|path| {
            crate::gpu::training::load_training_samples_json(path)
                .unwrap_or_else(|message| panic!("{message}"))
        })
        .unwrap_or_default();
    let samples = crate::gpu::training::load_training_samples_json(append_path)
        .unwrap_or_else(|message| panic!("{message}"));
    let retained = crate::gpu::training::append_replay_samples(
        &buffer,
        &samples,
        config.gpu_replay_max.max(1),
    );
    if let Some(out) = &config.out {
        crate::gpu::training::save_training_samples_json(out, &retained)
            .unwrap_or_else(|message| panic!("{message}"));
    }
    let summary = format!(
        "gpu_replay_append buffer={} appended={} retained={} max={} wrote={}",
        buffer.len(),
        samples.len(),
        retained.len(),
        config.gpu_replay_max.max(1),
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
    let (value_report, policy_report, _) =
        train_native_gpu_value_model_from_projected(&samples, &projected, model, config);
    format!(
        "gpu_train_projected_samples samples={} projected_values={} value_epochs={} value_initial_loss={} value_final_loss={} policy_samples={} policy_steps={} policy_initial_loss={} policy_final_loss={} wrote={}",
        value_report.samples, projected.len(), value_report.epochs, value_report.initial_loss,
        value_report.final_loss, policy_report.samples, policy_report.steps,
        policy_report.initial_loss, policy_report.final_loss, config.out.as_deref().unwrap_or("")
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
        snapshot_json: read_optional_snapshot(&config.gpu_search_snapshot, "GPU search"),
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
    config: &GpuCliConfig,
) -> crate::gpu::training::SearchLabelBatchRequest {
    gpu_label_batch_request(
        read_optional_snapshot(&config.gpu_sample_search_snapshot, "GPU sample search"),
        config,
    )
}

fn gpu_train_search_label_batch_request(
    config: &GpuCliConfig,
) -> crate::gpu::training::SearchLabelBatchRequest {
    gpu_label_batch_request(
        read_optional_snapshot(&config.gpu_train_search_snapshot, "GPU train search"),
        config,
    )
}

fn gpu_label_batch_request(
    snapshot_json: Option<String>,
    config: &GpuCliConfig,
) -> crate::gpu::training::SearchLabelBatchRequest {
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
    (config.gpu_sample_mode == crate::gpu::training::SearchLabelMode::Distilled)
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
    &config.gpu_value_model_path
}

pub(crate) fn value_head_training_config(
    config: &GpuCliConfig,
) -> crate::gpu::training::ValueHeadTrainingConfig {
    crate::gpu::training::ValueHeadTrainingConfig {
        learning_rate: config.gpu_learning_rate.clamp(0.0001, 0.1),
        epochs: config.gpu_epochs.clamp(1, 65_536),
        weight_decay: config.gpu_weight_decay.clamp(0.0, 0.01),
        momentum: config.gpu_momentum.clamp(0.0, 0.999),
    }
}

fn load_gpu_value_model(config: &GpuCliConfig) -> crate::gpu::training::CompactValueModel {
    crate::gpu::training::load_compact_value_model(gpu_value_model_path(config))
        .unwrap_or_else(|message| panic!("{message}"))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn train_gpu_value_model_from_samples(
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
    train_native_gpu_value_model_from_projected(&working_set, &projected, model, config)
}

#[cfg(not(feature = "neural-wgpu"))]
fn train_gpu_value_model_from_samples(
    samples: &[crate::gpu::training::TrainingSample],
    config: &GpuCliConfig,
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
        value_head_training_config(config),
    )
    .unwrap_or_else(|message| panic!("{message}"));
    let (trained, policy_report) = crate::gpu::training::train_policy_head_cpu(
        &value_trained,
        &samples,
        value_head_training_config(config),
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
    let (trained, policy_report) = crate::gpu::native::train_policy_head(
        crate::gpu::native::NativePolicyHeadTrainingRequest {
            model: value_trained,
            samples: samples.to_vec(),
            projected_features: projected.to_vec(),
            config: value_head_training_config(config),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_training_options_are_parsed_and_clamped_for_engine_training() {
        let config = crate::cpu::training::TrainerConfig::from_env(vec![
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
}
