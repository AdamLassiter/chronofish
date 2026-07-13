use std::{collections::BTreeMap, fmt, path::Path};

use super::{GpuKernel, GpuKernelSet, WgslShader};
use crate::{
    cpu::{search_deadline, MoveStep, SearchOptions, TurnPlan},
    wasm_api::parse_game_snapshot,
    BoardSnapshot,
    Color,
    Game,
    Piece,
    PieceType,
    Timeline,
};

pub const PROJECT_FEATURES_SHADER: &str = include_str!("training/shaders/project_features.wgsl");
pub const FORWARD_LAYER_SHADER: &str = include_str!("training/shaders/forward_layer.wgsl");
pub const FORWARD_INDEXED_LAYER_SHADER: &str =
    include_str!("training/shaders/forward_indexed_layer.wgsl");
pub const FORWARD_OUTPUT_SHADER: &str = include_str!("training/shaders/forward_output.wgsl");
pub const OUTPUT_DELTA_SHADER: &str = include_str!("training/shaders/output_delta.wgsl");
pub const HIDDEN_DELTA_SHADER: &str = include_str!("training/shaders/hidden_delta.wgsl");
pub const HIDDEN3_DELTA_SHADER: &str = include_str!("training/shaders/hidden3_delta.wgsl");
pub const APPLY_LAYER_SHADER: &str = include_str!("training/shaders/apply_layer.wgsl");
pub const APPLY_INDEXED_LAYER_SHADER: &str =
    include_str!("training/shaders/apply_indexed_layer.wgsl");
pub const APPLY_OUTPUT_SHADER: &str = include_str!("training/shaders/apply_output.wgsl");
pub const POLICY_SHADER: &str = include_str!("training/shaders/policy.wgsl");
pub const POLICY_LOSS_SHADER: &str = include_str!("training/shaders/policy_loss.wgsl");
pub const REDUCE_LOSS_SHADER: &str = include_str!("training/shaders/reduce_loss.wgsl");

pub const SHADERS: &[WgslShader] = &[
    WgslShader {
        name: "project_features.wgsl",
        source: PROJECT_FEATURES_SHADER,
    },
    WgslShader {
        name: "forward_layer.wgsl",
        source: FORWARD_LAYER_SHADER,
    },
    WgslShader {
        name: "forward_indexed_layer.wgsl",
        source: FORWARD_INDEXED_LAYER_SHADER,
    },
    WgslShader {
        name: "forward_output.wgsl",
        source: FORWARD_OUTPUT_SHADER,
    },
    WgslShader {
        name: "output_delta.wgsl",
        source: OUTPUT_DELTA_SHADER,
    },
    WgslShader {
        name: "hidden_delta.wgsl",
        source: HIDDEN_DELTA_SHADER,
    },
    WgslShader {
        name: "hidden3_delta.wgsl",
        source: HIDDEN3_DELTA_SHADER,
    },
    WgslShader {
        name: "apply_layer.wgsl",
        source: APPLY_LAYER_SHADER,
    },
    WgslShader {
        name: "apply_indexed_layer.wgsl",
        source: APPLY_INDEXED_LAYER_SHADER,
    },
    WgslShader {
        name: "apply_output.wgsl",
        source: APPLY_OUTPUT_SHADER,
    },
    WgslShader {
        name: "policy.wgsl",
        source: POLICY_SHADER,
    },
    WgslShader {
        name: "policy_loss.wgsl",
        source: POLICY_LOSS_SHADER,
    },
    WgslShader {
        name: "reduce_loss.wgsl",
        source: REDUCE_LOSS_SHADER,
    },
];

pub const KERNELS: &[GpuKernel] = &[
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "project_features",
        shader: "project_features.wgsl",
        entry_point: "project_features",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "forward_indexed_layer_naive",
        shader: "forward_indexed_layer.wgsl",
        entry_point: "forward_layer_naive",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "forward_indexed_layer_tiled",
        shader: "forward_indexed_layer.wgsl",
        entry_point: "forward_layer",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "forward_layer_naive",
        shader: "forward_layer.wgsl",
        entry_point: "forward_layer_naive",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "forward_layer_tiled",
        shader: "forward_layer.wgsl",
        entry_point: "forward_layer",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "forward_output",
        shader: "forward_output.wgsl",
        entry_point: "forward_output",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "reduce_loss",
        shader: "reduce_loss.wgsl",
        entry_point: "reduce_loss",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "output_delta",
        shader: "output_delta.wgsl",
        entry_point: "output_delta",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "hidden3_delta",
        shader: "hidden3_delta.wgsl",
        entry_point: "hidden3_delta",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "hidden_delta_naive",
        shader: "hidden_delta.wgsl",
        entry_point: "hidden_delta_naive",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "hidden_delta_tiled",
        shader: "hidden_delta.wgsl",
        entry_point: "hidden_delta",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "apply_indexed_layer_naive",
        shader: "apply_indexed_layer.wgsl",
        entry_point: "apply_layer_naive",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "apply_indexed_layer_tiled",
        shader: "apply_indexed_layer.wgsl",
        entry_point: "apply_layer",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "apply_layer_naive",
        shader: "apply_layer.wgsl",
        entry_point: "apply_layer_naive",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "apply_layer_tiled",
        shader: "apply_layer.wgsl",
        entry_point: "apply_layer",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "apply_output",
        shader: "apply_output.wgsl",
        entry_point: "apply_output",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "policy_forward_naive",
        shader: "policy.wgsl",
        entry_point: "forward_policy_naive",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "policy_forward_tiled",
        shader: "policy.wgsl",
        entry_point: "forward_policy",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "policy_delta",
        shader: "policy.wgsl",
        entry_point: "policy_delta",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "policy_apply_naive",
        shader: "policy.wgsl",
        entry_point: "apply_policy_naive",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "policy_apply_tiled",
        shader: "policy.wgsl",
        entry_point: "apply_policy",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuTraining,
        label: "policy_loss",
        shader: "policy_loss.wgsl",
        entry_point: "reduce_policy_loss",
        constants: &[],
    },
];

pub const DEFAULT_VALUE_MODEL_PATH: &str = "engine/models/gpu-v1/value-model.cfnn";
pub const VALUE_SCORE_SCALE: f32 = 20_000.0;
pub const POLICY_BUCKETS: u32 = 257;
pub const DEFAULT_PROJECTION_SIZE: usize = 2048;
pub const DEFAULT_PROJECTION_SEED: u32 = 2_166_136_261;
pub const DEFAULT_HIDDEN_LAYERS: &[u32] = &[1024, 512, 256];
pub const DEFAULT_SEARCH_SAMPLE_COUNT: usize = 1;
pub const DEFAULT_SEARCH_SAMPLE_MAX_PLIES: usize = 0;
pub const DEFAULT_SEARCH_SAMPLE_POSITION_DEPTH: i32 = 2;
pub const DEFAULT_SEARCH_SAMPLE_POSITION_NODES: i32 = 512;
pub const DEFAULT_SEARCH_SAMPLE_POSITION_TIME_MS: i32 = 3_000;
pub const VALUE_EPOCHS_PER_SUBMIT: usize = 64;
pub const POLICY_STEPS_PER_SUBMIT: usize = 64;
pub const TILED_TRAINING_MIN_BATCH: usize = 16;
pub const CPU_PREDICTION_MAX_BATCH: usize = 4;
pub const MIN_HIDDEN_TRAINING_POSITIONS: usize = 256;
pub const CPU_HEAD_TRAINING_MAX_POSITIONS: usize = 32;
pub const OPTIMIZER_MOMENTUM: f32 = 0.9;
pub const DEFAULT_BATCH_SIZE: usize = 1024;
pub const DEFAULT_VALIDATION_SPLIT: f32 = 0.1;
pub const DEFAULT_PATIENCE: usize = 12;
pub const DEFAULT_WEIGHT_DECAY: f32 = 0.00001;
pub const MAX_GPU_TRAINING_SAMPLES: usize = 16_384;
pub const MAX_GPU_TRAINING_BATCH: usize = 16_384;
pub const MAX_GPU_VALIDATION_INTERVAL: usize = 16_384;
pub const PROJECTION_CHUNK_SIZE: usize = 256;
pub const PROJECTION_TEMPORARY_BUDGET: usize = 128 * 1024 * 1024;
pub const DEFAULT_PROJECTED_WORKING_SET_BYTES: usize = 128 * 1024 * 1024;
pub const COMPACT_VALUE_MODEL_MAGIC: &[u8; 4] = b"CFNN";
pub const MAX_COMPACT_VALUE_MODEL_VERSION: u32 = 5;
pub const NEURAL_BOARD_PLANES: usize = 32;
pub const NEURAL_BOARD_SQUARES: usize = 64;
pub const AUXILIARY_VALUE_HEAD_COUNT: usize = 9;
pub const MIN_POLICY_REPLAY_FRACTION: f32 = 0.25;
pub const MIN_POLICY_WORKING_SET_FRACTION: f32 = 0.25;
pub const MAX_PLAYOUT_PLIES: usize = 10;
// Each sampler owns a browser worker and its WebGPU device.  Keeping this
// bounded avoids worker-loader/device failures on Firefox and lower-capability
// adapters while still providing useful parallel label collection.
pub const MAX_PARALLEL_GPU_TRAINING_WORKERS: usize = 8;
pub const GPU_WARMUP_MAX_TIME_MS: u64 = 5_000;
pub const GPU_POSITION_GENERATION_TIME_MS: u64 = 3_000;
pub const LABEL_REQUEST_MIN_TIMEOUT_MS: u64 = 30_000;
pub const LABEL_REQUEST_MAX_TIMEOUT_MS: u64 = 120_000;
pub const LABEL_REQUEST_NODE_TIMEOUT_FACTOR_MS: u64 = 3;
pub const TRAINING_IO_TIMEOUT_MS: u64 = 15_000;
pub const DEFAULT_PARTIAL_OUTCOME_LABEL_KIND: &str = "search-bootstrap";
pub const DEFAULT_PARTIAL_OUTCOME_LABEL_WEIGHT: f32 = 0.5;
pub const OUTCOME_LABEL_DECAY: f32 = 0.96;
pub const OUTCOME_LABEL_WEIGHT: f32 = 1.25;
pub const DUEL_LABEL_WEIGHT: f32 = 1.35;
pub const DUEL_DRAW_LABEL_WEIGHT: f32 = 1.1;
pub const DISTILLED_LABEL_WEIGHT: f32 = 0.25;

#[derive(Clone, Debug, PartialEq)]
pub struct CompactValueModel {
    pub version: u32,
    pub projection_size: u32,
    pub projection_seed: u32,
    pub hidden_layers: Vec<u32>,
    pub hidden_weights: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub policy_logits: Vec<f32>,
    pub policy_weights: Vec<f32>,
    pub auxiliary_value_weights: Vec<f32>,
    pub scale: f32,
    pub bias: f32,
    pub output_activation: OutputActivation,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct EncodableCompactValueModelJson {
    projection_size: u32,
    projection_seed: u32,
    hidden_layers: Vec<u32>,
    hidden_weights: Vec<f32>,
    output_weights: Vec<f32>,
    #[serde(default)]
    auxiliary_value_weights: Vec<f32>,
    #[serde(default)]
    policy_weights: Vec<f32>,
    #[serde(default)]
    policy_logits: Vec<f32>,
    #[serde(default)]
    scale: Option<f32>,
    #[serde(default)]
    bias: Option<f32>,
    #[serde(default)]
    output_activation: Option<OutputActivationJson>,
}

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
enum OutputActivationJson {
    Linear,
    Tanh,
}

impl CompactValueModel {
    pub fn summary(&self) -> CompactValueModelSummary {
        CompactValueModelSummary {
            version: self.version,
            projection_size: self.projection_size,
            projection_seed: self.projection_seed,
            hidden_layers: self.hidden_layers.clone(),
            hidden_weight_count: self.hidden_weights.len(),
            output_weight_count: self.output_weights.len(),
            policy_logit_count: self.policy_logits.len(),
            policy_weight_count: self.policy_weights.len(),
            auxiliary_value_weight_count: self.auxiliary_value_weights.len(),
            scale: self.scale,
            bias: self.bias,
            output_activation: self.output_activation,
        }
    }

    pub fn predict_value(&self, features: &[f32]) -> f32 {
        predict_value(features, self)
    }

    pub fn predict_values<'a>(&self, samples: impl IntoIterator<Item = &'a [f32]>) -> Vec<f32> {
        samples
            .into_iter()
            .map(|features| self.predict_value(features))
            .collect()
    }

    pub fn encode(&self) -> Vec<u8> {
        encode_compact_value_model(self)
    }
}

pub fn distill_training_samples(
    samples: &[TrainingSample],
    model: &CompactValueModel,
) -> Vec<TrainingSample> {
    samples
        .iter()
        .map(|sample| {
            let mut distilled = sample.clone();
            distilled.label = model.predict_value(&sample.features);
            distilled.policy = None;
            distilled.label_kind = Some("distilled".to_string());
            distilled.label_weight = DISTILLED_LABEL_WEIGHT;
            distilled.pseudo = Some(true);
            distilled
        })
        .collect()
}

pub fn distill_training_samples_with_labels(
    samples: &[TrainingSample],
    labels: &[Option<f32>],
) -> Vec<TrainingSample> {
    samples
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let mut distilled = sample.clone();
            distilled.label = labels
                .get(index)
                .and_then(|label| *label)
                .filter(|label| label.is_finite())
                .unwrap_or(0.0);
            distilled.policy = None;
            distilled.label_kind = Some("distilled".to_string());
            distilled.label_weight = DISTILLED_LABEL_WEIGHT;
            distilled.pseudo = Some(true);
            distilled
        })
        .collect()
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct DistillTrainingSamplesWithLabelsRequest {
    samples: Vec<TrainingSample>,
    labels: Vec<Option<f32>>,
}

pub fn distill_training_samples_with_labels_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<DistillTrainingSamplesWithLabelsRequest>(request_json)
        .map_err(|error| format!("Distilled sample request is not valid JSON: {error}"))?;
    serde_json::to_string(&distill_training_samples_with_labels(
        &request.samples,
        &request.labels,
    ))
    .map_err(|error| format!("Distilled sample response failed to encode: {error}"))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultLabelSampleRequest {
    sample: TrainingSample,
    score: Option<i32>,
    first_move: Option<SearchResultMoveJson>,
    label_kind: String,
    label_weight: f32,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultLabelSampleFromResultRequest {
    sample: TrainingSample,
    result: Option<SearchResultLabelResultJson>,
    label_kind: String,
    label_weight: f32,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultLabelResultJson {
    score: Option<i32>,
    moves: Option<Vec<SearchResultMoveJson>>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultTurnRequest {
    result: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultTurnResponse {
    moves: Vec<serde_json::Value>,
    score: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultMoveJson {
    from: SearchResultMovePositionJson,
    to: SearchResultMovePositionJson,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultMovePositionJson {
    timeline_id: i32,
    time: i32,
    x: i32,
    y: i32,
}

pub fn search_result_label_sample(
    mut sample: TrainingSample,
    score: Option<i32>,
    first_move: Option<impl Into<SearchResultPolicyMove>>,
    label_kind: impl Into<String>,
    label_weight: f32,
) -> TrainingSample {
    sample.label = normalized_search_score(score.unwrap_or(0));
    sample.policy = first_move.map(|movement| {
        let movement = movement.into();
        policy_bucket_from_move_values(
            movement.from.timeline_id,
            movement.from.time,
            movement.from.x,
            movement.from.y,
            movement.to.timeline_id,
            movement.to.time,
            movement.to.x,
            movement.to.y,
            0,
        )
    });
    sample.label_kind = Some(label_kind.into());
    sample.label_weight = label_weight;
    sample.pseudo = Some(false);
    sample
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchResultPolicyMove {
    from: SearchResultPolicyPosition,
    to: SearchResultPolicyPosition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SearchResultPolicyPosition {
    timeline_id: i32,
    time: i32,
    x: i32,
    y: i32,
}

impl From<SearchResultMoveJson> for SearchResultPolicyMove {
    fn from(movement: SearchResultMoveJson) -> Self {
        Self {
            from: SearchResultPolicyPosition {
                timeline_id: movement.from.timeline_id,
                time: movement.from.time,
                x: movement.from.x,
                y: movement.from.y,
            },
            to: SearchResultPolicyPosition {
                timeline_id: movement.to.timeline_id,
                time: movement.to.time,
                x: movement.to.x,
                y: movement.to.y,
            },
        }
    }
}

impl From<MoveStep> for SearchResultPolicyMove {
    fn from(movement: MoveStep) -> Self {
        Self {
            from: SearchResultPolicyPosition {
                timeline_id: movement.from.timeline_id,
                time: movement.from.time,
                x: movement.from.x,
                y: movement.from.y,
            },
            to: SearchResultPolicyPosition {
                timeline_id: movement.to.timeline_id,
                time: movement.to.time,
                x: movement.to.x,
                y: movement.to.y,
            },
        }
    }
}

pub fn search_result_label_sample_json(request_json: &str) -> Result<String, String> {
    let request =
        serde_json::from_str::<SearchResultLabelSampleRequest>(request_json).map_err(|error| {
            format!("Search result label sample request is not valid JSON: {error}")
        })?;
    serde_json::to_string(&search_result_label_sample(
        request.sample,
        request.score,
        request.first_move,
        request.label_kind,
        request.label_weight,
    ))
    .map_err(|error| format!("Search result label sample response failed to encode: {error}"))
}

pub fn search_result_label_sample_from_result_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<SearchResultLabelSampleFromResultRequest>(request_json)
        .map_err(|error| {
            format!("Search result label sample-from-result request is not valid JSON: {error}")
        })?;
    let Some(result) = request.result else {
        return Ok("null".to_string());
    };
    let Some(first_move) = result.moves.and_then(|moves| moves.into_iter().next()) else {
        return Ok("null".to_string());
    };
    serde_json::to_string(&Some(search_result_label_sample(
        request.sample,
        result.score,
        Some(first_move),
        request.label_kind,
        request.label_weight,
    )))
    .map_err(|error| {
        format!("Search result label sample-from-result response failed to encode: {error}")
    })
}

pub fn search_result_turn_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<SearchResultTurnRequest>(request_json)
        .map_err(|error| format!("Search result turn request is not valid JSON: {error}"))?;
    let result = request.result.unwrap_or(serde_json::Value::Null);
    let moves = result
        .get("moves")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let score = result
        .get("score")
        .filter(|score| score.is_number())
        .cloned();
    serde_json::to_string(&SearchResultTurnResponse { moves, score })
        .map_err(|error| format!("Search result turn response failed to encode: {error}"))
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingSample {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub side_to_move: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub board_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position_key: Option<String>,
    pub features: Vec<f32>,
    pub label: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_kind: Option<String>,
    #[serde(default = "default_label_weight")]
    pub label_weight: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_label_weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_mass: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pseudo: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationSplit {
    pub train_indices: Vec<usize>,
    pub validation_indices: Vec<usize>,
    pub seed: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseProjectionFeatures {
    pub offsets: Vec<u32>,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    pub byte_length: usize,
}

#[derive(Clone)]
struct ReplayEntry {
    sample: TrainingSample,
    index: usize,
    priority: f32,
}

#[derive(Clone, Debug)]
pub struct SearchLabelSampleRequest {
    pub snapshot_json: Option<String>,
    pub depth: i32,
    pub min_depth: Option<i32>,
    pub nodes: i32,
    pub time_ms: i32,
    pub label_weight: f32,
}

#[derive(Clone, Debug)]
pub struct SearchLabelBatchRequest {
    pub snapshot_json: Option<String>,
    pub mode: SearchLabelMode,
    pub distill_model: Option<CompactValueModel>,
    pub count: usize,
    pub max_plies: usize,
    pub position_depth: i32,
    pub position_nodes: i32,
    pub position_time_ms: i32,
    pub label_depth: i32,
    pub label_min_depth: Option<i32>,
    pub label_nodes: i32,
    pub label_time_ms: i32,
    pub label_weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchLabelMode {
    Search,
    Cpu,
    Curriculum,
    Tactical,
    Distilled,
    Outcome,
    Duel,
}

impl SearchLabelMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "search" => Ok(Self::Search),
            "cpu" | "cpu-search" | "baseline" | "cpu-baseline" => Ok(Self::Cpu),
            "curriculum" => Ok(Self::Curriculum),
            "tactical" => Ok(Self::Tactical),
            "distilled" | "distill" => Ok(Self::Distilled),
            "outcome" | "self" | "self-play" => Ok(Self::Outcome),
            "duel" | "duel-search" | "cpu-gpu" | "vs-cpu" => Ok(Self::Duel),
            _ => Err(format!(
                "Unknown GPU sample mode '{value}'. Expected search, cpu, curriculum, tactical, distilled, outcome, or duel."
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Search => "search",
            Self::Cpu => "cpu",
            Self::Curriculum => "curriculum",
            Self::Tactical => "tactical",
            Self::Distilled => "distilled",
            Self::Outcome => "outcome",
            Self::Duel => "duel",
        }
    }
}

impl Default for SearchLabelSampleRequest {
    fn default() -> Self {
        Self {
            snapshot_json: None,
            depth: crate::cpu::search::DEFAULT_CPU_SEARCH_DEPTH,
            min_depth: Some(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
            nodes: crate::cpu::search::DEFAULT_CPU_SEARCH_NODES,
            time_ms: crate::cpu::search::DEFAULT_CPU_SEARCH_TIME_MS,
            label_weight: 1.0,
        }
    }
}

impl Default for SearchLabelBatchRequest {
    fn default() -> Self {
        Self {
            snapshot_json: None,
            mode: SearchLabelMode::Search,
            distill_model: None,
            count: DEFAULT_SEARCH_SAMPLE_COUNT,
            max_plies: DEFAULT_SEARCH_SAMPLE_MAX_PLIES,
            position_depth: DEFAULT_SEARCH_SAMPLE_POSITION_DEPTH,
            position_nodes: DEFAULT_SEARCH_SAMPLE_POSITION_NODES,
            position_time_ms: DEFAULT_SEARCH_SAMPLE_POSITION_TIME_MS,
            label_depth: crate::cpu::search::DEFAULT_CPU_SEARCH_DEPTH,
            label_min_depth: Some(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
            label_nodes: crate::cpu::search::DEFAULT_CPU_SEARCH_NODES,
            label_time_ms: crate::cpu::search::DEFAULT_CPU_SEARCH_TIME_MS,
            label_weight: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchLabelSampleResponse {
    pub samples: Vec<TrainingSample>,
    pub source: &'static str,
    pub score: i32,
    pub depth: i32,
    pub nodes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchLabelBatchResponse {
    pub samples: Vec<TrainingSample>,
    pub source: &'static str,
    pub requested: usize,
    pub generated_positions: usize,
    pub labeled_positions: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingSearchConfig {
    pub depth: i32,
    pub nodes: i32,
    pub exploration_temperature: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingWorkerSearchConfig {
    pub depth: i32,
    pub nodes: i32,
    pub time_ms: u64,
    pub exploration_temperature: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ValueHeadTrainingConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub weight_decay: f32,
    pub momentum: f32,
}

impl Default for ValueHeadTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 128,
            weight_decay: 0.00001,
            momentum: 0.9,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValueHeadTrainingReport {
    pub initial_loss: f32,
    pub final_loss: f32,
    pub samples: usize,
    pub epochs: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyHeadTrainingReport {
    pub initial_loss: f32,
    pub final_loss: f32,
    pub samples: usize,
    pub steps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputActivation {
    Linear,
    Tanh,
}

impl fmt::Display for OutputActivation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Linear => "linear",
            Self::Tanh => "tanh",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactValueModelSummary {
    pub version: u32,
    pub projection_size: u32,
    pub projection_seed: u32,
    pub hidden_layers: Vec<u32>,
    pub hidden_weight_count: usize,
    pub output_weight_count: usize,
    pub policy_logit_count: usize,
    pub policy_weight_count: usize,
    pub auxiliary_value_weight_count: usize,
    pub scale: f32,
    pub bias: f32,
    pub output_activation: OutputActivation,
}

impl fmt::Display for CompactValueModelSummary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "CFNN version={} projection={} seed={} hidden_layers={:?} hidden_weights={} output_weights={} policy_logits={} policy_weights={} auxiliary_value_weights={} scale={} bias={} activation={}",
            self.version,
            self.projection_size,
            self.projection_seed,
            self.hidden_layers,
            self.hidden_weight_count,
            self.output_weight_count,
            self.policy_logit_count,
            self.policy_weight_count,
            self.auxiliary_value_weight_count,
            self.scale,
            self.bias,
            self.output_activation
        )
    }
}

#[derive(Debug, PartialEq)]
pub enum CompactValueModelError {
    Truncated {
        offset: usize,
        needed: usize,
        len: usize,
    },
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    NonFinite {
        section: &'static str,
        index: usize,
        value: f32,
    },
    TrailingBytes {
        parsed: usize,
        len: usize,
    },
}

impl fmt::Display for CompactValueModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated {
                offset,
                needed,
                len,
            } => write!(
                formatter,
                "compact value model is truncated at byte {offset}; needed {needed} bytes but file has {len}"
            ),
            Self::InvalidMagic(magic) => write!(
                formatter,
                "compact value model has invalid magic {:?}",
                String::from_utf8_lossy(magic)
            ),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported compact value model version {version}")
            }
            Self::NonFinite {
                section,
                index,
                value,
            } => write!(
                formatter,
                "compact value model has non-finite {section}[{index}]={value}"
            ),
            Self::TrailingBytes { parsed, len } => write!(
                formatter,
                "compact value model has trailing bytes: parsed {parsed} of {len}"
            ),
        }
    }
}

impl std::error::Error for CompactValueModelError {}

pub fn load_compact_value_model(path: impl AsRef<Path>) -> Result<CompactValueModel, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to read GPU value model {}: {error}", path.display()))?;
    decode_compact_value_model(&bytes).map_err(|error| {
        format!(
            "failed to parse GPU value model {}: {error}",
            path.display()
        )
    })
}

pub fn load_training_samples_json(path: impl AsRef<Path>) -> Result<Vec<TrainingSample>, String> {
    let path = path.as_ref();
    let json = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "failed to read GPU training samples {}: {error}",
            path.display()
        )
    })?;
    serde_json::from_str(&json).map_err(|error| {
        format!(
            "failed to parse GPU training samples {}: {error}",
            path.display()
        )
    })
}

pub fn save_training_samples_json(
    path: impl AsRef<Path>,
    samples: &[TrainingSample],
) -> Result<(), String> {
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(samples)
        .map_err(|error| format!("failed to encode GPU training samples: {error}"))?;
    std::fs::write(path, json).map_err(|error| {
        format!(
            "failed to write GPU training samples {}: {error}",
            path.display()
        )
    })
}

pub fn append_replay_samples(
    buffer: &[TrainingSample],
    samples: &[TrainingSample],
    max_buffer: usize,
) -> Vec<TrainingSample> {
    let mut combined = Vec::with_capacity(buffer.len() + samples.len());
    combined.extend_from_slice(buffer);
    combined.extend_from_slice(samples);
    let values = dedupe_training_samples(&combined);
    if values.len() <= max_buffer {
        return values;
    }
    let values_len = values.len();
    let mut ranked = values
        .into_iter()
        .enumerate()
        .map(|(index, sample)| {
            let priority = replay_sample_priority(&sample, index, values_len);
            ReplayEntry {
                sample,
                index,
                priority,
            }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .priority
            .total_cmp(&left.priority)
            .then_with(|| right.index.cmp(&left.index))
    });
    let mut selected = ranked.iter().take(max_buffer).cloned().collect::<Vec<_>>();
    let mut selected_indices = selected.iter().map(|entry| entry.index).collect::<Vec<_>>();
    let available_policy_count = ranked
        .iter()
        .filter(|entry| replay_has_policy_target(&entry.sample))
        .count();
    let required_policy_count = available_policy_count
        .min(1usize.max((max_buffer as f32 * MIN_POLICY_REPLAY_FRACTION).ceil() as usize));
    let mut selected_policy_count = selected
        .iter()
        .filter(|entry| replay_has_policy_target(&entry.sample))
        .count();
    for replacement in &ranked {
        if selected_policy_count >= required_policy_count {
            break;
        }
        if !replay_has_policy_target(&replacement.sample)
            || selected_indices.contains(&replacement.index)
        {
            continue;
        }
        let Some(replace_index) = selected
            .iter()
            .rposition(|entry| !replay_has_policy_target(&entry.sample))
        else {
            break;
        };
        if let Some(position) = selected_indices
            .iter()
            .position(|index| *index == selected[replace_index].index)
        {
            selected_indices.remove(position);
        }
        selected[replace_index] = replacement.clone();
        selected_indices.push(replacement.index);
        selected_policy_count += 1;
    }
    selected.sort_by_key(|entry| entry.index);
    selected.into_iter().map(|entry| entry.sample).collect()
}

pub fn dedupe_training_samples(samples: &[TrainingSample]) -> Vec<TrainingSample> {
    let merged = samples
        .iter()
        .filter(|sample| !sample.features.is_empty())
        .cloned()
        .collect::<Vec<_>>();
    let mut deduplicated: Vec<(String, TrainingSample, usize)> = Vec::new();
    let mut legacy_index = 0;
    for (index, sample) in merged.into_iter().enumerate() {
        let key = replay_sample_key(&sample, legacy_index);
        if sample.position_key.is_none() {
            legacy_index += 1;
        }
        if let Some(existing_index) = deduplicated
            .iter()
            .position(|(existing_key, _, _)| existing_key == &key)
        {
            let combined = merge_compatible_samples(&deduplicated[existing_index].1, &sample);
            deduplicated.remove(existing_index);
            deduplicated.push((key, combined, index));
        } else {
            deduplicated.push((key, sample, index));
        }
    }
    deduplicated
        .into_iter()
        .map(|(_, sample, _)| sample)
        .collect()
}

pub fn merge_compatible_samples(
    existing: &TrainingSample,
    incoming: &TrainingSample,
) -> TrainingSample {
    let existing_weight = existing
        .base_label_weight
        .unwrap_or(existing.label_weight)
        .max(0.0);
    let incoming_weight = incoming
        .base_label_weight
        .unwrap_or(incoming.label_weight)
        .max(0.0);
    let existing_mass = existing.label_mass.unwrap_or(existing_weight).max(0.0);
    let incoming_mass = incoming.label_mass.unwrap_or(incoming_weight).max(0.0);
    let total_mass = existing_mass + incoming_mass;
    let existing_count = existing.observation_count.unwrap_or(1).max(1);
    let incoming_count = incoming.observation_count.unwrap_or(1).max(1);
    let observation_count = existing_count + incoming_count;
    let strongest_weight = existing_weight.max(incoming_weight);
    let confidence = 2.0_f32.min((observation_count as f32).sqrt());
    let preferred = if incoming_weight >= existing_weight {
        incoming
    } else {
        existing
    };
    let mut merged = preferred.clone();
    merged.features = incoming.features.clone();
    merged.label = if total_mass > 0.0 {
        (existing.label * existing_mass + incoming.label * incoming_mass) / total_mass
    } else {
        incoming.label
    };
    merged.label_weight = strongest_weight * confidence;
    merged.base_label_weight = Some(strongest_weight);
    merged.label_mass = Some(total_mass.min(64.0));
    merged.observation_count = Some(observation_count.min(64));
    merged.policy = preferred.policy.or(existing.policy).or(incoming.policy);
    merged.pseudo = Some(existing.pseudo.unwrap_or(false) && incoming.pseudo.unwrap_or(false));
    merged
}

pub fn replay_sample_priority(sample: &TrainingSample, index: usize, total: usize) -> f32 {
    let recency = if total > 1 {
        index as f32 / (total - 1) as f32
    } else {
        1.0
    };
    training_label_priority(sample.label_kind.as_deref(), sample.pseudo.unwrap_or(false))
        + sample.label_weight.max(0.0)
        + recency * 0.25
}

pub fn label_source_counts(samples: &[TrainingSample]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for sample in samples {
        let key = sample.label_kind.clone().unwrap_or_else(|| {
            if sample.pseudo.unwrap_or(false) {
                "distilled"
            } else {
                "unknown"
            }
            .to_string()
        });
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

pub fn select_training_working_set(
    samples: &[TrainingSample],
    max_projected_bytes: usize,
) -> Vec<TrainingSample> {
    select_training_working_set_for_projection(
        samples,
        DEFAULT_PROJECTION_SIZE,
        max_projected_bytes,
    )
}

pub fn select_training_working_set_for_projection(
    samples: &[TrainingSample],
    projection_size: usize,
    max_projected_bytes: usize,
) -> Vec<TrainingSample> {
    let projected_bytes = samples
        .len()
        .saturating_mul(projection_size)
        .saturating_mul(std::mem::size_of::<f32>());
    if projected_bytes <= max_projected_bytes {
        return samples.to_vec();
    }
    let bytes_per_sample = projection_size.saturating_mul(std::mem::size_of::<f32>());
    let max_projected_samples = max_projected_bytes
        .checked_div(bytes_per_sample)
        .map(|x| x.max(1usize))
        .unwrap_or(samples.len());
    select_training_working_set_with_capacity(samples, max_projected_samples)
}

pub fn select_training_working_set_indices(
    samples: &[TrainingSample],
    max_projected_bytes: usize,
) -> Vec<usize> {
    select_training_working_set_indices_for_projection(
        samples,
        DEFAULT_PROJECTION_SIZE,
        max_projected_bytes,
    )
}

pub fn select_training_working_set_indices_for_projection(
    samples: &[TrainingSample],
    projection_size: usize,
    max_projected_bytes: usize,
) -> Vec<usize> {
    let projected_bytes = samples
        .len()
        .saturating_mul(projection_size)
        .saturating_mul(std::mem::size_of::<f32>());
    if projected_bytes <= max_projected_bytes {
        return (0..samples.len()).collect();
    }
    let bytes_per_sample = projection_size.saturating_mul(std::mem::size_of::<f32>());
    let max_projected_samples = max_projected_bytes
        .checked_div(bytes_per_sample)
        .map(|x| x.max(1usize))
        .unwrap_or(samples.len());
    select_training_working_set_indices_with_capacity(samples, max_projected_samples)
}

pub fn select_training_working_set_indices_with_capacity(
    samples: &[TrainingSample],
    max_projected_samples: usize,
) -> Vec<usize> {
    select_training_working_set_entries(samples, max_projected_samples)
        .into_iter()
        .map(|entry| entry.index)
        .collect()
}

pub fn select_training_working_set_with_capacity(
    samples: &[TrainingSample],
    max_projected_samples: usize,
) -> Vec<TrainingSample> {
    if samples.len() <= max_projected_samples {
        return samples.to_vec();
    }
    select_training_working_set_entries(samples, max_projected_samples)
        .into_iter()
        .map(|entry| entry.sample)
        .collect()
}

fn select_training_working_set_entries(
    samples: &[TrainingSample],
    max_projected_samples: usize,
) -> Vec<ReplayEntry> {
    if samples.len() <= max_projected_samples {
        return samples
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, sample)| ReplayEntry {
                sample,
                index,
                priority: 0.0,
            })
            .collect();
    }
    let target = 1usize.max(samples.len().min(max_projected_samples));
    let mut ranked = samples
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, sample)| {
            let priority = training_sample_priority(&sample, index, samples.len());
            ReplayEntry {
                sample,
                index,
                priority,
            }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .priority
            .total_cmp(&left.priority)
            .then_with(|| right.index.cmp(&left.index))
    });
    let mut selected = ranked.iter().take(target).cloned().collect::<Vec<_>>();
    let mut selected_indices = selected.iter().map(|entry| entry.index).collect::<Vec<_>>();
    let available_policy_count = ranked
        .iter()
        .filter(|entry| has_policy_training_target(&entry.sample))
        .count();
    let required_policy_count = available_policy_count
        .min(1usize.max((target as f32 * MIN_POLICY_WORKING_SET_FRACTION).ceil() as usize));
    let mut selected_policy_count = selected
        .iter()
        .filter(|entry| has_policy_training_target(&entry.sample))
        .count();
    if selected_policy_count < required_policy_count {
        let policy_replacements = ranked
            .iter()
            .filter(|entry| {
                has_policy_training_target(&entry.sample)
                    && !selected_indices.contains(&entry.index)
            })
            .cloned()
            .collect::<Vec<_>>();
        for replacement in policy_replacements {
            if selected_policy_count >= required_policy_count {
                break;
            }
            let Some(replace_index) = selected
                .iter()
                .rposition(|entry| !has_policy_training_target(&entry.sample))
            else {
                break;
            };
            if let Some(position) = selected_indices
                .iter()
                .position(|index| *index == selected[replace_index].index)
            {
                selected_indices.remove(position);
            }
            selected[replace_index] = replacement;
            selected_indices.push(selected[replace_index].index);
            selected_policy_count += 1;
        }
    }
    selected.sort_by_key(|entry| entry.index);
    selected
}

pub fn training_sample_priority(sample: &TrainingSample, index: usize, total: usize) -> f32 {
    replay_sample_priority(sample, index, total)
}

pub fn split_validation_samples(
    samples: &[TrainingSample],
    validation_split: f32,
) -> ValidationSplit {
    let mut train_indices = Vec::new();
    let mut validation_indices = Vec::new();
    let threshold = (validation_split * 10_000.0).floor().max(0.0) as u32;
    let seed = samples
        .iter()
        .enumerate()
        .fold(2_166_136_261_u32, |hash, (index, sample)| {
            (hash ^ stable_sample_hash(sample, index)).wrapping_mul(16_777_619)
        });
    for (index, sample) in samples.iter().enumerate() {
        let bucket = stable_sample_hash(sample, index) % 10_000;
        if threshold > 0 && bucket < threshold {
            validation_indices.push(index);
        } else {
            train_indices.push(index);
        }
    }
    if validation_split > 0.0 && validation_indices.is_empty() && train_indices.len() > 1 {
        move_position_group_to_validation(
            samples,
            &mut train_indices,
            &mut validation_indices,
            seed,
        );
    }
    if train_indices.is_empty() && !validation_indices.is_empty() {
        move_or_collapse_validation_group(samples, &mut train_indices, &mut validation_indices);
    }
    ValidationSplit {
        train_indices,
        validation_indices,
        seed,
    }
}

pub fn split_policy_training_indices(
    samples: &[TrainingSample],
    policy_indices: &[usize],
    split: &ValidationSplit,
    validation_split: f32,
) -> ValidationSplit {
    let mut train_indices = split
        .train_indices
        .iter()
        .copied()
        .filter(|index| policy_indices.contains(index))
        .collect::<Vec<_>>();
    let mut validation_indices = split
        .validation_indices
        .iter()
        .copied()
        .filter(|index| policy_indices.contains(index))
        .collect::<Vec<_>>();
    if validation_split > 0.0 && validation_indices.is_empty() && train_indices.len() > 1 {
        move_position_group_to_validation(
            samples,
            &mut train_indices,
            &mut validation_indices,
            split.seed,
        );
    }
    if train_indices.is_empty() && !validation_indices.is_empty() {
        move_or_collapse_validation_group(samples, &mut train_indices, &mut validation_indices);
    }
    ValidationSplit {
        train_indices,
        validation_indices,
        seed: split.seed,
    }
}

pub fn move_position_group_to_validation(
    samples: &[TrainingSample],
    train_indices: &mut Vec<usize>,
    validation_indices: &mut Vec<usize>,
    seed: u32,
) {
    let groups = group_training_indices_by_position(samples, train_indices);
    if groups.len() < 2 {
        return;
    }
    let representatives = groups
        .iter()
        .filter_map(|group| group.first().copied())
        .collect::<Vec<_>>();
    let selected_offset = fallback_validation_offset(samples, &representatives, seed);
    let Some(selected_group) = groups.get(selected_offset) else {
        return;
    };
    let mut offset = train_indices.len();
    while offset > 0 {
        offset -= 1;
        if selected_group.contains(&train_indices[offset]) {
            validation_indices.push(train_indices.remove(offset));
        }
    }
    validation_indices.sort_unstable();
}

pub fn move_or_collapse_validation_group(
    samples: &[TrainingSample],
    train_indices: &mut Vec<usize>,
    validation_indices: &mut Vec<usize>,
) {
    let groups = group_training_indices_by_position(samples, validation_indices);
    if groups.len() < 2 {
        train_indices.append(validation_indices);
        return;
    }
    let Some(selected_group) = groups.first() else {
        return;
    };
    let mut offset = validation_indices.len();
    while offset > 0 {
        offset -= 1;
        if selected_group.contains(&validation_indices[offset]) {
            train_indices.push(validation_indices.remove(offset));
        }
    }
    train_indices.sort_unstable();
}

pub fn stable_sample_hash(sample: &TrainingSample, _index: usize) -> u32 {
    let mut hash = 2_166_136_261_u32;
    for unit in training_position_identity(sample).encode_utf16() {
        hash ^= u32::from(unit);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

pub fn shuffled_indices(indices: &[usize], epoch: u32, seed: u32) -> Vec<usize> {
    let mut result = indices.to_vec();
    let mut state = seed ^ epoch.wrapping_mul(2_654_435_761);
    for index in (1..result.len()).rev() {
        state = xorshift32(state);
        let swap_index = state as usize % (index + 1);
        result.swap(index, swap_index);
    }
    result
}

pub fn shuffled_indices_bytes(indices: &[usize], epoch: u32, seed: u32) -> Result<Vec<u8>, String> {
    let shuffled = shuffled_indices(indices, epoch, seed);
    let mut bytes = Vec::with_capacity(shuffled.len() * 4);
    for index in shuffled {
        let index = u32::try_from(index)
            .map_err(|_| "Shuffled training index exceeds GPU parameter range.".to_string())?;
        push_u32(&mut bytes, index);
    }
    Ok(bytes)
}

pub fn group_training_indices_by_position(
    samples: &[TrainingSample],
    indices: &[usize],
) -> Vec<Vec<usize>> {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for &index in indices {
        let Some(sample) = samples.get(index) else {
            continue;
        };
        let identity = training_position_identity(sample);
        if let Some((_, group)) = groups.iter_mut().find(|(key, _)| *key == identity) {
            group.push(index);
        } else {
            groups.push((identity, vec![index]));
        }
    }
    groups.into_iter().map(|(_, group)| group).collect()
}

pub fn unique_training_position_count(samples: &[TrainingSample], indices: &[usize]) -> usize {
    let mut identities: Vec<String> = Vec::new();
    for &index in indices {
        let Some(sample) = samples.get(index) else {
            continue;
        };
        let identity = training_position_identity(sample);
        if !identities.iter().any(|existing| existing == &identity) {
            identities.push(identity);
        }
    }
    identities.len()
}

pub fn fill_grouped_training_batch_indices(
    batch: &mut [u32],
    train_groups: &[Vec<usize>],
    epoch: u32,
    seed: u32,
    label_weights: &[f32],
) -> Result<f32, String> {
    if train_groups.is_empty() {
        return Err("Training requires at least one train position.".to_string());
    }
    let mut state = seed ^ epoch.wrapping_mul(2_654_435_761);
    let mut batch_weight = 0.0;
    for slot in batch.iter_mut() {
        state = xorshift32(if state == 0 { 1 } else { state });
        let group = &train_groups[state as usize % train_groups.len()];
        if group.is_empty() {
            return Err("Training position group must not be empty.".to_string());
        }
        state = xorshift32(if state == 0 { 1 } else { state });
        let selected = group[state as usize % group.len()];
        *slot = selected as u32;
        batch_weight += label_weights.get(selected).copied().unwrap_or(1.0).max(0.0);
    }
    Ok(batch_weight)
}

pub fn fill_grouped_training_batch_indices_bytes(request: &[u8]) -> Result<Vec<u8>, String> {
    struct Cursor<'a> {
        bytes: &'a [u8],
        offset: usize,
    }

    impl<'a> Cursor<'a> {
        fn read_u32(&mut self, label: &str) -> Result<u32, String> {
            let end = self
                .offset
                .checked_add(4)
                .ok_or_else(|| "Grouped batch request is too large.".to_string())?;
            let bytes = self
                .bytes
                .get(self.offset..end)
                .ok_or_else(|| format!("Grouped batch request is missing {label}."))?;
            self.offset = end;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
        }

        fn read_f32(&mut self, label: &str) -> Result<f32, String> {
            let end = self
                .offset
                .checked_add(4)
                .ok_or_else(|| "Grouped batch request is too large.".to_string())?;
            let bytes = self
                .bytes
                .get(self.offset..end)
                .ok_or_else(|| format!("Grouped batch request is missing {label}."))?;
            self.offset = end;
            Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
        }
    }

    let mut cursor = Cursor {
        bytes: request,
        offset: 0,
    };
    let batch_len = cursor.read_u32("batch length")? as usize;
    let group_count = cursor.read_u32("group count")? as usize;
    let item_count = cursor.read_u32("group item count")? as usize;
    let label_weight_count = cursor.read_u32("label weight count")? as usize;
    let epoch = cursor.read_u32("epoch")?;
    let seed = cursor.read_u32("seed")?;
    let mut offsets = Vec::with_capacity(group_count + 1);
    for index in 0..=group_count {
        offsets.push(cursor.read_u32(&format!("group offset {index}"))? as usize);
    }
    if offsets.first().copied().unwrap_or(0) != 0
        || offsets.last().copied().unwrap_or(0) != item_count
    {
        return Err("Grouped batch request has inconsistent group offsets.".to_string());
    }
    let mut items = Vec::with_capacity(item_count);
    for index in 0..item_count {
        items.push(cursor.read_u32(&format!("group item {index}"))? as usize);
    }
    let mut label_weights = Vec::with_capacity(label_weight_count);
    for index in 0..label_weight_count {
        label_weights.push(cursor.read_f32(&format!("label weight {index}"))?);
    }
    if cursor.offset != request.len() {
        return Err("Grouped batch request has trailing bytes.".to_string());
    }
    let mut groups = Vec::with_capacity(group_count);
    for group_index in 0..group_count {
        let start = offsets[group_index];
        let end = offsets[group_index + 1];
        if start > end || end > items.len() {
            return Err("Grouped batch request has invalid group offsets.".to_string());
        }
        groups.push(items[start..end].to_vec());
    }
    let mut batch = vec![0; batch_len];
    let batch_weight =
        fill_grouped_training_batch_indices(&mut batch, &groups, epoch, seed, &label_weights)?;
    let mut bytes = Vec::with_capacity(8 + batch.len() * 4);
    push_f32(&mut bytes, batch_weight);
    push_u32(&mut bytes, batch.len() as u32);
    for index in batch {
        push_u32(&mut bytes, index);
    }
    Ok(bytes)
}

pub fn xorshift32(value: u32) -> u32 {
    let mut state = value;
    state ^= state.wrapping_shl(13);
    state ^= state >> 17;
    state ^= state.wrapping_shl(5);
    state
}

pub fn feature_length(samples: &[TrainingSample]) -> Result<usize, String> {
    let Some(length) = samples.first().map(|sample| sample.features.len()) else {
        return Err("Training samples have inconsistent feature lengths.".to_string());
    };
    if length == 0 || samples.iter().any(|sample| sample.features.len() != length) {
        return Err("Training samples have inconsistent feature lengths.".to_string());
    }
    Ok(length)
}

pub fn split_work(total: usize, workers: usize) -> Vec<usize> {
    if workers == 0 {
        return Vec::new();
    }
    (0..workers)
        .map(|index| total / workers + usize::from(index < total % workers))
        .filter(|count| *count > 0)
        .collect()
}

pub fn take_training_sample_batches(
    batches: &[Vec<TrainingSample>],
    target: usize,
) -> Vec<TrainingSample> {
    if target == 0 {
        return Vec::new();
    }
    batches
        .iter()
        .flat_map(|batch| batch.iter().cloned())
        .take(target)
        .collect()
}

pub fn compact_training_samples(samples: &[Option<TrainingSample>]) -> Vec<TrainingSample> {
    samples.iter().filter_map(Clone::clone).collect()
}

pub fn gpu_training_worker_count(total: usize, requested_workers: usize) -> usize {
    total.min(requested_workers.clamp(1, MAX_PARALLEL_GPU_TRAINING_WORKERS))
}

pub fn gpu_duel_training_worker_count(
    total: usize,
    search_workers: usize,
    self_play_workers: usize,
) -> usize {
    gpu_training_worker_count(total, search_workers.min(self_play_workers))
}

pub fn training_label_worker_count(
    job_count: usize,
    requested_workers: Option<usize>,
    hardware_cores: usize,
) -> usize {
    if job_count == 0 {
        return 0;
    }
    let auto_workers = hardware_cores
        .max(4)
        .saturating_sub(1)
        .clamp(1, MAX_PARALLEL_GPU_TRAINING_WORKERS);
    let requested = requested_workers.unwrap_or(auto_workers);
    job_count.min(requested.clamp(1, 8))
}

pub fn sample_plies(index: usize, encode_only: bool) -> usize {
    let stride = if encode_only { 2 } else { 1 };
    1 + ((index * stride) % MAX_PLAYOUT_PLIES)
}

pub fn gpu_warmup_plies(worker_index: usize) -> usize {
    if worker_index == 0 {
        0
    } else {
        1 + (worker_index % MAX_PLAYOUT_PLIES.saturating_sub(1).max(1))
    }
}

pub fn gpu_rollout_max_plies(target: usize, worker_index: usize) -> usize {
    MAX_PLAYOUT_PLIES.max(target.saturating_add(worker_index))
}

pub fn gpu_rollout_ply_offset(ply: usize, worker_index: usize) -> usize {
    ply.saturating_add(worker_index.saturating_mul(gpu_rollout_max_plies(0, 0)))
}

pub fn gpu_warmup_search_config(
    depth: i32,
    nodes: i32,
    search_time_ms: u64,
    exploration_temperature: f32,
) -> TrainingWorkerSearchConfig {
    TrainingWorkerSearchConfig {
        depth: depth.clamp(1, 2),
        nodes: nodes.clamp(1, 1024),
        time_ms: search_time_ms.min(GPU_WARMUP_MAX_TIME_MS),
        exploration_temperature,
    }
}

pub fn gpu_position_generation_search_config(
    depth: i32,
    nodes: i32,
    exploration_temperature: f32,
) -> TrainingWorkerSearchConfig {
    TrainingWorkerSearchConfig {
        depth: depth.clamp(1, 2),
        nodes: nodes.clamp(1, 512),
        time_ms: GPU_POSITION_GENERATION_TIME_MS,
        exploration_temperature,
    }
}

pub fn worker_request_timeout_ms(nodes: i64, time_ms: i64) -> u64 {
    let nodes = nodes.max(1) as u64;
    let time_ms = time_ms.max(0) as u64;
    let upper = LABEL_REQUEST_MAX_TIMEOUT_MS.max(time_ms + 5_000);
    let lower = LABEL_REQUEST_MIN_TIMEOUT_MS
        .max(time_ms + 1_000)
        .max(nodes.saturating_mul(LABEL_REQUEST_NODE_TIMEOUT_FACTOR_MS));
    upper.min(lower)
}

pub fn worker_search_time_ms(nodes: i64, time_ms: i64) -> u64 {
    worker_request_timeout_ms(nodes, time_ms)
        .saturating_sub(1_000)
        .max(1_000)
}

pub fn worker_request_timeout_ms_json(request_json: &str) -> Result<u64, String> {
    let (nodes, time_ms) = worker_timeout_request_values(request_json)?;
    Ok(worker_request_timeout_ms(nodes, time_ms))
}

pub fn worker_search_time_ms_json(request_json: &str) -> Result<u64, String> {
    let (nodes, time_ms) = worker_timeout_request_values(request_json)?;
    Ok(worker_search_time_ms(nodes, time_ms))
}

pub fn loss_log_replay_logs_json(request_json: &str) -> Result<String, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct LossLogReplayRequest {
        logs: Option<Vec<serde_json::Value>>,
        limit: Option<f64>,
    }

    let request = serde_json::from_str::<LossLogReplayRequest>(request_json)
        .map_err(|error| format!("loss-log replay request is invalid: {error}"))?;
    let limit = request
        .limit
        .filter(|limit| limit.is_finite())
        .map(|limit| limit.floor().max(0.0) as usize)
        .unwrap_or(0);
    let logs = request
        .logs
        .unwrap_or_default()
        .into_iter()
        .filter(|log| {
            log.get("decisions")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|decisions| !decisions.is_empty())
        })
        .take(limit)
        .collect::<Vec<_>>();
    serde_json::to_string(&logs)
        .map_err(|error| format!("loss-log replay response failed to encode: {error}"))
}

pub fn loss_log_validation_update_json(request_json: &str) -> Result<String, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct LossLogValidationUpdateRequest {
        validation: Option<serde_json::Value>,
        event: Option<String>,
        example: Option<serde_json::Value>,
    }

    let request = serde_json::from_str::<LossLogValidationUpdateRequest>(request_json)
        .map_err(|error| format!("loss-log validation update request is invalid: {error}"))?;
    let mut validation = loss_log_validation_value(request.validation.as_ref());
    match request.event.as_deref().unwrap_or("finalize") {
        "skip" => increment_json_counter(&mut validation, "skipped"),
        "unchanged" => {
            increment_json_counter(&mut validation, "checked");
            increment_json_counter(&mut validation, "unchanged");
        }
        "changed" => {
            increment_json_counter(&mut validation, "checked");
            increment_json_counter(&mut validation, "changed");
            if let Some(example) = request.example {
                if let Some(examples) = validation
                    .entry("examples".to_string())
                    .or_insert_with(|| serde_json::json!([]))
                    .as_array_mut()
                {
                    examples.push(example);
                }
            }
        }
        "finalize" => {}
        event => return Err(format!("loss-log validation event is invalid: {event}")),
    }
    let checked = json_counter(&validation, "checked");
    let changed = json_counter(&validation, "changed");
    validation.insert(
        "failed".to_string(),
        serde_json::json!(checked > 0 && changed == 0),
    );
    serde_json::to_string(&serde_json::Value::Object(validation))
        .map_err(|error| format!("loss-log validation update failed to encode: {error}"))
}

pub fn training_metrics_summary_json(request_json: &str) -> Result<String, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TrainingMetricsSummaryRequest {
        started_at: Option<f64>,
        now_ms: Option<f64>,
        phases: Option<serde_json::Map<String, serde_json::Value>>,
        sample_counts: Option<serde_json::Map<String, serde_json::Value>>,
        search_position_count: Option<f64>,
        search_label_count: Option<f64>,
        loss_log_validation: Option<serde_json::Value>,
    }

    let request = serde_json::from_str::<TrainingMetricsSummaryRequest>(request_json)
        .map_err(|error| format!("training metrics summary request is invalid: {error}"))?;
    let phases = request.phases.unwrap_or_default();
    let mut rounded_phases = serde_json::Map::new();
    for (name, value) in &phases {
        rounded_phases.insert(
            name.clone(),
            serde_json::json!(round_millis(json_number(value).unwrap_or(0.0))),
        );
    }

    let mut sample_rates = serde_json::Map::new();
    for (kind, count) in request.sample_counts.unwrap_or_default() {
        let phase_name = format!("{kind}Labels");
        let phase_ms = phases
            .get(&phase_name)
            .and_then(json_number)
            .or_else(|| phases.get("collect").and_then(json_number))
            .unwrap_or(0.0);
        if phase_ms > 0.0 {
            sample_rates.insert(
                kind,
                serde_json::json!(round_rate(json_number(&count).unwrap_or(0.0), phase_ms)),
            );
        }
    }

    if let (Some(count), Some(phase_ms)) = (
        finite_positive(request.search_position_count),
        phases
            .get("searchPositions")
            .and_then(json_number)
            .filter(|value| *value > 0.0),
    ) {
        sample_rates.insert(
            "searchPositions".to_string(),
            serde_json::json!(round_rate(count, phase_ms)),
        );
    }
    if let (Some(count), Some(phase_ms)) = (
        finite_positive(request.search_label_count),
        phases
            .get("searchLabels")
            .and_then(json_number)
            .filter(|value| *value > 0.0),
    ) {
        sample_rates.insert(
            "searchLabels".to_string(),
            serde_json::json!(round_rate(count, phase_ms)),
        );
    }

    let started_at = request
        .started_at
        .filter(|value| value.is_finite())
        .unwrap_or(0.0);
    let now_ms = request
        .now_ms
        .filter(|value| value.is_finite())
        .unwrap_or(started_at);
    serde_json::to_string(&serde_json::json!({
        "totalMs": round_millis(now_ms - started_at),
        "phases": rounded_phases,
        "sampleRates": sample_rates,
        "lossLogValidation": request.loss_log_validation.unwrap_or(serde_json::Value::Null),
    }))
    .map_err(|error| format!("training metrics summary response failed to encode: {error}"))
}

fn json_number(value: &serde_json::Value) -> Option<f64> {
    value.as_f64().filter(|value| value.is_finite())
}

fn finite_positive(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite() && *value > 0.0)
}

fn round_millis(value: f64) -> i64 {
    value.round() as i64
}

fn round_rate(count: f64, phase_ms: f64) -> f64 {
    ((count / (phase_ms / 1000.0)) * 100.0).round() / 100.0
}

fn loss_log_validation_value(
    value: Option<&serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut validation = serde_json::Map::new();
    if let Some(existing) = value.and_then(serde_json::Value::as_object) {
        validation.extend(
            existing
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    for key in ["checked", "changed", "unchanged", "skipped"] {
        validation
            .entry(key.to_string())
            .or_insert_with(|| serde_json::json!(0));
    }
    validation
        .entry("failed".to_string())
        .or_insert_with(|| serde_json::json!(false));
    validation
        .entry("examples".to_string())
        .or_insert_with(|| serde_json::json!([]));
    validation
}

fn json_counter(validation: &serde_json::Map<String, serde_json::Value>, key: &str) -> u64 {
    validation
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
}

fn increment_json_counter(validation: &mut serde_json::Map<String, serde_json::Value>, key: &str) {
    let value = json_counter(validation, key).saturating_add(1);
    validation.insert(key.to_string(), serde_json::json!(value));
}

fn worker_timeout_request_values(request_json: &str) -> Result<(i64, i64), String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct WorkerTimeoutRequest {
        nodes: Option<serde_json::Value>,
        time_ms: Option<serde_json::Value>,
    }

    let request = serde_json::from_str::<WorkerTimeoutRequest>(request_json)
        .map_err(|error| format!("training worker timeout request is invalid: {error}"))?;
    let nodes = browser_number_value(request.nodes.as_ref())
        .filter(|value| *value != 0.0)
        .unwrap_or(1.0);
    let time_ms = browser_number_value(request.time_ms.as_ref())
        .filter(|value| *value != 0.0)
        .unwrap_or(0.0);
    Ok((truncate_f64_to_i64(nodes)?, truncate_f64_to_i64(time_ms)?))
}

fn browser_number_value(value: Option<&serde_json::Value>) -> Option<f64> {
    match value? {
        serde_json::Value::Number(number) => number.as_f64().filter(|value| value.is_finite()),
        serde_json::Value::String(text) => {
            text.parse::<f64>().ok().filter(|value| value.is_finite())
        }
        serde_json::Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn truncate_f64_to_i64(value: f64) -> Result<i64, String> {
    let value = value.trunc();
    if value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err("training worker timeout value exceeds i64 range.".to_string());
    }
    Ok(value as i64)
}

pub fn clamp_training_number(value: Option<f64>, min: f64, max: f64, fallback: f64) -> f64 {
    let Some(value) = value.filter(|value| value.is_finite()) else {
        return fallback;
    };
    value.max(min).min(max)
}

pub fn clamp_training_integer(value: Option<f64>, min: i64, max: i64, fallback: i64) -> i64 {
    js_round(clamp_training_number(
        value,
        min as f64,
        max as f64,
        fallback as f64,
    )) as i64
}

fn js_round(value: f64) -> f64 {
    (value + 0.5).floor()
}

pub fn sample_seed(prefix: &str, index: u32, salt: u32) -> u32 {
    let mut hash = salt;
    for unit in prefix.encode_utf16() {
        hash ^= u32::from(unit);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash ^= index;
    hash.wrapping_mul(16_777_619)
}

pub fn search_seed_json(json_text: Option<&str>, salt: u32) -> u32 {
    let mut hash = salt;
    let text = json_text.unwrap_or("null");
    for unit in text.encode_utf16() {
        hash ^= u32::from(unit);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

pub fn tactical_position_priority_from_counts(
    checked_royal_count: usize,
    active_timeline_count: usize,
    timeline_count: usize,
    royal_exposure: usize,
    temporal_power_piece_count: usize,
) -> i32 {
    let mut priority = 0;
    priority += (checked_royal_count * 2).min(3) as i32;
    priority += active_timeline_count.saturating_sub(1) as i32;
    priority += timeline_count.saturating_sub(2) as i32;
    priority += royal_exposure.min(2) as i32;
    priority += i32::from(temporal_power_piece_count > 1);
    priority
}

pub fn curriculum_stage(index: usize) -> usize {
    index % 6
}

pub fn curriculum_timeline_limit(stage: usize, timeline_count: usize) -> usize {
    let limit = if stage <= 1 {
        1
    } else if stage <= 3 {
        2
    } else {
        2.max(timeline_count.min(4))
    };
    timeline_count.min(limit)
}

pub fn curriculum_board_times(board_times: &[i32], present_time: i32, stage: usize) -> Vec<i32> {
    if board_times.is_empty() {
        return Vec::new();
    }

    let latest = board_times.iter().copied().max();
    let present = board_times
        .iter()
        .rev()
        .copied()
        .find(|time| *time == present_time);
    let mut candidates = Vec::new();
    if stage <= 1 {
        if let Some(time) = present.or(latest) {
            candidates.push(time);
        }
    } else if stage <= 3 {
        candidates.extend(
            board_times[board_times.len().saturating_sub(2)..]
                .iter()
                .copied(),
        );
        candidates.extend(present);
    } else {
        candidates.extend(
            board_times[board_times.len().saturating_sub(stage + 1)..]
                .iter()
                .copied(),
        );
    }

    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

pub fn curriculum_piece_type(piece_type: &str, stage: usize) -> Option<String> {
    if stage != 0 {
        return Some(piece_type.to_string());
    }
    match piece_type {
        "king" | "queen" | "rook" | "bishop" | "knight" | "pawn" => Some(piece_type.to_string()),
        "royalQueen" => Some("queen".to_string()),
        _ => None,
    }
}

pub fn curriculum_timeline_priority(
    has_present_board: bool,
    active: bool,
    latest_time: Option<i32>,
) -> f64 {
    let has_present = if has_present_board { 4.0 } else { 0.0 };
    let active = if active { 2.0 } else { 0.0 };
    let latest = latest_time.map_or(f64::NEG_INFINITY, |time| time as f64 / 1000.0);
    has_present + active + latest
}

pub fn curriculum_search_config(
    depth: i32,
    nodes: i32,
    exploration_temperature: f32,
    index: usize,
) -> TrainingSearchConfig {
    let stage = curriculum_stage(index);
    TrainingSearchConfig {
        depth: depth.clamp(1, 1 + (stage / 2) as i32),
        nodes: nodes.clamp(1, 512 * (stage as i32 + 1)),
        exploration_temperature: exploration_temperature
            .max(if stage >= 3 { 0.35 } else { 0.15 })
            .min(0.6),
    }
}

pub fn curriculum_game_snapshot_json(snapshot_json: &str, index: usize) -> Result<String, String> {
    let game = parse_game_snapshot(snapshot_json)?;
    Ok(curriculum_game(&game, index).to_json())
}

pub fn tactical_search_config(
    depth: i32,
    nodes: i32,
    exploration_temperature: f32,
    attempt: usize,
) -> TrainingSearchConfig {
    TrainingSearchConfig {
        depth: depth.min(3 + attempt as i32).max(2),
        nodes: nodes.min(2048 * (attempt as i32 + 1)).max(1024),
        exploration_temperature: exploration_temperature
            .max(0.4 + attempt as f32 * 0.1)
            .min(0.8),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TacticalPositionSelection {
    pub use_generated: bool,
    pub next_priority: i32,
    pub complete: bool,
}

pub fn tactical_position_attempt_count(index: usize) -> usize {
    1 + (index % 4)
}

pub fn tactical_position_use_best_source(best_priority: i32) -> bool {
    best_priority > 0
}

pub fn tactical_position_selection(
    best_priority: i32,
    generated_priority: i32,
) -> TacticalPositionSelection {
    let use_generated = generated_priority > best_priority;
    TacticalPositionSelection {
        use_generated,
        next_priority: if use_generated {
            generated_priority
        } else {
            best_priority
        },
        complete: generated_priority >= 4,
    }
}

pub fn tactical_position_selection_json(
    best_priority: i32,
    generated_priority: i32,
) -> Result<String, String> {
    serde_json::to_string(&tactical_position_selection(
        best_priority,
        generated_priority,
    ))
    .map_err(|error| format!("Tactical position selection response failed to encode: {error}"))
}

pub fn is_training_subject(value: &str) -> bool {
    matches!(value, "gpu" | "cpu")
}

pub fn is_training_mode(value: &str) -> bool {
    matches!(
        value,
        "vsGpu" | "vsCpu" | "self" | "distill" | "curriculum" | "tactical"
    )
}

pub fn legacy_training_subject(training_target: Option<&str>) -> &'static str {
    if training_target == Some("trainCpu") {
        "cpu"
    } else {
        "gpu"
    }
}

pub fn legacy_training_modes(
    subject: &str,
    training_target: Option<&str>,
    cpu_training_target: Option<&str>,
    label_mode: Option<&str>,
) -> Vec<String> {
    if subject == "cpu" {
        return match cpu_training_target {
            Some("vsGpu") => vec!["vsGpu".to_string()],
            Some("vsBoth") => vec!["vsCpu".to_string(), "vsGpu".to_string()],
            _ => vec!["vsCpu".to_string()],
        };
    }

    let mut modes = Vec::new();
    if matches!(training_target, Some("trainCpu" | "trainBoth")) {
        modes.push("vsCpu".to_string());
    }
    if training_target != Some("trainCpu")
        && (training_target == Some("trainBoth")
            || matches!(label_mode, Some("mixed" | "search") | None))
    {
        modes.push("vsGpu".to_string());
    }
    if training_target != Some("trainCpu")
        && matches!(label_mode, Some("mixed" | "selfPlay") | None)
    {
        modes.push("self".to_string());
    }
    if training_target != Some("trainCpu") && matches!(label_mode, Some("mixed" | "distill") | None)
    {
        modes.push("distill".to_string());
    }
    if training_target != Some("trainCpu") && label_mode == Some("mixed") {
        modes.push("curriculum".to_string());
        modes.push("tactical".to_string());
    }
    modes
}

pub fn normalize_training_modes(
    explicit_modes: &[&str],
    subject: &str,
    training_target: Option<&str>,
    cpu_training_target: Option<&str>,
    label_mode: Option<&str>,
) -> Vec<String> {
    let legacy_modes;
    let source: Vec<&str> = if explicit_modes.is_empty() {
        legacy_modes =
            legacy_training_modes(subject, training_target, cpu_training_target, label_mode);
        legacy_modes.iter().map(String::as_str).collect()
    } else {
        explicit_modes
            .iter()
            .copied()
            .filter(|mode| is_training_mode(mode))
            .collect()
    };

    let mut deduped = Vec::<String>::new();
    for mode in source {
        if subject == "cpu" && mode == "distill" {
            continue;
        }
        if !deduped.iter().any(|existing| existing == mode) {
            deduped.push(mode.to_string());
        }
    }
    if deduped.is_empty() {
        if subject == "cpu" {
            vec!["vsCpu".to_string()]
        } else {
            vec!["vsGpu".to_string(), "self".to_string()]
        }
    } else {
        deduped
    }
}

pub fn training_mode_enabled(modes: &[String], mode: &str) -> bool {
    modes.iter().any(|candidate| candidate == mode)
}

pub fn cpu_baseline_mode_enabled(modes: &[String]) -> bool {
    training_mode_enabled(modes, "vsCpu") || training_mode_enabled(modes, "self")
}

pub fn training_mode_count(subject: &str, modes: &[String]) -> usize {
    if subject == "cpu" {
        modes
            .iter()
            .filter(|mode| mode.as_str() != "distill")
            .count()
    } else {
        modes.len()
    }
}

pub fn outcome_label_for_turns(
    winner: &str,
    outcome_turn: &str,
    ply: usize,
    max_ply: usize,
) -> Result<f32, String> {
    let winner = parse_training_color(winner)?;
    let outcome_turn = parse_training_color(outcome_turn)?;
    let sign = if outcome_turn == winner { 1.0 } else { -1.0 };
    Ok(sign * OUTCOME_LABEL_DECAY.powi(max_ply.saturating_sub(ply) as i32))
}

pub fn apply_outcome_label(
    sample: &mut TrainingSample,
    winner: &str,
    outcome_turn: &str,
    ply: usize,
    max_ply: usize,
) -> Result<(), String> {
    sample.label = outcome_label_for_turns(winner, outcome_turn, ply, max_ply)?;
    sample.label_kind = Some("outcome".to_string());
    sample.label_weight = OUTCOME_LABEL_WEIGHT;
    Ok(())
}

pub fn apply_draw_label(sample: &mut TrainingSample, label_kind: &str, label_weight: f32) {
    sample.label = 0.0;
    sample.label_kind = Some(label_kind.to_string());
    sample.label_weight = label_weight;
}

pub fn samples_from_partial_outcome(
    samples: &[TrainingSample],
    label_kind: Option<&str>,
    label_weight: Option<f32>,
) -> Vec<TrainingSample> {
    let label_kind = label_kind.unwrap_or(DEFAULT_PARTIAL_OUTCOME_LABEL_KIND);
    let label_weight = label_weight.unwrap_or(DEFAULT_PARTIAL_OUTCOME_LABEL_WEIGHT);
    samples
        .iter()
        .cloned()
        .map(|mut sample| {
            sample.label_kind = Some(label_kind.to_string());
            sample.label_weight = label_weight;
            sample
        })
        .collect()
}

pub fn tactical_position_priority_snapshot_json(snapshot_json: &str) -> Result<i32, String> {
    let game = parse_game_snapshot(snapshot_json)?;
    Ok(tactical_position_priority(&game))
}

pub fn royal_count_snapshot_json(snapshot_json: &str, color: &str) -> Result<usize, String> {
    let game = parse_game_snapshot(snapshot_json)?;
    let color = parse_training_color(color)?;
    Ok(royal_count(&game, color))
}

pub fn royal_capture_winner_snapshot_json(
    before_json: &str,
    after_json: &str,
    mover: &str,
) -> Result<Option<&'static str>, String> {
    let before = parse_game_snapshot(before_json)?;
    let after = parse_game_snapshot(after_json)?;
    let mover = parse_training_color(mover)?;
    Ok(royal_capture_winner(&before, &after, mover).map(color_name))
}

pub(crate) fn tactical_position_priority(game: &Game) -> i32 {
    tactical_position_priority_from_counts(
        game.checked_royal_positions().len(),
        game.timelines
            .iter()
            .filter(|timeline| game.is_active_timeline(timeline.id))
            .count(),
        game.timelines.len(),
        royal_exposure(game),
        temporal_power_piece_count(game),
    )
}

pub(crate) fn royal_capture_winner(before: &Game, after: &Game, mover: Color) -> Option<Color> {
    let opponent = match mover {
        Color::White => Color::Black,
        Color::Black => Color::White,
    };
    (royal_count(after, opponent) < royal_count(before, opponent)).then_some(mover)
}

pub(crate) fn royal_count(game: &Game, color: Color) -> usize {
    game.timelines
        .iter()
        .filter_map(|timeline| timeline.boards.last())
        .flat_map(|board| board.board.iter().flatten())
        .filter(|piece| {
            piece.is_some_and(|piece| {
                piece.color == color
                    && matches!(piece.piece_type, PieceType::King | PieceType::RoyalQueen)
            })
        })
        .count()
}

pub(crate) fn royal_exposure(game: &Game) -> usize {
    game.timelines
        .iter()
        .filter(|timeline| game.is_active_timeline(timeline.id))
        .filter_map(|timeline| timeline.boards.last())
        .map(|board| {
            board
                .board
                .iter()
                .flatten()
                .filter(|piece| {
                    piece.is_some_and(|piece| {
                        matches!(piece.piece_type, PieceType::King | PieceType::RoyalQueen)
                    })
                })
                .count()
        })
        .sum::<usize>()
        .min(2)
}

pub(crate) fn temporal_power_piece_count(game: &Game) -> usize {
    game.timelines
        .iter()
        .filter_map(|timeline| timeline.boards.last())
        .map(|board| {
            board
                .board
                .iter()
                .flatten()
                .filter(|piece| {
                    piece.is_some_and(|piece| {
                        matches!(
                            piece.piece_type,
                            PieceType::Queen
                                | PieceType::RoyalQueen
                                | PieceType::Unicorn
                                | PieceType::Dragon
                        )
                    })
                })
                .count()
        })
        .sum()
}

fn parse_training_color(color: &str) -> Result<Color, String> {
    match color {
        "white" => Ok(Color::White),
        "black" => Ok(Color::Black),
        other => Err(format!(
            "Training color must be white or black, got `{other}`."
        )),
    }
}

fn color_name(color: Color) -> &'static str {
    match color {
        Color::White => "white",
        Color::Black => "black",
    }
}

pub fn pack_sparse_projection_features(
    samples: &[TrainingSample],
    input_size: Option<usize>,
) -> Result<SparseProjectionFeatures, String> {
    let input_size = match input_size {
        Some(input_size) => input_size,
        None => feature_length(samples)?,
    };
    let rows = samples
        .iter()
        .map(|sample| {
            if sample.features.len() != input_size {
                Err("Training samples have inconsistent feature lengths.".to_string())
            } else {
                Ok(sample.features.as_slice())
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    pack_sparse_feature_rows(&rows, input_size)
}

pub fn sparse_projection_features_bytes(
    samples: &[TrainingSample],
    input_size: Option<usize>,
) -> Result<Vec<u8>, String> {
    let packed = pack_sparse_projection_features(samples, input_size)?;
    let offsets_len = u32::try_from(packed.offsets.len())
        .map_err(|_| "sparse projection offset count exceeds GPU parameter range".to_string())?;
    let indices_len = u32::try_from(packed.indices.len())
        .map_err(|_| "sparse projection index count exceeds GPU parameter range".to_string())?;
    let values_len = u32::try_from(packed.values.len())
        .map_err(|_| "sparse projection value count exceeds GPU parameter range".to_string())?;
    let byte_length = u32::try_from(packed.byte_length)
        .map_err(|_| "sparse projection byte length exceeds GPU parameter range".to_string())?;
    let mut bytes = Vec::with_capacity(
        16 + (packed.offsets.len() + packed.indices.len() + packed.values.len()) * 4,
    );
    push_u32(&mut bytes, offsets_len);
    push_u32(&mut bytes, indices_len);
    push_u32(&mut bytes, values_len);
    push_u32(&mut bytes, byte_length);
    for value in packed.offsets {
        push_u32(&mut bytes, value);
    }
    for value in packed.indices {
        push_u32(&mut bytes, value);
    }
    for value in packed.values {
        push_f32(&mut bytes, value);
    }
    Ok(bytes)
}

pub fn pack_sparse_feature_rows(
    features: &[&[f32]],
    input_size: usize,
) -> Result<SparseProjectionFeatures, String> {
    let mut offsets = Vec::with_capacity(features.len() + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    offsets.push(0);
    for row in features {
        if row.len() != input_size {
            return Err("Training samples have inconsistent feature lengths.".to_string());
        }
        for (index, value) in row.iter().copied().enumerate().take(input_size) {
            if value != 0.0 {
                indices.push(
                    index
                        .try_into()
                        .map_err(|_| "feature index exceeds GPU parameter range".to_string())?,
                );
                values.push(value);
            }
        }
        offsets.push(
            indices
                .len()
                .try_into()
                .map_err(|_| "nonzero feature count exceeds GPU parameter range".to_string())?,
        );
    }
    if indices.is_empty() {
        indices.push(0);
        values.push(0.0);
    }
    let byte_length = offsets.len() * std::mem::size_of::<u32>()
        + indices.len() * std::mem::size_of::<u32>()
        + values.len() * std::mem::size_of::<f32>();
    Ok(SparseProjectionFeatures {
        offsets,
        indices,
        values,
        byte_length,
    })
}

fn fallback_validation_offset(
    samples: &[TrainingSample],
    train_indices: &[usize],
    seed: u32,
) -> usize {
    let mut best_offset = 0;
    let mut best_priority = f32::NEG_INFINITY;
    for (offset, &sample_index) in train_indices.iter().enumerate() {
        let Some(sample) = samples.get(sample_index) else {
            continue;
        };
        let priority = validation_sample_priority(sample, sample_index, seed);
        if priority > best_priority {
            best_priority = priority;
            best_offset = offset;
        }
    }
    best_offset
}

fn replay_has_policy_target(sample: &TrainingSample) -> bool {
    sample.label_kind.as_deref() != Some("distilled") && sample.policy.is_some()
}

fn replay_sample_key(sample: &TrainingSample, legacy_index: usize) -> String {
    let label_kind = sample.label_kind.as_deref().unwrap_or("unknown");
    if let Some(position_key) = sample
        .position_key
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        return format!("{position_key}|{label_kind}");
    }
    if let Some(fingerprint) = replay_feature_fingerprint(&sample.features) {
        return format!(
            "{fingerprint}|{}|{}|{label_kind}",
            sample.side_to_move.as_deref().unwrap_or_default(),
            sample.board_count.unwrap_or(0)
        );
    }
    format!("legacy:{legacy_index}")
}

fn replay_feature_fingerprint(features: &[f32]) -> Option<String> {
    if features.is_empty() {
        return None;
    }
    let mut hash = 2_166_136_261_u32;
    let mut non_zero = 0usize;
    for (index, value) in features.iter().copied().enumerate() {
        if value == 0.0 {
            continue;
        }
        non_zero += 1;
        hash ^= index as u32;
        hash = hash.wrapping_mul(16_777_619);
        hash ^= (value * 1024.0).round() as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    (non_zero > 0).then(|| format!("features:{}:{hash:x}", features.len()))
}

fn validation_sample_priority(sample: &TrainingSample, index: usize, seed: u32) -> f32 {
    let hash_tie_break = (stable_sample_hash(sample, index) ^ seed) as f32 / u32::MAX as f32;
    training_label_priority(sample.label_kind.as_deref(), sample.pseudo.unwrap_or(false))
        + sample.label_weight.max(0.0)
        + hash_tie_break * 0.001
}

pub fn training_label_priority(label_kind: Option<&str>, pseudo: bool) -> f32 {
    match label_kind {
        Some("outcome" | "duel") => 4.0,
        Some("search" | "cpu") => 3.0,
        Some("duel-search" | "search-bootstrap") => 2.0,
        Some("distilled") => 1.0,
        _ if pseudo => 1.0,
        _ => 2.0,
    }
}

fn training_position_identity(sample: &TrainingSample) -> String {
    let side_to_move = sample.side_to_move.as_deref().unwrap_or_default();
    let board_count = sample.board_count.unwrap_or(0);
    if let Some(position_key) = sample
        .position_key
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        format!("{position_key}|{side_to_move}|{board_count}")
    } else {
        format!(
            "{}|{side_to_move}|{board_count}",
            feature_fingerprint(&sample.features)
        )
    }
}

fn feature_fingerprint(features: &[f32]) -> String {
    let mut hash = 2_166_136_261_u32;
    for (index, value) in features.iter().copied().enumerate() {
        if value == 0.0 {
            continue;
        }
        hash ^= index as u32;
        hash = hash.wrapping_mul(16_777_619);
        hash ^= (value * 1024.0).round() as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    format!("{hash:x}")
}

pub(crate) fn sample_from_game_label(game: &Game, label: f32, label_weight: f32) -> TrainingSample {
    let encoded = game.encode_neural_position(game.turn_color());
    TrainingSample {
        side_to_move: Some(game.turn_color().as_str().to_string()),
        board_count: Some(encoded.board_count),
        position_key: Some(game.position_key()),
        features: encoded.values,
        label: bounded_value(label),
        label_kind: None,
        label_weight,
        base_label_weight: None,
        label_mass: None,
        observation_count: None,
        policy: None,
        pseudo: None,
    }
}

pub fn sample_from_snapshot_label(
    snapshot_json: Option<&str>,
    label: f32,
    label_weight: f32,
) -> Result<TrainingSample, String> {
    let game = match snapshot_json {
        Some(snapshot) => parse_game_snapshot(snapshot)?,
        None => Game::new(),
    };
    Ok(sample_from_game_label(&game, label, label_weight))
}

pub fn search_label_sample(
    request: SearchLabelSampleRequest,
) -> Result<SearchLabelSampleResponse, String> {
    let game = match request.snapshot_json.as_deref() {
        Some(snapshot) => parse_game_snapshot(snapshot)?,
        None => Game::new(),
    };
    let (sample, score, depth, nodes) = search_label_sample_from_game(
        &game,
        request.depth,
        request.min_depth,
        request.nodes,
        request.time_ms,
        request.label_weight,
    );
    Ok(SearchLabelSampleResponse {
        samples: vec![sample],
        source: "heuristic-search",
        score,
        depth,
        nodes,
    })
}

pub fn collect_search_label_samples(
    request: SearchLabelBatchRequest,
) -> Result<SearchLabelBatchResponse, String> {
    let base = match request.snapshot_json.as_deref() {
        Some(snapshot) => parse_game_snapshot(snapshot)?,
        None => Game::new(),
    };
    let requested = request.count;
    if request.mode == SearchLabelMode::Distilled {
        let Some(model) = request.distill_model.as_ref() else {
            return Err("distilled sample mode requires a compact value model".to_string());
        };
        let mut generated_positions = 0;
        let mut samples = Vec::with_capacity(requested);
        for index in 0..requested {
            let game = generate_label_position(&base, index, &request);
            generated_positions += 1;
            samples.push(sample_from_game_label(&game, 0.0, DISTILLED_LABEL_WEIGHT));
        }
        let samples = distill_training_samples(&samples, model);
        let labeled_positions = samples.len();
        return Ok(SearchLabelBatchResponse {
            samples,
            source: request.mode.source_name(),
            requested,
            generated_positions,
            labeled_positions,
        });
    }
    if matches!(
        request.mode,
        SearchLabelMode::Outcome | SearchLabelMode::Duel
    ) {
        let samples = match request.mode {
            SearchLabelMode::Outcome => outcome_label_samples(&base, &request),
            SearchLabelMode::Duel => duel_label_samples(&base, &request),
            _ => unreachable!("terminal sample modes handled above"),
        };
        let labeled_positions = samples.len();
        return Ok(SearchLabelBatchResponse {
            samples,
            source: request.mode.source_name(),
            requested,
            generated_positions: requested,
            labeled_positions,
        });
    }
    let mut generated_positions = 0;
    let mut samples = Vec::with_capacity(requested);
    for index in 0..requested {
        let game = generate_label_position(&base, index, &request);
        generated_positions += 1;
        let (mut sample, _, _, _) = search_label_sample_from_game(
            &game,
            request.label_depth,
            request.label_min_depth,
            request.label_nodes,
            request.label_time_ms,
            request.mode.label_weight(request.label_weight, &game),
        );
        sample.label_kind = Some(request.mode.as_str().to_string());
        if sample.policy.is_some() {
            samples.push(sample);
        }
    }
    let labeled_positions = samples.len();
    Ok(SearchLabelBatchResponse {
        samples,
        source: request.mode.source_name(),
        requested,
        generated_positions,
        labeled_positions,
    })
}

/// Collect search labels through the engine's GPU-search API. This is kept
/// separate from `collect_search_label_samples`, whose explicit non-search
/// modes intentionally retain their CPU/self-play behavior.
pub fn collect_native_gpu_search_label_samples(
    request: SearchLabelBatchRequest,
    model_path: &str,
) -> Result<SearchLabelBatchResponse, String> {
    if request.mode != SearchLabelMode::Search {
        return Err("native GPU search labels require sample mode `search`".to_string());
    }
    let base = match request.snapshot_json.as_deref() {
        Some(snapshot) => parse_game_snapshot(snapshot)?,
        None => Game::new(),
    };
    let requested = request.count;
    let mut samples = Vec::with_capacity(requested);
    for index in 0..requested {
        let game = generate_label_position(&base, index, &request);
        let response = crate::gpu::search::search(crate::gpu::search::GpuSearchRequest {
            snapshot_json: Some(game.to_json()),
            model_path: Some(model_path.to_string()),
            depth: request.label_depth,
            min_depth: request.label_min_depth,
            nodes: request.label_nodes,
            time_ms: request.label_time_ms,
        })?;
        let sample = sample_from_game_label(&game, 0.0, request.label_weight);
        let labeled = search_result_label_sample_from_result_json(
            &serde_json::json!({
                "sample": sample,
                "result": serde_json::from_str::<serde_json::Value>(&response.result_json)
                    .map_err(|error| format!("native GPU search result is not valid JSON: {error}"))?,
                "labelKind": "gpu-search",
                "labelWeight": request.label_weight,
            })
            .to_string(),
        )?;
        if let Some(sample) = serde_json::from_str::<Option<TrainingSample>>(&labeled)
            .map_err(|error| format!("native GPU search label is not valid JSON: {error}"))?
        {
            samples.push(sample);
        }
    }
    let labeled_positions = samples.len();
    Ok(SearchLabelBatchResponse {
        samples,
        source: "native-gpu-search-batch",
        requested,
        generated_positions: requested,
        labeled_positions,
    })
}

fn search_label_sample_from_game(
    game: &Game,
    depth: i32,
    min_depth: Option<i32>,
    nodes: i32,
    time_ms: i32,
    label_weight: f32,
) -> (TrainingSample, i32, i32, usize) {
    let depth = depth.max(1);
    let min_depth = min_depth
        .unwrap_or(Game::DEFAULT_MIN_AI_SEARCH_DEPTH)
        .max(1);
    let nodes = nodes.max(1);
    let deadline = search_deadline(time_ms.max(1));
    let (result, _) = game.best_ai_turn_with_options_min_depth(
        depth,
        min_depth,
        nodes,
        deadline,
        SearchOptions::optimized(),
        Some("gpu-training-search-label"),
    );
    let label = result.score as f32 / VALUE_SCORE_SCALE;
    let mut sample = sample_from_game_label(game, label, label_weight);
    sample.label_kind = Some("search".to_string());
    sample.policy = result.moves.first().map(policy_bucket);
    sample.pseudo = Some(false);
    (sample, result.score, result.depth, result.nodes)
}

fn outcome_label_samples(base: &Game, request: &SearchLabelBatchRequest) -> Vec<TrainingSample> {
    let mut game = base.clone();
    let max_plies = request.max_plies.max(request.count).max(1);
    let mut samples = Vec::with_capacity(request.count);
    let mut outcome_turns = Vec::with_capacity(request.count);
    let mut plies = Vec::with_capacity(request.count);
    for ply in 0..max_plies {
        if samples.len() >= request.count || game.result.is_some() {
            break;
        }
        let before_turn = game.turn_color();
        let (result, _) = game.best_ai_turn_with_options_min_depth(
            request.label_depth.max(1),
            request.label_min_depth.unwrap_or(1).max(1),
            request.label_nodes.max(1),
            search_deadline(request.label_time_ms.max(1)),
            SearchOptions::optimized(),
            Some("gpu-training-outcome-rollout"),
        );
        if result.moves.is_empty() {
            break;
        }
        let mut sample = sample_from_game_label(
            &game,
            result.score as f32 / VALUE_SCORE_SCALE,
            OUTCOME_LABEL_WEIGHT,
        );
        sample.label_kind = Some("outcome".to_string());
        sample.policy = result.moves.first().map(policy_bucket);
        sample.pseudo = Some(false);
        samples.push(sample);
        outcome_turns.push(before_turn);
        plies.push(ply);

        let plan = TurnPlan {
            moves: result.moves,
            score_hint: result.score,
        };
        let Some(next) = game.apply_turn_plan_for_search(&plan) else {
            break;
        };
        game = next;
        if let Some(result) = game.result {
            if let Some(winner) = result.winner {
                let max_ply = plies.last().copied().unwrap_or(0);
                for ((sample, outcome_turn), ply) in samples
                    .iter_mut()
                    .zip(outcome_turns.iter())
                    .zip(plies.iter())
                {
                    let _ = apply_outcome_label(
                        sample,
                        color_name(winner),
                        color_name(*outcome_turn),
                        *ply,
                        max_ply,
                    );
                }
            } else {
                for sample in &mut samples {
                    apply_draw_label(sample, "outcome", 1.0);
                }
            }
            return samples;
        }
    }
    samples_from_partial_outcome(&samples, None, None)
}

fn duel_label_samples(base: &Game, request: &SearchLabelBatchRequest) -> Vec<TrainingSample> {
    let mut game = base.clone();
    let max_plies = request.max_plies.max(request.count).max(1);
    let cpu_color = game.turn_color();
    let mut samples = Vec::with_capacity(request.count);
    let mut outcome_turns = Vec::with_capacity(request.count);
    let mut plies = Vec::with_capacity(request.count);
    for ply in 0..max_plies {
        if samples.len() >= request.count || game.result.is_some() {
            break;
        }
        let before_turn = game.turn_color();
        let use_cpu = before_turn == cpu_color;
        let depth = if use_cpu {
            request.position_depth
        } else {
            request.label_depth
        }
        .max(1);
        let min_depth = if use_cpu {
            1
        } else {
            request.label_min_depth.unwrap_or(1).max(1)
        };
        let nodes = if use_cpu {
            request.position_nodes
        } else {
            request.label_nodes
        }
        .max(1);
        let time_ms = if use_cpu {
            request.position_time_ms
        } else {
            request.label_time_ms
        }
        .max(1);
        let (result, _) = game.best_ai_turn_with_options_min_depth(
            depth,
            min_depth,
            nodes,
            search_deadline(time_ms),
            SearchOptions::optimized(),
            Some(if use_cpu {
                "gpu-training-duel-cpu-rollout"
            } else {
                "gpu-training-duel-gpu-rollout"
            }),
        );
        if result.moves.is_empty() {
            break;
        }
        let mut sample = sample_from_game_label(
            &game,
            result.score as f32 / VALUE_SCORE_SCALE,
            DUEL_LABEL_WEIGHT,
        );
        sample.label_kind = Some("duel".to_string());
        sample.policy = result.moves.first().map(policy_bucket);
        sample.pseudo = Some(false);
        samples.push(sample);
        outcome_turns.push(before_turn);
        plies.push(ply);

        let plan = TurnPlan {
            moves: result.moves,
            score_hint: result.score,
        };
        let Some(next) = game.apply_turn_plan_for_search(&plan) else {
            break;
        };
        game = next;
        if let Some(result) = game.result {
            if let Some(winner) = result.winner {
                let max_ply = plies.last().copied().unwrap_or(0);
                for ((sample, outcome_turn), ply) in samples
                    .iter_mut()
                    .zip(outcome_turns.iter())
                    .zip(plies.iter())
                {
                    if let Ok(label) = outcome_label_for_turns(
                        color_name(winner),
                        color_name(*outcome_turn),
                        *ply,
                        max_ply,
                    ) {
                        sample.label = label;
                        sample.label_kind = Some("duel".to_string());
                        sample.label_weight = DUEL_LABEL_WEIGHT;
                    }
                }
            } else {
                for sample in &mut samples {
                    apply_draw_label(sample, "duel", DUEL_DRAW_LABEL_WEIGHT);
                }
            }
            return samples;
        }
    }
    samples_from_partial_outcome(&samples, Some("duel-search"), Some(1.0))
}

fn generate_label_position(base: &Game, index: usize, request: &SearchLabelBatchRequest) -> Game {
    match request.mode {
        SearchLabelMode::Search => playout_search_position(base, index, request),
        SearchLabelMode::Cpu => playout_search_position(base, index, request),
        SearchLabelMode::Curriculum => {
            let generated = playout_search_position_with_config(
                base,
                index,
                request,
                curriculum_search_config(
                    request.position_depth,
                    request.position_nodes,
                    0.0,
                    index,
                ),
            );
            curriculum_game(&generated, index)
        }
        SearchLabelMode::Tactical => tactical_search_position(base, index, request),
        SearchLabelMode::Distilled => playout_search_position(base, index, request),
        SearchLabelMode::Outcome => base.clone(),
        SearchLabelMode::Duel => base.clone(),
    }
}

fn playout_search_position(base: &Game, index: usize, request: &SearchLabelBatchRequest) -> Game {
    playout_search_position_with_config(
        base,
        index,
        request,
        TrainingSearchConfig {
            depth: request.position_depth,
            nodes: request.position_nodes,
            exploration_temperature: 0.0,
        },
    )
}

fn playout_search_position_with_config(
    base: &Game,
    index: usize,
    request: &SearchLabelBatchRequest,
    config: TrainingSearchConfig,
) -> Game {
    let mut game = base.clone();
    let plies = if request.max_plies == 0 {
        0
    } else {
        1 + (index % request.max_plies)
    };
    for _ in 0..plies {
        let (result, _) = game.best_ai_turn_with_options_min_depth(
            config.depth.max(1),
            1,
            config.nodes.max(1),
            search_deadline(request.position_time_ms.max(1)),
            SearchOptions::optimized(),
            Some("gpu-training-position-playout"),
        );
        let plan = TurnPlan {
            moves: result.moves,
            score_hint: result.score,
        };
        let Some(next) = game.apply_turn_plan_for_search(&plan) else {
            break;
        };
        game = next;
    }
    game
}

fn tactical_search_position(base: &Game, index: usize, request: &SearchLabelBatchRequest) -> Game {
    let mut best = base.clone();
    let mut best_priority = tactical_position_priority(&best);
    let attempts = tactical_position_attempt_count(index);
    for attempt in 0..attempts {
        let generated = playout_search_position_with_config(
            if tactical_position_use_best_source(best_priority) {
                &best
            } else {
                base
            },
            index + attempt * request.max_plies.max(1),
            request,
            tactical_search_config(request.position_depth, request.position_nodes, 0.0, attempt),
        );
        let priority = tactical_position_priority(&generated);
        let selection = tactical_position_selection(best_priority, priority);
        if selection.use_generated {
            best = generated;
            best_priority = selection.next_priority;
        }
        if selection.complete {
            break;
        }
    }
    best
}

fn curriculum_game(game: &Game, index: usize) -> Game {
    let stage = curriculum_stage(index);
    let present_time = game.present_time().unwrap_or(0);
    let mut timelines = game
        .timelines
        .iter()
        .filter_map(|timeline| {
            let boards = curriculum_timeline_boards(timeline, present_time, stage);
            (!boards.is_empty()).then(|| {
                let mut timeline = timeline.clone();
                timeline.boards = boards;
                timeline
            })
        })
        .collect::<Vec<_>>();
    timelines.sort_by(|left, right| {
        let left_priority = curriculum_timeline_priority_for_game(game, left, present_time);
        let right_priority = curriculum_timeline_priority_for_game(game, right, present_time);
        right_priority.total_cmp(&left_priority)
    });
    let timeline_limit = curriculum_timeline_limit(stage, timelines.len());
    timelines.truncate(timeline_limit);
    for (row, timeline) in timelines.iter_mut().enumerate() {
        timeline.row = row as i32;
    }
    let mut output = game.clone_for_search();
    output.timelines = timelines;
    output.staged_turn.clear();
    output.staged_notation.clear();
    output.staged_royal_capture_by = None;
    output.position_hash = output.recompute_position_hash();
    output
}

fn curriculum_timeline_boards(
    timeline: &Timeline,
    present_time: i32,
    stage: usize,
) -> Vec<BoardSnapshot> {
    let board_times = timeline
        .boards
        .iter()
        .map(|board| board.time)
        .collect::<Vec<_>>();
    let selected_times = curriculum_board_times(&board_times, present_time, stage);
    selected_times
        .iter()
        .filter_map(|time| {
            let mut board = timeline
                .boards
                .iter()
                .find(|board| board.time == *time)?
                .clone();
            if stage == 0 {
                board.board = board.board.map(|row| {
                    row.map(|piece| piece.and_then(|piece| curriculum_piece(piece, stage)))
                });
            }
            Some(board)
        })
        .collect()
}

fn curriculum_piece(piece: Piece, stage: usize) -> Option<Piece> {
    match curriculum_piece_type(training_piece_type_name(piece.piece_type), stage)?.as_str() {
        "king" => Some(Piece {
            piece_type: PieceType::King,
            ..piece
        }),
        "queen" => Some(Piece {
            piece_type: PieceType::Queen,
            ..piece
        }),
        "rook" => Some(Piece {
            piece_type: PieceType::Rook,
            ..piece
        }),
        "bishop" => Some(Piece {
            piece_type: PieceType::Bishop,
            ..piece
        }),
        "knight" => Some(Piece {
            piece_type: PieceType::Knight,
            ..piece
        }),
        "pawn" => Some(Piece {
            piece_type: PieceType::Pawn,
            ..piece
        }),
        _ => None,
    }
}

fn training_piece_type_name(piece_type: PieceType) -> &'static str {
    match piece_type {
        PieceType::King => "king",
        PieceType::CommonKing => "commonKing",
        PieceType::Queen => "queen",
        PieceType::RoyalQueen => "royalQueen",
        PieceType::Princess => "princess",
        PieceType::Rook => "rook",
        PieceType::Bishop => "bishop",
        PieceType::Unicorn => "unicorn",
        PieceType::Dragon => "dragon",
        PieceType::Knight => "knight",
        PieceType::Pawn => "pawn",
        PieceType::Brawn => "brawn",
    }
}

fn curriculum_timeline_priority_for_game(
    game: &Game,
    timeline: &Timeline,
    present_time: i32,
) -> f64 {
    curriculum_timeline_priority(
        timeline
            .boards
            .iter()
            .any(|board| board.time == present_time),
        game.is_active_timeline(timeline.id),
        timeline.boards.last().map(|board| board.time),
    )
}

impl SearchLabelMode {
    fn source_name(self) -> &'static str {
        match self {
            Self::Search => "heuristic-search-batch",
            Self::Cpu => "heuristic-cpu-batch",
            Self::Curriculum => "heuristic-curriculum-batch",
            Self::Tactical => "heuristic-tactical-batch",
            Self::Distilled => "heuristic-distilled-batch",
            Self::Outcome => "heuristic-outcome-batch",
            Self::Duel => "heuristic-duel-batch",
        }
    }

    fn label_weight(self, base: f32, game: &Game) -> f32 {
        match self {
            Self::Search => base,
            Self::Cpu => base,
            Self::Curriculum => base * 1.05,
            Self::Tactical => base * 1.6 * (1.0 + tactical_position_priority(game) as f32 * 0.2),
            Self::Distilled => DISTILLED_LABEL_WEIGHT,
            Self::Outcome => OUTCOME_LABEL_WEIGHT,
            Self::Duel => DUEL_LABEL_WEIGHT,
        }
    }
}

pub(crate) fn policy_bucket(step: &MoveStep) -> u32 {
    policy_bucket_from_move_values(
        step.from.timeline_id,
        step.from.time,
        step.from.x,
        step.from.y,
        step.to.timeline_id,
        step.to.time,
        step.to.x,
        step.to.y,
        0,
    )
}

pub fn policy_bucket_from_move_values(
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
    let values = [
        to_timeline_id - from_timeline_id,
        to_time - from_time,
        to_x - from_x,
        to_y - from_y,
        from_x,
        from_y,
        intent,
    ];
    policy_bucket_from_values(values)
}

pub fn policy_bucket_from_values(values: [i32; 7]) -> u32 {
    let mut hash = 2_166_136_261_u32;
    for value in values {
        let bits = value as u32;
        for shift in (0..32).step_by(8) {
            hash ^= (bits >> shift) & 0xff;
            hash = hash.wrapping_mul(16_777_619);
        }
    }
    hash % POLICY_BUCKETS
}

pub fn save_compact_value_model(
    path: impl AsRef<Path>,
    model: &CompactValueModel,
) -> Result<(), String> {
    let path = path.as_ref();
    std::fs::write(path, model.encode()).map_err(|error| {
        format!(
            "failed to write GPU value model {}: {error}",
            path.display()
        )
    })
}

pub fn encode_compact_value_model(model: &CompactValueModel) -> Vec<u8> {
    let policy_values = compact_value_model_policy_values(model);
    let mut bytes = Vec::with_capacity(compact_value_model_encoded_len(model));
    bytes.extend_from_slice(COMPACT_VALUE_MODEL_MAGIC);
    push_u32(&mut bytes, model.version);
    push_u32(&mut bytes, model.projection_size);
    push_u32(&mut bytes, model.projection_seed);
    push_u32(&mut bytes, model.hidden_layers.len() as u32);
    push_u32(&mut bytes, model.output_weights.len() as u32);
    if model.version >= 2 {
        push_u32(&mut bytes, policy_values.len() as u32);
    }
    if model.version >= 5 {
        push_u32(&mut bytes, model.auxiliary_value_weights.len() as u32);
    }
    push_f32(&mut bytes, model.scale);
    push_f32(&mut bytes, model.bias);
    for &layer in &model.hidden_layers {
        push_u32(&mut bytes, layer);
    }
    push_u32(&mut bytes, model.hidden_weights.len() as u32);
    for value in model
        .hidden_weights
        .iter()
        .chain(model.output_weights.iter())
        .chain(policy_values.iter())
        .chain(model.auxiliary_value_weights.iter())
    {
        push_f32(&mut bytes, *value);
    }
    bytes
}

pub fn compact_value_model_policy_values(model: &CompactValueModel) -> &[f32] {
    match model.version {
        2 => &model.policy_logits,
        3.. => &model.policy_weights,
        _ => &[],
    }
}

pub fn compact_value_model_encoded_len(model: &CompactValueModel) -> usize {
    let optional_policy_size = usize::from(model.version >= 2) * 4;
    let optional_auxiliary_size = usize::from(model.version >= 5) * 4;
    let float_count = 2
        + model.hidden_weights.len()
        + model.output_weights.len()
        + compact_value_model_policy_values(model).len()
        + model.auxiliary_value_weights.len();
    4 + 4 * 6
        + optional_policy_size
        + optional_auxiliary_size
        + 4 * model.hidden_layers.len()
        + 4 * float_count
}

pub fn train_value_head_cpu(
    model: &CompactValueModel,
    samples: &[TrainingSample],
    config: ValueHeadTrainingConfig,
) -> Result<(CompactValueModel, ValueHeadTrainingReport), String> {
    feature_length(samples)?;
    let features = samples
        .iter()
        .map(|sample| hidden_features(&sample.features, model))
        .collect::<Vec<_>>();
    train_value_head_from_features_cpu(model, samples, &features, config)
}

pub fn train_value_head_from_features_cpu(
    model: &CompactValueModel,
    samples: &[TrainingSample],
    features: &[Vec<f32>],
    config: ValueHeadTrainingConfig,
) -> Result<(CompactValueModel, ValueHeadTrainingReport), String> {
    if samples.is_empty() {
        return Err("GPU value-head training requires at least one sample.".to_string());
    }
    if features.len() != samples.len() {
        return Err(format!(
            "GPU value-head training got {} feature rows for {} samples.",
            features.len(),
            samples.len()
        ));
    }
    let mut trained = model.clone();
    let output_size = trained
        .hidden_layers
        .last()
        .map(|value| *value as usize)
        .unwrap_or(trained.projection_size as usize);
    if trained.output_weights.len() != output_size + 1 {
        return Err(format!(
            "GPU value model output head has {} weights but expected {}.",
            trained.output_weights.len(),
            output_size + 1
        ));
    }
    for (index, features) in features.iter().enumerate() {
        if features.len() != output_size {
            return Err(format!(
                "GPU value-head training feature row {index} has {} values but expected {output_size}.",
                features.len()
            ));
        }
    }
    let mut output_weights = trained.output_weights.clone();
    let mut velocity = vec![0.0; output_weights.len()];
    let initial_loss = value_head_loss(features, samples, &output_weights);
    let mut best_loss = initial_loss;
    let mut best_output_weights = output_weights.clone();
    for _ in 0..config.epochs.max(1) {
        apply_value_head_gradient(
            features,
            samples,
            &mut output_weights,
            &mut velocity,
            config,
        );
        let loss = value_head_loss(features, samples, &output_weights);
        if loss.is_finite() && loss < best_loss {
            best_loss = loss;
            best_output_weights.clone_from(&output_weights);
        }
    }
    trained.output_weights = best_output_weights;
    Ok((
        trained,
        ValueHeadTrainingReport {
            initial_loss,
            final_loss: best_loss,
            samples: samples.len(),
            epochs: config.epochs.max(1),
        },
    ))
}

pub fn train_policy_head_cpu(
    model: &CompactValueModel,
    samples: &[TrainingSample],
    config: ValueHeadTrainingConfig,
) -> Result<(CompactValueModel, PolicyHeadTrainingReport), String> {
    feature_length(samples)?;
    let features = samples
        .iter()
        .map(|sample| hidden_features(&sample.features, model))
        .collect::<Vec<_>>();
    train_policy_head_from_features_cpu(model, samples, &features, config)
}

pub fn train_policy_head_from_features_cpu(
    model: &CompactValueModel,
    samples: &[TrainingSample],
    features: &[Vec<f32>],
    config: ValueHeadTrainingConfig,
) -> Result<(CompactValueModel, PolicyHeadTrainingReport), String> {
    if features.len() != samples.len() {
        return Err(format!(
            "GPU policy-head training got {} feature rows for {} samples.",
            features.len(),
            samples.len()
        ));
    }
    let mut trained = model.clone();
    let input_size = model
        .hidden_layers
        .last()
        .map(|value| *value as usize)
        .unwrap_or(model.projection_size as usize);
    for (index, features) in features.iter().enumerate() {
        if features.len() != input_size {
            return Err(format!(
                "GPU policy-head training feature row {index} has {} values but expected {input_size}.",
                features.len()
            ));
        }
    }
    let expected_weights = POLICY_BUCKETS as usize * (input_size + 1);
    let mut weights = if trained.policy_weights.len() == expected_weights {
        trained.policy_weights.clone()
    } else {
        vec![0.0; expected_weights]
    };
    let indices = samples
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| {
            has_policy_training_target(sample)
                .then_some(index)
                .filter(|_| sample.label_weight.max(0.0) > 0.0)
        })
        .collect::<Vec<_>>();
    if indices.is_empty() {
        trained.policy_weights = weights;
        return Ok((
            trained,
            PolicyHeadTrainingReport {
                initial_loss: f32::NAN,
                final_loss: f32::NAN,
                samples: 0,
                steps: 0,
            },
        ));
    }
    let mut velocity = vec![0.0; weights.len()];
    let initial_loss = policy_head_loss(features, samples, &weights, &indices, input_size);
    let mut best_loss = initial_loss;
    let mut best_weights = weights.clone();
    let steps = policy_training_steps(config.epochs);
    for _ in 0..steps {
        apply_policy_head_gradient(
            features,
            samples,
            &indices,
            &mut weights,
            &mut velocity,
            input_size,
            config,
        );
        let loss = policy_head_loss(features, samples, &weights, &indices, input_size);
        if loss.is_finite() && loss + 1e-6 < best_loss {
            best_loss = loss;
            best_weights.clone_from(&weights);
        }
    }
    trained.policy_weights = best_weights;
    Ok((
        trained,
        PolicyHeadTrainingReport {
            initial_loss,
            final_loss: best_loss,
            samples: indices.len(),
            steps,
        },
    ))
}

pub fn predict_value(features: &[f32], model: &CompactValueModel) -> f32 {
    let activations = hidden_features(features, model);
    let mut prediction = model
        .output_weights
        .get(activations.len())
        .copied()
        .unwrap_or(0.0);
    for (input, activation) in activations.iter().enumerate() {
        prediction += activation * model.output_weights.get(input).copied().unwrap_or(0.0);
    }
    let activated = match model.output_activation {
        OutputActivation::Linear => prediction,
        OutputActivation::Tanh => prediction.tanh(),
    };
    bounded_value(activated * model.scale + model.bias)
}

pub fn hidden_features(features: &[f32], model: &CompactValueModel) -> Vec<f32> {
    let activations = project_features(
        features,
        model.projection_size as usize,
        model.projection_seed,
    );
    hidden_features_from_projected(activations, model)
}

pub fn hidden_features_from_projected(
    mut activations: Vec<f32>,
    model: &CompactValueModel,
) -> Vec<f32> {
    let mut weight_offset = 0usize;
    for &output_size in &model.hidden_layers {
        let output_size = output_size as usize;
        let input_size = activations.len();
        let row_size = input_size + 1;
        let mut next = Vec::with_capacity(output_size);
        for output in 0..output_size {
            let row = weight_offset + output * row_size;
            let mut sum = model
                .hidden_weights
                .get(row + input_size)
                .copied()
                .unwrap_or(0.0);
            for (input, activation) in activations.iter().enumerate() {
                sum += activation
                    * model
                        .hidden_weights
                        .get(row + input)
                        .copied()
                        .unwrap_or(0.0);
            }
            next.push(sum.max(0.0));
        }
        weight_offset += output_size * row_size;
        activations = next;
    }
    activations
}

pub fn project_features(features: &[f32], projection_size: usize, seed: u32) -> Vec<f32> {
    let active: Vec<(usize, f32)> = features
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value != 0.0)
        .collect();
    let mut projected = vec![0.0; projection_size];
    if active.is_empty() {
        return projected;
    }
    let scale = (active.len() as f32).sqrt();
    for (output, projected_value) in projected.iter_mut().enumerate() {
        let mut sum = 0.0;
        for &(input, value) in &active {
            let sign = if projection_hash(input as u32, output as u32, seed) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            sum += value * sign / scale;
        }
        *projected_value = sum;
    }
    projected
}

pub fn projection_hash(raw_index: u32, projection_index: u32, seed: u32) -> u32 {
    let mut hash = seed ^ raw_index;
    hash = hash.wrapping_mul(16_777_619);
    hash ^= projection_index;
    hash = hash.wrapping_mul(16_777_619);
    hash ^ (hash >> 16)
}

pub fn bounded_value(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

pub fn optimizer_velocity(previous: f32, gradient: f32, momentum: f32) -> f32 {
    momentum * previous + (1.0 - momentum) * gradient
}

pub fn dense_kernel_entry_point(entry_point: &str, sample_count: usize) -> String {
    if sample_count >= TILED_TRAINING_MIN_BATCH {
        entry_point.to_string()
    } else {
        format!("{entry_point}_naive")
    }
}

pub fn format_bytes(bytes: usize) -> String {
    let mib = bytes as f64 / (1024.0 * 1024.0);
    if mib >= 10.0 {
        format!("{mib:.0} MiB")
    } else {
        format!("{mib:.1} MiB")
    }
}

pub fn align4(value: usize) -> usize {
    value.div_ceil(4) * 4
}

pub fn normalized_search_score(score: i32) -> f32 {
    bounded_value(score as f32 / VALUE_SCORE_SCALE)
}

pub fn denormalized_search_score(value: f32) -> i32 {
    (bounded_value(value) * VALUE_SCORE_SCALE).round() as i32
}

pub fn inverse_tanh(value: f32) -> f32 {
    let bounded = value.clamp(-0.999_999, 0.999_999);
    0.5 * ((1.0 + bounded) / (1.0 - bounded)).ln()
}

pub fn loss_reduction_workgroup_count(sample_count: usize) -> usize {
    1usize.max(sample_count.saturating_add(63) / 64)
}

pub fn training_workgroups_16(item_count: usize) -> usize {
    item_count.div_ceil(16)
}

pub fn training_workgroups_64(item_count: usize) -> usize {
    item_count.div_ceil(64)
}

pub fn cpu_prediction_max_batch() -> usize {
    CPU_PREDICTION_MAX_BATCH
}

pub fn cpu_head_training_max_positions() -> usize {
    CPU_HEAD_TRAINING_MAX_POSITIONS
}

pub fn min_hidden_training_positions() -> usize {
    MIN_HIDDEN_TRAINING_POSITIONS
}

pub fn projection_chunk_size() -> usize {
    PROJECTION_CHUNK_SIZE
}

pub fn projection_temporary_budget(max_buffer_size: usize) -> usize {
    PROJECTION_TEMPORARY_BUDGET.min(max_buffer_size.saturating_div(2).max(1))
}

pub fn output_delta_params_bytes(sample_count: usize, total_weight: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    push_u32(&mut bytes, sample_count as u32);
    push_f32(&mut bytes, total_weight.max(0.0));
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    bytes
}

pub fn hidden_delta_params_bytes(
    sample_count: usize,
    current_size: usize,
    next_size: usize,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    push_u32(&mut bytes, sample_count as u32);
    push_u32(&mut bytes, current_size as u32);
    push_u32(&mut bytes, next_size as u32);
    push_u32(&mut bytes, 0);
    bytes
}

pub fn policy_params_bytes(
    batch_count: usize,
    input_size: usize,
    total_weight: f32,
    learning_rate: f32,
    weight_decay: f32,
    momentum: f32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    push_u32(&mut bytes, batch_count as u32);
    push_u32(&mut bytes, input_size as u32);
    push_u32(&mut bytes, POLICY_BUCKETS);
    push_u32(&mut bytes, 0);
    push_f32(&mut bytes, total_weight.max(0.0));
    push_f32(&mut bytes, learning_rate);
    push_f32(&mut bytes, weight_decay);
    push_f32(&mut bytes, momentum);
    bytes
}

pub fn layer_params_bytes(
    sample_count: usize,
    input_size: usize,
    output_size: usize,
    learning_rate: f32,
    weight_decay: f32,
    momentum: f32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    push_u32(&mut bytes, sample_count as u32);
    push_u32(&mut bytes, input_size as u32);
    push_u32(&mut bytes, output_size as u32);
    push_f32(&mut bytes, learning_rate);
    push_f32(&mut bytes, weight_decay);
    push_f32(&mut bytes, momentum);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    bytes
}

pub fn output_params_bytes(
    sample_count: usize,
    input_size: usize,
    learning_rate: f32,
    weight_decay: f32,
    momentum: f32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    push_u32(&mut bytes, sample_count as u32);
    push_u32(&mut bytes, input_size as u32);
    push_u32(&mut bytes, 0);
    push_f32(&mut bytes, learning_rate);
    push_f32(&mut bytes, weight_decay);
    push_f32(&mut bytes, momentum);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    bytes
}

pub fn projection_params_bytes(
    sample_count: usize,
    input_size: usize,
    projection_size: usize,
    seed: u32,
    output_offset: usize,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    push_u32(&mut bytes, sample_count as u32);
    push_u32(&mut bytes, input_size as u32);
    push_u32(&mut bytes, projection_size as u32);
    push_u32(&mut bytes, seed);
    push_u32(&mut bytes, output_offset as u32);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    bytes
}

pub fn split_hidden_weights(
    hidden_weights: &[f32],
    input_size: usize,
    hidden_layers: &[u32],
) -> Vec<Vec<f32>> {
    let mut layers = Vec::with_capacity(hidden_layers.len());
    let mut cursor = 0usize;
    let mut previous_size = input_size;
    for &layer_size in hidden_layers {
        let layer_size = layer_size as usize;
        let length = layer_size.saturating_mul(previous_size.saturating_add(1));
        let end = cursor.saturating_add(length).min(hidden_weights.len());
        layers.push(hidden_weights[cursor..end].to_vec());
        cursor = cursor.saturating_add(length);
        previous_size = layer_size;
    }
    layers
}

pub fn split_hidden_weights_bytes(request: &[u8]) -> Result<Vec<u8>, String> {
    struct Cursor<'a> {
        bytes: &'a [u8],
        offset: usize,
    }

    impl<'a> Cursor<'a> {
        fn read_u32(&mut self, label: &str) -> Result<u32, String> {
            let end = self
                .offset
                .checked_add(4)
                .ok_or_else(|| "Hidden weight split request is too large.".to_string())?;
            let bytes = self
                .bytes
                .get(self.offset..end)
                .ok_or_else(|| format!("Hidden weight split request is missing {label}."))?;
            self.offset = end;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
        }

        fn read_f32(&mut self, label: &str) -> Result<f32, String> {
            let end = self
                .offset
                .checked_add(4)
                .ok_or_else(|| "Hidden weight split request is too large.".to_string())?;
            let bytes = self
                .bytes
                .get(self.offset..end)
                .ok_or_else(|| format!("Hidden weight split request is missing {label}."))?;
            self.offset = end;
            Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
        }
    }

    let mut cursor = Cursor {
        bytes: request,
        offset: 0,
    };
    let input_size = cursor.read_u32("input size")? as usize;
    let layer_count = cursor.read_u32("layer count")? as usize;
    let weight_count = cursor.read_u32("weight count")? as usize;
    let mut hidden_layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        hidden_layers.push(cursor.read_u32("hidden layer size")?);
    }
    let mut hidden_weights = Vec::with_capacity(weight_count);
    for _ in 0..weight_count {
        hidden_weights.push(cursor.read_f32("hidden weight")?);
    }
    if cursor.offset != request.len() {
        return Err("Hidden weight split request has trailing bytes.".to_string());
    }

    let layers = split_hidden_weights(&hidden_weights, input_size, &hidden_layers);
    let layer_count = u32::try_from(layers.len())
        .map_err(|_| "Hidden weight layer count exceeds GPU parameter range.".to_string())?;
    let value_count = layers.iter().map(Vec::len).sum::<usize>();
    let mut bytes = Vec::with_capacity(4 + layers.len() * 4 + value_count * 4);
    push_u32(&mut bytes, layer_count);
    for layer in &layers {
        push_u32(
            &mut bytes,
            u32::try_from(layer.len()).map_err(|_| {
                "Hidden weight layer length exceeds GPU parameter range.".to_string()
            })?,
        );
    }
    for layer in layers {
        for value in layer {
            push_f32(&mut bytes, value);
        }
    }
    Ok(bytes)
}

pub fn concat_f32(arrays: &[Vec<f32>]) -> Vec<f32> {
    let length = arrays.iter().map(Vec::len).sum();
    let mut result = Vec::with_capacity(length);
    for array in arrays {
        result.extend_from_slice(array);
    }
    result
}

pub fn concat_f32_bytes(request: &[u8]) -> Result<Vec<u8>, String> {
    struct Cursor<'a> {
        bytes: &'a [u8],
        offset: usize,
    }

    impl<'a> Cursor<'a> {
        fn read_u32(&mut self, label: &str) -> Result<u32, String> {
            let end = self
                .offset
                .checked_add(4)
                .ok_or_else(|| "Float32 concat request is too large.".to_string())?;
            let bytes = self
                .bytes
                .get(self.offset..end)
                .ok_or_else(|| format!("Float32 concat request is missing {label}."))?;
            self.offset = end;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
        }

        fn read_f32(&mut self, label: &str) -> Result<f32, String> {
            let end = self
                .offset
                .checked_add(4)
                .ok_or_else(|| "Float32 concat request is too large.".to_string())?;
            let bytes = self
                .bytes
                .get(self.offset..end)
                .ok_or_else(|| format!("Float32 concat request is missing {label}."))?;
            self.offset = end;
            Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
        }
    }

    let mut cursor = Cursor {
        bytes: request,
        offset: 0,
    };
    let array_count = cursor.read_u32("array count")? as usize;
    let mut lengths = Vec::with_capacity(array_count);
    for _ in 0..array_count {
        lengths.push(cursor.read_u32("array length")? as usize);
    }
    let mut arrays = Vec::with_capacity(array_count);
    for length in lengths {
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            values.push(cursor.read_f32("array value")?);
        }
        arrays.push(values);
    }
    if cursor.offset != request.len() {
        return Err("Float32 concat request has trailing bytes.".to_string());
    }

    let values = concat_f32(&arrays);
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        push_f32(&mut bytes, value);
    }
    Ok(bytes)
}

pub fn count_non_zero(values: &[f32]) -> usize {
    values.iter().filter(|value| **value != 0.0).count()
}

pub fn count_non_zero_bytes(bytes: &[u8]) -> Result<usize, String> {
    let values = f32_values_from_bytes(bytes, "Non-zero count request")?;
    Ok(count_non_zero(&values))
}

fn f32_values_from_bytes(bytes: &[u8], label: &str) -> Result<Vec<f32>, String> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(format!("{label} length is not a multiple of f32 size."));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in chunks {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(values)
}

pub fn model_architecture_matches(model: &CompactValueModel) -> bool {
    model.projection_size == DEFAULT_PROJECTION_SIZE as u32
        && model.projection_seed == DEFAULT_PROJECTION_SEED
        && model.hidden_layers == DEFAULT_HIDDEN_LAYERS
        && !model.hidden_weights.is_empty()
        && model.output_activation == OutputActivation::Tanh
        && compact_model_is_finite(model)
}

pub fn compact_value_model_architecture_matches_bytes(bytes: &[u8]) -> bool {
    decode_compact_value_model(bytes)
        .map(|model| model_architecture_matches(&model))
        .unwrap_or(false)
}

pub fn output_layer_size(hidden_layers: &[u32]) -> Result<usize, String> {
    hidden_layers
        .last()
        .map(|value| *value as usize)
        .ok_or_else(|| "Model must have at least one hidden layer.".to_string())
}

pub fn previous_layer_size(hidden_layers: &[u32], layer_index: usize, input_size: usize) -> usize {
    if layer_index == 0 {
        input_size
    } else {
        hidden_layers
            .get(layer_index - 1)
            .copied()
            .unwrap_or_default() as usize
    }
}

pub fn policy_logits_array(model: Option<&CompactValueModel>) -> Option<Vec<f32>> {
    let logits = model?.policy_logits.as_slice();
    (!logits.is_empty()).then(|| {
        logits
            .iter()
            .copied()
            .take(POLICY_BUCKETS as usize)
            .collect()
    })
}

pub fn policy_weights_array(
    model: Option<&CompactValueModel>,
    input_size: usize,
) -> Option<Vec<f32>> {
    let model = model?;
    let expected = POLICY_BUCKETS as usize * (input_size + 1);
    if model.policy_weights.len() == expected {
        return Some(model.policy_weights.clone());
    }
    let logits = policy_logits_array(Some(model))?;
    let mut weights = vec![0.0; expected];
    for bucket in 0..POLICY_BUCKETS as usize {
        weights[bucket * (input_size + 1) + input_size] =
            logits.get(bucket).copied().unwrap_or(0.0);
    }
    Some(weights)
}

pub fn compact_value_model_policy_weights_bytes(
    bytes: &[u8],
    input_size: usize,
) -> Result<Option<Vec<u8>>, CompactValueModelError> {
    let model = decode_compact_value_model(bytes)?;
    let Some(weights) = policy_weights_array(Some(&model), input_size) else {
        return Ok(None);
    };
    let mut output = Vec::with_capacity(weights.len() * std::mem::size_of::<f32>());
    for value in weights {
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(Some(output))
}

pub fn quantized_policy_upload_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let weights = f32_values_from_bytes(bytes, "Policy quantization weights")?;
    let mut max_abs = 0.0_f32;
    for value in &weights {
        max_abs = max_abs.max(value.abs());
    }
    let scale = if max_abs > 0.0 {
        max_abs / 127.0
    } else {
        1.0 / 127.0
    };
    let mut dequantized = Vec::with_capacity(weights.len());
    let mut max_abs_error = 0.0_f32;
    for value in weights {
        let packed = (value / scale).round().clamp(-127.0, 127.0) as i8;
        let restored = packed as f32 * scale;
        dequantized.push(restored);
        max_abs_error = max_abs_error.max((value - restored).abs());
    }
    let mut output = Vec::with_capacity(8 + dequantized.len() * std::mem::size_of::<f32>());
    push_f32(&mut output, scale);
    push_f32(&mut output, max_abs_error);
    for value in dequantized {
        push_f32(&mut output, value);
    }
    Ok(output)
}

pub fn f32_to_f16_upload_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let values = f32_values_from_bytes(bytes, "f16 upload weights")?;
    let mut output = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
    for value in values {
        output.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
    }
    Ok(output)
}

pub fn f32_to_f16_bits(value: f32) -> u16 {
    if !value.is_finite() {
        return if value < 0.0 {
            0xfc00
        } else if value > 0.0 {
            0x7c00
        } else {
            0x7e00
        };
    }
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mut mantissa = bits & 0x7fffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return sign | (((mantissa + 0x1000) >> 13) as u16);
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exponent as u16) << 10) | (((mantissa + 0x1000) >> 13) as u16)
}

pub fn initial_hidden_weights(input_size: usize, hidden_layers: &[u32]) -> Vec<f32> {
    let mut weights = Vec::new();
    let mut previous = input_size;
    for (layer_index, &layer_size) in hidden_layers.iter().enumerate() {
        let layer_size = layer_size as usize;
        let scale = (2.0 / previous as f32).sqrt();
        for output in 0..layer_size {
            for input in 0..previous {
                let hash = projection_hash(
                    input as u32,
                    output as u32 + layer_index as u32 * 4099,
                    DEFAULT_PROJECTION_SEED,
                );
                weights.push((((hash as f32 / u32::MAX as f32) * 2.0) - 1.0) * scale);
            }
            weights.push(0.0);
        }
        previous = layer_size;
    }
    weights
}

pub fn initial_hidden_weights_bytes(request: &[u8]) -> Result<Vec<u8>, String> {
    struct Cursor<'a> {
        bytes: &'a [u8],
        offset: usize,
    }

    impl<'a> Cursor<'a> {
        fn read_u32(&mut self, label: &str) -> Result<u32, String> {
            let end = self
                .offset
                .checked_add(4)
                .ok_or_else(|| "Initial hidden weight request is too large.".to_string())?;
            let bytes = self
                .bytes
                .get(self.offset..end)
                .ok_or_else(|| format!("Initial hidden weight request is missing {label}."))?;
            self.offset = end;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
        }
    }

    let mut cursor = Cursor {
        bytes: request,
        offset: 0,
    };
    let input_size = cursor.read_u32("input size")? as usize;
    let layer_count = cursor.read_u32("layer count")? as usize;
    let mut hidden_layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        hidden_layers.push(cursor.read_u32("hidden layer size")?);
    }
    if cursor.offset != request.len() {
        return Err("Initial hidden weight request has trailing bytes.".to_string());
    }

    let weights = initial_hidden_weights(input_size, &hidden_layers);
    let mut bytes = Vec::with_capacity(weights.len() * std::mem::size_of::<f32>());
    for value in weights {
        push_f32(&mut bytes, value);
    }
    Ok(bytes)
}

pub fn default_initial_hidden_weights() -> Vec<f32> {
    initial_hidden_weights(DEFAULT_PROJECTION_SIZE, DEFAULT_HIDDEN_LAYERS)
}

pub fn compact_model_is_finite(model: &CompactValueModel) -> bool {
    f32_values_are_finite(&model.hidden_weights)
        && f32_values_are_finite(&model.output_weights)
        && f32_values_are_finite(&model.policy_logits)
        && f32_values_are_finite(&model.policy_weights)
        && f32_values_are_finite(&model.auxiliary_value_weights)
        && model.scale.is_finite()
        && model.bias.is_finite()
}

pub fn f32_values_are_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

pub fn byte_arrays_equal(left: Option<&[u8]>, right: Option<&[u8]>) -> bool {
    matches!((left, right), (Some(left), Some(right)) if left == right)
}

pub fn compact_value_model_json(bytes: &[u8]) -> Result<String, String> {
    let model = decode_compact_value_model(bytes).map_err(|error| error.to_string())?;
    serde_json::to_string(&compact_value_model_json_value(&model))
        .map_err(|error| format!("failed to encode compact value model JSON: {error}"))
}

pub fn compact_value_model_frontier_layout_json(bytes: &[u8]) -> Result<String, String> {
    let model = decode_compact_value_model(bytes).map_err(|error| error.to_string())?;
    if !model_architecture_matches(&model) {
        return Ok(serde_json::json!({
            "architectureMatches": false,
        })
        .to_string());
    }
    let output_size = output_layer_size(&model.hidden_layers)?;
    let hidden_layer_weights = frontier_hidden_layer_weights(&model)?;
    let policy_weights = policy_weights_array(Some(&model), output_size);
    serde_json::to_string(&serde_json::json!({
        "architectureMatches": true,
        "model": compact_value_model_json_value(&model),
        "outputLayerSize": output_size,
        "hiddenLayerWeights": hidden_layer_weights,
        "policyWeights": policy_weights,
    }))
    .map_err(|error| format!("failed to encode compact value model frontier layout JSON: {error}"))
}

fn compact_value_model_json_value(model: &CompactValueModel) -> serde_json::Value {
    serde_json::json!({
        "projectionSize": model.projection_size,
        "projectionSeed": model.projection_seed,
        "hiddenLayers": model.hidden_layers,
        "hiddenWeights": model.hidden_weights,
        "outputWeights": model.output_weights,
        "auxiliaryValueWeights": model.auxiliary_value_weights,
        "policyLogits": model.policy_logits,
        "policyWeights": model.policy_weights,
        "scale": model.scale,
        "bias": model.bias,
        "outputActivation": model.output_activation.to_string(),
    })
}

fn frontier_hidden_layer_weights(model: &CompactValueModel) -> Result<Vec<Vec<f32>>, String> {
    let mut expected_len = 0usize;
    let mut input_size = model.projection_size as usize;
    for &output_size in &model.hidden_layers {
        expected_len = expected_len
            .checked_add(output_size as usize * (input_size + 1))
            .ok_or_else(|| "GPU value model hidden-weight layout is too large.".to_string())?;
        input_size = output_size as usize;
    }
    if expected_len != model.hidden_weights.len() {
        return Err(format!(
            "GPU value model hidden-weight layout has {} weights but expected {expected_len}.",
            model.hidden_weights.len()
        ));
    }
    Ok(split_hidden_weights(
        &model.hidden_weights,
        model.projection_size as usize,
        &model.hidden_layers,
    ))
}

pub fn compact_value_model_bytes_from_json(text: &str) -> Result<Vec<u8>, String> {
    let input: EncodableCompactValueModelJson = serde_json::from_str(text)
        .map_err(|error| format!("Compact value model JSON is invalid: {error}"))?;
    let version = if !input.auxiliary_value_weights.is_empty() {
        5
    } else if matches!(input.output_activation, Some(OutputActivationJson::Tanh)) {
        4
    } else if !input.policy_weights.is_empty() {
        3
    } else if !input.policy_logits.is_empty() {
        2
    } else {
        1
    };
    let model = CompactValueModel {
        version,
        projection_size: input.projection_size,
        projection_seed: input.projection_seed,
        hidden_layers: input.hidden_layers,
        hidden_weights: input.hidden_weights,
        output_weights: input.output_weights,
        policy_logits: if version == 2 {
            input.policy_logits
        } else {
            Vec::new()
        },
        policy_weights: if version >= 3 {
            input.policy_weights
        } else {
            Vec::new()
        },
        auxiliary_value_weights: if version >= 5 {
            input.auxiliary_value_weights
        } else {
            Vec::new()
        },
        scale: input.scale.unwrap_or(1.0),
        bias: input.bias.unwrap_or(0.0),
        output_activation: if version >= 4 {
            OutputActivation::Tanh
        } else {
            OutputActivation::Linear
        },
    };
    Ok(encode_compact_value_model(&model))
}

pub fn compact_value_model_is_finite_bytes(bytes: &[u8]) -> bool {
    decode_compact_value_model(bytes)
        .map(|model| compact_model_is_finite(&model))
        .unwrap_or(false)
}

pub fn compact_value_model_training_layout_bytes(
    model_bytes: Option<&[u8]>,
    average_label: f32,
) -> Result<Vec<u8>, String> {
    let active = match model_bytes.filter(|bytes| !bytes.is_empty()) {
        Some(bytes) => Some(decode_compact_value_model(bytes).map_err(|error| error.to_string())?),
        None => None,
    };
    let architecture_matches = active
        .as_ref()
        .map(model_architecture_matches)
        .unwrap_or(false);
    let output_size = output_layer_size(DEFAULT_HIDDEN_LAYERS)?;
    let hidden_weights = if architecture_matches {
        active
            .as_ref()
            .map(|model| model.hidden_weights.clone())
            .unwrap_or_default()
    } else {
        default_initial_hidden_weights()
    };
    let mut output_weights = vec![0.0; output_size + 1];
    if architecture_matches {
        if let Some(model) = active.as_ref() {
            if model.output_weights.len() == output_weights.len() {
                output_weights.clone_from_slice(&model.output_weights);
            } else {
                output_weights[output_size] = inverse_tanh(average_label);
            }
        }
    } else {
        output_weights[output_size] = inverse_tanh(average_label);
    }
    let mut bytes = Vec::with_capacity(16 + (hidden_weights.len() + output_weights.len()) * 4);
    push_u32(&mut bytes, u32::from(architecture_matches));
    push_u32(&mut bytes, output_size as u32);
    push_u32(&mut bytes, hidden_weights.len() as u32);
    push_u32(&mut bytes, output_weights.len() as u32);
    for value in hidden_weights.iter().chain(output_weights.iter()) {
        push_f32(&mut bytes, *value);
    }
    Ok(bytes)
}

pub fn compact_value_model_hidden_features_json(
    model_bytes: &[u8],
    samples_json: &str,
) -> Result<String, String> {
    let model = decode_compact_value_model(model_bytes)
        .map_err(|error| format!("Compact model hidden-feature request is invalid: {error}"))?;
    let samples = serde_json::from_str::<Vec<TrainingSample>>(samples_json)
        .map_err(|error| format!("Compact model hidden-feature samples are invalid: {error}"))?;
    feature_length(&samples)?;
    let features = samples
        .iter()
        .map(|sample| hidden_features(&sample.features, &model))
        .collect::<Vec<_>>();
    serde_json::to_string(&features)
        .map_err(|error| format!("Compact model hidden features failed to encode: {error}"))
}

fn value_head_loss(features: &[Vec<f32>], samples: &[TrainingSample], weights: &[f32]) -> f32 {
    let mut total = 0.0;
    let mut total_weight = 0.0;
    let bias_index = weights.len().saturating_sub(1);
    for (feature, sample) in features.iter().zip(samples.iter()) {
        let prediction = value_head_prediction(feature, weights, bias_index);
        let weight = sample.label_weight.max(0.0);
        let error = prediction - bounded_value(sample.label);
        total += weight * error * error;
        total_weight += weight;
    }
    if total_weight > 0.0 {
        total / total_weight
    } else {
        0.0
    }
}

fn apply_value_head_gradient(
    features: &[Vec<f32>],
    samples: &[TrainingSample],
    weights: &mut [f32],
    velocity: &mut [f32],
    config: ValueHeadTrainingConfig,
) {
    let bias_index = weights.len().saturating_sub(1);
    let mut gradient = vec![0.0; weights.len()];
    let mut batch_weight = 0.0;
    for (feature, sample) in features.iter().zip(samples.iter()) {
        let prediction = value_head_prediction(feature, weights, bias_index);
        let sample_weight = sample.label_weight.max(0.0);
        let scale = 2.0
            * sample_weight
            * (prediction - bounded_value(sample.label))
            * (1.0 - prediction * prediction);
        for (input, value) in feature.iter().enumerate() {
            gradient[input] += scale * value;
        }
        gradient[bias_index] += scale;
        batch_weight += sample_weight;
    }
    let normalization = 1.0 / batch_weight.max(1e-6);
    for index in 0..weights.len() {
        let decay = if index == bias_index {
            0.0
        } else {
            config.weight_decay * weights[index]
        };
        let update = gradient[index] * normalization + decay;
        velocity[index] = config.momentum * velocity[index] + (1.0 - config.momentum) * update;
        weights[index] -= config.learning_rate * velocity[index];
    }
}

fn value_head_prediction(feature: &[f32], weights: &[f32], bias_index: usize) -> f32 {
    let mut logit = weights.get(bias_index).copied().unwrap_or(0.0);
    for (input, value) in feature.iter().enumerate() {
        logit += value * weights.get(input).copied().unwrap_or(0.0);
    }
    logit.tanh()
}

fn policy_head_loss(
    features: &[Vec<f32>],
    samples: &[TrainingSample],
    weights: &[f32],
    indices: &[usize],
    input_size: usize,
) -> f32 {
    let row_size = input_size + 1;
    let mut total = 0.0;
    let mut total_weight = 0.0;
    for &index in indices {
        let Some(feature) = features.get(index) else {
            continue;
        };
        let target = policy_target(samples.get(index));
        let sample_weight = samples
            .get(index)
            .map(|sample| sample.label_weight.max(0.0))
            .unwrap_or(0.0);
        if sample_weight <= 0.0 {
            continue;
        }
        let logits = policy_logits(feature, weights, input_size);
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator = logits
            .iter()
            .map(|logit| (*logit - max_logit).exp())
            .sum::<f32>()
            .max(1e-12);
        total += sample_weight * (denominator.ln() - (logits[target] - max_logit));
        total_weight += sample_weight;
        debug_assert_eq!(row_size * POLICY_BUCKETS as usize, weights.len());
    }
    if total_weight > 0.0 {
        total / total_weight
    } else {
        0.0
    }
}

fn apply_policy_head_gradient(
    features: &[Vec<f32>],
    samples: &[TrainingSample],
    indices: &[usize],
    weights: &mut [f32],
    velocity: &mut [f32],
    input_size: usize,
    config: ValueHeadTrainingConfig,
) {
    let row_size = input_size + 1;
    let mut gradient = vec![0.0; weights.len()];
    let mut batch_weight = 0.0;
    for &sample_index in indices {
        let Some(feature) = features.get(sample_index) else {
            continue;
        };
        let target = policy_target(samples.get(sample_index));
        let sample_weight = samples
            .get(sample_index)
            .map(|sample| sample.label_weight.max(0.0))
            .unwrap_or(0.0);
        if sample_weight <= 0.0 {
            continue;
        }
        let logits = policy_logits(feature, weights, input_size);
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits = logits
            .iter()
            .map(|logit| (*logit - max_logit).exp())
            .collect::<Vec<_>>();
        let denominator = exp_logits.iter().sum::<f32>().max(1e-12);
        for (bucket, item) in exp_logits.iter().enumerate().take(POLICY_BUCKETS as usize) {
            let target_value = if bucket == target { 1.0 } else { 0.0 };
            let delta = (item / denominator - target_value) * sample_weight;
            let row = bucket * row_size;
            for (input, value) in feature.iter().enumerate().take(input_size) {
                gradient[row + input] += delta * value;
            }
            gradient[row + input_size] += delta;
        }
        batch_weight += sample_weight;
    }
    let normalization = 1.0 / batch_weight.max(1e-6);
    for (index, weight) in weights.iter_mut().enumerate() {
        let input = index % row_size;
        let decay = if input == input_size {
            0.0
        } else {
            config.weight_decay * *weight
        };
        let update = gradient[index] * normalization + decay;
        velocity[index] = config.momentum * velocity[index] + (1.0 - config.momentum) * update;
        *weight -= config.learning_rate * velocity[index];
    }
}

pub fn auxiliary_value_targets(sample: &TrainingSample) -> [f32; AUXILIARY_VALUE_HEAD_COUNT] {
    let features = &sample.features;
    let board_stride = NEURAL_BOARD_PLANES * NEURAL_BOARD_SQUARES;
    let encoded_board_count = features.len() / board_stride;
    let board_count = sample
        .board_count
        .unwrap_or(encoded_board_count)
        .min(encoded_board_count)
        .max(1);
    let mut active_boards = 0.0;
    let mut present_boards = 0.0;
    let mut royal_danger: f32 = 0.0;
    let mut active_material = 0.0;
    let mut inactive_material = 0.0;
    for board in 0..board_count {
        let base = board * board_stride;
        let active = features
            .get(base + 25 * NEURAL_BOARD_SQUARES)
            .copied()
            .unwrap_or(0.0);
        let present = features
            .get(base + 27 * NEURAL_BOARD_SQUARES)
            .copied()
            .unwrap_or(0.0);
        let royal = features
            .get(base + 31 * NEURAL_BOARD_SQUARES)
            .copied()
            .unwrap_or(0.0);
        active_boards += if active > 0.0 { 1.0 } else { 0.0 };
        present_boards += if present > 0.0 { 1.0 } else { 0.0 };
        royal_danger = royal_danger.max(royal);
        let material = material_balance_for_encoded_board(features, base);
        if active > 0.0 {
            active_material += material;
        } else {
            inactive_material += material;
        }
    }
    let board_count = board_count as f32;
    [
        if sample.label > 0.05 {
            1.0
        } else if sample.label < -0.05 {
            -1.0
        } else {
            0.0
        },
        bounded_value(sample.label),
        royal_danger,
        bounded_value(1.0 - 2.0 * royal_danger),
        bounded_value(active_boards / board_count),
        bounded_value(present_boards / board_count),
        bounded_value(active_material / 16.0),
        bounded_value(inactive_material / 16.0),
        if sample.policy.is_some() { 0.0 } else { 1.0 },
    ]
}

pub fn auxiliary_value_targets_bytes(samples: &[TrainingSample]) -> Vec<u8> {
    let mut bytes =
        Vec::with_capacity(samples.len() * AUXILIARY_VALUE_HEAD_COUNT * std::mem::size_of::<f32>());
    for sample in samples {
        for value in auxiliary_value_targets(sample) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

pub fn material_balance_for_encoded_board(features: &[f32], board_base: usize) -> f32 {
    let mut balance = 0.0;
    for plane in 0..24 {
        let piece_value = encoded_piece_value(plane % 12);
        let sign = if plane < 12 { 1.0 } else { -1.0 };
        for square in 0..NEURAL_BOARD_SQUARES {
            balance += sign
                * piece_value
                * features
                    .get(board_base + plane * NEURAL_BOARD_SQUARES + square)
                    .copied()
                    .unwrap_or(0.0);
        }
    }
    balance
}

pub fn encoded_piece_value(piece_type: usize) -> f32 {
    match piece_type {
        0 | 3 => 8.0,
        2 => 5.0,
        4 | 5 => 4.0,
        6 | 9 => 3.0,
        7 | 8 => 2.0,
        _ => 1.0,
    }
}

fn policy_logits(feature: &[f32], weights: &[f32], input_size: usize) -> Vec<f32> {
    let row_size = input_size + 1;
    let mut logits = vec![0.0; POLICY_BUCKETS as usize];
    for (bucket, logit) in logits.iter_mut().enumerate() {
        let row = bucket * row_size;
        *logit = weights.get(row + input_size).copied().unwrap_or(0.0);
        for (input, value) in feature.iter().enumerate().take(input_size) {
            *logit += value * weights.get(row + input).copied().unwrap_or(0.0);
        }
    }
    logits
}

fn policy_target(sample: Option<&TrainingSample>) -> usize {
    sample
        .and_then(|sample| sample.policy)
        .map(policy_training_target)
        .unwrap_or(0)
}

pub fn policy_training_target(policy: u32) -> usize {
    policy.min(POLICY_BUCKETS - 1) as usize
}

pub fn training_label_weight(label_weight: f32) -> f32 {
    label_weight.max(0.0)
}

pub fn training_weighted_average(total: f64, total_weight: f64) -> f64 {
    if total_weight > 0.0 {
        total / total_weight
    } else {
        0.0
    }
}

pub fn training_batch_normalization(batch_weight: f64) -> f64 {
    1.0 / batch_weight.max(1e-6)
}

pub fn has_policy_training_target(sample: &TrainingSample) -> bool {
    sample.label_kind.as_deref() != Some("distilled") && sample.policy.is_some()
}

pub fn policy_training_indices(
    samples: &[TrainingSample],
    require_positive_weight: bool,
) -> Vec<usize> {
    samples
        .iter()
        .enumerate()
        .filter(|(_, sample)| {
            has_policy_training_target(sample)
                && (!require_positive_weight || training_label_weight(sample.label_weight) > 0.0)
        })
        .map(|(index, _)| index)
        .collect()
}

pub fn policy_training_steps(value_epochs: usize) -> usize {
    (value_epochs.saturating_add(63) / 64).clamp(16, 256)
}

pub fn value_training_batch_size(config_batch_size: usize, training_count: usize) -> usize {
    config_batch_size.min(training_count.max(1))
}

pub fn policy_training_batch_size(config_batch_size: usize, training_count: usize) -> usize {
    config_batch_size.min(training_count)
}

pub fn value_head_validation_interval(epochs: usize, validation_interval: Option<usize>) -> usize {
    validation_interval.unwrap_or(256).min(epochs).max(1)
}

pub fn value_gpu_batches_per_submit(epochs: usize) -> usize {
    VALUE_EPOCHS_PER_SUBMIT.min(epochs.max(1))
}

pub fn value_gpu_validation_interval(
    batches_per_submit: usize,
    validation_interval: Option<usize>,
) -> usize {
    batches_per_submit.max(validation_interval.unwrap_or(256))
}

pub fn policy_training_steps_per_submit(steps: usize) -> usize {
    POLICY_STEPS_PER_SUBMIT.min(steps)
}

fn default_label_weight() -> f32 {
    1.0
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub fn decode_compact_value_model(
    bytes: &[u8],
) -> Result<CompactValueModel, CompactValueModelError> {
    let mut cursor = CompactValueModelCursor::new(bytes);
    let magic = cursor.read_magic()?;
    if &magic != COMPACT_VALUE_MODEL_MAGIC {
        return Err(CompactValueModelError::InvalidMagic(magic));
    }
    let version = cursor.read_u32()?;
    if !(1..=MAX_COMPACT_VALUE_MODEL_VERSION).contains(&version) {
        return Err(CompactValueModelError::UnsupportedVersion(version));
    }
    let projection_size = cursor.read_u32()?;
    let projection_seed = cursor.read_u32()?;
    let layer_count = cursor.read_u32()? as usize;
    let output_size = cursor.read_u32()? as usize;
    let policy_size = if version >= 2 {
        cursor.read_u32()? as usize
    } else {
        0
    };
    let auxiliary_value_size = if version >= 5 {
        cursor.read_u32()? as usize
    } else {
        0
    };
    let scale = cursor.read_f32("scale", 0)?;
    let bias = cursor.read_f32("bias", 0)?;
    let mut hidden_layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        hidden_layers.push(cursor.read_u32()?);
    }
    let hidden_weight_count = cursor.read_u32()? as usize;
    let hidden_weights = cursor.read_f32_vec("hidden_weights", hidden_weight_count)?;
    let output_weights = cursor.read_f32_vec("output_weights", output_size)?;
    let policy_values = cursor.read_f32_vec("policy", policy_size)?;
    let auxiliary_value_weights =
        cursor.read_f32_vec("auxiliary_value_weights", auxiliary_value_size)?;
    if cursor.remaining() != 0 {
        return Err(CompactValueModelError::TrailingBytes {
            parsed: cursor.offset(),
            len: bytes.len(),
        });
    }
    let (policy_logits, policy_weights) = if version == 2 {
        (policy_values, Vec::new())
    } else {
        (Vec::new(), policy_values)
    };
    Ok(CompactValueModel {
        version,
        projection_size,
        projection_seed,
        hidden_layers,
        hidden_weights,
        output_weights,
        policy_logits,
        policy_weights,
        auxiliary_value_weights,
        scale,
        bias,
        output_activation: if version >= 4 {
            OutputActivation::Tanh
        } else {
            OutputActivation::Linear
        },
    })
}

struct CompactValueModelCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> CompactValueModelCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_magic(&mut self) -> Result<[u8; 4], CompactValueModelError> {
        let bytes = self.take(4)?;
        Ok([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn read_u32(&mut self) -> Result<u32, CompactValueModelError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f32(
        &mut self,
        section: &'static str,
        index: usize,
    ) -> Result<f32, CompactValueModelError> {
        let bytes = self.take(4)?;
        let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if !value.is_finite() {
            return Err(CompactValueModelError::NonFinite {
                section,
                index,
                value,
            });
        }
        Ok(value)
    }

    fn read_f32_vec(
        &mut self,
        section: &'static str,
        count: usize,
    ) -> Result<Vec<f32>, CompactValueModelError> {
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            values.push(self.read_f32(section, index)?);
        }
        Ok(values)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], CompactValueModelError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(CompactValueModelError::Truncated {
                offset: self.offset,
                needed: len,
                len: self.bytes.len(),
            })?;
        if end > self.bytes.len() {
            return Err(CompactValueModelError::Truncated {
                offset: self.offset,
                needed: len,
                len: self.bytes.len(),
            });
        }
        let bytes = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }
}
