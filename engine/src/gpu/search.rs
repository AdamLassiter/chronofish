use super::{training, GpuKernel, GpuKernelSet, WgslShader};
use crate::{
    cpu::{search_deadline, SearchOptions, ValueEvaluator},
    wasm_api::parse_game_snapshot,
    BoardSnapshot,
    CastlingRights,
    Color,
    Game,
    Origin,
    PieceType,
    Position,
    TimelineOwner,
};

pub const TURN_STATUS_SHADER: &str = include_str!("search/shaders/turn_status.wgsl");
pub const MOVEGEN_SHADER: &str = include_str!("search/shaders/movegen.wgsl");
pub const REPLY_SHADER: &str = include_str!("search/shaders/reply.wgsl");
pub const MUTATE_SHADER: &str = include_str!("search/shaders/mutate.wgsl");
pub const FRONTIER_SELECT_SHADER: &str = include_str!("search/shaders/frontier_select.wgsl");
pub const FRONTIER_STATE_SHADER: &str = include_str!("search/shaders/frontier_state.wgsl");
pub const FRONTIER_EXPAND_SHADER: &str = include_str!("search/shaders/frontier_expand.wgsl");
pub const FRONTIER_FORWARD_SHADER: &str = include_str!("search/shaders/frontier_forward.wgsl");
pub const FRONTIER_POLICY_SHADER: &str = include_str!("search/shaders/frontier_policy.wgsl");
pub const FRONTIER_NEURAL_SHADER: &str = include_str!("search/shaders/frontier_neural.wgsl");

pub const SHADERS: &[WgslShader] = &[
    WgslShader {
        name: "turn_status.wgsl",
        source: TURN_STATUS_SHADER,
    },
    WgslShader {
        name: "movegen.wgsl",
        source: MOVEGEN_SHADER,
    },
    WgslShader {
        name: "reply.wgsl",
        source: REPLY_SHADER,
    },
    WgslShader {
        name: "mutate.wgsl",
        source: MUTATE_SHADER,
    },
    WgslShader {
        name: "frontier_select.wgsl",
        source: FRONTIER_SELECT_SHADER,
    },
    WgslShader {
        name: "frontier_state.wgsl",
        source: FRONTIER_STATE_SHADER,
    },
    WgslShader {
        name: "frontier_expand.wgsl",
        source: FRONTIER_EXPAND_SHADER,
    },
    WgslShader {
        name: "frontier_forward.wgsl",
        source: FRONTIER_FORWARD_SHADER,
    },
    WgslShader {
        name: "frontier_policy.wgsl",
        source: FRONTIER_POLICY_SHADER,
    },
    WgslShader {
        name: "frontier_neural.wgsl",
        source: FRONTIER_NEURAL_SHADER,
    },
];

pub const DEFAULT_GPU_SEARCH_DEPTH: i32 = 2;
pub const DEFAULT_GPU_SEARCH_NODES: i32 = 1_024;
pub const DEFAULT_GPU_SEARCH_TIME_MS: i32 = 10_000;

pub const DEFAULT_STORAGE_LIMIT: usize = 128 * 1024 * 1024;
pub const DEFAULT_BUFFER_LIMIT: usize = 256 * 1024 * 1024;
pub const MIN_FRONTIER_WIDTH: usize = 8;
pub const MAX_FRONTIER_WIDTH: usize = 512;
pub const MIN_CANDIDATES: usize = 256;
pub const MAX_CANDIDATES: usize = 65_536;
pub const MAX_SELECTION_SCAN: usize = 2048;

pub const GPU_CANDIDATE_STRIDE: usize = 24;
pub const GPU_CANDIDATE_COLOR_OFFSET: usize = 1;
pub const GPU_SOURCE_STRIDE: usize = 10;
pub const GPU_TARGET_STRIDE: usize = 10;
pub const GPU_BOARD_STRIDE: usize = 73;
pub const GPU_MUTATION_BOARD_STRIDE: usize = 76;
pub const GPU_MUTATION_CHILD_STRIDE: usize = GPU_MUTATION_BOARD_STRIDE * 2;
pub const GPU_CANDIDATE_INPUT_HEADER_I32S: usize = 7;
pub const GPU_MUTATION_STATUS_OK: i32 = 1;
pub const GPU_MUTATION_STATUS_ROYAL_CAPTURE: i32 = 2;
pub const GPU_MUTATION_STATUS_BRANCH_OK: i32 = 3;
pub const GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE: i32 = 4;
pub const GPU_TURN_STATUS_RECORD_STRIDE: usize = 4;

pub const FRONTIER_HEADER_STRIDE: usize = 16;
pub const FRONTIER_BOARD_STRIDE: usize = 78;
pub const FRONTIER_BOARD_TIMELINE_ID: usize = 0;
pub const FRONTIER_BOARD_ROW: usize = 1;
pub const FRONTIER_BOARD_OWNER: usize = 2;
pub const FRONTIER_BOARD_TIME: usize = 3;
pub const FRONTIER_BOARD_SIDE_TO_MOVE: usize = 4;
pub const FRONTIER_BOARD_CASTLING: usize = 5;
pub const FRONTIER_BOARD_EN_PASSANT: usize = 6;
pub const FRONTIER_BOARD_LATEST: usize = 10;
pub const FRONTIER_BOARD_ORIGIN: usize = 11;
pub const FRONTIER_BOARD_SQUARES: usize = 12;
pub const FRONTIER_BOARD_ACTIVE: usize = 76;
pub const FRONTIER_BOARD_PENDING: usize = 77;
pub const FRONTIER_MAX_PLAN_MOVES: usize = 64;
pub const FRONTIER_MAX_DEPTH: usize = 16;
pub const FRONTIER_ANCESTRY_STRIDE: usize = FRONTIER_MAX_DEPTH;
pub const FRONTIER_MOVE_STRIDE: usize = 8;
pub const FRONTIER_PLAN_STRIDE: usize = FRONTIER_MAX_PLAN_MOVES * FRONTIER_MOVE_STRIDE;
pub const FRONTIER_PLAN_OFFSET: usize = FRONTIER_HEADER_STRIDE + FRONTIER_ANCESTRY_STRIDE;
pub const FRONTIER_BOARD_OFFSET: usize = FRONTIER_PLAN_OFFSET + FRONTIER_PLAN_STRIDE;
pub const FRONTIER_CANDIDATE_STRIDE: usize = 24;
pub const FRONTIER_DELTA_STRIDE: usize = FRONTIER_BOARD_STRIDE * 2;
pub const FRONTIER_SUMMARY_STRIDE: usize = 12;

pub const FRONTIER_HEADER_PARENT: usize = 0;
pub const FRONTIER_HEADER_ROOT: usize = 1;
pub const FRONTIER_HEADER_SCORE: usize = 2;
pub const FRONTIER_HEADER_DEPTH: usize = 3;
pub const FRONTIER_HEADER_TURN: usize = 4;
pub const FRONTIER_HEADER_BOARD_COUNT: usize = 5;
pub const FRONTIER_HEADER_PLAN_LENGTH: usize = 6;
pub const FRONTIER_HEADER_COMPLETE: usize = 7;
pub const FRONTIER_HEADER_TERMINAL: usize = 8;
pub const FRONTIER_HEADER_HASH_LOW: usize = 9;
pub const FRONTIER_HEADER_HASH_HIGH: usize = 10;
pub const FRONTIER_HEADER_NEXT_WHITE_TIMELINE: usize = 11;
pub const FRONTIER_HEADER_NEXT_BLACK_TIMELINE: usize = 12;
pub const FRONTIER_HEADER_PRESENT_TIME: usize = 13;
pub const FRONTIER_HEADER_PENDING_BOARDS: usize = 14;
pub const FRONTIER_HEADER_FLAGS: usize = 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontierTuningLimits {
    pub max_storage_buffer_binding_size: Option<usize>,
    pub max_buffer_size: Option<usize>,
    pub max_compute_invocations_per_workgroup: Option<usize>,
}

impl Default for FrontierTuningLimits {
    fn default() -> Self {
        Self {
            max_storage_buffer_binding_size: Some(DEFAULT_STORAGE_LIMIT),
            max_buffer_size: Some(DEFAULT_BUFFER_LIMIT),
            max_compute_invocations_per_workgroup: Some(256),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontierTuning {
    pub max_boards: usize,
    pub frontier_width: usize,
    pub candidate_capacity: usize,
    pub neural_batch_size: usize,
    pub candidate_workgroup_size: usize,
    pub mutation_tile_size: usize,
    pub dispatch_candidate_limit: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontierSelectionPlan {
    pub candidate_capacity: usize,
    pub selection_capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuFrontierActiveBoard {
    pub time: i32,
    pub side_to_move: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuTimelineSortKey {
    pub row: i32,
    pub id: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuCandidatePosition {
    pub timeline_id: i32,
    pub time: i32,
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuCandidateMove {
    pub from: GpuCandidatePosition,
    pub to: GpuCandidatePosition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuSquareRecordInput {
    pub piece_code: i32,
    pub timeline_id: i32,
    pub time: i32,
    pub x: i32,
    pub y: i32,
    pub timeline_row: i32,
    pub side_to_move: i32,
    pub owner: i32,
    pub latest: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuSquareRecordBoardInput {
    pub timeline_id: i32,
    pub time: i32,
    pub timeline_row: i32,
    pub side_to_move: i32,
    pub owner: i32,
    pub latest: bool,
    pub squares: Vec<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuCandidateSquareRecord {
    pub meta: GpuCandidatePosition,
    pub words: [i32; GPU_SOURCE_STRIDE],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuCandidateBoardInput {
    pub timeline_id: i32,
    pub timeline_row: i32,
    pub timeline_index: i32,
    pub time: i32,
    pub side_to_move: i32,
    pub owner: i32,
    pub castling: i32,
    pub en_passant: Option<GpuEnPassantRecord>,
    pub latest: bool,
    pub origin_kind: i32,
    pub squares: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuCandidateBoardRecords {
    pub board: Vec<i32>,
    pub mutation_board: Vec<i32>,
    pub sources: Vec<GpuCandidateSquareRecord>,
    pub targets: Vec<GpuCandidateSquareRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuCandidateInputBoard {
    pub time: i32,
    pub side_to_move: i32,
    pub castling: i32,
    pub en_passant: Option<GpuEnPassantRecord>,
    pub origin_kind: i32,
    pub squares: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuCandidateInputTimeline {
    pub id: i32,
    pub row: i32,
    pub owner: i32,
    pub boards: Vec<GpuCandidateInputBoard>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuCandidateInputs {
    pub source_meta: Vec<GpuCandidatePosition>,
    pub target_meta: Vec<GpuCandidatePosition>,
    pub source_count: usize,
    pub target_count: usize,
    pub board_count: usize,
    pub sources: Vec<i32>,
    pub targets: Vec<i32>,
    pub boards: Vec<i32>,
    pub mutation_boards: Vec<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuChildBoardRef {
    pub timeline_id: i32,
    pub time: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuEnPassantRecord {
    pub x: i32,
    pub y: i32,
    pub captured_x: i32,
    pub captured_y: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuBoardRecordInput {
    pub timeline_id: i32,
    pub timeline_row: i32,
    pub time: i32,
    pub side_to_move: i32,
    pub castling: i32,
    pub en_passant: Option<GpuEnPassantRecord>,
    pub squares: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuMutationBoardRecordInput {
    pub timeline_index: i32,
    pub timeline_id: i32,
    pub time: i32,
    pub side_to_move: i32,
    pub castling: i32,
    pub en_passant: Option<GpuEnPassantRecord>,
    pub latest: bool,
    pub origin_kind: i32,
    pub squares: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuMutationBoardSnapshot {
    pub timeline_index: i32,
    pub timeline_id: i32,
    pub time: i32,
    pub side_to_move: &'static str,
    pub castling: i32,
    pub en_passant: Option<GpuEnPassantRecord>,
    pub latest: bool,
    pub origin_kind: i32,
    pub squares: Vec<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuDecodedPiece {
    pub piece_type: &'static str,
    pub color: &'static str,
}

pub type GpuDecodedBoard = Vec<Vec<Option<GpuDecodedPiece>>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EncodedFrontierRoot {
    pub(crate) words: Vec<i32>,
    pub(crate) board_count: usize,
    pub(crate) hash_low: i32,
    pub(crate) hash_high: i32,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuSnapshotJson {
    format: Option<String>,
    turn: String,
    next_timeline_id: Option<i32>,
    next_black_timeline_id: Option<i32>,
    royal_capture_by: Option<String>,
    timelines: Vec<GpuTimelineJson>,
}

#[derive(serde::Deserialize)]
struct GpuTimelineJson {
    id: i32,
    row: i32,
    label: Option<String>,
    owner: String,
    boards: Vec<GpuBoardJson>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuBoardJson {
    time: i32,
    side_to_move: String,
    castling: Option<i32>,
    en_passant: Option<GpuEnPassantJson>,
    origin: Option<serde_json::Value>,
    origin_kind: Option<i32>,
    #[serde(rename = "timelineIndex")]
    timeline_index: Option<i32>,
    latest: Option<bool>,
    squares: Option<serde_json::Value>,
    board: Option<Vec<Vec<Option<GpuPieceJson>>>>,
}

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuEnPassantJson {
    x: i32,
    y: i32,
    captured_x: i32,
    captured_y: i32,
}

#[derive(serde::Deserialize)]
struct GpuPieceJson {
    color: String,
    #[serde(rename = "type")]
    piece_type: String,
}

pub fn frontier_state_stride(max_boards: usize) -> usize {
    FRONTIER_BOARD_OFFSET + max_boards * FRONTIER_BOARD_STRIDE
}

pub fn frontier_state_bytes(max_boards: usize) -> usize {
    frontier_state_stride(max_boards) * std::mem::size_of::<i32>()
}

pub fn frontier_neural_params_bytes(
    state_count: usize,
    state_stride: usize,
    board_offset: usize,
    max_boards: usize,
    state_offset: usize,
    projection_size: usize,
    projection_seed: u32,
    target_depth: usize,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    push_u32(&mut bytes, state_count as u32);
    push_u32(&mut bytes, state_stride as u32);
    push_u32(&mut bytes, board_offset as u32);
    push_u32(&mut bytes, max_boards as u32);
    push_u32(&mut bytes, state_offset as u32);
    push_u32(&mut bytes, projection_size as u32);
    push_u32(&mut bytes, projection_seed);
    push_u32(&mut bytes, target_depth as u32);
    bytes
}

pub fn frontier_neural_apply_params_bytes(
    state_count: usize,
    root_color: i32,
    value_scale: f32,
    value_bias: f32,
    state_offset: usize,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    push_u32(&mut bytes, state_count as u32);
    push_i32(&mut bytes, root_color);
    push_f32(&mut bytes, value_scale);
    push_f32(&mut bytes, value_bias);
    push_u32(&mut bytes, state_offset as u32);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    bytes
}

pub fn frontier_neural_layer_params_bytes(
    sample_count: usize,
    input_size: usize,
    output_size: usize,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    push_u32(&mut bytes, sample_count as u32);
    push_u32(&mut bytes, input_size as u32);
    push_u32(&mut bytes, output_size as u32);
    push_u32(&mut bytes, 0);
    bytes
}

pub fn frontier_neural_effective_batch_size(
    state_count: usize,
    requested_batch_size: f64,
) -> usize {
    if state_count == 0 {
        return 1;
    }
    let requested = if requested_batch_size.is_finite() {
        requested_batch_size.floor().max(1.0) as usize
    } else {
        1
    };
    state_count.min(requested).max(1)
}

pub fn frontier_neural_batch_count(
    state_count: usize,
    state_offset: usize,
    effective_batch_size: usize,
) -> usize {
    state_count
        .saturating_sub(state_offset)
        .min(effective_batch_size.max(1))
}

pub fn frontier_neural_cache_hit_rate(hits: f64, misses: f64) -> f64 {
    let lookups = hits + misses;
    if !hits.is_finite() || !misses.is_finite() || lookups <= 0.0 {
        return 0.0;
    }
    ((hits / lookups) * 1000.0).round() / 1000.0
}

pub fn frontier_cycle_state_count(frontier_width: usize, requested_state_count: usize) -> usize {
    frontier_width.min(requested_state_count.max(1)).max(1)
}

pub fn frontier_expansion_source_scan_limit(
    candidate_workgroup_size: usize,
    dispatch_candidate_limit: usize,
) -> usize {
    candidate_workgroup_size
        .max(dispatch_candidate_limit)
        .max(1)
}

pub fn frontier_expansion_source_scan_count(
    source_scan_limit: usize,
    source_scans: usize,
    source_scan_base: usize,
) -> usize {
    source_scans
        .saturating_sub(source_scan_base)
        .min(source_scan_limit)
}

pub fn frontier_minimax_bounded_depth(target_depth: i32, ancestry_stride: i32) -> i32 {
    target_depth.min(ancestry_stride)
}

pub fn frontier_neural_select_board_workgroups(batch_count: usize) -> usize {
    batch_count.saturating_mul(16).div_ceil(64)
}

pub fn frontier_neural_project_workgroups_x(batch_count: usize) -> usize {
    batch_count.div_ceil(16)
}

pub fn frontier_neural_project_workgroups_y(projection_size: usize) -> usize {
    projection_size.div_ceil(16)
}

pub fn frontier_neural_layer_workgroups_x(batch_count: usize) -> usize {
    batch_count.div_ceil(16)
}

pub fn frontier_neural_layer_workgroups_y(output_size: usize) -> usize {
    output_size.div_ceil(16)
}

pub fn frontier_neural_output_workgroups(batch_count: usize) -> usize {
    batch_count.div_ceil(64)
}

pub fn frontier_policy_workgroups(candidate_count: usize) -> usize {
    candidate_count.div_ceil(64)
}

pub fn frontier_expand_workgroups(count: usize, candidate_workgroup_size: usize) -> usize {
    count.div_ceil(candidate_workgroup_size.max(1))
}

pub fn frontier_selection_workgroups(capacity: usize, candidate_workgroup_size: usize) -> usize {
    capacity.div_ceil(candidate_workgroup_size.max(1))
}

pub fn frontier_materialize_workgroups(frontier_width: usize, mutation_tile_size: usize) -> usize {
    frontier_width.div_ceil(mutation_tile_size.max(1))
}

pub fn frontier_minimax_workgroups(frontier_width: usize) -> usize {
    frontier_width.div_ceil(64)
}

pub fn frontier_policy_params_bytes(
    candidate_count: usize,
    candidate_stride: usize,
    input_size: usize,
    policy_scale: f32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    push_u32(&mut bytes, candidate_count as u32);
    push_u32(&mut bytes, candidate_stride as u32);
    push_u32(&mut bytes, input_size as u32);
    push_f32(&mut bytes, policy_scale);
    bytes
}

pub fn derive_frontier_tuning(
    limits: FrontierTuningLimits,
    requested_nodes: usize,
    board_count: usize,
    additional_board_capacity: usize,
) -> FrontierTuning {
    let storage_limit = gpu_frontier_positive_limit(
        limits.max_storage_buffer_binding_size,
        DEFAULT_STORAGE_LIMIT,
    );
    let buffer_limit = gpu_frontier_positive_limit(limits.max_buffer_size, DEFAULT_BUFFER_LIMIT);
    let max_invocations =
        gpu_frontier_positive_limit(limits.max_compute_invocations_per_workgroup, 256);
    let min_device_limit = storage_limit.min(buffer_limit);
    let state_words_at_min_width = storage_limit / MIN_FRONTIER_WIDTH / std::mem::size_of::<i32>();
    let max_boards_by_state = state_words_at_min_width
        .saturating_sub(FRONTIER_HEADER_STRIDE + FRONTIER_PLAN_STRIDE)
        / FRONTIER_BOARD_STRIDE;
    let max_boards_by_state = max_boards_by_state.max(1);
    let desired_max_boards = 64.min(gpu_frontier_next_power_of_two(
        board_count.max(board_count.saturating_add(additional_board_capacity)),
    ));
    let max_boards = board_count.max(desired_max_boards.min(max_boards_by_state));
    let state_bytes = frontier_state_bytes(max_boards);
    let frontier_width = gpu_frontier_clamp_usize(
        min_device_limit / (state_bytes.saturating_mul(2)).max(1),
        MIN_FRONTIER_WIDTH,
        MAX_FRONTIER_WIDTH.min(MIN_FRONTIER_WIDTH.max(requested_nodes)),
    );
    let candidate_record_bytes =
        (FRONTIER_CANDIDATE_STRIDE + FRONTIER_DELTA_STRIDE) * std::mem::size_of::<i32>();
    let candidate_capacity = gpu_frontier_clamp_usize(
        min_device_limit / candidate_record_bytes.max(1),
        MIN_CANDIDATES,
        MAX_CANDIDATES.min(MIN_CANDIDATES.max(requested_nodes.saturating_mul(4))),
    );
    let neural_bytes_per_sample = 32 * 64 * 16 * std::mem::size_of::<f32>();
    let neural_batch_size =
        gpu_frontier_clamp_usize(storage_limit / neural_bytes_per_sample, 1, frontier_width);
    let candidate_workgroup_size = gpu_frontier_workgroup_size(max_invocations);
    let mutation_tile_size = if candidate_workgroup_size >= 128 {
        128
    } else if candidate_workgroup_size >= 64 {
        64
    } else {
        32
    };
    let dispatch_candidate_limit = candidate_workgroup_size
        .max(candidate_capacity.min(candidate_workgroup_size.saturating_mul(1024)));
    FrontierTuning {
        max_boards,
        frontier_width,
        candidate_capacity,
        neural_batch_size,
        candidate_workgroup_size,
        mutation_tile_size,
        dispatch_candidate_limit,
    }
}

pub fn frontier_selection_plan(
    tuning: &FrontierTuning,
    max_selection_scan: Option<usize>,
) -> FrontierSelectionPlan {
    let candidate_capacity = gpu_frontier_floor_power_of_two(tuning.candidate_capacity);
    let max_scan = max_selection_scan.unwrap_or(tuning.frontier_width.saturating_mul(4));
    let selection_capacity =
        gpu_frontier_floor_power_of_two(candidate_capacity.min(MAX_SELECTION_SCAN).min(max_scan));
    FrontierSelectionPlan {
        candidate_capacity,
        selection_capacity,
    }
}

pub fn frontier_max_cycles(requested_depth: i32, timeline_count: usize) -> i32 {
    let depth = requested_depth.max(1);
    let timeline_span = timeline_count.saturating_add(2).max(2) as i32;
    FRONTIER_MAX_PLAN_MOVES.min((depth * timeline_span).max(depth + 1) as usize) as i32
}

pub fn frontier_per_parent_limit(frontier_width: usize) -> i32 {
    gpu_frontier_clamp_usize(frontier_width.div_ceil(8), 2, 16) as i32
}

pub fn frontier_next_active_state_limit(
    frontier_width: usize,
    active_state_limit: usize,
    per_parent_limit: i32,
) -> usize {
    let per_parent_limit = per_parent_limit.max(1) as usize;
    frontier_width.min(active_state_limit.saturating_mul(per_parent_limit))
}

pub fn gpu_candidate_move_from_record(records: &[i32], index: usize) -> GpuCandidateMove {
    let offset = index.saturating_mul(GPU_CANDIDATE_STRIDE);
    GpuCandidateMove {
        from: GpuCandidatePosition {
            timeline_id: *records.get(offset + 11).unwrap_or(&0),
            time: *records.get(offset + 12).unwrap_or(&0),
            x: *records.get(offset + 13).unwrap_or(&0),
            y: *records.get(offset + 14).unwrap_or(&0),
        },
        to: GpuCandidatePosition {
            timeline_id: *records.get(offset + 15).unwrap_or(&0),
            time: *records.get(offset + 16).unwrap_or(&0),
            x: *records.get(offset + 17).unwrap_or(&0),
            y: *records.get(offset + 18).unwrap_or(&0),
        },
    }
}

pub fn gpu_child_is_source_advance(child: GpuChildBoardRef, source: GpuCandidatePosition) -> bool {
    child.timeline_id == source.timeline_id && child.time == source.time + 1
}

pub fn gpu_next_branch_row(
    occupied_rows: &[i32],
    source_row: i32,
    owner: &str,
) -> Result<i32, String> {
    let direction = match owner.to_ascii_lowercase().as_str() {
        "white" => 1,
        "black" => -1,
        _ => return Err(format!("unsupported GPU branch owner: {owner}")),
    };
    let mut row = source_row + direction;
    while occupied_rows.contains(&row) {
        row += direction;
    }
    Ok(row)
}

pub fn gpu_square_record_from_code(input: GpuSquareRecordInput) -> [i32; GPU_SOURCE_STRIDE] {
    [
        input.piece_code & 255,
        (input.piece_code >> 8) & 255,
        input.timeline_id,
        input.time,
        input.x,
        input.y,
        input.timeline_row,
        input.side_to_move,
        input.owner,
        i32::from(input.latest),
    ]
}

pub fn gpu_target_square_records_for_board(
    board: &GpuSquareRecordBoardInput,
) -> Vec<GpuCandidateSquareRecord> {
    gpu_square_records_for_board(board, false)
}

pub fn gpu_source_square_records_for_board(
    board: &GpuSquareRecordBoardInput,
) -> Vec<GpuCandidateSquareRecord> {
    gpu_square_records_for_board(board, true)
}

fn gpu_square_records_for_board(
    board: &GpuSquareRecordBoardInput,
    occupied_only: bool,
) -> Vec<GpuCandidateSquareRecord> {
    let mut records = Vec::with_capacity(if occupied_only { 0 } else { 64 });
    for y in 0..8 {
        for x in 0..8 {
            let piece_code = *board.squares.get(y * 8 + x).unwrap_or(&0);
            if occupied_only && (piece_code & 255) == 0 {
                continue;
            }
            let meta = GpuCandidatePosition {
                timeline_id: board.timeline_id,
                time: board.time,
                x: x as i32,
                y: y as i32,
            };
            records.push(GpuCandidateSquareRecord {
                meta,
                words: gpu_square_record_from_code(GpuSquareRecordInput {
                    piece_code,
                    timeline_id: board.timeline_id,
                    time: board.time,
                    x: x as i32,
                    y: y as i32,
                    timeline_row: board.timeline_row,
                    side_to_move: board.side_to_move,
                    owner: board.owner,
                    latest: board.latest,
                }),
            });
        }
    }
    records
}

pub fn gpu_candidate_board_records_from_snapshot(
    board: &GpuCandidateBoardInput,
) -> GpuCandidateBoardRecords {
    let board_record = gpu_board_record_from_snapshot(&GpuBoardRecordInput {
        timeline_id: board.timeline_id,
        timeline_row: board.timeline_row,
        time: board.time,
        side_to_move: board.side_to_move,
        castling: board.castling,
        en_passant: board.en_passant,
        squares: board.squares.clone(),
    });
    let mutation_board = gpu_mutation_board_record_from_snapshot(&GpuMutationBoardRecordInput {
        timeline_index: board.timeline_index,
        timeline_id: board.timeline_id,
        time: board.time,
        side_to_move: board.side_to_move,
        castling: board.castling,
        en_passant: board.en_passant,
        latest: board.latest,
        origin_kind: board.origin_kind,
        squares: board.squares.clone(),
    });
    let square_board = GpuSquareRecordBoardInput {
        timeline_id: board.timeline_id,
        time: board.time,
        timeline_row: board.timeline_row,
        side_to_move: board.side_to_move,
        owner: board.owner,
        latest: board.latest,
        squares: board.squares.clone(),
    };
    GpuCandidateBoardRecords {
        board: board_record,
        mutation_board,
        sources: gpu_source_square_records_for_board(&square_board),
        targets: gpu_target_square_records_for_board(&square_board),
    }
}

pub fn gpu_candidate_inputs_from_timelines(
    timelines: &[GpuCandidateInputTimeline],
) -> GpuCandidateInputs {
    let sort_keys = timelines
        .iter()
        .map(|timeline| GpuTimelineSortKey {
            row: timeline.row,
            id: timeline.id,
        })
        .collect::<Vec<_>>();
    let mut output = GpuCandidateInputs {
        source_meta: Vec::new(),
        target_meta: Vec::new(),
        source_count: 0,
        target_count: 0,
        board_count: 0,
        sources: Vec::new(),
        targets: Vec::new(),
        boards: Vec::new(),
        mutation_boards: Vec::new(),
    };

    for timeline_index in gpu_timeline_sort_order(&sort_keys) {
        let Some(timeline) = timelines.get(timeline_index) else {
            continue;
        };
        let latest_time = gpu_latest_board_index(
            &timeline
                .boards
                .iter()
                .map(|board| board.time)
                .collect::<Vec<_>>(),
        )
        .and_then(|index| timeline.boards.get(index))
        .map(|board| board.time);

        for board in &timeline.boards {
            let records = gpu_candidate_board_records_from_snapshot(&GpuCandidateBoardInput {
                timeline_id: timeline.id,
                timeline_row: timeline.row,
                timeline_index: 0,
                time: board.time,
                side_to_move: board.side_to_move,
                owner: timeline.owner,
                castling: board.castling,
                en_passant: board.en_passant,
                latest: latest_time.is_some_and(|latest| board.time == latest),
                origin_kind: board.origin_kind,
                squares: board.squares.clone(),
            });
            output.boards.extend(records.board);
            output.mutation_boards.extend(records.mutation_board);
            for target in records.targets {
                output.target_meta.push(target.meta);
                output.targets.extend(target.words);
            }
            for source in records.sources {
                output.source_meta.push(source.meta);
                output.sources.extend(source.words);
            }
            output.board_count += 1;
        }
    }
    output.source_count = output.source_meta.len();
    output.target_count = output.target_meta.len();
    output
}

pub fn gpu_candidate_inputs_from_snapshot_json(
    snapshot_json: &str,
) -> Result<GpuCandidateInputs, String> {
    let game = parse_game_snapshot(snapshot_json)?;
    Ok(gpu_candidate_inputs_from_game(&game))
}

pub fn gpu_candidate_inputs_json_from_snapshot_json(snapshot_json: &str) -> Result<String, String> {
    Ok(gpu_candidate_inputs_json(
        &gpu_candidate_inputs_from_snapshot_json(snapshot_json)?,
    ))
}

pub fn gpu_candidate_inputs_i32s_from_snapshot_json(
    snapshot_json: &str,
) -> Result<Vec<i32>, String> {
    Ok(gpu_candidate_inputs_i32s(
        &gpu_candidate_inputs_from_snapshot_json(snapshot_json)?,
    ))
}

pub fn gpu_snapshot_game_json(snapshot_json: &str) -> Result<String, String> {
    let snapshot = serde_json::from_str::<GpuSnapshotJson>(snapshot_json)
        .map_err(|error| format!("GPU snapshot game conversion JSON is invalid: {error}"))?;
    let timelines = snapshot
        .timelines
        .iter()
        .map(|timeline| {
            let mut boards = timeline
                .boards
                .iter()
                .map(gpu_game_board_json_from_json)
                .collect::<Result<Vec<_>, _>>()?;
            boards.sort_by_key(|board| {
                board
                    .get("time")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0)
            });
            Ok(serde_json::json!({
                "id": timeline.id,
                "row": timeline.row,
                "label": timeline.label.clone().unwrap_or_else(|| format!("T{}", timeline.id)),
                "owner": timeline.owner,
                "boards": boards,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let value = serde_json::json!({
        "turn": snapshot.turn,
        "nextTimelineId": snapshot.next_timeline_id.unwrap_or(1),
        "nextBlackTimelineId": snapshot.next_black_timeline_id.unwrap_or(-1),
        "royalCaptureBy": snapshot.royal_capture_by,
        "checkedRoyals": [],
        "timelines": timelines,
    });
    serde_json::to_string(&value)
        .map_err(|error| format!("GPU snapshot game conversion response failed to encode: {error}"))
}

pub fn gpu_snapshot_with_child_boards_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuSnapshotChildBoardsRequest>(request_json)
        .map_err(|error| format!("GPU child snapshot request is invalid: {error}"))?;
    let snapshot = request.snapshot;
    let movement = request.movement.as_ref();
    let branch_status = request.mutation_status == GPU_MUTATION_STATUS_BRANCH_OK
        || request.mutation_status == GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE;
    let royal_capture = request.mutation_status == GPU_MUTATION_STATUS_ROYAL_CAPTURE
        || request.mutation_status == GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE;
    let historical_branch = movement.is_some_and(|movement| {
        branch_status
            && !gpu_snapshot_latest_board_is(&snapshot, movement.to.timeline_id, movement.to.time)
    });

    let mut record_slices = vec![request
        .child_board_records
        .get(0..GPU_MUTATION_BOARD_STRIDE)
        .unwrap_or(&request.child_board_records[..])];
    if branch_status {
        record_slices.push(
            request
                .child_board_records
                .get(GPU_MUTATION_BOARD_STRIDE..GPU_MUTATION_CHILD_STRIDE)
                .unwrap_or(&[]),
        );
    }
    let next_turn = record_slices
        .last()
        .map(|record| gpu_search_color_from_code(*record.get(3).unwrap_or(&0)))
        .unwrap_or(snapshot.turn.as_str());

    let mut child_by_timeline =
        std::collections::BTreeMap::<i32, Vec<GpuMutationBoardSnapshot>>::new();
    let mut historical_branch_child = None;
    for record in record_slices {
        let child = gpu_mutation_board_record_to_snapshot(record);
        if historical_branch
            && movement.is_some_and(|movement| {
                !gpu_child_is_source_advance(
                    GpuChildBoardRef {
                        timeline_id: child.timeline_id,
                        time: child.time,
                    },
                    gpu_move_position_candidate(&movement.from),
                )
            })
        {
            historical_branch_child = Some(child);
            continue;
        }
        child_by_timeline
            .entry(child.timeline_id)
            .or_default()
            .push(child);
    }

    let mut timeline_values = Vec::with_capacity(snapshot.timelines.len() + 1);
    for (timeline_index, timeline) in snapshot.timelines.iter().enumerate() {
        let children = child_by_timeline
            .get(&timeline.id)
            .cloned()
            .unwrap_or_default();
        let mut boards = Vec::with_capacity(timeline.boards.len() + children.len());
        for board in &timeline.boards {
            boards.push(gpu_snapshot_board_json_from_json(
                board,
                if children.is_empty() {
                    None
                } else {
                    Some(false)
                },
            )?);
        }
        for child in children {
            boards.push(gpu_mutation_child_board_json(
                &child,
                timeline_index as i32,
                movement.map(|movement| gpu_child_origin_json(&child, movement)),
            ));
        }
        let latest_time = boards
            .iter()
            .filter_map(|board| board.get("time").and_then(serde_json::Value::as_i64))
            .max()
            .unwrap_or(0) as i32;
        timeline_values.push(serde_json::json!({
            "id": timeline.id,
            "row": timeline.row,
            "label": timeline.label.clone().unwrap_or_else(|| format!("T{}", timeline.id)),
            "owner": timeline.owner,
            "boardCount": boards.len(),
            "latestTime": latest_time,
            "boards": boards,
        }));
    }

    let mut next_timeline_id = snapshot.next_timeline_id.unwrap_or(1);
    let mut next_black_timeline_id = snapshot.next_black_timeline_id.unwrap_or(-1);
    if let (Some(child), Some(movement)) = (historical_branch_child.as_ref(), movement) {
        let new_timeline_id = if snapshot.turn == "white" {
            let id = next_timeline_id;
            next_timeline_id += 1;
            id
        } else {
            let id = next_black_timeline_id;
            next_black_timeline_id -= 1;
            id
        };
        let source_row = snapshot
            .timelines
            .iter()
            .find(|timeline| timeline.id == movement.from.timeline_id)
            .map(|timeline| timeline.row)
            .unwrap_or(0);
        let occupied_rows = timeline_values
            .iter()
            .filter_map(|timeline| timeline.get("row").and_then(serde_json::Value::as_i64))
            .map(|row| row as i32)
            .collect::<Vec<_>>();
        let row = gpu_next_branch_row(&occupied_rows, source_row, &snapshot.turn)?;
        let board = gpu_mutation_child_board_json(
            child,
            timeline_values.len() as i32,
            Some(gpu_branch_origin_json(movement)),
        );
        timeline_values.push(serde_json::json!({
            "id": new_timeline_id,
            "row": row,
            "label": format!("{} T{}", if snapshot.turn == "white" { "White" } else { "Black" }, new_timeline_id),
            "owner": snapshot.turn,
            "boardCount": 1,
            "latestTime": child.time,
            "boards": [board],
        }));
    }

    let flat_boards = timeline_values
        .iter()
        .flat_map(|timeline| {
            timeline
                .get("boards")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "format": snapshot.format.unwrap_or_else(|| "engine-gpu-snapshot-v1".to_string()),
        "turn": if request.advance_turn.unwrap_or(true) { next_turn } else { snapshot.turn.as_str() },
        "nextTimelineId": next_timeline_id,
        "nextBlackTimelineId": next_black_timeline_id,
        "royalCaptureBy": if royal_capture {
            serde_json::Value::String(snapshot.turn)
        } else {
            snapshot
                .royal_capture_by
                .map(serde_json::Value::String)
                .unwrap_or(serde_json::Value::Null)
        },
        "timelines": timeline_values,
        "boards": flat_boards,
    });
    serde_json::to_string(&value)
        .map_err(|error| format!("GPU child snapshot response failed to encode: {error}"))
}

pub(crate) fn gpu_candidate_inputs_json_from_game(game: &Game) -> String {
    gpu_candidate_inputs_json(&gpu_candidate_inputs_from_game(game))
}

pub(crate) fn gpu_candidate_inputs_i32s_from_game(game: &Game) -> Vec<i32> {
    gpu_candidate_inputs_i32s(&gpu_candidate_inputs_from_game(game))
}

fn gpu_candidate_inputs_json(inputs: &GpuCandidateInputs) -> String {
    let source_meta = inputs
        .source_meta
        .iter()
        .map(position_json)
        .collect::<Vec<_>>();
    let target_meta = inputs
        .target_meta
        .iter()
        .map(position_json)
        .collect::<Vec<_>>();
    serde_json::json!({
        "sourceMeta": source_meta,
        "targetMeta": target_meta,
        "sourceCount": inputs.source_count,
        "targetCount": inputs.target_count,
        "boardCount": inputs.board_count,
        "sources": inputs.sources,
        "targets": inputs.targets,
        "boards": inputs.boards,
        "mutationBoards": inputs.mutation_boards,
    })
    .to_string()
}

pub fn gpu_candidate_input_meta_json_from_i32s(words: &[i32]) -> Result<String, String> {
    if words.len() < GPU_CANDIDATE_INPUT_HEADER_I32S {
        return Err("GPU candidate input metadata is truncated.".to_string());
    }
    let source_length = candidate_input_length(words[3], "source")?;
    let target_length = candidate_input_length(words[4], "target")?;
    let board_length = candidate_input_length(words[5], "board")?;
    let mutation_board_length = candidate_input_length(words[6], "mutation board")?;
    let total_length = GPU_CANDIDATE_INPUT_HEADER_I32S
        .checked_add(source_length)
        .and_then(|value| value.checked_add(target_length))
        .and_then(|value| value.checked_add(board_length))
        .and_then(|value| value.checked_add(mutation_board_length))
        .ok_or_else(|| "GPU candidate input metadata length overflows.".to_string())?;
    if total_length != words.len() {
        return Err("GPU candidate input metadata length does not match header.".to_string());
    }
    if source_length % GPU_SOURCE_STRIDE != 0 {
        return Err("GPU candidate input source records are misaligned.".to_string());
    }
    if target_length % GPU_TARGET_STRIDE != 0 {
        return Err("GPU candidate input target records are misaligned.".to_string());
    }
    let source_start = GPU_CANDIDATE_INPUT_HEADER_I32S;
    let target_start = source_start + source_length;
    let source_meta =
        gpu_candidate_meta_from_records(&words[source_start..target_start], GPU_SOURCE_STRIDE);
    let target_meta = gpu_candidate_meta_from_records(
        &words[target_start..target_start + target_length],
        GPU_TARGET_STRIDE,
    );
    let value = serde_json::json!({
        "sourceMeta": source_meta.iter().map(position_json).collect::<Vec<_>>(),
        "targetMeta": target_meta.iter().map(position_json).collect::<Vec<_>>(),
    });
    serde_json::to_string(&value)
        .map_err(|error| format!("GPU candidate input metadata failed to encode: {error}"))
}

fn candidate_input_length(value: i32, label: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("GPU candidate input {label} length is negative."))
}

pub fn gpu_candidate_meta_from_records(
    records: &[i32],
    stride: usize,
) -> Vec<GpuCandidatePosition> {
    let mut meta = Vec::new();
    for offset in (0..records.len()).step_by(stride.max(1)) {
        if offset + stride > records.len() {
            break;
        }
        meta.push(GpuCandidatePosition {
            timeline_id: *records.get(offset + 2).unwrap_or(&0),
            time: *records.get(offset + 3).unwrap_or(&0),
            x: *records.get(offset + 4).unwrap_or(&0),
            y: *records.get(offset + 5).unwrap_or(&0),
        });
    }
    meta
}

fn position_json(position: &GpuCandidatePosition) -> serde_json::Value {
    serde_json::json!({
        "timelineId": position.timeline_id,
        "time": position.time,
        "x": position.x,
        "y": position.y,
    })
}

fn gpu_candidate_inputs_i32s(inputs: &GpuCandidateInputs) -> Vec<i32> {
    let mut words = Vec::with_capacity(
        GPU_CANDIDATE_INPUT_HEADER_I32S
            + inputs.sources.len()
            + inputs.targets.len()
            + inputs.boards.len()
            + inputs.mutation_boards.len(),
    );
    words.extend_from_slice(&[
        inputs.source_count as i32,
        inputs.target_count as i32,
        inputs.board_count as i32,
        inputs.sources.len() as i32,
        inputs.targets.len() as i32,
        inputs.boards.len() as i32,
        inputs.mutation_boards.len() as i32,
    ]);
    words.extend_from_slice(&inputs.sources);
    words.extend_from_slice(&inputs.targets);
    words.extend_from_slice(&inputs.boards);
    words.extend_from_slice(&inputs.mutation_boards);
    words
}

fn gpu_candidate_inputs_from_game(game: &Game) -> GpuCandidateInputs {
    let timelines = game
        .timelines
        .iter()
        .map(|timeline| GpuCandidateInputTimeline {
            id: timeline.id,
            row: timeline.row,
            owner: owner_code(timeline.owner),
            boards: timeline
                .boards
                .iter()
                .map(|board| GpuCandidateInputBoard {
                    time: board.time,
                    side_to_move: color_code(board.side_to_move),
                    castling: castling_code(board.castling),
                    en_passant: board.en_passant.map(|en_passant| GpuEnPassantRecord {
                        x: en_passant.x,
                        y: en_passant.y,
                        captured_x: en_passant.captured_x,
                        captured_y: en_passant.captured_y,
                    }),
                    origin_kind: 0,
                    squares: square_codes_for_board_snapshot(board),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    gpu_candidate_inputs_from_timelines(&timelines)
}

pub fn gpu_board_record_from_snapshot(board: &GpuBoardRecordInput) -> Vec<i32> {
    let mut record = Vec::with_capacity(GPU_BOARD_STRIDE);
    record.extend_from_slice(&[
        board.timeline_id,
        board.timeline_row,
        board.time,
        board.side_to_move,
        board.castling,
    ]);
    push_en_passant_record(&mut record, board.en_passant);
    for index in 0..64 {
        record.push(*board.squares.get(index).unwrap_or(&0));
    }
    record
}

pub fn gpu_mutation_board_record_from_snapshot(board: &GpuMutationBoardRecordInput) -> Vec<i32> {
    let mut record = Vec::with_capacity(GPU_MUTATION_BOARD_STRIDE);
    record.extend_from_slice(&[
        board.timeline_index,
        board.timeline_id,
        board.time,
        board.side_to_move,
        board.castling,
    ]);
    push_en_passant_record(&mut record, board.en_passant);
    record.extend_from_slice(&[i32::from(board.latest), board.origin_kind, 0]);
    for index in 0..64 {
        record.push(*board.squares.get(index).unwrap_or(&0));
    }
    record
}

pub fn gpu_mutation_board_record_to_snapshot(record: &[i32]) -> GpuMutationBoardSnapshot {
    let en_passant_x = *record.get(5).unwrap_or(&-1);
    GpuMutationBoardSnapshot {
        timeline_index: *record.first().unwrap_or(&0),
        timeline_id: *record.get(1).unwrap_or(&0),
        time: *record.get(2).unwrap_or(&0),
        side_to_move: gpu_search_color_from_code(*record.get(3).unwrap_or(&0)),
        castling: *record.get(4).unwrap_or(&0),
        en_passant: (en_passant_x >= 0).then(|| GpuEnPassantRecord {
            x: en_passant_x,
            y: *record.get(6).unwrap_or(&-1),
            captured_x: *record.get(7).unwrap_or(&-1),
            captured_y: *record.get(8).unwrap_or(&-1),
        }),
        latest: true,
        origin_kind: *record.get(10).unwrap_or(&0),
        squares: record
            .get(12..GPU_MUTATION_BOARD_STRIDE)
            .unwrap_or(&[])
            .to_vec(),
    }
}

pub(crate) fn encode_frontier_root(
    game: &Game,
    max_boards: usize,
) -> Result<EncodedFrontierRoot, String> {
    encode_frontier_root_from_timelines(
        color_code(game.turn),
        game.next_timeline_id,
        game.next_black_timeline_id,
        game.staged_royal_capture_by.is_some(),
        &game
            .timelines
            .iter()
            .map(|timeline| GpuCandidateInputTimeline {
                id: timeline.id,
                row: timeline.row,
                owner: owner_code(timeline.owner),
                boards: timeline
                    .boards
                    .iter()
                    .map(|board| GpuCandidateInputBoard {
                        time: board.time,
                        side_to_move: color_code(board.side_to_move),
                        castling: castling_code(board.castling),
                        en_passant: board.en_passant.map(|en_passant| GpuEnPassantRecord {
                            x: en_passant.x,
                            y: en_passant.y,
                            captured_x: en_passant.captured_x,
                            captured_y: en_passant.captured_y,
                        }),
                        origin_kind: origin_code(&board.origin),
                        squares: square_codes_for_board_snapshot(board),
                    })
                    .collect(),
            })
            .collect::<Vec<_>>(),
        max_boards,
    )
}

pub fn gpu_frontier_root_i32s_from_snapshot_json(
    snapshot_json: &str,
    max_boards: usize,
) -> Result<Vec<i32>, String> {
    Ok(encode_frontier_root_from_gpu_snapshot_json(snapshot_json, max_boards)?.words)
}

pub fn gpu_pending_present_boards_json_from_snapshot_json(
    snapshot_json: &str,
) -> Result<String, String> {
    let snapshot = serde_json::from_str::<GpuSnapshotJson>(snapshot_json)
        .map_err(|error| format!("GPU pending board snapshot JSON is invalid: {error}"))?;
    let max_boards = snapshot
        .timelines
        .iter()
        .map(|timeline| timeline.boards.len())
        .sum::<usize>()
        .max(1);
    let turn = gpu_search_color_code(&snapshot.turn)?;
    let timelines = snapshot
        .timelines
        .iter()
        .map(gpu_timeline_from_json)
        .collect::<Result<Vec<_>, _>>()?;
    let root = encode_frontier_root_from_timelines(
        turn,
        snapshot.next_timeline_id.unwrap_or(1),
        snapshot.next_black_timeline_id.unwrap_or(-1),
        snapshot.royal_capture_by.is_some(),
        &timelines,
        max_boards,
    )?;
    let pending = (0..root.board_count)
        .filter_map(|index| {
            let base = FRONTIER_BOARD_OFFSET + index * FRONTIER_BOARD_STRIDE;
            if root.words[base + FRONTIER_BOARD_PENDING] == 0 {
                return None;
            }
            Some(serde_json::json!({
                "timelineId": root.words[base + FRONTIER_BOARD_TIMELINE_ID],
                "time": root.words[base + FRONTIER_BOARD_TIME],
            }))
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&pending)
        .map_err(|error| format!("GPU pending board JSON serialization failed: {error}"))
}

#[derive(serde::Deserialize)]
struct GpuSearchSelectionRequest {
    candidates: Vec<GpuSearchSelectionCandidate>,
    temperature: Option<f64>,
    #[serde(rename = "randomSeed")]
    random_seed: Option<f64>,
}

#[derive(serde::Deserialize)]
struct GpuSearchChoiceSelectionRequest {
    candidates: Vec<serde_json::Value>,
    temperature: Option<f64>,
    #[serde(rename = "randomSeed")]
    random_seed: Option<f64>,
}

#[derive(Clone, serde::Deserialize)]
struct GpuSearchSelectionCandidate {
    index: usize,
    score: f64,
    key: String,
    #[serde(rename = "moveCount")]
    move_count: usize,
}

#[derive(serde::Serialize)]
struct GpuSearchSelectionResponse {
    #[serde(rename = "selectedIndex")]
    selected_index: Option<usize>,
    #[serde(rename = "rankedIndexes")]
    ranked_indexes: Vec<usize>,
}

pub fn gpu_search_select_candidate_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuSearchSelectionRequest>(request_json)
        .map_err(|error| format!("GPU search selection request is invalid: {error}"))?;
    let response = gpu_search_select_candidate(
        request.candidates,
        request.temperature.unwrap_or(0.0),
        gpu_search_random_seed(request.random_seed)?,
    );
    serde_json::to_string(&response)
        .map_err(|error| format!("GPU search selection response failed to encode: {error}"))
}

pub fn gpu_search_select_choice_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuSearchChoiceSelectionRequest>(request_json)
        .map_err(|error| format!("GPU search choice selection request is invalid: {error}"))?;
    let candidates = gpu_search_choice_selection_candidates(&request.candidates)?;
    let response = gpu_search_select_candidate(
        candidates,
        request.temperature.unwrap_or(0.0),
        gpu_search_random_seed(request.random_seed)?,
    );
    serde_json::to_string(&response)
        .map_err(|error| format!("GPU search choice selection response failed to encode: {error}"))
}

pub fn gpu_search_selected_choice_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuSearchChoiceSelectionRequest>(request_json)
        .map_err(|error| format!("GPU selected search choice request is invalid: {error}"))?;
    let selection_candidates = gpu_search_choice_selection_candidates(&request.candidates)?;
    let response = gpu_search_select_candidate(
        selection_candidates,
        request.temperature.unwrap_or(0.0),
        gpu_search_random_seed(request.random_seed)?,
    );
    let Some(selected_index) = response.selected_index else {
        return Ok("null".to_string());
    };
    let Some(selected) = request.candidates.get(selected_index) else {
        return Ok("null".to_string());
    };
    let mut selected = selected.clone();
    let supported = response
        .ranked_indexes
        .iter()
        .filter_map(|index| request.candidates.get(*index).cloned())
        .collect::<Vec<_>>();
    let choices = gpu_summarize_search_choices(&supported);
    if let serde_json::Value::Object(ref mut object) = selected {
        object.insert("choices".to_string(), serde_json::Value::Array(choices));
        serde_json::to_string(&selected).map_err(|error| {
            format!("GPU selected search choice response failed to encode: {error}")
        })
    } else {
        Ok("null".to_string())
    }
}

fn gpu_search_random_seed(random_seed: Option<f64>) -> Result<i64, String> {
    let Some(random_seed) = random_seed.filter(|random_seed| random_seed.is_finite()) else {
        return Ok(0);
    };
    let truncated = random_seed.trunc();
    if truncated < i64::MIN as f64 || truncated > i64::MAX as f64 {
        return Err("GPU search random seed exceeds i64 range.".to_string());
    }
    Ok(truncated as i64)
}

fn gpu_search_choice_selection_candidates(
    candidates: &[serde_json::Value],
) -> Result<Vec<GpuSearchSelectionCandidate>, String> {
    candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let moves = gpu_choice_moves(candidate)?;
            Ok(GpuSearchSelectionCandidate {
                index,
                score: candidate
                    .get("score")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0),
                key: gpu_move_plan_key(&moves),
                move_count: moves.len(),
            })
        })
        .collect::<Result<Vec<_>, String>>()
}

pub fn gpu_turn_status_records_i32s_from_snapshot_json(
    snapshot_json: &str,
) -> Result<Vec<i32>, String> {
    let snapshot = serde_json::from_str::<GpuSnapshotJson>(snapshot_json)
        .map_err(|error| format!("GPU turn-status snapshot JSON is invalid: {error}"))?;
    let mut timelines = snapshot.timelines.iter().collect::<Vec<_>>();
    timelines.sort_by_key(|timeline| (timeline.row, timeline.id));
    let mut records = Vec::with_capacity(timelines.len() * GPU_TURN_STATUS_RECORD_STRIDE);
    for timeline in timelines {
        let Some(board) = timeline.boards.iter().max_by_key(|board| board.time) else {
            continue;
        };
        records.extend([
            timeline.id,
            gpu_search_owner_code(&timeline.owner)?,
            board.time,
            gpu_search_color_code(&board.side_to_move)?,
        ]);
    }
    if records.is_empty() {
        records.extend([0, 0, 0, gpu_search_color_code(&snapshot.turn)?]);
    }
    Ok(records)
}

pub fn gpu_turn_status_json_from_i32s(words: &[i32]) -> Result<String, String> {
    if words.len() < 4 {
        return Err("GPU turn-status response is truncated.".to_string());
    }
    let value = serde_json::json!({
        "complete": words[0] == 0,
        "nextTurn": gpu_search_color_from_code(words[1]),
        "presentTime": words[2],
        "pendingPresentBoardCount": words[3],
    });
    serde_json::to_string(&value)
        .map_err(|error| format!("GPU turn-status response failed to encode: {error}"))
}

pub fn gpu_turn_status_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuTurnStatusRequest>(request_json)
        .map_err(|error| format!("GPU turn-status JSON request is invalid: {error}"))?;
    gpu_turn_status_json_from_i32s(&request.records)
}

pub fn gpu_full_search_precondition_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuFullSearchPreconditionRequest>(request_json)
        .map_err(|error| format!("GPU full-search precondition request is invalid: {error}"))?;
    serde_json::to_string(&gpu_full_search_precondition(
        request.status.pending_present_board_count,
    ))
    .map_err(|error| format!("GPU full-search precondition response failed to encode: {error}"))
}

fn gpu_full_search_precondition(
    pending_present_board_count: Option<i32>,
) -> GpuFullSearchPreconditionResponse {
    let supported = pending_present_board_count == Some(1);
    GpuFullSearchPreconditionResponse {
        supported,
        error: (!supported)
            .then_some("Full GPU search currently requires one pending present board."),
    }
}

pub fn gpu_ranked_candidate_indexes_from_i32s(words: &[i32]) -> Result<Vec<i32>, String> {
    Ok(gpu_ranked_candidate_entries_from_i32s(words)?
        .ranked
        .into_iter()
        .map(|(index, _)| index as i32)
        .collect())
}

pub fn gpu_ranked_candidates_json_from_i32s(words: &[i32]) -> Result<String, String> {
    let ranked = gpu_ranked_candidate_entries_from_i32s(words)?;
    let candidates = ranked
        .ranked
        .iter()
        .map(|(index, score)| {
            let movement = gpu_candidate_move_from_record(ranked.records, *index);
            serde_json::json!({
                "move": {
                    "from": position_json(&movement.from),
                    "to": position_json(&movement.to),
                },
                "index": index,
                "score": score,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&candidates)
        .map_err(|error| format!("GPU ranked candidate response failed to encode: {error}"))
}

pub fn gpu_ranked_candidates_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuRankedCandidatesRequest>(request_json)
        .map_err(|error| format!("GPU ranked candidates JSON request is invalid: {error}"))?;
    let candidate_count = request.scores.len();
    let expected_record_len = candidate_count
        .checked_mul(GPU_CANDIDATE_STRIDE)
        .ok_or_else(|| "GPU ranked candidates JSON record length overflowed.".to_string())?;
    if request.records.len() < expected_record_len {
        return Err(format!(
            "GPU ranked candidates JSON record length mismatch: expected at least {expected_record_len}, got {}.",
            request.records.len()
        ));
    }
    let mut words = Vec::with_capacity(
        4 + candidate_count + expected_record_len + request.pending_boards.len() * 2,
    );
    words.push(
        i32::try_from(candidate_count)
            .map_err(|_| "GPU ranked candidates count exceeds i32 range.".to_string())?,
    );
    words.push(
        i32::try_from(request.limit)
            .map_err(|_| "GPU ranked candidates limit exceeds i32 range.".to_string())?,
    );
    words.push(
        i32::try_from(request.pending_boards.len())
            .map_err(|_| "GPU ranked candidates pending count exceeds i32 range.".to_string())?,
    );
    words.push(if request.require_pending { 1 } else { 0 });
    words.extend(request.scores);
    words.extend(request.records.into_iter().take(expected_record_len));
    for board in request.pending_boards {
        words.push(board.timeline_id);
        words.push(board.time);
    }
    gpu_ranked_candidates_json_from_i32s(&words)
}

pub fn gpu_mutation_selected_candidates_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuMutationSelectedCandidatesRequest>(request_json)
        .map_err(|error| format!("GPU mutation selected candidates request is invalid: {error}"))?;
    let selected = request
        .ranked
        .into_iter()
        .take(request.limit)
        .collect::<Vec<_>>();
    serde_json::to_string(&selected).map_err(|error| {
        format!("GPU mutation selected candidates response failed to encode: {error}")
    })
}

pub fn gpu_candidate_indexes_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuCandidateIndexesRequest>(request_json)
        .map_err(|error| format!("GPU candidate indexes request is invalid: {error}"))?;
    let indexes = request
        .candidates
        .iter()
        .map(gpu_candidate_json_index)
        .collect::<Result<Vec<_>, _>>()?;
    serde_json::to_string(&indexes)
        .map_err(|error| format!("GPU candidate indexes response failed to encode: {error}"))
}

struct GpuRankedCandidateEntries<'a> {
    records: &'a [i32],
    ranked: Vec<(usize, i32)>,
}

fn gpu_ranked_candidate_entries_from_i32s(
    words: &[i32],
) -> Result<GpuRankedCandidateEntries<'_>, String> {
    if words.len() < 4 {
        return Err("GPU candidate ranking request is truncated.".to_string());
    }
    let candidate_count = non_negative_usize(words[0], "candidate count")?;
    let limit = non_negative_usize(words[1], "candidate ranking limit")?;
    let pending_count = non_negative_usize(words[2], "pending board count")?;
    let require_pending = words[3] != 0;
    let score_offset = 4;
    let record_offset = score_offset + candidate_count;
    let record_len = candidate_count
        .checked_mul(GPU_CANDIDATE_STRIDE)
        .ok_or_else(|| "GPU candidate ranking record length overflowed.".to_string())?;
    let pending_offset = record_offset + record_len;
    let pending_len = pending_count
        .checked_mul(2)
        .ok_or_else(|| "GPU candidate ranking pending length overflowed.".to_string())?;
    let expected_len = pending_offset + pending_len;
    if words.len() != expected_len {
        return Err(format!(
            "GPU candidate ranking request length mismatch: expected {expected_len}, got {}.",
            words.len()
        ));
    }
    let scores = &words[score_offset..record_offset];
    let records = &words[record_offset..pending_offset];
    let pending = words[pending_offset..]
        .chunks_exact(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect::<Vec<_>>();
    let mut ranked = scores
        .iter()
        .enumerate()
        .filter_map(|(index, score)| {
            if *score <= -2_147_480_000 {
                return None;
            }
            let candidate = gpu_candidate_move_from_record(records, index);
            if require_pending
                && !pending.iter().any(|(timeline_id, time)| {
                    *timeline_id == candidate.from.timeline_id && *time == candidate.from.time
                })
            {
                return None;
            }
            Some((index, *score))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranked.truncate(limit);
    Ok(GpuRankedCandidateEntries { records, ranked })
}

pub fn gpu_scoring_summary_from_i32s(words: &[i32]) -> Result<String, String> {
    if words.len() < 2 {
        return Err("GPU scoring summary request is truncated.".to_string());
    }
    let candidate_count = non_negative_usize(words[0], "candidate count")?;
    let pending_count = non_negative_usize(words[1], "pending board count")?;
    let score_offset = 2;
    let record_offset = score_offset + candidate_count;
    let record_len = candidate_count
        .checked_mul(GPU_CANDIDATE_STRIDE)
        .ok_or_else(|| "GPU scoring summary record length overflowed.".to_string())?;
    let pending_offset = record_offset + record_len;
    let pending_len = pending_count
        .checked_mul(2)
        .ok_or_else(|| "GPU scoring summary pending length overflowed.".to_string())?;
    let expected_len = pending_offset + pending_len;
    if words.len() != expected_len {
        return Err(format!(
            "GPU scoring summary request length mismatch: expected {expected_len}, got {}.",
            words.len()
        ));
    }
    let scores = &words[score_offset..record_offset];
    let records = &words[record_offset..pending_offset];
    let pending = words[pending_offset..]
        .chunks_exact(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect::<Vec<_>>();
    let mut valid_score_count = 0;
    let mut pending_start_count = 0;
    let mut best = -2_147_483_647;
    for (index, score) in scores.iter().enumerate() {
        if *score <= -2_147_480_000 {
            continue;
        }
        valid_score_count += 1;
        best = best.max(*score);
        let candidate = gpu_candidate_move_from_record(records, index);
        if pending.iter().any(|(timeline_id, time)| {
            *timeline_id == candidate.from.timeline_id && *time == candidate.from.time
        }) {
            pending_start_count += 1;
        }
    }
    Ok(format!(
        "validScores={valid_score_count}, pendingStarts={pending_start_count}, best={best}"
    ))
}

pub fn gpu_scoring_summary_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuScoringSummaryRequest>(request_json)
        .map_err(|error| format!("GPU scoring summary JSON request is invalid: {error}"))?;
    let candidate_count = request.scores.len();
    let expected_record_len = candidate_count
        .checked_mul(GPU_CANDIDATE_STRIDE)
        .ok_or_else(|| "GPU scoring summary JSON record length overflowed.".to_string())?;
    if request.records.len() < expected_record_len {
        return Err(format!(
            "GPU scoring summary JSON record length mismatch: expected at least {expected_record_len}, got {}.",
            request.records.len()
        ));
    }
    let mut words = Vec::with_capacity(
        2 + candidate_count + expected_record_len + request.pending_boards.len() * 2,
    );
    words.push(candidate_count as i32);
    words.push(
        i32::try_from(request.pending_boards.len()).map_err(|_| {
            "GPU scoring summary pending board count exceeds i32 range.".to_string()
        })?,
    );
    words.extend(request.scores);
    words.extend(request.records.into_iter().take(expected_record_len));
    for board in request.pending_boards {
        words.push(board.timeline_id);
        words.push(board.time);
    }
    gpu_scoring_summary_from_i32s(&words)
}

pub fn gpu_mutation_summary_from_i32s(statuses: &[i32]) -> String {
    if statuses.is_empty() {
        return "none".to_string();
    }
    let mut counts = std::collections::BTreeMap::<i32, usize>::new();
    for status in statuses {
        *counts.entry(*status).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(status, count)| format!("{status}:{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

pub fn gpu_mutation_summary_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuMutationSummaryRequest>(request_json)
        .map_err(|error| format!("GPU mutation summary JSON request is invalid: {error}"))?;
    Ok(gpu_mutation_summary_from_i32s(&request.statuses))
}

#[derive(serde::Deserialize)]
struct GpuPendingBoardRefJson {
    #[serde(rename = "timelineId")]
    timeline_id: i32,
    time: i32,
}

#[derive(Clone, Copy, serde::Deserialize, serde::Serialize)]
struct GpuMovePositionJson {
    #[serde(rename = "timelineId")]
    timeline_id: i32,
    time: i32,
    x: i32,
    y: i32,
}

#[derive(Clone, Copy, serde::Deserialize, serde::Serialize)]
struct GpuMoveJson {
    from: GpuMovePositionJson,
    to: GpuMovePositionJson,
}

impl From<GpuMovePositionJson> for Position {
    fn from(position: GpuMovePositionJson) -> Self {
        Position {
            timeline_id: position.timeline_id,
            time: position.time,
            x: position.x,
            y: position.y,
        }
    }
}

#[derive(serde::Deserialize)]
struct GpuChoiceAgreementRequest {
    selected: Vec<GpuMoveJson>,
    choices: Vec<Vec<GpuMoveJson>>,
    limits: Vec<usize>,
}

#[derive(serde::Deserialize)]
struct GpuChoiceAgreementChoicesRequest {
    selected: serde_json::Value,
    choices: Vec<serde_json::Value>,
    limits: Vec<usize>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuSnapshotChildBoardsRequest {
    snapshot: GpuSnapshotJson,
    child_board_records: Vec<i32>,
    mutation_status: i32,
    #[serde(rename = "move")]
    movement: Option<GpuMoveJson>,
    advance_turn: Option<bool>,
}

#[derive(serde::Serialize)]
struct GpuChoiceAgreementResponse {
    agreements: Vec<i32>,
}

#[derive(serde::Deserialize)]
struct GpuNonPostableResultSummaryJson {
    status: Option<String>,
    moves: Option<Vec<serde_json::Value>>,
    #[serde(rename = "incompleteMoves")]
    incomplete_moves: Option<Vec<serde_json::Value>>,
    #[serde(rename = "pendingPresentBoardCount")]
    pending_present_board_count: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
struct GpuPostableSearchResultJson {
    status: Option<String>,
    moves: Option<Vec<serde_json::Value>>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuValidateSearchResultRequest {
    game: serde_json::Value,
    result: serde_json::Value,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuReplyPressureRankedRootsRequest {
    ranked_roots: Vec<serde_json::Value>,
    pair_scores: Vec<i32>,
    reply_count: usize,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuScoringSummaryRequest {
    scores: Vec<i32>,
    records: Vec<i32>,
    pending_boards: Vec<GpuPendingBoardRefJson>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuRankedCandidatesRequest {
    scores: Vec<i32>,
    records: Vec<i32>,
    pending_boards: Vec<GpuPendingBoardRefJson>,
    require_pending: bool,
    limit: usize,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuMutationSelectedCandidatesRequest {
    ranked: Vec<serde_json::Value>,
    limit: usize,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuCandidateIndexesRequest {
    candidates: Vec<serde_json::Value>,
}

#[derive(serde::Deserialize)]
struct GpuTurnStatusRequest {
    records: Vec<i32>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuFullSearchPreconditionRequest {
    status: GpuTurnCompletionStepStatus,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuFullSearchPreconditionResponse {
    supported: bool,
    error: Option<&'static str>,
}

#[derive(serde::Deserialize)]
struct GpuMutationSummaryRequest {
    statuses: Vec<i32>,
}

#[derive(serde::Deserialize)]
struct GpuPickCandidateRecordsRequest {
    records: Vec<i32>,
    indices: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct GpuMutationTurnCodeRequest {
    records: Vec<i32>,
}

#[derive(serde::Deserialize)]
struct GpuCandidateIndexRequest {
    records: Vec<i32>,
    #[serde(rename = "move")]
    movement: GpuMoveJson,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuSupportedMutationCandidatesRequest {
    candidates: Vec<GpuSupportedMutationCandidateRequest>,
    limit: Option<f64>,
    require_child_boards: Option<bool>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuSupportedMutationCandidateRequest {
    mutation_status: i32,
    has_child_boards: bool,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuCompletedTurnChoiceRequest {
    result: serde_json::Value,
    moves: Vec<serde_json::Value>,
    gpu_search: Option<String>,
    principal_variation: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuValidateFirstFrontierTurnRequest {
    game: serde_json::Value,
    moves: Vec<GpuMoveJson>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuValidatedFrontierChoiceRequest {
    candidate: serde_json::Value,
    moves: Vec<serde_json::Value>,
    seen_keys: Vec<String>,
    choice_count: usize,
    choice_limit: usize,
    gpu_search: String,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuValidatedFrontierChoiceResponse {
    accepted: bool,
    key: Option<String>,
    choice: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuFrontierChoiceDiagnosticsRequest {
    selected: serde_json::Value,
    choices: Vec<serde_json::Value>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuFrontierChoiceDiagnosticsResponse {
    legal_choice_count: usize,
    legal_tactical_choice_count: usize,
    selected_move_pruned_risk: i32,
    selected_move_tactical: i32,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuTurnCompletionStepStatus {
    complete: bool,
    pending_present_board_count: Option<i32>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuTurnCompletionStepRequest {
    snapshot: GpuSnapshotJson,
    moves_length: usize,
    pending_boards: Vec<GpuPendingBoardRefJson>,
    status: GpuTurnCompletionStepStatus,
    visited_keys: Option<Vec<String>>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuIncompleteTurnPendingCountRequest {
    pending_boards: Vec<GpuPendingBoardRefJson>,
    status: GpuTurnCompletionStepStatus,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuTurnCompletionStepResponse {
    action: &'static str,
    state_key: Option<String>,
    max_moves: usize,
}

pub fn gpu_turn_completion_key_json(request_json: &str) -> Result<String, String> {
    let pending = serde_json::from_str::<Vec<GpuPendingBoardRefJson>>(request_json)
        .map_err(|error| format!("GPU turn completion key request is invalid: {error}"))?;
    Ok(gpu_turn_completion_key_from_refs(pending))
}

fn gpu_turn_completion_key_from_refs(mut pending: Vec<GpuPendingBoardRefJson>) -> String {
    pending.sort_by_key(|board| (board.timeline_id, board.time));
    pending
        .into_iter()
        .map(|board| format!("{}:{}", board.timeline_id, board.time))
        .collect::<Vec<_>>()
        .join("|")
}

pub fn gpu_turn_completion_step_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuTurnCompletionStepRequest>(request_json)
        .map_err(|error| format!("GPU turn completion step request is invalid: {error}"))?;
    let max_moves =
        gpu_turn_completion_max_moves(request.moves_length, request.snapshot.timelines.len());
    let response = if request.snapshot.royal_capture_by.is_some() {
        GpuTurnCompletionStepResponse {
            action: "terminal",
            state_key: None,
            max_moves,
        }
    } else if request.pending_boards.is_empty()
        && (request.status.complete || request.status.pending_present_board_count == Some(0))
    {
        GpuTurnCompletionStepResponse {
            action: "complete",
            state_key: None,
            max_moves,
        }
    } else if request.moves_length >= max_moves {
        GpuTurnCompletionStepResponse {
            action: "maxMoves",
            state_key: None,
            max_moves,
        }
    } else {
        let state_key = gpu_turn_completion_key_from_refs(request.pending_boards);
        let visited = request.visited_keys.unwrap_or_default();
        if visited.iter().any(|key| key == &state_key) {
            GpuTurnCompletionStepResponse {
                action: "loop",
                state_key: Some(state_key),
                max_moves,
            }
        } else {
            GpuTurnCompletionStepResponse {
                action: "search",
                state_key: Some(state_key),
                max_moves,
            }
        }
    };
    serde_json::to_string(&response)
        .map_err(|error| format!("GPU turn completion step response failed to encode: {error}"))
}

pub fn gpu_incomplete_turn_pending_present_board_count_json(
    request_json: &str,
) -> Result<usize, String> {
    let request = serde_json::from_str::<GpuIncompleteTurnPendingCountRequest>(request_json)
        .map_err(|error| {
            format!("GPU incomplete turn pending count request is invalid: {error}")
        })?;
    Ok(gpu_incomplete_turn_pending_present_board_count(
        request.status.pending_present_board_count,
        request.pending_boards.len(),
    ))
}

pub fn gpu_incomplete_turn_pending_present_board_count(
    status_pending_count: Option<i32>,
    pending_board_count: usize,
) -> usize {
    status_pending_count
        .and_then(|count| usize::try_from(count).ok())
        .unwrap_or(0)
        .max(pending_board_count)
}

pub fn gpu_choice_agreement_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuChoiceAgreementRequest>(request_json)
        .map_err(|error| format!("GPU choice agreement request is invalid: {error}"))?;
    let agreements = gpu_choice_agreements(&request.selected, &request.choices, &request.limits);
    serde_json::to_string(&GpuChoiceAgreementResponse { agreements })
        .map_err(|error| format!("GPU choice agreement response failed to encode: {error}"))
}

pub fn gpu_choice_agreement_choices_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuChoiceAgreementChoicesRequest>(request_json)
        .map_err(|error| format!("GPU choice agreement choices request is invalid: {error}"))?;
    let selected = gpu_choice_moves(&request.selected)?;
    let choices = request
        .choices
        .iter()
        .map(gpu_choice_moves)
        .collect::<Result<Vec<_>, _>>()?;
    let agreements = gpu_choice_agreements(&selected, &choices, &request.limits);
    serde_json::to_string(&GpuChoiceAgreementResponse { agreements })
        .map_err(|error| format!("GPU choice agreement response failed to encode: {error}"))
}

pub fn gpu_move_plan_key_json(request_json: &str) -> Result<String, String> {
    let moves = serde_json::from_str::<Vec<GpuMoveJson>>(request_json)
        .map_err(|error| format!("GPU move plan key request is invalid: {error}"))?;
    Ok(gpu_move_plan_key(&moves))
}

pub fn gpu_frontier_plan_json_from_i32s(
    words: &[i32],
    offset: usize,
    plan_length: usize,
) -> Result<String, String> {
    let moves = gpu_frontier_plan_from_i32s(words, offset, plan_length)?;
    serde_json::to_string(&moves)
        .map_err(|error| format!("GPU frontier plan response failed to encode: {error}"))
}

pub fn gpu_frontier_choices_json_from_i32s(
    words: &[i32],
    max_boards: usize,
    frontier_width: usize,
    requested_depth: i32,
    gpu_search: &str,
    choice_limit: usize,
) -> Result<String, String> {
    let stride = frontier_state_stride(max_boards);
    let mut ranked = Vec::new();
    for index in 0..frontier_width {
        let base = index
            .checked_mul(stride)
            .ok_or_else(|| "GPU frontier choice state offset overflows.".to_string())?;
        if base + FRONTIER_HEADER_STRIDE > words.len() {
            break;
        }
        let depth = words[base + FRONTIER_HEADER_DEPTH];
        let score = words[base + FRONTIER_HEADER_SCORE];
        let terminal = words[base + FRONTIER_HEADER_TERMINAL] != 0;
        let plan_length = (words[base + FRONTIER_HEADER_PLAN_LENGTH].max(0) as usize)
            .min(FRONTIER_MAX_PLAN_MOVES);
        if plan_length > 0 && depth > 0 {
            ranked.push((index, depth, score, terminal, plan_length));
        }
    }
    ranked.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut choices = Vec::new();
    let mut seen = Vec::<String>::new();
    for (index, depth, score, terminal, plan_length) in ranked {
        let offset = index
            .checked_mul(stride)
            .and_then(|base| base.checked_add(FRONTIER_PLAN_OFFSET))
            .ok_or_else(|| "GPU frontier choice plan offset overflows.".to_string())?;
        let moves = gpu_frontier_plan_from_i32s(words, offset, plan_length)?;
        if moves.is_empty() {
            continue;
        }
        let key = gpu_move_plan_key(&moves);
        if seen.iter().any(|existing| existing == &key) {
            continue;
        }
        seen.push(key);
        choices.push(serde_json::json!({
            "status": "ok",
            "moves": moves,
            "score": score,
            "depth": requested_depth.max(1).min(depth),
            "principalVariation": [moves],
            "gpu": true,
            "gpuMode": "full",
            "gpuTerminal": terminal,
            "gpuSearch": gpu_search,
            "tactical": terminal,
        }));
        if choices.len() >= choice_limit {
            break;
        }
    }
    serde_json::to_string(&choices)
        .map_err(|error| format!("GPU frontier choices response failed to encode: {error}"))
}

fn gpu_frontier_plan_from_i32s(
    words: &[i32],
    offset: usize,
    plan_length: usize,
) -> Result<Vec<GpuMoveJson>, String> {
    let plan_length = plan_length.min(FRONTIER_MAX_PLAN_MOVES);
    let end = offset
        .checked_add(plan_length.saturating_mul(FRONTIER_MOVE_STRIDE))
        .ok_or_else(|| "GPU frontier plan offset overflows.".to_string())?;
    if end > words.len() {
        return Err("GPU frontier plan response is truncated.".to_string());
    }
    Ok((0..plan_length)
        .map(|move_index| {
            let base = offset + move_index * FRONTIER_MOVE_STRIDE;
            GpuMoveJson {
                from: GpuMovePositionJson {
                    timeline_id: words[base],
                    time: words[base + 1],
                    x: words[base + 2],
                    y: words[base + 3],
                },
                to: GpuMovePositionJson {
                    timeline_id: words[base + 4],
                    time: words[base + 5],
                    x: words[base + 6],
                    y: words[base + 7],
                },
            }
        })
        .collect())
}

pub fn gpu_non_postable_result_summary_json(request_json: &str) -> Result<String, String> {
    let result = serde_json::from_str::<Option<GpuNonPostableResultSummaryJson>>(request_json)
        .map_err(|error| format!("GPU non-postable result summary request is invalid: {error}"))?;
    let Some(result) = result else {
        return Ok("status=unknown, moves=0, incomplete=0, pending=unknown".to_string());
    };
    let pending = result
        .pending_present_board_count
        .map(|value| match value {
            serde_json::Value::String(text) => text,
            other => other.to_string(),
        })
        .unwrap_or_else(|| "unknown".to_string());
    Ok(format!(
        "status={}, moves={}, incomplete={}, pending={}",
        result.status.unwrap_or_else(|| "unknown".to_string()),
        result.moves.map(|moves| moves.len()).unwrap_or(0),
        result
            .incomplete_moves
            .map(|moves| moves.len())
            .unwrap_or(0),
        pending
    ))
}

pub fn gpu_postable_search_result_json(request_json: &str) -> Result<bool, String> {
    let result = serde_json::from_str::<Option<GpuPostableSearchResultJson>>(request_json)
        .map_err(|error| format!("GPU postable search result request is invalid: {error}"))?;
    Ok(result
        .map(|result| {
            result.status.as_deref() == Some("ok")
                && result.moves.is_some_and(|moves| !moves.is_empty())
        })
        .unwrap_or(false))
}

pub fn gpu_validate_first_frontier_turn_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuValidateFirstFrontierTurnRequest>(request_json)
        .map_err(|error| format!("GPU validate first frontier turn request is invalid: {error}"))?;
    let mut game = parse_game_snapshot(&request.game.to_string())?;
    let mut accepted = Vec::new();
    for movement in request.moves {
        if game.apply_move(movement.from.into(), movement.to.into()) == 0 {
            return Ok("[]".to_string());
        }
        accepted.push(movement);
        if game.submit_turn() != 0 {
            return serde_json::to_string(&accepted).map_err(|error| {
                format!("GPU validate first frontier turn response failed to encode: {error}")
            });
        }
    }
    Ok("[]".to_string())
}

pub fn gpu_validate_search_result_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuValidateSearchResultRequest>(request_json)
        .map_err(|error| format!("GPU validate search result request is invalid: {error}"))?;
    if !gpu_postable_search_result_json(&request.result.to_string())? {
        return Ok("null".to_string());
    }
    let mut result = request.result;
    let Some(moves_value) = result.get("moves").and_then(|moves| moves.as_array()) else {
        return Ok("null".to_string());
    };
    let moves = moves_value
        .iter()
        .cloned()
        .map(serde_json::from_value::<GpuMoveJson>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("GPU validate search result moves are invalid: {error}"))?;
    let mut game = parse_game_snapshot(&request.game.to_string())?;
    for (index, movement) in moves.iter().enumerate() {
        if game.apply_move(movement.from.into(), movement.to.into()) == 0 {
            return Ok("null".to_string());
        }
        if game.submit_turn() != 0 {
            if index != moves.len().saturating_sub(1) {
                return Ok("null".to_string());
            }
            let Some(object) = result.as_object_mut() else {
                return Ok("null".to_string());
            };
            object.insert("authoritativeReplay".to_string(), serde_json::json!(true));
            object.insert(
                "terminal".to_string(),
                serde_json::json!(game.result.is_some()),
            );
            if let Some(result) = game.result {
                if let Some(winner) = result.winner {
                    object.insert("winner".to_string(), serde_json::json!(winner.as_str()));
                }
                object.insert(
                    "resultReason".to_string(),
                    serde_json::json!(result.reason.as_str()),
                );
                let was_gpu_terminal = object
                    .get("gpuTerminal")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                object.insert(
                    "gpuTerminal".to_string(),
                    serde_json::json!(
                        was_gpu_terminal || result.reason.as_str() == "royal-capture"
                    ),
                );
            } else {
                let was_gpu_terminal = object
                    .get("gpuTerminal")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                object.insert(
                    "gpuTerminal".to_string(),
                    serde_json::json!(was_gpu_terminal),
                );
            }
            return serde_json::to_string(&object).map_err(|error| {
                format!("GPU validate search result response failed to encode: {error}")
            });
        }
    }
    Ok("null".to_string())
}

pub fn gpu_search_failure_summary_json(snapshot_json: &str) -> Result<String, String> {
    let snapshot = serde_json::from_str::<GpuSnapshotJson>(snapshot_json)
        .map_err(|error| format!("GPU search failure summary snapshot JSON is invalid: {error}"))?;
    let timeline_count = snapshot.timelines.len();
    let board_count = snapshot
        .timelines
        .iter()
        .map(|timeline| timeline.boards.len())
        .sum::<usize>()
        .max(1);
    let inputs = gpu_candidate_inputs_from_snapshot_json(snapshot_json)?;
    let root = encode_frontier_root_from_gpu_snapshot_json(snapshot_json, board_count)?;
    let pending = root
        .words
        .get(FRONTIER_HEADER_PENDING_BOARDS)
        .copied()
        .unwrap_or(0)
        .max(0);
    Ok(format!(
        "sources={}, targets={}, pending={}, timelines={}",
        inputs.source_count, inputs.target_count, pending, timeline_count
    ))
}

pub fn gpu_completed_turn_choice_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuCompletedTurnChoiceRequest>(request_json)
        .map_err(|error| format!("GPU completed-turn choice request is invalid: {error}"))?;
    let mut result = request
        .result
        .as_object()
        .cloned()
        .ok_or_else(|| "GPU completed-turn choice result must be an object.".to_string())?;
    let gpu_search = request.gpu_search.or_else(|| {
        result
            .get("gpuSearch")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    });
    let principal_variation = request.principal_variation.unwrap_or_else(|| {
        let mut variation = vec![serde_json::Value::Array(request.moves.clone())];
        if let Some(existing) = result
            .get("principalVariation")
            .and_then(serde_json::Value::as_array)
        {
            variation.extend(existing.iter().skip(1).cloned());
        }
        serde_json::Value::Array(variation)
    });

    let mut completed_choice = serde_json::Map::new();
    completed_choice.insert("rank".to_string(), serde_json::json!(1));
    if let Some(score) = result.get("score") {
        completed_choice.insert("score".to_string(), score.clone());
    }
    completed_choice.insert(
        "moves".to_string(),
        serde_json::Value::Array(request.moves.clone()),
    );
    completed_choice.insert(
        "principalVariation".to_string(),
        principal_variation.clone(),
    );
    if let Some(depth) = result.get("depth") {
        completed_choice.insert("depth".to_string(), depth.clone());
    }
    if let Some(nodes) = result.get("nodes") {
        completed_choice.insert("nodes".to_string(), nodes.clone());
    }
    if let Some(gpu_search) = gpu_search.as_deref() {
        completed_choice.insert(
            "gpuSearch".to_string(),
            serde_json::Value::String(gpu_search.to_string()),
        );
    }

    let mut choices = vec![serde_json::Value::Object(completed_choice)];
    if let Some(existing) = result.get("choices").and_then(serde_json::Value::as_array) {
        choices.extend(
            existing
                .iter()
                .filter(|choice| {
                    let choice_moves = choice
                        .get("moves")
                        .and_then(serde_json::Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    !gpu_same_move_sequence_values(&choice_moves, &request.moves)
                })
                .take(11)
                .cloned(),
        );
    }

    result.insert(
        "moves".to_string(),
        serde_json::Value::Array(request.moves.clone()),
    );
    if let Some(gpu_search) = gpu_search {
        result.insert(
            "gpuSearch".to_string(),
            serde_json::Value::String(gpu_search),
        );
    }
    result.insert("principalVariation".to_string(), principal_variation);
    result.insert("choices".to_string(), serde_json::Value::Array(choices));

    serde_json::to_string(&serde_json::Value::Object(result))
        .map_err(|error| format!("GPU completed-turn choice response failed to encode: {error}"))
}

pub fn gpu_validated_frontier_choice_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuValidatedFrontierChoiceRequest>(request_json)
        .map_err(|error| format!("GPU validated frontier choice request is invalid: {error}"))?;
    let moves = normalize_move_values(&request.moves);
    let response = if request.choice_count >= request.choice_limit || moves.is_empty() {
        GpuValidatedFrontierChoiceResponse {
            accepted: false,
            key: None,
            choice: None,
        }
    } else {
        let parsed_moves =
            serde_json::from_value::<Vec<GpuMoveJson>>(serde_json::Value::Array(moves.clone()))
                .map_err(|error| {
                    format!("GPU validated frontier choice moves are invalid: {error}")
                })?;
        let key = gpu_move_plan_key(&parsed_moves);
        if request.seen_keys.iter().any(|seen| seen == &key) {
            GpuValidatedFrontierChoiceResponse {
                accepted: false,
                key: Some(key),
                choice: None,
            }
        } else {
            let mut choice = request
                .candidate
                .as_object()
                .cloned()
                .unwrap_or_else(serde_json::Map::new);
            choice.insert("status".to_string(), serde_json::json!("ok"));
            choice.insert("moves".to_string(), serde_json::Value::Array(moves.clone()));
            choice.insert(
                "principalVariation".to_string(),
                serde_json::Value::Array(vec![serde_json::Value::Array(moves)]),
            );
            choice.insert("gpu".to_string(), serde_json::Value::Bool(true));
            choice.insert("gpuMode".to_string(), serde_json::json!("full"));
            choice.insert(
                "gpuSearch".to_string(),
                serde_json::Value::String(request.gpu_search),
            );
            GpuValidatedFrontierChoiceResponse {
                accepted: true,
                key: Some(key),
                choice: Some(serde_json::Value::Object(choice)),
            }
        }
    };
    serde_json::to_string(&response).map_err(|error| {
        format!("GPU validated frontier choice response failed to encode: {error}")
    })
}

pub fn gpu_frontier_choice_diagnostics_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuFrontierChoiceDiagnosticsRequest>(request_json)
        .map_err(|error| format!("GPU frontier choice diagnostics request is invalid: {error}"))?;
    let selected_tactical = request
        .selected
        .get("tactical")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let response = GpuFrontierChoiceDiagnosticsResponse {
        legal_choice_count: request.choices.len(),
        legal_tactical_choice_count: request
            .choices
            .iter()
            .filter(|choice| {
                choice
                    .get("tactical")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            })
            .count(),
        selected_move_pruned_risk: if selected_tactical { 0 } else { 1 },
        selected_move_tactical: if selected_tactical { 1 } else { 0 },
    };
    serde_json::to_string(&response).map_err(|error| {
        format!("GPU frontier choice diagnostics response failed to encode: {error}")
    })
}

pub fn gpu_normalize_principal_variation_json(request_json: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(request_json)
        .map_err(|error| format!("GPU principal variation request is invalid: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "GPU principal variation request must be an object.".to_string())?;
    let mut cleaned = Vec::new();
    if let Some(turns) = object
        .get("variation")
        .and_then(serde_json::Value::as_array)
    {
        for turn in turns {
            let Some(moves) = turn.as_array() else {
                continue;
            };
            let valid = normalize_move_values(moves);
            if !valid.is_empty() {
                cleaned.push(serde_json::Value::Array(valid));
            }
        }
    }
    if cleaned.is_empty() {
        cleaned.push(serde_json::Value::Array(normalize_move_values(
            object
                .get("fallback")
                .and_then(serde_json::Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        )));
    }
    serde_json::to_string(&serde_json::Value::Array(cleaned))
        .map_err(|error| format!("GPU principal variation response failed to encode: {error}"))
}

pub fn gpu_summarize_search_choices_json(request_json: &str) -> Result<String, String> {
    let candidates = serde_json::from_str::<Vec<serde_json::Value>>(request_json)
        .map_err(|error| format!("GPU search choice summary request is invalid: {error}"))?;
    let choices = gpu_summarize_search_choices(&candidates);
    serde_json::to_string(&choices)
        .map_err(|error| format!("GPU search choice summary response failed to encode: {error}"))
}

pub fn bot_ranked_choices_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<BotRankedChoicesRequest>(request_json)
        .map_err(|error| format!("Bot ranked choices request is invalid: {error}"))?;
    let selected_key = gpu_move_plan_key(&request.selected_moves);
    let mut by_moves: Vec<(String, serde_json::Value)> = Vec::new();
    for entry in request.results {
        let raw_choices = bot_raw_choices(&entry.result);
        for choice in raw_choices {
            let moves = gpu_choice_moves(&choice)?;
            let key = gpu_move_plan_key(&moves);
            if key.is_empty() {
                continue;
            }
            let next = bot_ranked_choice_value(&entry, &choice, &moves, key == selected_key)?;
            if let Some((_, current)) = by_moves
                .iter_mut()
                .find(|(existing_key, _)| existing_key == &key)
            {
                if bot_compare_choice_values(&next, current).is_lt() {
                    *current = next;
                } else if key == selected_key {
                    if let Some(object) = current.as_object_mut() {
                        object.insert("selected".to_string(), serde_json::Value::Bool(true));
                    }
                }
            } else {
                by_moves.push((key, next));
            }
        }
    }
    by_moves.sort_by(|left, right| bot_compare_choice_values(&left.1, &right.1));
    let choices = by_moves
        .into_iter()
        .take(16)
        .map(|(_, value)| value)
        .collect::<Vec<_>>();
    serde_json::to_string(&choices)
        .map_err(|error| format!("Bot ranked choices response failed to encode: {error}"))
}

pub fn bot_select_best_result_json(request_json: &str) -> Result<String, String> {
    let results = serde_json::from_str::<Vec<serde_json::Value>>(request_json)
        .map_err(|error| format!("Bot search result selection request is invalid: {error}"))?;
    let mut best: Option<serde_json::Value> = None;
    for result in results {
        if !bot_result_has_legal_moves(&result) {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|current| bot_compare_result_values(&result, current).is_lt())
        {
            best = Some(result);
        }
    }
    serde_json::to_string(&best)
        .map_err(|error| format!("Bot search result selection response failed to encode: {error}"))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct BotRankedChoicesRequest {
    results: Vec<BotPendingResultJson>,
    #[serde(default)]
    selected_moves: Vec<GpuMoveJson>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct BotPendingResultJson {
    result: serde_json::Value,
    partition_index: Option<i32>,
}

fn bot_raw_choices(result: &serde_json::Value) -> Vec<serde_json::Value> {
    if let Some(choices) = result
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .filter(|choices| !choices.is_empty())
    {
        return choices.clone();
    }
    let moves = gpu_choice_moves_value(result);
    if moves.as_array().is_some_and(|moves| !moves.is_empty()) {
        let mut choice = serde_json::Map::new();
        choice.insert("moves".to_string(), moves);
        for key in [
            "score",
            "depth",
            "nodes",
            "gpuSearch",
            "cpuSearch",
            "principalVariation",
        ] {
            if let Some(value) = result.get(key) {
                choice.insert(key.to_string(), value.clone());
            }
        }
        return vec![serde_json::Value::Object(choice)];
    }
    Vec::new()
}

fn bot_result_has_legal_moves(result: &serde_json::Value) -> bool {
    result.get("status").and_then(serde_json::Value::as_str) == Some("ok")
        && gpu_choice_moves_value(result)
            .as_array()
            .is_some_and(|moves| !moves.is_empty())
}

fn bot_ranked_choice_value(
    entry: &BotPendingResultJson,
    choice: &serde_json::Value,
    moves: &[GpuMoveJson],
    selected: bool,
) -> Result<serde_json::Value, String> {
    let mut output = serde_json::Map::new();
    let moves_value = serde_json::to_value(moves)
        .map_err(|error| format!("Bot ranked choice moves failed to encode: {error}"))?;
    output.insert("moves".to_string(), moves_value.clone());
    for key in ["score", "depth", "nodes", "gpuSearch", "cpuSearch"] {
        let value = choice.get(key).or_else(|| entry.result.get(key));
        if let Some(value) = value {
            output.insert(key.to_string(), value.clone());
        }
    }
    output.insert(
        "principalVariation".to_string(),
        bot_normalized_principal_variation(
            choice
                .get("principalVariation")
                .or_else(|| entry.result.get("principalVariation")),
            moves_value,
        ),
    );
    if let Some(partition_index) = entry.partition_index {
        output.insert(
            "partitionIndex".to_string(),
            serde_json::json!(partition_index),
        );
    }
    output.insert("selected".to_string(), serde_json::Value::Bool(selected));
    Ok(serde_json::Value::Object(output))
}

fn bot_normalized_principal_variation(
    variation: Option<&serde_json::Value>,
    fallback_moves: serde_json::Value,
) -> serde_json::Value {
    let turns = variation
        .and_then(serde_json::Value::as_array)
        .map(|turns| {
            turns
                .iter()
                .filter_map(|turn| {
                    let moves = turn
                        .as_array()?
                        .iter()
                        .filter(|movement| {
                            movement.get("from").is_some() && movement.get("to").is_some()
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    if moves.is_empty() {
                        None
                    } else {
                        Some(serde_json::Value::Array(moves))
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if turns.is_empty() {
        serde_json::Value::Array(vec![fallback_moves])
    } else {
        serde_json::Value::Array(turns)
    }
}

fn bot_compare_choice_values(
    left: &serde_json::Value,
    right: &serde_json::Value,
) -> std::cmp::Ordering {
    let left_depth = bot_choice_number(left, "depth", 0.0);
    let right_depth = bot_choice_number(right, "depth", 0.0);
    right_depth
        .total_cmp(&left_depth)
        .then_with(|| {
            let left_score = bot_choice_number(left, "score", f64::NEG_INFINITY);
            let right_score = bot_choice_number(right, "score", f64::NEG_INFINITY);
            right_score.total_cmp(&left_score)
        })
        .then_with(|| {
            let left_nodes = bot_choice_number(left, "nodes", 0.0);
            let right_nodes = bot_choice_number(right, "nodes", 0.0);
            right_nodes.total_cmp(&left_nodes)
        })
        .then_with(|| {
            let left_moves = gpu_choice_moves(left).unwrap_or_default();
            let right_moves = gpu_choice_moves(right).unwrap_or_default();
            gpu_move_plan_key(&left_moves).cmp(&gpu_move_plan_key(&right_moves))
        })
}

fn bot_compare_result_values(
    left: &serde_json::Value,
    right: &serde_json::Value,
) -> std::cmp::Ordering {
    let left_depth = bot_choice_number(left, "depth", 0.0);
    let right_depth = bot_choice_number(right, "depth", 0.0);
    right_depth
        .total_cmp(&left_depth)
        .then_with(|| {
            let left_score = bot_choice_number(left, "score", f64::NEG_INFINITY);
            let right_score = bot_choice_number(right, "score", f64::NEG_INFINITY);
            right_score.total_cmp(&left_score)
        })
        .then_with(|| {
            let left_nodes = bot_choice_number(left, "nodes", 0.0);
            let right_nodes = bot_choice_number(right, "nodes", 0.0);
            right_nodes.total_cmp(&left_nodes)
        })
}

fn bot_choice_number(value: &serde_json::Value, key: &str, fallback: f64) -> f64 {
    value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(fallback)
}

fn gpu_summarize_search_choices(candidates: &[serde_json::Value]) -> Vec<serde_json::Value> {
    candidates
        .iter()
        .take(12)
        .enumerate()
        .map(|(index, candidate)| {
            let mut choice = serde_json::Map::new();
            choice.insert("rank".to_string(), serde_json::json!(index + 1));
            for key in [
                "score",
                "principalVariation",
                "depth",
                "nodes",
                "gpuSearch",
                "gpuTerminal",
                "tactical",
            ] {
                if let Some(value) = candidate.get(key) {
                    choice.insert(key.to_string(), value.clone());
                }
            }
            choice.insert("moves".to_string(), gpu_choice_moves_value(candidate));
            serde_json::Value::Object(choice)
        })
        .collect()
}

fn gpu_move_plan_key(moves: &[GpuMoveJson]) -> String {
    moves
        .iter()
        .map(|movement| {
            format!(
                "{}:{}:{}:{}:{}:{}:{}:{}",
                movement.from.timeline_id,
                movement.from.time,
                movement.from.x,
                movement.from.y,
                movement.to.timeline_id,
                movement.to.time,
                movement.to.x,
                movement.to.y
            )
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn gpu_choice_moves_value(candidate: &serde_json::Value) -> serde_json::Value {
    if let Some(moves) = candidate.get("moves").filter(|value| value.is_array()) {
        return moves.clone();
    }
    if let Some(move_value) = candidate.get("move").filter(|value| !value.is_null()) {
        return serde_json::Value::Array(vec![move_value.clone()]);
    }
    serde_json::Value::Array(Vec::new())
}

fn gpu_choice_moves(candidate: &serde_json::Value) -> Result<Vec<GpuMoveJson>, String> {
    serde_json::from_value::<Vec<GpuMoveJson>>(gpu_choice_moves_value(candidate))
        .map_err(|error| format!("GPU search choice moves are invalid: {error}"))
}

fn normalize_move_values(moves: &[serde_json::Value]) -> Vec<serde_json::Value> {
    moves
        .iter()
        .filter(|movement| serde_json::from_value::<GpuMoveJson>((*movement).clone()).is_ok())
        .cloned()
        .collect()
}

fn gpu_choice_agreements(
    selected: &[GpuMoveJson],
    choices: &[Vec<GpuMoveJson>],
    limits: &[usize],
) -> Vec<i32> {
    let selected_key = gpu_move_plan_key(selected);
    limits
        .iter()
        .map(|limit| {
            if selected_key.is_empty() {
                return 0;
            }
            i32::from(
                choices
                    .iter()
                    .take(*limit)
                    .any(|choice| gpu_move_plan_key(choice) == selected_key),
            )
        })
        .collect()
}

fn gpu_same_move_sequence_values(left: &[serde_json::Value], right: &[serde_json::Value]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().zip(right.iter()).all(|(left, right)| {
        let Ok(left) = serde_json::from_value::<GpuMoveJson>(left.clone()) else {
            return false;
        };
        let Ok(right) = serde_json::from_value::<GpuMoveJson>(right.clone()) else {
            return false;
        };
        gpu_same_move(&left, &right)
    })
}

fn gpu_same_move(left: &GpuMoveJson, right: &GpuMoveJson) -> bool {
    left.from.timeline_id == right.from.timeline_id
        && left.from.time == right.from.time
        && left.from.x == right.from.x
        && left.from.y == right.from.y
        && left.to.timeline_id == right.to.timeline_id
        && left.to.time == right.to.time
        && left.to.x == right.to.x
        && left.to.y == right.to.y
}

pub fn gpu_pick_candidate_records_from_i32s(words: &[i32]) -> Result<Vec<i32>, String> {
    if words.len() < 2 {
        return Err("GPU candidate record pick request is truncated.".to_string());
    }
    let record_count = non_negative_usize(words[0], "record count")?;
    let index_count = non_negative_usize(words[1], "index count")?;
    let record_offset = 2;
    let record_len = record_count
        .checked_mul(GPU_CANDIDATE_STRIDE)
        .ok_or_else(|| "GPU candidate record pick record length overflowed.".to_string())?;
    let index_offset = record_offset + record_len;
    let expected_len = index_offset + index_count;
    if words.len() != expected_len {
        return Err(format!(
            "GPU candidate record pick request length mismatch: expected {expected_len}, got {}.",
            words.len()
        ));
    }
    let records = &words[record_offset..index_offset];
    let mut picked = Vec::with_capacity(index_count * GPU_CANDIDATE_STRIDE);
    for raw_index in &words[index_offset..] {
        let index = non_negative_usize(*raw_index, "record index")?;
        let offset = index.saturating_mul(GPU_CANDIDATE_STRIDE);
        if offset + GPU_CANDIDATE_STRIDE > records.len() {
            return Err(format!(
                "GPU candidate record index {index} is out of range."
            ));
        }
        picked.extend_from_slice(&records[offset..offset + GPU_CANDIDATE_STRIDE]);
    }
    Ok(picked)
}

pub fn gpu_pick_candidate_records_json(request_json: &str) -> Result<Vec<i32>, String> {
    let request = serde_json::from_str::<GpuPickCandidateRecordsRequest>(request_json)
        .map_err(|error| format!("GPU candidate record pick JSON request is invalid: {error}"))?;
    let record_count = request.records.len() / GPU_CANDIDATE_STRIDE;
    let record_len = record_count * GPU_CANDIDATE_STRIDE;
    let mut words = Vec::with_capacity(2 + record_len + request.indices.len());
    words.push(
        i32::try_from(record_count)
            .map_err(|_| "GPU candidate record pick count exceeds i32 range.".to_string())?,
    );
    words.push(
        i32::try_from(request.indices.len())
            .map_err(|_| "GPU candidate record pick index count exceeds i32 range.".to_string())?,
    );
    words.extend(request.records.into_iter().take(record_len));
    let indices = request
        .indices
        .into_iter()
        .map(gpu_candidate_record_pick_index)
        .collect::<Result<Vec<_>, _>>()?;
    words.extend(indices);
    gpu_pick_candidate_records_from_i32s(&words)
}

pub fn gpu_mutation_turn_code_from_records(records: &[i32]) -> i32 {
    records
        .get(GPU_CANDIDATE_COLOR_OFFSET)
        .copied()
        .unwrap_or(0)
}

pub fn gpu_mutation_turn_code_json(request_json: &str) -> Result<i32, String> {
    let request = serde_json::from_str::<GpuMutationTurnCodeRequest>(request_json)
        .map_err(|error| format!("GPU mutation turn-code request is invalid: {error}"))?;
    Ok(gpu_mutation_turn_code_from_records(&request.records))
}

fn gpu_candidate_record_pick_index(index: f64) -> Result<i32, String> {
    if !index.is_finite() {
        return Err("GPU candidate record pick index is not finite.".to_string());
    }
    let truncated = index.trunc();
    if truncated < i32::MIN as f64 || truncated > i32::MAX as f64 {
        return Err("GPU candidate record pick index exceeds i32 range.".to_string());
    }
    Ok(truncated as i32)
}

fn gpu_candidate_json_index(candidate: &serde_json::Value) -> Result<i32, String> {
    let index = candidate
        .get("index")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| "GPU candidate index is missing.".to_string())?;
    gpu_candidate_record_pick_index(index)
}

pub fn gpu_candidate_index_from_i32s(words: &[i32]) -> Result<i32, String> {
    if words.len() < 10 {
        return Err("GPU candidate index request is truncated.".to_string());
    }
    let record_count = non_negative_usize(words[0], "record count")?;
    let target = &words[1..9];
    let record_offset = 9;
    let record_len = record_count
        .checked_mul(GPU_CANDIDATE_STRIDE)
        .ok_or_else(|| "GPU candidate index record length overflowed.".to_string())?;
    let expected_len = record_offset + record_len;
    if words.len() != expected_len {
        return Err(format!(
            "GPU candidate index request length mismatch: expected {expected_len}, got {}.",
            words.len()
        ));
    }
    let records = &words[record_offset..];
    for index in 0..record_count {
        let offset = index * GPU_CANDIDATE_STRIDE;
        if records.get(offset + 11..offset + 19) == Some(target) {
            return Ok(index as i32);
        }
    }
    Ok(-1)
}

pub fn gpu_candidate_index_json(request_json: &str) -> Result<i32, String> {
    let request = serde_json::from_str::<GpuCandidateIndexRequest>(request_json)
        .map_err(|error| format!("GPU candidate index JSON request is invalid: {error}"))?;
    let record_count = request.records.len() / GPU_CANDIDATE_STRIDE;
    if record_count * GPU_CANDIDATE_STRIDE != request.records.len() {
        return Err(format!(
            "GPU candidate index JSON records are not stride-aligned: got {} i32s.",
            request.records.len()
        ));
    }
    let mut words = Vec::with_capacity(9 + request.records.len());
    words.push(
        i32::try_from(record_count)
            .map_err(|_| "GPU candidate index record count exceeds i32 range.".to_string())?,
    );
    words.extend([
        request.movement.from.timeline_id,
        request.movement.from.time,
        request.movement.from.x,
        request.movement.from.y,
        request.movement.to.timeline_id,
        request.movement.to.time,
        request.movement.to.x,
        request.movement.to.y,
    ]);
    words.extend(request.records);
    gpu_candidate_index_from_i32s(&words)
}

pub fn gpu_reply_pressure_ranked_roots_from_i32s(words: &[i32]) -> Result<Vec<i32>, String> {
    if words.len() < 2 {
        return Err("GPU reply pressure request is truncated.".to_string());
    }
    let root_count = non_negative_usize(words[0], "root count")?;
    let reply_count = non_negative_usize(words[1], "reply count")?;
    let root_index_offset = 2;
    let root_score_offset = root_index_offset + root_count;
    let pair_score_offset = root_score_offset + root_count;
    let pair_count = root_count
        .checked_mul(reply_count)
        .ok_or_else(|| "GPU reply pressure pair count overflowed.".to_string())?;
    let expected_len = pair_score_offset + pair_count;
    if words.len() != expected_len {
        return Err(format!(
            "GPU reply pressure request length mismatch: expected {expected_len}, got {}.",
            words.len()
        ));
    }
    let root_indexes = &words[root_index_offset..root_score_offset];
    let root_scores = &words[root_score_offset..pair_score_offset];
    let pair_scores = &words[pair_score_offset..];
    let mut ranked = root_indexes
        .iter()
        .enumerate()
        .map(|(root_index, candidate_index)| {
            let offset = root_index * reply_count;
            let max_pressure = pair_scores
                .get(offset..offset + reply_count)
                .unwrap_or(&[])
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .max(0);
            let score = root_scores
                .get(root_index)
                .copied()
                .unwrap_or(-2_147_483_647)
                .saturating_sub(max_pressure);
            (*candidate_index, score)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    Ok(ranked
        .into_iter()
        .flat_map(|(index, score)| [index, score])
        .collect())
}

pub fn gpu_reply_pressure_ranked_roots_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuReplyPressureRankedRootsRequest>(request_json)
        .map_err(|error| format!("GPU reply pressure JSON request is invalid: {error}"))?;
    let pair_count = request
        .ranked_roots
        .len()
        .checked_mul(request.reply_count)
        .ok_or_else(|| "GPU reply pressure pair count overflowed.".to_string())?;
    if request.pair_scores.len() != pair_count {
        return Err(format!(
            "GPU reply pressure JSON request length mismatch: expected {pair_count} pair scores, got {}.",
            request.pair_scores.len()
        ));
    }
    let mut ranked = request
        .ranked_roots
        .into_iter()
        .enumerate()
        .filter_map(|(root_index, mut root)| {
            let index = root.get("index").and_then(serde_json::Value::as_i64)?;
            let score = root
                .get("score")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(-2_147_483_647);
            let offset = root_index * request.reply_count;
            let max_pressure = request
                .pair_scores
                .get(offset..offset + request.reply_count)
                .unwrap_or(&[])
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .max(0) as i64;
            let adjusted = score.saturating_sub(max_pressure);
            if let Some(object) = root.as_object_mut() {
                object.insert("score".to_string(), serde_json::json!(adjusted));
            }
            Some((index, adjusted, root))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    serde_json::to_string(
        &ranked
            .into_iter()
            .map(|(_, _, root)| root)
            .collect::<Vec<_>>(),
    )
    .map_err(|error| format!("GPU reply pressure JSON response failed to encode: {error}"))
}

fn non_negative_usize(value: i32, label: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("GPU candidate ranking {label} is negative."))
}

fn gpu_search_select_candidate(
    candidates: Vec<GpuSearchSelectionCandidate>,
    temperature: f64,
    random_seed: i64,
) -> GpuSearchSelectionResponse {
    let mut supported = candidates
        .into_iter()
        .filter(|candidate| candidate.move_count > 0 && candidate.score.is_finite())
        .collect::<Vec<_>>();
    supported.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.key.cmp(&right.key))
    });
    let ranked_indexes = supported
        .iter()
        .map(|candidate| candidate.index)
        .collect::<Vec<_>>();
    let selected_index = if supported.is_empty() {
        None
    } else if !temperature.is_finite() || temperature <= 0.0 {
        supported.first().map(|candidate| candidate.index)
    } else {
        let candidate_limit = supported.len().min(32);
        let top = &supported[..candidate_limit];
        let max_score = top.first().map(|candidate| candidate.score).unwrap_or(0.0);
        let score_scale = (temperature * 100.0).max(1.0);
        let weights = top
            .iter()
            .map(|candidate| {
                ((candidate.score - max_score) / score_scale)
                    .clamp(-50.0, 0.0)
                    .exp()
            })
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        let mut pick = gpu_search_seeded_unit(random_seed) * total;
        let mut selected = top.last().map(|candidate| candidate.index);
        for (index, weight) in weights.iter().enumerate() {
            pick -= weight;
            if pick <= 0.0 {
                selected = top.get(index).map(|candidate| candidate.index);
                break;
            }
        }
        selected
    };
    GpuSearchSelectionResponse {
        selected_index,
        ranked_indexes,
    }
}

fn gpu_search_seeded_unit(seed: i64) -> f64 {
    let mut state = seed as u32;
    state ^= state.wrapping_shl(13);
    state ^= state.wrapping_shr(17);
    state ^= state.wrapping_shl(5);
    f64::from(if state == 0 { 1 } else { state }) / 0xffff_ffffu32 as f64
}

fn encode_frontier_root_from_gpu_snapshot_json(
    snapshot_json: &str,
    max_boards: usize,
) -> Result<EncodedFrontierRoot, String> {
    let snapshot = serde_json::from_str::<GpuSnapshotJson>(snapshot_json)
        .map_err(|error| format!("GPU frontier snapshot JSON is invalid: {error}"))?;
    let turn = gpu_search_color_code(&snapshot.turn)?;
    let timelines = snapshot
        .timelines
        .iter()
        .map(gpu_timeline_from_json)
        .collect::<Result<Vec<_>, _>>()?;
    encode_frontier_root_from_timelines(
        turn,
        snapshot.next_timeline_id.unwrap_or(1),
        snapshot.next_black_timeline_id.unwrap_or(-1),
        snapshot.royal_capture_by.is_some(),
        &timelines,
        max_boards,
    )
}

fn encode_frontier_root_from_timelines(
    root_color: i32,
    next_timeline_id: i32,
    next_black_timeline_id: i32,
    terminal: bool,
    timelines: &[GpuCandidateInputTimeline],
    max_boards: usize,
) -> Result<EncodedFrontierRoot, String> {
    let boards = sorted_gpu_timeline_boards(timelines);
    if boards.len() > max_boards {
        return Err(format!(
            "GPU frontier snapshot has {} boards but the adapter limit is {max_boards}.",
            boards.len()
        ));
    }

    let mut words = vec![0; frontier_state_stride(max_boards)];
    words[FRONTIER_HEADER_PARENT] = -1;
    words[FRONTIER_HEADER_ROOT] = 0;
    words[FRONTIER_HEADER_SCORE] = 0;
    words[FRONTIER_HEADER_DEPTH] = 0;
    words[FRONTIER_HEADER_TURN] = root_color;
    words[FRONTIER_HEADER_BOARD_COUNT] = boards.len() as i32;
    words[FRONTIER_HEADER_PLAN_LENGTH] = 0;
    words[FRONTIER_HEADER_COMPLETE] = 0;
    words[FRONTIER_HEADER_TERMINAL] = i32::from(terminal);
    words[FRONTIER_HEADER_NEXT_WHITE_TIMELINE] = next_timeline_id;
    words[FRONTIER_HEADER_NEXT_BLACK_TIMELINE] = next_black_timeline_id;

    let ids = timelines
        .iter()
        .map(|timeline| timeline.id)
        .collect::<Vec<_>>();
    let active_distance = gpu_frontier_active_timeline_distance(&ids);
    let active_board_summaries = timelines
        .iter()
        .filter(|timeline| {
            gpu_frontier_timeline_active(
                gpu_search_owner_from_code(timeline.owner),
                timeline.id,
                active_distance,
            )
            .unwrap_or(false)
        })
        .filter_map(|timeline| {
            latest_gpu_board(timeline).map(|board| GpuFrontierActiveBoard {
                time: board.time,
                side_to_move: board.side_to_move,
            })
        })
        .collect::<Vec<_>>();
    let present = gpu_frontier_present_time(&active_board_summaries);
    let pending = gpu_frontier_pending_board_count(&active_board_summaries, present, root_color);
    words[FRONTIER_HEADER_PRESENT_TIME] = present;
    words[FRONTIER_HEADER_PENDING_BOARDS] = pending as i32;
    words[FRONTIER_HEADER_COMPLETE] = i32::from(pending == 0);

    for (index, (timeline, board)) in boards.iter().enumerate() {
        let base = FRONTIER_BOARD_OFFSET + index * FRONTIER_BOARD_STRIDE;
        let latest = latest_gpu_board(timeline).is_some_and(|latest| latest.time == board.time);
        let active = gpu_frontier_timeline_active(
            gpu_search_owner_from_code(timeline.owner),
            timeline.id,
            active_distance,
        )
        .unwrap_or(false);
        let pending_board =
            latest && active && board.time == present && board.side_to_move == root_color;
        words[base + FRONTIER_BOARD_TIMELINE_ID] = timeline.id;
        words[base + FRONTIER_BOARD_ROW] = timeline.row;
        words[base + FRONTIER_BOARD_OWNER] = timeline.owner;
        words[base + FRONTIER_BOARD_TIME] = board.time;
        words[base + FRONTIER_BOARD_SIDE_TO_MOVE] = board.side_to_move;
        words[base + FRONTIER_BOARD_CASTLING] = board.castling;
        write_gpu_en_passant(
            &mut words,
            base + FRONTIER_BOARD_EN_PASSANT,
            board.en_passant,
        );
        words[base + FRONTIER_BOARD_LATEST] = i32::from(latest);
        words[base + FRONTIER_BOARD_ORIGIN] = board.origin_kind;
        words[base + FRONTIER_BOARD_ACTIVE] = i32::from(active);
        words[base + FRONTIER_BOARD_PENDING] = i32::from(pending_board);
        for square in 0..64 {
            words[base + FRONTIER_BOARD_SQUARES + square] =
                *board.squares.get(square).unwrap_or(&0);
        }
    }

    let board_start = FRONTIER_BOARD_OFFSET;
    let board_end = board_start + boards.len() * FRONTIER_BOARD_STRIDE;
    let (hash_low, hash_high) = hash_frontier_words(&words[board_start..board_end]);
    words[FRONTIER_HEADER_HASH_LOW] = hash_low;
    words[FRONTIER_HEADER_HASH_HIGH] = hash_high;

    Ok(EncodedFrontierRoot {
        words,
        board_count: boards.len(),
        hash_low,
        hash_high,
    })
}

fn gpu_timeline_from_json(timeline: &GpuTimelineJson) -> Result<GpuCandidateInputTimeline, String> {
    Ok(GpuCandidateInputTimeline {
        id: timeline.id,
        row: timeline.row,
        owner: gpu_search_owner_code(&timeline.owner)?,
        boards: timeline
            .boards
            .iter()
            .map(gpu_board_from_json)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn gpu_board_from_json(board: &GpuBoardJson) -> Result<GpuCandidateInputBoard, String> {
    Ok(GpuCandidateInputBoard {
        time: board.time,
        side_to_move: gpu_search_color_code(&board.side_to_move)?,
        castling: board.castling.unwrap_or(0),
        en_passant: board.en_passant.map(|en_passant| GpuEnPassantRecord {
            x: en_passant.x,
            y: en_passant.y,
            captured_x: en_passant.captured_x,
            captured_y: en_passant.captured_y,
        }),
        origin_kind: board.origin_kind.unwrap_or_else(|| {
            gpu_frontier_origin_code(
                board
                    .origin
                    .as_ref()
                    .and_then(|origin| origin.get("type"))
                    .and_then(serde_json::Value::as_str),
            )
        }),
        squares: gpu_board_squares_from_json(board)?,
    })
}

fn gpu_game_board_json_from_json(board: &GpuBoardJson) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "time": board.time,
        "sideToMove": board.side_to_move,
        "castling": board.castling.unwrap_or(0),
        "enPassant": board.en_passant.map(|en_passant| serde_json::json!({
            "x": en_passant.x,
            "y": en_passant.y,
            "capturedX": en_passant.captured_x,
            "capturedY": en_passant.captured_y,
        })),
        "origin": board.origin.clone(),
        "board": gpu_game_board_squares_json(&gpu_board_squares_from_json(board)?),
    }))
}

fn gpu_snapshot_board_json_from_json(
    board: &GpuBoardJson,
    latest_override: Option<bool>,
) -> Result<serde_json::Value, String> {
    let mut value = serde_json::json!({
        "time": board.time,
        "sideToMove": board.side_to_move,
        "castling": board.castling.unwrap_or(0),
        "enPassant": board.en_passant.map(|en_passant| serde_json::json!({
            "x": en_passant.x,
            "y": en_passant.y,
            "capturedX": en_passant.captured_x,
            "capturedY": en_passant.captured_y,
        })),
        "origin": board.origin.clone(),
        "originKind": board.origin_kind.unwrap_or(0),
        "latest": latest_override.unwrap_or_else(|| board.latest.unwrap_or(false)),
        "squares": gpu_square_codes_json(&gpu_board_squares_from_json(board)?),
    });
    if let Some(timeline_index) = board.timeline_index {
        value["timelineIndex"] = serde_json::json!(timeline_index);
    }
    Ok(value)
}

fn gpu_mutation_child_board_json(
    child: &GpuMutationBoardSnapshot,
    timeline_index: i32,
    origin: Option<serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "timelineIndex": timeline_index,
        "timelineId": child.timeline_id,
        "time": child.time,
        "sideToMove": child.side_to_move,
        "castling": child.castling,
        "enPassant": child.en_passant.map(|en_passant| serde_json::json!({
            "x": en_passant.x,
            "y": en_passant.y,
            "capturedX": en_passant.captured_x,
            "capturedY": en_passant.captured_y,
        })),
        "origin": origin,
        "latest": true,
        "originKind": child.origin_kind,
        "squares": gpu_square_codes_json(&child.squares),
    })
}

fn gpu_child_origin_json(
    child: &GpuMutationBoardSnapshot,
    movement: &GpuMoveJson,
) -> serde_json::Value {
    let source_advance = gpu_child_is_source_advance(
        GpuChildBoardRef {
            timeline_id: child.timeline_id,
            time: child.time,
        },
        gpu_move_position_candidate(&movement.from),
    );
    serde_json::json!({
        "type": if source_advance { "source-advance" } else { "cross-board" },
        "from": move_position_json(&movement.from),
        "to": move_position_json(&movement.to),
    })
}

fn gpu_branch_origin_json(movement: &GpuMoveJson) -> serde_json::Value {
    serde_json::json!({
        "type": "branch",
        "from": move_position_json(&movement.from),
        "to": move_position_json(&movement.to),
    })
}

fn gpu_move_position_candidate(position: &GpuMovePositionJson) -> GpuCandidatePosition {
    GpuCandidatePosition {
        timeline_id: position.timeline_id,
        time: position.time,
        x: position.x,
        y: position.y,
    }
}

fn move_position_json(position: &GpuMovePositionJson) -> serde_json::Value {
    serde_json::json!({
        "timelineId": position.timeline_id,
        "time": position.time,
        "x": position.x,
        "y": position.y,
    })
}

fn gpu_snapshot_latest_board_is(snapshot: &GpuSnapshotJson, timeline_id: i32, time: i32) -> bool {
    snapshot
        .timelines
        .iter()
        .find(|timeline| timeline.id == timeline_id)
        .and_then(|timeline| timeline.boards.iter().max_by_key(|board| board.time))
        .is_some_and(|board| board.time == time)
}

fn gpu_game_board_squares_json(squares: &[i32]) -> serde_json::Value {
    let rows = (0..8)
        .map(|y| {
            serde_json::Value::Array(
                (0..8)
                    .map(|x| {
                        gpu_search_piece_from_code(*squares.get(y * 8 + x).unwrap_or(&0))
                            .map(|piece| {
                                serde_json::json!({
                                    "type": piece.piece_type,
                                    "color": piece.color,
                                })
                            })
                            .unwrap_or(serde_json::Value::Null)
                    })
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    serde_json::Value::Array(rows)
}

fn gpu_square_codes_json(squares: &[i32]) -> serde_json::Value {
    serde_json::Value::Array(
        (0..64)
            .map(|index| serde_json::json!(*squares.get(index).unwrap_or(&0)))
            .collect(),
    )
}

fn gpu_board_squares_from_json(board: &GpuBoardJson) -> Result<Vec<i32>, String> {
    if let Some(squares) = &board.squares {
        let mut output = vec![0; 64];
        match squares {
            serde_json::Value::Array(values) => {
                for (index, value) in values.iter().take(64).enumerate() {
                    output[index] = value.as_i64().unwrap_or(0) as i32;
                }
            }
            serde_json::Value::Object(values) => {
                for index in 0..64 {
                    output[index] = values
                        .get(&index.to_string())
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(0) as i32;
                }
            }
            _ => {}
        }
        return Ok(output);
    }
    let Some(rows) = &board.board else {
        return Ok(vec![0; 64]);
    };
    let mut output = vec![0; 64];
    for (y, row) in rows.iter().take(8).enumerate() {
        for (x, piece) in row.iter().take(8).enumerate() {
            output[y * 8 + x] = match piece {
                Some(piece) => gpu_search_piece_code(&piece.piece_type, &piece.color)?,
                None => 0,
            };
        }
    }
    Ok(output)
}

fn sorted_gpu_timeline_boards(
    timelines: &[GpuCandidateInputTimeline],
) -> Vec<(&GpuCandidateInputTimeline, &GpuCandidateInputBoard)> {
    let sort_keys = timelines
        .iter()
        .map(|timeline| GpuTimelineSortKey {
            row: timeline.row,
            id: timeline.id,
        })
        .collect::<Vec<_>>();
    gpu_timeline_sort_order(&sort_keys)
        .into_iter()
        .filter_map(|index| timelines.get(index))
        .flat_map(|timeline| {
            let mut order = (0..timeline.boards.len()).collect::<Vec<_>>();
            order.sort_by_key(|index| timeline.boards[*index].time);
            order
                .into_iter()
                .filter_map(|index| timeline.boards.get(index))
                .map(move |board| (timeline, board))
        })
        .collect()
}

fn latest_gpu_board(timeline: &GpuCandidateInputTimeline) -> Option<&GpuCandidateInputBoard> {
    gpu_latest_board_index(
        &timeline
            .boards
            .iter()
            .map(|board| board.time)
            .collect::<Vec<_>>(),
    )
    .and_then(|index| timeline.boards.get(index))
}

fn write_gpu_en_passant(words: &mut [i32], offset: usize, en_passant: Option<GpuEnPassantRecord>) {
    if let Some(en_passant) = en_passant {
        words[offset] = en_passant.x;
        words[offset + 1] = en_passant.y;
        words[offset + 2] = en_passant.captured_x;
        words[offset + 3] = en_passant.captured_y;
    } else {
        words[offset] = -1;
        words[offset + 1] = -1;
        words[offset + 2] = -1;
        words[offset + 3] = -1;
    }
}

pub const KERNELS: &[GpuKernel] = &[
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "turn_status",
        shader: "turn_status.wgsl",
        entry_point: "turn_status",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "score_candidates",
        shader: "movegen.wgsl",
        entry_point: "score_candidates",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "mutate_candidates",
        shader: "mutate.wgsl",
        entry_point: "mutate_candidates",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "score_replies",
        shader: "reply.wgsl",
        entry_point: "score_replies",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_expand",
        shader: "frontier_expand.wgsl",
        entry_point: "expand_frontier",
        constants: &[("EXPAND_WORKGROUP_SIZE", 128)],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_hash",
        shader: "frontier_select.wgsl",
        entry_point: "hash_candidates",
        constants: &[("SELECT_WORKGROUP_SIZE", 128)],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_order",
        shader: "frontier_select.wgsl",
        entry_point: "bucket_order",
        constants: &[("SELECT_WORKGROUP_SIZE", 128)],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_sort",
        shader: "frontier_select.wgsl",
        entry_point: "bitonic_sort",
        constants: &[("SELECT_WORKGROUP_SIZE", 128)],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_unique",
        shader: "frontier_select.wgsl",
        entry_point: "mark_unique",
        constants: &[("SELECT_WORKGROUP_SIZE", 128)],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_parent_quota",
        shader: "frontier_select.wgsl",
        entry_point: "mark_parent_quota",
        constants: &[("SELECT_WORKGROUP_SIZE", 128)],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_compact",
        shader: "frontier_select.wgsl",
        entry_point: "compact_selected",
        constants: &[("SELECT_WORKGROUP_SIZE", 128)],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_select",
        shader: "frontier_select.wgsl",
        entry_point: "fill_selection_underflow",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_materialize",
        shader: "frontier_state.wgsl",
        entry_point: "materialize_selected",
        constants: &[("MATERIALIZE_WORKGROUP_SIZE", 128)],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_reduce",
        shader: "frontier_state.wgsl",
        entry_point: "minimax_reduce_stage",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_reduce_copy",
        shader: "frontier_state.wgsl",
        entry_point: "minimax_copy_scores",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_forward_layer",
        shader: "frontier_forward.wgsl",
        entry_point: "forward_layer_masked",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_forward_output",
        shader: "frontier_forward.wgsl",
        entry_point: "forward_output_masked",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_forward_output_linear",
        shader: "frontier_forward.wgsl",
        entry_point: "forward_output_masked_linear",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_apply_neural",
        shader: "frontier_neural.wgsl",
        entry_point: "apply_neural_values",
        constants: &[],
    },
    GpuKernel {
        set: GpuKernelSet::GpuSearch,
        label: "frontier_apply_policy",
        shader: "frontier_policy.wgsl",
        entry_point: "apply_policy_prior",
        constants: &[],
    },
];

#[derive(Clone, Debug)]
pub struct GpuSearchRequest {
    pub snapshot_json: Option<String>,
    pub model_path: Option<String>,
    pub depth: i32,
    pub min_depth: Option<i32>,
    pub nodes: i32,
    pub time_ms: i32,
}

impl Default for GpuSearchRequest {
    fn default() -> Self {
        Self {
            snapshot_json: None,
            model_path: Some(training::DEFAULT_VALUE_MODEL_PATH.to_string()),
            depth: DEFAULT_GPU_SEARCH_DEPTH,
            min_depth: Some(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
            nodes: DEFAULT_GPU_SEARCH_NODES,
            time_ms: DEFAULT_GPU_SEARCH_TIME_MS,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuSearchResponse {
    pub result_json: String,
    pub gpu_search: &'static str,
    pub backend: &'static str,
    pub native_frontier_round: Option<String>,
}

struct NativeFrontierSearchResult {
    result_json: Option<String>,
    metadata: Option<String>,
    gpu_search: Option<&'static str>,
    backend: Option<&'static str>,
}

pub fn search(request: GpuSearchRequest) -> Result<GpuSearchResponse, String> {
    let model_path = request
        .model_path
        .unwrap_or_else(|| training::DEFAULT_VALUE_MODEL_PATH.to_string());
    let model = training::load_compact_value_model(&model_path)?;
    let evaluator = ValueEvaluator::compact_value_model_from_path(model_path, model);
    let game = match request.snapshot_json.as_deref() {
        Some(snapshot) => parse_game_snapshot(snapshot)?,
        None => Game::new(),
    };
    let max_boards = game
        .timelines
        .iter()
        .map(|timeline| timeline.boards.len())
        .sum::<usize>()
        .max(1);
    let _frontier_root = encode_frontier_root(&game, max_boards)?;
    let depth = request.depth.max(1);
    let min_depth = request
        .min_depth
        .unwrap_or(Game::DEFAULT_MIN_AI_SEARCH_DEPTH)
        .max(1);
    let native_frontier = native_frontier_search_result(&game, depth, min_depth)?;
    if let Some(result_json) = native_frontier.result_json {
        return Ok(GpuSearchResponse {
            result_json,
            gpu_search: native_frontier.gpu_search.unwrap_or("native-wgpu-frontier"),
            backend: native_frontier.backend.unwrap_or("wgpu-frontier"),
            native_frontier_round: native_frontier.metadata,
        });
    }
    let nodes = request.nodes.max(1);
    let deadline = search_deadline(request.time_ms.max(1));
    let (result, _) = game.best_ai_turn_with_value_evaluator_min_depth(
        depth,
        min_depth,
        nodes,
        deadline,
        SearchOptions::optimized(),
        evaluator,
        Some("gpu-compact-value"),
    );
    Ok(GpuSearchResponse {
        result_json: result.to_json(),
        gpu_search: "cpu-orchestrated-compact-value-model",
        backend: "cpu-search-with-gpu-model",
        native_frontier_round: native_frontier.metadata,
    })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
fn native_frontier_search_result(
    game: &Game,
    depth: i32,
    min_depth: i32,
) -> Result<NativeFrontierSearchResult, String> {
    let search_depth = depth.min(2);
    let search = crate::gpu::native::run_frontier_search(game, search_depth)?;
    if search.rounds.is_empty() {
        return Ok(NativeFrontierSearchResult {
            result_json: None,
            metadata: None,
            gpu_search: None,
            backend: None,
        });
    };
    let candidate_count = search
        .rounds
        .iter()
        .map(|round| round.candidate_count.max(0) as usize)
        .sum::<usize>();
    let selected_count = search
        .rounds
        .iter()
        .map(|round| round.selected_count.max(0) as usize)
        .sum::<usize>();
    let state_count = search
        .rounds
        .iter()
        .map(|round| round.materialized_state_count())
        .sum::<usize>();
    let plan_count = search
        .rounds
        .iter()
        .map(|round| round.planned_root_moves().len())
        .sum::<usize>();
    let mut metadata = format!(
        "wgpu-frontier-search rounds={} candidates={} selected={} states={} plans={}",
        search.rounds.len(),
        candidate_count,
        selected_count,
        state_count,
        plan_count
    );
    if search.rounds.len() >= 2 {
        metadata.push_str(&format!(" minimax_roots={}", search.minimax_root_count()));
    }
    let metadata = Some(metadata);
    if depth > 2 || min_depth > depth || search.rounds.len() < depth as usize {
        return Ok(NativeFrontierSearchResult {
            result_json: None,
            metadata,
            gpu_search: None,
            backend: None,
        });
    }
    let Some(best) = search.best_minimax_root_move() else {
        return Ok(NativeFrontierSearchResult {
            result_json: None,
            metadata,
            gpu_search: None,
            backend: None,
        });
    };
    let step = crate::MoveStep {
        from: crate::Position {
            timeline_id: best.move_record[0],
            time: best.move_record[1],
            x: best.move_record[2],
            y: best.move_record[3],
        },
        to: crate::Position {
            timeline_id: best.move_record[4],
            time: best.move_record[5],
            x: best.move_record[6],
            y: best.move_record[7],
        },
    };
    if !game.can_move_to(step.from, step.to) {
        return Err(format!(
            "native GPU frontier selected illegal move from=({}, {}, {}, {}) to=({}, {}, {}, {})",
            step.from.timeline_id,
            step.from.time,
            step.from.x,
            step.from.y,
            step.to.timeline_id,
            step.to.time,
            step.to.x,
            step.to.y
        ));
    }
    let result = crate::AiSearchResult {
        moves: vec![step],
        score: best.score,
        depth: search.rounds.len().min(depth as usize) as i32,
        nodes: candidate_count,
        status: "ok",
        principal_variation: vec![vec![step]],
        terminal_royal_capture: false,
    };
    Ok(NativeFrontierSearchResult {
        result_json: Some(result.to_json()),
        metadata,
        gpu_search: Some(if search.rounds.len() >= 2 {
            "native-wgpu-frontier-depth2-minimax"
        } else {
            "native-wgpu-frontier-depth1"
        }),
        backend: Some("wgpu-frontier"),
    })
}

#[cfg(any(target_arch = "wasm32", not(feature = "neural-wgpu")))]
fn native_frontier_search_result(
    _game: &Game,
    _depth: i32,
    _min_depth: i32,
) -> Result<NativeFrontierSearchResult, String> {
    Ok(NativeFrontierSearchResult {
        result_json: None,
        metadata: None,
        gpu_search: None,
        backend: None,
    })
}

pub fn gpu_frontier_active_timeline_distance(timeline_ids: &[i32]) -> i32 {
    let min_timeline = timeline_ids
        .iter()
        .copied()
        .chain(std::iter::once(0))
        .min()
        .unwrap_or(0);
    let max_timeline = timeline_ids
        .iter()
        .copied()
        .chain(std::iter::once(0))
        .max()
        .unwrap_or(0);
    (-min_timeline).min(max_timeline).max(0) + 1
}

pub fn gpu_frontier_timeline_active(
    owner: &str,
    timeline_id: i32,
    active_distance: i32,
) -> Result<bool, String> {
    match owner.to_ascii_lowercase().as_str() {
        "neutral" => Ok(true),
        "white" | "black" => Ok(timeline_id.abs() <= active_distance),
        _ => Err(format!("unsupported GPU frontier timeline owner: {owner}")),
    }
}

pub fn gpu_frontier_present_time(active_latest: &[GpuFrontierActiveBoard]) -> i32 {
    active_latest
        .iter()
        .map(|board| board.time)
        .min()
        .unwrap_or(0)
}

pub fn gpu_frontier_pending_board_count(
    active_latest: &[GpuFrontierActiveBoard],
    present_time: i32,
    root_color: i32,
) -> usize {
    active_latest
        .iter()
        .filter(|board| board.time == present_time && board.side_to_move == root_color)
        .count()
}

pub fn gpu_timeline_sort_order(timelines: &[GpuTimelineSortKey]) -> Vec<usize> {
    let mut order = (0..timelines.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| (timelines[*index].row, timelines[*index].id));
    order
}

pub fn gpu_latest_board_index(board_times: &[i32]) -> Option<usize> {
    let first = board_times.first()?;
    let mut latest_index = 0;
    let mut latest_time = *first;
    for (index, time) in board_times.iter().copied().enumerate().skip(1) {
        if time > latest_time {
            latest_time = time;
            latest_index = index;
        }
    }
    Some(latest_index)
}

fn push_en_passant_record(out: &mut Vec<i32>, en_passant: Option<GpuEnPassantRecord>) {
    out.extend_from_slice(&en_passant_record(en_passant));
}

fn en_passant_record(en_passant: Option<GpuEnPassantRecord>) -> [i32; 4] {
    en_passant.map_or([-1, -1, -1, -1], |en_passant| {
        [
            en_passant.x,
            en_passant.y,
            en_passant.captured_x,
            en_passant.captured_y,
        ]
    })
}

fn square_codes_for_board_snapshot(board: &BoardSnapshot) -> Vec<i32> {
    let pieces = board
        .board
        .iter()
        .flat_map(|row| {
            row.iter().map(|piece| {
                piece.map(|piece| (piece_type_name(piece.piece_type), color_name(piece.color)))
            })
        })
        .collect::<Vec<_>>();
    gpu_search_board_to_square_codes(&pieces).unwrap_or_else(|_| vec![0; 64])
}

fn hash_frontier_words(words: &[i32]) -> (i32, i32) {
    gpu_frontier_hash_words(words)
}

pub fn gpu_frontier_hash_words(words: &[i32]) -> (i32, i32) {
    let mut low = 0x811c9dc5u32;
    let mut high = 0x9e3779b9u32;
    for (index, value) in words.iter().copied().enumerate() {
        low = (low ^ value as u32).wrapping_mul(0x01000193);
        high = high
            .wrapping_add(value as u32)
            .wrapping_add(index as u32)
            .wrapping_mul(0x85ebca6b);
    }
    (low as i32, high as i32)
}

pub fn gpu_frontier_positive_limit(value: Option<usize>, fallback: usize) -> usize {
    value.filter(|value| *value > 0).unwrap_or(fallback)
}

pub fn gpu_frontier_workgroup_size(max_invocations: usize) -> usize {
    if max_invocations >= 256 {
        256
    } else if max_invocations >= 128 {
        128
    } else if max_invocations >= 64 {
        64
    } else {
        32
    }
}

pub fn gpu_frontier_clamp_usize(value: usize, minimum: usize, maximum: usize) -> usize {
    value.max(minimum).min(maximum)
}

pub fn gpu_frontier_floor_power_of_two(value: usize) -> usize {
    let value = value.max(1);
    1usize << (usize::BITS - 1 - value.leading_zeros())
}

pub fn gpu_frontier_next_power_of_two(value: usize) -> usize {
    value.max(1).next_power_of_two()
}

pub fn bot_search_depth_at_least_one(depth: f64) -> i32 {
    if depth.is_finite() {
        (depth.floor() as i32).max(1)
    } else {
        1
    }
}

pub const DEFAULT_BOT_SEARCH_DEPTH: i32 = 2;
pub const DEFAULT_BOT_SEARCH_NODES: f64 = 64.0;
pub const DEFAULT_BOT_SEARCH_TIME_MS: f64 = 10_000.0;

pub fn bot_search_config_json(
    depth: f64,
    min_depth: f64,
    nodes: f64,
    time_ms: f64,
) -> Result<String, String> {
    let target_depth = bot_search_depth_at_least_one(if depth.is_finite() {
        depth
    } else {
        f64::from(DEFAULT_BOT_SEARCH_DEPTH)
    });
    let min_depth = bot_search_depth_at_least_one(if min_depth.is_finite() {
        min_depth
    } else {
        f64::from(DEFAULT_BOT_SEARCH_DEPTH)
    })
    .min(target_depth);
    let nodes = if nodes.is_finite() {
        nodes.max(1.0)
    } else {
        DEFAULT_BOT_SEARCH_NODES
    };
    let time_ms = if time_ms.is_finite() {
        time_ms.max(1.0)
    } else {
        DEFAULT_BOT_SEARCH_TIME_MS
    };
    serde_json::to_string(&serde_json::json!({
        "targetDepth": target_depth,
        "minDepth": min_depth,
        "nodes": nodes,
        "timeMs": time_ms,
    }))
    .map_err(|error| format!("Bot search config response failed to encode: {error}"))
}

pub fn gpu_worker_search_config_json(
    depth: f64,
    min_depth: f64,
    time_ms: f64,
) -> Result<String, String> {
    let requested_depth =
        bot_search_depth_at_least_one(if depth.is_finite() { depth } else { 1.0 });
    let minimum_depth = bot_search_depth_at_least_one(if min_depth.is_finite() {
        min_depth
    } else {
        1.0
    })
    .min(requested_depth);
    let search_time_ms = if time_ms.is_finite() {
        time_ms.max(1.0)
    } else {
        DEFAULT_BOT_SEARCH_TIME_MS
    };
    let deadline_delay_ms = if minimum_depth >= requested_depth {
        serde_json::Value::Null
    } else {
        serde_json::json!((search_time_ms * 0.8).floor().max(1.0))
    };
    serde_json::to_string(&serde_json::json!({
        "requestedDepth": requested_depth,
        "minimumDepth": minimum_depth,
        "searchTimeMs": search_time_ms,
        "deadlineDelayMs": deadline_delay_ms,
    }))
    .map_err(|error| format!("GPU worker search config response failed to encode: {error}"))
}

fn gpu_nodes_limit(nodes: f64, fallback: f64, minimum: usize, maximum: usize) -> usize {
    let nodes = if nodes.is_finite() { nodes } else { fallback };
    (nodes.max(minimum as f64).min(maximum as f64).floor() as usize).clamp(minimum, maximum)
}

pub fn gpu_search_ranking_limit(nodes: f64) -> usize {
    gpu_nodes_limit(nodes, DEFAULT_BOT_SEARCH_NODES, 16, 128)
}

pub fn gpu_search_reply_limit(nodes: f64) -> usize {
    gpu_nodes_limit(nodes, DEFAULT_BOT_SEARCH_NODES, 4, 12)
}

pub fn gpu_reply_pressure_reply_limit() -> usize {
    512
}

pub fn gpu_search_validation_limit(nodes: f64) -> usize {
    gpu_nodes_limit(nodes, DEFAULT_BOT_SEARCH_NODES, 8, 32)
}

pub fn gpu_supported_mutation_candidate_indexes_from_i32s(
    words: &[i32],
) -> Result<Vec<i32>, String> {
    if words.len() < 3 {
        return Err("GPU supported mutation request is truncated.".to_string());
    }
    let candidate_count = usize::try_from(words[0])
        .map_err(|_| "GPU supported mutation candidate count is negative.".to_string())?;
    let limit = usize::try_from(words[1])
        .map_err(|_| "GPU supported mutation limit is negative.".to_string())?;
    let require_child_boards = words[2] != 0;
    let expected = 3 + candidate_count.saturating_mul(2);
    if words.len() < expected {
        return Err("GPU supported mutation request does not contain every candidate.".to_string());
    }
    let mut indexes = Vec::new();
    for index in 0..candidate_count {
        if limit > 0 && indexes.len() >= limit {
            break;
        }
        let offset = 3 + index * 2;
        let status = words[offset];
        let has_child_boards = words[offset + 1] != 0;
        if status >= GPU_MUTATION_STATUS_OK && (!require_child_boards || has_child_boards) {
            indexes.push(i32::try_from(index).map_err(|_| {
                "GPU supported mutation candidate index exceeds i32 range.".to_string()
            })?);
        }
    }
    Ok(indexes)
}

pub fn gpu_supported_mutation_candidate_indexes_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuSupportedMutationCandidatesRequest>(request_json)
        .map_err(|error| format!("GPU supported mutation JSON request is invalid: {error}"))?;
    let limit = gpu_supported_mutation_limit(request.limit)?;
    let mut words = Vec::with_capacity(3 + request.candidates.len() * 2);
    words
        .push(i32::try_from(request.candidates.len()).map_err(|_| {
            "GPU supported mutation candidate count exceeds i32 range.".to_string()
        })?);
    words.push(limit);
    words.push(i32::from(request.require_child_boards.unwrap_or(true)));
    for candidate in request.candidates {
        words.push(candidate.mutation_status);
        words.push(i32::from(candidate.has_child_boards));
    }
    let indexes = gpu_supported_mutation_candidate_indexes_from_i32s(&words)?;
    serde_json::to_string(&indexes)
        .map_err(|error| format!("GPU supported mutation JSON response failed to encode: {error}"))
}

fn gpu_supported_mutation_limit(limit: Option<f64>) -> Result<i32, String> {
    let Some(limit) = limit.filter(|limit| limit.is_finite()) else {
        return Ok(0);
    };
    if limit <= 0.0 {
        return Ok(0);
    }
    if limit.floor() > i32::MAX as f64 {
        return Err("GPU supported mutation limit exceeds i32 range.".to_string());
    }
    Ok(limit.floor() as i32)
}

pub fn gpu_mutation_status_is_terminal(status: i32) -> bool {
    status == GPU_MUTATION_STATUS_ROYAL_CAPTURE
        || status == GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE
}

pub fn gpu_full_search_reported_depth(requested_depth: i32) -> i32 {
    requested_depth.min(2)
}

pub fn gpu_completed_reply_should_search(
    royal_capture_present: bool,
    now_ms: f64,
    deadline_at_ms: f64,
) -> bool {
    !royal_capture_present && now_ms < deadline_at_ms
}

pub fn gpu_diagnostic_rate(numerator: f64, denominator: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= 0.0 {
        return 0.0;
    }
    ((numerator / denominator) * 1000.0).round() / 1000.0
}

pub fn gpu_effective_branching_factor(selected_count: f64, cycles_completed: f64) -> f64 {
    if cycles_completed > 0.0 {
        ((selected_count / cycles_completed) * 100.0).round() / 100.0
    } else {
        selected_count
    }
}

pub fn gpu_reported_latency_ms(latency_ms: f64) -> f64 {
    if latency_ms.is_finite() {
        latency_ms.max(0.0).round()
    } else {
        0.0
    }
}

pub fn gpu_nodes_per_second(nodes: f64, latency_ms: f64) -> f64 {
    let latency_ms = if latency_ms.is_finite() {
        latency_ms.max(0.0)
    } else {
        0.0
    };
    if latency_ms > 0.0 {
        ((nodes * 1000.0) / latency_ms).round()
    } else {
        nodes
    }
}

pub fn gpu_search_nodes(nodes: f64) -> f64 {
    if nodes.is_finite() {
        nodes
    } else {
        64.0
    }
}

pub fn gpu_mutation_candidate_limit(candidate_count: usize) -> usize {
    candidate_count.min(64)
}

pub fn gpu_mutation_candidate_workgroups(candidate_limit: usize) -> usize {
    candidate_limit.div_ceil(64)
}

pub fn gpu_turn_completion_max_moves(existing_moves: usize, timeline_count: usize) -> usize {
    existing_moves.max(timeline_count.saturating_add(4))
}

pub fn gpu_candidate_max_dispatch_workgroups() -> usize {
    65_535
}

pub fn gpu_candidate_max_candidates_per_dispatch() -> usize {
    gpu_candidate_max_dispatch_workgroups().saturating_mul(64)
}

pub fn gpu_candidate_max_candidates_per_batch(max_binding_size: usize) -> usize {
    let max_by_binding = max_binding_size
        .checked_div(GPU_CANDIDATE_STRIDE.saturating_mul(std::mem::size_of::<i32>()))
        .unwrap_or(0);
    gpu_candidate_max_candidates_per_dispatch()
        .min(max_by_binding)
        .max(1)
}

pub fn gpu_candidate_source_batch_size(
    max_candidates_per_batch: usize,
    target_count: usize,
) -> usize {
    max_candidates_per_batch
        .checked_div(target_count.max(1))
        .unwrap_or(0)
        .max(1)
}

pub fn gpu_candidate_batch_source_count(
    source_count: usize,
    source_start: usize,
    source_batch_size: usize,
) -> usize {
    source_count
        .saturating_sub(source_start)
        .min(source_batch_size.max(1))
}

pub fn gpu_candidate_batch_candidate_count(source_count: usize, target_count: usize) -> usize {
    source_count.saturating_mul(target_count)
}

pub fn gpu_candidate_score_workgroups(batch_candidate_count: usize) -> usize {
    gpu_candidate_max_dispatch_workgroups().min(batch_candidate_count.div_ceil(64))
}

pub fn gpu_reply_score_workgroups_x(root_count: usize) -> usize {
    root_count.div_ceil(16)
}

pub fn gpu_reply_score_workgroups_y(reply_count: usize) -> usize {
    reply_count.div_ceil(16)
}

pub fn bot_next_search_depth(current_depth: i32, target_depth: i32) -> i32 {
    let target_depth = target_depth.max(1);
    if current_depth <= 0 {
        target_depth.min(2)
    } else {
        target_depth.min(current_depth.saturating_add(2).max(1))
    }
}

pub fn bot_worker_search_time_ms(time_ms: i32) -> i32 {
    let time_ms = time_ms.max(1);
    let margin = (time_ms / 20).clamp(100, 1_000);
    (time_ms - margin).max(1)
}

pub fn bot_completed_search_depth(
    result_depth: f64,
    requested_depth: i32,
    result_ends_in_royal_capture: bool,
) -> i32 {
    if !result_depth.is_finite() {
        return 0;
    }
    let requested_depth = requested_depth.max(1);
    let completed_depth = requested_depth.min(result_depth.floor() as i32);
    if completed_depth != requested_depth {
        return 0;
    }
    if (completed_depth >= 2 && completed_depth % 2 == 0)
        || (completed_depth >= 1 && result_ends_in_royal_capture)
    {
        completed_depth
    } else {
        0
    }
}

pub fn bot_result_ends_in_royal_capture(
    result_reason: Option<&str>,
    gpu_terminal: bool,
    terminal: bool,
) -> bool {
    result_reason == Some("royal-capture")
        || gpu_terminal
        || (terminal && result_reason == Some("royal-capture"))
}

pub fn bot_result_ends_in_royal_capture_json(request_json: &str) -> Result<bool, String> {
    let value: serde_json::Value = serde_json::from_str(request_json)
        .map_err(|error| format!("Bot result terminal request is not valid JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Bot result terminal request must be an object.".to_string())?;
    Ok(bot_result_ends_in_royal_capture(
        object
            .get("resultReason")
            .or_else(|| object.get("reason"))
            .and_then(serde_json::Value::as_str),
        object
            .get("gpuTerminal")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        object
            .get("terminal")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    ))
}

fn origin_code(origin: &Origin) -> i32 {
    match origin {
        Origin::None => 0,
        Origin::Move { move_type, .. } => gpu_frontier_origin_code(Some(move_type)),
    }
}

pub fn gpu_frontier_origin_code(origin: Option<&str>) -> i32 {
    match origin {
        Some("source-advance") => 1,
        Some("branch") => 2,
        Some("cross-board") => 3,
        Some(_) => 4,
        None => 0,
    }
}

fn castling_code(castling: CastlingRights) -> i32 {
    castling.white_kingside as i32
        | ((castling.white_queenside as i32) << 1)
        | ((castling.black_kingside as i32) << 2)
        | ((castling.black_queenside as i32) << 3)
}

pub fn gpu_search_board_to_square_codes(
    board: &[Option<(&str, &str)>],
) -> Result<Vec<i32>, String> {
    let mut squares = Vec::with_capacity(64);
    for index in 0..64 {
        squares.push(match board.get(index).copied().flatten() {
            Some((piece_type, color)) => gpu_search_piece_code(piece_type, color)?,
            None => 0,
        });
    }
    Ok(squares)
}

pub fn gpu_search_piece_code(piece_type: &str, color: &str) -> Result<i32, String> {
    let piece_type = gpu_search_piece_type_code(piece_type)
        .ok_or_else(|| format!("unsupported GPU search piece type: {piece_type}"))?;
    let color = gpu_search_color_code(color)?;
    Ok(piece_type | (color << 8))
}

pub fn gpu_search_piece_type_code(piece_type: &str) -> Option<i32> {
    match piece_type {
        "king" => Some(1),
        "commonKing" => Some(2),
        "queen" => Some(3),
        "royalQueen" => Some(4),
        "princess" => Some(5),
        "rook" => Some(6),
        "bishop" => Some(7),
        "unicorn" => Some(8),
        "dragon" => Some(9),
        "knight" => Some(10),
        "pawn" => Some(11),
        "brawn" => Some(12),
        _ => None,
    }
}

pub fn gpu_search_piece_type_from_code(code: i32) -> Option<&'static str> {
    match code {
        1 => Some("king"),
        2 => Some("commonKing"),
        3 => Some("queen"),
        4 => Some("royalQueen"),
        5 => Some("princess"),
        6 => Some("rook"),
        7 => Some("bishop"),
        8 => Some("unicorn"),
        9 => Some("dragon"),
        10 => Some("knight"),
        11 => Some("pawn"),
        12 => Some("brawn"),
        _ => None,
    }
}

pub fn gpu_search_piece_from_code(code: i32) -> Option<GpuDecodedPiece> {
    Some(GpuDecodedPiece {
        piece_type: gpu_search_piece_type_from_code(code & 255)?,
        color: gpu_search_color_from_code((code >> 8) & 255),
    })
}

pub fn gpu_search_square_codes_to_board(squares: &[i32]) -> GpuDecodedBoard {
    let mut board = Vec::with_capacity(8);
    for y in 0..8 {
        let mut row = Vec::with_capacity(8);
        for x in 0..8 {
            row.push(gpu_search_piece_from_code(
                *squares.get(y * 8 + x).unwrap_or(&0),
            ));
        }
        board.push(row);
    }
    board
}

pub fn gpu_search_color_code(color: &str) -> Result<i32, String> {
    match color.to_ascii_lowercase().as_str() {
        "white" => Ok(0),
        "black" => Ok(1),
        _ => Err(format!("unsupported GPU search color: {color}")),
    }
}

pub fn gpu_search_color_from_code(code: i32) -> &'static str {
    if code == 1 {
        "black"
    } else {
        "white"
    }
}

pub fn gpu_search_opposite_color(color: &str) -> Result<&'static str, String> {
    match color.to_ascii_lowercase().as_str() {
        "white" => Ok("black"),
        "black" => Ok("white"),
        _ => Err(format!("unsupported GPU search color: {color}")),
    }
}

pub fn gpu_search_owner_code(owner: &str) -> Result<i32, String> {
    match owner.to_ascii_lowercase().as_str() {
        "neutral" => Ok(0),
        "white" => Ok(1),
        "black" => Ok(2),
        _ => Err(format!("unsupported GPU search timeline owner: {owner}")),
    }
}

pub fn gpu_search_owner_from_code(code: i32) -> &'static str {
    match code {
        1 => "white",
        2 => "black",
        _ => "neutral",
    }
}

fn color_code(color: Color) -> i32 {
    gpu_search_color_code(color_name(color)).unwrap_or(0)
}

fn owner_code(owner: TimelineOwner) -> i32 {
    gpu_search_owner_code(owner_name(owner)).unwrap_or(0)
}

fn piece_type_name(piece_type: PieceType) -> &'static str {
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

fn color_name(color: Color) -> &'static str {
    match color {
        Color::White => "white",
        Color::Black => "black",
    }
}

fn owner_name(owner: TimelineOwner) -> &'static str {
    match owner {
        TimelineOwner::Neutral => "neutral",
        TimelineOwner::White => "white",
        TimelineOwner::Black => "black",
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontier_layout_matches_web_contract() {
        assert_eq!(FRONTIER_HEADER_STRIDE, 16);
        assert_eq!(FRONTIER_BOARD_STRIDE, 78);
        assert_eq!(FRONTIER_PLAN_OFFSET, 32);
        assert_eq!(FRONTIER_BOARD_OFFSET, 544);
        assert_eq!(FRONTIER_CANDIDATE_STRIDE, 24);
        assert_eq!(FRONTIER_DELTA_STRIDE, 156);
        assert_eq!(FRONTIER_SUMMARY_STRIDE, 12);
        assert_eq!(frontier_state_stride(1), 622);
    }

    #[test]
    fn frontier_neural_param_bytes_match_shader_layouts() {
        assert_eq!(
            frontier_neural_params_bytes(1, 2, 3, 4, 5, 6, 7, 8),
            [1_u32, 2, 3, 4, 5, 6, 7, 8]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        let apply = frontier_neural_apply_params_bytes(9, -1, 2.5, -3.0, 10);
        assert_eq!(u32::from_le_bytes(apply[0..4].try_into().unwrap()), 9);
        assert_eq!(i32::from_le_bytes(apply[4..8].try_into().unwrap()), -1);
        assert_eq!(f32::from_le_bytes(apply[8..12].try_into().unwrap()), 2.5);
        assert_eq!(f32::from_le_bytes(apply[12..16].try_into().unwrap()), -3.0);
        assert_eq!(u32::from_le_bytes(apply[16..20].try_into().unwrap()), 10);
        assert_eq!(apply.len(), 32);
        assert_eq!(
            frontier_neural_layer_params_bytes(11, 12, 13),
            [11_u32, 12, 13, 0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        let policy = frontier_policy_params_bytes(14, 15, 16, 25.0);
        assert_eq!(u32::from_le_bytes(policy[0..4].try_into().unwrap()), 14);
        assert_eq!(u32::from_le_bytes(policy[4..8].try_into().unwrap()), 15);
        assert_eq!(u32::from_le_bytes(policy[8..12].try_into().unwrap()), 16);
        assert_eq!(f32::from_le_bytes(policy[12..16].try_into().unwrap()), 25.0);

        assert_eq!(frontier_neural_effective_batch_size(100, 32.9), 32);
        assert_eq!(frontier_neural_effective_batch_size(8, 32.0), 8);
        assert_eq!(frontier_neural_effective_batch_size(8, f64::NAN), 1);
        assert_eq!(frontier_neural_batch_count(100, 64, 32), 32);
        assert_eq!(frontier_neural_batch_count(100, 96, 32), 4);
        assert_eq!(frontier_neural_select_board_workgroups(5), 2);
        assert_eq!(frontier_neural_project_workgroups_x(17), 2);
        assert_eq!(frontier_neural_project_workgroups_y(33), 3);
        assert_eq!(frontier_neural_layer_workgroups_x(17), 2);
        assert_eq!(frontier_neural_layer_workgroups_y(33), 3);
        assert_eq!(frontier_neural_output_workgroups(65), 2);
    }

    #[test]
    fn encodes_default_game_as_frontier_root() {
        let root = encode_frontier_root(&Game::new(), 1).expect("encode default game");
        assert_eq!(root.board_count, 1);
        assert_eq!(root.words.len(), frontier_state_stride(1));
        assert_eq!(root.words[FRONTIER_HEADER_PARENT], -1);
        assert_eq!(root.words[FRONTIER_HEADER_ROOT], 0);
        assert_eq!(root.words[FRONTIER_HEADER_TURN], 0);
        assert_eq!(root.words[FRONTIER_HEADER_BOARD_COUNT], 1);
        assert_eq!(root.words[FRONTIER_HEADER_COMPLETE], 0);
        assert_eq!(root.words[FRONTIER_HEADER_TERMINAL], 0);
        assert_eq!(root.words[FRONTIER_HEADER_NEXT_WHITE_TIMELINE], 1);
        assert_eq!(root.words[FRONTIER_HEADER_NEXT_BLACK_TIMELINE], -1);
        assert_eq!(root.words[FRONTIER_HEADER_PRESENT_TIME], 0);
        assert_eq!(root.words[FRONTIER_HEADER_PENDING_BOARDS], 1);

        let board = FRONTIER_BOARD_OFFSET;
        assert_eq!(root.words[board + FRONTIER_BOARD_TIMELINE_ID], 0);
        assert_eq!(root.words[board + FRONTIER_BOARD_ROW], 0);
        assert_eq!(root.words[board + FRONTIER_BOARD_OWNER], 0);
        assert_eq!(root.words[board + FRONTIER_BOARD_TIME], 0);
        assert_eq!(root.words[board + FRONTIER_BOARD_SIDE_TO_MOVE], 0);
        assert_eq!(root.words[board + FRONTIER_BOARD_CASTLING], 15);
        assert_eq!(root.words[board + FRONTIER_BOARD_EN_PASSANT], -1);
        assert_eq!(root.words[board + FRONTIER_BOARD_LATEST], 1);
        assert_eq!(root.words[board + FRONTIER_BOARD_ORIGIN], 0);
        assert_eq!(root.words[board + FRONTIER_BOARD_ACTIVE], 1);
        assert_eq!(root.words[board + FRONTIER_BOARD_PENDING], 1);
        assert_eq!(root.words[board + FRONTIER_BOARD_SQUARES], 6);
        assert_eq!(root.words[board + FRONTIER_BOARD_SQUARES + 4], 1);
        assert_eq!(
            root.words[board + FRONTIER_BOARD_SQUARES + 60],
            1 | (1 << 8)
        );

        let hashed = hash_frontier_words(
            &root.words[FRONTIER_BOARD_OFFSET..FRONTIER_BOARD_OFFSET + FRONTIER_BOARD_STRIDE],
        );
        assert_eq!((root.hash_low, root.hash_high), hashed);
        assert_eq!(root.words[FRONTIER_HEADER_HASH_LOW], root.hash_low);
        assert_eq!(root.words[FRONTIER_HEADER_HASH_HIGH], root.hash_high);
    }

    #[test]
    fn root_encoding_rejects_adapter_board_limit() {
        let error = encode_frontier_root(&Game::new(), 0).expect_err("max board limit should fail");
        assert!(error.contains("GPU frontier snapshot has 1 boards"));
    }
}
