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
    origin: Option<GpuOriginJson>,
    origin_kind: Option<i32>,
    squares: Option<Vec<i32>>,
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
struct GpuOriginJson {
    #[serde(rename = "type")]
    kind: String,
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

#[derive(serde::Deserialize)]
struct GpuSearchSelectionRequest {
    candidates: Vec<GpuSearchSelectionCandidate>,
    temperature: Option<f64>,
    #[serde(rename = "randomSeed")]
    random_seed: Option<i64>,
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
        request.random_seed.unwrap_or(0),
    );
    serde_json::to_string(&response)
        .map_err(|error| format!("GPU search selection response failed to encode: {error}"))
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

pub fn gpu_ranked_candidate_indexes_from_i32s(words: &[i32]) -> Result<Vec<i32>, String> {
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
    Ok(ranked.into_iter().map(|(index, _)| index as i32).collect())
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

#[derive(serde::Deserialize)]
struct GpuPendingBoardRefJson {
    #[serde(rename = "timelineId")]
    timeline_id: i32,
    time: i32,
}

#[derive(serde::Deserialize)]
struct GpuMovePositionJson {
    #[serde(rename = "timelineId")]
    timeline_id: i32,
    time: i32,
    x: i32,
    y: i32,
}

#[derive(serde::Deserialize)]
struct GpuMoveJson {
    from: GpuMovePositionJson,
    to: GpuMovePositionJson,
}

#[derive(serde::Deserialize)]
struct GpuChoiceAgreementRequest {
    selected: Vec<GpuMoveJson>,
    choices: Vec<Vec<GpuMoveJson>>,
    limits: Vec<usize>,
}

#[derive(serde::Serialize)]
struct GpuChoiceAgreementResponse {
    agreements: Vec<i32>,
}

pub fn gpu_turn_completion_key_json(request_json: &str) -> Result<String, String> {
    let mut pending = serde_json::from_str::<Vec<GpuPendingBoardRefJson>>(request_json)
        .map_err(|error| format!("GPU turn completion key request is invalid: {error}"))?;
    pending.sort_by_key(|board| (board.timeline_id, board.time));
    Ok(pending
        .into_iter()
        .map(|board| format!("{}:{}", board.timeline_id, board.time))
        .collect::<Vec<_>>()
        .join("|"))
}

pub fn gpu_choice_agreement_json(request_json: &str) -> Result<String, String> {
    let request = serde_json::from_str::<GpuChoiceAgreementRequest>(request_json)
        .map_err(|error| format!("GPU choice agreement request is invalid: {error}"))?;
    let selected_key = gpu_move_plan_key(&request.selected);
    let agreements = request
        .limits
        .iter()
        .map(|limit| {
            if selected_key.is_empty() {
                return 0;
            }
            i32::from(
                request
                    .choices
                    .iter()
                    .take(*limit)
                    .any(|choice| gpu_move_plan_key(choice) == selected_key),
            )
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&GpuChoiceAgreementResponse { agreements })
        .map_err(|error| format!("GPU choice agreement response failed to encode: {error}"))
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
            gpu_frontier_origin_code(board.origin.as_ref().map(|origin| origin.kind.as_str()))
        }),
        squares: gpu_board_squares_from_json(board)?,
    })
}

fn gpu_board_squares_from_json(board: &GpuBoardJson) -> Result<Vec<i32>, String> {
    if let Some(squares) = &board.squares {
        let mut output = vec![0; 64];
        for (index, value) in squares.iter().take(64).enumerate() {
            output[index] = *value;
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
    if completed_depth >= 2 && completed_depth % 2 == 0 {
        completed_depth
    } else if completed_depth >= 1 && result_ends_in_royal_capture {
        completed_depth
    } else {
        0
    }
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
