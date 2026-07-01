struct Params {
  state_count: u32,
  state_stride: u32,
  board_offset: u32,
  max_boards: u32,
  state_offset: u32,
  projection_size: u32,
  projection_seed: u32,
  target_depth: u32,
};

struct ApplyParams {
  state_count: u32,
  root_color: i32,
  value_scale: f32,
  value_bias: f32,
  state_offset: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> states: array<i32>;
@group(0) @binding(1) var<storage, read_write> selected_boards: array<i32>;
@group(0) @binding(2) var<storage, read_write> features: array<f32>;
@group(0) @binding(3) var<storage, read_write> predictions: array<f32>;
@group(0) @binding(4) var<storage, read_write> summaries: array<i32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<uniform> apply_params: ApplyParams;
@group(0) @binding(7) var<storage, read_write> active_states: array<u32>;

const HEADER_SCORE: u32 = 2u;
const HEADER_DEPTH: u32 = 3u;
const HEADER_TURN: u32 = 4u;
const HEADER_BOARD_COUNT: u32 = 5u;
const HEADER_TERMINAL: u32 = 8u;
const HEADER_PRESENT: u32 = 13u;
const HEADER_LAST_NEURAL: u32 = 15u;
const BOARD_STRIDE: u32 = 78u;
const BOARD_TIMELINE: u32 = 0u;
const BOARD_OWNER: u32 = 2u;
const BOARD_TIME: u32 = 3u;
const BOARD_SIDE: u32 = 4u;
const BOARD_LATEST: u32 = 10u;
const BOARD_ORIGIN: u32 = 11u;
const BOARD_SQUARES: u32 = 12u;
const BOARD_ACTIVE: u32 = 76u;
const MAX_NEURAL_BOARDS: u32 = 16u;
const BOARD_PLANES: u32 = 32u;
const BOARD_SQUARE_COUNT: u32 = 64u;
const SUMMARY_STRIDE: u32 = 12u;

fn state_base(state: u32) -> u32 { return (params.state_offset + state) * params.state_stride; }
fn board_base(state: u32, board: u32) -> u32 { return state_base(state) + params.board_offset + board * BOARD_STRIDE; }
fn abs_i32(value: i32) -> i32 { return select(value, -value, value < 0); }

fn has_royal(state: u32, board: u32) -> bool {
  let base = board_base(state, board);
  for (var square = 0u; square < 64u; square = square + 1u) {
    let piece = states[base + BOARD_SQUARES + square] & 255;
    if (piece == 1 || piece == 4) { return true; }
  }
  return false;
}

fn category(state: u32, board: u32) -> i32 {
  let base = board_base(state, board);
  if (states[base + BOARD_LATEST] != 0 && states[base + BOARD_ACTIVE] != 0) { return 0; }
  if (states[base + BOARD_LATEST] != 0) { return 1; }
  if (has_royal(state, board)) { return 2; }
  if (states[base + BOARD_ORIGIN] != 0) { return 3; }
  return 4;
}

fn board_before(state: u32, left: u32, right: u32) -> bool {
  let left_base = board_base(state, left); let right_base = board_base(state, right);
  let left_category = category(state, left); let right_category = category(state, right);
  if (left_category != right_category) { return left_category < right_category; }
  let left_time = states[left_base + BOARD_TIME]; let right_time = states[right_base + BOARD_TIME];
  if (left_time != right_time) { return left_time > right_time; }
  let left_timeline = states[left_base + BOARD_TIMELINE]; let right_timeline = states[right_base + BOARD_TIMELINE];
  if (abs_i32(left_timeline) != abs_i32(right_timeline)) { return abs_i32(left_timeline) < abs_i32(right_timeline); }
  if (left_timeline != right_timeline) { return left_timeline < right_timeline; }
  return left < right;
}

@compute @workgroup_size(64)
fn select_neural_boards(@builtin(global_invocation_id) id: vec3<u32>) {
  let state = id.x / MAX_NEURAL_BOARDS; let slot = id.x % MAX_NEURAL_BOARDS;
  if (state >= params.state_count) { return; }
  if (slot == 0u) {
    let base = state_base(state);
    active_states[state] = select(1u, 0u,
      states[base + HEADER_BOARD_COUNT] == 0
      || states[base + HEADER_TERMINAL] != 0
      || states[base + HEADER_LAST_NEURAL] != 0
    );
  }
  let board_count = min(params.max_boards, u32(max(0, states[state_base(state) + HEADER_BOARD_COUNT])));
  var chosen = -1;
  for (var board = 0u; board < board_count; board = board + 1u) {
    if (category(state, board) >= 4) { continue; }
    var rank = 0u;
    for (var other = 0u; other < board_count; other = other + 1u) {
      if (category(state, other) < 4 && board_before(state, other, board)) { rank = rank + 1u; }
    }
    if (rank == slot) { chosen = i32(board); break; }
  }
  selected_boards[state * MAX_NEURAL_BOARDS + slot] = chosen;
}

fn piece_plane(code: i32) -> i32 {
  let piece = code & 255;
  if (piece <= 0 || piece > 12) { return -1; }
  return ((code >> 8) & 255) * 12 + piece - 1;
}

fn projection_hash(raw_index: u32, projection_index: u32, seed: u32) -> u32 {
  var hash = seed ^ raw_index;
  hash = hash * 16777619u;
  hash = hash ^ projection_index;
  hash = hash * 16777619u;
  return hash ^ (hash >> 16u);
}

fn projection_sign(feature_index: u32, projection: u32) -> f32 {
  return select(-1.0, 1.0, (projection_hash(feature_index, projection, params.projection_seed) & 1u) == 0u);
}

@compute @workgroup_size(16, 16)
fn project_neural_features(@builtin(global_invocation_id) id: vec3<u32>) {
  let state = id.x;
  let projection = id.y;
  if (state >= params.state_count || projection >= params.projection_size) { return; }
  let base_state = state_base(state);
  let output = state * params.projection_size + projection;
  if (active_states[state] == 0u) {
    features[output] = 0.0;
    return;
  }

  let present = states[base_state + HEADER_PRESENT];
  let perspective = states[base_state + HEADER_TURN];
  var active_count = 0u;
  for (var slot = 0u; slot < MAX_NEURAL_BOARDS; slot = slot + 1u) {
    let selected = selected_boards[state * MAX_NEURAL_BOARDS + slot];
    if (selected < 0) { continue; }
    let base = board_base(state, u32(selected));
    for (var square = 0u; square < BOARD_SQUARE_COUNT; square = square + 1u) {
      if (piece_plane(states[base + BOARD_SQUARES + square]) >= 0) { active_count = active_count + 1u; }
    }
    let owner = states[base + BOARD_OWNER];
    let time_delta = f32(clamp(states[base + BOARD_TIME] - present, -16, 16)) / 16.0;
    let metadata = array<f32, 7>(
      select(-1.0, 1.0, states[base + BOARD_SIDE] == perspective),
      select(0.0, 1.0, states[base + BOARD_ACTIVE] != 0),
      select(0.0, 1.0, states[base + BOARD_LATEST] != 0),
      select(0.0, 1.0, states[base + BOARD_TIME] == present),
      select(0.0, select(-1.0, 1.0, owner - 1 == perspective), owner != 0),
      time_delta,
      1.0
    );
    for (var metadata_index = 0u; metadata_index < 7u; metadata_index = metadata_index + 1u) {
      if (metadata[metadata_index] != 0.0) { active_count = active_count + BOARD_SQUARE_COUNT; }
    }
  }

  if (active_count == 0u) {
    features[output] = 0.0;
    return;
  }
  let scale = sqrt(f32(active_count));
  var sum = 0.0;
  for (var slot = 0u; slot < MAX_NEURAL_BOARDS; slot = slot + 1u) {
    let selected = selected_boards[state * MAX_NEURAL_BOARDS + slot];
    if (selected < 0) { continue; }
    let base = board_base(state, u32(selected));
    for (var square = 0u; square < BOARD_SQUARE_COUNT; square = square + 1u) {
      let plane = piece_plane(states[base + BOARD_SQUARES + square]);
      if (plane >= 0) {
        let feature_index = slot * BOARD_PLANES * BOARD_SQUARE_COUNT + u32(plane) * BOARD_SQUARE_COUNT + square;
        sum = sum + projection_sign(feature_index, projection) / scale;
      }
    }
    let owner = states[base + BOARD_OWNER];
    let metadata = array<f32, 7>(
      select(-1.0, 1.0, states[base + BOARD_SIDE] == perspective),
      select(0.0, 1.0, states[base + BOARD_ACTIVE] != 0),
      select(0.0, 1.0, states[base + BOARD_LATEST] != 0),
      select(0.0, 1.0, states[base + BOARD_TIME] == present),
      select(0.0, select(-1.0, 1.0, owner - 1 == perspective), owner != 0),
      f32(clamp(states[base + BOARD_TIME] - present, -16, 16)) / 16.0,
      1.0
    );
    for (var metadata_index = 0u; metadata_index < 7u; metadata_index = metadata_index + 1u) {
      let value = metadata[metadata_index];
      if (value == 0.0) { continue; }
      let plane = 24u + metadata_index;
      for (var square = 0u; square < BOARD_SQUARE_COUNT; square = square + 1u) {
        let feature_index = slot * BOARD_PLANES * BOARD_SQUARE_COUNT + plane * BOARD_SQUARE_COUNT + square;
        sum = sum + value * projection_sign(feature_index, projection) / scale;
      }
    }
  }
  features[output] = sum;
}

@compute @workgroup_size(64)
fn apply_neural_values(@builtin(global_invocation_id) id: vec3<u32>) {
  let state = id.x;
  if (state >= apply_params.state_count) { return; }
  let base = (apply_params.state_offset + state) * params.state_stride;
  if (active_states[state] == 0u) { return; }
  let perspective = select(-1.0, 1.0, states[base + HEADER_TURN] == apply_params.root_color);
  let neural = clamp(predictions[state] * apply_params.value_scale + apply_params.value_bias, -1.0, 1.0) * perspective;
  let neural_score = i32(round(neural * 20000.0));
  let score = states[base + HEADER_SCORE] - states[base + HEADER_LAST_NEURAL] + neural_score;
  states[base + HEADER_SCORE] = score;
  states[base + HEADER_LAST_NEURAL] = neural_score;
  summaries[(apply_params.state_offset + state) * SUMMARY_STRIDE + 1u] = score;
}
