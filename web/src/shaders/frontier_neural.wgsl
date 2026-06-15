struct Params {
  state_count: u32,
  state_stride: u32,
  board_offset: u32,
  max_boards: u32,
  state_offset: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
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

const HEADER_SCORE: u32 = 2u;
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
const INPUT_SIZE: u32 = MAX_NEURAL_BOARDS * BOARD_PLANES * BOARD_SQUARE_COUNT;
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
  if (states[base + BOARD_LATEST] != 0) { return 0; }
  if (has_royal(state, board)) { return 1; }
  if (states[base + BOARD_ORIGIN] != 0) { return 2; }
  return 3;
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
  let board_count = min(params.max_boards, u32(max(0, states[state_base(state) + HEADER_BOARD_COUNT])));
  var chosen = -1;
  for (var board = 0u; board < board_count; board = board + 1u) {
    if (category(state, board) >= 3) { continue; }
    var rank = 0u;
    for (var other = 0u; other < board_count; other = other + 1u) {
      if (category(state, other) < 3 && board_before(state, other, board)) { rank = rank + 1u; }
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

@compute @workgroup_size(256)
fn encode_neural_features(@builtin(global_invocation_id) id: vec3<u32>) {
  let state = id.x / INPUT_SIZE; let feature = id.x % INPUT_SIZE;
  if (state >= params.state_count) { return; }
  let square = feature % 64u; let plane = (feature / 64u) % 32u; let slot = feature / (64u * 32u);
  let selected = selected_boards[state * MAX_NEURAL_BOARDS + slot]; let output = state * INPUT_SIZE + feature;
  if (selected < 0) { features[output] = 0.0; return; }
  let base = board_base(state, u32(selected));
  if (plane < 24u) { features[output] = select(0.0, 1.0, piece_plane(states[base + BOARD_SQUARES + square]) == i32(plane)); return; }
  let present = states[state_base(state) + HEADER_PRESENT];
  let perspective = states[state_base(state) + HEADER_TURN]; let side = states[base + BOARD_SIDE]; let owner = states[base + BOARD_OWNER];
  let time = states[base + BOARD_TIME];
  let relative_side = select(-1.0, 1.0, side == perspective);
  let is_active = states[base + BOARD_ACTIVE] != 0; let owner_sign = select(0.0, select(-1.0, 1.0, owner - 1 == perspective), owner != 0);
  if (plane == 24u) { features[output] = relative_side; }
  else if (plane == 25u) { features[output] = select(0.0, 1.0, is_active); }
  else if (plane == 26u) { features[output] = select(0.0, 1.0, states[base + BOARD_LATEST] != 0); }
  else if (plane == 27u) { features[output] = select(0.0, 1.0, time == present); }
  else if (plane == 28u) { features[output] = owner_sign; }
  else if (plane == 29u) { features[output] = f32(clamp(time - present, -16, 16)) / 16.0; }
  else if (plane == 30u) { features[output] = 1.0; }
  else { features[output] = 0.0; }
}

@compute @workgroup_size(64)
fn apply_neural_values(@builtin(global_invocation_id) id: vec3<u32>) {
  let state = id.x;
  if (state >= apply_params.state_count) { return; }
  let base = (apply_params.state_offset + state) * params.state_stride;
  if (states[base + HEADER_BOARD_COUNT] == 0 || states[base + HEADER_TERMINAL] != 0) { return; }
  let perspective = select(-1.0, 1.0, states[base + HEADER_TURN] == apply_params.root_color);
  let neural = (predictions[state] * apply_params.value_scale + apply_params.value_bias) * perspective;
  let neural_score = i32(round(neural * 100.0));
  let score = states[base + HEADER_SCORE] - states[base + HEADER_LAST_NEURAL] + neural_score;
  states[base + HEADER_SCORE] = score;
  states[base + HEADER_LAST_NEURAL] = neural_score;
  summaries[(apply_params.state_offset + state) * SUMMARY_STRIDE + 1u] = score;
}
