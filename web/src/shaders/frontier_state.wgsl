struct Params {
  selected_count: u32,
  max_boards: u32,
  state_stride: u32,
  board_offset: u32,
  plan_offset: u32,
  delta_stride: u32,
  candidate_stride: u32,
  max_plan_moves: u32,
  ancestry_offset: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> parent_states: array<i32>;
@group(0) @binding(1) var<storage, read> candidates: array<i32>;
@group(0) @binding(2) var<storage, read> deltas: array<i32>;
@group(0) @binding(3) var<storage, read> selected: array<i32>;
@group(0) @binding(4) var<storage, read_write> next_states: array<i32>;
@group(0) @binding(5) var<storage, read_write> summaries: array<i32>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;

override MATERIALIZE_WORKGROUP_SIZE: u32 = 64u;

const HEADER_PARENT: u32 = 0u;
const HEADER_ROOT: u32 = 1u;
const HEADER_SCORE: u32 = 2u;
const HEADER_DEPTH: u32 = 3u;
const HEADER_TURN: u32 = 4u;
const HEADER_BOARD_COUNT: u32 = 5u;
const HEADER_PLAN_LENGTH: u32 = 6u;
const HEADER_COMPLETE: u32 = 7u;
const HEADER_TERMINAL: u32 = 8u;
const HEADER_HASH_LOW: u32 = 9u;
const HEADER_HASH_HIGH: u32 = 10u;
const HEADER_NEXT_WHITE: u32 = 11u;
const HEADER_NEXT_BLACK: u32 = 12u;
const HEADER_PRESENT: u32 = 13u;
const HEADER_PENDING: u32 = 14u;
const HEADER_LAST_NEURAL: u32 = 15u;

const CANDIDATE_PARENT: u32 = 0u;
const CANDIDATE_ROOT: u32 = 1u;
const CANDIDATE_SCORE: u32 = 2u;
const CANDIDATE_LAST_NEURAL: u32 = 3u;
const CANDIDATE_STATUS: u32 = 5u;
const CANDIDATE_HASH_LOW: u32 = 6u;
const CANDIDATE_HASH_HIGH: u32 = 7u;
const CANDIDATE_MOVE: u32 = 8u;
const CANDIDATE_DELTA_COUNT: u32 = 16u;
const CANDIDATE_CARRY: u32 = 19u;
const CANDIDATE_NODE_ID: u32 = 20u;
const CANDIDATE_POLICY_PRIOR: u32 = 21u;

const BOARD_STRIDE: u32 = 78u;
const BOARD_TIMELINE: u32 = 0u;
const BOARD_OWNER: u32 = 2u;
const BOARD_TIME: u32 = 3u;
const BOARD_SIDE: u32 = 4u;
const BOARD_LATEST: u32 = 10u;
const BOARD_ACTIVE: u32 = 76u;
const BOARD_PENDING: u32 = 77u;
const SUMMARY_STRIDE: u32 = 12u;

fn abs_i32(value: i32) -> i32 {
  return select(value, -value, value < 0);
}

fn owner_active(owner: i32, timeline_id: i32, active_distance: i32) -> bool {
  return owner == 0 || abs_i32(timeline_id) <= active_distance;
}

fn board_base(state_base: u32, board_index: u32) -> u32 {
  return state_base + params.board_offset + board_index * BOARD_STRIDE;
}

fn recompute_turn_status(state_base: u32) {
  let board_count = u32(max(0, next_states[state_base + HEADER_BOARD_COUNT]));
  let turn = next_states[state_base + HEADER_TURN];
  if (board_count == 0u) {
    next_states[state_base + HEADER_COMPLETE] = 1;
    next_states[state_base + HEADER_PRESENT] = 0;
    next_states[state_base + HEADER_PENDING] = 0;
    return;
  }
  var min_timeline = next_states[board_base(state_base, 0u) + BOARD_TIMELINE];
  var max_timeline = min_timeline;
  for (var index = 1u; index < board_count; index = index + 1u) {
    let timeline = next_states[board_base(state_base, index) + BOARD_TIMELINE];
    min_timeline = min(min_timeline, timeline);
    max_timeline = max(max_timeline, timeline);
  }
  let active_distance = max(0, min(-min_timeline, max_timeline)) + 1;
  var present = 2147483647;
  for (var index = 0u; index < board_count; index = index + 1u) {
    let base = board_base(state_base, index);
    let timeline = next_states[base + BOARD_TIMELINE];
    let owner = next_states[base + BOARD_OWNER];
    let time = next_states[base + BOARD_TIME];
    let latest = next_states[base + BOARD_LATEST] != 0;
    let active_flag = owner_active(owner, timeline, active_distance);
    next_states[base + BOARD_ACTIVE] = select(0, 1, latest && active_flag);
    next_states[base + BOARD_PENDING] = 0;
    if (latest && active_flag) {
      present = min(present, time);
    }
  }
  var pending = 0;
  for (var index = 0u; index < board_count; index = index + 1u) {
    let base = board_base(state_base, index);
    let time = next_states[base + BOARD_TIME];
    let side = next_states[base + BOARD_SIDE];
    let board_pending = next_states[base + BOARD_ACTIVE] != 0 && time == present && side == turn;
    next_states[base + BOARD_PENDING] = select(0, 1, board_pending);
    if (board_pending) {
      pending = pending + 1;
    }
  }
  next_states[state_base + HEADER_PRESENT] = select(present, 0, present == 2147483647);
  next_states[state_base + HEADER_PENDING] = pending;
  next_states[state_base + HEADER_COMPLETE] = select(0, 1, pending == 0);
  if (pending == 0 && next_states[state_base + HEADER_TERMINAL] == 0) {
    next_states[state_base + HEADER_DEPTH] = next_states[state_base + HEADER_DEPTH] + 1;
    next_states[state_base + HEADER_TURN] = 1 - turn;
  }
}

@compute @workgroup_size(MATERIALIZE_WORKGROUP_SIZE)
fn materialize_selected(@builtin(global_invocation_id) id: vec3<u32>) {
  let output_index = id.x;
  if (output_index >= min(params.selected_count, atomicLoad(&counters[1]))) {
    return;
  }
  let candidate_index = u32(selected[output_index]);
  let candidate_base = candidate_index * params.candidate_stride;
  let parent_index = u32(max(0, candidates[candidate_base + CANDIDATE_PARENT]));
  let parent_base = parent_index * params.state_stride;
  let output_base = output_index * params.state_stride;
  for (var word = 0u; word < params.state_stride; word = word + 1u) {
    next_states[output_base + word] = parent_states[parent_base + word];
  }

  next_states[output_base + HEADER_PARENT] = i32(parent_index);
  next_states[output_base + HEADER_ROOT] = candidates[candidate_base + CANDIDATE_ROOT];
  next_states[output_base + HEADER_SCORE] =
    candidates[candidate_base + CANDIDATE_SCORE] - candidates[candidate_base + CANDIDATE_POLICY_PRIOR];
  next_states[output_base + HEADER_LAST_NEURAL] = candidates[candidate_base + CANDIDATE_LAST_NEURAL];
  next_states[output_base + HEADER_HASH_LOW] = candidates[candidate_base + CANDIDATE_HASH_LOW];
  next_states[output_base + HEADER_HASH_HIGH] = candidates[candidate_base + CANDIDATE_HASH_HIGH];
  let carry = candidates[candidate_base + CANDIDATE_CARRY] != 0;
  let status = candidates[candidate_base + CANDIDATE_STATUS];
  if (status == 2 || status == 4) {
    next_states[output_base + HEADER_TERMINAL] = 1;
  }
  if (!carry && (status == 3 || status == 4)) {
    let turn = next_states[output_base + HEADER_TURN];
    if (turn == 0) {
      next_states[output_base + HEADER_NEXT_WHITE] = next_states[output_base + HEADER_NEXT_WHITE] + 1;
    } else {
      next_states[output_base + HEADER_NEXT_BLACK] = next_states[output_base + HEADER_NEXT_BLACK] - 1;
    }
  }

  let plan_length = u32(max(0, next_states[output_base + HEADER_PLAN_LENGTH]));
  if (!carry && plan_length < params.max_plan_moves) {
    let plan_base = output_base + params.plan_offset + plan_length * 8u;
    for (var word = 0u; word < 8u; word = word + 1u) {
      next_states[plan_base + word] = candidates[candidate_base + CANDIDATE_MOVE + word];
    }
    next_states[output_base + HEADER_PLAN_LENGTH] = i32(plan_length + 1u);
  }

  var board_count = u32(max(0, next_states[output_base + HEADER_BOARD_COUNT]));
  let delta_count = min(2u, u32(max(0, candidates[candidate_base + CANDIDATE_DELTA_COUNT])));
  let delta_base = candidate_index * params.delta_stride;
  for (var delta_index = 0u; delta_index < delta_count && board_count < params.max_boards; delta_index = delta_index + 1u) {
    let source = delta_base + delta_index * BOARD_STRIDE;
    let timeline = deltas[source + BOARD_TIMELINE];
    for (var existing = 0u; existing < board_count; existing = existing + 1u) {
      let existing_base = board_base(output_base, existing);
      if (next_states[existing_base + BOARD_TIMELINE] == timeline) {
        next_states[existing_base + BOARD_LATEST] = 0;
      }
    }
    let target_base = board_base(output_base, board_count);
    for (var word = 0u; word < BOARD_STRIDE; word = word + 1u) {
      next_states[target_base + word] = deltas[source + word];
    }
    next_states[target_base + BOARD_LATEST] = 1;
    board_count = board_count + 1u;
  }
  next_states[output_base + HEADER_BOARD_COUNT] = i32(board_count);
  let old_depth = next_states[output_base + HEADER_DEPTH];
  if (!carry) {
    recompute_turn_status(output_base);
  }
  let new_depth = next_states[output_base + HEADER_DEPTH];
  if (!carry && new_depth > old_depth && old_depth >= 0 && u32(old_depth) < 16u) {
    next_states[output_base + params.ancestry_offset + u32(old_depth)] = candidates[candidate_base + CANDIDATE_NODE_ID];
  }

  let summary = output_index * SUMMARY_STRIDE;
  summaries[summary + 0u] = next_states[output_base + HEADER_ROOT];
  summaries[summary + 1u] = next_states[output_base + HEADER_SCORE];
  summaries[summary + 2u] = next_states[output_base + HEADER_DEPTH];
  summaries[summary + 3u] = next_states[output_base + HEADER_TURN];
  summaries[summary + 4u] = next_states[output_base + HEADER_PLAN_LENGTH];
  summaries[summary + 5u] = next_states[output_base + HEADER_COMPLETE];
  summaries[summary + 6u] = next_states[output_base + HEADER_TERMINAL];
  summaries[summary + 7u] = next_states[output_base + HEADER_HASH_LOW];
  summaries[summary + 8u] = next_states[output_base + HEADER_HASH_HIGH];
  summaries[summary + 9u] = next_states[output_base + HEADER_PRESENT];
  summaries[summary + 10u] = next_states[output_base + HEADER_PENDING];
  summaries[summary + 11u] = i32(output_index);
}

struct ReduceParams {
  state_count: u32,
  state_stride: u32,
  ancestry_offset: u32,
  target_depth: u32,
  level: u32,
  read_from_summaries: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> reduce_states: array<i32>;
@group(0) @binding(1) var<storage, read_write> reduce_summaries: array<i32>;
@group(0) @binding(2) var<uniform> reduce_params: ReduceParams;

fn reduction_value(state: u32) -> i32 {
  if (reduce_params.read_from_summaries != 0u) {
    return reduce_summaries[state * SUMMARY_STRIDE + 1u];
  }
  return reduce_states[state * reduce_params.state_stride + HEADER_SCORE];
}

fn reduction_state_valid(state: u32) -> bool {
  let base = state * reduce_params.state_stride;
  return reduce_states[base + HEADER_DEPTH] >= i32(reduce_params.target_depth)
    && reduce_states[base + reduce_params.ancestry_offset + reduce_params.level] != 0;
}

@compute @workgroup_size(64)
fn minimax_reduce_stage(@builtin(global_invocation_id) id: vec3<u32>) {
  let state = id.x;
  if (state >= reduce_params.state_count || reduce_params.level == 0u || !reduction_state_valid(state)) {
    return;
  }
  let base = state * reduce_params.state_stride;
  let parent = reduce_states[base + reduce_params.ancestry_offset + reduce_params.level - 1u];
  let maximize = (reduce_params.level & 1u) == 0u;
  var value = select(2147483647, -2147483647, maximize);
  var found = false;
  for (var peer = 0u; peer < reduce_params.state_count; peer = peer + 1u) {
    if (!reduction_state_valid(peer)) { continue; }
    let peer_base = peer * reduce_params.state_stride;
    if (reduce_states[peer_base + reduce_params.ancestry_offset + reduce_params.level - 1u] != parent) {
      continue;
    }
    let peer_value = reduction_value(peer);
    value = select(min(value, peer_value), max(value, peer_value), maximize);
    found = true;
  }
  if (!found) { return; }
  if (reduce_params.read_from_summaries != 0u) {
    reduce_states[base + HEADER_SCORE] = value;
  } else {
    reduce_summaries[state * SUMMARY_STRIDE + 1u] = value;
  }
}

@compute @workgroup_size(64)
fn minimax_copy_scores(@builtin(global_invocation_id) id: vec3<u32>) {
  let state = id.x;
  if (state >= reduce_params.state_count || !reduction_state_valid(state)) { return; }
  let base = state * reduce_params.state_stride;
  reduce_states[base + HEADER_SCORE] = reduce_summaries[state * SUMMARY_STRIDE + 1u];
}
