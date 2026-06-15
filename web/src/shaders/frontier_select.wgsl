struct Params {
  candidate_count: u32,
  selected_limit: u32,
  per_parent_limit: u32,
  max_scan: u32,
  state_stride: u32,
  delta_stride: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> candidates: array<i32>;
@group(0) @binding(1) var<storage, read> parent_states: array<i32>;
@group(0) @binding(2) var<storage, read> deltas: array<i32>;
@group(0) @binding(3) var<storage, read_write> order: array<i32>;
@group(0) @binding(4) var<storage, read_write> selected: array<i32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<uniform> sort_stage: vec4<u32>;

override SELECT_WORKGROUP_SIZE: u32 = 256u;

const CANDIDATE_STRIDE: u32 = 24u;
const CANDIDATE_PARENT: u32 = 0u;
const CANDIDATE_SCORE: u32 = 2u;
const CANDIDATE_HASH_LOW: u32 = 6u;
const CANDIDATE_HASH_HIGH: u32 = 7u;
const CANDIDATE_MOVE: u32 = 8u;
const CANDIDATE_DELTA_COUNT: u32 = 16u;
const CANDIDATE_CARRY: u32 = 19u;
const HEADER_HASH_LOW: u32 = 9u;
const HEADER_HASH_HIGH: u32 = 10u;

fn mix32(value: u32) -> u32 {
  var result = value;
  result = result ^ (result >> 16u);
  result = result * 0x7feb352du;
  result = result ^ (result >> 15u);
  result = result * 0x846ca68bu;
  return result ^ (result >> 16u);
}

fn candidate_base(index: u32) -> u32 {
  return index * CANDIDATE_STRIDE;
}

fn better(left_index: i32, right_index: i32) -> bool {
  if (left_index < 0) {
    return false;
  }
  if (right_index < 0) {
    return true;
  }
  let left = candidate_base(u32(left_index));
  let right = candidate_base(u32(right_index));
  let left_score = candidates[left + CANDIDATE_SCORE];
  let right_score = candidates[right + CANDIDATE_SCORE];
  if (left_score != right_score) {
    return left_score > right_score;
  }
  for (var word = 0u; word < 8u; word = word + 1u) {
    let left_word = candidates[left + CANDIDATE_MOVE + word];
    let right_word = candidates[right + CANDIDATE_MOVE + word];
    if (left_word != right_word) {
      return left_word < right_word;
    }
  }
  return left_index < right_index;
}

@compute @workgroup_size(SELECT_WORKGROUP_SIZE)
fn hash_candidates(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= min(params.candidate_count, atomicLoad(&counters[0]))) {
    return;
  }
  let base = candidate_base(index);
  let parent = u32(max(0, candidates[base + CANDIDATE_PARENT]));
  let parent_base = parent * params.state_stride;
  var low = u32(parent_states[parent_base + HEADER_HASH_LOW]);
  var high = u32(parent_states[parent_base + HEADER_HASH_HIGH]);
  if (candidates[base + CANDIDATE_CARRY] != 0) {
    candidates[base + CANDIDATE_HASH_LOW] = i32(low);
    candidates[base + CANDIDATE_HASH_HIGH] = i32(high);
    return;
  }
  let delta_count = u32(max(0, candidates[base + CANDIDATE_DELTA_COUNT]));
  let delta_base = index * params.delta_stride;
  let board_stride = params.delta_stride / 2u;
  for (var word = 0u; word < delta_count * board_stride; word = word + 1u) {
    let value = u32(deltas[delta_base + word]);
    low = mix32(low ^ value ^ (word * 0x9e3779b9u));
    high = mix32(high + value + (word * 0x85ebca6bu));
  }
  for (var word = 0u; word < 8u; word = word + 1u) {
    let value = u32(candidates[base + CANDIDATE_MOVE + word]);
    low = mix32(low ^ value);
    high = mix32(high + value);
  }
  candidates[base + CANDIDATE_HASH_LOW] = i32(low);
  candidates[base + CANDIDATE_HASH_HIGH] = i32(high);
}

@compute @workgroup_size(SELECT_WORKGROUP_SIZE)
fn initialize_order(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index < params.candidate_count) {
    order[index] = select(-1, i32(index), index < atomicLoad(&counters[0]));
  }
}

// A host-encoded bitonic network invokes this entry point once per (k, j)
// stage. Each stage has its own uniform, so all stages share one command buffer.
@compute @workgroup_size(SELECT_WORKGROUP_SIZE)
fn bitonic_sort(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let k = sort_stage.x;
  let j = sort_stage.y;
  let partner = index ^ j;
  if (index >= params.candidate_count || partner >= params.candidate_count || partner <= index) {
    return;
  }
  let ascending = (index & k) != 0u;
  let left = order[index];
  let right = order[partner];
  let swap = select(better(right, left), better(left, right), ascending);
  if (swap) {
    order[index] = right;
    order[partner] = left;
  }
}

fn already_selected(hash_low: i32, hash_high: i32, selected_count: u32) -> bool {
  for (var index = 0u; index < selected_count; index = index + 1u) {
    let selected_index = u32(selected[index]);
    let base = candidate_base(selected_index);
    if (candidates[base + CANDIDATE_HASH_LOW] == hash_low && candidates[base + CANDIDATE_HASH_HIGH] == hash_high) {
      return true;
    }
  }
  return false;
}

fn parent_selected_count(parent: i32, selected_count: u32) -> u32 {
  var count = 0u;
  for (var index = 0u; index < selected_count; index = index + 1u) {
    let selected_index = u32(selected[index]);
    if (candidates[candidate_base(selected_index) + CANDIDATE_PARENT] == parent) {
      count = count + 1u;
    }
  }
  return count;
}

@compute @workgroup_size(1)
fn select_top_k(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x != 0u) {
    return;
  }
  var selected_count = 0u;
  let scan_limit = min(min(params.candidate_count, atomicLoad(&counters[0])), params.max_scan);
  for (var rank = 0u; rank < scan_limit && selected_count < params.selected_limit; rank = rank + 1u) {
    let candidate_index = order[rank];
    if (candidate_index < 0) {
      continue;
    }
    let base = candidate_base(u32(candidate_index));
    let parent = candidates[base + CANDIDATE_PARENT];
    let hash_low = candidates[base + CANDIDATE_HASH_LOW];
    let hash_high = candidates[base + CANDIDATE_HASH_HIGH];
    if (already_selected(hash_low, hash_high, selected_count)) {
      continue;
    }
    if (parent_selected_count(parent, selected_count) >= params.per_parent_limit) {
      continue;
    }
    selected[selected_count] = candidate_index;
    selected_count = selected_count + 1u;
  }
  atomicStore(&counters[1], selected_count);
}
