struct Params {
  candidate_count: u32,
  selected_limit: u32,
  per_parent_limit: u32,
  max_scan: u32,
  state_stride: u32,
  delta_stride: u32,
  cycle_index: u32,
  puct_scale: u32,
};

@group(0) @binding(0) var<storage, read_write> candidates: array<i32>;
@group(0) @binding(1) var<storage, read> parent_states: array<i32>;
@group(0) @binding(2) var<storage, read> deltas: array<i32>;
@group(0) @binding(3) var<storage, read_write> order: array<i32>;
@group(0) @binding(4) var<storage, read_write> selected: array<i32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<uniform> sort_stage: vec4<u32>;
@group(0) @binding(8) var<storage, read_write> eligibility: array<atomic<u32>>;

override SELECT_WORKGROUP_SIZE: u32 = 256u;

const CANDIDATE_STRIDE: u32 = 24u;
const CANDIDATE_PARENT: u32 = 0u;
const CANDIDATE_SCORE: u32 = 2u;
const CANDIDATE_DEPTH: u32 = 4u;
const CANDIDATE_HASH_LOW: u32 = 6u;
const CANDIDATE_HASH_HIGH: u32 = 7u;
const CANDIDATE_MOVE: u32 = 8u;
const CANDIDATE_DELTA_COUNT: u32 = 16u;
const CANDIDATE_CARRY: u32 = 19u;
const CANDIDATE_POLICY_PRIOR: u32 = 21u;
const CANDIDATE_TACTICAL_PRIORITY: u32 = 22u;
const CANDIDATE_INTENT: u32 = 23u;
const HEADER_DEPTH: u32 = 3u;
const HEADER_HASH_LOW: u32 = 9u;
const HEADER_HASH_HIGH: u32 = 10u;

const INTENT_ROYAL_CAPTURE: i32 = 0;
const INTENT_CHECK_ROYAL: i32 = 1;
const INTENT_HIGH_VALUE_CAPTURE: i32 = 2;
const INTENT_CREATE_TIMELINE: i32 = 3;
const INTENT_AFFECT_PRESENT: i32 = 4;
const INTENT_QUIET_TEMPORAL: i32 = 5;
const INTENT_QUIET: i32 = 6;

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

fn puct_score(base: u32) -> i32 {
  let parent = u32(max(0, candidates[base + CANDIDATE_PARENT]));
  let parent_base = parent * params.state_stride;
  let parent_depth = u32(max(0, parent_states[parent_base + HEADER_DEPTH]));
  let parent_visits = f32(max(1u, params.per_parent_limit) * max(1u, params.cycle_index + 1u) * max(1u, parent_depth + 1u));
  let child_visits = f32(max(1, candidates[base + CANDIDATE_DEPTH] + 1));
  let prior = f32(max(0, candidates[base + CANDIDATE_POLICY_PRIOR]));
  let exploration = prior * sqrt(parent_visits) / child_visits;
  return candidates[base + CANDIDATE_SCORE] + i32(round(exploration * f32(max(1u, params.puct_scale)) / 100.0));
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
  let left_tactical = candidates[left + CANDIDATE_TACTICAL_PRIORITY];
  let right_tactical = candidates[right + CANDIDATE_TACTICAL_PRIORITY];
  if (left_tactical != right_tactical) {
    return left_tactical > right_tactical;
  }
  let left_score = puct_score(left);
  let right_score = puct_score(right);
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
fn bucket_order(@builtin(global_invocation_id) id: vec3<u32>) {
  let bucket = id.x;
  if (bucket >= params.max_scan) {
    return;
  }
  let actual_count = min(params.candidate_count, atomicLoad(&counters[0]));
  var best_index = -1;
  for (var index = bucket; index < actual_count; index = index + params.max_scan) {
    let candidate_index = i32(index);
    if (better(candidate_index, best_index)) {
      best_index = candidate_index;
    }
  }
  order[bucket] = best_index;
}

// A host-encoded bitonic network invokes this entry point once per (k, j) stage
// over the bounded shortlist produced by bucket_order.
@compute @workgroup_size(SELECT_WORKGROUP_SIZE)
fn bitonic_sort(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let k = sort_stage.x;
  let j = sort_stage.y;
  let partner = index ^ j;
  if (index >= params.max_scan || partner >= params.max_scan || partner <= index) {
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

fn parent_intent_selected_count(parent: i32, intent: i32, selected_count: u32) -> u32 {
  var count = 0u;
  for (var index = 0u; index < selected_count; index = index + 1u) {
    let selected_index = u32(selected[index]);
    let base = candidate_base(selected_index);
    if (candidates[base + CANDIDATE_PARENT] == parent && candidates[base + CANDIDATE_INTENT] == intent) {
      count = count + 1u;
    }
  }
  return count;
}

fn intent_cap(intent: i32) -> u32 {
  if (intent == INTENT_ROYAL_CAPTURE) { return 32u; }
  if (intent == INTENT_CHECK_ROYAL) { return 32u; }
  if (intent == INTENT_HIGH_VALUE_CAPTURE) { return 16u; }
  if (intent == INTENT_CREATE_TIMELINE) { return 16u; }
  if (intent == INTENT_AFFECT_PRESENT) { return 16u; }
  if (intent == INTENT_QUIET_TEMPORAL) { return 4u; }
  if (intent == INTENT_QUIET) { return 4u; }
  return 4u;
}

@compute @workgroup_size(SELECT_WORKGROUP_SIZE)
fn mark_unique(@builtin(global_invocation_id) id: vec3<u32>) {
  let rank = id.x;
  if (rank >= params.max_scan) { return; }
  let candidate_index = order[rank];
  if (candidate_index < 0) {
    atomicStore(&eligibility[rank], 0u);
    return;
  }
  let base = candidate_base(u32(candidate_index));
  let hash_low = candidates[base + CANDIDATE_HASH_LOW];
  let hash_high = candidates[base + CANDIDATE_HASH_HIGH];
  for (var earlier = 0u; earlier < rank; earlier = earlier + 1u) {
    let earlier_index = order[earlier];
    if (earlier_index < 0) { continue; }
    let earlier_base = candidate_base(u32(earlier_index));
    if (candidates[earlier_base + CANDIDATE_HASH_LOW] == hash_low
      && candidates[earlier_base + CANDIDATE_HASH_HIGH] == hash_high) {
      atomicStore(&eligibility[rank], 0u);
      return;
    }
  }
  atomicStore(&eligibility[rank], 1u);
}

@compute @workgroup_size(SELECT_WORKGROUP_SIZE)
fn mark_parent_quota(@builtin(global_invocation_id) id: vec3<u32>) {
  let rank = id.x;
  if (rank >= params.max_scan || atomicLoad(&eligibility[rank]) == 0u) { return; }
  let candidate_index = order[rank];
  let base = candidate_base(u32(candidate_index));
  let parent = candidates[base + CANDIDATE_PARENT];
  let intent = candidates[base + CANDIDATE_INTENT];
  var count = 0u;
  var intent_count = 0u;
  for (var earlier = 0u; earlier < rank; earlier = earlier + 1u) {
    if (atomicLoad(&eligibility[earlier]) == 0u) { continue; }
    let earlier_index = order[earlier];
    if (earlier_index < 0) { continue; }
    let earlier_base = candidate_base(u32(earlier_index));
    if (candidates[earlier_base + CANDIDATE_PARENT] == parent) {
      count = count + 1u;
      if (candidates[earlier_base + CANDIDATE_INTENT] == intent) {
        intent_count = intent_count + 1u;
      }
    }
  }
  atomicStore(&eligibility[rank], select(2u, 0u, count >= params.per_parent_limit || intent_count >= intent_cap(intent)));
}

@compute @workgroup_size(SELECT_WORKGROUP_SIZE)
fn compact_selected(@builtin(global_invocation_id) id: vec3<u32>) {
  let rank = id.x;
  if (rank >= params.max_scan || atomicLoad(&eligibility[rank]) != 2u) { return; }
  var output = 0u;
  for (var earlier = 0u; earlier < rank; earlier = earlier + 1u) {
    if (atomicLoad(&eligibility[earlier]) == 2u) {
      output = output + 1u;
    }
  }
  if (output < params.selected_limit) {
    selected[output] = order[rank];
    atomicMax(&counters[1], output + 1u);
  }
}

fn try_select_candidate(candidate_index: i32, selected_count: ptr<function, u32>) {
  if (candidate_index < 0 || (*selected_count) >= params.selected_limit) {
    return;
  }
  let base = candidate_base(u32(candidate_index));
  let parent = candidates[base + CANDIDATE_PARENT];
  let hash_low = candidates[base + CANDIDATE_HASH_LOW];
  let hash_high = candidates[base + CANDIDATE_HASH_HIGH];
  if (already_selected(hash_low, hash_high, *selected_count)) {
    return;
  }
  if (parent_selected_count(parent, *selected_count) >= params.per_parent_limit) {
    return;
  }
  if (parent_intent_selected_count(parent, candidates[base + CANDIDATE_INTENT], *selected_count) >= intent_cap(candidates[base + CANDIDATE_INTENT])) {
    return;
  }
  selected[*selected_count] = candidate_index;
  *selected_count = (*selected_count) + 1u;
}

@compute @workgroup_size(1)
fn fill_selection_underflow(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x != 0u) {
    return;
  }
  var selected_count = atomicLoad(&counters[1]);
  if (selected_count >= params.selected_limit) { return; }
  let actual_count = min(params.candidate_count, atomicLoad(&counters[0]));
  for (var index = 0u; index < actual_count && selected_count < params.selected_limit; index = index + 1u) {
    try_select_candidate(i32(index), &selected_count);
  }
  atomicStore(&counters[1], selected_count);
}
