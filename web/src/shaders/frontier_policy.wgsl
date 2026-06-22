struct Params {
  candidate_count: u32,
  candidate_stride: u32,
  input_size: u32,
  policy_scale: f32,
};

@group(0) @binding(0) var<storage, read_write> candidates: array<i32>;
@group(0) @binding(1) var<storage, read> hidden_features: array<f32>;
@group(0) @binding(2) var<storage, read> policy_weights: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const CANDIDATE_PARENT: u32 = 0u;
const CANDIDATE_SCORE: u32 = 2u;
const CANDIDATE_STATUS: u32 = 5u;
const CANDIDATE_MOVE: u32 = 8u;
const CANDIDATE_CARRY: u32 = 19u;
const CANDIDATE_POLICY_PRIOR: u32 = 21u;
const POLICY_BUCKETS: u32 = 257u;

fn hash_value(hash: u32, value: i32) -> u32 {
  var result = hash;
  let bits = bitcast<u32>(value);
  for (var shift = 0u; shift < 32u; shift = shift + 8u) {
    result = (result ^ ((bits >> shift) & 255u)) * 16777619u;
  }
  return result;
}

fn policy_bucket(base: u32) -> u32 {
  let from_timeline = candidates[base + CANDIDATE_MOVE];
  let from_time = candidates[base + CANDIDATE_MOVE + 1u];
  let from_x = candidates[base + CANDIDATE_MOVE + 2u];
  let from_y = candidates[base + CANDIDATE_MOVE + 3u];
  var hash = 2166136261u;
  hash = hash_value(hash, candidates[base + CANDIDATE_MOVE + 4u] - from_timeline);
  hash = hash_value(hash, candidates[base + CANDIDATE_MOVE + 5u] - from_time);
  hash = hash_value(hash, candidates[base + CANDIDATE_MOVE + 6u] - from_x);
  hash = hash_value(hash, candidates[base + CANDIDATE_MOVE + 7u] - from_y);
  hash = hash_value(hash, from_x);
  hash = hash_value(hash, from_y);
  return hash % POLICY_BUCKETS;
}

@compute @workgroup_size(64)
fn apply_policy_prior(@builtin(global_invocation_id) id: vec3<u32>) {
  let candidate = id.x;
  if (candidate >= params.candidate_count) {
    return;
  }
  let base = candidate * params.candidate_stride;
  if (candidates[base + CANDIDATE_STATUS] == 0 || candidates[base + CANDIDATE_CARRY] != 0) {
    return;
  }
  let parent = u32(max(0, candidates[base + CANDIDATE_PARENT]));
  let bucket = policy_bucket(base);
  let row = bucket * (params.input_size + 1u);
  var logit = policy_weights[row + params.input_size];
  for (var input = 0u; input < params.input_size; input = input + 1u) {
    logit = logit + hidden_features[parent * params.input_size + input] * policy_weights[row + input];
  }
  logit = clamp(logit, -4.0, 4.0);
  let prior = i32(round(logit * params.policy_scale));
  candidates[base + CANDIDATE_POLICY_PRIOR] = prior;
  candidates[base + CANDIDATE_SCORE] =
    candidates[base + CANDIDATE_SCORE] + prior;
}
