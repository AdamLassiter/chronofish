struct Params {
  batch_count: u32,
  input_size: u32,
  bucket_count: u32,
  _pad0: u32,
  total_weight: f32,
  learning_rate: f32,
  weight_decay: f32,
  momentum: f32,
};

@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read> sample_weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> policy_weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(6) var<storage, read> batch_indices: array<u32>;
@group(0) @binding(7) var<uniform> params: Params;
@group(0) @binding(8) var<storage, read_write> velocity: array<f32>;

var<workgroup> policy_feature_tile: array<f32, 256>;
var<workgroup> policy_weight_tile: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn forward_policy_naive(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  let bucket = id.y;
  if (sample >= params.batch_count || bucket >= params.bucket_count) {
    return;
  }
  let dataset_sample = batch_indices[sample];
  let row = bucket * (params.input_size + 1u);
  var sum = policy_weights[row + params.input_size];
  for (var input = 0u; input < params.input_size; input = input + 1u) {
    sum = sum + features[dataset_sample * params.input_size + input] * policy_weights[row + input];
  }
  logits[sample * params.bucket_count + bucket] = sum;
}

@compute @workgroup_size(16, 16)
fn forward_policy(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>
) {
  let sample = id.x;
  let bucket = id.y;
  let dataset_sample = select(0u, batch_indices[sample], sample < params.batch_count);
  let row = bucket * (params.input_size + 1u);
  var sum = 0.0;
  for (var base = 0u; base < params.input_size; base = base + 16u) {
    let feature_index = base + local.y;
    let weight_index = base + local.x;
    policy_feature_tile[local.x * 16u + local.y] = select(
      0.0,
      features[dataset_sample * params.input_size + feature_index],
      sample < params.batch_count && feature_index < params.input_size
    );
    policy_weight_tile[local.y * 16u + local.x] = select(
      0.0,
      policy_weights[row + weight_index],
      bucket < params.bucket_count && weight_index < params.input_size
    );
    workgroupBarrier();
    for (var offset = 0u; offset < 16u; offset = offset + 1u) {
      sum = sum
        + policy_feature_tile[local.x * 16u + offset]
        * policy_weight_tile[local.y * 16u + offset];
    }
    workgroupBarrier();
  }
  if (sample < params.batch_count && bucket < params.bucket_count) {
    logits[sample * params.bucket_count + bucket] = sum + policy_weights[row + params.input_size];
  }
}

@compute @workgroup_size(64)
fn policy_delta(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  if (sample >= params.batch_count) {
    return;
  }
  let dataset_sample = batch_indices[sample];
  let base = sample * params.bucket_count;
  var max_logit = logits[base];
  for (var bucket = 1u; bucket < params.bucket_count; bucket = bucket + 1u) {
    max_logit = max(max_logit, logits[base + bucket]);
  }
  var denominator = 0.0;
  for (var bucket = 0u; bucket < params.bucket_count; bucket = bucket + 1u) {
    denominator = denominator + exp(logits[base + bucket] - max_logit);
  }
  let normalization = sample_weights[dataset_sample] / max(params.total_weight, 0.000001);
  for (var bucket = 0u; bucket < params.bucket_count; bucket = bucket + 1u) {
    let probability = exp(logits[base + bucket] - max_logit) / denominator;
    let target_value = select(0.0, 1.0, bucket == targets[dataset_sample]);
    deltas[base + bucket] = (probability - target_value) * normalization;
  }
}

@compute @workgroup_size(16, 16)
fn apply_policy_naive(@builtin(global_invocation_id) id: vec3<u32>) {
  let input = id.x;
  let bucket = id.y;
  if (input > params.input_size || bucket >= params.bucket_count) {
    return;
  }
  var gradient = 0.0;
  for (var sample = 0u; sample < params.batch_count; sample = sample + 1u) {
    let dataset_sample = batch_indices[sample];
    let feature = select(1.0, features[dataset_sample * params.input_size + input], input < params.input_size);
    gradient = gradient + deltas[sample * params.bucket_count + bucket] * feature;
  }
  let weight = bucket * (params.input_size + 1u) + input;
  let decay = select(params.weight_decay * policy_weights[weight], 0.0, input == params.input_size);
  velocity[weight] = params.momentum * velocity[weight] + (1.0 - params.momentum) * (gradient + decay);
  policy_weights[weight] = policy_weights[weight] - params.learning_rate * velocity[weight];
}

@compute @workgroup_size(16, 16)
fn apply_policy(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>
) {
  let input = id.x;
  let bucket = id.y;
  var gradient = 0.0;
  for (var base = 0u; base < params.batch_count; base = base + 16u) {
    let feature_sample = base + local.y;
    let delta_sample = base + local.x;
    let dataset_sample = select(0u, batch_indices[feature_sample], feature_sample < params.batch_count);
    policy_feature_tile[local.y * 16u + local.x] = select(
      0.0,
      select(1.0, features[dataset_sample * params.input_size + input], input < params.input_size),
      feature_sample < params.batch_count && input <= params.input_size
    );
    policy_weight_tile[local.x * 16u + local.y] = select(
      0.0,
      deltas[delta_sample * params.bucket_count + bucket],
      delta_sample < params.batch_count && bucket < params.bucket_count
    );
    workgroupBarrier();
    for (var offset = 0u; offset < 16u; offset = offset + 1u) {
      gradient = gradient
        + policy_feature_tile[offset * 16u + local.x]
        * policy_weight_tile[offset * 16u + local.y];
    }
    workgroupBarrier();
  }
  if (input <= params.input_size && bucket < params.bucket_count) {
    let weight = bucket * (params.input_size + 1u) + input;
    let decay = select(params.weight_decay * policy_weights[weight], 0.0, input == params.input_size);
    velocity[weight] = params.momentum * velocity[weight] + (1.0 - params.momentum) * (gradient + decay);
    policy_weights[weight] = policy_weights[weight] - params.learning_rate * velocity[weight];
  }
}
