struct Params {
  sample_count: u32,
  input_size: u32,
  bucket_count: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read> sample_weights: array<f32>;
@group(0) @binding(3) var<storage, read> policy_weights: array<f32>;
@group(0) @binding(4) var<storage, read> sample_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> partial_sums: array<vec2<f32>>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> reductions: array<vec2<f32>, 64>;

fn logit(dataset_sample: u32, bucket: u32) -> f32 {
  let row = bucket * (params.input_size + 1u);
  var result = policy_weights[row + params.input_size];
  for (var input = 0u; input < params.input_size; input = input + 1u) {
    result = result + features[dataset_sample * params.input_size + input] * policy_weights[row + input];
  }
  return result;
}

@compute @workgroup_size(64)
fn reduce_policy_loss(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let sample = global_id.x;
  var contribution = vec2<f32>(0.0, 0.0);
  if (sample < params.sample_count) {
    let dataset_sample = sample_indices[sample];
    let weight = max(sample_weights[dataset_sample], 0.0);
    var max_logit = logit(dataset_sample, 0u);
    for (var bucket = 1u; bucket < params.bucket_count; bucket = bucket + 1u) {
      max_logit = max(max_logit, logit(dataset_sample, bucket));
    }
    var denominator = 0.0;
    for (var bucket = 0u; bucket < params.bucket_count; bucket = bucket + 1u) {
      denominator = denominator + exp(logit(dataset_sample, bucket) - max_logit);
    }
    let target_logit = logit(dataset_sample, targets[dataset_sample]);
    let loss = max_logit + log(max(denominator, 0.000001)) - target_logit;
    contribution = vec2<f32>(weight * loss, weight);
  }
  reductions[local_id.x] = contribution;
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (local_id.x < stride) {
      reductions[local_id.x] = reductions[local_id.x] + reductions[local_id.x + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  if (local_id.x == 0u) {
    partial_sums[workgroup_id.x] = reductions[0];
  }
}
