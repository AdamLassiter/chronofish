struct Params {
  sample_count: u32,
  bucket_count: u32,
  learning_rate: f32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> targets: array<u32>;
@group(0) @binding(1) var<storage, read> logits_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits_out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> label_weights: array<f32>;

@compute @workgroup_size(64)
fn train_policy(@builtin(global_invocation_id) id: vec3<u32>) {
  let bucket = id.x;
  if (bucket >= params.bucket_count) {
    return;
  }

  var max_logit = logits_in[0];
  for (var index = 1u; index < params.bucket_count; index = index + 1u) {
    max_logit = max(max_logit, logits_in[index]);
  }
  var total = 0.0;
  for (var index = 0u; index < params.bucket_count; index = index + 1u) {
    total = total + exp(logits_in[index] - max_logit);
  }
  let probability = exp(logits_in[bucket] - max_logit) / total;

  var target_weight = 0.0;
  var total_weight = 0.0;
  for (var sample = 0u; sample < params.sample_count; sample = sample + 1u) {
    let weight = max(0.0, label_weights[sample]);
    total_weight = total_weight + weight;
    if (targets[sample] == bucket) {
      target_weight = target_weight + weight;
    }
  }

  let normalized_target = select(0.0, target_weight / total_weight, total_weight > 0.0);
  let gradient = probability - normalized_target;
  logits_out[bucket] = logits_in[bucket] - params.learning_rate * gradient;
}
