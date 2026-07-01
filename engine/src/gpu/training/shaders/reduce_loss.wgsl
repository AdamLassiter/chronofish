struct Params {
  sample_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> predictions: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<f32>;
@group(0) @binding(2) var<storage, read> label_weights: array<f32>;
@group(0) @binding(3) var<storage, read> sample_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> partial_sums: array<vec2<f32>>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> reductions: array<vec2<f32>, 64>;

@compute @workgroup_size(64)
fn reduce_loss(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let sample = global_id.x;
  var contribution = vec2<f32>(0.0, 0.0);
  if (sample < params.sample_count) {
    let dataset_sample = sample_indices[sample];
    let weight = max(label_weights[dataset_sample], 0.0);
    let error = predictions[sample] - labels[dataset_sample];
    contribution = vec2<f32>(weight * error * error, weight);
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
