struct Params {
  batch_count: u32,
  total_weight: f32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> predictions: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<f32>;
@group(0) @binding(2) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> batch_indices: array<u32>;
@group(0) @binding(5) var<storage, read> label_weights: array<f32>;

@compute @workgroup_size(64)
fn output_delta(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  if (sample >= params.batch_count) {
    return;
  }
  let dataset_sample = batch_indices[sample];
  let normalization = f32(params.batch_count) / max(params.total_weight, 0.000001);
  deltas[sample] = (predictions[sample] - labels[dataset_sample])
    * label_weights[dataset_sample]
    * normalization;
}
