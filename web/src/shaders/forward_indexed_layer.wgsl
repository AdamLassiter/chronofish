struct Params {
  batch_count: u32,
  input_size: u32,
  output_size: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> batch_indices: array<u32>;

@compute @workgroup_size(16, 16)
fn forward_layer(@builtin(global_invocation_id) id: vec3<u32>) {
  let batch_sample = id.x;
  let unit = id.y;
  if (batch_sample >= params.batch_count || unit >= params.output_size) {
    return;
  }

  let dataset_sample = batch_indices[batch_sample];
  let row = unit * (params.input_size + 1u);
  var sum = weights[row + params.input_size];
  let input_base = dataset_sample * params.input_size;
  for (var input_index = 0u; input_index < params.input_size; input_index = input_index + 1u) {
    sum = sum + input_values[input_base + input_index] * weights[row + input_index];
  }
  output_values[batch_sample * params.output_size + unit] = max(sum, 0.0);
}