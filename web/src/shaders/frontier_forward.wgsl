struct LayerParams {
  sample_count: u32,
  input_size: u32,
  output_size: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<uniform> layer_params: LayerParams;
@group(0) @binding(4) var<storage, read> active_states: array<u32>;

@compute @workgroup_size(16, 16)
fn forward_layer_masked(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  let unit = id.y;
  if (sample >= layer_params.sample_count || unit >= layer_params.output_size || active_states[sample] == 0u) {
    return;
  }

  let row = unit * (layer_params.input_size + 1u);
  var sum = weights[row + layer_params.input_size];
  let input_base = sample * layer_params.input_size;
  for (var input_index = 0u; input_index < layer_params.input_size; input_index = input_index + 1u) {
    sum = sum + input_values[input_base + input_index] * weights[row + input_index];
  }
  output_values[sample * layer_params.output_size + unit] = max(sum, 0.0);
}

@compute @workgroup_size(64)
fn forward_output_masked_linear(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  if (sample >= layer_params.sample_count || active_states[sample] == 0u) {
    return;
  }

  var sum = weights[layer_params.input_size];
  let input_base = sample * layer_params.input_size;
  for (var input_index = 0u; input_index < layer_params.input_size; input_index = input_index + 1u) {
    sum = sum + input_values[input_base + input_index] * weights[input_index];
  }
  output_values[sample] = sum;
}

@compute @workgroup_size(64)
fn forward_output_masked(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  if (sample >= layer_params.sample_count || active_states[sample] == 0u) {
    return;
  }

  var sum = weights[layer_params.input_size];
  let input_base = sample * layer_params.input_size;
  for (var input_index = 0u; input_index < layer_params.input_size; input_index = input_index + 1u) {
    sum = sum + input_values[input_base + input_index] * weights[input_index];
  }
  output_values[sample] = tanh(sum);
}
