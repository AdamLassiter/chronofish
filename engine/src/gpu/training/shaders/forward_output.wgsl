struct Params {
  sample_count: u32,
  input_size: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> predictions: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn forward_output(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  if (sample >= params.sample_count) {
    return;
  }

  var sum = weights[params.input_size];
  let input_base = sample * params.input_size;
  for (var input_index = 0u; input_index < params.input_size; input_index = input_index + 1u) {
    sum = sum + input_values[input_base + input_index] * weights[input_index];
  }
  predictions[sample] = tanh(sum);
}
