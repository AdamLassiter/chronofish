struct Params {
  sample_count: u32,
  input_size: u32,
  output_size: u32,
  learning_rate: f32,
  weight_decay: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> batch_indices: array<u32>;

@compute @workgroup_size(16, 16)
fn apply_layer(@builtin(global_invocation_id) id: vec3<u32>) {
  let input_index = id.x;
  let output_index = id.y;
  if (input_index > params.input_size || output_index >= params.output_size) {
    return;
  }

  var gradient = 0.0;
  for (var sample = 0u; sample < params.sample_count; sample = sample + 1u) {
    let delta = deltas[sample * params.output_size + output_index];
    if (input_index == params.input_size) {
      gradient = gradient + delta;
    } else {
      let dataset_sample = batch_indices[sample];
      gradient = gradient + delta * features[dataset_sample * params.input_size + input_index];
    }
  }

  let weight_index = output_index * (params.input_size + 1u) + input_index;
  let decay = select(params.weight_decay * weights[weight_index], 0.0, input_index == params.input_size);
  weights[weight_index] = weights[weight_index]
    - params.learning_rate * ((2.0 * gradient / f32(params.sample_count)) + decay);
}