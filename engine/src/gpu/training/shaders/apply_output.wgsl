struct Params {
  sample_count: u32,
  input_size: u32,
  _pad0: u32,
  learning_rate: f32,
  weight_decay: f32,
  momentum: f32,
  _pad2: u32,
  _pad3: u32,
};

@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> velocity: array<f32>;

@compute @workgroup_size(64)
fn apply_output(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index > params.input_size) {
    return;
  }
  var gradient = 0.0;
  for (var sample = 0u; sample < params.sample_count; sample = sample + 1u) {
    if (index == params.input_size) {
      gradient = gradient + deltas[sample];
    } else {
      gradient = gradient + deltas[sample] * features[sample * params.input_size + index];
    }
  }
  let decay = select(params.weight_decay * weights[index], 0.0, index == params.input_size);
  let update = (2.0 * gradient / f32(params.sample_count)) + decay;
  velocity[index] = params.momentum * velocity[index] + (1.0 - params.momentum) * update;
  weights[index] = weights[index] - params.learning_rate * velocity[index];
}
