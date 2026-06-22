struct Params {
  sample_count: u32,
  input_size: u32,
  output_size: u32,
  learning_rate: f32,
  weight_decay: f32,
  momentum: f32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> velocity: array<f32>;

var<workgroup> feature_tile: array<f32, 256>;
var<workgroup> delta_tile: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn apply_layer_naive(@builtin(global_invocation_id) id: vec3<u32>) {
  let input_index = id.x;
  let output_index = id.y;
  if (input_index > params.input_size || output_index >= params.output_size) {
    return;
  }
  var gradient = 0.0;
  for (var sample = 0u; sample < params.sample_count; sample = sample + 1u) {
    let delta = deltas[sample * params.output_size + output_index];
    gradient = gradient + delta * select(1.0, features[sample * params.input_size + input_index], input_index < params.input_size);
  }
  let weight_index = output_index * (params.input_size + 1u) + input_index;
  let decay = select(params.weight_decay * weights[weight_index], 0.0, input_index == params.input_size);
  let update = (2.0 * gradient / f32(params.sample_count)) + decay;
  velocity[weight_index] = params.momentum * velocity[weight_index] + (1.0 - params.momentum) * update;
  weights[weight_index] = weights[weight_index] - params.learning_rate * velocity[weight_index];
}

@compute @workgroup_size(16, 16)
fn apply_layer(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>
) {
  let input_index = id.x;
  let output_index = id.y;
  var gradient = 0.0;
  for (var base = 0u; base < params.sample_count; base = base + 16u) {
    let feature_sample = base + local.y;
    let delta_sample = base + local.x;
    feature_tile[local.y * 16u + local.x] = select(
      0.0,
      select(1.0, features[feature_sample * params.input_size + input_index], input_index < params.input_size),
      feature_sample < params.sample_count && input_index <= params.input_size
    );
    delta_tile[local.x * 16u + local.y] = select(
      0.0,
      deltas[delta_sample * params.output_size + output_index],
      delta_sample < params.sample_count && output_index < params.output_size
    );
    workgroupBarrier();
    for (var offset = 0u; offset < 16u; offset = offset + 1u) {
      gradient = gradient
        + feature_tile[offset * 16u + local.x]
        * delta_tile[offset * 16u + local.y];
    }
    workgroupBarrier();
  }
  if (input_index <= params.input_size && output_index < params.output_size) {
    let weight_index = output_index * (params.input_size + 1u) + input_index;
    let decay = select(params.weight_decay * weights[weight_index], 0.0, input_index == params.input_size);
    let update = (2.0 * gradient / f32(params.sample_count)) + decay;
    velocity[weight_index] = params.momentum * velocity[weight_index] + (1.0 - params.momentum) * update;
    weights[weight_index] = weights[weight_index] - params.learning_rate * velocity[weight_index];
  }
}
