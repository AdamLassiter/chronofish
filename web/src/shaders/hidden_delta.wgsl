struct Params {
  sample_count: u32,
  current_size: u32,
  next_size: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> next_deltas: array<f32>;
@group(0) @binding(2) var<storage, read> next_weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn hidden_delta(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  let unit = id.y;
  if (sample >= params.sample_count || unit >= params.current_size) {
    return;
  }
  let activation = activations[sample * params.current_size + unit];
  if (activation <= 0.0) {
    deltas[sample * params.current_size + unit] = 0.0;
    return;
  }

  var sum = 0.0;
  for (var next = 0u; next < params.next_size; next = next + 1u) {
    sum = sum + next_deltas[sample * params.next_size + next]
      * next_weights[next * (params.current_size + 1u) + unit];
  }
  deltas[sample * params.current_size + unit] = sum;
}