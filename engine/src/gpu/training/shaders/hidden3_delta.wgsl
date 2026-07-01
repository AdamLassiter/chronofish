struct Params {
  sample_count: u32,
  current_size: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> output_deltas: array<f32>;
@group(0) @binding(2) var<storage, read> output_weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn hidden3_delta(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  let unit = id.y;
  if (sample >= params.sample_count || unit >= params.current_size) {
    return;
  }
  let activation = activations[sample * params.current_size + unit];
  deltas[sample * params.current_size + unit] = select(
    0.0,
    output_deltas[sample] * output_weights[unit],
    activation > 0.0
  );
}