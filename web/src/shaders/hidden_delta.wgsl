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

var<workgroup> delta_tile: array<f32, 256>;
var<workgroup> weight_tile: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn hidden_delta_naive(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  let unit = id.y;
  if (sample >= params.sample_count || unit >= params.current_size) {
    return;
  }
  let activation = activations[sample * params.current_size + unit];
  var sum = 0.0;
  for (var next = 0u; next < params.next_size; next = next + 1u) {
    sum = sum + next_deltas[sample * params.next_size + next]
      * next_weights[next * (params.current_size + 1u) + unit];
  }
  deltas[sample * params.current_size + unit] = select(0.0, sum, activation > 0.0);
}

@compute @workgroup_size(16, 16)
fn hidden_delta(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>
) {
  let sample = id.x;
  let unit = id.y;
  var sum = 0.0;
  for (var base = 0u; base < params.next_size; base = base + 16u) {
    let delta_index = base + local.y;
    let weight_index = base + local.x;
    delta_tile[local.x * 16u + local.y] = select(
      0.0,
      next_deltas[sample * params.next_size + delta_index],
      sample < params.sample_count && delta_index < params.next_size
    );
    weight_tile[local.y * 16u + local.x] = select(
      0.0,
      next_weights[weight_index * (params.current_size + 1u) + unit],
      unit < params.current_size && weight_index < params.next_size
    );
    workgroupBarrier();
    for (var offset = 0u; offset < 16u; offset = offset + 1u) {
      sum = sum
        + delta_tile[local.x * 16u + offset]
        * weight_tile[local.y * 16u + offset];
    }
    workgroupBarrier();
  }
  if (sample < params.sample_count && unit < params.current_size) {
    let activation = activations[sample * params.current_size + unit];
    deltas[sample * params.current_size + unit] = select(0.0, sum, activation > 0.0);
  }
}
