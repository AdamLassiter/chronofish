struct Params {
  sample_count: u32,
  input_size: u32,
  output_size: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> input_tile: array<f32, 256>;
var<workgroup> weight_tile: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn forward_layer_naive(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  let unit = id.y;
  if (sample >= params.sample_count || unit >= params.output_size) {
    return;
  }
  let row = unit * (params.input_size + 1u);
  var sum = weights[row + params.input_size];
  let input_base = sample * params.input_size;
  for (var input_index = 0u; input_index < params.input_size; input_index = input_index + 1u) {
    sum = sum + input_values[input_base + input_index] * weights[row + input_index];
  }
  output_values[sample * params.output_size + unit] = max(sum, 0.0);
}

@compute @workgroup_size(16, 16)
fn forward_layer(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>
) {
  let sample = id.x;
  let unit = id.y;
  let row = unit * (params.input_size + 1u);
  var sum = 0.0;
  let input_base = sample * params.input_size;
  for (var base = 0u; base < params.input_size; base = base + 16u) {
    let input_index = base + local.y;
    let weight_index = base + local.x;
    input_tile[local.x * 16u + local.y] = select(
      0.0,
      input_values[input_base + input_index],
      sample < params.sample_count && input_index < params.input_size
    );
    weight_tile[local.y * 16u + local.x] = select(
      0.0,
      weights[row + weight_index],
      unit < params.output_size && weight_index < params.input_size
    );
    workgroupBarrier();
    for (var offset = 0u; offset < 16u; offset = offset + 1u) {
      sum = sum
        + input_tile[local.x * 16u + offset]
        * weight_tile[local.y * 16u + offset];
    }
    workgroupBarrier();
  }
  if (sample < params.sample_count && unit < params.output_size) {
    output_values[sample * params.output_size + unit] = max(sum + weights[row + params.input_size], 0.0);
  }
}
