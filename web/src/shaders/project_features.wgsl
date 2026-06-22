struct Params {
  sample_count: u32,
  input_size: u32,
  projection_size: u32,
  seed: u32,
  output_offset: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> feature_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> feature_indices: array<u32>;
@group(0) @binding(2) var<storage, read> feature_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> projected_features: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn projection_hash(raw_index: u32, projection_index: u32, seed: u32) -> u32 {
  var hash = seed ^ raw_index;
  hash = hash * 16777619u;
  hash = hash ^ projection_index;
  hash = hash * 16777619u;
  hash = hash ^ (hash >> 16u);
  return hash;
}

@compute @workgroup_size(16, 16)
fn project_features(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  let projection = id.y;
  if (sample >= params.sample_count || projection >= params.projection_size) {
    return;
  }

  let sparse_start = feature_offsets[sample];
  let sparse_end = feature_offsets[sample + 1u];
  let active_count = sparse_end - sparse_start;
  var sum = 0.0;
  if (active_count > 0u) {
    let scale = sqrt(f32(active_count));
    for (var sparse_index = sparse_start; sparse_index < sparse_end; sparse_index = sparse_index + 1u) {
      let feature_index = feature_indices[sparse_index];
      let value = feature_values[sparse_index];
      let sign = select(-1.0, 1.0, (projection_hash(feature_index, projection, params.seed) & 1u) == 0u);
      sum = sum + value * sign / scale;
    }
  }

  projected_features[(params.output_offset + sample) * params.projection_size + projection] = sum;
}
