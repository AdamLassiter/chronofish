struct Params {
  sample_count: u32,
  input_size: u32,
  projection_size: u32,
  seed: u32,
};

@group(0) @binding(0) var<storage, read> raw_features: array<f32>;
@group(0) @binding(1) var<storage, read_write> projected_features: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

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

  let raw_base = sample * params.input_size;
  var active_count = 0u;
  for (var feature_index = 0u; feature_index < params.input_size; feature_index = feature_index + 1u) {
    if (raw_features[raw_base + feature_index] != 0.0) {
      active_count = active_count + 1u;
    }
  }

  var sum = 0.0;
  if (active_count > 0u) {
    let scale = sqrt(f32(active_count));
    for (var feature_index = 0u; feature_index < params.input_size; feature_index = feature_index + 1u) {
      let value = raw_features[raw_base + feature_index];
      if (value != 0.0) {
        let sign = select(-1.0, 1.0, (projection_hash(feature_index, projection, params.seed) & 1u) == 0u);
        sum = sum + value * sign / scale;
      }
    }
  }

  projected_features[sample * params.projection_size + projection] = sum;
}