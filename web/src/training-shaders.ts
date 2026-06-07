export const PROJECT_FEATURES_SHADER = `
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
`;

export const FORWARD_LAYER_SHADER = `
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

@compute @workgroup_size(16, 16)
fn forward_layer(@builtin(global_invocation_id) id: vec3<u32>) {
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
`;

export const FORWARD_INDEXED_LAYER_SHADER = `
struct Params {
  batch_count: u32,
  input_size: u32,
  output_size: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> batch_indices: array<u32>;

@compute @workgroup_size(16, 16)
fn forward_layer(@builtin(global_invocation_id) id: vec3<u32>) {
  let batch_sample = id.x;
  let unit = id.y;
  if (batch_sample >= params.batch_count || unit >= params.output_size) {
    return;
  }

  let dataset_sample = batch_indices[batch_sample];
  let row = unit * (params.input_size + 1u);
  var sum = weights[row + params.input_size];
  let input_base = dataset_sample * params.input_size;
  for (var input_index = 0u; input_index < params.input_size; input_index = input_index + 1u) {
    sum = sum + input_values[input_base + input_index] * weights[row + input_index];
  }
  output_values[batch_sample * params.output_size + unit] = max(sum, 0.0);
}
`;

export const FORWARD_OUTPUT_SHADER = `
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
  predictions[sample] = sum;
}
`;

export const OUTPUT_DELTA_SHADER = `
struct Params {
  batch_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> predictions: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<f32>;
@group(0) @binding(2) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> batch_indices: array<u32>;
@group(0) @binding(5) var<storage, read> label_weights: array<f32>;

@compute @workgroup_size(64)
fn output_delta(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  if (sample >= params.batch_count) {
    return;
  }
  let dataset_sample = batch_indices[sample];
  deltas[sample] = (predictions[sample] - labels[dataset_sample]) * label_weights[dataset_sample];
}
`;

export const HIDDEN_DELTA_SHADER = `
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
`;

export const HIDDEN3_DELTA_SHADER = `
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
`;

export const APPLY_LAYER_SHADER = `
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
      gradient = gradient + delta * features[sample * params.input_size + input_index];
    }
  }

  let weight_index = output_index * (params.input_size + 1u) + input_index;
  let decay = select(params.weight_decay * weights[weight_index], 0.0, input_index == params.input_size);
  weights[weight_index] = weights[weight_index]
    - params.learning_rate * ((2.0 * gradient / f32(params.sample_count)) + decay);
}
`;

export const APPLY_INDEXED_LAYER_SHADER = `
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
`;

export const APPLY_OUTPUT_SHADER = `
struct Params {
  sample_count: u32,
  input_size: u32,
  _pad0: u32,
  learning_rate: f32,
  weight_decay: f32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};

@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

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
  weights[index] = weights[index] - params.learning_rate * ((2.0 * gradient / f32(params.sample_count)) + decay);
}
`;

export const POLICY_SHADER = `
struct Params {
  sample_count: u32,
  bucket_count: u32,
  learning_rate: f32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> targets: array<u32>;
@group(0) @binding(1) var<storage, read> logits_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits_out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn train_policy(@builtin(global_invocation_id) id: vec3<u32>) {
  let bucket = id.x;
  if (bucket >= params.bucket_count) {
    return;
  }

  var max_logit = logits_in[0];
  for (var index = 1u; index < params.bucket_count; index = index + 1u) {
    max_logit = max(max_logit, logits_in[index]);
  }
  var total = 0.0;
  for (var index = 0u; index < params.bucket_count; index = index + 1u) {
    total = total + exp(logits_in[index] - max_logit);
  }
  let probability = exp(logits_in[bucket] - max_logit) / total;

  var target_count = 0.0;
  for (var sample = 0u; sample < params.sample_count; sample = sample + 1u) {
    if (targets[sample] == bucket) {
      target_count = target_count + 1.0;
    }
  }

  let gradient = probability - target_count / f32(params.sample_count);
  logits_out[bucket] = logits_in[bucket] - params.learning_rate * gradient;
}
`;
