const POLICY_BUCKETS = 257;
const BUFFER_KEY = "value-policy-buffer";
const PROJECTION_SIZE = 2048;
const PROJECTION_SEED = 2166136261;
const MAX_PLAYOUT_PLIES = 10;
const HIDDEN_LAYERS = [1024, 512, 256];
const VALUE_EPOCHS_PER_SUBMIT = 64;
const POLICY_STEPS_PER_SUBMIT = 64;
const DEFAULT_BATCH_SIZE = 1024;
const DEFAULT_VALIDATION_SPLIT = 0.1;
const DEFAULT_PATIENCE = 12;
const DEFAULT_WEIGHT_DECAY = 0.00001;
const PROJECTION_CHUNK_SIZE = 256;
const LABEL_REQUEST_MIN_TIMEOUT_MS = 30000;
const LABEL_REQUEST_MAX_TIMEOUT_MS = 120000;
const LABEL_REQUEST_NODE_TIMEOUT_FACTOR_MS = 3;
const TRAINING_IO_TIMEOUT_MS = 15000;
let cachedGpuAdapter = null;
let cachedGpuDevice = null;
const pipelineCache = new Map();

const PROJECT_FEATURES_SHADER = `
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

const FORWARD_LAYER_SHADER = `
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

const FORWARD_INDEXED_LAYER_SHADER = `
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

const FORWARD_OUTPUT_SHADER = `
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

const OUTPUT_DELTA_SHADER = `
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

const HIDDEN_DELTA_SHADER = `
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

const HIDDEN3_DELTA_SHADER = `
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

const APPLY_LAYER_SHADER = `
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

const APPLY_INDEXED_LAYER_SHADER = `
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

const APPLY_OUTPUT_SHADER = `
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

const POLICY_SHADER = `
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

self.addEventListener("message", async (event) => {
  const { id, type = "train", game, config } = event.data;
  try {
    const metrics = createTrainingMetrics();
    const normalizedConfig = normalizeTrainingConfig(config);
    normalizedConfig.metrics = metrics;
    if (type === "validateLossLogs") {
      const validation = await timed(metrics, "lossLogValidation", () =>
        validateLossLogs(normalizedConfig, (message) => {
          self.postMessage({ id, ok: true, ...message });
        })
      );
      metrics.lossLogValidation = validation;
      self.postMessage({
        id,
        ok: true,
        type: "lossLogValidation",
        validation,
        metrics: metricsSummary(metrics)
      });
      return;
    }
    const [activeModel, loadedBuffer] = await timed(metrics, "load", () => Promise.all([
      fetchActiveModel(),
      loadReplayBuffer()
    ]));
    let buffer = loadedBuffer;
    const samples = await timed(metrics, "collect", () => collectTrainingSamples(game, normalizedConfig, activeModel, (message) => {
      self.postMessage({ id, ok: true, ...message });
    }, metrics));
    metrics.sampleCounts = labelSourceCounts(samples);
    buffer = appendReplaySamples(buffer, samples, normalizedConfig.maxBuffer);
    await timed(metrics, "saveReplay", () => saveReplayBuffer(buffer));
    const labelCounts = labelSourceCounts(buffer);
    self.postMessage({
      id,
      ok: true,
      gpuPhase: true,
      bufferSize: buffer.length,
      labelCounts,
      batchSize: normalizedConfig.batchSize,
      selfPlayWorkers: normalizedConfig.selfPlayWorkers,
      searchWorkers: normalizedConfig.searchWorkers,
      metrics: metricsSummary(metrics)
    });
    const model = await timed(metrics, "train", () => train(buffer, normalizedConfig, activeModel, (progressMetrics) => {
      self.postMessage({ id, ok: true, ...progressMetrics, metrics: metricsSummary(metrics) });
    }));
    model.metrics = metricsSummary(metrics);
    self.postMessage({
      id,
      ok: true,
      model,
      loss: model.trainingLoss,
      validationLoss: model.validationLoss,
      bestValidationLoss: model.bestValidationLoss,
      earlyStopReason: model.earlyStopReason,
      labelCounts: model.labelCounts,
      replaySize: model.replayBufferSize,
      nonZeroWeights: model.nonZeroWeights,
      metrics: model.metrics
    });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error.message });
  }
});

function normalizeTrainingConfig(config = {}) {
  return {
    ...config,
    labelMode: ["mixed", "search", "selfPlay", "distill"].includes(config.labelMode) ? config.labelMode : "mixed",
    runSeed: randomRunSeed(),
    samples: clampInteger(config.samples, 1, 1024, 64),
    selfPlayWorkers: clampInteger(config.selfPlayWorkers, 1, 8, 2),
    searchWorkers: clampInteger(config.searchWorkers, 1, 16, 2),
    explorationTemperature: clampNumber(config.explorationTemperature, 0, 2, 0.25),
    depth: clampInteger(config.depth, 1, 8, 5),
    nodes: clampInteger(config.nodes, 1, 131072, 16384),
    epochs: clampInteger(config.epochs, 1, 65536, 8192),
    maxBuffer: clampInteger(config.maxBuffer, 16, 16384, 4096),
    batchSize: clampInteger(config.batchSize, 16, 8192, DEFAULT_BATCH_SIZE),
    validationSplit: clampNumber(config.validationSplit, 0, 0.3, DEFAULT_VALIDATION_SPLIT),
    validationInterval: clampInteger(config.validationInterval, 16, 4096, 256),
    patience: clampInteger(config.patience, 1, 64, DEFAULT_PATIENCE),
    weightDecay: clampNumber(config.weightDecay, 0, 0.01, DEFAULT_WEIGHT_DECAY),
    lossLogReplay: clampInteger(config.lossLogReplay, 0, 32, 4)
  };
}

function createTrainingMetrics() {
  return {
    startedAt: performance.now(),
    phases: Object.create(null)
  };
}

async function timed(metrics, name, fn) {
  if (!metrics) {
    return fn();
  }
  const startedAt = performance.now();
  try {
    return await fn();
  } finally {
    const elapsed = performance.now() - startedAt;
    metrics.phases[name] = (metrics.phases[name] ?? 0) + elapsed;
  }
}

function metricsSummary(metrics) {
  if (!metrics) {
    return null;
  }
  const phases = {};
  for (const [name, ms] of Object.entries(metrics.phases)) {
    phases[name] = Math.round(ms);
  }
  const sampleRates = {};
  for (const [kind, count] of Object.entries(metrics.sampleCounts ?? {})) {
    const phaseName = `${kind}Labels`;
    const phaseMs = metrics.phases[phaseName] ?? metrics.phases.collect;
    if (phaseMs > 0) {
      sampleRates[kind] = Number((count / (phaseMs / 1000)).toFixed(2));
    }
  }
  if (metrics.searchPositionCount && metrics.phases.searchPositions > 0) {
    sampleRates.searchPositions = Number((metrics.searchPositionCount / (metrics.phases.searchPositions / 1000)).toFixed(2));
  }
  if (metrics.searchLabelCount && metrics.phases.searchLabels > 0) {
    sampleRates.searchLabels = Number((metrics.searchLabelCount / (metrics.phases.searchLabels / 1000)).toFixed(2));
  }
  return {
    totalMs: Math.round(performance.now() - metrics.startedAt),
    phases,
    sampleRates,
    lossLogValidation: metrics.lossLogValidation ?? null
  };
}

async function collectTrainingSamples(game, config, activeModel, progress, metrics = null) {
  const collectors = [];
  if (config.labelMode === "mixed" || config.labelMode === "search") {
    collectors.push(() => collectSearchSamples(game, config, progress));
  }
  if (config.labelMode === "mixed" || config.labelMode === "selfPlay") {
    collectors.push(() => timed(metrics, "outcomeLabels", () => collectOutcomeSamples(game, config, progress)));
  }
  if (config.labelMode === "mixed" || config.labelMode === "distill") {
    collectors.push(() => timed(metrics, "distillLabels", () => collectDistilledSamples(game, config, activeModel, progress)));
  }

  const collected = await Promise.allSettled(collectors.map((collector) => collector()));
  const results = collected
    .filter((result) => result.status === "fulfilled")
    .flatMap((result) => result.value);
  if (results.length > 0) {
    return results;
  }
  if (activeModel?.outputWeights?.length && config.labelMode !== "distill") {
    return collectDistilledSamples(game, config, activeModel, progress);
  }
  throw new Error("No GPU training labels were collected.");
}

async function collectSearchSamples(game, config, progress) {
  const target = config.labelMode === "mixed" ? Math.ceil(config.samples * 0.6) : config.samples;
  const positions = await timed(config.metrics, "searchPositions", () => collectGpuPositions(game, config, target, progress, "search"));
  if (config.metrics) {
    config.metrics.searchPositionCount = positions.length;
  }
  const workerCount = Math.min(positions.length, config.searchWorkers ?? 1);
  const samples = new Array(positions.length);
  let nextPosition = 0;
  let collected = 0;
  progress({ sampleCount: positions.length, labelWorkers: workerCount, labelKind: "search", labelPhase: "labels" });
  await timed(config.metrics, "searchLabels", () =>
    Promise.all(Array.from({ length: workerCount }, (_, workerIndex) => runSearchWorker(workerIndex)))
  );
  const filtered = samples.filter(Boolean);
  if (config.metrics) {
    config.metrics.searchLabelCount = filtered.length;
  }
  return filtered;

  async function runSearchWorker(workerIndex) {
    const ai = new Worker("./ai-worker.js", { type: "module" });
    try {
      while (nextPosition < positions.length) {
        const index = nextPosition;
        nextPosition += 1;
        const position = positions[index];
        samples[index] = await searchLabelSample(ai, position, index, workerIndex);
        collected += samples[index] ? 1 : 0;
        progress({ collected, sampleCount: positions.length, labelWorkers: workerCount, labelKind: "search", labelPhase: "labels" });
      }
    } finally {
      ai.terminate();
    }
  }

  async function searchLabelSample(ai, position, index, workerIndex) {
    try {
      const response = await requestWorker(ai, {
        type: "search",
        game: position.game,
        depth: config.depth,
        nodes: config.nodes,
        timeMs: workerSearchTimeMs(config),
        gpuMode: "full",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("search", index, searchSeed(position.sample, config.runSeed ^ workerIndex ^ 0x51a7_0001))
      }, workerRequestTimeout(config));
      const result = response.result;
      if (!result?.moves?.length) {
        return null;
      }
      return {
        ...position.sample,
        label: normalizeSearchScore(result.score ?? 0),
        policy: policyBucket(result.moves[0]),
        labelKind: "search",
        labelWeight: 1.0,
        pseudo: false
      };
    } catch {
      return null;
    }
  }
}

async function collectOutcomeSamples(game, config, progress) {
  const target = config.labelMode === "mixed" ? Math.floor(config.samples * 0.4) : config.samples;
  if (target <= 0) {
    return [];
  }
  const workerCount = Math.min(target, config.selfPlayWorkers ?? 1);
  const targets = splitWork(target, workerCount);
  let collected = 0;
  const report = (count) => {
    collected += count;
    progress({
      collected,
      sampleCount: target,
      labelWorkers: workerCount,
      labelKind: "outcome"
    });
  };
  progress({
    sampleCount: target,
    labelWorkers: workerCount,
    labelKind: "outcome"
  });
  const results = await Promise.all(targets.map((count, workerIndex) =>
    collectOutcomeRollout(game, config, count, workerIndex, report)
  ));
  return results.flat().slice(0, target);
}

async function collectOutcomeRollout(game, config, target, workerIndex, progress) {
  const ai = new Worker("./ai-worker.js", { type: "module" });
  const encoder = new Worker("./training-label-worker.js", { type: "module" });
  const samples = [];
  try {
    let current = cloneGame(game);
    current = await warmupSelfPlayPosition(ai, current, config, workerIndex);
    const maxPlies = Math.max(MAX_PLAYOUT_PLIES, target + workerIndex);
    for (let ply = 0; ply < maxPlies && samples.length < target; ply += 1) {
      const beforeTurn = current.turn;
      const encoded = await encodePosition(encoder, current);
      const response = await requestWorker(ai, {
        type: "search",
        game: current,
        depth: config.depth,
        nodes: config.nodes,
        timeMs: workerSearchTimeMs(config),
        gpuMode: "full",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("outcome", ply, searchSeed(encoded, config.runSeed ^ workerIndex ^ 0x0c70_0001))
      }, workerRequestTimeout(config));
      const result = response.result;
      const move = result?.moves?.[0];
      if (!move) {
        break;
      }
      samples.push({
        ...encoded,
        label: normalizeSearchScore(result.score ?? 0),
        policy: policyBucket(move),
        labelKind: "outcome",
        labelWeight: 1.25,
        outcomeTurn: beforeTurn,
        ply: ply + workerIndex * MAX_PLAYOUT_PLIES
      });
      progress(1);
      const previous = current;
      const applied = await requestWorker(ai, {
        type: "applyMove",
        game: current,
        move
      }, workerRequestTimeout(config));
      current = applied.game;
      const winner = royalCaptureWinner(previous, current, beforeTurn);
      if (winner) {
        return backfillOutcomeLabels(samples, winner);
      }
      const status = await requestWorker(ai, {
        type: "submitTurn",
        game: current
      }, workerRequestTimeout(config));
      if (status.status?.terminal && status.status.winner) {
        return backfillOutcomeLabels(samples, status.status.winner);
      }
      if (status.status?.complete) {
        current = { ...current, turn: status.status.nextTurn };
      }
    }
    return samples.map(({ outcomeTurn, ply, ...sample }) => sample);
  } catch {
    return samplesFromPartialOutcome(samples);
  } finally {
    ai.terminate();
    encoder.terminate();
  }
}

function splitWork(total, workers) {
  return Array.from({ length: workers }, (_, index) =>
    Math.floor(total / workers) + (index < total % workers ? 1 : 0)
  ).filter((count) => count > 0);
}

async function warmupSelfPlayPosition(ai, game, config, workerIndex) {
  let current = cloneGame(game);
  const warmupPlies = workerIndex === 0 ? 0 : 1 + (workerIndex % Math.max(1, MAX_PLAYOUT_PLIES - 1));
  for (let ply = 0; ply < warmupPlies; ply += 1) {
    try {
      const response = await requestWorker(ai, {
        type: "search",
        game: current,
        depth: Math.max(1, Math.min(2, config.depth)),
        nodes: Math.max(1, Math.min(1024, config.nodes)),
        timeMs: Math.min(5000, workerSearchTimeMs(config)),
        gpuMode: "full",
        partitionIndex: workerIndex,
        partitionCount: config.selfPlayWorkers ?? 1,
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("warmup", ply, config.runSeed ^ workerIndex ^ 0x0aa5_0001)
      }, workerRequestTimeout({ ...config, nodes: 1024 }));
      const move = response.result?.moves?.[0];
      if (!move) {
        break;
      }
      const applied = await requestWorker(ai, {
        type: "applyMove",
        game: current,
        move
      }, workerRequestTimeout(config));
      current = applied.game;
      const status = await requestWorker(ai, {
        type: "submitTurn",
        game: current
      }, workerRequestTimeout(config));
      if (status.status?.terminal) {
        break;
      }
      if (status.status?.complete) {
        current = { ...current, turn: status.status.nextTurn };
      }
    } catch {
      break;
    }
  }
  return current;
}

async function collectDistilledSamples(game, config, activeModel, progress) {
  if (!activeModel?.outputWeights?.length) {
    return [];
  }
  const positions = await collectSamples(game, config, true, (collected, sampleCount, labelWorkers) => {
    progress({ collected, sampleCount, labelWorkers, labelKind: "distilled" });
  });
  const labels = await predictValues(positions, activeModel);
  return positions.map((sample, index) => ({
    ...sample,
    label: labels[index],
    policy: null,
    labelKind: "distilled",
    labelWeight: 0.25,
    pseudo: true
  }));
}

async function collectGpuPositions(game, config, target, progress, labelKind) {
  if (target <= 0) {
    return [];
  }
  const workerCount = Math.min(target, Math.max(1, config.searchWorkers ?? 1));
  const positions = new Array(target);
  let nextJob = 0;
  let generated = 0;
  progress({ sampleCount: target, labelWorkers: workerCount, labelKind, labelPhase: "positions" });
  await Promise.all(Array.from({ length: workerCount }, (_, workerIndex) => runPositionWorker(workerIndex)));
  return positions.filter((position) => position?.sample?.features?.length);

  async function runPositionWorker(workerIndex) {
    const ai = new Worker("./ai-worker.js", { type: "module" });
    const encoder = new Worker("./training-label-worker.js", { type: "module" });
    const local = [];
    try {
      while (nextJob < target) {
        const index = nextJob;
        nextJob += 1;
        const positionGame = await generatePositionGame(ai, game, config, index, workerIndex);
        local.push({ index, game: positionGame });
        generated += 1;
        progress({ collected: generated, sampleCount: target, labelWorkers: workerCount, labelKind, labelPhase: "positions" });
      }
      let samples = [];
      try {
        samples = await encodePositions(encoder, local.map((entry) => entry.game));
      } catch {
        samples = [];
      }
      for (let index = 0; index < local.length; index += 1) {
        const entry = local[index];
        if (!samples[index]?.features?.length) {
          continue;
        }
        positions[entry.index] = {
          game: entry.game,
          sample: samples[index]
        };
      }
    } finally {
      ai.terminate();
      encoder.terminate();
    }
  }
}

async function generatePositionGame(ai, game, config, index, workerIndex) {
  let current = cloneGame(game);
  const plies = samplePlies(index, false);
  for (let ply = 0; ply < plies; ply += 1) {
    try {
      const shallowConfig = { ...config, nodes: Math.max(1, Math.min(512, config.nodes)) };
      const response = await requestWorker(ai, {
        type: "search",
        game: current,
        depth: Math.max(1, Math.min(2, config.depth)),
        nodes: shallowConfig.nodes,
        timeMs: 3000,
        gpuMode: "full",
        temperature: config.explorationTemperature,
        randomSeed: sampleSeed("position", index * MAX_PLAYOUT_PLIES + ply, config.runSeed ^ workerIndex ^ 0x9051_0001)
      }, workerRequestTimeout(shallowConfig));
      const move = response.result?.moves?.[0];
      if (!move) {
        break;
      }
      const applied = await requestWorker(ai, { type: "applyMove", game: current, move }, workerRequestTimeout(config));
      current = applied.game;
      const status = await requestWorker(ai, { type: "submitTurn", game: current }, workerRequestTimeout(config));
      if (status.status?.terminal) {
        break;
      }
      if (status.status?.complete) {
        current = { ...current, turn: status.status.nextTurn };
      }
    } catch {
      break;
    }
  }
  return current;
}

function samplesFromPartialOutcome(samples) {
  return samples.map(({ outcomeTurn, ply, ...sample }) => sample);
}

function backfillOutcomeLabels(samples, winner) {
  const maxPly = samples.at(-1)?.ply ?? 0;
  return samples.map(({ outcomeTurn, ply, ...sample }) => ({
    ...sample,
    label: (outcomeTurn === winner ? 1 : -1) * Math.pow(0.96, maxPly - ply),
    labelKind: "outcome",
    labelWeight: 1.25
  }));
}

function royalCaptureWinner(before, after, mover) {
  const opponent = mover === "white" ? "black" : "white";
  return royalCount(after, opponent) < royalCount(before, opponent) ? mover : null;
}

function royalCount(game, color) {
  let count = 0;
  for (const timeline of game.timelines ?? []) {
    const board = latestBoard(timeline);
    for (const row of board?.board ?? []) {
      for (const piece of row ?? []) {
        if (piece?.color === color && ["king", "royalQueen"].includes(piece.type)) {
          count += 1;
        }
      }
    }
  }
  return count;
}

function latestBoard(timeline) {
  return (timeline.boards ?? []).reduce(
    (latest, board) => !latest || board.time > latest.time ? board : latest,
    null
  );
}

async function encodePosition(worker, game) {
  const response = await requestWorker(worker, {
    type: "sample",
    game,
    encodeOnly: true
  }, workerRequestTimeout({ nodes: 1 }));
  return response.sample;
}

async function encodePositions(worker, games) {
  if (!games.length) {
    return [];
  }
  const response = await requestWorker(worker, {
    type: "batchSample",
    games
  }, workerRequestTimeout({ nodes: games.length }));
  return response.samples ?? [];
}

async function collectSamples(game, config, encodeOnly, progress) {
  const jobs = Array.from({ length: config.samples }, (_, index) => ({
    game,
    index,
    seed: sampleSeed(JSON.stringify(game), index, encodeOnly ? 0xa11c_e000 : 0x5eed_1000),
    plies: samplePlies(index, encodeOnly)
  }));
  const workerCount = Math.min(
    jobs.length,
    Math.max(1, Math.min(config.labelWorkers ?? autoLabelWorkers(), 8))
  );
  progress(0, jobs.length, workerCount);

  const samples = new Array(jobs.length);
  let nextJob = 0;
  let collected = 0;

  await Promise.all(Array.from({ length: workerCount }, () => runLabelWorker()));
  return samples;

  async function runLabelWorker() {
    const worker = new Worker("./training-label-worker.js", { type: "module" });
    try {
      while (nextJob < jobs.length) {
        const job = jobs[nextJob];
        nextJob += 1;
        samples[job.index] = await labelSample(worker, job, config, encodeOnly);
        collected += 1;
        progress(collected, jobs.length, workerCount);
      }
    } finally {
      worker.terminate();
    }
  }
}

function labelSample(worker, job, config, encodeOnly) {
  const payload = {
    type: "sample",
    game: job.game,
    depth: config.depth,
    nodes: config.nodes,
    encodeOnly,
    seed: job.seed,
    plies: job.plies
  };
  return requestWorker(worker, {
    ...payload,
    timeMs: workerSearchTimeMs(payload)
  }).then((response) => response.sample);
}

function requestWorker(worker, payload, timeoutMs = workerRequestTimeout(payload)) {
  return new Promise((resolve, reject) => {
    const messageId = crypto.randomUUID();
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error("Position worker timed out."));
    }, timeoutMs);
    const handleMessage = (event) => {
      if (event.data.id !== messageId) {
        return;
      }
      cleanup();
      if (event.data.ok) {
        resolve(event.data);
      } else {
        reject(new Error(event.data.error));
      }
    };
    const handleError = (event) => {
      cleanup();
      reject(new Error(event.message || "Label worker failed."));
    };
    const cleanup = () => {
      clearTimeout(timeout);
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
      worker.removeEventListener("messageerror", handleError);
    };
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.addEventListener("messageerror", handleError);
    worker.postMessage({
      id: messageId,
      ...payload
    });
  });
}

function normalizeSearchScore(score) {
  return Math.max(-1, Math.min(1, score / 20000));
}

function policyBucket(move) {
  if (!move) {
    return null;
  }
  const values = [
    move.from?.timelineId ?? 0,
    move.from?.time ?? 0,
    move.from?.x ?? 0,
    move.from?.y ?? 0,
    move.to?.timelineId ?? 0,
    move.to?.time ?? 0,
    move.to?.x ?? 0,
    move.to?.y ?? 0
  ];
  let hash = 2166136261;
  for (const value of values) {
    hash ^= value & 0xff;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash % POLICY_BUCKETS;
}

function cloneGame(game) {
  return JSON.parse(JSON.stringify(game));
}

function labelSourceCounts(samples) {
  return samples.reduce((counts, sample) => {
    const key = sample.labelKind ?? (sample.pseudo ? "distilled" : "unknown");
    counts[key] = (counts[key] ?? 0) + 1;
    return counts;
  }, {});
}

function clampInteger(value, min, max, fallback) {
  return Math.round(clampNumber(value, min, max, fallback));
}

function clampNumber(value, min, max, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, number));
}

function randomRunSeed() {
  const values = new Uint32Array(1);
  crypto.getRandomValues(values);
  return values[0] >>> 0;
}

function workerRequestTimeout(payload) {
  const nodes = Math.max(1, Number(payload.nodes) || 1);
  return Math.min(
    LABEL_REQUEST_MAX_TIMEOUT_MS,
    Math.max(
      LABEL_REQUEST_MIN_TIMEOUT_MS,
      nodes * LABEL_REQUEST_NODE_TIMEOUT_FACTOR_MS
    )
  );
}

function workerSearchTimeMs(payload) {
  const timeout = workerRequestTimeout(payload);
  return Math.max(1000, timeout - 1000);
}

function withTimeout(promise, timeoutMs, message) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      (value) => {
        clearTimeout(timeout);
        resolve(value);
      },
      (error) => {
        clearTimeout(timeout);
        reject(error);
      }
    );
  });
}

function samplePlies(index, encodeOnly) {
  const stride = encodeOnly ? 2 : 1;
  return 1 + ((index * stride) % MAX_PLAYOUT_PLIES);
}

function sampleSeed(prefix, index, salt) {
  let hash = salt >>> 0;
  for (let offset = 0; offset < prefix.length; offset += 1) {
    hash ^= prefix.charCodeAt(offset);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  hash ^= index;
  hash = Math.imul(hash, 16777619) >>> 0;
  return hash >>> 0;
}

function searchSeed(value, salt) {
  let hash = salt >>> 0;
  const text = JSON.stringify(value ?? null);
  for (let offset = 0; offset < text.length; offset += 1) {
    hash ^= text.charCodeAt(offset);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

async function train(samples, config, activeModel, progress) {
  if (!globalThis.navigator?.gpu) {
    throw new Error("WebGPU is unavailable in this browser.");
  }
  if (!samples?.length) {
    throw new Error("No samples were collected.");
  }

  const device = await getGpuDevice();
  if (!device) {
    throw new Error("No WebGPU adapter is available.");
  }
  const value = await trainValue(device, samples, config, activeModel, progress);
  const policy = await trainPolicy(device, samples, config, activeModel);
  let model = encodeCompactModel({
    projectionSize: PROJECTION_SIZE,
    projectionSeed: PROJECTION_SEED,
    hiddenLayers: HIDDEN_LAYERS,
    hiddenWeights: value.hiddenWeights,
    outputWeights: value.weights,
    policyLogits: policy,
    scale: 1,
    bias: 0
  });
  if (activeModel?.bytes && byteArraysEqual(model, activeModel.bytes) && value.finalHiddenWeights && value.finalWeights) {
    const finalModel = encodeCompactModel({
      projectionSize: PROJECTION_SIZE,
      projectionSeed: PROJECTION_SEED,
      hiddenLayers: HIDDEN_LAYERS,
      hiddenWeights: value.finalHiddenWeights,
      outputWeights: value.finalWeights,
      policyLogits: policy,
      scale: 1,
      bias: 0
    });
    if (!byteArraysEqual(finalModel, activeModel.bytes)) {
      model = finalModel;
      value.earlyStopReason = value.earlyStopReason
        ? `${value.earlyStopReason}; exported final changed checkpoint`
        : "exported final changed checkpoint";
    }
  }
  model.trainingLoss = value.loss;
  model.validationLoss = value.validationLoss;
  model.bestValidationLoss = value.bestValidationLoss;
  model.earlyStopReason = value.earlyStopReason;
  model.labelCounts = labelSourceCounts(samples);
  model.nonZeroWeights = countNonZero(value.weights) + countNonZero(value.hiddenWeights);
  model.replayBufferSize = samples.length;
  return model;
}

async function trainValue(device, samples, config, activeModel, progress) {
  if (!samples.every((sample) => sample.features.length === samples[0].features.length)) {
    throw new Error("Training samples have inconsistent feature lengths.");
  }

  const sampleCount = samples.length;
  const labels = new Float32Array(sampleCount);
  const labelWeights = new Float32Array(sampleCount);
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    labels[sampleIndex] = samples[sampleIndex].label;
    labelWeights[sampleIndex] = samples[sampleIndex].labelWeight ?? 1;
  }
  const split = splitValidationSamples(samples, config.validationSplit);
  const trainIndices = split.trainIndices.length ? split.trainIndices : split.validationIndices;
  const validationIndices = split.validationIndices;
  const batchSize = Math.min(config.batchSize, Math.max(1, trainIndices.length));

  const initialHidden = modelArchitectureMatches(activeModel)
    ? activeModel.hiddenWeights
    : initialHiddenWeights(PROJECTION_SIZE, HIDDEN_LAYERS);
  const layerWeights = splitHiddenWeights(initialHidden, PROJECTION_SIZE, HIDDEN_LAYERS);
  const outputSize = HIDDEN_LAYERS.at(-1);
  const outputWeights = new Float32Array(outputSize + 1);
  if (modelArchitectureMatches(activeModel) && activeModel.outputWeights?.length === outputWeights.length) {
    outputWeights.set(activeModel.outputWeights);
  } else {
    outputWeights[outputSize] = labels.reduce((sum, value) => sum + value, 0) / labels.length;
  }

  const featureBuffer = await timed(config.metrics, "projection", () =>
    projectSamplesToBuffer(device, samples, PROJECTION_SIZE, PROJECTION_SEED)
  );
  const labelBuffer = storageBuffer(device, labels, GPUBufferUsage.STORAGE);
  const labelWeightBuffer = storageBuffer(device, labelWeights, GPUBufferUsage.STORAGE);
  const weightBuffers = layerWeights.map((weights) =>
    storageBuffer(device, weights, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
  );
  const outputWeightBuffer = storageBuffer(
    device,
    outputWeights,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const activationBuffers = HIDDEN_LAYERS.map((layerSize) => device.createBuffer({
    size: align4(batchSize * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE
  }));
  const deltaBuffers = HIDDEN_LAYERS.map((layerSize) => device.createBuffer({
    size: align4(batchSize * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE
  }));
  const predictionBuffer = device.createBuffer({
    size: align4(batchSize * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE
  });
  const outputDeltaBuffer = device.createBuffer({
    size: align4(batchSize * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE
  });
  const batchesPerSubmit = Math.min(VALUE_EPOCHS_PER_SUBMIT, Math.max(1, config.epochs));
  const validationInterval = Math.max(batchesPerSubmit, config.validationInterval ?? 256);
  const batchIndexBuffers = Array.from({ length: batchesPerSubmit }, () => device.createBuffer({
    size: align4(batchSize * Uint32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  }));
  const forwardLayerParams = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      batchSize,
      layerIndex === 0 ? PROJECTION_SIZE : HIDDEN_LAYERS[layerIndex - 1],
      layerSize,
      0
    )
  );
  const applyLayerParams = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      batchSize,
      layerIndex === 0 ? PROJECTION_SIZE : HIDDEN_LAYERS[layerIndex - 1],
      layerSize,
      config.learningRate,
      config.weightDecay
    )
  );
  const forwardOutputParams = outputParamsBuffer(device, batchSize, outputSize, 0);
  const applyOutputParams = outputParamsBuffer(device, batchSize, outputSize, config.learningRate, config.weightDecay);
  const outputDeltaParams = outputDeltaParamsBuffer(device, batchSize);
  const lastHiddenDeltaParams = hiddenDeltaParamsBuffer(
    device,
    batchSize,
    HIDDEN_LAYERS.at(-1),
    0
  );
  const hiddenDeltaParams = HIDDEN_LAYERS.slice(0, -1).map((layerSize, layerIndex) =>
    hiddenDeltaParamsBuffer(device, batchSize, layerSize, HIDDEN_LAYERS[layerIndex + 1])
  );

  const forwardIndexedLayerPipeline = await createComputePipelineChecked(device, "forward_indexed_layer", FORWARD_INDEXED_LAYER_SHADER, "forward_layer");
  const forwardLayerPipeline = await createComputePipelineChecked(device, "forward_layer", FORWARD_LAYER_SHADER, "forward_layer");
  const forwardOutputPipeline = await createComputePipelineChecked(device, "forward_output", FORWARD_OUTPUT_SHADER, "forward_output");
  const outputDeltaPipeline = await createComputePipelineChecked(device, "output_delta", OUTPUT_DELTA_SHADER, "output_delta");
  const lastHiddenDeltaPipeline = await createComputePipelineChecked(device, "hidden3_delta", HIDDEN3_DELTA_SHADER, "hidden3_delta");
  const hiddenDeltaPipeline = await createComputePipelineChecked(device, "hidden_delta", HIDDEN_DELTA_SHADER, "hidden_delta");
  const applyIndexedLayerPipeline = await createComputePipelineChecked(device, "apply_indexed_layer", APPLY_INDEXED_LAYER_SHADER, "apply_layer");
  const applyLayerPipeline = await createComputePipelineChecked(device, "apply_layer", APPLY_LAYER_SHADER, "apply_layer");
  const applyOutputPipeline = await createComputePipelineChecked(device, "apply_output", APPLY_OUTPUT_SHADER, "apply_output");

  let bestValidationLoss = Number.POSITIVE_INFINITY;
  let bestOutputWeights = new Float32Array(outputWeights);
  let bestLayerWeights = layerWeights.map((weights) => new Float32Array(weights));
  let latestOutputWeights = new Float32Array(outputWeights);
  let latestLayerWeights = layerWeights.map((weights) => new Float32Array(weights));
  let epochsWithoutImprovement = 0;
  let lastTrainLoss = Number.NaN;
  let lastValidationLoss = Number.NaN;
  let earlyStopReason = "";

  for (let epoch = 1; epoch <= config.epochs;) {
    const batchEnd = Math.min(config.epochs, epoch + batchesPerSubmit - 1);
    const encoder = device.createCommandEncoder();
    for (; epoch <= batchEnd; epoch += 1) {
      const batchSlot = (epoch - 1) % batchesPerSubmit;
      const batchIndexBuffer = batchIndexBuffers[batchSlot];
      const epochOrder = shuffledIndices(trainIndices, epoch, split.seed);
      const batchStart = ((epoch - 1) * batchSize) % epochOrder.length;
      const batch = new Uint32Array(batchSize);
      for (let index = 0; index < batchSize; index += 1) {
        batch[index] = epochOrder[(batchStart + index) % epochOrder.length];
      }
      device.queue.writeBuffer(batchIndexBuffer, 0, batch);
      for (let layerIndex = 0; layerIndex < HIDDEN_LAYERS.length; layerIndex += 1) {
        const inputSize = layerIndex === 0 ? PROJECTION_SIZE : HIDDEN_LAYERS[layerIndex - 1];
        const inputBuffer = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1];
        const outputSizeForLayer = HIDDEN_LAYERS[layerIndex];
        encodePipeline(device, encoder, layerIndex === 0 ? forwardIndexedLayerPipeline : forwardLayerPipeline, [
          inputBuffer,
          weightBuffers[layerIndex],
          activationBuffers[layerIndex],
          forwardLayerParams[layerIndex],
          ...(layerIndex === 0 ? [batchIndexBuffer] : [])
        ], Math.ceil(batchSize / 16), Math.ceil(outputSizeForLayer / 16));
      }

      encodePipeline(device, encoder, forwardOutputPipeline, [
        activationBuffers.at(-1),
        outputWeightBuffer,
        predictionBuffer,
        forwardOutputParams
      ], Math.ceil(batchSize / 64));

      encodePipeline(device, encoder, outputDeltaPipeline, [
        predictionBuffer,
        labelBuffer,
        outputDeltaBuffer,
        outputDeltaParams,
        batchIndexBuffer,
        labelWeightBuffer
      ], Math.ceil(batchSize / 64));

      const lastLayerIndex = HIDDEN_LAYERS.length - 1;
      encodePipeline(device, encoder, lastHiddenDeltaPipeline, [
        activationBuffers[lastLayerIndex],
        outputDeltaBuffer,
        outputWeightBuffer,
        deltaBuffers[lastLayerIndex],
        lastHiddenDeltaParams
      ], Math.ceil(batchSize / 16), Math.ceil(HIDDEN_LAYERS[lastLayerIndex] / 16));

      for (let layerIndex = HIDDEN_LAYERS.length - 2; layerIndex >= 0; layerIndex -= 1) {
        encodePipeline(device, encoder, hiddenDeltaPipeline, [
          activationBuffers[layerIndex],
          deltaBuffers[layerIndex + 1],
          weightBuffers[layerIndex + 1],
          deltaBuffers[layerIndex],
          hiddenDeltaParams[layerIndex]
        ], Math.ceil(batchSize / 16), Math.ceil(HIDDEN_LAYERS[layerIndex] / 16));
      }

      encodePipeline(device, encoder, applyOutputPipeline, [
        activationBuffers.at(-1),
        outputDeltaBuffer,
        outputWeightBuffer,
        applyOutputParams
      ], Math.ceil((outputSize + 1) / 64));

      for (let layerIndex = HIDDEN_LAYERS.length - 1; layerIndex >= 0; layerIndex -= 1) {
        const inputSize = layerIndex === 0 ? PROJECTION_SIZE : HIDDEN_LAYERS[layerIndex - 1];
        const inputBuffer = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1];
        const outputSizeForLayer = HIDDEN_LAYERS[layerIndex];
        encodePipeline(device, encoder, layerIndex === 0 ? applyIndexedLayerPipeline : applyLayerPipeline, [
          inputBuffer,
          deltaBuffers[layerIndex],
          weightBuffers[layerIndex],
          applyLayerParams[layerIndex],
          ...(layerIndex === 0 ? [batchIndexBuffer] : [])
        ], Math.ceil((inputSize + 1) / 16), Math.ceil(outputSizeForLayer / 16));
      }
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (batchEnd % validationInterval !== 0 && batchEnd < config.epochs) {
      continue;
    }
    const { currentOutput, currentLayers, currentModel } = await timed(config.metrics, "weightReadback", async () => {
      const readOutput = await readFloats(device, outputWeightBuffer, outputWeights.byteLength);
      const readLayers = [];
      for (let layerIndex = 0; layerIndex < weightBuffers.length; layerIndex += 1) {
        readLayers.push(await readFloats(device, weightBuffers[layerIndex], layerWeights[layerIndex].byteLength));
      }
      return {
        currentOutput: readOutput,
        currentLayers: readLayers,
        currentModel: {
          projectionSize: PROJECTION_SIZE,
          projectionSeed: PROJECTION_SEED,
          hiddenLayers: HIDDEN_LAYERS,
          hiddenWeights: concatFloat32(readLayers),
          outputWeights: readOutput,
          scale: 1
        }
      };
    });
    latestOutputWeights = currentOutput;
    latestLayerWeights = currentLayers.map((weights) => new Float32Array(weights));
    lastTrainLoss = await timed(config.metrics, "trainLoss", () =>
      predictionLossOnGpu(device, indexSamples(samples, trainIndices), currentModel)
    );
    lastValidationLoss = validationIndices.length
      ? await timed(config.metrics, "validationLoss", () =>
        predictionLossOnGpu(device, indexSamples(samples, validationIndices), currentModel)
      )
      : lastTrainLoss;
    if (lastValidationLoss + 1e-6 < bestValidationLoss) {
      bestValidationLoss = lastValidationLoss;
      bestOutputWeights = currentOutput;
      bestLayerWeights = currentLayers.map((weights) => new Float32Array(weights));
      epochsWithoutImprovement = 0;
    } else {
      epochsWithoutImprovement += 1;
    }
    progress({
      epoch: batchEnd,
      loss: lastTrainLoss,
      validationLoss: lastValidationLoss,
      bestValidationLoss,
      epochsWithoutImprovement,
      batchSize,
      batchesPerSubmit,
      validationInterval,
      replaySize: sampleCount,
      labelCounts: labelSourceCounts(samples)
    });
    if (epochsWithoutImprovement >= config.patience) {
      earlyStopReason = `validation did not improve for ${config.patience} checks`;
      break;
    }
  }

  const trainedOutput = bestOutputWeights;
  const trainedHidden = concatFloat32(bestLayerWeights);
  return {
    featureCount: outputSize,
    weights: trainedOutput,
    hiddenWeights: trainedHidden,
    finalWeights: latestOutputWeights,
    finalHiddenWeights: concatFloat32(latestLayerWeights),
    loss: lastTrainLoss,
    validationLoss: lastValidationLoss,
    bestValidationLoss,
    earlyStopReason
  };
}

async function trainPolicy(device, samples, config, activeModel) {
  const policySamples = samples.filter((sample) =>
    sample.labelKind !== "distilled" && Number.isInteger(sample.policy) && sample.policy >= 0
  );
  if (!policySamples.length) {
    return activeModel?.policy_logits?.length
      ? new Float32Array(activeModel.policy_logits.slice(0, POLICY_BUCKETS))
      : new Float32Array(POLICY_BUCKETS);
  }
  const targets = new Uint32Array(policySamples.map((sample) => Math.min(POLICY_BUCKETS - 1, sample.policy)));
  const logits = new Float32Array(POLICY_BUCKETS);
  if (activeModel?.policy_logits?.length) {
    logits.set(activeModel.policy_logits.slice(0, POLICY_BUCKETS));
  }
  const params = paramsBuffer([policySamples.length, POLICY_BUCKETS], config.learningRate, 1);
  const targetBuffer = storageBuffer(device, targets, GPUBufferUsage.STORAGE);
  let inputLogits = storageBuffer(device, logits, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  let outputLogits = device.createBuffer({
    size: logits.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const paramsGpuBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
  const pipeline = await createComputePipelineChecked(device, "train_policy", POLICY_SHADER, "train_policy");
  const steps = Math.max(1, Math.ceil(config.epochs / 4));
  for (let step = 0; step < steps;) {
    const batchEnd = Math.min(steps, step + POLICY_STEPS_PER_SUBMIT);
    const encoder = device.createCommandEncoder();
    for (; step < batchEnd; step += 1) {
      encodePipeline(
        device,
        encoder,
        pipeline,
        [targetBuffer, inputLogits, outputLogits, paramsGpuBuffer],
        Math.ceil(POLICY_BUCKETS / 64)
      );
      [inputLogits, outputLogits] = [outputLogits, inputLogits];
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  return readFloats(device, inputLogits, logits.byteLength);
}

function runPipeline(device, pipeline, buffers, workgroupsX, workgroupsY = 1) {
  const encoder = device.createCommandEncoder();
  encodePipeline(device, encoder, pipeline, buffers, workgroupsX, workgroupsY);
  device.queue.submit([encoder.finish()]);
}

function encodePipeline(device, encoder, pipeline, buffers, workgroupsX, workgroupsY = 1) {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupsX, workgroupsY);
  pass.end();
}

function paramsBuffer([first, second], learningRate, fourth) {
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, first, true);
  view.setUint32(4, second, true);
  view.setFloat32(8, learningRate, true);
  view.setUint32(12, fourth, true);
  return params;
}

function layerParamsBuffer(device, sampleCount, inputSize, outputSize, learningRate, weightDecay = 0) {
  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setUint32(8, outputSize, true);
  view.setFloat32(12, learningRate, true);
  view.setFloat32(16, weightDecay, true);
  return storageBuffer(device, params, GPUBufferUsage.UNIFORM);
}

function outputParamsBuffer(device, sampleCount, inputSize, learningRate, weightDecay = 0) {
  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setFloat32(12, learningRate, true);
  view.setFloat32(16, weightDecay, true);
  return storageBuffer(device, params, GPUBufferUsage.UNIFORM);
}

function outputDeltaParamsBuffer(device, sampleCount) {
  const params = new ArrayBuffer(16);
  new DataView(params).setUint32(0, sampleCount, true);
  return storageBuffer(device, params, GPUBufferUsage.UNIFORM);
}

function hiddenDeltaParamsBuffer(device, sampleCount, currentSize, nextSize) {
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, currentSize, true);
  view.setUint32(8, nextSize, true);
  return storageBuffer(device, params, GPUBufferUsage.UNIFORM);
}

function storageBuffer(device, data, usage) {
  const bytes = data instanceof ArrayBuffer ? data : data.buffer;
  const buffer = device.createBuffer({
    size: align4(bytes.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

async function projectSamplesToBuffer(device, samples, projectionSize, seed = PROJECTION_SEED) {
  const inputSize = featureLength(samples);
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const projectedBytes = samples.length * projectionSize * Float32Array.BYTES_PER_ELEMENT;
  if (projectedBytes > maxBindingSize) {
    throw new Error(`Projected replay buffer exceeds this WebGPU device's storage binding limit (${formatBytes(projectedBytes)} > ${formatBytes(maxBindingSize)}). Reduce replay buffer or projection size.`);
  }
  const projectedBuffer = device.createBuffer({
    size: align4(projectedBytes),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  const pipeline = await createComputePipelineChecked(device, "project_features", PROJECT_FEATURES_SHADER, "project_features");
  for (let offset = 0; offset < samples.length; offset += PROJECTION_CHUNK_SIZE) {
    const chunkSamples = samples.slice(offset, offset + PROJECTION_CHUNK_SIZE);
    const rawFeatures = new Float32Array(chunkSamples.length * inputSize);
    for (let sampleIndex = 0; sampleIndex < chunkSamples.length; sampleIndex += 1) {
      rawFeatures.set(chunkSamples[sampleIndex].features, sampleIndex * inputSize);
    }
    if (rawFeatures.byteLength > maxBindingSize) {
      throw new Error(`Projection chunk exceeds this WebGPU device's storage binding limit (${formatBytes(rawFeatures.byteLength)} > ${formatBytes(maxBindingSize)}). Reduce batch size or feature size.`);
    }
    const rawBuffer = storageBuffer(device, rawFeatures, GPUBufferUsage.STORAGE);
    const chunkProjectedBuffer = device.createBuffer({
      size: align4(chunkSamples.length * projectionSize * Float32Array.BYTES_PER_ELEMENT),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const paramsBuffer = projectionParamsBuffer(device, chunkSamples.length, inputSize, projectionSize, seed);
    const encoder = device.createCommandEncoder();
    encodePipeline(
      device,
      encoder,
      pipeline,
      [rawBuffer, chunkProjectedBuffer, paramsBuffer],
      Math.ceil(chunkSamples.length / 16),
      Math.ceil(projectionSize / 16)
    );
    encoder.copyBufferToBuffer(
      chunkProjectedBuffer,
      0,
      projectedBuffer,
      offset * projectionSize * Float32Array.BYTES_PER_ELEMENT,
      chunkSamples.length * projectionSize * Float32Array.BYTES_PER_ELEMENT
    );
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  return projectedBuffer;
}

function splitValidationSamples(samples, validationSplit) {
  const trainIndices = [];
  const validationIndices = [];
  const threshold = Math.floor(validationSplit * 10000);
  const seed = samples.reduce((hash, sample, index) => {
    hash ^= stableSampleHash(sample, index);
    return Math.imul(hash, 16777619) >>> 0;
  }, 2166136261);
  for (let index = 0; index < samples.length; index += 1) {
    const bucket = stableSampleHash(samples[index], index) % 10000;
    if (threshold > 0 && bucket < threshold) {
      validationIndices.push(index);
    } else {
      trainIndices.push(index);
    }
  }
  if (!trainIndices.length && validationIndices.length > 1) {
    trainIndices.push(validationIndices.pop());
  }
  return { trainIndices, validationIndices, seed };
}

function stableSampleHash(sample, index) {
  let hash = 2166136261;
  const text = `${sample.labelKind ?? ""}|${sample.sideToMove ?? ""}|${sample.boardCount ?? 0}|${index}`;
  for (let offset = 0; offset < text.length; offset += 1) {
    hash ^= text.charCodeAt(offset);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function shuffledIndices(indices, epoch, seed) {
  const result = indices.slice();
  let state = (seed ^ Math.imul(epoch, 2654435761)) >>> 0;
  for (let index = result.length - 1; index > 0; index -= 1) {
    state = xorshift32(state);
    const swapIndex = state % (index + 1);
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }
  return result;
}

function xorshift32(value) {
  let state = value >>> 0;
  state ^= state << 13;
  state ^= state >>> 17;
  state ^= state << 5;
  return state >>> 0;
}

function indexSamples(samples, indices) {
  return indices.map((index) => samples[index]);
}

function featureLength(samples) {
  const length = samples[0]?.features?.length;
  if (!length || !samples.every((sample) => sample.features.length === length)) {
    throw new Error("Training samples have inconsistent feature lengths.");
  }
  return length;
}

function projectionParamsBuffer(device, sampleCount, inputSize, projectionSize, seed) {
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setUint32(8, projectionSize, true);
  view.setUint32(12, seed >>> 0, true);
  return storageBuffer(device, params, GPUBufferUsage.UNIFORM);
}

async function readFloats(device, buffer, byteLength) {
  const readBuffer = device.createBuffer({
    size: align4(byteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const copy = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  return copy;
}

async function predictionLossOnGpu(device, samples, model) {
  const predictions = await predictValuesOnGpu(device, samples, model);
  let total = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const error = predictions[index] - samples[index].label;
    total += error * error;
  }
  return total / samples.length;
}

function splitHiddenWeights(hiddenWeights, inputSize, hiddenLayers) {
  const layers = [];
  let cursor = 0;
  let previousSize = inputSize;
  for (const layerSize of hiddenLayers) {
    const length = layerSize * (previousSize + 1);
    layers.push(hiddenWeights.slice(cursor, cursor + length));
    cursor += length;
    previousSize = layerSize;
  }
  return layers;
}

function concatFloat32(arrays) {
  const length = arrays.reduce((sum, array) => sum + array.length, 0);
  const result = new Float32Array(length);
  let cursor = 0;
  for (const array of arrays) {
    result.set(array, cursor);
    cursor += array.length;
  }
  return result;
}

function countNonZero(values) {
  let count = 0;
  for (const value of values) {
    if (value !== 0) {
      count += 1;
    }
  }
  return count;
}

async function predictValues(samples, model) {
  if (!samples.length) {
    return [];
  }
  if (!globalThis.navigator?.gpu || !modelArchitectureMatches(model)) {
    throw new Error("GPU batch prediction unavailable.");
  }
  const device = await getGpuDevice();
  if (!device) {
    throw new Error("No WebGPU adapter is available.");
  }
  return Array.from(await predictValuesOnGpu(device, samples, model));
}

async function predictValuesOnGpu(device, samples, model) {
  const sampleCount = samples.length;
  const featureBuffer = await projectSamplesToBuffer(
    device,
    samples,
    model.projectionSize,
    model.projectionSeed
  );
  const hiddenLayers = model.hiddenLayers;
  const layerWeights = splitHiddenWeights(model.hiddenWeights, model.projectionSize, hiddenLayers);
  const weightBuffers = layerWeights.map((weights) => storageBuffer(device, weights, GPUBufferUsage.STORAGE));
  const outputWeightBuffer = storageBuffer(device, model.outputWeights, GPUBufferUsage.STORAGE);
  const activationBuffers = hiddenLayers.map((layerSize) => device.createBuffer({
    size: align4(sampleCount * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE
  }));
  const predictionBuffer = device.createBuffer({
    size: align4(sampleCount * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const forwardLayerParams = hiddenLayers.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      sampleCount,
      layerIndex === 0 ? model.projectionSize : hiddenLayers[layerIndex - 1],
      layerSize,
      0
    )
  );
  const forwardOutputParams = outputParamsBuffer(device, sampleCount, hiddenLayers.at(-1), 0);
  const forwardLayerPipeline = await createComputePipelineChecked(device, "forward_layer", FORWARD_LAYER_SHADER, "forward_layer");
  const forwardOutputPipeline = await createComputePipelineChecked(device, "forward_output", FORWARD_OUTPUT_SHADER, "forward_output");
  const encoder = device.createCommandEncoder();
  for (let layerIndex = 0; layerIndex < hiddenLayers.length; layerIndex += 1) {
    const inputBuffer = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1];
    encodePipeline(device, encoder, forwardLayerPipeline, [
      inputBuffer,
      weightBuffers[layerIndex],
      activationBuffers[layerIndex],
      forwardLayerParams[layerIndex]
    ], Math.ceil(sampleCount / 16), Math.ceil(hiddenLayers[layerIndex] / 16));
  }
  encodePipeline(device, encoder, forwardOutputPipeline, [
    activationBuffers.at(-1),
    outputWeightBuffer,
    predictionBuffer,
    forwardOutputParams
  ], Math.ceil(sampleCount / 64));
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const predictions = await readFloats(device, predictionBuffer, sampleCount * Float32Array.BYTES_PER_ELEMENT);
  const scale = model.scale ?? 1;
  for (let index = 0; index < predictions.length; index += 1) {
    predictions[index] *= scale;
  }
  return predictions;
}

function modelArchitectureMatches(model) {
  return model
    && model.projectionSize === PROJECTION_SIZE
    && model.projectionSeed === PROJECTION_SEED
    && JSON.stringify(model.hiddenLayers) === JSON.stringify(HIDDEN_LAYERS)
    && model.hiddenWeights?.length;
}

function projectionHash(rawIndex, projectionIndex, seed) {
  let hash = (seed ^ rawIndex) >>> 0;
  hash = Math.imul(hash, 16777619) >>> 0;
  hash = (hash ^ projectionIndex) >>> 0;
  hash = Math.imul(hash, 16777619) >>> 0;
  hash = (hash ^ (hash >>> 16)) >>> 0;
  return hash;
}

function initialHiddenWeights(inputSize, hiddenLayers) {
  const weights = [];
  let previous = inputSize;
  for (let layerIndex = 0; layerIndex < hiddenLayers.length; layerIndex += 1) {
    const layerSize = hiddenLayers[layerIndex];
    const scale = Math.sqrt(2 / previous);
    for (let output = 0; output < layerSize; output += 1) {
      for (let input = 0; input < previous; input += 1) {
        const hash = projectionHash(input, output + layerIndex * 4099, PROJECTION_SEED);
        weights.push((((hash / 0xffffffff) * 2) - 1) * scale);
      }
      weights.push(0);
    }
    previous = layerSize;
  }
  return new Float32Array(weights);
}

function encodeCompactModel(model) {
  const hiddenWeights = Array.from(model.hiddenWeights);
  const outputWeights = Array.from(model.outputWeights);
  const floats = [model.scale, model.bias, ...hiddenWeights, ...outputWeights];
  const byteLength = 4
    + 4 * 7
    + 4 * model.hiddenLayers.length
    + 4 * floats.length;
  const buffer = new ArrayBuffer(byteLength);
  const view = new DataView(buffer);
  let cursor = 0;
  writeAscii(view, cursor, "CFNN");
  cursor += 4;
  cursor = writeU32(view, cursor, 1);
  cursor = writeU32(view, cursor, model.projectionSize);
  cursor = writeU32(view, cursor, model.projectionSeed);
  cursor = writeU32(view, cursor, model.hiddenLayers.length);
  cursor = writeU32(view, cursor, outputWeights.length);
  cursor = writeF32(view, cursor, model.scale);
  cursor = writeF32(view, cursor, model.bias);
  for (const layer of model.hiddenLayers) {
    cursor = writeU32(view, cursor, layer);
  }
  cursor = writeU32(view, cursor, hiddenWeights.length);
  for (const value of hiddenWeights) {
    cursor = writeF32(view, cursor, value);
  }
  for (const value of outputWeights) {
    cursor = writeF32(view, cursor, value);
  }
  return new Uint8Array(buffer);
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function writeU32(view, offset, value) {
  view.setUint32(offset, value, true);
  return offset + 4;
}

function writeF32(view, offset, value) {
  view.setFloat32(offset, value, true);
  return offset + 4;
}

function appendReplaySamples(buffer, samples, maxBuffer) {
  const merged = buffer.concat(samples).filter((sample) => Array.isArray(sample.features));
  return merged.slice(Math.max(0, merged.length - maxBuffer));
}

async function validateLossLogs(config, progress) {
  const logs = await fetchLossLogs(config.lossLogReplay);
  const validation = {
    checked: 0,
    changed: 0,
    unchanged: 0,
    skipped: 0,
    failed: false,
    examples: []
  };
  if (!logs.length || config.lossLogReplay <= 0) {
    return validation;
  }

  const ai = new Worker("./ai-worker.js", { type: "module" });
  try {
    for (const log of logs) {
      const decisions = Array.isArray(log.decisions) ? log.decisions : [];
      let logChanged = false;
      for (const decision of decisions) {
        const previousKey = movesKey(decision.selectedMoves);
        if (!decision.game || !previousKey) {
          validation.skipped += 1;
          continue;
        }
        validation.checked += 1;
        progress?.({
          lossLogValidation: {
            checked: validation.checked,
            changed: validation.changed,
            logPath: log.logPath ?? null
          }
        });
        try {
          const response = await requestWorker(ai, {
            type: "search",
            game: decision.game,
            depth: config.depth,
            nodes: config.nodes,
            timeMs: workerSearchTimeMs(config),
            gpuMode: "full",
            temperature: config.explorationTemperature,
            randomSeed: sampleSeed("loss-log", validation.checked, config.runSeed ^ 0x1055_1000)
          }, workerRequestTimeout(config));
          const currentMoves = response.result?.moves ?? [];
          const currentKey = movesKey(currentMoves);
          if (!currentKey) {
            validation.skipped += 1;
            continue;
          }
          if (currentKey !== previousKey) {
            validation.changed += 1;
            logChanged = true;
            validation.examples.push({
              logPath: log.logPath ?? null,
              ply: decision.ply ?? null,
              botColor: decision.botColor ?? null,
              previous: previousKey,
              current: currentKey,
              previousScore: decision.selectedScore ?? null,
              currentScore: response.result?.score ?? null
            });
            break;
          }
          validation.unchanged += 1;
        } catch {
          validation.skipped += 1;
        }
      }
      if (logChanged) {
        continue;
      }
    }
  } finally {
    ai.terminate();
  }
  validation.failed = validation.checked > 0 && validation.changed === 0;
  return validation;
}

async function fetchLossLogs(limit) {
  if (limit <= 0) {
    return [];
  }
  try {
    const response = await withTimeout(
      fetch("/api/training/loss-logs"),
      TRAINING_IO_TIMEOUT_MS,
      "Timed out loading loss logs."
    );
    if (!response.ok) {
      return [];
    }
    const payload = await response.json();
    return (payload.logs ?? [])
      .filter((log) => Array.isArray(log.decisions) && log.decisions.length > 0)
      .slice(0, limit);
  } catch {
    return [];
  }
}

function movesKey(moves) {
  return (moves ?? []).map((move) =>
    `${move?.from?.timelineId}:${move?.from?.time}:${move?.from?.x}:${move?.from?.y}->${move?.to?.timelineId}:${move?.to?.time}:${move?.to?.x}:${move?.to?.y}`
  ).join("|");
}

async function fetchActiveModel() {
  try {
    const response = await withTimeout(
      fetch("/api/training/model"),
      TRAINING_IO_TIMEOUT_MS,
      "Timed out loading active model."
    );
    if (!response.ok) {
      return null;
    }
    const buffer = await response.arrayBuffer();
    const model = decodeCompactModel(buffer);
    if (model) {
      model.bytes = new Uint8Array(buffer);
    }
    return model;
  } catch {
    return null;
  }
}

function byteArraysEqual(left, right) {
  if (!left || !right || left.byteLength !== right.byteLength) {
    return false;
  }
  for (let index = 0; index < left.byteLength; index += 1) {
    if (left[index] !== right[index]) {
      return false;
    }
  }
  return true;
}

function decodeCompactModel(buffer) {
  const view = new DataView(buffer);
  let cursor = 0;
  const magic = String.fromCharCode(
    view.getUint8(cursor),
    view.getUint8(cursor + 1),
    view.getUint8(cursor + 2),
    view.getUint8(cursor + 3)
  );
  cursor += 4;
  if (magic !== "CFNN") {
    return null;
  }
  const version = view.getUint32(cursor, true);
  cursor += 4;
  if (version !== 1) {
    return null;
  }
  const projectionSize = view.getUint32(cursor, true);
  cursor += 4;
  const projectionSeed = view.getUint32(cursor, true);
  cursor += 4;
  const layerCount = view.getUint32(cursor, true);
  cursor += 4;
  const outputSize = view.getUint32(cursor, true);
  cursor += 4;
  const scale = view.getFloat32(cursor, true);
  cursor += 4;
  const bias = view.getFloat32(cursor, true);
  cursor += 4;
  const hiddenLayers = [];
  for (let index = 0; index < layerCount; index += 1) {
    hiddenLayers.push(view.getUint32(cursor, true));
    cursor += 4;
  }
  const hiddenWeightCount = view.getUint32(cursor, true);
  cursor += 4;
  const hiddenWeights = new Float32Array(hiddenWeightCount);
  for (let index = 0; index < hiddenWeightCount; index += 1) {
    hiddenWeights[index] = view.getFloat32(cursor, true);
    cursor += 4;
  }
  const outputWeights = new Float32Array(outputSize);
  for (let index = 0; index < outputSize; index += 1) {
    outputWeights[index] = view.getFloat32(cursor, true);
    cursor += 4;
  }
  return {
    projectionSize,
    projectionSeed,
    hiddenLayers,
    hiddenWeights,
    outputWeights,
    scale,
    bias
  };
}

async function loadReplayBuffer() {
  try {
    const db = await withTimeout(openReplayDb(), TRAINING_IO_TIMEOUT_MS, "Timed out opening replay buffer.");
    return (await withTimeout(idbGet(db, BUFFER_KEY), TRAINING_IO_TIMEOUT_MS, "Timed out reading replay buffer.")) ?? [];
  } catch {
    return [];
  }
}

async function saveReplayBuffer(samples) {
  try {
    const db = await withTimeout(openReplayDb(), TRAINING_IO_TIMEOUT_MS, "Timed out opening replay buffer.");
    await withTimeout(idbPut(db, BUFFER_KEY, samples), TRAINING_IO_TIMEOUT_MS, "Timed out saving replay buffer.");
  } catch {
    // IndexedDB is an optimization; an in-memory run still works without it.
  }
}

function openReplayDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("chronofish-training", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("buffers");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function idbGet(db, key) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("buffers", "readonly");
    const request = tx.objectStore("buffers").get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function idbPut(db, key, value) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("buffers", "readwrite");
    tx.objectStore("buffers").put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

function autoLabelWorkers() {
  const cores = navigator.hardwareConcurrency ?? 4;
  return Math.max(1, Math.min(cores - 1, 16));
}

async function getGpuDevice() {
  if (!navigator.gpu) {
    return null;
  }
  if (cachedGpuDevice) {
    return cachedGpuDevice;
  }
  cachedGpuAdapter = cachedGpuAdapter ?? await navigator.gpu.requestAdapter();
  if (!cachedGpuAdapter) {
    return null;
  }
  cachedGpuDevice = await requestHighLimitDevice(cachedGpuAdapter);
  cachedGpuDevice.lost?.then(() => {
    cachedGpuDevice = null;
    pipelineCache.clear();
  });
  return cachedGpuDevice;
}

async function requestHighLimitDevice(adapter) {
  const requiredLimits = {};
  for (const key of ["maxStorageBufferBindingSize", "maxBufferSize"]) {
    const value = adapter.limits?.[key];
    if (Number.isFinite(value) && value > 0) {
      requiredLimits[key] = value;
    }
  }
  if (Object.keys(requiredLimits).length === 0) {
    return adapter.requestDevice();
  }
  try {
    return await adapter.requestDevice({ requiredLimits });
  } catch {
    return adapter.requestDevice();
  }
}

async function createComputePipelineChecked(device, label, code, entryPoint) {
  const cacheKey = `${label}:${entryPoint}`;
  const cached = pipelineCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const module = device.createShaderModule({ label: `${label}.module`, code });
  if (module.compilationInfo) {
    const info = await module.compilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(formatShaderErrors(label, errors));
    }
  }
  const pipeline = device.createComputePipeline({
    label,
    layout: "auto",
    compute: { module, entryPoint }
  });
  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

function formatShaderErrors(label, errors) {
  return `${label} shader compilation failed: ${errors.map((error) =>
    `line ${error.lineNum ?? "?"}, column ${error.linePos ?? "?"}: ${error.message}`
  ).join("; ")}`;
}

function formatBytes(bytes) {
  const mib = bytes / (1024 * 1024);
  return `${mib.toFixed(mib >= 10 ? 0 : 1)} MiB`;
}

function align4(value) {
  return Math.ceil(value / 4) * 4;
}
