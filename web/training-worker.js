const POLICY_BUCKETS = 257;
const BUFFER_KEY = "value-policy-buffer";
const PROJECTION_SIZE = 2048;
const PROJECTION_SEED = 2166136261;
const MAX_PLAYOUT_PLIES = 10;
const HIDDEN_LAYERS = [1024, 512, 256];
const VALUE_EPOCHS_PER_SUBMIT = 8;
const POLICY_STEPS_PER_SUBMIT = 64;

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
  let output = id.y;
  if (sample >= params.sample_count || output >= params.output_size) {
    return;
  }

  let row = output * (params.input_size + 1u);
  var sum = weights[row + params.input_size];
  let input_base = sample * params.input_size;
  for (var input = 0u; input < params.input_size; input = input + 1u) {
    sum = sum + input_values[input_base + input] * weights[row + input];
  }
  output_values[sample * params.output_size + output] = max(sum, 0.0);
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
  for (var input = 0u; input < params.input_size; input = input + 1u) {
    sum = sum + input_values[input_base + input] * weights[input];
  }
  predictions[sample] = sum;
}
`;

const OUTPUT_DELTA_SHADER = `
struct Params {
  sample_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> predictions: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<f32>;
@group(0) @binding(2) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn output_delta(@builtin(global_invocation_id) id: vec3<u32>) {
  let sample = id.x;
  if (sample >= params.sample_count) {
    return;
  }
  deltas[sample] = predictions[sample] - labels[sample];
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
};

@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn apply_layer(@builtin(global_invocation_id) id: vec3<u32>) {
  let input = id.x;
  let output = id.y;
  if (input > params.input_size || output >= params.output_size) {
    return;
  }

  var gradient = 0.0;
  for (var sample = 0u; sample < params.sample_count; sample = sample + 1u) {
    let delta = deltas[sample * params.output_size + output];
    if (input == params.input_size) {
      gradient = gradient + delta;
    } else {
      gradient = gradient + delta * features[sample * params.input_size + input];
    }
  }

  let weight_index = output * (params.input_size + 1u) + input;
  weights[weight_index] = weights[weight_index]
    - params.learning_rate * (2.0 * gradient / f32(params.sample_count));
}
`;

const APPLY_OUTPUT_SHADER = `
struct Params {
  sample_count: u32,
  input_size: u32,
  _pad0: u32,
  learning_rate: f32,
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
  weights[index] = weights[index] - params.learning_rate * (2.0 * gradient / f32(params.sample_count));
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
  const { id, notation, config } = event.data;
  try {
    const activeModel = await fetchActiveModel();
    let buffer = await loadReplayBuffer();
    const expertSamples = await collectSamples(notation ?? "", config, false, (collected, sampleCount, workers) => {
      self.postMessage({ id, ok: true, collected, sampleCount });
      if (collected === 0) {
        self.postMessage({ id, ok: true, labelWorkers: workers, sampleCount });
      }
    });
    const pseudoTarget = Math.max(0, config.maxBuffer - buffer.length - expertSamples.length);
    const pseudoSamples = await collectPseudoSamples(notation ?? "", config, activeModel, pseudoTarget);
    buffer = appendReplaySamples(buffer, expertSamples.concat(pseudoSamples), config.maxBuffer);
    await saveReplayBuffer(buffer);
    self.postMessage({
      id,
      ok: true,
      gpuPhase: true,
      bufferSize: buffer.length,
      pseudoCount: buffer.filter((sample) => sample.pseudo).length
    });
    const model = await train(buffer, config, activeModel, (epoch, loss) => {
      self.postMessage({ id, ok: true, epoch, loss });
    });
    self.postMessage({ id, ok: true, model, loss: model.trainingLoss });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error.message });
  }
});

async function collectPseudoSamples(notation, config, activeModel, targetCount) {
  if (!activeModel?.outputWeights?.length) {
    return [];
  }
  const samples = Math.min(config.maxBuffer, Math.max(targetCount, config.samples * 8));
  const pseudoConfig = { ...config, samples };
  const positions = await collectSamples(notation, pseudoConfig, true, () => {});
  return positions.map((sample) => ({
    ...sample,
    label: predictValue(sample.features, activeModel),
    policy: 0,
    pseudo: true
  }));
}

async function collectSamples(notation, config, encodeOnly, progress) {
  const prefixes = notationPrefixes(notation, config.samples);
  const jobs = prefixes.map((prefix, index) => ({
    prefix,
    index,
    seed: sampleSeed(prefix, index, encodeOnly ? 0xa11c_e000 : 0x5eed_1000),
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

function notationPrefixes(notation, count) {
  const lines = notation.split(/\n/).filter(Boolean);
  const prefixes = [];
  for (let index = 0; index < lines.length; index += 1) {
    prefixes.push(lines.slice(0, index + 1).join("\n"));
  }
  if (prefixes.length === 0 || prefixes.at(-1) !== notation) {
    prefixes.push(notation);
  }
  while (prefixes.length < count) {
    prefixes.push(notation);
  }
  return prefixes.slice(0, count);
}

function labelSample(worker, job, config, encodeOnly) {
  return new Promise((resolve, reject) => {
    const messageId = crypto.randomUUID();
    const handleMessage = (event) => {
      if (event.data.id !== messageId) {
        return;
      }
      cleanup();
      if (event.data.ok) {
        resolve(event.data.sample);
      } else {
        reject(new Error(event.data.error));
      }
    };
    const handleError = (event) => {
      cleanup();
      reject(new Error(event.message || "Label worker failed."));
    };
    const cleanup = () => {
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
    };
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.postMessage({
      id: messageId,
      notation: job.prefix,
      depth: config.depth,
      nodes: config.nodes,
      encodeOnly,
      seed: job.seed,
      plies: job.plies
    });
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

async function train(samples, config, activeModel, progress) {
  if (!globalThis.navigator?.gpu) {
    throw new Error("WebGPU is unavailable in this browser.");
  }
  if (!samples?.length) {
    throw new Error("No samples were collected.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter is available.");
  }
  const device = await adapter.requestDevice();
  const value = await trainValue(device, samples, config, activeModel, progress);
  const policy = await trainPolicy(device, samples, config, activeModel);
  const model = encodeCompactModel({
    projectionSize: PROJECTION_SIZE,
    projectionSeed: PROJECTION_SEED,
    hiddenLayers: HIDDEN_LAYERS,
    hiddenWeights: value.hiddenWeights,
    outputWeights: value.weights,
    policyLogits: policy,
    scale: 1,
    bias: 0
  });
  model.trainingLoss = value.loss;
  model.nonZeroWeights = countNonZero(value.weights) + countNonZero(value.hiddenWeights);
  model.replayBufferSize = samples.length;
  return model;
}

async function trainValue(device, samples, config, activeModel, progress) {
  if (!samples.every((sample) => sample.features.length === samples[0].features.length)) {
    throw new Error("Training samples have inconsistent feature lengths.");
  }

  const sampleCount = samples.length;
  const projectedFeatures = new Float32Array(sampleCount * PROJECTION_SIZE);
  const labels = new Float32Array(sampleCount);
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    projectedFeatures.set(
      projectFeatures(samples[sampleIndex].features, PROJECTION_SIZE),
      sampleIndex * PROJECTION_SIZE
    );
    labels[sampleIndex] = samples[sampleIndex].label;
  }

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

  const featureBuffer = storageBuffer(device, projectedFeatures, GPUBufferUsage.STORAGE);
  const labelBuffer = storageBuffer(device, labels, GPUBufferUsage.STORAGE);
  const weightBuffers = layerWeights.map((weights) =>
    storageBuffer(device, weights, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
  );
  const outputWeightBuffer = storageBuffer(
    device,
    outputWeights,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const activationBuffers = HIDDEN_LAYERS.map((layerSize) => device.createBuffer({
    size: align4(sampleCount * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE
  }));
  const deltaBuffers = HIDDEN_LAYERS.map((layerSize) => device.createBuffer({
    size: align4(sampleCount * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE
  }));
  const predictionBuffer = device.createBuffer({
    size: align4(sampleCount * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE
  });
  const outputDeltaBuffer = device.createBuffer({
    size: align4(sampleCount * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE
  });
  const forwardLayerParams = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      sampleCount,
      layerIndex === 0 ? PROJECTION_SIZE : HIDDEN_LAYERS[layerIndex - 1],
      layerSize,
      0
    )
  );
  const applyLayerParams = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      sampleCount,
      layerIndex === 0 ? PROJECTION_SIZE : HIDDEN_LAYERS[layerIndex - 1],
      layerSize,
      config.learningRate
    )
  );
  const forwardOutputParams = outputParamsBuffer(device, sampleCount, outputSize, 0);
  const applyOutputParams = outputParamsBuffer(device, sampleCount, outputSize, config.learningRate);
  const outputDeltaParams = outputDeltaParamsBuffer(device, sampleCount);
  const lastHiddenDeltaParams = hiddenDeltaParamsBuffer(
    device,
    sampleCount,
    HIDDEN_LAYERS.at(-1),
    0
  );
  const hiddenDeltaParams = HIDDEN_LAYERS.slice(0, -1).map((layerSize, layerIndex) =>
    hiddenDeltaParamsBuffer(device, sampleCount, layerSize, HIDDEN_LAYERS[layerIndex + 1])
  );

  const forwardLayerPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: FORWARD_LAYER_SHADER }), entryPoint: "forward_layer" }
  });
  const forwardOutputPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: FORWARD_OUTPUT_SHADER }), entryPoint: "forward_output" }
  });
  const outputDeltaPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: OUTPUT_DELTA_SHADER }), entryPoint: "output_delta" }
  });
  const lastHiddenDeltaPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: HIDDEN3_DELTA_SHADER }), entryPoint: "hidden3_delta" }
  });
  const hiddenDeltaPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: HIDDEN_DELTA_SHADER }), entryPoint: "hidden_delta" }
  });
  const applyLayerPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: APPLY_LAYER_SHADER }), entryPoint: "apply_layer" }
  });
  const applyOutputPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: APPLY_OUTPUT_SHADER }), entryPoint: "apply_output" }
  });

  for (let epoch = 1; epoch <= config.epochs;) {
    const batchEnd = Math.min(config.epochs, epoch + VALUE_EPOCHS_PER_SUBMIT - 1);
    const encoder = device.createCommandEncoder();
    for (; epoch <= batchEnd; epoch += 1) {
      for (let layerIndex = 0; layerIndex < HIDDEN_LAYERS.length; layerIndex += 1) {
        const inputSize = layerIndex === 0 ? PROJECTION_SIZE : HIDDEN_LAYERS[layerIndex - 1];
        const inputBuffer = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1];
        const outputSizeForLayer = HIDDEN_LAYERS[layerIndex];
        encodePipeline(device, encoder, forwardLayerPipeline, [
          inputBuffer,
          weightBuffers[layerIndex],
          activationBuffers[layerIndex],
          forwardLayerParams[layerIndex]
        ], Math.ceil(sampleCount / 16), Math.ceil(outputSizeForLayer / 16));
      }

      encodePipeline(device, encoder, forwardOutputPipeline, [
        activationBuffers.at(-1),
        outputWeightBuffer,
        predictionBuffer,
        forwardOutputParams
      ], Math.ceil(sampleCount / 64));

      encodePipeline(device, encoder, outputDeltaPipeline, [
        predictionBuffer,
        labelBuffer,
        outputDeltaBuffer,
        outputDeltaParams
      ], Math.ceil(sampleCount / 64));

      const lastLayerIndex = HIDDEN_LAYERS.length - 1;
      encodePipeline(device, encoder, lastHiddenDeltaPipeline, [
        activationBuffers[lastLayerIndex],
        outputDeltaBuffer,
        outputWeightBuffer,
        deltaBuffers[lastLayerIndex],
        lastHiddenDeltaParams
      ], Math.ceil(sampleCount / 16), Math.ceil(HIDDEN_LAYERS[lastLayerIndex] / 16));

      for (let layerIndex = HIDDEN_LAYERS.length - 2; layerIndex >= 0; layerIndex -= 1) {
        encodePipeline(device, encoder, hiddenDeltaPipeline, [
          activationBuffers[layerIndex],
          deltaBuffers[layerIndex + 1],
          weightBuffers[layerIndex + 1],
          deltaBuffers[layerIndex],
          hiddenDeltaParams[layerIndex]
        ], Math.ceil(sampleCount / 16), Math.ceil(HIDDEN_LAYERS[layerIndex] / 16));
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
        encodePipeline(device, encoder, applyLayerPipeline, [
          inputBuffer,
          deltaBuffers[layerIndex],
          weightBuffers[layerIndex],
          applyLayerParams[layerIndex]
        ], Math.ceil((inputSize + 1) / 16), Math.ceil(outputSizeForLayer / 16));
      }
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    progress(batchEnd, 0);
  }

  const trainedOutput = await readFloats(device, outputWeightBuffer, outputWeights.byteLength);
  const trainedLayers = [];
  for (let layerIndex = 0; layerIndex < weightBuffers.length; layerIndex += 1) {
    trainedLayers.push(await readFloats(device, weightBuffers[layerIndex], layerWeights[layerIndex].byteLength));
  }
  const trainedHidden = concatFloat32(trainedLayers);
  return {
    featureCount: outputSize,
    weights: trainedOutput,
    hiddenWeights: trainedHidden,
    loss: mse(samples, trainedOutput, trainedHidden)
  };
}

async function trainPolicy(device, samples, config, activeModel) {
  const targets = new Uint32Array(samples.map((sample) => Math.min(POLICY_BUCKETS - 1, sample.policy ?? 0)));
  const logits = new Float32Array(POLICY_BUCKETS);
  if (activeModel?.policy_logits?.length) {
    logits.set(activeModel.policy_logits.slice(0, POLICY_BUCKETS));
  }
  const params = paramsBuffer([samples.length, POLICY_BUCKETS], config.learningRate, 1);
  const targetBuffer = storageBuffer(device, targets, GPUBufferUsage.STORAGE);
  let inputLogits = storageBuffer(device, logits, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  let outputLogits = device.createBuffer({
    size: logits.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const paramsGpuBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: POLICY_SHADER }), entryPoint: "train_policy" }
  });
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

function layerParamsBuffer(device, sampleCount, inputSize, outputSize, learningRate) {
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setUint32(8, outputSize, true);
  view.setFloat32(12, learningRate, true);
  return storageBuffer(device, params, GPUBufferUsage.UNIFORM);
}

function outputParamsBuffer(device, sampleCount, inputSize, learningRate) {
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setFloat32(12, learningRate, true);
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

function mse(samples, weights, hiddenWeights) {
  const featureCount = HIDDEN_LAYERS.at(-1);
  let total = 0;
  for (const sample of samples) {
    const hidden = evaluateHidden(projectFeatures(sample.features, PROJECTION_SIZE), HIDDEN_LAYERS, hiddenWeights);
    let prediction = weights[featureCount];
    for (let index = 0; index < hidden.length; index += 1) {
      prediction += hidden[index] * weights[index];
    }
    const error = prediction - sample.label;
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

function predictValue(features, model) {
  const projected = projectFeatures(features, model.projectionSize, model.projectionSeed);
  const hidden = evaluateHidden(projected, model.hiddenLayers, model.hiddenWeights);
  let prediction = model.outputWeights[hidden.length] ?? 0;
  for (let index = 0; index < hidden.length; index += 1) {
    prediction += hidden[index] * model.outputWeights[index];
  }
  return prediction * (model.scale ?? 1);
}

function modelArchitectureMatches(model) {
  return model
    && model.projectionSize === PROJECTION_SIZE
    && model.projectionSeed === PROJECTION_SEED
    && JSON.stringify(model.hiddenLayers) === JSON.stringify(HIDDEN_LAYERS)
    && model.hiddenWeights?.length;
}

function projectFeatures(features, projectionSize, seed = PROJECTION_SEED) {
  const active = [];
  for (let index = 0; index < features.length; index += 1) {
    if (features[index] !== 0) {
      active.push([index, features[index]]);
    }
  }
  const projected = new Float32Array(projectionSize);
  if (active.length === 0) {
    return projected;
  }
  const scale = Math.sqrt(active.length);
  for (const [rawIndex, value] of active) {
    for (let projectionIndex = 0; projectionIndex < projectionSize; projectionIndex += 1) {
      const sign = (projectionHash(rawIndex, projectionIndex, seed) & 1) === 0 ? 1 : -1;
      projected[projectionIndex] += value * sign / scale;
    }
  }
  return projected;
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

function evaluateHidden(input, hiddenLayers, hiddenWeights) {
  let values = input;
  let cursor = 0;
  for (const layerSize of hiddenLayers) {
    const next = new Float32Array(layerSize);
    for (let output = 0; output < layerSize; output += 1) {
      const row = cursor + output * (values.length + 1);
      let sum = hiddenWeights[row + values.length];
      for (let inputIndex = 0; inputIndex < values.length; inputIndex += 1) {
        sum += values[inputIndex] * hiddenWeights[row + inputIndex];
      }
      next[output] = Math.max(0, sum);
    }
    cursor += layerSize * (values.length + 1);
    values = next;
  }
  return values;
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

async function fetchActiveModel() {
  try {
    const response = await fetch("/api/training/model");
    return response.ok ? decodeCompactModel(await response.arrayBuffer()) : null;
  } catch {
    return null;
  }
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
    const db = await openReplayDb();
    return (await idbGet(db, BUFFER_KEY)) ?? [];
  } catch {
    return [];
  }
}

async function saveReplayBuffer(samples) {
  try {
    const db = await openReplayDb();
    await idbPut(db, BUFFER_KEY, samples);
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

function align4(value) {
  return Math.ceil(value / 4) * 4;
}
