import { PROJECT_FEATURES_SHADER, FORWARD_LAYER_SHADER, FORWARD_INDEXED_LAYER_SHADER, FORWARD_OUTPUT_SHADER, OUTPUT_DELTA_SHADER, HIDDEN_DELTA_SHADER, HIDDEN3_DELTA_SHADER, APPLY_LAYER_SHADER, APPLY_INDEXED_LAYER_SHADER, APPLY_OUTPUT_SHADER, POLICY_SHADER } from "./training-shaders.js";
import type { Color } from "./types.js";

const POLICY_BUCKETS = 257;
const PROJECTION_SIZE = 2048;
const PROJECTION_SEED = 2166136261;
const HIDDEN_LAYERS = [1024, 512, 256];
const VALUE_EPOCHS_PER_SUBMIT = 64;
const POLICY_STEPS_PER_SUBMIT = 64;
const PROJECTION_CHUNK_SIZE = 256;
let cachedGpuAdapter: GPUAdapter | null = null;
let cachedGpuDevice: GPUDevice | null = null;
const pipelineCache = new Map<string, GPUComputePipeline>();

interface GpuBufferUsageConstants {
  MAP_READ: number;
  COPY_SRC: number;
  COPY_DST: number;
  UNIFORM: number;
  STORAGE: number;
}

interface GpuMapModeConstants {
  READ: number;
}

const gpuBufferUsage: GpuBufferUsageConstants = (globalThis as unknown as { GPUBufferUsage?: GpuBufferUsageConstants }).GPUBufferUsage ?? {
  MAP_READ: 1,
  COPY_SRC: 4,
  COPY_DST: 8,
  UNIFORM: 64,
  STORAGE: 128
};
const gpuMapMode: GpuMapModeConstants = (globalThis as unknown as { GPUMapMode?: GpuMapModeConstants }).GPUMapMode ?? {
  READ: 1
};

export type TrainingLabelKind = "search" | "outcome" | "distilled" | "unknown" | string;

export interface TrainingSample {
  sideToMove?: Color;
  boardCount?: number;
  positionKey?: string;
  features: number[] | Float32Array;
  label: number;
  labelKind?: TrainingLabelKind;
  labelWeight?: number;
  policy?: number | null;
  pseudo?: boolean;
}

export interface TrainingMetrics {
  phases?: Record<string, number>;
  [key: string]: unknown;
}

export interface TrainingConfig {
  learningRate: number;
  epochs: number;
  batchSize: number;
  validationSplit?: number;
  validationInterval?: number;
  patience: number;
  weightDecay: number;
  metrics?: TrainingMetrics | null;
}

export interface CompactValueModel {
  projectionSize: number;
  projectionSeed: number;
  hiddenLayers: number[];
  hiddenWeights: Float32Array;
  outputWeights: Float32Array;
  policy_logits?: ArrayLike<number> & { slice?: (start?: number, end?: number) => ArrayLike<number> };
  scale?: number;
  bias?: number;
  bytes?: Uint8Array;
}

export interface EncodedCompactModel extends Uint8Array {
  trainingLoss?: number;
  validationLoss?: number;
  bestValidationLoss?: number;
  earlyStopReason?: string;
  labelCounts?: Record<string, number>;
  nonZeroWeights?: number;
  replayBufferSize?: number;
  metrics?: unknown;
}

interface EncodableCompactModel {
  projectionSize: number;
  projectionSeed: number;
  hiddenLayers: number[];
  hiddenWeights: ArrayLike<number>;
  outputWeights: ArrayLike<number>;
  policyLogits?: ArrayLike<number>;
  scale?: number;
  bias?: number;
}

interface TrainedValueWeights {
  featureCount: number | undefined;
  weights: Float32Array;
  hiddenWeights: Float32Array;
  finalWeights: Float32Array;
  finalHiddenWeights: Float32Array;
  loss: number;
  validationLoss: number;
  bestValidationLoss: number;
  earlyStopReason: string;
}

interface ValidationSplit {
  trainIndices: number[];
  validationIndices: number[];
  seed: number;
}

interface PredictionModel {
  projectionSize: number;
  projectionSeed: number;
  hiddenLayers: number[];
  hiddenWeights: Float32Array;
  outputWeights: Float32Array;
  scale?: number;
}

function outputLayerSize(layers: number[] = HIDDEN_LAYERS): number {
  const size = layers[layers.length - 1];
  if (size === undefined) {
    throw new Error("Model must have at least one hidden layer.");
  }
  return size;
}

function previousLayerSize(layers: number[], layerIndex: number, inputSize: number): number {
  return layerIndex === 0 ? inputSize : layers[layerIndex - 1]!;
}

function policyLogitsArray(model: CompactValueModel | null): Float32Array | null {
  if (!model?.policy_logits?.length) {
    return null;
  }
  return new Float32Array(Array.from(model.policy_logits).slice(0, POLICY_BUCKETS));
}

function timed<T>(_metrics: TrainingMetrics | null | undefined, _phase: string, task: () => Promise<T>): Promise<T>;
function timed<T>(_metrics: TrainingMetrics | null | undefined, _phase: string, task: () => T): T;
function timed<T>(_metrics: TrainingMetrics | null | undefined, _phase: string, task: () => T | Promise<T>): T | Promise<T> {
  return task();
}

function labelSourceCounts(samples: TrainingSample[]): Record<string, number> {
  return samples.reduce<Record<string, number>>((counts, sample) => {
    const key = sample.labelKind ?? (sample.pseudo ? "distilled" : "unknown");
    counts[key] = (counts[key] ?? 0) + 1;
    return counts;
  }, {});
}

export async function train(
  samples: TrainingSample[],
  config: TrainingConfig,
  activeModel: CompactValueModel | null,
  progress: (message: Record<string, unknown>) => void
): Promise<EncodedCompactModel> {
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
  let model: EncodedCompactModel = encodeCompactModel({
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

export async function trainValue(
  device: GPUDevice,
  samples: TrainingSample[],
  config: TrainingConfig,
  activeModel: CompactValueModel | null,
  progress: (message: Record<string, unknown>) => void
): Promise<TrainedValueWeights> {
  const firstSample = samples[0];
  if (!firstSample || !samples.every((sample) => sample.features.length === firstSample.features.length)) {
    throw new Error("Training samples have inconsistent feature lengths.");
  }

  const sampleCount = samples.length;
  const labels = new Float32Array(sampleCount);
  const labelWeights = new Float32Array(sampleCount);
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    const sample = samples[sampleIndex]!;
    labels[sampleIndex] = sample.label;
    labelWeights[sampleIndex] = sample.labelWeight ?? 1;
  }
  const split = splitValidationSamples(samples, config.validationSplit);
  const trainIndices = split.trainIndices.length ? split.trainIndices : split.validationIndices;
  const validationIndices = split.validationIndices;
  const batchSize = Math.min(config.batchSize, Math.max(1, trainIndices.length));

  const initialHidden = modelArchitectureMatches(activeModel)
    ? activeModel.hiddenWeights
    : initialHiddenWeights(PROJECTION_SIZE, HIDDEN_LAYERS);
  const layerWeights = splitHiddenWeights(initialHidden, PROJECTION_SIZE, HIDDEN_LAYERS);
  const outputSize = outputLayerSize();
  const outputWeights = new Float32Array(outputSize + 1);
  if (modelArchitectureMatches(activeModel) && activeModel.outputWeights?.length === outputWeights.length) {
    outputWeights.set(activeModel.outputWeights);
  } else {
    outputWeights[outputSize] = labels.reduce((sum, value) => sum + value, 0) / labels.length;
  }

  const featureBuffer = await timed(config.metrics, "projection", () =>
    projectSamplesToBuffer(device, samples, PROJECTION_SIZE, PROJECTION_SEED)
  );
  const labelBuffer = storageBuffer(device, labels, gpuBufferUsage.STORAGE);
  const labelWeightBuffer = storageBuffer(device, labelWeights, gpuBufferUsage.STORAGE);
  const weightBuffers = layerWeights.map((weights) =>
    storageBuffer(device, weights, gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC)
  );
  const outputWeightBuffer = storageBuffer(
    device,
    outputWeights,
    gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  );
  const activationBuffers = HIDDEN_LAYERS.map((layerSize) => device.createBuffer({
    size: align4(batchSize * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  }));
  const deltaBuffers = HIDDEN_LAYERS.map((layerSize) => device.createBuffer({
    size: align4(batchSize * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  }));
  const predictionBuffer = device.createBuffer({
    size: align4(batchSize * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  });
  const outputDeltaBuffer = device.createBuffer({
    size: align4(batchSize * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  });
  const batchesPerSubmit = Math.min(VALUE_EPOCHS_PER_SUBMIT, Math.max(1, config.epochs));
  const validationInterval = Math.max(batchesPerSubmit, config.validationInterval ?? 256);
  const batchIndexBuffers = Array.from({ length: batchesPerSubmit }, () => device.createBuffer({
    size: align4(batchSize * Uint32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_DST
  }));
  const forwardLayerParams = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      batchSize,
      previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE),
      layerSize,
      0
    )
  );
  const applyLayerParams = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      batchSize,
      previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE),
      layerSize,
      config.learningRate,
      config.weightDecay
    )
  );
  const forwardOutputParams = outputParamsBuffer(device, batchSize, outputSize, 0);
  const applyOutputParams = outputParamsBuffer(device, batchSize, outputSize, config.learningRate, config.weightDecay);
  const outputDeltaParams = Array.from({ length: batchesPerSubmit }, () =>
    outputDeltaParamsBuffer(device, batchSize, batchSize)
  );
  const lastHiddenDeltaParams = hiddenDeltaParamsBuffer(
    device,
    batchSize,
    outputLayerSize(),
    0
  );
  const hiddenDeltaParams = HIDDEN_LAYERS.slice(0, -1).map((layerSize, layerIndex) =>
    hiddenDeltaParamsBuffer(device, batchSize, layerSize, HIDDEN_LAYERS[layerIndex + 1]!)
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
      const batchIndexBuffer = batchIndexBuffers[batchSlot]!;
      const epochOrder = shuffledIndices(trainIndices, epoch, split.seed);
      const batchStart = ((epoch - 1) * batchSize) % epochOrder.length;
      const batch = new Uint32Array(batchSize);
      let batchWeight = 0;
      for (let index = 0; index < batchSize; index += 1) {
        batch[index] = epochOrder[(batchStart + index) % epochOrder.length]!;
        batchWeight += Math.max(0, labelWeights[batch[index]!] ?? 1);
      }
      device.queue.writeBuffer(batchIndexBuffer, 0, batch);
      device.queue.writeBuffer(
        outputDeltaParams[batchSlot]!,
        0,
        outputDeltaParamsData(batchSize, batchWeight)
      );
      for (let layerIndex = 0; layerIndex < HIDDEN_LAYERS.length; layerIndex += 1) {
        const inputSize = previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE);
        const inputBuffer = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1]!;
        const outputSizeForLayer = HIDDEN_LAYERS[layerIndex]!;
        encodePipeline(device, encoder, layerIndex === 0 ? forwardIndexedLayerPipeline : forwardLayerPipeline, [
          inputBuffer,
          weightBuffers[layerIndex]!,
          activationBuffers[layerIndex]!,
          forwardLayerParams[layerIndex]!,
          ...(layerIndex === 0 ? [batchIndexBuffer] : [])
        ], Math.ceil(batchSize / 16), Math.ceil(outputSizeForLayer / 16));
      }

      encodePipeline(device, encoder, forwardOutputPipeline, [
        activationBuffers[activationBuffers.length - 1]!,
        outputWeightBuffer,
        predictionBuffer,
        forwardOutputParams
      ], Math.ceil(batchSize / 64));

      encodePipeline(device, encoder, outputDeltaPipeline, [
        predictionBuffer,
        labelBuffer,
        outputDeltaBuffer,
        outputDeltaParams[batchSlot]!,
        batchIndexBuffer,
        labelWeightBuffer
      ], Math.ceil(batchSize / 64));

      const lastLayerIndex = HIDDEN_LAYERS.length - 1;
      encodePipeline(device, encoder, lastHiddenDeltaPipeline, [
        activationBuffers[lastLayerIndex]!,
        outputDeltaBuffer,
        outputWeightBuffer,
        deltaBuffers[lastLayerIndex]!,
        lastHiddenDeltaParams
      ], Math.ceil(batchSize / 16), Math.ceil(HIDDEN_LAYERS[lastLayerIndex]! / 16));

      for (let layerIndex = HIDDEN_LAYERS.length - 2; layerIndex >= 0; layerIndex -= 1) {
        encodePipeline(device, encoder, hiddenDeltaPipeline, [
          activationBuffers[layerIndex]!,
          deltaBuffers[layerIndex + 1]!,
          weightBuffers[layerIndex + 1]!,
          deltaBuffers[layerIndex]!,
          hiddenDeltaParams[layerIndex]!
        ], Math.ceil(batchSize / 16), Math.ceil(HIDDEN_LAYERS[layerIndex]! / 16));
      }

      encodePipeline(device, encoder, applyOutputPipeline, [
        activationBuffers[activationBuffers.length - 1]!,
        outputDeltaBuffer,
        outputWeightBuffer,
        applyOutputParams
      ], Math.ceil((outputSize + 1) / 64));

      for (let layerIndex = HIDDEN_LAYERS.length - 1; layerIndex >= 0; layerIndex -= 1) {
        const inputSize = previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE);
        const inputBuffer = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1]!;
        const outputSizeForLayer = HIDDEN_LAYERS[layerIndex]!;
        encodePipeline(device, encoder, layerIndex === 0 ? applyIndexedLayerPipeline : applyLayerPipeline, [
          inputBuffer,
          deltaBuffers[layerIndex]!,
          weightBuffers[layerIndex]!,
          applyLayerParams[layerIndex]!,
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
      const readLayers: Float32Array[] = [];
      for (let layerIndex = 0; layerIndex < weightBuffers.length; layerIndex += 1) {
        readLayers.push(await readFloats(device, weightBuffers[layerIndex]!, layerWeights[layerIndex]!.byteLength));
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

export async function trainPolicy(
  device: GPUDevice,
  samples: TrainingSample[],
  config: TrainingConfig,
  activeModel: CompactValueModel | null
): Promise<Float32Array> {
  const policySamples = samples.filter((sample): sample is TrainingSample & { policy: number } =>
    sample.labelKind !== "distilled" && Number.isInteger(sample.policy) && (sample.policy ?? -1) >= 0
  );
  if (!policySamples.length) {
    return policyLogitsArray(activeModel) ?? new Float32Array(POLICY_BUCKETS);
  }
  const targets = new Uint32Array(policySamples.map((sample) => Math.min(POLICY_BUCKETS - 1, sample.policy)));
  const labelWeights = new Float32Array(policySamples.map((sample) => Math.max(0, sample.labelWeight ?? 1)));
  const logits = new Float32Array(POLICY_BUCKETS);
  const activePolicy = policyLogitsArray(activeModel);
  if (activePolicy) {
    logits.set(activePolicy);
  }
  const params = paramsBuffer([policySamples.length, POLICY_BUCKETS], config.learningRate, 1);
  const targetBuffer = storageBuffer(device, targets, gpuBufferUsage.STORAGE);
  const labelWeightBuffer = storageBuffer(device, labelWeights, gpuBufferUsage.STORAGE);
  let inputLogits = storageBuffer(device, logits, gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC);
  let outputLogits = device.createBuffer({
    size: logits.byteLength,
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  });
  const paramsGpuBuffer = storageBuffer(device, params, gpuBufferUsage.UNIFORM);
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
        [targetBuffer, inputLogits, outputLogits, paramsGpuBuffer, labelWeightBuffer],
        Math.ceil(POLICY_BUCKETS / 64)
      );
      [inputLogits, outputLogits] = [outputLogits, inputLogits];
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  return readFloats(device, inputLogits, logits.byteLength);
}

export function runPipeline(device: GPUDevice, pipeline: GPUComputePipeline, buffers: GPUBuffer[], workgroupsX: number, workgroupsY = 1): void {
  const encoder = device.createCommandEncoder();
  encodePipeline(device, encoder, pipeline, buffers, workgroupsX, workgroupsY);
  device.queue.submit([encoder.finish()]);
}

export function encodePipeline(device: GPUDevice, encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, buffers: GPUBuffer[], workgroupsX: number, workgroupsY = 1): void {
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

export function paramsBuffer([first, second]: [number, number], learningRate: number, fourth: number): ArrayBuffer {
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, first, true);
  view.setUint32(4, second, true);
  view.setFloat32(8, learningRate, true);
  view.setUint32(12, fourth, true);
  return params;
}

export function layerParamsBuffer(device: GPUDevice, sampleCount: number, inputSize: number, outputSize: number, learningRate: number, weightDecay = 0): GPUBuffer {
  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setUint32(8, outputSize, true);
  view.setFloat32(12, learningRate, true);
  view.setFloat32(16, weightDecay, true);
  return storageBuffer(device, params, gpuBufferUsage.UNIFORM);
}

export function outputParamsBuffer(device: GPUDevice, sampleCount: number, inputSize: number, learningRate: number, weightDecay = 0): GPUBuffer {
  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setFloat32(12, learningRate, true);
  view.setFloat32(16, weightDecay, true);
  return storageBuffer(device, params, gpuBufferUsage.UNIFORM);
}

export function outputDeltaParamsBuffer(device: GPUDevice, sampleCount: number, totalWeight = sampleCount): GPUBuffer {
  return storageBuffer(device, outputDeltaParamsData(sampleCount, totalWeight), gpuBufferUsage.UNIFORM);
}

function outputDeltaParamsData(sampleCount: number, totalWeight: number): ArrayBuffer {
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setFloat32(4, Math.max(0, totalWeight), true);
  return params;
}

export function hiddenDeltaParamsBuffer(device: GPUDevice, sampleCount: number, currentSize: number, nextSize: number): GPUBuffer {
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, currentSize, true);
  view.setUint32(8, nextSize, true);
  return storageBuffer(device, params, gpuBufferUsage.UNIFORM);
}

export function storageBuffer(device: GPUDevice, data: ArrayBuffer | ArrayBufferView, usage: number): GPUBuffer {
  const bytes = data instanceof ArrayBuffer
    ? data
    : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const buffer = device.createBuffer({
    size: align4(bytes.byteLength),
    usage: usage | gpuBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

export async function projectSamplesToBuffer(device: GPUDevice, samples: TrainingSample[], projectionSize: number, seed = PROJECTION_SEED): Promise<GPUBuffer> {
  const inputSize = featureLength(samples);
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const projectedBytes = samples.length * projectionSize * Float32Array.BYTES_PER_ELEMENT;
  if (projectedBytes > maxBindingSize) {
    throw new Error(`Projected replay buffer exceeds this WebGPU device's storage binding limit (${formatBytes(projectedBytes)} > ${formatBytes(maxBindingSize)}). Reduce replay buffer or projection size.`);
  }
  const projectedBuffer = device.createBuffer({
    size: align4(projectedBytes),
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC | gpuBufferUsage.COPY_DST
  });
  const pipeline = await createComputePipelineChecked(device, "project_features", PROJECT_FEATURES_SHADER, "project_features");
  for (let offset = 0; offset < samples.length; offset += PROJECTION_CHUNK_SIZE) {
    const chunkSamples = samples.slice(offset, offset + PROJECTION_CHUNK_SIZE);
    const rawFeatures = new Float32Array(chunkSamples.length * inputSize);
    for (let sampleIndex = 0; sampleIndex < chunkSamples.length; sampleIndex += 1) {
      rawFeatures.set(chunkSamples[sampleIndex]!.features, sampleIndex * inputSize);
    }
    if (rawFeatures.byteLength > maxBindingSize) {
      throw new Error(`Projection chunk exceeds this WebGPU device's storage binding limit (${formatBytes(rawFeatures.byteLength)} > ${formatBytes(maxBindingSize)}). Reduce batch size or feature size.`);
    }
    const rawBuffer = storageBuffer(device, rawFeatures, gpuBufferUsage.STORAGE);
    const chunkProjectedBuffer = device.createBuffer({
      size: align4(chunkSamples.length * projectionSize * Float32Array.BYTES_PER_ELEMENT),
      usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
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

export function splitValidationSamples(samples: TrainingSample[], validationSplit = 0): ValidationSplit {
  const trainIndices: number[] = [];
  const validationIndices: number[] = [];
  const threshold = Math.floor(validationSplit * 10000);
  const seed = samples.reduce((hash, sample, index) => {
    hash ^= stableSampleHash(sample, index);
    return Math.imul(hash, 16777619) >>> 0;
  }, 2166136261);
  for (let index = 0; index < samples.length; index += 1) {
    const bucket = stableSampleHash(samples[index]!, index) % 10000;
    if (threshold > 0 && bucket < threshold) {
      validationIndices.push(index);
    } else {
      trainIndices.push(index);
    }
  }
  if (!trainIndices.length && validationIndices.length > 1) {
    trainIndices.push(validationIndices.pop()!);
  }
  return { trainIndices, validationIndices, seed };
}

export function stableSampleHash(sample: TrainingSample, _index: number): number {
  let hash = 2166136261;
  const text = sample.positionKey
    ? `${sample.positionKey}|${sample.sideToMove ?? ""}|${sample.boardCount ?? 0}`
    : `${featureFingerprint(sample.features)}|${sample.sideToMove ?? ""}|${sample.boardCount ?? 0}`;
  for (let offset = 0; offset < text.length; offset += 1) {
    hash ^= text.charCodeAt(offset);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function featureFingerprint(features: number[] | Float32Array): string {
  let hash = 2166136261;
  for (let index = 0; index < features.length; index += 1) {
    const value = features[index] ?? 0;
    if (value === 0) { continue; }
    hash ^= index;
    hash = Math.imul(hash, 16777619) >>> 0;
    hash ^= Math.round(value * 1024);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash.toString(16);
}

export function shuffledIndices(indices: number[], epoch: number, seed: number): number[] {
  const result = indices.slice();
  let state = (seed ^ Math.imul(epoch, 2654435761)) >>> 0;
  for (let index = result.length - 1; index > 0; index -= 1) {
    state = xorshift32(state);
    const swapIndex = state % (index + 1);
    const current = result[index]!;
    result[index] = result[swapIndex]!;
    result[swapIndex] = current;
  }
  return result;
}

export function xorshift32(value: number): number {
  let state = value >>> 0;
  state ^= state << 13;
  state ^= state >>> 17;
  state ^= state << 5;
  return state >>> 0;
}

export function indexSamples(samples: TrainingSample[], indices: number[]): TrainingSample[] {
  return indices.map((index) => samples[index]).filter((sample): sample is TrainingSample => Boolean(sample));
}

export function featureLength(samples: TrainingSample[]): number {
  const length = samples[0]?.features?.length;
  if (!length || !samples.every((sample) => sample.features.length === length)) {
    throw new Error("Training samples have inconsistent feature lengths.");
  }
  return length;
}

export function projectionParamsBuffer(device: GPUDevice, sampleCount: number, inputSize: number, projectionSize: number, seed: number): GPUBuffer {
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setUint32(8, projectionSize, true);
  view.setUint32(12, seed >>> 0, true);
  return storageBuffer(device, params, gpuBufferUsage.UNIFORM);
}

export async function readFloats(device: GPUDevice, buffer: GPUBuffer, byteLength: number): Promise<Float32Array<ArrayBuffer>> {
  const readBuffer = device.createBuffer({
    size: align4(byteLength),
    usage: gpuBufferUsage.COPY_DST | gpuBufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(gpuMapMode.READ);
  const mapped = readBuffer.getMappedRange().slice(0, byteLength) as ArrayBuffer;
  const copy = new Float32Array(mapped);
  readBuffer.unmap();
  return copy;
}

export async function predictionLossOnGpu(device: GPUDevice, samples: TrainingSample[], model: PredictionModel): Promise<number> {
  const predictions = await predictValuesOnGpu(device, samples, model);
  let total = 0;
  let totalWeight = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const error = predictions[index]! - samples[index]!.label;
    const weight = Math.max(0, samples[index]!.labelWeight ?? 1);
    total += weight * error * error;
    totalWeight += weight;
  }
  return totalWeight > 0 ? total / totalWeight : 0;
}

export function splitHiddenWeights(hiddenWeights: Float32Array, inputSize: number, hiddenLayers: number[]): Float32Array[] {
  const layers: Float32Array[] = [];
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

export function concatFloat32(arrays: Float32Array[]): Float32Array {
  const length = arrays.reduce((sum, array) => sum + array.length, 0);
  const result = new Float32Array(length);
  let cursor = 0;
  for (const array of arrays) {
    result.set(array, cursor);
    cursor += array.length;
  }
  return result;
}

export function countNonZero(values: ArrayLike<number>): number {
  let count = 0;
  for (let index = 0; index < values.length; index += 1) {
    if ((values[index] ?? 0) !== 0) {
      count += 1;
    }
  }
  return count;
}

export async function predictValues(samples: TrainingSample[], model: CompactValueModel | null): Promise<number[]> {
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

export async function predictValuesOnGpu(device: GPUDevice, samples: TrainingSample[], model: CompactValueModel): Promise<Float32Array> {
  const sampleCount = samples.length;
  const featureBuffer = await projectSamplesToBuffer(
    device,
    samples,
    model.projectionSize,
    model.projectionSeed
  );
  const hiddenLayers = model.hiddenLayers;
  const finalHiddenSize = outputLayerSize(hiddenLayers);
  const layerWeights = splitHiddenWeights(model.hiddenWeights, model.projectionSize, hiddenLayers);
  const weightBuffers = layerWeights.map((weights) => storageBuffer(device, weights, gpuBufferUsage.STORAGE));
  const outputWeightBuffer = storageBuffer(device, model.outputWeights, gpuBufferUsage.STORAGE);
  const activationBuffers = hiddenLayers.map((layerSize) => device.createBuffer({
    size: align4(sampleCount * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  }));
  const predictionBuffer = device.createBuffer({
    size: align4(sampleCount * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  });
  const forwardLayerParams = hiddenLayers.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      sampleCount,
      previousLayerSize(hiddenLayers, layerIndex, model.projectionSize),
      layerSize,
      0
    )
  );
  const forwardOutputParams = outputParamsBuffer(device, sampleCount, finalHiddenSize, 0);
  const forwardLayerPipeline = await createComputePipelineChecked(device, "forward_layer", FORWARD_LAYER_SHADER, "forward_layer");
  const forwardOutputPipeline = await createComputePipelineChecked(device, "forward_output", FORWARD_OUTPUT_SHADER, "forward_output");
  const encoder = device.createCommandEncoder();
  for (let layerIndex = 0; layerIndex < hiddenLayers.length; layerIndex += 1) {
    const inputBuffer = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1]!;
    encodePipeline(device, encoder, forwardLayerPipeline, [
      inputBuffer,
      weightBuffers[layerIndex]!,
      activationBuffers[layerIndex]!,
      forwardLayerParams[layerIndex]!
    ], Math.ceil(sampleCount / 16), Math.ceil(hiddenLayers[layerIndex]! / 16));
  }
  encodePipeline(device, encoder, forwardOutputPipeline, [
    activationBuffers[activationBuffers.length - 1]!,
    outputWeightBuffer,
    predictionBuffer,
    forwardOutputParams
  ], Math.ceil(sampleCount / 64));
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const predictions = await readFloats(device, predictionBuffer, sampleCount * Float32Array.BYTES_PER_ELEMENT);
  const scale = model.scale ?? 1;
  for (let index = 0; index < predictions.length; index += 1) {
    predictions[index] = (predictions[index] ?? 0) * scale;
  }
  return predictions;
}

export function modelArchitectureMatches(model: CompactValueModel | null | undefined): model is CompactValueModel {
  return Boolean(model
    && model.projectionSize === PROJECTION_SIZE
    && model.projectionSeed === PROJECTION_SEED
    && JSON.stringify(model.hiddenLayers) === JSON.stringify(HIDDEN_LAYERS)
    && model.hiddenWeights?.length);
}

export function projectionHash(rawIndex: number, projectionIndex: number, seed: number): number {
  let hash = (seed ^ rawIndex) >>> 0;
  hash = Math.imul(hash, 16777619) >>> 0;
  hash = (hash ^ projectionIndex) >>> 0;
  hash = Math.imul(hash, 16777619) >>> 0;
  hash = (hash ^ (hash >>> 16)) >>> 0;
  return hash;
}

export function initialHiddenWeights(inputSize: number, hiddenLayers: number[]): Float32Array {
  const weights: number[] = [];
  let previous = inputSize;
  for (let layerIndex = 0; layerIndex < hiddenLayers.length; layerIndex += 1) {
    const layerSize = hiddenLayers[layerIndex]!;
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

export function encodeCompactModel(model: EncodableCompactModel): EncodedCompactModel {
  const hiddenWeights = Array.from(model.hiddenWeights);
  const outputWeights = Array.from(model.outputWeights);
  const scale = model.scale ?? 1;
  const bias = model.bias ?? 0;
  const floats = [scale, bias, ...hiddenWeights, ...outputWeights];
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
  cursor = writeF32(view, cursor, scale);
  cursor = writeF32(view, cursor, bias);
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
  return new Uint8Array(buffer) as EncodedCompactModel;
}

export function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

export function writeU32(view: DataView, offset: number, value: number): number {
  view.setUint32(offset, value, true);
  return offset + 4;
}

export function writeF32(view: DataView, offset: number, value: number): number {
  view.setFloat32(offset, value, true);
  return offset + 4;
}

export function byteArraysEqual(left: Uint8Array | null | undefined, right: Uint8Array | null | undefined): boolean {
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

export function decodeCompactModel(buffer: ArrayBuffer): CompactValueModel | null {
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

export async function getGpuDevice(): Promise<GPUDevice | null> {
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

export async function requestHighLimitDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  const requiredLimits: Record<string, number> = {};
  for (const key of ["maxStorageBufferBindingSize", "maxBufferSize"] as const) {
    const value = adapter.limits[key];
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

export async function createComputePipelineChecked(device: GPUDevice, label: string, code: string, entryPoint: string): Promise<GPUComputePipeline> {
  const cacheKey = `${label}:${entryPoint}`;
  const cached = pipelineCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const module = device.createShaderModule({ label: `${label}.module`, code });
  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message: GPUCompilationMessage) => message.type === "error");
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

export function formatShaderErrors(label: string, errors: GPUCompilationMessage[]): string {
  return `${label} shader compilation failed: ${errors.map((error) =>
    `line ${error.lineNum ?? "?"}, column ${error.linePos ?? "?"}: ${error.message}`
  ).join("; ")}`;
}

export function formatBytes(bytes: number): string {
  const mib = bytes / (1024 * 1024);
  return `${mib.toFixed(mib >= 10 ? 0 : 1)} MiB`;
}

export function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}
