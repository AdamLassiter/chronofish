import { PROJECT_FEATURES_SHADER, FORWARD_LAYER_SHADER, FORWARD_INDEXED_LAYER_SHADER, FORWARD_OUTPUT_SHADER, OUTPUT_DELTA_SHADER, HIDDEN_DELTA_SHADER, HIDDEN3_DELTA_SHADER, APPLY_LAYER_SHADER, APPLY_INDEXED_LAYER_SHADER, APPLY_OUTPUT_SHADER, POLICY_SHADER, POLICY_LOSS_SHADER, REDUCE_LOSS_SHADER } from "./training-shaders.js";
import { POLICY_BUCKETS } from "./training-policy.js";
import { trainingLabelPriority } from "./training-replay.js";
import { byteArraysEqual, compactModelIsFinite, encodeCompactModel } from "./training-gpu-model.js";
import type { CompactValueModel, EncodableCompactModel, EncodedCompactModel } from "./training-gpu-model.js";
import type { Color } from "./types.js";
export { byteArraysEqual, compactModelIsFinite, decodeCompactModel, encodeCompactModel, writeAscii, writeF32, writeU32 } from "./training-gpu-model.js";
export type { CompactValueModel, EncodableCompactModel, EncodedCompactModel } from "./training-gpu-model.js";

const PROJECTION_SIZE = 2048;
const PROJECTION_SEED = 2166136261;
export const VALUE_SCORE_SCALE = 20_000;
const HIDDEN_LAYERS = [1024, 512, 256];
const VALUE_EPOCHS_PER_SUBMIT = 64;
const POLICY_STEPS_PER_SUBMIT = 64;
const TILED_TRAINING_MIN_BATCH = 16;
const CPU_PREDICTION_MAX_BATCH = 4;
const MIN_HIDDEN_TRAINING_POSITIONS = 256;
const CPU_HEAD_TRAINING_MAX_POSITIONS = 32;
const OPTIMIZER_MOMENTUM = 0.9;
const MIN_POLICY_WORKING_SET_FRACTION = 0.25;
const PROJECTION_CHUNK_SIZE = 256;
const PROJECTION_TEMPORARY_BUDGET = 128 * 1024 * 1024;
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
  baseLabelWeight?: number;
  labelMass?: number;
  observationCount?: number;
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

interface TrainedValueWeights {
  featureCount: number | undefined;
  weights: Float32Array;
  hiddenWeights: Float32Array;
  loss: number;
  initialValidationLoss: number;
  validationLoss: number;
  bestValidationLoss: number;
  checkpointImproved: boolean;
  hiddenLayersTrained: boolean;
  earlyStopReason: string;
  policyFeatureBuffer: GPUBuffer;
  resources: GPUBuffer[];
}

interface TrainedPolicyWeights {
  weights: Float32Array;
  initialValidationLoss: number;
  validationLoss: number;
  bestValidationLoss: number;
  checkpointImproved: boolean;
}

interface ValidationSplit {
  trainIndices: number[];
  validationIndices: number[];
  seed: number;
}

export interface SparseProjectionFeatures {
  offsets: Uint32Array;
  indices: Uint32Array;
  values: Float32Array;
  byteLength: number;
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
  const logits = model?.policyLogits ?? model?.policy_logits;
  if (!logits?.length) {
    return null;
  }
  return new Float32Array(Array.from(logits).slice(0, POLICY_BUCKETS));
}

function policyWeightsArray(model: CompactValueModel | null, inputSize: number): Float32Array | null {
  const expected = POLICY_BUCKETS * (inputSize + 1);
  if (model?.policyWeights?.length === expected) {
    return new Float32Array(model.policyWeights);
  }
  const logits = policyLogitsArray(model);
  if (!logits) {
    return null;
  }
  const weights = new Float32Array(expected);
  for (let bucket = 0; bucket < POLICY_BUCKETS; bucket += 1) {
    weights[bucket * (inputSize + 1) + inputSize] = logits[bucket] ?? 0;
  }
  return weights;
}

function timed<T>(metrics: TrainingMetrics | null | undefined, phase: string, task: () => Promise<T>): Promise<T>;
function timed<T>(metrics: TrainingMetrics | null | undefined, phase: string, task: () => T): T;
function timed<T>(metrics: TrainingMetrics | null | undefined, phase: string, task: () => T | Promise<T>): T | Promise<T> {
  if (!metrics) {
    return task();
  }
  const startedAt = performance.now();
  const record = (): void => {
    metrics.phases ??= {};
    metrics.phases[phase] = (metrics.phases[phase] ?? 0) + performance.now() - startedAt;
  };
  try {
    const result = task();
    if (isPromiseLike(result)) {
      return result.finally(record);
    }
    record();
    return result;
  } catch (error) {
    record();
    throw error;
  }
}

function isPromiseLike<T>(value: T | Promise<T>): value is Promise<T> {
  return Boolean(value && typeof (value as Promise<T>).then === "function");
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
  if (!samples?.length) {
    throw new Error("No samples were collected.");
  }
  if (uniqueTrainingPositionCount(samples, samples.map((_, index) => index)) <= CPU_HEAD_TRAINING_MAX_POSITIONS) {
    return timed(config.metrics, "cpuHeadTrain", () =>
      trainHeadsOnCpu(samples, config, activeModel, progress)
    );
  }
  if (!globalThis.navigator?.gpu) {
    throw new Error("WebGPU is unavailable in this browser.");
  }

  const device = await getGpuDevice();
  if (!device) {
    throw new Error("No WebGPU adapter is available.");
  }
  const trainingSamples = selectTrainingWorkingSet(samples, device);
  const value = await timed(config.metrics, "valueTrain", () =>
    trainValue(device, trainingSamples, config, activeModel, progress)
  );
  try {
    const policy = await timed(config.metrics, "policyTrain", () =>
      trainPolicy(
        device,
        trainingSamples,
        config,
        activeModel,
        value.policyFeatureBuffer,
        outputLayerSize()
      )
    );
    const model: EncodedCompactModel = encodeCompactModel({
      projectionSize: PROJECTION_SIZE,
      projectionSeed: PROJECTION_SEED,
      hiddenLayers: HIDDEN_LAYERS,
      hiddenWeights: value.hiddenWeights,
      outputWeights: value.weights,
      policyWeights: policy.weights,
      scale: 1,
      bias: 0,
      outputActivation: "tanh"
    });
    model.trainingLoss = value.loss;
    model.initialValidationLoss = value.initialValidationLoss;
    model.validationLoss = value.validationLoss;
    model.bestValidationLoss = value.bestValidationLoss;
    model.initialPolicyValidationLoss = policy.initialValidationLoss;
    model.policyValidationLoss = policy.validationLoss;
    model.bestPolicyValidationLoss = policy.bestValidationLoss;
    model.valueCheckpointImproved = value.checkpointImproved;
    model.policyCheckpointImproved = policy.checkpointImproved;
    model.modelChanged = !activeModel?.bytes || !byteArraysEqual(model, activeModel.bytes);
    model.earlyStopReason = value.earlyStopReason;
    model.labelCounts = labelSourceCounts(trainingSamples);
    model.nonZeroWeights = countNonZero(value.weights) + countNonZero(value.hiddenWeights) + countNonZero(policy.weights);
    model.replayBufferSize = samples.length;
    model.trainingSampleCount = trainingSamples.length;
    model.policyTrainingSampleCount = trainingSamples.filter(hasPolicyTrainingTarget).length;
    model.hiddenLayersTrained = value.hiddenLayersTrained;
    return model;
  } finally {
    destroyBuffers(value.resources);
  }
}

export function trainHeadsOnCpu(
  samples: TrainingSample[],
  config: TrainingConfig,
  activeModel: CompactValueModel | null,
  progress: (message: Record<string, unknown>) => void = () => {}
): EncodedCompactModel {
  const split = splitValidationSamples(samples, config.validationSplit);
  const trainIndices = split.trainIndices.length ? split.trainIndices : split.validationIndices;
  const validationIndices = split.validationIndices;
  const trainGroups = groupTrainingIndicesByPosition(samples, trainIndices);
  const batchSize = Math.min(config.batchSize, Math.max(1, trainIndices.length));
  const hiddenWeights = modelArchitectureMatches(activeModel)
    ? activeModel.hiddenWeights.slice()
    : initialHiddenWeights(PROJECTION_SIZE, HIDDEN_LAYERS);
  const hiddenFeatures = samples.map((sample) =>
    hiddenFeaturesOnCpu(sample, PROJECTION_SIZE, PROJECTION_SEED, HIDDEN_LAYERS, hiddenWeights)
  );
  const outputSize = outputLayerSize();
  const outputWeights = modelArchitectureMatches(activeModel)
    && activeModel.outputWeights.length === outputSize + 1
    ? activeModel.outputWeights.slice()
    : new Float32Array(outputSize + 1);
  if (!modelArchitectureMatches(activeModel)) {
    outputWeights[outputSize] = inverseTanh(
      samples.reduce((sum, sample) => sum + sample.label, 0) / samples.length
    );
  }
  const labelWeights = Float32Array.from(samples, (sample) => Math.max(0, sample.labelWeight ?? 1));
  const lossIndices = validationIndices.length ? validationIndices : trainIndices;
  const initialValidationLoss = valueHeadLossOnCpu(hiddenFeatures, samples, outputWeights, lossIndices);
  let bestValidationLoss = initialValidationLoss;
  let lastValidationLoss = initialValidationLoss;
  let lastTrainLoss = initialValidationLoss;
  let bestOutputWeights = outputWeights.slice();
  const outputVelocity = new Float32Array(outputWeights.length);
  let checkpointImproved = false;
  let epochsWithoutImprovement = 0;
  let earlyStopReason = "";
  const batchIndices = new Uint32Array(batchSize);
  const validationInterval = Math.max(1, Math.min(config.epochs, config.validationInterval ?? 256));

  for (let epoch = 1; epoch <= config.epochs; epoch += 1) {
    const batchWeight = fillGroupedTrainingBatchIndices(batchIndices, trainGroups, epoch, split.seed, labelWeights);
    applyValueHeadGradientOnCpu(
      hiddenFeatures,
      samples,
      outputWeights,
      batchIndices,
      batchWeight,
      config.learningRate,
      config.weightDecay,
      outputVelocity
    );
    if (epoch % validationInterval !== 0 && epoch < config.epochs) {
      continue;
    }
    lastTrainLoss = valueHeadLossOnCpu(hiddenFeatures, samples, outputWeights, trainIndices);
    lastValidationLoss = validationIndices.length
      ? valueHeadLossOnCpu(hiddenFeatures, samples, outputWeights, validationIndices)
      : lastTrainLoss;
    if (lastValidationLoss + 1e-6 < bestValidationLoss) {
      bestValidationLoss = lastValidationLoss;
      bestOutputWeights = outputWeights.slice();
      checkpointImproved = true;
      epochsWithoutImprovement = 0;
    } else {
      epochsWithoutImprovement += 1;
    }
    progress({
      epoch,
      loss: lastTrainLoss,
      validationLoss: lastValidationLoss,
      bestValidationLoss,
      initialValidationLoss,
      checkpointImproved,
      epochsWithoutImprovement,
      batchSize,
      replaySize: samples.length,
      hiddenLayersTrained: false,
      trainingBackend: "cpu-heads",
      labelCounts: labelSourceCounts(samples)
    });
    if (epochsWithoutImprovement >= config.patience) {
      earlyStopReason = `validation did not improve for ${config.patience} checks`;
      break;
    }
  }

  const policy = trainPolicyHeadOnCpu(samples, hiddenFeatures, config, activeModel, split);
  const model: EncodedCompactModel = encodeCompactModel({
    projectionSize: PROJECTION_SIZE,
    projectionSeed: PROJECTION_SEED,
    hiddenLayers: HIDDEN_LAYERS,
    hiddenWeights,
    outputWeights: bestOutputWeights,
    policyWeights: policy.weights,
    scale: 1,
    bias: 0,
    outputActivation: "tanh"
  });
  model.trainingLoss = lastTrainLoss;
  model.initialValidationLoss = initialValidationLoss;
  model.validationLoss = lastValidationLoss;
  model.bestValidationLoss = bestValidationLoss;
  model.initialPolicyValidationLoss = policy.initialValidationLoss;
  model.policyValidationLoss = policy.validationLoss;
  model.bestPolicyValidationLoss = policy.bestValidationLoss;
  model.valueCheckpointImproved = checkpointImproved;
  model.policyCheckpointImproved = policy.checkpointImproved;
  model.modelChanged = !activeModel?.bytes || !byteArraysEqual(model, activeModel.bytes);
  model.earlyStopReason = earlyStopReason;
  model.labelCounts = labelSourceCounts(samples);
  model.nonZeroWeights = countNonZero(bestOutputWeights) + countNonZero(hiddenWeights) + countNonZero(policy.weights);
  model.replayBufferSize = samples.length;
  model.trainingSampleCount = samples.length;
  model.policyTrainingSampleCount = samples.filter(hasPolicyTrainingTarget).length;
  model.hiddenLayersTrained = false;
  return model;
}

export function selectTrainingWorkingSet(samples: TrainingSample[], device: GPUDevice): TrainingSample[] {
  const projectedBytes = samples.length * PROJECTION_SIZE * Float32Array.BYTES_PER_ELEMENT;
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  if (projectedBytes <= maxBindingSize) {
    return samples;
  }
  const maxProjectedSamples = Math.max(1, Math.floor(maxBindingSize / (PROJECTION_SIZE * Float32Array.BYTES_PER_ELEMENT)));
  const target = Math.max(1, Math.min(samples.length, maxProjectedSamples));
  const ranked = samples
    .map((sample, index) => ({
      sample,
      index,
      priority: trainingSamplePriority(sample, index, samples.length)
    }))
    .sort((left, right) => right.priority - left.priority || right.index - left.index);
  const selected = ranked.slice(0, target);
  const selectedIndices = new Set(selected.map((entry) => entry.index));
  const availablePolicyCount = ranked.reduce(
    (count, entry) => count + Number(hasPolicyTrainingTarget(entry.sample)),
    0
  );
  const requiredPolicyCount = Math.min(
    availablePolicyCount,
    Math.max(1, Math.ceil(target * MIN_POLICY_WORKING_SET_FRACTION))
  );
  let selectedPolicyCount = selected.reduce(
    (count, entry) => count + Number(hasPolicyTrainingTarget(entry.sample)),
    0
  );
  if (selectedPolicyCount < requiredPolicyCount) {
    const policyReplacements = ranked.filter(
      (entry) => hasPolicyTrainingTarget(entry.sample) && !selectedIndices.has(entry.index)
    );
    for (const replacement of policyReplacements) {
      let replaceIndex = -1;
      for (let index = selected.length - 1; index >= 0; index -= 1) {
        if (!hasPolicyTrainingTarget(selected[index]!.sample)) {
          replaceIndex = index;
          break;
        }
      }
      if (replaceIndex < 0 || selectedPolicyCount >= requiredPolicyCount) {
        break;
      }
      selectedIndices.delete(selected[replaceIndex]!.index);
      selected[replaceIndex] = replacement;
      selectedIndices.add(replacement.index);
      selectedPolicyCount += 1;
    }
  }
  return selected
    .sort((left, right) => left.index - right.index)
    .map((entry) => entry.sample);
}

function trainingSamplePriority(sample: TrainingSample, index: number, total: number): number {
  const recency = total > 1 ? index / (total - 1) : 1;
  return trainingLabelPriority(sample.labelKind, sample.pseudo) +
    Math.max(0, sample.labelWeight ?? 1) +
    recency * 0.25;
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
  const trainGroups = groupTrainingIndicesByPosition(samples, trainIndices);
  const hiddenLayersTrained = uniqueTrainingPositionCount(samples, trainIndices) >= MIN_HIDDEN_TRAINING_POSITIONS;
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
    outputWeights[outputSize] = inverseTanh(
      labels.reduce((sum, value) => sum + value, 0) / labels.length
    );
  }

  const featureBuffer = await timed(config.metrics, "projection", () =>
    projectSamplesToBuffer(device, samples, PROJECTION_SIZE, PROJECTION_SEED)
  );
  const labelBuffer = storageBuffer(device, labels, gpuBufferUsage.STORAGE);
  const labelWeightBuffer = storageBuffer(device, labelWeights, gpuBufferUsage.STORAGE);
  const weightBuffers = layerWeights.map((weights) =>
    storageBuffer(device, weights, gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC)
  );
  const velocityBuffers = hiddenLayersTrained
    ? layerWeights.map((weights) => zeroStorageBuffer(device, weights.byteLength))
    : [];
  const outputWeightBuffer = storageBuffer(
    device,
    outputWeights,
    gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  );
  const outputVelocityBuffer = zeroStorageBuffer(device, outputWeights.byteLength);
  const bestWeightBuffers = layerWeights.map((weights) =>
    storageBuffer(device, weights, gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC | gpuBufferUsage.COPY_DST)
  );
  const bestOutputWeightBuffer = storageBuffer(
    device,
    outputWeights,
    gpuBufferUsage.COPY_SRC | gpuBufferUsage.COPY_DST
  );
  const activationBuffers = HIDDEN_LAYERS.map((layerSize) => device.createBuffer({
    size: align4(batchSize * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  }));
  const deltaBuffers = hiddenLayersTrained
    ? HIDDEN_LAYERS.map((layerSize) => device.createBuffer({
      size: align4(batchSize * layerSize * Float32Array.BYTES_PER_ELEMENT),
      usage: gpuBufferUsage.STORAGE
    }))
    : [];
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
  const batchIndices = new Uint32Array(batchSize);
  const forwardLayerParams = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      batchSize,
      previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE),
      layerSize,
      0
    )
  );
  const applyLayerParams = hiddenLayersTrained
    ? HIDDEN_LAYERS.map((layerSize, layerIndex) =>
      layerParamsBuffer(
        device,
        batchSize,
        previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE),
        layerSize,
        config.learningRate,
        config.weightDecay,
        OPTIMIZER_MOMENTUM
      )
    )
    : [];
  const forwardOutputParams = outputParamsBuffer(device, batchSize, outputSize, 0);
  const applyOutputParams = outputParamsBuffer(
    device,
    batchSize,
    outputSize,
    config.learningRate,
    config.weightDecay,
    OPTIMIZER_MOMENTUM
  );
  const outputDeltaParams = Array.from({ length: batchesPerSubmit }, () =>
    outputDeltaParamsBuffer(device, batchSize, batchSize)
  );
  const lastHiddenDeltaParams = hiddenLayersTrained
    ? hiddenDeltaParamsBuffer(device, batchSize, outputLayerSize(), 0)
    : null;
  const hiddenDeltaParams = hiddenLayersTrained
    ? HIDDEN_LAYERS.slice(0, -1).map((layerSize, layerIndex) =>
      hiddenDeltaParamsBuffer(device, batchSize, layerSize, HIDDEN_LAYERS[layerIndex + 1]!)
    )
    : [];

  const denseKernelSuffix = batchSize >= TILED_TRAINING_MIN_BATCH ? "tiled" : "naive";
  const forwardLayerEntryPoint = denseKernelEntryPoint("forward_layer", batchSize);
  const applyLayerEntryPoint = denseKernelEntryPoint("apply_layer", batchSize);
  const hiddenDeltaEntryPoint = denseKernelEntryPoint("hidden_delta", batchSize);
  const forwardIndexedLayerPipeline = await createComputePipelineChecked(device, `forward_indexed_layer_${denseKernelSuffix}`, FORWARD_INDEXED_LAYER_SHADER, forwardLayerEntryPoint);
  const forwardLayerPipeline = await createComputePipelineChecked(device, `forward_layer_${denseKernelSuffix}`, FORWARD_LAYER_SHADER, forwardLayerEntryPoint);
  const forwardOutputPipeline = await createComputePipelineChecked(device, "forward_output", FORWARD_OUTPUT_SHADER, "forward_output");
  const reduceLossPipeline = await createComputePipelineChecked(device, "reduce_loss", REDUCE_LOSS_SHADER, "reduce_loss");
  const outputDeltaPipeline = await createComputePipelineChecked(device, "output_delta", OUTPUT_DELTA_SHADER, "output_delta");
  const lastHiddenDeltaPipeline = hiddenLayersTrained
    ? await createComputePipelineChecked(device, "hidden3_delta", HIDDEN3_DELTA_SHADER, "hidden3_delta")
    : null;
  const hiddenDeltaPipeline = hiddenLayersTrained
    ? await createComputePipelineChecked(device, `hidden_delta_${denseKernelSuffix}`, HIDDEN_DELTA_SHADER, hiddenDeltaEntryPoint)
    : null;
  const applyIndexedLayerPipeline = hiddenLayersTrained
    ? await createComputePipelineChecked(device, `apply_indexed_layer_${denseKernelSuffix}`, APPLY_INDEXED_LAYER_SHADER, applyLayerEntryPoint)
    : null;
  const applyLayerPipeline = hiddenLayersTrained
    ? await createComputePipelineChecked(device, `apply_layer_${denseKernelSuffix}`, APPLY_LAYER_SHADER, applyLayerEntryPoint)
    : null;
  const applyOutputPipeline = await createComputePipelineChecked(device, "apply_output", APPLY_OUTPUT_SHADER, "apply_output");

  const lossIndices = validationIndices.length ? validationIndices : trainIndices;
  const initialValidationLoss = await timed(config.metrics, "initialValidationLoss", () =>
    predictionLossOnProjectedGpu(
      device,
      featureBuffer,
      weightBuffers,
      outputWeightBuffer,
      lossIndices,
      labelBuffer,
      labelWeightBuffer,
      forwardIndexedLayerPipeline,
      forwardLayerPipeline,
      forwardOutputPipeline,
      reduceLossPipeline
    )
  );
  let bestValidationLoss = initialValidationLoss;
  let checkpointImproved = false;
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
      const batchWeight = fillGroupedTrainingBatchIndices(batchIndices, trainGroups, epoch, split.seed, labelWeights);
      device.queue.writeBuffer(batchIndexBuffer, 0, batchIndices);
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

      if (hiddenLayersTrained) {
        const lastLayerIndex = HIDDEN_LAYERS.length - 1;
        encodePipeline(device, encoder, lastHiddenDeltaPipeline!, [
          activationBuffers[lastLayerIndex]!,
          outputDeltaBuffer,
          outputWeightBuffer,
          deltaBuffers[lastLayerIndex]!,
          lastHiddenDeltaParams!
        ], Math.ceil(batchSize / 16), Math.ceil(HIDDEN_LAYERS[lastLayerIndex]! / 16));

        for (let layerIndex = HIDDEN_LAYERS.length - 2; layerIndex >= 0; layerIndex -= 1) {
          encodePipeline(device, encoder, hiddenDeltaPipeline!, [
            activationBuffers[layerIndex]!,
            deltaBuffers[layerIndex + 1]!,
            weightBuffers[layerIndex + 1]!,
            deltaBuffers[layerIndex]!,
            hiddenDeltaParams[layerIndex]!
          ], Math.ceil(batchSize / 16), Math.ceil(HIDDEN_LAYERS[layerIndex]! / 16));
        }
      }

      encodePipeline(device, encoder, applyOutputPipeline, [
        activationBuffers[activationBuffers.length - 1]!,
        outputDeltaBuffer,
        outputWeightBuffer,
        applyOutputParams,
        outputVelocityBuffer
      ], Math.ceil((outputSize + 1) / 64));

      if (hiddenLayersTrained) {
        for (let layerIndex = HIDDEN_LAYERS.length - 1; layerIndex >= 0; layerIndex -= 1) {
          const inputSize = previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE);
          const inputBuffer = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1]!;
          const outputSizeForLayer = HIDDEN_LAYERS[layerIndex]!;
          encodePipeline(device, encoder, layerIndex === 0 ? applyIndexedLayerPipeline! : applyLayerPipeline!, [
            inputBuffer,
            deltaBuffers[layerIndex]!,
            weightBuffers[layerIndex]!,
            applyLayerParams[layerIndex]!,
            ...(layerIndex === 0
              ? [batchIndexBuffer, velocityBuffers[layerIndex]!]
              : [velocityBuffers[layerIndex]!])
          ], Math.ceil((inputSize + 1) / 16), Math.ceil(outputSizeForLayer / 16));
        }
      }
    }
    device.queue.submit([encoder.finish()]);
    if (batchEnd % validationInterval !== 0 && batchEnd < config.epochs) {
      continue;
    }
    lastTrainLoss = await timed(config.metrics, "trainLoss", () =>
      predictionLossOnProjectedGpu(
        device,
        featureBuffer,
        weightBuffers,
        outputWeightBuffer,
        trainIndices,
        labelBuffer,
        labelWeightBuffer,
        forwardIndexedLayerPipeline,
        forwardLayerPipeline,
        forwardOutputPipeline,
        reduceLossPipeline
      )
    );
    lastValidationLoss = validationIndices.length
      ? await timed(config.metrics, "validationLoss", () =>
        predictionLossOnProjectedGpu(
          device,
          featureBuffer,
          weightBuffers,
          outputWeightBuffer,
          validationIndices,
          labelBuffer,
          labelWeightBuffer,
          forwardIndexedLayerPipeline,
          forwardLayerPipeline,
          forwardOutputPipeline,
          reduceLossPipeline
        )
      )
      : lastTrainLoss;
    if (lastValidationLoss + 1e-6 < bestValidationLoss) {
      bestValidationLoss = lastValidationLoss;
      checkpointImproved = true;
      await timed(config.metrics, "bestCheckpointCopy", () =>
        copyTrainingWeights(device, outputWeightBuffer, weightBuffers, bestOutputWeightBuffer, bestWeightBuffers, layerWeights, outputWeights.byteLength)
      );
      epochsWithoutImprovement = 0;
    } else {
      epochsWithoutImprovement += 1;
    }
    progress({
      epoch: batchEnd,
      loss: lastTrainLoss,
      validationLoss: lastValidationLoss,
      bestValidationLoss,
      initialValidationLoss,
      checkpointImproved,
      epochsWithoutImprovement,
      batchSize,
      batchesPerSubmit,
      validationInterval,
      replaySize: sampleCount,
      hiddenLayersTrained,
      labelCounts: labelSourceCounts(samples)
    });
    if (epochsWithoutImprovement >= config.patience) {
      earlyStopReason = `validation did not improve for ${config.patience} checks`;
      break;
    }
  }

  const {
    output: trainedOutput,
    layers: trainedLayerWeights
  } = await timed(config.metrics, "bestWeightReadback", () =>
    readTrainingWeights(device, bestOutputWeightBuffer, bestWeightBuffers, layerWeights, outputWeights.byteLength)
  );
  const policyFeatureKernelSuffix = sampleCount >= TILED_TRAINING_MIN_BATCH ? "tiled" : "naive";
  const policyFeatureForwardPipeline = await createComputePipelineChecked(
    device,
    `forward_layer_${policyFeatureKernelSuffix}`,
    FORWARD_LAYER_SHADER,
    denseKernelEntryPoint("forward_layer", sampleCount)
  );
  const policyFeatures = forwardHiddenFeaturesOnProjectedGpu(
    device,
    featureBuffer,
    bestWeightBuffers,
    sampleCount,
    policyFeatureForwardPipeline
  );
  const trainedHidden = concatFloat32(trainedLayerWeights);
  return {
    featureCount: outputSize,
    weights: trainedOutput,
    hiddenWeights: trainedHidden,
    loss: lastTrainLoss,
    initialValidationLoss,
    validationLoss: lastValidationLoss,
    bestValidationLoss,
    checkpointImproved,
    hiddenLayersTrained,
    earlyStopReason,
    policyFeatureBuffer: policyFeatures.featureBuffer,
    resources: [
      featureBuffer,
      labelBuffer,
      labelWeightBuffer,
      ...weightBuffers,
      ...velocityBuffers,
      outputWeightBuffer,
      outputVelocityBuffer,
      ...bestWeightBuffers,
      bestOutputWeightBuffer,
      ...activationBuffers,
      ...deltaBuffers,
      predictionBuffer,
      outputDeltaBuffer,
      ...batchIndexBuffers,
      ...forwardLayerParams,
      ...applyLayerParams,
      forwardOutputParams,
      applyOutputParams,
      ...outputDeltaParams,
      ...(lastHiddenDeltaParams ? [lastHiddenDeltaParams] : []),
      ...hiddenDeltaParams,
      ...policyFeatures.resources
    ]
  };
}

export async function trainPolicy(
  device: GPUDevice,
  samples: TrainingSample[],
  config: TrainingConfig,
  activeModel: CompactValueModel | null,
  featureBuffer: GPUBuffer,
  inputSize: number
): Promise<TrainedPolicyWeights> {
  const policyIndices: number[] = [];
  const targets = new Uint32Array(samples.length);
  const labelWeights = new Float32Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index]!;
    if (!hasPolicyTrainingTarget(sample)) {
      continue;
    }
    targets[index] = Math.min(POLICY_BUCKETS - 1, sample.policy ?? 0);
    labelWeights[index] = Math.max(0, sample.labelWeight ?? 1);
    if (labelWeights[index]! > 0) {
      policyIndices.push(index);
    }
  }
  const weightCount = POLICY_BUCKETS * (inputSize + 1);
  const initialWeights = policyWeightsArray(activeModel, inputSize) ?? new Float32Array(weightCount);
  if (!policyIndices.length) {
    return {
      weights: initialWeights,
      initialValidationLoss: Number.NaN,
      validationLoss: Number.NaN,
      bestValidationLoss: Number.NaN,
      checkpointImproved: false
    };
  }
  const split = splitValidationSamples(samples, config.validationSplit);
  const policySplit = splitPolicyTrainingIndices(samples, policyIndices, split, config.validationSplit ?? 0);
  const trainIndices = policySplit.trainIndices.length ? policySplit.trainIndices : policyIndices;
  const validationIndices = policySplit.validationIndices;
  const trainGroups = groupTrainingIndicesByPosition(samples, trainIndices);
  const batchSize = Math.min(config.batchSize, trainIndices.length);
  const steps = policyTrainingSteps(config.epochs);
  const stepsPerSubmit = Math.min(POLICY_STEPS_PER_SUBMIT, steps);
  const targetBuffer = storageBuffer(device, targets, gpuBufferUsage.STORAGE);
  const labelWeightBuffer = storageBuffer(device, labelWeights, gpuBufferUsage.STORAGE);
  const policyWeightBuffer = storageBuffer(
    device,
    initialWeights,
    gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  );
  const policyVelocityBuffer = zeroStorageBuffer(device, initialWeights.byteLength);
  const bestPolicyWeightBuffer = storageBuffer(
    device,
    initialWeights,
    gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC | gpuBufferUsage.COPY_DST
  );
  const logitsBuffer = device.createBuffer({
    size: align4(batchSize * POLICY_BUCKETS * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  });
  const deltaBuffer = device.createBuffer({
    size: align4(batchSize * POLICY_BUCKETS * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  });
  const batchIndexBuffers = Array.from({ length: stepsPerSubmit }, () => device.createBuffer({
    size: align4(batchSize * Uint32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_DST
  }));
  const paramsBuffers = Array.from({ length: stepsPerSubmit }, () =>
    storageBuffer(device, policyParamsData(batchSize, inputSize, 1, config), gpuBufferUsage.UNIFORM)
  );
  const batchIndices = new Uint32Array(batchSize);
  const policyKernelSuffix = batchSize >= TILED_TRAINING_MIN_BATCH ? "tiled" : "naive";
  const forwardPipeline = await createComputePipelineChecked(device, `policy_forward_${policyKernelSuffix}`, POLICY_SHADER, denseKernelEntryPoint("forward_policy", batchSize));
  const deltaPipeline = await createComputePipelineChecked(device, "policy_delta", POLICY_SHADER, "policy_delta");
  const applyPipeline = await createComputePipelineChecked(device, `policy_apply_${policyKernelSuffix}`, POLICY_SHADER, denseKernelEntryPoint("apply_policy", batchSize));
  const lossPipeline = await createComputePipelineChecked(device, "policy_loss", POLICY_LOSS_SHADER, "reduce_policy_loss");
  const lossIndices = validationIndices.length ? validationIndices : trainIndices;
  const initialValidationLoss = await policyLossOnGpu(
    device,
    featureBuffer,
    targetBuffer,
    labelWeightBuffer,
    policyWeightBuffer,
    lossIndices,
    inputSize,
    lossPipeline
  );
  let lastValidationLoss = initialValidationLoss;
  let bestValidationLoss = initialValidationLoss;
  let checkpointImproved = false;
  for (let step = 0; step < steps;) {
    const batchEnd = Math.min(steps, step + stepsPerSubmit);
    const encoder = device.createCommandEncoder();
    for (; step < batchEnd; step += 1) {
      const slot = step % stepsPerSubmit;
      const batchWeight = fillGroupedTrainingBatchIndices(
        batchIndices,
        trainGroups,
        step + 1,
        0x9e3779b9,
        labelWeights
      );
      device.queue.writeBuffer(batchIndexBuffers[slot]!, 0, batchIndices);
      device.queue.writeBuffer(
        paramsBuffers[slot]!,
        0,
        policyParamsData(batchSize, inputSize, batchWeight, config)
      );
      encodePipelineBindings(device, encoder, forwardPipeline, [
        [0, featureBuffer],
        [3, policyWeightBuffer],
        [4, logitsBuffer],
        [6, batchIndexBuffers[slot]!],
        [7, paramsBuffers[slot]!]
      ], Math.ceil(batchSize / 16), Math.ceil(POLICY_BUCKETS / 16));
      encodePipelineBindings(device, encoder, deltaPipeline, [
        [1, targetBuffer],
        [2, labelWeightBuffer],
        [4, logitsBuffer],
        [5, deltaBuffer],
        [6, batchIndexBuffers[slot]!],
        [7, paramsBuffers[slot]!]
      ], Math.ceil(batchSize / 64));
      encodePipelineBindings(device, encoder, applyPipeline, [
        [0, featureBuffer],
        [3, policyWeightBuffer],
        [5, deltaBuffer],
        [6, batchIndexBuffers[slot]!],
        [7, paramsBuffers[slot]!],
        [8, policyVelocityBuffer]
      ], Math.ceil((inputSize + 1) / 16), Math.ceil(POLICY_BUCKETS / 16));
    }
    device.queue.submit([encoder.finish()]);
    lastValidationLoss = await policyLossOnGpu(
      device,
      featureBuffer,
      targetBuffer,
      labelWeightBuffer,
      policyWeightBuffer,
      lossIndices,
      inputSize,
      lossPipeline
    );
    if (lastValidationLoss + 1e-6 < bestValidationLoss) {
      bestValidationLoss = lastValidationLoss;
      checkpointImproved = true;
      const checkpoint = device.createCommandEncoder();
      checkpoint.copyBufferToBuffer(
        policyWeightBuffer,
        0,
        bestPolicyWeightBuffer,
        0,
        initialWeights.byteLength
      );
      device.queue.submit([checkpoint.finish()]);
    }
  }
  const weights = await readFloats(device, bestPolicyWeightBuffer, initialWeights.byteLength);
  destroyBuffers([
    targetBuffer,
    labelWeightBuffer,
    policyWeightBuffer,
    policyVelocityBuffer,
    bestPolicyWeightBuffer,
    logitsBuffer,
    deltaBuffer,
    ...batchIndexBuffers,
    ...paramsBuffers
  ]);
  return {
    weights,
    initialValidationLoss,
    validationLoss: lastValidationLoss,
    bestValidationLoss,
    checkpointImproved
  };
}

export function hasPolicyTrainingTarget(sample: TrainingSample): boolean {
  return sample.labelKind !== "distilled"
    && Number.isInteger(sample.policy)
    && (sample.policy ?? -1) >= 0;
}

export function splitPolicyTrainingIndices(
  samples: TrainingSample[],
  policyIndices: number[],
  split: ValidationSplit,
  validationSplit: number
): ValidationSplit {
  const policySet = new Set(policyIndices);
  const trainIndices = split.trainIndices.filter((index) => policySet.has(index));
  const validationIndices = split.validationIndices.filter((index) => policySet.has(index));
  if (validationSplit > 0 && !validationIndices.length && trainIndices.length > 1) {
    movePositionGroupToValidation(samples, trainIndices, validationIndices);
  }
  if (!trainIndices.length && validationIndices.length) {
    moveOrCollapseValidationGroup(samples, trainIndices, validationIndices);
  }
  return { trainIndices, validationIndices, seed: split.seed };
}

export function policyTrainingSteps(valueEpochs: number): number {
  return Math.max(16, Math.min(256, Math.ceil(valueEpochs / 64)));
}

async function policyLossOnGpu(
  device: GPUDevice,
  featureBuffer: GPUBuffer,
  targetBuffer: GPUBuffer,
  labelWeightBuffer: GPUBuffer,
  policyWeightBuffer: GPUBuffer,
  indices: number[],
  inputSize: number,
  pipeline: GPUComputePipeline
): Promise<number> {
  if (!indices.length) {
    return 0;
  }
  const indexBuffer = storageBuffer(device, new Uint32Array(indices), gpuBufferUsage.STORAGE);
  const partialCount = lossReductionWorkgroupCount(indices.length);
  const partialBuffer = device.createBuffer({
    size: align4(partialCount * 2 * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  });
  const params = new Uint32Array([indices.length, inputSize, POLICY_BUCKETS, 0]);
  const paramsBuffer = storageBuffer(device, params, gpuBufferUsage.UNIFORM);
  const encoder = device.createCommandEncoder();
  encodePipeline(device, encoder, pipeline, [
    featureBuffer,
    targetBuffer,
    labelWeightBuffer,
    policyWeightBuffer,
    indexBuffer,
    partialBuffer,
    paramsBuffer
  ], partialCount);
  device.queue.submit([encoder.finish()]);
  const partials = await readFloats(
    device,
    partialBuffer,
    partialCount * 2 * Float32Array.BYTES_PER_ELEMENT
  );
  let total = 0;
  let totalWeight = 0;
  for (let index = 0; index < partials.length; index += 2) {
    total += partials[index] ?? 0;
    totalWeight += partials[index + 1] ?? 0;
  }
  destroyBuffers([indexBuffer, partialBuffer, paramsBuffer]);
  return totalWeight > 0 ? total / totalWeight : 0;
}

export function runPipeline(device: GPUDevice, pipeline: GPUComputePipeline, buffers: GPUBuffer[], workgroupsX: number, workgroupsY = 1): void {
  const encoder = device.createCommandEncoder();
  encodePipeline(device, encoder, pipeline, buffers, workgroupsX, workgroupsY);
  device.queue.submit([encoder.finish()]);
}

export function encodePipeline(device: GPUDevice, encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, buffers: GPUBuffer[], workgroupsX: number, workgroupsY = 1): void {
  encodePipelineBindings(
    device,
    encoder,
    pipeline,
    buffers.map((buffer, binding) => [binding, buffer]),
    workgroupsX,
    workgroupsY
  );
}

function encodePipelineBindings(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  buffers: Array<[number, GPUBuffer]>,
  workgroupsX: number,
  workgroupsY = 1
): void {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map(([binding, buffer]) => ({ binding, resource: { buffer } }))
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupsX, workgroupsY);
  pass.end();
}

function copyTrainingWeights(
  device: GPUDevice,
  outputWeightBuffer: GPUBuffer,
  weightBuffers: GPUBuffer[],
  bestOutputWeightBuffer: GPUBuffer,
  bestWeightBuffers: GPUBuffer[],
  layerWeights: Float32Array[],
  outputByteLength: number
): void {
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(outputWeightBuffer, 0, bestOutputWeightBuffer, 0, outputByteLength);
  for (let layerIndex = 0; layerIndex < weightBuffers.length; layerIndex += 1) {
    encoder.copyBufferToBuffer(
      weightBuffers[layerIndex]!,
      0,
      bestWeightBuffers[layerIndex]!,
      0,
      layerWeights[layerIndex]!.byteLength
    );
  }
  device.queue.submit([encoder.finish()]);
}

async function readTrainingWeights(
  device: GPUDevice,
  outputWeightBuffer: GPUBuffer,
  weightBuffers: GPUBuffer[],
  layerWeights: Float32Array[],
  outputByteLength: number
): Promise<{ output: Float32Array; layers: Float32Array[] }> {
  const layerOffsets: number[] = [];
  let totalByteLength = outputByteLength;
  for (let layerIndex = 0; layerIndex < weightBuffers.length; layerIndex += 1) {
    layerOffsets.push(totalByteLength);
    totalByteLength += layerWeights[layerIndex]!.byteLength;
  }
  const readBuffer = device.createBuffer({
    size: align4(totalByteLength),
    usage: gpuBufferUsage.COPY_DST | gpuBufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(outputWeightBuffer, 0, readBuffer, 0, outputByteLength);
  for (let layerIndex = 0; layerIndex < weightBuffers.length; layerIndex += 1) {
    encoder.copyBufferToBuffer(
      weightBuffers[layerIndex]!,
      0,
      readBuffer,
      layerOffsets[layerIndex]!,
      layerWeights[layerIndex]!.byteLength
    );
  }
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(gpuMapMode.READ);
  const mapped = readBuffer.getMappedRange();
  const output = new Float32Array(mapped.slice(0, outputByteLength));
  const layers = layerWeights.map((weights, layerIndex) =>
    new Float32Array(mapped.slice(
      layerOffsets[layerIndex]!,
      layerOffsets[layerIndex]! + weights.byteLength
    ))
  );
  readBuffer.unmap();
  readBuffer.destroy();
  return { output, layers };
}

function destroyBuffers(buffers: GPUBuffer[]): void {
  const destroyed = new Set<GPUBuffer>();
  for (const buffer of buffers) {
    if (destroyed.has(buffer)) {
      continue;
    }
    destroyed.add(buffer);
    buffer.destroy();
  }
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

function policyParamsData(
  batchCount: number,
  inputSize: number,
  totalWeight: number,
  config: TrainingConfig
): ArrayBuffer {
  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, batchCount, true);
  view.setUint32(4, inputSize, true);
  view.setUint32(8, POLICY_BUCKETS, true);
  view.setFloat32(16, Math.max(0, totalWeight), true);
  view.setFloat32(20, config.learningRate, true);
  view.setFloat32(24, config.weightDecay, true);
  view.setFloat32(28, OPTIMIZER_MOMENTUM, true);
  return params;
}

export function layerParamsBuffer(device: GPUDevice, sampleCount: number, inputSize: number, outputSize: number, learningRate: number, weightDecay = 0, momentum = 0): GPUBuffer {
  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setUint32(8, outputSize, true);
  view.setFloat32(12, learningRate, true);
  view.setFloat32(16, weightDecay, true);
  view.setFloat32(20, momentum, true);
  return storageBuffer(device, params, gpuBufferUsage.UNIFORM);
}

export function outputParamsBuffer(device: GPUDevice, sampleCount: number, inputSize: number, learningRate: number, weightDecay = 0, momentum = 0): GPUBuffer {
  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setFloat32(12, learningRate, true);
  view.setFloat32(16, weightDecay, true);
  view.setFloat32(20, momentum, true);
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

function zeroStorageBuffer(device: GPUDevice, byteLength: number): GPUBuffer {
  return device.createBuffer({
    size: align4(byteLength),
    usage: gpuBufferUsage.STORAGE
  });
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
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  });
  const pipeline = await createComputePipelineChecked(device, "project_features", PROJECT_FEATURES_SHADER, "project_features");
  const temporaryBudget = projectionTemporaryBudget(device);
  let batchOffset = 0;
  while (batchOffset < samples.length) {
    const encoder = device.createCommandEncoder();
    const temporaryBuffers: GPUBuffer[] = [];
    let temporaryBytes = 0;
    let offset = batchOffset;
    while (offset < samples.length) {
      const chunkSamples = samples.slice(offset, offset + PROJECTION_CHUNK_SIZE);
      const sparseFeatures = packSparseProjectionFeatures(chunkSamples, inputSize);
      if (temporaryBuffers.length && temporaryBytes + sparseFeatures.byteLength > temporaryBudget) {
        break;
      }
      if (sparseFeatures.indices.byteLength > maxBindingSize || sparseFeatures.values.byteLength > maxBindingSize) {
        throw new Error(`Sparse projection chunk exceeds this WebGPU device's storage binding limit (${formatBytes(Math.max(sparseFeatures.indices.byteLength, sparseFeatures.values.byteLength))} > ${formatBytes(maxBindingSize)}). Reduce batch size or feature size.`);
      }
      const offsetBuffer = storageBuffer(device, sparseFeatures.offsets, gpuBufferUsage.STORAGE);
      const indexBuffer = storageBuffer(device, sparseFeatures.indices, gpuBufferUsage.STORAGE);
      const valueBuffer = storageBuffer(device, sparseFeatures.values, gpuBufferUsage.STORAGE);
      const paramsBuffer = projectionParamsBuffer(
        device,
        chunkSamples.length,
        inputSize,
        projectionSize,
        seed,
        offset
      );
      temporaryBuffers.push(offsetBuffer, indexBuffer, valueBuffer, paramsBuffer);
      temporaryBytes += sparseFeatures.byteLength;
      encodePipeline(
        device,
        encoder,
        pipeline,
        [offsetBuffer, indexBuffer, valueBuffer, projectedBuffer, paramsBuffer],
        Math.ceil(chunkSamples.length / 16),
        Math.ceil(projectionSize / 16)
      );
      offset += chunkSamples.length;
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    for (const buffer of temporaryBuffers) {
      buffer.destroy();
    }
    batchOffset = offset;
  }
  return projectedBuffer;
}

function projectionTemporaryBudget(device: GPUDevice): number {
  const maxBufferSize = device.limits?.maxBufferSize ?? 256 * 1024 * 1024;
  return Math.min(
    PROJECTION_TEMPORARY_BUDGET,
    Math.max(1, Math.floor(maxBufferSize / 2))
  );
}

export function packSparseProjectionFeatures(samples: TrainingSample[], inputSize = featureLength(samples)): SparseProjectionFeatures {
  const offsets = new Uint32Array(samples.length + 1);
  let nonZeroCount = 0;
  for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex += 1) {
    const features = samples[sampleIndex]!.features;
    if (features.length !== inputSize) {
      throw new Error("Training samples have inconsistent feature lengths.");
    }
    for (let featureIndex = 0; featureIndex < inputSize; featureIndex += 1) {
      if (features[featureIndex] !== 0) {
        nonZeroCount += 1;
      }
    }
    offsets[sampleIndex + 1] = nonZeroCount;
  }
  const allocatedCount = Math.max(1, nonZeroCount);
  const indices = new Uint32Array(allocatedCount);
  const values = new Float32Array(allocatedCount);
  let cursor = 0;
  for (const sample of samples) {
    for (let featureIndex = 0; featureIndex < inputSize; featureIndex += 1) {
      const value = sample.features[featureIndex] ?? 0;
      if (value === 0) {
        continue;
      }
      indices[cursor] = featureIndex;
      values[cursor] = value;
      cursor += 1;
    }
  }
  return {
    offsets,
    indices,
    values,
    byteLength: offsets.byteLength + indices.byteLength + values.byteLength
  };
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
  if (validationSplit > 0 && !validationIndices.length && trainIndices.length > 1) {
    movePositionGroupToValidation(samples, trainIndices, validationIndices, seed);
  }
  if (!trainIndices.length && validationIndices.length) {
    moveOrCollapseValidationGroup(samples, trainIndices, validationIndices);
  }
  return { trainIndices, validationIndices, seed };
}

function fallbackValidationOffset(samples: TrainingSample[], trainIndices: number[], seed: number): number {
  let bestOffset = 0;
  let bestPriority = Number.NEGATIVE_INFINITY;
  for (let offset = 0; offset < trainIndices.length; offset += 1) {
    const sampleIndex = trainIndices[offset]!;
    const sample = samples[sampleIndex]!;
    const priority = validationSamplePriority(sample, sampleIndex, seed);
    if (priority > bestPriority) {
      bestPriority = priority;
      bestOffset = offset;
    }
  }
  return bestOffset;
}

function movePositionGroupToValidation(
  samples: TrainingSample[],
  trainIndices: number[],
  validationIndices: number[],
  seed = 0
): void {
  const groups = groupTrainingIndicesByPosition(samples, trainIndices);
  if (groups.length < 2) {
    return;
  }
  const representatives = groups.map((group) => group[0]!);
  const selectedOffset = fallbackValidationOffset(samples, representatives, seed);
  const selected = new Set(groups[selectedOffset]!);
  for (let offset = trainIndices.length - 1; offset >= 0; offset -= 1) {
    if (selected.has(trainIndices[offset]!)) {
      validationIndices.push(trainIndices.splice(offset, 1)[0]!);
    }
  }
  validationIndices.sort((left, right) => left - right);
}

function moveOrCollapseValidationGroup(
  samples: TrainingSample[],
  trainIndices: number[],
  validationIndices: number[]
): void {
  const groups = groupTrainingIndicesByPosition(samples, validationIndices);
  if (groups.length < 2) {
    trainIndices.push(...validationIndices);
    validationIndices.length = 0;
    return;
  }
  const selected = new Set(groups[0]!);
  for (let offset = validationIndices.length - 1; offset >= 0; offset -= 1) {
    if (selected.has(validationIndices[offset]!)) {
      trainIndices.push(validationIndices.splice(offset, 1)[0]!);
    }
  }
  trainIndices.sort((left, right) => left - right);
}

function validationSamplePriority(sample: TrainingSample, index: number, seed: number): number {
  const hashTieBreak = ((stableSampleHash(sample, index) ^ seed) >>> 0) / 0xffffffff;
  return trainingLabelPriority(sample.labelKind, sample.pseudo) +
    Math.max(0, sample.labelWeight ?? 1) +
    hashTieBreak * 0.001;
}

export function stableSampleHash(sample: TrainingSample, _index: number): number {
  let hash = 2166136261;
  const text = trainingPositionIdentity(sample);
  for (let offset = 0; offset < text.length; offset += 1) {
    hash ^= text.charCodeAt(offset);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function trainingPositionIdentity(sample: TrainingSample): string {
  return sample.positionKey
    ? `${sample.positionKey}|${sample.sideToMove ?? ""}|${sample.boardCount ?? 0}`
    : `${featureFingerprint(sample.features)}|${sample.sideToMove ?? ""}|${sample.boardCount ?? 0}`;
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

export function groupTrainingIndicesByPosition(samples: TrainingSample[], indices: number[]): number[][] {
  const groups = new Map<string, number[]>();
  for (const index of indices) {
    const identity = trainingPositionIdentity(samples[index]!);
    const group = groups.get(identity);
    if (group) {
      group.push(index);
    } else {
      groups.set(identity, [index]);
    }
  }
  return Array.from(groups.values());
}

export function uniqueTrainingPositionCount(samples: TrainingSample[], indices: number[]): number {
  return new Set(indices.map((index) => trainingPositionIdentity(samples[index]!))).size;
}

export function fillGroupedTrainingBatchIndices(
  batch: Uint32Array,
  trainGroups: number[][],
  epoch: number,
  seed: number,
  labelWeights: Float32Array
): number {
  if (!trainGroups.length) {
    throw new Error("Training requires at least one train position.");
  }
  let state = (seed ^ Math.imul(epoch, 2654435761)) >>> 0;
  let batchWeight = 0;
  for (let index = 0; index < batch.length; index += 1) {
    state = xorshift32(state || 1);
    const group = trainGroups[state % trainGroups.length]!;
    state = xorshift32(state || 1);
    const selected = group[state % group.length]!;
    batch[index] = selected;
    batchWeight += Math.max(0, labelWeights[selected] ?? 1);
  }
  return batchWeight;
}

export function xorshift32(value: number): number {
  let state = value >>> 0;
  state ^= state << 13;
  state ^= state >>> 17;
  state ^= state << 5;
  return state >>> 0;
}

export function featureLength(samples: TrainingSample[]): number {
  const length = samples[0]?.features?.length;
  if (!length || !samples.every((sample) => sample.features.length === length)) {
    throw new Error("Training samples have inconsistent feature lengths.");
  }
  return length;
}

export function projectionParamsBuffer(device: GPUDevice, sampleCount: number, inputSize: number, projectionSize: number, seed: number, outputOffset = 0): GPUBuffer {
  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setUint32(8, projectionSize, true);
  view.setUint32(12, seed >>> 0, true);
  view.setUint32(16, outputOffset, true);
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
  readBuffer.destroy();
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

export async function predictionLossOnProjectedGpu(
  device: GPUDevice,
  featureBuffer: GPUBuffer,
  weightBuffers: GPUBuffer[],
  outputWeightBuffer: GPUBuffer,
  indices: number[],
  labelBuffer: GPUBuffer,
  labelWeightBuffer: GPUBuffer,
  forwardIndexedLayerPipeline: GPUComputePipeline,
  forwardLayerPipeline: GPUComputePipeline,
  forwardOutputPipeline: GPUComputePipeline,
  reduceLossPipeline: GPUComputePipeline
): Promise<number> {
  if (!indices.length) {
    return 0;
  }
  const sampleCount = indices.length;
  const batchIndices = new Uint32Array(indices);
  const batchIndexBuffer = storageBuffer(device, batchIndices, gpuBufferUsage.STORAGE);
  const activationBuffers = HIDDEN_LAYERS.map((layerSize) => device.createBuffer({
    size: align4(sampleCount * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  }));
  const predictionBuffer = device.createBuffer({
    size: align4(sampleCount * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  });
  const partialCount = lossReductionWorkgroupCount(sampleCount);
  const partialLossBuffer = device.createBuffer({
    size: align4(partialCount * 2 * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  });
  const forwardLayerParams = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      sampleCount,
      previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE),
      layerSize,
      0
    )
  );
  const forwardOutputParams = outputParamsBuffer(device, sampleCount, outputLayerSize(), 0);
  const reduceLossParams = lossParamsBuffer(device, sampleCount);
  const encoder = device.createCommandEncoder();
  for (let layerIndex = 0; layerIndex < HIDDEN_LAYERS.length; layerIndex += 1) {
    const inputBuffer = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1]!;
    const outputSizeForLayer = HIDDEN_LAYERS[layerIndex]!;
    encodePipeline(device, encoder, layerIndex === 0 ? forwardIndexedLayerPipeline : forwardLayerPipeline, [
      inputBuffer,
      weightBuffers[layerIndex]!,
      activationBuffers[layerIndex]!,
      forwardLayerParams[layerIndex]!,
      ...(layerIndex === 0 ? [batchIndexBuffer] : [])
    ], Math.ceil(sampleCount / 16), Math.ceil(outputSizeForLayer / 16));
  }
  encodePipeline(device, encoder, forwardOutputPipeline, [
    activationBuffers[activationBuffers.length - 1]!,
    outputWeightBuffer,
    predictionBuffer,
    forwardOutputParams
  ], Math.ceil(sampleCount / 64));
  encodePipeline(device, encoder, reduceLossPipeline, [
    predictionBuffer,
    labelBuffer,
    labelWeightBuffer,
    batchIndexBuffer,
    partialLossBuffer,
    reduceLossParams
  ], partialCount);
  device.queue.submit([encoder.finish()]);
  const partials = await readFloats(
    device,
    partialLossBuffer,
    partialCount * 2 * Float32Array.BYTES_PER_ELEMENT
  );
  let total = 0;
  let totalWeight = 0;
  for (let index = 0; index < partials.length; index += 2) {
    total += partials[index] ?? 0;
    totalWeight += partials[index + 1] ?? 0;
  }
  destroyBuffers([
    batchIndexBuffer,
    ...activationBuffers,
    predictionBuffer,
    partialLossBuffer,
    ...forwardLayerParams,
    forwardOutputParams,
    reduceLossParams
  ]);
  return totalWeight > 0 ? total / totalWeight : 0;
}

function forwardHiddenFeaturesOnProjectedGpu(
  device: GPUDevice,
  featureBuffer: GPUBuffer,
  weightBuffers: GPUBuffer[],
  sampleCount: number,
  forwardLayerPipeline: GPUComputePipeline
): { featureBuffer: GPUBuffer; resources: GPUBuffer[] } {
  const activationBuffers = HIDDEN_LAYERS.map((layerSize) => device.createBuffer({
    size: align4(sampleCount * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  }));
  const params = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      sampleCount,
      previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE),
      layerSize,
      0
    )
  );
  const encoder = device.createCommandEncoder();
  for (let layerIndex = 0; layerIndex < HIDDEN_LAYERS.length; layerIndex += 1) {
    const input = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1]!;
    encodePipeline(device, encoder, forwardLayerPipeline, [
      input,
      weightBuffers[layerIndex]!,
      activationBuffers[layerIndex]!,
      params[layerIndex]!
    ], Math.ceil(sampleCount / 16), Math.ceil(HIDDEN_LAYERS[layerIndex]! / 16));
  }
  device.queue.submit([encoder.finish()]);
  return {
    featureBuffer: activationBuffers[activationBuffers.length - 1]!,
    resources: [...activationBuffers, ...params]
  };
}

export function lossReductionWorkgroupCount(sampleCount: number): number {
  return Math.max(1, Math.ceil(sampleCount / 64));
}

function lossParamsBuffer(device: GPUDevice, sampleCount: number): GPUBuffer {
  return storageBuffer(device, new Uint32Array([sampleCount, 0, 0, 0]), gpuBufferUsage.UNIFORM);
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
  if (!modelArchitectureMatches(model)) {
    throw new Error("GPU batch prediction unavailable.");
  }
  if (samples.length <= CPU_PREDICTION_MAX_BATCH) {
    return Array.from(predictValuesOnCpu(samples, model));
  }
  if (!globalThis.navigator?.gpu) {
    throw new Error("GPU batch prediction unavailable.");
  }
  const device = await getGpuDevice();
  if (!device) {
    throw new Error("No WebGPU adapter is available.");
  }
  return Array.from(await predictValuesOnGpu(device, samples, model));
}

function hiddenFeaturesOnCpu(
  sample: TrainingSample,
  projectionSize: number,
  projectionSeed: number,
  hiddenLayers: number[],
  hiddenWeights: Float32Array
): Float32Array {
  let activations = projectSampleOnCpu(sample, projectionSize, projectionSeed);
  let weightOffset = 0;
  for (const outputSize of hiddenLayers) {
    const inputSize = activations.length;
    const next = new Float32Array(outputSize);
    const rowSize = inputSize + 1;
    for (let output = 0; output < outputSize; output += 1) {
      const row = weightOffset + output * rowSize;
      let sum = hiddenWeights[row + inputSize] ?? 0;
      for (let input = 0; input < inputSize; input += 1) {
        sum += activations[input]! * (hiddenWeights[row + input] ?? 0);
      }
      next[output] = Math.max(0, sum);
    }
    weightOffset += outputSize * rowSize;
    activations = next;
  }
  return activations;
}

export function predictValuesOnCpu(samples: TrainingSample[], model: CompactValueModel): Float32Array {
  const predictions = new Float32Array(samples.length);
  for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex += 1) {
    const activations = hiddenFeaturesOnCpu(
      samples[sampleIndex]!,
      model.projectionSize,
      model.projectionSeed,
      model.hiddenLayers,
      model.hiddenWeights
    );
    let prediction = model.outputWeights[activations.length] ?? 0;
    for (let input = 0; input < activations.length; input += 1) {
      prediction += activations[input]! * (model.outputWeights[input] ?? 0);
    }
    const activated = model.outputActivation === "tanh" ? Math.tanh(prediction) : prediction;
    predictions[sampleIndex] = boundedValue(activated * (model.scale ?? 1) + (model.bias ?? 0));
  }
  return predictions;
}

function valueHeadLossOnCpu(
  features: Float32Array[],
  samples: TrainingSample[],
  weights: Float32Array,
  indices: number[]
): number {
  let total = 0;
  let totalWeight = 0;
  const biasIndex = weights.length - 1;
  for (const index of indices) {
    const feature = features[index]!;
    let logit = weights[biasIndex] ?? 0;
    for (let input = 0; input < feature.length; input += 1) {
      logit += feature[input]! * (weights[input] ?? 0);
    }
    const prediction = Math.tanh(logit);
    const weight = Math.max(0, samples[index]!.labelWeight ?? 1);
    const error = prediction - samples[index]!.label;
    total += weight * error * error;
    totalWeight += weight;
  }
  return totalWeight > 0 ? total / totalWeight : 0;
}

function applyValueHeadGradientOnCpu(
  features: Float32Array[],
  samples: TrainingSample[],
  weights: Float32Array,
  batchIndices: Uint32Array,
  batchWeight: number,
  learningRate: number,
  weightDecay: number,
  velocity: Float32Array
): void {
  const gradient = new Float32Array(weights.length);
  const biasIndex = weights.length - 1;
  for (const sampleIndex of batchIndices) {
    const feature = features[sampleIndex]!;
    let logit = weights[biasIndex] ?? 0;
    for (let input = 0; input < feature.length; input += 1) {
      logit += feature[input]! * (weights[input] ?? 0);
    }
    const prediction = Math.tanh(logit);
    const scale = 2 * Math.max(0, samples[sampleIndex]!.labelWeight ?? 1)
      * (prediction - samples[sampleIndex]!.label)
      * (1 - prediction * prediction);
    for (let input = 0; input < feature.length; input += 1) {
      gradient[input] = (gradient[input] ?? 0) + scale * feature[input]!;
    }
    gradient[biasIndex] = (gradient[biasIndex] ?? 0) + scale;
  }
  const normalization = 1 / Math.max(batchWeight, 1e-6);
  for (let input = 0; input < weights.length; input += 1) {
    const decay = input === biasIndex ? 0 : weightDecay * weights[input]!;
    const update = gradient[input]! * normalization + decay;
    velocity[input] = optimizerVelocity(velocity[input] ?? 0, update);
    weights[input] = (weights[input] ?? 0) - learningRate * velocity[input]!;
  }
}

function trainPolicyHeadOnCpu(
  samples: TrainingSample[],
  features: Float32Array[],
  config: TrainingConfig,
  activeModel: CompactValueModel | null,
  valueSplit: ValidationSplit
): TrainedPolicyWeights {
  const policyIndices = samples
    .map((sample, index) => ({ sample, index }))
    .filter(({ sample }) =>
      hasPolicyTrainingTarget(sample) && Math.max(0, sample.labelWeight ?? 1) > 0
    )
    .map(({ index }) => index);
  const inputSize = features[0]?.length ?? outputLayerSize();
  const initialWeights = policyWeightsArray(activeModel, inputSize)
    ?? new Float32Array(POLICY_BUCKETS * (inputSize + 1));
  if (!policyIndices.length) {
    return {
      weights: initialWeights,
      initialValidationLoss: Number.NaN,
      validationLoss: Number.NaN,
      bestValidationLoss: Number.NaN,
      checkpointImproved: false
    };
  }
  const split = splitPolicyTrainingIndices(samples, policyIndices, valueSplit, config.validationSplit ?? 0);
  const trainIndices = split.trainIndices.length ? split.trainIndices : policyIndices;
  const validationIndices = split.validationIndices;
  const trainGroups = groupTrainingIndicesByPosition(samples, trainIndices);
  const batchSize = Math.min(config.batchSize, trainIndices.length);
  const batchIndices = new Uint32Array(batchSize);
  const labelWeights = Float32Array.from(samples, (sample) => Math.max(0, sample.labelWeight ?? 1));
  const weights = initialWeights.slice();
  const velocity = new Float32Array(weights.length);
  let bestWeights = weights.slice();
  const lossIndices = validationIndices.length ? validationIndices : trainIndices;
  const initialValidationLoss = policyHeadLossOnCpu(samples, features, weights, lossIndices);
  let validationLoss = initialValidationLoss;
  let bestValidationLoss = initialValidationLoss;
  let checkpointImproved = false;
  const steps = policyTrainingSteps(config.epochs);
  for (let step = 1; step <= steps; step += 1) {
    const batchWeight = fillGroupedTrainingBatchIndices(batchIndices, trainGroups, step, split.seed, labelWeights);
    applyPolicyHeadGradientOnCpu(samples, features, weights, batchIndices, batchWeight, config, velocity);
    validationLoss = policyHeadLossOnCpu(samples, features, weights, lossIndices);
    if (validationLoss + 1e-6 < bestValidationLoss) {
      bestValidationLoss = validationLoss;
      bestWeights = weights.slice();
      checkpointImproved = true;
    }
  }
  return {
    weights: bestWeights,
    initialValidationLoss,
    validationLoss,
    bestValidationLoss,
    checkpointImproved
  };
}

function policyHeadLossOnCpu(
  samples: TrainingSample[],
  features: Float32Array[],
  weights: Float32Array,
  indices: number[]
): number {
  const inputSize = features[0]?.length ?? 0;
  const rowSize = inputSize + 1;
  let total = 0;
  let totalWeight = 0;
  const logits = new Float32Array(POLICY_BUCKETS);
  for (const index of indices) {
    const feature = features[index]!;
    let maxLogit = Number.NEGATIVE_INFINITY;
    for (let bucket = 0; bucket < POLICY_BUCKETS; bucket += 1) {
      const row = bucket * rowSize;
      let logit = weights[row + inputSize] ?? 0;
      for (let input = 0; input < inputSize; input += 1) {
        logit += feature[input]! * (weights[row + input] ?? 0);
      }
      logits[bucket] = logit;
      maxLogit = Math.max(maxLogit, logit);
    }
    let denominator = 0;
    for (const logit of logits) {
      denominator += Math.exp(logit - maxLogit);
    }
    const target = Math.min(POLICY_BUCKETS - 1, samples[index]!.policy ?? 0);
    const weight = Math.max(0, samples[index]!.labelWeight ?? 1);
    total += weight * (Math.log(Math.max(denominator, 1e-12)) - (logits[target]! - maxLogit));
    totalWeight += weight;
  }
  return totalWeight > 0 ? total / totalWeight : 0;
}

function applyPolicyHeadGradientOnCpu(
  samples: TrainingSample[],
  features: Float32Array[],
  weights: Float32Array,
  batchIndices: Uint32Array,
  batchWeight: number,
  config: TrainingConfig,
  velocity: Float32Array
): void {
  const inputSize = features[0]?.length ?? 0;
  const rowSize = inputSize + 1;
  const gradient = new Float32Array(weights.length);
  const logits = new Float32Array(POLICY_BUCKETS);
  for (const sampleIndex of batchIndices) {
    const feature = features[sampleIndex]!;
    let maxLogit = Number.NEGATIVE_INFINITY;
    for (let bucket = 0; bucket < POLICY_BUCKETS; bucket += 1) {
      const row = bucket * rowSize;
      let logit = weights[row + inputSize] ?? 0;
      for (let input = 0; input < inputSize; input += 1) {
        logit += feature[input]! * (weights[row + input] ?? 0);
      }
      logits[bucket] = logit;
      maxLogit = Math.max(maxLogit, logit);
    }
    let denominator = 0;
    for (let bucket = 0; bucket < POLICY_BUCKETS; bucket += 1) {
      logits[bucket] = Math.exp(logits[bucket]! - maxLogit);
      denominator += logits[bucket]!;
    }
    const target = Math.min(POLICY_BUCKETS - 1, samples[sampleIndex]!.policy ?? 0);
    const sampleWeight = Math.max(0, samples[sampleIndex]!.labelWeight ?? 1);
    for (let bucket = 0; bucket < POLICY_BUCKETS; bucket += 1) {
      const delta = ((logits[bucket]! / denominator) - (bucket === target ? 1 : 0)) * sampleWeight;
      const row = bucket * rowSize;
      for (let input = 0; input < inputSize; input += 1) {
        gradient[row + input] = (gradient[row + input] ?? 0) + delta * feature[input]!;
      }
      gradient[row + inputSize] = (gradient[row + inputSize] ?? 0) + delta;
    }
  }
  const normalization = 1 / Math.max(batchWeight, 1e-6);
  for (let index = 0; index < weights.length; index += 1) {
    const isBias = index % rowSize === inputSize;
    const decay = isBias ? 0 : config.weightDecay * weights[index]!;
    const update = gradient[index]! * normalization + decay;
    velocity[index] = optimizerVelocity(velocity[index] ?? 0, update);
    weights[index] = (weights[index] ?? 0) - config.learningRate * velocity[index]!;
  }
}

export function optimizerVelocity(previous: number, gradient: number, momentum = OPTIMIZER_MOMENTUM): number {
  return momentum * previous + (1 - momentum) * gradient;
}

export function boundedValue(value: number): number {
  return Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0));
}

export function normalizedSearchScore(score: number): number {
  return boundedValue(score / VALUE_SCORE_SCALE);
}

export function denormalizedSearchScore(value: number): number {
  return Math.round(boundedValue(value) * VALUE_SCORE_SCALE);
}

export function inverseTanh(value: number): number {
  const bounded = Math.max(-0.999999, Math.min(0.999999, value));
  return 0.5 * Math.log((1 + bounded) / (1 - bounded));
}

function projectSampleOnCpu(sample: TrainingSample, projectionSize: number, seed: number): Float32Array {
  const active: Array<[number, number]> = [];
  for (let input = 0; input < sample.features.length; input += 1) {
    const value = sample.features[input] ?? 0;
    if (value !== 0) {
      active.push([input, value]);
    }
  }
  const projected = new Float32Array(projectionSize);
  if (!active.length) {
    return projected;
  }
  const scale = Math.sqrt(active.length);
  for (let output = 0; output < projectionSize; output += 1) {
    let sum = 0;
    for (const [input, value] of active) {
      sum += value * ((projectionHash(input, output, seed) & 1) === 0 ? 1 : -1) / scale;
    }
    projected[output] = sum;
  }
  return projected;
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
  const kernelSuffix = sampleCount >= TILED_TRAINING_MIN_BATCH ? "tiled" : "naive";
  const forwardLayerPipeline = await createComputePipelineChecked(device, `forward_layer_${kernelSuffix}`, FORWARD_LAYER_SHADER, denseKernelEntryPoint("forward_layer", sampleCount));
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
  const predictions = await readFloats(device, predictionBuffer, sampleCount * Float32Array.BYTES_PER_ELEMENT);
  const scale = model.scale ?? 1;
  const bias = model.bias ?? 0;
  for (let index = 0; index < predictions.length; index += 1) {
    predictions[index] = boundedValue((predictions[index] ?? 0) * scale + bias);
  }
  destroyBuffers([
    featureBuffer,
    ...weightBuffers,
    outputWeightBuffer,
    ...activationBuffers,
    predictionBuffer,
    ...forwardLayerParams,
    forwardOutputParams
  ]);
  return predictions;
}

export function modelArchitectureMatches(model: CompactValueModel | null | undefined): model is CompactValueModel {
  return Boolean(model
    && model.projectionSize === PROJECTION_SIZE
    && model.projectionSeed === PROJECTION_SEED
    && JSON.stringify(model.hiddenLayers) === JSON.stringify(HIDDEN_LAYERS)
    && model.hiddenWeights?.length
    && model.outputActivation === "tanh"
    && compactModelIsFinite(model));
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

export function denseKernelEntryPoint(entryPoint: string, sampleCount: number): string {
  return sampleCount >= TILED_TRAINING_MIN_BATCH ? entryPoint : `${entryPoint}_naive`;
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
