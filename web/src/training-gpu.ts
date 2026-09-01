import { loadTrainingShaders } from "./training-shaders.js";
import { CPU_HEAD_TRAINING_MAX_POSITIONS, CPU_PREDICTION_MAX_BATCH, HIDDEN_LAYERS, MIN_HIDDEN_TRAINING_POSITIONS, MIN_POLICY_WORKING_SET_FRACTION, NEURAL_BOARD_PLANES, NEURAL_BOARD_SQUARES, OPTIMIZER_MOMENTUM, POLICY_BUCKETS, PROJECTION_CHUNK_SIZE, PROJECTION_SEED, PROJECTION_SIZE, PROJECTION_TEMPORARY_BUDGET, VALUE_SCORE_SCALE } from "./training-gpu-constants.js";
import { align4, createComputePipelineChecked, denseKernelEntryPoint as fallbackDenseKernelEntryPoint, formatBytes, getGpuDevice } from "./training-gpu-device.js";
import { byteArraysEqual, compactModelIsFinite, encodeCompactModel } from "./training-gpu-model.js";
import * as engineGpuTrainingPolicy from "./engine-gpu-training-policy.js";
import { featureLength, fillGroupedTrainingBatchIndices, groupTrainingIndicesByPosition, moveOrCollapseValidationGroup, movePositionGroupToValidation, shuffledIndices, splitValidationSamples, trainingLabelPriority, uniqueTrainingPositionCount, xorshift32 } from "./training-gpu-samples.js";
import type { CompactValueModel, EncodableCompactModel, EncodedCompactModel } from "./training-gpu-model.js";
import type { ChronofishEngine } from "./types.js";
import type { SparseProjectionFeatures, TrainingConfig, TrainingMetrics, TrainingSample } from "./training-gpu-types.js";
import type { ValidationSplit } from "./training-gpu-samples.js";
export { VALUE_SCORE_SCALE } from "./training-gpu-constants.js";
export { align4, createComputePipelineChecked, formatBytes, formatShaderErrors, getGpuDevice, requestHighLimitDevice } from "./training-gpu-device.js";
export { byteArraysEqual, compactModelBytesAreFiniteWithEngine, compactModelIsFinite, decodeCompactFrontierModelLayoutWithEngine, decodeCompactModel, decodeCompactModelWithEngine, encodeCompactModel, encodeCompactModelWithEngine, writeAscii, writeF32, writeU32 } from "./training-gpu-model.js";
export { featureLength, fillGroupedTrainingBatchIndices, groupTrainingIndicesByPosition, shuffledIndices, splitValidationSamples, stableSampleHash, trainingLabelPriority, uniqueTrainingPositionCount, xorshift32 } from "./training-gpu-samples.js";
export type { CompactFrontierModelLayout, CompactValueModel, EncodableCompactModel, EncodedCompactModel } from "./training-gpu-model.js";
export type { ValidationSplit } from "./training-gpu-samples.js";
export type { SparseProjectionFeatures, TrainingConfig, TrainingLabelKind, TrainingMetrics, TrainingSample } from "./training-gpu-types.js";

export const AUXILIARY_VALUE_HEADS = [
  "value_wdl",
  "value_scalar",
  "mate_or_forced_loss_risk",
  "royal_safety_score",
  "active_timeline_advantage",
  "present_control",
  "material_active",
  "material_inactive",
  "policy_uncertainty"
] as const;

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

interface TrainedAuxiliaryValueWeights {
  weights: Float32Array;
  validationLoss: number;
}

interface PredictionModel {
  projectionSize: number;
  projectionSeed: number;
  hiddenLayers: number[];
  hiddenWeights: Float32Array;
  outputWeights: Float32Array;
  scale?: number;
}

interface CompactTrainingLayout {
  architectureMatches: boolean;
  outputSize: number;
  hiddenWeights: Float32Array;
  outputWeights: Float32Array;
}

export function outputLayerSize(layers: number[] = HIDDEN_LAYERS, engine?: ChronofishEngine): number {
  if (engine && isDefaultHiddenLayers(layers)) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_default_output_layer_size");
  }
  const size = layers[layers.length - 1];
  if (size === undefined) {
    throw new Error("Model must have at least one hidden layer.");
  }
  return size;
}

export function previousLayerSize(layers: number[], layerIndex: number, inputSize: number, engine?: ChronofishEngine): number {
  if (engine && isDefaultHiddenLayers(layers)) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_default_previous_layer_size", layerIndex, inputSize);
  }
  return layerIndex === 0 ? inputSize : layers[layerIndex - 1]!;
}

function isDefaultHiddenLayers(layers: number[]): boolean {
  return layers.length === HIDDEN_LAYERS.length
    && layers.every((layer, index) => layer === HIDDEN_LAYERS[index]);
}

function policyLogitsArray(model: CompactValueModel | null): Float32Array | null {
  const logits = model?.policyLogits ?? model?.policy_logits;
  if (!logits?.length) {
    return null;
  }
  return new Float32Array(Array.from(logits).slice(0, POLICY_BUCKETS));
}

export function policyWeightsArray(model: CompactValueModel | null, inputSize: number, engine?: ChronofishEngine): Float32Array | null {
  if (model && engine) {
    const modelBytes = model.bytes ?? encodeCompactModel(model, engine);
    const bytes = engineGpuTrainingPolicy.bytesResult(
      engine,
      "chronofish_compact_value_model_policy_weights_bytes",
      modelBytes,
      inputSize
    );
    if (bytes) {
      return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Float32Array.BYTES_PER_ELEMENT).slice();
    }
  }
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

function configuredLabelCounts(config: TrainingConfig): Record<string, number> {
  return config.labelCounts ?? {};
}

export async function train(
  samples: TrainingSample[],
  config: TrainingConfig,
  activeModel: CompactValueModel | null,
  progress: (message: Record<string, unknown>) => void,
  engine?: ChronofishEngine
): Promise<EncodedCompactModel> {
  if (!samples?.length) {
    throw new Error("No samples were collected.");
  }
  if (uniqueTrainingPositionCount(samples, samples.map((_, index) => index), engine) <= cpuHeadTrainingMaxPositions(engine)) {
    return timed(config.metrics, "cpuHeadTrain", () =>
      trainHeadsOnCpu(samples, config, activeModel, progress, engine)
    );
  }
  if (!globalThis.navigator?.gpu) {
    throw new Error("WebGPU is unavailable in this browser.");
  }

  const device = await getGpuDevice();
  if (!device) {
    throw new Error("No WebGPU adapter is available.");
  }
  const trainingSamples = selectTrainingWorkingSet(samples, device, engine);
  const value = await timed(config.metrics, "valueTrain", () =>
    trainValue(device, trainingSamples, config, activeModel, progress, engine)
  );
  try {
    const valueSplit = splitValidationSamples(trainingSamples, config.validationSplit, engine);
    const hiddenFeatures = trainingSamples.map((sample) =>
      hiddenFeaturesOnCpu(sample, PROJECTION_SIZE, PROJECTION_SEED, HIDDEN_LAYERS, value.hiddenWeights)
    );
    const auxiliary = trainAuxiliaryValueHeadsOnCpu(trainingSamples, hiddenFeatures, config, activeModel, valueSplit, engine);
    const policy = await timed(config.metrics, "policyTrain", () =>
      trainPolicy(
        device,
        trainingSamples,
        config,
        activeModel,
        value.policyFeatureBuffer,
        outputLayerSize(HIDDEN_LAYERS, engine),
        engine
      )
    );
    const model: EncodedCompactModel = encodeCompactModel({
      projectionSize: PROJECTION_SIZE,
      projectionSeed: PROJECTION_SEED,
      hiddenLayers: HIDDEN_LAYERS,
      hiddenWeights: value.hiddenWeights,
      outputWeights: value.weights,
      auxiliaryValueWeights: auxiliary.weights,
      policyWeights: policy.weights,
      scale: 1,
      bias: 0,
      outputActivation: "tanh"
    }, engine);
    model.trainingLoss = value.loss;
    model.initialValidationLoss = value.initialValidationLoss;
    model.validationLoss = value.validationLoss;
    model.bestValidationLoss = value.bestValidationLoss;
    model.initialPolicyValidationLoss = policy.initialValidationLoss;
    model.policyValidationLoss = policy.validationLoss;
    model.bestPolicyValidationLoss = policy.bestValidationLoss;
    model.auxiliaryValidationLoss = auxiliary.validationLoss;
    model.auxiliaryHeadCount = AUXILIARY_VALUE_HEADS.length;
    model.valueCheckpointImproved = value.checkpointImproved;
    model.policyCheckpointImproved = policy.checkpointImproved;
    model.modelChanged = !activeModel?.bytes || !byteArraysEqual(model, activeModel.bytes);
    model.earlyStopReason = value.earlyStopReason;
    model.labelCounts = configuredLabelCounts(config);
    model.nonZeroWeights = countNonZero(value.weights, engine) + countNonZero(auxiliary.weights, engine) + countNonZero(value.hiddenWeights, engine) + countNonZero(policy.weights, engine);
    model.replayBufferSize = samples.length;
    model.trainingSampleCount = trainingSamples.length;
    model.policyTrainingSampleCount = policyTrainingIndices(trainingSamples, false, engine).length;
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
  progress: (message: Record<string, unknown>) => void = () => {},
  engine?: ChronofishEngine
): EncodedCompactModel {
  const split = splitValidationSamples(samples, config.validationSplit, engine);
  const trainIndices = split.trainIndices.length ? split.trainIndices : split.validationIndices;
  const validationIndices = split.validationIndices;
  const trainGroups = groupTrainingIndicesByPosition(samples, trainIndices, engine);
  const batchSize = valueTrainingBatchSize(config.batchSize, trainIndices.length, engine);
  const averageLabel = samples.reduce((sum, sample) => sum + sample.label, 0) / samples.length;
  const layout = compactTrainingLayout(activeModel, averageLabel, engine);
  const hiddenWeights = layout.hiddenWeights;
  const hiddenFeatures = hiddenFeaturesForSamples(samples, hiddenWeights, engine);
  const outputWeights = layout.outputWeights;
  const labelWeights = Float32Array.from(samples, (sample) => trainingLabelWeight(sample.labelWeight, engine));
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
  const validationInterval = valueHeadValidationInterval(config.epochs, config.validationInterval, engine);

  for (let epoch = 1; epoch <= config.epochs; epoch += 1) {
    const batchWeight = fillGroupedTrainingBatchIndices(batchIndices, trainGroups, epoch, split.seed, labelWeights, engine);
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
      labelCounts: configuredLabelCounts(config)
    });
    if (epochsWithoutImprovement >= config.patience) {
      earlyStopReason = `validation did not improve for ${config.patience} checks`;
      break;
    }
  }

  const auxiliary = trainAuxiliaryValueHeadsOnCpu(samples, hiddenFeatures, config, activeModel, split, engine);
  const policy = trainPolicyHeadOnCpu(samples, hiddenFeatures, config, activeModel, split, engine);
  const model: EncodedCompactModel = encodeCompactModel({
    projectionSize: PROJECTION_SIZE,
    projectionSeed: PROJECTION_SEED,
    hiddenLayers: HIDDEN_LAYERS,
    hiddenWeights,
    outputWeights: bestOutputWeights,
    auxiliaryValueWeights: auxiliary.weights,
    policyWeights: policy.weights,
    scale: 1,
    bias: 0,
    outputActivation: "tanh"
  }, engine);
  model.trainingLoss = lastTrainLoss;
  model.initialValidationLoss = initialValidationLoss;
  model.validationLoss = lastValidationLoss;
  model.bestValidationLoss = bestValidationLoss;
  model.initialPolicyValidationLoss = policy.initialValidationLoss;
  model.policyValidationLoss = policy.validationLoss;
  model.bestPolicyValidationLoss = policy.bestValidationLoss;
  model.auxiliaryValidationLoss = auxiliary.validationLoss;
  model.auxiliaryHeadCount = AUXILIARY_VALUE_HEADS.length;
  model.valueCheckpointImproved = checkpointImproved;
  model.policyCheckpointImproved = policy.checkpointImproved;
  model.modelChanged = !activeModel?.bytes || !byteArraysEqual(model, activeModel.bytes);
  model.earlyStopReason = earlyStopReason;
  model.labelCounts = configuredLabelCounts(config);
  model.nonZeroWeights = countNonZero(bestOutputWeights, engine) + countNonZero(auxiliary.weights, engine) + countNonZero(hiddenWeights, engine) + countNonZero(policy.weights, engine);
  model.replayBufferSize = samples.length;
  model.trainingSampleCount = samples.length;
  model.policyTrainingSampleCount = policyTrainingIndices(samples, false, engine).length;
  model.hiddenLayersTrained = false;
  return model;
}

export function selectTrainingWorkingSet(samples: TrainingSample[], device: GPUDevice, engine?: ChronofishEngine): TrainingSample[] {
  const maxProjectedBytes = trainingWorkingSetMaxProjectedBytes(device);
  if (engine) {
    return selectTrainingWorkingSetWithEngine(samples, maxProjectedBytes, engine);
  }
  return selectTrainingWorkingSetFallback(samples, maxProjectedBytes);
}

function trainingWorkingSetMaxProjectedBytes(device: GPUDevice): number {
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  return maxBindingSize;
}

function selectTrainingWorkingSetWithEngine(samples: TrainingSample[], maxProjectedBytes: number, engine: ChronofishEngine): TrainingSample[] {
  const bytes = engineGpuTrainingPolicy.jsonBytes(
    engine,
    "chronofish_select_training_working_set_indexes_bytes",
    samples,
    maxProjectedBytes
  );
  if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Training working set index response is not i32-aligned.");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const selected: TrainingSample[] = [];
  for (let offset = 0; offset < bytes.byteLength; offset += Int32Array.BYTES_PER_ELEMENT) {
    const index = view.getInt32(offset, true);
    const sample = samples[index];
    if (!sample) {
      throw new Error(`Training working set index ${index} is out of range.`);
    }
    selected.push(sample);
  }
  return selected;
}

function selectTrainingWorkingSetFallback(samples: TrainingSample[], maxProjectedBytes: number): TrainingSample[] {
  const projectedBytes = samples.length * PROJECTION_SIZE * Float32Array.BYTES_PER_ELEMENT;
  if (projectedBytes <= maxProjectedBytes) {
    return samples;
  }
  const maxProjectedSamples = Math.max(1, Math.floor(maxProjectedBytes / (PROJECTION_SIZE * Float32Array.BYTES_PER_ELEMENT)));
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

function trainingSamplePriority(sample: TrainingSample, index: number, total: number, engine?: ChronofishEngine): number {
  const recency = total > 1 ? index / (total - 1) : 1;
  return trainingLabelPriority(sample.labelKind, sample.pseudo, engine) +
    trainingLabelWeight(sample.labelWeight, engine) +
    recency * 0.25;
}

export async function trainValue(
  device: GPUDevice,
  samples: TrainingSample[],
  config: TrainingConfig,
  activeModel: CompactValueModel | null,
  progress: (message: Record<string, unknown>) => void,
  engine?: ChronofishEngine
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
  const split = splitValidationSamples(samples, config.validationSplit, engine);
  const trainIndices = split.trainIndices.length ? split.trainIndices : split.validationIndices;
  const validationIndices = split.validationIndices;
  const trainGroups = groupTrainingIndicesByPosition(samples, trainIndices, engine);
  const hiddenLayersTrained = uniqueTrainingPositionCount(samples, trainIndices, engine) >= minHiddenTrainingPositions(engine);
  const batchSize = valueTrainingBatchSize(config.batchSize, trainIndices.length, engine);

  const averageLabel = labels.reduce((sum, value) => sum + value, 0) / labels.length;
  const layout = compactTrainingLayout(activeModel, averageLabel, engine);
  const initialHidden = layout.hiddenWeights;
  const layerWeights = splitHiddenWeights(initialHidden, PROJECTION_SIZE, HIDDEN_LAYERS, engine);
  const outputSize = layout.outputSize;
  const outputWeights = layout.outputWeights;

  const featureBuffer = await timed(config.metrics, "projection", () =>
    projectSamplesToBuffer(device, samples, PROJECTION_SIZE, PROJECTION_SEED, engine)
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
  const batchesPerSubmit = valueGpuBatchesPerSubmit(config.epochs, engine);
  const validationInterval = valueGpuValidationInterval(batchesPerSubmit, config.validationInterval, engine);
  const batchIndexBuffers = Array.from({ length: batchesPerSubmit }, () => device.createBuffer({
    size: align4(batchSize * Uint32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_DST
  }));
  const batchIndices = new Uint32Array(batchSize);
  const forwardLayerParams = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      batchSize,
      previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE, engine),
      layerSize,
      0,
      0,
      0,
      engine
    )
  );
  const applyLayerParams = hiddenLayersTrained
    ? HIDDEN_LAYERS.map((layerSize, layerIndex) =>
      layerParamsBuffer(
        device,
        batchSize,
        previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE, engine),
        layerSize,
        config.learningRate,
        config.weightDecay,
        OPTIMIZER_MOMENTUM,
        engine
      )
    )
    : [];
  const forwardOutputParams = outputParamsBuffer(device, batchSize, outputSize, 0, 0, 0, engine);
  const applyOutputParams = outputParamsBuffer(
    device,
    batchSize,
    outputSize,
    config.learningRate,
    config.weightDecay,
    OPTIMIZER_MOMENTUM,
    engine
  );
  const outputDeltaParams = Array.from({ length: batchesPerSubmit }, () =>
    outputDeltaParamsBuffer(device, batchSize, batchSize, engine)
  );
  const lastHiddenDeltaParams = hiddenLayersTrained
    ? hiddenDeltaParamsBuffer(device, batchSize, outputLayerSize(HIDDEN_LAYERS, engine), 0, engine)
    : null;
  const hiddenDeltaParams = hiddenLayersTrained
    ? HIDDEN_LAYERS.slice(0, -1).map((layerSize, layerIndex) =>
      hiddenDeltaParamsBuffer(device, batchSize, layerSize, HIDDEN_LAYERS[layerIndex + 1]!, engine)
    )
    : [];

  const forwardLayerEntryPoint = denseKernelEntryPoint("forward_layer", batchSize, engine);
  const applyLayerEntryPoint = denseKernelEntryPoint("apply_layer", batchSize, engine);
  const hiddenDeltaEntryPoint = denseKernelEntryPoint("hidden_delta", batchSize, engine);
  const denseKernelSuffix = denseKernelLabelSuffix("forward_layer", forwardLayerEntryPoint);
  const shaders = await loadTrainingShaders();
  const forwardIndexedLayerPipeline = await createComputePipelineChecked(device, `forward_indexed_layer_${denseKernelSuffix}`, shaders.forwardIndexedLayer, forwardLayerEntryPoint);
  const forwardLayerPipeline = await createComputePipelineChecked(device, `forward_layer_${denseKernelSuffix}`, shaders.forwardLayer, forwardLayerEntryPoint);
  const forwardOutputPipeline = await createComputePipelineChecked(device, "forward_output", shaders.forwardOutput, "forward_output");
  const reduceLossPipeline = await createComputePipelineChecked(device, "reduce_loss", shaders.reduceLoss, "reduce_loss");
  const outputDeltaPipeline = await createComputePipelineChecked(device, "output_delta", shaders.outputDelta, "output_delta");
  const lastHiddenDeltaPipeline = hiddenLayersTrained
    ? await createComputePipelineChecked(device, "hidden3_delta", shaders.hidden3Delta, "hidden3_delta")
    : null;
  const hiddenDeltaPipeline = hiddenLayersTrained
    ? await createComputePipelineChecked(device, `hidden_delta_${denseKernelSuffix}`, shaders.hiddenDelta, hiddenDeltaEntryPoint)
    : null;
  const applyIndexedLayerPipeline = hiddenLayersTrained
    ? await createComputePipelineChecked(device, `apply_indexed_layer_${denseKernelSuffix}`, shaders.applyIndexedLayer, applyLayerEntryPoint)
    : null;
  const applyLayerPipeline = hiddenLayersTrained
    ? await createComputePipelineChecked(device, `apply_layer_${denseKernelSuffix}`, shaders.applyLayer, applyLayerEntryPoint)
    : null;
  const applyOutputPipeline = await createComputePipelineChecked(device, "apply_output", shaders.applyOutput, "apply_output");

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
      reduceLossPipeline,
      engine
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
      const batchWeight = fillGroupedTrainingBatchIndices(batchIndices, trainGroups, epoch, split.seed, labelWeights, engine);
      device.queue.writeBuffer(batchIndexBuffer, 0, batchIndices);
      device.queue.writeBuffer(
        outputDeltaParams[batchSlot]!,
        0,
        outputDeltaParamsData(batchSize, batchWeight, engine)
      );
      for (let layerIndex = 0; layerIndex < HIDDEN_LAYERS.length; layerIndex += 1) {
        const inputSize = previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE, engine);
        const inputBuffer = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1]!;
        const outputSizeForLayer = HIDDEN_LAYERS[layerIndex]!;
        encodePipeline(device, encoder, layerIndex === 0 ? forwardIndexedLayerPipeline : forwardLayerPipeline, [
          inputBuffer,
          weightBuffers[layerIndex]!,
          activationBuffers[layerIndex]!,
          forwardLayerParams[layerIndex]!,
          ...(layerIndex === 0 ? [batchIndexBuffer] : [])
        ], trainingWorkgroups16(batchSize, engine), trainingWorkgroups16(outputSizeForLayer, engine));
      }

      encodePipeline(device, encoder, forwardOutputPipeline, [
        activationBuffers[activationBuffers.length - 1]!,
        outputWeightBuffer,
        predictionBuffer,
        forwardOutputParams
      ], trainingWorkgroups64(batchSize, engine));

      encodePipeline(device, encoder, outputDeltaPipeline, [
        predictionBuffer,
        labelBuffer,
        outputDeltaBuffer,
        outputDeltaParams[batchSlot]!,
        batchIndexBuffer,
        labelWeightBuffer
      ], trainingWorkgroups64(batchSize, engine));

      if (hiddenLayersTrained) {
        const lastLayerIndex = HIDDEN_LAYERS.length - 1;
        encodePipeline(device, encoder, lastHiddenDeltaPipeline!, [
          activationBuffers[lastLayerIndex]!,
          outputDeltaBuffer,
          outputWeightBuffer,
          deltaBuffers[lastLayerIndex]!,
          lastHiddenDeltaParams!
        ], trainingWorkgroups16(batchSize, engine), trainingWorkgroups16(HIDDEN_LAYERS[lastLayerIndex]!, engine));

        for (let layerIndex = HIDDEN_LAYERS.length - 2; layerIndex >= 0; layerIndex -= 1) {
          encodePipeline(device, encoder, hiddenDeltaPipeline!, [
            activationBuffers[layerIndex]!,
            deltaBuffers[layerIndex + 1]!,
            weightBuffers[layerIndex + 1]!,
            deltaBuffers[layerIndex]!,
            hiddenDeltaParams[layerIndex]!
          ], trainingWorkgroups16(batchSize, engine), trainingWorkgroups16(HIDDEN_LAYERS[layerIndex]!, engine));
        }
      }

      encodePipeline(device, encoder, applyOutputPipeline, [
        activationBuffers[activationBuffers.length - 1]!,
        outputDeltaBuffer,
        outputWeightBuffer,
        applyOutputParams,
        outputVelocityBuffer
      ], trainingWorkgroups64(outputSize + 1, engine));

      if (hiddenLayersTrained) {
        for (let layerIndex = HIDDEN_LAYERS.length - 1; layerIndex >= 0; layerIndex -= 1) {
          const inputSize = previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE, engine);
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
          ], trainingWorkgroups16(inputSize + 1, engine), trainingWorkgroups16(outputSizeForLayer, engine));
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
        reduceLossPipeline,
        engine
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
          reduceLossPipeline,
          engine
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
      labelCounts: configuredLabelCounts(config)
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
  const policyFeatureEntryPoint = denseKernelEntryPoint("forward_layer", sampleCount, engine);
  const policyFeatureKernelSuffix = denseKernelLabelSuffix("forward_layer", policyFeatureEntryPoint);
  const policyFeatureForwardPipeline = await createComputePipelineChecked(
    device,
    `forward_layer_${policyFeatureKernelSuffix}`,
    shaders.forwardLayer,
    policyFeatureEntryPoint
  );
  const policyFeatures = forwardHiddenFeaturesOnProjectedGpu(
    device,
    featureBuffer,
    bestWeightBuffers,
    sampleCount,
    policyFeatureForwardPipeline,
    engine
  );
  const trainedHidden = concatFloat32(trainedLayerWeights, engine);
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
  inputSize: number,
  engine?: ChronofishEngine
): Promise<TrainedPolicyWeights> {
  const policyIndices = policyTrainingIndices(samples, true, engine);
  const targets = new Uint32Array(samples.length);
  const labelWeights = new Float32Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index]!;
    if (!hasPolicyTrainingTarget(sample, engine)) {
      continue;
    }
    targets[index] = policyTrainingTarget(sample.policy ?? undefined, engine);
    labelWeights[index] = trainingLabelWeight(sample.labelWeight, engine);
  }
  const weightCount = POLICY_BUCKETS * (inputSize + 1);
  const initialWeights = policyWeightsArray(activeModel, inputSize, engine) ?? new Float32Array(weightCount);
  if (!policyIndices.length) {
    return {
      weights: initialWeights,
      initialValidationLoss: Number.NaN,
      validationLoss: Number.NaN,
      bestValidationLoss: Number.NaN,
      checkpointImproved: false
    };
  }
  const split = splitValidationSamples(samples, config.validationSplit, engine);
  const policySplit = splitPolicyTrainingIndices(samples, policyIndices, split, config.validationSplit ?? 0, engine);
  const trainIndices = policySplit.trainIndices.length ? policySplit.trainIndices : policyIndices;
  const validationIndices = policySplit.validationIndices;
  const trainGroups = groupTrainingIndicesByPosition(samples, trainIndices, engine);
  const batchSize = policyTrainingBatchSize(config.batchSize, trainIndices.length, engine);
  const steps = policyTrainingSteps(config.epochs, engine);
  const stepsPerSubmit = policyTrainingStepsPerSubmit(steps, engine);
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
    storageBuffer(device, policyParamsData(batchSize, inputSize, 1, config, engine), gpuBufferUsage.UNIFORM)
  );
  const batchIndices = new Uint32Array(batchSize);
  const policyForwardEntryPoint = denseKernelEntryPoint("forward_policy", batchSize, engine);
  const policyApplyEntryPoint = denseKernelEntryPoint("apply_policy", batchSize, engine);
  const policyKernelSuffix = denseKernelLabelSuffix("forward_policy", policyForwardEntryPoint);
  const shaders = await loadTrainingShaders();
  const forwardPipeline = await createComputePipelineChecked(device, `policy_forward_${policyKernelSuffix}`, shaders.policy, policyForwardEntryPoint);
  const deltaPipeline = await createComputePipelineChecked(device, "policy_delta", shaders.policy, "policy_delta");
  const applyPipeline = await createComputePipelineChecked(device, `policy_apply_${policyKernelSuffix}`, shaders.policy, policyApplyEntryPoint);
  const lossPipeline = await createComputePipelineChecked(device, "policy_loss", shaders.policyLoss, "reduce_policy_loss");
  const lossIndices = validationIndices.length ? validationIndices : trainIndices;
  const initialValidationLoss = await policyLossOnGpu(
    device,
    featureBuffer,
    targetBuffer,
    labelWeightBuffer,
    policyWeightBuffer,
    lossIndices,
    inputSize,
    lossPipeline,
    engine
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
        labelWeights,
        engine
      );
      device.queue.writeBuffer(batchIndexBuffers[slot]!, 0, batchIndices);
      device.queue.writeBuffer(
        paramsBuffers[slot]!,
        0,
        policyParamsData(batchSize, inputSize, batchWeight, config, engine)
      );
      encodePipelineBindings(device, encoder, forwardPipeline, [
        [0, featureBuffer],
        [3, policyWeightBuffer],
        [4, logitsBuffer],
        [6, batchIndexBuffers[slot]!],
        [7, paramsBuffers[slot]!]
      ], trainingWorkgroups16(batchSize, engine), trainingWorkgroups16(POLICY_BUCKETS, engine));
      encodePipelineBindings(device, encoder, deltaPipeline, [
        [1, targetBuffer],
        [2, labelWeightBuffer],
        [4, logitsBuffer],
        [5, deltaBuffer],
        [6, batchIndexBuffers[slot]!],
        [7, paramsBuffers[slot]!]
      ], trainingWorkgroups64(batchSize, engine));
      encodePipelineBindings(device, encoder, applyPipeline, [
        [0, featureBuffer],
        [3, policyWeightBuffer],
        [5, deltaBuffer],
        [6, batchIndexBuffers[slot]!],
        [7, paramsBuffers[slot]!],
        [8, policyVelocityBuffer]
      ], trainingWorkgroups16(inputSize + 1, engine), trainingWorkgroups16(POLICY_BUCKETS, engine));
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
      lossPipeline,
      engine
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

export function hasPolicyTrainingTarget(sample: TrainingSample, engine?: ChronofishEngine): boolean {
  if (engine) {
    return engineGpuTrainingPolicy.jsonBoolean(engine, "chronofish_has_policy_training_target_json", sample);
  }
  return sample.labelKind !== "distilled"
    && Number.isInteger(sample.policy)
    && (sample.policy ?? -1) >= 0;
}

export function policyTrainingIndices(samples: TrainingSample[], requirePositiveWeight: boolean, engine?: ChronofishEngine): number[] {
  if (engine) {
    return policyTrainingIndicesWithEngine(samples, requirePositiveWeight, engine);
  }
  return samples
    .map((sample, index) => ({ sample, index }))
    .filter(({ sample }) =>
      hasPolicyTrainingTarget(sample) && (!requirePositiveWeight || trainingLabelWeight(sample.labelWeight, engine) > 0)
    )
    .map(({ index }) => index);
}

function policyTrainingIndicesWithEngine(samples: TrainingSample[], requirePositiveWeight: boolean, engine: ChronofishEngine): number[] {
  const bytes = engineGpuTrainingPolicy.jsonBytes(
    engine,
    "chronofish_policy_training_indices_bytes",
    samples,
    requirePositiveWeight ? 1 : 0
  );
  if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Policy training index response is not i32-aligned.");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const indices: number[] = [];
  for (let offset = 0; offset < bytes.byteLength; offset += Int32Array.BYTES_PER_ELEMENT) {
    indices.push(view.getInt32(offset, true));
  }
  return indices;
}

export function splitPolicyTrainingIndices(
  samples: TrainingSample[],
  policyIndices: number[],
  split: ValidationSplit,
  validationSplit: number,
  engine?: ChronofishEngine
): ValidationSplit {
  if (engine) {
    return splitPolicyTrainingIndicesWithEngine(samples, policyIndices, split, validationSplit, engine);
  }
  return splitPolicyTrainingIndicesFallback(samples, policyIndices, split, validationSplit);
}

function splitPolicyTrainingIndicesWithEngine(
  samples: TrainingSample[],
  policyIndices: number[],
  split: ValidationSplit,
  validationSplit: number,
  engine: ChronofishEngine
): ValidationSplit {
  return engineGpuTrainingPolicy.jsonValue<ValidationSplit>(
    engine,
    "chronofish_split_policy_training_indices_json",
    {
    samples,
    policyIndices,
    split
    },
    validationSplit
  );
}

function splitPolicyTrainingIndicesFallback(
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

export function policyTrainingSteps(valueEpochs: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_policy_training_steps", valueEpochs);
  }
  return Math.max(16, Math.min(256, Math.ceil(valueEpochs / 64)));
}

export function policyTrainingTarget(policy: number | undefined, engine?: ChronofishEngine): number {
  const value = policy ?? 0;
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_policy_training_target", value);
  }
  return Math.min(POLICY_BUCKETS - 1, value);
}

export function trainingLabelWeight(labelWeight: number | undefined, engine?: ChronofishEngine): number {
  const value = labelWeight ?? 1;
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_training_label_weight", value);
  }
  return Math.max(0, value);
}

export function trainingWeightedAverage(total: number, totalWeight: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_training_weighted_average", total, totalWeight);
  }
  return totalWeight > 0 ? total / totalWeight : 0;
}

export function trainingBatchNormalization(batchWeight: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_training_batch_normalization", batchWeight);
  }
  return 1 / Math.max(batchWeight, 1e-6);
}

export function valueTrainingBatchSize(configBatchSize: number, trainingCount: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_value_training_batch_size", configBatchSize, trainingCount);
  }
  return Math.min(configBatchSize, Math.max(1, trainingCount));
}

export function policyTrainingBatchSize(configBatchSize: number, trainingCount: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_policy_training_batch_size", configBatchSize, trainingCount);
  }
  return Math.min(configBatchSize, trainingCount);
}

export function valueHeadValidationInterval(epochs: number, validationInterval?: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_value_head_validation_interval", epochs, validationInterval ?? -1);
  }
  return Math.max(1, Math.min(epochs, validationInterval ?? 256));
}

export function valueGpuBatchesPerSubmit(epochs: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_value_gpu_batches_per_submit", epochs);
  }
  return Math.min(64, Math.max(1, epochs));
}

export function valueGpuValidationInterval(batchesPerSubmit: number, validationInterval?: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(
      engine,
      "chronofish_value_gpu_validation_interval",
      batchesPerSubmit,
      validationInterval ?? -1
    );
  }
  return Math.max(batchesPerSubmit, validationInterval ?? 256);
}

export function policyTrainingStepsPerSubmit(steps: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_policy_training_steps_per_submit", steps);
  }
  return Math.min(64, steps);
}

async function policyLossOnGpu(
  device: GPUDevice,
  featureBuffer: GPUBuffer,
  targetBuffer: GPUBuffer,
  labelWeightBuffer: GPUBuffer,
  policyWeightBuffer: GPUBuffer,
  indices: number[],
  inputSize: number,
  pipeline: GPUComputePipeline,
  engine?: ChronofishEngine
): Promise<number> {
  if (!indices.length) {
    return 0;
  }
  const indexBuffer = storageBuffer(device, new Uint32Array(indices), gpuBufferUsage.STORAGE);
  const partialCount = lossReductionWorkgroupCount(indices.length, engine);
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
  return trainingWeightedAverage(total, totalWeight);
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

export function policyParamsData(
  batchCount: number,
  inputSize: number,
  totalWeight: number,
  config: TrainingConfig,
  engine?: ChronofishEngine
): ArrayBuffer {
  if (engine) {
    return engineGpuTrainingPolicy.byteBuffer(
      engine,
      "chronofish_policy_params_bytes",
      batchCount,
      inputSize,
      totalWeight,
      config.learningRate,
      config.weightDecay,
      OPTIMIZER_MOMENTUM
    );
  }
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

export function layerParamsData(sampleCount: number, inputSize: number, outputSize: number, learningRate: number, weightDecay = 0, momentum = 0, engine?: ChronofishEngine): ArrayBuffer {
  if (engine) {
    return engineGpuTrainingPolicy.byteBuffer(
      engine,
      "chronofish_layer_params_bytes",
      sampleCount,
      inputSize,
      outputSize,
      learningRate,
      weightDecay,
      momentum
    );
  }
  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setUint32(8, outputSize, true);
  view.setFloat32(12, learningRate, true);
  view.setFloat32(16, weightDecay, true);
  view.setFloat32(20, momentum, true);
  return params;
}

export function layerParamsBuffer(device: GPUDevice, sampleCount: number, inputSize: number, outputSize: number, learningRate: number, weightDecay = 0, momentum = 0, engine?: ChronofishEngine): GPUBuffer {
  return storageBuffer(device, layerParamsData(sampleCount, inputSize, outputSize, learningRate, weightDecay, momentum, engine), gpuBufferUsage.UNIFORM);
}

export function outputParamsData(sampleCount: number, inputSize: number, learningRate: number, weightDecay = 0, momentum = 0, engine?: ChronofishEngine): ArrayBuffer {
  if (engine) {
    return engineGpuTrainingPolicy.byteBuffer(
      engine,
      "chronofish_output_params_bytes",
      sampleCount,
      inputSize,
      learningRate,
      weightDecay,
      momentum
    );
  }
  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setFloat32(12, learningRate, true);
  view.setFloat32(16, weightDecay, true);
  view.setFloat32(20, momentum, true);
  return params;
}

export function outputParamsBuffer(device: GPUDevice, sampleCount: number, inputSize: number, learningRate: number, weightDecay = 0, momentum = 0, engine?: ChronofishEngine): GPUBuffer {
  return storageBuffer(device, outputParamsData(sampleCount, inputSize, learningRate, weightDecay, momentum, engine), gpuBufferUsage.UNIFORM);
}

export function outputDeltaParamsBuffer(device: GPUDevice, sampleCount: number, totalWeight = sampleCount, engine?: ChronofishEngine): GPUBuffer {
  return storageBuffer(device, outputDeltaParamsData(sampleCount, totalWeight, engine), gpuBufferUsage.UNIFORM);
}

export function outputDeltaParamsData(sampleCount: number, totalWeight: number, engine?: ChronofishEngine): ArrayBuffer {
  if (engine) {
    return engineGpuTrainingPolicy.byteBuffer(engine, "chronofish_output_delta_params_bytes", sampleCount, totalWeight);
  }
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setFloat32(4, Math.max(0, totalWeight), true);
  return params;
}

export function hiddenDeltaParamsBuffer(device: GPUDevice, sampleCount: number, currentSize: number, nextSize: number, engine?: ChronofishEngine): GPUBuffer {
  return storageBuffer(device, hiddenDeltaParamsData(sampleCount, currentSize, nextSize, engine), gpuBufferUsage.UNIFORM);
}

export function hiddenDeltaParamsData(sampleCount: number, currentSize: number, nextSize: number, engine?: ChronofishEngine): ArrayBuffer {
  if (engine) {
    return engineGpuTrainingPolicy.byteBuffer(
      engine,
      "chronofish_hidden_delta_params_bytes",
      sampleCount,
      currentSize,
      nextSize
    );
  }
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, currentSize, true);
  view.setUint32(8, nextSize, true);
  return params;
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

export async function projectSamplesToBuffer(device: GPUDevice, samples: TrainingSample[], projectionSize: number, seed = PROJECTION_SEED, engine?: ChronofishEngine): Promise<GPUBuffer> {
  const inputSize = featureLength(samples, engine);
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const projectedBytes = samples.length * projectionSize * Float32Array.BYTES_PER_ELEMENT;
  if (projectedBytes > maxBindingSize) {
    throw new Error(`Projected replay buffer exceeds this WebGPU device's storage binding limit (${formatBytes(projectedBytes)} > ${formatBytes(maxBindingSize)}). Reduce replay buffer or projection size.`);
  }
  const projectedBuffer = device.createBuffer({
    size: align4(projectedBytes),
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  });
  const pipeline = await createComputePipelineChecked(device, "project_features", (await loadTrainingShaders()).projectFeatures, "project_features");
  const temporaryBudget = projectionTemporaryBudget(device, engine);
  const projectionChunkSize = projectionBatchChunkSize(engine);
  let batchOffset = 0;
  while (batchOffset < samples.length) {
    const encoder = device.createCommandEncoder();
    const temporaryBuffers: GPUBuffer[] = [];
    let temporaryBytes = 0;
    let offset = batchOffset;
    while (offset < samples.length) {
      const chunkSamples = samples.slice(offset, offset + projectionChunkSize);
      const sparseFeatures = packSparseProjectionFeatures(chunkSamples, inputSize, engine);
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
        offset,
        engine
      );
      temporaryBuffers.push(offsetBuffer, indexBuffer, valueBuffer, paramsBuffer);
      temporaryBytes += sparseFeatures.byteLength;
      encodePipeline(
        device,
        encoder,
        pipeline,
        [offsetBuffer, indexBuffer, valueBuffer, projectedBuffer, paramsBuffer],
        trainingWorkgroups16(chunkSamples.length, engine),
        trainingWorkgroups16(projectionSize, engine)
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

export function projectionTemporaryBudget(device: GPUDevice, engine?: ChronofishEngine): number {
  const maxBufferSize = device.limits?.maxBufferSize ?? 256 * 1024 * 1024;
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_projection_temporary_budget", maxBufferSize);
  }
  return Math.min(
    PROJECTION_TEMPORARY_BUDGET,
    Math.max(1, Math.floor(maxBufferSize / 2))
  );
}

export function cpuPredictionMaxBatch(engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_cpu_prediction_max_batch");
  }
  return CPU_PREDICTION_MAX_BATCH;
}

export function cpuHeadTrainingMaxPositions(engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_cpu_head_training_max_positions");
  }
  return CPU_HEAD_TRAINING_MAX_POSITIONS;
}

export function minHiddenTrainingPositions(engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_min_hidden_training_positions");
  }
  return MIN_HIDDEN_TRAINING_POSITIONS;
}

export function projectionBatchChunkSize(engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_projection_chunk_size");
  }
  return PROJECTION_CHUNK_SIZE;
}

export function packSparseProjectionFeatures(samples: TrainingSample[], inputSize?: number, engine?: ChronofishEngine): SparseProjectionFeatures {
  const resolvedInputSize = inputSize ?? featureLength(samples, engine);
  if (engine) {
    return packSparseProjectionFeaturesWithEngine(samples, resolvedInputSize, engine);
  }
  return packSparseProjectionFeaturesFallback(samples, resolvedInputSize);
}

function packSparseProjectionFeaturesWithEngine(samples: TrainingSample[], inputSize: number, engine: ChronofishEngine): SparseProjectionFeatures {
  return readSparseProjectionFeatures(engineGpuTrainingPolicy.jsonBytes(
    engine,
    "chronofish_sparse_projection_features_bytes",
    samples,
    inputSize
  ));
}

function readSparseProjectionFeatures(bytes: Uint8Array): SparseProjectionFeatures {
  if (bytes.byteLength < 16) {
    throw new Error("Sparse projection feature response is truncated.");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const offsetsLength = view.getUint32(0, true);
  const indicesLength = view.getUint32(4, true);
  const valuesLength = view.getUint32(8, true);
  const byteLength = view.getUint32(12, true);
  const expectedLength = 16 + (offsetsLength + indicesLength + valuesLength) * Uint32Array.BYTES_PER_ELEMENT;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(`Sparse projection feature response has ${bytes.byteLength} bytes but expected ${expectedLength}.`);
  }
  const offsets = new Uint32Array(offsetsLength);
  const indices = new Uint32Array(indicesLength);
  const values = new Float32Array(valuesLength);
  let cursor = 16;
  for (let index = 0; index < offsetsLength; index += 1) {
    offsets[index] = view.getUint32(cursor, true);
    cursor += 4;
  }
  for (let index = 0; index < indicesLength; index += 1) {
    indices[index] = view.getUint32(cursor, true);
    cursor += 4;
  }
  for (let index = 0; index < valuesLength; index += 1) {
    values[index] = view.getFloat32(cursor, true);
    cursor += 4;
  }
  return { offsets, indices, values, byteLength };
}

function packSparseProjectionFeaturesFallback(samples: TrainingSample[], inputSize: number): SparseProjectionFeatures {
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

export function projectionParamsData(sampleCount: number, inputSize: number, projectionSize: number, seed: number, outputOffset = 0, engine?: ChronofishEngine): ArrayBuffer {
  if (engine) {
    return engineGpuTrainingPolicy.byteBuffer(
      engine,
      "chronofish_projection_params_bytes",
      sampleCount,
      inputSize,
      projectionSize,
      seed >>> 0,
      outputOffset
    );
  }
  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, sampleCount, true);
  view.setUint32(4, inputSize, true);
  view.setUint32(8, projectionSize, true);
  view.setUint32(12, seed >>> 0, true);
  view.setUint32(16, outputOffset, true);
  return params;
}

export function projectionParamsBuffer(device: GPUDevice, sampleCount: number, inputSize: number, projectionSize: number, seed: number, outputOffset = 0, engine?: ChronofishEngine): GPUBuffer {
  return storageBuffer(device, projectionParamsData(sampleCount, inputSize, projectionSize, seed, outputOffset, engine), gpuBufferUsage.UNIFORM);
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

export async function predictionLossOnGpu(device: GPUDevice, samples: TrainingSample[], model: PredictionModel, engine?: ChronofishEngine): Promise<number> {
  const predictions = await predictValuesOnGpu(device, samples, model, engine);
  let total = 0;
  let totalWeight = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const error = predictions[index]! - samples[index]!.label;
    const weight = trainingLabelWeight(samples[index]!.labelWeight);
    total += weight * error * error;
    totalWeight += weight;
  }
  return trainingWeightedAverage(total, totalWeight);
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
  reduceLossPipeline: GPUComputePipeline,
  engine?: ChronofishEngine
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
  const partialCount = lossReductionWorkgroupCount(sampleCount, engine);
  const partialLossBuffer = device.createBuffer({
    size: align4(partialCount * 2 * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  });
  const forwardLayerParams = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      sampleCount,
      previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE, engine),
      layerSize,
      0,
      0,
      0,
      engine
    )
  );
  const forwardOutputParams = outputParamsBuffer(device, sampleCount, outputLayerSize(HIDDEN_LAYERS, engine), 0, 0, 0, engine);
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
    ], trainingWorkgroups16(sampleCount, engine), trainingWorkgroups16(outputSizeForLayer, engine));
  }
  encodePipeline(device, encoder, forwardOutputPipeline, [
    activationBuffers[activationBuffers.length - 1]!,
    outputWeightBuffer,
    predictionBuffer,
    forwardOutputParams
  ], trainingWorkgroups64(sampleCount, engine));
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
  return trainingWeightedAverage(total, totalWeight);
}

function forwardHiddenFeaturesOnProjectedGpu(
  device: GPUDevice,
  featureBuffer: GPUBuffer,
  weightBuffers: GPUBuffer[],
  sampleCount: number,
  forwardLayerPipeline: GPUComputePipeline,
  engine?: ChronofishEngine
): { featureBuffer: GPUBuffer; resources: GPUBuffer[] } {
  const activationBuffers = HIDDEN_LAYERS.map((layerSize) => device.createBuffer({
    size: align4(sampleCount * layerSize * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE
  }));
  const params = HIDDEN_LAYERS.map((layerSize, layerIndex) =>
    layerParamsBuffer(
      device,
      sampleCount,
      previousLayerSize(HIDDEN_LAYERS, layerIndex, PROJECTION_SIZE, engine),
      layerSize,
      0,
      0,
      0,
      engine
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
    ], trainingWorkgroups16(sampleCount, engine), trainingWorkgroups16(HIDDEN_LAYERS[layerIndex]!, engine));
  }
  device.queue.submit([encoder.finish()]);
  return {
    featureBuffer: activationBuffers[activationBuffers.length - 1]!,
    resources: [...activationBuffers, ...params]
  };
}

export function lossReductionWorkgroupCount(sampleCount: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_loss_reduction_workgroup_count", sampleCount);
  }
  return Math.max(1, Math.ceil(sampleCount / 64));
}

export function trainingWorkgroups16(itemCount: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_training_workgroups_16", itemCount);
  }
  return Math.ceil(itemCount / 16);
}

export function trainingWorkgroups64(itemCount: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_training_workgroups_64", itemCount);
  }
  return Math.ceil(itemCount / 64);
}

export function denseKernelEntryPoint(entryPoint: string, sampleCount: number, engine?: ChronofishEngine): string {
  if (engine) {
    return engineGpuTrainingPolicy.textValue(engine, "chronofish_dense_kernel_entry_point_bytes", entryPoint, sampleCount);
  }
  return fallbackDenseKernelEntryPoint(entryPoint, sampleCount);
}

function denseKernelLabelSuffix(entryPoint: string, selectedEntryPoint: string): string {
  return selectedEntryPoint === entryPoint ? "tiled" : "naive";
}

function lossParamsBuffer(device: GPUDevice, sampleCount: number): GPUBuffer {
  return storageBuffer(device, new Uint32Array([sampleCount, 0, 0, 0]), gpuBufferUsage.UNIFORM);
}

export function splitHiddenWeights(hiddenWeights: Float32Array, inputSize: number, hiddenLayers: number[], engine?: ChronofishEngine): Float32Array[] {
  if (engine) {
    const request = hiddenWeightSplitRequestBytes(hiddenWeights, inputSize, hiddenLayers);
    const output = engineGpuTrainingPolicy.bytesRequired(engine, "chronofish_split_hidden_weights_bytes", request);
    return readSplitHiddenWeights(output);
  }
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

function hiddenWeightSplitRequestBytes(hiddenWeights: Float32Array, inputSize: number, hiddenLayers: number[]): Uint8Array {
  const byteLength = 12 + hiddenLayers.length * Uint32Array.BYTES_PER_ELEMENT + hiddenWeights.byteLength;
  const bytes = new Uint8Array(byteLength);
  const view = new DataView(bytes.buffer);
  let cursor = 0;
  view.setUint32(cursor, inputSize, true);
  cursor += 4;
  view.setUint32(cursor, hiddenLayers.length, true);
  cursor += 4;
  view.setUint32(cursor, hiddenWeights.length, true);
  cursor += 4;
  for (const layerSize of hiddenLayers) {
    view.setUint32(cursor, layerSize, true);
    cursor += 4;
  }
  bytes.set(new Uint8Array(hiddenWeights.buffer, hiddenWeights.byteOffset, hiddenWeights.byteLength), cursor);
  return bytes;
}

function readSplitHiddenWeights(bytes: Uint8Array): Float32Array[] {
  if (bytes.byteLength < 4) {
    throw new Error("Hidden weight split response is truncated.");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const layerCount = view.getUint32(0, true);
  const headerLength = 4 + layerCount * Uint32Array.BYTES_PER_ELEMENT;
  if (bytes.byteLength < headerLength) {
    throw new Error("Hidden weight split response is missing layer lengths.");
  }
  const lengths: number[] = [];
  let cursor = 4;
  let valueCount = 0;
  for (let index = 0; index < layerCount; index += 1) {
    const length = view.getUint32(cursor, true);
    lengths.push(length);
    valueCount += length;
    cursor += 4;
  }
  const expectedLength = headerLength + valueCount * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(`Hidden weight split response has ${bytes.byteLength} bytes but expected ${expectedLength}.`);
  }
  const layers: Float32Array[] = [];
  for (const length of lengths) {
    const layer = new Float32Array(length);
    for (let index = 0; index < length; index += 1) {
      layer[index] = view.getFloat32(cursor, true);
      cursor += 4;
    }
    layers.push(layer);
  }
  return layers;
}

export function concatFloat32(arrays: Float32Array[], engine?: ChronofishEngine): Float32Array {
  if (engine) {
    const request = concatFloat32RequestBytes(arrays);
    const output = engineGpuTrainingPolicy.bytesRequired(engine, "chronofish_concat_f32_bytes", request);
    return float32ArrayFromBytes(output);
  }
  const length = arrays.reduce((sum, array) => sum + array.length, 0);
  const result = new Float32Array(length);
  let cursor = 0;
  for (const array of arrays) {
    result.set(array, cursor);
    cursor += array.length;
  }
  return result;
}

function concatFloat32RequestBytes(arrays: Float32Array[]): Uint8Array {
  const valueBytes = arrays.reduce((sum, array) => sum + array.byteLength, 0);
  const bytes = new Uint8Array(4 + arrays.length * Uint32Array.BYTES_PER_ELEMENT + valueBytes);
  const view = new DataView(bytes.buffer);
  let cursor = 0;
  view.setUint32(cursor, arrays.length, true);
  cursor += 4;
  for (const array of arrays) {
    view.setUint32(cursor, array.length, true);
    cursor += 4;
  }
  for (const array of arrays) {
    bytes.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength), cursor);
    cursor += array.byteLength;
  }
  return bytes;
}

function float32ArrayFromBytes(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Float32 response length is not a multiple of f32 size.");
  }
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Float32Array.BYTES_PER_ELEMENT).slice();
}

export function countNonZero(values: ArrayLike<number>, engine?: ChronofishEngine): number {
  if (engine) {
    const array = values instanceof Float32Array ? values : Float32Array.from(Array.from(values));
    return engineGpuTrainingPolicy.bytesNumeric(
      engine,
      "chronofish_count_non_zero_f32_bytes",
      new Uint8Array(array.buffer, array.byteOffset, array.byteLength)
    );
  }
  let count = 0;
  for (let index = 0; index < values.length; index += 1) {
    if ((values[index] ?? 0) !== 0) {
      count += 1;
    }
  }
  return count;
}

export async function predictValues(samples: TrainingSample[], model: CompactValueModel | null, engine?: ChronofishEngine): Promise<number[]> {
  if (!samples.length) {
    return [];
  }
  if (!modelArchitectureMatches(model, engine)) {
    throw new Error("GPU batch prediction unavailable.");
  }
  if (samples.length <= cpuPredictionMaxBatch(engine)) {
    if (engine) {
      return predictValuesWithEngine(engine, samples, model);
    }
    return Array.from(predictValuesOnCpu(samples, model));
  }
  if (!globalThis.navigator?.gpu) {
    throw new Error("GPU batch prediction unavailable.");
  }
  const device = await getGpuDevice();
  if (!device) {
    throw new Error("No WebGPU adapter is available.");
  }
  return Array.from(await predictValuesOnGpu(device, samples, model, engine));
}

function compactTrainingLayoutWithEngine(engine: ChronofishEngine, activeModel: CompactValueModel | null, averageLabel: number): CompactTrainingLayout {
  const modelBytes = activeModel?.bytes ?? (activeModel ? encodeCompactModel(activeModel, engine) : new Uint8Array());
  const output = engineGpuTrainingPolicy.bytesRequired(
    engine,
    "chronofish_compact_value_model_training_layout_bytes",
    modelBytes,
    averageLabel
  );
  return readCompactTrainingLayout(output);
}

function compactTrainingLayoutFallback(activeModel: CompactValueModel | null, averageLabel: number): CompactTrainingLayout {
  const active = modelArchitectureMatches(activeModel);
  const hiddenWeights = active
    ? activeModel.hiddenWeights.slice()
    : initialHiddenWeights(PROJECTION_SIZE, HIDDEN_LAYERS);
  const outputSize = outputLayerSize();
  const outputWeights = active && activeModel.outputWeights.length === outputSize + 1
    ? activeModel.outputWeights.slice()
    : new Float32Array(outputSize + 1);
  if (!active || activeModel.outputWeights.length !== outputSize + 1) {
    outputWeights[outputSize] = inverseTanh(averageLabel);
  }
  return {
    architectureMatches: active,
    outputSize,
    hiddenWeights,
    outputWeights
  };
}

function compactTrainingLayout(activeModel: CompactValueModel | null, averageLabel: number, engine?: ChronofishEngine): CompactTrainingLayout {
  return engine
    ? compactTrainingLayoutWithEngine(engine, activeModel, averageLabel)
    : compactTrainingLayoutFallback(activeModel, averageLabel);
}

function readCompactTrainingLayout(bytes: Uint8Array): CompactTrainingLayout {
  if (bytes.byteLength < 16) {
    throw new Error("Compact value model training layout is truncated.");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const architectureMatches = view.getUint32(0, true) !== 0;
  const outputSize = view.getUint32(4, true);
  const hiddenLength = view.getUint32(8, true);
  const outputLength = view.getUint32(12, true);
  const expectedLength = 16 + (hiddenLength + outputLength) * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(`Compact value model training layout has ${bytes.byteLength} bytes but expected ${expectedLength}.`);
  }
  const hiddenWeights = new Float32Array(hiddenLength);
  const outputWeights = new Float32Array(outputLength);
  let cursor = 16;
  for (let index = 0; index < hiddenLength; index += 1) {
    hiddenWeights[index] = view.getFloat32(cursor, true);
    cursor += 4;
  }
  for (let index = 0; index < outputLength; index += 1) {
    outputWeights[index] = view.getFloat32(cursor, true);
    cursor += 4;
  }
  return {
    architectureMatches,
    outputSize,
    hiddenWeights,
    outputWeights
  };
}

function predictValuesWithEngine(engine: ChronofishEngine, samples: TrainingSample[], model: CompactValueModel): number[] {
  const modelBytes = model.bytes ?? encodeCompactModel(model, engine);
  return engineGpuTrainingPolicy.bytesJsonValue(
    engine,
    "chronofish_compact_value_model_predict_values_json",
    modelBytes,
    samples
  );
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

function hiddenFeaturesForSamples(samples: TrainingSample[], hiddenWeights: Float32Array, engine?: ChronofishEngine): Float32Array[] {
  if (engine) {
    const model = encodeCompactModel({
      projectionSize: PROJECTION_SIZE,
      projectionSeed: PROJECTION_SEED,
      hiddenLayers: HIDDEN_LAYERS,
      hiddenWeights,
      outputWeights: new Float32Array(outputLayerSize(HIDDEN_LAYERS, engine) + 1),
      scale: 1,
      bias: 0,
      outputActivation: "tanh"
    }, engine);
    return engineGpuTrainingPolicy.bytesJsonValue<number[][]>(
      engine,
      "chronofish_compact_value_model_hidden_features_json",
      model,
      samples
    ).map((features) => Float32Array.from(features));
  }
  return samples.map((sample) =>
    hiddenFeaturesOnCpu(sample, PROJECTION_SIZE, PROJECTION_SEED, HIDDEN_LAYERS, hiddenWeights)
  );
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
    const weight = trainingLabelWeight(samples[index]!.labelWeight);
    const error = prediction - samples[index]!.label;
    total += weight * error * error;
    totalWeight += weight;
  }
  return trainingWeightedAverage(total, totalWeight);
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
    const scale = 2 * trainingLabelWeight(samples[sampleIndex]!.labelWeight)
      * (prediction - samples[sampleIndex]!.label)
      * (1 - prediction * prediction);
    for (let input = 0; input < feature.length; input += 1) {
      gradient[input] = (gradient[input] ?? 0) + scale * feature[input]!;
    }
    gradient[biasIndex] = (gradient[biasIndex] ?? 0) + scale;
  }
  const normalization = trainingBatchNormalization(batchWeight);
  for (let input = 0; input < weights.length; input += 1) {
    const decay = input === biasIndex ? 0 : weightDecay * weights[input]!;
    const update = gradient[input]! * normalization + decay;
    velocity[input] = optimizerVelocity(velocity[input] ?? 0, update);
    weights[input] = (weights[input] ?? 0) - learningRate * velocity[input]!;
  }
}

function trainAuxiliaryValueHeadsOnCpu(
  samples: TrainingSample[],
  features: Float32Array[],
  config: TrainingConfig,
  activeModel: CompactValueModel | null,
  valueSplit: ValidationSplit,
  engine?: ChronofishEngine
): TrainedAuxiliaryValueWeights {
  const inputSize = features[0]?.length ?? outputLayerSize(HIDDEN_LAYERS, engine);
  const rowSize = inputSize + 1;
  const expected = AUXILIARY_VALUE_HEADS.length * rowSize;
  const weights = modelArchitectureMatches(activeModel, engine)
    && activeModel?.auxiliaryValueWeights?.length === expected
    ? activeModel.auxiliaryValueWeights.slice()
    : new Float32Array(expected);
  const velocity = new Float32Array(weights.length);
  const trainIndices = valueSplit.trainIndices.length ? valueSplit.trainIndices : valueSplit.validationIndices;
  const validationIndices = valueSplit.validationIndices.length ? valueSplit.validationIndices : trainIndices;
  if (!trainIndices.length) {
    return { weights, validationLoss: 0 };
  }
  const trainGroups = groupTrainingIndicesByPosition(samples, trainIndices, engine);
  const batchSize = valueTrainingBatchSize(config.batchSize, trainIndices.length, engine);
  const batchIndices = new Uint32Array(batchSize);
  const labelWeights = Float32Array.from(samples, (sample) => trainingLabelWeight(sample.labelWeight, engine));
  const targets = auxiliaryValueTargetsForSamples(samples, engine);
  const steps = Math.max(1, Math.min(config.epochs, policyTrainingSteps(config.epochs, engine)));
  for (let step = 1; step <= steps; step += 1) {
    const batchWeight = fillGroupedTrainingBatchIndices(batchIndices, trainGroups, step, valueSplit.seed ^ 0xa77a11, labelWeights, engine);
    applyAuxiliaryValueHeadGradientOnCpu(features, samples, targets, weights, batchIndices, batchWeight, config, velocity);
  }
  return {
    weights,
    validationLoss: auxiliaryValueHeadLossOnCpu(features, samples, targets, weights, validationIndices)
  };
}

export function auxiliaryValueTargetsForSamples(samples: TrainingSample[], engine?: ChronofishEngine): Float32Array {
  if (engine) {
    const bytes = engineGpuTrainingPolicy.jsonBytes(engine, "chronofish_auxiliary_value_targets_bytes", samples);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Float32Array.BYTES_PER_ELEMENT).slice();
  }
  const targets = new Float32Array(samples.length * AUXILIARY_VALUE_HEADS.length);
  for (let index = 0; index < samples.length; index += 1) {
    targets.set(auxiliaryValueTargets(samples[index]!), index * AUXILIARY_VALUE_HEADS.length);
  }
  return targets;
}

function auxiliaryValueTargets(sample: TrainingSample): number[] {
  const features = sample.features;
  const boardStride = NEURAL_BOARD_PLANES * NEURAL_BOARD_SQUARES;
  const boardCount = Math.max(1, Math.min(sample.boardCount ?? Math.floor(features.length / boardStride), Math.floor(features.length / boardStride)));
  let activeBoards = 0;
  let presentBoards = 0;
  let royalDanger = 0;
  let activeMaterial = 0;
  let inactiveMaterial = 0;
  for (let board = 0; board < boardCount; board += 1) {
    const base = board * boardStride;
    const active = features[base + 25 * NEURAL_BOARD_SQUARES] ?? 0;
    const present = features[base + 27 * NEURAL_BOARD_SQUARES] ?? 0;
    const royal = features[base + 31 * NEURAL_BOARD_SQUARES] ?? 0;
    activeBoards += active > 0 ? 1 : 0;
    presentBoards += present > 0 ? 1 : 0;
    royalDanger = Math.max(royalDanger, royal);
    const material = materialBalanceForEncodedBoard(features, base);
    if (active > 0) {
      activeMaterial += material;
    } else {
      inactiveMaterial += material;
    }
  }
  return [
    sample.label > 0.05 ? 1 : sample.label < -0.05 ? -1 : 0,
    boundedValue(sample.label),
    royalDanger,
    boundedValue(1 - 2 * royalDanger),
    boundedValue(activeBoards / boardCount),
    boundedValue(presentBoards / boardCount),
    boundedValue(activeMaterial / 16),
    boundedValue(inactiveMaterial / 16),
    Number.isInteger(sample.policy) ? 0 : 1
  ];
}

function materialBalanceForEncodedBoard(features: number[] | Float32Array, boardBase: number): number {
  let balance = 0;
  for (let plane = 0; plane < 24; plane += 1) {
    const pieceValue = encodedPieceValue(plane % 12);
    const sign = plane < 12 ? 1 : -1;
    for (let square = 0; square < NEURAL_BOARD_SQUARES; square += 1) {
      balance += sign * pieceValue * (features[boardBase + plane * NEURAL_BOARD_SQUARES + square] ?? 0);
    }
  }
  return balance;
}

function encodedPieceValue(pieceType: number): number {
  if (pieceType === 0 || pieceType === 3) return 8;
  if (pieceType === 2) return 5;
  if (pieceType === 4 || pieceType === 5) return 4;
  if (pieceType === 6 || pieceType === 9) return 3;
  if (pieceType === 7 || pieceType === 8) return 2;
  return 1;
}

function auxiliaryValueHeadLossOnCpu(
  features: Float32Array[],
  samples: TrainingSample[],
  targets: Float32Array,
  weights: Float32Array,
  indices: number[]
): number {
  const inputSize = features[0]?.length ?? 0;
  const rowSize = inputSize + 1;
  let total = 0;
  let totalWeight = 0;
  for (const index of indices) {
    const feature = features[index]!;
    const weight = trainingLabelWeight(samples[index]!.labelWeight);
    const targetOffset = index * AUXILIARY_VALUE_HEADS.length;
    for (let head = 0; head < AUXILIARY_VALUE_HEADS.length; head += 1) {
      const row = head * rowSize;
      let logit = weights[row + inputSize] ?? 0;
      for (let input = 0; input < inputSize; input += 1) {
        logit += feature[input]! * (weights[row + input] ?? 0);
      }
      const prediction = Math.tanh(logit);
      const error = prediction - (targets[targetOffset + head] ?? 0);
      total += weight * error * error;
      totalWeight += weight;
    }
  }
  return trainingWeightedAverage(total, totalWeight);
}

function applyAuxiliaryValueHeadGradientOnCpu(
  features: Float32Array[],
  samples: TrainingSample[],
  targets: Float32Array,
  weights: Float32Array,
  batchIndices: Uint32Array,
  batchWeight: number,
  config: TrainingConfig,
  velocity: Float32Array
): void {
  const inputSize = features[0]?.length ?? 0;
  const rowSize = inputSize + 1;
  const gradient = new Float32Array(weights.length);
  for (const sampleIndex of batchIndices) {
    const feature = features[sampleIndex]!;
    const sampleWeight = trainingLabelWeight(samples[sampleIndex]!.labelWeight);
    const targetOffset = sampleIndex * AUXILIARY_VALUE_HEADS.length;
    for (let head = 0; head < AUXILIARY_VALUE_HEADS.length; head += 1) {
      const row = head * rowSize;
      let logit = weights[row + inputSize] ?? 0;
      for (let input = 0; input < inputSize; input += 1) {
        logit += feature[input]! * (weights[row + input] ?? 0);
      }
      const prediction = Math.tanh(logit);
      const scale = 2 * sampleWeight * (prediction - (targets[targetOffset + head] ?? 0)) * (1 - prediction * prediction);
      for (let input = 0; input < inputSize; input += 1) {
        gradient[row + input] = (gradient[row + input] ?? 0) + scale * feature[input]!;
      }
      gradient[row + inputSize] = (gradient[row + inputSize] ?? 0) + scale;
    }
  }
  const normalization = trainingBatchNormalization(batchWeight);
  for (let index = 0; index < weights.length; index += 1) {
    const isBias = index % rowSize === inputSize;
    const decay = isBias ? 0 : config.weightDecay * weights[index]!;
    const update = gradient[index]! * normalization + decay;
    velocity[index] = optimizerVelocity(velocity[index] ?? 0, update);
    weights[index] = (weights[index] ?? 0) - config.learningRate * velocity[index]!;
  }
}

function trainPolicyHeadOnCpu(
  samples: TrainingSample[],
  features: Float32Array[],
  config: TrainingConfig,
  activeModel: CompactValueModel | null,
  valueSplit: ValidationSplit,
  engine?: ChronofishEngine
): TrainedPolicyWeights {
  const policyIndices = policyTrainingIndices(samples, true, engine);
  const inputSize = features[0]?.length ?? outputLayerSize(HIDDEN_LAYERS, engine);
  const initialWeights = policyWeightsArray(activeModel, inputSize, engine)
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
  const split = splitPolicyTrainingIndices(samples, policyIndices, valueSplit, config.validationSplit ?? 0, engine);
  const trainIndices = split.trainIndices.length ? split.trainIndices : policyIndices;
  const validationIndices = split.validationIndices;
  const trainGroups = groupTrainingIndicesByPosition(samples, trainIndices, engine);
  const batchSize = policyTrainingBatchSize(config.batchSize, trainIndices.length, engine);
  const batchIndices = new Uint32Array(batchSize);
  const labelWeights = Float32Array.from(samples, (sample) => trainingLabelWeight(sample.labelWeight, engine));
  const weights = initialWeights.slice();
  const velocity = new Float32Array(weights.length);
  let bestWeights = weights.slice();
  const lossIndices = validationIndices.length ? validationIndices : trainIndices;
  const initialValidationLoss = policyHeadLossOnCpu(samples, features, weights, lossIndices);
  let validationLoss = initialValidationLoss;
  let bestValidationLoss = initialValidationLoss;
  let checkpointImproved = false;
  const steps = policyTrainingSteps(config.epochs, engine);
  for (let step = 1; step <= steps; step += 1) {
    const batchWeight = fillGroupedTrainingBatchIndices(batchIndices, trainGroups, step, split.seed, labelWeights, engine);
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
    const target = policyTrainingTarget(samples[index]!.policy ?? undefined);
    const weight = trainingLabelWeight(samples[index]!.labelWeight);
    total += weight * (Math.log(Math.max(denominator, 1e-12)) - (logits[target]! - maxLogit));
    totalWeight += weight;
  }
  return trainingWeightedAverage(total, totalWeight);
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
    const target = policyTrainingTarget(samples[sampleIndex]!.policy ?? undefined);
    const sampleWeight = trainingLabelWeight(samples[sampleIndex]!.labelWeight);
    for (let bucket = 0; bucket < POLICY_BUCKETS; bucket += 1) {
      const delta = ((logits[bucket]! / denominator) - (bucket === target ? 1 : 0)) * sampleWeight;
      const row = bucket * rowSize;
      for (let input = 0; input < inputSize; input += 1) {
        gradient[row + input] = (gradient[row + input] ?? 0) + delta * feature[input]!;
      }
      gradient[row + inputSize] = (gradient[row + inputSize] ?? 0) + delta;
    }
  }
  const normalization = trainingBatchNormalization(batchWeight);
  for (let index = 0; index < weights.length; index += 1) {
    const isBias = index % rowSize === inputSize;
    const decay = isBias ? 0 : config.weightDecay * weights[index]!;
    const update = gradient[index]! * normalization + decay;
    velocity[index] = optimizerVelocity(velocity[index] ?? 0, update);
    weights[index] = (weights[index] ?? 0) - config.learningRate * velocity[index]!;
  }
}

export function optimizerVelocity(previous: number, gradient: number, momentum = OPTIMIZER_MOMENTUM, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_optimizer_velocity", previous, gradient, momentum);
  }
  return momentum * previous + (1 - momentum) * gradient;
}

export function boundedValue(value: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_bounded_value", value);
  }
  return Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0));
}

export function normalizedSearchScore(score: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_normalized_search_score", score);
  }
  return boundedValue(score / VALUE_SCORE_SCALE);
}

export function denormalizedSearchScore(value: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_denormalized_search_score", value);
  }
  return Math.round(boundedValue(value) * VALUE_SCORE_SCALE);
}

export function inverseTanh(value: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_inverse_tanh", value);
  }
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

export async function predictValuesOnGpu(device: GPUDevice, samples: TrainingSample[], model: CompactValueModel, engine?: ChronofishEngine): Promise<Float32Array> {
  const sampleCount = samples.length;
  const featureBuffer = await projectSamplesToBuffer(
    device,
    samples,
    model.projectionSize,
    model.projectionSeed,
    engine
  );
  const hiddenLayers = model.hiddenLayers;
  const finalHiddenSize = outputLayerSize(hiddenLayers, engine);
  const layerWeights = splitHiddenWeights(model.hiddenWeights, model.projectionSize, hiddenLayers, engine);
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
      previousLayerSize(hiddenLayers, layerIndex, model.projectionSize, engine),
      layerSize,
      0
    )
  );
  const forwardOutputParams = outputParamsBuffer(device, sampleCount, finalHiddenSize, 0, 0, 0, engine);
  const forwardLayerEntryPoint = denseKernelEntryPoint("forward_layer", sampleCount, engine);
  const shaders = await loadTrainingShaders();
  const forwardLayerPipeline = await createComputePipelineChecked(device, `forward_layer_${denseKernelLabelSuffix("forward_layer", forwardLayerEntryPoint)}`, shaders.forwardLayer, forwardLayerEntryPoint);
  const forwardOutputPipeline = await createComputePipelineChecked(device, "forward_output", shaders.forwardOutput, "forward_output");
  const encoder = device.createCommandEncoder();
  for (let layerIndex = 0; layerIndex < hiddenLayers.length; layerIndex += 1) {
    const inputBuffer = layerIndex === 0 ? featureBuffer : activationBuffers[layerIndex - 1]!;
    encodePipeline(device, encoder, forwardLayerPipeline, [
      inputBuffer,
      weightBuffers[layerIndex]!,
      activationBuffers[layerIndex]!,
      forwardLayerParams[layerIndex]!
    ], trainingWorkgroups16(sampleCount, engine), trainingWorkgroups16(hiddenLayers[layerIndex]!, engine));
  }
  encodePipeline(device, encoder, forwardOutputPipeline, [
    activationBuffers[activationBuffers.length - 1]!,
    outputWeightBuffer,
    predictionBuffer,
    forwardOutputParams
  ], trainingWorkgroups64(sampleCount, engine));
  device.queue.submit([encoder.finish()]);
  const predictions = await readFloats(device, predictionBuffer, sampleCount * Float32Array.BYTES_PER_ELEMENT);
  const scale = model.scale ?? 1;
  const bias = model.bias ?? 0;
  for (let index = 0; index < predictions.length; index += 1) {
    predictions[index] = boundedValue((predictions[index] ?? 0) * scale + bias, engine);
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

export function modelArchitectureMatches(model: CompactValueModel | null | undefined, engine?: ChronofishEngine): model is CompactValueModel {
  if (model && engine) {
    const modelBytes = model.bytes ?? encodeCompactModel(model, engine);
    return Boolean(engineGpuTrainingPolicy.bytesNumeric(
      engine,
      "chronofish_compact_value_model_architecture_matches_bytes",
      modelBytes
    ));
  }
  return Boolean(model
    && model.projectionSize === PROJECTION_SIZE
    && model.projectionSeed === PROJECTION_SEED
    && JSON.stringify(model.hiddenLayers) === JSON.stringify(HIDDEN_LAYERS)
    && model.hiddenWeights?.length
    && model.outputActivation === "tanh"
    && compactModelIsFinite(model));
}

export function projectionHash(rawIndex: number, projectionIndex: number, seed: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engineGpuTrainingPolicy.numeric(engine, "chronofish_projection_hash", rawIndex, projectionIndex, seed) >>> 0;
  }
  let hash = (seed ^ rawIndex) >>> 0;
  hash = Math.imul(hash, 16777619) >>> 0;
  hash = (hash ^ projectionIndex) >>> 0;
  hash = Math.imul(hash, 16777619) >>> 0;
  hash = (hash ^ (hash >>> 16)) >>> 0;
  return hash;
}

export function initialHiddenWeights(inputSize: number, hiddenLayers: number[], engine?: ChronofishEngine): Float32Array {
  if (engine) {
    if (inputSize === PROJECTION_SIZE && isDefaultHiddenLayers(hiddenLayers)) {
      return new Float32Array(engineGpuTrainingPolicy.byteBuffer(engine, "chronofish_default_initial_hidden_weights_bytes"));
    }
    const request = initialHiddenWeightsRequestBytes(inputSize, hiddenLayers);
    return new Float32Array(engineGpuTrainingPolicy.bytesBuffer(engine, "chronofish_initial_hidden_weights_bytes", request));
  }
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

function initialHiddenWeightsRequestBytes(inputSize: number, hiddenLayers: number[]): Uint8Array {
  const bytes = new Uint8Array((2 + hiddenLayers.length) * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  view.setUint32(offset, inputSize, true);
  offset += 4;
  view.setUint32(offset, hiddenLayers.length, true);
  offset += 4;
  for (const layerSize of hiddenLayers) {
    view.setUint32(offset, layerSize, true);
    offset += 4;
  }
  return bytes;
}
