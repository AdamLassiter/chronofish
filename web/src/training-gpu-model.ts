import * as engineGpuModel from "./engine-gpu-model-binding.js";
import type { ChronofishEngine } from "./types.js";

export interface CompactValueModel {
  projectionSize: number;
  projectionSeed: number;
  hiddenLayers: number[];
  hiddenWeights: Float32Array;
  outputWeights: Float32Array;
  auxiliaryValueWeights?: Float32Array;
  policyWeights?: Float32Array;
  policyLogits?: Float32Array;
  policy_logits?: ArrayLike<number> & { slice?: (start?: number, end?: number) => ArrayLike<number> };
  scale?: number;
  bias?: number;
  outputActivation?: "linear" | "tanh";
  bytes?: Uint8Array;
}

export interface CompactFrontierModelLayout {
  model: CompactValueModel;
  outputLayerSize: number;
  hiddenLayerWeights: Float32Array[];
  policyWeights: Float32Array | null;
}

export interface EncodedCompactModel extends Uint8Array {
  trainingLoss?: number;
  initialValidationLoss?: number;
  validationLoss?: number;
  bestValidationLoss?: number;
  initialPolicyValidationLoss?: number;
  policyValidationLoss?: number;
  bestPolicyValidationLoss?: number;
  auxiliaryValidationLoss?: number;
  auxiliaryHeadCount?: number;
  valueCheckpointImproved?: boolean;
  policyCheckpointImproved?: boolean;
  modelChanged?: boolean;
  earlyStopReason?: string;
  labelCounts?: Record<string, number>;
  nonZeroWeights?: number;
  replayBufferSize?: number;
  trainingSampleCount?: number;
  policyTrainingSampleCount?: number;
  metrics?: unknown;
  hiddenLayersTrained?: boolean;
}

export interface EncodableCompactModel {
  projectionSize: number;
  projectionSeed: number;
  hiddenLayers: number[];
  hiddenWeights: ArrayLike<number>;
  outputWeights: ArrayLike<number>;
  auxiliaryValueWeights?: ArrayLike<number>;
  policyWeights?: ArrayLike<number>;
  policyLogits?: ArrayLike<number>;
  scale?: number;
  bias?: number;
  outputActivation?: "linear" | "tanh";
}

const finiteModelCache = new WeakMap<CompactValueModel, boolean>();

export function compactModelIsFinite(model: CompactValueModel): boolean {
  const cached = finiteModelCache.get(model);
  if (cached !== undefined) {
    return cached;
  }
  const finite = finiteArray(model.hiddenWeights)
    && finiteArray(model.outputWeights)
    && finiteArray(model.auxiliaryValueWeights ?? [])
    && finiteArray(model.policyWeights ?? [])
    && finiteArray(model.policyLogits ?? [])
    && Number.isFinite(model.scale ?? 1)
    && Number.isFinite(model.bias ?? 0);
  finiteModelCache.set(model, finite);
  return finite;
}

export function compactModelBytesAreFiniteWithEngine(engine: ChronofishEngine, bytes: ArrayBuffer | Uint8Array): boolean {
  return engineGpuModel.compactModelBytesAreFinite(engine, bytes);
}

export function encodeCompactModelWithEngine(engine: ChronofishEngine, model: EncodableCompactModel): EncodedCompactModel | null {
  const payload = {
    projectionSize: model.projectionSize,
    projectionSeed: model.projectionSeed,
    hiddenLayers: Array.from(model.hiddenLayers),
    hiddenWeights: Array.from(model.hiddenWeights),
    outputWeights: Array.from(model.outputWeights),
    auxiliaryValueWeights: Array.from(model.auxiliaryValueWeights ?? []),
    policyWeights: Array.from(model.policyWeights ?? []),
    policyLogits: Array.from(model.policyLogits ?? []),
    scale: model.scale,
    bias: model.bias,
    outputActivation: model.outputActivation
  };
  return engineGpuModel.encodeCompactModel(engine, payload) as EncodedCompactModel | null;
}

export function decodeCompactModelWithEngine(engine: ChronofishEngine, buffer: ArrayBuffer | Uint8Array): CompactValueModel | null {
  const value = engineGpuModel.decodeCompactModel<CompactValueModelJson>(engine, buffer);
  return value ? compactModelFromJson(value) : null;
}

export function decodeCompactFrontierModelLayoutWithEngine(engine: ChronofishEngine, buffer: ArrayBuffer | Uint8Array): CompactFrontierModelLayout | null {
  const value = engineGpuModel.decodeCompactFrontierModelLayout<CompactFrontierModelLayoutJson>(engine, buffer);
  if (!value?.architectureMatches || !value.model) {
    return null;
  }
  return {
    model: compactModelFromJson(value.model),
    outputLayerSize: value.outputLayerSize ?? 0,
    hiddenLayerWeights: (value.hiddenLayerWeights ?? []).map((weights) => new Float32Array(weights)),
    policyWeights: value.policyWeights ? new Float32Array(value.policyWeights) : null
  };
}

function finiteArray(values: ArrayLike<number>): boolean {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      return false;
    }
  }
  return true;
}

export function encodeCompactModel(model: EncodableCompactModel, engine?: ChronofishEngine): EncodedCompactModel {
  if (engine && engineGpuModel.supportsCompactModelEncoding(engine)) {
    const encoded = encodeCompactModelWithEngine(engine, model);
    if (encoded) {
      return encoded;
    }
  }
  const hiddenWeights = Array.from(model.hiddenWeights);
  const outputWeights = Array.from(model.outputWeights);
  const auxiliaryValueWeights = Array.from(model.auxiliaryValueWeights ?? []);
  const policyWeights = Array.from(model.policyWeights ?? []);
  const policyLogits = Array.from(model.policyLogits ?? []);
  const scale = model.scale ?? 1;
  const bias = model.bias ?? 0;
  const version = auxiliaryValueWeights.length
    ? 5
    : model.outputActivation === "tanh"
    ? 4
    : policyWeights.length
      ? 3
      : policyLogits.length
        ? 2
        : 1;
  const policyValues = version >= 3 ? policyWeights : policyLogits;
  const floats = [scale, bias, ...hiddenWeights, ...outputWeights, ...policyValues, ...auxiliaryValueWeights];
  const byteLength = 4
    + 4 * 6
    + (version >= 2 ? 4 : 0)
    + (version >= 5 ? 4 : 0)
    + 4 * model.hiddenLayers.length
    + 4 * floats.length;
  const buffer = new ArrayBuffer(byteLength);
  const view = new DataView(buffer);
  let cursor = 0;
  writeAscii(view, cursor, "CFNN");
  cursor += 4;
  cursor = writeU32(view, cursor, version);
  cursor = writeU32(view, cursor, model.projectionSize);
  cursor = writeU32(view, cursor, model.projectionSeed);
  cursor = writeU32(view, cursor, model.hiddenLayers.length);
  cursor = writeU32(view, cursor, outputWeights.length);
  if (version >= 2) {
    cursor = writeU32(view, cursor, policyValues.length);
  }
  if (version >= 5) {
    cursor = writeU32(view, cursor, auxiliaryValueWeights.length);
  }
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
  for (const value of policyValues) {
    cursor = writeF32(view, cursor, value);
  }
  for (const value of auxiliaryValueWeights) {
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
  if (version < 1 || version > 5) {
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
  const policySize = version >= 2 ? view.getUint32(cursor, true) : 0;
  cursor += version >= 2 ? 4 : 0;
  const auxiliaryValueSize = version >= 5 ? view.getUint32(cursor, true) : 0;
  cursor += version >= 5 ? 4 : 0;
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
  const policyValues = new Float32Array(policySize);
  for (let index = 0; index < policySize; index += 1) {
    policyValues[index] = view.getFloat32(cursor, true);
    cursor += 4;
  }
  const auxiliaryValueWeights = new Float32Array(auxiliaryValueSize);
  for (let index = 0; index < auxiliaryValueSize; index += 1) {
    auxiliaryValueWeights[index] = view.getFloat32(cursor, true);
    cursor += 4;
  }
  const model: CompactValueModel = {
    projectionSize,
    projectionSeed,
    hiddenLayers,
    hiddenWeights,
    outputWeights,
    auxiliaryValueWeights,
    policyLogits: version === 2 ? policyValues : new Float32Array(),
    policyWeights: version >= 3 ? policyValues : new Float32Array(),
    scale,
    bias,
    outputActivation: version >= 4 ? "tanh" : "linear"
  };
  return compactModelIsFinite(model) ? model : null;
}

interface CompactValueModelJson {
  projectionSize: number;
  projectionSeed: number;
  hiddenLayers: number[];
  hiddenWeights: number[];
  outputWeights: number[];
  auxiliaryValueWeights?: number[];
  policyWeights?: number[];
  policyLogits?: number[];
  scale?: number;
  bias?: number;
  outputActivation?: "linear" | "tanh";
}

interface CompactFrontierModelLayoutJson {
  architectureMatches: boolean;
  model?: CompactValueModelJson;
  outputLayerSize?: number;
  hiddenLayerWeights?: number[][];
  policyWeights?: number[] | null;
}

function compactModelFromJson(value: CompactValueModelJson): CompactValueModel {
  return {
    projectionSize: value.projectionSize,
    projectionSeed: value.projectionSeed,
    hiddenLayers: value.hiddenLayers,
    hiddenWeights: new Float32Array(value.hiddenWeights),
    outputWeights: new Float32Array(value.outputWeights),
    auxiliaryValueWeights: new Float32Array(value.auxiliaryValueWeights ?? []),
    policyWeights: new Float32Array(value.policyWeights ?? []),
    policyLogits: new Float32Array(value.policyLogits ?? []),
    scale: value.scale,
    bias: value.bias,
    outputActivation: value.outputActivation
  };
}
