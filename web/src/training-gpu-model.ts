export interface CompactValueModel {
  projectionSize: number;
  projectionSeed: number;
  hiddenLayers: number[];
  hiddenWeights: Float32Array;
  outputWeights: Float32Array;
  policyWeights?: Float32Array;
  policyLogits?: Float32Array;
  policy_logits?: ArrayLike<number> & { slice?: (start?: number, end?: number) => ArrayLike<number> };
  scale?: number;
  bias?: number;
  outputActivation?: "linear" | "tanh";
  bytes?: Uint8Array;
}

export interface EncodedCompactModel extends Uint8Array {
  trainingLoss?: number;
  initialValidationLoss?: number;
  validationLoss?: number;
  bestValidationLoss?: number;
  initialPolicyValidationLoss?: number;
  policyValidationLoss?: number;
  bestPolicyValidationLoss?: number;
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
    && finiteArray(model.policyWeights ?? [])
    && finiteArray(model.policyLogits ?? [])
    && Number.isFinite(model.scale ?? 1)
    && Number.isFinite(model.bias ?? 0);
  finiteModelCache.set(model, finite);
  return finite;
}

function finiteArray(values: ArrayLike<number>): boolean {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      return false;
    }
  }
  return true;
}

export function encodeCompactModel(model: EncodableCompactModel): EncodedCompactModel {
  const hiddenWeights = Array.from(model.hiddenWeights);
  const outputWeights = Array.from(model.outputWeights);
  const policyWeights = Array.from(model.policyWeights ?? []);
  const policyLogits = Array.from(model.policyLogits ?? []);
  const scale = model.scale ?? 1;
  const bias = model.bias ?? 0;
  const version = model.outputActivation === "tanh"
    ? 4
    : policyWeights.length
      ? 3
      : policyLogits.length
        ? 2
        : 1;
  const policyValues = version >= 3 ? policyWeights : policyLogits;
  const floats = [scale, bias, ...hiddenWeights, ...outputWeights, ...policyValues];
  const byteLength = 4
    + 4 * 6
    + (version >= 2 ? 4 : 0)
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
  if (version < 1 || version > 4) {
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
  const model: CompactValueModel = {
    projectionSize,
    projectionSeed,
    hiddenLayers,
    hiddenWeights,
    outputWeights,
    policyLogits: version === 2 ? policyValues : new Float32Array(),
    policyWeights: version >= 3 ? policyValues : new Float32Array(),
    scale,
    bias,
    outputActivation: version === 4 ? "tanh" : "linear"
  };
  return compactModelIsFinite(model) ? model : null;
}
