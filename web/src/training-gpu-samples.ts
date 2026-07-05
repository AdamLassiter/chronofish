import { readWasmBytes, readWasmString, writeWasmBytes, writeWasmString } from "./engine-io.js";
import type { ChronofishEngine } from "./types.js";
import type { TrainingSample } from "./training-gpu-types.js";

export interface ValidationSplit {
  trainIndices: number[];
  validationIndices: number[];
  seed: number;
}

export function splitValidationSamples(samples: TrainingSample[], validationSplit = 0, engine?: ChronofishEngine): ValidationSplit {
  if (engine) {
    return splitValidationSamplesWithEngine(samples, validationSplit, engine);
  }
  return splitValidationSamplesFallback(samples, validationSplit);
}

function splitValidationSamplesWithEngine(samples: TrainingSample[], validationSplit: number, engine: ChronofishEngine): ValidationSplit {
  const input = writeWasmString(engine, JSON.stringify(samples));
  try {
    const output = engine.chronofish_split_validation_samples_json(input.ptr, input.len, validationSplit);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as ValidationSplit;
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

function splitValidationSamplesFallback(samples: TrainingSample[], validationSplit = 0): ValidationSplit {
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

export function movePositionGroupToValidation(
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

export function moveOrCollapseValidationGroup(
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

export function trainingLabelPriority(labelKind: string | undefined, pseudo: boolean | undefined, engine?: ChronofishEngine): number {
  if (engine) {
    const input = writeWasmString(engine, labelKind ?? "");
    try {
      return engine.chronofish_training_label_priority(input.ptr, input.len, pseudo ? 1 : 0);
    } finally {
      engine.chronofish_dealloc(input.ptr, input.len);
    }
  }
  if (labelKind === "outcome" || labelKind === "duel") {
    return 4;
  }
  if (labelKind === "search" || labelKind === "cpu") {
    return 3;
  }
  if (labelKind === "duel-search" || labelKind === "search-bootstrap") {
    return 2;
  }
  if (labelKind === "distilled" || pseudo) {
    return 1;
  }
  return 2;
}

export function stableSampleHash(sample: TrainingSample, index: number, engine?: ChronofishEngine): number {
  if (engine) {
    return stableSampleHashWithEngine(sample, index, engine);
  }
  return stableSampleHashFallback(sample);
}

function stableSampleHashWithEngine(sample: TrainingSample, index: number, engine: ChronofishEngine): number {
  const input = writeWasmString(engine, JSON.stringify(sample));
  try {
    return engine.chronofish_stable_sample_hash_json(input.ptr, input.len, index) >>> 0;
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

function stableSampleHashFallback(sample: TrainingSample): number {
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

export function shuffledIndices(indices: number[], epoch: number, seed: number, engine?: ChronofishEngine): number[] {
  if (engine) {
    return shuffledIndicesWithEngine(indices, epoch, seed, engine);
  }
  return shuffledIndicesFallback(indices, epoch, seed);
}

function shuffledIndicesWithEngine(indices: number[], epoch: number, seed: number, engine: ChronofishEngine): number[] {
  const input = writeWasmBytes(engine, u32ArrayBytes(indices));
  try {
    const output = engine.chronofish_shuffled_training_indices_bytes(input.ptr, input.len, epoch, seed);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return u32ArrayFromBytes(readWasmBytes(engine, output));
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

function shuffledIndicesFallback(indices: number[], epoch: number, seed: number): number[] {
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

function u32ArrayBytes(values: number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setUint32(index * Uint32Array.BYTES_PER_ELEMENT, values[index] ?? 0, true);
  }
  return bytes;
}

function u32ArrayFromBytes(bytes: Uint8Array): number[] {
  if (bytes.byteLength % Uint32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Shuffled training index response has an unexpected length.");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const result: number[] = [];
  for (let offset = 0; offset < bytes.byteLength; offset += Uint32Array.BYTES_PER_ELEMENT) {
    result.push(view.getUint32(offset, true));
  }
  return result;
}

export function groupTrainingIndicesByPosition(samples: TrainingSample[], indices: number[], engine?: ChronofishEngine): number[][] {
  if (engine) {
    return groupTrainingIndicesByPositionWithEngine(samples, indices, engine);
  }
  return groupTrainingIndicesByPositionFallback(samples, indices);
}

function groupTrainingIndicesByPositionWithEngine(samples: TrainingSample[], indices: number[], engine: ChronofishEngine): number[][] {
  const input = writeWasmString(engine, JSON.stringify({ samples, indices }));
  try {
    const output = engine.chronofish_group_training_indices_by_position_json(input.ptr, input.len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as number[][];
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

function groupTrainingIndicesByPositionFallback(samples: TrainingSample[], indices: number[]): number[][] {
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

export function uniqueTrainingPositionCount(samples: TrainingSample[], indices: number[], engine?: ChronofishEngine): number {
  if (engine) {
    return uniqueTrainingPositionCountWithEngine(samples, indices, engine);
  }
  return uniqueTrainingPositionCountFallback(samples, indices);
}

function uniqueTrainingPositionCountWithEngine(samples: TrainingSample[], indices: number[], engine: ChronofishEngine): number {
  const input = writeWasmString(engine, JSON.stringify({ samples, indices }));
  try {
    const output = engine.chronofish_unique_training_position_count_json(input.ptr, input.len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const count = Number(readWasmString(engine, output));
    if (!Number.isInteger(count) || count < 0) {
      throw new Error("Unique training position count response is invalid.");
    }
    return count;
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

function uniqueTrainingPositionCountFallback(samples: TrainingSample[], indices: number[]): number {
  return new Set(indices.map((index) => trainingPositionIdentity(samples[index]!))).size;
}

export function fillGroupedTrainingBatchIndices(
  batch: Uint32Array,
  trainGroups: number[][],
  epoch: number,
  seed: number,
  labelWeights: Float32Array,
  engine?: ChronofishEngine
): number {
  if (engine) {
    return fillGroupedTrainingBatchIndicesWithEngine(batch, trainGroups, epoch, seed, labelWeights, engine);
  }
  return fillGroupedTrainingBatchIndicesFallback(batch, trainGroups, epoch, seed, labelWeights);
}

function fillGroupedTrainingBatchIndicesWithEngine(
  batch: Uint32Array,
  trainGroups: number[][],
  epoch: number,
  seed: number,
  labelWeights: Float32Array,
  engine: ChronofishEngine
): number {
  const request = groupedBatchRequestBytes(batch.length, trainGroups, epoch, seed, labelWeights);
  const input = writeWasmBytes(engine, request);
  try {
    const output = engine.chronofish_fill_grouped_training_batch_indices_bytes(input.ptr, input.len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readGroupedBatchResponse(readWasmBytes(engine, output), batch);
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

function groupedBatchRequestBytes(
  batchLength: number,
  trainGroups: number[][],
  epoch: number,
  seed: number,
  labelWeights: Float32Array
): Uint8Array {
  const itemCount = trainGroups.reduce((sum, group) => sum + group.length, 0);
  const byteLength = 24 + (trainGroups.length + 1 + itemCount + labelWeights.length) * Uint32Array.BYTES_PER_ELEMENT;
  const bytes = new Uint8Array(byteLength);
  const view = new DataView(bytes.buffer);
  let cursor = 0;
  const writeU32 = (value: number): void => {
    view.setUint32(cursor, value >>> 0, true);
    cursor += 4;
  };
  const writeF32 = (value: number): void => {
    view.setFloat32(cursor, value, true);
    cursor += 4;
  };
  writeU32(batchLength);
  writeU32(trainGroups.length);
  writeU32(itemCount);
  writeU32(labelWeights.length);
  writeU32(epoch);
  writeU32(seed);
  let offset = 0;
  writeU32(offset);
  for (const group of trainGroups) {
    offset += group.length;
    writeU32(offset);
  }
  for (const group of trainGroups) {
    for (const index of group) {
      writeU32(index);
    }
  }
  for (const weight of labelWeights) {
    writeF32(weight);
  }
  return bytes;
}

function readGroupedBatchResponse(bytes: Uint8Array, batch: Uint32Array): number {
  if (bytes.byteLength !== 8 + batch.length * Uint32Array.BYTES_PER_ELEMENT) {
    throw new Error("Grouped training batch response has an unexpected length.");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const batchWeight = view.getFloat32(0, true);
  const batchLength = view.getUint32(4, true);
  if (batchLength !== batch.length) {
    throw new Error("Grouped training batch response length does not match the request.");
  }
  let cursor = 8;
  for (let index = 0; index < batch.length; index += 1) {
    batch[index] = view.getUint32(cursor, true);
    cursor += 4;
  }
  return batchWeight;
}

function fillGroupedTrainingBatchIndicesFallback(
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

export function featureLength(samples: TrainingSample[], engine?: ChronofishEngine): number {
  if (engine) {
    return featureLengthWithEngine(samples, engine);
  }
  return featureLengthFallback(samples);
}

function featureLengthWithEngine(samples: TrainingSample[], engine: ChronofishEngine): number {
  const input = writeWasmString(engine, JSON.stringify(samples));
  try {
    const output = engine.chronofish_feature_length_json(input.ptr, input.len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const length = Number(readWasmString(engine, output));
    if (!Number.isInteger(length) || length <= 0) {
      throw new Error("Training feature length response is invalid.");
    }
    return length;
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

function featureLengthFallback(samples: TrainingSample[]): number {
  const length = samples[0]?.features?.length;
  if (!length || !samples.every((sample) => sample.features.length === length)) {
    throw new Error("Training samples have inconsistent feature lengths.");
  }
  return length;
}
