import type { TrainingSample } from "./training-gpu-types.js";

export interface ValidationSplit {
  trainIndices: number[];
  validationIndices: number[];
  seed: number;
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

export function trainingLabelPriority(labelKind: string | undefined, pseudo: boolean | undefined): number {
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
