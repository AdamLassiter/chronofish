import type { TrainingLabelKind, TrainingSample } from "./training-gpu.js";

const MIN_POLICY_REPLAY_FRACTION = 0.25;

export function appendReplaySamples(buffer: TrainingSample[], samples: TrainingSample[], maxBuffer: number): TrainingSample[] {
  const values = dedupeTrainingSamples(buffer.concat(samples));
  if (values.length <= maxBuffer) {
    return values;
  }
  const ranked = values
    .map((sample, index) => ({
      sample,
      index,
      priority: replaySamplePriority(sample, index, values.length)
    }))
    .sort((left, right) => right.priority - left.priority || right.index - left.index);
  const selected = ranked.slice(0, maxBuffer);
  const selectedIndices = new Set(selected.map((entry) => entry.index));
  const availablePolicyCount = ranked.reduce(
    (count, entry) => count + Number(replayHasPolicyTarget(entry.sample)),
    0
  );
  const requiredPolicyCount = Math.min(
    availablePolicyCount,
    Math.max(1, Math.ceil(maxBuffer * MIN_POLICY_REPLAY_FRACTION))
  );
  let selectedPolicyCount = selected.reduce(
    (count, entry) => count + Number(replayHasPolicyTarget(entry.sample)),
    0
  );
  for (const replacement of ranked) {
    if (selectedPolicyCount >= requiredPolicyCount) {
      break;
    }
    if (!replayHasPolicyTarget(replacement.sample) || selectedIndices.has(replacement.index)) {
      continue;
    }
    let replaceIndex = -1;
    for (let index = selected.length - 1; index >= 0; index -= 1) {
      if (!replayHasPolicyTarget(selected[index]!.sample)) {
        replaceIndex = index;
        break;
      }
    }
    if (replaceIndex < 0) {
      break;
    }
    selectedIndices.delete(selected[replaceIndex]!.index);
    selected[replaceIndex] = replacement;
    selectedIndices.add(replacement.index);
    selectedPolicyCount += 1;
  }
  return selected
    .sort((left, right) => left.index - right.index)
    .map((entry) => entry.sample);
}

function replayHasPolicyTarget(sample: TrainingSample): boolean {
  return sample.labelKind !== "distilled"
    && Number.isInteger(sample.policy)
    && (sample.policy ?? -1) >= 0;
}

export function dedupeTrainingSamples(samples: TrainingSample[]): TrainingSample[] {
  const merged = samples.filter((sample) =>
    Array.isArray(sample.features) || sample.features instanceof Float32Array
  );
  const deduplicated = new Map<string, { sample: TrainingSample; index: number }>();
  let legacyIndex = 0;
  for (let index = 0; index < merged.length; index += 1) {
    const sample = merged[index]!;
    const key = replaySampleKey(sample, legacyIndex);
    if (!sample.positionKey) {
      legacyIndex += 1;
    }
    const existing = deduplicated.get(key);
    if (existing) {
      const combined = mergeCompatibleSamples(existing.sample, sample);
      deduplicated.delete(key);
      deduplicated.set(key, { sample: combined, index });
      continue;
    }
    deduplicated.set(key, { sample, index });
  }
  return Array.from(deduplicated.values()).map((entry) => entry.sample);
}

export function mergeCompatibleSamples(
  existing: TrainingSample,
  incoming: TrainingSample
): TrainingSample {
  const existingWeight = Math.max(0, existing.baseLabelWeight ?? existing.labelWeight ?? 1);
  const incomingWeight = Math.max(0, incoming.baseLabelWeight ?? incoming.labelWeight ?? 1);
  const existingMass = Math.max(0, existing.labelMass ?? existingWeight);
  const incomingMass = Math.max(0, incoming.labelMass ?? incomingWeight);
  const totalMass = existingMass + incomingMass;
  const existingCount = Math.max(1, existing.observationCount ?? 1);
  const incomingCount = Math.max(1, incoming.observationCount ?? 1);
  const observationCount = existingCount + incomingCount;
  const strongestWeight = Math.max(existingWeight, incomingWeight);
  const confidence = Math.min(2, Math.sqrt(observationCount));
  const preferred = incomingWeight >= existingWeight ? incoming : existing;
  return {
    ...preferred,
    features: incoming.features,
    label: totalMass > 0
      ? (existing.label * existingMass + incoming.label * incomingMass) / totalMass
      : incoming.label,
    labelWeight: strongestWeight * confidence,
    baseLabelWeight: strongestWeight,
    labelMass: Math.min(totalMass, 64),
    observationCount: Math.min(observationCount, 64),
    policy: preferred.policy ?? existing.policy ?? incoming.policy ?? null,
    pseudo: Boolean(existing.pseudo && incoming.pseudo)
  };
}

function replaySampleKey(sample: TrainingSample, legacyIndex: number): string {
  const labelKind = sample.labelKind ?? "unknown";
  if (sample.positionKey) {
    return `${sample.positionKey}|${labelKind}`;
  }
  const fingerprint = featureFingerprint(sample.features);
  if (fingerprint) {
    return `${fingerprint}|${sample.sideToMove ?? ""}|${sample.boardCount ?? 0}|${labelKind}`;
  }
  return `legacy:${legacyIndex}`;
}

function featureFingerprint(features: number[] | Float32Array): string | null {
  if (!features.length) {
    return null;
  }
  let hash = 2166136261;
  let nonZero = 0;
  for (let index = 0; index < features.length; index += 1) {
    const value = features[index] ?? 0;
    if (value === 0) {
      continue;
    }
    nonZero += 1;
    hash ^= index;
    hash = Math.imul(hash, 16777619) >>> 0;
    hash ^= Math.round(value * 1024);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return nonZero > 0 ? `features:${features.length}:${hash.toString(16)}` : null;
}

export function replaySamplePriority(sample: TrainingSample, index: number, total: number): number {
  const recency = total > 1 ? index / (total - 1) : 1;
  return trainingLabelPriority(sample.labelKind, sample.pseudo) +
    Math.max(0, sample.labelWeight ?? 1) +
    recency * 0.25;
}

export function trainingLabelPriority(labelKind: TrainingLabelKind | undefined, pseudo: boolean | undefined): number {
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
