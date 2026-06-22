export type CpuParameters = Record<string, number>;

export function cpuParametersKey(parameters: CpuParameters): string {
  return Object.keys(parameters)
    .sort()
    .map((key) => `${key}:${parameters[key]}`)
    .join("|");
}

export function uniqueCpuParameters(values: CpuParameters[]): CpuParameters[] {
  const seen = new Set<string>();
  const unique: CpuParameters[] = [];
  for (const parameters of values) {
    const key = cpuParametersKey(parameters);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    unique.push(parameters);
  }
  return unique;
}

export function cpuReferenceWorkerCount(
  gameCount: number,
  requestedWorkers: number,
  pairBatch: number
): number {
  return Math.min(
    Math.max(0, gameCount),
    Math.max(1, Math.floor(requestedWorkers) || 1),
    Math.max(1, Math.floor(pairBatch) || 1)
  );
}

export function breedCpuPopulation(
  baseline: CpuParameters,
  elites: CpuParameters[],
  target: number,
  seed: number,
  generation: number,
  stagnation: number
): CpuParameters[] {
  const parents = uniqueCpuParameters([baseline, ...elites]);
  const population = [...parents.slice(0, Math.max(1, Math.min(target, parents.length)))];
  const mutationScale = Math.min(3, 1 + stagnation * 0.4);
  const maxAttempts = Math.max(64, target * 64);
  for (let attempt = 0; population.length < target && attempt < maxAttempts; attempt += 1) {
    const left = parents[(attempt + generation) % parents.length] ?? baseline;
    const right = parents[(attempt * 5 + generation + 1) % parents.length] ?? baseline;
    const childSeed = (seed ^ Math.imul(generation + 1, 0x9e3779b1) ^ Math.imul(attempt + 1, 0x85ebca6b)) >>> 0;
    const child = mutateCpuParameters(
      crossoverCpuParameters(left, right, childSeed ^ 0xc2b2ae35),
      childSeed,
      mutationScale
    );
    if (!population.some((candidate) => cpuParametersKey(candidate) === cpuParametersKey(child))) {
      population.push(child);
    }
  }
  return population;
}

export function mutateCpuParameters(base: CpuParameters, seed: number, scale = 1): CpuParameters {
  let state = seed >>> 0 || 1;
  const nextRandom = (): number => {
    state = Math.imul(state, 1664525) + 1013904223 >>> 0;
    return state / 0xffffffff;
  };
  const next = { ...base };
  const mutable = Object.entries(base)
    .filter(([key, value]) => Number.isFinite(value) && key !== "king" && key !== "royal_queen");
  if (!mutable.length) {
    return next;
  }
  for (let index = mutable.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(nextRandom() * (index + 1));
    [mutable[index], mutable[swapIndex]] = [mutable[swapIndex]!, mutable[index]!];
  }
  const broadMutation = nextRandom() < 0.125;
  const sparseTarget = Math.max(1, Math.min(
    mutable.length,
    Math.round((1 + nextRandom() * 3) * Math.sqrt(Math.max(0.25, scale)))
  ));
  const mutationTarget = broadMutation
    ? Math.max(sparseTarget, Math.ceil(mutable.length * Math.min(0.8, 0.2 * Math.max(1, scale))))
    : sparseTarget;
  let changed = 0;
  for (const [key, value] of mutable) {
    if (changed >= mutationTarget) {
      break;
    }
    const spread = Math.max(1, Math.round(Math.abs(value) * 0.08 * Math.max(0.25, scale)));
    let delta = Math.round((nextRandom() * 2 - 1) * spread);
    if (delta === 0) {
      delta = nextRandom() < 0.5 ? -1 : 1;
    }
    const mutated = Math.max(-10_000, Math.min(10_000, Math.round(value + delta)));
    if (mutated !== value) {
      next[key] = mutated;
      changed += 1;
    }
  }
  if (changed === 0) {
    const [key, value] = mutable[0]!;
    next[key] = value >= 10_000 ? value - 1 : value + 1;
  }
  return next;
}

export function crossoverCpuParameters(left: CpuParameters, right: CpuParameters, seed: number): CpuParameters {
  let state = seed >>> 0 || 1;
  const nextRandom = (): number => {
    state = Math.imul(state, 1103515245) + 12345 >>> 0;
    return state / 0xffffffff;
  };
  const child: CpuParameters = {};
  const keys = new Set([...Object.keys(left), ...Object.keys(right)]);
  for (const key of keys) {
    const leftValue = left[key];
    const rightValue = right[key];
    if (typeof leftValue !== "number" || !Number.isFinite(leftValue)) {
      child[key] = typeof rightValue === "number" && Number.isFinite(rightValue) ? rightValue : 0;
    } else if (typeof rightValue !== "number" || !Number.isFinite(rightValue)) {
      child[key] = leftValue;
    } else if (key === "king" || key === "royal_queen") {
      child[key] = leftValue;
    } else {
      const blend = nextRandom();
      child[key] = Math.round(leftValue * blend + rightValue * (1 - blend));
    }
  }
  return child;
}
