import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import * as esbuild from "esbuild";
import { glsl } from "esbuild-plugin-glsl";

const root = path.resolve(import.meta.dirname, "..");
const modules = await buildTrainingModules();
const {
  decodeCompactModel,
  denormalizedSearchScore,
  encodeCompactModel,
  fillGroupedTrainingBatchIndices,
  groupTrainingIndicesByPosition,
  inverseTanh,
  lossReductionWorkgroupCount,
  normalizedSearchScore,
  optimizerVelocity,
  packSparseProjectionFeatures,
  policyTrainingSteps,
  predictValuesOnCpu,
  projectionHash,
  selectTrainingWorkingSet,
  splitPolicyTrainingIndices,
  splitValidationSamples,
  trainHeadsOnCpu,
  uniqueTrainingPositionCount
} = await import(modules.trainingGpu);
const { appendReplaySamples, dedupeTrainingSamples } = await import(modules.trainingReplay);
const { policyBucket } = await import(modules.trainingPolicy);
const {
  encodeNeuralPositionFeatures,
  NEURAL_BOARD_PLANES,
  NEURAL_BOARD_SQUARES,
  NEURAL_INPUT_SIZE,
  neuralBoardSelection
} = await import(modules.trainingEncoding);
const {
  breedCpuPopulation,
  cpuParametersKey,
  cpuReferenceWorkerCount,
  crossoverCpuParameters,
  mutateCpuParameters,
  uniqueCpuParameters
} = await import(modules.trainingCpu);

test("browser CPU candidate identity is stable and removes duplicates", () => {
  const first = { mobility: 2, queen: 10 };
  const reordered = { queen: 10, mobility: 2 };
  const different = { queen: 11, mobility: 2 };

  assert.equal(cpuParametersKey(first), cpuParametersKey(reordered));
  assert.deepEqual(uniqueCpuParameters([first, reordered, different]), [first, different]);
});

test("browser CPU reference workers respect games, workers, and pair batch", () => {
  assert.equal(cpuReferenceWorkerCount(0, 8, 8), 0);
  assert.equal(cpuReferenceWorkerCount(12, 8, 3), 3);
  assert.equal(cpuReferenceWorkerCount(2, 8, 3), 2);
  assert.equal(cpuReferenceWorkerCount(12, 0, 0), 1);
});

test("browser CPU evolution breeds deterministic unique populations around elites", () => {
  const baseline = { mobility: 100, queen: 900, king: 10000 };
  const elite = { mobility: 108, queen: 880, king: 10000 };
  const first = breedCpuPopulation(baseline, [elite], 8, 1234, 2, 1);
  const second = breedCpuPopulation(baseline, [elite], 8, 1234, 2, 1);

  assert.deepEqual(first, second);
  assert.equal(first.length, 8);
  assert.equal(new Set(first.map(cpuParametersKey)).size, 8);
  assert.deepEqual(first[0], baseline);
  assert.deepEqual(first[1], elite);
  assert.ok(first.every((parameters) => parameters.king === 10000));
});

test("browser CPU crossover preserves protected royal values", () => {
  const child = crossoverCpuParameters(
    { mobility: 100, king: 10000, royal_queen: 9000 },
    { mobility: 200, king: -1, royal_queen: -1 },
    99
  );

  assert.ok(child.mobility >= 100 && child.mobility <= 200);
  assert.equal(child.king, 10000);
  assert.equal(child.royal_queen, 9000);
});

test("browser CPU mutations are usually sparse but always change a tunable coordinate", () => {
  const baseline = Object.fromEntries([
    ["king", 10000],
    ["royal_queen", 9000],
    ...Array.from({ length: 32 }, (_, index) => [`weight${index}`, 100 + index])
  ]);
  const changedCounts = Array.from({ length: 64 }, (_, seed) => {
    const candidate = mutateCpuParameters(baseline, seed + 1, 1);
    assert.equal(candidate.king, baseline.king);
    assert.equal(candidate.royal_queen, baseline.royal_queen);
    return Object.keys(baseline).filter((key) => candidate[key] !== baseline[key]).length;
  });

  assert.ok(changedCounts.every((count) => count >= 1));
  assert.ok(changedCounts.filter((count) => count <= 4).length >= 48);
});

test("policy head training scales sublinearly with value optimization steps", () => {
  assert.equal(policyTrainingSteps(1), 16);
  assert.equal(policyTrainingSteps(1024), 16);
  assert.equal(policyTrainingSteps(4096), 64);
  assert.equal(policyTrainingSteps(16384), 256);
  assert.equal(policyTrainingSteps(65536), 256);
});

test("training momentum accumulates consistent gradients and reacts to reversals", () => {
  const first = optimizerVelocity(0, 2);
  const second = optimizerVelocity(first, 2);
  const reversed = optimizerVelocity(second, -2);

  assert.ok(Math.abs(first - 0.2) < 1e-12);
  assert.ok(Math.abs(second - 0.38) < 1e-12);
  assert.ok(reversed > 0);
  assert.ok(reversed < second);
});

test("value targets and search scores use an invertible bounded scale", () => {
  assert.equal(normalizedSearchScore(20_000), 1);
  assert.equal(normalizedSearchScore(-10_000), -0.5);
  assert.equal(normalizedSearchScore(100_000), 1);
  assert.equal(denormalizedSearchScore(0.75), 15_000);
  assert.equal(denormalizedSearchScore(-2), -20_000);
  assert.ok(Math.abs(Math.tanh(inverseTanh(0.75)) - 0.75) < 1e-12);
});

test("policy holdout reuses the position split and falls back for sparse labels", () => {
  const samples = ["a", "b", "c", "d", "e"].map((key) => sample(key));
  const split = {
    trainIndices: [0, 1, 2, 3],
    validationIndices: [4],
    seed: 123
  };
  assert.deepEqual(
    splitPolicyTrainingIndices(samples, [1, 3, 4], split, 0.2),
    { trainIndices: [1, 3], validationIndices: [4], seed: 123 }
  );
  const fallback = splitPolicyTrainingIndices(samples, [1, 3], split, 0.2);
  assert.equal(fallback.trainIndices.length, 1);
  assert.equal(fallback.validationIndices.length, 1);
  assert.notEqual(fallback.trainIndices[0], fallback.validationIndices[0]);
});

test("minibatch selection is uniform by position while gradient normalization remains weighted", () => {
  const firstBatch = new Uint32Array(32);
  const secondBatch = new Uint32Array(32);
  const samples = [
    sample("same", { labelKind: "search" }),
    sample("same", { labelKind: "outcome" }),
    sample("other")
  ];
  const trainGroups = groupTrainingIndicesByPosition(samples, [0, 1, 2]);
  const uniformWeights = new Float32Array([1, 1, 1]);
  const skewedWeights = new Float32Array([1, 10, 1]);

  const uniformBatchWeight = fillGroupedTrainingBatchIndices(firstBatch, trainGroups, 1, 1234, uniformWeights);
  const skewedBatchWeight = fillGroupedTrainingBatchIndices(secondBatch, trainGroups, 1, 1234, skewedWeights);

  assert.deepEqual(trainGroups, [[0, 1], [2]]);
  assert.deepEqual(secondBatch, firstBatch);
  assert.equal(uniformBatchWeight, firstBatch.length);
  assert.ok(skewedBatchWeight > uniformBatchWeight);
  assert.ok(new Set(firstBatch).size > 1);
});

test("hidden-layer training counts distinct positions rather than duplicate labels", () => {
  const samples = [
    sample("same", { labelKind: "search" }),
    sample("same", { labelKind: "outcome" }),
    sample("other")
  ];

  assert.equal(uniqueTrainingPositionCount(samples, [0, 1, 2]), 2);
  assert.equal(uniqueTrainingPositionCount(samples, [0, 1]), 1);
});

test("fallback validation moves every label for the selected position", () => {
  const samples = [
    sample("same", { labelKind: "search" }),
    sample("same", { labelKind: "outcome" }),
    sample("other", { labelKind: "search" })
  ];
  const split = splitValidationSamples(samples, 0.000001);
  const trainKeys = new Set(split.trainIndices.map((index) => samples[index].positionKey));
  const validationKeys = new Set(split.validationIndices.map((index) => samples[index].positionKey));

  assert.equal([...trainKeys].some((key) => validationKeys.has(key)), false);
  assert.ok(split.validationIndices.length >= 1);
});

test("GPU loss reduction emits one partial per 64 validation samples", () => {
  assert.equal(lossReductionWorkgroupCount(1), 1);
  assert.equal(lossReductionWorkgroupCount(64), 1);
  assert.equal(lossReductionWorkgroupCount(65), 2);
  assert.equal(lossReductionWorkgroupCount(4096), 64);
});

test("feature projection packs dense samples into stable sparse rows", () => {
  const packed = packSparseProjectionFeatures([
    sample("first", { features: [0, 2, 0, -1] }),
    sample("empty", { features: [0, 0, 0, 0] }),
    sample("last", { features: [3, 0, 4, 0] })
  ]);

  assert.deepEqual(Array.from(packed.offsets), [0, 2, 2, 4]);
  assert.deepEqual(Array.from(packed.indices.slice(0, 4)), [1, 3, 0, 2]);
  assert.deepEqual(Array.from(packed.values.slice(0, 4)), [2, -1, 3, 4]);
  assert.equal(packed.byteLength, 16 + 16 + 16);
});

test("feature projection allocates valid empty storage buffers", () => {
  const packed = packSparseProjectionFeatures([
    sample("empty", { features: [0, 0, 0] })
  ]);

  assert.deepEqual(Array.from(packed.offsets), [0, 0]);
  assert.equal(packed.indices.length, 1);
  assert.equal(packed.values.length, 1);
});

test("small-batch CPU prediction preserves compact-model projection and layer layout", () => {
  const seed = 17;
  const projected = [0, 1].map((output) =>
    (projectionHash(0, output, seed) & 1) === 0 ? 1 : -1
  );
  const hidden = Math.max(0, projected[0] + projected[1] * 2 + 0.5);
  const predictions = predictValuesOnCpu([
    sample("cpu-predict", { features: [1, 0] })
  ], {
    projectionSize: 2,
    projectionSeed: seed,
    hiddenLayers: [1],
    hiddenWeights: new Float32Array([1, 2, 0.5]),
    outputWeights: new Float32Array([3, 1]),
    scale: 0.1,
    bias: 0.2,
    outputActivation: "tanh"
  });

  assert.equal(predictions.length, 1);
  assert.ok(Math.abs(predictions[0] - Math.max(-1, Math.min(1, Math.tanh(hidden * 3 + 1) * 0.1 + 0.2))) < 1e-6);
});

test("CPU position encoding preserves neural plane layout and perspective", () => {
  const game = {
    turn: "white",
    nextTimelineId: 2,
    nextBlackTimelineId: -2,
    checkedRoyals: [],
    timelines: [
      {
        id: -1,
        row: 0,
        label: "-1",
        owner: "black",
        boards: [board(4, "black", pieceAt(0, 0, "white", "king"))]
      },
      {
        id: 1,
        row: 1,
        label: "+1",
        owner: "white",
        boards: [board(6, "white", pieceAt(7, 7, "black", "pawn"))]
      },
      {
        id: 3,
        row: 2,
        label: "+3",
        owner: "white",
        boards: [board(8, "black", pieceAt(3, 3, "black", "queen"))]
      }
    ]
  };

  const encoded = encodeNeuralPositionFeatures(game, "white");
  const boardStride = NEURAL_BOARD_PLANES * NEURAL_BOARD_SQUARES;
  const plane = (boardIndex, planeIndex, square = 0) =>
    boardIndex * boardStride + planeIndex * NEURAL_BOARD_SQUARES + square;

  assert.equal(encoded.boardCount, 3);
  assert.equal(encoded.values.length, NEURAL_INPUT_SIZE);
  assert.equal(encoded.values[plane(0, 0, 0)], 1);
  assert.equal(encoded.values[plane(0, 24)], -1);
  assert.equal(encoded.values[plane(0, 27)], 1);
  assert.equal(encoded.values[plane(0, 28)], -1);
  assert.equal(encoded.values[plane(0, 29)], 0);
  assert.equal(encoded.values[plane(1, 22, 63)], 1);
  assert.equal(encoded.values[plane(1, 24)], 1);
  assert.equal(encoded.values[plane(1, 27)], 0);
  assert.equal(encoded.values[plane(1, 28)], 1);
  assert.equal(encoded.values[plane(1, 29)], 0.125);
  assert.equal(encoded.values[plane(2, 14, 27)], 1);
  assert.equal(encoded.values[plane(2, 25)], 0);
  assert.equal(encoded.values[plane(0, 31)], 0);
});

test("CPU neural board selection prefers canonical structure before raw timeline ids", () => {
  const quietBoard = board(4, "white", pieceAt(7, 7, "black", "pawn"));
  const royalBoard = board(4, "white", pieceAt(0, 0, "white", "king"));
  const game = {
    turn: "white",
    presentTime: 4,
    nextTimelineId: 10,
    nextBlackTimelineId: -10,
    checkedRoyals: [],
    timelines: [
      {
        id: -9,
        row: -1,
        label: "raw-first",
        owner: "white",
        active: true,
        boards: [quietBoard]
      },
      {
        id: 9,
        row: 1,
        label: "structural-first",
        owner: "white",
        active: true,
        boards: [royalBoard]
      }
    ]
  };

  const selected = neuralBoardSelection(game);

  assert.equal(selected[0].timeline.id, 9);
  assert.equal(selected[1].timeline.id, -9);
});

test("CFNN v3 preserves position-conditioned policy weights and older versions remain readable", () => {
  const base = {
    projectionSize: 4,
    projectionSeed: 9,
    hiddenLayers: [2],
    hiddenWeights: new Float32Array(10),
    outputWeights: new Float32Array(3),
    scale: 1,
    bias: 0
  };
  const v1 = encodeCompactModel(base);
  const policyLogits = Float32Array.from({ length: 257 }, (_, index) => index / 10);
  const v2 = encodeCompactModel({ ...base, policyLogits });
  const policyWeights = Float32Array.from({ length: 257 * 3 }, (_, index) => index / 100);
  const v3 = encodeCompactModel({ ...base, policyWeights });
  const v4 = encodeCompactModel({ ...base, policyWeights, outputActivation: "tanh" });

  assert.equal(v1.byteLength, 36 + 4 + (10 + 3) * 4);
  assert.equal(new DataView(v1.buffer).getUint32(4, true), 1);
  assert.equal(new DataView(v2.buffer).getUint32(4, true), 2);
  assert.equal(new DataView(v3.buffer).getUint32(4, true), 3);
  assert.equal(new DataView(v4.buffer).getUint32(4, true), 4);
  assert.deepEqual(Array.from(decodeCompactModel(v1.buffer).policyLogits), []);
  assert.deepEqual(
    Array.from(decodeCompactModel(v2.buffer).policyLogits),
    Array.from(policyLogits)
  );
  assert.deepEqual(
    Array.from(decodeCompactModel(v3.buffer).policyWeights),
    Array.from(policyWeights)
  );
  assert.equal(decodeCompactModel(v3.buffer).outputActivation, "linear");
  assert.equal(decodeCompactModel(v4.buffer).outputActivation, "tanh");
});

test("CFNN decoding rejects non-finite checkpoints and the bundled model is finite", async () => {
  const invalid = encodeCompactModel({
    projectionSize: 4,
    projectionSeed: 9,
    hiddenLayers: [2],
    hiddenWeights: new Float32Array(10),
    outputWeights: new Float32Array(3)
  });
  new DataView(invalid.buffer).setFloat32(40, Number.NaN, true);
  assert.equal(decodeCompactModel(invalid.buffer), null);

  const bundled = await readFile(path.join(root, "../engine/models/gpu-v1/value-model.cfnn"));
  assert.ok(decodeCompactModel(
    bundled.buffer.slice(bundled.byteOffset, bundled.byteOffset + bundled.byteLength)
  ));
});

test("small-replay CPU head training emits finite losses and a valid checkpoint", async () => {
  const bundled = await readFile(path.join(root, "../engine/models/gpu-v1/value-model.cfnn"));
  const active = decodeCompactModel(
    bundled.buffer.slice(bundled.byteOffset, bundled.byteOffset + bundled.byteLength)
  );
  const trained = trainHeadsOnCpu([
    sample("single", { features: [1, 0, -1], label: 0.5, labelWeight: 1 })
  ], {
    learningRate: 0.005,
    epochs: 1,
    batchSize: 16,
    validationSplit: 0.2,
    validationInterval: 1,
    patience: 2,
    weightDecay: 0.00001
  }, active);

  assert.ok(Number.isFinite(trained.trainingLoss));
  assert.ok(Number.isFinite(trained.initialValidationLoss));
  assert.ok(Number.isFinite(trained.bestValidationLoss));
  assert.equal(trained.hiddenLayersTrained, false);
  assert.ok(decodeCompactModel(
    trained.buffer.slice(trained.byteOffset, trained.byteOffset + trained.byteLength)
  ));
});

test("policy buckets describe move geometry rather than absolute time coordinates", () => {
  const first = move(1, 4, 2, 2, 1, 4, 4, 3);
  const translated = move(9, 40, 2, 2, 9, 40, 4, 3);
  const differentGeometry = move(1, 4, 2, 2, 1, 4, 3, 4);

  assert.equal(policyBucket(first), policyBucket(translated));
  assert.notEqual(policyBucket(first), policyBucket(differentGeometry));
});

test("validation split is stable for equivalent sample identities", () => {
  const samples = [
    sample("a"),
    sample("b"),
    sample("c"),
    sample("d"),
    sample("e"),
    sample("f"),
    sample("g"),
    sample("h")
  ];
  const first = splitValidationSamples(samples, 0.25);
  const second = splitValidationSamples(samples.map((entry) => ({ ...entry, features: [9, 9, 9] })), 0.25);

  assert.deepEqual(first, second);
  assert.ok(first.trainIndices.length > 0);
  assert.ok(first.validationIndices.length > 0);
});

test("validation split keeps a high-signal holdout for small requested splits", () => {
  const samples = [
    sample("distilled", { labelKind: "distilled", labelWeight: 1, pseudo: true }),
    sample("search", { labelKind: "search", labelWeight: 1 }),
    sample("outcome", { labelKind: "outcome", labelWeight: 5 })
  ];

  const split = splitValidationSamples(samples, 0.000001);

  assert.deepEqual(split.validationIndices, [2]);
  assert.deepEqual(split.trainIndices, [0, 1]);
});

test("working set selection respects adapter buffer limits and keeps stronger labels", () => {
  const samples = [
    sample("distilled-old", { labelKind: "distilled", labelWeight: 1 }),
    sample("search-low", { labelKind: "search", labelWeight: 1 }),
    sample("outcome", { labelKind: "outcome", labelWeight: 1 }),
    sample("search-high", { labelKind: "search", labelWeight: 10 }),
    sample("unknown-recent", { labelKind: "unknown", labelWeight: 1 })
  ];
  const device = {
    limits: {
      maxStorageBufferBindingSize: 3 * 2048 * Float32Array.BYTES_PER_ELEMENT
    }
  };

  const selected = selectTrainingWorkingSet(samples, device);

  assert.deepEqual(selected.map((entry) => entry.positionKey), [
    "search-low",
    "outcome",
    "search-high"
  ]);
});

test("working set selection reserves policy supervision when value labels dominate", () => {
  const samples = [
    ...Array.from({ length: 6 }, (_, index) =>
      sample(`outcome-${index}`, {
        labelKind: "outcome",
        labelWeight: 10 - index,
        policy: null
      })
    ),
    sample("policy-search", {
      labelKind: "search",
      labelWeight: 1,
      policy: 42
    }),
    sample("distilled", {
      labelKind: "distilled",
      labelWeight: 1,
      policy: null,
      pseudo: true
    })
  ];
  const device = {
    limits: {
      maxStorageBufferBindingSize: 4 * 2048 * Float32Array.BYTES_PER_ELEMENT
    }
  };

  const selected = selectTrainingWorkingSet(samples, device);

  assert.equal(selected.length, 4);
  assert.equal(selected.filter((entry) => Number.isInteger(entry.policy)).length, 1);
  assert.ok(selected.some((entry) => entry.positionKey === "policy-search"));
  assert.equal(selected.filter((entry) => entry.labelKind === "outcome").length, 3);
});

test("replay dedupe averages compatible labels and keeps the strongest policy target", () => {
  const retained = dedupeTrainingSamples([
    sample("same-position", { label: -1, labelKind: "search", policy: 1, labelWeight: 1 }),
    sample("same-position", { label: 1, labelKind: "search", policy: 2, labelWeight: 5 }),
    sample("same-position", { labelKind: "outcome", policy: 3, labelWeight: 1 })
  ]);

  assert.equal(retained.length, 2);
  assert.equal(retained[0].labelKind, "search");
  assert.equal(retained[0].policy, 2);
  assert.ok(Math.abs(retained[0].label - (4 / 6)) < 1e-9);
  assert.ok(Math.abs(retained[0].labelWeight - (5 * Math.sqrt(2))) < 1e-9);
  assert.equal(retained[0].baseLabelWeight, 5);
  assert.equal(retained[0].labelMass, 6);
  assert.equal(retained[0].observationCount, 2);
  assert.equal(retained[1].labelKind, "outcome");
});

test("replay confidence is bounded across repeated observations", () => {
  const repeated = Array.from({ length: 100 }, (_, index) =>
    sample("repeated", { label: index % 2 ? 1 : -1, labelWeight: 2 })
  );
  const [retained] = dedupeTrainingSamples(repeated);

  assert.equal(retained.observationCount, 64);
  assert.equal(retained.labelMass, 64);
  assert.equal(retained.labelWeight, 4);
  assert.ok(Math.abs(retained.label) < 0.02);
});

test("collected batch dedupe drops samples without feature vectors", () => {
  const retained = dedupeTrainingSamples([
    sample("valid", { features: [1, 2, 3] }),
    sample("typed", { features: new Float32Array([1, 2, 3]) }),
    sample("invalid", { features: null }),
    sample("also-invalid", { features: undefined })
  ]);

  assert.deepEqual(retained.map((entry) => entry.positionKey), ["valid", "typed"]);
});

test("replay dedupe fingerprints legacy samples without position keys", () => {
  const retained = dedupeTrainingSamples([
    sample(undefined, { features: [0, 2, 0], labelKind: "search", labelWeight: 1, policy: 1 }),
    sample(undefined, { features: [0, 2, 0], labelKind: "search", labelWeight: 4, policy: 2 }),
    sample(undefined, { features: [0, 2, 0], labelKind: "outcome", labelWeight: 1, policy: 3 }),
    sample(undefined, { features: [0, 0, 0], labelKind: "search", labelWeight: 1, policy: 4 }),
    sample(undefined, { features: [0, 0, 0], labelKind: "search", labelWeight: 2, policy: 5 })
  ]);

  assert.equal(retained.length, 4);
  assert.equal(retained[0].labelKind, "search");
  assert.equal(retained[0].policy, 2);
  assert.ok(Math.abs(retained[0].labelWeight - (4 * Math.sqrt(2))) < 1e-9);
  assert.deepEqual(retained.slice(1).map((entry) => [entry.labelKind, entry.policy, entry.labelWeight]), [
    ["outcome", 3, 1],
    ["search", 4, 1],
    ["search", 5, 2]
  ]);
});

test("replay retention keeps high-signal samples when the buffer is capped", () => {
  const retained = appendReplaySamples([], [
    sample("distilled", { labelKind: "distilled", labelWeight: 1, pseudo: true }),
    sample("search", { labelKind: "search", labelWeight: 1 }),
    sample("outcome", { labelKind: "outcome", labelWeight: 1 }),
    sample("cpu", { labelKind: "cpu", labelWeight: 2 })
  ], 2);

  assert.deepEqual(retained.map((entry) => entry.positionKey), ["outcome", "cpu"]);
});

test("replay retention preserves policy supervision when outcomes fill the buffer", () => {
  const retained = appendReplaySamples([], [
    ...Array.from({ length: 6 }, (_, index) =>
      sample(`outcome-${index}`, {
        labelKind: "outcome",
        labelWeight: 10 - index,
        policy: null
      })
    ),
    sample("policy-search", {
      labelKind: "search",
      labelWeight: 1,
      policy: 13
    })
  ], 4);

  assert.equal(retained.length, 4);
  assert.ok(retained.some((entry) => entry.positionKey === "policy-search"));
  assert.equal(retained.filter((entry) => entry.labelKind === "outcome").length, 3);
});

function sample(positionKey, overrides = {}) {
  return {
    positionKey,
    sideToMove: "white",
    boardCount: 1,
    features: [1, 0, 0],
    label: 0,
    labelKind: "search",
    labelWeight: 1,
    ...overrides
  };
}

function move(fromTimeline, fromTime, fromX, fromY, toTimeline, toTime, toX, toY) {
  return {
    from: { timelineId: fromTimeline, time: fromTime, x: fromX, y: fromY },
    to: { timelineId: toTimeline, time: toTime, x: toX, y: toY }
  };
}

function board(time, sideToMove, boardSquares) {
  return {
    time,
    sideToMove,
    castling: 0,
    enPassant: null,
    origin: null,
    board: boardSquares
  };
}

function pieceAt(x, y, color, type) {
  const squares = Array.from({ length: 8 }, () => Array(8).fill(null));
  squares[y][x] = { color, type };
  return squares;
}

async function buildTrainingModules() {
  const outdir = await mkdtemp(path.join(os.tmpdir(), "chronofish-web-training-test-"));
  await esbuild.build({
    entryPoints: [
      path.join(root, "src/training-gpu.ts"),
      path.join(root, "src/training-replay.ts"),
      path.join(root, "src/training-policy.ts"),
      path.join(root, "src/training-cpu.ts"),
      path.join(root, "src/training-encoding.ts")
    ],
    outdir,
    bundle: true,
    format: "esm",
    platform: "node",
    target: "es2022",
    plugins: [glsl()],
    logLevel: "silent"
  });
  return {
    trainingGpu: pathToFileURL(path.join(outdir, "training-gpu.js")).href,
    trainingReplay: pathToFileURL(path.join(outdir, "training-replay.js")).href,
    trainingPolicy: pathToFileURL(path.join(outdir, "training-policy.js")).href,
    trainingCpu: pathToFileURL(path.join(outdir, "training-cpu.js")).href,
    trainingEncoding: pathToFileURL(path.join(outdir, "training-encoding.js")).href
  };
}
