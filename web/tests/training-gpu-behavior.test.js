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
  uniqueTrainingPositionCount,
  AUXILIARY_VALUE_HEADS
} = await import(modules.trainingGpu);
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

test("CFNN v5 preserves auxiliary value heads while older versions remain readable", () => {
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
  const auxiliaryValueWeights = Float32Array.from({ length: AUXILIARY_VALUE_HEADS.length * 3 }, (_, index) => index / 50);
  const v5 = encodeCompactModel({ ...base, policyWeights, auxiliaryValueWeights, outputActivation: "tanh" });

  assert.equal(v1.byteLength, 36 + 4 + (10 + 3) * 4);
  assert.equal(new DataView(v1.buffer).getUint32(4, true), 1);
  assert.equal(new DataView(v2.buffer).getUint32(4, true), 2);
  assert.equal(new DataView(v3.buffer).getUint32(4, true), 3);
  assert.equal(new DataView(v4.buffer).getUint32(4, true), 4);
  assert.equal(new DataView(v5.buffer).getUint32(4, true), 5);
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
  assert.equal(decodeCompactModel(v5.buffer).outputActivation, "tanh");
  assert.deepEqual(
    Array.from(decodeCompactModel(v5.buffer).auxiliaryValueWeights),
    Array.from(auxiliaryValueWeights)
  );
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
  assert.ok(Number.isFinite(trained.auxiliaryValidationLoss));
  assert.equal(trained.auxiliaryHeadCount, AUXILIARY_VALUE_HEADS.length);
  assert.equal(trained.hiddenLayersTrained, false);
  const checkpoint = decodeCompactModel(
    trained.buffer.slice(trained.byteOffset, trained.byteOffset + trained.byteLength)
  );
  assert.ok(checkpoint);
  assert.equal(checkpoint.auxiliaryValueWeights.length, AUXILIARY_VALUE_HEADS.length * (checkpoint.hiddenLayers.at(-1) + 1));
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
      path.join(root, "src/training-gpu.ts")
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
    trainingGpu: pathToFileURL(path.join(outdir, "training-gpu.js")).href
  };
}
