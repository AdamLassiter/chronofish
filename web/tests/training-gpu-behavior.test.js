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
  auxiliaryValueTargetsForSamples,
  boundedValue,
  concatFloat32,
  cpuHeadTrainingMaxPositions,
  cpuPredictionMaxBatch,
  denseKernelEntryPoint,
  countNonZero,
  decodeCompactModel,
  denormalizedSearchScore,
  encodeCompactModel,
  featureLength,
  fillGroupedTrainingBatchIndices,
  groupTrainingIndicesByPosition,
  hasPolicyTrainingTarget,
  hiddenDeltaParamsData,
  initialHiddenWeights,
  inverseTanh,
  layerParamsData,
  lossReductionWorkgroupCount,
  minHiddenTrainingPositions,
  modelArchitectureMatches,
  normalizedSearchScore,
  optimizerVelocity,
  outputParamsData,
  outputLayerSize,
  outputDeltaParamsData,
  packSparseProjectionFeatures,
  policyParamsData,
  policyTrainingTarget,
  policyTrainingIndices,
  policyTrainingSteps,
  trainingBatchNormalization,
  trainingLabelPriority,
  trainingLabelWeight,
  trainingWeightedAverage,
  trainingWorkgroups16,
  trainingWorkgroups64,
  policyWeightsArray,
  projectionBatchChunkSize,
  projectionTemporaryBudget,
  projectionParamsData,
  predictValuesOnCpu,
  previousLayerSize,
  projectionHash,
  selectTrainingWorkingSet,
  splitHiddenWeights,
  splitPolicyTrainingIndices,
  splitValidationSamples,
  trainHeadsOnCpu,
  uniqueTrainingPositionCount,
  valueTrainingBatchSize,
  valueHeadValidationInterval,
  valueGpuBatchesPerSubmit,
  valueGpuValidationInterval,
  policyTrainingBatchSize,
  policyTrainingStepsPerSubmit,
  AUXILIARY_VALUE_HEADS
} = await import(modules.trainingGpu);
test("policy head training scales sublinearly with value optimization steps", () => {
  assert.equal(policyTrainingSteps(1), 16);
  assert.equal(policyTrainingSteps(1024), 16);
  assert.equal(policyTrainingSteps(4096), 64);
  assert.equal(policyTrainingSteps(16384), 256);
  assert.equal(policyTrainingSteps(65536), 256);

  let calls = 0;
  const engine = {
    chronofish_policy_training_steps(valueEpochs) {
      calls += 1;
      assert.equal(valueEpochs, 4096);
      return 64;
    }
  };
  assert.equal(policyTrainingSteps(4096, engine), 64);
  assert.equal(calls, 1);
});

test("policy target and label weight normalization use engine policy when available", () => {
  assert.equal(policyTrainingTarget(undefined), 0);
  assert.equal(policyTrainingTarget(7), 7);
  assert.equal(policyTrainingTarget(999), 256);
  assert.equal(trainingLabelWeight(undefined), 1);
  assert.equal(trainingLabelWeight(-0.5), 0);
  assert.equal(trainingLabelWeight(1.5), 1.5);

  let targetCalls = 0;
  let weightCalls = 0;
  const engine = {
    chronofish_policy_training_target(policy) {
      targetCalls += 1;
      assert.equal(policy, 7);
      return 6;
    },
    chronofish_training_label_weight(labelWeight) {
      weightCalls += 1;
      assert.equal(labelWeight, -0.5);
      return 0.25;
    }
  };
  assert.equal(policyTrainingTarget(7, engine), 6);
  assert.equal(trainingLabelWeight(-0.5, engine), 0.25);
  assert.equal(targetCalls, 1);
  assert.equal(weightCalls, 1);
});

test("weighted averages and batch normalization use engine policy when available", () => {
  assert.equal(trainingWeightedAverage(10, 2), 5);
  assert.equal(trainingWeightedAverage(10, 0), 0);
  assert.equal(trainingBatchNormalization(2), 0.5);
  assert.equal(trainingBatchNormalization(0), 1_000_000);

  let averageCalls = 0;
  let normalizationCalls = 0;
  const engine = {
    chronofish_training_weighted_average(total, totalWeight) {
      averageCalls += 1;
      assert.equal(total, 10);
      assert.equal(totalWeight, 2);
      return 4;
    },
    chronofish_training_batch_normalization(batchWeight) {
      normalizationCalls += 1;
      assert.equal(batchWeight, 2);
      return 0.25;
    }
  };
  assert.equal(trainingWeightedAverage(10, 2, engine), 4);
  assert.equal(trainingBatchNormalization(2, engine), 0.25);
  assert.equal(averageCalls, 1);
  assert.equal(normalizationCalls, 1);
});

test("training batch sizes use engine policy when available", () => {
  assert.equal(valueTrainingBatchSize(64, 0), 1);
  assert.equal(valueTrainingBatchSize(64, 12), 12);
  assert.equal(valueTrainingBatchSize(64, 128), 64);
  assert.equal(policyTrainingBatchSize(64, 0), 0);
  assert.equal(policyTrainingBatchSize(64, 12), 12);
  assert.equal(policyTrainingBatchSize(64, 128), 64);

  let valueCalls = 0;
  let policyCalls = 0;
  const engine = {
    chronofish_value_training_batch_size(configBatchSize, trainingCount) {
      valueCalls += 1;
      assert.equal(configBatchSize, 64);
      assert.equal(trainingCount, 12);
      return 11;
    },
    chronofish_policy_training_batch_size(configBatchSize, trainingCount) {
      policyCalls += 1;
      assert.equal(configBatchSize, 64);
      assert.equal(trainingCount, 12);
      return 10;
    }
  };
  assert.equal(valueTrainingBatchSize(64, 12, engine), 11);
  assert.equal(policyTrainingBatchSize(64, 12, engine), 10);
  assert.equal(valueCalls, 1);
  assert.equal(policyCalls, 1);
});

test("training submit schedules use engine policy when available", () => {
  assert.equal(valueHeadValidationInterval(0), 1);
  assert.equal(valueHeadValidationInterval(100, 16), 16);
  assert.equal(valueHeadValidationInterval(10, 256), 10);
  assert.equal(valueGpuBatchesPerSubmit(0), 1);
  assert.equal(valueGpuBatchesPerSubmit(12), 12);
  assert.equal(valueGpuBatchesPerSubmit(128), 64);
  assert.equal(valueGpuValidationInterval(12), 256);
  assert.equal(valueGpuValidationInterval(64, 16), 64);
  assert.equal(valueGpuValidationInterval(64, 128), 128);
  assert.equal(policyTrainingStepsPerSubmit(0), 0);
  assert.equal(policyTrainingStepsPerSubmit(12), 12);
  assert.equal(policyTrainingStepsPerSubmit(128), 64);

  let calls = 0;
  const engine = {
    chronofish_value_head_validation_interval(epochs, validationInterval) {
      calls += 1;
      assert.equal(epochs, 100);
      assert.equal(validationInterval, 16);
      return 15;
    },
    chronofish_value_gpu_batches_per_submit(epochs) {
      calls += 1;
      assert.equal(epochs, 128);
      return 63;
    },
    chronofish_value_gpu_validation_interval(batchesPerSubmit, validationInterval) {
      calls += 1;
      assert.equal(batchesPerSubmit, 63);
      assert.equal(validationInterval, 16);
      return 62;
    },
    chronofish_policy_training_steps_per_submit(steps) {
      calls += 1;
      assert.equal(steps, 128);
      return 61;
    }
  };
  assert.equal(valueHeadValidationInterval(100, 16, engine), 15);
  assert.equal(valueGpuBatchesPerSubmit(128, engine), 63);
  assert.equal(valueGpuValidationInterval(63, 16, engine), 62);
  assert.equal(policyTrainingStepsPerSubmit(128, engine), 61);
  assert.equal(calls, 4);
});

test("training momentum accumulates consistent gradients and reacts to reversals", () => {
  const first = optimizerVelocity(0, 2);
  const second = optimizerVelocity(first, 2);
  const reversed = optimizerVelocity(second, -2);

  assert.ok(Math.abs(first - 0.2) < 1e-12);
  assert.ok(Math.abs(second - 0.38) < 1e-12);
  assert.ok(reversed > 0);
  assert.ok(reversed < second);

  let calls = 0;
  const engine = {
    chronofish_optimizer_velocity(previous, gradient, momentum) {
      calls += 1;
      assert.equal(previous, 3);
      assert.equal(gradient, -2);
      assert.equal(momentum, 0.75);
      return 1.75;
    }
  };
  assert.equal(optimizerVelocity(3, -2, 0.75, engine), 1.75);
  assert.equal(calls, 1);
});

test("default layer-size helpers delegate to engine when available", () => {
  const calls = [];
  const engine = {
    chronofish_default_output_layer_size() {
      calls.push(["output"]);
      return 256;
    },
    chronofish_default_previous_layer_size(layerIndex, inputSize) {
      calls.push(["previous", layerIndex, inputSize]);
      return layerIndex === 0 ? inputSize : 1024;
    }
  };

  assert.equal(outputLayerSize(undefined, engine), 256);
  assert.equal(previousLayerSize([1024, 512, 256], 0, 2048, engine), 2048);
  assert.equal(previousLayerSize([1024, 512, 256], 2, 2048, engine), 1024);
  assert.deepEqual(calls, [
    ["output"],
    ["previous", 0, 2048],
    ["previous", 2, 2048]
  ]);
});

test("default model initialization helpers delegate to engine when available", () => {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const outputPtr = 32768;
  const weights = Float32Array.from([0.25, -0.5, 0, 0.75]);
  const customWeights = Float32Array.from([0.5, -0.25, 0]);
  const calls = [];
  let outputLen = 0;
  let nextPtr = 1024;
  const engine = {
    memory,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      return outputPtr;
    },
    chronofish_projection_hash(rawIndex, projectionIndex, seed) {
      calls.push(["hash", rawIndex, projectionIndex, seed]);
      return 0xffffffff;
    },
    chronofish_default_initial_hidden_weights_bytes() {
      calls.push(["weights"]);
      new Uint8Array(memory.buffer, outputPtr, weights.byteLength).set(new Uint8Array(weights.buffer));
      outputLen = weights.byteLength;
      return outputPtr;
    },
    chronofish_initial_hidden_weights_bytes(ptr, length) {
      const view = new DataView(memory.buffer, ptr, length);
      calls.push([
        "customWeights",
        view.getUint32(0, true),
        view.getUint32(4, true),
        view.getUint32(8, true)
      ]);
      new Uint8Array(memory.buffer, outputPtr, customWeights.byteLength).set(new Uint8Array(customWeights.buffer));
      outputLen = customWeights.byteLength;
      return outputPtr;
    }
  };

  assert.equal(projectionHash(1, 2, 3, engine), 0xffffffff);
  assert.deepEqual(Array.from(initialHiddenWeights(2048, [1024, 512, 256], engine)), Array.from(weights));
  assert.deepEqual(Array.from(initialHiddenWeights(7, [2], engine)), Array.from(customWeights));
  assert.deepEqual(calls, [
    ["hash", 1, 2, 3],
    ["weights"],
    ["customWeights", 7, 1, 2]
  ]);
});

test("training vector utilities delegate to engine when available", () => {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const outputPtr = 32768;
  let outputLen = 0;
  let nextPtr = 1024;
  const calls = [];
  const writeOutput = (bytes) => {
    new Uint8Array(memory.buffer, outputPtr, bytes.byteLength).set(bytes);
    outputLen = bytes.byteLength;
    return outputPtr;
  };
  const engine = {
    memory,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      return outputPtr;
    },
    chronofish_split_hidden_weights_bytes(ptr, length) {
      const view = new DataView(memory.buffer, ptr, length);
      calls.push(["split", view.getUint32(0, true), view.getUint32(4, true), view.getUint32(8, true)]);
      assert.equal(view.getUint32(12, true), 2);
      assert.equal(view.getUint32(16, true), 1);
      const response = new Uint8Array(4 + 2 * 4 + 3 * 4);
      const output = new DataView(response.buffer);
      output.setUint32(0, 2, true);
      output.setUint32(4, 2, true);
      output.setUint32(8, 1, true);
      output.setFloat32(12, 1, true);
      output.setFloat32(16, 2, true);
      output.setFloat32(20, 3, true);
      return writeOutput(response);
    },
    chronofish_concat_f32_bytes(ptr, length) {
      const view = new DataView(memory.buffer, ptr, length);
      calls.push(["concat", view.getUint32(0, true), view.getUint32(4, true), view.getUint32(8, true)]);
      const values = Float32Array.from([4, 5, 6]);
      return writeOutput(new Uint8Array(values.buffer));
    },
    chronofish_count_non_zero_f32_bytes(ptr, length) {
      const values = new Float32Array(memory.buffer, ptr, length / Float32Array.BYTES_PER_ELEMENT);
      calls.push(["count", Array.from(values)]);
      return 2;
    }
  };

  const layers = splitHiddenWeights(Float32Array.from([1, 2, 3]), 4, [2, 1], engine);
  assert.deepEqual(layers.map((layer) => Array.from(layer)), [[1, 2], [3]]);
  assert.deepEqual(Array.from(concatFloat32([Float32Array.from([4]), Float32Array.from([5, 6])], engine)), [4, 5, 6]);
  assert.equal(countNonZero(Float32Array.from([0, -1, 0, 2]), engine), 2);
  assert.deepEqual(calls, [
    ["split", 4, 2, 3],
    ["concat", 2, 1, 2],
    ["count", [0, -1, 0, 2]]
  ]);
});

test("value targets and search scores use an invertible bounded scale", () => {
  assert.equal(normalizedSearchScore(20_000), 1);
  assert.equal(normalizedSearchScore(-10_000), -0.5);
  assert.equal(normalizedSearchScore(100_000), 1);
  assert.equal(denormalizedSearchScore(0.75), 15_000);
  assert.equal(denormalizedSearchScore(-2), -20_000);
  assert.ok(Math.abs(Math.tanh(inverseTanh(0.75)) - 0.75) < 1e-12);

  const calls = [];
  const engine = {
    chronofish_normalized_search_score(score) {
      calls.push(["normalized", score]);
      return 0.25;
    },
    chronofish_denormalized_search_score(value) {
      calls.push(["denormalized", value]);
      return 5_000;
    },
    chronofish_bounded_value(value) {
      calls.push(["bounded", value]);
      return -1;
    },
    chronofish_inverse_tanh(value) {
      calls.push(["inverse", value]);
      return 0.5;
    }
  };
  assert.equal(normalizedSearchScore(1234, engine), 0.25);
  assert.equal(denormalizedSearchScore(0.25, engine), 5_000);
  assert.equal(boundedValue(3, engine), -1);
  assert.equal(inverseTanh(0.25, engine), 0.5);
  assert.deepEqual(calls, [
    ["normalized", 1234],
    ["denormalized", 0.25],
    ["bounded", 3],
    ["inverse", 0.25]
  ]);
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

test("policy target index selection delegates to engine when available", () => {
  const samples = [
    sample("policy", { policy: 7 }),
    sample("distilled", { policy: 9, labelKind: "distilled" }),
    sample("zero", { policy: 11, labelWeight: 0 })
  ];
  const engine = policyIndexEngine([0], ({ inputSamples, requirePositiveWeight }) => {
    assert.equal(inputSamples.length, samples.length);
    assert.equal(requirePositiveWeight, 1);
  });

  assert.deepEqual(policyTrainingIndices(samples, true, engine), [0]);
  assert.equal(engine.calls, 1);
});

test("single policy target checks delegate to engine when available", () => {
  const memory = new WebAssembly.Memory({ initial: 1 });
  let nextPtr = 1024;
  let calls = 0;
  const engine = {
    memory,
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_has_policy_training_target_json(ptr, length) {
      calls += 1;
      const request = JSON.parse(new TextDecoder("utf-8").decode(new Uint8Array(memory.buffer, ptr, length)));
      assert.equal(request.positionKey, "single");
      assert.equal(request.policy, 12);
      return 1;
    }
  };

  assert.equal(hasPolicyTrainingTarget(sample("single", { policy: 12 })), true);
  assert.equal(hasPolicyTrainingTarget(sample("single", { policy: 12 }), engine), true);
  assert.equal(calls, 1);
});

test("policy holdout delegates to engine when available", () => {
  const samples = ["a", "b", "c"].map((key) => sample(key));
  const split = { trainIndices: [0, 1], validationIndices: [2], seed: 99 };
  const engine = policySplitEngine(
    { trainIndices: [1], validationIndices: [2], seed: 99 },
    ({ request, validationSplit }) => {
      assert.equal(request.samples.length, samples.length);
      assert.deepEqual(request.policyIndices, [1, 2]);
      assert.deepEqual(request.split, split);
      assert.equal(validationSplit, 0.2);
    }
  );

  assert.deepEqual(splitPolicyTrainingIndices(samples, [1, 2], split, 0.2, engine), {
    trainIndices: [1],
    validationIndices: [2],
    seed: 99
  });
  assert.equal(engine.calls, 1);
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

  const engineBatch = new Uint32Array(32);
  const engine = groupedBatchFillEngine(Array.from(firstBatch), uniformBatchWeight, ({ request }) => {
    assert.equal(request.batchLength, 32);
    assert.equal(request.groupCount, 2);
    assert.equal(request.itemCount, 3);
    assert.equal(request.labelWeightCount, 3);
    assert.equal(request.epoch, 1);
    assert.equal(request.seed, 1234);
  });
  assert.equal(fillGroupedTrainingBatchIndices(engineBatch, trainGroups, 1, 1234, uniformWeights, engine), uniformBatchWeight);
  assert.deepEqual(engineBatch, firstBatch);
  assert.equal(engine.calls, 1);
});

test("position grouping delegates to engine when available", () => {
  const samples = [
    sample("same", { labelKind: "search" }),
    sample("same", { labelKind: "outcome" }),
    sample("other")
  ];
  const engine = groupIndicesEngine([[0, 1], [2]], ({ request }) => {
    assert.equal(request.samples.length, samples.length);
    assert.deepEqual(request.indices, [0, 1, 2]);
  });

  assert.deepEqual(groupTrainingIndicesByPosition(samples, [0, 1, 2], engine), [[0, 1], [2]]);
  assert.equal(engine.calls, 1);
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

test("unique position counting delegates to engine when available", () => {
  const samples = [
    sample("same", { labelKind: "search" }),
    sample("same", { labelKind: "outcome" }),
    sample("other")
  ];
  const engine = uniqueCountEngine(2, ({ request }) => {
    assert.equal(request.samples.length, samples.length);
    assert.deepEqual(request.indices, [0, 1, 2]);
  });

  assert.equal(uniqueTrainingPositionCount(samples, [0, 1, 2], engine), 2);
  assert.equal(engine.calls, 1);
});

test("feature length delegates to engine when available", () => {
  const samples = [
    sample("first", { features: [0, 2, 0, -1] }),
    sample("second", { features: [3, 0, 4, 0] })
  ];
  const engine = featureLengthEngine(4, ({ inputSamples }) => {
    assert.equal(inputSamples.length, samples.length);
  });

  assert.equal(featureLength(samples, engine), 4);
  assert.equal(engine.calls, 1);
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

test("training label priority delegates to engine when available", () => {
  assert.equal(trainingLabelPriority("outcome", false), 4);
  assert.equal(trainingLabelPriority("cpu", true), 3);
  assert.equal(trainingLabelPriority("distilled", false), 1);
  assert.equal(trainingLabelPriority(undefined, true), 1);
  assert.equal(trainingLabelPriority("unknown", false), 2);

  const memory = new WebAssembly.Memory({ initial: 1 });
  const decoder = new TextDecoder("utf-8");
  let nextPtr = 1024;
  const calls = [];
  const engine = {
    memory,
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_training_label_priority(ptr, length, pseudo) {
      calls.push([decoder.decode(new Uint8Array(memory.buffer, ptr, length)), pseudo]);
      return 7;
    }
  };

  assert.equal(trainingLabelPriority("duel-search", true, engine), 7);
  assert.equal(trainingLabelPriority(undefined, false, engine), 7);
  assert.deepEqual(calls, [
    ["duel-search", 1],
    ["", 0]
  ]);
});

test("validation split delegates to engine when available", () => {
  const samples = [sample("a"), sample("b"), sample("c")];
  const engine = validationSplitEngine(
    { trainIndices: [0, 2], validationIndices: [1], seed: 123 },
    ({ samples: inputSamples, validationSplit }) => {
      assert.equal(inputSamples.length, samples.length);
      assert.equal(validationSplit, 0.25);
    }
  );

  assert.deepEqual(splitValidationSamples(samples, 0.25, engine), {
    trainIndices: [0, 2],
    validationIndices: [1],
    seed: 123
  });
  assert.equal(engine.calls, 1);
});

test("GPU loss reduction emits one partial per 64 validation samples", () => {
  assert.equal(lossReductionWorkgroupCount(1), 1);
  assert.equal(lossReductionWorkgroupCount(64), 1);
  assert.equal(lossReductionWorkgroupCount(65), 2);
  assert.equal(lossReductionWorkgroupCount(4096), 64);

  let calls = 0;
  const engine = {
    chronofish_loss_reduction_workgroup_count(sampleCount) {
      calls += 1;
      assert.equal(sampleCount, 65);
      return 2;
    }
  };
  assert.equal(lossReductionWorkgroupCount(65, engine), 2);
  assert.equal(calls, 1);
});

test("GPU training dispatch workgroups delegate to engine when available", async () => {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const outputPtr = 32768;
  let outputLen = 0;
  assert.equal(cpuHeadTrainingMaxPositions(), 32);
  assert.equal(cpuPredictionMaxBatch(), 4);
  assert.equal(minHiddenTrainingPositions(), 256);
  assert.equal(projectionBatchChunkSize(), 256);
  assert.equal(denseKernelEntryPoint("forward_layer", 15), "forward_layer_naive");
  assert.equal(denseKernelEntryPoint("forward_layer", 16), "forward_layer");
  assert.equal(trainingWorkgroups16(0), 0);
  assert.equal(trainingWorkgroups16(1), 1);
  assert.equal(trainingWorkgroups16(16), 1);
  assert.equal(trainingWorkgroups16(17), 2);
  assert.equal(trainingWorkgroups64(0), 0);
  assert.equal(trainingWorkgroups64(64), 1);
  assert.equal(trainingWorkgroups64(65), 2);

  const calls = [];
  const engine = {
    memory,
    chronofish_alloc(length) {
      calls.push(["alloc", length]);
      return 1024;
    },
    chronofish_dealloc(ptr, length) {
      calls.push(["dealloc", ptr, length]);
    },
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_cpu_head_training_max_positions() {
      calls.push(["cpu-head"]);
      return 31;
    },
    chronofish_cpu_prediction_max_batch() {
      calls.push(["cpu-predict"]);
      return 3;
    },
    chronofish_min_hidden_training_positions() {
      calls.push(["min-hidden"]);
      return 255;
    },
    chronofish_projection_chunk_size() {
      calls.push(["projection-chunk"]);
      return 12;
    },
    chronofish_dense_kernel_entry_point_bytes(ptr, length, sampleCount) {
      const input = new TextDecoder().decode(new Uint8Array(memory.buffer, ptr, length));
      calls.push(["dense", input, sampleCount]);
      const output = new TextEncoder().encode(`${input}:engine:${sampleCount}`);
      new Uint8Array(memory.buffer, outputPtr, output.length).set(output);
      outputLen = output.length;
      return outputPtr;
    },
    chronofish_training_workgroups_16(itemCount) {
      calls.push(["x16", itemCount]);
      return itemCount + 100;
    },
    chronofish_training_workgroups_64(itemCount) {
      calls.push(["x64", itemCount]);
      return itemCount + 200;
    }
  };
  assert.equal(cpuHeadTrainingMaxPositions(engine), 31);
  assert.equal(cpuPredictionMaxBatch(engine), 3);
  assert.equal(minHiddenTrainingPositions(engine), 255);
  assert.equal(projectionBatchChunkSize(engine), 12);
  assert.equal(denseKernelEntryPoint("forward_layer", 7, engine), "forward_layer:engine:7");
  assert.equal(trainingWorkgroups16(7, engine), 107);
  assert.equal(trainingWorkgroups64(9, engine), 209);
  assert.deepEqual(calls, [
    ["cpu-head"],
    ["cpu-predict"],
    ["min-hidden"],
    ["projection-chunk"],
    ["alloc", 13],
    ["dense", "forward_layer", 7],
    ["dealloc", 1024, 13],
    ["x16", 7],
    ["x64", 9]
  ]);

  const trainingGpu = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const wasmApi = await readFile(path.join(path.dirname(root), "engine/src/wasm_api.rs"), "utf8");
  const engineTraining = await readFile(path.join(path.dirname(root), "engine/src/gpu/training.rs"), "utf8");
  assert.match(trainingGpu, /trainingWorkgroups16\(batchSize, engine\)/);
  assert.match(trainingGpu, /trainingWorkgroups64\(batchSize, engine\)/);
  assert.match(trainingGpu, /denseKernelEntryPoint\("forward_layer", batchSize, engine\)/);
  assert.match(trainingGpu, /denseKernelEntryPoint\("apply_layer", batchSize, engine\)/);
  assert.match(trainingGpu, /denseKernelEntryPoint\("hidden_delta", batchSize, engine\)/);
  assert.match(trainingGpu, /denseKernelEntryPoint\("forward_policy", batchSize, engine\)/);
  assert.match(trainingGpu, /denseKernelEntryPoint\("apply_policy", batchSize, engine\)/);
  assert.match(trainingGpu, /predictValuesOnGpu\(device, samples, model, engine\)/);
  assert.match(trainingGpu, /denseKernelEntryPoint\("forward_layer", sampleCount, engine\)/);
  assert.match(trainingGpu, /trainingWorkgroups16\(projectionSize, engine\)/);
  assert.match(trainingGpu, /trainingWorkgroups16\(sampleCount, engine\)/);
  assert.match(trainingGpu, /trainingWorkgroups64\(sampleCount, engine\)/);
  assert.match(trainingGpu, /boundedValue\(\(predictions\[index\] \?\? 0\) \* scale \+ bias, engine\)/);
  assert.match(trainingGpu, /uniqueTrainingPositionCount\(samples, samples\.map\(\(_, index\) => index\), engine\) <= cpuHeadTrainingMaxPositions\(engine\)/);
  assert.doesNotMatch(trainingGpu, /uniqueTrainingPositionCount\(samples, samples\.map\(\(_, index\) => index\), engine\) <= CPU_HEAD_TRAINING_MAX_POSITIONS/);
  assert.match(trainingGpu, /uniqueTrainingPositionCount\(samples, trainIndices, engine\) >= minHiddenTrainingPositions\(engine\)/);
  assert.doesNotMatch(trainingGpu, /uniqueTrainingPositionCount\(samples, trainIndices, engine\) >= MIN_HIDDEN_TRAINING_POSITIONS/);
  assert.match(trainingGpu, /const projectionChunkSize = projectionBatchChunkSize\(engine\)/);
  assert.match(trainingGpu, /samples\.slice\(offset, offset \+ projectionChunkSize\)/);
  assert.doesNotMatch(trainingGpu, /samples\.slice\(offset, offset \+ PROJECTION_CHUNK_SIZE\)/);
  assert.match(trainingGpu, /samples\.length <= cpuPredictionMaxBatch\(engine\)/);
  assert.doesNotMatch(trainingGpu, /samples\.length <= CPU_PREDICTION_MAX_BATCH/);
  assert.match(engineTypes, /chronofish_cpu_head_training_max_positions\(\): number/);
  assert.match(engineTypes, /chronofish_cpu_prediction_max_batch\(\): number/);
  assert.match(engineTypes, /chronofish_min_hidden_training_positions\(\): number/);
  assert.match(engineTypes, /chronofish_projection_chunk_size\(\): number/);
  assert.match(engineTypes, /chronofish_dense_kernel_entry_point_bytes\(ptr: number, length: number, sampleCount: number\): number/);
  assert.match(engineTypes, /chronofish_training_workgroups_16\(itemCount: number\): number/);
  assert.match(engineTypes, /chronofish_training_workgroups_64\(itemCount: number\): number/);
  assert.match(engineTypes, /chronofish_align4\(value: number\): number/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_head_training_max_positions/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_prediction_max_batch/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_min_hidden_training_positions/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_projection_chunk_size/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_dense_kernel_entry_point_bytes/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_training_workgroups_16/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_training_workgroups_64/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_align4/);
  assert.match(engineTraining, /pub fn cpu_head_training_max_positions/);
  assert.match(engineTraining, /pub fn cpu_prediction_max_batch/);
  assert.match(engineTraining, /pub fn min_hidden_training_positions/);
  assert.match(engineTraining, /pub fn projection_chunk_size/);
  assert.match(engineTraining, /pub fn training_workgroups_16/);
  assert.match(engineTraining, /pub fn training_workgroups_64/);
  assert.match(engineTraining, /pub fn align4/);
});

test("projection temporary budget delegates device-limit policy to engine", async () => {
  const smallDevice = { limits: { maxBufferSize: 64 } };
  assert.equal(projectionTemporaryBudget(smallDevice), 32);
  assert.equal(projectionTemporaryBudget({ limits: { maxBufferSize: 512 * 1024 * 1024 } }), 128 * 1024 * 1024);

  let calls = 0;
  const engine = {
    chronofish_projection_temporary_budget(maxBufferSize) {
      calls += 1;
      assert.equal(maxBufferSize, 64);
      return 17;
    }
  };
  assert.equal(projectionTemporaryBudget(smallDevice, engine), 17);
  assert.equal(calls, 1);

  const trainingGpu = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const wasmApi = await readFile(path.join(path.dirname(root), "engine/src/wasm_api.rs"), "utf8");
  const engineTraining = await readFile(path.join(path.dirname(root), "engine/src/gpu/training.rs"), "utf8");
  assert.match(trainingGpu, /const temporaryBudget = projectionTemporaryBudget\(device, engine\)/);
  assert.match(trainingGpu, /engine\.chronofish_projection_temporary_budget\(maxBufferSize\)/);
  assert.match(engineTypes, /chronofish_projection_temporary_budget\(maxBufferSize: number\): number/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_projection_temporary_budget/);
  assert.match(engineTraining, /pub fn projection_temporary_budget/);
});

test("training uniform parameter packing delegates to engine when available", () => {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const outputPtr = 32768;
  let outputLen = 0;
  const calls = [];
  const engine = {
    memory,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_output_delta_params_bytes(sampleCount, totalWeight) {
      calls.push(["delta", sampleCount, totalWeight]);
      const bytes = new Uint8Array(16);
      const view = new DataView(bytes.buffer);
      view.setUint32(0, sampleCount, true);
      view.setFloat32(4, Math.max(0, totalWeight), true);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_hidden_delta_params_bytes(sampleCount, currentSize, nextSize) {
      calls.push(["hidden-delta", sampleCount, currentSize, nextSize]);
      const bytes = new Uint8Array(16);
      const view = new DataView(bytes.buffer);
      view.setUint32(0, sampleCount, true);
      view.setUint32(4, currentSize, true);
      view.setUint32(8, nextSize, true);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_policy_params_bytes(batchCount, inputSize, totalWeight, learningRate, weightDecay, momentum) {
      calls.push(["policy", batchCount, inputSize, totalWeight, learningRate, weightDecay, momentum]);
      const bytes = new Uint8Array(32);
      const view = new DataView(bytes.buffer);
      view.setUint32(0, batchCount, true);
      view.setUint32(4, inputSize, true);
      view.setUint32(8, 257, true);
      view.setFloat32(16, Math.max(0, totalWeight), true);
      view.setFloat32(20, learningRate, true);
      view.setFloat32(24, weightDecay, true);
      view.setFloat32(28, momentum, true);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_layer_params_bytes(sampleCount, inputSize, outputSize, learningRate, weightDecay, momentum) {
      calls.push(["layer", sampleCount, inputSize, outputSize, learningRate, weightDecay, momentum]);
      const bytes = new Uint8Array(32);
      const view = new DataView(bytes.buffer);
      view.setUint32(0, sampleCount, true);
      view.setUint32(4, inputSize, true);
      view.setUint32(8, outputSize, true);
      view.setFloat32(12, learningRate, true);
      view.setFloat32(16, weightDecay, true);
      view.setFloat32(20, momentum, true);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_output_params_bytes(sampleCount, inputSize, learningRate, weightDecay, momentum) {
      calls.push(["output", sampleCount, inputSize, learningRate, weightDecay, momentum]);
      const bytes = new Uint8Array(32);
      const view = new DataView(bytes.buffer);
      view.setUint32(0, sampleCount, true);
      view.setUint32(4, inputSize, true);
      view.setFloat32(12, learningRate, true);
      view.setFloat32(16, weightDecay, true);
      view.setFloat32(20, momentum, true);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_projection_params_bytes(sampleCount, inputSize, projectionSize, seed, outputOffset) {
      calls.push(["projection", sampleCount, inputSize, projectionSize, seed, outputOffset]);
      const bytes = new Uint8Array(32);
      const view = new DataView(bytes.buffer);
      view.setUint32(0, sampleCount, true);
      view.setUint32(4, inputSize, true);
      view.setUint32(8, projectionSize, true);
      view.setUint32(12, seed, true);
      view.setUint32(16, outputOffset, true);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    }
  };

  const delta = new DataView(outputDeltaParamsData(32, -4, engine));
  assert.equal(delta.getUint32(0, true), 32);
  assert.equal(delta.getFloat32(4, true), 0);
  const hiddenDelta = new DataView(hiddenDeltaParamsData(48, 64, 128, engine));
  assert.equal(hiddenDelta.getUint32(0, true), 48);
  assert.equal(hiddenDelta.getUint32(4, true), 64);
  assert.equal(hiddenDelta.getUint32(8, true), 128);
  const policy = new DataView(policyParamsData(16, 2048, 12.5, {
    learningRate: 0.01,
    epochs: 1,
    batchSize: 16,
    patience: 1,
    weightDecay: 0.00001
  }, engine));
  assert.equal(policy.getUint32(8, true), 257);
  assert.equal(policy.getFloat32(16, true), 12.5);
  const layer = new DataView(layerParamsData(8, 32, 64, 0.25, 0.01, 0.9, engine));
  assert.equal(layer.getUint32(8, true), 64);
  assert.equal(layer.getFloat32(12, true), 0.25);
  const output = new DataView(outputParamsData(8, 64, 0.5, 0.02, 0.75, engine));
  assert.equal(output.getUint32(4, true), 64);
  assert.equal(output.getFloat32(20, true), 0.75);
  const projection = new DataView(projectionParamsData(4, 128, 2048, -1, 16, engine));
  assert.equal(projection.getUint32(8, true), 2048);
  assert.equal(projection.getUint32(12, true), 0xffffffff);
  assert.equal(projection.getUint32(16, true), 16);
  assert.deepEqual(calls, [
    ["delta", 32, -4],
    ["hidden-delta", 48, 64, 128],
    ["policy", 16, 2048, 12.5, 0.01, 0.00001, 0.9],
    ["layer", 8, 32, 64, 0.25, 0.01, 0.9],
    ["output", 8, 64, 0.5, 0.02, 0.75],
    ["projection", 4, 128, 2048, 0xffffffff, 16]
  ]);
});

test("auxiliary value targets delegate to engine when available", () => {
  const targets = Float32Array.from([1, 0.2, 0.25, 0.5, 1, 1, 7 / 16, 0, 0]);
  const memory = new WebAssembly.Memory({ initial: 1 });
  const outputPtr = 32768;
  let outputLen = 0;
  let calls = 0;
  let nextPtr = 1024;
  const engine = {
    memory,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      return outputPtr;
    },
    chronofish_auxiliary_value_targets_bytes(ptr, length) {
      calls += 1;
      const request = JSON.parse(new TextDecoder("utf-8").decode(new Uint8Array(memory.buffer, ptr, length)));
      assert.equal(request.length, 1);
      assert.equal(request[0].positionKey, "aux");
      new Uint8Array(memory.buffer, outputPtr, targets.byteLength).set(new Uint8Array(targets.buffer));
      outputLen = targets.byteLength;
      return outputPtr;
    }
  };

  assert.deepEqual(Array.from(auxiliaryValueTargetsForSamples([sample("aux")], engine)), Array.from(targets));
  assert.equal(calls, 1);
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

test("feature projection packing delegates to engine when available", () => {
  const samples = [
    sample("first", { features: [0, 2, 0, -1] }),
    sample("empty", { features: [0, 0, 0, 0] }),
    sample("last", { features: [3, 0, 4, 0] })
  ];
  const engine = sparseProjectionEngine({
    offsets: [0, 2, 2, 4],
    indices: [1, 3, 0, 2],
    values: [2, -1, 3, 4],
    byteLength: 48
  }, ({ inputSamples, inputSize }) => {
    assert.equal(inputSamples.length, samples.length);
    assert.equal(inputSize, 4);
  });

  const packed = packSparseProjectionFeatures(samples, 4, engine);

  assert.deepEqual(Array.from(packed.offsets), [0, 2, 2, 4]);
  assert.deepEqual(Array.from(packed.indices), [1, 3, 0, 2]);
  assert.deepEqual(Array.from(packed.values), [2, -1, 3, 4]);
  assert.equal(packed.byteLength, 48);
  assert.equal(engine.calls, 1);
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

test("compact model encoding delegates to engine when available", () => {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const encoded = Uint8Array.from([67, 70, 78, 78, 99, 1, 2, 3]);
  const outputPtr = 32768;
  let outputLen = 0;
  let nextPtr = 1024;
  let calls = 0;
  const decoder = new TextDecoder("utf-8");
  const engine = {
    memory,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_compact_value_model_bytes_json(ptr, length) {
      calls += 1;
      const request = JSON.parse(decoder.decode(new Uint8Array(memory.buffer, ptr, length)));
      assert.equal(request.projectionSize, 4);
      assert.equal(request.projectionSeed, 9);
      assert.deepEqual(request.hiddenLayers, [2]);
      assert.deepEqual(request.hiddenWeights, [1, 2]);
      assert.deepEqual(request.outputWeights, [3, 4, 5]);
      assert.deepEqual(request.policyWeights, [6, 7]);
      assert.equal(request.outputActivation, "tanh");
      new Uint8Array(memory.buffer, outputPtr, encoded.byteLength).set(encoded);
      outputLen = encoded.byteLength;
      return outputPtr;
    }
  };

  const result = encodeCompactModel({
    projectionSize: 4,
    projectionSeed: 9,
    hiddenLayers: [2],
    hiddenWeights: new Float32Array([1, 2]),
    outputWeights: new Float32Array([3, 4, 5]),
    policyWeights: new Float32Array([6, 7]),
    outputActivation: "tanh"
  }, engine);

  assert.deepEqual(Array.from(result), Array.from(encoded));
  assert.equal(calls, 1);
});

test("compact model architecture matching delegates to engine when available", () => {
  const model = {
    projectionSize: 4,
    projectionSeed: 9,
    hiddenLayers: [2],
    hiddenWeights: new Float32Array(10),
    outputWeights: new Float32Array(3),
    scale: 1,
    bias: 0,
    outputActivation: "tanh"
  };
  const memory = new WebAssembly.Memory({ initial: 1 });
  let calls = 0;
  let nextPtr = 1024;
  const engine = {
    memory,
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_compact_value_model_architecture_matches_bytes(ptr, length) {
      calls += 1;
      const bytes = new Uint8Array(memory.buffer, ptr, length);
      assert.equal(new TextDecoder("ascii").decode(bytes.slice(0, 4)), "CFNN");
      return 1;
    }
  };

  assert.equal(modelArchitectureMatches(model, engine), true);
  assert.equal(calls, 1);
});

test("compact model policy weights delegate to engine when available", () => {
  const model = {
    projectionSize: 4,
    projectionSeed: 9,
    hiddenLayers: [2],
    hiddenWeights: new Float32Array(10),
    outputWeights: new Float32Array(3),
    policyLogits: Float32Array.from([0.25, -0.5]),
    scale: 1,
    bias: 0
  };
  const outputWeights = Float32Array.from([0, 0, 0.25, 0, 0, -0.5]);
  const memory = new WebAssembly.Memory({ initial: 1 });
  const outputPtr = 32768;
  let outputLen = 0;
  let calls = 0;
  let nextPtr = 1024;
  const engine = {
    memory,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_compact_value_model_policy_weights_bytes(ptr, length, inputSize) {
      calls += 1;
      const bytes = new Uint8Array(memory.buffer, ptr, length);
      assert.equal(new TextDecoder("ascii").decode(bytes.slice(0, 4)), "CFNN");
      assert.equal(inputSize, 2);
      new Uint8Array(memory.buffer, outputPtr, outputWeights.byteLength).set(new Uint8Array(outputWeights.buffer));
      outputLen = outputWeights.byteLength;
      return outputPtr;
    }
  };

  assert.deepEqual(Array.from(policyWeightsArray(model, 2, engine)), Array.from(outputWeights));
  assert.equal(calls, 1);
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

  const engine = workingSetEngine([1, 2, 3], ({ samples: inputSamples, maxProjectedBytes }) => {
    assert.equal(inputSamples.length, samples.length);
    assert.equal(maxProjectedBytes, 3 * 2048 * Float32Array.BYTES_PER_ELEMENT);
  });
  const selected = selectTrainingWorkingSet(samples, device, engine);

  assert.deepEqual(selected.map((entry) => entry.positionKey), [
    "search-low",
    "outcome",
    "search-high"
  ]);
  assert.equal(engine.calls, 1);
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

  const engine = workingSetEngine([0, 1, 2, 6], ({ samples: inputSamples, maxProjectedBytes }) => {
    assert.equal(inputSamples.length, samples.length);
    assert.equal(maxProjectedBytes, 4 * 2048 * Float32Array.BYTES_PER_ELEMENT);
  });
  const selected = selectTrainingWorkingSet(samples, device, engine);

  assert.equal(selected.length, 4);
  assert.equal(selected.filter((entry) => Number.isInteger(entry.policy)).length, 1);
  assert.ok(selected.some((entry) => entry.positionKey === "policy-search"));
  assert.equal(selected.filter((entry) => entry.labelKind === "outcome").length, 3);
  assert.equal(engine.calls, 1);
});

function workingSetEngine(indexes, inspect) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  let nextPtr = 1024;
  let outputLen = 0;
  let lastMessage = "";
  const outputPtr = 32768;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8");
  const engine = {
    memory,
    calls: 0,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      const bytes = encoder.encode(lastMessage);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_select_training_working_set_indexes_bytes(ptr, length, maxProjectedBytes) {
      engine.calls += 1;
      const text = decoder.decode(new Uint8Array(memory.buffer, ptr, length));
      inspect?.({ samples: JSON.parse(text), maxProjectedBytes });
      const bytes = new Uint8Array(indexes.length * Int32Array.BYTES_PER_ELEMENT);
      const view = new DataView(bytes.buffer);
      indexes.forEach((index, offset) => view.setInt32(offset * Int32Array.BYTES_PER_ELEMENT, index, true));
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    }
  };
  return engine;
}

function validationSplitEngine(split, inspect) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  let nextPtr = 1024;
  let outputLen = 0;
  let lastMessage = "";
  const outputPtr = 32768;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8");
  const engine = {
    memory,
    calls: 0,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      const bytes = encoder.encode(lastMessage);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_split_validation_samples_json(ptr, length, validationSplit) {
      engine.calls += 1;
      const text = decoder.decode(new Uint8Array(memory.buffer, ptr, length));
      inspect?.({ samples: JSON.parse(text), validationSplit });
      const bytes = encoder.encode(JSON.stringify(split));
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    }
  };
  return engine;
}

function policySplitEngine(split, inspect) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  let nextPtr = 1024;
  let outputLen = 0;
  let lastMessage = "";
  const outputPtr = 32768;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8");
  const engine = {
    memory,
    calls: 0,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      const bytes = encoder.encode(lastMessage);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_split_policy_training_indices_json(ptr, length, validationSplit) {
      engine.calls += 1;
      const text = decoder.decode(new Uint8Array(memory.buffer, ptr, length));
      inspect?.({ request: JSON.parse(text), validationSplit });
      const bytes = encoder.encode(JSON.stringify(split));
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    }
  };
  return engine;
}

function groupIndicesEngine(groups, inspect) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  let nextPtr = 1024;
  let outputLen = 0;
  let lastMessage = "";
  const outputPtr = 32768;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8");
  const engine = {
    memory,
    calls: 0,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      const bytes = encoder.encode(lastMessage);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_group_training_indices_by_position_json(ptr, length) {
      engine.calls += 1;
      const text = decoder.decode(new Uint8Array(memory.buffer, ptr, length));
      inspect?.({ request: JSON.parse(text) });
      const bytes = encoder.encode(JSON.stringify(groups));
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    }
  };
  return engine;
}

function groupedBatchFillEngine(batch, batchWeight, inspect) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  let nextPtr = 1024;
  let outputLen = 0;
  let lastMessage = "";
  const outputPtr = 32768;
  const encoder = new TextEncoder();
  const engine = {
    memory,
    calls: 0,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      const bytes = encoder.encode(lastMessage);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_fill_grouped_training_batch_indices_bytes(ptr, length) {
      engine.calls += 1;
      const view = new DataView(memory.buffer, ptr, length);
      inspect?.({
        request: {
          batchLength: view.getUint32(0, true),
          groupCount: view.getUint32(4, true),
          itemCount: view.getUint32(8, true),
          labelWeightCount: view.getUint32(12, true),
          epoch: view.getUint32(16, true),
          seed: view.getUint32(20, true)
        }
      });
      const bytes = new Uint8Array(8 + batch.length * Uint32Array.BYTES_PER_ELEMENT);
      const output = new DataView(bytes.buffer);
      output.setFloat32(0, batchWeight, true);
      output.setUint32(4, batch.length, true);
      batch.forEach((value, index) => output.setUint32(8 + index * 4, value, true));
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    }
  };
  return engine;
}

function policyIndexEngine(indices, inspect) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  let nextPtr = 1024;
  let outputLen = 0;
  let lastMessage = "";
  const outputPtr = 32768;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8");
  const engine = {
    memory,
    calls: 0,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      const bytes = encoder.encode(lastMessage);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_policy_training_indices_bytes(ptr, length, requirePositiveWeight) {
      engine.calls += 1;
      const text = decoder.decode(new Uint8Array(memory.buffer, ptr, length));
      inspect?.({ inputSamples: JSON.parse(text), requirePositiveWeight });
      const bytes = new Uint8Array(indices.length * Int32Array.BYTES_PER_ELEMENT);
      const view = new DataView(bytes.buffer);
      indices.forEach((index, offset) => view.setInt32(offset * Int32Array.BYTES_PER_ELEMENT, index, true));
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    }
  };
  return engine;
}

function uniqueCountEngine(count, inspect) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  let nextPtr = 1024;
  let outputLen = 0;
  let lastMessage = "";
  const outputPtr = 32768;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8");
  const engine = {
    memory,
    calls: 0,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      const bytes = encoder.encode(lastMessage);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_unique_training_position_count_json(ptr, length) {
      engine.calls += 1;
      const text = decoder.decode(new Uint8Array(memory.buffer, ptr, length));
      inspect?.({ request: JSON.parse(text) });
      const bytes = encoder.encode(String(count));
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    }
  };
  return engine;
}

function featureLengthEngine(length, inspect) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  let nextPtr = 1024;
  let outputLen = 0;
  let lastMessage = "";
  const outputPtr = 32768;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8");
  const engine = {
    memory,
    calls: 0,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(byteLength) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, byteLength);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      const bytes = encoder.encode(lastMessage);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_feature_length_json(ptr, byteLength) {
      engine.calls += 1;
      const text = decoder.decode(new Uint8Array(memory.buffer, ptr, byteLength));
      inspect?.({ inputSamples: JSON.parse(text) });
      const bytes = encoder.encode(String(length));
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    }
  };
  return engine;
}

function sparseProjectionEngine(packed, inspect) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  let nextPtr = 1024;
  let outputLen = 0;
  let lastMessage = "";
  const outputPtr = 32768;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8");
  const engine = {
    memory,
    calls: 0,
    chronofish_output_len() {
      return outputLen;
    },
    chronofish_alloc(length) {
      const ptr = nextPtr;
      nextPtr += Math.max(1, length);
      return ptr;
    },
    chronofish_dealloc() {},
    chronofish_last_message() {
      const bytes = encoder.encode(lastMessage);
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    },
    chronofish_sparse_projection_features_bytes(ptr, length, inputSize) {
      engine.calls += 1;
      const text = decoder.decode(new Uint8Array(memory.buffer, ptr, length));
      inspect?.({ inputSamples: JSON.parse(text), inputSize });
      const byteLength = 16 + (packed.offsets.length + packed.indices.length + packed.values.length) * Uint32Array.BYTES_PER_ELEMENT;
      const bytes = new Uint8Array(byteLength);
      const view = new DataView(bytes.buffer);
      view.setUint32(0, packed.offsets.length, true);
      view.setUint32(4, packed.indices.length, true);
      view.setUint32(8, packed.values.length, true);
      view.setUint32(12, packed.byteLength, true);
      let cursor = 16;
      for (const value of packed.offsets) {
        view.setUint32(cursor, value, true);
        cursor += 4;
      }
      for (const value of packed.indices) {
        view.setUint32(cursor, value, true);
        cursor += 4;
      }
      for (const value of packed.values) {
        view.setFloat32(cursor, value, true);
        cursor += 4;
      }
      new Uint8Array(memory.buffer, outputPtr, bytes.length).set(bytes);
      outputLen = bytes.length;
      return outputPtr;
    }
  };
  return engine;
}

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
