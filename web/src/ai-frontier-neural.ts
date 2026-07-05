import { decodeCompactFrontierModelLayoutWithEngine } from "./training-gpu.js";
import type { CompactValueModel } from "./training-gpu.js";
import { readWasmBytes, readWasmString, writeWasmBytes } from "./engine-io.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import type { ChronofishEngine } from "./types.js";
import { FRONTIER_NEURAL_SHADER, FRONTIER_POLICY_SHADER } from "./training-shaders.js";
import frontierForward from "../../engine/src/gpu/search/shaders/frontier_forward.wgsl";

interface BufferUsageConstants {
  COPY_SRC: number;
  COPY_DST: number;
  STORAGE: number;
  UNIFORM: number;
}

const usage: BufferUsageConstants = (globalThis as unknown as { GPUBufferUsage?: BufferUsageConstants }).GPUBufferUsage ?? {
  COPY_SRC: 4,
  COPY_DST: 8,
  STORAGE: 128,
  UNIFORM: 64
};

let modelEnginePromise: Promise<ChronofishEngine> | null = null;

interface ModelBuffers {
  engine: ChronofishEngine;
  model: CompactValueModel;
  inferencePrecision: InferencePrecision;
  sharedBoardEncoder: GPUBuffer[];
  fastNet: {
    policyWeights: GPUBuffer | null;
    inputSize: number;
    policyQuantization: PolicyQuantizationStats | null;
  };
  bigNet: {
    outputWeights: GPUBuffer;
    auxiliaryValueWeights: GPUBuffer | null;
  };
}

interface PolicyQuantizationStats {
  format: "int8-dequantized-upload" | "int8-to-fp16-upload";
  scale: number;
  maxAbsError: number;
}

type InferencePrecision = "fp32" | "fp16";

interface NeuralWorkspace {
  capacity: number;
  selectedBoards: GPUBuffer;
  activeStates: GPUBuffer;
  projected: GPUBuffer;
  activations: GPUBuffer[];
  predictions: GPUBuffer;
}

interface Pipelines {
  selectBoards: GPUComputePipeline;
  project: GPUComputePipeline;
  forwardLayer: GPUComputePipeline;
  forwardOutput: GPUComputePipeline;
  forwardOutputLinear: GPUComputePipeline;
  applyValues: GPUComputePipeline;
  applyPolicy: GPUComputePipeline;
}

export class FrontierNeuralEvaluator {
  readonly device: GPUDevice;
  #model: Promise<ModelBuffers | null> | null = null;
  #engine: ChronofishEngine | null = null;
  #pipelines: Promise<Pipelines> | null = null;
  #workspace: NeuralWorkspace | null = null;
  #currentPolicyFeatures: { buffer: GPUBuffer; capacity: number; count: number } | null = null;
  #nextPolicyFeatures: { buffer: GPUBuffer; capacity: number; count: number } | null = null;
  #cacheStats = { hits: 0, misses: 0, stores: 0 };
  #temporaries: GPUBuffer[] = [];

  constructor(device: GPUDevice, modelBytes: ArrayBuffer | null = null) {
    this.device = device;
    if (modelBytes) {
      this.#model = Promise.resolve(modelBuffersFromBytes(device, modelBytes));
    }
  }

  async available(): Promise<boolean> {
    return Boolean(await this.model());
  }

  beginSearch(): void {
    this.destroyPolicyFeatures();
    this.#cacheStats = { hits: 0, misses: 0, stores: 0 };
  }

  cacheStats(): { hits: number; misses: number; stores: number; hitRate: number } {
    return {
      ...this.#cacheStats,
      hitRate: frontierNeuralCacheHitRate(this.#cacheStats.hits, this.#cacheStats.misses, this.#engine ?? undefined)
    };
  }

  async quantizationStats(): Promise<{ inferencePrecision: InferencePrecision | null; fastNetPolicy: PolicyQuantizationStats | null }> {
    const modelBuffers = await this.model();
    return {
      inferencePrecision: modelBuffers?.inferencePrecision ?? null,
      fastNetPolicy: modelBuffers?.fastNet.policyQuantization ?? null
    };
  }

  networkRoles(): { fastNet: string; bigNet: string } {
    return {
      fastNet: "policy-prior-candidate-pruning",
      bigNet: "retained-state-value-and-auxiliary-heads"
    };
  }

  async encode(
    encoder: GPUCommandEncoder,
    states: GPUBuffer,
    summaries: GPUBuffer,
    stateCount: number,
    stateStride: number,
    boardOffset: number,
    maxBoards: number,
    rootColor: number,
    batchSize: number,
    targetDepth: number
  ): Promise<boolean> {
    const modelBuffers = await this.model();
    if (!modelBuffers) {
      return false;
    }
    const pipelines = await this.pipelines();
    const effectiveBatchSize = frontierNeuralEffectiveBatchSize(modelBuffers.engine, stateCount, batchSize);
    const workspace = this.workspace(effectiveBatchSize, modelBuffers.model, modelBuffers.engine);
    const nextPolicyFeatures = modelBuffers.fastNet.policyWeights
      ? this.policyFeatureBuffer("next", stateCount, modelBuffers.fastNet.inputSize, modelBuffers.engine)
      : null;
    if (nextPolicyFeatures) {
      encoder.clearBuffer(nextPolicyFeatures);
    }
    for (let stateOffset = 0; stateOffset < stateCount; stateOffset += effectiveBatchSize) {
      const batchCount = frontierNeuralBatchCount(modelBuffers.engine, stateCount, stateOffset, effectiveBatchSize);
      const common = this.keep(frontierNeuralParams(this.device, modelBuffers.engine,
        batchCount,
        stateStride,
        boardOffset,
        maxBoards,
        stateOffset,
        modelBuffers.model.projectionSize,
        modelBuffers.model.projectionSeed,
        targetDepth
      ));
      const apply = this.keep(frontierNeuralApplyParams(this.device, modelBuffers.engine, batchCount, rootColor, modelBuffers.model.scale ?? 1, modelBuffers.model.bias ?? 0, stateOffset));

      encodeBindings(this.device, encoder, pipelines.selectBoards, [
        [0, states], [1, workspace.selectedBoards], [5, common], [7, workspace.activeStates]
      ], frontierNeuralSelectBoardWorkgroups(modelBuffers.engine, batchCount));
      encodeBindings(this.device, encoder, pipelines.project, [
        [0, states], [1, workspace.selectedBoards], [2, workspace.projected], [5, common], [7, workspace.activeStates]
      ], frontierNeuralProjectWorkgroupsX(modelBuffers.engine, batchCount), frontierNeuralProjectWorkgroupsY(modelBuffers.engine, modelBuffers.model.projectionSize));

      for (let layer = 0; layer < modelBuffers.model.hiddenLayers.length; layer += 1) {
        const inputSize = layer === 0 ? modelBuffers.model.projectionSize : modelBuffers.model.hiddenLayers[layer - 1]!;
        const outputSize = modelBuffers.model.hiddenLayers[layer]!;
        const layerParams = this.keep(frontierNeuralLayerParams(this.device, modelBuffers.engine, batchCount, inputSize, outputSize));
        encodeBindings(this.device, encoder, pipelines.forwardLayer, [
          [0, layer === 0 ? workspace.projected : workspace.activations[layer - 1]!],
          [1, modelBuffers.sharedBoardEncoder[layer]!],
          [2, workspace.activations[layer]!],
          [3, layerParams],
          [4, workspace.activeStates]
        ], frontierNeuralLayerWorkgroupsX(modelBuffers.engine, batchCount), frontierNeuralLayerWorkgroupsY(modelBuffers.engine, outputSize));
      }
      if (nextPolicyFeatures) {
        encoder.copyBufferToBuffer(
          workspace.activations.at(-1)!,
          0,
          nextPolicyFeatures,
          stateOffset * modelBuffers.fastNet.inputSize * Float32Array.BYTES_PER_ELEMENT,
          batchCount * modelBuffers.fastNet.inputSize * Float32Array.BYTES_PER_ELEMENT
        );
        this.#cacheStats.stores += batchCount;
      }

      const finalSize = modelBuffers.model.hiddenLayers.at(-1)!;
      const outputParams = this.keep(frontierNeuralLayerParams(this.device, modelBuffers.engine, batchCount, finalSize, 0));
      encodeBindings(this.device, encoder, modelBuffers.model.outputActivation === "tanh"
        ? pipelines.forwardOutput
        : pipelines.forwardOutputLinear, [
        [0, workspace.activations.at(-1)!],
        [1, modelBuffers.bigNet.outputWeights],
        [2, workspace.predictions],
        [3, outputParams],
        [4, workspace.activeStates]
      ], frontierNeuralOutputWorkgroups(modelBuffers.engine, batchCount));
      encodeBindings(this.device, encoder, pipelines.applyValues, [
        [0, states], [3, workspace.predictions], [4, summaries], [5, common], [6, apply], [7, workspace.activeStates]
      ], frontierNeuralOutputWorkgroups(modelBuffers.engine, batchCount));
    }
    return true;
  }

  async encodePolicyPrior(
    encoder: GPUCommandEncoder,
    states: GPUBuffer,
    candidates: GPUBuffer,
    candidateCount: number,
    candidateStride: number,
    stateCount: number,
    stateStride: number,
    boardOffset: number,
    maxBoards: number,
    batchSize: number,
    targetDepth: number
  ): Promise<boolean> {
    const modelBuffers = await this.model();
    if (!modelBuffers?.fastNet.policyWeights) {
      return false;
    }
    const pipelines = await this.pipelines();
    const inputSize = modelBuffers.fastNet.inputSize;
    if (!this.#currentPolicyFeatures || this.#currentPolicyFeatures.count < stateCount) {
      this.#cacheStats.misses += stateCount;
      const current = this.policyFeatureBuffer("current", stateCount, inputSize, modelBuffers.engine);
      encoder.clearBuffer(current);
      this.encodeHiddenFeatures(
        encoder,
        states,
        current,
        stateCount,
        stateStride,
        boardOffset,
        maxBoards,
        batchSize,
        targetDepth,
        modelBuffers,
        pipelines
      );
    } else {
      this.#cacheStats.hits += stateCount;
    }
    const params = this.keep(policyParams(this.device, modelBuffers.engine, candidateCount, candidateStride, inputSize, 25));
    encodeBindings(this.device, encoder, pipelines.applyPolicy, [
      [0, candidates],
      [1, this.#currentPolicyFeatures!.buffer],
      [2, modelBuffers.fastNet.policyWeights],
      [3, params]
    ], frontierPolicyWorkgroups(modelBuffers.engine, candidateCount));
    return true;
  }

  advancePolicyFeatures(): void {
    this.#currentPolicyFeatures?.buffer.destroy();
    this.#currentPolicyFeatures = this.#nextPolicyFeatures;
    this.#nextPolicyFeatures = null;
  }

  destroy(): void {
    this.releaseTemporaries();
    if (this.#workspace) {
      destroyWorkspace(this.#workspace);
    }
    this.#workspace = null;
    void this.#model?.then((buffers) => {
      buffers?.sharedBoardEncoder.forEach((buffer) => buffer.destroy());
      buffers?.bigNet.outputWeights.destroy();
      buffers?.bigNet.auxiliaryValueWeights?.destroy();
      buffers?.fastNet.policyWeights?.destroy();
    });
    this.#model = null;
    this.#pipelines = null;
    this.destroyPolicyFeatures();
  }

  releaseTemporaries(): void {
    this.#temporaries.forEach((buffer) => buffer.destroy());
    this.#temporaries = [];
  }

  private async model(): Promise<ModelBuffers | null> {
    this.#model ??= loadModel(this.device);
    const model = await this.#model;
    this.#engine = model?.engine ?? null;
    return model;
  }

  private pipelines(): Promise<Pipelines> {
    this.#pipelines ??= createPipelines(this.device);
    return this.#pipelines;
  }

  private workspace(stateCount: number, model: CompactValueModel, engine: ChronofishEngine): NeuralWorkspace {
    if (this.#workspace && this.#workspace.capacity >= stateCount) {
      return this.#workspace;
    }
    if (this.#workspace) {
      destroyWorkspace(this.#workspace);
    }
    const capacity = stateCount;
    this.#workspace = {
      capacity,
      selectedBoards: gpuBuffer(this.device, capacity * 16 * 4, 0, engine),
      activeStates: gpuBuffer(this.device, capacity * 4, 0, engine),
      projected: gpuBuffer(this.device, capacity * model.projectionSize * 4, 0, engine),
      activations: model.hiddenLayers.map((size) =>
        gpuBuffer(this.device, capacity * size * 4, usage.COPY_SRC, engine)
      ),
      predictions: gpuBuffer(this.device, capacity * 4, 0, engine)
    };
    return this.#workspace;
  }

  private keep(buffer: GPUBuffer): GPUBuffer {
    this.#temporaries.push(buffer);
    return buffer;
  }

  private encodeHiddenFeatures(
    encoder: GPUCommandEncoder,
    states: GPUBuffer,
    destination: GPUBuffer,
    stateCount: number,
    stateStride: number,
    boardOffset: number,
    maxBoards: number,
    batchSize: number,
    targetDepth: number,
    modelBuffers: ModelBuffers,
    pipelines: Pipelines
  ): void {
    const effectiveBatchSize = frontierNeuralEffectiveBatchSize(modelBuffers.engine, stateCount, batchSize);
    const workspace = this.workspace(effectiveBatchSize, modelBuffers.model, modelBuffers.engine);
    const inputSize = modelBuffers.fastNet.inputSize;
    for (let stateOffset = 0; stateOffset < stateCount; stateOffset += effectiveBatchSize) {
      const batchCount = frontierNeuralBatchCount(modelBuffers.engine, stateCount, stateOffset, effectiveBatchSize);
      const common = this.keep(frontierNeuralParams(this.device, modelBuffers.engine,
        batchCount,
        stateStride,
        boardOffset,
        maxBoards,
        stateOffset,
        modelBuffers.model.projectionSize,
        modelBuffers.model.projectionSeed,
        targetDepth
      ));
      encodeBindings(this.device, encoder, pipelines.selectBoards, [
        [0, states], [1, workspace.selectedBoards], [5, common], [7, workspace.activeStates]
      ], frontierNeuralSelectBoardWorkgroups(modelBuffers.engine, batchCount));
      encodeBindings(this.device, encoder, pipelines.project, [
        [0, states], [1, workspace.selectedBoards], [2, workspace.projected], [5, common], [7, workspace.activeStates]
      ], frontierNeuralProjectWorkgroupsX(modelBuffers.engine, batchCount), frontierNeuralProjectWorkgroupsY(modelBuffers.engine, modelBuffers.model.projectionSize));
      for (let layer = 0; layer < modelBuffers.model.hiddenLayers.length; layer += 1) {
        const layerInputSize = layer === 0
          ? modelBuffers.model.projectionSize
          : modelBuffers.model.hiddenLayers[layer - 1]!;
        const outputSize = modelBuffers.model.hiddenLayers[layer]!;
        const layerParams = this.keep(frontierNeuralLayerParams(this.device, modelBuffers.engine, batchCount, layerInputSize, outputSize));
        encodeBindings(this.device, encoder, pipelines.forwardLayer, [
          [0, layer === 0 ? workspace.projected : workspace.activations[layer - 1]!],
          [1, modelBuffers.sharedBoardEncoder[layer]!],
          [2, workspace.activations[layer]!],
          [3, layerParams],
          [4, workspace.activeStates]
        ], frontierNeuralLayerWorkgroupsX(modelBuffers.engine, batchCount), frontierNeuralLayerWorkgroupsY(modelBuffers.engine, outputSize));
      }
      encoder.copyBufferToBuffer(
        workspace.activations.at(-1)!,
        0,
        destination,
        stateOffset * inputSize * Float32Array.BYTES_PER_ELEMENT,
        batchCount * inputSize * Float32Array.BYTES_PER_ELEMENT
      );
    }
  }

  private policyFeatureBuffer(
    target: "current" | "next",
    stateCount: number,
    inputSize: number,
    engine: ChronofishEngine
  ): GPUBuffer {
    const field = target === "current" ? this.#currentPolicyFeatures : this.#nextPolicyFeatures;
    if (field && field.capacity >= stateCount) {
      field.count = stateCount;
      return field.buffer;
    }
    field?.buffer.destroy();
    const created = {
      buffer: gpuBuffer(this.device, stateCount * inputSize * Float32Array.BYTES_PER_ELEMENT, 0, engine),
      capacity: stateCount,
      count: stateCount
    };
    if (target === "current") {
      this.#currentPolicyFeatures = created;
    } else {
      this.#nextPolicyFeatures = created;
    }
    return created.buffer;
  }

  private destroyPolicyFeatures(): void {
    this.#currentPolicyFeatures?.buffer.destroy();
    this.#nextPolicyFeatures?.buffer.destroy();
    this.#currentPolicyFeatures = null;
    this.#nextPolicyFeatures = null;
  }
}

async function loadModel(device: GPUDevice): Promise<ModelBuffers | null> {
  try {
    const response = await fetch("/ai/value-model.cfnn", { cache: "no-cache" });
    if (!response.ok) {
      return null;
    }
    return modelBuffersFromBytes(device, await response.arrayBuffer());
  } catch (error) {
    console.warn("GPU value model is unavailable; using heuristic frontier scores.", error);
    return null;
  }
}

async function modelBuffersFromBytes(device: GPUDevice, bytes: ArrayBuffer): Promise<ModelBuffers | null> {
  const engine = await compactModelEngine();
  const layout = decodeCompactFrontierModelLayoutWithEngine(engine, bytes);
  if (!layout) {
    return null;
  }
  const model = layout.model;
  const inferencePrecision = device.features?.has("shader-f16" as GPUFeatureName) ? "fp16" : "fp32";
  const sharedBoardEncoder = await Promise.all(
    layout.hiddenLayerWeights.map((weights) => initializedWeightBuffer(device, weights, inferencePrecision, engine))
  );
  const policyWeights = layout.policyWeights;
  const policyQuantization = policyWeights ? quantizePolicyWeights(policyWeights, engine) : null;
  return {
    engine,
    model,
    inferencePrecision,
    sharedBoardEncoder,
    fastNet: {
      policyWeights: policyQuantization ? await initializedWeightBuffer(device, policyQuantization.dequantized, inferencePrecision, engine) : null,
      inputSize: layout.outputLayerSize,
      policyQuantization: policyQuantization
        ? {
          format: inferencePrecision === "fp16" ? "int8-to-fp16-upload" : "int8-dequantized-upload",
          scale: policyQuantization.scale,
          maxAbsError: policyQuantization.maxAbsError
        }
        : null
    },
    bigNet: {
      outputWeights: await initializedWeightBuffer(device, model.outputWeights, inferencePrecision, engine),
      auxiliaryValueWeights: model.auxiliaryValueWeights?.length
        ? await initializedWeightBuffer(device, model.auxiliaryValueWeights, inferencePrecision, engine)
        : null
    }
  };
}

function compactModelEngine(): Promise<ChronofishEngine> {
  modelEnginePromise ??= instantiateChronofishWasm("./chronofish_engine.wasm")
    .then((instance) => instance.exports as unknown as ChronofishEngine);
  return modelEnginePromise;
}

function quantizePolicyWeights(weights: Float32Array, engine: ChronofishEngine): {
  dequantized: Float32Array;
  scale: number;
  maxAbsError: number;
} {
  const input = writeWasmBytes(engine, new Uint8Array(weights.buffer, weights.byteOffset, weights.byteLength));
  try {
    const output = engine.chronofish_quantized_policy_upload_bytes(input.ptr, input.len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return readQuantizedPolicyUpload(readWasmBytes(engine, output), weights.length);
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

function readQuantizedPolicyUpload(bytes: Uint8Array, weightCount: number): {
  dequantized: Float32Array;
  scale: number;
  maxAbsError: number;
} {
  const expectedLength = 8 + weightCount * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(`Quantized policy upload has ${bytes.byteLength} bytes but expected ${expectedLength}.`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const scale = view.getFloat32(0, true);
  const maxAbsError = view.getFloat32(4, true);
  const dequantized = new Float32Array(weightCount);
  let cursor = 8;
  for (let index = 0; index < weightCount; index += 1) {
    dequantized[index] = view.getFloat32(cursor, true);
    cursor += 4;
  }
  return { dequantized, scale, maxAbsError };
}

async function createPipelines(device: GPUDevice): Promise<Pipelines> {
  const f16 = device.features?.has("shader-f16" as GPUFeatureName) ?? false;
  const forwardShader = f16 ? frontierForwardF16 : frontierForward;
  const policyShader = f16 ? frontierPolicyF16 : FRONTIER_POLICY_SHADER;
  const suffix = f16 ? "_f16" : "";
  const [selectBoards, project, forwardLayer, forwardOutput, forwardOutputLinear, applyValues, applyPolicy] = await Promise.all([
    pipeline(device, "frontier_select_neural_boards", FRONTIER_NEURAL_SHADER, "select_neural_boards"),
    pipeline(device, "frontier_project", FRONTIER_NEURAL_SHADER, "project_neural_features"),
    pipeline(device, `frontier_forward_layer${suffix}`, forwardShader, "forward_layer_masked"),
    pipeline(device, `frontier_forward_output${suffix}`, forwardShader, "forward_output_masked"),
    pipeline(device, `frontier_forward_output_linear${suffix}`, forwardShader, "forward_output_masked_linear"),
    pipeline(device, "frontier_apply_neural", FRONTIER_NEURAL_SHADER, "apply_neural_values"),
    pipeline(device, `frontier_apply_policy${suffix}`, policyShader, "apply_policy_prior")
  ]);
  return { selectBoards, project, forwardLayer, forwardOutput, forwardOutputLinear, applyValues, applyPolicy };
}

async function pipeline(device: GPUDevice, label: string, code: string, entryPoint: string): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label: `${label}.module`, code });
  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) {
      throw new Error(`${label} shader compilation failed: ${errors.map((error) => error.message).join("; ")}`);
    }
  }
  return device.createComputePipeline({ label, layout: "auto", compute: { module, entryPoint } });
}

function encodeBindings(device: GPUDevice, encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bindings: Array<[number, GPUBuffer]>, x: number, y = 1): void {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map(([binding, buffer]) => ({ binding, resource: { buffer } }))
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, x), Math.max(1, y));
  pass.end();
}

function gpuBuffer(device: GPUDevice, byteLength: number, extraUsage = 0, engine?: ChronofishEngine): GPUBuffer {
  return device.createBuffer({
    size: align4(byteLength, engine),
    usage: usage.STORAGE | usage.COPY_DST | extraUsage
  });
}

function initializedBuffer(device: GPUDevice, values: ArrayBufferView, engine?: ChronofishEngine): GPUBuffer {
  const buffer = gpuBuffer(device, values.byteLength, 0, engine);
  device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

async function initializedWeightBuffer(device: GPUDevice, values: Float32Array, precision: InferencePrecision, engine: ChronofishEngine): Promise<GPUBuffer> {
  return initializedBuffer(device, precision === "fp16" ? float32ToFloat16Array(values, engine) : values, engine);
}

function float32ToFloat16Array(values: Float32Array, engine: ChronofishEngine): Uint16Array {
  const input = writeWasmBytes(engine, new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  try {
    const output = engine.chronofish_f32_to_f16_upload_bytes(input.ptr, input.len);
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    const bytes = readWasmBytes(engine, output);
    if (bytes.byteLength !== values.length * Uint16Array.BYTES_PER_ELEMENT) {
      throw new Error("f16 upload response length does not match the request.");
    }
    return new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Uint16Array.BYTES_PER_ELEMENT).slice();
  } finally {
    engine.chronofish_dealloc(input.ptr, input.len);
  }
}

const frontierForwardF16 = `enable f16;
${frontierForward
  .replace("var<storage, read> weights: array<f32>;", "var<storage, read> weights: array<f16>;")
  .replace(/var sum = weights\[row \+ layer_params\.input_size\];/g, "var sum = f32(weights[row + layer_params.input_size]);")
  .replace(/var sum = weights\[layer_params\.input_size\];/g, "var sum = f32(weights[layer_params.input_size]);")
  .replace(/weights\[row \+ input_index\]/g, "f32(weights[row + input_index])")
  .replace(/weights\[input_index\]/g, "f32(weights[input_index])")}`;

const frontierPolicyF16 = `enable f16;
${FRONTIER_POLICY_SHADER
  .replace("var<storage, read> policy_weights: array<f32>;", "var<storage, read> policy_weights: array<f16>;")
  .replace(/var logit = policy_weights\[row \+ params\.input_size\];/g, "var logit = f32(policy_weights[row + params.input_size]);")
  .replace(/policy_weights\[row \+ input\]/g, "f32(policy_weights[row + input])")}`;

function uniformBytes(device: GPUDevice, bytes: Uint8Array, engine?: ChronofishEngine): GPUBuffer {
  const buffer = device.createBuffer({ size: align4(bytes.byteLength, engine), usage: usage.COPY_DST | usage.UNIFORM });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

function u32Bytes(values: number[]): Uint8Array {
  const data = new Uint32Array(values.map((value) => value >>> 0));
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

function frontierNeuralParams(
  device: GPUDevice,
  engine: ChronofishEngine | undefined,
  stateCount: number,
  stateStride: number,
  boardOffset: number,
  maxBoards: number,
  stateOffset: number,
  projectionSize: number,
  projectionSeed: number,
  targetDepth: number
): GPUBuffer {
  if (engine && typeof engine.chronofish_frontier_neural_params_bytes === "function") {
    return uniformBytes(device, readWasmBytes(engine, engine.chronofish_frontier_neural_params_bytes(
      stateCount,
      stateStride,
      boardOffset,
      maxBoards,
      stateOffset,
      projectionSize,
      projectionSeed,
      targetDepth
    )), engine);
  }
  return uniformBytes(device, u32Bytes([
    stateCount,
    stateStride,
    boardOffset,
    maxBoards,
    stateOffset,
    projectionSize,
    projectionSeed,
    targetDepth
  ]));
}

function frontierNeuralApplyParams(
  device: GPUDevice,
  engine: ChronofishEngine | undefined,
  stateCount: number,
  rootColor: number,
  scale: number,
  bias: number,
  stateOffset: number
): GPUBuffer {
  if (engine && typeof engine.chronofish_frontier_neural_apply_params_bytes === "function") {
    return uniformBytes(device, readWasmBytes(engine, engine.chronofish_frontier_neural_apply_params_bytes(
      stateCount,
      rootColor,
      scale,
      bias,
      stateOffset
    )), engine);
  }
  const data = new ArrayBuffer(32);
  const view = new DataView(data);
  view.setUint32(0, stateCount, true);
  view.setInt32(4, rootColor, true);
  view.setFloat32(8, scale, true);
  view.setFloat32(12, bias, true);
  view.setUint32(16, stateOffset, true);
  return uniformBytes(device, new Uint8Array(data));
}

function frontierNeuralLayerParams(device: GPUDevice, engine: ChronofishEngine | undefined, sampleCount: number, inputSize: number, outputSize: number): GPUBuffer {
  if (engine && typeof engine.chronofish_frontier_neural_layer_params_bytes === "function") {
    return uniformBytes(device, readWasmBytes(engine, engine.chronofish_frontier_neural_layer_params_bytes(sampleCount, inputSize, outputSize)), engine);
  }
  return uniformBytes(device, u32Bytes([sampleCount, inputSize, outputSize, 0]));
}

function frontierNeuralEffectiveBatchSize(engine: ChronofishEngine, stateCount: number, requestedBatchSize: number): number {
  return engine.chronofish_frontier_neural_effective_batch_size(stateCount, requestedBatchSize);
}

function frontierNeuralBatchCount(engine: ChronofishEngine, stateCount: number, stateOffset: number, effectiveBatchSize: number): number {
  return engine.chronofish_frontier_neural_batch_count(stateCount, stateOffset, effectiveBatchSize);
}

function frontierNeuralCacheHitRate(hits: number, misses: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_frontier_neural_cache_hit_rate(hits, misses);
  }
  const lookups = hits + misses;
  if (!Number.isFinite(hits) || !Number.isFinite(misses) || lookups <= 0) {
    return 0;
  }
  return Math.round((hits / lookups) * 1000) / 1000;
}

function frontierNeuralSelectBoardWorkgroups(engine: ChronofishEngine, batchCount: number): number {
  return engine.chronofish_frontier_neural_select_board_workgroups(batchCount);
}

function frontierNeuralProjectWorkgroupsX(engine: ChronofishEngine, batchCount: number): number {
  return engine.chronofish_frontier_neural_project_workgroups_x(batchCount);
}

function frontierNeuralProjectWorkgroupsY(engine: ChronofishEngine, projectionSize: number): number {
  return engine.chronofish_frontier_neural_project_workgroups_y(projectionSize);
}

function frontierNeuralLayerWorkgroupsX(engine: ChronofishEngine, batchCount: number): number {
  return engine.chronofish_frontier_neural_layer_workgroups_x(batchCount);
}

function frontierNeuralLayerWorkgroupsY(engine: ChronofishEngine, outputSize: number): number {
  return engine.chronofish_frontier_neural_layer_workgroups_y(outputSize);
}

function frontierNeuralOutputWorkgroups(engine: ChronofishEngine, batchCount: number): number {
  return engine.chronofish_frontier_neural_output_workgroups(batchCount);
}

function frontierPolicyWorkgroups(engine: ChronofishEngine, candidateCount: number): number {
  return engine.chronofish_frontier_policy_workgroups(candidateCount);
}

function policyParams(
  device: GPUDevice,
  engine: ChronofishEngine | undefined,
  candidateCount: number,
  candidateStride: number,
  inputSize: number,
  scale: number
): GPUBuffer {
  if (engine && typeof engine.chronofish_frontier_policy_params_bytes === "function") {
    return uniformBytes(device, readWasmBytes(engine, engine.chronofish_frontier_policy_params_bytes(candidateCount, candidateStride, inputSize, scale)), engine);
  }
  const data = new ArrayBuffer(16);
  const view = new DataView(data);
  view.setUint32(0, candidateCount, true);
  view.setUint32(4, candidateStride, true);
  view.setUint32(8, inputSize, true);
  view.setFloat32(12, scale, true);
  return uniformBytes(device, new Uint8Array(data));
}

function destroyWorkspace(workspace: NeuralWorkspace): void {
  workspace.selectedBoards.destroy();
  workspace.activeStates.destroy();
  workspace.projected.destroy();
  workspace.activations.forEach((buffer) => buffer.destroy());
  workspace.predictions.destroy();
}

function align4(value: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_align4(value);
  }
  return Math.ceil(value / 4) * 4;
}
