import { decodeCompactModel, modelArchitectureMatches } from "./training-gpu.js";
import type { CompactValueModel } from "./training-gpu.js";
import { FRONTIER_NEURAL_SHADER, FRONTIER_POLICY_SHADER } from "./training-shaders.js";
import frontierForward from "./shaders/frontier_forward.wgsl";

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

interface ModelBuffers {
  model: CompactValueModel;
  hiddenWeights: GPUBuffer[];
  outputWeights: GPUBuffer;
  policyWeights: GPUBuffer | null;
}

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
  #pipelines: Promise<Pipelines> | null = null;
  #workspace: NeuralWorkspace | null = null;
  #currentPolicyFeatures: { buffer: GPUBuffer; capacity: number; count: number } | null = null;
  #nextPolicyFeatures: { buffer: GPUBuffer; capacity: number; count: number } | null = null;
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
    const effectiveBatchSize = Math.max(1, Math.min(stateCount, Math.floor(batchSize)));
    const workspace = this.workspace(effectiveBatchSize, modelBuffers.model);
    const nextPolicyFeatures = modelBuffers.policyWeights
      ? this.policyFeatureBuffer("next", stateCount, outputLayerSize(modelBuffers.model))
      : null;
    if (nextPolicyFeatures) {
      encoder.clearBuffer(nextPolicyFeatures);
    }
    for (let stateOffset = 0; stateOffset < stateCount; stateOffset += effectiveBatchSize) {
      const batchCount = Math.min(effectiveBatchSize, stateCount - stateOffset);
      const common = this.keep(uniformU32(this.device, [
        batchCount,
        stateStride,
        boardOffset,
        maxBoards,
        stateOffset,
        modelBuffers.model.projectionSize,
        modelBuffers.model.projectionSeed,
        targetDepth
      ]));
      const apply = this.keep(uniformMixed(this.device, batchCount, rootColor, modelBuffers.model.scale ?? 1, modelBuffers.model.bias ?? 0, stateOffset));

      encodeBindings(this.device, encoder, pipelines.selectBoards, [
        [0, states], [1, workspace.selectedBoards], [5, common], [7, workspace.activeStates]
      ], Math.ceil(batchCount * 16 / 64));
      encodeBindings(this.device, encoder, pipelines.project, [
        [0, states], [1, workspace.selectedBoards], [2, workspace.projected], [5, common], [7, workspace.activeStates]
      ], Math.ceil(batchCount / 16), Math.ceil(modelBuffers.model.projectionSize / 16));

      for (let layer = 0; layer < modelBuffers.model.hiddenLayers.length; layer += 1) {
        const inputSize = layer === 0 ? modelBuffers.model.projectionSize : modelBuffers.model.hiddenLayers[layer - 1]!;
        const outputSize = modelBuffers.model.hiddenLayers[layer]!;
        const layerParams = this.keep(uniformU32(this.device, [batchCount, inputSize, outputSize, 0]));
        encodeBindings(this.device, encoder, pipelines.forwardLayer, [
          [0, layer === 0 ? workspace.projected : workspace.activations[layer - 1]!],
          [1, modelBuffers.hiddenWeights[layer]!],
          [2, workspace.activations[layer]!],
          [3, layerParams],
          [4, workspace.activeStates]
        ], Math.ceil(batchCount / 16), Math.ceil(outputSize / 16));
      }
      if (nextPolicyFeatures) {
        encoder.copyBufferToBuffer(
          workspace.activations.at(-1)!,
          0,
          nextPolicyFeatures,
          stateOffset * outputLayerSize(modelBuffers.model) * Float32Array.BYTES_PER_ELEMENT,
          batchCount * outputLayerSize(modelBuffers.model) * Float32Array.BYTES_PER_ELEMENT
        );
      }

      const finalSize = modelBuffers.model.hiddenLayers.at(-1)!;
      const outputParams = this.keep(uniformU32(this.device, [batchCount, finalSize, 0, 0]));
      encodeBindings(this.device, encoder, modelBuffers.model.outputActivation === "tanh"
        ? pipelines.forwardOutput
        : pipelines.forwardOutputLinear, [
        [0, workspace.activations.at(-1)!],
        [1, modelBuffers.outputWeights],
        [2, workspace.predictions],
        [3, outputParams],
        [4, workspace.activeStates]
      ], Math.ceil(batchCount / 64));
      encodeBindings(this.device, encoder, pipelines.applyValues, [
        [0, states], [3, workspace.predictions], [4, summaries], [5, common], [6, apply], [7, workspace.activeStates]
      ], Math.ceil(batchCount / 64));
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
    if (!modelBuffers?.policyWeights) {
      return false;
    }
    const pipelines = await this.pipelines();
    const inputSize = outputLayerSize(modelBuffers.model);
    if (!this.#currentPolicyFeatures || this.#currentPolicyFeatures.count < stateCount) {
      const current = this.policyFeatureBuffer("current", stateCount, inputSize);
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
    }
    const params = this.keep(policyParams(this.device, candidateCount, candidateStride, inputSize, 25));
    encodeBindings(this.device, encoder, pipelines.applyPolicy, [
      [0, candidates],
      [1, this.#currentPolicyFeatures!.buffer],
      [2, modelBuffers.policyWeights],
      [3, params]
    ], Math.ceil(candidateCount / 64));
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
      buffers?.hiddenWeights.forEach((buffer) => buffer.destroy());
      buffers?.outputWeights.destroy();
      buffers?.policyWeights?.destroy();
    });
    this.#model = null;
    this.#pipelines = null;
    this.destroyPolicyFeatures();
  }

  releaseTemporaries(): void {
    this.#temporaries.forEach((buffer) => buffer.destroy());
    this.#temporaries = [];
  }

  private model(): Promise<ModelBuffers | null> {
    this.#model ??= loadModel(this.device);
    return this.#model;
  }

  private pipelines(): Promise<Pipelines> {
    this.#pipelines ??= createPipelines(this.device);
    return this.#pipelines;
  }

  private workspace(stateCount: number, model: CompactValueModel): NeuralWorkspace {
    if (this.#workspace && this.#workspace.capacity >= stateCount) {
      return this.#workspace;
    }
    if (this.#workspace) {
      destroyWorkspace(this.#workspace);
    }
    const capacity = stateCount;
    this.#workspace = {
      capacity,
      selectedBoards: gpuBuffer(this.device, capacity * 16 * 4),
      activeStates: gpuBuffer(this.device, capacity * 4),
      projected: gpuBuffer(this.device, capacity * model.projectionSize * 4),
      activations: model.hiddenLayers.map((size) =>
        gpuBuffer(this.device, capacity * size * 4, usage.COPY_SRC)
      ),
      predictions: gpuBuffer(this.device, capacity * 4)
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
    const effectiveBatchSize = Math.max(1, Math.min(stateCount, Math.floor(batchSize)));
    const workspace = this.workspace(effectiveBatchSize, modelBuffers.model);
    const inputSize = outputLayerSize(modelBuffers.model);
    for (let stateOffset = 0; stateOffset < stateCount; stateOffset += effectiveBatchSize) {
      const batchCount = Math.min(effectiveBatchSize, stateCount - stateOffset);
      const common = this.keep(uniformU32(this.device, [
        batchCount,
        stateStride,
        boardOffset,
        maxBoards,
        stateOffset,
        modelBuffers.model.projectionSize,
        modelBuffers.model.projectionSeed,
        targetDepth
      ]));
      encodeBindings(this.device, encoder, pipelines.selectBoards, [
        [0, states], [1, workspace.selectedBoards], [5, common], [7, workspace.activeStates]
      ], Math.ceil(batchCount * 16 / 64));
      encodeBindings(this.device, encoder, pipelines.project, [
        [0, states], [1, workspace.selectedBoards], [2, workspace.projected], [5, common], [7, workspace.activeStates]
      ], Math.ceil(batchCount / 16), Math.ceil(modelBuffers.model.projectionSize / 16));
      for (let layer = 0; layer < modelBuffers.model.hiddenLayers.length; layer += 1) {
        const layerInputSize = layer === 0
          ? modelBuffers.model.projectionSize
          : modelBuffers.model.hiddenLayers[layer - 1]!;
        const outputSize = modelBuffers.model.hiddenLayers[layer]!;
        const layerParams = this.keep(uniformU32(this.device, [batchCount, layerInputSize, outputSize, 0]));
        encodeBindings(this.device, encoder, pipelines.forwardLayer, [
          [0, layer === 0 ? workspace.projected : workspace.activations[layer - 1]!],
          [1, modelBuffers.hiddenWeights[layer]!],
          [2, workspace.activations[layer]!],
          [3, layerParams],
          [4, workspace.activeStates]
        ], Math.ceil(batchCount / 16), Math.ceil(outputSize / 16));
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
    inputSize: number
  ): GPUBuffer {
    const field = target === "current" ? this.#currentPolicyFeatures : this.#nextPolicyFeatures;
    if (field && field.capacity >= stateCount) {
      field.count = stateCount;
      return field.buffer;
    }
    field?.buffer.destroy();
    const created = {
      buffer: gpuBuffer(this.device, stateCount * inputSize * Float32Array.BYTES_PER_ELEMENT),
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

function modelBuffersFromBytes(device: GPUDevice, bytes: ArrayBuffer): ModelBuffers | null {
  const model = decodeCompactModel(bytes);
  if (!modelArchitectureMatches(model)) {
    return null;
  }
  const hiddenWeights = splitHiddenWeights(model).map((weights) => initializedBuffer(device, weights));
  const policyWeights = policyWeightsForModel(model);
  return {
    model,
    hiddenWeights,
    outputWeights: initializedBuffer(device, model.outputWeights),
    policyWeights: policyWeights ? initializedBuffer(device, policyWeights) : null
  };
}

function outputLayerSize(model: CompactValueModel): number {
  const size = model.hiddenLayers.at(-1);
  if (!size) {
    throw new Error("GPU value model has no hidden output layer.");
  }
  return size;
}

function policyWeightsForModel(model: CompactValueModel): Float32Array | null {
  const inputSize = outputLayerSize(model);
  const expected = 257 * (inputSize + 1);
  if (model.policyWeights?.length === expected) {
    return model.policyWeights;
  }
  if (model.policyLogits?.length !== 257) {
    return null;
  }
  const weights = new Float32Array(expected);
  for (let bucket = 0; bucket < 257; bucket += 1) {
    weights[bucket * (inputSize + 1) + inputSize] = model.policyLogits[bucket] ?? 0;
  }
  return weights;
}

function splitHiddenWeights(model: CompactValueModel): Float32Array[] {
  const layers: Float32Array[] = [];
  let cursor = 0;
  let inputSize = model.projectionSize;
  for (const outputSize of model.hiddenLayers) {
    const count = outputSize * (inputSize + 1);
    layers.push(model.hiddenWeights.slice(cursor, cursor + count));
    cursor += count;
    inputSize = outputSize;
  }
  if (cursor !== model.hiddenWeights.length) {
    throw new Error("GPU value model hidden-weight layout is incompatible with search.");
  }
  return layers;
}

async function createPipelines(device: GPUDevice): Promise<Pipelines> {
  const [selectBoards, project, forwardLayer, forwardOutput, forwardOutputLinear, applyValues, applyPolicy] = await Promise.all([
    pipeline(device, "frontier_select_neural_boards", FRONTIER_NEURAL_SHADER, "select_neural_boards"),
    pipeline(device, "frontier_project", FRONTIER_NEURAL_SHADER, "project_neural_features"),
    pipeline(device, "frontier_forward_layer", frontierForward, "forward_layer_masked"),
    pipeline(device, "frontier_forward_output", frontierForward, "forward_output_masked"),
    pipeline(device, "frontier_forward_output_linear", frontierForward, "forward_output_masked_linear"),
    pipeline(device, "frontier_apply_neural", FRONTIER_NEURAL_SHADER, "apply_neural_values"),
    pipeline(device, "frontier_apply_policy", FRONTIER_POLICY_SHADER, "apply_policy_prior")
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

function gpuBuffer(device: GPUDevice, byteLength: number, extraUsage = 0): GPUBuffer {
  return device.createBuffer({
    size: align4(byteLength),
    usage: usage.STORAGE | usage.COPY_DST | extraUsage
  });
}

function initializedBuffer(device: GPUDevice, values: Float32Array): GPUBuffer {
  const buffer = gpuBuffer(device, values.byteLength);
  device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

function uniformU32(device: GPUDevice, values: number[]): GPUBuffer {
  const data = new Uint32Array(values.map((value) => value >>> 0));
  const buffer = device.createBuffer({ size: align4(data.byteLength), usage: usage.COPY_DST | usage.UNIFORM });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function uniformMixed(device: GPUDevice, stateCount: number, rootColor: number, scale: number, bias: number, stateOffset: number): GPUBuffer {
  const data = new ArrayBuffer(32);
  const view = new DataView(data);
  view.setUint32(0, stateCount, true);
  view.setInt32(4, rootColor, true);
  view.setFloat32(8, scale, true);
  view.setFloat32(12, bias, true);
  view.setUint32(16, stateOffset, true);
  const buffer = device.createBuffer({ size: 32, usage: usage.COPY_DST | usage.UNIFORM });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function policyParams(
  device: GPUDevice,
  candidateCount: number,
  candidateStride: number,
  inputSize: number,
  scale: number
): GPUBuffer {
  const data = new ArrayBuffer(16);
  const view = new DataView(data);
  view.setUint32(0, candidateCount, true);
  view.setUint32(4, candidateStride, true);
  view.setUint32(8, inputSize, true);
  view.setFloat32(12, scale, true);
  const buffer = device.createBuffer({ size: 16, usage: usage.COPY_DST | usage.UNIFORM });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function destroyWorkspace(workspace: NeuralWorkspace): void {
  workspace.selectedBoards.destroy();
  workspace.activeStates.destroy();
  workspace.projected.destroy();
  workspace.activations.forEach((buffer) => buffer.destroy());
  workspace.predictions.destroy();
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}
