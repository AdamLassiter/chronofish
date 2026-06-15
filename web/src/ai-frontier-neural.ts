import { decodeCompactModel, modelArchitectureMatches } from "./training-gpu.js";
import type { CompactValueModel } from "./training-gpu.js";
import { FORWARD_LAYER_SHADER, FORWARD_OUTPUT_SHADER, FRONTIER_NEURAL_SHADER, PROJECT_FEATURES_SHADER } from "./training-shaders.js";

interface BufferUsageConstants {
  COPY_DST: number;
  STORAGE: number;
  UNIFORM: number;
}

const usage: BufferUsageConstants = (globalThis as unknown as { GPUBufferUsage?: BufferUsageConstants }).GPUBufferUsage ?? {
  COPY_DST: 8,
  STORAGE: 128,
  UNIFORM: 64
};

const NEURAL_INPUT_SIZE = 16 * 32 * 64;

interface ModelBuffers {
  model: CompactValueModel;
  hiddenWeights: GPUBuffer[];
  outputWeights: GPUBuffer;
}

interface NeuralWorkspace {
  capacity: number;
  selectedBoards: GPUBuffer;
  rawFeatures: GPUBuffer;
  projected: GPUBuffer;
  activations: GPUBuffer[];
  predictions: GPUBuffer;
}

interface Pipelines {
  selectBoards: GPUComputePipeline;
  encodeFeatures: GPUComputePipeline;
  project: GPUComputePipeline;
  forwardLayer: GPUComputePipeline;
  forwardOutput: GPUComputePipeline;
  applyValues: GPUComputePipeline;
}

export class FrontierNeuralEvaluator {
  readonly device: GPUDevice;
  #model: Promise<ModelBuffers | null> | null = null;
  #pipelines: Promise<Pipelines> | null = null;
  #workspace: NeuralWorkspace | null = null;
  #temporaries: GPUBuffer[] = [];

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async available(): Promise<boolean> {
    return Boolean(await this.model());
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
    batchSize: number
  ): Promise<boolean> {
    const modelBuffers = await this.model();
    if (!modelBuffers) {
      return false;
    }
    const pipelines = await this.pipelines();
    const effectiveBatchSize = Math.max(1, Math.min(stateCount, Math.floor(batchSize)));
    const workspace = this.workspace(effectiveBatchSize, modelBuffers.model);
    for (let stateOffset = 0; stateOffset < stateCount; stateOffset += effectiveBatchSize) {
      const batchCount = Math.min(effectiveBatchSize, stateCount - stateOffset);
      const common = this.keep(uniformU32(this.device, [batchCount, stateStride, boardOffset, maxBoards, stateOffset, 0, 0, 0]));
      const apply = this.keep(uniformMixed(this.device, batchCount, rootColor, modelBuffers.model.scale ?? 1, modelBuffers.model.bias ?? 0, stateOffset));

      encodeBindings(this.device, encoder, pipelines.selectBoards, [
        [0, states], [1, workspace.selectedBoards], [5, common]
      ], Math.ceil(batchCount * 16 / 64));
      encodeBindings(this.device, encoder, pipelines.encodeFeatures, [
        [0, states], [1, workspace.selectedBoards], [2, workspace.rawFeatures], [5, common]
      ], Math.ceil(batchCount * NEURAL_INPUT_SIZE / 256));

      const projectionParams = this.keep(uniformU32(this.device, [
        batchCount,
        NEURAL_INPUT_SIZE,
        modelBuffers.model.projectionSize,
        modelBuffers.model.projectionSeed
      ]));
      encodeBindings(this.device, encoder, pipelines.project, [
        [0, workspace.rawFeatures], [1, workspace.projected], [2, projectionParams]
      ], Math.ceil(batchCount / 16), Math.ceil(modelBuffers.model.projectionSize / 16));

      for (let layer = 0; layer < modelBuffers.model.hiddenLayers.length; layer += 1) {
        const inputSize = layer === 0 ? modelBuffers.model.projectionSize : modelBuffers.model.hiddenLayers[layer - 1]!;
        const outputSize = modelBuffers.model.hiddenLayers[layer]!;
        const layerParams = this.keep(uniformU32(this.device, [batchCount, inputSize, outputSize, 0]));
        encodeBindings(this.device, encoder, pipelines.forwardLayer, [
          [0, layer === 0 ? workspace.projected : workspace.activations[layer - 1]!],
          [1, modelBuffers.hiddenWeights[layer]!],
          [2, workspace.activations[layer]!],
          [3, layerParams]
        ], Math.ceil(batchCount / 16), Math.ceil(outputSize / 16));
      }

      const finalSize = modelBuffers.model.hiddenLayers.at(-1)!;
      const outputParams = this.keep(uniformU32(this.device, [batchCount, finalSize, 0, 0]));
      encodeBindings(this.device, encoder, pipelines.forwardOutput, [
        [0, workspace.activations.at(-1)!],
        [1, modelBuffers.outputWeights],
        [2, workspace.predictions],
        [3, outputParams]
      ], Math.ceil(batchCount / 64));
      encodeBindings(this.device, encoder, pipelines.applyValues, [
        [0, states], [3, workspace.predictions], [4, summaries], [5, common], [6, apply]
      ], Math.ceil(batchCount / 64));
    }
    return true;
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
    });
    this.#model = null;
    this.#pipelines = null;
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
      rawFeatures: gpuBuffer(this.device, capacity * NEURAL_INPUT_SIZE * 4),
      projected: gpuBuffer(this.device, capacity * model.projectionSize * 4),
      activations: model.hiddenLayers.map((size) => gpuBuffer(this.device, capacity * size * 4)),
      predictions: gpuBuffer(this.device, capacity * 4)
    };
    return this.#workspace;
  }

  private keep(buffer: GPUBuffer): GPUBuffer {
    this.#temporaries.push(buffer);
    return buffer;
  }
}

async function loadModel(device: GPUDevice): Promise<ModelBuffers | null> {
  try {
    const response = await fetch("/ai/value-model.cfnn", { cache: "no-cache" });
    if (!response.ok) {
      return null;
    }
    const model = decodeCompactModel(await response.arrayBuffer());
    if (!modelArchitectureMatches(model)) {
      return null;
    }
    const hiddenWeights = splitHiddenWeights(model).map((weights) => initializedBuffer(device, weights));
    return { model, hiddenWeights, outputWeights: initializedBuffer(device, model.outputWeights) };
  } catch (error) {
    console.warn("GPU value model is unavailable; using heuristic frontier scores.", error);
    return null;
  }
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
  const [selectBoards, encodeFeatures, project, forwardLayer, forwardOutput, applyValues] = await Promise.all([
    pipeline(device, "frontier_select_neural_boards", FRONTIER_NEURAL_SHADER, "select_neural_boards"),
    pipeline(device, "frontier_encode_neural", FRONTIER_NEURAL_SHADER, "encode_neural_features"),
    pipeline(device, "frontier_project", PROJECT_FEATURES_SHADER, "project_features"),
    pipeline(device, "frontier_forward_layer", FORWARD_LAYER_SHADER, "forward_layer"),
    pipeline(device, "frontier_forward_output", FORWARD_OUTPUT_SHADER, "forward_output"),
    pipeline(device, "frontier_apply_neural", FRONTIER_NEURAL_SHADER, "apply_neural_values")
  ]);
  return { selectBoards, encodeFeatures, project, forwardLayer, forwardOutput, applyValues };
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

function gpuBuffer(device: GPUDevice, byteLength: number): GPUBuffer {
  return device.createBuffer({ size: align4(byteLength), usage: usage.STORAGE | usage.COPY_DST });
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

function destroyWorkspace(workspace: NeuralWorkspace): void {
  workspace.selectedBoards.destroy();
  workspace.rawFeatures.destroy();
  workspace.projected.destroy();
  workspace.activations.forEach((buffer) => buffer.destroy());
  workspace.predictions.destroy();
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}
