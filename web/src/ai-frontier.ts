import {
  GPU_FRONTIER_BOARD_OFFSET,
  GPU_FRONTIER_BOARD_STRIDE,
  GPU_FRONTIER_CANDIDATE_STRIDE,
  GPU_FRONTIER_DELTA_STRIDE,
  GPU_FRONTIER_HEADER_PLAN_LENGTH,
  GPU_FRONTIER_HEADER_STRIDE,
  GPU_FRONTIER_ANCESTRY_STRIDE,
  GPU_FRONTIER_MAX_PLAN_MOVES,
  GPU_FRONTIER_PLAN_STRIDE,
  GPU_FRONTIER_PLAN_OFFSET,
  GPU_FRONTIER_SUMMARY_STRIDE
} from "./ai-layout.js";
import { GPU_FRONTIER_EXPAND_SHADER, GPU_FRONTIER_SELECT_SHADER, GPU_FRONTIER_STATE_SHADER } from "./ai-shaders.js";
import { readWasmString } from "./engine-io.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import type { ChronofishEngine } from "./types.js";

interface GpuBufferUsageConstants {
  MAP_READ: number;
  COPY_SRC: number;
  COPY_DST: number;
  QUERY_RESOLVE: number;
  UNIFORM: number;
  STORAGE: number;
  INDIRECT: number;
}

interface GpuMapModeConstants {
  READ: number;
}

const gpuBufferUsage: GpuBufferUsageConstants = (globalThis as unknown as { GPUBufferUsage?: GpuBufferUsageConstants }).GPUBufferUsage ?? {
  MAP_READ: 1,
  COPY_SRC: 4,
  COPY_DST: 8,
  QUERY_RESOLVE: 512,
  UNIFORM: 64,
  STORAGE: 128,
  INDIRECT: 256
};
const gpuMapMode: GpuMapModeConstants = (globalThis as unknown as { GPUMapMode?: GpuMapModeConstants }).GPUMapMode ?? {
  READ: 1
};

const I32_BYTES = Int32Array.BYTES_PER_ELEMENT;
const GPU_SHADER_STAGE_COMPUTE = 4;
const tuningCache = new Map<string, Promise<FrontierTuning>>();
let tuningEnginePromise: Promise<ChronofishEngine> | null = null;

export interface FrontierTuning {
  maxBoards: number;
  frontierWidth: number;
  candidateCapacity: number;
  neuralBatchSize: number;
  candidateWorkgroupSize: 32 | 64 | 128 | 256;
  mutationTileSize: 32 | 64 | 128;
  dispatchCandidateLimit: number;
}

interface FrontierSelectionPlan {
  candidateCapacity: number;
  selectionCapacity: number;
}

export interface FrontierBufferSet {
  states: GPUBuffer;
  nextStates: GPUBuffer;
  candidates: GPUBuffer;
  deltas: GPUBuffer;
  counters: GPUBuffer;
  order: GPUBuffer;
  eligibility: GPUBuffer;
  selected: GPUBuffer;
  summaries: GPUBuffer;
  indirect: GPUBuffer;
}

export type FrontierCandidateScorer = (
  encoder: GPUCommandEncoder,
  buffers: FrontierBufferSet,
  candidateCapacity: number
) => void | Promise<void>;

export interface EncodedFrontierRoot {
  words: Int32Array;
  boardCount: number;
  hashLow: number;
  hashHigh: number;
}

export interface FrontierPassOptions {
  rootColor: number;
  targetDepth: number;
  cycleIndex: number;
  stateCount?: number;
  perParentLimit?: number;
  maxSelectionScan?: number;
}

interface FrontierPipelines {
  expand: GPUComputePipeline;
  hash: GPUComputePipeline;
  bucketOrder: GPUComputePipeline;
  bitonicSort: GPUComputePipeline;
  markUnique: GPUComputePipeline;
  markParentQuota: GPUComputePipeline;
  compactSelected: GPUComputePipeline;
  select: GPUComputePipeline;
  materialize: GPUComputePipeline;
  reduce: GPUComputePipeline;
  copyReduced: GPUComputePipeline;
}

interface FrontierPipelineLayouts {
  expand: GPUBindGroupLayout;
  select: GPUBindGroupLayout;
  materialize: GPUBindGroupLayout;
  reduce: GPUBindGroupLayout;
}

interface PooledBuffer {
  buffer: GPUBuffer;
  size: number;
  usage: number;
  inUse: boolean;
}

export class FrontierBufferPool {
  readonly device: GPUDevice;
  readonly tuning: FrontierTuning;
  readonly engine: ChronofishEngine | undefined;
  #buffers: PooledBuffer[] = [];
  #destroyed = false;

  constructor(device: GPUDevice, tuning: FrontierTuning, engine?: ChronofishEngine) {
    this.device = device;
    this.tuning = tuning;
    this.engine = engine;
  }

  createSearchBuffers(): FrontierBufferSet {
    const stateBytes = frontierStateBytes(this.tuning.maxBoards, this.engine) * this.tuning.frontierWidth;
    const candidateBytes = GPU_FRONTIER_CANDIDATE_STRIDE * I32_BYTES * this.tuning.candidateCapacity;
    const deltaBytes = GPU_FRONTIER_DELTA_STRIDE * I32_BYTES * this.tuning.candidateCapacity;
    const storageCopy = gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC | gpuBufferUsage.COPY_DST;
    return {
      states: this.acquire(stateBytes, storageCopy),
      nextStates: this.acquire(stateBytes, storageCopy),
      candidates: this.acquire(candidateBytes, storageCopy),
      deltas: this.acquire(deltaBytes, storageCopy),
      counters: this.acquire(64, storageCopy),
      order: this.acquire(this.tuning.candidateCapacity * I32_BYTES, storageCopy),
      eligibility: this.acquire(this.tuning.candidateCapacity * I32_BYTES, storageCopy),
      selected: this.acquire(this.tuning.frontierWidth * I32_BYTES, storageCopy),
      summaries: this.acquire(this.tuning.frontierWidth * GPU_FRONTIER_SUMMARY_STRIDE * I32_BYTES, storageCopy),
      indirect: this.acquire(16, gpuBufferUsage.STORAGE | gpuBufferUsage.INDIRECT | gpuBufferUsage.COPY_DST)
    };
  }

  acquire(byteLength: number, usage: number): GPUBuffer {
    if (this.#destroyed) {
      throw new Error("GPU frontier buffer pool was destroyed.");
    }
    const size = align4(Math.max(4, byteLength), this.engine);
    const existing = this.#buffers.find((entry) => !entry.inUse && entry.usage === usage && entry.size >= size);
    if (existing) {
      existing.inUse = true;
      return existing.buffer;
    }
    const buffer = this.device.createBuffer({ size, usage });
    this.#buffers.push({ buffer, size, usage, inUse: true });
    return buffer;
  }

  release(buffer: GPUBuffer): void {
    const entry = this.#buffers.find((candidate) => candidate.buffer === buffer);
    if (entry) {
      entry.inUse = false;
    }
  }

  releaseSearchBuffers(buffers: FrontierBufferSet): void {
    for (const buffer of Object.values(buffers)) {
      this.release(buffer);
    }
  }

  destroy(): void {
    if (this.#destroyed) {
      return;
    }
    this.#destroyed = true;
    for (const entry of this.#buffers) {
      entry.buffer.destroy();
    }
    this.#buffers = [];
  }
}

export class FrontierGpuPipeline {
  readonly device: GPUDevice;
  readonly tuning: FrontierTuning;
  readonly pool: FrontierBufferPool;
  readonly engine: ChronofishEngine | undefined;
  #pipelines: Promise<FrontierPipelines> | null = null;
  #cycleTemporaries: GPUBuffer[] = [];

  constructor(device: GPUDevice, tuning: FrontierTuning, pool?: FrontierBufferPool, engine?: ChronofishEngine) {
    this.device = device;
    this.tuning = tuning;
    this.engine = engine ?? pool?.engine;
    this.pool = pool ?? new FrontierBufferPool(device, tuning, this.engine);
  }

  uploadRoot(buffers: FrontierBufferSet, root: EncodedFrontierRoot): void {
    this.device.queue.writeBuffer(buffers.states, 0, root.words);
  }

  async encodeExpansionCycle(
    encoder: GPUCommandEncoder,
    buffers: FrontierBufferSet,
    options: FrontierPassOptions,
    scoreCandidates?: FrontierCandidateScorer
  ): Promise<void> {
    const pipelines = await this.pipelines();
    const selectionPlan = await frontierSelectionPlan(this.tuning, options.maxSelectionScan);
    const { candidateCapacity, selectionCapacity } = selectionPlan;
    const stateStride = frontierStateStride(this.tuning.maxBoards, this.engine);
    const boardOffset = GPU_FRONTIER_BOARD_OFFSET;
    encoder.clearBuffer(buffers.counters, 0, 12);
    encoder.clearBuffer(buffers.nextStates);
    encoder.clearBuffer(buffers.summaries);
    encoder.clearBuffer(buffers.candidates);
    encoder.clearBuffer(buffers.deltas);
    encoder.clearBuffer(buffers.order);
    encoder.clearBuffer(buffers.eligibility);
    encoder.clearBuffer(buffers.selected);

    const stateCount = frontierCycleStateCount(this.tuning.frontierWidth, options.stateCount ?? this.tuning.frontierWidth, this.engine);
    const sourceScans = stateCount * this.tuning.maxBoards * 64;
    const sourceScanLimit = frontierExpansionSourceScanLimit(this.tuning.candidateWorkgroupSize, this.tuning.dispatchCandidateLimit, this.engine);
    for (let base = 0; base < sourceScans; base += sourceScanLimit) {
      const count = frontierExpansionSourceScanCount(sourceScanLimit, sourceScans, base, this.engine);
      const expandParams = u32Uniform([
        stateCount,
        this.tuning.maxBoards,
        stateStride,
        boardOffset,
        candidateCapacity,
        GPU_FRONTIER_CANDIDATE_STRIDE,
        GPU_FRONTIER_DELTA_STRIDE,
        options.rootColor,
        options.targetDepth,
        options.cycleIndex,
        base,
        count,
        0,
        0,
        0,
        0
      ]);
      const expandParamsBuffer = this.temporaryUniform(expandParams);
      encodePass(this.device, encoder, pipelines.expand, [
        buffers.states, buffers.candidates, buffers.deltas, buffers.counters, expandParamsBuffer
      ], frontierExpandWorkgroups(count, this.tuning.candidateWorkgroupSize, this.engine), 1, `frontier_expand_${options.cycleIndex}_${base}`);
    }
    await scoreCandidates?.(encoder, buffers, candidateCapacity);

    const selectParams = u32Uniform([
      candidateCapacity,
      this.tuning.frontierWidth,
      options.perParentLimit ?? 8,
      selectionCapacity,
      stateStride,
      GPU_FRONTIER_DELTA_STRIDE,
      options.cycleIndex,
      100
    ]);
    const selectParamsBuffer = this.temporaryUniform(selectParams);
    const inertSortBuffer = this.temporaryUniform(new Uint32Array(4));
    const selectionBuffers = selectionPassBuffers(buffers, selectParamsBuffer, inertSortBuffer);
    encodePass(this.device, encoder, pipelines.hash, selectionBuffers, frontierSelectionWorkgroups(candidateCapacity, this.tuning.candidateWorkgroupSize, this.engine), 1, "frontier_hash");
    encodePass(this.device, encoder, pipelines.bucketOrder, selectionBuffers, frontierSelectionWorkgroups(selectionCapacity, this.tuning.candidateWorkgroupSize, this.engine), 1, "frontier_bucket_order");
    for (let k = 2; k <= selectionCapacity; k *= 2) {
      for (let j = k / 2; j > 0; j = Math.floor(j / 2)) {
        const stageBuffer = this.temporaryUniform(new Uint32Array([k, j, 0, 0]));
        encodePass(
          this.device,
          encoder,
          pipelines.bitonicSort,
          selectionPassBuffers(buffers, selectParamsBuffer, stageBuffer),
          frontierSelectionWorkgroups(selectionCapacity, this.tuning.candidateWorkgroupSize, this.engine),
          1,
          `frontier_bitonic_sort_${k}_${j}`
        );
      }
    }
    encodePass(this.device, encoder, pipelines.markUnique, selectionBuffers, frontierSelectionWorkgroups(selectionCapacity, this.tuning.candidateWorkgroupSize, this.engine), 1, "frontier_mark_unique");
    encodePass(this.device, encoder, pipelines.markParentQuota, selectionBuffers, frontierSelectionWorkgroups(selectionCapacity, this.tuning.candidateWorkgroupSize, this.engine), 1, "frontier_mark_parent_quota");
    encodePass(this.device, encoder, pipelines.compactSelected, selectionBuffers, frontierSelectionWorkgroups(selectionCapacity, this.tuning.candidateWorkgroupSize, this.engine), 1, "frontier_compact_selected");
    encodePass(this.device, encoder, pipelines.select, selectionBuffers, 1, 1, "frontier_fill_selection_underflow");

    const stateParams = u32Uniform([
      this.tuning.frontierWidth,
      this.tuning.maxBoards,
      stateStride,
      boardOffset,
      GPU_FRONTIER_PLAN_OFFSET,
      GPU_FRONTIER_DELTA_STRIDE,
      GPU_FRONTIER_CANDIDATE_STRIDE,
      GPU_FRONTIER_MAX_PLAN_MOVES,
      GPU_FRONTIER_HEADER_STRIDE,
      0,
      0,
      0
    ]);
    const stateParamsBuffer = this.temporaryUniform(stateParams);
    encodePass(this.device, encoder, pipelines.materialize, [
      buffers.states,
      buffers.candidates,
      buffers.deltas,
      buffers.selected,
      buffers.nextStates,
      buffers.summaries,
      buffers.counters,
      stateParamsBuffer
    ], frontierMaterializeWorkgroups(this.tuning.frontierWidth, this.tuning.mutationTileSize, this.engine), 1, "frontier_materialize_selected");
  }

  swapFrontiers(buffers: FrontierBufferSet): void {
    const current = buffers.states;
    buffers.states = buffers.nextStates;
    buffers.nextStates = current;
  }

  async encodeMinimax(encoder: GPUCommandEncoder, buffers: FrontierBufferSet, targetDepth: number): Promise<void> {
    const pipelines = await this.pipelines();
    const boundedDepth = frontierMinimaxBoundedDepth(targetDepth, GPU_FRONTIER_ANCESTRY_STRIDE, this.engine);
    let readFromSummaries = true;
    for (let level = boundedDepth - 1; level > 0; level -= 1) {
      const params = this.temporaryUniform(u32Uniform([
        this.tuning.frontierWidth,
        frontierStateStride(this.tuning.maxBoards, this.engine),
        GPU_FRONTIER_HEADER_STRIDE,
        boundedDepth,
        level,
        readFromSummaries ? 1 : 0,
        0,
        0
      ]));
      encodePass(
        this.device,
        encoder,
        pipelines.reduce,
        [buffers.states, buffers.summaries, params],
        frontierMinimaxWorkgroups(this.tuning.frontierWidth, this.engine),
        1,
        `frontier_minimax_reduce_${level}`
      );
      readFromSummaries = !readFromSummaries;
    }
    if (boundedDepth > 1 && readFromSummaries) {
      const params = this.temporaryUniform(u32Uniform([
        this.tuning.frontierWidth,
        frontierStateStride(this.tuning.maxBoards, this.engine),
        GPU_FRONTIER_HEADER_STRIDE,
        boundedDepth,
        0,
        1,
        0,
        0
      ]));
      encodePass(
        this.device,
        encoder,
        pipelines.copyReduced,
        [buffers.states, buffers.summaries, params],
        frontierMinimaxWorkgroups(this.tuning.frontierWidth, this.engine),
        1,
        "frontier_minimax_copy_scores"
      );
    }
  }

  releaseCycleTemporaries(): void {
    this.#cycleTemporaries.forEach((buffer) => buffer.destroy());
    this.#cycleTemporaries = [];
  }

  destroy(): void {
    this.releaseCycleTemporaries();
    this.pool.destroy();
    this.#pipelines = null;
  }

  private pipelines(): Promise<FrontierPipelines> {
    this.#pipelines ??= createFrontierPipelines(this.device, this.tuning);
    return this.#pipelines;
  }

  private temporaryUniform(data: ArrayBufferView): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: align4(data.byteLength, this.engine),
      usage: gpuBufferUsage.UNIFORM | gpuBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    this.#cycleTemporaries.push(buffer);
    return buffer;
  }
}

export async function deriveFrontierTuning(
  device: GPUDevice,
  requestedNodes: number,
  boardCount: number,
  additionalBoardCapacity = 0
): Promise<FrontierTuning> {
  const engine = await frontierTuningEngine();
  return JSON.parse(readWasmString(engine, engine.chronofish_derive_frontier_tuning_json(
    device.limits?.maxStorageBufferBindingSize ?? 0,
    device.limits?.maxBufferSize ?? 0,
    device.limits?.maxComputeInvocationsPerWorkgroup ?? 0,
    requestedNodes,
    boardCount,
    additionalBoardCapacity
  ))) as FrontierTuning;
}

export async function autotuneFrontier(
  adapter: GPUAdapter,
  device: GPUDevice,
  requestedNodes: number,
  boardCount: number,
  modelVersion: string,
  additionalBoardCapacity = 0
): Promise<FrontierTuning> {
  const base = await deriveFrontierTuning(device, requestedNodes, boardCount, additionalBoardCapacity);
  const key = adapterTuningCacheKey(adapter, modelVersion, base);
  const cached = tuningCache.get(key);
  if (cached) {
    return cached;
  }
  const tuning = tuneWorkgroups(device, base);
  tuningCache.set(key, tuning);
  return tuning;
}

export function frontierStateStride(maxBoards: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_frontier_state_stride(maxBoards);
  }
  return GPU_FRONTIER_BOARD_OFFSET + maxBoards * GPU_FRONTIER_BOARD_STRIDE;
}

export function frontierStateBytes(maxBoards: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_frontier_state_bytes(maxBoards);
  }
  return frontierStateStride(maxBoards) * I32_BYTES;
}

export function frontierExpandWorkgroups(count: number, candidateWorkgroupSize: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_frontier_expand_workgroups(count, candidateWorkgroupSize);
  }
  return Math.ceil(count / Math.max(1, candidateWorkgroupSize));
}

export function frontierSelectionWorkgroups(capacity: number, candidateWorkgroupSize: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_frontier_selection_workgroups(capacity, candidateWorkgroupSize);
  }
  return Math.ceil(capacity / Math.max(1, candidateWorkgroupSize));
}

export function frontierMaterializeWorkgroups(frontierWidth: number, mutationTileSize: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_frontier_materialize_workgroups(frontierWidth, mutationTileSize);
  }
  return Math.ceil(frontierWidth / Math.max(1, mutationTileSize));
}

export function frontierMinimaxWorkgroups(frontierWidth: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_frontier_minimax_workgroups(frontierWidth);
  }
  return Math.ceil(frontierWidth / 64);
}

export function frontierCycleStateCount(frontierWidth: number, requestedStateCount: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_frontier_cycle_state_count(frontierWidth, requestedStateCount);
  }
  return Math.max(1, Math.min(frontierWidth, requestedStateCount));
}

export function frontierExpansionSourceScanLimit(candidateWorkgroupSize: number, dispatchCandidateLimit: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_frontier_expansion_source_scan_limit(candidateWorkgroupSize, dispatchCandidateLimit);
  }
  return Math.max(candidateWorkgroupSize, dispatchCandidateLimit, 1);
}

export function frontierExpansionSourceScanCount(sourceScanLimit: number, sourceScans: number, sourceScanBase: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_frontier_expansion_source_scan_count(sourceScanLimit, sourceScans, sourceScanBase);
  }
  return Math.min(sourceScanLimit, sourceScans - sourceScanBase);
}

export function frontierMinimaxBoundedDepth(targetDepth: number, ancestryStride: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_frontier_minimax_bounded_depth(targetDepth, ancestryStride);
  }
  return Math.min(targetDepth, ancestryStride);
}

export function adapterTuningCacheKey(adapter: GPUAdapter, modelVersion: string, tuning: FrontierTuning): string {
  const info = (adapter as GPUAdapter & { info?: { vendor?: string; architecture?: string; device?: string; description?: string } }).info;
  return [
    "chronofish.frontier-tuning.v1",
    info?.vendor ?? "unknown",
    info?.architecture ?? "unknown",
    info?.device ?? info?.description ?? "unknown",
    modelVersion,
    tuning.maxBoards
  ].join(":");
}

async function frontierTuningEngine(): Promise<ChronofishEngine> {
  tuningEnginePromise ??= instantiateChronofishWasm("./chronofish_engine.wasm")
    .then((instance) => instance.exports as unknown as ChronofishEngine);
  return tuningEnginePromise;
}

async function frontierSelectionPlan(tuning: FrontierTuning, maxSelectionScan: number | undefined): Promise<FrontierSelectionPlan> {
  const engine = await frontierTuningEngine();
  return JSON.parse(readWasmString(engine, engine.chronofish_frontier_selection_plan_json(
    tuning.maxBoards,
    tuning.frontierWidth,
    tuning.candidateCapacity,
    tuning.neuralBatchSize,
    tuning.candidateWorkgroupSize,
    tuning.mutationTileSize,
    tuning.dispatchCandidateLimit,
    maxSelectionScan ?? 0
  ))) as FrontierSelectionPlan;
}

export function align4(value: number, engine?: ChronofishEngine): number {
  if (engine) {
    return engine.chronofish_align4(value);
  }
  return Math.ceil(value / 4) * 4;
}

async function createFrontierPipelines(device: GPUDevice, tuning: FrontierTuning): Promise<FrontierPipelines> {
  const layouts = frontierPipelineLayouts(device);
  const [
    expand,
    hash,
    bucketOrder,
    bitonicSort,
    markUnique,
    markParentQuota,
    compactSelected,
    select,
    materialize,
    reduce,
    copyReduced
  ] = await Promise.all([
    createPipeline(device, "frontier_expand", GPU_FRONTIER_EXPAND_SHADER, "expand_frontier", { EXPAND_WORKGROUP_SIZE: tuning.candidateWorkgroupSize }, layouts.expand),
    createPipeline(device, "frontier_hash", GPU_FRONTIER_SELECT_SHADER, "hash_candidates", { SELECT_WORKGROUP_SIZE: tuning.candidateWorkgroupSize }, layouts.select),
    createPipeline(device, "frontier_order", GPU_FRONTIER_SELECT_SHADER, "bucket_order", { SELECT_WORKGROUP_SIZE: tuning.candidateWorkgroupSize }, layouts.select),
    createPipeline(device, "frontier_sort", GPU_FRONTIER_SELECT_SHADER, "bitonic_sort", { SELECT_WORKGROUP_SIZE: tuning.candidateWorkgroupSize }, layouts.select),
    createPipeline(device, "frontier_unique", GPU_FRONTIER_SELECT_SHADER, "mark_unique", { SELECT_WORKGROUP_SIZE: tuning.candidateWorkgroupSize }, layouts.select),
    createPipeline(device, "frontier_parent_quota", GPU_FRONTIER_SELECT_SHADER, "mark_parent_quota", { SELECT_WORKGROUP_SIZE: tuning.candidateWorkgroupSize }, layouts.select),
    createPipeline(device, "frontier_compact", GPU_FRONTIER_SELECT_SHADER, "compact_selected", { SELECT_WORKGROUP_SIZE: tuning.candidateWorkgroupSize }, layouts.select),
    createPipeline(device, "frontier_select", GPU_FRONTIER_SELECT_SHADER, "fill_selection_underflow", undefined, layouts.select),
    createPipeline(device, "frontier_materialize", GPU_FRONTIER_STATE_SHADER, "materialize_selected", { MATERIALIZE_WORKGROUP_SIZE: tuning.mutationTileSize }, layouts.materialize),
    createPipeline(device, "frontier_reduce", GPU_FRONTIER_STATE_SHADER, "minimax_reduce_stage", undefined, layouts.reduce),
    createPipeline(device, "frontier_reduce_copy", GPU_FRONTIER_STATE_SHADER, "minimax_copy_scores", undefined, layouts.reduce)
  ]);
  return {
    expand,
    hash,
    bucketOrder,
    bitonicSort,
    markUnique,
    markParentQuota,
    compactSelected,
    select,
    materialize,
    reduce,
    copyReduced
  };
}

function frontierPipelineLayouts(device: GPUDevice): FrontierPipelineLayouts {
  return {
    expand: device.createBindGroupLayout({
      label: "frontier_expand.layout",
      entries: [
        storageLayout(0, "read-only-storage"),
        storageLayout(1, "storage"),
        storageLayout(2, "storage"),
        storageLayout(3, "storage"),
        storageLayout(4, "uniform")
      ]
    }),
    select: device.createBindGroupLayout({
      label: "frontier_select.layout",
      entries: [
        storageLayout(0, "storage"),
        storageLayout(1, "read-only-storage"),
        storageLayout(2, "read-only-storage"),
        storageLayout(3, "storage"),
        storageLayout(4, "storage"),
        storageLayout(5, "storage"),
        storageLayout(6, "uniform"),
        storageLayout(7, "uniform"),
        storageLayout(8, "storage")
      ]
    }),
    materialize: device.createBindGroupLayout({
      label: "frontier_materialize.layout",
      entries: [
        storageLayout(0, "read-only-storage"),
        storageLayout(1, "read-only-storage"),
        storageLayout(2, "read-only-storage"),
        storageLayout(3, "read-only-storage"),
        storageLayout(4, "storage"),
        storageLayout(5, "storage"),
        storageLayout(6, "storage"),
        storageLayout(7, "uniform")
      ]
    }),
    reduce: device.createBindGroupLayout({
      label: "frontier_reduce.layout",
      entries: [
        storageLayout(0, "storage"),
        storageLayout(1, "storage"),
        storageLayout(2, "uniform")
      ]
    })
  };
}

function storageLayout(binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPU_SHADER_STAGE_COMPUTE,
    buffer: { type }
  };
}

async function createPipeline(
  device: GPUDevice,
  label: string,
  code: string,
  entryPoint: string,
  constants?: Record<string, number>,
  bindGroupLayout?: GPUBindGroupLayout
): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label: `${label}.module`, code });
  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) {
      throw new Error(`${label} shader compilation failed: ${errors.map((error) => error.message).join("; ")}`);
    }
  }
  const compute: GPUProgrammableStage = constants ? { module, entryPoint, constants } : { module, entryPoint };
  const layout = bindGroupLayout
    ? device.createPipelineLayout({ label: `${label}.layout`, bindGroupLayouts: [bindGroupLayout] })
    : "auto";
  return device.createComputePipeline({ label, layout, compute });
}

async function tuneWorkgroups(device: GPUDevice, base: FrontierTuning): Promise<FrontierTuning> {
  const candidates = ([32, 64, 128, 256] as const)
    .filter((size) => size <= (device.limits?.maxComputeInvocationsPerWorkgroup ?? 256));
  let best = base.candidateWorkgroupSize;
  let bestTime = Number.POSITIVE_INFINITY;
  const code = "override TUNE_SIZE: u32 = 64u; @compute @workgroup_size(TUNE_SIZE) fn tune(@builtin(global_invocation_id) id: vec3<u32>) { if (id.x == 0xffffffffu) {} }";
  for (const size of candidates) {
    try {
      const pipeline = await createPipeline(device, `frontier_tune_${size}`, code, "tune", { TUNE_SIZE: size });
      const elapsed = await measureTunedPipeline(device, pipeline, Math.max(64, Math.ceil(base.dispatchCandidateLimit / size)));
      if (elapsed < bestTime) {
        bestTime = elapsed;
        best = size;
      }
    } catch {
      // Ignore unsupported override sizes and keep the best supported candidate.
    }
  }
  return {
    ...base,
    candidateWorkgroupSize: best,
    mutationTileSize: best >= 128 ? 128 : best >= 64 ? 64 : 32,
    neuralBatchSize: Math.max(1, Math.min(base.frontierWidth, base.neuralBatchSize))
  };
}

async function measureTunedPipeline(device: GPUDevice, pipeline: GPUComputePipeline, workgroups: number): Promise<number> {
  if (device.features?.has("timestamp-query" as GPUFeatureName)) {
    try {
      const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
      const resolve = device.createBuffer({ size: 16, usage: gpuBufferUsage.QUERY_RESOLVE | gpuBufferUsage.COPY_SRC });
      const staging = device.createBuffer({ size: 16, usage: gpuBufferUsage.COPY_DST | gpuBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass({
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1
        }
      });
      pass.setPipeline(pipeline);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, staging, 0, 16);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(gpuMapMode.READ);
      const timestamps = new BigUint64Array(staging.getMappedRange().slice(0));
      const elapsed = Number((timestamps[1] ?? 0n) - (timestamps[0] ?? 0n));
      staging.unmap();
      staging.destroy();
      resolve.destroy();
      querySet.destroy();
      return elapsed;
    } catch {
      // Fall through to wall-clock timing if timestamp queries are blocked.
    }
  }
  const started = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - started;
}

function selectionPassBuffers(buffers: FrontierBufferSet, paramsBuffer: GPUBuffer, sortStageBuffer: GPUBuffer): GPUBuffer[] {
  return [
    buffers.candidates,
    buffers.states,
    buffers.deltas,
    buffers.order,
    buffers.selected,
    buffers.counters,
    paramsBuffer,
    sortStageBuffer,
    buffers.eligibility
  ];
}

function encodePass(device: GPUDevice, encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, buffers: GPUBuffer[], x: number, y = 1, label = pipeline.label): void {
  const bindGroup = device.createBindGroup({
    label: `${label}.bindGroup`,
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const pass = encoder.beginComputePass({ label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, x), Math.max(1, y));
  pass.end();
}

function u32Uniform(values: number[]): Uint32Array {
  return new Uint32Array(values.map((value) => value >>> 0));
}

export const frontierLayout = Object.freeze({
  headerStride: GPU_FRONTIER_HEADER_STRIDE,
  boardStride: GPU_FRONTIER_BOARD_STRIDE,
  planStride: GPU_FRONTIER_PLAN_STRIDE,
  maxPlanMoves: GPU_FRONTIER_MAX_PLAN_MOVES,
  candidateStride: GPU_FRONTIER_CANDIDATE_STRIDE,
  deltaStride: GPU_FRONTIER_DELTA_STRIDE,
  summaryStride: GPU_FRONTIER_SUMMARY_STRIDE
});
