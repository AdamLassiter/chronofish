import {
  GPU_FRONTIER_BOARD_CASTLING,
  GPU_FRONTIER_BOARD_OFFSET,
  GPU_FRONTIER_BOARD_EN_PASSANT,
  GPU_FRONTIER_BOARD_LATEST,
  GPU_FRONTIER_BOARD_ORIGIN,
  GPU_FRONTIER_BOARD_OWNER,
  GPU_FRONTIER_BOARD_ACTIVE,
  GPU_FRONTIER_BOARD_PENDING,
  GPU_FRONTIER_BOARD_ROW,
  GPU_FRONTIER_BOARD_SIDE_TO_MOVE,
  GPU_FRONTIER_BOARD_SQUARES,
  GPU_FRONTIER_BOARD_STRIDE,
  GPU_FRONTIER_BOARD_TIME,
  GPU_FRONTIER_BOARD_TIMELINE_ID,
  GPU_FRONTIER_CANDIDATE_STRIDE,
  GPU_FRONTIER_DELTA_STRIDE,
  GPU_FRONTIER_HEADER_BOARD_COUNT,
  GPU_FRONTIER_HEADER_COMPLETE,
  GPU_FRONTIER_HEADER_DEPTH,
  GPU_FRONTIER_HEADER_HASH_HIGH,
  GPU_FRONTIER_HEADER_HASH_LOW,
  GPU_FRONTIER_HEADER_NEXT_BLACK_TIMELINE,
  GPU_FRONTIER_HEADER_NEXT_WHITE_TIMELINE,
  GPU_FRONTIER_HEADER_PARENT,
  GPU_FRONTIER_HEADER_PENDING_BOARDS,
  GPU_FRONTIER_HEADER_PLAN_LENGTH,
  GPU_FRONTIER_HEADER_PRESENT_TIME,
  GPU_FRONTIER_HEADER_ROOT,
  GPU_FRONTIER_HEADER_SCORE,
  GPU_FRONTIER_HEADER_STRIDE,
  GPU_FRONTIER_HEADER_TERMINAL,
  GPU_FRONTIER_HEADER_TURN,
  GPU_FRONTIER_ANCESTRY_STRIDE,
  GPU_FRONTIER_MAX_PLAN_MOVES,
  GPU_FRONTIER_PLAN_STRIDE,
  GPU_FRONTIER_PLAN_OFFSET,
  GPU_FRONTIER_SUMMARY_STRIDE
} from "./ai-layout.js";
import { colorCode, latestBoard, ownerCode, sortedTimelines, squareCodesForBoard } from "./ai-snapshot.js";
import type { GpuSnapshot } from "./ai-snapshot.js";
import { GPU_FRONTIER_EXPAND_SHADER, GPU_FRONTIER_SELECT_SHADER, GPU_FRONTIER_STATE_SHADER } from "./ai-shaders.js";

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
const DEFAULT_STORAGE_LIMIT = 128 * 1024 * 1024;
const DEFAULT_BUFFER_LIMIT = 256 * 1024 * 1024;
const MIN_FRONTIER_WIDTH = 8;
const MAX_FRONTIER_WIDTH = 128;
const MIN_CANDIDATES = 256;
const MAX_CANDIDATES = 32_768;
const tuningCache = new Map<string, Promise<FrontierTuning>>();

export interface FrontierTuning {
  maxBoards: number;
  frontierWidth: number;
  candidateCapacity: number;
  neuralBatchSize: number;
  candidateWorkgroupSize: 32 | 64 | 128 | 256;
  mutationTileSize: 32 | 64 | 128;
  dispatchCandidateLimit: number;
}

export interface FrontierBufferSet {
  states: GPUBuffer;
  nextStates: GPUBuffer;
  candidates: GPUBuffer;
  deltas: GPUBuffer;
  counters: GPUBuffer;
  order: GPUBuffer;
  selected: GPUBuffer;
  summaries: GPUBuffer;
  indirect: GPUBuffer;
}

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
  perParentLimit?: number;
  maxSelectionScan?: number;
}

interface FrontierPipelines {
  expand: GPUComputePipeline;
  hash: GPUComputePipeline;
  initializeOrder: GPUComputePipeline;
  bitonicSort: GPUComputePipeline;
  select: GPUComputePipeline;
  materialize: GPUComputePipeline;
  reduce: GPUComputePipeline;
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
  #buffers: PooledBuffer[] = [];
  #destroyed = false;

  constructor(device: GPUDevice, tuning: FrontierTuning) {
    this.device = device;
    this.tuning = tuning;
  }

  createSearchBuffers(): FrontierBufferSet {
    const stateBytes = frontierStateBytes(this.tuning.maxBoards) * this.tuning.frontierWidth;
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
      selected: this.acquire(this.tuning.frontierWidth * I32_BYTES, storageCopy),
      summaries: this.acquire(this.tuning.frontierWidth * GPU_FRONTIER_SUMMARY_STRIDE * I32_BYTES, storageCopy),
      indirect: this.acquire(16, gpuBufferUsage.STORAGE | gpuBufferUsage.INDIRECT | gpuBufferUsage.COPY_DST)
    };
  }

  acquire(byteLength: number, usage: number): GPUBuffer {
    if (this.#destroyed) {
      throw new Error("GPU frontier buffer pool was destroyed.");
    }
    const size = align4(Math.max(4, byteLength));
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
  #pipelines: Promise<FrontierPipelines> | null = null;
  #cycleTemporaries: GPUBuffer[] = [];

  constructor(device: GPUDevice, tuning: FrontierTuning, pool = new FrontierBufferPool(device, tuning)) {
    this.device = device;
    this.tuning = tuning;
    this.pool = pool;
  }

  uploadRoot(buffers: FrontierBufferSet, root: EncodedFrontierRoot): void {
    this.device.queue.writeBuffer(buffers.states, 0, root.words);
  }

  async encodeExpansionCycle(
    encoder: GPUCommandEncoder,
    buffers: FrontierBufferSet,
    options: FrontierPassOptions
  ): Promise<void> {
    const pipelines = await this.pipelines();
    const candidateCapacity = floorPowerOfTwo(this.tuning.candidateCapacity);
    const stateStride = frontierStateStride(this.tuning.maxBoards);
    const boardOffset = GPU_FRONTIER_BOARD_OFFSET;
    encoder.clearBuffer(buffers.counters, 0, 12);
    encoder.clearBuffer(buffers.nextStates);
    encoder.clearBuffer(buffers.summaries);
    encoder.clearBuffer(buffers.candidates);
    encoder.clearBuffer(buffers.deltas);
    encoder.clearBuffer(buffers.order);
    encoder.clearBuffer(buffers.selected);

    const sourceScans = this.tuning.frontierWidth * this.tuning.maxBoards * 64;
    const sourceScanLimit = Math.max(this.tuning.candidateWorkgroupSize, this.tuning.dispatchCandidateLimit);
    for (let base = 0; base < sourceScans; base += sourceScanLimit) {
      const count = Math.min(sourceScanLimit, sourceScans - base);
      const expandParams = u32Uniform([
        this.tuning.frontierWidth,
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
      ], Math.ceil(count / this.tuning.candidateWorkgroupSize));
    }

    const selectParams = u32Uniform([
      candidateCapacity,
      this.tuning.frontierWidth,
      options.perParentLimit ?? 8,
      Math.min(candidateCapacity, options.maxSelectionScan ?? this.tuning.frontierWidth * 16),
      stateStride,
      GPU_FRONTIER_DELTA_STRIDE,
      0,
      0
    ]);
    const selectParamsBuffer = this.temporaryUniform(selectParams);
    const inertSortBuffer = this.temporaryUniform(new Uint32Array(4));
    const selectionBuffers = [
      buffers.candidates,
      buffers.states,
      buffers.deltas,
      buffers.order,
      buffers.selected,
      buffers.counters,
      selectParamsBuffer,
      inertSortBuffer
    ];
    encodePass(this.device, encoder, pipelines.hash, selectionBuffers, Math.ceil(candidateCapacity / this.tuning.candidateWorkgroupSize));
    encodePass(this.device, encoder, pipelines.initializeOrder, selectionBuffers, Math.ceil(candidateCapacity / this.tuning.candidateWorkgroupSize));
    for (let k = 2; k <= candidateCapacity; k *= 2) {
      for (let j = k / 2; j > 0; j = Math.floor(j / 2)) {
        const stageBuffer = this.temporaryUniform(new Uint32Array([k, j, 0, 0]));
        encodePass(this.device, encoder, pipelines.bitonicSort, [
          ...selectionBuffers.slice(0, 7), stageBuffer
        ], Math.ceil(candidateCapacity / this.tuning.candidateWorkgroupSize));
      }
    }
    encodePass(this.device, encoder, pipelines.select, selectionBuffers, 1);

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
    ], Math.ceil(this.tuning.frontierWidth / this.tuning.mutationTileSize));
  }

  swapFrontiers(buffers: FrontierBufferSet): void {
    const current = buffers.states;
    buffers.states = buffers.nextStates;
    buffers.nextStates = current;
  }

  async encodeMinimax(encoder: GPUCommandEncoder, buffers: FrontierBufferSet, targetDepth: number): Promise<void> {
    const pipelines = await this.pipelines();
    const params = this.temporaryUniform(u32Uniform([
      this.tuning.frontierWidth,
      frontierStateStride(this.tuning.maxBoards),
      GPU_FRONTIER_HEADER_STRIDE,
      Math.min(targetDepth, GPU_FRONTIER_ANCESTRY_STRIDE)
    ]));
    encodePass(this.device, encoder, pipelines.reduce, [buffers.states, buffers.summaries, params], 1);
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
      size: align4(data.byteLength),
      usage: gpuBufferUsage.UNIFORM | gpuBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    this.#cycleTemporaries.push(buffer);
    return buffer;
  }
}

export function deriveFrontierTuning(device: GPUDevice, requestedNodes: number, boardCount: number): FrontierTuning {
  const storageLimit = finiteLimit(device.limits?.maxStorageBufferBindingSize, DEFAULT_STORAGE_LIMIT);
  const bufferLimit = finiteLimit(device.limits?.maxBufferSize, DEFAULT_BUFFER_LIMIT);
  const maxInvocations = finiteLimit(device.limits?.maxComputeInvocationsPerWorkgroup, 256);
  const maxBoardsByState = Math.max(1, Math.floor((storageLimit / MIN_FRONTIER_WIDTH / I32_BYTES - GPU_FRONTIER_HEADER_STRIDE - GPU_FRONTIER_PLAN_STRIDE) / GPU_FRONTIER_BOARD_STRIDE));
  const maxBoards = Math.max(boardCount, Math.min(64, maxBoardsByState));
  const stateBytes = frontierStateBytes(maxBoards);
  const frontierWidth = clamp(
    Math.floor(Math.min(storageLimit, bufferLimit) / Math.max(1, stateBytes * 2)),
    MIN_FRONTIER_WIDTH,
    Math.min(MAX_FRONTIER_WIDTH, Math.max(MIN_FRONTIER_WIDTH, requestedNodes))
  );
  const candidateRecordBytes = (GPU_FRONTIER_CANDIDATE_STRIDE + GPU_FRONTIER_DELTA_STRIDE) * I32_BYTES;
  const candidateCapacity = clamp(
    Math.floor(Math.min(storageLimit, bufferLimit) / candidateRecordBytes),
    MIN_CANDIDATES,
    Math.min(MAX_CANDIDATES, Math.max(MIN_CANDIDATES, requestedNodes * 4))
  );
  const neuralBytesPerSample = 32 * 64 * 16 * Float32Array.BYTES_PER_ELEMENT;
  const neuralBatchSize = clamp(Math.floor(storageLimit / neuralBytesPerSample), 1, frontierWidth);
  const candidateWorkgroupSize = workgroupSize(maxInvocations);
  return {
    maxBoards,
    frontierWidth,
    candidateCapacity,
    neuralBatchSize,
    candidateWorkgroupSize,
    mutationTileSize: candidateWorkgroupSize >= 128 ? 128 : candidateWorkgroupSize >= 64 ? 64 : 32,
    dispatchCandidateLimit: Math.max(candidateWorkgroupSize, Math.min(candidateCapacity, candidateWorkgroupSize * 1024))
  };
}

export function autotuneFrontier(
  adapter: GPUAdapter,
  device: GPUDevice,
  requestedNodes: number,
  boardCount: number,
  modelVersion: string
): Promise<FrontierTuning> {
  const base = deriveFrontierTuning(device, requestedNodes, boardCount);
  const key = adapterTuningCacheKey(adapter, modelVersion, base);
  const cached = tuningCache.get(key);
  if (cached) {
    return cached;
  }
  const tuning = tuneWorkgroups(device, base);
  tuningCache.set(key, tuning);
  return tuning;
}

export function frontierStateStride(maxBoards: number): number {
  return GPU_FRONTIER_BOARD_OFFSET + maxBoards * GPU_FRONTIER_BOARD_STRIDE;
}

export function frontierStateBytes(maxBoards: number): number {
  return frontierStateStride(maxBoards) * I32_BYTES;
}

export function encodeFrontierRoot(snapshot: GpuSnapshot, maxBoards: number): EncodedFrontierRoot {
  const boards = sortedTimelines(snapshot)
    .flatMap((timeline) => timeline.boards
      .slice()
      .sort((left, right) => left.time - right.time)
      .map((board) => ({ timeline, board })));
  if (boards.length > maxBoards) {
    throw new Error(`GPU frontier snapshot has ${boards.length} boards but the adapter limit is ${maxBoards}.`);
  }
  const words = new Int32Array(frontierStateStride(maxBoards));
  words[GPU_FRONTIER_HEADER_PARENT] = -1;
  words[GPU_FRONTIER_HEADER_ROOT] = 0;
  words[GPU_FRONTIER_HEADER_SCORE] = 0;
  words[GPU_FRONTIER_HEADER_DEPTH] = 0;
  words[GPU_FRONTIER_HEADER_TURN] = colorCode(snapshot.turn);
  words[GPU_FRONTIER_HEADER_BOARD_COUNT] = boards.length;
  words[GPU_FRONTIER_HEADER_PLAN_LENGTH] = 0;
  words[GPU_FRONTIER_HEADER_COMPLETE] = 0;
  words[GPU_FRONTIER_HEADER_TERMINAL] = snapshot.royalCaptureBy ? 1 : 0;
  words[GPU_FRONTIER_HEADER_NEXT_WHITE_TIMELINE] = snapshot.nextTimelineId ?? 1;
  words[GPU_FRONTIER_HEADER_NEXT_BLACK_TIMELINE] = snapshot.nextBlackTimelineId ?? -1;
  const ids = snapshot.timelines.map((timeline) => timeline.id);
  const activeDistance = Math.max(0, Math.min(-Math.min(...ids, 0), Math.max(...ids, 0))) + 1;
  const activeLatest = snapshot.timelines
    .filter((timeline) => timelineActive(timeline, activeDistance))
    .map((timeline) => ({ timeline, board: latestBoard(timeline) }))
    .filter((entry): entry is typeof entry & { board: NonNullable<typeof entry.board> } => Boolean(entry.board));
  const present = activeLatest.reduce<number | null>((value, entry) => value === null ? entry.board.time : Math.min(value, entry.board.time), null) ?? 0;
  const pending = activeLatest.filter(({ board }) => board.time === present && board.sideToMove === snapshot.turn).length;
  words[GPU_FRONTIER_HEADER_PRESENT_TIME] = present;
  words[GPU_FRONTIER_HEADER_PENDING_BOARDS] = pending;
  words[GPU_FRONTIER_HEADER_COMPLETE] = pending === 0 ? 1 : 0;

  const boardOffset = GPU_FRONTIER_BOARD_OFFSET;
  boards.forEach(({ timeline, board }, index) => {
    const base = boardOffset + index * GPU_FRONTIER_BOARD_STRIDE;
    const latest = latestBoard(timeline)?.time === board.time;
    const active = timelineActive(timeline, activeDistance);
    const pendingBoard = latest && active && board.time === present && board.sideToMove === snapshot.turn;
    words[base + GPU_FRONTIER_BOARD_TIMELINE_ID] = timeline.id;
    words[base + GPU_FRONTIER_BOARD_ROW] = timeline.row;
    words[base + GPU_FRONTIER_BOARD_OWNER] = ownerCode(timeline.owner);
    words[base + GPU_FRONTIER_BOARD_TIME] = board.time;
    words[base + GPU_FRONTIER_BOARD_SIDE_TO_MOVE] = colorCode(board.sideToMove);
    words[base + GPU_FRONTIER_BOARD_CASTLING] = board.castling ?? 0;
    words[base + GPU_FRONTIER_BOARD_EN_PASSANT] = board.enPassant?.x ?? -1;
    words[base + GPU_FRONTIER_BOARD_EN_PASSANT + 1] = board.enPassant?.y ?? -1;
    words[base + GPU_FRONTIER_BOARD_EN_PASSANT + 2] = board.enPassant?.capturedX ?? -1;
    words[base + GPU_FRONTIER_BOARD_EN_PASSANT + 3] = board.enPassant?.capturedY ?? -1;
    words[base + GPU_FRONTIER_BOARD_LATEST] = latest ? 1 : 0;
    words[base + GPU_FRONTIER_BOARD_ORIGIN] = originCode(board.origin?.type);
    words[base + GPU_FRONTIER_BOARD_ACTIVE] = active ? 1 : 0;
    words[base + GPU_FRONTIER_BOARD_PENDING] = pendingBoard ? 1 : 0;
    words.set(squareCodesForBoard(board), base + GPU_FRONTIER_BOARD_SQUARES);
  });

  const [hashLow, hashHigh] = hashFrontierWords(words.subarray(boardOffset, boardOffset + boards.length * GPU_FRONTIER_BOARD_STRIDE));
  words[GPU_FRONTIER_HEADER_HASH_LOW] = hashLow;
  words[GPU_FRONTIER_HEADER_HASH_HIGH] = hashHigh;
  return { words, boardCount: boards.length, hashLow, hashHigh };
}

function timelineActive(timeline: { id: number; owner?: string }, activeDistance: number): boolean {
  return timeline.owner === "neutral" || Math.abs(timeline.id) <= activeDistance;
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

function finiteLimit(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) && (value ?? 0) > 0 ? value! : fallback;
}

function hashFrontierWords(words: Int32Array): [number, number] {
  let low = 0x811c9dc5 >>> 0;
  let high = 0x9e3779b9 >>> 0;
  for (let index = 0; index < words.length; index += 1) {
    const value = words[index] ?? 0;
    low = Math.imul((low ^ value) >>> 0, 0x01000193) >>> 0;
    high = Math.imul((high + value + index) >>> 0, 0x85ebca6b) >>> 0;
  }
  return [low | 0, high | 0];
}

function originCode(origin: string | undefined): number {
  if (origin === "source-advance") return 1;
  if (origin === "branch") return 2;
  if (origin === "cross-board") return 3;
  return origin ? 4 : 0;
}

function workgroupSize(maxInvocations: number): 32 | 64 | 128 | 256 {
  if (maxInvocations >= 256) return 256;
  if (maxInvocations >= 128) return 128;
  if (maxInvocations >= 64) return 64;
  return 32;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

async function createFrontierPipelines(device: GPUDevice, tuning: FrontierTuning): Promise<FrontierPipelines> {
  const [expand, hash, initializeOrder, bitonicSort, select, materialize, reduce] = await Promise.all([
    createPipeline(device, "frontier_expand", GPU_FRONTIER_EXPAND_SHADER, "expand_frontier", { EXPAND_WORKGROUP_SIZE: tuning.candidateWorkgroupSize }),
    createPipeline(device, "frontier_hash", GPU_FRONTIER_SELECT_SHADER, "hash_candidates", { SELECT_WORKGROUP_SIZE: tuning.candidateWorkgroupSize }),
    createPipeline(device, "frontier_order", GPU_FRONTIER_SELECT_SHADER, "initialize_order", { SELECT_WORKGROUP_SIZE: tuning.candidateWorkgroupSize }),
    createPipeline(device, "frontier_sort", GPU_FRONTIER_SELECT_SHADER, "bitonic_sort", { SELECT_WORKGROUP_SIZE: tuning.candidateWorkgroupSize }),
    createPipeline(device, "frontier_select", GPU_FRONTIER_SELECT_SHADER, "select_top_k"),
    createPipeline(device, "frontier_materialize", GPU_FRONTIER_STATE_SHADER, "materialize_selected", { MATERIALIZE_WORKGROUP_SIZE: tuning.mutationTileSize }),
    createPipeline(device, "frontier_reduce", GPU_FRONTIER_STATE_SHADER, "minimax_reduce")
  ]);
  return { expand, hash, initializeOrder, bitonicSort, select, materialize, reduce };
}

async function createPipeline(device: GPUDevice, label: string, code: string, entryPoint: string, constants?: Record<string, number>): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label: `${label}.module`, code });
  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) {
      throw new Error(`${label} shader compilation failed: ${errors.map((error) => error.message).join("; ")}`);
    }
  }
  const compute: GPUProgrammableStage = constants ? { module, entryPoint, constants } : { module, entryPoint };
  return device.createComputePipeline({ label, layout: "auto", compute });
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

function encodePass(device: GPUDevice, encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, buffers: GPUBuffer[], x: number, y = 1): void {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, x), Math.max(1, y));
  pass.end();
}

function u32Uniform(values: number[]): Uint32Array {
  return new Uint32Array(values.map((value) => value >>> 0));
}

function floorPowerOfTwo(value: number): number {
  return 2 ** Math.floor(Math.log2(Math.max(1, value)));
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
