import type { BoardSnapshot, Color, GameSnapshot, Piece, PieceType, Timeline, TimelineOwner } from "./types.js";
import encode_neural_position from "./shaders/encode_neural_position.wgsl";

// Matches constants in encode_neural_position.wgsl
const NEURAL_MAX_BOARDS = 16;
const NEURAL_BOARD_PLANES = 32;
const NEURAL_BOARD_SQUARES = 64;
const NEURAL_INPUT_SIZE = NEURAL_MAX_BOARDS * NEURAL_BOARD_PLANES * NEURAL_BOARD_SQUARES;

const ENCODE_META_STRIDE = 6;
let cachedGpuAdapter: GPUAdapter | null = null;
let cachedGpuDevice: GPUDevice | null = null;
const pipelineCache = new Map<string, GPUComputePipeline>();
interface GpuBufferUsageConstants {
  MAP_READ: number;
  COPY_SRC: number;
  COPY_DST: number;
  UNIFORM: number;
  STORAGE: number;
}

interface GpuMapModeConstants {
  READ: number;
}

const gpuBufferUsage: GpuBufferUsageConstants = (globalThis as unknown as { GPUBufferUsage?: GpuBufferUsageConstants }).GPUBufferUsage ?? {
  MAP_READ: 1,
  COPY_SRC: 4,
  COPY_DST: 8,
  UNIFORM: 64,
  STORAGE: 128
};
const gpuMapMode: GpuMapModeConstants = (globalThis as unknown as { GPUMapMode?: GpuMapModeConstants }).GPUMapMode ?? {
  READ: 1
};

interface WorkerScope {
  addEventListener(type: "message", listener: (event: MessageEvent<TrainingLabelRequest>) => void | Promise<void>): void;
  postMessage(message: TrainingLabelResponse): void;
}

interface TrainingLabelRequest {
  id: number;
  type?: "batchSample" | "selfPlay" | string;
  game?: GameSnapshot;
  games?: GameSnapshot[];
  encodeOnly?: boolean;
}

type TrainingLabelResponse =
  | { id: number; ok: true; sample: NeuralSample }
  | { id: number; ok: true; samples: NeuralSample[] }
  | { id: number; ok: false; error: string };

interface NeuralSample {
  sideToMove: Color;
  boardCount: number;
  features: number[];
}

interface EncodedPosition {
  values: number[];
  boardCount: number;
}

interface SelectedBoard {
  category: number;
  negativeTime: number;
  absTimeline: number;
  timelineId: number;
  timelineIndex: number;
  boardIndex: number;
  timeline: Timeline;
  board: BoardSnapshot;
}

const ENCODE_NEURAL_POSITION_SHADER = encode_neural_position;

const workerSelf = self as unknown as WorkerScope;

workerSelf.addEventListener("message", async (event) => {
  const { id, type, game, games, encodeOnly } = event.data;
  try {
    if (type === "batchSample") {
      if (!Array.isArray(games)) {
        throw new Error("Batch training position encoding requires game snapshots.");
      }
      const samples: NeuralSample[] = [];
      for (const snapshot of games) {
        if (!snapshot.timelines.length) {
          throw new Error("Training position encoding requires a client game snapshot.");
        }
        samples.push(await neuralPosition(snapshot));
      }
      workerSelf.postMessage({ id, ok: true, samples });
      return;
    }
    if (type === "selfPlay" || !encodeOnly) {
      throw new Error("CPU search labels are disabled; training labels must come from GPU/model prediction.");
    }
    if (!game?.timelines.length) {
      throw new Error("Training position encoding requires a client game snapshot.");
    }
    workerSelf.postMessage({ id, ok: true, sample: await neuralPosition(game) });
  } catch (error: unknown) {
    workerSelf.postMessage({ id, ok: false, error: errorMessage(error) });
  }
});

async function neuralPosition(game: GameSnapshot): Promise<NeuralSample> {
  const encoded = await encodeNeuralPositionOnGpu(game, game.turn);
  return {
    sideToMove: game.turn,
    boardCount: encoded.boardCount,
    features: encoded.values
  };
}

async function encodeNeuralPositionOnGpu(game: GameSnapshot, color: Color): Promise<EncodedPosition> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable for training position encoding.");
  }
  const selected = neuralBoardSelection(game);
  const boardCount = selected.length;
  const squares = new Int32Array(NEURAL_MAX_BOARDS * NEURAL_BOARD_SQUARES);
  const boardMeta = new Int32Array(NEURAL_MAX_BOARDS * ENCODE_META_STRIDE);

  selected.forEach(({ timeline, board }, boardIndex) => {
    const latest = latestBoard(timeline)?.time === board.time;
    boardMeta.set([
      timeline.id,
      ownerCode(timeline.owner),
      board.time,
      latest ? 1 : 0,
      colorCode(board.sideToMove),
      colorCode(color)
    ], boardIndex * ENCODE_META_STRIDE);

    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        squares[boardIndex * NEURAL_BOARD_SQUARES + y * 8 + x] = pieceCode(board.board[y]?.[x]);
      }
    }
  });

  const device = await getGpuDevice();
  if (!device) {
    throw new Error("No WebGPU adapter is available.");
  }
  const squareBuffer = storageBuffer(device, squares, gpuBufferUsage.STORAGE);
  const metaBuffer = storageBuffer(device, boardMeta, gpuBufferUsage.STORAGE);
  const featureBuffer = device.createBuffer({
    size: align4(NEURAL_INPUT_SIZE * Float32Array.BYTES_PER_ELEMENT),
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  });
  const params = new ArrayBuffer(16);
  new DataView(params).setUint32(0, boardCount, true);
  const paramsBuffer = storageBuffer(device, params, gpuBufferUsage.UNIFORM);
  const pipeline = await createComputePipelineChecked(device, "encode_neural_position", ENCODE_NEURAL_POSITION_SHADER, "encode");
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [squareBuffer, metaBuffer, featureBuffer, paramsBuffer]
      .map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(NEURAL_INPUT_SIZE / 256));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const values = await readFloats(device, featureBuffer, NEURAL_INPUT_SIZE * Float32Array.BYTES_PER_ELEMENT);
  return { values: Array.from(values), boardCount };
}

function neuralBoardSelection(game: GameSnapshot): SelectedBoard[] {
  const candidates: SelectedBoard[] = [];
  game.timelines.forEach((timeline, timelineIndex) => {
    const latestTime = latestBoard(timeline)?.time;
    timeline.boards.forEach((board, boardIndex) => {
      const latest = board.time === latestTime;
      const hasRoyal = board.board.some((row) => row.some((piece) => Boolean(piece && isRoyalPiece(piece.type))));
      const hasRecentOrigin = Boolean(board.origin);
      if (!latest && !hasRoyal && !hasRecentOrigin) {
        return;
      }
      const category = latest ? 0 : hasRoyal ? 1 : 2;
      candidates.push({
        category,
        negativeTime: -board.time,
        absTimeline: Math.abs(timeline.id),
        timelineId: timeline.id,
        timelineIndex,
        boardIndex,
        timeline,
        board
      });
    });
  });

  candidates.sort((left, right) =>
    left.category - right.category ||
    left.negativeTime - right.negativeTime ||
    left.absTimeline - right.absTimeline ||
    left.timelineId - right.timelineId ||
    left.timelineIndex - right.timelineIndex ||
    left.boardIndex - right.boardIndex
  );
  return candidates.slice(0, NEURAL_MAX_BOARDS);
}

function pieceCode(piece: Piece | null | undefined): number {
  if (!piece) {
    return 0;
  }
  return pieceTypeCode(piece.type) | ((piece.color === "black" ? 1 : 0) << 8);
}

function colorCode(color: Color): number {
  return color === "black" ? 1 : 0;
}

function ownerCode(owner: TimelineOwner): number {
  if (owner === "white") {
    return 1;
  }
  if (owner === "black") {
    return 2;
  }
  return 0;
}

function pieceTypeCode(type: PieceType): number {
  const codes: Record<PieceType, number> = {
    king: 0,
    commonKing: 1,
    queen: 2,
    royalQueen: 3,
    princess: 4,
    rook: 5,
    bishop: 6,
    unicorn: 7,
    dragon: 8,
    knight: 9,
    pawn: 10,
    brawn: 11
  };
  return codes[type] + 1;
}

function isRoyalPiece(type: PieceType): boolean {
  return type === "king" || type === "royalQueen";
}

function latestBoard(timeline: Timeline): BoardSnapshot | undefined {
  const first = timeline.boards[0];
  return first ? timeline.boards.reduce((latest, board) => board.time > latest.time ? board : latest, first) : undefined;
}

function storageBuffer(device: GPUDevice, data: ArrayBuffer | ArrayBufferView, usage: number): GPUBuffer {
  const bytes = data instanceof ArrayBuffer
    ? data
    : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const buffer = device.createBuffer({
    size: align4(bytes.byteLength),
    usage: usage | gpuBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

async function getGpuDevice(): Promise<GPUDevice | null> {
  if (!navigator.gpu) {
    return null;
  }
  if (cachedGpuDevice) {
    return cachedGpuDevice;
  }
  cachedGpuAdapter = cachedGpuAdapter ?? await navigator.gpu.requestAdapter();
  if (!cachedGpuAdapter) {
    return null;
  }
  cachedGpuDevice = await requestHighLimitDevice(cachedGpuAdapter);
  cachedGpuDevice.lost?.then(() => {
    cachedGpuDevice = null;
    pipelineCache.clear();
  });
  return cachedGpuDevice;
}

async function requestHighLimitDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  const requiredLimits: Record<string, number> = {};
  for (const key of ["maxStorageBufferBindingSize", "maxBufferSize"] as const) {
    const value = adapter.limits[key];
    if (Number.isFinite(value) && value > 0) {
      requiredLimits[key] = value;
    }
  }
  if (Object.keys(requiredLimits).length === 0) {
    return adapter.requestDevice();
  }
  try {
    return await adapter.requestDevice({ requiredLimits });
  } catch {
    return adapter.requestDevice();
  }
}

async function createComputePipelineChecked(device: GPUDevice, label: string, code: string, entryPoint: string): Promise<GPUComputePipeline> {
  const cacheKey = `${label}:${entryPoint}`;
  const cached = pipelineCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const module = device.createShaderModule({ label: `${label}.module`, code });
  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message: GPUCompilationMessage) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(formatShaderErrors(label, errors));
    }
  }
  const pipeline = device.createComputePipeline({
    label,
    layout: "auto",
    compute: { module, entryPoint }
  });
  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

function formatShaderErrors(label: string, errors: GPUCompilationMessage[]): string {
  return `${label} shader compilation failed: ${errors.map((error) =>
    `line ${error.lineNum ?? "?"}, column ${error.linePos ?? "?"}: ${error.message}`
  ).join("; ")}`;
}

async function readFloats(device: GPUDevice, buffer: GPUBuffer, byteLength: number): Promise<Float32Array> {
  const readBuffer = device.createBuffer({
    size: align4(byteLength),
    usage: gpuBufferUsage.COPY_DST | gpuBufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(gpuMapMode.READ);
  const copy = new Float32Array(readBuffer.getMappedRange().slice(0, byteLength));
  readBuffer.unmap();
  return copy;
}

function align4(value: number): number {
  return Math.max(4, Math.ceil(value / 4) * 4);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
