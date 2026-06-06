const NEURAL_MAX_BOARDS = 16;
const NEURAL_BOARD_PLANES = 32;
const NEURAL_BOARD_SQUARES = 64;
const NEURAL_INPUT_SIZE = NEURAL_MAX_BOARDS * NEURAL_BOARD_PLANES * NEURAL_BOARD_SQUARES;
const ENCODE_META_STRIDE = 6;
let cachedGpuAdapter = null;
let cachedGpuDevice = null;
const pipelineCache = new Map();

const ENCODE_NEURAL_POSITION_SHADER = `
struct Params {
  board_count: u32,
};

@group(0) @binding(0) var<storage, read> squares: array<i32>;
@group(0) @binding(1) var<storage, read> board_meta: array<i32>;
@group(0) @binding(2) var<storage, read_write> features: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BOARD_PLANES: u32 = 32u;
const BOARD_SQUARES: u32 = 64u;
const META_STRIDE: u32 = 6u;

fn piece_plane(code: i32) -> i32 {
  let piece_type = code & 255;
  if (piece_type <= 0 || piece_type > 12) {
    return -1;
  }
  let color = (code >> 8) & 255;
  return color * 12 + piece_type - 1;
}

fn abs_i32(value: i32) -> i32 {
  return select(value, -value, value < 0);
}

fn min_i32(left: i32, right: i32) -> i32 {
  return select(right, left, left < right);
}

fn max_i32(left: i32, right: i32) -> i32 {
  return select(right, left, left > right);
}

fn owner_active(owner: i32, timeline_id: i32, active_distance: i32) -> bool {
  if (owner == 0) {
    return true;
  }
  return abs_i32(timeline_id) <= active_distance;
}

fn relative_color_value(color: i32, perspective: i32) -> f32 {
  return select(-1.0, 1.0, color == perspective);
}

@compute @workgroup_size(256)
fn encode(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= ${NEURAL_INPUT_SIZE}u) {
    return;
  }
  let square = index % BOARD_SQUARES;
  let plane = (index / BOARD_SQUARES) % BOARD_PLANES;
  let board = index / (BOARD_SQUARES * BOARD_PLANES);
  if (board >= params.board_count) {
    features[index] = 0.0;
    return;
  }

  if (plane < 24u) {
    let expected = piece_plane(squares[board * BOARD_SQUARES + square]);
    features[index] = select(0.0, 1.0, expected == i32(plane));
    return;
  }

  let meta_base = board * META_STRIDE;
  var min_timeline = board_meta[0];
  var max_timeline = board_meta[0];
  var present = 2147483647;
  let perspective = board_meta[5u];
  for (var meta_board = 0u; meta_board < params.board_count; meta_board = meta_board + 1u) {
    let base = meta_board * META_STRIDE;
    let timeline_id = board_meta[base];
    min_timeline = min_i32(min_timeline, timeline_id);
    max_timeline = max_i32(max_timeline, timeline_id);
  }
  let active_distance = max_i32(0, min_i32(-min_timeline, max_timeline)) + 1;
  for (var meta_board = 0u; meta_board < params.board_count; meta_board = meta_board + 1u) {
    let base = meta_board * META_STRIDE;
    let timeline_id = board_meta[base];
    let owner = board_meta[base + 1u];
    let time = board_meta[base + 2u];
    if (owner_active(owner, timeline_id, active_distance) && time < present) {
      present = time;
    }
  }
  let timeline_id = board_meta[meta_base];
  let owner = board_meta[meta_base + 1u];
  let time = board_meta[meta_base + 2u];
  let latest = board_meta[meta_base + 3u] != 0;
  let side_to_move = board_meta[meta_base + 4u];
  let is_active = owner_active(owner, timeline_id, active_distance);
  let owner_sign = select(0.0, relative_color_value(owner - 1, perspective), owner != 0);
  let time_distance = f32(max_i32(-16, min_i32(16, time - present))) / 16.0;
  if (plane == 24u) {
    features[index] = relative_color_value(side_to_move, perspective);
  } else if (plane == 25u) {
    features[index] = select(0.0, 1.0, is_active);
  } else if (plane == 26u) {
    features[index] = select(0.0, 1.0, latest);
  } else if (plane == 27u) {
    features[index] = select(0.0, 1.0, time == present);
  } else if (plane == 28u) {
    features[index] = owner_sign;
  } else if (plane == 29u) {
    features[index] = time_distance;
  } else if (plane == 30u) {
    features[index] = 1.0;
  } else {
    features[index] = 0.0;
  }
}
`;

self.addEventListener("message", async (event) => {
  const { id, type, game, games, encodeOnly } = event.data;
  try {
    if (type === "batchSample") {
      if (!Array.isArray(games)) {
        throw new Error("Batch training position encoding requires game snapshots.");
      }
      const samples = [];
      for (const snapshot of games) {
        if (!snapshot?.timelines?.length) {
          throw new Error("Training position encoding requires a client game snapshot.");
        }
        samples.push(await neuralPosition(snapshot));
      }
      self.postMessage({ id, ok: true, samples });
      return;
    }
    if (type === "selfPlay" || !encodeOnly) {
      throw new Error("CPU search labels are disabled; training labels must come from GPU/model prediction.");
    }
    if (!game?.timelines?.length) {
      throw new Error("Training position encoding requires a client game snapshot.");
    }
    self.postMessage({ id, ok: true, sample: await neuralPosition(game) });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error.message });
  }
});

async function neuralPosition(game) {
  const encoded = await encodeNeuralPositionOnGpu(game, game.turn);
  return {
    sideToMove: game.turn,
    boardCount: encoded.boardCount,
    features: encoded.values
  };
}

async function encodeNeuralPositionOnGpu(game, color) {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable for training position encoding.");
  }
  const selected = neuralBoardSelection(game);
  const boardCount = selected.length;
  const squares = new Int32Array(NEURAL_MAX_BOARDS * NEURAL_BOARD_SQUARES);
  const meta = new Int32Array(NEURAL_MAX_BOARDS * ENCODE_META_STRIDE);

  selected.forEach(({ timeline, board }, boardIndex) => {
    const latest = latestBoard(timeline)?.time === board.time;
    meta.set([
      timeline.id,
      ownerCode(timeline.owner),
      board.time,
      latest ? 1 : 0,
      colorCode(board.sideToMove),
      colorCode(color)
    ], boardIndex * ENCODE_META_STRIDE);

    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        squares[boardIndex * NEURAL_BOARD_SQUARES + y * 8 + x] = pieceCode(board.board?.[y]?.[x]);
      }
    }
  });

  const device = await getGpuDevice();
  if (!device) {
    throw new Error("No WebGPU adapter is available.");
  }
  const squareBuffer = storageBuffer(device, squares, GPUBufferUsage.STORAGE);
  const metaBuffer = storageBuffer(device, meta, GPUBufferUsage.STORAGE);
  const featureBuffer = device.createBuffer({
    size: align4(NEURAL_INPUT_SIZE * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const params = new ArrayBuffer(16);
  new DataView(params).setUint32(0, boardCount, true);
  const paramsBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
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

function neuralBoardSelection(game) {
  const candidates = [];
  game.timelines.forEach((timeline, timelineIndex) => {
    const latestTime = latestBoard(timeline)?.time;
    timeline.boards.forEach((board, boardIndex) => {
      const latest = board.time === latestTime;
      const hasRoyal = board.board?.some((row) => row.some((piece) => piece && isRoyalPiece(piece.type)));
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

function pieceCode(piece) {
  if (!piece) {
    return 0;
  }
  return pieceTypeCode(piece.type) | ((piece.color === "black" ? 1 : 0) << 8);
}

function colorCode(color) {
  return color === "black" ? 1 : 0;
}

function ownerCode(owner) {
  if (owner === "white") {
    return 1;
  }
  if (owner === "black") {
    return 2;
  }
  return 0;
}

function pieceTypeCode(type) {
  return {
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
  }[type] + 1 || 0;
}

function isRoyalPiece(type) {
  return type === "king" || type === "royalQueen";
}

function latestBoard(timeline) {
  return timeline.boards.reduce((latest, board) => board.time > latest.time ? board : latest, timeline.boards[0]);
}

function storageBuffer(device, data, usage) {
  const bytes = data instanceof ArrayBuffer
    ? data
    : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const buffer = device.createBuffer({
    size: align4(bytes.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

async function getGpuDevice() {
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

async function requestHighLimitDevice(adapter) {
  const requiredLimits = {};
  for (const key of ["maxStorageBufferBindingSize", "maxBufferSize"]) {
    const value = adapter.limits?.[key];
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

async function createComputePipelineChecked(device, label, code, entryPoint) {
  const cacheKey = `${label}:${entryPoint}`;
  const cached = pipelineCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const module = device.createShaderModule({ label: `${label}.module`, code });
  if (module.compilationInfo) {
    const info = await module.compilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
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

function formatShaderErrors(label, errors) {
  return `${label} shader compilation failed: ${errors.map((error) =>
    `line ${error.lineNum ?? "?"}, column ${error.linePos ?? "?"}: ${error.message}`
  ).join("; ")}`;
}

async function readFloats(device, buffer, byteLength) {
  const readBuffer = device.createBuffer({
    size: align4(byteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const copy = new Float32Array(readBuffer.getMappedRange().slice(0, byteLength));
  readBuffer.unmap();
  return copy;
}

function align4(value) {
  return Math.max(4, Math.ceil(value / 4) * 4);
}
