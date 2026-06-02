import { instantiateChronofishWasm } from "./wasm-loader.js";

let engine = null;
let activeModelLoaded = false;
let activeModelLoad = null;
const GPU_CANDIDATE_STRIDE = 24;
const GPU_BOARD_STRIDE = 68;
const GPU_SNAPSHOT_MAGIC = 0x43464750;
const GPU_SNAPSHOT_VERSION = 1;
const GPU_SNAPSHOT_HEADER_I32S = 16;

function readWasmString(ptr) {
  // Same shared-output convention as the main thread: copy before another export
  // overwrites the buffer.
  const bytes = new Uint8Array(engine.memory.buffer, ptr, engine.chronofish_output_len());
  return new TextDecoder("utf-8").decode(bytes);
}

function readWasmBytes(ptr) {
  return new Uint8Array(engine.memory.buffer, ptr, engine.chronofish_output_len()).slice();
}

function writeWasmString(value) {
  const bytes = new TextEncoder().encode(value ?? "");
  const ptr = engine.chronofish_alloc(bytes.length);
  new Uint8Array(engine.memory.buffer, ptr, bytes.length).set(bytes);
  return { ptr, len: bytes.length };
}

async function loadEngine() {
  // The worker owns a separate WASM instance so AI search cannot block the UI
  // thread or mutate the visible game state.
  if (engine) {
    return;
  }

  const instance = await instantiateChronofishWasm("./chronofish_engine.wasm");
  engine = instance.exports;
}

async function loadActiveModel() {
  if (activeModelLoaded) {
    return;
  }
  if (activeModelLoad) {
    return activeModelLoad;
  }
  activeModelLoad = loadActiveModelOnce().finally(() => {
    activeModelLoaded = true;
    activeModelLoad = null;
  });
  return activeModelLoad;
}

async function loadActiveModelOnce() {
  if (!engine?.chronofish_set_neural_model_bytes) {
    return;
  }
  try {
    const response = await fetch("/api/training/model");
    if (!response.ok) {
      engine.chronofish_clear_neural_model?.();
      return;
    }
    const model = new Uint8Array(await response.arrayBuffer());
    const { ptr, len } = writeWasmBytes(model);
    try {
      engine.chronofish_set_neural_model_bytes(ptr, len);
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  } catch {
    engine.chronofish_clear_neural_model?.();
  }
}

function writeWasmBytes(bytes) {
  const ptr = engine.chronofish_alloc(bytes.length);
  new Uint8Array(engine.memory.buffer, ptr, bytes.length).set(bytes);
  return { ptr, len: bytes.length };
}

function replayTurns(turns) {
  // Rebuild state from submitted turns so the worker evaluates the same game the
  // main thread would reconstruct locally.
  engine.chronofish_reset();

  for (const turn of turns) {
    for (const move of turn) {
      engine.chronofish_apply_move(
        move.from.timelineId,
        move.from.time,
        move.from.x,
        move.from.y,
        move.to.timelineId,
        move.to.time,
        move.to.x,
        move.to.y
      );
    }
    engine.chronofish_submit_turn();
  }
}

function replayNotation(notation) {
  const { ptr, len } = writeWasmString(notation);
  try {
    if (!engine.chronofish_load_notation(ptr, len)) {
      throw new Error(readWasmString(engine.chronofish_last_message()));
    }
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

const GPU_MOVEGEN_SHADER = `
struct Params {
  source_count: u32,
  target_count: u32,
  root_color: u32,
  board_count: u32,
};

@group(0) @binding(0) var<storage, read> sources: array<i32>;
@group(0) @binding(1) var<storage, read> targets: array<i32>;
@group(0) @binding(2) var<storage, read_write> candidates: array<i32>;
@group(0) @binding(3) var<storage, read_write> scores: array<i32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> boards: array<i32>;

const BOARD_STRIDE: u32 = 68u;
const BOARD_SQUARE_OFFSET: u32 = 4u;

fn abs_i32(value: i32) -> i32 {
  return select(value, -value, value < 0);
}

fn sign_i32(value: i32) -> i32 {
  if (value > 0) { return 1; }
  if (value < 0) { return -1; }
  return 0;
}

fn same_distance(a: i32, b: i32, c: i32, d: i32, count: i32) -> bool {
  var first = 0;
  if (a > 0) { first = a; }
  if (first == 0 && b > 0) { first = b; }
  if (first == 0 && c > 0) { first = c; }
  if (first == 0 && d > 0) { first = d; }
  if (first == 0) { return false; }
  if (a > 0 && a != first) { return false; }
  if (b > 0 && b != first) { return false; }
  if (c > 0 && c != first) { return false; }
  if (d > 0 && d != first) { return false; }
  return count > 0;
}

fn piece_value(piece_type: i32) -> i32 {
  switch piece_type {
    case 1: { return 20000; }
    case 2: { return 10000; }
    case 3: { return 900; }
    case 4: { return 20000; }
    case 5: { return 700; }
    case 6: { return 500; }
    case 7: { return 330; }
    case 8: { return 500; }
    case 9: { return 900; }
    case 10: { return 320; }
    case 11: { return 100; }
    case 12: { return 130; }
    default: { return 0; }
  }
}

fn royal_move_penalty(piece_type: i32, target_piece: i32) -> i32 {
  if (piece_type == 1 || piece_type == 4) {
    return select(18000, 6000, target_piece != 0);
  }
  if (piece_type == 2) {
    return select(9000, 3000, target_piece != 0);
  }
  return 0;
}

fn legal_shape(piece_type: i32, color: i32, dx: i32, dy: i32, dt: i32, dl: i32, same_board: bool, target_piece: i32, target_color: i32) -> bool {
  let ax = abs_i32(dx);
  let ay = abs_i32(dy);
  let at = abs_i32(dt);
  let al = abs_i32(dl);
  let changed = select(0, 1, ax > 0) + select(0, 1, ay > 0) + select(0, 1, at > 0) + select(0, 1, al > 0);
  if (changed == 0) {
    return false;
  }
  let forward = select(1, -1, color == 1);
  let timeline_forward = forward;

  if (piece_type == 1 || piece_type == 2) {
    return ax <= 1 && ay <= 1 && at <= 1 && al <= 1;
  }
  if (piece_type == 10) {
    return (max(max(ax, ay), max(at, al)) == 2) && (ax + ay + at + al == 3);
  }
  if (piece_type == 11) {
    if (same_board && dx == 0 && dy == forward && target_piece == 0) { return true; }
    if (same_board && ax == 1 && dy == forward && target_piece != 0 && target_color != color) { return true; }
    if (!same_board && dx == 0 && dy == 0 && dt == 0 && (dl == timeline_forward || dl == timeline_forward * 2) && target_piece == 0) { return true; }
    return at == 1 && dl == timeline_forward && dx == 0 && dy == 0 && target_piece != 0 && target_color != color;
  }
  if (piece_type == 12) {
    if (target_piece != 0 && changed >= 2 && ax <= 1 && ay <= 1 && at <= 1 && al <= 1 && (dy == forward || dl == timeline_forward) && dy != -forward && dl != -timeline_forward) {
      return true;
    }
    return same_board && dx == 0 && dy == forward && target_piece == 0;
  }
  if (piece_type == 6) {
    return changed == 1;
  }
  if (piece_type == 7) {
    return changed == 2 && same_distance(ax, ay, at, al, changed);
  }
  if (piece_type == 8) {
    return changed == 3 && same_distance(ax, ay, at, al, changed);
  }
  if (piece_type == 9) {
    return changed == 4 && same_distance(ax, ay, at, al, changed);
  }
  if (piece_type == 5) {
    return changed == 1 || (changed == 2 && same_distance(ax, ay, at, al, changed));
  }
  if (piece_type == 3 || piece_type == 4) {
    return same_distance(ax, ay, at, al, changed);
  }
  return false;
}

fn board_base_by_row_time(row: i32, time: i32) -> i32 {
  for (var index = 0u; index < params.board_count; index = index + 1u) {
    let base = index * BOARD_STRIDE;
    if (boards[base + 1u] == row && boards[base + 2u] == time) {
      return i32(base);
    }
  }
  return -1;
}

fn square_code_at(row: i32, time: i32, x: i32, y: i32) -> i32 {
  if (x < 0 || x >= 8 || y < 0 || y >= 8) {
    return -1;
  }
  let base = board_base_by_row_time(row, time);
  if (base < 0) {
    return -1;
  }
  return boards[u32(base) + BOARD_SQUARE_OFFSET + u32(y * 8 + x)];
}

fn board_side_at(row: i32, time: i32) -> i32 {
  let base = board_base_by_row_time(row, time);
  if (base < 0) {
    return -1;
  }
  return boards[u32(base) + 3u];
}

fn is_sliding_piece(piece_type: i32) -> bool {
  return piece_type == 3 || piece_type == 4 || piece_type == 5 || piece_type == 6 || piece_type == 7 || piece_type == 8 || piece_type == 9;
}

fn path_clear(piece_type: i32, color: i32, from_row: i32, from_time: i32, from_x: i32, from_y: i32, dx: i32, dy: i32, raw_dt: i32, dt: i32, dl: i32) -> bool {
  if (!is_sliding_piece(piece_type) && piece_type != 11 && piece_type != 12) {
    return true;
  }
  let distance = max(max(abs_i32(dx), abs_i32(dy)), max(abs_i32(dt), abs_i32(dl)));
  if (distance <= 1) {
    return true;
  }
  let step_x = sign_i32(dx);
  let step_y = sign_i32(dy);
  let step_t = raw_dt / distance;
  let step_l = sign_i32(dl);
  for (var step = 1; step < distance; step = step + 1) {
    let row = from_row + step_l * step;
    let time = from_time + step_t * step;
    if (step_t != 0 && board_side_at(row, time) != color) {
      continue;
    }
    let x = from_x + step_x * step;
    let y = from_y + step_y * step;
    let code = square_code_at(row, time, x, y);
    if (code < 0 || code != 0) {
      return false;
    }
  }
  return true;
}

@compute @workgroup_size(64)
fn score_candidates(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let candidate_count = params.source_count * params.target_count;
  if (index >= candidate_count) {
    return;
  }
  let source_index = index / params.target_count;
  let target_index = index % params.target_count;
  let source_base = source_index * 8u;
  let target_base = target_index * 8u;
  let base = index * 24u;
  let piece_type = sources[source_base + 0u];
  let color = sources[source_base + 1u];
  let from_timeline = sources[source_base + 2u];
  let from_time = sources[source_base + 3u];
  let from_x = sources[source_base + 4u];
  let from_y = sources[source_base + 5u];
  let from_row = sources[source_base + 6u];
  let target_piece = targets[target_base + 0u];
  let target_color = targets[target_base + 1u];
  let to_timeline = targets[target_base + 2u];
  let to_time = targets[target_base + 3u];
  let to_x = targets[target_base + 4u];
  let to_y = targets[target_base + 5u];
  let to_row = targets[target_base + 6u];
  let target_side_matches = targets[target_base + 7u];
  let dx = to_x - from_x;
  let dy = to_y - from_y;
  let raw_dt = to_time - from_time;
  var dt = raw_dt;
  if (from_timeline != to_timeline || from_time != to_time) {
    if (dt % 2 == 0) {
      dt = dt / 2;
    }
  }
  let dl = to_row - from_row;
  let same_board = from_timeline == to_timeline && from_time == to_time;
  let centrality = 14 - (abs_i32(2 * to_x - 7) + abs_i32(2 * to_y - 7));
  let branch = select(1, 0, same_board);

  candidates[base + 0u] = piece_type;
  candidates[base + 1u] = color;
  candidates[base + 2u] = dx;
  candidates[base + 3u] = dy;
  candidates[base + 4u] = dt;
  candidates[base + 5u] = dl;
  candidates[base + 6u] = select(0, 1, same_board);
  candidates[base + 7u] = target_piece;
  candidates[base + 8u] = target_color;
  candidates[base + 9u] = centrality;
  candidates[base + 10u] = branch;
  candidates[base + 11u] = from_timeline;
  candidates[base + 12u] = from_time;
  candidates[base + 13u] = from_x;
  candidates[base + 14u] = from_y;
  candidates[base + 15u] = to_timeline;
  candidates[base + 16u] = to_time;
  candidates[base + 17u] = to_x;
  candidates[base + 18u] = to_y;

  if (color != i32(params.root_color) || (target_piece != 0 && target_color == color)) {
    scores[index] = -2147483647;
    return;
  }
  if (!same_board && target_side_matches == 0) {
    scores[index] = -2147483647;
    return;
  }
  if (!legal_shape(piece_type, color, dx, dy, dt, dl, same_board, target_piece, target_color)) {
    scores[index] = -2147483647;
    return;
  }
  if (!path_clear(piece_type, color, from_row, from_time, from_x, from_y, dx, dy, raw_dt, dt, dl)) {
    scores[index] = -2147483647;
    return;
  }
  scores[index] = piece_value(target_piece) * 16
    + centrality * 4
    - branch * 25
    + piece_value(piece_type) / 64
    - royal_move_penalty(piece_type, target_piece);
}
`;

const GPU_REPLY_SHADER = `
struct ReplyParams {
  root_count: u32,
  reply_count: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> roots: array<i32>;
@group(0) @binding(1) var<storage, read> replies: array<i32>;
@group(0) @binding(2) var<storage, read> root_scores: array<i32>;
@group(0) @binding(3) var<storage, read> reply_scores: array<i32>;
@group(0) @binding(4) var<storage, read_write> pair_scores: array<i32>;
@group(0) @binding(5) var<uniform> params: ReplyParams;

fn same_square(left_base: u32, right_base: u32, left_offset: u32, right_offset: u32) -> bool {
  return roots[left_base + left_offset] == replies[right_base + right_offset]
    && roots[left_base + left_offset + 1u] == replies[right_base + right_offset + 1u]
    && roots[left_base + left_offset + 2u] == replies[right_base + right_offset + 2u]
    && roots[left_base + left_offset + 3u] == replies[right_base + right_offset + 3u];
}

fn piece_value(piece_type: i32) -> i32 {
  switch piece_type {
    case 1: { return 20000; }
    case 2: { return 10000; }
    case 3: { return 900; }
    case 4: { return 20000; }
    case 5: { return 700; }
    case 6: { return 500; }
    case 7: { return 330; }
    case 8: { return 500; }
    case 9: { return 900; }
    case 10: { return 320; }
    case 11: { return 100; }
    case 12: { return 130; }
    default: { return 0; }
  }
}

@compute @workgroup_size(16, 16)
fn score_replies(@builtin(global_invocation_id) id: vec3<u32>) {
  let root_index = id.x;
  let reply_index = id.y;
  if (root_index >= params.root_count || reply_index >= params.reply_count) {
    return;
  }
  let root_base = root_index * 24u;
  let reply_base = reply_index * 24u;
  let out_index = root_index * params.reply_count + reply_index;
  if (root_scores[root_index] < -2147480000 || reply_scores[reply_index] < -2147480000) {
    pair_scores[out_index] = -2147483647;
    return;
  }
  if (same_square(root_base, reply_base, 15u, 11u)) {
    pair_scores[out_index] = -2147483647;
    return;
  }

  var pressure = reply_scores[reply_index];
  if (same_square(root_base, reply_base, 15u, 15u)) {
    pressure = pressure + piece_value(roots[root_base + 0u]) * 16;
  }
  if (same_square(root_base, reply_base, 11u, 15u)) {
    pressure = pressure - piece_value(roots[root_base + 0u]) * 8;
  }
  pair_scores[out_index] = pressure;
}
`;

async function tryGpuSearch({ depth, nodes, timeMs }) {
  if (!navigator.gpu) {
    return null;
  }
  const requestedDepth = Math.max(1, depth ?? 1);
  if (requestedDepth > 1) {
    return null;
  }
  const snapshot = readGpuSnapshot();
  if (!snapshot) {
    return null;
  }
  const candidates = buildGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (candidates.sourceCount === 0 || candidates.targetCount === 0) {
    return null;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    return null;
  }
  const device = await adapter.requestDevice();
  const scored = await scoreCandidatesOnGpu(device, candidates, snapshot.turn);
  let ranked = Array.from(scored.scores, (score, index) => ({
    move: moveFromCandidateRecord(scored.records, index),
    index,
    score: score ?? -2147483647
  }))
    .filter((entry) => entry.score > -2147480000)
    .sort((left, right) => right.score - left.score)
    .slice(0, Math.min(128, Math.max(16, nodes ?? 64)));

  for (const entry of ranked) {
    if (validateSingleMove(entry.move)) {
      return {
        moves: [entry.move],
        score: entry.score,
        depth: requestedDepth,
        nodes: ranked.length,
        status: "ok",
        gpu: true,
        gpuSnapshot: snapshot.format
      };
    }
    replayCurrentState();
    if (Date.now() >= (gpuDeadlineAt ?? 0)) {
      break;
    }
  }
  return null;
}

let currentReplay = null;
let gpuDeadlineAt = 0;

function replayCurrentState() {
  if (!currentReplay) {
    return;
  }
  if (currentReplay.notation) {
    replayNotation(currentReplay.notation);
  } else {
    replayTurns(currentReplay.turns ?? []);
  }
}

function validateSingleMove(move) {
  const ok = engine.chronofish_apply_move(
    move.from.timelineId,
    move.from.time,
    move.from.x,
    move.from.y,
    move.to.timelineId,
    move.to.time,
    move.to.x,
    move.to.y
  ) && engine.chronofish_submit_turn();
  return Boolean(ok);
}

async function scoreCandidatesOnGpu(device, inputs, turn) {
  const candidateCount = inputs.sourceCount * inputs.targetCount;
  const sourceBuffer = storageBuffer(device, inputs.sources, GPUBufferUsage.STORAGE);
  const targetBuffer = storageBuffer(device, inputs.targets, GPUBufferUsage.STORAGE);
  const boardBuffer = storageBuffer(device, inputs.boards ?? new Int32Array(GPU_BOARD_STRIDE), GPUBufferUsage.STORAGE);
  const candidateBuffer = device.createBuffer({
    size: align4(candidateCount * GPU_CANDIDATE_STRIDE * Int32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const scoreBuffer = device.createBuffer({
    size: align4(candidateCount * Int32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, inputs.sourceCount, true);
  view.setUint32(4, inputs.targetCount, true);
  view.setUint32(8, colorCode(turn), true);
  view.setUint32(12, inputs.boardCount ?? 0, true);
  const paramsBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: GPU_MOVEGEN_SHADER }), entryPoint: "score_candidates" }
  });
  const encoder = device.createCommandEncoder();
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [sourceBuffer, targetBuffer, candidateBuffer, scoreBuffer, paramsBuffer, boardBuffer]
      .map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(candidateCount / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const [records, scores] = await Promise.all([
    readInts(device, candidateBuffer, candidateCount * GPU_CANDIDATE_STRIDE * Int32Array.BYTES_PER_ELEMENT),
    readInts(device, scoreBuffer, candidateCount * Int32Array.BYTES_PER_ELEMENT)
  ]);
  return { records, scores };
}

async function scoreRootCandidatesWithReplies(
  device,
  allRootRecords,
  rankedRoots,
  allRootScores,
  allReplyRecords,
  allReplyScores
) {
  const replyLimit = 512;
  const rankedReplies = Array.from(allReplyScores, (score, index) => ({ index, score }))
    .filter((entry) => entry.score > -2147480000)
    .sort((left, right) => right.score - left.score)
    .slice(0, replyLimit);
  if (rankedReplies.length === 0) {
    return rankedRoots;
  }

  const rootRecords = pickCandidateRecords(allRootRecords, rankedRoots.map((entry) => entry.index));
  const replyRecords = pickCandidateRecords(allReplyRecords, rankedReplies.map((entry) => entry.index));
  const rootScores = new Int32Array(rankedRoots.map((entry) => allRootScores[entry.index] ?? -2147483647));
  const replyScores = new Int32Array(rankedReplies.map((entry) => allReplyScores[entry.index] ?? -2147483647));
  const pairCount = rankedRoots.length * rankedReplies.length;
  const rootBuffer = storageBuffer(device, rootRecords, GPUBufferUsage.STORAGE);
  const replyBuffer = storageBuffer(device, replyRecords, GPUBufferUsage.STORAGE);
  const rootScoreBuffer = storageBuffer(device, rootScores, GPUBufferUsage.STORAGE);
  const replyScoreBuffer = storageBuffer(device, replyScores, GPUBufferUsage.STORAGE);
  const pairBuffer = device.createBuffer({
    size: align4(pairCount * Int32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, rankedRoots.length, true);
  view.setUint32(4, rankedReplies.length, true);
  const paramsBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: GPU_REPLY_SHADER }), entryPoint: "score_replies" }
  });
  const encoder = device.createCommandEncoder();
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [rootBuffer, replyBuffer, rootScoreBuffer, replyScoreBuffer, pairBuffer, paramsBuffer]
      .map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(rankedRoots.length / 16), Math.ceil(rankedReplies.length / 16));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const pairScores = await readInts(device, pairBuffer, pairCount * Int32Array.BYTES_PER_ELEMENT);

  return rankedRoots
    .map((entry, rootIndex) => {
      let maxPressure = 0;
      const offset = rootIndex * rankedReplies.length;
      for (let replyIndex = 0; replyIndex < rankedReplies.length; replyIndex += 1) {
        maxPressure = Math.max(maxPressure, pairScores[offset + replyIndex] ?? -2147483647);
      }
      return { ...entry, score: entry.score - maxPressure };
    })
    .sort((left, right) => right.score - left.score);
}

function pickCandidateRecords(records, indices) {
  const picked = new Int32Array(indices.length * GPU_CANDIDATE_STRIDE);
  for (let outputIndex = 0; outputIndex < indices.length; outputIndex += 1) {
    const sourceOffset = indices[outputIndex] * GPU_CANDIDATE_STRIDE;
    picked.set(
      records.subarray(sourceOffset, sourceOffset + GPU_CANDIDATE_STRIDE),
      outputIndex * GPU_CANDIDATE_STRIDE
    );
  }
  return picked;
}

function readGpuSnapshot() {
  if (engine.chronofish_gpu_snapshot_bytes) {
    const bytes = readWasmBytes(engine.chronofish_gpu_snapshot_bytes());
    const parsed = parseGpuSnapshotBytes(bytes);
    if (parsed) {
      return parsed;
    }
  }
  if (!engine.chronofish_snapshot_json) {
    return null;
  }
  return {
    ...JSON.parse(readWasmString(engine.chronofish_snapshot_json())),
    format: "json"
  };
}

function parseGpuSnapshotBytes(bytes) {
  if (bytes.byteLength < GPU_SNAPSHOT_HEADER_I32S * Int32Array.BYTES_PER_ELEMENT) {
    return null;
  }
  const words = new Int32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
  if (words[0] !== GPU_SNAPSHOT_MAGIC || words[1] !== GPU_SNAPSHOT_VERSION) {
    return null;
  }
  const timelineRecordSize = words[9];
  const boardRecordSize = words[10];
  const boardSquareCount = words[11];
  const timelineCount = words[3];
  const boardCount = words[4];
  const timelines = [];
  const boards = [];
  let offset = GPU_SNAPSHOT_HEADER_I32S;

  for (let index = 0; index < timelineCount; index += 1) {
    const record = words.subarray(offset, offset + timelineRecordSize);
    timelines.push({
      id: record[0],
      row: record[1],
      owner: ownerFromCode(record[2]),
      firstBoard: record[3],
      boardCount: record[4],
      active: record[5] !== 0,
      latestTime: record[6],
      boards: []
    });
    offset += timelineRecordSize;
  }

  for (let index = 0; index < boardCount; index += 1) {
    const record = words.subarray(offset, offset + boardRecordSize);
    offset += boardRecordSize;
    const squares = words.subarray(offset, offset + boardSquareCount);
    offset += boardSquareCount;
    const board = {
      timelineIndex: record[0],
      timelineId: record[1],
      time: record[2],
      sideToMove: colorFromCode(record[3]),
      castling: record[4],
      enPassant: record[5] >= 0 ? {
        x: record[5],
        y: record[6],
        capturedX: record[7],
        capturedY: record[8]
      } : null,
      latest: record[9] !== 0,
      originKind: record[10],
      squares
    };
    boards.push(board);
    timelines[board.timelineIndex]?.boards.push(board);
  }

  return {
    format: "binary",
    words,
    turn: colorFromCode(words[2]),
    presentTime: words[8],
    timelines,
    boards,
    maxTurnPlans: words[12],
    maxMovesPerNode: words[13],
    requiredMovesPerBoard: words[14],
    maxQuiescenceDepth: words[15]
  };
}

function buildGpuCandidateInputsFromSnapshot(snapshot, color) {
  return snapshot.format === "binary"
    ? buildGpuCandidateInputsFromBinarySnapshot(snapshot, color)
    : buildGpuCandidateInputs(snapshot, color);
}

function buildGpuCandidateInputsFromBinarySnapshot(snapshot, color) {
  const sourceMeta = [];
  const targetMeta = [];
  const sources = [];
  const targets = [];
  const boards = [];
  const expectedColor = colorCode(color);

  for (const timeline of snapshot.timelines) {
    for (const board of timeline.boards) {
      pushGpuBoardRecord(boards, timeline, board);
      for (let y = 0; y < 8; y += 1) {
        for (let x = 0; x < 8; x += 1) {
          const code = board.squares[y * 8 + x] ?? 0;
          targetMeta.push({ timelineId: timeline.id, time: board.time, x, y });
          targets.push(
            code & 0xff,
            (code >> 8) & 0xff,
            timeline.id,
            board.time,
            x,
            y,
            timeline.row,
            board.sideToMove === color ? 1 : 0
          );
        }
      }
    }
  }

  for (const timeline of snapshot.timelines) {
    if (!timeline.active) {
      continue;
    }
    const board = timeline.boards.find((candidate) => candidate.latest);
    if (!board || board.sideToMove !== color) {
      continue;
    }
    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        const code = board.squares[y * 8 + x] ?? 0;
        if ((code & 0xff) === 0 || ((code >> 8) & 0xff) !== expectedColor) {
          continue;
        }
        sourceMeta.push({ timelineId: timeline.id, time: board.time, x, y });
        sources.push(
          code & 0xff,
          (code >> 8) & 0xff,
          timeline.id,
          board.time,
          x,
          y,
          timeline.row,
          board.castling
        );
      }
    }
  }

  return {
    sourceMeta,
    targetMeta,
    sourceCount: sourceMeta.length,
    targetCount: targetMeta.length,
    boardCount: boards.length / GPU_BOARD_STRIDE,
    sources: new Int32Array(sources),
    targets: new Int32Array(targets),
    boards: new Int32Array(boards)
  };
}

function buildGpuCandidateInputs(game, color) {
  const sourceMeta = [];
  const targetMeta = [];
  const sources = [];
  const targets = [];
  const boards = [];
  const timelines = sortedTimelines(game);

  for (const timeline of timelines) {
    for (const board of timeline.boards) {
      pushGpuBoardRecord(boards, timeline, {
        time: board.time,
        sideToMove: board.sideToMove,
        squares: board.board.flat().map((piece) => piece ? pieceTypeCode(piece.type) | (colorCode(piece.color) << 8) : 0)
      });
      for (let y = 0; y < 8; y += 1) {
        for (let x = 0; x < 8; x += 1) {
          const piece = board.board[y][x];
          targetMeta.push({ timelineId: timeline.id, time: board.time, x, y });
          targets.push(
            piece ? pieceTypeCode(piece.type) : 0,
            piece ? colorCode(piece.color) : 0,
            timeline.id,
            board.time,
            x,
            y,
            timeline.row,
            board.sideToMove === color ? 1 : 0
          );
        }
      }
    }
  }

  for (const timeline of timelines) {
    if (!isActiveTimeline(game, timeline)) {
      continue;
    }
    const board = latestBoard(timeline);
    if (!board || board.sideToMove !== color) {
      continue;
    }
    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        const piece = board.board[y][x];
        if (!piece || piece.color !== color) {
          continue;
        }
        sourceMeta.push({ timelineId: timeline.id, time: board.time, x, y });
        sources.push(
          pieceTypeCode(piece.type),
          colorCode(piece.color),
          timeline.id,
          board.time,
          x,
          y,
          timeline.row,
          0
        );
      }
    }
  }
  return {
    sourceMeta,
    targetMeta,
    sourceCount: sourceMeta.length,
    targetCount: targetMeta.length,
    boardCount: boards.length / GPU_BOARD_STRIDE,
    sources: new Int32Array(sources),
    targets: new Int32Array(targets),
    boards: new Int32Array(boards)
  };
}

function pushGpuBoardRecord(out, timeline, board) {
  out.push(
    timeline.id,
    timeline.row,
    board.time,
    colorCode(board.sideToMove)
  );
  for (let index = 0; index < 64; index += 1) {
    out.push(board.squares?.[index] ?? 0);
  }
}

function colorFromCode(code) {
  return code === 1 ? "black" : "white";
}

function ownerFromCode(code) {
  if (code === 1) {
    return "white";
  }
  if (code === 2) {
    return "black";
  }
  return "neutral";
}

function moveFromCandidateRecord(records, index) {
  const offset = index * GPU_CANDIDATE_STRIDE;
  return {
    from: {
      timelineId: records[offset + 11],
      time: records[offset + 12],
      x: records[offset + 13],
      y: records[offset + 14]
    },
    to: {
      timelineId: records[offset + 15],
      time: records[offset + 16],
      x: records[offset + 17],
      y: records[offset + 18]
    }
  };
}

function oppositeColor(color) {
  return color === "white" ? "black" : "white";
}

function sortedTimelines(game) {
  return [...game.timelines].sort((left, right) => left.row - right.row || left.id - right.id);
}

function latestBoard(timeline) {
  return timeline.boards.reduce((latest, board) => board.time > latest.time ? board : latest, timeline.boards[0]);
}

function isActiveTimeline(game, timeline) {
  if (timeline.owner === "neutral") {
    return true;
  }
  const ids = game.timelines.map((candidate) => candidate.id);
  const activeDistance = Math.max(0, Math.min(-Math.min(...ids), Math.max(...ids))) + 1;
  return Math.abs(timeline.id) <= activeDistance;
}

function pieceTypeCode(type) {
  return {
    king: 1,
    commonKing: 2,
    queen: 3,
    royalQueen: 4,
    princess: 5,
    rook: 6,
    bishop: 7,
    unicorn: 8,
    dragon: 9,
    knight: 10,
    pawn: 11,
    brawn: 12
  }[type] ?? 0;
}

function colorCode(color) {
  return color === "black" ? 1 : 0;
}

function storageBuffer(device, data, usage) {
  const bytes = data instanceof ArrayBuffer ? data : data.buffer;
  const buffer = device.createBuffer({
    size: align4(bytes.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

async function readInts(device, buffer, byteLength) {
  const readBuffer = device.createBuffer({
    size: align4(byteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const copy = new Int32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  return copy;
}

function align4(value) {
  return Math.ceil(value / 4) * 4;
}

self.addEventListener("message", async (event) => {
  // id is echoed back so the main thread can discard stale search results.
  const { id, notation, turns, depth, nodes, timeMs, partitionIndex, partitionCount } = event.data;

  try {
    await loadEngine();
    await loadActiveModel();
    currentReplay = { notation, turns };
    if (notation) {
      replayNotation(notation);
    } else {
      replayTurns(turns ?? []);
    }
    const searchTimeMs = Math.max(1, timeMs ?? 10_000);
    gpuDeadlineAt = Date.now() + Math.max(1, Math.floor(searchTimeMs * 0.8));
    try {
      const gpuResult = await tryGpuSearch({ depth, nodes, timeMs: searchTimeMs });
      if (gpuResult?.status === "ok" && gpuResult.moves?.length) {
        self.postMessage({ id, ok: true, result: gpuResult, partitionIndex: partitionIndex ?? 0 });
        return;
      }
    } catch (gpuError) {
      console.debug?.("GPU search fallback", gpuError);
      replayCurrentState();
    }

    const searchPartitionCount = Math.max(1, partitionCount ?? 1);
    const searchPartitionIndex = Math.min(
      searchPartitionCount - 1,
      Math.max(0, partitionIndex ?? 0)
    );
    let pointer;
    if (searchPartitionCount > 1 && engine.chronofish_ai_turn_partitioned_timed_json) {
      pointer = engine.chronofish_ai_turn_partitioned_timed_json(
        depth,
        nodes,
        searchTimeMs,
        searchPartitionIndex,
        searchPartitionCount
      );
    } else {
      const fn = engine.chronofish_ai_turn_timed_json ?? engine.chronofish_ai_turn_json;
      pointer = engine.chronofish_ai_turn_timed_json
        ? fn(depth, nodes, searchTimeMs)
        : fn(depth, nodes);
    }
    const result = JSON.parse(readWasmString(pointer));
    self.postMessage({ id, ok: true, result, partitionIndex: searchPartitionIndex });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error.message, partitionIndex: partitionIndex ?? 0 });
  }
});
