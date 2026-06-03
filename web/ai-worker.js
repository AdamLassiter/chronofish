const GPU_CANDIDATE_STRIDE = 24;
const GPU_SOURCE_STRIDE = 10;
const GPU_TARGET_STRIDE = 10;
const GPU_BOARD_STRIDE = 68;
const GPU_MUTATION_BOARD_STRIDE = 76;
const GPU_MUTATION_CHILD_STRIDE = GPU_MUTATION_BOARD_STRIDE * 2;
const GPU_MUTATION_STATUS_OK = 1;
const GPU_MUTATION_STATUS_ROYAL_CAPTURE = 2;
const GPU_MUTATION_STATUS_BRANCH_OK = 3;
const GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE = 4;
const GPU_TURN_STATUS_RECORD_STRIDE = 4;

const GPU_TURN_STATUS_SHADER = `
struct Params {
  board_count: u32,
  turn: i32,
};

@group(0) @binding(0) var<storage, read> boards: array<i32>;
@group(0) @binding(1) var<storage, read_write> result: array<i32>;
@group(0) @binding(2) var<uniform> params: Params;

const RECORD_STRIDE: u32 = 4u;

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

@compute @workgroup_size(1)
fn turn_status(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x != 0u) {
    return;
  }
  if (params.board_count == 0u) {
    result[0] = 0;
    result[1] = params.turn;
    result[2] = 0;
    result[3] = 0;
    return;
  }

  var min_timeline = boards[0];
  var max_timeline = boards[0];
  for (var index = 0u; index < params.board_count; index = index + 1u) {
    let offset = index * RECORD_STRIDE;
    let timeline_id = boards[offset];
    min_timeline = min_i32(min_timeline, timeline_id);
    max_timeline = max_i32(max_timeline, timeline_id);
  }
  let active_distance = max_i32(0, min_i32(-min_timeline, max_timeline)) + 1;

  var present = 2147483647;
  var next_turn = params.turn;
  var unplayed = 0;
  var pending_count = 0;
  for (var index = 0u; index < params.board_count; index = index + 1u) {
    let offset = index * RECORD_STRIDE;
    let timeline_id = boards[offset];
    let owner = boards[offset + 1u];
    let time = boards[offset + 2u];
    let side = boards[offset + 3u];
    if (!owner_active(owner, timeline_id, active_distance)) {
      continue;
    }
    if (time < present) {
      present = time;
      next_turn = side;
    }
  }

  for (var index = 0u; index < params.board_count; index = index + 1u) {
    let offset = index * RECORD_STRIDE;
    let timeline_id = boards[offset];
    let owner = boards[offset + 1u];
    let time = boards[offset + 2u];
    let side = boards[offset + 3u];
    if (!owner_active(owner, timeline_id, active_distance)) {
      continue;
    }
    if (time == present && side == params.turn) {
      unplayed = 1;
      pending_count = pending_count + 1;
    }
  }

  if (present == 2147483647) {
    present = 0;
  }

  result[0] = unplayed;
  result[1] = next_turn;
  result[2] = present;
  result[3] = pending_count;
}
`;
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
const SOURCE_STRIDE: u32 = 10u;
const TARGET_STRIDE: u32 = 10u;

fn abs_i32(value: i32) -> i32 {
  return select(value, -value, value < 0);
}

fn min_i32(left: i32, right: i32) -> i32 {
  return select(right, left, left < right);
}

fn max_i32(left: i32, right: i32) -> i32 {
  return select(right, left, left > right);
}

fn sign_i32(value: i32) -> i32 {
  if (value > 0) { return 1; }
  if (value < 0) { return -1; }
  return 0;
}

fn owner_active(owner: i32, timeline_id: i32, active_distance: i32) -> bool {
  if (owner == 0) {
    return true;
  }
  return abs_i32(timeline_id) <= active_distance;
}

fn active_distance_from_targets() -> i32 {
  if (params.target_count == 0u) {
    return 0;
  }
  var min_timeline = targets[2u];
  var max_timeline = targets[2u];
  for (var index = 0u; index < params.target_count; index = index + 1u) {
    let base = index * TARGET_STRIDE;
    let timeline_id = targets[base + 2u];
    min_timeline = min_i32(min_timeline, timeline_id);
    max_timeline = max_i32(max_timeline, timeline_id);
  }
  return max_i32(0, min_i32(-min_timeline, max_timeline)) + 1;
}

fn present_time_from_targets(active_distance: i32) -> i32 {
  var present = 2147483647;
  for (var index = 0u; index < params.target_count; index = index + 1u) {
    let base = index * TARGET_STRIDE;
    let timeline_id = targets[base + 2u];
    let time = targets[base + 3u];
    let owner = targets[base + 8u];
    let latest = targets[base + 9u] != 0;
    if (latest && owner_active(owner, timeline_id, active_distance) && time < present) {
      present = time;
    }
  }
  return present;
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
  let source_base = source_index * SOURCE_STRIDE;
  let target_base = target_index * TARGET_STRIDE;
  let base = index * 24u;
  let piece_type = sources[source_base + 0u];
  let color = sources[source_base + 1u];
  let from_timeline = sources[source_base + 2u];
  let from_time = sources[source_base + 3u];
  let from_x = sources[source_base + 4u];
  let from_y = sources[source_base + 5u];
  let from_row = sources[source_base + 6u];
  let source_side_to_move = sources[source_base + 7u];
  let source_owner = sources[source_base + 8u];
  let source_latest = sources[source_base + 9u] != 0;
  let target_piece = targets[target_base + 0u];
  let target_color = targets[target_base + 1u];
  let to_timeline = targets[target_base + 2u];
  let to_time = targets[target_base + 3u];
  let to_x = targets[target_base + 4u];
  let to_y = targets[target_base + 5u];
  let to_row = targets[target_base + 6u];
  let target_side_to_move = targets[target_base + 7u];
  let target_latest = targets[target_base + 9u] != 0;
  let active_distance = active_distance_from_targets();
  let present = present_time_from_targets(active_distance);
  let source_active = owner_active(source_owner, from_timeline, active_distance);
  let source_present = present != 2147483647 && from_time == present;
  let source_side_matches = source_side_to_move == i32(params.root_color);
  let target_side_matches = target_side_to_move == i32(params.root_color);
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

  if (!source_side_matches || !source_latest || !source_active || !source_present || color != i32(params.root_color) || (target_piece != 0 && target_color == color)) {
    scores[index] = -2147483647;
    return;
  }
  if (!same_board && !target_side_matches) {
    scores[index] = -2147483647;
    return;
  }
  if (!same_board && !target_latest && !(piece_type == 1 || piece_type == 4)) {
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

const GPU_MUTATE_SHADER = `
struct MutateParams {
  candidate_count: u32,
  board_count: u32,
  turn: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> candidates: array<i32>;
@group(0) @binding(1) var<storage, read> parent_boards: array<i32>;
@group(0) @binding(2) var<storage, read_write> child_boards: array<i32>;
@group(0) @binding(3) var<storage, read_write> statuses: array<i32>;
@group(0) @binding(4) var<uniform> params: MutateParams;

const CANDIDATE_STRIDE: u32 = 24u;
const BOARD_STRIDE: u32 = 76u;
const BOARD_SQUARE_OFFSET: u32 = 12u;
const STATUS_OK: i32 = 1;
const STATUS_ROYAL_CAPTURE: i32 = 2;
const STATUS_BRANCH_OK: i32 = 3;
const STATUS_BRANCH_ROYAL_CAPTURE: i32 = 4;
const STATUS_UNSUPPORTED_HISTORICAL_BRANCH: i32 = -1;
const STATUS_UNSUPPORTED_EN_PASSANT: i32 = -3;
const STATUS_MISSING_BOARD: i32 = -4;
const STATUS_SOURCE_MISMATCH: i32 = -5;
const STATUS_TARGET_MISMATCH: i32 = -6;
const STATUS_MISSING_EN_PASSANT: i32 = -7;
const STATUS_MISSING_ROOK: i32 = -8;

fn abs_i32(value: i32) -> i32 {
  return select(value, -value, value < 0);
}

fn board_base_by_timeline_time(timeline_id: i32, time: i32) -> i32 {
  for (var index = 0u; index < params.board_count; index = index + 1u) {
    let base = index * BOARD_STRIDE;
    if (parent_boards[base + 1u] == timeline_id && parent_boards[base + 2u] == time) {
      return i32(base);
    }
  }
  return -1;
}

fn board_base_by_timeline_time_or_latest(timeline_id: i32, time: i32) -> i32 {
  for (var index = 0u; index < params.board_count; index = index + 1u) {
    let base = index * BOARD_STRIDE;
    if (parent_boards[base + 1u] == timeline_id && parent_boards[base + 2u] == time) {
      return i32(base);
    }
  }
  return -1;
}

fn promoted_piece_type(piece_type: i32, color: i32, y: i32) -> i32 {
  if ((piece_type == 11 || piece_type == 12) && ((color == 0 && y == 7) || (color == 1 && y == 0))) {
    return 3;
  }
  return piece_type;
}

fn update_castling_rights(castling: i32, piece_type: i32, color: i32, from_x: i32, from_y: i32, to_x: i32, to_y: i32, target_code: i32) -> i32 {
  var rights = castling;
  if (piece_type == 1 && color == 0) {
    rights = rights & ~3;
  }
  if (piece_type == 1 && color == 1) {
    rights = rights & ~12;
  }
  if (piece_type == 6 && color == 0 && from_y == 0 && from_x == 0) {
    rights = rights & ~2;
  }
  if (piece_type == 6 && color == 0 && from_y == 0 && from_x == 7) {
    rights = rights & ~1;
  }
  if (piece_type == 6 && color == 1 && from_y == 7 && from_x == 0) {
    rights = rights & ~8;
  }
  if (piece_type == 6 && color == 1 && from_y == 7 && from_x == 7) {
    rights = rights & ~4;
  }
  let captured_type = target_code & 255;
  let captured_color = (target_code >> 8) & 255;
  if (captured_type == 6 && captured_color == 0 && to_y == 0 && to_x == 0) {
    rights = rights & ~2;
  }
  if (captured_type == 6 && captured_color == 0 && to_y == 0 && to_x == 7) {
    rights = rights & ~1;
  }
  if (captured_type == 6 && captured_color == 1 && to_y == 7 && to_x == 0) {
    rights = rights & ~8;
  }
  if (captured_type == 6 && captured_color == 1 && to_y == 7 && to_x == 7) {
    rights = rights & ~4;
  }
  return rights;
}

@compute @workgroup_size(64)
fn mutate_candidates(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.candidate_count) {
    return;
  }

  let candidate_base = index * CANDIDATE_STRIDE;
  let out_base = index * BOARD_STRIDE * 2u;
  let piece_type = candidates[candidate_base + 0u];
  let color = candidates[candidate_base + 1u];
  let dx = candidates[candidate_base + 2u];
  let dy = candidates[candidate_base + 3u];
  let same_board = candidates[candidate_base + 6u] == 1;
  let target_piece = candidates[candidate_base + 7u];
  let target_color = candidates[candidate_base + 8u];
  let from_timeline = candidates[candidate_base + 11u];
  let from_time = candidates[candidate_base + 12u];
  let from_x = candidates[candidate_base + 13u];
  let from_y = candidates[candidate_base + 14u];
  let to_timeline = candidates[candidate_base + 15u];
  let to_time = candidates[candidate_base + 16u];
  let to_x = candidates[candidate_base + 17u];
  let to_y = candidates[candidate_base + 18u];

  let parent_base_i32 = board_base_by_timeline_time(from_timeline, from_time);
  if (parent_base_i32 < 0) {
    statuses[index] = STATUS_MISSING_BOARD;
    return;
  }
  let parent_base = u32(parent_base_i32);
  let target_base_i32 = board_base_by_timeline_time_or_latest(to_timeline, to_time);
  if (target_base_i32 < 0) {
    statuses[index] = STATUS_MISSING_BOARD;
    return;
  }
  let target_base = u32(target_base_i32);
  for (var cell = 0u; cell < BOARD_STRIDE; cell = cell + 1u) {
    child_boards[out_base + cell] = parent_boards[parent_base + cell];
  }
  if (!same_board) {
    for (var cell = 0u; cell < BOARD_STRIDE; cell = cell + 1u) {
      child_boards[out_base + BOARD_STRIDE + cell] = parent_boards[target_base + cell];
    }
  }

  let source_square = BOARD_SQUARE_OFFSET + u32(from_y * 8 + from_x);
  let target_square = BOARD_SQUARE_OFFSET + u32(to_y * 8 + to_x);
  let expected_source = piece_type | (color << 8);
  let actual_source = parent_boards[parent_base + source_square];
  let actual_target = parent_boards[target_base + target_square];
  if (actual_source != expected_source) {
    statuses[index] = STATUS_SOURCE_MISMATCH;
    return;
  }
  if ((actual_target & 255) != target_piece || (target_piece != 0 && ((actual_target >> 8) & 255) != target_color)) {
    statuses[index] = STATUS_TARGET_MISMATCH;
    return;
  }

  let next_turn = 1 - i32(params.turn);
  let placed_type = promoted_piece_type(piece_type, color, to_y);
  let placed_code = placed_type | (color << 8);
  let source_castling = parent_boards[parent_base + 4u];
  let target_castling = parent_boards[target_base + 4u];

  if (same_board) {
    child_boards[out_base + 2u] = from_time + 1;
    child_boards[out_base + 3u] = next_turn;
    child_boards[out_base + source_square] = 0;
    child_boards[out_base + target_square] = placed_code;
    child_boards[out_base + 3u] = next_turn;
    child_boards[out_base + 2u] = from_time + 1;
    child_boards[out_base + 3u] = next_turn;
    child_boards[out_base + 4u] = update_castling_rights(source_castling, piece_type, color, from_x, from_y, to_x, to_y, actual_target);
    child_boards[out_base + 5u] = -1;
    child_boards[out_base + 6u] = -1;
    child_boards[out_base + 7u] = -1;
    child_boards[out_base + 8u] = -1;
    if ((piece_type == 11 || piece_type == 12) && abs_i32(dx) == 1 && target_piece == 0) {
      let ep_x = parent_boards[parent_base + 5u];
      let ep_y = parent_boards[parent_base + 6u];
      let captured_x = parent_boards[parent_base + 7u];
      let captured_y = parent_boards[parent_base + 8u];
      if (ep_x != to_x || ep_y != to_y) {
        statuses[index] = STATUS_MISSING_EN_PASSANT;
        return;
      }
      child_boards[out_base + BOARD_SQUARE_OFFSET + u32(captured_y * 8 + captured_x)] = 0;
    }
    if (piece_type == 1 && abs_i32(dx) == 2 && dy == 0) {
      let rook_from_x = select(0, 7, dx > 0);
      let rook_to_x = select(3, 5, dx > 0);
      let rook_square = BOARD_SQUARE_OFFSET + u32(from_y * 8 + rook_from_x);
      let rook_code = parent_boards[parent_base + rook_square];
      if ((rook_code & 255) != 6 || ((rook_code >> 8) & 255) != color) {
        statuses[index] = STATUS_MISSING_ROOK;
        return;
      }
      child_boards[out_base + rook_square] = 0;
      child_boards[out_base + BOARD_SQUARE_OFFSET + u32(from_y * 8 + rook_to_x)] = rook_code;
    }
    if ((piece_type == 11 || piece_type == 12) && dx == 0 && abs_i32(dy) == 2) {
      child_boards[out_base + 5u] = from_x;
      child_boards[out_base + 6u] = from_y + select(1, -1, color == 1);
      child_boards[out_base + 7u] = to_x;
      child_boards[out_base + 8u] = to_y;
    }
    statuses[index] = select(STATUS_OK, STATUS_ROYAL_CAPTURE, target_piece == 1 || target_piece == 4);
    return;
  }

  let target_latest = parent_boards[target_base + 9u] == 1;
  if (!target_latest) {
    statuses[index] = STATUS_UNSUPPORTED_HISTORICAL_BRANCH;
    return;
  }
  child_boards[out_base + 2u] = from_time + 1;
  child_boards[out_base + 3u] = next_turn;
  child_boards[out_base + 4u] = update_castling_rights(source_castling, piece_type, color, from_x, from_y, to_x, to_y, 0);
  child_boards[out_base + 5u] = -1;
  child_boards[out_base + 6u] = -1;
  child_boards[out_base + 7u] = -1;
  child_boards[out_base + 8u] = -1;
  child_boards[out_base + source_square] = 0;
  let branch_base = out_base + BOARD_STRIDE;
  child_boards[branch_base + 2u] = to_time + 1;
  child_boards[branch_base + 3u] = next_turn;
  child_boards[branch_base + 4u] = update_castling_rights(target_castling, piece_type, color, from_x, from_y, to_x, to_y, actual_target);
  child_boards[branch_base + 5u] = -1;
  child_boards[branch_base + 6u] = -1;
  child_boards[branch_base + 7u] = -1;
  child_boards[branch_base + 8u] = -1;
  child_boards[branch_base + target_square] = placed_code;
  statuses[index] = select(STATUS_BRANCH_OK, STATUS_BRANCH_ROYAL_CAPTURE, target_piece == 1 || target_piece == 4);
}
`;

async function tryGpuSearch({ depth, nodes, timeMs, gpuMode = "hybrid", snapshotOverride = null }) {
  if (!navigator.gpu) {
    return null;
  }
  const requestedDepth = Math.max(1, depth ?? 1);
  const snapshot = snapshotOverride ?? readGpuSnapshot();
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
  const turnStatus = await turnStatusOnGpu(device, snapshot);
  if (gpuMode === "full") {
    return tryFullGpuSearch(device, snapshot, candidates, { requestedDepth, nodes: nodes ?? 64, turnStatus });
  }
  const scored = await scoreCandidatesOnGpu(device, candidates, snapshot.turn);
  let ranked = Array.from(scored.scores, (score, index) => ({
    move: moveFromCandidateRecord(scored.records, index),
    index,
    score: score ?? -2147483647
  }))
    .filter((entry) => entry.score > -2147480000)
    .sort((left, right) => right.score - left.score)
    .slice(0, Math.min(128, Math.max(16, nodes ?? 64)));

  if (requestedDepth > 1) {
    return searchSingleMoveRepliesOnGpu(device, snapshot, candidates, scored.records, ranked, {
      requestedDepth,
      nodes: nodes ?? 64
    });
  }

  if (turnStatus.pendingPresentBoardCount === 1 && ranked.length > 0) {
    const mutated = await mutateRankedCandidatesOnGpu(device, candidates, scored.records, ranked);
    const selected = mutated.find((entry) => entry.mutationStatus >= GPU_MUTATION_STATUS_OK);
    if (selected) {
      return {
        moves: [selected.move],
        score: selected.score,
        depth: requestedDepth,
        nodes: ranked.length,
        status: "ok",
        gpu: true,
        gpuTerminal: selected.mutationStatus === GPU_MUTATION_STATUS_ROYAL_CAPTURE || selected.mutationStatus === GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE,
        gpuSnapshot: snapshot.format,
        gpuSearch: "single-present-gpu-mutated"
      };
    }
  }

  return null;
}

async function searchSingleMoveRepliesOnGpu(device, snapshot, inputs, allCandidateRecords, ranked, { requestedDepth, nodes }) {
  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, allCandidateRecords, ranked, { readChildren: true });
  let best = null;
  for (const entry of mutated.filter((candidate) => candidate.mutationStatus >= GPU_MUTATION_STATUS_OK && candidate.childBoards)) {
    let score = entry.score;
    if (entry.mutationStatus !== GPU_MUTATION_STATUS_ROYAL_CAPTURE && entry.mutationStatus !== GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
      const childSnapshot = snapshotWithGpuChildBoards(snapshot, entry.childBoards, entry.mutationStatus, { advanceTurn: true });
      score -= await bestReplyScoreOnGpu(device, childSnapshot, nodes);
    }
    const candidate = {
      moves: [entry.move],
      score,
      depth: Math.min(requestedDepth, 2),
      nodes: mutated.length,
      status: "ok",
      gpu: true,
      gpuSnapshot: snapshot.format,
      gpuSearch: "single-move-replies"
    };
    if (!best || candidate.score > best.score || candidate.score === best.score && turnPlanKey(candidate.moves) < turnPlanKey(best.moves)) {
      best = candidate;
    }
  }
  return best;
}

async function legalTargetsOnGpu(position, snapshotOverride = null) {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable for legal target calculation.");
  }
  const snapshot = snapshotOverride ?? readGpuSnapshot();
  if (!snapshot) {
    throw new Error("GPU snapshot is unavailable.");
  }
  const inputs = buildGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    return { source: null, targets: [] };
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter is available.");
  }
  const device = await adapter.requestDevice();
  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  const targets = [];
  const seen = new Set();
  let source = null;

  for (let index = 0; index < scored.scores.length; index += 1) {
    const score = scored.scores[index] ?? -2147483647;
    if (score <= -2147480000) {
      continue;
    }
    const offset = index * GPU_CANDIDATE_STRIDE;
    if (
      scored.records[offset + 11] !== position.timelineId ||
      scored.records[offset + 12] !== position.time ||
      scored.records[offset + 13] !== position.x ||
      scored.records[offset + 14] !== position.y
    ) {
      continue;
    }
    source ??= {
      piece: {
        type: pieceTypeFromCode(scored.records[offset + 0]),
        color: colorFromCode(scored.records[offset + 1])
      },
      position: { ...position }
    };
    const target = {
      timelineId: scored.records[offset + 15],
      time: scored.records[offset + 16],
      x: scored.records[offset + 17],
      y: scored.records[offset + 18]
    };
    const key = `${target.timelineId}:${target.time}:${target.x}:${target.y}`;
    if (!seen.has(key)) {
      seen.add(key);
      targets.push(target);
    }
  }

  targets.sort((left, right) =>
    left.timelineId - right.timelineId ||
    left.time - right.time ||
    left.y - right.y ||
    left.x - right.x
  );
  return { source, targets };
}

async function applyMoveOnGpu(move, snapshotOverride = null) {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable for move application.");
  }
  const snapshot = snapshotOverride ?? readGpuSnapshot();
  if (!snapshot) {
    throw new Error("GPU snapshot is unavailable.");
  }
  const inputs = buildGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    throw new Error("No GPU move candidates are available.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter is available.");
  }
  const device = await adapter.requestDevice();
  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  const index = findCandidateIndex(scored, move);
  if (index < 0 || (scored.scores[index] ?? -2147483647) <= -2147480000) {
    throw new Error("GPU rejected that move.");
  }
  const candidateRecords = pickCandidateRecords(scored.records, [index]);
  const ranked = [{ move, index: 0, score: scored.scores[index] ?? 0 }];
  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, candidateRecords, ranked, { readChildren: true });
  const selected = mutated[0];
  if (!selected || selected.mutationStatus < GPU_MUTATION_STATUS_OK || !selected.childBoards) {
    throw new Error("GPU move mutation is unsupported for that move.");
  }
  const nextSnapshot = snapshotWithGpuChildBoards(snapshot, selected.childBoards, selected.mutationStatus, { move, advanceTurn: false });
  return gpuSnapshotToGame(nextSnapshot);
}

async function submitTurnOnGpu(snapshotOverride = null) {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable for turn submission.");
  }
  const snapshot = snapshotOverride ?? readGpuSnapshot();
  if (!snapshot) {
    throw new Error("GPU snapshot is unavailable.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter is available.");
  }
  const device = await adapter.requestDevice();
  return turnStatusOnGpu(device, snapshot);
}

async function turnStatusOnGpu(device, snapshot) {
  const records = [];
  for (const timeline of sortedTimelines(snapshot)) {
    const board = latestBoard(timeline);
    if (!board) {
      continue;
    }
    records.push(
      timeline.id,
      ownerCode(timeline.owner),
      board.time,
      colorCode(board.sideToMove)
    );
  }
  const boardRecords = new Int32Array(records.length > 0 ? records : [0, 0, 0, colorCode(snapshot.turn)]);
  const boardBuffer = storageBuffer(device, boardRecords, GPUBufferUsage.STORAGE);
  const resultBuffer = device.createBuffer({
    size: align4(4 * Int32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, records.length / GPU_TURN_STATUS_RECORD_STRIDE, true);
  view.setInt32(4, colorCode(snapshot.turn), true);
  const paramsBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: GPU_TURN_STATUS_SHADER }), entryPoint: "turn_status" }
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [boardBuffer, resultBuffer, paramsBuffer]
      .map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const result = await readInts(device, resultBuffer, 4 * Int32Array.BYTES_PER_ELEMENT);
  return {
    complete: result[0] === 0,
    nextTurn: colorFromCode(result[1]),
    presentTime: result[2],
    pendingPresentBoardCount: result[3]
  };
}

function findCandidateIndex(scored, move) {
  for (let index = 0; index < scored.scores.length; index += 1) {
    const offset = index * GPU_CANDIDATE_STRIDE;
    if (
      scored.records[offset + 11] === move.from.timelineId &&
      scored.records[offset + 12] === move.from.time &&
      scored.records[offset + 13] === move.from.x &&
      scored.records[offset + 14] === move.from.y &&
      scored.records[offset + 15] === move.to.timelineId &&
      scored.records[offset + 16] === move.to.time &&
      scored.records[offset + 17] === move.to.x &&
      scored.records[offset + 18] === move.to.y
    ) {
      return index;
    }
  }
  return -1;
}

async function tryFullGpuSearch(device, snapshot, inputs, { requestedDepth, nodes, turnStatus }) {
  if (turnStatus.pendingPresentBoardCount !== 1) {
    throw new Error("Full GPU search currently requires one pending present board.");
  }

  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  const ranked = Array.from(scored.scores, (score, index) => ({
    move: moveFromCandidateRecord(scored.records, index),
    index,
    score: score ?? -2147483647
  }))
    .filter((entry) => entry.score > -2147480000)
    .sort((left, right) => right.score - left.score)
    .slice(0, Math.min(128, Math.max(16, nodes ?? 64)));
  if (ranked.length === 0) {
    throw new Error("Full GPU search found no candidate moves.");
  }

  const mutated = await mutateRankedCandidatesOnGpu(device, inputs, scored.records, ranked, { readChildren: true });
  const supported = mutated.filter((entry) => entry.mutationStatus >= GPU_MUTATION_STATUS_OK && entry.childBoards);
  if (supported.length === 0) {
    throw new Error("Full GPU mutation produced no supported child states.");
  }

  let best = null;
  for (const entry of supported.slice(0, Math.min(32, Math.max(8, nodes ?? 64)))) {
    let score = entry.score;
    if (requestedDepth > 1 && entry.mutationStatus !== GPU_MUTATION_STATUS_ROYAL_CAPTURE && entry.mutationStatus !== GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
      const childSnapshot = snapshotWithGpuChildBoards(snapshot, entry.childBoards, entry.mutationStatus, { advanceTurn: true });
      const replyScore = await bestReplyScoreOnGpu(device, childSnapshot, nodes);
      score -= replyScore;
    }
    const candidate = {
      moves: [entry.move],
      score,
      depth: Math.min(requestedDepth, 2),
      nodes: supported.length,
      status: "ok",
      gpu: true,
      gpuMode: "full",
      gpuTerminal: entry.mutationStatus === GPU_MUTATION_STATUS_ROYAL_CAPTURE || entry.mutationStatus === GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE,
      gpuSnapshot: snapshot.format,
      gpuSearch: "full-single-present"
    };
    if (!best || candidate.score > best.score || candidate.score === best.score && turnPlanKey(candidate.moves) < turnPlanKey(best.moves)) {
      best = candidate;
    }
  }
  if (!best) {
    throw new Error("Full GPU search produced no legal result.");
  }
  return best;
}

async function bestReplyScoreOnGpu(device, snapshot, nodes) {
  const inputs = buildGpuCandidateInputsFromSnapshot(snapshot, snapshot.turn);
  if (inputs.sourceCount === 0 || inputs.targetCount === 0) {
    return 0;
  }
  const scored = await scoreCandidatesOnGpu(device, inputs, snapshot.turn);
  let best = 0;
  for (let index = 0; index < scored.scores.length; index += 1) {
    const score = scored.scores[index] ?? -2147483647;
    if (score > best) {
      best = score;
    }
  }
  return best;
}

function turnPlanKey(moves) {
  return moves.map((move) => [
    move.from.timelineId,
    move.from.time,
    move.from.x,
    move.from.y,
    move.to.timelineId,
    move.to.time,
    move.to.x,
    move.to.y
  ].join(":")).join("/");
}

let gpuDeadlineAt = 0;

async function scoreCandidatesOnGpu(device, inputs, turn) {
  const candidateCount = inputs.sourceCount * inputs.targetCount;
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const maxCandidatesPerBatch = Math.max(1, Math.floor(maxBindingSize / (GPU_CANDIDATE_STRIDE * Int32Array.BYTES_PER_ELEMENT)));
  if (inputs.targetCount > maxCandidatesPerBatch) {
    throw new Error(`GPU move generation target set is too large for this device (${inputs.targetCount} targets).`);
  }
  const targetBuffer = storageBuffer(device, inputs.targets, GPUBufferUsage.STORAGE);
  const boardBuffer = storageBuffer(device, inputs.boards ?? new Int32Array(GPU_BOARD_STRIDE), GPUBufferUsage.STORAGE);
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: GPU_MOVEGEN_SHADER }), entryPoint: "score_candidates" }
  });
  const records = new Int32Array(candidateCount * GPU_CANDIDATE_STRIDE);
  const scores = new Int32Array(candidateCount);
  const sourceBatchSize = Math.max(1, Math.floor(maxCandidatesPerBatch / inputs.targetCount));

  for (let sourceStart = 0; sourceStart < inputs.sourceCount; sourceStart += sourceBatchSize) {
    const sourceCount = Math.min(sourceBatchSize, inputs.sourceCount - sourceStart);
    const batchCandidateCount = sourceCount * inputs.targetCount;
    const sourceBuffer = storageBuffer(
      device,
      inputs.sources.subarray(sourceStart * GPU_SOURCE_STRIDE, (sourceStart + sourceCount) * GPU_SOURCE_STRIDE),
      GPUBufferUsage.STORAGE
    );
    const candidateBuffer = device.createBuffer({
      size: align4(batchCandidateCount * GPU_CANDIDATE_STRIDE * Int32Array.BYTES_PER_ELEMENT),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const scoreBuffer = device.createBuffer({
      size: align4(batchCandidateCount * Int32Array.BYTES_PER_ELEMENT),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const params = new ArrayBuffer(16);
    const view = new DataView(params);
    view.setUint32(0, sourceCount, true);
    view.setUint32(4, inputs.targetCount, true);
    view.setUint32(8, colorCode(turn), true);
    view.setUint32(12, inputs.boardCount ?? 0, true);
    const paramsBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
    const encoder = device.createCommandEncoder();
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [sourceBuffer, targetBuffer, candidateBuffer, scoreBuffer, paramsBuffer, boardBuffer]
        .map((buffer, binding) => ({ binding, resource: { buffer } }))
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(batchCandidateCount / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const [batchRecords, batchScores] = await Promise.all([
      readInts(device, candidateBuffer, batchCandidateCount * GPU_CANDIDATE_STRIDE * Int32Array.BYTES_PER_ELEMENT),
      readInts(device, scoreBuffer, batchCandidateCount * Int32Array.BYTES_PER_ELEMENT)
    ]);
    const candidateOffset = sourceStart * inputs.targetCount;
    records.set(batchRecords, candidateOffset * GPU_CANDIDATE_STRIDE);
    scores.set(batchScores, candidateOffset);
  }

  return { records, scores };
}

async function mutateRankedCandidatesOnGpu(device, inputs, allCandidateRecords, ranked, { readChildren = false } = {}) {
  const limit = Math.min(ranked.length, 64);
  if (limit === 0 || !inputs.mutationBoards || inputs.boardCount === 0) {
    return [];
  }
  const selected = ranked.slice(0, limit);
  const candidateRecords = pickCandidateRecords(allCandidateRecords, selected.map((entry) => entry.index));
  const candidateBuffer = storageBuffer(device, candidateRecords, GPUBufferUsage.STORAGE);
  const boardBuffer = storageBuffer(device, inputs.mutationBoards, GPUBufferUsage.STORAGE);
  const childBoardBuffer = device.createBuffer({
    size: align4(limit * GPU_MUTATION_CHILD_STRIDE * Int32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const statusBuffer = device.createBuffer({
    size: align4(limit * Int32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, limit, true);
  view.setUint32(4, inputs.boardCount, true);
  view.setUint32(8, candidateRecords[1] ?? 0, true);
  const paramsBuffer = storageBuffer(device, params, GPUBufferUsage.UNIFORM);
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: GPU_MUTATE_SHADER }), entryPoint: "mutate_candidates" }
  });
  const encoder = device.createCommandEncoder();
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [candidateBuffer, boardBuffer, childBoardBuffer, statusBuffer, paramsBuffer]
      .map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(limit / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const [statuses, childBoards] = await Promise.all([
    readInts(device, statusBuffer, limit * Int32Array.BYTES_PER_ELEMENT),
    readChildren
      ? readInts(device, childBoardBuffer, limit * GPU_MUTATION_CHILD_STRIDE * Int32Array.BYTES_PER_ELEMENT)
      : Promise.resolve(null)
  ]);
  return selected.map((entry, index) => ({
    ...entry,
    mutationStatus: statuses[index] ?? 0,
    childBoards: childBoards?.subarray(index * GPU_MUTATION_CHILD_STRIDE, (index + 1) * GPU_MUTATION_CHILD_STRIDE) ?? null
  }));
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
  return null;
}

function buildGpuCandidateInputsFromSnapshot(snapshot, color) {
  return buildGpuCandidateInputs(snapshot, color);
}

function snapshotWithGpuChildBoards(snapshot, childBoardRecords, mutationStatus, options = {}) {
  const { move = null, advanceTurn = true } = options;
  const records = [
    childBoardRecords.subarray(0, GPU_MUTATION_BOARD_STRIDE)
  ];
  if (mutationStatus === GPU_MUTATION_STATUS_BRANCH_OK || mutationStatus === GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
    records.push(childBoardRecords.subarray(GPU_MUTATION_BOARD_STRIDE, GPU_MUTATION_CHILD_STRIDE));
  }
  const childByTimeline = new Map();
  for (const record of records) {
    const timelineId = record[1];
    if (!childByTimeline.has(timelineId)) {
      childByTimeline.set(timelineId, []);
    }
    childByTimeline.get(timelineId).push(gpuMutationBoardRecordToSnapshot(record));
  }

  const timelines = snapshot.timelines.map((timeline) => {
    const children = childByTimeline.get(timeline.id) ?? [];
    if (children.length === 0) {
      return {
        ...timeline,
        boards: timeline.boards.map((board) => ({ ...board }))
      };
    }
    const oldBoards = timeline.boards.map((board) => ({ ...board, latest: false }));
    const boards = [...oldBoards, ...children.map((child) => ({
      ...child,
      timelineIndex: snapshot.timelines.indexOf(timeline),
      origin: move ? originForGpuChild(child, move) : child.origin
    }))];
    const latest = latestBoard({ ...timeline, boards });
    return {
      ...timeline,
      boardCount: boards.length,
      latestTime: latest.time,
      boards
    };
  });
  const boards = timelines.flatMap((timeline) => timeline.boards);
  return {
    ...snapshot,
    turn: advanceTurn ? colorFromCode(records[records.length - 1][3]) : snapshot.turn,
    timelines,
    boards
  };
}

function originForGpuChild(child, move) {
  const sourceAdvance = child.timelineId === move.from.timelineId && child.time === move.from.time + 1;
  return {
    type: sourceAdvance ? "source-advance" : "cross-board",
    from: { ...move.from },
    to: { ...move.to }
  };
}

function gpuMutationBoardRecordToSnapshot(record) {
  return {
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
    latest: true,
    originKind: record[10],
    squares: record.slice(12, 76)
  };
}

function gpuSnapshotToGame(snapshot) {
  return {
    turn: snapshot.turn,
    nextTimelineId: snapshot.nextTimelineId ?? 1,
    nextBlackTimelineId: snapshot.nextBlackTimelineId ?? -1,
    checkedRoyals: [],
    timelines: snapshot.timelines.map((timeline) => ({
      id: timeline.id,
      row: timeline.row,
      label: timeline.label ?? `T${timeline.id}`,
      owner: timeline.owner,
      boards: timeline.boards
        .map((board) => gpuBoardToGameBoard(board))
        .sort((left, right) => left.time - right.time)
    }))
  };
}

function gpuBoardToGameBoard(board) {
  if (board.board) {
    return {
      ...board,
      board: board.board.map((row) => row.map((piece) => piece ? { ...piece } : null))
    };
  }
  return {
    time: board.time,
    sideToMove: board.sideToMove,
    castling: board.castling,
    enPassant: board.enPassant,
    origin: board.origin,
    board: squaresToGameBoard(board.squares)
  };
}

function squaresToGameBoard(squares) {
  const board = [];
  for (let y = 0; y < 8; y += 1) {
    const row = [];
    for (let x = 0; x < 8; x += 1) {
      row.push(pieceFromCode(squares?.[y * 8 + x] ?? 0));
    }
    board.push(row);
  }
  return board;
}

function pieceFromCode(code) {
  const type = pieceTypeFromCode(code & 255);
  if (!type) {
    return null;
  }
  return {
    type,
    color: colorFromCode((code >> 8) & 255)
  };
}

function buildGpuCandidateInputs(game, color) {
  const sourceMeta = [];
  const targetMeta = [];
  const sources = [];
  const targets = [];
  const boards = [];
  const mutationBoards = [];
  const timelines = sortedTimelines(game);

  for (const timeline of timelines) {
    const latest = latestBoard(timeline);
    for (const board of timeline.boards) {
      const squares = squareCodesForBoard(board);
      const isLatest = board.time === latest?.time;
      pushGpuBoardRecord(boards, timeline, {
        time: board.time,
        sideToMove: board.sideToMove,
        squares
      });
      pushGpuMutationBoardRecord(mutationBoards, timeline, {
        time: board.time,
        sideToMove: board.sideToMove,
        castling: board.castling ?? 0,
        enPassant: board.enPassant ?? null,
        latest: isLatest,
        originKind: 0,
        squares
      });
      for (let y = 0; y < 8; y += 1) {
        for (let x = 0; x < 8; x += 1) {
          const code = squares[y * 8 + x] ?? 0;
          targetMeta.push({ timelineId: timeline.id, time: board.time, x, y });
          targets.push(
            code & 255,
            (code >> 8) & 255,
            timeline.id,
            board.time,
            x,
            y,
            timeline.row,
            colorCode(board.sideToMove),
            ownerCode(timeline.owner),
            isLatest ? 1 : 0
          );
        }
      }
      for (let y = 0; y < 8; y += 1) {
        for (let x = 0; x < 8; x += 1) {
          const code = squares[y * 8 + x] ?? 0;
          if ((code & 255) === 0) {
            continue;
          }
          sourceMeta.push({ timelineId: timeline.id, time: board.time, x, y });
          sources.push(
            code & 255,
            (code >> 8) & 255,
            timeline.id,
            board.time,
            x,
            y,
            timeline.row,
            colorCode(board.sideToMove),
            ownerCode(timeline.owner),
            isLatest ? 1 : 0
          );
        }
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
    boards: new Int32Array(boards),
    mutationBoards: new Int32Array(mutationBoards)
  };
}

function squareCodesForBoard(board) {
  if (board.squares) {
    return board.squares;
  }
  return board.board.flat().map((piece) => piece ? pieceTypeCode(piece.type) | (colorCode(piece.color) << 8) : 0);
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

function pushGpuMutationBoardRecord(out, timeline, board) {
  out.push(
    board.timelineIndex ?? 0,
    timeline.id,
    board.time,
    colorCode(board.sideToMove),
    board.castling ?? 0,
    board.enPassant?.x ?? -1,
    board.enPassant?.y ?? -1,
    board.enPassant?.capturedX ?? -1,
    board.enPassant?.capturedY ?? -1,
    board.latest ? 1 : 0,
    board.originKind ?? 0,
    0
  );
  for (let index = 0; index < 64; index += 1) {
    out.push(board.squares?.[index] ?? 0);
  }
}

function colorFromCode(code) {
  return code === 1 ? "black" : "white";
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

function pieceTypeFromCode(code) {
  return {
    1: "king",
    2: "commonKing",
    3: "queen",
    4: "royalQueen",
    5: "princess",
    6: "rook",
    7: "bishop",
    8: "unicorn",
    9: "dragon",
    10: "knight",
    11: "pawn",
    12: "brawn"
  }[code] ?? null;
}

function colorCode(color) {
  return color === "black" ? 1 : 0;
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
  const {
    id,
    type = "search",
    notation,
    turns,
    stagedMoves,
    game: clientGame,
    position,
    move,
    depth,
    nodes,
    timeMs,
    partitionIndex,
    partitionCount,
    gpuMode = "hybrid"
  } = event.data;

  try {
    const snapshotOverride = clientGame ? { ...clientGame, format: "json" } : null;
    if (!snapshotOverride) {
      throw new Error("GPU worker calculations require a client game snapshot.");
    }

    if (type === "legalTargets") {
      const selection = await legalTargetsOnGpu(position, snapshotOverride);
      self.postMessage({ id, ok: true, selection });
      return;
    }

    if (type === "applyMove") {
      const game = await applyMoveOnGpu(move, snapshotOverride);
      self.postMessage({ id, ok: true, game });
      return;
    }

    if (type === "submitTurn") {
      const status = await submitTurnOnGpu(snapshotOverride);
      self.postMessage({ id, ok: true, status });
      return;
    }

    const searchTimeMs = Math.max(1, timeMs ?? 10_000);
    gpuDeadlineAt = Date.now() + Math.max(1, Math.floor(searchTimeMs * 0.8));
    try {
      const gpuResult = await tryGpuSearch({ depth, nodes, timeMs: searchTimeMs, gpuMode, snapshotOverride });
      if (gpuResult?.status === "ok" && gpuResult.moves?.length) {
        self.postMessage({ id, ok: true, result: gpuResult, partitionIndex: partitionIndex ?? 0 });
        return;
      }
    } catch (gpuError) {
      console.debug?.("GPU search failed", gpuError);
      if (gpuMode === "full") {
        try {
          const hybridResult = await tryGpuSearch({ depth, nodes, timeMs: searchTimeMs, gpuMode: "hybrid", snapshotOverride });
          if (hybridResult?.status === "ok" && hybridResult.moves?.length) {
            self.postMessage({ id, ok: true, result: hybridResult, partitionIndex: partitionIndex ?? 0 });
            return;
          }
        } catch (hybridError) {
          console.debug?.("Hybrid GPU search failed", hybridError);
        }
      }
      throw gpuError;
    }

    throw new Error("GPU search did not produce a legal turn.");
  } catch (error) {
    self.postMessage({ id, ok: false, error: error.message, partitionIndex: partitionIndex ?? 0 });
  }
});
