export const GPU_TURN_STATUS_SHADER = `
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

export const GPU_MOVEGEN_SHADER = `
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

const BOARD_STRIDE: u32 = 69u;
const BOARD_SQUARE_OFFSET: u32 = 5u;
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
  if (piece_type == 1) { return 20000; }
  if (piece_type == 2) { return 10000; }
  if (piece_type == 3) { return 900; }
  if (piece_type == 4) { return 20000; }
  if (piece_type == 5) { return 700; }
  if (piece_type == 6) { return 500; }
  if (piece_type == 7) { return 330; }
  if (piece_type == 8) { return 500; }
  if (piece_type == 9) { return 900; }
  if (piece_type == 10) { return 320; }
  if (piece_type == 11) { return 100; }
  if (piece_type == 12) { return 130; }
  return 0;
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

fn legal_shape(piece_type: i32, color: i32, from_y: i32, dx: i32, dy: i32, dt: i32, dl: i32, same_board: bool, target_piece: i32, target_color: i32, castling: i32) -> bool {
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
  let has_moved = select(from_y != 1, from_y != 6, color == 1);

  if (piece_type == 1 || piece_type == 2) {
    if (piece_type == 1 && same_board && dy == 0 && dt == 0 && dl == 0 && ax == 2 && target_piece == 0) {
      if (color == 0 && from_y == 0 && dx == 2 && (castling & 1) != 0) { return true; }
      if (color == 0 && from_y == 0 && dx == -2 && (castling & 2) != 0) { return true; }
      if (color == 1 && from_y == 7 && dx == 2 && (castling & 4) != 0) { return true; }
      if (color == 1 && from_y == 7 && dx == -2 && (castling & 8) != 0) { return true; }
    }
    return ax <= 1 && ay <= 1 && at <= 1 && al <= 1;
  }
  if (piece_type == 10) {
    return (max(max(ax, ay), max(at, al)) == 2) && (ax + ay + at + al == 3);
  }
  if (piece_type == 11) {
    if (same_board && dx == 0 && dy == forward && target_piece == 0) { return true; }
    if (same_board && dx == 0 && dy == forward * 2 && !has_moved && target_piece == 0) { return true; }
    if (same_board && ax == 1 && dy == forward && target_piece != 0 && target_color != color) { return true; }
    if (!same_board && dx == 0 && dy == 0 && dt == 0 && (dl == timeline_forward || (dl == timeline_forward * 2 && !has_moved)) && target_piece == 0) { return true; }
    return at == 1 && dl == timeline_forward && dx == 0 && dy == 0 && target_piece != 0 && target_color != color;
  }
  if (piece_type == 12) {
    if (target_piece != 0 && changed >= 2 && ax <= 1 && ay <= 1 && at <= 1 && al <= 1 && (dy == forward || dl == timeline_forward) && dy != -forward && dl != -timeline_forward) {
      return true;
    }
    if (same_board && dx == 0 && dy == forward && target_piece == 0) { return true; }
    if (same_board && dx == 0 && dy == forward * 2 && !has_moved && target_piece == 0) { return true; }
    return !same_board && dx == 0 && dy == 0 && dt == 0 && (dl == timeline_forward || (dl == timeline_forward * 2 && !has_moved)) && target_piece == 0;
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

fn castling_path_clear(castling: i32, piece_type: i32, color: i32, from_row: i32, from_time: i32, from_x: i32, from_y: i32, dx: i32, dy: i32, dt: i32, dl: i32) -> bool {
  if (piece_type != 1 || dy != 0 || dt != 0 || dl != 0 || abs_i32(dx) != 2) {
    return true;
  }
  let expected_y = select(0, 7, color == 1);
  if (from_x != 4 || from_y != expected_y) {
    return false;
  }
  let rook_x = select(0, 7, dx > 0);
  let rook_code = square_code_at(from_row, from_time, rook_x, from_y);
  if ((rook_code & 255) != 6 || ((rook_code >> 8) & 255) != color) {
    return false;
  }
  if (color == 0 && dx == 2 && (castling & 1) == 0) { return false; }
  if (color == 0 && dx == -2 && (castling & 2) == 0) { return false; }
  if (color == 1 && dx == 2 && (castling & 4) == 0) { return false; }
  if (color == 1 && dx == -2 && (castling & 8) == 0) { return false; }
  let step = sign_i32(dx);
  var x = from_x + step;
  loop {
    if (x == rook_x) {
      break;
    }
    if (square_code_at(from_row, from_time, x, from_y) != 0) {
      return false;
    }
    x = x + step;
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
  let source_board_base_i32 = board_base_by_row_time(from_row, from_time);
  var source_castling = 0;
  if (source_board_base_i32 >= 0) {
    source_castling = boards[u32(source_board_base_i32) + 4u];
  }
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
  if (!legal_shape(piece_type, color, from_y, dx, dy, dt, dl, same_board, target_piece, target_color, source_castling)) {
    scores[index] = -2147483647;
    return;
  }
  if (!path_clear(piece_type, color, from_row, from_time, from_x, from_y, dx, dy, raw_dt, dt, dl)) {
    scores[index] = -2147483647;
    return;
  }
  if (!castling_path_clear(source_castling, piece_type, color, from_row, from_time, from_x, from_y, dx, dy, dt, dl)) {
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

export const GPU_REPLY_SHADER = `
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
  if (piece_type == 1) { return 20000; }
  if (piece_type == 2) { return 10000; }
  if (piece_type == 3) { return 900; }
  if (piece_type == 4) { return 20000; }
  if (piece_type == 5) { return 700; }
  if (piece_type == 6) { return 500; }
  if (piece_type == 7) { return 330; }
  if (piece_type == 8) { return 500; }
  if (piece_type == 9) { return 900; }
  if (piece_type == 10) { return 320; }
  if (piece_type == 11) { return 100; }
  if (piece_type == 12) { return 130; }
  return 0;
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

export const GPU_MUTATE_SHADER = `
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
    rights = rights & 12;
  }
  if (piece_type == 1 && color == 1) {
    rights = rights & 3;
  }
  if (piece_type == 6 && color == 0 && from_y == 0 && from_x == 0) {
    rights = rights & 13;
  }
  if (piece_type == 6 && color == 0 && from_y == 0 && from_x == 7) {
    rights = rights & 14;
  }
  if (piece_type == 6 && color == 1 && from_y == 7 && from_x == 0) {
    rights = rights & 7;
  }
  if (piece_type == 6 && color == 1 && from_y == 7 && from_x == 7) {
    rights = rights & 11;
  }
  let captured_type = target_code & 255;
  let captured_color = (target_code >> 8) & 255;
  if (captured_type == 6 && captured_color == 0 && to_y == 0 && to_x == 0) {
    rights = rights & 13;
  }
  if (captured_type == 6 && captured_color == 0 && to_y == 0 && to_x == 7) {
    rights = rights & 14;
  }
  if (captured_type == 6 && captured_color == 1 && to_y == 7 && to_x == 0) {
    rights = rights & 7;
  }
  if (captured_type == 6 && captured_color == 1 && to_y == 7 && to_x == 7) {
    rights = rights & 11;
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
