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

const BOARD_STRIDE: u32 = 73u;
const BOARD_EP: u32 = 5u;
const BOARD_SQUARE_OFFSET: u32 = 9u;
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

fn legal_shape(piece_type: i32, color: i32, from_y: i32, to_x: i32, to_y: i32, dx: i32, dy: i32, dt: i32, dl: i32, same_board: bool, target_piece: i32, target_color: i32, castling: i32, ep_x: i32, ep_y: i32) -> bool {
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
    if (same_board && ax == 1 && dy == forward && target_piece == 0 && ep_x == to_x && ep_y == to_y) { return true; }
    if (!same_board && dx == 0 && dy == 0 && dt == 0 && (dl == timeline_forward || (dl == timeline_forward * 2 && !has_moved)) && target_piece == 0) { return true; }
    return at == 1 && dl == timeline_forward && dx == 0 && dy == 0 && target_piece != 0 && target_color != color;
  }
  if (piece_type == 12) {
    if (target_piece != 0 && changed >= 2 && ax <= 1 && ay <= 1 && at <= 1 && al <= 1 && (dy == forward || dl == timeline_forward) && dy != -forward && dl != -timeline_forward) {
      return true;
    }
    if (same_board && dx == 0 && dy == forward && target_piece == 0) { return true; }
    if (same_board && dx == 0 && dy == forward * 2 && !has_moved && target_piece == 0) { return true; }
    if (same_board && ax == 1 && dy == forward && target_piece == 0 && ep_x == to_x && ep_y == to_y) { return true; }
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
  var source_ep_x = -1;
  var source_ep_y = -1;
  if (source_board_base_i32 >= 0) {
    let source_board_base = u32(source_board_base_i32);
    source_castling = boards[source_board_base + 4u];
    source_ep_x = boards[source_board_base + BOARD_EP];
    source_ep_y = boards[source_board_base + BOARD_EP + 1u];
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
  if (!legal_shape(piece_type, color, from_y, to_x, to_y, dx, dy, dt, dl, same_board, target_piece, target_color, source_castling, source_ep_x, source_ep_y)) {
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
