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