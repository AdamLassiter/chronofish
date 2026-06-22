struct Params {
  state_count: u32,
  max_boards: u32,
  state_stride: u32,
  board_offset: u32,
  candidate_capacity: u32,
  candidate_stride: u32,
  delta_stride: u32,
  root_color: i32,
  target_depth: i32,
  cycle_index: u32,
  dispatch_base: u32,
  dispatch_count: u32,
  _pad2: u32,
  _pad3: u32,
  _pad4: u32,
  _pad5: u32,
};

@group(0) @binding(0) var<storage, read> states: array<i32>;
@group(0) @binding(1) var<storage, read_write> candidates: array<i32>;
@group(0) @binding(2) var<storage, read_write> deltas: array<i32>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

override EXPAND_WORKGROUP_SIZE: u32 = 64u;

const HEADER_ROOT: u32 = 1u;
const HEADER_SCORE: u32 = 2u;
const HEADER_DEPTH: u32 = 3u;
const HEADER_TURN: u32 = 4u;
const HEADER_BOARD_COUNT: u32 = 5u;
const HEADER_TERMINAL: u32 = 8u;
const HEADER_NEXT_WHITE: u32 = 11u;
const HEADER_NEXT_BLACK: u32 = 12u;
const HEADER_PRESENT: u32 = 13u;
const HEADER_LAST_NEURAL: u32 = 15u;

const BOARD_STRIDE: u32 = 78u;
const BOARD_TIMELINE: u32 = 0u;
const BOARD_ROW: u32 = 1u;
const BOARD_OWNER: u32 = 2u;
const BOARD_TIME: u32 = 3u;
const BOARD_SIDE: u32 = 4u;
const BOARD_CASTLING: u32 = 5u;
const BOARD_EP: u32 = 6u;
const BOARD_LATEST: u32 = 10u;
const BOARD_ORIGIN: u32 = 11u;
const BOARD_SQUARES: u32 = 12u;
const BOARD_ACTIVE: u32 = 76u;
const BOARD_PENDING: u32 = 77u;

const CANDIDATE_PARENT: u32 = 0u;
const CANDIDATE_ROOT: u32 = 1u;
const CANDIDATE_SCORE: u32 = 2u;
const CANDIDATE_LAST_NEURAL: u32 = 3u;
const CANDIDATE_DEPTH: u32 = 4u;
const CANDIDATE_STATUS: u32 = 5u;
const CANDIDATE_MOVE: u32 = 8u;
const CANDIDATE_DELTA_COUNT: u32 = 16u;
const CANDIDATE_TERMINAL: u32 = 18u;
const CANDIDATE_CARRY: u32 = 19u;
const CANDIDATE_NODE_ID: u32 = 20u;
const CANDIDATE_POLICY_PRIOR: u32 = 21u;

fn abs_i32(value: i32) -> i32 { return select(value, -value, value < 0); }
fn sign_i32(value: i32) -> i32 { return select(select(0, 1, value > 0), -1, value < 0); }

fn state_base(state: u32) -> u32 { return state * params.state_stride; }
fn board_base(state: u32, board: u32) -> u32 { return state_base(state) + params.board_offset + board * BOARD_STRIDE; }

fn piece_value(piece_type: i32) -> i32 {
  if (piece_type == 1 || piece_type == 4) { return 20000; }
  if (piece_type == 2) { return 10000; }
  if (piece_type == 3 || piece_type == 9) { return 900; }
  if (piece_type == 5) { return 700; }
  if (piece_type == 6 || piece_type == 8) { return 500; }
  if (piece_type == 7) { return 330; }
  if (piece_type == 10) { return 320; }
  if (piece_type == 11) { return 100; }
  if (piece_type == 12) { return 130; }
  return 0;
}

fn same_distance(a: i32, b: i32, c: i32, d: i32, count: i32) -> bool {
  var first = 0;
  if (a > 0) { first = a; }
  if (first == 0 && b > 0) { first = b; }
  if (first == 0 && c > 0) { first = c; }
  if (first == 0 && d > 0) { first = d; }
  if (first == 0 || count == 0) { return false; }
  return (a == 0 || a == first) && (b == 0 || b == first) && (c == 0 || c == first) && (d == 0 || d == first);
}

fn legal_shape(piece: i32, color: i32, from_y: i32, to_x: i32, to_y: i32, dx: i32, dy: i32, dt: i32, dl: i32, same_board: bool, target_piece: i32, target_color: i32, castling: i32, ep_x: i32, ep_y: i32) -> bool {
  let ax = abs_i32(dx); let ay = abs_i32(dy); let at = abs_i32(dt); let al = abs_i32(dl);
  let changed = select(0, 1, ax > 0) + select(0, 1, ay > 0) + select(0, 1, at > 0) + select(0, 1, al > 0);
  if (changed == 0) { return false; }
  let forward = select(1, -1, color == 1);
  let has_moved = select(from_y != 1, from_y != 6, color == 1);
  if (piece == 1 || piece == 2) {
    if (piece == 1 && same_board && dy == 0 && dt == 0 && dl == 0 && ax == 2 && target_piece == 0) {
      if (color == 0 && from_y == 0 && dx == 2 && (castling & 1) != 0) { return true; }
      if (color == 0 && from_y == 0 && dx == -2 && (castling & 2) != 0) { return true; }
      if (color == 1 && from_y == 7 && dx == 2 && (castling & 4) != 0) { return true; }
      if (color == 1 && from_y == 7 && dx == -2 && (castling & 8) != 0) { return true; }
    }
    return ax <= 1 && ay <= 1 && at <= 1 && al <= 1;
  }
  if (piece == 10) { return max(max(ax, ay), max(at, al)) == 2 && ax + ay + at + al == 3; }
  if (piece == 11) {
    if (same_board && dx == 0 && dy == forward && target_piece == 0) { return true; }
    if (same_board && dx == 0 && dy == forward * 2 && !has_moved && target_piece == 0) { return true; }
    if (same_board && ax == 1 && dy == forward && target_piece != 0 && target_color != color) { return true; }
    if (same_board && ax == 1 && dy == forward && target_piece == 0 && ep_x == to_x && ep_y == to_y) { return true; }
    if (!same_board && dx == 0 && dy == 0 && dt == 0 && (dl == forward || (dl == forward * 2 && !has_moved)) && target_piece == 0) { return true; }
    return at == 1 && dl == forward && dx == 0 && dy == 0 && target_piece != 0 && target_color != color;
  }
  if (piece == 12) {
    if (target_piece != 0 && changed >= 2 && ax <= 1 && ay <= 1 && at <= 1 && al <= 1 && (dy == forward || dl == forward) && dy != -forward && dl != -forward) { return true; }
    if (same_board && ax == 1 && dy == forward && target_piece == 0 && ep_x == to_x && ep_y == to_y) { return true; }
    if (same_board && dx == 0 && dy == forward && target_piece == 0) { return true; }
    if (same_board && dx == 0 && dy == forward * 2 && !has_moved && target_piece == 0) { return true; }
    return !same_board && dx == 0 && dy == 0 && dt == 0 && (dl == forward || (dl == forward * 2 && !has_moved)) && target_piece == 0;
  }
  if (piece == 6) { return changed == 1; }
  if (piece == 7) { return changed == 2 && same_distance(ax, ay, at, al, changed); }
  if (piece == 8) { return changed == 3 && same_distance(ax, ay, at, al, changed); }
  if (piece == 9) { return changed == 4 && same_distance(ax, ay, at, al, changed); }
  if (piece == 5) { return changed == 1 || (changed == 2 && same_distance(ax, ay, at, al, changed)); }
  return (piece == 3 || piece == 4) && same_distance(ax, ay, at, al, changed);
}

fn find_board_by_row_time(state: u32, row: i32, time: i32) -> i32 {
  let count = u32(max(0, states[state_base(state) + HEADER_BOARD_COUNT]));
  for (var index = 0u; index < count; index = index + 1u) {
    let base = board_base(state, index);
    if (states[base + BOARD_ROW] == row && states[base + BOARD_TIME] == time) { return i32(index); }
  }
  return -1;
}

fn square_at(state: u32, row: i32, time: i32, x: i32, y: i32) -> i32 {
  if (x < 0 || x >= 8 || y < 0 || y >= 8) { return -1; }
  let board = find_board_by_row_time(state, row, time);
  if (board < 0) { return -1; }
  return states[board_base(state, u32(board)) + BOARD_SQUARES + u32(y * 8 + x)];
}

fn board_side_at(state: u32, row: i32, time: i32) -> i32 {
  let board = find_board_by_row_time(state, row, time);
  return select(states[board_base(state, u32(max(0, board))) + BOARD_SIDE], -1, board < 0);
}

fn path_clear(state: u32, piece: i32, color: i32, row: i32, time: i32, x: i32, y: i32, dx: i32, dy: i32, raw_dt: i32, dt: i32, dl: i32) -> bool {
  let sliding = piece == 3 || piece == 4 || piece == 5 || piece == 6 || piece == 7 || piece == 8 || piece == 9;
  if (!sliding && piece != 11 && piece != 12) { return true; }
  let distance = max(max(abs_i32(dx), abs_i32(dy)), max(abs_i32(dt), abs_i32(dl)));
  if (distance <= 1) { return true; }
  let step_x = sign_i32(dx); let step_y = sign_i32(dy); let step_t = raw_dt / distance; let step_l = sign_i32(dl);
  for (var step = 1; step < distance; step = step + 1) {
    let check_row = row + step_l * step; let check_time = time + step_t * step;
    if (step_t != 0 && board_side_at(state, check_row, check_time) != color) { continue; }
    if (square_at(state, check_row, check_time, x + step_x * step, y + step_y * step) != 0) { return false; }
  }
  return true;
}

fn castling_clear(state: u32, piece: i32, color: i32, row: i32, time: i32, x: i32, y: i32, dx: i32, dy: i32, dt: i32, dl: i32, castling: i32) -> bool {
  if (piece != 1 || dy != 0 || dt != 0 || dl != 0 || abs_i32(dx) != 2) { return true; }
  let expected_y = select(0, 7, color == 1); let rook_x = select(0, 7, dx > 0);
  if (x != 4 || y != expected_y) { return false; }
  let rook = square_at(state, row, time, rook_x, y);
  if ((rook & 255) != 6 || ((rook >> 8) & 255) != color) { return false; }
  if (color == 0 && dx == 2 && (castling & 1) == 0) { return false; }
  if (color == 0 && dx == -2 && (castling & 2) == 0) { return false; }
  if (color == 1 && dx == 2 && (castling & 4) == 0) { return false; }
  if (color == 1 && dx == -2 && (castling & 8) == 0) { return false; }
  let step = sign_i32(dx); var check_x = x + step;
  loop { if (check_x == rook_x) { break; } if (square_at(state, row, time, check_x, y) != 0) { return false; } check_x = check_x + step; }
  return true;
}

fn update_castling(rights_in: i32, piece: i32, color: i32, from_x: i32, from_y: i32, to_x: i32, to_y: i32, captured: i32) -> i32 {
  var rights = rights_in;
  if (piece == 1 && color == 0) { rights = rights & 12; }
  if (piece == 1 && color == 1) { rights = rights & 3; }
  if (piece == 6 && color == 0 && from_y == 0 && from_x == 0) { rights = rights & 13; }
  if (piece == 6 && color == 0 && from_y == 0 && from_x == 7) { rights = rights & 14; }
  if (piece == 6 && color == 1 && from_y == 7 && from_x == 0) { rights = rights & 7; }
  if (piece == 6 && color == 1 && from_y == 7 && from_x == 7) { rights = rights & 11; }
  let captured_piece = captured & 255; let captured_color = (captured >> 8) & 255;
  if (captured_piece == 6 && captured_color == 0 && to_y == 0 && to_x == 0) { rights = rights & 13; }
  if (captured_piece == 6 && captured_color == 0 && to_y == 0 && to_x == 7) { rights = rights & 14; }
  if (captured_piece == 6 && captured_color == 1 && to_y == 7 && to_x == 0) { rights = rights & 7; }
  if (captured_piece == 6 && captured_color == 1 && to_y == 7 && to_x == 7) { rights = rights & 11; }
  return rights;
}

fn next_branch_row(state: u32, source_row: i32, color: i32) -> i32 {
  let direction = select(1, -1, color == 1); var row = source_row + direction;
  let count = u32(max(0, states[state_base(state) + HEADER_BOARD_COUNT]));
  loop {
    var occupied = false;
    for (var index = 0u; index < count; index = index + 1u) { if (states[board_base(state, index) + BOARD_ROW] == row) { occupied = true; } }
    if (!occupied) { return row; }
    row = row + direction;
  }
  return row;
}

fn copy_board(source: u32, destination: u32) { for (var word = 0u; word < BOARD_STRIDE; word = word + 1u) { deltas[destination + word] = states[source + word]; } }

fn write_carry(state: u32) {
  let slot = atomicAdd(&counters[0], 1u);
  if (slot >= params.candidate_capacity) { atomicStore(&counters[2], 1u); return; }
  let base = slot * params.candidate_stride; let parent = state_base(state);
  candidates[base + CANDIDATE_PARENT] = i32(state);
  candidates[base + CANDIDATE_ROOT] = states[parent + HEADER_ROOT];
  candidates[base + CANDIDATE_SCORE] = states[parent + HEADER_SCORE];
  candidates[base + CANDIDATE_LAST_NEURAL] = states[parent + HEADER_LAST_NEURAL];
  candidates[base + CANDIDATE_DEPTH] = states[parent + HEADER_DEPTH];
  candidates[base + CANDIDATE_STATUS] = 0;
  candidates[base + CANDIDATE_DELTA_COUNT] = 0;
  candidates[base + CANDIDATE_TERMINAL] = states[parent + HEADER_TERMINAL];
  candidates[base + CANDIDATE_CARRY] = 1;
  candidates[base + CANDIDATE_NODE_ID] = 0;
  candidates[base + CANDIDATE_POLICY_PRIOR] = 0;
}

fn write_candidate(state: u32, source_board: u32, target_board: u32, from_x: i32, from_y: i32, to_x: i32, to_y: i32, piece: i32, color: i32, captured: i32, heuristic: i32) {
  let slot = atomicAdd(&counters[0], 1u);
  atomicAdd(&counters[3], 1u);
  if (slot >= params.candidate_capacity) { atomicStore(&counters[2], 1u); return; }
  let state_offset = state_base(state); let source = board_base(state, source_board); let target_base = board_base(state, target_board);
  let candidate = slot * params.candidate_stride; let delta = slot * params.delta_stride;
  let same_board = source_board == target_board; let target_latest = states[target_base + BOARD_LATEST] != 0;
  let terminal = (captured & 255) == 1 || (captured & 255) == 4;
  let historical_branch = !same_board && !target_latest;
  let status = select(select(1, 3, historical_branch), select(2, 4, historical_branch), terminal);
  candidates[candidate + CANDIDATE_PARENT] = i32(state);
  let inherited_root = states[state_offset + HEADER_ROOT];
  candidates[candidate + CANDIDATE_ROOT] = select(inherited_root, i32(slot + 1u), states[state_offset + HEADER_DEPTH] == 0 && inherited_root == 0);
  let perspective = select(-1, 1, states[state_offset + HEADER_TURN] == params.root_color);
  candidates[candidate + CANDIDATE_SCORE] = states[state_offset + HEADER_SCORE] - states[state_offset + HEADER_LAST_NEURAL] + heuristic * perspective;
  candidates[candidate + CANDIDATE_LAST_NEURAL] = 0;
  candidates[candidate + CANDIDATE_DEPTH] = states[state_offset + HEADER_DEPTH];
  candidates[candidate + CANDIDATE_STATUS] = status;
  candidates[candidate + CANDIDATE_MOVE + 0u] = states[source + BOARD_TIMELINE];
  candidates[candidate + CANDIDATE_MOVE + 1u] = states[source + BOARD_TIME];
  candidates[candidate + CANDIDATE_MOVE + 2u] = from_x;
  candidates[candidate + CANDIDATE_MOVE + 3u] = from_y;
  candidates[candidate + CANDIDATE_MOVE + 4u] = states[target_base + BOARD_TIMELINE];
  candidates[candidate + CANDIDATE_MOVE + 5u] = states[target_base + BOARD_TIME];
  candidates[candidate + CANDIDATE_MOVE + 6u] = to_x;
  candidates[candidate + CANDIDATE_MOVE + 7u] = to_y;
  candidates[candidate + CANDIDATE_DELTA_COUNT] = select(2, 1, same_board);
  candidates[candidate + CANDIDATE_TERMINAL] = select(0, 1, terminal);
  candidates[candidate + CANDIDATE_CARRY] = 0;
  candidates[candidate + CANDIDATE_NODE_ID] = i32(params.cycle_index * params.candidate_capacity + slot + 1u);
  candidates[candidate + CANDIDATE_POLICY_PRIOR] = 0;

  copy_board(source, delta);
  let source_square = BOARD_SQUARES + u32(from_y * 8 + from_x); let target_square = BOARD_SQUARES + u32(to_y * 8 + to_x);
  let next_turn = 1 - color; let placed_piece = select(piece, 3, (piece == 11 || piece == 12) && ((color == 0 && to_y == 7) || (color == 1 && to_y == 0)));
  let placed = placed_piece | (color << 8);
  deltas[delta + BOARD_TIME] = states[source + BOARD_TIME] + 1;
  deltas[delta + BOARD_SIDE] = next_turn;
  deltas[delta + BOARD_LATEST] = 1;
  deltas[delta + BOARD_ORIGIN] = 1;
  deltas[delta + BOARD_ACTIVE] = 0;
  deltas[delta + BOARD_PENDING] = 0;
  deltas[delta + BOARD_EP] = -1; deltas[delta + BOARD_EP + 1u] = -1; deltas[delta + BOARD_EP + 2u] = -1; deltas[delta + BOARD_EP + 3u] = -1;
  deltas[delta + source_square] = 0;
  if (same_board) {
    deltas[delta + target_square] = placed;
    deltas[delta + BOARD_CASTLING] = update_castling(states[source + BOARD_CASTLING], piece, color, from_x, from_y, to_x, to_y, captured);
    if ((piece == 11 || piece == 12) && from_x != to_x && (captured & 255) == 0) {
      let ep_x = states[source + BOARD_EP]; let ep_y = states[source + BOARD_EP + 1u];
      if (ep_x == to_x && ep_y == to_y) { let captured_x = states[source + BOARD_EP + 2u]; let captured_y = states[source + BOARD_EP + 3u]; deltas[delta + BOARD_SQUARES + u32(captured_y * 8 + captured_x)] = 0; }
    }
    if (piece == 1 && abs_i32(to_x - from_x) == 2) {
      let rook_from = select(0, 7, to_x > from_x); let rook_to = select(3, 5, to_x > from_x);
      let rook_square = BOARD_SQUARES + u32(from_y * 8 + rook_from); let rook = states[source + rook_square];
      deltas[delta + rook_square] = 0; deltas[delta + BOARD_SQUARES + u32(from_y * 8 + rook_to)] = rook;
    }
    if ((piece == 11 || piece == 12) && from_x == to_x && abs_i32(to_y - from_y) == 2) {
      deltas[delta + BOARD_EP] = from_x; deltas[delta + BOARD_EP + 1u] = from_y + select(1, -1, color == 1); deltas[delta + BOARD_EP + 2u] = to_x; deltas[delta + BOARD_EP + 3u] = to_y;
    }
    return;
  }

  let target_delta = delta + BOARD_STRIDE; copy_board(target_base, target_delta);
  deltas[target_delta + BOARD_TIME] = states[target_base + BOARD_TIME] + 1;
  deltas[target_delta + BOARD_SIDE] = next_turn;
  deltas[target_delta + BOARD_LATEST] = 1;
  deltas[target_delta + BOARD_ORIGIN] = select(3, 2, historical_branch);
  deltas[target_delta + BOARD_ACTIVE] = 0;
  deltas[target_delta + BOARD_PENDING] = 0;
  deltas[target_delta + BOARD_EP] = -1; deltas[target_delta + BOARD_EP + 1u] = -1; deltas[target_delta + BOARD_EP + 2u] = -1; deltas[target_delta + BOARD_EP + 3u] = -1;
  deltas[target_delta + target_square] = placed;
  deltas[target_delta + BOARD_CASTLING] = update_castling(states[target_base + BOARD_CASTLING], piece, color, from_x, from_y, to_x, to_y, captured);
  if (historical_branch) {
    deltas[target_delta + BOARD_TIMELINE] = select(states[state_offset + HEADER_NEXT_WHITE], states[state_offset + HEADER_NEXT_BLACK], color == 1);
    deltas[target_delta + BOARD_ROW] = next_branch_row(state, states[source + BOARD_ROW], color);
    deltas[target_delta + BOARD_OWNER] = select(1, 2, color == 1);
  }
}

@compute @workgroup_size(EXPAND_WORKGROUP_SIZE)
fn expand_frontier(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.dispatch_count) { return; }
  let source_index = params.dispatch_base + id.x;
  let sources_per_state = params.max_boards * 64u; let state = source_index / sources_per_state;
  if (state >= params.state_count) { return; }
  let state_offset = state_base(state);
  let source_slot = source_index % sources_per_state; let source_board = source_slot / 64u; let source_square = source_slot % 64u;
  if (states[state_offset + HEADER_TERMINAL] != 0 || states[state_offset + HEADER_DEPTH] >= params.target_depth) {
    if (source_slot == 0u) { write_carry(state); }
    return;
  }
  let board_count = u32(max(0, states[state_offset + HEADER_BOARD_COUNT]));
  if (source_board >= board_count) { return; }
  let source = board_base(state, source_board); let turn = states[state_offset + HEADER_TURN];
  if (states[source + BOARD_PENDING] == 0 || states[source + BOARD_SIDE] != turn) { return; }
  let code = states[source + BOARD_SQUARES + source_square]; let piece = code & 255; let color = (code >> 8) & 255;
  if (piece == 0 || color != turn) { return; }
  let from_x = i32(source_square % 8u); let from_y = i32(source_square / 8u);
  for (var target_board = 0u; target_board < board_count; target_board = target_board + 1u) {
    let target_base = board_base(state, target_board); let same_board = source_board == target_board;
    if (!same_board && states[target_base + BOARD_SIDE] != turn) { continue; }
    for (var target_square = 0u; target_square < 64u; target_square = target_square + 1u) {
      let captured = states[target_base + BOARD_SQUARES + target_square]; let target_color = (captured >> 8) & 255;
      if ((captured & 255) != 0 && target_color == color) { continue; }
      let to_x = i32(target_square % 8u); let to_y = i32(target_square / 8u); let dx = to_x - from_x; let dy = to_y - from_y;
      let raw_dt = states[target_base + BOARD_TIME] - states[source + BOARD_TIME]; var dt = raw_dt;
      if (!same_board && dt % 2 == 0) { dt = dt / 2; }
      let dl = states[target_base + BOARD_ROW] - states[source + BOARD_ROW];
      let castling = states[source + BOARD_CASTLING];
      if (!legal_shape(piece, color, from_y, to_x, to_y, dx, dy, dt, dl, same_board, captured & 255, target_color, castling, states[source + BOARD_EP], states[source + BOARD_EP + 1u])) { continue; }
      if (!path_clear(state, piece, color, states[source + BOARD_ROW], states[source + BOARD_TIME], from_x, from_y, dx, dy, raw_dt, dt, dl)) { continue; }
      if (!castling_clear(state, piece, color, states[source + BOARD_ROW], states[source + BOARD_TIME], from_x, from_y, dx, dy, dt, dl, castling)) { continue; }
      let centrality = 14 - (abs_i32(2 * to_x - 7) + abs_i32(2 * to_y - 7));
      let heuristic = piece_value(captured & 255) * 16 + centrality * 4 - select(25, 0, same_board) + piece_value(piece) / 64;
      write_candidate(state, source_board, target_board, from_x, from_y, to_x, to_y, piece, color, captured, heuristic);
    }
  }
}
