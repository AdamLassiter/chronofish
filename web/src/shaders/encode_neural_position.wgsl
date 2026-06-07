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
const MAX_BOARDS: u32 = 16;
const INPUT_SIZE: u32 = MAX_BOARDS * BOARD_PLANES * BOARD_SQUARES;

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
  if (index >= INPUT_SIZE) {
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