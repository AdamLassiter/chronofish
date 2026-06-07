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
