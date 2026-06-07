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