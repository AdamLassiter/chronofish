import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const repoRoot = path.resolve(root, "..");
const searchShaderRoot = path.join(repoRoot, "engine/src/gpu/search/shaders");
const trainingShaderRoot = path.join(repoRoot, "engine/src/gpu/training/shaders");

async function fileExists(file) {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}

test("GPU frontier uses parent-plus-delta candidates and pooled retained states", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const layout = await readFile(path.join(root, "src/ai-layout.ts"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(layout, /GPU_CANDIDATE_STRIDE = 24/);
  assert.match(layout, /GPU_SOURCE_STRIDE = 10/);
  assert.match(layout, /GPU_TARGET_STRIDE = 10/);
  assert.match(layout, /GPU_BOARD_STRIDE = 73/);
  assert.match(layout, /GPU_MUTATION_BOARD_STRIDE = 76/);
  assert.match(layout, /GPU_MUTATION_CHILD_STRIDE = GPU_MUTATION_BOARD_STRIDE \* 2/);
  assert.match(layout, /GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE = 4/);
  assert.match(layout, /GPU_FRONTIER_CANDIDATE_STRIDE/);
  assert.match(layout, /GPU_FRONTIER_BOARD_STRIDE = 78/);
  assert.match(layout, /GPU_FRONTIER_BOARD_ACTIVE = 76/);
  assert.match(layout, /GPU_FRONTIER_BOARD_PENDING = 77/);
  assert.match(layout, /GPU_FRONTIER_DELTA_STRIDE = GPU_FRONTIER_BOARD_STRIDE \* 2/);
  assert.match(engineSearch, /pub const GPU_CANDIDATE_STRIDE: usize = 24/);
  assert.match(engineSearch, /pub const GPU_SOURCE_STRIDE: usize = 10/);
  assert.match(engineSearch, /pub const GPU_TARGET_STRIDE: usize = 10/);
  assert.match(engineSearch, /pub const GPU_BOARD_STRIDE: usize = 73/);
  assert.match(engineSearch, /pub const GPU_MUTATION_BOARD_STRIDE: usize = 76/);
  assert.match(engineSearch, /pub const GPU_MUTATION_CHILD_STRIDE: usize = GPU_MUTATION_BOARD_STRIDE \* 2/);
  assert.match(engineSearch, /pub const GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE: i32 = 4/);
  assert.match(source, /class FrontierBufferPool/);
  assert.match(source, /deriveFrontierTuning/);
  assert.doesNotMatch(source, /encodeFrontierRoot/);
  assert.doesNotMatch(source, /function timelineActive/);
  assert.doesNotMatch(source, /function hashFrontierWords/);
  assert.doesNotMatch(source, /function originCode/);
  assert.match(worker, /engineFrontierRootFromSnapshot\(snapshot, tuning\.maxBoards\)/);
  assert.match(worker, /chronofish_frontier_root_snapshot_bytes\(ptr, len, maxBoards\)/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_frontier_root_snapshot_bytes/);
  assert.match(engineTypes, /chronofish_frontier_root_snapshot_bytes\(ptr: number, length: number, maxBoards: number\): number/);
  assert.match(engineSearch, /pub fn gpu_frontier_root_i32s_from_snapshot_json/);
  assert.match(engineSearch, /fn encode_frontier_root_from_timelines/);
  assert.match(engineSearch, /gpu_frontier_active_timeline_distance\(&ids\)/);
  assert.match(engineSearch, /gpu_frontier_present_time\(&active_board_summaries\)/);
  assert.match(engineSearch, /gpu_frontier_pending_board_count\(&active_board_summaries, present, root_color\)/);
  assert.match(source, /maxStorageBufferBindingSize/);
  assert.match(source, /entry\.buffer\.destroy\(\)/);
  assert.match(engineSearch, /pub fn gpu_frontier_hash_words/);
  assert.match(engineSearch, /pub fn gpu_frontier_origin_code/);
  assert.match(engineSearch, /pub fn gpu_frontier_active_timeline_distance/);
  assert.match(engineSearch, /pub fn gpu_frontier_timeline_active/);
  assert.match(engineSearch, /pub struct GpuFrontierActiveBoard/);
  assert.match(engineSearch, /pub fn gpu_frontier_present_time/);
  assert.match(engineSearch, /pub fn gpu_frontier_pending_board_count/);
  assert.match(engineSearch, /fn hash_frontier_words\(words: &\[i32\]\) -> \(i32, i32\) \{\s*gpu_frontier_hash_words\(words\)\s*\}/s);
  assert.match(engineSearch, /gpu_frontier_active_timeline_distance\(&ids\)/);
  assert.match(engineSearch, /gpu_frontier_timeline_active\(\s*gpu_search_owner_from_code\(timeline\.owner\),\s*timeline\.id,\s*active_distance,\s*\)/s);
  assert.match(engineSearch, /let active_board_summaries = timelines/);
  assert.match(engineSearch, /gpu_frontier_present_time\(&active_board_summaries\)/);
  assert.match(engineSearch, /gpu_frontier_pending_board_count\(&active_board_summaries, present, root_color\)/);
  assert.match(engineSearch, /origin_kind: origin_code\(&board\.origin\)/);
  assert.match(engineSearch, /gpu_frontier_origin_code\(board\.origin\.as_ref\(\)\.map\(\|origin\| origin\.kind\.as_str\(\)\)\)/);
});

test("GPU frontier pruning is deterministic and diversity bounded", async () => {
  const shader = await readFile(path.join(searchShaderRoot, "frontier_select.wgsl"), "utf8");
  const expand = await readFile(path.join(searchShaderRoot, "frontier_expand.wgsl"), "utf8");
  const policy = await readFile(path.join(searchShaderRoot, "frontier_policy.wgsl"), "utf8");

  assert.match(shader, /fn hash_candidates/);
  assert.match(shader, /fn bucket_order/);
  assert.match(shader, /fn bitonic_sort/);
  assert.match(shader, /fn mark_unique/);
  assert.match(shader, /fn mark_parent_quota/);
  assert.match(shader, /fn compact_selected/);
  assert.match(shader, /fn fill_selection_underflow/);
  assert.match(shader, /already_selected/);
  assert.match(shader, /parent_selected_count/);
  assert.match(shader, /parent_intent_selected_count/);
  assert.match(shader, /fn puct_score/);
  assert.match(shader, /CANDIDATE_POLICY_PRIOR/);
  assert.match(shader, /sqrt\(parent_visits\)/);
  assert.match(shader, /params\.puct_scale/);
  assert.match(shader, /CANDIDATE_MOVE/);
  assert.match(shader, /CANDIDATE_TACTICAL_PRIORITY/);
  assert.match(shader, /CANDIDATE_INTENT/);
  assert.match(shader, /fn intent_cap/);
  assert.match(shader, /INTENT_CHECK_ROYAL/);
  assert.match(shader, /intent_count >= intent_cap\(intent\)/);
  assert.match(shader, /left_tactical != right_tactical/);
  assert.match(shader, /return left_tactical > right_tactical/);
  assert.match(shader, /let left_score = puct_score\(left\)/);
  assert.match(expand, /fn move_checks_royal_after_move/);
  assert.match(expand, /path_clear_after_move/);
  assert.match(expand, /let tactical_priority = select/);
  assert.match(expand, /let intent = select/);
  assert.match(expand, /INTENT_CREATE_TIMELINE/);
  assert.match(expand, /INTENT_AFFECT_PRESENT/);
  assert.match(expand, /INTENT_QUIET_TEMPORAL/);
  assert.match(expand, /royal_check/);
  assert.match(expand, /captured_value >= 900/);
  assert.match(expand, /CANDIDATE_TACTICAL_PRIORITY\] = tactical_priority/);
  assert.match(expand, /CANDIDATE_INTENT\] = intent/);
  assert.match(policy, /CANDIDATE_TACTICAL_PRIORITY/);
  assert.match(policy, /prior < 0/);
});

test("GPU frontier pipelines use explicit superset bind layouts", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");

  assert.match(source, /frontierPipelineLayouts/);
  assert.match(source, /frontier_select\.layout/);
  assert.match(source, /storageLayout\(8, "storage"\)/);
  assert.match(source, /device\.createPipelineLayout/);
  assert.match(source, /selectionPassBuffers/);
  assert.match(source, /buffers\.eligibility/);
  assert.match(source, /frontier_bitonic_sort/);
  assert.doesNotMatch(source, /selectionBuffers\.slice\(0, 7\)/);
  assert.doesNotMatch(source, /layout: "auto", compute/);
});

test("GPU frontier reports scoped validation failures by cycle", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");

  assert.match(worker, /pushGpuValidationScope/);
  assert.match(worker, /popGpuValidationScope\(device, validationScope, `GPU frontier cycle \$\{cycle\}`\)/);
  assert.match(worker, /GPU frontier minimax reduction/);
  assert.match(source, /\.bindGroup/);
  assert.match(source, /beginComputePass\(\{ label \}\)/);
});

test("GPU frontier clears transient buffers before each expansion cycle", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");

  assert.match(source, /encoder\.clearBuffer\(buffers\.candidates\)/);
  assert.match(source, /encoder\.clearBuffer\(buffers\.deltas\)/);
  assert.match(source, /encoder\.clearBuffer\(buffers\.order\)/);
  assert.match(source, /encoder\.clearBuffer\(buffers\.selected\)/);
});

test("GPU frontier chunks expansion dispatches by tuned adapter limit", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const shader = await readFile(path.join(searchShaderRoot, "frontier_expand.wgsl"), "utf8");

  assert.match(source, /dispatchCandidateLimit/);
  assert.match(source, /for \(let base = 0; base < sourceScans; base \+= sourceScanLimit\)/);
  assert.match(source, /Math\.ceil\(count \/ this\.tuning\.candidateWorkgroupSize\)/);
  assert.match(source, /const selectionPlan = await frontierSelectionPlan\(this\.tuning, options\.maxSelectionScan\)/);
  assert.match(source, /pipelines\.bucketOrder/);
  assert.match(shader, /dispatch_base/);
  assert.match(shader, /dispatch_count/);
  assert.match(shader, /source_index = params\.dispatch_base \+ id\.x/);
});

test("GPU frontier board capacity scales with search growth instead of always reserving 64 boards", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");

  assert.match(source, /additionalBoardCapacity = 0/);
  assert.match(source, /chronofish_derive_frontier_tuning_json/);
  assert.doesNotMatch(source, /nextPowerOfTwo\(Math\.max\(boardCount, boardCount \+ Math\.max\(0, additionalBoardCapacity\)\)\)/);
  assert.match(engineSearch, /board_count\.max\(board_count\.saturating_add\(additional_board_capacity\)\)/);
  assert.match(wasmApi, /additional_board_capacity: usize/);
  assert.match(worker, /maxCycles \* 2/);
});

test("GPU frontier materializes only retained deltas and completes whole turns", async () => {
  const shader = await readFile(path.join(searchShaderRoot, "frontier_state.wgsl"), "utf8");

  assert.match(shader, /fn materialize_selected/);
  assert.match(shader, /recompute_turn_status/);
  assert.match(shader, /HEADER_DEPTH.*\+ 1/s);
  assert.match(shader, /HEADER_TURN.*1 - turn/s);
  assert.match(shader, /delta_count/);
});

test("GPU frontier expands all retained states without CPU source-target products", async () => {
  const shader = await readFile(path.join(searchShaderRoot, "frontier_expand.wgsl"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");

  assert.match(shader, /fn expand_frontier/);
  assert.match(shader, /atomicAdd\(&counters\[0\]/);
  assert.match(shader, /fn write_candidate/);
  assert.match(shader, /historical_branch/);
  assert.match(shader, /next_branch_row/);
  assert.match(shader, /CANDIDATE_DELTA_COUNT/);
  assert.match(shader, /BOARD_PENDING/);
  assert.match(shader, /states\[source \+ BOARD_PENDING\] == 0/);
  assert.match(engineSearch, /FRONTIER_BOARD_ACTIVE/);
  assert.match(engineSearch, /FRONTIER_BOARD_PENDING/);
});

test("GPU frontier move generation includes special pawn cases", async () => {
  const shader = await readFile(path.join(searchShaderRoot, "frontier_expand.wgsl"), "utf8");

  assert.match(shader, /ep_x == to_x && ep_y == to_y/);
  assert.match(shader, /captured_x = states\[source \+ BOARD_EP \+ 2u\]/);
  assert.match(shader, /placed_piece = select\(piece, 3,/);
});

test("GPU snapshot wire codes have engine-owned counterparts", async () => {
  const snapshot = await readFile(path.join(root, "src/ai-snapshot.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");

  assert.match(snapshot, /export function moveFromCandidateRecord/);
  assert.match(snapshot, /export function buildGpuCandidateInputs/);
  assert.match(snapshot, /sourceCount: sourceMeta\.length/);
  assert.match(snapshot, /targetCount: targetMeta\.length/);
  assert.match(snapshot, /boardCount: boards\.length \/ GPU_BOARD_STRIDE/);
  assert.match(snapshot, /const offset = index \* GPU_CANDIDATE_STRIDE/);
  assert.match(snapshot, /timelineId: records\[offset \+ 11\] \?\? 0/);
  assert.match(snapshot, /function isSourceAdvanceChild/);
  assert.match(snapshot, /child\.timelineId === move\.from\.timelineId && child\.time === move\.from\.time \+ 1/);
  assert.match(snapshot, /function nextBranchRow/);
  assert.match(snapshot, /const direction = owner === "white" \? 1 : -1/);
  assert.match(snapshot, /export function sortedTimelines/);
  assert.match(snapshot, /left\.row - right\.row \|\| left\.id - right\.id/);
  assert.match(snapshot, /export function latestBoard/);
  assert.match(snapshot, /board\.time > latest\.time \? board : latest/);
  assert.match(snapshot, /function pushGpuBoardRecord/);
  assert.doesNotMatch(snapshot, /export function pushGpuBoardRecord/);
  assert.match(snapshot, /pushGpuBoardRecord\(boards, timeline,/);
  assert.match(snapshot, /out\.push\(\s*timeline\.id,\s*timeline\.row,\s*board\.time,\s*colorCode\(board\.sideToMove\),\s*board\.castling \?\? 0,\s*board\.enPassant\?\.x \?\? -1/s);
  assert.match(snapshot, /for \(let index = 0; index < 64; index \+= 1\)/);
  assert.match(snapshot, /function pushGpuMutationBoardRecord/);
  assert.doesNotMatch(snapshot, /export function pushGpuMutationBoardRecord/);
  assert.match(snapshot, /pushGpuMutationBoardRecord\(mutationBoards, timeline,/);
  assert.match(snapshot, /board\.timelineIndex \?\? 0,\s*timeline\.id,\s*board\.time,\s*colorCode\(board\.sideToMove\),\s*board\.castling \?\? 0/s);
  assert.match(snapshot, /board\.latest \? 1 : 0,\s*board\.originKind \?\? 0,\s*0/s);
  assert.match(snapshot, /targets\.push\(\s*code & 255,\s*\(code >> 8\) & 255,\s*timeline\.id,\s*board\.time,\s*x,\s*y,\s*timeline\.row,\s*colorCode\(board\.sideToMove\),\s*ownerCode\(timeline\.owner\),\s*isLatest \? 1 : 0/s);
  assert.match(snapshot, /targetMeta\.push\(\{ timelineId: timeline\.id, time: board\.time, x, y \}\)/);
  assert.match(snapshot, /sources\.push\(\s*code & 255,\s*\(code >> 8\) & 255,\s*timeline\.id,\s*board\.time,\s*x,\s*y,\s*timeline\.row,\s*colorCode\(board\.sideToMove\),\s*ownerCode\(timeline\.owner\),\s*isLatest \? 1 : 0/s);
  assert.match(snapshot, /if \(\(code & 255\) === 0\) \{\s*continue;\s*\}/s);
  assert.match(snapshot, /sourceMeta\.push\(\{ timelineId: timeline\.id, time: board\.time, x, y \}\)/);
  assert.match(snapshot, /function gpuMutationBoardRecordToSnapshot/);
  assert.doesNotMatch(snapshot, /export function gpuMutationBoardRecordToSnapshot/);
  assert.match(snapshot, /timelineIndex: record\[0\] \?\? 0/);
  assert.match(snapshot, /sideToMove: colorFromCode\(record\[3\] \?\? 0\)/);
  assert.match(snapshot, /squares: record\.slice\(12, 76\)/);
  assert.match(snapshot, /function squaresToGameBoard/);
  assert.doesNotMatch(snapshot, /export function squaresToGameBoard/);
  assert.match(snapshot, /row\.push\(pieceFromCode\(squares\?\.\[y \* 8 \+ x\] \?\? 0\)\)/);
  assert.match(snapshot, /export function squareCodesForBoard/);
  assert.match(snapshot, /\(board\.board \?\? \[\]\)\.flat\(\)\.map\(\(piece\) => piece \? pieceTypeCode\(piece\.type\) \| \(colorCode\(piece\.color\) << 8\) : 0\)/);
  assert.match(snapshot, /function pieceTypeCode/);
  assert.match(snapshot, /function pieceTypeFromCode/);
  assert.match(snapshot, /function pieceFromCode/);
  assert.doesNotMatch(snapshot, /export function pieceTypeCode/);
  assert.doesNotMatch(snapshot, /export function pieceTypeFromCode/);
  assert.doesNotMatch(snapshot, /export function pieceFromCode/);
  assert.match(snapshot, /export function colorCode/);
  assert.match(snapshot, /export function colorFromCode/);
  assert.match(snapshot, /export function ownerCode/);
  assert.match(snapshot, /function ownerFromCode/);
  assert.match(snapshot, /export function oppositeColor/);
  assert.match(snapshot, /pieceTypeCode\(piece\.type\) \| \(colorCode\(piece\.color\) << 8\)/);
  assert.doesNotMatch(snapshot, /export function capitalize/);
  assert.match(worker, /engine\.chronofish_gpu_turn_status_records_snapshot_bytes\(ptr, len\)/);
  assert.match(worker, /const boardRecords = await engineTurnStatusRecords\(snapshot\)/);
  assert.match(worker, /boardRecords\.length \/ GPU_TURN_STATUS_RECORD_STRIDE/);
  assert.doesNotMatch(worker, /records\.length \/ GPU_TURN_STATUS_RECORD_STRIDE/);
  assert.match(engineTypes, /chronofish_gpu_turn_status_records_snapshot_bytes\(ptr: number, length: number\): number/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_turn_status_records_snapshot_bytes/);
  assert.match(engineSearch, /pub fn gpu_turn_status_records_i32s_from_snapshot_json/);
  assert.match(engineSearch, /pub struct GpuCandidatePosition/);
  assert.match(engineSearch, /pub struct GpuCandidateMove/);
  assert.match(engineSearch, /pub struct GpuCandidateBoardInput/);
  assert.match(engineSearch, /pub struct GpuCandidateBoardRecords/);
  assert.match(engineSearch, /pub struct GpuCandidateInputBoard/);
  assert.match(engineSearch, /pub struct GpuCandidateInputTimeline/);
  assert.match(engineSearch, /pub struct GpuCandidateInputs/);
  assert.match(engineSearch, /pub struct GpuCandidateSquareRecord/);
  assert.match(engineSearch, /pub struct GpuSquareRecordBoardInput/);
  assert.match(engineSearch, /pub struct GpuSquareRecordInput/);
  assert.match(engineSearch, /pub struct GpuTimelineSortKey/);
  assert.match(engineSearch, /pub struct GpuBoardRecordInput/);
  assert.match(engineSearch, /pub struct GpuMutationBoardRecordInput/);
  assert.match(engineSearch, /pub struct GpuChildBoardRef/);
  assert.match(engineSearch, /pub struct GpuMutationBoardSnapshot/);
  assert.match(engineSearch, /pub struct GpuEnPassantRecord/);
  assert.match(engineSearch, /pub fn gpu_timeline_sort_order/);
  assert.match(engineSearch, /pub fn gpu_latest_board_index/);
  assert.match(engineSearch, /gpu_timeline_sort_order\(&sort_keys\)/);
  assert.match(engineSearch, /gpu_latest_board_index\(/);
  assert.match(engineSearch, /pub fn gpu_candidate_board_records_from_snapshot/);
  assert.match(engineSearch, /pub fn gpu_candidate_inputs_from_timelines/);
  assert.match(engineSearch, /pub fn gpu_candidate_inputs_from_snapshot_json/);
  assert.match(engineSearch, /pub fn gpu_candidate_inputs_json_from_snapshot_json/);
  assert.match(engineSearch, /pub fn gpu_candidate_inputs_i32s_from_snapshot_json/);
  assert.match(engineSearch, /pub const GPU_CANDIDATE_INPUT_HEADER_I32S: usize = 7/);
  assert.match(engineSearch, /parse_game_snapshot\(snapshot_json\)/);
  assert.match(engineSearch, /gpu_candidate_inputs_from_game/);
  assert.match(engineSearch, /gpu_candidate_inputs_i32s_from_game/);
  assert.match(engineSearch, /source_count: usize/);
  assert.match(engineSearch, /target_count: usize/);
  assert.match(engineSearch, /board_count: usize/);
  assert.match(engineSearch, /pub fn gpu_candidate_move_from_record/);
  assert.match(engineSearch, /pub fn gpu_target_square_records_for_board/);
  assert.match(engineSearch, /pub fn gpu_source_square_records_for_board/);
  assert.match(engineSearch, /pub fn gpu_square_record_from_code/);
  assert.match(engineSearch, /pub fn gpu_board_record_from_snapshot/);
  assert.match(engineSearch, /pub fn gpu_mutation_board_record_from_snapshot/);
  assert.match(engineSearch, /pub fn gpu_child_is_source_advance/);
  assert.match(engineSearch, /pub fn gpu_next_branch_row/);
  assert.match(engineSearch, /pub fn gpu_mutation_board_record_to_snapshot/);
  assert.match(engineSearch, /pub fn gpu_search_color_from_code/);
  assert.match(engineSearch, /pub fn gpu_search_opposite_color/);
  assert.match(engineSearch, /pub fn gpu_search_owner_from_code/);
  assert.match(engineSearch, /squares: record\s*\.get\(12\.\.GPU_MUTATION_BOARD_STRIDE\)\s*\.unwrap_or\(&\[\]\)\s*\.to_vec\(\)/);
  assert.match(engineSearch, /records\.get\(offset \+ 11\)\.unwrap_or\(&0\)/);
  assert.match(engineSearch, /pub fn gpu_search_piece_type_code/);
  assert.match(engineSearch, /pub fn gpu_search_board_to_square_codes/);
  assert.match(engineSearch, /pub fn gpu_search_piece_type_from_code/);
  assert.match(engineSearch, /pub fn gpu_search_piece_from_code/);
  assert.match(engineSearch, /pub fn gpu_search_square_codes_to_board/);
  assert.match(engineSearch, /gpu_search_board_to_square_codes\(&pieces\)/);
  assert.match(engineSearch, /pub struct GpuDecodedPiece/);
  assert.match(engineSearch, /pub type GpuDecodedBoard/);
  assert.match(engineSearch, /pub fn gpu_search_color_code/);
  assert.match(engineSearch, /pub fn gpu_search_owner_code/);
  assert.match(engineSearch, /pub fn gpu_search_piece_code/);
  assert.match(engineSearch, /"commonKing" => Some\(2\)/);
  assert.match(engineSearch, /"brawn" => Some\(12\)/);
  assert.match(engineSearch, /Ok\(piece_type \| \(color << 8\)\)/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_candidate_inputs_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_candidate_inputs_bytes/);
  assert.match(wasmApi, /gpu_candidate_inputs_json_from_game/);
  assert.match(wasmApi, /gpu_candidate_inputs_i32s_from_game/);
  assert.match(engineTypes, /chronofish_gpu_candidate_inputs_json\(\): number/);
  assert.match(engineTypes, /chronofish_gpu_candidate_inputs_bytes\(\): number/);
  assert.match(worker, /async function engineGpuCandidateInputs\(game: GameSnapshot\): Promise<GpuCandidateInputs>/);
  assert.match(worker, /engine\.chronofish_gpu_candidate_inputs_bytes\(\)/);
  assert.match(worker, /const GPU_CANDIDATE_INPUT_HEADER_I32S = 7/);
  assert.match(worker, /function candidateInputsFromWords\(words: Int32Array\): GpuCandidateInputs/);
  assert.match(worker, /sourceMeta: candidateMetaFromRecords\(sources, GPU_SOURCE_STRIDE\)/);
  assert.match(worker, /targetMeta: candidateMetaFromRecords\(targets, GPU_TARGET_STRIDE\)/);
  assert.match(worker, /function candidateMetaFromRecords\(records: Int32Array, stride: number\): Position\[\]/);
  assert.match(worker, /sourceGame\s*\?\s*await engineGpuCandidateInputs\(sourceGame\)\s*:\s*buildGpuCandidateInputsFromSnapshot\(snapshot, snapshot\.turn\)/);
});

test("GPU frontier prunes quiet moves on low-relevance inactive boards", async () => {
  const shader = await readFile(path.join(searchShaderRoot, "frontier_expand.wgsl"), "utf8");

  assert.match(shader, /fn board_relevance/);
  assert.match(shader, /fn board_contains_royal/);
  assert.match(shader, /BOARD_PENDING/);
  assert.match(shader, /BOARD_ACTIVE/);
  assert.match(shader, /BOARD_ORIGIN/);
  assert.match(shader, /prune_low_relevance_quiet_target/);
  assert.match(shader, /tactical_priority == 0 && inactive && non_present/);
  assert.match(shader, /board_relevance\(state, target_board\) < 4/);
  assert.match(shader, /if \(prune_low_relevance_quiet_target\(state, target_board, tactical_priority\)\) \{\s*return;\s*\}/);
});

test("normal server exposes the committed GPU value model read-only", async () => {
  const server = await readFile(path.resolve(root, "../server/src/static_files.rs"), "utf8");

  assert.match(server, /ai\/value-model\.cfnn/);
  assert.match(server, /engine\/models\/gpu-v1\/value-model\.cfnn/);
});

test("GPU frontier projects retained states sparsely for neural evaluation", async () => {
  const shader = await readFile(path.join(searchShaderRoot, "frontier_neural.wgsl"), "utf8");
  const source = await readFile(path.join(root, "src/ai-frontier-neural.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const frontier = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const forward = await readFile(path.join(searchShaderRoot, "frontier_forward.wgsl"), "utf8");

  assert.match(shader, /select_neural_boards/);
  assert.match(shader, /project_neural_features/);
  assert.match(shader, /projection_hash/);
  assert.match(shader, /active_states\[state\]/);
  assert.match(shader, /HEADER_LAST_NEURAL/);
  assert.match(shader, /apply_neural_values/);
  assert.match(shader, /MAX_NEURAL_BOARDS: u32 = 16u/);
  assert.match(shader, /perspective/);
  assert.match(shader, /BOARD_LATEST\] != 0 && states\[base \+ BOARD_ACTIVE\] != 0/);
  assert.match(shader, /if \(states\[base \+ BOARD_ORIGIN\] != 0\) \{ return 3; \}/);
  assert.match(shader, /category\(state, board\) >= 4/);
  assert.doesNotMatch(source, /PROJECT_FEATURES_SHADER/);
  assert.doesNotMatch(source, /rawFeatures/);
  assert.match(source, /activeStates/);
  assert.match(source, /#currentPolicyFeatures/);
  assert.match(source, /#nextPolicyFeatures/);
  assert.match(source, /cacheStats\(\)/);
  assert.match(source, /#cacheStats\.hits \+= stateCount/);
  assert.match(source, /#cacheStats\.misses \+= stateCount/);
  assert.match(source, /#cacheStats\.stores \+= batchCount/);
  assert.match(source, /sharedBoardEncoder/);
  assert.match(source, /fastNet: \{/);
  assert.match(source, /policy-prior-candidate-pruning/);
  assert.match(source, /bigNet: \{/);
  assert.match(source, /retained-state-value-and-auxiliary-heads/);
  assert.match(forward, /forward_layer_masked/);
  assert.match(forward, /active_states\[sample\] == 0u/);
  assert.match(worker, /let activeStateLimit = 1/);
  assert.match(worker, /stateCount: activeStateLimit/);
  assert.match(frontier, /const sourceScans = stateCount \* this\.tuning\.maxBoards \* 64/);
});

test("GPU frontier neural evaluation uses adapter-sized batches", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier-neural.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const shader = await readFile(path.join(searchShaderRoot, "frontier_neural.wgsl"), "utf8");

  assert.match(source, /effectiveBatchSize/);
  assert.match(source, /for \(let stateOffset = 0; stateOffset < stateCount; stateOffset \+= effectiveBatchSize\)/);
  assert.match(worker, /tuning\.neuralBatchSize/);
  assert.match(shader, /state_offset/);
  assert.match(shader, /summaries\[\(apply_params\.state_offset \+ state\) \* SUMMARY_STRIDE \+ 1u\]/);
});

test("GPU frontier loads CFNN once and evaluates without prediction readback", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier-neural.ts"), "utf8");
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");

  assert.match(source, /fetch\("\/ai\/value-model\.cfnn"/);
  assert.match(source, /modelArchitectureMatches/);
  assert.match(source, /FrontierNeuralEvaluator/);
  assert.match(source, /modelBuffersFromBytes/);
  assert.match(trainer, /function outputLayerSize/);
  assert.match(trainer, /function previousLayerSize/);
  assert.match(trainer, /function policyLogitsArray/);
  assert.match(trainer, /function policyWeightsArray/);
  assert.match(engineTraining, /pub fn model_architecture_matches/);
  assert.match(engineTraining, /pub fn output_layer_size/);
  assert.match(engineTraining, /pub fn previous_layer_size/);
  assert.match(engineTraining, /pub fn policy_logits_array/);
  assert.match(engineTraining, /pub fn policy_weights_array/);
  assert.match(engineTraining, /pub const DEFAULT_HIDDEN_LAYERS: &\[u32\] = &\[1024, 512, 256\]/);
  assert.match(engineTraining, /pub const DEFAULT_PROJECTION_SEED: u32 = 2_166_136_261/);
  assert.doesNotMatch(source, /mapAsync/);
});

test("GPU workers validate an in-memory candidate model before promotion", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const trainer = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const ui = await readFile(path.join(root, "src/training-ui.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(worker, /type === "setModel"/);
  assert.match(worker, /frontierModelOverride = modelBytes/);
  assert.match(worker, /new FrontierNeuralEvaluator\(device, frontierModelOverride\)/);
  assert.match(trainer, /type: "setModel"/);
  assert.match(trainer, /modelBytes: candidateModel/);
  assert.match(trainer, /temperature: 0/);
  assert.match(trainer, /sampleSeed\("loss-log"/);
  assert.match(trainer, /chronofish_training_sample_seed\(ptr, len, index, salt\)/);
  assert.match(trainer, /chronofish_training_search_seed_json\(ptr, len, salt\)/);
  assert.doesNotMatch(trainer, /Math\.imul\(hash, 16777619\)/);
  assert.match(engineTraining, /pub fn sample_seed/);
  assert.match(engineTraining, /pub fn search_seed_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_training_sample_seed/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_training_search_seed_json/);
  assert.match(engineTypes, /chronofish_training_sample_seed\(ptr: number, length: number, index: number, salt: number\): number/);
  assert.match(engineTypes, /chronofish_training_search_seed_json\(ptr: number, length: number, salt: number\): number/);
  assert.match(ui, /const modelBytes = exactArrayBuffer\(model\)/);
  assert.match(ui, /validateTrainingLossLogs\(trainingConfig\(\), modelBytes\)/);
  assert.match(ui, /if \(logValidation\?\.failed\)/);
  assert.match(ui, /title: "Model Rejected"/);
});

test("GPU frontier applies serialized policy priors before candidate pruning", async () => {
  const frontier = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const neural = await readFile(path.join(root, "src/ai-frontier-neural.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const shader = await readFile(path.join(searchShaderRoot, "frontier_policy.wgsl"), "utf8");
  const stateShader = await readFile(path.join(searchShaderRoot, "frontier_state.wgsl"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(frontier, /await scoreCandidates\?\.\(encoder, buffers, candidateCapacity\)/);
  assert.match(frontier, /options\.cycleIndex,\s*100/s);
  assert.match(neural, /async encodePolicyPrior/);
  assert.match(neural, /model\.policyWeights\?\.length === expected/);
  assert.match(neural, /policyWeightsForModel/);
  assert.match(neural, /quantizePolicyWeights/);
  assert.match(neural, /format: "int8-dequantized-upload"/);
  assert.match(neural, /format: inferencePrecision === "fp16" \? "int8-to-fp16-upload" : "int8-dequantized-upload"/);
  assert.match(neural, /device\.features\?\.has\("shader-f16" as GPUFeatureName\)/);
  assert.match(neural, /const frontierForwardF16 = `enable f16;/);
  assert.match(neural, /var<storage, read> weights: array<f16>/);
  assert.match(neural, /const frontierPolicyF16 = `enable f16;/);
  assert.match(neural, /var<storage, read> policy_weights: array<f16>/);
  assert.match(neural, /float32ToFloat16Array/);
  assert.match(neural, /policyWeights: policyQuantization \? initializedWeightBuffer\(device, policyQuantization\.dequantized, inferencePrecision\) : null/);
  assert.match(neural, /modelBuffers\.fastNet\.policyWeights/);
  assert.match(neural, /modelBuffers\.bigNet\.outputWeights/);
  assert.match(neural, /#currentPolicyFeatures/);
  assert.match(neural, /advancePolicyFeatures/);
  assert.match(neural, /FRONTIER_POLICY_SHADER/);
  assert.match(worker, /runtime\.neural\.encodePolicyPrior/);
  assert.match(worker, /"gpu-v1-cfnn-v3-policy-head"/);
  assert.match(shader, /fn policy_bucket/);
  assert.match(shader, /CANDIDATE_CARRY/);
  assert.match(shader, /CANDIDATE_INTENT/);
  assert.match(shader, /hash_value\(hash, candidates\[base \+ CANDIDATE_INTENT\]\)/);
  assert.match(shader, /hidden_features\[parent \* params\.input_size \+ input\]/);
  assert.match(shader, /policy_weights\[row \+ input\]/);
  assert.match(shader, /CANDIDATE_POLICY_PRIOR/);
  assert.match(shader, /CANDIDATE_SCORE.*\+ prior/s);
  assert.match(stateShader, /CANDIDATE_SCORE\] - candidates\[candidate_base \+ CANDIDATE_POLICY_PRIOR\]/);
});

test("GPU frontier result labels neural mode only when the neural pass ran", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /let modelUsed = false/);
  assert.match(worker, /modelUsed = await runtime\.neural\.encode/);
  assert.match(worker, /const gpuSearch = modelUsed \? "neural-frontier" : "heuristic-frontier"/);
  assert.match(worker, /validatedFrontierChoices\(snapshot, readback\.states, tuning, requestedDepth, gpuSearch, sourceGame\)/);
  assert.doesNotMatch(worker, /readback\.modelUsed/);
});

test("GPU worker replays every posted search result through authoritative WASM", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /validateSearchResultBeforePost/);
  assert.match(worker, /validateSearchResultBeforePost\(snapshotOverride, gpuResult, clientGame\)/);
  assert.match(worker, /validateSearchResultBeforePost\(snapshotOverride, hybridResult, clientGame\)/);
  assert.match(worker, /sourceGame \?\? gpuSnapshotToGame\(snapshot\)/);
  assert.match(worker, /GPU search produced a turn that failed authoritative WASM replay/);
  assert.match(worker, /Hybrid GPU search produced a turn that failed authoritative WASM replay/);
  assert.match(worker, /chronofish_apply_move/);
  assert.match(worker, /chronofish_submit_turn/);
  assert.match(worker, /authoritativeReplay: true/);
});

test("GPU frontier publishes diagnostics needed for rollout gates", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const workerTypes = await readFile(path.join(root, "src/ai-worker-types.ts"), "utf8");
  const controller = await readFile(path.join(root, "src/bot-controller.ts"), "utf8");
  const source = await readFile(path.join(root, "scripts/gpu-frontier-smoke.mjs"), "utf8");
  const expandShader = await readFile(path.join(searchShaderRoot, "frontier_expand.wgsl"), "utf8");
  const stateShader = await readFile(path.join(searchShaderRoot, "frontier_state.wgsl"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(workerTypes, /interface GpuSearchDiagnostics/);
  assert.match(worker, /frontierWidth: tuning\.frontierWidth/);
  assert.match(worker, /candidateCapacity: tuning\.candidateCapacity/);
  assert.match(worker, /selectedCount: readback\.selectedCount/);
  assert.match(worker, /maxBoards: tuning\.maxBoards/);
  assert.match(worker, /dispatchCandidateLimit: tuning\.dispatchCandidateLimit/);
  assert.match(worker, /nodes: readback\.nodes/);
  assert.match(worker, /readbacks: 1/);
  assert.match(worker, /candidateOverflow: readback\.candidateOverflow \? 1 : 0/);
  assert.match(worker, /tacticalCandidates: readback\.tacticalCandidates/);
  assert.match(worker, /selectedTacticalCandidates: readback\.selectedTacticalCandidates/);
  assert.match(worker, /candidateSelectionRate: ratio\(readback\.selectedCount, readback\.nodes\)/);
  assert.match(worker, /tacticalSelectionRate: ratio\(readback\.selectedTacticalCandidates, readback\.tacticalCandidates\)/);
  assert.match(worker, /effectiveBranchingFactor/);
  assert.match(worker, /searchController: "puct-frontier-graph"/);
  assert.match(worker, /const maxCycles = await engineFrontierMaxCycles\(requestedDepth, snapshot\.timelines\.length\)/);
  assert.match(worker, /const perParentLimit = await engineFrontierPerParentLimit\(tuning\.frontierWidth\)/);
  assert.match(worker, /progressiveWideningLimit: perParentLimit/);
  assert.doesNotMatch(worker, /Math\.max\(2, Math\.min\(16, Math\.ceil\(tuning\.frontierWidth \/ 8\)\)\)/);
  assert.match(engineSearch, /pub fn frontier_max_cycles/);
  assert.match(engineSearch, /pub fn frontier_per_parent_limit/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_frontier_max_cycles/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_frontier_per_parent_limit/);
  assert.match(engineTypes, /chronofish_frontier_max_cycles\(requestedDepth: number, timelineCount: number\): number/);
  assert.match(engineTypes, /chronofish_frontier_per_parent_limit\(frontierWidth: number\): number/);
  assert.match(worker, /graphDeduplication: 1/);
  assert.match(worker, /legalChoiceCount: selected\.choices\.length/);
  assert.match(worker, /legalTacticalChoiceCount: selected\.choices\.filter\(\(choice\) => choice\.tactical\)\.length/);
  assert.match(worker, /const policyChoiceAgreement = await engineChoiceAgreement\(selected, selected\.choices, \[1, 5, 20\]\)/);
  assert.match(worker, /topPolicyChoiceAgreement: policyChoiceAgreement\[0\] \?\? 0/);
  assert.match(worker, /top5PolicyChoiceAgreement: policyChoiceAgreement\[1\] \?\? 0/);
  assert.match(worker, /top20PolicyChoiceAgreement: policyChoiceAgreement\[2\] \?\? 0/);
  assert.match(worker, /engine\.chronofish_gpu_choice_agreement_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /function choiceAgreement/);
  assert.match(engineTypes, /chronofish_gpu_choice_agreement_json\(ptr: number, length: number\): number/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_choice_agreement_json/);
  assert.match(engineSearch, /pub fn gpu_choice_agreement_json/);
  assert.match(worker, /selectedMovePrunedRisk: selected\.tactical \? 0 : 1/);
  assert.match(worker, /selectedMoveTactical: selected\.tactical \? 1 : 0/);
  assert.match(worker, /const neuralCache = runtime\.neural\.cacheStats\(\)/);
  assert.match(worker, /const quantization = await runtime\.neural\.quantizationStats\(\)/);
  assert.match(worker, /inferencePrecision: quantization\.inferencePrecision \?\? undefined/);
  assert.match(worker, /fastNetPolicyFormat: quantization\.fastNetPolicy\?\.format/);
  assert.match(workerTypes, /inferencePrecision\?: string/);
  assert.match(workerTypes, /fastNetPolicyMaxAbsError/);
  assert.match(worker, /nnCacheHits: neuralCache\.hits/);
  assert.match(worker, /nnCacheMisses: neuralCache\.misses/);
  assert.match(worker, /nnCacheStores: neuralCache\.stores/);
  assert.match(worker, /nnCacheHitRate: neuralCache\.hitRate/);
  assert.match(worker, /const networkRoles = runtime\.neural\.networkRoles\(\)/);
  assert.match(worker, /fastNet: networkRoles\.fastNet/);
  assert.match(worker, /bigNet: networkRoles\.bigNet/);
  assert.match(worker, /const counterByteLength = 24/);
  assert.match(worker, /nodesPerSecond/);
  assert.match(expandShader, /atomicAdd\(&counters\[4\], 1u\)/);
  assert.match(stateShader, /atomicAdd\(&counters\[5\], 1u\)/);
  assert.match(source, /selectedCount < Math\.min/);
  assert.match(controller, /Selected GPU frontier diagnostics/);
});

test("GPU frontier keeps pruned overflow searches when selected states exist", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const shader = await readFile(path.join(searchShaderRoot, "frontier_expand.wgsl"), "utf8");

  assert.match(shader, /atomicStore\(&counters\[2\], 1u\)/);
  assert.match(worker, /candidateOverflow: \(counters\[2\] \?\? 0\) !== 0/);
  assert.match(worker, /if \(readback\.candidateOverflow && readback\.selectedCount === 0\)/);
  assert.match(worker, /GPU frontier candidate capacity overflowed before completing search/);
});

test("GPU readbacks copy mapped ranges before unmapping", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /const statesCopy = bytes\.slice\(0, stateByteLength\)/);
  assert.match(worker, /const countersCopy = bytes\.slice\(stateByteLength, stateByteLength \+ counterByteLength\)/);
  assert.match(worker, /return new Int32Array\(bytes\)/);
  assert.match(worker, /readBuffer\.destroy\(\)/);
  assert.match(worker, /clearCachedGpuState\(\)/);
});

test("GPU frontier smoke harness covers board-count and tactical fixtures", async () => {
  const source = await readFile(path.join(root, "scripts/gpu-frontier-smoke.mjs"), "utf8");

  assert.match(source, /one-board-initial/);
  assert.match(source, /three-board-present/);
  assert.match(source, /five-board-present/);
  assert.match(source, /forced-multi-move-turn/);
  assert.match(source, /historical-branch/);
  assert.match(source, /expectedTarget/);
  assert.match(source, /legalTargetExpression/);
  assert.match(source, /type: "legalTargets"/);
  assert.match(source, /expected target missing from GPU legal targets/);
  assert.match(source, /capture/);
  assert.match(source, /castling/);
  assert.match(source, /en-passant/);
  assert.match(source, /promotion/);
  assert.match(source, /terminal-pressure/);
  assert.match(source, /stale-generation/);
  assert.match(source, /device-loss/);
  assert.match(source, /debugLoseDevice/);
  assert.match(source, /performance-gates/);
  assert.match(source, /--skip-performance-gates/);
  assert.match(source, /maxRegression: 1\.10/);
  assert.match(source, /minSpeedup: 2/);
  assert.match(source, /const smokeGpuMode = optionValue\("--gpu-mode"\) \?\? "full"/);
  assert.match(source, /fallback software adapter/);
  assert.match(source, /--allow-software-adapter/);
  assert.match(source, /--disable-neural/);
  assert.match(source, /--cpu-min-depth-only/);
  assert.match(source, /cpuMinimumDepthExpression/);
  assert.match(source, /minDepth: 3/);
  assert.match(source, /disableNeural/);
  assert.match(source, /gpuMode: smokeGpuMode/);
  assert.match(source, /readbacks !== 1/);
  assert.match(source, /candidateOverflow/);
  assert.match(source, /authoritativeReplay !== true/);
  assert.match(source, /gpuSearch !== "neural-frontier"/);
  assert.match(source, /gpuSearch !== "heuristic-frontier"/);
});

test("full GPU mode enters the resident frontier before legacy CPU candidate products", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const body = worker.slice(worker.indexOf("async function tryGpuSearch"), worker.indexOf("async function tryGpuResidentFrontierSearch"));

  assert.ok(body.indexOf("if (gpuMode === \"full\")") < body.indexOf("buildGpuCandidateInputsFromSnapshot"));
});

test("GPU bot search uses one worker and one device queue", async () => {
  const controller = await readFile(path.join(root, "src/bot-controller.ts"), "utf8");

  assert.match(controller, /WebGPU work is intentionally serialized through one worker\/device queue/);
  assert.match(controller, /function botSearchWorkerCount[\s\S]*return 1;/);
  assert.doesNotMatch(controller, /Math\.min\(2, hardwareThreads - 1\)/);
});

test("GPU training harness uses bounded parallel WebGPU workers", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const workerTypes = await readFile(path.join(root, "src/training-worker-types.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(worker, /function gpuTrainingWorkerCount/);
  assert.match(worker, /function splitWork/);
  assert.doesNotMatch(workerTypes, /const MAX_PARALLEL_GPU_TRAINING_WORKERS/);
  assert.match(constants, /export const MAX_PLAYOUT_PLIES = 10/);
  assert.match(worker, /MAX_PLAYOUT_PLIES[\s\S]*from "\.\/training-gpu-constants\.js"/);
  assert.match(worker, /chronofish_gpu_training_worker_count\(total, requestedWorkers\)/);
  assert.match(worker, /chronofish_training_split_work_json\(total, workers\)/);
  assert.match(worker, /chronofish_training_sample_plies\(index, encodeOnly \? 1 : 0\)/);
  assert.match(worker, /chronofish_training_worker_request_timeout_ms/);
  assert.match(worker, /chronofish_training_worker_search_time_ms/);
  assert.doesNotMatch(worker, /Math\.min\(MAX_PARALLEL_GPU_TRAINING_WORKERS, Math\.floor\(requestedWorkers\) \|\| 1\)/);
  assert.doesNotMatch(worker, /Math\.floor\(total \/ workers\) \+ \(index < total % workers \? 1 : 0\)/);
  assert.match(worker, /const workerCount = gpuTrainingWorkerCount\(positions\.length, config\.searchWorkers\)/);
  assert.match(worker, /const workerCount = gpuTrainingWorkerCount\(target, config\.selfPlayWorkers\)/);
  assert.match(worker, /collectGpuPositions\(game, config, target, progress, "search", config\.searchWorkers\)/);
  assert.match(worker, /const warmupPlies = gpuWarmupPlies\(workerIndex\)/);
  assert.match(worker, /const warmupConfig = gpuWarmupSearchConfig\(config\)/);
  assert.match(worker, /depth: warmupConfig\.depth/);
  assert.match(worker, /nodes: warmupConfig\.nodes/);
  assert.match(worker, /timeMs: warmupConfig\.timeMs/);
  assert.doesNotMatch(worker, /workerIndex === 0 \? 0 : 1 \+ \(workerIndex % Math\.max\(1, MAX_PLAYOUT_PLIES - 1\)\)/);
  assert.doesNotMatch(worker, /nodes: Math\.max\(1, Math\.min\(1024, config\.nodes\)\)/);
  assert.doesNotMatch(worker, /timeMs: Math\.min\(5000, workerSearchTimeMs\(config\)\)/);
  assert.match(worker, /const searchConfig = gpuPositionGenerationSearchConfig\(config\)/);
  assert.match(worker, /depth: searchConfig\.depth/);
  assert.match(worker, /nodes: searchConfig\.nodes/);
  assert.match(worker, /timeMs: searchConfig\.timeMs/);
  assert.doesNotMatch(worker, /const shallowConfig = \{ \.\.\.config, nodes: Math\.max\(1, Math\.min\(512, config\.nodes\)\) \}/);
  assert.doesNotMatch(worker, /timeMs: 3000/);
  assert.match(engineTraining, /pub const MAX_PARALLEL_GPU_TRAINING_WORKERS: usize = 16/);
  assert.match(engineTraining, /pub const MAX_PLAYOUT_PLIES: usize = 10/);
  assert.match(engineTraining, /pub const GPU_WARMUP_MAX_TIME_MS: u64 = 5_000/);
  assert.match(engineTraining, /pub const GPU_POSITION_GENERATION_TIME_MS: u64 = 3_000/);
  assert.match(engineTraining, /pub fn split_work/);
  assert.match(engineTraining, /pub fn gpu_training_worker_count/);
  assert.match(engineTraining, /pub fn sample_plies/);
  assert.match(engineTraining, /pub fn worker_request_timeout_ms/);
  assert.match(engineTraining, /pub fn worker_search_time_ms/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_training_worker_count/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_training_split_work_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_training_sample_plies/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_warmup_plies/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_warmup_search_config_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_position_generation_search_config_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_training_worker_request_timeout_ms/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_training_worker_search_time_ms/);
  assert.match(engineTypes, /chronofish_gpu_training_worker_count\(total: number, requestedWorkers: number\): number/);
  assert.match(engineTypes, /chronofish_training_split_work_json\(total: number, workers: number\): number/);
  assert.match(engineTypes, /chronofish_training_sample_plies\(index: number, encodeOnly: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_warmup_plies\(workerIndex: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_warmup_search_config_json\(depth: number, nodes: number, searchTimeMs: number, explorationTemperature: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_position_generation_search_config_json\(depth: number, nodes: number, explorationTemperature: number\): number/);
  assert.match(engineTraining, /pub fn sample_plies/);
  assert.match(engineTraining, /pub fn gpu_warmup_plies/);
  assert.match(engineTraining, /pub fn gpu_warmup_search_config/);
  assert.match(engineTraining, /pub fn gpu_position_generation_search_config/);
});

test("GPU training rollouts apply complete returned turns", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");

  assert.match(worker, /async function applyWorkerTurn/);
  assert.match(worker, /for \(const move of moves\)/);
  assert.match(worker, /type: "applyMove"[\s\S]*type: "submitTurn"/);
  assert.match(worker, /chronofish_royal_capture_winner_snapshot_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /function royalCount/);
  assert.doesNotMatch(worker, /function latestBoard/);
  assert.doesNotMatch(worker, /\["king", "royalQueen"\]\.includes\(piece\.type\)/);
  assert.doesNotMatch(worker, /result\.moves\?\.\[0\]/);
  assert.ok((worker.match(/applyWorkerTurn\(/g) ?? []).length >= 5);
});

test("GPU worker non-search commands use engine WASM instead of WebGPU", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const handler = worker.slice(worker.indexOf("self.addEventListener"), worker.indexOf("const snapshotOverride"));

  assert.match(handler, /type === "legalTargets"[\s\S]*engineLegalTargets/);
  assert.match(handler, /type === "applyMove"[\s\S]*engineApplyMove/);
  assert.match(handler, /type === "submitTurn"[\s\S]*engineSubmitTurn/);
  assert.doesNotMatch(handler, /getGpuDevice/);
  assert.doesNotMatch(handler, /legalTargetsOnGpu/);
  assert.doesNotMatch(handler, /applyMoveOnGpu/);
  assert.doesNotMatch(handler, /submitTurnOnGpu/);
  assert.doesNotMatch(worker, /function legalTargetsOnGpu/);
  assert.doesNotMatch(worker, /function applyMoveOnGpu/);
  assert.doesNotMatch(worker, /function submitTurnOnGpu/);
});

test("GPU worker gets search snapshots from the engine boundary", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const snapshotModule = await readFile(path.join(root, "src/ai-snapshot.ts"), "utf8");

  assert.match(worker, /const snapshotOverride = await engineGpuSnapshot\(clientGame\)/);
  assert.match(worker, /engine\.chronofish_gpu_snapshot_bytes\(\)/);
  assert.doesNotMatch(worker, /readGpuSnapshot/);
  assert.doesNotMatch(snapshotModule, /function readGpuSnapshot/);
});

test("stale GPU search generations cannot publish results", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /activeSearchGeneration/);
  assert.match(worker, /searchGeneration !== activeSearchGeneration/);
});

test("GPU frontier deadline cannot interrupt the requested depth", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /cyclesCompleted >= requestedDepth && Date\.now\(\) >= gpuDeadlineAt/);
});

test("hybrid GPU depth-one search uses engine pending boards for turn completion", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /const pendingBoards = await enginePendingPresentBoards\(snapshot, snapshot\.turn\)/);
  assert.match(worker, /async function enginePendingPresentBoards/);
  assert.match(worker, /engineFrontierRootFromSnapshot\(\{ \.\.\.snapshot, turn: color \}, boardCount\)/);
  assert.doesNotMatch(worker, /function pendingPresentBoardsForSnapshot/);
  assert.doesNotMatch(worker, /function activePresentTimeForSnapshot/);
  assert.doesNotMatch(worker, /function isActiveSnapshotTimeline/);
  assert.match(worker, /if \(pendingBoards\.length >= 1 && ranked\.length > 0\)/);
});

test("GPU candidate selection accepts single-move choices", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");

  assert.match(worker, /await selectSearchCandidate/);
  assert.match(worker, /engine\.chronofish_gpu_select_candidate_json\(ptr, len\)/);
  assert.match(worker, /moveCount: moves\.length/);
  assert.match(worker, /function choiceMoves/);
  assert.match(worker, /candidate\.moves \?\? \(candidate\.move \? \[candidate\.move\] : \[\]\)/);
  assert.doesNotMatch(worker, /function seededUnit/);
  assert.match(worker, /engine\.chronofish_gpu_pick_candidate_records_bytes\(ptr, byteLength\)/);
  assert.doesNotMatch(worker, /function pickCandidateRecords/);
  assert.match(worker, /engine\.chronofish_gpu_candidate_index_bytes\(ptr, byteLength\)/);
  assert.doesNotMatch(worker, /function findCandidateIndex/);
  assert.match(engineTypes, /chronofish_gpu_select_candidate_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_pick_candidate_records_bytes\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_candidate_index_bytes\(ptr: number, length: number\): number/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_select_candidate_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_pick_candidate_records_bytes/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_candidate_index_bytes/);
  assert.match(engineSearch, /pub fn gpu_search_select_candidate_json/);
  assert.match(engineSearch, /pub fn gpu_pick_candidate_records_from_i32s/);
  assert.match(engineSearch, /pub fn gpu_candidate_index_from_i32s/);
});

test("GPU frontier smoke harness can force device-loss cleanup and rebuild", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const gpuDevice = await readFile(path.join(root, "src/ai-gpu-device.ts"), "utf8");

  assert.match(worker, /debugLoseDevice/);
  assert.match(worker, /destroyCachedGpuDeviceForSmoke/);
  assert.match(worker, /device\.destroy\(\)/);
  assert.match(worker, /cachedGpuAdapter = null/);
  assert.match(worker, /clearComputePipelineCache\(\)/);
  assert.match(gpuDevice, /pipelineCache\.clear\(\)/);
});

test("GPU frontier tuning uses timestamp queries when available", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");

  assert.match(source, /autotuneFrontier/);
  assert.match(source, /timestamp-query/);
  assert.match(source, /beginningOfPassWriteIndex/);
  assert.match(source, /adapterTuningCacheKey/);
});

test("GPU frontier tuning stays below browser watchdog-sized passes", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const shader = await readFile(path.join(searchShaderRoot, "frontier_state.wgsl"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(source, /export async function deriveFrontierTuning/);
  assert.match(source, /function finiteLimit/);
  assert.match(source, /instantiateChronofishWasm\("\.\/chronofish_engine\.wasm"\)/);
  assert.match(source, /chronofish_derive_frontier_tuning_json/);
  assert.match(source, /const base = await deriveFrontierTuning/);
  assert.doesNotMatch(source, /const MAX_FRONTIER_WIDTH = 512/);
  assert.doesNotMatch(source, /const MAX_CANDIDATES = 65_536/);
  assert.doesNotMatch(source, /const MAX_SELECTION_SCAN = 2048/);
  assert.doesNotMatch(source, /function workgroupSize/);
  assert.doesNotMatch(source, /function clamp/);
  assert.doesNotMatch(source, /function nextPowerOfTwo/);
  assert.match(engineSearch, /pub const MAX_FRONTIER_WIDTH: usize = 512/);
  assert.match(engineSearch, /pub const MAX_CANDIDATES: usize = 65_536/);
  assert.match(engineSearch, /pub const MAX_SELECTION_SCAN: usize = 2048/);
  assert.match(engineSearch, /pub struct FrontierTuningLimits/);
  assert.match(engineSearch, /pub struct FrontierTuning/);
  assert.match(engineSearch, /pub fn derive_frontier_tuning/);
  assert.match(engineSearch, /pub fn gpu_frontier_positive_limit/);
  assert.match(engineSearch, /pub fn gpu_frontier_workgroup_size/);
  assert.match(engineSearch, /pub fn gpu_frontier_clamp_usize/);
  assert.match(engineSearch, /pub fn gpu_frontier_next_power_of_two/);
  assert.match(engineSearch, /gpu_frontier_workgroup_size\(max_invocations\)/);
  assert.match(engineSearch, /gpu_frontier_next_power_of_two\(/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_derive_frontier_tuning_json/);
  assert.match(engineTypes, /chronofish_derive_frontier_tuning_json\(/);
  assert.match(source, /minimax_reduce_stage/);
  assert.match(source, /Math\.ceil\(this\.tuning\.frontierWidth \/ 64\)/);
  assert.match(shader, /fn minimax_reduce_stage/);
  assert.match(shader, /peer < reduce_params\.state_count/);
  assert.doesNotMatch(shader, /array<i32, 128>/);
  assert.doesNotMatch(shader, /min\(128u, reduce_params\.state_count\)/);
});

test("GPU frontier sorts a bounded shortlist instead of full candidate capacity", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const shader = await readFile(path.join(searchShaderRoot, "frontier_select.wgsl"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(source, /async function frontierSelectionPlan/);
  assert.match(source, /chronofish_frontier_selection_plan_json/);
  assert.match(source, /const \{ candidateCapacity, selectionCapacity \} = selectionPlan/);
  assert.doesNotMatch(source, /selectionCapacity = floorPowerOfTwo\(Math\.min/);
  assert.doesNotMatch(source, /function floorPowerOfTwo/);
  assert.match(engineSearch, /pub struct FrontierSelectionPlan/);
  assert.match(engineSearch, /pub fn frontier_selection_plan/);
  assert.match(engineSearch, /pub fn gpu_frontier_floor_power_of_two/);
  assert.match(engineSearch, /gpu_frontier_floor_power_of_two\(tuning\.candidate_capacity\)/);
  assert.match(engineSearch, /candidate_capacity\.min\(MAX_SELECTION_SCAN\)/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_frontier_selection_plan_json/);
  assert.match(wasmApi, /max_selection_scan: usize/);
  assert.match(engineTypes, /chronofish_frontier_selection_plan_json\(/);
  assert.match(source, /for \(let k = 2; k <= selectionCapacity; k \*= 2\)/);
  assert.match(source, /Math\.ceil\(selectionCapacity \/ this\.tuning\.candidateWorkgroupSize\)/);
  assert.match(shader, /index = index \+ params\.max_scan/);
  assert.match(shader, /index >= params\.max_scan/);
});

test("GPU frontier fills from unsorted candidates when shortlist pruning underfills", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const shader = await readFile(path.join(searchShaderRoot, "frontier_select.wgsl"), "utf8");

  assert.match(source, /markUnique/);
  assert.match(source, /markParentQuota/);
  assert.match(source, /compactSelected/);
  assert.match(shader, /fn mark_unique/);
  assert.match(shader, /fn mark_parent_quota/);
  assert.match(shader, /fn compact_selected/);
  assert.match(shader, /atomicMax\(&counters\[1\], output \+ 1u\)/);
  assert.match(shader, /fn try_select_candidate/);
  assert.match(shader, /fn fill_selection_underflow/);
  assert.match(shader, /if \(selected_count >= params\.selected_limit\) \{ return; \}/);
  assert.match(shader, /for \(var index = 0u; index < actual_count && selected_count < params\.selected_limit/);
  assert.match(shader, /try_select_candidate\(i32\(index\), &selected_count\)/);
});

test("GPU policy training applies label weights to move priors", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  const shader = await readFile(path.join(trainingShaderRoot, "policy.wgsl"), "utf8");
  const lossShader = await readFile(path.join(trainingShaderRoot, "policy_loss.wgsl"), "utf8");
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");

  assert.equal(await fileExists(path.join(root, "src/training-policy.ts")), false);
  assert.doesNotMatch(worker, /import \{ policyBucket \} from "\.\/training-policy\.js"/);
  assert.doesNotMatch(trainer, /from "\.\/training-policy\.js"/);
  assert.match(trainer, /POLICY_BUCKETS[\s\S]*from "\.\/training-gpu-constants\.js"/);
  assert.match(constants, /export const POLICY_BUCKETS = 257/);
  assert.match(engineTraining, /pub const POLICY_BUCKETS: u32 = 257/);
  assert.match(worker, /chronofish_policy_bucket_from_move_values\(/);
  assert.match(engineTypes, /chronofish_policy_bucket_from_move_values\(/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_policy_bucket_from_move_values/);
  assert.match(trainer, /const policyIndices: number\[\] = \[\]/);
  assert.match(trainer, /labelWeights\[index\] = Math\.max\(0, sample\.labelWeight \?\? 1\)/);
  assert.match(trainer, /fillGroupedTrainingBatchIndices\(/);
  assert.match(trainer, /policyTrainingSteps\(config\.epochs\)/);
  assert.match(trainer, /forwardHiddenFeaturesOnProjectedGpu/);
  assert.match(trainer, /splitPolicyTrainingIndices/);
  assert.match(trainer, /policyLossOnGpu/);
  assert.match(trainer, /bestPolicyWeightBuffer/);
  assert.match(shader, /sample_weights\[dataset_sample\] \/ max\(params\.total_weight/);
  assert.match(shader, /features\[dataset_sample \* params\.input_size \+ input\]/);
  assert.match(shader, /policy_weights\[weight\] = policy_weights\[weight\] - params\.learning_rate/);
  assert.match(lossShader, /fn reduce_policy_loss/);
  assert.match(lossShader, /max_logit \+ log\(max\(denominator, 0\.000001\)\) - target_logit/);
  assert.match(engineTraining, /pub fn split_policy_training_indices/);
  assert.match(engineTraining, /pub fn has_policy_training_target/);
  assert.match(engineTraining, /pub fn policy_training_steps/);
  assert.match(engineTraining, /pub fn policy_bucket_from_move_values/);
  assert.match(engineTraining, /pub fn policy_bucket_from_values/);
});

test("GPU training distinguishes completed outcomes from search bootstraps", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const delta = await readFile(path.join(trainingShaderRoot, "output_delta.wgsl"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(worker, /kind: "partial"/);
  assert.match(worker, /"duel-search"/);
  assert.match(worker, /relabelOutcomeSamplesWithEngine/);
  assert.match(worker, /chronofish_relabel_outcome_samples_json\(ptr, len\)/);
  assert.match(worker, /const labelPolicy = trainingLabelPolicy\(\)/);
  assert.match(worker, /labelWeight: labelPolicy\.duelLabelWeight/);
  assert.match(worker, /labelWeight: labelPolicy\.duelDrawLabelWeight/);
  assert.match(worker, /labelWeight: labelPolicy\.outcomeLabelWeight/);
  assert.match(worker, /labelWeight: trainingLabelPolicy\(\)\.distilledLabelWeight/);
  assert.doesNotMatch(worker, /labelWeight: 1\.35/);
  assert.doesNotMatch(worker, /labelWeight: 1\.25/);
  assert.doesNotMatch(worker, /labelWeight: 1\.1/);
  assert.doesNotMatch(worker, /labelWeight: 0\.25/);
  assert.doesNotMatch(worker, /function backfillDrawLabels/);
  assert.doesNotMatch(worker, /function backfillOutcomeLabels/);
  assert.match(engineTraining, /pub fn outcome_label_for_turns/);
  assert.match(engineTraining, /pub fn apply_outcome_label/);
  assert.match(engineTraining, /pub fn apply_draw_label/);
  assert.match(engineTraining, /pub fn samples_from_partial_outcome/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_relabel_outcome_samples_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_training_label_policy_json/);
  assert.match(engineTypes, /chronofish_relabel_outcome_samples_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_training_label_policy_json\(\): number/);
  assert.match(engineTraining, /pub const DEFAULT_PARTIAL_OUTCOME_LABEL_KIND: &str = "search-bootstrap"/);
  assert.match(engineTraining, /pub const DEFAULT_PARTIAL_OUTCOME_LABEL_WEIGHT: f32 = 0\.5/);
  assert.match(engineTraining, /pub const OUTCOME_LABEL_DECAY: f32 = 0\.96/);
  assert.match(engineTraining, /pub const OUTCOME_LABEL_WEIGHT: f32 = 1\.25/);
  assert.match(engineTraining, /pub const DUEL_LABEL_WEIGHT: f32 = 1\.35/);
  assert.match(engineTraining, /pub const DUEL_DRAW_LABEL_WEIGHT: f32 = 1\.1/);
  assert.match(engineTraining, /pub const DISTILLED_LABEL_WEIGHT: f32 = 0\.25/);
  assert.match(trainer, /const weight = Math\.max\(0, samples\[index\]!\.labelWeight \?\? 1\)/);
  assert.match(trainer, /totalWeight > 0 \? total \/ totalWeight : 0/);
  assert.match(trainer, /const batchIndices = new Uint32Array\(batchSize\)/);
  assert.match(trainer, /const batchWeight = fillGroupedTrainingBatchIndices\(batchIndices, trainGroups, epoch, split\.seed, labelWeights\)/);
  assert.match(trainer, /outputDeltaParamsData\(batchSize, batchWeight\)/);
  assert.match(delta, /f32\(params\.batch_count\) \/ max\(params\.total_weight, 0\.000001\)/);
  assert.match(delta, /\* label_weights\[dataset_sample\]/);
});

test("GPU training has staged curriculum and tactical adversarial label modes", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const workerTypes = await readFile(path.join(root, "src/training-worker-types.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const trainingCli = await readFile(path.join(repoRoot, "engine/src/training/cli.rs"), "utf8");
  const ui = await readFile(path.join(root, "src/training-ui.ts"), "utf8");
  const html = await readFile(path.join(root, "src/index.html"), "utf8");

  assert.match(workerTypes, /"curriculum" \| "tactical"/);
  assert.match(worker, /collectCurriculumSamples/);
  assert.match(worker, /collectTacticalSamples/);
  assert.match(worker, /generateCurriculumPositionGame/);
  assert.match(worker, /curriculumGame/);
  assert.match(worker, /generateTacticalPositionGame/);
  assert.match(worker, /tacticalPositionPriority/);
  assert.match(worker, /chronofish_curriculum_game_snapshot_json\(ptr, len, index\)/);
  assert.match(worker, /chronofish_tactical_position_priority_snapshot_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /function curriculumBoards/);
  assert.doesNotMatch(worker, /function curriculumBoard/);
  assert.doesNotMatch(worker, /function curriculumTimelinePriority/);
  assert.doesNotMatch(worker, /const timelineLimit = stage <= 1 \? 1 : stage <= 3 \? 2 : Math\.max\(2, Math\.min\(timelines\.length, 4\)\)/);
  assert.doesNotMatch(worker, /priority \+= Math\.min\(3, game\.checkedRoyals\.length \* 2\)/);
  assert.doesNotMatch(worker, /function royalExposure/);
  assert.doesNotMatch(worker, /function temporalPowerPieceCount/);
  assert.match(worker, /chronofish_curriculum_search_config_json/);
  assert.match(worker, /chronofish_tactical_search_config_json/);
  assert.doesNotMatch(worker, /const stage = index % 6;\n  return \{\n    \.\.\.config,\n    depth: Math\.max\(1, Math\.min\(config\.depth, 1 \+ Math\.floor\(stage \/ 2\)\)\)/);
  assert.doesNotMatch(worker, /depth: Math\.max\(2, Math\.min\(config\.depth, 3 \+ attempt\)\)/);
  assert.match(engineTraining, /pub fn curriculum_stage/);
  assert.match(engineTraining, /pub fn curriculum_search_config/);
  assert.match(engineTraining, /pub fn tactical_search_config/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_curriculum_search_config_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_curriculum_game_snapshot_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_tactical_search_config_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_tactical_position_priority_snapshot_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_royal_capture_winner_snapshot_json/);
  assert.match(engineTypes, /chronofish_curriculum_search_config_json\(depth: number, nodes: number, explorationTemperature: number, index: number\): number/);
  assert.match(engineTypes, /chronofish_curriculum_game_snapshot_json\(ptr: number, length: number, index: number\): number/);
  assert.match(engineTypes, /chronofish_tactical_search_config_json\(depth: number, nodes: number, explorationTemperature: number, attempt: number\): number/);
  assert.match(engineTypes, /chronofish_tactical_position_priority_snapshot_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_royal_capture_winner_snapshot_json\(ptr: number, length: number\): number/);
  assert.match(engineTraining, /pub fn curriculum_timeline_limit/);
  assert.match(engineTraining, /pub fn curriculum_board_times/);
  assert.match(engineTraining, /pub fn curriculum_piece_type/);
  assert.match(engineTraining, /pub fn curriculum_timeline_priority/);
  assert.match(engineTraining, /pub fn tactical_position_priority_from_counts/);
  assert.match(engineTraining, /pub fn tactical_position_priority_snapshot_json/);
  assert.match(engineTraining, /pub fn royal_count_snapshot_json/);
  assert.match(engineTraining, /pub fn royal_capture_winner_snapshot_json/);
  assert.match(engineTraining, /pub enum SearchLabelMode/);
  assert.match(engineTraining, /Cpu/);
  assert.match(engineTraining, /Curriculum/);
  assert.match(engineTraining, /Tactical/);
  assert.match(engineTraining, /Distilled/);
  assert.match(engineTraining, /Outcome/);
  assert.match(engineTraining, /Duel/);
  assert.match(engineTraining, /heuristic-cpu-batch/);
  assert.match(engineTraining, /heuristic-curriculum-batch/);
  assert.match(engineTraining, /heuristic-tactical-batch/);
  assert.match(engineTraining, /heuristic-distilled-batch/);
  assert.match(engineTraining, /heuristic-outcome-batch/);
  assert.match(engineTraining, /heuristic-duel-batch/);
  assert.match(engineTraining, /fn outcome_label_samples/);
  assert.match(engineTraining, /fn duel_label_samples/);
  assert.match(trainingCli, /--gpu-sample-mode/);
  assert.match(trainingCli, /config\.gpu_sample_mode/);
  assert.match(worker, /collectGpuSearchLabels\(positions, config, progress, "curriculum"/);
  assert.match(worker, /collectGpuSearchLabels\([\s\S]*"tactical"/);
  assert.match(html, /<option value="curriculum" selected/);
  assert.match(html, /<option value="tactical"/);
  assert.match(ui, /gpuModes: \["vsGpu", "self", "curriculum"\]/);
  assert.match(ui, /gpuModes: \["vsGpu", "vsCpu", "self", "curriculum", "tactical", "distill"\]/);
  assert.match(ui, /curriculum: "Curriculum"/);
  assert.match(ui, /tactical: "Tactics"/);
});

test("GPU distillation labels searched positions instead of duplicating the root snapshot", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const trainingCli = await readFile(path.join(repoRoot, "engine/src/training/cli.rs"), "utf8");

  assert.match(worker, /const positions = await collectGpuPositions\([\s\S]*"distilled"/);
  assert.match(worker, /const samples = positions\.map\(\(position\) => position\.sample\)/);
  assert.match(worker, /const labels = await predictValues\(samples, activeModel\)/);
  assert.match(engineTraining, /pub fn distill_training_samples/);
  assert.match(engineTraining, /pub const DISTILLED_LABEL_WEIGHT: f32 = 0\.25/);
  assert.match(engineTraining, /SearchLabelMode::Distilled/);
  assert.match(engineTraining, /distilled sample mode requires a compact value model/);
  assert.match(trainingCli, /--gpu-distill-samples/);
  assert.match(trainingCli, /"--gpu-model" \| "--gpu-value-model"/);
  assert.match(trainingCli, /config\.gpu_value_model_path = model_path/);
  assert.match(trainingCli, /fn gpu_value_model_path\(config: &TrainerConfig\) -> &str/);
  assert.match(trainingCli, /fn load_gpu_value_model\(config: &TrainerConfig\)/);
  assert.match(trainingCli, /gpu_sample_distill_model/);
  assert.match(trainingCli, /load_gpu_value_model\(config\)/);
  assert.match(trainingCli, /distill_training_samples\(&samples, &model\)/);
  assert.match(trainingCli, /source_model=\{\}/);
  assert.doesNotMatch(worker, /const positions = await collectSamples\(game, config, true/);
});

test("GPU training samples uniform minibatches and applies label weights once", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const sampleHelpers = await readFile(path.join(root, "src/training-gpu-samples.ts"), "utf8");

  assert.match(sampleHelpers, /export function fillGroupedTrainingBatchIndices/);
  assert.match(sampleHelpers, /export function groupTrainingIndicesByPosition/);
  assert.match(sampleHelpers, /state = xorshift32\(state \|\| 1\)/);
  assert.match(sampleHelpers, /const group = trainGroups\[state % trainGroups\.length\]!/);
  assert.match(sampleHelpers, /const selected = group\[state % group\.length\]!/);
  assert.match(sampleHelpers, /batch\[index\] = selected/);
  assert.match(sampleHelpers, /batchWeight \+= Math\.max\(0, labelWeights\[selected\] \?\? 1\)/);
  assert.doesNotMatch(`${trainer}\n${sampleHelpers}`, /trainingWeightCdf/);
  assert.doesNotMatch(`${trainer}\n${sampleHelpers}`, /weightedTrainingIndex/);
  assert.doesNotMatch(`${trainer}\n${sampleHelpers}`, /const epochOrder = shuffledIndices\(trainIndices, epoch, split\.seed\)/);
});

test("GPU training sample utilities have engine-owned counterparts", async () => {
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");

  assert.match(engineTraining, /pub fn split_validation_samples/);
  assert.match(engineTraining, /pub fn stable_sample_hash/);
  assert.match(engineTraining, /pub fn shuffled_indices/);
  assert.match(engineTraining, /pub fn group_training_indices_by_position/);
  assert.match(engineTraining, /pub fn unique_training_position_count/);
  assert.match(engineTraining, /pub fn fill_grouped_training_batch_indices/);
  assert.match(engineTraining, /pub fn feature_length/);
  assert.match(engineTraining, /pub fn xorshift32/);
});

test("GPU training validation split falls back to a high-signal holdout", async () => {
  const sampleHelpers = await readFile(path.join(root, "src/training-gpu-samples.ts"), "utf8");

  assert.match(sampleHelpers, /validationSplit > 0 && !validationIndices\.length && trainIndices\.length > 1/);
  assert.match(sampleHelpers, /movePositionGroupToValidation\(samples, trainIndices, validationIndices, seed\)/);
  assert.match(sampleHelpers, /groupTrainingIndicesByPosition\(samples, trainIndices\)/);
  assert.match(sampleHelpers, /function validationSamplePriority/);
  assert.match(sampleHelpers, /trainingLabelPriority\(sample\.labelKind, sample\.pseudo\)/);
  assert.match(sampleHelpers, /Math\.max\(0, sample\.labelWeight \?\? 1\)/);
});

test("GPU training selects a device-sized high-signal working set", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  const sampleHelpers = await readFile(path.join(root, "src/training-gpu-samples.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const trainingCli = await readFile(path.join(repoRoot, "engine/src/training/cli.rs"), "utf8");

  assert.equal(await fileExists(path.join(root, "src/training-replay.ts")), false);
  assert.match(trainer, /const trainingSamples = selectTrainingWorkingSet\(samples, device\)/);
  assert.match(trainer, /trainValue\(device, trainingSamples, config, activeModel, progress\)/);
  assert.match(trainer, /value\.policyFeatureBuffer/);
  assert.match(trainer, /model\.replayBufferSize = samples\.length/);
  assert.match(trainer, /model\.trainingSampleCount = trainingSamples\.length/);
  assert.match(trainer, /export function selectTrainingWorkingSet/);
  assert.match(trainer, /device\.limits\?\.maxStorageBufferBindingSize/);
  assert.match(trainer, /maxProjectedSamples = Math\.max\(1, Math\.floor\(maxBindingSize \/ \(PROJECTION_SIZE \* Float32Array\.BYTES_PER_ELEMENT\)\)\)/);
  assert.match(trainer, /trainingSamplePriority\(sample, index, samples\.length\)/);
  assert.match(constants, /export const MIN_POLICY_WORKING_SET_FRACTION = 0\.25/);
  assert.match(trainer, /const requiredPolicyCount = Math\.min/);
  assert.match(trainer, /for \(let index = selected\.length - 1; index >= 0; index -= 1\)/);
  assert.match(trainer, /trainingLabelPriority[\s\S]*from "\.\/training-gpu-samples\.js"/);
  assert.match(sampleHelpers, /export function trainingLabelPriority/);
  assert.doesNotMatch(trainer, /from "\.\/training-replay\.js"/);
  assert.match(engineTraining, /pub fn select_training_working_set/);
  assert.match(engineTraining, /pub fn select_training_working_set_for_projection/);
  assert.match(engineTraining, /pub fn select_training_working_set_with_capacity/);
  assert.match(engineTraining, /pub fn training_sample_priority/);
  assert.match(engineTraining, /pub fn training_label_priority/);
  assert.match(engineTraining, /pub const MIN_POLICY_WORKING_SET_FRACTION: f32 = 0\.25/);
  assert.match(engineTraining, /pub const DEFAULT_PROJECTION_SIZE: usize = 2048/);
  assert.match(engineTraining, /pub const DEFAULT_PROJECTED_WORKING_SET_BYTES: usize = 128 \* 1024 \* 1024/);
  assert.match(trainingCli, /select_training_working_set_for_projection\(/);
  assert.match(trainingCli, /DEFAULT_PROJECTED_WORKING_SET_BYTES/);
  assert.match(trainingCli, /train_native_gpu_value_model_from_projected\(&working_set,/);
});

test("GPU training checkpoint loss reuses projected replay buffers", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const lossShader = await readFile(path.join(trainingShaderRoot, "reduce_loss.wgsl"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");

  assert.match(trainer, /predictionLossOnProjectedGpu\(/);
  assert.match(trainer, /featureBuffer,/);
  assert.match(trainer, /weightBuffers,/);
  assert.match(trainer, /outputWeightBuffer,/);
  assert.match(trainer, /forwardIndexedLayerPipeline,/);
  assert.match(trainer, /export async function predictionLossOnProjectedGpu/);
  assert.match(trainer, /const batchIndexBuffer = storageBuffer/);
  assert.match(trainer, /layerIndex === 0 \? forwardIndexedLayerPipeline : forwardLayerPipeline/);
  assert.match(trainer, /const partials = await readFloats/);
  assert.match(trainer, /lossReductionWorkgroupCount\(sampleCount\)/);
  assert.match(lossShader, /var<workgroup> reductions: array<vec2<f32>, 64>/);
  assert.match(lossShader, /weight \* error \* error/);
  assert.match(engineTraining, /pub fn loss_reduction_workgroup_count/);
  assert.doesNotMatch(trainer, /predictionLossOnGpu\(device, indexSamples/);
  assert.doesNotMatch(trainer, /function indexSamples/);
});

test("GPU training records internal phase timings for hardware benchmarks", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const benchmark = await readFile(path.join(root, "scripts/gpu-frontier-smoke.mjs"), "utf8");

  assert.match(trainer, /metrics\.phases\[phase\] = \(metrics\.phases\[phase\] \?\? 0\) \+ performance\.now\(\) - startedAt/);
  assert.match(trainer, /timed\(config\.metrics, "valueTrain"/);
  assert.match(trainer, /timed\(config\.metrics, "policyTrain"/);
  assert.match(trainer, /timed\(config\.metrics, "projection"/);
  assert.match(trainer, /timed\(config\.metrics, "initialValidationLoss"/);
  assert.match(benchmark, /phasePercentages/);
  assert.match(benchmark, /trainingSamplesPerSecond/);
});

test("GPU dense training kernels use workgroup-tiled matrix operations", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  const device = await readFile(path.join(root, "src/training-gpu-device.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const shaders = await Promise.all([
    "forward_layer.wgsl",
    "forward_indexed_layer.wgsl",
    "apply_layer.wgsl",
    "apply_indexed_layer.wgsl",
    "hidden_delta.wgsl",
    "policy.wgsl"
  ].map((name) => readFile(path.join(trainingShaderRoot, name), "utf8")));

  for (const shader of shaders) {
    assert.match(shader, /var<workgroup>/);
    assert.match(shader, /workgroupBarrier\(\)/);
    assert.match(shader, /base = base \+ 16u/);
  }
  assert.match(shaders[0], /input_tile\[local\.x \* 16u \+ offset\]/);
  assert.match(shaders[2], /feature_tile\[offset \* 16u \+ local\.x\]/);
  assert.match(shaders[4], /delta_tile\[local\.x \* 16u \+ offset\]/);
  assert.match(shaders[5], /policy_feature_tile/);
  for (const shader of shaders.slice(0, 5)) {
    assert.match(shader, /fn \w+_naive\(/);
  }
  assert.match(shaders[5], /fn forward_policy_naive\(/);
  assert.match(shaders[5], /fn apply_policy_naive\(/);
  assert.match(constants, /export const TILED_TRAINING_MIN_BATCH = 16/);
  assert.match(device, /sampleCount >= TILED_TRAINING_MIN_BATCH \? entryPoint : `\$\{entryPoint\}_naive`/);
  assert.match(device, /function denseKernelEntryPoint/);
  assert.match(device, /function formatBytes/);
  assert.match(device, /function align4/);
  assert.match(engineTraining, /pub const TILED_TRAINING_MIN_BATCH: usize = 16/);
  assert.match(engineTraining, /pub fn dense_kernel_entry_point/);
  assert.match(engineTraining, /pub fn format_bytes/);
  assert.match(engineTraining, /pub fn align4/);
  assert.match(trainer, /denseKernelEntryPoint\("forward_layer", batchSize\)/);
  assert.match(trainer, /denseKernelEntryPoint\("apply_layer", batchSize\)/);
  assert.match(trainer, /denseKernelEntryPoint\("hidden_delta", batchSize\)/);
  assert.match(trainer, /denseKernelEntryPoint\("forward_policy", batchSize\)/);
  assert.match(trainer, /denseKernelEntryPoint\("apply_policy", batchSize\)/);
});

test("GPU training unlocks hidden-layer backpropagation only with enough unique positions", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  const model = await readFile(path.join(root, "src/training-gpu-model.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");

  assert.match(constants, /export const MIN_HIDDEN_TRAINING_POSITIONS = 256/);
  assert.match(constants, /export const CPU_HEAD_TRAINING_MAX_POSITIONS = 32/);
  assert.match(constants, /export const CPU_PREDICTION_MAX_BATCH = 4/);
  assert.match(engineTraining, /pub const MIN_HIDDEN_TRAINING_POSITIONS: usize = 256/);
  assert.match(engineTraining, /pub const CPU_HEAD_TRAINING_MAX_POSITIONS: usize = 32/);
  assert.match(engineTraining, /pub const CPU_PREDICTION_MAX_BATCH: usize = 4/);
  assert.match(trainer, /const hiddenLayersTrained = uniqueTrainingPositionCount\(samples, trainIndices\) >= MIN_HIDDEN_TRAINING_POSITIONS/);
  assert.match(trainer, /const deltaBuffers = hiddenLayersTrained/);
  assert.match(trainer, /const hiddenDeltaPipeline = hiddenLayersTrained/);
  assert.match(trainer, /const applyLayerPipeline = hiddenLayersTrained/);
  assert.match(trainer, /if \(hiddenLayersTrained\) \{\s*const lastLayerIndex/);
  assert.match(trainer, /model\.hiddenLayersTrained = value\.hiddenLayersTrained/);
  assert.match(trainer, /AUXILIARY_VALUE_HEADS/);
  assert.match(trainer, /trainAuxiliaryValueHeadsOnCpu/);
  assert.match(trainer, /auxiliaryValueWeights: auxiliary\.weights/);
  assert.match(model, /auxiliaryValueWeights/);
  assert.match(model, /export function compactModelIsFinite/);
  assert.match(model, /function finiteArray/);
  assert.match(model, /export function byteArraysEqual/);
  assert.match(engineTraining, /pub fn compact_model_is_finite/);
  assert.match(engineTraining, /pub fn f32_values_are_finite/);
  assert.match(engineTraining, /pub fn byte_arrays_equal/);
  assert.match(engineTraining, /pub fn encode_compact_value_model/);
  assert.match(engineTraining, /pub fn decode_compact_value_model/);
  assert.match(engineTraining, /pub fn compact_value_model_policy_values/);
  assert.match(engineTraining, /pub fn compact_value_model_encoded_len/);
  assert.match(model, /export function encodeCompactModel/);
  assert.match(model, /export function decodeCompactModel/);
  assert.match(model, /const byteLength = 4/);
});

test("GPU and CPU-head optimizers retain momentum without checkpoint readbacks", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const shaders = await Promise.all([
    "apply_output.wgsl",
    "apply_layer.wgsl",
    "apply_indexed_layer.wgsl",
    "policy.wgsl"
  ].map((name) => readFile(path.join(trainingShaderRoot, name), "utf8")));

  assert.match(constants, /export const OPTIMIZER_MOMENTUM = 0\.9/);
  assert.match(engineTraining, /pub const OPTIMIZER_MOMENTUM: f32 = 0\.9/);
  assert.match(engineTraining, /pub fn optimizer_velocity/);
  assert.match(trainer, /const velocityBuffers = hiddenLayersTrained/);
  assert.match(trainer, /const outputVelocityBuffer = zeroStorageBuffer/);
  assert.match(trainer, /const policyVelocityBuffer = zeroStorageBuffer/);
  assert.match(trainer, /encodePipelineBindings\(device, encoder, forwardPipeline/);
  assert.match(trainer, /\[8, policyVelocityBuffer\]/);
  assert.match(trainer, /optimizerVelocity\(velocity\[input\] \?\? 0, update\)/);
  assert.match(trainer, /optimizerVelocity\(velocity\[index\] \?\? 0, update\)/);
  assert.match(engineTraining, /pub fn split_hidden_weights/);
  assert.match(engineTraining, /pub fn concat_f32/);
  assert.match(engineTraining, /pub fn count_non_zero/);
  assert.match(engineTraining, /pub fn initial_hidden_weights/);
  assert.match(engineTraining, /pub fn default_initial_hidden_weights/);
  for (const shader of shaders) {
    assert.match(shader, /momentum: f32/);
    assert.match(shader, /var<storage, read_write> velocity: array<f32>/);
    assert.match(shader, /params\.momentum \* velocity/);
  }
});

test("GPU value inference restores the normalized training score scale", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const shader = await readFile(path.join(searchShaderRoot, "frontier_neural.wgsl"), "utf8");
  const forward = await readFile(path.join(trainingShaderRoot, "forward_output.wgsl"), "utf8");
  const frontierForward = await readFile(path.join(searchShaderRoot, "frontier_forward.wgsl"), "utf8");
  const delta = await readFile(path.join(trainingShaderRoot, "output_delta.wgsl"), "utf8");

  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  assert.match(constants, /export const VALUE_SCORE_SCALE = 20_000/);
  assert.match(worker, /return loadedEngine\(\)\.chronofish_normalized_search_score\(score\)/);
  assert.doesNotMatch(worker, /import \{ train, predictValues, normalizedSearchScore \}/);
  assert.match(shader, /clamp\(predictions\[state\] \* apply_params\.value_scale \+ apply_params\.value_bias, -1\.0, 1\.0\)/);
  assert.match(shader, /neural \* 20000\.0/);
  assert.doesNotMatch(shader, /neural \* 100\.0/);
  assert.match(forward, /predictions\[sample\] = tanh\(sum\)/);
  assert.match(frontierForward, /output_values\[sample\] = tanh\(sum\)/);
  assert.match(delta, /\* \(1\.0 - prediction \* prediction\)/);
  assert.match(trainer, /outputWeights\[outputSize\] = inverseTanh/);
  assert.match(engineTraining, /pub fn normalized_search_score/);
  assert.match(engineTraining, /pub fn denormalized_search_score/);
  assert.match(engineTraining, /pub fn inverse_tanh/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_normalized_search_score/);
  assert.match(engineTypes, /chronofish_normalized_search_score\(score: number\): number/);
});

test("GPU training keeps best checkpoints on GPU until final export", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");

  assert.match(trainer, /const bestWeightBuffers = layerWeights\.map/);
  assert.match(trainer, /const bestOutputWeightBuffer = storageBuffer/);
  assert.match(trainer, /"initialValidationLoss"/);
  assert.match(trainer, /let bestValidationLoss = initialValidationLoss/);
  assert.match(trainer, /let checkpointImproved = false/);
  assert.match(trainer, /checkpointImproved = true/);
  assert.match(trainer, /"bestCheckpointCopy"/);
  assert.match(trainer, /copyTrainingWeights\(device, outputWeightBuffer, weightBuffers, bestOutputWeightBuffer, bestWeightBuffers, layerWeights, outputWeights\.byteLength\)/);
  assert.match(trainer, /function copyTrainingWeights/);
  assert.match(trainer, /encoder\.copyBufferToBuffer\(outputWeightBuffer, 0, bestOutputWeightBuffer, 0, outputByteLength\)/);
  assert.match(trainer, /"bestWeightReadback"/);
  assert.doesNotMatch(trainer, /"finalWeightReadback"/);
  assert.doesNotMatch(trainer, /exported final changed checkpoint/);
  assert.match(trainer, /async function readTrainingWeights/);
  assert.match(trainer, /const readBuffer = device\.createBuffer/);
  assert.match(trainer, /layerOffsets\[layerIndex\]!/);
  assert.equal(trainer.match(/onSubmittedWorkDone/g)?.length, 1);
  assert.doesNotMatch(trainer, /"weightReadback"/);
});

test("GPU training destroys per-run and validation buffers after export", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");

  assert.match(trainer, /finally \{\s*destroyBuffers\(value\.resources\)/);
  assert.match(trainer, /resources: \[\s*featureBuffer,/);
  assert.match(trainer, /destroyBuffers\(\[\s*targetBuffer,/);
  assert.match(trainer, /destroyBuffers\(\[indexBuffer, partialBuffer, paramsBuffer\]\)/);
  assert.match(trainer, /destroyBuffers\(\[\s*batchIndexBuffer,/);
  assert.match(trainer, /readBuffer\.destroy\(\)/);
  assert.match(trainer, /function destroyBuffers/);
  assert.match(trainer, /destroyed\.has\(buffer\)/);
});

test("GPU training batches feature projection directly into the final buffer", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  const shader = await readFile(path.join(trainingShaderRoot, "project_features.wgsl"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");

  assert.match(constants, /export const PROJECTION_TEMPORARY_BUDGET = 128 \* 1024 \* 1024/);
  assert.match(trainer, /packSparseProjectionFeatures\(chunkSamples, inputSize\)/);
  assert.match(trainer, /temporaryBytes \+ sparseFeatures\.byteLength > temporaryBudget/);
  assert.match(trainer, /\[offsetBuffer, indexBuffer, valueBuffer, projectedBuffer, paramsBuffer\]/);
  assert.match(trainer, /temporaryBuffers\.push\(offsetBuffer, indexBuffer, valueBuffer, paramsBuffer\)/);
  assert.match(trainer, /await device\.queue\.onSubmittedWorkDone\(\)/);
  assert.match(trainer, /buffer\.destroy\(\)/);
  assert.doesNotMatch(trainer, /chunkProjectedBuffer/);
  assert.doesNotMatch(shader, /raw_features/);
  assert.match(shader, /feature_offsets: array<u32>/);
  assert.match(shader, /feature_indices: array<u32>/);
  assert.match(shader, /feature_values: array<f32>/);
  assert.match(shader, /sparse_start = feature_offsets\[sample\]/);
  assert.match(shader, /sparse_end = feature_offsets\[sample \+ 1u\]/);
  assert.match(shader, /output_offset: u32/);
  assert.match(shader, /\(params\.output_offset \+ sample\) \* params\.projection_size/);
  assert.match(engineTraining, /pub struct SparseProjectionFeatures/);
  assert.match(engineTraining, /pub fn pack_sparse_projection_features/);
  assert.match(engineTraining, /pub fn pack_sparse_feature_rows/);
  assert.match(engineTraining, /pub const PROJECTION_TEMPORARY_BUDGET: usize = 128 \* 1024 \* 1024/);
});

test("training label workers encode positions through engine WASM", async () => {
  const worker = await readFile(path.join(root, "src/training-label-worker.ts"), "utf8");
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineEvaluator = await readFile(path.join(repoRoot, "engine/src/ai/evaluator.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.equal(await fileExists(path.join(root, "src/training-encoding.ts")), false);
  assert.match(worker, /instantiateChronofishWasm\("\.\/chronofish_engine\.wasm"\)/);
  assert.match(worker, /chronofish_training_samples_json\(ptr, len\)/);
  assert.match(worker, /readWasmString\(engine, output\)/);
  assert.match(trainer, /NEURAL_BOARD_PLANES[\s\S]*from "\.\/training-gpu-constants\.js"/);
  assert.match(trainer, /base \+ 31 \* NEURAL_BOARD_SQUARES/);
  assert.match(constants, /export const NEURAL_BOARD_PLANES = 32/);
  assert.match(constants, /export const NEURAL_BOARD_SQUARES = 64/);
  assert.match(engineEvaluator, /NEURAL_BOARD_PLANES: usize = 32/);
  assert.match(engineEvaluator, /NEURAL_BOARD_SQUARES: usize = 64/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_training_sample_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_training_samples_json/);
  assert.match(wasmApi, /sample_from_snapshot_label\(Some\(text\), 0\.0, 1\.0\)/);
  assert.match(engineTypes, /chronofish_training_sample_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_training_samples_json\(ptr: number, length: number\): number/);
  assert.doesNotMatch(trainer, /from "\.\/training-encoding\.js"/);
  assert.doesNotMatch(worker, /encodeNeuralPositionFeatures/);
  assert.doesNotMatch(worker, /samples\.push\(await neuralPosition\(snapshot\)\)/);
  assert.doesNotMatch(worker, /features\.buffer/);
  assert.doesNotMatch(worker, /navigator\.gpu/);
  assert.doesNotMatch(worker, /onSubmittedWorkDone/);
  assert.doesNotMatch(worker, /readFloats/);
});

test("GPU replay deduplicates positions and keeps validation groups separate", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const labels = await readFile(path.join(root, "src/training-label-worker.ts"), "utf8");
  const sampleHelpers = await readFile(path.join(root, "src/training-gpu-samples.ts"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");

  assert.equal(await fileExists(path.join(root, "src/training-replay.ts")), false);
  assert.match(labels, /chronofish_training_samples_json\(ptr, len\)/);
  assert.doesNotMatch(labels, /function positionKey/);
  assert.match(worker, /chronofish_dedupe_training_samples_json\(ptr, len\)/);
  assert.match(worker, /chronofish_append_replay_samples_json\(ptr, len, maxBuffer\)/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_dedupe_training_samples_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_append_replay_samples_json/);
  assert.match(engineTypes, /chronofish_dedupe_training_samples_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_append_replay_samples_json\(ptr: number, length: number, maxBuffer: number\): number/);
  assert.doesNotMatch(worker, /import \{ appendReplaySamples, dedupeTrainingSamples \} from "\.\/training-replay\.js"/);
  assert.doesNotMatch(worker, /from "\.\/training-replay\.js"/);
  assert.match(worker, /const collectedSamples = await timed\(metrics, "collect"/);
  assert.match(worker, /const samples = await dedupeTrainingSamplesWithEngine\(collectedSamples\)/);
  assert.match(engineTraining, /pub fn dedupe_training_samples/);
  assert.match(engineTraining, /fn replay_sample_key/);
  assert.match(engineTraining, /fn feature_fingerprint/);
  assert.match(sampleHelpers, /sample\.positionKey/);
  assert.doesNotMatch(sampleHelpers, /boardCount \?\? 0\}\|\$\{index\}/);
});

test("GPU replay retention prioritizes stronger label sources", async () => {
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const trainingCli = await readFile(path.join(repoRoot, "engine/src/training/cli.rs"), "utf8");
  const engineModelTests = await readFile(path.join(repoRoot, "engine/tests/gpu_model.rs"), "utf8");

  assert.equal(await fileExists(path.join(root, "src/training-replay.ts")), false);
  assert.match(engineTraining, /pub fn append_replay_samples/);
  assert.match(engineTraining, /pub fn dedupe_training_samples/);
  assert.match(engineTraining, /pub fn merge_compatible_samples/);
  assert.match(engineTraining, /pub fn replay_sample_priority/);
  assert.match(engineTraining, /pub fn training_label_priority/);
  assert.match(engineTraining, /pub const MIN_POLICY_REPLAY_FRACTION: f32 = 0\.25/);
  assert.match(engineTraining, /base_label_weight: Option<f32>/);
  assert.match(engineTraining, /label_mass: Option<f32>/);
  assert.match(engineTraining, /observation_count: Option<u32>/);
  assert.match(engineModelTests, /fn replay_dedupe_averages_labels_and_keeps_strongest_policy_target/);
  assert.match(engineModelTests, /fn replay_confidence_is_bounded_across_repeated_observations/);
  assert.match(engineModelTests, /fn replay_dedupe_fingerprints_legacy_samples_without_position_keys/);
  assert.match(engineModelTests, /fn replay_retention_keeps_high_signal_samples_and_policy_supervision/);
  assert.match(engineModelTests, /fn training_label_priority_matches_browser_replay_policy/);
  assert.match(trainingCli, /let samples = crate::gpu::training::dedupe_training_samples\(samples\);/);
  assert.match(trainingCli, /train_value_head_cpu\(\s*&model,\s*&samples,/s);
  assert.match(trainingCli, /select_training_working_set_for_projection\(\s*&samples,/s);
  assert.match(trainingCli, /append_replay_samples\(&buffer, &samples, max_buffer\)/);
  assert.match(trainingCli, /model_path: Some\(gpu_value_model_path\(config\)\.to_string\(\)\)/);
  assert.match(trainingCli, /let model = load_gpu_value_model\(config\);/);
  assert.match(trainingCli, /--gpu-replay-append/);
  assert.match(trainingCli, /--gpu-replay-buffer/);
});
