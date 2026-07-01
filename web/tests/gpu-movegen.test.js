import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const repoRoot = path.resolve(root, "..");
const searchShaderRoot = path.join(repoRoot, "engine/src/gpu/search/shaders");

test("GPU move generation permits historical branch mutation", async () => {
  const shader = await readFile(path.join(searchShaderRoot, "mutate.wgsl"), "utf8");

  assert.doesNotMatch(shader, /STATUS_UNSUPPORTED_HISTORICAL_BRANCH/);
  assert.doesNotMatch(shader, /!same_board\s*&&\s*!target_latest/);
  assert.match(shader, /statuses\[index\]\s*=\s*select\(STATUS_BRANCH_OK,\s*STATUS_BRANCH_ROYAL_CAPTURE/);
});

test("legacy GPU move generation carries en-passant board metadata", async () => {
  const layout = await readFile(path.join(root, "src/ai-layout.ts"), "utf8");
  const snapshot = await readFile(path.join(root, "src/ai-snapshot.ts"), "utf8");
  const shader = await readFile(path.join(searchShaderRoot, "movegen.wgsl"), "utf8");

  assert.match(layout, /GPU_BOARD_STRIDE = 73/);
  assert.match(snapshot, /board\.enPassant\?\.x \?\? -1/);
  assert.match(shader, /const BOARD_EP: u32 = 5u/);
  assert.match(shader, /const BOARD_SQUARE_OFFSET: u32 = 9u/);
  assert.match(shader, /source_ep_x = boards\[source_board_base \+ BOARD_EP\]/);
  assert.match(shader, /ep_x == to_x && ep_y == to_y/);
});

test("full GPU mode uses the resident frontier for parallel timelines", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const caller = worker.slice(0, worker.indexOf("async function tryFullGpuSearch"));

  assert.doesNotMatch(caller, /throw new Error\("Full GPU search currently requires one pending present board\."\)/);
  assert.match(caller, /gpuMode === "full"/);
  assert.match(caller, /tryGpuResidentFrontierSearch/);
  assert.match(worker, /validatedFrontierChoices/);
  assert.match(worker, /validateFirstFrontierTurn/);
  assert.match(caller, /falling back to hybrid GPU search/);
});

test("GPU search returns complete turn plans for post-match review", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");

  assert.match(worker, /const principalVariation = \[moves\]/);
  assert.match(worker, /completedGpuReplyTurn\(device, current/);
  assert.match(worker, /principalVariation\.push\(reply\)/);
  assert.match(worker, /depth:\s*1,[\s\S]*gpuSearch:\s*"projected-reply"/);
  assert.match(worker, /principalVariation:\s*Move\[\]\[\]/);
  assert.match(worker, /const \[best\] = await engineRankedCandidates\(scored, \{\s*pendingBoards,\s*requirePending: true,\s*limit: 1/s);
  assert.match(worker, /engine\.chronofish_gpu_reply_pressure_ranked_roots_bytes\(ptr, byteLength\)/);
  assert.match(worker, /return engineReplyPressureRankedRoots\(rankedRoots, pairScores, rankedReplies\.length\)/);
  assert.match(engineTypes, /chronofish_gpu_reply_pressure_ranked_roots_bytes\(ptr: number, length: number\): number/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_reply_pressure_ranked_roots_bytes/);
  assert.match(engineSearch, /pub fn gpu_reply_pressure_ranked_roots_from_i32s/);
});

test("GPU turn completion only spends moves on pending present boards", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");

  assert.match(worker, /engineRankedCandidates\(scored, \{\s*pendingBoards,\s*requirePending: true/s);
  assert.match(worker, /engine\.chronofish_gpu_ranked_candidate_indexes_bytes\(ptr, byteLength\)/);
  assert.match(worker, /await engineGpuScoringSummary\(scored, pendingBoards\)/);
  assert.match(worker, /engine\.chronofish_gpu_scoring_summary_bytes\(ptr, byteLength\)/);
  assert.match(worker, /await engineGpuMutationSummary\(mutated\)/);
  assert.match(worker, /engine\.chronofish_gpu_mutation_summary_bytes\(ptr, byteLength\)/);
  assert.match(worker, /await engineGpuTurnCompletionKey\(pendingBoards\)/);
  assert.match(worker, /engine\.chronofish_gpu_turn_completion_key_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /function gpuScoringSummary/);
  assert.doesNotMatch(worker, /function moveStartsOnPendingBoard/);
  assert.doesNotMatch(worker, /function gpuMutationSummary/);
  assert.doesNotMatch(worker, /function gpuTurnCompletionKey/);
  assert.match(engineTypes, /chronofish_gpu_ranked_candidate_indexes_bytes\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_scoring_summary_bytes\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_mutation_summary_bytes\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_turn_completion_key_json\(ptr: number, length: number\): number/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_ranked_candidate_indexes_bytes/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_scoring_summary_bytes/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_mutation_summary_bytes/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_turn_completion_key_json/);
  assert.match(engineSearch, /pub fn gpu_ranked_candidate_indexes_from_i32s/);
  assert.match(engineSearch, /pub fn gpu_scoring_summary_from_i32s/);
  assert.match(engineSearch, /pub fn gpu_mutation_summary_from_i32s/);
  assert.match(engineSearch, /pub fn gpu_turn_completion_key_json/);
  assert.match(worker, /findCompleteGpuTurn\(device, snapshot, rootTurn/);
  assert.match(worker, /`\$\{result\.gpuSearch \?\? "gpu"\}-turn-fallback`/);
  assert.match(worker, /choices: \[\]/);
  assert.match(worker, /return Boolean\(result\?\.status === "ok" && result\.moves\?\.length\)/);
  assert.doesNotMatch(worker, /\|\| result\?\.status === "incompleteTurn"/);
});

test("GPU reply sentinel is never used as an evaluation", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /if \(reply\.move\) \{\s*score -= reply\.score/);
  assert.match(worker, /return best \? \{ score: best\.score, move: best\.move \} : \{ score: 0 \}/);
});

test("GPU pending-board filters use engine numeric color normalization", async () => {
  const snapshot = await readFile(path.join(root, "src/ai-snapshot.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const frontier = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");

  assert.match(snapshot, /export function colorCode\(color: Color \| string \| number \| null \| undefined\)/);
  assert.match(snapshot, /typeof color === "number"/);
  assert.match(worker, /const pendingBoards = await enginePendingPresentBoards\(snapshot, snapshot\.turn\)/);
  assert.match(worker, /engineFrontierRootFromSnapshot\(\{ \.\.\.snapshot, turn: color \}, boardCount\)/);
  assert.match(worker, /root\.words\[base \+ GPU_FRONTIER_BOARD_PENDING\]/);
  assert.match(worker, /engineFrontierRootFromSnapshot\(snapshot, tuning\.maxBoards\)/);
  assert.match(engineSearch, /board\.side_to_move == root_color/);
  assert.match(engineSearch, /let turn = gpu_search_color_code\(&snapshot\.turn\)/);
  assert.doesNotMatch(frontier, /board\.sideToMove === snapshot\.turn/);
});

test("hybrid GPU scoring batches dispatches under WebGPU limits", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /const maxDispatchWorkgroups = 65_535/);
  assert.match(worker, /const maxCandidatesPerDispatch = maxDispatchWorkgroups \* 64/);
  assert.match(worker, /Math\.min\(\s*maxDispatchWorkgroups,\s*Math\.ceil\(batchCandidateCount \/ 64\)\s*\)/);
});
