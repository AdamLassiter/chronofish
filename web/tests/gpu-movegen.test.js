import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const repoRoot = path.resolve(root, "..");
const searchShaderRoot = path.join(repoRoot, "engine/src/gpu/search/shaders");

async function readGpuSearchSources() {
  const [worker, bindings] = await Promise.all([
    readFile(path.join(root, "src/ai-worker.ts"), "utf8"),
    readFile(path.join(root, "src/engine-gpu-search.ts"), "utf8")
  ]);
  return `${worker}\n${bindings}`;
}

test("GPU move generation permits historical branch mutation", async () => {
  const shader = await readFile(path.join(searchShaderRoot, "mutate.wgsl"), "utf8");

  assert.doesNotMatch(shader, /STATUS_UNSUPPORTED_HISTORICAL_BRANCH/);
  assert.doesNotMatch(shader, /!same_board\s*&&\s*!target_latest/);
  assert.match(shader, /statuses\[index\]\s*=\s*select\(STATUS_BRANCH_OK,\s*STATUS_BRANCH_ROYAL_CAPTURE/);
});

test("legacy GPU move generation carries en-passant board metadata", async () => {
  const layout = await readFile(path.join(root, "src/ai-layout.ts"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");
  const shader = await readFile(path.join(searchShaderRoot, "movegen.wgsl"), "utf8");

  assert.match(layout, /GPU_BOARD_STRIDE = 73/);
  assert.match(engineSearch, /push_en_passant_record\(&mut record, en_passant\)/);
  assert.match(shader, /const BOARD_EP: u32 = 5u/);
  assert.match(shader, /const BOARD_SQUARE_OFFSET: u32 = 9u/);
  assert.match(shader, /source_ep_x = boards\[source_board_base \+ BOARD_EP\]/);
  assert.match(shader, /ep_x == to_x && ep_y == to_y/);
});

test("full GPU mode uses the resident frontier for parallel timelines", async () => {
  const worker = await readGpuSearchSources();
  const caller = worker;

  assert.doesNotMatch(caller, /throw new Error\("Full GPU search currently requires one pending present board\."\)/);
  assert.match(caller, /gpuMode === "full"/);
  assert.match(caller, /tryGpuResidentFrontierSearch/);
  assert.match(worker, /validatedFrontierChoices/);
  assert.match(worker, /validateFirstFrontierTurn/);
  assert.match(worker, /engine\.chronofish_gpu_validate_first_frontier_turn_json\(ptr, len\)/);
  assert.doesNotMatch(worker.slice(worker.indexOf("async function validateFirstFrontierTurn"), worker.indexOf("async function validateSearchResultBeforePost")), /chronofish_apply_move|chronofish_submit_turn/);
  assert.match(worker, /engineValidatedFrontierChoice\(candidate, moves, seenKeys, choices\.length, 12, gpuSearch\)/);
  assert.doesNotMatch(worker, /seen\.has\(key\)/);
  assert.match(caller, /falling back to hybrid GPU search/);
});

test("GPU search returns complete turn plans for post-match review", async () => {
  const worker = await readGpuSearchSources();
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");

  assert.match(worker, /const principalVariation = \[moves\]/);
  assert.match(worker, /completedGpuReplyTurn\(device, current/);
  assert.match(worker, /principalVariation\.push\(reply\)/);
  assert.match(worker, /depth:\s*1,[\s\S]*gpuSearch:\s*"projected-reply"/);
  assert.match(worker, /principalVariation:\s*Move\[\]\[\]/);
  assert.match(worker, /const \[best\] = await engineGpuSearch\.engineRankedCandidates\(scored, \{\s*pendingBoards,\s*requirePending: true,\s*limit: 1/s);
  assert.match(worker, /engine\.chronofish_gpu_reply_pressure_ranked_roots_json\(ptr, len\)/);
  assert.match(worker, /return engineGpuSearch\.engineReplyPressureRankedRoots\(rankedRoots, pairScores, rankedReplies\.length\)/);
  const replyPressureHelper = worker.slice(worker.indexOf("async function engineReplyPressureRankedRoots"), worker.indexOf("function readEngineFrontierRoot"));
  assert.match(replyPressureHelper, /pairScores: Array\.from\(pairScores\)/);
  assert.doesNotMatch(replyPressureHelper, /pairScores\.subarray\(0, rankedRoots\.length \* replyCount\)/);
  assert.doesNotMatch(worker, /new Map\(rankedRoots\.map/);
  assert.doesNotMatch(worker, /ranked\.push\(\{ \.\.\.root, score:/);
  assert.match(engineTypes, /chronofish_gpu_reply_pressure_ranked_roots_bytes\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_reply_pressure_ranked_roots_json\(ptr: number, length: number\): number/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_reply_pressure_ranked_roots_bytes/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_reply_pressure_ranked_roots_json/);
  assert.match(engineSearch, /pub fn gpu_reply_pressure_ranked_roots_from_i32s/);
  assert.match(engineSearch, /pub fn gpu_reply_pressure_ranked_roots_json/);
});

test("GPU turn completion only spends moves on pending present boards", async () => {
  const worker = await readGpuSearchSources();
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");

  assert.match(worker, /engineGpuSearch\.engineRankedCandidates\(scored, \{\s*pendingBoards,\s*requirePending: true/s);
  assert.match(worker, /engine\.chronofish_gpu_ranked_candidates_json\(ptr, len\)/);
  const rankedCandidatesHelper = worker.slice(worker.indexOf("async function engineRankedCandidates"), worker.indexOf("async function engineGpuScoringSummary"));
  assert.match(rankedCandidatesHelper, /records: Array\.from\(scored\.records\)/);
  assert.doesNotMatch(rankedCandidatesHelper, /candidateCount/);
  assert.doesNotMatch(rankedCandidatesHelper, /subarray\(0, candidateCount \* GPU_CANDIDATE_STRIDE\)/);
  assert.doesNotMatch(worker, /engine\.chronofish_gpu_ranked_candidates_json_bytes\(ptr, byteLength\)/);
  assert.doesNotMatch(worker, /moveFromCandidateRecord/);
  assert.match(worker, /await engineGpuSearch\.engineGpuScoringSummary\(scored, pendingBoards\)/);
  assert.match(worker, /engine\.chronofish_gpu_scoring_summary_json\(ptr, len\)/);
  const scoringSummaryHelper = worker.slice(worker.indexOf("async function engineGpuScoringSummary"), worker.indexOf("async function engineGpuMutationSummary"));
  assert.match(scoringSummaryHelper, /records: Array\.from\(scored\.records\)/);
  assert.doesNotMatch(scoringSummaryHelper, /candidateCount/);
  assert.doesNotMatch(scoringSummaryHelper, /subarray\(0, candidateCount \* GPU_CANDIDATE_STRIDE\)/);
  assert.doesNotMatch(worker, /engine\.chronofish_gpu_scoring_summary_bytes\(ptr, byteLength\)/);
  assert.match(worker, /await engineGpuSearch\.engineGpuMutationSummary\(mutated\)/);
  assert.match(worker, /engine\.chronofish_gpu_mutation_summary_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /engine\.chronofish_gpu_mutation_summary_bytes\(ptr, byteLength\)/);
  assert.match(worker, /supportedMutatedCandidates\(mutated/);
  assert.match(worker, /engine\.chronofish_gpu_supported_mutation_candidate_indexes_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /engine\.chronofish_gpu_supported_mutation_candidate_indexes_bytes\(ptr, byteLength\)/);
  assert.match(worker, /candidates,\s*options\.limit,\s*requireChildBoards/s);
  assert.doesNotMatch(worker, /Math\.max\(0, Math\.floor\(options\.limit \?\? 0\)\)/);
  assert.match(worker, /gpuMutationStatusIsTerminal\(.*mutationStatus/);
  assert.match(worker, /engine\.chronofish_gpu_mutation_status_is_terminal\(status\)/);
  assert.match(worker, /await engineGpuSearch\.engineGpuTurnCompletionStep\(current, moves\.length, pendingBoards, status\)/);
  assert.match(worker, /await engineGpuSearch\.engineGpuTurnCompletionStep\(snapshot, moves\.length, pendingBoards, status, visited\)/);
  assert.match(worker, /engine\.chronofish_gpu_turn_status_json\(ptr, len\)/);
  assert.match(worker, /await engineGpuSearch\.engineFullSearchPrecondition\(turnStatus\)/);
  assert.match(worker, /engine\.chronofish_gpu_full_search_precondition_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /turnStatus\.pendingPresentBoardCount !== 1/);
  assert.doesNotMatch(worker, /engine\.chronofish_gpu_turn_status_json_bytes\(ptr, byteLength\)/);
  assert.match(worker, /engine\.chronofish_gpu_turn_completion_step_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /async function engineGpuTurnCompletionKey/);
  assert.doesNotMatch(worker, /async function engineGpuTurnCompletionMaxMoves/);
  assert.match(worker, /await engineGpuSearch\.engineNonPostableResultSummary\(gpuResult\)/);
  assert.match(worker, /engine\.chronofish_gpu_non_postable_result_summary_json\(ptr, len\)/);
  assert.match(worker, /gpuResult && await engineGpuSearch\.enginePostableSearchResult\(gpuResult\)/);
  assert.match(worker, /engine\.chronofish_gpu_postable_search_result_json\(ptr, len\)/);
  assert.match(worker, /return await engineGpuSearch\.engineGpuSearchFailureSummary\(snapshot\)/);
  assert.match(worker, /engine\.chronofish_gpu_search_failure_summary_json\(ptr, len\)/);
  assert.match(worker, /await engineGpuSearch\.engineWithCompletedTurnChoice\(result, result\.moves, result\.gpuSearch\)/);
  assert.match(worker, /engine\.chronofish_gpu_completed_turn_choice_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /function gpuScoringSummary/);
  assert.doesNotMatch(worker, /function moveStartsOnPendingBoard/);
  assert.doesNotMatch(worker, /function gpuMutationSummary/);
  assert.doesNotMatch(worker, /function hasSupportedChildBoards/);
  assert.doesNotMatch(worker, /\.filter\(hasSupportedChildBoards\)/);
  assert.doesNotMatch(worker, /mutationStatus >= GPU_MUTATION_STATUS_OK/);
  assert.doesNotMatch(worker, /mutationStatus === GPU_MUTATION_STATUS_ROYAL_CAPTURE/);
  assert.doesNotMatch(worker, /mutationStatus !== GPU_MUTATION_STATUS_ROYAL_CAPTURE/);
  assert.doesNotMatch(worker, /function gpuTurnCompletionKey/);
  assert.doesNotMatch(worker, /function nonPostableResultSummary/);
  assert.doesNotMatch(worker, /function isPostableSearchResult/);
  assert.doesNotMatch(worker, /function withCompletedTurnChoice/);
  assert.doesNotMatch(worker, /function sameMoveSequence/);
  assert.match(engineTypes, /chronofish_gpu_ranked_candidate_indexes_bytes\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_ranked_candidates_json_bytes\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_ranked_candidates_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_turn_status_json_bytes\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_turn_status_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_full_search_precondition_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_scoring_summary_bytes\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_scoring_summary_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_mutation_summary_bytes\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_mutation_summary_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_supported_mutation_candidate_indexes_bytes\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_supported_mutation_candidate_indexes_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_mutation_status_is_terminal\(status: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_turn_completion_key_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_turn_completion_step_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_non_postable_result_summary_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_postable_search_result_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_search_failure_summary_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_completed_turn_choice_json\(ptr: number, length: number\): number/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_ranked_candidate_indexes_bytes/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_ranked_candidates_json_bytes/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_ranked_candidates_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_turn_status_json_bytes/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_turn_status_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_full_search_precondition_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_scoring_summary_bytes/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_scoring_summary_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_mutation_summary_bytes/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_mutation_summary_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_supported_mutation_candidate_indexes_bytes/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_supported_mutation_candidate_indexes_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_mutation_status_is_terminal/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_turn_completion_key_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_turn_completion_step_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_non_postable_result_summary_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_postable_search_result_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_search_failure_summary_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_completed_turn_choice_json/);
  assert.match(engineSearch, /pub fn gpu_ranked_candidate_indexes_from_i32s/);
  assert.match(engineSearch, /pub fn gpu_ranked_candidates_json_from_i32s/);
  assert.match(engineSearch, /pub fn gpu_ranked_candidates_json/);
  assert.match(engineSearch, /pub fn gpu_turn_status_json_from_i32s/);
  assert.match(engineSearch, /pub fn gpu_turn_status_json/);
  assert.match(engineSearch, /pub fn gpu_full_search_precondition_json/);
  assert.match(engineSearch, /pub fn gpu_scoring_summary_from_i32s/);
  assert.match(engineSearch, /pub fn gpu_scoring_summary_json/);
  assert.match(engineSearch, /pub fn gpu_mutation_summary_from_i32s/);
  assert.match(engineSearch, /pub fn gpu_mutation_summary_json/);
  assert.match(engineSearch, /pub fn gpu_supported_mutation_candidate_indexes_from_i32s/);
  assert.match(engineSearch, /pub fn gpu_supported_mutation_candidate_indexes_json/);
  assert.match(engineSearch, /pub fn gpu_mutation_status_is_terminal/);
  assert.match(engineSearch, /pub fn gpu_turn_completion_key_json/);
  assert.match(engineSearch, /pub fn gpu_turn_completion_step_json/);
  assert.match(engineSearch, /pub fn gpu_non_postable_result_summary_json/);
  assert.match(engineSearch, /pub fn gpu_postable_search_result_json/);
  assert.match(engineSearch, /pub fn gpu_search_failure_summary_json/);
  assert.match(engineSearch, /pub fn gpu_completed_turn_choice_json/);
  assert.match(worker, /findCompleteGpuTurn\(device, snapshot, rootTurn/);
  assert.match(worker, /`\$\{result\.gpuSearch \?\? "gpu"\}-turn-fallback`/);
  assert.match(worker, /choices: \[\]/);
  assert.doesNotMatch(worker, /return Boolean\(result\?\.status === "ok" && result\.moves\?\.length\)/);
  assert.doesNotMatch(worker, /\|\| result\?\.status === "incompleteTurn"/);
});

test("GPU reply sentinel is never used as an evaluation", async () => {
  const worker = await readGpuSearchSources();

  assert.match(worker, /if \(reply\.move\) \{\s*score -= reply\.score/);
  assert.match(worker, /return best \? \{ score: best\.score, move: best\.move \} : \{ score: 0 \}/);
});

test("GPU pending-board filters use engine numeric color normalization", async () => {
  const [snapshotSource, bindings] = await Promise.all([
    readFile(path.join(root, "src/ai-snapshot.ts"), "utf8"),
    readFile(path.join(root, "src/engine-gpu-search.ts"), "utf8")
  ]);
  const snapshot = `${snapshotSource}\n${bindings}`;
  const worker = await readGpuSearchSources();
  const frontier = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");

  assert.match(snapshot, /export function colorCode\(color: Color \| string \| number \| null \| undefined, engine\?: ChronofishEngine\)/);
  assert.match(snapshot, /return engineGpuSearchColorCode\(engine, color\)/);
  assert.match(snapshot, /engine\.chronofish_gpu_search_color_code_json\(ptr, len\)/);
  assert.match(snapshot, /typeof color === "number"/);
  assert.match(worker, /const pendingBoards = await engineGpuSearch\.enginePendingPresentBoards\(snapshot, snapshot\.turn\)/);
  assert.match(worker, /chronofish_gpu_pending_present_boards_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /root\.words\[base \+ GPU_FRONTIER_BOARD_PENDING\]/);
  assert.match(worker, /engineGpuSearch\.engineFrontierRootFromSnapshot\(snapshot, tuning\.maxBoards\)/);
  assert.match(engineSearch, /gpu_pending_present_boards_json_from_snapshot_json/);
  assert.match(engineSearch, /board\.side_to_move == root_color/);
  assert.match(engineSearch, /let turn = gpu_search_color_code\(&snapshot\.turn\)/);
  assert.doesNotMatch(frontier, /board\.sideToMove === snapshot\.turn/);
});

test("hybrid GPU scoring batches dispatches under WebGPU limits", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");

  assert.match(worker, /engineGpuCandidateMaxCandidatesPerBatch\(engine, maxBindingSize\)/);
  assert.match(worker, /engineGpuCandidateSourceBatchSize\(engine, maxCandidatesPerBatch, inputs\.targetCount\)/);
  assert.match(worker, /engineGpuCandidateScoreWorkgroups\(engine, batchCandidateCount\)/);
  assert.match(worker, /engineGpuMutationCandidateWorkgroups\(engine, limit\)/);
  assert.match(worker, /engineGpuReplyScoreWorkgroupsX\(engine, rankedRoots\.length\)/);
  assert.match(worker, /engineGpuReplyScoreWorkgroupsY\(engine, rankedReplies\.length\)/);
  assert.doesNotMatch(worker, /const maxDispatchWorkgroups = 65_535/);
  assert.doesNotMatch(worker, /const maxCandidatesPerDispatch = maxDispatchWorkgroups \* 64/);
  assert.doesNotMatch(worker, /Math\.min\(\s*maxDispatchWorkgroups,\s*Math\.ceil\(batchCandidateCount \/ 64\)\s*\)/);
  assert.doesNotMatch(worker, /Math\.ceil\(limit \/ 64\)/);
  assert.doesNotMatch(worker, /Math\.ceil\(rankedRoots\.length \/ 16\)/);
  assert.doesNotMatch(worker, /Math\.ceil\(rankedReplies\.length \/ 16\)/);
  assert.match(engineTypes, /chronofish_gpu_candidate_score_workgroups\(batchCandidateCount: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_mutation_candidate_workgroups\(candidateLimit: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_reply_score_workgroups_x\(rootCount: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_reply_score_workgroups_y\(replyCount: number\): number/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_candidate_score_workgroups/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_mutation_candidate_workgroups/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_reply_score_workgroups_x/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_reply_score_workgroups_y/);
  assert.match(engineSearch, /pub fn gpu_candidate_score_workgroups/);
  assert.match(engineSearch, /pub fn gpu_mutation_candidate_workgroups/);
  assert.match(engineSearch, /pub fn gpu_reply_score_workgroups_x/);
  assert.match(engineSearch, /pub fn gpu_reply_score_workgroups_y/);
});
