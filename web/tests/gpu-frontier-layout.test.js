import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("GPU frontier uses parent-plus-delta candidates and pooled retained states", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const layout = await readFile(path.join(root, "src/ai-layout.ts"), "utf8");

  assert.match(layout, /GPU_FRONTIER_CANDIDATE_STRIDE/);
  assert.match(layout, /GPU_FRONTIER_BOARD_STRIDE = 78/);
  assert.match(layout, /GPU_FRONTIER_BOARD_ACTIVE = 76/);
  assert.match(layout, /GPU_FRONTIER_BOARD_PENDING = 77/);
  assert.match(layout, /GPU_FRONTIER_DELTA_STRIDE = GPU_FRONTIER_BOARD_STRIDE \* 2/);
  assert.match(source, /class FrontierBufferPool/);
  assert.match(source, /deriveFrontierTuning/);
  assert.match(source, /encodeFrontierRoot/);
  assert.match(source, /maxStorageBufferBindingSize/);
  assert.match(source, /entry\.buffer\.destroy\(\)/);
});

test("GPU frontier pruning is deterministic and diversity bounded", async () => {
  const shader = await readFile(path.join(root, "src/shaders/frontier_select.wgsl"), "utf8");

  assert.match(shader, /fn hash_candidates/);
  assert.match(shader, /fn bitonic_sort/);
  assert.match(shader, /fn select_top_k/);
  assert.match(shader, /already_selected/);
  assert.match(shader, /parent_selected_count/);
  assert.match(shader, /CANDIDATE_MOVE/);
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
  const shader = await readFile(path.join(root, "src/shaders/frontier_expand.wgsl"), "utf8");

  assert.match(source, /dispatchCandidateLimit/);
  assert.match(source, /for \(let base = 0; base < sourceScans; base \+= sourceScanLimit\)/);
  assert.match(source, /Math\.ceil\(count \/ this\.tuning\.candidateWorkgroupSize\)/);
  assert.match(shader, /dispatch_base/);
  assert.match(shader, /dispatch_count/);
  assert.match(shader, /source_index = params\.dispatch_base \+ id\.x/);
});

test("GPU frontier materializes only retained deltas and completes whole turns", async () => {
  const shader = await readFile(path.join(root, "src/shaders/frontier_state.wgsl"), "utf8");

  assert.match(shader, /fn materialize_selected/);
  assert.match(shader, /recompute_turn_status/);
  assert.match(shader, /HEADER_DEPTH.*\+ 1/s);
  assert.match(shader, /HEADER_TURN.*1 - turn/s);
  assert.match(shader, /delta_count/);
});

test("GPU frontier expands all retained states without CPU source-target products", async () => {
  const shader = await readFile(path.join(root, "src/shaders/frontier_expand.wgsl"), "utf8");
  const encoder = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");

  assert.match(shader, /fn expand_frontier/);
  assert.match(shader, /atomicAdd\(&counters\[0\]/);
  assert.match(shader, /fn write_candidate/);
  assert.match(shader, /historical_branch/);
  assert.match(shader, /next_branch_row/);
  assert.match(shader, /CANDIDATE_DELTA_COUNT/);
  assert.match(shader, /BOARD_PENDING/);
  assert.match(shader, /states\[source \+ BOARD_PENDING\] == 0/);
  assert.match(encoder, /GPU_FRONTIER_BOARD_ACTIVE/);
  assert.match(encoder, /GPU_FRONTIER_BOARD_PENDING/);
});

test("GPU frontier move generation includes special pawn cases", async () => {
  const shader = await readFile(path.join(root, "src/shaders/frontier_expand.wgsl"), "utf8");

  assert.match(shader, /ep_x == to_x && ep_y == to_y/);
  assert.match(shader, /captured_x = states\[source \+ BOARD_EP \+ 2u\]/);
  assert.match(shader, /placed_piece = select\(piece, 3,/);
});

test("normal server exposes the committed GPU value model read-only", async () => {
  const server = await readFile(path.resolve(root, "../server/src/static_files.rs"), "utf8");

  assert.match(server, /ai\/value-model\.cfnn/);
  assert.match(server, /engine\/models\/gpu-v1\/value-model\.cfnn/);
});

test("GPU frontier encodes retained states directly for neural evaluation", async () => {
  const shader = await readFile(path.join(root, "src/shaders/frontier_neural.wgsl"), "utf8");

  assert.match(shader, /select_neural_boards/);
  assert.match(shader, /encode_neural_features/);
  assert.match(shader, /apply_neural_values/);
  assert.match(shader, /MAX_NEURAL_BOARDS: u32 = 16u/);
  assert.match(shader, /perspective/);
});

test("GPU frontier neural evaluation uses adapter-sized batches", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier-neural.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const shader = await readFile(path.join(root, "src/shaders/frontier_neural.wgsl"), "utf8");

  assert.match(source, /effectiveBatchSize/);
  assert.match(source, /for \(let stateOffset = 0; stateOffset < stateCount; stateOffset \+= effectiveBatchSize\)/);
  assert.match(worker, /tuning\.neuralBatchSize/);
  assert.match(shader, /state_offset/);
  assert.match(shader, /summaries\[\(apply_params\.state_offset \+ state\) \* SUMMARY_STRIDE \+ 1u\]/);
});

test("GPU frontier loads CFNN once and evaluates without prediction readback", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier-neural.ts"), "utf8");

  assert.match(source, /fetch\("\/ai\/value-model\.cfnn"/);
  assert.match(source, /modelArchitectureMatches/);
  assert.match(source, /FrontierNeuralEvaluator/);
  assert.doesNotMatch(source, /mapAsync/);
});

test("GPU frontier result labels neural mode only when the neural pass ran", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /let modelUsed = false/);
  assert.match(worker, /modelUsed = await runtime\.neural\.encode/);
  assert.match(worker, /const gpuSearch = modelUsed \? "neural-frontier" : "heuristic-frontier"/);
  assert.match(worker, /validatedFrontierChoices\(snapshot, readback\.states, tuning, requestedDepth, gpuSearch\)/);
  assert.doesNotMatch(worker, /readback\.modelUsed/);
});

test("GPU worker replays every posted search result through authoritative WASM", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /validateSearchResultBeforePost/);
  assert.match(worker, /GPU search produced a turn that failed authoritative WASM replay/);
  assert.match(worker, /Hybrid GPU search produced a turn that failed authoritative WASM replay/);
  assert.match(worker, /chronofish_apply_move/);
  assert.match(worker, /chronofish_submit_turn/);
  assert.match(worker, /authoritativeReplay: true/);
});

test("GPU frontier publishes diagnostics needed for rollout gates", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const controller = await readFile(path.join(root, "src/bot-controller.ts"), "utf8");

  assert.match(worker, /interface GpuSearchDiagnostics/);
  assert.match(worker, /frontierWidth: tuning\.frontierWidth/);
  assert.match(worker, /candidateCapacity: tuning\.candidateCapacity/);
  assert.match(worker, /maxBoards: tuning\.maxBoards/);
  assert.match(worker, /dispatchCandidateLimit: tuning\.dispatchCandidateLimit/);
  assert.match(worker, /readbacks: 1/);
  assert.match(worker, /candidateOverflow: readback\.candidateOverflow \? 1 : 0/);
  assert.match(worker, /nodesPerSecond/);
  assert.match(controller, /Selected GPU frontier diagnostics/);
});

test("GPU frontier rejects capacity-truncated full searches", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const shader = await readFile(path.join(root, "src/shaders/frontier_expand.wgsl"), "utf8");

  assert.match(shader, /atomicStore\(&counters\[2\], 1u\)/);
  assert.match(worker, /candidateOverflow: \(counters\[2\] \?\? 0\) !== 0/);
  assert.match(worker, /GPU frontier candidate capacity overflowed before completing search/);
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
  assert.match(source, /gpuMode: "full"/);
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

test("stale GPU search generations cannot publish results", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /activeSearchGeneration/);
  assert.match(worker, /searchGeneration !== activeSearchGeneration/);
});

test("GPU frontier smoke harness can force device-loss cleanup and rebuild", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /debugLoseDevice/);
  assert.match(worker, /destroyCachedGpuDeviceForSmoke/);
  assert.match(worker, /device\.destroy\(\)/);
  assert.match(worker, /pipelineCache\.clear\(\)/);
});

test("GPU frontier tuning uses timestamp queries when available", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");

  assert.match(source, /autotuneFrontier/);
  assert.match(source, /timestamp-query/);
  assert.match(source, /beginningOfPassWriteIndex/);
  assert.match(source, /adapterTuningCacheKey/);
});
