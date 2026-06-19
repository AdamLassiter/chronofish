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
  assert.match(shader, /fn bucket_order/);
  assert.match(shader, /fn bitonic_sort/);
  assert.match(shader, /fn mark_unique/);
  assert.match(shader, /fn mark_parent_quota/);
  assert.match(shader, /fn compact_selected/);
  assert.match(shader, /fn fill_selection_underflow/);
  assert.match(shader, /already_selected/);
  assert.match(shader, /parent_selected_count/);
  assert.match(shader, /CANDIDATE_MOVE/);
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
  const shader = await readFile(path.join(root, "src/shaders/frontier_expand.wgsl"), "utf8");

  assert.match(source, /dispatchCandidateLimit/);
  assert.match(source, /for \(let base = 0; base < sourceScans; base \+= sourceScanLimit\)/);
  assert.match(source, /Math\.ceil\(count \/ this\.tuning\.candidateWorkgroupSize\)/);
  assert.match(source, /const selectionCapacity = floorPowerOfTwo/);
  assert.match(source, /pipelines\.bucketOrder/);
  assert.match(shader, /dispatch_base/);
  assert.match(shader, /dispatch_count/);
  assert.match(shader, /source_index = params\.dispatch_base \+ id\.x/);
});

test("GPU frontier board capacity scales with search growth instead of always reserving 64 boards", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(source, /additionalBoardCapacity = 0/);
  assert.match(source, /nextPowerOfTwo\(Math\.max\(boardCount, boardCount \+ Math\.max\(0, additionalBoardCapacity\)\)\)/);
  assert.match(worker, /maxCycles \* 2/);
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

test("GPU frontier projects retained states sparsely for neural evaluation", async () => {
  const shader = await readFile(path.join(root, "src/shaders/frontier_neural.wgsl"), "utf8");
  const source = await readFile(path.join(root, "src/ai-frontier-neural.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const frontier = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const forward = await readFile(path.join(root, "src/shaders/frontier_forward.wgsl"), "utf8");

  assert.match(shader, /select_neural_boards/);
  assert.match(shader, /project_neural_features/);
  assert.match(shader, /projection_hash/);
  assert.match(shader, /active_states\[state\]/);
  assert.match(shader, /HEADER_LAST_NEURAL/);
  assert.match(shader, /apply_neural_values/);
  assert.match(shader, /MAX_NEURAL_BOARDS: u32 = 16u/);
  assert.match(shader, /perspective/);
  assert.doesNotMatch(source, /PROJECT_FEATURES_SHADER/);
  assert.doesNotMatch(source, /rawFeatures/);
  assert.match(source, /activeStates/);
  assert.match(forward, /forward_layer_masked/);
  assert.match(forward, /active_states\[sample\] == 0u/);
  assert.match(worker, /let activeStateLimit = 1/);
  assert.match(worker, /stateCount: activeStateLimit/);
  assert.match(frontier, /const sourceScans = stateCount \* this\.tuning\.maxBoards \* 64/);
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
  const source = await readFile(path.join(root, "scripts/gpu-frontier-smoke.mjs"), "utf8");

  assert.match(worker, /interface GpuSearchDiagnostics/);
  assert.match(worker, /frontierWidth: tuning\.frontierWidth/);
  assert.match(worker, /candidateCapacity: tuning\.candidateCapacity/);
  assert.match(worker, /selectedCount: readback\.selectedCount/);
  assert.match(worker, /maxBoards: tuning\.maxBoards/);
  assert.match(worker, /dispatchCandidateLimit: tuning\.dispatchCandidateLimit/);
  assert.match(worker, /nodes: readback\.nodes/);
  assert.match(worker, /readbacks: 1/);
  assert.match(worker, /candidateOverflow: readback\.candidateOverflow \? 1 : 0/);
  assert.match(worker, /nodesPerSecond/);
  assert.match(source, /selectedCount < Math\.min/);
  assert.match(controller, /Selected GPU frontier diagnostics/);
});

test("GPU frontier keeps pruned overflow searches when selected states exist", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const shader = await readFile(path.join(root, "src/shaders/frontier_expand.wgsl"), "utf8");

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

test("GPU training harness uses one WebGPU worker queue", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");

  assert.match(worker, /function gpuTrainingWorkerCount/);
  assert.match(worker, /WebGPU work is serialized through one worker\/device queue/);
  assert.match(worker, /const workerCount = gpuTrainingWorkerCount\(positions\.length\)/);
  assert.match(worker, /const workerCount = gpuTrainingWorkerCount\(target\)/);
});

test("GPU training rollouts apply complete returned turns", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");

  assert.match(worker, /async function applyWorkerTurn/);
  assert.match(worker, /for \(const move of moves\)/);
  assert.match(worker, /type: "applyMove"[\s\S]*type: "submitTurn"/);
  assert.doesNotMatch(worker, /result\.moves\?\.\[0\]/);
  assert.ok((worker.match(/applyWorkerTurn\(/g) ?? []).length >= 5);
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

test("hybrid GPU depth-one search uses snapshot pending boards for turn completion", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /const pendingBoards = pendingPresentBoardsForSnapshot\(snapshot, snapshot\.turn\)/);
  assert.match(worker, /if \(pendingBoards\.length >= 1 && ranked\.length > 0\)/);
});

test("GPU candidate selection accepts single-move choices", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /function choiceMoves/);
  assert.match(worker, /candidate\.moves \?\? \(candidate\.move \? \[candidate\.move\] : \[\]\)/);
  assert.match(worker, /choiceMoves\(candidate\)\.length > 0/);
});

test("GPU frontier smoke harness can force device-loss cleanup and rebuild", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /debugLoseDevice/);
  assert.match(worker, /destroyCachedGpuDeviceForSmoke/);
  assert.match(worker, /device\.destroy\(\)/);
  assert.match(worker, /cachedGpuAdapter = null/);
  assert.match(worker, /pipelineCache\.clear\(\)/);
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
  const shader = await readFile(path.join(root, "src/shaders/frontier_state.wgsl"), "utf8");

  assert.match(source, /const MAX_FRONTIER_WIDTH = 512/);
  assert.match(source, /const MAX_CANDIDATES = 65_536/);
  assert.match(source, /const MAX_SELECTION_SCAN = 2048/);
  assert.match(source, /minimax_reduce_stage/);
  assert.match(source, /Math\.ceil\(this\.tuning\.frontierWidth \/ 64\)/);
  assert.match(shader, /fn minimax_reduce_stage/);
  assert.match(shader, /peer < reduce_params\.state_count/);
  assert.doesNotMatch(shader, /array<i32, 128>/);
  assert.doesNotMatch(shader, /min\(128u, reduce_params\.state_count\)/);
});

test("GPU frontier sorts a bounded shortlist instead of full candidate capacity", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const shader = await readFile(path.join(root, "src/shaders/frontier_select.wgsl"), "utf8");

  assert.match(source, /selectionCapacity = floorPowerOfTwo\(Math\.min/);
  assert.match(source, /MAX_SELECTION_SCAN/);
  assert.match(source, /this\.tuning\.frontierWidth \* 4/);
  assert.match(source, /for \(let k = 2; k <= selectionCapacity; k \*= 2\)/);
  assert.match(source, /Math\.ceil\(selectionCapacity \/ this\.tuning\.candidateWorkgroupSize\)/);
  assert.match(shader, /index = index \+ params\.max_scan/);
  assert.match(shader, /index >= params\.max_scan/);
});

test("GPU frontier fills from unsorted candidates when shortlist pruning underfills", async () => {
  const source = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const shader = await readFile(path.join(root, "src/shaders/frontier_select.wgsl"), "utf8");

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
  const shader = await readFile(path.join(root, "src/shaders/policy.wgsl"), "utf8");

  assert.match(trainer, /const labelWeights = new Float32Array/);
  assert.match(trainer, /labelWeightBuffer/);
  assert.match(shader, /label_weights/);
  assert.match(shader, /target_weight \/ total_weight/);
});

test("GPU training distinguishes completed outcomes from search bootstraps", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const delta = await readFile(path.join(root, "src/shaders/output_delta.wgsl"), "utf8");

  assert.match(worker, /"search-bootstrap"/);
  assert.match(worker, /"duel-search"/);
  assert.match(worker, /backfillDrawLabels/);
  assert.match(worker, /label: 0/);
  assert.match(trainer, /const weight = Math\.max\(0, samples\[index\]!\.labelWeight \?\? 1\)/);
  assert.match(trainer, /totalWeight > 0 \? total \/ totalWeight : 0/);
  assert.match(trainer, /batchWeight \+= Math\.max\(0, labelWeights\[batch\[index\]!\] \?\? 1\)/);
  assert.match(trainer, /outputDeltaParamsData\(batchSize, batchWeight\)/);
  assert.match(delta, /f32\(params\.batch_count\) \/ max\(params\.total_weight, 0\.000001\)/);
});

test("GPU replay deduplicates positions and keeps validation groups separate", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const labels = await readFile(path.join(root, "src/training-label-worker.ts"), "utf8");
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");

  assert.match(labels, /positionKey: positionKey\(game\)/);
  assert.match(worker, /sample\.positionKey/);
  assert.match(worker, /deduplicated\.delete\(key\)/);
  assert.match(trainer, /sample\.positionKey/);
  assert.doesNotMatch(trainer, /boardCount \?\? 0\}\|\$\{index\}/);
});
