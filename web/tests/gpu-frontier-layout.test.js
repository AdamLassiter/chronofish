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
  assert.match(shader, /BOARD_LATEST\] != 0 && states\[base \+ BOARD_ACTIVE\] != 0/);
  assert.match(shader, /if \(states\[base \+ BOARD_ORIGIN\] != 0\) \{ return 3; \}/);
  assert.match(shader, /category\(state, board\) >= 4/);
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
  assert.match(source, /modelBuffersFromBytes/);
  assert.doesNotMatch(source, /mapAsync/);
});

test("GPU workers validate an in-memory candidate model before promotion", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const trainer = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const ui = await readFile(path.join(root, "src/training-ui.ts"), "utf8");

  assert.match(worker, /type === "setModel"/);
  assert.match(worker, /frontierModelOverride = modelBytes/);
  assert.match(worker, /new FrontierNeuralEvaluator\(device, frontierModelOverride\)/);
  assert.match(trainer, /type: "setModel"/);
  assert.match(trainer, /modelBytes: candidateModel/);
  assert.match(trainer, /temperature: 0/);
  assert.match(trainer, /sampleSeed\("loss-log"/);
  assert.match(ui, /const modelBytes = exactArrayBuffer\(model\)/);
  assert.match(ui, /validateTrainingLossLogs\(trainingConfig\(\), modelBytes\)/);
  assert.match(ui, /if \(logValidation\?\.failed\)/);
  assert.match(ui, /title: "Model Rejected"/);
});

test("GPU frontier applies serialized policy priors before candidate pruning", async () => {
  const frontier = await readFile(path.join(root, "src/ai-frontier.ts"), "utf8");
  const neural = await readFile(path.join(root, "src/ai-frontier-neural.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const shader = await readFile(path.join(root, "src/shaders/frontier_policy.wgsl"), "utf8");
  const stateShader = await readFile(path.join(root, "src/shaders/frontier_state.wgsl"), "utf8");

  assert.match(frontier, /await scoreCandidates\?\.\(encoder, buffers, candidateCapacity\)/);
  assert.match(neural, /async encodePolicyPrior/);
  assert.match(neural, /model\.policyWeights\?\.length === expected/);
  assert.match(neural, /policyWeightsForModel/);
  assert.match(neural, /#currentPolicyFeatures/);
  assert.match(neural, /advancePolicyFeatures/);
  assert.match(neural, /FRONTIER_POLICY_SHADER/);
  assert.match(worker, /runtime\.neural\.encodePolicyPrior/);
  assert.match(worker, /"gpu-v1-cfnn-v3-policy-head"/);
  assert.match(shader, /fn policy_bucket/);
  assert.match(shader, /CANDIDATE_CARRY/);
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

test("GPU training harness uses bounded parallel WebGPU workers", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");

  assert.match(worker, /function gpuTrainingWorkerCount/);
  assert.match(worker, /const MAX_PARALLEL_GPU_TRAINING_WORKERS = 16/);
  assert.match(worker, /Math\.min\(MAX_PARALLEL_GPU_TRAINING_WORKERS, Math\.floor\(requestedWorkers\) \|\| 1\)/);
  assert.match(worker, /const workerCount = gpuTrainingWorkerCount\(positions\.length, config\.searchWorkers\)/);
  assert.match(worker, /const workerCount = gpuTrainingWorkerCount\(target, config\.selfPlayWorkers\)/);
  assert.match(worker, /collectGpuPositions\(game, config, target, progress, "search", config\.searchWorkers\)/);
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
  const lossShader = await readFile(path.join(root, "src/shaders/policy_loss.wgsl"), "utf8");

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
  assert.match(trainer, /const batchIndices = new Uint32Array\(batchSize\)/);
  assert.match(trainer, /const batchWeight = fillGroupedTrainingBatchIndices\(batchIndices, trainGroups, epoch, split\.seed, labelWeights\)/);
  assert.match(trainer, /outputDeltaParamsData\(batchSize, batchWeight\)/);
  assert.match(delta, /f32\(params\.batch_count\) \/ max\(params\.total_weight, 0\.000001\)/);
  assert.match(delta, /\* label_weights\[dataset_sample\]/);
});

test("GPU distillation labels searched positions instead of duplicating the root snapshot", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");

  assert.match(worker, /const positions = await collectGpuPositions\([\s\S]*"distilled"/);
  assert.match(worker, /const samples = positions\.map\(\(position\) => position\.sample\)/);
  assert.match(worker, /const labels = await predictValues\(samples, activeModel\)/);
  assert.doesNotMatch(worker, /const positions = await collectSamples\(game, config, true/);
});

test("GPU training samples uniform minibatches and applies label weights once", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");

  assert.match(trainer, /export function fillGroupedTrainingBatchIndices/);
  assert.match(trainer, /export function groupTrainingIndicesByPosition/);
  assert.match(trainer, /state = xorshift32\(state \|\| 1\)/);
  assert.match(trainer, /const group = trainGroups\[state % trainGroups\.length\]!/);
  assert.match(trainer, /const selected = group\[state % group\.length\]!/);
  assert.match(trainer, /batch\[index\] = selected/);
  assert.match(trainer, /batchWeight \+= Math\.max\(0, labelWeights\[selected\] \?\? 1\)/);
  assert.doesNotMatch(trainer, /trainingWeightCdf/);
  assert.doesNotMatch(trainer, /weightedTrainingIndex/);
  assert.doesNotMatch(trainer, /const epochOrder = shuffledIndices\(trainIndices, epoch, split\.seed\)/);
});

test("GPU training validation split falls back to a high-signal holdout", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");

  assert.match(trainer, /validationSplit > 0 && !validationIndices\.length && trainIndices\.length > 1/);
  assert.match(trainer, /movePositionGroupToValidation\(samples, trainIndices, validationIndices, seed\)/);
  assert.match(trainer, /groupTrainingIndicesByPosition\(samples, trainIndices\)/);
  assert.match(trainer, /function validationSamplePriority/);
  assert.match(trainer, /trainingLabelPriority\(sample\.labelKind, sample\.pseudo\)/);
  assert.match(trainer, /Math\.max\(0, sample\.labelWeight \?\? 1\)/);
});

test("GPU training selects a device-sized high-signal working set", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  const replay = await readFile(path.join(root, "src/training-replay.ts"), "utf8");

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
  assert.match(trainer, /import \{ trainingLabelPriority \} from "\.\/training-replay\.js"/);
  assert.match(replay, /export function trainingLabelPriority/);
});

test("GPU training checkpoint loss reuses projected replay buffers", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const lossShader = await readFile(path.join(root, "src/shaders/reduce_loss.wgsl"), "utf8");

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
  const shaders = await Promise.all([
    "forward_layer.wgsl",
    "forward_indexed_layer.wgsl",
    "apply_layer.wgsl",
    "apply_indexed_layer.wgsl",
    "hidden_delta.wgsl",
    "policy.wgsl"
  ].map((name) => readFile(path.join(root, "src/shaders", name), "utf8")));

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
  assert.match(trainer, /denseKernelEntryPoint\("forward_layer", batchSize\)/);
  assert.match(trainer, /denseKernelEntryPoint\("apply_layer", batchSize\)/);
  assert.match(trainer, /denseKernelEntryPoint\("hidden_delta", batchSize\)/);
  assert.match(trainer, /denseKernelEntryPoint\("forward_policy", batchSize\)/);
  assert.match(trainer, /denseKernelEntryPoint\("apply_policy", batchSize\)/);
});

test("GPU training unlocks hidden-layer backpropagation only with enough unique positions", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");

  assert.match(constants, /export const MIN_HIDDEN_TRAINING_POSITIONS = 256/);
  assert.match(trainer, /const hiddenLayersTrained = uniqueTrainingPositionCount\(samples, trainIndices\) >= MIN_HIDDEN_TRAINING_POSITIONS/);
  assert.match(trainer, /const deltaBuffers = hiddenLayersTrained/);
  assert.match(trainer, /const hiddenDeltaPipeline = hiddenLayersTrained/);
  assert.match(trainer, /const applyLayerPipeline = hiddenLayersTrained/);
  assert.match(trainer, /if \(hiddenLayersTrained\) \{\s*const lastLayerIndex/);
  assert.match(trainer, /model\.hiddenLayersTrained = value\.hiddenLayersTrained/);
});

test("GPU and CPU-head optimizers retain momentum without checkpoint readbacks", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  const shaders = await Promise.all([
    "apply_output.wgsl",
    "apply_layer.wgsl",
    "apply_indexed_layer.wgsl",
    "policy.wgsl"
  ].map((name) => readFile(path.join(root, "src/shaders", name), "utf8")));

  assert.match(constants, /export const OPTIMIZER_MOMENTUM = 0\.9/);
  assert.match(trainer, /const velocityBuffers = hiddenLayersTrained/);
  assert.match(trainer, /const outputVelocityBuffer = zeroStorageBuffer/);
  assert.match(trainer, /const policyVelocityBuffer = zeroStorageBuffer/);
  assert.match(trainer, /encodePipelineBindings\(device, encoder, forwardPipeline/);
  assert.match(trainer, /\[8, policyVelocityBuffer\]/);
  assert.match(trainer, /optimizerVelocity\(velocity\[input\] \?\? 0, update\)/);
  assert.match(trainer, /optimizerVelocity\(velocity\[index\] \?\? 0, update\)/);
  for (const shader of shaders) {
    assert.match(shader, /momentum: f32/);
    assert.match(shader, /var<storage, read_write> velocity: array<f32>/);
    assert.match(shader, /params\.momentum \* velocity/);
  }
});

test("GPU value inference restores the normalized training score scale", async () => {
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const shader = await readFile(path.join(root, "src/shaders/frontier_neural.wgsl"), "utf8");
  const forward = await readFile(path.join(root, "src/shaders/forward_output.wgsl"), "utf8");
  const frontierForward = await readFile(path.join(root, "src/shaders/frontier_forward.wgsl"), "utf8");
  const delta = await readFile(path.join(root, "src/shaders/output_delta.wgsl"), "utf8");

  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  assert.match(constants, /export const VALUE_SCORE_SCALE = 20_000/);
  assert.match(worker, /return normalizedSearchScore\(score\)/);
  assert.match(shader, /clamp\(predictions\[state\] \* apply_params\.value_scale \+ apply_params\.value_bias, -1\.0, 1\.0\)/);
  assert.match(shader, /neural \* 20000\.0/);
  assert.doesNotMatch(shader, /neural \* 100\.0/);
  assert.match(forward, /predictions\[sample\] = tanh\(sum\)/);
  assert.match(frontierForward, /output_values\[sample\] = tanh\(sum\)/);
  assert.match(delta, /\* \(1\.0 - prediction \* prediction\)/);
  assert.match(trainer, /outputWeights\[outputSize\] = inverseTanh/);
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
  const shader = await readFile(path.join(root, "src/shaders/project_features.wgsl"), "utf8");

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
});

test("training label workers encode positions on CPU and transfer typed feature buffers", async () => {
  const worker = await readFile(path.join(root, "src/training-label-worker.ts"), "utf8");
  const encoding = await readFile(path.join(root, "src/training-encoding.ts"), "utf8");

  assert.match(worker, /encodeNeuralPositionFeatures\(game, game\.turn\)/);
  assert.match(worker, /samples\.map\(\(sample\) => sample\.features\.buffer\)/);
  assert.match(worker, /\[sample\.features\.buffer\]/);
  assert.doesNotMatch(worker, /navigator\.gpu/);
  assert.doesNotMatch(worker, /onSubmittedWorkDone/);
  assert.doesNotMatch(worker, /readFloats/);
  assert.match(encoding, /new Float32Array\(NEURAL_INPUT_SIZE\)/);
  assert.match(encoding, /values\.fill/);
  assert.match(encoding, /neuralBoardSelection/);
});

test("GPU replay deduplicates positions and keeps validation groups separate", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const replay = await readFile(path.join(root, "src/training-replay.ts"), "utf8");
  const labels = await readFile(path.join(root, "src/training-label-worker.ts"), "utf8");
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");

  assert.match(labels, /positionKey: positionKey\(game\)/);
  assert.match(worker, /import \{ appendReplaySamples, dedupeTrainingSamples \} from "\.\/training-replay\.js"/);
  assert.match(worker, /const collectedSamples = await timed\(metrics, "collect"/);
  assert.match(worker, /const samples = dedupeTrainingSamples\(collectedSamples\)/);
  assert.match(replay, /sample\.positionKey/);
  assert.match(replay, /deduplicated\.delete\(key\)/);
  assert.match(replay, /function replaySampleKey/);
  assert.match(replay, /function featureFingerprint/);
  assert.match(replay, /\$\{sample\.positionKey\}\|\$\{labelKind\}/);
  assert.match(replay, /features:\$\{features\.length\}:\$\{hash\.toString\(16\)\}/);
  assert.doesNotMatch(replay, /\$\{sample\.positionKey\}\|\$\{sample\.labelKind \?\? "unknown"\}\|\$\{sample\.policy/);
  assert.match(trainer, /sample\.positionKey/);
  assert.doesNotMatch(trainer, /boardCount \?\? 0\}\|\$\{index\}/);
});

test("GPU replay retention prioritizes stronger label sources", async () => {
  const replay = await readFile(path.join(root, "src/training-replay.ts"), "utf8");

  assert.match(replay, /const MIN_POLICY_REPLAY_FRACTION = 0\.25/);
  assert.match(replay, /const requiredPolicyCount = Math\.min/);
  assert.match(replay, /replayHasPolicyTarget/);
  assert.match(replay, /replaySamplePriority\(sample, index, values\.length\)/);
  assert.match(replay, /trainingLabelPriority\(sample\.labelKind, sample\.pseudo\)/);
  assert.match(replay, /labelKind === "outcome" \|\| labelKind === "duel"/);
  assert.match(replay, /labelKind === "search" \|\| labelKind === "cpu"/);
  assert.match(replay, /labelKind === "distilled" \|\| pseudo/);
  assert.match(replay, /const existing = deduplicated\.get\(key\)/);
  assert.match(replay, /mergeCompatibleSamples\(existing\.sample, sample\)/);
  assert.match(replay, /label: totalMass > 0/);
  assert.match(replay, /labelWeight: strongestWeight \* confidence/);
  assert.match(replay, /observationCount: Math\.min\(observationCount, 64\)/);
  assert.match(replay, /\.sort\(\(left, right\) => right\.priority - left\.priority \|\| right\.index - left\.index\)/);
  assert.match(replay, /\.sort\(\(left, right\) => left\.index - right\.index\)/);
});
