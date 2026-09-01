import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const webRoot = path.resolve(import.meta.dirname, "..");
const repoRoot = path.resolve(webRoot, "..");

async function readBotControllerSources() {
  const [controller, binding] = await Promise.all([
    readFile(path.join(webRoot, "src/bot-controller.ts"), "utf8"),
    readFile(path.join(webRoot, "src/engine-bot-policy.ts"), "utf8")
  ]);
  return `${controller}\n${binding}`;
}

test("GPU effort presets are model data with minimum depths", async () => {
  const effort = JSON.parse(await readFile(
    path.join(repoRoot, "engine/models/gpu-v1/effort.json"),
    "utf8"
  ));

  for (const name of ["fast", "balanced", "expert"]) {
    assert.equal(typeof effort[name].depth, "number");
    assert.equal(typeof effort[name].minDepth, "number");
    assert.ok(effort[name].minDepth <= effort[name].depth);
    assert.equal(typeof effort[name].nodes, "number");
    assert.equal(typeof effort[name].timeMs, "number");
    assert.equal(effort[name].minDepth, 2);
  }
  assert.deepEqual(
    [effort.fast.timeMs, effort.balanced.timeMs, effort.expert.timeMs],
    [1_500, 5_000, 15_000]
  );
});

test("frontend loads GPU effort separately from CPU effort", async () => {
  const main = await readFile(path.join(webRoot, "src/main.ts"), "utf8");

  assert.match(main, /fetch\("\/ai\/gpu-effort\.json"\)/);
  assert.match(main, /gpuEffortConfigs = await gpuEffortResponse\.json/);
  assert.match(main, /bot-gpu-custom/);
});

test("CPU efforts use bounded alpha-beta search and custom CPU difficulty can override it", async () => {
  const [effortText, main, controller, worker, binding, wasmApi, engineTypes] = await Promise.all([
    readFile(path.join(repoRoot, "engine/models/cpu-v1/effort.json"), "utf8"),
    readFile(path.join(webRoot, "src/main.ts"), "utf8"),
    readFile(path.join(webRoot, "src/bot-controller.ts"), "utf8"),
    readFile(path.join(webRoot, "src/cpu-ai-worker.ts"), "utf8"),
    readFile(path.join(webRoot, "src/engine-cpu-search.ts"), "utf8"),
    readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8"),
    readFile(path.join(webRoot, "src/types.ts"), "utf8")
  ]);
  const effort = JSON.parse(effortText);

  for (const name of ["fast", "balanced", "expert"]) {
    assert.equal(effort[name].searchStrategy, "alpha-beta");
    assert.equal(effort[name].minDepth, 2);
  }
  assert.deepEqual(
    [effort.fast.timeMs, effort.balanced.timeMs, effort.expert.timeMs],
    [1_500, 5_000, 15_000]
  );
  assert.match(main, /timeMs: 5_000,[\s\S]*searchStrategy: "alpha-beta"/);
  assert.match(main, /candidate\.searchStrategy === "beam" \? "beam" : "alpha-beta"/);
  assert.match(main, /customCpuSearchStrategyInput/);
  assert.match(controller, /searchStrategy: effort\.searchStrategy/);
  assert.match(controller, /effort\.searchStrategy === "beam"[\s\S]*targetDepth: 1, minDepth: 1/);
  assert.match(worker, /searchStrategy\?: "alpha-beta" \| "beam"/);
  assert.match(binding, /engine\.chronofish_cpu_search_json\(ptr, len\)/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_cpu_search_json/);
  assert.match(engineTypes, /chronofish_cpu_search_json\(ptr: number, length: number\): number/);
});

test("bot timeout preserves minimum depth and a completed legal result", async () => {
  const controller = await readBotControllerSources();
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(webRoot, "src/types.ts"), "utf8");

  assert.match(controller, /const searchConfig = botSearchConfig\(effort, backend, gpuMode\)/);
  assert.match(controller, /targetDepth: searchConfig\.targetDepth/);
  assert.match(controller, /minDepth: searchConfig\.minDepth/);
  assert.match(controller, /nodes: searchConfig\.nodes/);
  assert.match(controller, /timeMs: searchConfig\.timeMs/);
  assert.match(controller, /"chronofish_bot_search_config_json"/);
  assert.match(controller, /const nextDepth = nextBotSearchDepth\(pending\.currentDepth, pending\.targetDepth\)/);
  assert.match(controller, /chronofish_bot_next_search_depth\(currentDepth, targetDepth\)/);
  assert.match(controller, /chronofish_bot_worker_search_time_ms\(timeMs\)/);
  assert.match(controller, /chronofish_bot_completed_search_depth/);
  assert.doesNotMatch(controller, /DEFAULT_MIN_BOT_SEARCH_DEPTH/);
  assert.doesNotMatch(controller, /effort\.timeMs \?\? 10_000/);
  assert.doesNotMatch(controller, /Math\.max\(1, effort\.nodes \?\? 64\)/);
  assert.doesNotMatch(controller, /return currentDepth <= 0 \? Math\.min\(2, targetDepth\) : Math\.min\(targetDepth, currentDepth \+ 2\)/);
  assert.doesNotMatch(controller, /const margin = Math\.min\(1000, Math\.max\(100, Math\.floor\(timeMs \* 0\.05\)\)\)/);
  assert.match(controller, /minDepth: Math\.min\(nextDepth, pending\.minDepth\)/);
  assert.match(controller, /pending\.currentDepth <= pending\.minDepth && pending\.depthReceived < pending\.depthExpected/);
  assert.match(controller, /function completedSearchDepth/);
  assert.match(controller, /function resultEndsInRoyalCapture/);
  assert.match(controller, /"chronofish_bot_result_ends_in_royal_capture_json"/);
  assert.match(controller, /pending\.currentDepth <= pending\.minDepth/);
  assert.match(controller, /completedDepth >= pending\.minDepth/);
  assert.match(controller, /if \(bestResult && \(bestResult\.depth \?\? 0\) >= pending\.minDepth\)/);
  assert.doesNotMatch(controller, /completedDepth >= 2 && completedDepth % 2 === 0/);
  assert.match(controller, /resultEndsInRoyalCapture\(result\) \? 1 : 0/);
  assert.doesNotMatch(controller, /result\.resultReason === "royal-capture"/);
  assert.doesNotMatch(controller, /result\.gpuTerminal === true \|\| result\.terminal === true/);
  assert.match(controller, /pending\.incompleteDepthAttempt = true/);
  assert.match(controller, /pending\.incompleteDepthAttempt && pending\.currentDepth >= pending\.minDepth/);
  assert.match(controller, /is completing depth/);
  assert.match(controller, /selectDeepestStoredResult\(pending\)/);
  assert.match(controller, /startMinimumDepthCpuFallback\(pending\)/);
  assert.match(controller, /searchStrategy: "alpha-beta"/);
  assert.match(controller, /if \(\(bestResult\?\.depth \?\? 0\) >= pending\.minDepth\)[\s\S]*finishBotSearch\(pending, "complete"\)/);
  assert.match(controller, /is completing minimum depth/);
  assert.match(controller, /minDepth: pending\.minDepth/);
  assert.match(controller, /pending\.bestByDepth\.set\(entry\.depth, depthBest\)/);
  assert.match(controller, /\(bestResult\?\.depth \?\? 0\) >= pending\.targetDepth/);
  assert.match(engineSearch, /pub fn bot_search_depth_at_least_one/);
  assert.match(engineSearch, /pub fn bot_search_config_json/);
  assert.match(engineSearch, /pub fn bot_next_search_depth/);
  assert.match(engineSearch, /pub fn bot_worker_search_time_ms/);
  assert.match(engineSearch, /pub fn bot_completed_search_depth/);
  assert.match(engineSearch, /pub fn bot_result_ends_in_royal_capture_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_bot_search_depth_at_least_one/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_bot_search_config_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_bot_next_search_depth/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_bot_worker_search_time_ms/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_bot_completed_search_depth/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_bot_result_ends_in_royal_capture_json/);
  assert.match(engineTypes, /chronofish_bot_search_depth_at_least_one\(depth: number\): number/);
  assert.match(engineTypes, /chronofish_bot_search_config_json\(depth: number, minDepth: number, nodes: number, timeMs: number\): number/);
  assert.match(engineTypes, /chronofish_bot_next_search_depth\(currentDepth: number, targetDepth: number\): number/);
  assert.match(engineTypes, /chronofish_bot_worker_search_time_ms\(timeMs: number\): number/);
  assert.match(engineTypes, /chronofish_bot_completed_search_depth\(resultDepth: number, requestedDepth: number, resultEndsInRoyalCapture: number\): number/);
  assert.match(engineTypes, /chronofish_bot_result_ends_in_royal_capture_json\(ptr: number, length: number\): number/);
});

test("GPU bots default to the deep frontier while hybrid mode stops honestly at depth two", async () => {
  const controller = await readFile(path.join(webRoot, "src/bot-controller.ts"), "utf8");

  assert.match(controller, /localStorage\.getItem\(GPU_MODE_STORAGE_KEY\) === "hybrid" \? "hybrid" : "full"/);
  assert.match(controller, /backend === "gpu" && gpuMode === "hybrid"/);
  assert.match(controller, /targetDepth: Math\.min\(config\.targetDepth, 2\)/);
  assert.match(controller, /minDepth: Math\.min\(config\.minDepth, 2\)/);
  assert.match(controller, /gpuMode: pending\.gpuMode/);
  assert.match(controller, /result\.gpuMode === "hybrid"/);
  assert.match(controller, /pending\.targetDepth = Math\.min\(pending\.targetDepth, 2\)/);
});

test("bot countdown switches to overtime after the nominal deadline", async () => {
  const controller = await readBotControllerSources();

  assert.match(controller, /function formatBotCountdown\(deadlineAt: number, now = Date\.now\(\)\)/);
  assert.match(controller, /return `\$\{formatBotTimeLimit\(deltaMs\)\} left`/);
  assert.match(controller, /return `\$\{formatBotTimeLimit\(-deltaMs\)\} overtime`/);
  assert.match(controller, /formatBotCountdown\(pending\.deadlineAt\)/);
  assert.doesNotMatch(controller, /Math\.max\(0, pending\.deadlineAt - Date\.now\(\)\)/);
});

test("bot search result ranking prefers deeper completed searches before score", async () => {
  const controller = await readBotControllerSources();
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(webRoot, "src/types.ts"), "utf8");

  assert.match(controller, /"chronofish_bot_ranked_choices_json"/);
  assert.match(controller, /"chronofish_bot_select_best_result_json"/);
  assert.doesNotMatch(controller, /function compareAiResultPreference/);
  assert.doesNotMatch(controller, /function compareBotChoicePreference/);
  assert.doesNotMatch(controller, /function botChoiceScore/);
  assert.doesNotMatch(controller, /function botChoiceDepth/);
  assert.match(engineSearch, /pub fn bot_ranked_choices_json/);
  assert.match(engineSearch, /pub fn bot_select_best_result_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_bot_ranked_choices_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_bot_select_best_result_json/);
  assert.match(engineTypes, /chronofish_bot_ranked_choices_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_bot_select_best_result_json\(ptr: number, length: number\): number/);
});

test("GPU worker honors minimum depth before applying its internal deadline", async () => {
  const worker = `${await readFile(path.join(webRoot, "src/ai-worker.ts"), "utf8")}\n${await readFile(path.join(webRoot, "src/engine-gpu-search.ts"), "utf8")}`;
  const workerTypes = await readFile(path.join(webRoot, "src/ai-worker-types.ts"), "utf8");
  const cpuWorker = `${await readFile(path.join(webRoot, "src/cpu-ai-worker.ts"), "utf8")}\n${await readFile(path.join(webRoot, "src/engine-cpu-search.ts"), "utf8")}`;
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(webRoot, "src/types.ts"), "utf8");

  assert.match(workerTypes, /minDepth\?: number/);
  assert.match(worker, /const searchConfig = await engineGpuSearch\.engineGpuWorkerSearchConfig\(depth, minDepth, timeMs\)/);
  assert.match(worker, /chronofish_gpu_worker_search_config_json\(/);
  assert.match(worker, /engineGpuSearch\.setGpuSearchDeadline\(searchConfig\.deadlineDelayMs == null/);
  assert.match(worker, /Number\.POSITIVE_INFINITY/);
  assert.match(worker, /depth: searchConfig\.requestedDepth/);
  assert.match(worker, /timeMs: searchConfig\.searchTimeMs/);
  assert.doesNotMatch(worker, /const requestedDepth = Math\.max\(1, depth \?\? 1\)/);
  assert.doesNotMatch(worker, /const minimumDepth = Math\.min\(requestedDepth, Math\.max\(1, Math\.floor\(minDepth \?\? 1\)\)\)/);
  assert.doesNotMatch(worker, /Math\.floor\(searchTimeMs \* 0\.8\)/);
  assert.match(engineSearch, /pub fn gpu_worker_search_config_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_worker_search_config_json/);
  assert.match(engineTypes, /chronofish_gpu_worker_search_config_json\(depth: number, minDepth: number, timeMs: number\): number/);
  assert.match(cpuWorker, /const config = this\.searchConfig\(engine, request\)/);
  assert.match(cpuWorker, /engine\.chronofish_cpu_worker_search_config_json\(ptr, len\)/);
  assert.match(cpuWorker, /return this\.searchResult\(engine, JSON\.parse\(readWasmString\(engine, output\)\) as CpuAiResult\)/);
  assert.match(cpuWorker, /engine\.chronofish_cpu_worker_search_result_json\(ptr, len\)/);
  assert.doesNotMatch(cpuWorker, /Math\.max\(1, depth\)/);
  assert.doesNotMatch(cpuWorker, /Math\.max\(1, Math\.min\(depth, minDepth\)\)/);
  assert.doesNotMatch(cpuWorker, /Math\.max\(1, Math\.floor\(timeMs\)\)/);
  assert.doesNotMatch(cpuWorker, /principalVariation \?\?=/);
  assert.doesNotMatch(cpuWorker, /cpuSearch = "heuristic"/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_cpu_worker_search_config_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_cpu_worker_search_result_json/);
  assert.match(engineTypes, /chronofish_cpu_worker_search_config_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_worker_search_result_json\(ptr: number, length: number\): number/);
  assert.match(workerTypes, /resultReason\?: SearchResultReason/);
  assert.match(worker, /chronofish_gpu_validate_search_result_json\(ptr, len\)/);
  assert.match(engineSearch, /"resultReason"/);
  assert.match(engineSearch, /"gpuTerminal"/);
  assert.match(engineSearch, /result\.reason\.as_str\(\) == "royal-capture"/);
  assert.match(cpuWorker, /resultReason\?: "royal-capture" \| "threefold-repetition" \| "stalemate" \| null/);
});

test("bot move choice logging includes full principal variation plans", async () => {
  const controller = await readBotControllerSources();
  const worker = `${await readFile(path.join(webRoot, "src/ai-worker.ts"), "utf8")}\n${await readFile(path.join(webRoot, "src/engine-gpu-search.ts"), "utf8")}`;
  const engineSearch = await readFile(path.join(repoRoot, "engine/src/gpu/search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(webRoot, "src/types.ts"), "utf8");

  assert.match(controller, /plan: formatBotPlan\(choice\.principalVariation \?\? \[choice\.moves\], pending\.game\)/);
  assert.match(controller, /function formatBotPlan/);
  assert.match(controller, /"chronofish_bot_ranked_choices_json"/);
  assert.match(engineSearch, /fn bot_normalized_principal_variation/);
  assert.match(controller, /principalVariation: normalizePrincipalVariation\(choice\.principalVariation, choice\.moves\)/);
  assert.match(controller, /"chronofish_gpu_normalize_principal_variation_json"/);
  assert.doesNotMatch(controller, /\.map\(\(turn\) => turn\.filter\(\(move\) => move\?\.from && move\.to\)\.map\(cloneMove\)\)/);
  assert.match(await readFile(path.join(webRoot, "src/ai-worker-types.ts"), "utf8"), /principalVariation\?: Move\[\]\[\] \| undefined/);
  assert.match(worker, /principalVariation,/);
  assert.match(worker, /engine\.chronofish_gpu_selected_choice_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /await engineSummarizeSearchChoices\(candidates\)/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_gpu_normalize_principal_variation_json/);
  assert.match(engineTypes, /chronofish_gpu_normalize_principal_variation_json\(ptr: number, length: number\): number/);
  assert.match(engineSearch, /pub fn gpu_normalize_principal_variation_json/);
  assert.match(engineSearch, /"principalVariation"/);
});

test("post-match bot review opens principal variation suffixes from clicked boards", async () => {
  const main = await readFile(path.join(webRoot, "src/main.ts"), "utf8");

  assert.match(main, /function botReviewPlanForBoard\(position: Position, snapshot: GameSnapshot\): BotReviewPlanMatch \| null/);
  assert.match(main, /let bestMatch: BotReviewPlanMatch \| null = null/);
  assert.match(main, /let bestReplayOffset = Number\.POSITIVE_INFINITY/);
  assert.match(main, /let replayOffset = 0/);
  assert.match(main, /for \(const decision of botController\.allDecisions\(\)\)/);
  assert.doesNotMatch(main, /allDecisions\(\)\.slice\(\)\.reverse\(\)/);
  assert.match(main, /for \(let turnIndex = 0; turnIndex < decision\.principalVariation\.length; turnIndex \+= 1\)/);
  assert.match(main, /for \(let moveIndex = 0; moveIndex < turn\.length; moveIndex \+= 1\)/);
  assert.match(main, /const decisionBoard = boardAt\(baseSnapshot, move\.from\.timelineId, move\.from\.time\)/);
  assert.match(main, /move\.from\.timelineId === position\.timelineId/);
  assert.match(main, /move\.from\.time === position\.time/);
  assert.match(main, /boardSnapshotKey\(decisionBoard\) === boardSnapshotKey\(clickedBoard\)/);
  assert.match(main, /if \(replayOffset < bestReplayOffset\)/);
  assert.match(main, /if \(bestReplayOffset === 0\)/);
  assert.match(main, /replayOffset \+= 1/);
  assert.match(main, /skipTurns: turnIndex/);
  assert.match(main, /skipMovesInFirstTurn: moveIndex/);
  assert.match(main, /const reviewSnapshot = botReviewProjection\?\.finalGame \?\? game/);
  assert.match(main, /const baseSnapshot = cloneGame\(match\.baseSnapshot\)/);
  assert.match(main, /buildBotReviewPlan\(match\.decision, baseSnapshot, match\.skipTurns, match\.skipMovesInFirstTurn\)/);
  assert.doesNotMatch(main, /selectedTurnClicked/);
  assert.doesNotMatch(main, /applyBotReviewTurn/);
  assert.doesNotMatch(main, /botReviewProjection && snapshotHasBoard\(game, position\)/);
});
