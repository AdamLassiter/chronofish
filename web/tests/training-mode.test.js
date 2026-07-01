import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const repoRoot = path.resolve(root, "..");

async function fileExists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

test("training UI uses mode multiselects instead of target selectors", async () => {
  const html = await readFile(path.join(root, "src/index.html"), "utf8");
  const dom = await readFile(path.join(root, "src/dom.ts"), "utf8");
  const ui = await readFile(path.join(root, "src/training-ui.ts"), "utf8");

  assert.match(html, /id="training-mode" multiple/);
  assert.match(html, /id="training-cpu-mode" multiple/);
  assert.match(html, /value="vsGpu"[\s\S]*vs GPU search/);
  assert.match(html, /value="vsCpu"[\s\S]*vs CPU heuristic/);
  assert.match(html, /value="self"[\s\S]*vs Self/);
  assert.match(html, /value="distill"[\s\S]*Distill/);
  assert.doesNotMatch(html, /id="training-target"/);
  assert.doesNotMatch(html, /id="training-label-mode"/);
  assert.doesNotMatch(html, /id="training-cpu-target"/);
  assert.match(dom, /trainingModeSelect/);
  assert.match(dom, /trainingCpuModeSelect/);
  assert.match(ui, /trainingModes: selectedTrainingModes/);
});

test("training UI exposes low medium high setting presets", async () => {
  const html = await readFile(path.join(root, "src/index.html"), "utf8");
  const dom = await readFile(path.join(root, "src/dom.ts"), "utf8");
  const ui = await readFile(path.join(root, "src/training-ui.ts"), "utf8");

  assert.match(html, /data-training-preset="low"[\s\S]*Low/);
  assert.match(html, /data-training-preset="med" aria-pressed="true"[\s\S]*Med/);
  assert.match(html, /data-training-preset="high"[\s\S]*High/);
  assert.match(dom, /trainingPresetButtons/);
  assert.match(ui, /type TrainingPresetName = "low" \| "med" \| "high"/);
  assert.match(ui, /const TRAINING_PRESETS: Record<TrainingPresetName, TrainingPreset>/);
  assert.match(ui, /button\.addEventListener\("click", \(\) => applyTrainingPreset\(button\.dataset\.trainingPreset\)\)/);
  assert.match(ui, /let defaultTrainingPresetApplied = false/);
  assert.match(ui, /function applyDefaultTrainingPreset/);
  assert.match(ui, /if \(defaultTrainingPresetApplied\) \{/);
  assert.match(ui, /applyTrainingPreset\("med", \{ announce: false \}\)/);
  assert.match(ui, /function applyTrainingPreset/);
  assert.match(ui, /function setActiveTrainingPreset/);
  assert.match(ui, /button\.setAttribute\("aria-pressed", button\.dataset\.trainingPreset === name \? "true" : "false"\)/);
  assert.match(ui, /selectTrainingModes\(trainingModeSelect, preset\.gpuModes\)/);
  assert.match(ui, /selectTrainingModes\(cpuModeSelect, preset\.cpuModes\)/);
  assert.match(ui, /setPresetInput\(cpuSecondsInput, preset\.cpu\.seconds, 1, 86400\)/);
});

test("training worker normalizes modes and hides distill from CPU training", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const workerTypes = await readFile(path.join(root, "src/training-worker-types.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(workerTypes, /type TrainingMode = "vsGpu" \| "vsCpu" \| "self" \| "distill"/);
  assert.match(worker, /const normalizedConfig = await normalizeTrainingConfig\(config\)/);
  assert.match(worker, /async function normalizeTrainingConfigWithEngine/);
  assert.match(worker, /chronofish_normalize_training_config_json\(ptr, len\)/);
  assert.match(worker, /\.\.\.await normalizeTrainingConfigWithEngine\(config\)/);
  assert.doesNotMatch(worker, /function legacyTrainingModes/);
  assert.doesNotMatch(worker, /function normalizeTrainingModes\(/);
  assert.doesNotMatch(worker, /function isTrainingMode/);
  assert.match(worker, /function trainingModeEnabled/);
  assert.match(worker, /function cpuBaselineModeEnabled/);
  assert.match(worker, /function trainingModePolicy/);
  assert.match(worker, /chronofish_training_mode_policy_json\(ptr, len\)/);
  assert.match(worker, /trainingModePolicyCache/);
  assert.doesNotMatch(worker, /return config\.trainingModes\.includes\(mode\)/);
  assert.doesNotMatch(worker, /config\.trainingModes\.filter\(\(mode\) => mode !== "distill"\)\.length/);
  assert.doesNotMatch(worker, /return trainingModeEnabled\(config, "vsCpu"\) \|\| trainingModeEnabled\(config, "self"\)/);
  assert.match(worker, /function modeLabelTarget/);
  assert.match(engineTraining, /pub fn is_training_subject/);
  assert.match(engineTraining, /pub fn is_training_mode/);
  assert.match(engineTraining, /pub fn legacy_training_subject/);
  assert.match(engineTraining, /pub fn legacy_training_modes/);
  assert.match(engineTraining, /pub fn normalize_training_modes/);
  assert.match(engineTraining, /pub fn training_mode_enabled/);
  assert.match(engineTraining, /pub fn cpu_baseline_mode_enabled/);
  assert.match(engineTraining, /pub fn training_mode_count/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_normalize_training_modes_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_training_mode_policy_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_normalize_training_config_json/);
  assert.match(engineTypes, /chronofish_normalize_training_modes_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_training_mode_policy_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_normalize_training_config_json\(ptr: number, length: number\): number/);
  assert.doesNotMatch(worker, /type TrainingLabelMode/);
  assert.doesNotMatch(worker, /type CpuTrainingTarget/);
});

test("GPU worker caps preserve medium and high preset budgets", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const workerTypes = await readFile(path.join(root, "src/training-worker-types.ts"), "utf8");
  const ui = await readFile(path.join(root, "src/training-ui.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.doesNotMatch(workerTypes, /const MAX_GPU_TRAINING_SAMPLES/);
  assert.doesNotMatch(workerTypes, /const MAX_GPU_TRAINING_BATCH/);
  assert.doesNotMatch(workerTypes, /const MAX_GPU_VALIDATION_INTERVAL/);
  assert.doesNotMatch(workerTypes, /const MAX_PARALLEL_GPU_TRAINING_WORKERS/);
  assert.doesNotMatch(workerTypes, /const DEFAULT_BATCH_SIZE/);
  assert.doesNotMatch(workerTypes, /const DEFAULT_VALIDATION_SPLIT/);
  assert.doesNotMatch(workerTypes, /const DEFAULT_PATIENCE/);
  assert.doesNotMatch(workerTypes, /const DEFAULT_WEIGHT_DECAY/);
  assert.match(worker, /chronofish_normalize_training_config_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /function clampInteger/);
  assert.doesNotMatch(worker, /function clampNumber/);
  assert.doesNotMatch(worker, /samples: clampInteger/);
  assert.doesNotMatch(worker, /batchSize: clampInteger/);
  assert.doesNotMatch(worker, /validationInterval: clampInteger/);
  assert.match(worker, /const warmupPlies = gpuWarmupPlies\(workerIndex\)/);
  assert.match(worker, /const warmupConfig = gpuWarmupSearchConfig\(config\)/);
  assert.match(worker, /depth: warmupConfig\.depth/);
  assert.match(worker, /nodes: warmupConfig\.nodes/);
  assert.match(worker, /timeMs: warmupConfig\.timeMs/);
  assert.doesNotMatch(worker, /workerIndex === 0 \? 0 : 1 \+ \(workerIndex % Math\.max\(1, MAX_PLAYOUT_PLIES - 1\)\)/);
  assert.doesNotMatch(worker, /nodes: Math\.max\(1, Math\.min\(1024, config\.nodes\)\)/);
  assert.doesNotMatch(worker, /timeMs: Math\.min\(5000, workerSearchTimeMs\(config\)\)/);
  assert.match(ui, /samples: 8192/);
  assert.match(ui, /batch: 8192/);
  assert.match(ui, /validationInterval: 8192/);
  assert.match(ui, /Math\.min\(highMemory \? 16 : 8, hardwareThreads - 1\)/);
  assert.match(engineTraining, /pub fn clamp_training_integer/);
  assert.match(engineTraining, /pub fn clamp_training_number/);
  assert.match(engineTraining, /pub const MAX_GPU_TRAINING_SAMPLES: usize = 16_384/);
  assert.match(engineTraining, /pub const MAX_GPU_TRAINING_BATCH: usize = 16_384/);
  assert.match(engineTraining, /pub const MAX_GPU_VALIDATION_INTERVAL: usize = 16_384/);
  assert.match(engineTraining, /pub const DEFAULT_BATCH_SIZE: usize = 1024/);
  assert.match(engineTraining, /pub const DEFAULT_VALIDATION_SPLIT: f32 = 0\.1/);
  assert.match(engineTraining, /pub const DEFAULT_PATIENCE: usize = 12/);
  assert.match(engineTraining, /pub const DEFAULT_WEIGHT_DECAY: f32 = 0\.00001/);
  assert.match(engineTraining, /pub fn gpu_warmup_plies/);
  assert.match(engineTraining, /pub fn gpu_warmup_search_config/);
  assert.match(wasmApi, /"samples": clamp_config_integer/);
  assert.match(wasmApi, /"batchSize": clamp_config_integer/);
  assert.match(wasmApi, /"validationInterval": clamp_config_integer/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_warmup_plies/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_gpu_warmup_search_config_json/);
  assert.match(engineTypes, /chronofish_gpu_warmup_plies\(workerIndex: number\): number/);
  assert.match(engineTypes, /chronofish_gpu_warmup_search_config_json\(depth: number, nodes: number, searchTimeMs: number, explorationTemperature: number\): number/);
});

test("browser CPU training screens candidates before full finalist scoring", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const engineCpuSearch = await readFile(path.join(repoRoot, "engine/src/cpu_search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(worker, /const screeningGames = sampleGames\.slice/);
  assert.match(worker, /collectCpuTrainingGames\(game, baseline, config, target, progress\)/);
  assert.match(worker, /const workerCount = cpuTrainingPositionWorkerCount\(target, config\.cpuWorkers\)/);
  assert.match(worker, /const searchConfig = cpuTrainingPositionSearchConfig\(config\)/);
  assert.match(worker, /depth: searchConfig\.depth/);
  assert.match(worker, /nodes: searchConfig\.nodes/);
  assert.doesNotMatch(worker, /const workerCount = Math\.min\(target, Math\.max\(1, config\.cpuWorkers\)\)/);
  assert.doesNotMatch(worker, /depth: Math\.max\(1, Math\.min\(2, config\.cpuDepth\)\)/);
  assert.doesNotMatch(worker, /nodes: Math\.max\(1, Math\.min\(512, config\.cpuNodes\)\)/);
  assert.doesNotMatch(worker, /collectGpuPositions\(game, config, target, progress, "cpu"\)/);
  assert.match(worker, /applyCpuWorkerTurn\(cpu, current, moves, config\)/);
  assert.match(worker, /const screeningConfig = cpuScreeningTrainingConfig\(config\)/);
  assert.match(worker, /`cpu-screen-\$\{generation \+ 1\}`/);
  assert.match(worker, /const finalistCandidates = uniqueCpuParameters/);
  assert.match(worker, /`cpu-train-\$\{generation \+ 1\}`/);
  assert.match(worker, /function cpuScreeningTrainingConfig/);
  assert.match(worker, /const screening = cpuScreeningTrainingConfigWithEngine\(config\)/);
  assert.match(worker, /cpuTrainingTimeMs: screening\.cpuTrainingTimeMs/);
  assert.doesNotMatch(worker, /cpuTrainingTimeMs: Math\.max\(1, Math\.min\(config\.cpuTrainingTimeMs, Math\.ceil\(config\.cpuTrainingTimeMs \/ 4\)\)\)/);
  assert.match(engineCpuSearch, /pub struct CpuScreeningTrainingConfig/);
  assert.match(engineCpuSearch, /pub fn cpu_screening_training_config/);
  assert.match(engineCpuSearch, /pub struct CpuTrainingPositionSearchConfig/);
  assert.match(engineCpuSearch, /pub fn cpu_training_position_worker_count/);
  assert.match(engineCpuSearch, /pub fn cpu_training_position_search_config/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_training_position_worker_count/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_training_position_search_config_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_screening_training_config_json/);
  assert.match(engineTypes, /chronofish_cpu_training_position_worker_count\(target: number, cpuWorkers: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_training_position_search_config_json\(cpuDepth: number, cpuNodes: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_screening_training_config_json\(cpuDepth: number, depth: number, cpuNodes: number, nodes: number, cpuTrainingTimeMs: number\): number/);
});

test("browser CPU training reuses baseline and GPU reference searches", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const workerTypes = await readFile(path.join(root, "src/training-worker-types.ts"), "utf8");
  const engineCpuSearch = await readFile(path.join(repoRoot, "engine/src/cpu_search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(workerTypes, /interface CpuReferenceScore/);
  assert.match(worker, /const screeningReferences = await timed\(config\.metrics, "cpuScreeningReferences"/);
  assert.doesNotMatch(worker, /const finalistReferences = await precomputeCpuReferenceScores/);
  assert.match(worker, /cpuReferenceWorkerCount\(games\.length, config\.cpuWorkers, config\.cpuPairBatch\)/);
  assert.match(worker, /chronofish_cpu_reference_worker_count\(gameCount, requestedWorkers, pairBatch\)/);
  assert.match(worker, /Promise\.all\(Array\.from\(\{ length: workerCount \}, \(\) => runReferenceWorker\(\)\)\)/);
  assert.match(worker, /const uniqueCandidates = uniqueCpuParameters\(stageCandidates\)/);
  assert.match(worker, /const workerCount = cpuCandidateWorkerCount\(uniqueCandidates\.length, stageConfig\.cpuWorkers, stageConfig\.cpuPairBatch\)/);
  assert.match(worker, /chronofish_cpu_candidate_worker_count\(candidateCount, cpuWorkers, pairBatch\)/);
  assert.match(worker, /const workerCount = cpuLabelWorkerCount\(positions\.length, config\.cpuWorkers\)/);
  assert.match(worker, /labelWeight: cpuSearchLabelWeight\(config\)/);
  assert.match(worker, /chronofish_cpu_label_worker_count\(positionCount, cpuWorkers\)/);
  assert.match(worker, /chronofish_cpu_search_label_weight\(trainingModeCount\(config\)\)/);
  assert.doesNotMatch(worker, /Math\.min\(uniqueCandidates\.length, Math\.max\(1, stageConfig\.cpuWorkers\), Math\.max\(1, stageConfig\.cpuPairBatch\)\)/);
  assert.doesNotMatch(worker, /Math\.min\(positions\.length, Math\.max\(1, config\.cpuWorkers \?\? 1\)\)/);
  assert.doesNotMatch(worker, /labelWeight: trainingModeCount\(config\) > 1 \? 1\.1 : 1\.0/);
  assert.match(worker, /scoreCpuCandidate\(candidate, stageGames, references, stageConfig, candidateWorker, stageDeadlineAt\)/);
  assert.match(worker, /fitnessCache\.set\(cpuParametersKey\(candidate\), score\)/);
  assert.match(worker, /cacheHits/);
  assert.match(worker, /async function precomputeCpuReferenceScores/);
  assert.match(worker, /labelKind: "cpu-reference"/);
  assert.match(worker, /reference\.baselineScore = baselineResult\.result\?\.score \?\? 0/);
  assert.match(worker, /reference\.gpuScore = gpuResult\.result\?\.score \?\? 0/);
  assert.match(worker, /const reference = references\[index\] \?\? \{\}/);
  assert.match(worker, /cpuReferenceScoreDelta\(\s*candidateScore,\s*reference\.baselineScore,/);
  assert.match(worker, /cpuReferenceScoreDelta\(\s*candidateScore,\s*reference\.gpuScore,/);
  assert.match(worker, /cpuReferenceCandidateAverage\(score, compared, nearDraws, config\.cpuDrawRateLimit\)/);
  assert.doesNotMatch(worker, /const delta = candidateScore - reference\.baselineScore/);
  assert.doesNotMatch(worker, /if \(Math\.abs\(delta\) <= config\.cpuDrawWindow\)/);
  assert.doesNotMatch(worker, /nearDrawRate > config\.cpuDrawRateLimit \? average \* 0\.5 : average/);
  assert.doesNotMatch(worker, /function moveAgreementBonus/);
  assert.doesNotMatch(worker, /function botTrainingMovesKey/);
  assert.match(engineCpuSearch, /pub struct CpuReferenceScoreDelta/);
  assert.match(engineCpuSearch, /pub fn cpu_reference_score_delta/);
  assert.match(engineCpuSearch, /pub fn cpu_reference_candidate_average/);
  assert.match(engineCpuSearch, /pub fn cpu_candidate_worker_count/);
  assert.match(engineCpuSearch, /pub fn cpu_label_worker_count/);
  assert.match(engineCpuSearch, /pub fn cpu_search_label_weight/);
  assert.match(engineCpuSearch, /pub fn move_agreement_bonus/);
  assert.match(engineCpuSearch, /pub fn bot_training_moves_key/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_cpu_reference_score_delta_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_reference_candidate_average/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_reference_worker_count/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_candidate_worker_count/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_label_worker_count/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_search_label_weight/);
  assert.match(engineTypes, /chronofish_cpu_reference_score_delta_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_reference_candidate_average\(score: number, compared: number, nearDraws: number, drawRateLimit: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_reference_worker_count\(gameCount: number, requestedWorkers: number, pairBatch: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_candidate_worker_count\(candidateCount: number, cpuWorkers: number, pairBatch: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_label_worker_count\(positionCount: number, cpuWorkers: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_search_label_weight\(trainingModeCount: number\): number/);
});

test("browser CPU finalists use paired matches on a common score scale", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const engineCpuSearch = await readFile(path.join(repoRoot, "engine/src/cpu_search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(worker, /scoreCpuCandidateByPairedMatches/);
  assert.match(worker, /for \(const candidateColor of \[game\.turn, oppositeColor\(game\.turn\)\]\)/);
  assert.match(worker, /candidateTurn \? candidateWorker : baselineWorker/);
  assert.match(worker, /applyCpuWorkerTurn\(baselineWorker, current, moves, config\)/);
  assert.match(worker, /parametersJson: candidateTurn \? candidateJson : baselineJson/);
  assert.match(worker, /return cpuTrainingNoMoveScore\(candidateTurn\)/);
  assert.match(worker, /return cpuTrainingWinnerScore\(applied\.winner, candidateColor\)/);
  assert.match(worker, /return cpuTrainingWinnerScore\(applied\.status\.winner \?\? null, candidateColor\)/);
  assert.match(worker, /return cpuTrainingAdjudicationScore\(current\.turn, candidateColor, baselineScore\)/);
  assert.doesNotMatch(worker, /return candidateTurn \? -100_000 : 100_000/);
  assert.doesNotMatch(worker, /applied\.winner === candidateColor \? 100_000 : -100_000/);
  assert.match(worker, /parametersJson: baselineJson/);
  assert.doesNotMatch(worker, /current\.turn === candidateColor \? baselineScore : -baselineScore/);
  assert.match(worker, /chronofish_cpu_match_turn_time_ms/);
  assert.match(worker, /config\.cpuMaxMatchPlies - ply \+ 1/);
  assert.doesNotMatch(worker, /function cpuMatchTurnTimeMs[\s\S]*Math\.floor\(\(deadlineAt - performance\.now\(\)\)/);
  assert.match(engineCpuSearch, /pub const CPU_TRAINING_WIN_SCORE: i32 = 100_000/);
  assert.match(engineCpuSearch, /pub fn cpu_training_no_move_score/);
  assert.match(engineCpuSearch, /pub fn cpu_training_winner_score/);
  assert.match(engineCpuSearch, /pub fn cpu_training_adjudication_score/);
  assert.match(engineCpuSearch, /pub fn cpu_match_turn_time_ms/);
  assert.match(engineCpuSearch, /pub fn cpu_training_budget_ms/);
  assert.match(engineCpuSearch, /pub fn cpu_training_position_target/);
  assert.match(engineCpuSearch, /pub fn mode_label_target/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_training_no_move_score/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_cpu_training_winner_score_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_cpu_training_adjudication_score_json/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_match_turn_time_ms/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_mode_label_target/);
  assert.match(engineTypes, /chronofish_cpu_training_no_move_score\(candidateTurn: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_training_winner_score_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_training_adjudication_score_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_match_turn_time_ms\(cpuTrainingTimeMs: number, nowMs: number, deadlineAtMs: number, remainingSearches: number\): number/);
  assert.match(engineTypes, /chronofish_mode_label_target\(samples: number, trainingModeCount: number, divisor: number\): number/);
});

test("browser CPU training performs bounded evolutionary generations", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const engineCpuSearch = await readFile(path.join(repoRoot, "engine/src/cpu_search.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(worker, /breedCpuPopulation\(baseline, \[\], candidateCount/);
  assert.match(worker, /chronofish_breed_cpu_population_json\(ptr, len\)/);
  assert.match(worker, /chronofish_unique_cpu_parameters_json\(ptr, len\)/);
  assert.match(worker, /chronofish_cpu_parameters_key_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /from "\.\/training-cpu\.js"/);
  assert.equal(await fileExists(path.join(root, "src/training-cpu.ts")), false);
  assert.match(worker, /const candidateCount = cpuTrainingCandidateCount\(config\)/);
  assert.match(worker, /const finalistTarget = cpuTrainingFinalistTarget\(config, population\.length, screened\.length\)/);
  assert.match(worker, /\.slice\(0, cpuTrainingEliteCount\(config\)\)/);
  assert.doesNotMatch(worker, /const candidateCount = Math\.max\(1, Math\.min\(256, config\.cpuCandidates\)\)/);
  assert.doesNotMatch(worker, /const finalistTarget = Math\.min/);
  assert.doesNotMatch(worker, /\.slice\(0, Math\.max\(1, Math\.min\(4, config\.cpuFinalists\)\)\)/);
  assert.match(worker, /generationsWithoutCandidate < config\.cpuMaxGenerationsWithoutCandidate/);
  assert.match(worker, /labelKind: "cpu-generation"/);
  assert.match(worker, /bestCandidate \? \[bestCandidate\.parameters, \.\.\.elites\] : elites/);
  assert.match(worker, /winner\.score > baselineScore/);
  assert.match(worker, /timed\(config\.metrics, "cpuPositions"/);
  assert.match(worker, /chronofish_cpu_training_position_target/);
  assert.match(worker, /chronofish_cpu_training_budget_ms/);
  assert.doesNotMatch(worker, /const variantPairs = \(config\.cpuOpponentVariants \+ config\.cpuScreeningOpponentVariants\) \* config\.cpuRoundsPerVariant/);
  assert.doesNotMatch(worker, /const fallbackMs = Math\.min\(/);
  assert.match(worker, /timed\(config\.metrics, "cpuScreeningReferences"/);
  assert.match(worker, /timed\(config\.metrics, "cpuScreening"/);
  assert.match(worker, /timed\(config\.metrics, "cpuFinalists"/);
  assert.match(worker, /chronofish_mode_label_target/);
  assert.doesNotMatch(worker, /return Math\.max\(1, Math\.ceil\(config\.samples \/ divisor\)\)/);
  assert.match(engineCpuSearch, /pub fn breed_cpu_population/);
  assert.match(engineCpuSearch, /pub fn mutate_cpu_parameters/);
  assert.match(engineCpuSearch, /pub fn crossover_cpu_parameters/);
  assert.match(engineCpuSearch, /pub fn unique_cpu_parameters/);
  assert.match(engineCpuSearch, /pub fn cpu_parameters_key/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_breed_cpu_population_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_unique_cpu_parameters_json/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_cpu_parameters_key_json/);
  assert.match(engineCpuSearch, /pub fn cpu_reference_worker_count/);
  assert.match(engineCpuSearch, /pub const MAX_CPU_TRAINING_CANDIDATES: usize = 256/);
  assert.match(engineCpuSearch, /pub const MAX_CPU_TRAINING_ELITES: usize = 4/);
  assert.match(engineCpuSearch, /pub fn cpu_training_candidate_count/);
  assert.match(engineCpuSearch, /pub fn cpu_training_finalist_target/);
  assert.match(engineCpuSearch, /pub fn cpu_training_elite_count/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_training_position_target/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_training_budget_ms/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_training_candidate_count/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_training_finalist_target/);
  assert.match(wasmApi, /pub extern "C" fn chronofish_cpu_training_elite_count/);
  assert.match(engineTypes, /chronofish_cpu_training_position_target\(/);
  assert.match(engineTypes, /chronofish_breed_cpu_population_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_unique_cpu_parameters_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_parameters_key_json\(ptr: number, length: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_training_budget_ms\(cpuTrainSeconds: number, cpuTrainingTimeMs: number, cpuMaxMatchPlies: number, cpuMaxMatchTimeMs: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_training_candidate_count\(cpuCandidates: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_training_finalist_target\(populationLen: number, cpuFinalists: number, cpuPairBatch: number, screenedLen: number\): number/);
  assert.match(engineTypes, /chronofish_cpu_training_elite_count\(cpuFinalists: number\): number/);
});

test("training UI reports GPU working set size separately from replay size", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const ui = await readFile(path.join(root, "src/training-ui.ts"), "utf8");

  assert.match(worker, /trainingSampleCount: model\.trainingSampleCount/);
  assert.match(worker, /policyTrainingSampleCount: model\.policyTrainingSampleCount/);
  assert.match(ui, /trainingSampleCount\?: number/);
  assert.match(ui, /policyTrainingSampleCount\?: number/);
  assert.match(ui, /trainingSampleCount,/);
  assert.match(ui, /trainingMetric\("Train", trainingSampleCount\)/);
  assert.match(ui, /trainingMetric\("Policy N", policyTrainingSampleCount\)/);
  assert.match(ui, /trainingMetric\("Replay", replaySize \?\? "\?"/);
});

test("training reports active-model validation baselines and checkpoint decisions", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const ui = await readFile(path.join(root, "src/training-ui.ts"), "utf8");

  assert.match(worker, /initialValidationLoss: model\.initialValidationLoss/);
  assert.match(worker, /valueCheckpointImproved: model\.valueCheckpointImproved/);
  assert.match(worker, /policyCheckpointImproved: model\.policyCheckpointImproved/);
  assert.match(worker, /modelChanged: model\.modelChanged/);
  assert.match(ui, /trainingMetric\("Initial Val", formatLoss\(initialValidationLoss\)\)/);
  assert.match(ui, /valueCheckpointImproved \? "improved" : "unchanged"/);
  assert.match(ui, /if \(modelChanged === false\)/);
  assert.match(ui, /title: "Model Unchanged"/);
});

test("training label source counts have engine-owned policy", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const trainer = await readFile(path.join(root, "src/training-gpu.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");
  const wasmApi = await readFile(path.join(repoRoot, "engine/src/wasm_api.rs"), "utf8");
  const engineTypes = await readFile(path.join(root, "src/types.ts"), "utf8");

  assert.match(worker, /labelSourceCountsWithEngine/);
  assert.match(worker, /chronofish_label_source_counts_json\(ptr, len\)/);
  assert.match(worker, /normalizedConfig\.labelCounts = labelCounts/);
  assert.doesNotMatch(worker, /function labelSourceCounts\(samples/);
  assert.match(trainer, /function configuredLabelCounts\(config: TrainingConfig\)/);
  assert.match(trainer, /model\.labelCounts = configuredLabelCounts\(config\)/);
  assert.doesNotMatch(trainer, /function labelSourceCounts/);
  assert.doesNotMatch(trainer, /sample\.labelKind \?\? \(sample\.pseudo \? "distilled" : "unknown"\)/);
  assert.match(engineTraining, /pub fn label_source_counts/);
  assert.match(wasmApi, /pub unsafe extern "C" fn chronofish_label_source_counts_json/);
  assert.match(engineTypes, /chronofish_label_source_counts_json\(ptr: number, length: number\): number/);
});

test("training closes replay databases and transfers candidate models to validation workers", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const storage = await readFile(path.join(root, "src/training-worker-storage.ts"), "utf8");
  const constants = await readFile(path.join(root, "src/training-gpu-constants.ts"), "utf8");
  const engineTraining = await readFile(path.join(repoRoot, "engine/src/gpu/training.rs"), "utf8");

  assert.match(constants, /export const TRAINING_IO_TIMEOUT_MS = 15_000/);
  assert.match(worker, /TRAINING_IO_TIMEOUT_MS[\s\S]*from "\.\/training-gpu-constants\.js"/);
  assert.match(storage, /TRAINING_IO_TIMEOUT_MS[\s\S]*from "\.\/training-gpu-constants\.js"/);
  assert.match(worker, /TRAINING_IO_TIMEOUT_MS, \[candidateModel\]/);
  assert.match(engineTraining, /pub const TRAINING_IO_TIMEOUT_MS: u64 = 15_000/);
  assert.match(engineTraining, /pub const LABEL_REQUEST_MIN_TIMEOUT_MS: u64 = 30_000/);
  assert.match(engineTraining, /pub const LABEL_REQUEST_MAX_TIMEOUT_MS: u64 = 120_000/);
  assert.match(engineTraining, /pub const LABEL_REQUEST_NODE_TIMEOUT_FACTOR_MS: u64 = 3/);
  assert.match(engineTraining, /pub fn worker_request_timeout_ms/);
  assert.match(engineTraining, /pub fn worker_search_time_ms/);
  assert.match(worker, /transfer: Transferable\[\] = \[\]/);
  assert.match(worker, /worker\.postMessage\(\{[\s\S]*\}, transfer\)/);
  assert.equal(storage.match(/db\?\.close\(\)/g)?.length, 2);
});

test("training UI labels CPU reference and screening phases", async () => {
  const ui = await readFile(path.join(root, "src/training-ui.ts"), "utf8");

  assert.match(ui, /"cpu-positions": 5/);
  assert.match(ui, /"cpu-reference": 6/);
  assert.match(ui, /"cpu-screen": 7/);
  assert.match(ui, /"cpu-reference": "CPU References"/);
  assert.match(ui, /"cpu-positions": "CPU Positions"/);
  assert.match(ui, /"cpu-screen": "CPU Screening"/);
  assert.match(ui, /entry\.labelKind === "cpu-reference"/);
  assert.match(ui, /return "references"/);
});
