import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

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

  assert.match(worker, /type TrainingMode = "vsGpu" \| "vsCpu" \| "self" \| "distill"/);
  assert.match(worker, /trainingModes: normalizeTrainingModes/);
  assert.match(worker, /subject === "cpu" \? legacy\.filter\(\(mode\) => mode !== "distill"\)/);
  assert.match(worker, /function legacyTrainingModes/);
  assert.match(worker, /function trainingModeEnabled/);
  assert.match(worker, /function cpuBaselineModeEnabled/);
  assert.match(worker, /function modeLabelTarget/);
  assert.doesNotMatch(worker, /type TrainingLabelMode/);
  assert.doesNotMatch(worker, /type CpuTrainingTarget/);
});

test("GPU worker caps preserve medium and high preset budgets", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");
  const ui = await readFile(path.join(root, "src/training-ui.ts"), "utf8");

  assert.match(worker, /const MAX_GPU_TRAINING_SAMPLES = 16384/);
  assert.match(worker, /const MAX_GPU_TRAINING_BATCH = 16384/);
  assert.match(worker, /const MAX_GPU_VALIDATION_INTERVAL = 16384/);
  assert.match(worker, /const MAX_PARALLEL_GPU_TRAINING_WORKERS = 16/);
  assert.match(worker, /samples: clampInteger\(config\.samples, 1, MAX_GPU_TRAINING_SAMPLES, 64\)/);
  assert.match(worker, /batchSize: clampInteger\(config\.batchSize, 16, MAX_GPU_TRAINING_BATCH, DEFAULT_BATCH_SIZE\)/);
  assert.match(worker, /validationInterval: clampInteger\(config\.validationInterval, 16, MAX_GPU_VALIDATION_INTERVAL, 256\)/);
  assert.match(ui, /samples: 8192/);
  assert.match(ui, /batch: 8192/);
  assert.match(ui, /validationInterval: 8192/);
  assert.match(ui, /Math\.min\(highMemory \? 16 : 8, hardwareThreads - 1\)/);
});

test("browser CPU training screens candidates before full finalist scoring", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");

  assert.match(worker, /const screeningGames = sampleGames\.slice/);
  assert.match(worker, /collectCpuTrainingGames\(game, baseline, config, target, progress\)/);
  assert.doesNotMatch(worker, /collectGpuPositions\(game, config, target, progress, "cpu"\)/);
  assert.match(worker, /applyCpuWorkerTurn\(cpu, current, moves, config\)/);
  assert.match(worker, /const screeningConfig = cpuScreeningTrainingConfig\(config\)/);
  assert.match(worker, /`cpu-screen-\$\{generation \+ 1\}`/);
  assert.match(worker, /const finalistCandidates = uniqueCpuParameters/);
  assert.match(worker, /`cpu-train-\$\{generation \+ 1\}`/);
  assert.match(worker, /function cpuScreeningTrainingConfig/);
  assert.match(worker, /cpuTrainingTimeMs: Math\.max\(1, Math\.min\(config\.cpuTrainingTimeMs, Math\.ceil\(config\.cpuTrainingTimeMs \/ 4\)\)\)/);
});

test("browser CPU training reuses baseline and GPU reference searches", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");

  assert.match(worker, /interface CpuReferenceScore/);
  assert.match(worker, /const screeningReferences = await timed\(config\.metrics, "cpuScreeningReferences"/);
  assert.doesNotMatch(worker, /const finalistReferences = await precomputeCpuReferenceScores/);
  assert.match(worker, /cpuReferenceWorkerCount\(games\.length, config\.cpuWorkers, config\.cpuPairBatch\)/);
  assert.match(worker, /Promise\.all\(Array\.from\(\{ length: workerCount \}, \(\) => runReferenceWorker\(\)\)\)/);
  assert.match(worker, /const uniqueCandidates = uniqueCpuParameters\(stageCandidates\)/);
  assert.match(worker, /scoreCpuCandidate\(candidate, stageGames, references, stageConfig, candidateWorker, stageDeadlineAt\)/);
  assert.match(worker, /fitnessCache\.set\(cpuParametersKey\(candidate\), score\)/);
  assert.match(worker, /cacheHits/);
  assert.match(worker, /async function precomputeCpuReferenceScores/);
  assert.match(worker, /labelKind: "cpu-reference"/);
  assert.match(worker, /reference\.baselineScore = baselineResult\.result\?\.score \?\? 0/);
  assert.match(worker, /reference\.gpuScore = gpuResult\.result\?\.score \?\? 0/);
  assert.match(worker, /const reference = references\[index\] \?\? \{\}/);
  assert.match(worker, /moveAgreementBonus\(candidateResult\.result\?\.moves, reference\.baselineMoves\)/);
  assert.match(worker, /moveAgreementBonus\(candidateResult\.result\?\.moves, reference\.gpuMoves\)/);
});

test("browser CPU finalists use paired matches on a common score scale", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");

  assert.match(worker, /scoreCpuCandidateByPairedMatches/);
  assert.match(worker, /for \(const candidateColor of \[game\.turn, oppositeColor\(game\.turn\)\]\)/);
  assert.match(worker, /candidateTurn \? candidateWorker : baselineWorker/);
  assert.match(worker, /applyCpuWorkerTurn\(baselineWorker, current, moves, config\)/);
  assert.match(worker, /parametersJson: candidateTurn \? candidateJson : baselineJson/);
  assert.match(worker, /applied\.winner === candidateColor \? 100_000 : -100_000/);
  assert.match(worker, /parametersJson: baselineJson/);
  assert.match(worker, /current\.turn === candidateColor \? baselineScore : -baselineScore/);
  assert.match(worker, /const remainingMatches = Math\.max\(1, totalMatches - completed\)/);
  assert.match(worker, /config\.cpuMaxMatchPlies - ply \+ 1/);
});

test("browser CPU training performs bounded evolutionary generations", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");

  assert.match(worker, /breedCpuPopulation\(baseline, \[\], candidateCount/);
  assert.match(worker, /generationsWithoutCandidate < config\.cpuMaxGenerationsWithoutCandidate/);
  assert.match(worker, /labelKind: "cpu-generation"/);
  assert.match(worker, /bestCandidate \? \[bestCandidate\.parameters, \.\.\.elites\] : elites/);
  assert.match(worker, /winner\.score > baselineScore/);
  assert.match(worker, /timed\(config\.metrics, "cpuPositions"/);
  assert.match(worker, /timed\(config\.metrics, "cpuScreeningReferences"/);
  assert.match(worker, /timed\(config\.metrics, "cpuScreening"/);
  assert.match(worker, /timed\(config\.metrics, "cpuFinalists"/);
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

test("training closes replay databases and transfers candidate models to validation workers", async () => {
  const worker = await readFile(path.join(root, "src/training-worker.ts"), "utf8");

  assert.match(worker, /TRAINING_IO_TIMEOUT_MS, \[candidateModel\]/);
  assert.match(worker, /transfer: Transferable\[\] = \[\]/);
  assert.match(worker, /worker\.postMessage\(\{[\s\S]*\}, transfer\)/);
  assert.equal(worker.match(/db\?\.close\(\)/g)?.length, 2);
});

test("training UI labels CPU reference and screening phases", async () => {
  const ui = await readFile(path.join(root, "src/training-ui.ts"), "utf8");

  assert.match(ui, /"cpu-positions": 3/);
  assert.match(ui, /"cpu-reference": 4/);
  assert.match(ui, /"cpu-screen": 5/);
  assert.match(ui, /"cpu-reference": "CPU References"/);
  assert.match(ui, /"cpu-positions": "CPU Positions"/);
  assert.match(ui, /"cpu-screen": "CPU Screening"/);
  assert.match(ui, /entry\.labelKind === "cpu-reference"/);
  assert.match(ui, /return "references"/);
});
