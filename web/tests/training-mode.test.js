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
