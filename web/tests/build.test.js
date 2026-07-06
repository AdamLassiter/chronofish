import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const packageJson = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));

test("package exposes the browser training benchmark", () => {
  assert.equal(
    packageJson.scripts["training:benchmark"],
    "node scripts/gpu-frontier-smoke.mjs --training-benchmark gpu"
  );
  assert.equal(
    packageJson.scripts["training:benchmark:cpu"],
    "node scripts/gpu-frontier-smoke.mjs --training-benchmark cpu"
  );
});

test("CPU worker supports authoritative whole-turn application", async () => {
  const worker = await readFile(path.join(root, "src/cpu-ai-worker.ts"), "utf8");

  assert.match(worker, /type\?: "search" \| "applyTurn"/);
  assert.match(worker, /if \(type === "applyTurn"\)/);
  assert.match(worker, /const applied = cpuApplyTurn\(engine, game, moves \?\? \[\]\)/);
  assert.match(worker, /engine\.chronofish_cpu_apply_turn_json\(ptr, len\)/);
  assert.doesNotMatch(worker, /engine\.chronofish_apply_move/);
  assert.doesNotMatch(worker, /engine\.chronofish_submit_turn\(\)/);
  assert.doesNotMatch(worker, /const snapshot = snapshotJson\(engine\)/);
  assert.match(worker, /function cpuWorkerSearchConfig/);
  assert.match(worker, /engine\.chronofish_cpu_worker_search_config_json\(ptr, len\)/);
});

test("build output declares package version on the main page", async () => {
  const html = await readFile(path.join(root, "dist/index.html"), "utf8");
  const appVersion = await readFile(path.join(root, "dist/app-version.js"), "utf8");

  assert.match(html, new RegExp(`🌐 v${escapeRegExp(packageJson.version)}`));
  assert.equal(appVersion.trim(), `export const APP_VERSION = ${JSON.stringify(packageJson.version)};`);
});

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
