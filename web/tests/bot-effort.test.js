import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const webRoot = path.resolve(import.meta.dirname, "..");
const repoRoot = path.resolve(webRoot, "..");

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
  }
});

test("frontend loads GPU effort separately from CPU effort", async () => {
  const main = await readFile(path.join(webRoot, "src/main.ts"), "utf8");

  assert.match(main, /fetch\("\/ai\/gpu-effort\.json"\)/);
  assert.match(main, /gpuEffortConfigs = await gpuEffortResponse\.json/);
  assert.match(main, /bot-gpu-custom/);
});

test("bot timeout preserves minimum depth and a completed legal result", async () => {
  const controller = await readFile(path.join(webRoot, "src/bot-controller.ts"), "utf8");

  assert.match(controller, /pending\.currentDepth <= pending\.minDepth/);
  assert.match(controller, /if \(bestResult && \(bestResult\.depth \?\? 0\) >= pending\.minDepth\)/);
  assert.match(controller, /is completing depth/);
  assert.match(controller, /selectDeepestStoredResult\(pending\)/);
});
