import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("GPU move generation permits historical branch mutation", async () => {
  const shader = await readFile(path.join(root, "src/ai-shaders.ts"), "utf8");

  assert.doesNotMatch(shader, /STATUS_UNSUPPORTED_HISTORICAL_BRANCH/);
  assert.doesNotMatch(shader, /!same_board\s*&&\s*!target_latest/);
  assert.match(shader, /statuses\[index\]\s*=\s*select\(STATUS_BRANCH_OK,\s*STATUS_BRANCH_ROYAL_CAPTURE/);
});

test("full GPU mode falls back when parallel timelines need multiple replies", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const caller = worker.slice(0, worker.indexOf("async function tryFullGpuSearch"));

  assert.doesNotMatch(caller, /throw new Error\("Full GPU search currently requires one pending present board\."\)/);
  assert.match(caller, /gpuMode === "full" && turnStatus\.pendingPresentBoardCount === 1/);
  assert.match(caller, /falling back to hybrid GPU search/);
  assert.match(caller, /pendingPresentBoardsForSnapshot\(current,\s*rootTurn\)/);
});
