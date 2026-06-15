import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("GPU move generation permits historical branch mutation", async () => {
  const shader = await readFile(path.join(root, "src/shaders/mutate.wgsl"), "utf8");

  assert.doesNotMatch(shader, /STATUS_UNSUPPORTED_HISTORICAL_BRANCH/);
  assert.doesNotMatch(shader, /!same_board\s*&&\s*!target_latest/);
  assert.match(shader, /statuses\[index\]\s*=\s*select\(STATUS_BRANCH_OK,\s*STATUS_BRANCH_ROYAL_CAPTURE/);
});

test("legacy GPU move generation carries en-passant board metadata", async () => {
  const layout = await readFile(path.join(root, "src/ai-layout.ts"), "utf8");
  const snapshot = await readFile(path.join(root, "src/ai-snapshot.ts"), "utf8");
  const shader = await readFile(path.join(root, "src/shaders/movegen.wgsl"), "utf8");

  assert.match(layout, /GPU_BOARD_STRIDE = 73/);
  assert.match(snapshot, /board\.enPassant\?\.x \?\? -1/);
  assert.match(shader, /const BOARD_EP: u32 = 5u/);
  assert.match(shader, /const BOARD_SQUARE_OFFSET: u32 = 9u/);
  assert.match(shader, /source_ep_x = boards\[source_board_base \+ BOARD_EP\]/);
  assert.match(shader, /ep_x == to_x && ep_y == to_y/);
});

test("full GPU mode uses the resident frontier for parallel timelines", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");
  const caller = worker.slice(0, worker.indexOf("async function tryFullGpuSearch"));

  assert.doesNotMatch(caller, /throw new Error\("Full GPU search currently requires one pending present board\."\)/);
  assert.match(caller, /gpuMode === "full"/);
  assert.match(caller, /tryGpuResidentFrontierSearch/);
  assert.match(worker, /validatedFrontierChoices/);
  assert.match(worker, /validateFirstFrontierTurn/);
  assert.match(caller, /falling back to hybrid GPU search/);
});

test("GPU search returns complete turn plans for post-match review", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /const principalVariation = \[moves\]/);
  assert.match(worker, /completedGpuReplyTurn\(device, current/);
  assert.match(worker, /principalVariation\.push\(reply\)/);
  assert.match(worker, /depth:\s*1,[\s\S]*gpuSearch:\s*"projected-reply"/);
  assert.match(worker, /principalVariation:\s*Move\[\]\[\]/);
  assert.match(worker, /let best = -2147483647/);
});

test("GPU turn completion only spends moves on pending present boards", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /moveStartsOnPendingBoard\(entry\.move, pendingBoards\)/);
  assert.match(worker, /findCompleteGpuTurn\(device, snapshot, rootTurn/);
  assert.match(worker, /`\$\{result\.gpuSearch \?\? "gpu"\}-turn-fallback`/);
  assert.match(worker, /choices: \[\]/);
  assert.match(worker, /return Boolean\(result\?\.status === "ok" && result\.moves\?\.length\)/);
  assert.doesNotMatch(worker, /\|\| result\?\.status === "incompleteTurn"/);
});

test("GPU reply sentinel is never used as an evaluation", async () => {
  const worker = await readFile(path.join(root, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /if \(reply\.move\) \{\s*score -= reply\.score/);
  assert.match(worker, /return bestMove \? \{ score: best, move: bestMove \} : \{ score: 0 \}/);
});
