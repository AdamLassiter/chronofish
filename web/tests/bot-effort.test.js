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

  assert.match(controller, /const targetDepth = evenSearchDepthAtMost\(effort\.depth \?\? DEFAULT_MIN_BOT_SEARCH_DEPTH\)/);
  assert.match(controller, /evenSearchDepthAtLeast\(effort\.minDepth \?\? DEFAULT_MIN_BOT_SEARCH_DEPTH\)/);
  assert.match(controller, /const nextDepth = nextBotSearchDepth\(pending\.currentDepth\)/);
  assert.match(controller, /return currentDepth <= 0 \? 2 : currentDepth \+ 2/);
  assert.match(controller, /minDepth: Math\.min\(nextDepth, pending\.minDepth\)/);
  assert.match(controller, /pending\.currentDepth <= pending\.minDepth && pending\.depthReceived < pending\.depthExpected/);
  assert.match(controller, /function completedEvenSearchDepth/);
  assert.match(controller, /pending\.currentDepth <= pending\.minDepth/);
  assert.match(controller, /completedDepth >= pending\.minDepth/);
  assert.match(controller, /if \(bestResult && \(bestResult\.depth \?\? 0\) >= pending\.minDepth\)/);
  assert.match(controller, /completedDepth === requestedDepth && completedDepth >= 2 && completedDepth % 2 === 0/);
  assert.match(controller, /pending\.incompleteDepthAttempt = true/);
  assert.match(controller, /pending\.incompleteDepthAttempt && pending\.currentDepth >= pending\.minDepth/);
  assert.match(controller, /is completing depth/);
  assert.match(controller, /selectDeepestStoredResult\(pending\)/);
  assert.match(controller, /startMinimumDepthCpuFallback\(pending\)/);
  assert.match(controller, /is completing minimum depth/);
  assert.match(controller, /minDepth: pending\.minDepth/);
  assert.match(controller, /pending\.bestByDepth\.set\(entry\.depth, depthBest\)/);
  assert.match(controller, /\(bestResult\?\.depth \?\? 0\) >= pending\.targetDepth/);
});

test("bot countdown switches to overtime after the nominal deadline", async () => {
  const controller = await readFile(path.join(webRoot, "src/bot-controller.ts"), "utf8");

  assert.match(controller, /function formatBotCountdown\(deadlineAt: number, now = Date\.now\(\)\)/);
  assert.match(controller, /return `\$\{formatBotTimeLimit\(deltaMs\)\} left`/);
  assert.match(controller, /return `\$\{formatBotTimeLimit\(-deltaMs\)\} overtime`/);
  assert.match(controller, /formatBotCountdown\(pending\.deadlineAt\)/);
  assert.doesNotMatch(controller, /Math\.max\(0, pending\.deadlineAt - Date\.now\(\)\)/);
});

test("bot search result ranking prefers deeper completed searches before score", async () => {
  const controller = await readFile(path.join(webRoot, "src/bot-controller.ts"), "utf8");

  assert.match(controller, /function compareAiResultPreference/);
  assert.match(controller, /function compareBotChoicePreference/);
  assert.match(controller, /const depth = \(right\.depth \?\? 0\) - \(left\.depth \?\? 0\)/);
  assert.match(controller, /const depth = botChoiceDepth\(right\) - botChoiceDepth\(left\)/);
  assert.match(controller, /sort\(compareAiResultPreference\)/);
  assert.match(controller, /compareBotChoicePreference\(next, current\) < 0/);
  assert.match(controller, /sort\(compareBotChoicePreference\)/);
});

test("GPU worker honors minimum depth before applying its internal deadline", async () => {
  const worker = await readFile(path.join(webRoot, "src/ai-worker.ts"), "utf8");

  assert.match(worker, /minDepth\?: number/);
  assert.match(worker, /const requestedDepth = Math\.max\(1, depth \?\? 1\)/);
  assert.match(worker, /const minimumDepth = Math\.min\(requestedDepth, Math\.max\(1, Math\.floor\(minDepth \?\? 1\)\)\)/);
  assert.match(worker, /gpuDeadlineAt = minimumDepth >= requestedDepth/);
  assert.match(worker, /Number\.POSITIVE_INFINITY/);
  assert.match(worker, /depth: requestedDepth/);
});

test("bot move choice logging includes full principal variation plans", async () => {
  const controller = await readFile(path.join(webRoot, "src/bot-controller.ts"), "utf8");
  const worker = await readFile(path.join(webRoot, "src/ai-worker.ts"), "utf8");

  assert.match(controller, /plan: formatBotPlan\(choice\.principalVariation \?\? \[choice\.moves\], pending\.game\)/);
  assert.match(controller, /function formatBotPlan/);
  assert.match(controller, /principalVariation: normalizePrincipalVariation\(choice\.principalVariation \?\? result\.principalVariation, moves\)/);
  assert.match(controller, /principalVariation: normalizePrincipalVariation\(choice\.principalVariation, choice\.moves\)/);
  assert.match(worker, /principalVariation\?: Move\[\]\[\] \| undefined/);
  assert.match(worker, /principalVariation,/);
  assert.match(worker, /principalVariation: candidate\.principalVariation/);
});

test("post-match bot review opens plans from matching side-to-move boards", async () => {
  const main = await readFile(path.join(webRoot, "src/main.ts"), "utf8");

  assert.match(main, /function botReviewDecisionForBoard\(position: Position, snapshot: GameSnapshot\): BotDecisionRecord \| null/);
  assert.match(main, /decision\.botColor !== clickedBoard\.sideToMove/);
  assert.match(main, /boardSnapshotKey\(decisionBoard\) === boardSnapshotKey\(clickedBoard\)/);
  assert.match(main, /const reviewSnapshot = botReviewProjection\?\.finalGame \?\? game/);
  assert.match(main, /const baseSnapshot = cloneGame\(decision\.game\)/);
  assert.match(main, /buildBotReviewPlan\(decision, baseSnapshot\)/);
  assert.doesNotMatch(main, /selectedTurnClicked/);
  assert.doesNotMatch(main, /applyBotReviewTurn/);
});
