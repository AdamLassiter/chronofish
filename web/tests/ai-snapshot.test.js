import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import * as esbuild from "esbuild";

const root = path.resolve(import.meta.dirname, "..");
const modules = await buildTestModules();
const { GPU_MUTATION_BOARD_STRIDE, GPU_MUTATION_CHILD_STRIDE, GPU_MUTATION_STATUS_BRANCH_OK } = await import(modules.aiLayout);
const { buildGpuCandidateInputs, colorCode, parseGpuSnapshotBytes, snapshotWithGpuChildBoards } = await import(modules.aiSnapshot);

test("initial position encodes GPU move candidates", () => {
  const inputs = buildGpuCandidateInputs(initialGame(), "white");

  assert.equal(inputs.sourceCount, 32);
  assert.equal(inputs.targetCount, 64);
  assert.equal(inputs.boardCount, 1);
  assert.equal(inputs.sources.length, inputs.sourceCount * 10);
  assert.equal(inputs.targets.length, inputs.targetCount * 10);
});

test("GPU snapshot encoding normalizes numeric and cased colors", () => {
  const game = initialGame();
  game.turn = "WHITE";
  game.timelines[0].boards[0].sideToMove = 0;
  game.timelines[0].boards[0].board[0][0] = { color: "BLACK", type: "rook" };

  const inputs = buildGpuCandidateInputs(game, "white");

  assert.equal(colorCode("WHITE"), 0);
  assert.equal(colorCode("BLACK"), 1);
  assert.equal(colorCode(0), 0);
  assert.equal(colorCode(1), 1);
  assert.equal(inputs.targets[7], 0);
  assert.equal(inputs.sources[1], 1);
});

test("engine GPU snapshot bytes parse into GPU candidate inputs", () => {
  const snapshot = parseGpuSnapshotBytes(initialEngineGpuSnapshotBytes());
  const inputs = buildGpuCandidateInputs(snapshot, "white");

  assert.equal(snapshot.format, "engine-gpu-snapshot-v1");
  assert.equal(snapshot.turn, "white");
  assert.equal(snapshot.timelines.length, 1);
  assert.equal(snapshot.timelines[0].id, 0);
  assert.equal(snapshot.timelines[0].boards.length, 1);
  assert.equal(snapshot.timelines[0].boards[0].latest, true);
  assert.equal(inputs.sourceCount, 32);
  assert.equal(inputs.targetCount, 64);
  assert.equal(inputs.boardCount, 1);
});

test("historical GPU branch creates a new owned timeline", () => {
  const snapshot = {
    turn: "white",
    nextTimelineId: 1,
    nextBlackTimelineId: -1,
    timelines: [{
      id: 0,
      row: 0,
      owner: "neutral",
      boards: [
        emptyGpuBoard({ timelineId: 0, time: 2, sideToMove: "white", latest: false }),
        emptyGpuBoard({ timelineId: 0, time: 3, sideToMove: "white", latest: true })
      ]
    }]
  };
  const move = {
    from: { timelineId: 0, time: 3, x: 3, y: 7 },
    to: { timelineId: 0, time: 2, x: 3, y: 5 }
  };
  const records = new Int32Array(GPU_MUTATION_CHILD_STRIDE);
  writeMutationBoard(records, 0, { timelineId: 0, time: 4, sideToMove: 1 });
  writeMutationBoard(records, GPU_MUTATION_BOARD_STRIDE, { timelineId: 0, time: 3, sideToMove: 1 });

  const next = snapshotWithGpuChildBoards(snapshot, records, GPU_MUTATION_STATUS_BRANCH_OK, {
    move,
    advanceTurn: true
  });

  assert.equal(next.timelines.length, 2);
  assert.equal(next.nextTimelineId, 2);
  assert.equal(next.timelines[0].boards.at(-1).time, 4);
  assert.equal(next.timelines[1].id, 1);
  assert.equal(next.timelines[1].row, 1);
  assert.equal(next.timelines[1].owner, "white");
  assert.equal(next.timelines[1].boards[0].time, 3);
  assert.equal(next.timelines[1].boards[0].origin.type, "branch");
});

function emptyGpuBoard({ timelineId, time, sideToMove, latest }) {
  return {
    timelineId,
    time,
    sideToMove,
    castling: 0,
    enPassant: null,
    latest,
    squares: new Int32Array(64)
  };
}

function initialGame() {
  const board = Array.from({ length: 8 }, () => Array(8).fill(null));
  const backRank = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"];
  for (let x = 0; x < 8; x += 1) {
    board[0][x] = { color: "white", type: backRank[x] };
    board[1][x] = { color: "white", type: "pawn" };
    board[6][x] = { color: "black", type: "pawn" };
    board[7][x] = { color: "black", type: backRank[x] };
  }
  return {
    turn: "white",
    nextTimelineId: 1,
    nextBlackTimelineId: -1,
    timelines: [{
      id: 0,
      row: 0,
      owner: "neutral",
      boards: [{ time: 0, sideToMove: "white", castling: 15, enPassant: null, board }]
    }]
  };
}

function writeMutationBoard(records, offset, { timelineId, time, sideToMove }) {
  records[offset + 1] = timelineId;
  records[offset + 2] = time;
  records[offset + 3] = sideToMove;
  records[offset + 5] = -1;
  records[offset + 6] = -1;
  records[offset + 7] = -1;
  records[offset + 8] = -1;
  records[offset + 9] = 1;
}

function initialEngineGpuSnapshotBytes() {
  const words = [];
  const push = (...values) => words.push(...values);
  push(
    0x4346_4750,
    1,
    0,
    1,
    1,
    1,
    -1,
    -1,
    0,
    8,
    12,
    64,
    128,
    64,
    1,
    0
  );
  push(0, 0, 0, 0, 1, 1, 0, 0);
  push(0, 0, 0, 0, 15, -1, -1, -1, -1, 1, 0, 0);
  const backRank = [6, 10, 7, 3, 1, 7, 10, 6];
  for (let index = 0; index < 64; index += 1) {
    const y = Math.floor(index / 8);
    const x = index % 8;
    if (y === 0) {
      words.push(backRank[x]);
    } else if (y === 1) {
      words.push(11);
    } else if (y === 6) {
      words.push(11 | (1 << 8));
    } else if (y === 7) {
      words.push(backRank[x] | (1 << 8));
    } else {
      words.push(0);
    }
  }
  const bytes = new Uint8Array(words.length * Int32Array.BYTES_PER_ELEMENT);
  new Int32Array(bytes.buffer).set(words);
  return bytes;
}

async function buildTestModules() {
  const outdir = await mkdtemp(path.join(os.tmpdir(), "chronofish-web-test-"));
  await esbuild.build({
    entryPoints: [
      path.join(root, "src/ai-layout.ts"),
      path.join(root, "src/ai-snapshot.ts")
    ],
    outdir,
    bundle: false,
    format: "esm",
    platform: "node",
    target: "es2022",
    logLevel: "silent"
  });
  return {
    aiLayout: pathToFileURL(path.join(outdir, "ai-layout.js")).href,
    aiSnapshot: pathToFileURL(path.join(outdir, "ai-snapshot.js")).href
  };
}
