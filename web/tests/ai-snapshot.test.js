import assert from "node:assert/strict";
import test from "node:test";
import { initialGame } from "../dist/initial-game.js";
import { GPU_MUTATION_BOARD_STRIDE, GPU_MUTATION_CHILD_STRIDE, GPU_MUTATION_STATUS_BRANCH_OK } from "../dist/ai-layout.js";
import { buildGpuCandidateInputs, snapshotWithGpuChildBoards } from "../dist/ai-snapshot.js";

test("initial position encodes GPU move candidates", () => {
  const inputs = buildGpuCandidateInputs(initialGame(), "white");

  assert.equal(inputs.sourceCount, 32);
  assert.equal(inputs.targetCount, 64);
  assert.equal(inputs.boardCount, 1);
  assert.equal(inputs.sources.length, inputs.sourceCount * 10);
  assert.equal(inputs.targets.length, inputs.targetCount * 10);
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
