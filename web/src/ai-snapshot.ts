import { GPU_CANDIDATE_STRIDE, GPU_SOURCE_STRIDE, GPU_TARGET_STRIDE, GPU_BOARD_STRIDE, GPU_MUTATION_BOARD_STRIDE, GPU_MUTATION_CHILD_STRIDE, GPU_MUTATION_STATUS_OK, GPU_MUTATION_STATUS_ROYAL_CAPTURE, GPU_MUTATION_STATUS_BRANCH_OK, GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE, GPU_TURN_STATUS_RECORD_STRIDE } from "./ai-layout.js";
import type { BoardSnapshot, BoardSquares, Color, EnPassantTarget, GameSnapshot, Move, MoveOrigin, Piece, PieceType, Position, Timeline, TimelineOwner } from "./types.js";

export interface GpuBoardSnapshot {
  timelineIndex?: number;
  timelineId?: number;
  time: number;
  sideToMove: Color;
  castling: number;
  enPassant: EnPassantTarget | null;
  origin?: MoveOrigin | null;
  board?: BoardSquares;
  squares?: ArrayLike<number>;
  latest?: boolean;
  originKind?: number;
}

export interface GpuTimeline {
  id: number;
  row: number;
  label?: string;
  owner: TimelineOwner;
  boardCount?: number;
  latestTime?: number;
  boards: GpuBoardSnapshot[];
}

export interface GpuSnapshot {
  format?: string;
  turn: Color;
  nextTimelineId?: number;
  nextBlackTimelineId?: number;
  royalCaptureBy?: Color | null;
  timelines: GpuTimeline[];
  boards?: GpuBoardSnapshot[];
}

export interface CandidateMeta extends Position {}

export interface GpuCandidateInputs {
  sourceMeta: CandidateMeta[];
  targetMeta: CandidateMeta[];
  sourceCount: number;
  targetCount: number;
  boardCount: number;
  sources: Int32Array;
  targets: Int32Array;
  boards: Int32Array;
  mutationBoards: Int32Array;
}

interface SnapshotChildOptions {
  move?: Move | null;
  advanceTurn?: boolean;
}

export function readGpuSnapshot(): GpuSnapshot | null {
  return null;
}

export function buildGpuCandidateInputsFromSnapshot(snapshot: GpuSnapshot, color: Color): GpuCandidateInputs {
  return buildGpuCandidateInputs(snapshot, color);
}

export function snapshotWithGpuChildBoards(
  snapshot: GpuSnapshot,
  childBoardRecords: Int32Array,
  mutationStatus: number,
  options: SnapshotChildOptions = {}
): GpuSnapshot {
  const { move = null, advanceTurn = true } = options;
  const royalCapture = mutationStatus === GPU_MUTATION_STATUS_ROYAL_CAPTURE
    || mutationStatus === GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE;
  const historicalBranch = move
    && (mutationStatus === GPU_MUTATION_STATUS_BRANCH_OK || mutationStatus === GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE)
    && !isLatestSnapshotBoard(snapshot, move.to.timelineId, move.to.time);
  const records = [
    childBoardRecords.subarray(0, GPU_MUTATION_BOARD_STRIDE)
  ];
  if (mutationStatus === GPU_MUTATION_STATUS_BRANCH_OK || mutationStatus === GPU_MUTATION_STATUS_BRANCH_ROYAL_CAPTURE) {
    records.push(childBoardRecords.subarray(GPU_MUTATION_BOARD_STRIDE, GPU_MUTATION_CHILD_STRIDE));
  }
  const childByTimeline = new Map<number, GpuBoardSnapshot[]>();
  let historicalBranchChild: GpuBoardSnapshot | null = null;
  for (const record of records) {
    let child = gpuMutationBoardRecordToSnapshot(record);
    let timelineId = child.timelineId;
    if (timelineId === undefined) {
      continue;
    }
    if (historicalBranch && move && !isSourceAdvanceChild(child, move)) {
      historicalBranchChild = child;
      continue;
    }
    if (!childByTimeline.has(timelineId)) {
      childByTimeline.set(timelineId, []);
    }
    childByTimeline.get(timelineId)?.push(child);
  }

  const timelines: GpuTimeline[] = snapshot.timelines.map((timeline) => {
    const children = childByTimeline.get(timeline.id) ?? [];
    if (children.length === 0) {
      return {
        ...timeline,
        boards: timeline.boards.map((board) => ({ ...board }))
      };
    }
    const oldBoards: GpuBoardSnapshot[] = timeline.boards.map((board) => ({ ...board, latest: false }));
    const boards: GpuBoardSnapshot[] = [...oldBoards, ...children.map((child) => ({
      ...child,
      timelineIndex: snapshot.timelines.indexOf(timeline),
      origin: move ? originForGpuChild(child, move) : child.origin ?? null
    }))];
    const latest = latestBoard({ ...timeline, boards });
    return {
      ...timeline,
      boardCount: boards.length,
      latestTime: latest?.time ?? 0,
      boards
    };
  });
  let nextTimelineId = snapshot.nextTimelineId ?? 1;
  let nextBlackTimelineId = snapshot.nextBlackTimelineId ?? -1;
  if (historicalBranchChild && move) {
    const sourceTimeline = snapshot.timelines.find((timeline) => timeline.id === move.from.timelineId);
    const owner = snapshot.turn;
    const newTimelineId = owner === "white" ? nextTimelineId : nextBlackTimelineId;
    if (owner === "white") {
      nextTimelineId += 1;
    } else {
      nextBlackTimelineId -= 1;
    }
    const row = nextBranchRow(timelines, sourceTimeline?.row ?? 0, owner);
    const branchBoard: GpuBoardSnapshot = {
      ...historicalBranchChild,
      timelineId: newTimelineId,
      timelineIndex: timelines.length,
      origin: {
        type: "branch",
        from: { ...move.from },
        to: { ...move.to }
      }
    };
    timelines.push({
      id: newTimelineId,
      row,
      label: `${owner === "white" ? "White" : "Black"} T${newTimelineId}`,
      owner,
      boardCount: 1,
      latestTime: branchBoard.time,
      boards: [branchBoard]
    });
  }
  const boards: GpuBoardSnapshot[] = timelines.flatMap((timeline) => timeline.boards);
  return {
    ...snapshot,
    turn: advanceTurn ? colorFromCode(records[records.length - 1]?.[3] ?? 0) : snapshot.turn,
    royalCaptureBy: royalCapture ? snapshot.turn : snapshot.royalCaptureBy ?? null,
    nextTimelineId,
    nextBlackTimelineId,
    timelines,
    boards
  };
}

export function originForGpuChild(child: GpuBoardSnapshot, move: Move): MoveOrigin {
  const sourceAdvance = isSourceAdvanceChild(child, move);
  return {
    type: sourceAdvance ? "source-advance" : "cross-board",
    from: { ...move.from },
    to: { ...move.to }
  };
}

function isLatestSnapshotBoard(snapshot: GpuSnapshot, timelineId: number, time: number): boolean {
  const timeline = snapshot.timelines.find((candidate) => candidate.id === timelineId);
  const latest = timeline ? latestBoard(timeline) : null;
  return latest?.time === time;
}

function isSourceAdvanceChild(child: GpuBoardSnapshot, move: Move): boolean {
  return child.timelineId === move.from.timelineId && child.time === move.from.time + 1;
}

function nextBranchRow(timelines: GpuTimeline[], sourceRow: number, owner: Color): number {
  const direction = owner === "white" ? 1 : -1;
  let row = sourceRow + direction;
  while (timelines.some((timeline) => timeline.row === row)) {
    row += direction;
  }
  return row;
}

export function gpuMutationBoardRecordToSnapshot(record: Int32Array): GpuBoardSnapshot {
  return {
    timelineIndex: record[0] ?? 0,
    timelineId: record[1] ?? 0,
    time: record[2] ?? 0,
    sideToMove: colorFromCode(record[3] ?? 0),
    castling: record[4] ?? 0,
    enPassant: (record[5] ?? -1) >= 0 ? {
      x: record[5] ?? -1,
      y: record[6] ?? -1,
      capturedX: record[7] ?? -1,
      capturedY: record[8] ?? -1
    } : null,
    latest: true,
    originKind: record[10] ?? 0,
    squares: record.slice(12, 76)
  };
}

export function gpuSnapshotToGame(snapshot: GpuSnapshot): GameSnapshot {
  return {
    turn: snapshot.turn,
    nextTimelineId: snapshot.nextTimelineId ?? 1,
    nextBlackTimelineId: snapshot.nextBlackTimelineId ?? -1,
    royalCaptureBy: snapshot.royalCaptureBy ?? null,
    checkedRoyals: [],
    timelines: snapshot.timelines.map((timeline) => ({
      id: timeline.id,
      row: timeline.row,
      label: timeline.label ?? `T${timeline.id}`,
      owner: timeline.owner,
      boards: timeline.boards
        .map((board) => gpuBoardToGameBoard(board))
        .sort((left, right) => left.time - right.time)
    }))
  };
}

export function gpuBoardToGameBoard(board: GpuBoardSnapshot): BoardSnapshot {
  if (board.board) {
    return {
      time: board.time,
      sideToMove: board.sideToMove,
      castling: board.castling,
      enPassant: board.enPassant,
      origin: board.origin ?? null,
      board: board.board.map((row) => row.map((piece) => piece ? { ...piece } : null))
    };
  }
  return {
    time: board.time,
    sideToMove: board.sideToMove,
    castling: board.castling,
    enPassant: board.enPassant,
    origin: board.origin ?? null,
    board: squaresToGameBoard(board.squares)
  };
}

export function squaresToGameBoard(squares: ArrayLike<number> | undefined): BoardSquares {
  const board: BoardSquares = [];
  for (let y = 0; y < 8; y += 1) {
    const row: Array<Piece | null> = [];
    for (let x = 0; x < 8; x += 1) {
      row.push(pieceFromCode(squares?.[y * 8 + x] ?? 0));
    }
    board.push(row);
  }
  return board;
}

export function pieceFromCode(code: number): Piece | null {
  const type = pieceTypeFromCode(code & 255);
  if (!type) {
    return null;
  }
  return {
    type,
    color: colorFromCode((code >> 8) & 255)
  };
}

export function buildGpuCandidateInputs(game: GpuSnapshot | GameSnapshot, color: Color): GpuCandidateInputs {
  const sourceMeta: CandidateMeta[] = [];
  const targetMeta: CandidateMeta[] = [];
  const sources: number[] = [];
  const targets: number[] = [];
  const boards: number[] = [];
  const mutationBoards: number[] = [];
  const timelines = sortedTimelines(game);

  for (const timeline of timelines) {
    const latest = latestBoard(timeline);
    for (const board of timeline.boards) {
      const squares = squareCodesForBoard(board);
      const isLatest = board.time === latest?.time;
      pushGpuBoardRecord(boards, timeline, {
        time: board.time,
        sideToMove: board.sideToMove,
        castling: board.castling ?? 0,
        squares
      });
      pushGpuMutationBoardRecord(mutationBoards, timeline, {
        time: board.time,
        sideToMove: board.sideToMove,
        castling: board.castling ?? 0,
        enPassant: board.enPassant ?? null,
        latest: isLatest,
        originKind: 0,
        squares
      });
      for (let y = 0; y < 8; y += 1) {
        for (let x = 0; x < 8; x += 1) {
          const code = squares[y * 8 + x] ?? 0;
          targetMeta.push({ timelineId: timeline.id, time: board.time, x, y });
          targets.push(
            code & 255,
            (code >> 8) & 255,
            timeline.id,
            board.time,
            x,
            y,
            timeline.row,
            colorCode(board.sideToMove),
            ownerCode(timeline.owner),
            isLatest ? 1 : 0
          );
        }
      }
      for (let y = 0; y < 8; y += 1) {
        for (let x = 0; x < 8; x += 1) {
          const code = squares[y * 8 + x] ?? 0;
          if ((code & 255) === 0) {
            continue;
          }
          sourceMeta.push({ timelineId: timeline.id, time: board.time, x, y });
          sources.push(
            code & 255,
            (code >> 8) & 255,
            timeline.id,
            board.time,
            x,
            y,
            timeline.row,
            colorCode(board.sideToMove),
            ownerCode(timeline.owner),
            isLatest ? 1 : 0
          );
        }
      }
    }
  }
  return {
    sourceMeta,
    targetMeta,
    sourceCount: sourceMeta.length,
    targetCount: targetMeta.length,
    boardCount: boards.length / GPU_BOARD_STRIDE,
    sources: new Int32Array(sources),
    targets: new Int32Array(targets),
    boards: new Int32Array(boards),
    mutationBoards: new Int32Array(mutationBoards)
  };
}

export function squareCodesForBoard(board: GpuBoardSnapshot | BoardSnapshot): ArrayLike<number> {
  if ("squares" in board && board.squares) {
    return board.squares;
  }
  return (board.board ?? []).flat().map((piece) => piece ? pieceTypeCode(piece.type) | (colorCode(piece.color) << 8) : 0);
}

export function pushGpuBoardRecord(out: number[], timeline: GpuTimeline | Timeline, board: Pick<GpuBoardSnapshot, "time" | "sideToMove" | "castling" | "squares">): void {
  out.push(
    timeline.id,
    timeline.row,
    board.time,
    colorCode(board.sideToMove),
    board.castling ?? 0
  );
  for (let index = 0; index < 64; index += 1) {
    out.push(board.squares?.[index] ?? 0);
  }
}

export function pushGpuMutationBoardRecord(out: number[], timeline: GpuTimeline | Timeline, board: GpuBoardSnapshot): void {
  out.push(
    board.timelineIndex ?? 0,
    timeline.id,
    board.time,
    colorCode(board.sideToMove),
    board.castling ?? 0,
    board.enPassant?.x ?? -1,
    board.enPassant?.y ?? -1,
    board.enPassant?.capturedX ?? -1,
    board.enPassant?.capturedY ?? -1,
    board.latest ? 1 : 0,
    board.originKind ?? 0,
    0
  );
  for (let index = 0; index < 64; index += 1) {
    out.push(board.squares?.[index] ?? 0);
  }
}

export function colorFromCode(code: number): Color {
  return code === 1 ? "black" : "white";
}

export function ownerCode(owner: TimelineOwner): number {
  if (owner === "white") {
    return 1;
  }
  if (owner === "black") {
    return 2;
  }
  return 0;
}

export function moveFromCandidateRecord(records: Int32Array, index: number): Move {
  const offset = index * GPU_CANDIDATE_STRIDE;
  return {
    from: {
      timelineId: records[offset + 11] ?? 0,
      time: records[offset + 12] ?? 0,
      x: records[offset + 13] ?? 0,
      y: records[offset + 14] ?? 0
    },
    to: {
      timelineId: records[offset + 15] ?? 0,
      time: records[offset + 16] ?? 0,
      x: records[offset + 17] ?? 0,
      y: records[offset + 18] ?? 0
    }
  };
}

export function oppositeColor(color: Color): Color {
  return color === "white" ? "black" : "white";
}

export function sortedTimelines(game: GpuSnapshot | GameSnapshot): Array<GpuTimeline | Timeline> {
  return [...game.timelines].sort((left, right) => left.row - right.row || left.id - right.id);
}

export function latestBoard(timeline: GpuTimeline | Timeline): GpuBoardSnapshot | BoardSnapshot | undefined {
  const first = timeline.boards[0];
  return first ? timeline.boards.reduce((latest, board) => board.time > latest.time ? board : latest, first) : undefined;
}

export function presentTimeForSnapshot(snapshot: GpuSnapshot | GameSnapshot): number | null {
  let present: number | null = null;
  for (const timeline of sortedTimelines(snapshot)) {
    const board = latestBoard(timeline);
    if (!board) {
      continue;
    }
    if (present === null || board.time < present) {
      present = board.time;
    }
  }
  return present;
}

export function capitalize(value: string): string {
  return value ? `${value.charAt(0).toUpperCase()}${value.slice(1)}` : "";
}

export function pieceTypeCode(type: PieceType): number {
  const pieceCodes: Record<PieceType, number> = {
    king: 1,
    commonKing: 2,
    queen: 3,
    royalQueen: 4,
    princess: 5,
    rook: 6,
    bishop: 7,
    unicorn: 8,
    dragon: 9,
    knight: 10,
    pawn: 11,
    brawn: 12
  };
  return pieceCodes[type] ?? 0;
}

export function pieceTypeFromCode(code: number): PieceType | null {
  const pieceTypes: Record<number, PieceType> = {
    1: "king",
    2: "commonKing",
    3: "queen",
    4: "royalQueen",
    5: "princess",
    6: "rook",
    7: "bishop",
    8: "unicorn",
    9: "dragon",
    10: "knight",
    11: "pawn",
    12: "brawn"
  };
  return pieceTypes[code] ?? null;
}

export function colorCode(color: Color): number {
  return color === "black" ? 1 : 0;
}
