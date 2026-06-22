import type { BoardSnapshot, Color, GameSnapshot, Piece, PieceType, Timeline } from "./types.js";

export const NEURAL_MAX_BOARDS = 16;
export const NEURAL_BOARD_PLANES = 32;
export const NEURAL_BOARD_SQUARES = 64;
export const NEURAL_INPUT_SIZE = NEURAL_MAX_BOARDS * NEURAL_BOARD_PLANES * NEURAL_BOARD_SQUARES;

interface SelectedBoard {
  category: number;
  negativeTime: number;
  absTimeline: number;
  timelineId: number;
  timelineIndex: number;
  boardIndex: number;
  timeline: Timeline;
  board: BoardSnapshot;
}

export interface EncodedNeuralPosition {
  values: Float32Array;
  boardCount: number;
}

export function encodeNeuralPositionFeatures(game: GameSnapshot, perspective: Color): EncodedNeuralPosition {
  const selected = neuralBoardSelection(game);
  const values = new Float32Array(NEURAL_INPUT_SIZE);
  if (!selected.length) {
    return { values, boardCount: 0 };
  }

  const activeDistance = timelineActiveDistance(game);
  const derivedPresent = game.timelines.reduce((earliest, timeline) => {
    if (!timelineActive(timeline, activeDistance)) {
      return earliest;
    }
    const time = latestBoard(timeline)?.time;
    return Number.isInteger(time) ? Math.min(earliest, time!) : earliest;
  }, Number.MAX_SAFE_INTEGER);
  const present = Number.isInteger(game.presentTime)
    ? game.presentTime!
    : derivedPresent === Number.MAX_SAFE_INTEGER ? 0 : derivedPresent;

  selected.forEach(({ timeline, board }, boardIndex) => {
    const boardBase = boardIndex * NEURAL_BOARD_PLANES * NEURAL_BOARD_SQUARES;
    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        const plane = piecePlane(board.board[y]?.[x]);
        if (plane >= 0) {
          values[boardBase + plane * NEURAL_BOARD_SQUARES + y * 8 + x] = 1;
        }
      }
    }

    const active = timelineActive(timeline, activeDistance);
    const metadata = [
      relativeColorValue(board.sideToMove, perspective),
      active ? 1 : 0,
      latestBoard(timeline)?.time === board.time ? 1 : 0,
      board.time === present ? 1 : 0,
      timeline.owner === "neutral" ? 0 : relativeColorValue(timeline.owner, perspective),
      Math.max(-16, Math.min(16, board.time - present)) / 16,
      1
    ];
    for (let metadataIndex = 0; metadataIndex < metadata.length; metadataIndex += 1) {
      values.fill(
        metadata[metadataIndex]!,
        boardBase + (24 + metadataIndex) * NEURAL_BOARD_SQUARES,
        boardBase + (25 + metadataIndex) * NEURAL_BOARD_SQUARES
      );
    }
  });

  return { values, boardCount: selected.length };
}

export function neuralBoardSelection(game: GameSnapshot): SelectedBoard[] {
  const candidates: SelectedBoard[] = [];
  const activeDistance = timelineActiveDistance(game);
  game.timelines.forEach((timeline, timelineIndex) => {
    const latestTime = latestBoard(timeline)?.time;
    timeline.boards.forEach((board, boardIndex) => {
      const latest = board.time === latestTime;
      const hasRoyal = board.board.some((row) => row.some((piece) => Boolean(piece && isRoyalPiece(piece.type))));
      const hasRecentOrigin = Boolean(board.origin);
      if (!latest && !hasRoyal && !hasRecentOrigin) {
        return;
      }
      const active = timelineActive(timeline, activeDistance);
      candidates.push({
        category: latest && active ? 0 : latest ? 1 : hasRoyal ? 2 : 3,
        negativeTime: -board.time,
        absTimeline: Math.abs(timeline.id),
        timelineId: timeline.id,
        timelineIndex,
        boardIndex,
        timeline,
        board
      });
    });
  });
  candidates.sort((left, right) =>
    left.category - right.category ||
    left.negativeTime - right.negativeTime ||
    left.absTimeline - right.absTimeline ||
    left.timelineId - right.timelineId ||
    left.timelineIndex - right.timelineIndex ||
    left.boardIndex - right.boardIndex
  );
  return candidates.slice(0, NEURAL_MAX_BOARDS);
}

function piecePlane(piece: Piece | null | undefined): number {
  if (!piece) {
    return -1;
  }
  return colorCode(piece.color) * 12 + pieceTypeCode(piece.type);
}

function colorCode(color: Color): number {
  return color === "black" ? 1 : 0;
}

function pieceTypeCode(type: PieceType): number {
  const codes: Record<PieceType, number> = {
    king: 0,
    commonKing: 1,
    queen: 2,
    royalQueen: 3,
    princess: 4,
    rook: 5,
    bishop: 6,
    unicorn: 7,
    dragon: 8,
    knight: 9,
    pawn: 10,
    brawn: 11
  };
  return codes[type];
}

function timelineActiveDistance(game: GameSnapshot): number {
  const ids = game.timelines.map((timeline) => timeline.id);
  const minTimeline = Math.min(...ids, 0);
  const maxTimeline = Math.max(...ids, 0);
  return Math.max(0, Math.min(-minTimeline, maxTimeline)) + 1;
}

function timelineActive(timeline: Timeline, activeDistance: number): boolean {
  if (typeof timeline.active === "boolean") {
    return timeline.active;
  }
  return timeline.owner === "neutral" || Math.abs(timeline.id) <= activeDistance;
}

function relativeColorValue(color: Color, perspective: Color): number {
  return color === perspective ? 1 : -1;
}

function isRoyalPiece(type: PieceType): boolean {
  return type === "king" || type === "royalQueen";
}

function latestBoard(timeline: Timeline): BoardSnapshot | undefined {
  const first = timeline.boards[0];
  return first ? timeline.boards.reduce((latest, board) => board.time > latest.time ? board : latest, first) : undefined;
}
