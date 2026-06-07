import type { BoardSquares, GameSnapshot, PieceType } from "./types.js";

export function initialGame(): GameSnapshot {
  const board: BoardSquares = Array.from({ length: 8 }, () => Array(8).fill(null));
  const backRank: PieceType[] = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"];
  for (let x = 0; x < 8; x += 1) {
    const whiteBackRank = backRank[x];
    if (!whiteBackRank) {
      throw new Error(`Missing back-rank piece at file ${x}`);
    }
    board[0]![x] = { color: "white", type: whiteBackRank };
    board[1]![x] = { color: "white", type: "pawn" };
    board[6]![x] = { color: "black", type: "pawn" };
    board[7]![x] = { color: "black", type: whiteBackRank };
  }

  return {
    turn: "white",
    nextTimelineId: 1,
    nextBlackTimelineId: -1,
    checkedRoyals: [],
    timelines: [{
      id: 0,
      row: 0,
      label: "Sacred T0",
      owner: "neutral",
      boards: [{
        time: 0,
        sideToMove: "white",
        castling: 15,
        enPassant: null,
        origin: null,
        board
      }]
    }]
  };
}
