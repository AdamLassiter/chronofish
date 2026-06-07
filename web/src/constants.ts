import type { Color, PieceType } from "./types.js";

export const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];

// Unicode glyphs keep rendering lightweight. CSS adds shadows/glow so pieces
// remain legible against both square colors.
export const PIECES: Record<Color, Record<PieceType, string>> = {
  white: {
    king: "♔",
    commonKing: "♔",
    queen: "♕",
    royalQueen: "♕",
    princess: "♖",
    rook: "♖",
    bishop: "♗",
    unicorn: "♘",
    dragon: "♗",
    knight: "♘",
    pawn: "♙",
    brawn: "♙"
  },
  black: {
    king: "♚",
    commonKing: "♚",
    queen: "♛",
    royalQueen: "♛",
    princess: "♜",
    rook: "♜",
    bishop: "♝",
    unicorn: "♞",
    dragon: "♝",
    knight: "♞",
    pawn: "♟",
    brawn: "♟"
  }
};
