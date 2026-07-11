import { engineGpuSearchColorCode } from "./engine-gpu-search.js";
import type { BoardSquares, ChronofishEngine, Color, EnPassantTarget, MoveOrigin, Position, TimelineOwner } from "./types.js";

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

export function colorCode(color: Color | string | number | null | undefined, engine?: ChronofishEngine): number {
  if (typeof color === "number") {
    return color === 1 ? 1 : 0;
  }
  if (typeof color === "string") {
    if (engine) {
      return engineGpuSearchColorCode(engine, color);
    }
    const normalized = color.toLowerCase();
    if (normalized === "black") {
      return 1;
    }
    if (normalized === "white") {
      return 0;
    }
  }
  throw new Error(`Unsupported color value: ${String(color)}`);
}
