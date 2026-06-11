export type Color = "white" | "black";
export type TimelineOwner = Color | "neutral";

export type PieceType =
  | "king"
  | "commonKing"
  | "queen"
  | "royalQueen"
  | "princess"
  | "rook"
  | "bishop"
  | "unicorn"
  | "dragon"
  | "knight"
  | "pawn"
  | "brawn";

export interface Piece {
  color: Color;
  type: PieceType;
}

export type BoardSquares = Array<Array<Piece | null>>;

export interface EnPassantTarget {
  x: number;
  y: number;
  capturedX: number;
  capturedY: number;
}

export interface MoveOrigin {
  type: string;
  from?: Position;
  to?: Position;
}

export interface BoardSnapshot {
  time: number;
  sideToMove: Color;
  castling: number;
  enPassant: EnPassantTarget | null;
  origin: MoveOrigin | null;
  board: BoardSquares;
}

export interface Timeline {
  id: number;
  row: number;
  label: string;
  owner: TimelineOwner;
  active?: boolean;
  boards: BoardSnapshot[];
}

export interface GameSnapshot {
  turn: Color;
  presentTime?: number;
  nextTimelineId: number;
  nextBlackTimelineId: number;
  checkedRoyals: Position[];
  royalCaptureBy?: Color | null;
  result?: GameResult | null;
  timelines: Timeline[];
}

export interface GameResult {
  terminal: true;
  outcome: "win" | "draw";
  winner: Color | null;
  reason: "royal-capture" | "threefold-repetition" | "stalemate";
}

export interface Position {
  timelineId: number;
  time: number;
  x: number;
  y: number;
}

export interface Move {
  from: Position;
  to: Position;
}

export interface WasmString {
  ptr: number;
  len: number;
}

export interface ChronofishEngine {
  memory: WebAssembly.Memory;
  chronofish_output_len(): number;
  chronofish_alloc(length: number): number;
  chronofish_dealloc(ptr: number, length: number): void;
  chronofish_version(): number;
  chronofish_reset(): void;
  chronofish_snapshot_json(): number;
  chronofish_staged_turn_notation(): number;
  chronofish_evaluation_json(): number;
  chronofish_last_message(): number;
  chronofish_load_snapshot_json(ptr: number, length: number): number;
  chronofish_load_ai_parameters_json(ptr: number, length: number): number;
  chronofish_apply_move(
    fromTimelineId: number,
    fromTime: number,
    fromX: number,
    fromY: number,
    toTimelineId: number,
    toTime: number,
    toX: number,
    toY: number
  ): number;
  chronofish_legal_targets_json(timelineId: number, time: number, x: number, y: number): number;
  chronofish_legal_selection_json(timelineId: number, time: number, x: number, y: number): number;
  chronofish_submit_turn(): number;
  chronofish_ai_turn_json(maxDepth: number, maxNodes: number): number;
  chronofish_ai_turn_timed_json(maxDepth: number, maxNodes: number, millis: number): number;
}
