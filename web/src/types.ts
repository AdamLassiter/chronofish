/** Shapes of the server API and of client-side view data. */

export type Side = "white" | "black";

export type Ruleset = "standard" | "multiverse" | "multiverse-variant";

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

export interface User {
  id: number;
  username: string;
  displayName: string;
  isBot: boolean;
  rating: number;
  gamesPlayed: number;
  supportedRulesets?: Ruleset[];
  referenceRules?: boolean;
}

export interface Position {
  timelineId: number;
  time: number;
  x: number;
  y: number;
}

export interface Movement {
  from: Position;
  to: Position;
  promotion?: PieceType;
}

export type PlayerAction =
  { type: "move"; movement: Movement } | { type: "submitTurn" };

export interface ConcedeAction {
  type: "concede";
}

export interface UndoFailedAiAction {
  type: "undoFailedAiAction";
}

export interface UndoStagedMoveAction {
  type: "undoStagedMove";
}

export interface AiDecision {
  action: PlayerAction;
  principalVariation: PlayerAction[];
  rootValue: number | null;
}

export interface Piece {
  color: Side;
  type: PieceType;
}

export interface CastlingRights {
  whiteKingside: boolean;
  whiteQueenside: boolean;
  blackKingside: boolean;
  blackQueenside: boolean;
}

export interface EnPassant {
  x: number;
  y: number;
  capturedX: number;
  capturedY: number;
}

export type BoardOrigin =
  | { type: "none" }
  | {
      type: "move";
      from: Position;
      to: Position;
      move_type: string;
    };

export interface BoardSnapshot {
  time: number;
  sideToMove: Side;
  board: Array<Array<Piece | null>>;
  castling: CastlingRights;
  enPassant: EnPassant | null;
  origin: BoardOrigin;
  halfmoveClock: number;
}

export interface Timeline {
  id: number;
  row: number;
  label: string;
  owner: "neutral" | Side;
  accepted: boolean;
  boards: BoardSnapshot[];
}

export type OutcomeReason =
  | "checkmate"
  | "stalemate"
  | "royal-capture"
  | "threefold-repetition"
  | "fifty-move-rule"
  | "insufficient-material"
  | "no-legal-turn"
  | "concession";

export interface GameView {
  ruleset: Ruleset;
  turn: Side;
  outcome: { winner: Side | null; reason: OutcomeReason } | null;
  message: string;
  timelines: Timeline[];
  legalActions: PlayerAction[];
  canSubmit: boolean;
  hasStagedMoves: boolean;
  presentTime: number | null;
  submittedPresentTime: number | null;
  activeTimelines: number[];
  checkedRoyals: Position[];
}

export interface GameSummary {
  id: number;
  ruleset: Ruleset;
  status: "active" | "finished";
  white: User;
  black: User;
  winnerId: number | null;
  outcomeReason: string | null;
  version: number;
  createdAt: number;
  updatedAt: number;
  finishedAt: number | null;
}

export interface MoveRecord {
  sequence: number;
  actor: User;
  action:
    | PlayerAction
    | ConcedeAction
    | UndoFailedAiAction
    | UndoStagedMoveAction
    | AiDecision;
  createdAt: number;
}

export interface HistoryTurn {
  actor: User;
  entries: MoveRecord[];
}

export interface GameDetail {
  summary: GameSummary;
  game: GameView;
  viewerSide: Side | null;
  history: MoveRecord[];
}

export interface Evaluation {
  sequence: number;
  /** White's advantage on [-1, 1]. */
  whiteScore: number;
  centipawns: number | null;
  winner: Side | null;
  finished: boolean;
  engine: string;
}

export interface BugReportReceipt {
  id: number;
  status: "unresolved" | "closed";
}

export interface Session {
  token: string;
  user: User;
}

export interface AppState {
  token: string | null;
  user: User | null;
  users: User[];
  current: GameDetail | null;
  selected: Position | null;
  poll: ReturnType<typeof setInterval> | null;
}

export interface ApiOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
}

export type BoardStatus =
  "Optional" | "Required" | "Inactive" | "Future" | "Latest" | "Past";

export type SquareHighlight =
  "arrived" | "move-source" | "move-target" | "check";

export type HighlightMap = Map<string, Set<SquareHighlight>>;

export interface Point {
  x: number;
  y: number;
}

export interface MoveArrow {
  from: Position;
  to: Position;
}
