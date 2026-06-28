import type { BoardSnapshot, Color, GameSnapshot, Piece, PieceType, Timeline } from "./types.js";

export const NEURAL_MAX_BOARDS = 16;
export const NEURAL_BOARD_PLANES = 36;
export const NEURAL_BOARD_SQUARES = 64;
export const NEURAL_INPUT_SIZE = NEURAL_MAX_BOARDS * NEURAL_BOARD_PLANES * NEURAL_BOARD_SQUARES;

export type NeuralBoardGraphEdgeKind =
  | "same-timeline"
  | "branch-origin"
  | "causal-origin"
  | "present-frontier";

export interface SelectedBoard {
  category: number;
  negativeTime: number;
  presentDistance: number;
  activeRank: number;
  ownerRank: number;
  royalRank: number;
  structuralHash: number;
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
  graph: NeuralBoardGraph;
}

export interface NeuralBoardGraphNode {
  index: number;
  timelineId: number;
  timelineRow: number;
  relativeTime: number;
  presentOffset: number;
  active: boolean;
  latest: boolean;
  sideToMove: Color;
  owner: Timeline["owner"];
  containsRoyal: boolean;
  royalInCheck: boolean;
}

export interface NeuralBoardGraphEdge {
  from: number;
  to: number;
  kind: NeuralBoardGraphEdgeKind;
  deltaTime: number;
  deltaTimeline: number;
}

export interface NeuralBoardGraph {
  nodes: NeuralBoardGraphNode[];
  edges: NeuralBoardGraphEdge[];
}

export function encodeNeuralPositionFeatures(game: GameSnapshot, perspective: Color): EncodedNeuralPosition {
  const selected = neuralBoardSelection(game);
  const values = new Float32Array(NEURAL_INPUT_SIZE);
  if (!selected.length) {
    return { values, boardCount: 0, graph: { nodes: [], edges: [] } };
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
  const graph = buildNeuralBoardGraph(game, selected, present, activeDistance);
  const graphSummaries = graph.nodes.map((_, nodeIndex) => graphNodeSummary(graph, nodeIndex));

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
    const graphMetadata = graphSummaries[boardIndex] ?? [0, 0, 0, 0, 0];
    for (let metadataIndex = 0; metadataIndex < graphMetadata.length; metadataIndex += 1) {
      values.fill(
        graphMetadata[metadataIndex]!,
        boardBase + (31 + metadataIndex) * NEURAL_BOARD_SQUARES,
        boardBase + (32 + metadataIndex) * NEURAL_BOARD_SQUARES
      );
    }
  });

  return { values, boardCount: selected.length, graph };
}

export function buildNeuralBoardGraph(
  game: GameSnapshot,
  selected = neuralBoardSelection(game),
  present = presentTime(game, timelineActiveDistance(game)),
  activeDistance = timelineActiveDistance(game)
): NeuralBoardGraph {
  const indexByBoard = new Map<string, number>();
  const nodes = selected.map(({ timeline, board }, index): NeuralBoardGraphNode => {
    indexByBoard.set(boardKey(timeline.id, board.time), index);
    const latest = latestBoard(timeline)?.time === board.time;
    return {
      index,
      timelineId: timeline.id,
      timelineRow: timeline.row,
      relativeTime: board.time - present,
      presentOffset: board.time - present,
      active: timelineActive(timeline, activeDistance),
      latest,
      sideToMove: board.sideToMove,
      owner: timeline.owner,
      containsRoyal: boardContainsRoyal(board),
      royalInCheck: game.checkedRoyals.some((position) => position.timelineId === timeline.id && position.time === board.time)
    };
  });
  const edges: NeuralBoardGraphEdge[] = [];
  const addEdge = (from: number | undefined, to: number | undefined, kind: NeuralBoardGraphEdgeKind) => {
    if (from === undefined || to === undefined || from === to) {
      return;
    }
    if (edges.some((edge) => edge.from === from && edge.to === to && edge.kind === kind)) {
      return;
    }
    const left = nodes[from];
    const right = nodes[to];
    if (!left || !right) {
      return;
    }
    edges.push({
      from,
      to,
      kind,
      deltaTime: right.relativeTime - left.relativeTime,
      deltaTimeline: right.timelineRow - left.timelineRow
    });
  };

  const byTimeline = new Map<number, SelectedBoard[]>();
  for (const board of selected) {
    const group = byTimeline.get(board.timelineId) ?? [];
    group.push(board);
    byTimeline.set(board.timelineId, group);
  }
  for (const boards of byTimeline.values()) {
    boards.sort((left, right) => left.board.time - right.board.time);
    for (let index = 1; index < boards.length; index += 1) {
      const previous = boards[index - 1]!;
      const next = boards[index]!;
      addEdge(indexByBoard.get(boardKey(previous.timelineId, previous.board.time)), indexByBoard.get(boardKey(next.timelineId, next.board.time)), "same-timeline");
    }
  }

  for (const { timeline, board } of selected) {
    const target = indexByBoard.get(boardKey(timeline.id, board.time));
    if (board.time === present) {
      for (const node of nodes) {
        if (node.latest && node.active) {
          addEdge(node.index, target, "present-frontier");
        }
      }
    }
    if (!board.origin?.from || !board.origin.to) {
      continue;
    }
    const source = indexByBoard.get(boardKey(board.origin.from.timelineId, board.origin.from.time));
    const destination = indexByBoard.get(boardKey(board.origin.to.timelineId, board.origin.to.time));
    addEdge(source, target, board.origin.type === "branch" ? "branch-origin" : "causal-origin");
    addEdge(destination, target, "causal-origin");
  }

  return { nodes, edges };
}

function graphNodeSummary(graph: NeuralBoardGraph, nodeIndex: number): number[] {
  const incident = graph.edges.filter((edge) => edge.from === nodeIndex || edge.to === nodeIndex);
  const count = (kind: NeuralBoardGraphEdgeKind) => incident.filter((edge) => edge.kind === kind).length;
  return [
    Math.min(1, count("same-timeline") / 2),
    Math.min(1, count("branch-origin")),
    Math.min(1, count("causal-origin") / 2),
    Math.min(1, count("present-frontier") / Math.max(1, graph.nodes.length - 1)),
    graph.nodes[nodeIndex]?.royalInCheck ? 1 : graph.nodes[nodeIndex]?.containsRoyal ? 0.5 : 0
  ];
}

function boardKey(timelineId: number, time: number): string {
  return `${timelineId}:${time}`;
}

export function neuralBoardSelection(game: GameSnapshot): SelectedBoard[] {
  const candidates: SelectedBoard[] = [];
  const activeDistance = timelineActiveDistance(game);
  const present = presentTime(game, activeDistance);
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
        presentDistance: Math.abs(board.time - present),
        activeRank: active ? 0 : 1,
        ownerRank: ownerRank(timeline.owner),
        royalRank: hasRoyal ? 0 : 1,
        structuralHash: boardStructuralHash(board),
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
    left.presentDistance - right.presentDistance ||
    left.negativeTime - right.negativeTime ||
    left.activeRank - right.activeRank ||
    left.ownerRank - right.ownerRank ||
    left.royalRank - right.royalRank ||
    left.structuralHash - right.structuralHash ||
    left.absTimeline - right.absTimeline ||
    left.timelineId - right.timelineId ||
    left.timelineIndex - right.timelineIndex ||
    left.boardIndex - right.boardIndex
  );
  return candidates.slice(0, NEURAL_MAX_BOARDS);
}

function presentTime(game: GameSnapshot, activeDistance: number): number {
  const derivedPresent = game.timelines.reduce((earliest, timeline) => {
    if (!timelineActive(timeline, activeDistance)) {
      return earliest;
    }
    const time = latestBoard(timeline)?.time;
    return Number.isInteger(time) ? Math.min(earliest, time!) : earliest;
  }, Number.MAX_SAFE_INTEGER);
  return Number.isInteger(game.presentTime)
    ? game.presentTime!
    : derivedPresent === Number.MAX_SAFE_INTEGER ? 0 : derivedPresent;
}

function ownerRank(owner: Timeline["owner"]): number {
  return owner === "neutral" ? 0 : owner === "white" ? 1 : 2;
}

function boardStructuralHash(board: BoardSnapshot): number {
  let hash = 2166136261;
  hash = mixHash(hash, board.sideToMove === "white" ? 1 : 2);
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      const piece = board.board[y]?.[x];
      if (!piece) {
        continue;
      }
      hash = mixHash(hash, (piecePlane(piece) + 1) * 67 + y * 8 + x);
    }
  }
  return hash >>> 0;
}

function mixHash(hash: number, value: number): number {
  hash ^= value >>> 0;
  return Math.imul(hash, 16777619) >>> 0;
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

function boardContainsRoyal(board: BoardSnapshot): boolean {
  return board.board.some((row) => row.some((piece) => Boolean(piece && isRoyalPiece(piece.type))));
}

function latestBoard(timeline: Timeline): BoardSnapshot | undefined {
  const first = timeline.boards[0];
  return first ? timeline.boards.reduce((latest, board) => board.time > latest.time ? board : latest, first) : undefined;
}
