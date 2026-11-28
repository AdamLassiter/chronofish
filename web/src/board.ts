/** Renders the multiverse of boards for the current game. */

import { chooseSquare } from "./actions.js";
import { element } from "./dom.js";
import { renderEvaluationFrame } from "./evaluation.js";
import { groupHistory, renderHistoryTurn } from "./history.js";
import { stopPolling } from "./lobby.js";
import { formatPosition, legalTargets, samePosition } from "./moves.js";
import { followPresentLine } from "./scroll.js";
import { currentGame, requests, state } from "./state.js";
import {
  BoardSnapshot,
  BoardStatus,
  GameView,
  HighlightMap,
  Piece,
  PieceType,
  Position,
  Side,
  Timeline,
} from "./types.js";
import { highlightKey, movementVisuals, renderMoveArrows } from "./visuals.js";

export function latestTime(timeline: Timeline): number | undefined {
  return timeline.boards.reduce<number | undefined>(
    (latest, board) =>
      latest === undefined || board.time > latest ? board.time : latest,
    undefined,
  );
}

export function currentPresentTime(game: GameView): number {
  if (game.submittedPresentTime !== null) return game.submittedPresentTime;
  if (game.presentTime !== null) return game.presentTime;
  const activeTimes = game.timelines
    .filter((timeline) => game.activeTimelines.includes(timeline.id))
    .map(latestTime)
    .filter((time): time is number => time !== undefined);
  return activeTimes.length > 0 ? Math.min(...activeTimes) : 0;
}

export function boardStatus(
  game: GameView,
  timeline: Timeline,
  snapshot: BoardSnapshot,
  presentTime: number,
): BoardStatus {
  if (!game.activeTimelines.includes(timeline.id)) return "Inactive";
  if (
    game.ruleset !== "standard" &&
    snapshot.time === latestTime(timeline) &&
    snapshot.sideToMove === game.turn
  ) {
    return snapshot.time === presentTime ? "Required" : "Optional";
  }
  if (snapshot.time > presentTime) return "Future";
  return snapshot.time === latestTime(timeline) ? "Latest" : "Past";
}

export function renderGame(): void {
  const { summary, game, viewerSide, history } = currentGame();
  element("#game-title").textContent =
    `${summary.white.displayName} (White) vs ${summary.black.displayName} (Black)`;
  element("#game-status").textContent = game.outcome
    ? `${game.outcome.winner ?? "Nobody"} · ${game.outcome.reason}`
    : `${game.turn} to move · ${game.message}`;
  const mine = viewerSide === game.turn;
  const finished = game.outcome !== null;
  const currentPlayer = game.turn === "white" ? summary.white : summary.black;
  element<HTMLButtonElement>("#submit-turn").disabled =
    requests.turnPending || finished || !mine || !game.canSubmit;
  const undo = element<HTMLButtonElement>("#undo-staged-move");
  undo.disabled =
    requests.turnPending || finished || !mine || !game.hasStagedMoves;
  undo.title = game.hasStagedMoves
    ? "Undo the most recent move in this unsubmitted turn"
    : "There is no staged move to undo";
  const retryAi = element<HTMLButtonElement>("#retry-ai");
  retryAi.disabled = finished || !currentPlayer.isBot;
  retryAi.title = currentPlayer.isBot
    ? "Ask the server to try the AI turn again"
    : "The current player is not an AI";
  element<HTMLButtonElement>("#abandon-game").disabled = finished;
  if (finished) stopPolling();
  renderEvaluationFrame(summary, viewerSide);
  const presentTime = currentPresentTime(game);
  const maxTime = Math.max(
    presentTime,
    0,
    ...game.timelines.flatMap((timeline) =>
      timeline.boards.map((snapshot) => snapshot.time),
    ),
  );
  const { highlights, arrows } = movementVisuals(game);
  const grid = element("#timeline-grid");
  const targets = state.selected
    ? legalTargets(state.selected).map((move) => move.to)
    : [];
  const timelines = [...game.timelines].sort(
    (left, right) => left.row - right.row || left.id - right.id,
  );
  grid.replaceChildren(
    ...timelines.map((timeline) =>
      renderTimeline(game, timeline, maxTime, presentTime, targets, highlights),
    ),
  );
  element<HTMLOListElement>("#move-history").replaceChildren(
    ...groupHistory(history).map(renderHistoryTurn),
  );
  requestAnimationFrame(() => {
    renderMoveArrows(grid, arrows);
    const stage = element("#multiverse");
    const origin = grid.querySelector<HTMLElement>('[data-board="0:0"]');
    const firstLayout = Boolean(origin) && !stage.dataset.positioned;
    if (origin && firstLayout) {
      origin.scrollIntoView({ block: "center", inline: "start" });
      stage.dataset.positioned = "true";
    }
    if (stage.dataset.presentTime !== String(presentTime)) {
      stage.dataset.presentTime = String(presentTime);
      followPresentLine(stage, grid, !firstLayout);
    }
  });
}

export function renderTimeline(
  game: GameView,
  timeline: Timeline,
  maxTime: number,
  presentTime: number,
  targets: Position[],
  highlights: HighlightMap,
): HTMLElement {
  const row = document.createElement("div");
  row.className = "timeline-row";
  row.dataset.owner = timeline.owner;
  row.dataset.active = String(game.activeTimelines.includes(timeline.id));
  row.style.setProperty("--time-columns", String(maxTime + 1));

  const marker = document.createElement("div");
  marker.className = "present-line";
  marker.style.gridColumn = String(presentTime + 1);
  row.append(marker);

  const snapshots = [...timeline.boards].sort(
    (left, right) => left.time - right.time,
  );
  for (const snapshot of snapshots) {
    const board = renderBoard(
      game,
      timeline,
      snapshot,
      presentTime,
      targets,
      highlights,
    );
    board.style.gridColumn = String(snapshot.time + 1);
    row.append(board);
  }
  return row;
}

export function renderBoard(
  game: GameView,
  timeline: Timeline,
  snapshot: BoardSnapshot,
  presentTime: number,
  targets: Position[],
  highlights: HighlightMap,
): HTMLElement {
  const card = document.createElement("article");
  card.className = "board-card";
  card.dataset.turn = snapshot.sideToMove;
  card.dataset.latest = String(snapshot.time === latestTime(timeline));
  card.dataset.present = String(snapshot.time === presentTime);
  card.dataset.board = `${timeline.id}:${snapshot.time}`;
  const board = document.createElement("div");
  board.className = "chessboard";
  for (let y = 7; y >= 0; y -= 1) {
    for (let x = 0; x < 8; x += 1) {
      const position: Position = {
        timelineId: timeline.id,
        time: snapshot.time,
        x,
        y,
      };
      const square = document.createElement("button");
      const piece = snapshot.board[y]?.[x] ?? null;
      const squareHighlights = highlights.get(highlightKey(position));
      square.type = "button";
      square.className = "square";
      square.dataset.light = String((x + y) % 2 === 1);
      square.dataset.positionKey = highlightKey(position);
      square.ariaLabel = `${String.fromCharCode(97 + x)}${y + 1}`;
      if (piece) {
        square.textContent = pieceGlyph(piece);
        square.dataset.pieceColor = piece.color;
      }
      if (state.selected && samePosition(state.selected, position))
        square.classList.add("is-selected");
      if (targets.some((target) => samePosition(target, position)))
        square.classList.add("is-target");
      if (squareHighlights?.has("move-source"))
        square.classList.add("was-move-source");
      if (squareHighlights?.has("arrived")) square.classList.add("was-arrival");
      if (squareHighlights?.has("move-target"))
        square.classList.add("was-move-target");
      if (squareHighlights?.has("check")) square.classList.add("is-check");
      square.title = `${formatPosition(position)}${piece ? ` ${piece.color} ${piece.type}` : ""}`;
      square.onclick = () => chooseSquare(position);
      board.append(square);
    }
  }
  const footer = document.createElement("footer");
  footer.className = "board-footer";
  footer.innerHTML = `
    <span>${boardStatus(game, timeline, snapshot, presentTime)}</span>
    <span class="board-side-to-move">${snapshot.sideToMove}</span>
    <strong>T${snapshot.time} L${timeline.id}</strong>
  `;
  card.append(board, footer);
  return card;
}

export const PIECE_GLYPHS: Record<Side, Record<PieceType, string>> = {
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
    brawn: "♙",
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
    brawn: "♟",
  },
};

export function pieceGlyph(piece: Piece): string {
  return PIECE_GLYPHS[piece.color][piece.type];
}
