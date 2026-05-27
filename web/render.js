import { FILES, PIECES } from "./constants.js";
import {
  isActiveTimeline,
  isLatestBoard,
  presentTime,
  samePosition,
  sortedBoards,
  sortedTimelines
} from "./board.js";

function boardStatus({ game, presentGame, timeline, board, currentPresentTime }) {
  // Status uses the committed snapshot so staged moves do not relabel boards as
  // past/future before the player submits the turn.
  if (!isActiveTimeline(game, timeline)) {
    return "Inactive";
  }

  const committedTimeline = presentGame.timelines.find((candidate) => candidate.id === timeline.id);
  const committedLatestTime = committedTimeline
    ? Math.max(...committedTimeline.boards.map((candidate) => candidate.time))
    : null;

  if (committedLatestTime === null || board.time > committedLatestTime || board.time > currentPresentTime) {
    return "Future";
  }

  return board.time === committedLatestTime ? "Latest" : "Past";
}

function highlightKey(position) {
  return `${position.timelineId}:${position.time}:${position.x}:${position.y}`;
}

function addHighlight(highlights, position, type) {
  const key = highlightKey(position);
  const existing = highlights.get(key) ?? new Set();
  existing.add(type);
  highlights.set(key, existing);
}

function movementHighlights(game) {
  const highlights = new Map();

  for (const timeline of game.timelines) {
    for (const board of timeline.boards) {
      const origin = board.origin;
      if (!origin) {
        continue;
      }

      addHighlight(highlights, origin.from, "departed");

      if (origin.type === "branch") {
        addHighlight(highlights, origin.to, "branch-target");
      }

      if (origin.type !== "source-advance") {
        addHighlight(highlights, {
          timelineId: timeline.id,
          time: board.time,
          x: origin.to.x,
          y: origin.to.y
        }, "arrived");
      }
    }
  }

  return highlights;
}

function renderSquare({ position, board, selected, legalTargets, highlights, onSquareClick }) {
  const square = document.createElement("button");
  const piece = board.board[position.y][position.x];
  const target = legalTargets.find((candidate) => samePosition(candidate, position));
  const squareHighlights = highlights.get(highlightKey(position));

  square.type = "button";
  square.className = "square";
  square.dataset.light = String((position.x + position.y) % 2 === 1);
  square.ariaLabel = `${FILES[position.x]}${position.y + 1}`;

  if (piece) {
    // Pieces are text glyphs; CSS handles color-specific shadow/glow contrast.
    square.textContent = PIECES[piece.color][piece.type];
    square.dataset.pieceColor = piece.color;
  }

  if (selected && samePosition(selected, position)) {
    square.classList.add("is-selected");
  }

  if (target) {
    square.classList.add("is-target");
  }

  if (squareHighlights?.has("departed")) {
    square.classList.add("was-departure");
  }

  if (squareHighlights?.has("arrived")) {
    square.classList.add("was-arrival");
  }

  if (squareHighlights?.has("branch-target")) {
    square.classList.add("was-branch-target");
  }

  square.addEventListener("click", () => onSquareClick(position));
  return square;
}

function renderBoard({ game, presentGame, timeline, board, currentPresentTime, selected, legalTargets, highlights, onSquareClick }) {
  const boardEl = document.createElement("article");
  const latest = isLatestBoard(game, timeline.id, board.time);
  const status = boardStatus({ game, presentGame, timeline, board, currentPresentTime });
  boardEl.className = "board-card";
  boardEl.dataset.turn = board.sideToMove;
  boardEl.dataset.latest = String(latest);
  boardEl.dataset.present = String(board.time === currentPresentTime);

  const chessboard = document.createElement("div");
  chessboard.className = "chessboard";

  for (let y = 7; y >= 0; y -= 1) {
    for (let x = 0; x < 8; x += 1) {
      chessboard.append(renderSquare({
        position: { timelineId: timeline.id, time: board.time, x, y },
        board,
        selected,
        legalTargets,
        highlights,
        onSquareClick
      }));
    }
  }

  const footer = document.createElement("footer");
  // Footer carries all board metadata after the old board header was removed to
  // save vertical space.
  footer.className = "board-footer";
  footer.innerHTML = `
    <span>${status}</span>
    <span class="board-side-to-move">${board.sideToMove}</span>
    <strong>T${board.time} L${timeline.id}</strong>
  `;

  boardEl.append(chessboard, footer);
  return boardEl;
}

function renderTimeline({ game, presentGame, timeline, maxTime, currentPresentTime, selected, legalTargets, highlights, onSquareClick }) {
  const row = document.createElement("div");
  row.className = "timeline-row";
  row.dataset.owner = timeline.owner;
  row.dataset.active = String(isActiveTimeline(game, timeline));
  row.style.setProperty("--time-columns", String(maxTime + 1));

  const lane = document.createElement("div");
  lane.className = "timeline-label";
  lane.textContent = timeline.label;
  row.append(lane);

  const marker = document.createElement("div");
  // The marker follows committed present time, not speculative staged moves.
  marker.className = "present-line";
  marker.style.gridColumn = String(currentPresentTime + 2);
  row.append(marker);

  for (const board of sortedBoards(timeline)) {
    const boardEl = renderBoard({ game, presentGame, timeline, board, currentPresentTime, selected, legalTargets, highlights, onSquareClick });
    boardEl.style.gridColumn = String(board.time + 2);
    row.append(boardEl);
  }

  return row;
}

export function renderGame({ game, presentGame, selected, legalTargets, multiplayer, elements, onSquareClick, setMultiplayerStatus }) {
  // The timeline grid is small enough that replacing DOM children is clearer than
  // incremental reconciliation.
  const maxTime = Math.max(0, ...game.timelines.flatMap((timeline) => timeline.boards.map((board) => board.time)));
  const currentPresentTime = presentTime(presentGame);
  const highlights = movementHighlights(game);
  elements.timelineGrid.replaceChildren(
    ...sortedTimelines(game).map((timeline) => renderTimeline({
      game,
      presentGame,
      timeline,
      maxTime,
      currentPresentTime,
      selected,
      legalTargets,
      highlights,
      onSquareClick
    }))
  );

  if (multiplayer.connected) {
    const role = multiplayer.color === "spectator" ? "spectating" : `playing ${multiplayer.color}`;
    setMultiplayerStatus(`Room ${multiplayer.roomId} · ${role}`);
  }
}
