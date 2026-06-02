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

function samePiece(left, right) {
  if (!left || !right) {
    return left === right;
  }

  return left.color === right.color && left.type === right.type;
}

function previousBoard(timeline, time) {
  return timeline.boards.find((board) => board.time === time - 1) ?? null;
}

function addArrivalHighlights(highlights, timeline, board) {
  const previous = previousBoard(timeline, board.time);

  if (!previous) {
    if (board.origin && board.origin.type !== "source-advance") {
      addHighlight(highlights, {
        timelineId: timeline.id,
        time: board.time,
        x: board.origin.to.x,
        y: board.origin.to.y
      }, "arrived");
    }
    return;
  }

  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      const piece = board.board[y][x];
      if (piece && !samePiece(piece, previous.board[y][x])) {
        addHighlight(highlights, { timelineId: timeline.id, time: board.time, x, y }, "arrived");
      }
    }
  }
}

function movementVisuals(game) {
  const highlights = new Map();
  const arrows = [];

  for (const timeline of game.timelines) {
    for (const board of timeline.boards) {
      const origin = board.origin;
      addArrivalHighlights(highlights, timeline, board);

      if (origin && origin.type !== "source-advance") {
        addHighlight(highlights, origin.from, "move-source");
        addHighlight(highlights, origin.to, "move-target");
        arrows.push({
          from: origin.from,
          to: origin.to
        });
      }
    }
  }

  for (const position of game.checkedRoyals ?? []) {
    addHighlight(highlights, position, "check");
  }

  return { highlights, arrows };
}

function renderSquare({ position, board, selected, legalTargets, highlights, onSquareClick }) {
  const square = document.createElement("button");
  const piece = board.board[position.y][position.x];
  const target = legalTargets.find((candidate) => samePosition(candidate, position));
  const squareHighlights = highlights.get(highlightKey(position));

  square.type = "button";
  square.className = "square";
  square.dataset.light = String((position.x + position.y) % 2 === 1);
  square.dataset.positionKey = highlightKey(position);
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

  if (squareHighlights?.has("move-source")) {
    square.classList.add("was-move-source");
  }

  if (squareHighlights?.has("arrived")) {
    square.classList.add("was-arrival");
  }

  if (squareHighlights?.has("move-target")) {
    square.classList.add("was-move-target");
  }

  if (squareHighlights?.has("check")) {
    square.classList.add("is-check");
  }

  square.addEventListener("click", (event) => {
    event.preventDefault();
    onSquareClick(position);
  });
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

function measureSquareCenters(grid) {
  const gridRect = grid.getBoundingClientRect();
  const centers = new Map();

  for (const square of grid.querySelectorAll("[data-position-key]")) {
    const squareRect = square.getBoundingClientRect();
    centers.set(square.dataset.positionKey, {
      x: squareRect.left - gridRect.left + squareRect.width / 2,
      y: squareRect.top - gridRect.top + squareRect.height / 2
    });
  }

  return centers;
}

function squareCenter(centers, position) {
  return centers.get(highlightKey(position)) ?? null;
}

function appendArrowPath(svg, from, to, className) {
  const pathData = arrowPath(from, to);
  if (!pathData) {
    return;
  }

  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("class", className);
  path.setAttribute("d", pathData);
  path.setAttribute("marker-end", "url(#move-arrow-head)");
  svg.append(path);
}

function arrowPath(from, to) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);
  if (distance === 0) {
    return "";
  }

  // Pull endpoints inward so arrowheads do not obscure piece glyphs or highlight
  // rings on the source/target squares.
  const inset = Math.min(18, distance / 3);
  const start = {
    x: from.x + dx / distance * inset,
    y: from.y + dy / distance * inset
  };
  const end = {
    x: to.x - dx / distance * inset,
    y: to.y - dy / distance * inset
  };
  return `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} L ${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
}

function renderMoveArrows(grid, arrows) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const width = grid.scrollWidth;
  const height = grid.scrollHeight;
  const centers = measureSquareCenters(grid);
  svg.setAttribute("class", "move-arrows");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = `
    <defs>
      <marker id="move-arrow-head" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="3" markerHeight="3" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z"></path>
      </marker>
    </defs>
  `;
  grid.append(svg);

  for (const arrow of arrows) {
    const from = squareCenter(centers, arrow.from);
    const to = squareCenter(centers, arrow.to);
    if (!from || !to) {
      continue;
    }

    appendArrowPath(svg, from, to, "move-arrow");
  }
}

function renderTimeline({ game, presentGame, timeline, maxTime, currentPresentTime, selected, legalTargets, highlights, onSquareClick }) {
  const row = document.createElement("div");
  row.className = "timeline-row";
  row.dataset.owner = timeline.owner;
  row.dataset.active = String(isActiveTimeline(game, timeline));
  row.style.setProperty("--time-columns", String(maxTime + 1));

  const lane = document.createElement("div");
  lane.className = "timeline-label";
  lane.textContent = `L${timeline.id}`;
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
  const { highlights, arrows } = movementVisuals(game);
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
  renderMoveArrows(elements.timelineGrid, arrows);

  if (multiplayer.connected) {
    const role = multiplayer.color === "spectator" ? "spectating" : `playing ${multiplayer.color}`;
    setMultiplayerStatus(`Room ${multiplayer.roomId} · ${role}`);
  }
}
