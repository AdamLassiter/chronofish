const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const PIECES = {
  white: { king: "♔", queen: "♕", rook: "♖", bishop: "♗", knight: "♘", pawn: "♙" },
  black: { king: "♚", queen: "♛", rook: "♜", bishop: "♝", knight: "♞", pawn: "♟" }
};

const timelineGrid = document.querySelector("#timeline-grid");
const messageEl = document.querySelector("#message");
const turnLabel = document.querySelector("#turn-label");
const wasmStatus = document.querySelector("#wasm-status");
const serverStatus = document.querySelector("#server-status");
const resetButton = document.querySelector("#reset-game");
const roomInput = document.querySelector("#room-id");
const joinWhiteButton = document.querySelector("#join-white");
const joinBlackButton = document.querySelector("#join-black");
const joinSpectatorButton = document.querySelector("#join-spectator");
const multiplayerStatus = document.querySelector("#multiplayer-status");
const shareLink = document.querySelector("#share-link");

let engine = null;
let game = { turn: "white", timelines: [], nextTimelineId: 1 };
let selected = null;
let legalTargets = [];
let moveHistory = [];
let multiplayer = {
  roomId: new URLSearchParams(window.location.search).get("room") ?? makeRoomId(),
  token: localStorage.getItem("chronofish.playerToken") ?? crypto.randomUUID(),
  color: localStorage.getItem("chronofish.playerColor") ?? "local",
  events: null,
  connected: false
};

localStorage.setItem("chronofish.playerToken", multiplayer.token);
roomInput.value = multiplayer.roomId;

function makeRoomId() {
  return Math.random().toString(36).slice(2, 8);
}

function roomUrl(roomId) {
  const url = new URL(window.location.href);
  url.searchParams.set("room", roomId);
  return url;
}

function normalizeRoomId(value) {
  return value.trim().replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 48) || makeRoomId();
}

function canControlTurn() {
  return engine && (!multiplayer.connected || multiplayer.color === game.turn);
}

function setMultiplayerStatus(text) {
  multiplayerStatus.textContent = text;
}

function updateShareLink() {
  if (!multiplayer.connected) {
    shareLink.textContent = "";
    return;
  }

  const link = roomUrl(multiplayer.roomId);
  shareLink.innerHTML = `<a href="${link.href}">Share room</a>`;
}

function readWasmString(ptr) {
  const bytes = new Uint8Array(engine.memory.buffer, ptr, engine.chronofish_output_len());
  return new TextDecoder("utf-8").decode(bytes);
}

function engineSnapshot() {
  return JSON.parse(readWasmString(engine.chronofish_snapshot_json()));
}

function engineLastMessage() {
  return readWasmString(engine.chronofish_last_message());
}

function resetEngine() {
  engine.chronofish_reset();
  game = engineSnapshot();
  selected = null;
  legalTargets = [];
  moveHistory = [];
}

function replayMoves(moves) {
  engine.chronofish_reset();

  for (const move of moves) {
    engine.chronofish_apply_move(
      move.from.timelineId,
      move.from.time,
      move.from.x,
      move.from.y,
      move.to.timelineId,
      move.to.time,
      move.to.x,
      move.to.y
    );
  }

  moveHistory = moves.map((move) => ({
    from: { ...move.from },
    to: { ...move.to }
  }));
  game = engineSnapshot();
  selected = null;
  legalTargets = [];
}

function networkGame() {
  return {
    moves: moveHistory,
    snapshot: game
  };
}

function getTimeline(timelineId) {
  return game.timelines.find((timeline) => timeline.id === timelineId);
}

function getBoard(timelineId, time) {
  return getTimeline(timelineId)?.boards.find((board) => board.time === time) ?? null;
}

function getLatestBoard(timelineId) {
  const timeline = getTimeline(timelineId);
  return timeline?.boards.reduce((latest, board) => (board.time > latest.time ? board : latest), timeline.boards[0]);
}

function isLatestBoard(timelineId, time) {
  return getLatestBoard(timelineId)?.time === time;
}

function sortedTimelines() {
  return [...game.timelines].sort((a, b) => a.row - b.row || a.id - b.id);
}

function sortedBoards(timeline) {
  return [...timeline.boards].sort((a, b) => a.time - b.time);
}

function pieceAt(position) {
  return getBoard(position.timelineId, position.time)?.board[position.y]?.[position.x] ?? null;
}

function targetFor(position) {
  return legalTargets.find((target) => samePosition(target, position));
}

function samePosition(a, b) {
  return a.timelineId === b.timelineId && a.time === b.time && a.x === b.x && a.y === b.y;
}

function legalTargetsFor(position) {
  if (!engine) {
    return [];
  }

  return JSON.parse(readWasmString(engine.chronofish_legal_targets_json(
    position.timelineId,
    position.time,
    position.x,
    position.y
  )));
}

function applyEngineMove(from, to) {
  const ok = engine.chronofish_apply_move(
    from.timelineId,
    from.time,
    from.x,
    from.y,
    to.timelineId,
    to.time,
    to.x,
    to.y
  );
  const message = engineLastMessage();

  if (!ok) {
    messageEl.textContent = message;
    return null;
  }

  moveHistory.push({
    from: { ...from },
    to: { ...to }
  });
  game = engineSnapshot();
  selected = null;
  legalTargets = [];
  messageEl.textContent = message;
  return message;
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function handleSquareClick(position) {
  if (!engine) {
    messageEl.textContent = "Build the WASM engine first with `npm run build:wasm`.";
    return;
  }

  if (!canControlTurn()) {
    messageEl.textContent = multiplayer.connected
      ? `You are ${multiplayer.color}; waiting for ${game.turn}.`
      : `Select a ${game.turn} piece on a latest board.`;
    return;
  }

  const piece = pieceAt(position);
  const existingTarget = targetFor(position);

  if (selected && existingTarget) {
    const moveMessage = applyEngineMove(selected, position);
    render();

    if (moveMessage) {
      syncState("state", moveMessage);
    }
    return;
  }

  if (piece?.color === game.turn && isLatestBoard(position.timelineId, position.time)) {
    const board = getBoard(position.timelineId, position.time);

    if (board.sideToMove !== game.turn) {
      messageEl.textContent = `That board is waiting for ${board.sideToMove}.`;
      return;
    }

    selected = position;
    legalTargets = legalTargetsFor(position);
    messageEl.textContent = `${capitalize(piece.color)} ${piece.type} selected. ${legalTargets.length} legal target${legalTargets.length === 1 ? "" : "s"}.`;
    render();
    return;
  }

  selected = null;
  legalTargets = [];
  messageEl.textContent = `Select a ${game.turn} piece on a latest board.`;
  render();
}

function renderSquare(position, board) {
  const square = document.createElement("button");
  const piece = board.board[position.y][position.x];
  const target = targetFor(position);

  square.type = "button";
  square.className = "square";
  square.dataset.light = String((position.x + position.y) % 2 === 0);
  square.ariaLabel = `${FILES[position.x]}${position.y + 1}`;

  if (piece) {
    square.textContent = PIECES[piece.color][piece.type];
    square.dataset.pieceColor = piece.color;
  }

  if (selected && samePosition(selected, position)) {
    square.classList.add("is-selected");
  }

  if (target) {
    square.classList.add("is-target");
  }

  square.addEventListener("click", () => handleSquareClick(position));
  return square;
}

function renderBoard(timeline, board) {
  const boardEl = document.createElement("article");
  const latest = isLatestBoard(timeline.id, board.time);
  boardEl.className = "board-card";
  boardEl.dataset.turn = board.sideToMove;
  boardEl.dataset.latest = String(latest);

  const header = document.createElement("header");
  header.className = "board-header";
  header.innerHTML = `<span>T${timeline.id} · ${timeline.label}</span><strong>${board.sideToMove}</strong>`;

  const chessboard = document.createElement("div");
  chessboard.className = "chessboard";

  for (let y = 7; y >= 0; y -= 1) {
    for (let x = 0; x < 8; x += 1) {
      chessboard.append(renderSquare({ timelineId: timeline.id, time: board.time, x, y }, board));
    }
  }

  const footer = document.createElement("footer");
  footer.className = "board-footer";
  footer.innerHTML = `<span>${latest ? "Latest" : "Archived"}</span><strong>(${board.time}, ${timeline.row})</strong>`;

  boardEl.append(header, chessboard, footer);
  return boardEl;
}

function renderTimeline(timeline, maxTime) {
  const row = document.createElement("div");
  row.className = "timeline-row";
  row.dataset.owner = timeline.owner;
  row.style.setProperty("--time-columns", String(maxTime + 1));

  const lane = document.createElement("div");
  lane.className = "timeline-label";
  lane.textContent = timeline.label;
  row.append(lane);

  for (const board of sortedBoards(timeline)) {
    const boardEl = renderBoard(timeline, board);
    boardEl.style.gridColumn = String(board.time + 2);
    row.append(boardEl);
  }

  return row;
}

function render() {
  turnLabel.textContent = `${capitalize(game.turn)} to move`;
  const maxTime = Math.max(0, ...game.timelines.flatMap((timeline) => timeline.boards.map((board) => board.time)));
  timelineGrid.replaceChildren(...sortedTimelines().map((timeline) => renderTimeline(timeline, maxTime)));

  if (multiplayer.connected) {
    const role = multiplayer.color === "spectator" ? "spectating" : `playing ${multiplayer.color}`;
    setMultiplayerStatus(`Room ${multiplayer.roomId} · ${role}`);
  }
}

async function postRoom(action, body) {
  const response = await fetch(`/api/rooms/${encodeURIComponent(multiplayer.roomId)}/${action}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body)
  });
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.error ?? "Room request failed");
  }

  return payload;
}

function applyRemoteRoom(room, message = "") {
  if (room?.game?.moves && engine) {
    replayMoves(room.game.moves);
  } else if (room?.game?.snapshot) {
    game = room.game.snapshot;
    moveHistory = room.game.moves ?? [];
  } else if (room?.game?.timelines) {
    game = room.game;
    moveHistory = [];
  }

  selected = null;
  legalTargets = [];
  updateShareLink();
  render();

  if (message) {
    messageEl.textContent = message;
  }
}

function connectEvents() {
  multiplayer.events?.close();
  multiplayer.events = new EventSource(`/api/rooms/${encodeURIComponent(multiplayer.roomId)}/events`);

  multiplayer.events.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);

    if (payload.type === "sync") {
      applyRemoteRoom(payload.room);
      return;
    }

    if (payload.type === "players") {
      render();
      return;
    }

    if (payload.type === "state" || payload.type === "reset") {
      applyRemoteRoom(payload.room, payload.message);
    }
  });

  multiplayer.events.addEventListener("error", () => {
    setMultiplayerStatus(`Room ${multiplayer.roomId} · reconnecting`);
  });
}

async function joinRoom(color) {
  if (!engine) {
    messageEl.textContent = "WASM engine is not loaded yet.";
    return;
  }

  multiplayer.roomId = normalizeRoomId(roomInput.value);
  multiplayer.color = color;
  roomInput.value = multiplayer.roomId;

  const payload = await postRoom("join", {
    color,
    token: multiplayer.token,
    game: networkGame()
  });

  multiplayer.connected = true;
  multiplayer.color = payload.color;
  localStorage.setItem("chronofish.playerColor", payload.color);
  window.history.replaceState({}, "", roomUrl(multiplayer.roomId));
  applyRemoteRoom(payload.room, payload.color === "spectator" ? "Spectating room." : `Joined as ${payload.color}.`);
  connectEvents();
}

async function syncState(action, message) {
  if (!multiplayer.connected || multiplayer.color === "spectator") {
    return;
  }

  try {
    await postRoom(action, {
      token: multiplayer.token,
      color: multiplayer.color,
      game: networkGame(),
      message
    });
  } catch (error) {
    messageEl.textContent = error.message;
  }
}

async function loadWasmStatus() {
  try {
    const wasmPath = "../target/wasm32-unknown-unknown/debug/chronofish_engine.wasm";
    const { instance } = await WebAssembly.instantiateStreaming(fetch(wasmPath), {});
    engine = instance.exports;
    resetEngine();
    wasmStatus.textContent = `WASM ready (version ${readWasmString(engine.chronofish_version())})`;
    wasmStatus.dataset.state = "ready";
    messageEl.textContent = "Select a white piece on a latest board.";
    render();
  } catch (error) {
    console.error(error);
    wasmStatus.textContent = "WASM not built";
    wasmStatus.dataset.state = "error";
    messageEl.textContent = "Build the WASM engine first with `npm run build:wasm`.";
    render();
  }
}

async function loadServerStatus() {
  try {
    const response = await fetch("/api/version");
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error ?? "Server version unavailable");
    }

    serverStatus.textContent = `Server ready (version ${payload.version})`;
    serverStatus.dataset.state = "ready";
  } catch (error) {
    console.error(error);
    serverStatus.textContent = "Server unavailable";
    serverStatus.dataset.state = "error";
  }
}

resetButton.addEventListener("click", () => {
  if (!engine) {
    messageEl.textContent = "WASM engine is not loaded yet.";
    return;
  }

  if (multiplayer.connected && multiplayer.color === "spectator") {
    messageEl.textContent = "Spectators cannot reset a multiplayer room.";
    return;
  }

  resetEngine();
  messageEl.textContent = "Select a white piece on a latest board.";
  render();
  syncState("reset", `${capitalize(multiplayer.color)} reset the room.`);
});

joinWhiteButton.addEventListener("click", () => {
  joinRoom("white").catch((error) => {
    messageEl.textContent = error.message;
  });
});

joinBlackButton.addEventListener("click", () => {
  joinRoom("black").catch((error) => {
    messageEl.textContent = error.message;
  });
});

joinSpectatorButton.addEventListener("click", () => {
  joinRoom("spectator").catch((error) => {
    messageEl.textContent = error.message;
  });
});

loadWasmStatus();
loadServerStatus();
render();
