import { elements } from "./dom.js";
import { capitalize, getBoard, getLatestBoard, isLatestBoard, samePosition } from "./board.js";
import { renderGame } from "./render.js";

let engine = null;
let game = { turn: "white", timelines: [], nextTimelineId: 1 };
// Last submitted snapshot. While a turn is staged, rendering compares against
// this so the present line and board status labels do not jump before Submit.
let committedGame = game;
let selected = null;
let legalTargets = [];
// submittedTurns is replayable room history; stagedMoves is local undo state for
// the current unsubmitted turn only.
let submittedTurns = [];
let stagedMoves = [];
let aiWorker = null;
let aiRequestId = 0;
let phase = "lobby";
let assignments = {
  white: localStorage.getItem("chronofish.whitePlayer") ?? "local",
  black: localStorage.getItem("chronofish.blackPlayer") ?? "local"
};
let bot = {
  // Bot sides are chosen in the lobby. In multiplayer, a bot explicitly occupies
  // its side with its own token so it follows the same room seating rules.
  thinking: false,
  tokens: {}
};
let multiplayer = {
  // Room id lives in the URL so sharing the address reconstructs the room.
  roomId: new URLSearchParams(window.location.search).get("room") ?? makeRoomId(),
  token: localStorage.getItem("chronofish.playerToken") ?? crypto.randomUUID(),
  color: localStorage.getItem("chronofish.playerColor") ?? "local",
  events: null,
  connected: false
};
let currentRoom = null;

localStorage.setItem("chronofish.playerToken", multiplayer.token);
elements.roomInput.value = multiplayer.roomId;
elements.whitePlayerSelect.value = assignments.white;
elements.blackPlayerSelect.value = assignments.black;

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
  if (!engine || phase !== "game" || assignments[game.turn] === "bot") {
    return false;
  }

  if (!multiplayer.connected) {
    return assignments[game.turn] === "local";
  }

  return assignments[game.turn] === "human" && multiplayer.color === game.turn;
}

function setMultiplayerStatus(text) {
  elements.multiplayerStatus.textContent = text;
}

function updateShareLink() {
  if (!multiplayer.connected) {
    elements.shareLink.textContent = "";
    return;
  }

  const link = roomUrl(multiplayer.roomId);
  elements.shareLink.innerHTML = `<a href="${link.href}">Share room</a>`;
}

function normalizeAssignment(value, fallback = "local") {
  return ["local", "human", "bot", "open"].includes(value) ? value : fallback;
}

function readAssignments() {
  return {
    white: normalizeAssignment(elements.whitePlayerSelect.value, "local"),
    black: normalizeAssignment(elements.blackPlayerSelect.value, "local")
  };
}

function writeAssignments(nextAssignments) {
  assignments = {
    white: normalizeAssignment(nextAssignments?.white, "local"),
    black: normalizeAssignment(nextAssignments?.black, "local")
  };
  elements.whitePlayerSelect.value = assignments.white;
  elements.blackPlayerSelect.value = assignments.black;
  localStorage.setItem("chronofish.whitePlayer", assignments.white);
  localStorage.setItem("chronofish.blackPlayer", assignments.black);
  updateBotStatus();
}

function gamePayload(nextPhase = phase) {
  return {
    phase: nextPhase,
    assignments,
    turns: submittedTurns,
    snapshot: game
  };
}

function lobbyPayload() {
  return {
    phase: "lobby",
    assignments,
    snapshot: committedGame
  };
}

function readWasmString(ptr) {
  // Rust string exports share one output buffer. Copy the bytes immediately after
  // receiving a pointer because the next engine string call overwrites it.
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
  // Engine reset clears both visible and committed state plus all local history.
  engine.chronofish_reset();
  game = engineSnapshot();
  committedGame = game;
  selected = null;
  legalTargets = [];
  submittedTurns = [];
  stagedMoves = [];
}

function replayTurns(turns) {
  // Multiplayer sync stores submitted turns. Replaying them through Rust rebuilds
  // authoritative engine state instead of trusting an arbitrary remote snapshot.
  engine.chronofish_reset();

  for (const turn of turns) {
    for (const move of turn) {
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
    engine.chronofish_submit_turn();
  }

  submittedTurns = turns.map((turn) => turn.map(cloneMove));
  stagedMoves = [];
  game = engineSnapshot();
  committedGame = game;
  selected = null;
  legalTargets = [];
}

function cloneMove(move) {
  return {
    from: { ...move.from },
    to: { ...move.to }
  };
}

function botToken(color) {
  const key = `chronofish.botToken.${multiplayer.roomId}.${color}`;
  let token = localStorage.getItem(key);
  if (!token) {
    token = crypto.randomUUID();
    localStorage.setItem(key, token);
  }
  return token;
}

function botColors() {
  return ["white", "black"].filter((color) => assignments[color] === "bot");
}

function ensureAiWorker() {
  // Lazily create the worker so normal local play avoids WASM worker startup.
  if (!aiWorker) {
    aiWorker = new Worker("./ai-worker.js", { type: "module" });
    aiWorker.addEventListener("message", handleAiWorkerMessage);
  }
  return aiWorker;
}

function turnSignature() {
  // Used to ignore stale AI replies if the position changes while the worker is
  // thinking.
  return `${game.turn}:${submittedTurns.length}:${JSON.stringify(submittedTurns.at(-1) ?? [])}`;
}

function updateBotStatus(text = null) {
  if (text) {
    elements.botStatus.textContent = text;
    return;
  }
  const colors = botColors();
  elements.botStatus.textContent = colors.length ? `Bot ${colors.join(" + ")}` : "Bot idle";
}

function pieceAt(position) {
  return getBoard(game, position.timelineId, position.time)?.board[position.y]?.[position.x] ?? null;
}

function targetFor(position) {
  return legalTargets.find((target) => samePosition(target, position));
}

function legalTargetsFor(position) {
  if (!engine) {
    return [];
  }

  // Target highlighting is delegated to Rust so previews and final application
  // use the same legality code.
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
    elements.message.textContent = message;
    return null;
  }

  // Successful moves stay staged until Submit. Undo and Reset operate on this
  // list, not on the whole room/game history.
  stagedMoves.push({
    from: { ...from },
    to: { ...to }
  });
  game = engineSnapshot();
  selected = null;
  legalTargets = [];
  elements.message.textContent = message;
  return message;
}

function handleSquareClick(position) {
  if (!engine) {
    elements.message.textContent = "Build the WASM engine first with `cargo build --manifest-path engine/Cargo.toml --target wasm32-unknown-unknown`.";
    return;
  }

  if (!canControlTurn()) {
    elements.message.textContent = phase !== "game"
      ? "Start the game from the lobby first."
      : multiplayer.connected
        ? `You are ${multiplayer.color}; waiting for ${game.turn}.`
        : `Waiting for ${game.turn}.`;
    return;
  }

  const piece = pieceAt(position);
  const existingTarget = targetFor(position);

  // Click a highlighted target to move; click a latest own piece to select and
  // request legal targets from the engine.
  if (selected && existingTarget) {
    const moveMessage = applyEngineMove(selected, position);
    render();
    return;
  }

  if (piece?.color === game.turn && isLatestBoard(game, position.timelineId, position.time)) {
    const board = getBoard(game, position.timelineId, position.time);

    if (board.sideToMove !== game.turn) {
      elements.message.textContent = `That board is waiting for ${board.sideToMove}.`;
      return;
    }

    selected = position;
    legalTargets = legalTargetsFor(position);
    elements.message.textContent = `${capitalize(piece.color)} ${piece.type} selected. ${legalTargets.length} legal target${legalTargets.length === 1 ? "" : "s"}.`;
    render();
    return;
  }

  selected = null;
  legalTargets = [];
  elements.message.textContent = `Select a ${game.turn} piece on a latest board.`;
  render();
}

function render() {
  const inGame = phase === "game";
  elements.startGameButton.disabled = !engine || inGame || multiplayer.color === "spectator";
  elements.joinWhiteButton.disabled = inGame;
  elements.joinBlackButton.disabled = inGame;
  elements.whitePlayerSelect.disabled = inGame || multiplayer.color === "spectator";
  elements.blackPlayerSelect.disabled = inGame || multiplayer.color === "spectator";
  elements.resetButton.disabled = !inGame;
  elements.undoMoveButton.disabled = !inGame;
  elements.submitTurnButton.disabled = !inGame;

  // State and IO live here; renderGame only rebuilds the DOM from supplied data.
  renderGame({
    game,
    presentGame: committedGame,
    selected,
    legalTargets,
    multiplayer,
    elements,
    onSquareClick: handleSquareClick,
    setMultiplayerStatus
  });
}

function setHudCollapsed(collapsed) {
  // Preserve the space-saving preference across reloads.
  elements.hud.dataset.collapsed = String(collapsed);
  elements.toggleHudButton.textContent = collapsed ? "Show" : "Hide";
  elements.toggleHudButton.setAttribute("aria-expanded", String(!collapsed));
  localStorage.setItem("chronofish.hudCollapsed", String(collapsed));
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
  currentRoom = room;
  if (bot.thinking) {
    aiRequestId += 1;
    bot.thinking = false;
    updateBotStatus("Bot stale");
  }

  if (room?.game?.phase) {
    phase = room.game.phase;
  }
  if (room?.game?.assignments) {
    writeAssignments(room.game.assignments);
  }

  if (room?.game?.turns && engine) {
    replayTurns(room.game.turns);
  } else if (room?.game?.snapshot) {
    game = room.game.snapshot;
    committedGame = game;
    submittedTurns = room.game.turns ?? [];
    stagedMoves = [];
  } else if (room?.game?.timelines) {
    game = room.game;
    committedGame = game;
    submittedTurns = [];
    stagedMoves = [];
  }

  selected = null;
  legalTargets = [];
  updateShareLink();
  render();

  if (message) {
    elements.message.textContent = message;
  }

  maybeStartBotTurn();
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
      applyRemoteRoom(payload.room);
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
    elements.message.textContent = "WASM engine is not loaded yet.";
    return;
  }

  multiplayer.roomId = normalizeRoomId(elements.roomInput.value);
  multiplayer.color = color;
  elements.roomInput.value = multiplayer.roomId;
  if (color === "white" || color === "black") {
    const nextAssignments = readAssignments();
    nextAssignments[color] = "human";
    const other = color === "white" ? "black" : "white";
    if (nextAssignments[other] === "local") {
      nextAssignments[other] = "open";
    }
    writeAssignments(nextAssignments);
  }

  const payload = await postRoom("join", {
    color,
    token: multiplayer.token,
    game: lobbyPayload()
  });

  multiplayer.connected = true;
  multiplayer.color = payload.color;
  localStorage.setItem("chronofish.playerColor", payload.color);
  window.history.replaceState({}, "", roomUrl(multiplayer.roomId));
  applyRemoteRoom(payload.room, payload.color === "spectator" ? "Spectating room." : `Joined as ${payload.color}.`);
  connectEvents();
}

async function syncState(action, message, credentials = null) {
  const actor = credentials ?? { color: multiplayer.color, token: multiplayer.token };
  if (!multiplayer.connected || actor.color === "spectator") {
    return;
  }

  try {
    await postRoom(action, {
      token: actor.token,
      color: actor.color,
      game: gamePayload("game"),
      message
    });
  } catch (error) {
    elements.message.textContent = error.message;
  }
}

async function syncLobby(message = "Lobby updated.") {
  if (!multiplayer.connected || multiplayer.color === "spectator") {
    return;
  }

  try {
    await postRoom("state", {
      token: multiplayer.token,
      color: multiplayer.color,
      game: lobbyPayload(),
      message
    });
  } catch (error) {
    elements.message.textContent = error.message;
  }
}

function validateAssignments(nextAssignments) {
  for (const color of ["white", "black"]) {
    if (nextAssignments[color] === "open") {
      throw new Error(`${capitalize(color)} needs a player or bot before starting.`);
    }
    if (!multiplayer.connected && nextAssignments[color] === "human") {
      throw new Error("Join a room before starting with online humans.");
    }
    if (multiplayer.connected && nextAssignments[color] === "local") {
      throw new Error("Use Online human or Bot for room games.");
    }
    if (
      multiplayer.connected &&
      nextAssignments[color] === "human" &&
      !currentRoom?.players?.[color]
    ) {
      throw new Error(`${capitalize(color)} is set to online human but no player is seated.`);
    }
  }
}

async function startGame() {
  if (!engine) {
    elements.message.textContent = "WASM engine is not loaded yet.";
    return;
  }

  writeAssignments(readAssignments());
  validateAssignments(assignments);
  resetEngine();
  phase = "game";

  if (!multiplayer.connected) {
    elements.message.textContent = "Local game started.";
    render();
    maybeStartBotTurn();
    return;
  }

  if (multiplayer.color === "spectator") {
    throw new Error("Spectators cannot start the game.");
  }

  for (const color of botColors()) {
    await seatBot(color);
  }

  await postRoom("state", {
    token: multiplayer.token,
    color: multiplayer.color,
    game: gamePayload("game"),
    message: "Game started."
  });
  elements.message.textContent = "Game started.";
  render();
  maybeStartBotTurn();
}

async function seatBot(color) {
  bot.tokens[color] = botToken(color);
  const payload = await postRoom("join", {
    color,
    token: bot.tokens[color],
    game: lobbyPayload()
  });

  applyRemoteRoom(payload.room);
}

function maybeStartBotTurn() {
  if (
    !engine ||
    phase !== "game" ||
    assignments[game.turn] !== "bot" ||
    bot.thinking ||
    stagedMoves.length > 0
  ) {
    updateBotStatus();
    return;
  }

  const id = ++aiRequestId;
  bot.thinking = true;
  updateBotStatus(`Bot ${game.turn} thinking`);
  ensureAiWorker().postMessage({
    id,
    turns: submittedTurns.map((turn) => turn.map(cloneMove)),
    depth: 2,
    nodes: 25_000
  });
}

function handleAiWorkerMessage(event) {
  const { id, ok, result, error } = event.data;
  if (id !== aiRequestId) {
    return;
  }

  bot.thinking = false;
  if (!ok) {
    updateBotStatus("Bot error");
    elements.message.textContent = error;
    return;
  }

  const botColor = game.turn;
  if (assignments[botColor] !== "bot" || stagedMoves.length > 0) {
    updateBotStatus("Bot stale");
    return;
  }

  if (result.status !== "ok" || result.moves.length === 0) {
    updateBotStatus("Bot no legal turn");
    elements.message.textContent = "Bot found no legal turn.";
    return;
  }

  const before = turnSignature();
  for (const move of result.moves) {
    if (!applyEngineMove(move.from, move.to)) {
      updateBotStatus("Bot move failed");
      return;
    }
  }

  if (before !== turnSignature()) {
    updateBotStatus("Bot stale");
    return;
  }

  if (!engine.chronofish_submit_turn()) {
    elements.message.textContent = engineLastMessage();
    updateBotStatus("Bot submit failed");
    return;
  }

  game = engineSnapshot();
  submittedTurns.push(stagedMoves.map(cloneMove));
  stagedMoves = [];
  committedGame = game;
  selected = null;
  legalTargets = [];
  const message = `Bot ${botColor} moved. ${engineLastMessage()}`;
  elements.message.textContent = message;
  updateBotStatus(`Bot ${botColor} moved`);
  render();
  syncState("state", message, { color: botColor, token: bot.tokens[botColor] ?? botToken(botColor) });
}

async function loadWasmStatus() {
  try {
    const wasmPath = "./chronofish_engine.wasm";
    const { instance } = await WebAssembly.instantiateStreaming(fetch(wasmPath), {});
    engine = instance.exports;
    resetEngine();
    elements.wasmStatus.textContent = `Engine v${readWasmString(engine.chronofish_version())}`;
    elements.wasmStatus.dataset.state = "ready";
    elements.message.textContent = "Configure the lobby, then start the game.";
    render();
  } catch (error) {
    console.error(error);
    elements.wasmStatus.textContent = "WASM not built";
    elements.wasmStatus.dataset.state = "error";
    elements.message.textContent = "Build the WASM engine first with `cargo build --manifest-path engine/Cargo.toml --target wasm32-unknown-unknown`.";
    render();
  }
}

async function loadServerStatus() {
  try {
    const response = await fetch("/api/version");
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error ?? "Server unavailable");
    }

    elements.serverStatus.textContent = `Server v${payload.version}`;
    elements.serverStatus.dataset.state = "ready";
  } catch (error) {
    console.error(error);
    elements.serverStatus.textContent = "Server unavailable";
    elements.serverStatus.dataset.state = "error";
  }
}

elements.resetButton.addEventListener("click", () => {
  if (!engine) {
    elements.message.textContent = "WASM engine is not loaded yet.";
    return;
  }

  let undone = 0;
  while (stagedMoves.length > 0 && engine.chronofish_undo_staged_move()) {
    stagedMoves.pop();
    undone += 1;
  }

  game = engineSnapshot();
  committedGame = game;
  selected = null;
  legalTargets = [];
  elements.message.textContent = undone > 0 ? "Reset staged moves." : "No staged moves to reset.";
  render();
});

elements.undoMoveButton.addEventListener("click", () => {
  if (!engine) {
    elements.message.textContent = "WASM engine is not loaded yet.";
    return;
  }

  if (!engine.chronofish_undo_staged_move()) {
    elements.message.textContent = engineLastMessage();
    return;
  }

  stagedMoves.pop();
  game = engineSnapshot();
  if (stagedMoves.length === 0) {
    committedGame = game;
  }
  selected = null;
  legalTargets = [];
  elements.message.textContent = engineLastMessage();
  render();
});

elements.submitTurnButton.addEventListener("click", () => {
  if (!engine) {
    elements.message.textContent = "WASM engine is not loaded yet.";
    return;
  }

  if (!engine.chronofish_submit_turn()) {
    elements.message.textContent = engineLastMessage();
    return;
  }

  game = engineSnapshot();
  if (stagedMoves.length > 0) {
    submittedTurns.push(stagedMoves.map(cloneMove));
    stagedMoves = [];
  }
  committedGame = game;
  selected = null;
  legalTargets = [];
  elements.message.textContent = engineLastMessage();
  render();
  syncState("state", engineLastMessage());
  maybeStartBotTurn();
});

elements.toggleHudButton.addEventListener("click", () => {
  setHudCollapsed(elements.hud.dataset.collapsed !== "true");
});

elements.joinWhiteButton.addEventListener("click", () => {
  joinRoom("white").catch((error) => {
    elements.message.textContent = error.message;
  });
});

elements.joinBlackButton.addEventListener("click", () => {
  joinRoom("black").catch((error) => {
    elements.message.textContent = error.message;
  });
});

elements.joinSpectatorButton.addEventListener("click", () => {
  joinRoom("spectator").catch((error) => {
    elements.message.textContent = error.message;
  });
});

elements.startGameButton.addEventListener("click", () => {
  startGame().catch((error) => {
    elements.message.textContent = error.message;
  });
});

for (const select of [elements.whitePlayerSelect, elements.blackPlayerSelect]) {
  select.addEventListener("change", () => {
    writeAssignments(readAssignments());
    render();
    syncLobby();
  });
}

loadWasmStatus();
loadServerStatus();
setHudCollapsed(localStorage.getItem("chronofish.hudCollapsed") === "true");
updateBotStatus();
render();
