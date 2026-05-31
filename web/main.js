import { elements } from "./dom.js";
import { capitalize, getBoard, hasUnplayedBoards, isLatestBoard, presentTime, samePosition } from "./board.js";
import { renderGame } from "./render.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import { appendNotationLine, postMatchLog } from "./match-log.js";
import { readWasmString, writeWasmBytes, writeWasmString } from "./engine-io.js";

let engine = null;
let aiParameters = null;
let aiEffortConfigs = {
  fast: {
    label: "Bot: Fast",
    displayNames: ["Bullet Fischer", "Speedrun Steinitz", "Blitz Botvinnik"],
    depth: 2,
    nodes: 20_000,
    timeMs: 1500
  },
  balanced: {
    label: "Bot: Balanced",
    displayNames: ["Multiverse Magnus", "Timeline Tal", "Causality Capablanca"],
    depth: 4,
    nodes: 80_000,
    timeMs: 5000
  },
  expert: {
    label: "Bot: Expert",
    displayNames: ["Kasparadox", "Deep Blue Shift", "Premovaru Checkamura"],
    depth: 5,
    nodes: 200_000,
    timeMs: 15000
  }
};
let game = { turn: "white", timelines: [], nextTimelineId: 1 };
// Last submitted snapshot. While a turn is staged, rendering compares against
// this so the present line and board status labels do not jump before Submit.
let committedGame = game;
let selected = null;
let legalTargets = [];
// submittedTurns is replayable room history; stagedMoves is local undo state for
// the current unsubmitted turn only.
let submittedTurns = [];
let submittedNotation = "";
let stagedMoves = [];
let aiWorker = null;
let aiRequestId = 0;
let trainingWorker = null;
let trainingRequestId = 0;
let trainingEnabled = false;
let trainingRunning = false;
let trainingCycle = 0;
let phase = "lobby";
let lastScrolledPresentTime = null;
let lastMatchAlertMessage = "";
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
  if (!engine || phase !== "game" || isBotAssignment(assignments[game.turn])) {
    return false;
  }

  if (!multiplayer.connected) {
    return assignments[game.turn] === "local";
  }

  return assignments[game.turn] === "human" && multiplayer.color === game.turn;
}

function canActNow() {
  return phase === "game" && canControlTurn();
}

function hasStagedMoves() {
  return stagedMoves.length > 0;
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
  if (value === "bot") {
    return "bot-balanced";
  }
  if (typeof value === "string" && value.startsWith("nn-bot-")) {
    return value.replace("nn-bot-", "bot-");
  }
  return [
    "local",
    "human",
    "open",
    "bot-fast",
    "bot-balanced",
    "bot-expert"
  ].includes(value) ? value : fallback;
}

function isBotAssignment(value) {
  return typeof value === "string" && value.startsWith("bot-");
}

function botEffortName(value) {
  return isBotAssignment(value) ? value.slice("bot-".length) : "balanced";
}

function botEffort(value) {
  return aiEffortConfigs[botEffortName(value)] ?? aiEffortConfigs.balanced;
}

function stableIndex(value, count) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash) % count;
}

function botDisplayName(color) {
  const effortName = botEffortName(assignments[color]);
  const effort = botEffort(assignments[color]);
  const names = effort.displayNames ?? [effort.label ?? "Bot"];
  return names[stableIndex(`${multiplayer.roomId}:${color}:${effortName}`, names.length)];
}

function playerDisplayName(color) {
  return isBotAssignment(assignments[color]) ? botDisplayName(color) : capitalize(color);
}

function displayGameMessage(message) {
  return String(message ?? "")
    .replace(/\bWhite\b/g, playerDisplayName("white"))
    .replace(/\bBlack\b/g, playerDisplayName("black"));
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
}

writeAssignments(assignments);

function gamePayload(nextPhase = phase) {
  return {
    phase: nextPhase,
    assignments,
    notation: submittedNotation,
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

function engineSnapshot() {
  return JSON.parse(readWasmString(engine, engine.chronofish_snapshot_json()));
}

function engineLastMessage() {
  return readWasmString(engine, engine.chronofish_last_message());
}

function engineDisplayMessage() {
  return displayGameMessage(engineLastMessage());
}

function stagedTurnNotation() {
  return readWasmString(engine, engine.chronofish_staged_turn_notation());
}

async function loadActiveModelIntoEngine() {
  if (!engine?.chronofish_set_neural_model_bytes) {
    return false;
  }
  try {
    const response = await fetch("/api/training/model");
    if (!response.ok) {
      engine.chronofish_clear_neural_model?.();
      return false;
    }
    const model = new Uint8Array(await response.arrayBuffer());
    const { ptr, len } = writeWasmBytes(engine, model);
    try {
      return Boolean(engine.chronofish_set_neural_model_bytes(ptr, len));
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
  } catch {
    engine.chronofish_clear_neural_model?.();
    return false;
  }
}

function isMatchOverMessage(message) {
  return /\bwins\b/i.test(message);
}

function resetEngine() {
  // Engine reset clears both visible and committed state plus all local history.
  engine.chronofish_reset();
  game = engineSnapshot();
  committedGame = game;
  selected = null;
  legalTargets = [];
  submittedTurns = [];
  submittedNotation = "";
  stagedMoves = [];
  lastMatchAlertMessage = "";
}

function replayTurns(turns) {
  // Multiplayer sync stores submitted turns. Replaying them through Rust rebuilds
  // authoritative engine state instead of trusting an arbitrary remote snapshot.
  engine.chronofish_reset();
  const notationLines = [];

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
    const turnNotation = stagedTurnNotation();
    engine.chronofish_submit_turn();
    if (turnNotation) {
      notationLines.push(`${notationLines.length + 1}. ${turnNotation}`);
    }
  }

  submittedTurns = turns.map((turn) => turn.map(cloneMove));
  submittedNotation = notationLines.join("\n");
  stagedMoves = [];
  game = engineSnapshot();
  committedGame = game;
  selected = null;
  legalTargets = [];
}

function replayNotation(notation) {
  const text = notation ?? "";
  const { ptr, len } = writeWasmString(engine, text);
  try {
    if (!engine.chronofish_load_notation(ptr, len)) {
      throw new Error(engineDisplayMessage());
    }
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }

  submittedNotation = text;
  submittedTurns = [];
  stagedMoves = [];
  game = engineSnapshot();
  committedGame = game;
  selected = null;
  legalTargets = [];
}

function appendSubmittedNotation(turnNotation, actor = game.turn) {
  if (!turnNotation) {
    return;
  }
  const previous = submittedNotation;
  submittedNotation = appendNotationLine({ submittedNotation, turnNotation });
  if (submittedNotation !== previous) {
    postMatchLog(multiplayer.roomId, submittedNotation.split(/\n/).at(-1) ?? "");
  }
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
  return ["white", "black"].filter((color) => isBotAssignment(assignments[color]));
}

function ensureAiWorker() {
  // Lazily create the worker so normal local play avoids WASM worker startup.
  if (!aiWorker) {
    aiWorker = new Worker("./ai-worker.js", { type: "module" });
    aiWorker.addEventListener("message", handleAiWorkerMessage);
  }
  return aiWorker;
}

function ensureTrainingWorker() {
  if (!trainingWorker) {
    trainingWorker = new Worker("./training-worker.js", { type: "module" });
    trainingWorker.addEventListener("message", handleTrainingWorkerMessage);
  }
  return trainingWorker;
}

function turnSignature() {
  // Used to ignore stale AI replies if the position changes while the worker is
  // thinking.
  return `${game.turn}:${submittedNotation}`;
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
  return JSON.parse(readWasmString(engine, engine.chronofish_legal_targets_json(
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
  const message = engineDisplayMessage();

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
        ? `You are ${multiplayer.color}; waiting for ${playerDisplayName(game.turn)}.`
        : `Waiting for ${playerDisplayName(game.turn)}.`;
    return;
  }

  const piece = pieceAt(position);
  const existingTarget = targetFor(position);

  // Click a highlighted target to move; click a latest own piece to select and
  // request legal targets from the engine.
  if (selected && existingTarget) {
    const moveMessage = applyEngineMove(selected, position);
    render({ preserveScroll: true });
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
    render({ preserveScroll: true });
    return;
  }

  selected = null;
  legalTargets = [];
  elements.message.textContent = `Select a ${game.turn} piece on a latest board.`;
  render({ preserveScroll: true });
}

function render(options = {}) {
  const scrollState = options.preserveScroll ? captureScrollState() : null;
  const previousPresentTime = lastScrolledPresentTime;
  const nextPresentTime = committedGame.timelines.length ? presentTime(committedGame) : null;
  const inGame = phase === "game";
  elements.startGameButton.disabled = !engine || inGame || multiplayer.color === "spectator";
  elements.joinWhiteButton.disabled = inGame;
  elements.joinBlackButton.disabled = inGame;
  elements.whitePlayerSelect.disabled = inGame || multiplayer.color === "spectator";
  elements.blackPlayerSelect.disabled = inGame || multiplayer.color === "spectator";
  elements.resetButton.disabled = !canActNow() || !hasStagedMoves();
  elements.undoMoveButton.disabled = !canActNow() || !hasStagedMoves();
  elements.submitTurnButton.disabled = !canActNow() || !hasStagedMoves() || hasUnplayedBoards(game);
  elements.concedeButton.disabled = !canActNow();
  renderTrainingButtons();

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
  renderEvaluationBar();

  if (scrollState) {
    restoreScrollState(scrollState);
  } else if (phase === "game" && nextPresentTime !== null && nextPresentTime !== previousPresentTime) {
    scrollMultiverseToPresent();
  }
  lastScrolledPresentTime = nextPresentTime;
}

function captureScrollState() {
  return {
    windowX: window.scrollX,
    windowY: window.scrollY,
    multiverseX: elements.multiverse?.scrollLeft ?? 0,
    multiverseY: elements.multiverse?.scrollTop ?? 0
  };
}

function restoreScrollState(state) {
  if (elements.multiverse) {
    elements.multiverse.scrollLeft = state.multiverseX;
    elements.multiverse.scrollTop = state.multiverseY;
  }
  window.scrollTo(state.windowX, state.windowY);
}

function renderEvaluationBar() {
  if (!elements.evaluationBar || !elements.evaluationWhite || !elements.evaluationScore) {
    return;
  }
  if (!engine?.chronofish_evaluation_json || phase !== "game") {
    elements.evaluationBar.hidden = true;
    return;
  }

  try {
    const evaluation = JSON.parse(readWasmString(engine, engine.chronofish_evaluation_json()));
    const score = Number(evaluation.score);
    if (!Number.isFinite(score)) {
      elements.evaluationBar.hidden = true;
      return;
    }
    const whiteShare = 0.5 + 0.5 * normalizedEvaluation(score);
    const whitePercent = Math.max(3, Math.min(97, whiteShare * 100));
    elements.evaluationWhite.style.height = `${whitePercent}%`;
    elements.evaluationScore.textContent = formatEvaluation(score);
    elements.evaluationBar.dataset.leader = score >= 0 ? "white" : "black";
    elements.evaluationBar.title = `White ${formatSignedPawns(score)}. Source: ${evaluation.source ?? "evaluation"}.`;
    elements.evaluationBar.hidden = false;
  } catch {
    elements.evaluationBar.hidden = true;
  }
}

function formatEvaluation(score) {
  if (Math.abs(score) >= 90000) {
    return score > 0 ? "M" : "-M";
  }
  return formatSignedPawns(score);
}

function normalizedEvaluation(score) {
  const maxCentipawns = 100000;
  const kneeCentipawns = 100;
  const magnitude = Math.min(Math.abs(score), maxCentipawns);
  const scaled = Math.log1p(magnitude / kneeCentipawns)
    / Math.log1p(maxCentipawns / kneeCentipawns);
  return Math.sign(score) * scaled;
}

function formatSignedPawns(score) {
  const pawns = score / 100;
  if (Math.abs(pawns) < 0.05) {
    return "0.0";
  }
  return `${pawns > 0 ? "+" : ""}${pawns.toFixed(1)}`;
}

function scrollMultiverseToPresent() {
  const marker = elements.timelineGrid.querySelector(".present-line");
  if (!marker) {
    return;
  }

  const markerBox = marker.getBoundingClientRect();
  const containerBox = elements.multiverse.getBoundingClientRect();
  elements.multiverse.scrollTo({
    left: elements.multiverse.scrollLeft
      + markerBox.left
      - containerBox.left
      - containerBox.width / 2
      + markerBox.width / 2,
    top: elements.multiverse.scrollTop
      + markerBox.top
      - containerBox.top
      - containerBox.height / 2
      + markerBox.height / 2,
    behavior: "smooth"
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
  }

  if (room?.game?.phase) {
    phase = room.game.phase;
  }
  if (room?.game?.assignments) {
    writeAssignments(room.game.assignments);
  }

  if (room?.game?.notation && engine) {
    replayNotation(room.game.notation);
  } else if (room?.game?.turns && engine) {
    replayTurns(room.game.turns);
  } else if (room?.game?.snapshot) {
    game = room.game.snapshot;
    committedGame = game;
    submittedTurns = room.game.turns ?? [];
    submittedNotation = room.game.notation ?? "";
    stagedMoves = [];
  } else if (room?.game?.timelines) {
    game = room.game;
    committedGame = game;
    submittedTurns = [];
    submittedNotation = "";
    stagedMoves = [];
  }

  selected = null;
  legalTargets = [];
  updateShareLink();
  render();

  if (message) {
    elements.message.textContent = message;
    showMatchDialog(message);
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
      game: gamePayload(),
      message
    });
  } catch (error) {
    elements.message.textContent = error.message;
  }
}

function showMatchDialog(message) {
  if (!isMatchOverMessage(message) || lastMatchAlertMessage === message) {
    return;
  }
  lastMatchAlertMessage = message;
  window.alert(message);
}

async function enterPostMatchReview(message, credentials = null) {
  phase = "review";
  selected = null;
  legalTargets = [];
  elements.message.textContent = message;
  render();
  showMatchDialog(message);
  postMatchLog(multiplayer.roomId, submittedNotation.split(/\n/).at(-1) ?? "");
  await syncState("state", message, credentials);
}

function victoryMessage(loser) {
  const winner = loser === "white" ? "black" : "white";
  return `${playerDisplayName(winner)} wins. ${playerDisplayName(loser)} conceded.`;
}

async function concede(color = game.turn, credentials = null) {
  const message = victoryMessage(color);
  await enterPostMatchReview(message, credentials ?? { color, token: multiplayer.token });
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
      throw new Error("Join a room before starting with Online humans.");
    }
    if (multiplayer.connected && nextAssignments[color] === "local") {
      throw new Error("Use Online or Bot for room games.");
    }
    if (
      multiplayer.connected &&
      nextAssignments[color] === "human" &&
      !currentRoom?.players?.[color]
    ) {
      throw new Error(`${capitalize(color)} is set to Online but no player is seated.`);
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
    !isBotAssignment(assignments[game.turn]) ||
    bot.thinking ||
    stagedMoves.length > 0
  ) {
    return;
  }

  const id = ++aiRequestId;
  const effort = botEffort(assignments[game.turn]);
  bot.thinking = true;
  elements.message.textContent = `${botDisplayName(game.turn)} thinking.`;
  ensureAiWorker().postMessage({
    id,
    notation: submittedNotation,
    depth: effort.depth,
    nodes: effort.nodes,
    timeMs: effort.timeMs
  });
}

function handleAiWorkerMessage(event) {
  const { id, ok, result, error } = event.data;
  if (id !== aiRequestId) {
    return;
  }

  bot.thinking = false;
  if (!ok) {
    elements.message.textContent = error;
    concede(game.turn, { color: game.turn, token: bot.tokens[game.turn] ?? botToken(game.turn) });
    return;
  }

  const botColor = game.turn;
  if (!isBotAssignment(assignments[botColor]) || stagedMoves.length > 0) {
    return;
  }

  if (result.status !== "ok" || result.moves.length === 0) {
    elements.message.textContent = `${botDisplayName(botColor)} found no legal turn and conceded.`;
    concede(botColor, { color: botColor, token: bot.tokens[botColor] ?? botToken(botColor) });
    return;
  }

  const before = turnSignature();
  for (const move of result.moves) {
    if (!applyEngineMove(move.from, move.to)) {
      concede(botColor, { color: botColor, token: bot.tokens[botColor] ?? botToken(botColor) });
      return;
    }
  }

  if (before !== turnSignature()) {
    return;
  }

  const turnNotation = stagedTurnNotation();
  if (!engine.chronofish_submit_turn()) {
    elements.message.textContent = engineDisplayMessage();
    concede(botColor, { color: botColor, token: bot.tokens[botColor] ?? botToken(botColor) });
    return;
  }

  const message = engineDisplayMessage();
  game = engineSnapshot();
  submittedTurns.push(stagedMoves.map(cloneMove));
  appendSubmittedNotation(turnNotation, botColor);
  stagedMoves = [];
  committedGame = game;
  selected = null;
  legalTargets = [];
  const botMessage = `${botDisplayName(botColor)} moved. ${message}`;
  elements.message.textContent = botMessage;
  render();
  if (isMatchOverMessage(message)) {
    enterPostMatchReview(message, { color: botColor, token: bot.tokens[botColor] ?? botToken(botColor) });
    return;
  }
  syncState("state", botMessage, { color: botColor, token: bot.tokens[botColor] ?? botToken(botColor) });
  maybeStartBotTurn();
}

async function loadWasmStatus() {
  try {
    const wasmPath = "./chronofish_engine.wasm";
    const instance = await instantiateChronofishWasm(wasmPath);
    engine = instance.exports;
    resetEngine();
    await loadActiveModelIntoEngine();
    elements.wasmStatus.textContent = `Engine v${readWasmString(engine, engine.chronofish_version())}`;
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
    const [versionResponse, parametersResponse, effortResponse] = await Promise.all([
      fetch("/api/version"),
      fetch("/ai/parameters.json"),
      fetch("/ai/effort.json")
    ]);
    const payload = await versionResponse.json();

    if (!versionResponse.ok) {
      throw new Error(payload.error ?? "Server unavailable");
    }
    if (parametersResponse.ok) {
      aiParameters = await parametersResponse.json();
    }
    if (effortResponse.ok) {
      aiEffortConfigs = await effortResponse.json();
    }

    elements.serverStatus.textContent = `Server v${payload.version}`;
    elements.serverStatus.dataset.state = "ready";
    await loadTrainingStatus();
  } catch (error) {
    console.error(error);
    elements.serverStatus.textContent = "Server unavailable";
    elements.serverStatus.dataset.state = "error";
  }
}

async function loadTrainingStatus() {
  try {
    const response = await fetch("/api/training/status");
    if (!response.ok) {
      throw new Error("Training endpoints disabled");
    }
    const payload = await response.json();
    trainingEnabled = payload.enabled === true;
    elements.trainingPanel.hidden = !trainingEnabled;
    elements.trainingStatus.textContent = payload.modelPresent
      ? `Active model ${payload.modelBytes ?? 0} bytes`
      : "No active model";
    renderTrainingButtons();
  } catch {
    trainingEnabled = false;
    elements.trainingPanel.hidden = true;
  }
}

function trainingConfig() {
  return {
    samples: clampNumber(elements.trainingSamplesInput.value, 1, 64, 4),
    depth: clampNumber(elements.trainingDepthInput.value, 1, 5, 5),
    nodes: clampNumber(elements.trainingNodesInput.value, 1, 200000, 200000),
    learningRate: clampNumber(elements.trainingRateInput.value, 0.0001, 0.1, 0.001),
    epochs: clampNumber(elements.trainingEpochsInput.value, 1, 5000, 1000),
    maxBuffer: clampNumber(elements.trainingBufferInput.value, 16, 2048, 256),
    labelWorkers: autoTrainingWorkers()
  };
}

function autoTrainingWorkers() {
  const cores = navigator.hardwareConcurrency ?? 4;
  return Math.max(1, Math.min(cores - 1, 16));
}

function clampNumber(value, min, max, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, number));
}

function renderTrainingButtons() {
  if (!elements.trainingPanel || elements.trainingPanel.hidden) {
    return;
  }
  elements.startTrainingButton.disabled = !engine || !trainingEnabled || trainingRunning;
  elements.stopTrainingButton.disabled = !trainingRunning;
}

async function startFrontendTraining() {
  if (!trainingEnabled || trainingRunning) {
    return;
  }
  if (!engine?.chronofish_neural_sample_json) {
    elements.trainingStatus.textContent = "Rebuild WASM for neural sampling.";
    return;
  }
  trainingRunning = true;
  trainingCycle = 0;
  renderTrainingButtons();
  runFrontendTrainingCycle();
}

function runFrontendTrainingCycle() {
  if (!trainingRunning) {
    return;
  }
  const config = trainingConfig();
  const cycle = trainingCycle + 1;
  elements.trainingStatus.textContent = `Run ${cycle}: collecting samples.`;
  try {
    const id = ++trainingRequestId;
    ensureTrainingWorker().postMessage({
      id,
      type: "train",
      notation: submittedNotation,
      config
    });
  } catch (error) {
    trainingRunning = false;
    elements.trainingStatus.textContent = error.message;
    renderTrainingButtons();
  }
}

function stopFrontendTraining() {
  if (!trainingRunning) {
    return;
  }
  trainingRequestId += 1;
  trainingWorker?.terminate();
  trainingWorker = null;
  trainingRunning = false;
  elements.trainingStatus.textContent = "Training stopped.";
  renderTrainingButtons();
}

async function replaceActiveModel(model) {
  const response = await fetch("/api/training/model", {
    method: "PUT",
    headers: { "content-type": "application/octet-stream" },
    body: model
  });
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(payload?.error ?? `Failed to replace model (${response.status})`);
  }
  await loadActiveModelIntoEngine();
  resetAiWorker();
  await loadTrainingStatus();
}

function resetAiWorker() {
  aiRequestId += 1;
  bot.thinking = false;
  aiWorker?.terminate();
  aiWorker = null;
}

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return { error: text };
  }
}

async function handleTrainingWorkerMessage(event) {
  const {
    id,
    ok,
    error,
    model,
    loss,
    epoch,
    collected,
    sampleCount,
    labelWorkers,
    bufferSize,
    pseudoCount,
    gpuPhase
  } = event.data;
  if (id !== trainingRequestId) {
    return;
  }
  if (!ok) {
    trainingRunning = false;
    elements.trainingStatus.textContent = error;
    renderTrainingButtons();
    return;
  }
  if (labelWorkers !== undefined) {
    elements.trainingStatus.textContent = `Run ${trainingCycle + 1}: labeling with ${labelWorkers} workers.`;
    return;
  }
  if (model) {
    trainingCycle += 1;
    elements.trainingStatus.textContent = `Run ${trainingCycle}: replacing model. Loss ${formatLoss(loss)}.`;
    try {
      await replaceActiveModel(model);
    } catch (replaceError) {
      trainingRunning = false;
      elements.trainingStatus.textContent = replaceError.message;
      renderTrainingButtons();
      return;
    }
    if (!trainingRunning) {
      renderTrainingButtons();
      return;
    }
    elements.trainingStatus.textContent = `Run ${trainingCycle}: ${model.nonZeroWeights ?? 0} weights changed. Restarting.`;
    setTimeout(runFrontendTrainingCycle, 0);
    return;
  }
  if (collected !== undefined) {
    elements.trainingStatus.textContent = `Run ${trainingCycle + 1}: collected ${collected}/${sampleCount}.`;
    return;
  }
  if (gpuPhase) {
    elements.trainingStatus.textContent = `Run ${trainingCycle + 1}: GPU training ${bufferSize} samples (${pseudoCount} pseudo).`;
    return;
  }
  if (epoch !== undefined) {
    elements.trainingStatus.textContent = `Run ${trainingCycle + 1}: epoch ${epoch}. Loss ${formatLoss(loss)}.`;
  }
}

function formatLoss(loss) {
  return Number.isFinite(loss) ? loss.toFixed(2) : "pending";
}

elements.resetButton.addEventListener("click", () => {
  if (!engine) {
    elements.message.textContent = "WASM engine is not loaded yet.";
    return;
  }
  if (!canActNow()) {
    elements.message.textContent = `Waiting for ${playerDisplayName(game.turn)}.`;
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
  if (!canActNow()) {
    elements.message.textContent = `Waiting for ${playerDisplayName(game.turn)}.`;
    return;
  }

  if (!engine.chronofish_undo_staged_move()) {
    elements.message.textContent = engineDisplayMessage();
    return;
  }

  stagedMoves.pop();
  game = engineSnapshot();
  if (stagedMoves.length === 0) {
    committedGame = game;
  }
  selected = null;
  legalTargets = [];
  elements.message.textContent = engineDisplayMessage();
  render();
});

elements.submitTurnButton.addEventListener("click", () => {
  if (!engine) {
    elements.message.textContent = "WASM engine is not loaded yet.";
    return;
  }
  if (!canActNow()) {
    elements.message.textContent = `Waiting for ${playerDisplayName(game.turn)}.`;
    return;
  }

  const actor = game.turn;
  const turnNotation = stagedTurnNotation();
  if (!engine.chronofish_submit_turn()) {
    elements.message.textContent = engineDisplayMessage();
    return;
  }

  const message = engineDisplayMessage();
  game = engineSnapshot();
  if (stagedMoves.length > 0) {
    submittedTurns.push(stagedMoves.map(cloneMove));
    appendSubmittedNotation(turnNotation, actor);
    stagedMoves = [];
  }
  committedGame = game;
  selected = null;
  legalTargets = [];
  elements.message.textContent = message;
  render();
  if (isMatchOverMessage(message)) {
    enterPostMatchReview(message);
    return;
  }
  syncState("state", message);
  maybeStartBotTurn();
});

elements.concedeButton.addEventListener("click", () => {
  if (!engine) {
    elements.message.textContent = "WASM engine is not loaded yet.";
    return;
  }
  if (!canActNow()) {
    elements.message.textContent = `Waiting for ${playerDisplayName(game.turn)}.`;
    return;
  }

  if (!window.confirm(`${capitalize(game.turn)} will concede. Continue?`)) {
    elements.message.textContent = "Concession cancelled.";
    return;
  }

  concede(game.turn);
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

elements.startTrainingButton.addEventListener("click", () => {
  startFrontendTraining();
});

elements.stopTrainingButton.addEventListener("click", () => {
  stopFrontendTraining();
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
render();
