import { elements } from "./dom.js";
import { capitalize, getBoard, getLatestBoard, isActiveTimeline, isLatestBoard, presentTime, samePosition } from "./board.js";
import { renderGame } from "./render.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import { appendNotationLine, postMatchLog, postBotLossLog } from "./match-log.js";
import { readWasmString, writeWasmString } from "./engine-io.js";

const LOCAL_GAME_STORAGE_KEY = "chronofish.localGameState.v1";
const GPU_MODE_STORAGE_KEY = "chronofish.gpuMode";
const initialSearchParams = new URLSearchParams(window.location.search);
const initialUrlHasRoom = initialSearchParams.has("room");

function initialGame() {
  const board = Array.from({ length: 8 }, () => Array(8).fill(null));
  const backRank = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"];
  for (let x = 0; x < 8; x += 1) {
    board[0][x] = { color: "white", type: backRank[x] };
    board[1][x] = { color: "white", type: "pawn" };
    board[6][x] = { color: "black", type: "pawn" };
    board[7][x] = { color: "black", type: backRank[x] };
  }

  return {
    turn: "white",
    nextTimelineId: 1,
    nextBlackTimelineId: -1,
    checkedRoyals: [],
    timelines: [{
      id: 0,
      row: 0,
      label: "Sacred T0",
      owner: "neutral",
      boards: [{
        time: 0,
        sideToMove: "white",
        castling: 15,
        enPassant: null,
        origin: null,
        board
      }]
    }]
  };
}

let engine = null;
let aiParameters = null;
let aiEffortConfigs = {
  fast: {
    label: "Bot: Fast",
    displayNames: ["Bullet Fischer", "Speedrun Steinitz", "Blitz Botvinnik"],
    depth: 2,
    nodes: 20_000,
    timeMs: 10_000
  },
  balanced: {
    label: "Bot: Balanced",
    displayNames: ["Multiverse Magnus", "Timeline Tal", "Causality Capablanca"],
    depth: 4,
    nodes: 80_000,
    timeMs: 50_000
  },
  expert: {
    label: "Bot: Expert",
    displayNames: ["Kasparadox", "Deep Blue Shift", "Premovaru Checkamura"],
    depth: 5,
    nodes: 200_000,
    timeMs: 250_000
  }
};
let game = initialGame();
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
let botDecisionLog = [];
let aiWorkers = [];
let aiRequestId = 0;
let legalTargetRequestId = 0;
let trainingWorker = null;
let trainingRequestId = 0;
let trainingEnabled = false;
let trainingRunning = false;
let trainingCycle = 0;
let trainingGpuProfile = null;
let trainingGpuProfileApplied = false;
let phase = "lobby";
let lastRenderedPhase = phase;
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
  timeoutId: null,
  countdownId: null,
  pendingSearch: null,
  tokens: {}
};
let multiplayer = {
  // Room id lives in the URL so sharing the address reconstructs the room.
  roomId: initialSearchParams.get("room") ?? makeRoomId(),
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

function shouldPersistLocalGame() {
  return !multiplayer.connected && (phase === "game" || phase === "review");
}

function persistLocalGameState() {
  if (!engine) {
    return;
  }
  if (!shouldPersistLocalGame()) {
    localStorage.removeItem(LOCAL_GAME_STORAGE_KEY);
    return;
  }

  localStorage.setItem(LOCAL_GAME_STORAGE_KEY, JSON.stringify({
    phase,
    assignments,
    notation: submittedNotation,
    turns: submittedTurns,
    snapshot: game,
    committedSnapshot: committedGame,
    stagedMoves,
    message: elements.message.textContent,
    savedAt: Date.now()
  }));
}

function clearLocalGameState() {
  localStorage.removeItem(LOCAL_GAME_STORAGE_KEY);
}

async function restoreLocalGameState() {
  if (!engine || multiplayer.connected || initialUrlHasRoom) {
    return false;
  }

  const saved = localStorage.getItem(LOCAL_GAME_STORAGE_KEY);
  if (!saved) {
    return false;
  }

  try {
    const state = JSON.parse(saved);
    if (!state || !["game", "review"].includes(state.phase)) {
      clearLocalGameState();
      return false;
    }

    writeAssignments(state.assignments ?? assignments);
    phase = state.phase;
    const restoredSnapshot = Boolean(state.snapshot?.timelines);
    if (state.snapshot?.timelines) {
      game = state.snapshot;
      committedGame = state.committedSnapshot?.timelines ? state.committedSnapshot : state.snapshot;
      submittedNotation = state.notation ?? "";
      submittedTurns = Array.isArray(state.turns) ? state.turns.map((turn) => turn.map(cloneMove)) : [];
      stagedMoves = (state.stagedMoves ?? []).map(cloneMove);
    } else {
      clearLocalGameState();
      resetEngine();
    }

    if (!restoredSnapshot) {
      committedGame = game;
      for (const move of state.stagedMoves ?? []) {
        if (!move?.from || !move?.to || !(await applyEngineMove(move.from, move.to))) {
          break;
        }
      }
    }
    selected = null;
    legalTargets = [];
    elements.message.textContent = state.message || (phase === "game"
      ? `${playerDisplayName(game.turn)} to move.`
      : "Reviewing completed game.");
    return true;
  } catch (error) {
    console.error(error);
    clearLocalGameState();
    resetEngine();
    return false;
  }
}

function lobbyPayload() {
  return {
    phase: "lobby",
    assignments,
    snapshot: committedGame
  };
}

function clientTurnNotation(moves = stagedMoves) {
  return moves
    .map((move) => `${positionNotation(move.from)}-${positionNotation(move.to)}`)
    .join(" ");
}

function positionNotation(position) {
  return `T${position.time}L${position.timelineId}${String.fromCharCode(97 + position.x)}${position.y + 1}`;
}

function isMatchOverMessage(message) {
  return /\bwins\b/i.test(message);
}

function resetEngine() {
  // Engine reset clears both visible and committed state plus all local history.
  engine?.chronofish_reset?.();
  game = initialGame();
  committedGame = game;
  selected = null;
  legalTargets = [];
  submittedTurns = [];
  submittedNotation = "";
  stagedMoves = [];
  botDecisionLog = [];
  lastMatchAlertMessage = "";
}

function engineSnapshot() {
  return JSON.parse(readWasmString(engine, engine.chronofish_snapshot_json()));
}

function engineLastMessage() {
  return readWasmString(engine, engine.chronofish_last_message());
}

function syncEngineSnapshot(snapshot = game) {
  if (!engine?.chronofish_load_snapshot_json) {
    throw new Error("CPU rule engine is unavailable in this WASM build.");
  }
  const { ptr, len } = writeWasmString(engine, JSON.stringify(snapshot));
  try {
    if (!engine.chronofish_load_snapshot_json(ptr, len)) {
      throw new Error(engineLastMessage());
    }
  } finally {
    engine.chronofish_dealloc(ptr, len);
  }
}

function syncEngineToStagedState() {
  syncEngineSnapshot(committedGame);
  for (const move of stagedMoves) {
    const ok = engine.chronofish_apply_move(
      move.from.timelineId,
      move.from.time,
      move.from.x,
      move.from.y,
      move.to.timelineId,
      move.to.time,
      move.to.x,
      move.to.y
    );
    if (!ok) {
      throw new Error(engineLastMessage());
    }
  }
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

function cloneGame(snapshot) {
  return JSON.parse(JSON.stringify(snapshot));
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

function createAiWorker() {
  const worker = new Worker("./ai-worker.js", { type: "module" });
  worker.addEventListener("message", handleAiWorkerMessage);
  aiWorkers.push(worker);
  return worker;
}

function terminateAiWorkers() {
  for (const worker of aiWorkers) {
    worker.terminate();
  }
  aiWorkers = [];
  bot.pendingSearch = null;
}

function botSearchWorkerCount(effortName) {
  if (effortName !== "expert") {
    return 1;
  }
  const hardwareThreads = Math.max(1, navigator.hardwareConcurrency ?? 2);
  return Math.max(1, Math.min(2, hardwareThreads - 1));
}

function clearBotTimeout() {
  if (bot.timeoutId !== null) {
    clearTimeout(bot.timeoutId);
    bot.timeoutId = null;
  }
  if (bot.countdownId !== null) {
    clearInterval(bot.countdownId);
    bot.countdownId = null;
  }
}

function formatBotTimeLimit(ms) {
  const seconds = Math.max(0, Math.ceil(ms / 1000));
  return `${seconds}s`;
}

function botMoveCredentials(color) {
  return { color, token: bot.tokens[color] ?? botToken(color) };
}

function updateBotCountdownMessage(id) {
  const pending = bot.pendingSearch;
  if (!bot.thinking || !pending || pending.id !== id) {
    return;
  }
  const remainingMs = Math.max(0, pending.deadlineAt - Date.now());
  const workerText = `${pending.expected} worker${pending.expected === 1 ? "" : "s"}`;
  elements.message.textContent = `${botDisplayName(pending.botColor)} thinking across ${workerText}. ${formatBotTimeLimit(remainingMs)} left.`;
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

function targetFor(position) {
  return legalTargets.find((target) => samePosition(target, position));
}

function pieceAt(position) {
  return getBoard(game, position.timelineId, position.time)?.board[position.y]?.[position.x] ?? null;
}

function legalTargetsFor(position) {
  if (!engine) {
    return Promise.resolve({ source: null, targets: [] });
  }
  syncEngineSnapshot(game);
  const piece = pieceAt(position);
  const board = getBoard(game, position.timelineId, position.time);
  const selectable = piece?.color === game.turn
    && board?.sideToMove === game.turn
    && isLatestBoard(game, position.timelineId, position.time);
  return Promise.resolve({
    source: selectable ? { ...position, piece } : null,
    targets: JSON.parse(readWasmString(engine, engine.chronofish_legal_targets_json(
      position.timelineId,
      position.time,
      position.x,
      position.y
    )))
  });
}

async function applyEngineMove(from, to) {
  let ok;
  try {
    syncEngineSnapshot(game);
    ok = engine.chronofish_apply_move(
      from.timelineId,
      from.time,
      from.x,
      from.y,
      to.timelineId,
      to.time,
      to.x,
      to.y
    );
  } catch (error) {
    elements.message.textContent = error.message;
    return null;
  }
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
  persistLocalGameState();
  return elements.message.textContent;
}

async function submitVisibleTurn(actor) {
  if (stagedMoves.length === 0) {
    elements.message.textContent = "Make at least one move before submitting.";
    return null;
  }

  let ok;
  try {
    syncEngineToStagedState();
    ok = engine.chronofish_submit_turn();
  } catch (error) {
    elements.message.textContent = error.message;
    return null;
  }
  const engineMessage = engineLastMessage();
  if (!ok) {
    elements.message.textContent = engineMessage;
    return null;
  }

  const turnNotation = clientTurnNotation();
  const submitted = stagedMoves.map(cloneMove);
  game = engineSnapshot();
  committedGame = game;
  submittedTurns.push(submitted);
  appendSubmittedNotation(turnNotation, actor);
  stagedMoves = [];
  selected = null;
  legalTargets = [];

  const message = engineMessage || `${capitalize(game.turn)} to move.`;
  elements.message.textContent = message;
  persistLocalGameState();
  return message;
}

function resetStagedClientState() {
  const committed = committedGame;
  game = committed;
  committedGame = committed;
  stagedMoves = [];
  selected = null;
  legalTargets = [];
}

async function rebuildStagedClientState(moves) {
  resetStagedClientState();
  for (const move of moves) {
    if (!(await applyEngineMove(move.from, move.to))) {
      break;
    }
  }
}

async function handleSquareClick(position) {
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

  const existingTarget = targetFor(position);

  // Click a highlighted target to move; otherwise ask the GPU worker whether the
  // clicked square is a selectable source and what it can legally target.
  if (selected && existingTarget) {
    const moveMessage = await applyEngineMove(selected, position);
    render({ preserveScroll: true });
    return;
  }

  selected = position;
  legalTargets = [];
  const requestId = ++legalTargetRequestId;
  elements.message.textContent = "Checking legal moves.";
  render({ preserveScroll: true });
  try {
    const selection = await legalTargetsFor(position);
    if (requestId !== legalTargetRequestId || !selected || !samePosition(selected, position)) {
      return;
    }
    if (!selection.source) {
      legalTargetRequestId += 1;
      selected = null;
      legalTargets = [];
      elements.message.textContent = `Select a ${game.turn} piece on a playable board.`;
      render({ preserveScroll: true });
      return;
    }
    const piece = selection.source.piece;
    legalTargets = selection.targets ?? [];
    elements.message.textContent = `${capitalize(piece.color)} ${piece.type} selected. ${legalTargets.length} legal target${legalTargets.length === 1 ? "" : "s"}.`;
    render({ preserveScroll: true });
  } catch (error) {
    if (requestId !== legalTargetRequestId || !selected || !samePosition(selected, position)) {
      return;
    }
    selected = null;
    legalTargets = [];
    elements.message.textContent = error.message;
    render({ preserveScroll: true });
    return;
  }
}

function render(options = {}) {
  const scrollState = options.preserveScroll ? captureScrollState() : null;
  const previousPresentTime = lastScrolledPresentTime;
  const nextPresentTime = committedGame.timelines.length ? presentTime(committedGame) : null;
  const enteredGame = lastRenderedPhase !== "game" && phase === "game";
  const inGame = phase === "game";
  elements.startGameButton.disabled = !engine || inGame || multiplayer.color === "spectator";
  elements.joinWhiteButton.disabled = inGame;
  elements.joinBlackButton.disabled = inGame;
  elements.whitePlayerSelect.disabled = inGame || multiplayer.color === "spectator";
  elements.blackPlayerSelect.disabled = inGame || multiplayer.color === "spectator";
  elements.resetButton.disabled = !canActNow() || !hasStagedMoves();
  elements.undoMoveButton.disabled = !canActNow() || !hasStagedMoves();
  elements.submitTurnButton.disabled = !canActNow() || !hasStagedMoves();
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
  } else if (phase === "game" && nextPresentTime !== null && (enteredGame || nextPresentTime !== previousPresentTime)) {
    scrollMultiverseToPresent();
  }
  if (phase === "game") {
    lastScrolledPresentTime = nextPresentTime;
  } else {
    lastScrolledPresentTime = null;
  }
  lastRenderedPhase = phase;
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
  const evaluation = evaluateClientPosition(game);
  if (!evaluation) {
    elements.evaluationBar.hidden = true;
    return;
  }

  const { score, source } = evaluation;
  const whiteShare = 0.5 + 0.5 * normalizedEvaluation(score);
  const whitePercent = Math.max(3, Math.min(97, whiteShare * 100));
  elements.evaluationWhite.style.height = `${whitePercent}%`;
  elements.evaluationScore.textContent = formatEvaluation(score);
  elements.evaluationBar.dataset.leader = score >= 0 ? "white" : "black";
  elements.evaluationBar.title = `White ${formatSignedPawns(score)}. Source: ${source}.`;
  elements.evaluationBar.hidden = false;
}

function evaluateClientPosition(position) {
  if (!position?.timelines?.length) {
    return null;
  }
  let score = 0;
  let boardCount = 0;
  for (const timeline of position.timelines) {
    if (!isActiveTimeline(position, timeline)) {
      continue;
    }
    const board = getLatestBoard(position, timeline.id);
    if (!board?.board) {
      continue;
    }
    boardCount += 1;
    score += evaluateBoardMaterial(board);
    score += board.sideToMove === "white" ? 12 : -12;
  }
  if (boardCount === 0) {
    return null;
  }
  return {
    score: Math.round(score / boardCount),
    source: "client material"
  };
}

function evaluateBoardMaterial(board) {
  let score = 0;
  for (const row of board.board) {
    for (const piece of row) {
      if (!piece) {
        continue;
      }
      const value = pieceValue(piece.type);
      score += piece.color === "white" ? value : -value;
    }
  }
  return score;
}

function pieceValue(type) {
  return {
    king: 20000,
    commonKing: 10000,
    queen: 900,
    royalQueen: 20000,
    princess: 700,
    rook: 500,
    bishop: 330,
    unicorn: 500,
    dragon: 900,
    knight: 320,
    pawn: 100,
    brawn: 130
  }[type] ?? 0;
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
  const marker = elements.timelineGrid.querySelector('.timeline-row[data-active="true"] .present-line')
    ?? elements.timelineGrid.querySelector(".present-line");
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
    clearBotTimeout();
    terminateAiWorkers();
  }

  if (room?.game?.phase) {
    phase = room.game.phase;
  }
  if (room?.game?.assignments) {
    writeAssignments(room.game.assignments);
  }

  if (room?.game?.snapshot) {
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
  persistLocalGameState();
  render();
  showMatchDialog(message);
  postMatchLog(multiplayer.roomId, submittedNotation.split(/\n/).at(-1) ?? "");
  recordBotLossLog(message);
  await syncState("state", message, credentials);
}

function recordBotLossLog(message) {
  const winner = winnerFromMatchMessage(message);
  if (!winner) {
    return;
  }
  const loser = winner === "white" ? "black" : "white";
  if (!isBotAssignment(assignments[loser]) || isBotAssignment(assignments[winner])) {
    return;
  }
  const decisions = botDecisionLog.filter((decision) => decision.botColor === loser);
  if (!decisions.length) {
    return;
  }
  postBotLossLog(multiplayer.roomId, {
    roomId: multiplayer.roomId,
    recordedAt: Date.now(),
    winner,
    loser,
    reason: message,
    assignments,
    notation: submittedNotation,
    finalGame: game,
    decisions
  });
}

function winnerFromMatchMessage(message) {
  const normalized = String(message ?? "").toLowerCase();
  if (/\bwhite\b.*\bwins\b/.test(normalized)) {
    return "white";
  }
  if (/\bblack\b.*\bwins\b/.test(normalized)) {
    return "black";
  }
  return null;
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
    persistLocalGameState();
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
  const botColor = game.turn;
  const effortName = botEffortName(assignments[botColor]);
  const effort = botEffort(assignments[botColor]);
  const timeMs = Math.max(1, effort.timeMs ?? 10_000);
  const workerTimeMs = botWorkerSearchTimeMs(timeMs);
  const workerCount = botSearchWorkerCount(effortName);
  terminateAiWorkers();
  bot.thinking = true;
  bot.pendingSearch = {
    id,
    botColor,
    game: cloneGame(game),
    expected: 0,
    deadlineAt: Date.now() + timeMs,
    results: [],
    errors: []
  };
  clearBotTimeout();
  bot.timeoutId = setTimeout(() => handleBotTimeout(id, botColor, timeMs), timeMs);
  bot.countdownId = setInterval(() => updateBotCountdownMessage(id), 250);
  for (let partitionIndex = 0; partitionIndex < workerCount; partitionIndex += 1) {
    try {
      createAiWorker().postMessage({
        id,
        game,
        depth: effort.depth,
        nodes: effort.nodes,
        timeMs: workerTimeMs,
        gpuMode: botGpuMode(),
        partitionIndex,
        partitionCount: workerCount
      });
      bot.pendingSearch.expected += 1;
    } catch (error) {
      console.error(error);
      bot.pendingSearch.errors.push(error.message);
    }
  }
  if (bot.pendingSearch.expected === 0) {
    bot.pendingSearch.expected = 1;
    setTimeout(() => {
      handleAiWorkerMessage({
        data: {
          id,
          ok: false,
          error: "GPU worker search is unavailable.",
          partitionIndex: 0
        }
      });
    }, 0);
  }
  updateBotCountdownMessage(id);
}

function botWorkerSearchTimeMs(timeMs) {
  const margin = Math.min(1000, Math.max(100, Math.floor(timeMs * 0.05)));
  return Math.max(1, timeMs - margin);
}

function botGpuMode() {
  return localStorage.getItem(GPU_MODE_STORAGE_KEY) === "full" ? "full" : "hybrid";
}

function handleBotTimeout(id, botColor, timeMs) {
  if (id !== aiRequestId || !bot.thinking) {
    return;
  }

  const pending = bot.pendingSearch;
  const bestResult = selectBestAiResult(pending?.results.map((entry) => entry.result) ?? []);
  aiRequestId += 1;
  bot.thinking = false;
  clearBotTimeout();
  if (bestResult) {
    logBotSearchChoices(pending, bestResult, "timeout");
    bestResult.trainingDecision = buildBotDecisionRecord(pending, bestResult);
    terminateAiWorkers();
    elements.message.textContent = `${botDisplayName(botColor)} used the best move found in ${formatBotTimeLimit(timeMs)}.`;
    void completeBotTurn(botColor, bestResult);
    return;
  }

  terminateAiWorkers();
  elements.message.textContent = `${botDisplayName(botColor)} found no legal turn in ${formatBotTimeLimit(timeMs)}.`;
  void completeBotTurn(botColor, { status: "noLegalTurn", moves: [] });
}

function handleAiWorkerMessage(event) {
  const { id, ok, result, error, partitionIndex } = event.data;
  if (id !== aiRequestId) {
    return;
  }

  const pending = bot.pendingSearch;
  if (!pending || pending.id !== id) {
    return;
  }

  if (ok) {
    pending.results.push({ result, partitionIndex });
  } else {
    pending.errors.push(error);
  }

  if (pending.results.length + pending.errors.length < pending.expected) {
    return;
  }

  bot.thinking = false;
  clearBotTimeout();

  const bestResult = selectBestAiResult(pending.results.map((entry) => entry.result));
  logBotSearchChoices(pending, bestResult, "complete");
  if (!bestResult && pending.errors.length > 0) {
    terminateAiWorkers();
    elements.message.textContent = `${botDisplayName(pending.botColor)} search failed: ${pending.errors[0]}`;
    render();
    return;
  }

  if (bestResult) {
    bestResult.trainingDecision = buildBotDecisionRecord(pending, bestResult);
  }
  terminateAiWorkers();
  void completeBotTurn(pending.botColor, bestResult ?? { status: "noLegalTurn", moves: [] });
}

function selectBestAiResult(results) {
  return results
    .filter((result) => result?.status === "ok" && result.moves?.length > 0)
    .sort((left, right) => {
      const score = (right.score ?? -Infinity) - (left.score ?? -Infinity);
      if (score !== 0) {
        return score;
      }
      const depth = (right.depth ?? 0) - (left.depth ?? 0);
      if (depth !== 0) {
        return depth;
      }
      return (right.nodes ?? 0) - (left.nodes ?? 0);
    })[0] ?? null;
}

function logBotSearchChoices(pending, selectedResult, reason) {
  if (!pending) {
    return;
  }
  const choices = rankedBotChoices(pending.results, selectedResult);
  const botName = botDisplayName(pending.botColor);
  if (!choices.length) {
    console.info(`${botName} search ${reason}: no legal move choices`, {
      errors: pending.errors
    });
    return;
  }
  console.groupCollapsed(`${botName} search ${reason}: ${choices.length} move choice${choices.length === 1 ? "" : "s"}`);
  console.table(choices.map((choice, index) => ({
    rank: index + 1,
    selected: choice.selected ? "yes" : "",
    eval: formatBotEvaluation(choice.score),
    moves: choice.moves.map(formatBotMove).join(" | "),
    depth: choice.depth ?? "",
    nodes: choice.nodes ?? "",
    worker: choice.partitionIndex ?? "",
    search: choice.gpuSearch ?? ""
  })));
  if (pending.errors.length) {
    console.info("Bot search worker errors", pending.errors);
  }
  console.groupEnd();
}

function rankedBotChoices(results, selectedResult) {
  const selectedKey = botMovesKey(selectedResult?.moves ?? []);
  const byMoves = new Map();
  for (const entry of results) {
    const result = entry.result;
    const rawChoices = Array.isArray(result?.choices) && result.choices.length
      ? result.choices
      : result?.moves?.length
        ? [{ moves: result.moves, score: result.score, depth: result.depth, nodes: result.nodes, gpuSearch: result.gpuSearch }]
        : [];
    for (const choice of rawChoices) {
      const moves = choice.moves ?? [];
      const key = botMovesKey(moves);
      if (!key) {
        continue;
      }
      const current = byMoves.get(key);
      const next = {
        moves,
        score: choice.score,
        depth: choice.depth ?? result?.depth,
        nodes: choice.nodes ?? result?.nodes,
        gpuSearch: choice.gpuSearch ?? result?.gpuSearch,
        partitionIndex: entry.partitionIndex,
        selected: key === selectedKey
      };
      if (!current || (next.score ?? -Infinity) > (current.score ?? -Infinity)) {
        byMoves.set(key, next);
      } else if (key === selectedKey) {
        current.selected = true;
      }
    }
  }
  return Array.from(byMoves.values())
    .sort((left, right) => {
      const score = botChoiceScore(right) - botChoiceScore(left);
      if (score !== 0) {
        return score;
      }
      return botMovesKey(left.moves).localeCompare(botMovesKey(right.moves));
    })
    .slice(0, 16);
}

function buildBotDecisionRecord(pending, result) {
  if (!pending || result?.status !== "ok" || !result.moves?.length) {
    return null;
  }
  return {
    ply: submittedTurns.length + 1,
    botColor: pending.botColor,
    effort: botEffortName(assignments[pending.botColor]),
    game: cloneGame(pending.game ?? game),
    selectedMoves: result.moves.map(cloneMove),
    selectedScore: result.score ?? null,
    selectedDepth: result.depth ?? null,
    selectedNodes: result.nodes ?? null,
    choices: rankedBotChoices(pending.results, result).map((choice) => ({
      moves: choice.moves.map(cloneMove),
      score: choice.score ?? null,
      depth: choice.depth ?? null,
      nodes: choice.nodes ?? null,
      gpuSearch: choice.gpuSearch ?? null
    }))
  };
}

function recordBotDecision(result) {
  if (result?.trainingDecision) {
    botDecisionLog.push(result.trainingDecision);
  }
}

function botChoiceScore(choice) {
  return Number.isFinite(choice?.score) ? choice.score : -Infinity;
}

function botMovesKey(moves) {
  return (moves ?? []).map(formatBotMove).join("|");
}

function formatBotMove(move) {
  if (!move?.from || !move?.to) {
    return "?";
  }
  return `${formatBotSquare(move.from)} -> ${formatBotSquare(move.to)}`;
}

function formatBotSquare(square) {
  const file = typeof square.x === "number" ? String.fromCharCode(97 + square.x) : "?";
  const rank = typeof square.y === "number" ? String(8 - square.y) : "?";
  return `T${square.timelineId ?? "?"}@${square.time ?? "?"}:${file}${rank}`;
}

function formatBotEvaluation(score) {
  if (!Number.isFinite(score)) {
    return "?";
  }
  return `${score > 0 ? "+" : ""}${(score / 100).toFixed(2)}`;
}

async function completeBotTurn(botColor, result) {
  if (!isBotAssignment(assignments[botColor]) || stagedMoves.length > 0) {
    return;
  }

  if (result.status !== "ok" || result.moves.length === 0) {
    elements.message.textContent = `${botDisplayName(botColor)} found no legal turn and conceded.`;
    concede(botColor, botMoveCredentials(botColor));
    return;
  }

  recordBotDecision(result);
  const before = turnSignature();
  for (const move of result.moves) {
    if (!(await applyEngineMove(move.from, move.to))) {
      concede(botColor, botMoveCredentials(botColor));
      return;
    }
  }

  if (before !== turnSignature()) {
    return;
  }

  const message = await submitVisibleTurn(botColor);
  if (!message) {
    concede(botColor, botMoveCredentials(botColor));
    return;
  }

  const botMessage = `${botDisplayName(botColor)} moved. ${message}`;
  elements.message.textContent = botMessage;
  persistLocalGameState();
  render();
  if (isMatchOverMessage(message)) {
    enterPostMatchReview(message, botMoveCredentials(botColor));
    return;
  }
  syncState("state", botMessage, botMoveCredentials(botColor));
  maybeStartBotTurn();
}

async function loadWasmStatus() {
  try {
    const wasmPath = "./chronofish_engine.wasm";
    const instance = await instantiateChronofishWasm(wasmPath);
    engine = instance.exports;
    resetEngine();
    const restored = await restoreLocalGameState();
    elements.wasmStatus.textContent = `Engine v${readWasmString(engine, engine.chronofish_version())}`;
    elements.wasmStatus.dataset.state = "ready";
    if (!restored) {
      elements.message.textContent = "Configure the lobby, then start the game.";
    }
    render();
    if (restored) {
      maybeStartBotTurn();
    }
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
    if (payload?.enabled !== true || payload.modelPath !== "engine/models/value-v1/value-model.cfnn") {
      throw new Error("Training endpoints disabled");
    }
    trainingEnabled = true;
    elements.openTrainingButton.hidden = false;
    elements.trainingPanel.hidden = false;
    await applyTrainingGpuProfile();
    const modelHash = payload.modelHash ? `, hash ${payload.modelHash}` : "";
    const modelPath = payload.resolvedModelPath ? ` (${payload.resolvedModelPath})` : "";
    const gpuText = trainingGpuProfile ? `, GPU ${trainingGpuProfile.name}` : "";
    elements.trainingStatus.textContent = payload.modelPresent
      ? `Active model ${payload.modelBytes ?? 0} bytes${modelHash}${gpuText}${modelPath}`
      : `No active model${gpuText}`;
    renderTrainingButtons();
  } catch {
    trainingEnabled = false;
    elements.openTrainingButton.hidden = true;
    elements.trainingModal.hidden = true;
    elements.trainingPanel.hidden = true;
  }
}

function trainingConfig() {
  const caps = trainingGpuProfile?.config ?? {};
  return {
    labelMode: elements.trainingLabelModeSelect.value,
    samples: clampNumber(elements.trainingSamplesInput.value, 1, caps.maxSamples ?? 512, caps.samples ?? 64),
    selfPlayWorkers: clampNumber(elements.trainingSelfPlayWorkersInput.value, 1, caps.maxSelfPlayWorkers ?? 8, caps.selfPlayWorkers ?? 2),
    searchWorkers: clampNumber(elements.trainingSearchWorkersInput.value, 1, caps.maxSearchWorkers ?? 16, caps.searchWorkers ?? 2),
    explorationTemperature: clampNumber(elements.trainingTemperatureInput.value, 0, 2, 0.25),
    depth: clampNumber(elements.trainingDepthInput.value, 1, 8, 5),
    nodes: clampNumber(elements.trainingNodesInput.value, 1, caps.maxNodes ?? 65536, caps.nodes ?? 16384),
    learningRate: clampNumber(elements.trainingRateInput.value, 0.0001, 0.1, 0.01),
    epochs: clampNumber(elements.trainingEpochsInput.value, 1, caps.maxEpochs ?? 65536, caps.epochs ?? 8192),
    maxBuffer: clampNumber(elements.trainingBufferInput.value, 16, caps.maxBuffer ?? 8192, caps.buffer ?? 4096),
    batchSize: clampNumber(elements.trainingBatchInput.value, 16, caps.maxBatch ?? 4096, caps.batch ?? 1024),
    validationSplit: clampNumber(elements.trainingValidationInput.value, 0, 0.3, 0.1),
    lossLogReplay: clampNumber(elements.trainingLossLogReplayInput.value, 0, 32, 4),
    validationInterval: clampNumber(elements.trainingValidationIntervalInput.value, 16, caps.maxValidationInterval ?? 4096, caps.validationInterval ?? 256),
    patience: clampNumber(elements.trainingPatienceInput.value, 1, 64, 12),
    weightDecay: clampNumber(elements.trainingDecayInput.value, 0, 0.01, 0.00001),
    labelWorkers: autoTrainingWorkers()
  };
}

function autoTrainingWorkers() {
  const cores = navigator.hardwareConcurrency ?? 4;
  return Math.max(1, Math.min(cores - 1, 4));
}

async function applyTrainingGpuProfile() {
  if (trainingGpuProfileApplied) {
    return;
  }
  trainingGpuProfileApplied = true;
  trainingGpuProfile = await detectTrainingGpuProfile();
  if (!trainingGpuProfile) {
    return;
  }
  const config = trainingGpuProfile.config;
  applyTrainingInputProfile(elements.trainingSamplesInput, config.samples, config.maxSamples);
  applyTrainingInputProfile(elements.trainingSelfPlayWorkersInput, config.selfPlayWorkers, config.maxSelfPlayWorkers);
  applyTrainingInputProfile(elements.trainingSearchWorkersInput, config.searchWorkers, config.maxSearchWorkers);
  applyTrainingInputProfile(elements.trainingNodesInput, config.nodes, config.maxNodes);
  applyTrainingInputProfile(elements.trainingEpochsInput, config.epochs, config.maxEpochs);
  applyTrainingInputProfile(elements.trainingBufferInput, config.buffer, config.maxBuffer);
  applyTrainingInputProfile(elements.trainingBatchInput, config.batch, config.maxBatch);
  applyTrainingInputProfile(elements.trainingValidationIntervalInput, config.validationInterval, config.maxValidationInterval);
}

async function detectTrainingGpuProfile() {
  if (!navigator.gpu?.requestAdapter) {
    return null;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return null;
    }
    const limits = adapter.limits ?? {};
    const maxStorageBinding = limits.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    const maxBufferSize = limits.maxBufferSize ?? maxStorageBinding;
    const hardwareThreads = navigator.hardwareConcurrency ?? 4;
    const maxProjectedReplay = clampPowerOfTwo(Math.floor(maxStorageBinding / (2048 * Float32Array.BYTES_PER_ELEMENT)), 512, 16384);
    const maxBatchByActivation = clampPowerOfTwo(Math.floor(maxStorageBinding / (1024 * Float32Array.BYTES_PER_ELEMENT)), 512, 8192);
    const highMemory = maxStorageBinding >= 512 * 1024 * 1024 || maxBufferSize >= 1024 * 1024 * 1024;
    const mediumMemory = maxStorageBinding >= 256 * 1024 * 1024 || maxBufferSize >= 512 * 1024 * 1024;
    const maxWorkerBudget = Math.max(1, Math.min(highMemory ? 16 : 8, hardwareThreads - 1));
    const config = {
      maxSamples: highMemory ? 1024 : 512,
      samples: highMemory ? 256 : mediumMemory ? 128 : 64,
      maxSelfPlayWorkers: maxWorkerBudget,
      selfPlayWorkers: Math.max(1, Math.min(maxWorkerBudget, highMemory ? 4 : mediumMemory ? 3 : 2)),
      maxSearchWorkers: maxWorkerBudget,
      searchWorkers: Math.max(1, Math.min(maxWorkerBudget, highMemory ? 8 : mediumMemory ? 4 : 2)),
      maxNodes: highMemory ? 131072 : 65536,
      nodes: highMemory ? 32768 : 16384,
      maxEpochs: 65536,
      epochs: highMemory ? 16384 : 8192,
      maxBuffer: Math.min(maxProjectedReplay, highMemory ? 16384 : 8192),
      buffer: Math.min(maxProjectedReplay, highMemory ? 8192 : mediumMemory ? 6144 : 4096),
      maxBatch: Math.min(maxBatchByActivation, highMemory ? 8192 : 4096),
      batch: Math.min(maxBatchByActivation, highMemory ? 4096 : mediumMemory ? 2048 : 1024),
      maxValidationInterval: 4096,
      validationInterval: highMemory ? 256 : 128
    };
    return {
      name: adapter.info?.description || adapter.info?.device || adapter.info?.vendor || "WebGPU",
      limits,
      config
    };
  } catch (error) {
    console.warn("GPU capability detection failed", error);
    return null;
  }
}

function applyTrainingInputProfile(input, value, max) {
  if (!input) {
    return;
  }
  input.max = String(max);
  input.value = String(value);
}

function clampPowerOfTwo(value, min, max) {
  const clamped = Math.max(min, Math.min(max, value || min));
  return 2 ** Math.floor(Math.log2(clamped));
}

function clampNumber(value, min, max, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, number));
}

function renderTrainingButtons() {
  if (!elements.trainingPanel || !trainingEnabled) {
    return;
  }
  elements.startTrainingButton.disabled = !engine || !trainingEnabled || trainingRunning;
  elements.stopTrainingButton.disabled = !trainingRunning;
}

function openTrainingModal() {
  if (!trainingEnabled) {
    return;
  }
  elements.trainingModal.hidden = false;
  elements.openTrainingButton.setAttribute("aria-expanded", "true");
  renderTrainingButtons();
}

function closeTrainingModal() {
  elements.trainingModal.hidden = true;
  elements.openTrainingButton.setAttribute("aria-expanded", "false");
}

async function startFrontendTraining() {
  if (!trainingEnabled || trainingRunning) {
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
      game,
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
  resetAiWorker();
  await loadTrainingStatus();
  return payload;
}

function resetAiWorker() {
  aiRequestId += 1;
  bot.thinking = false;
  clearBotTimeout();
  terminateAiWorkers();
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
    labelKind,
    labelPhase,
    selfPlayWorkers,
    searchWorkers,
    bufferSize,
    labelCounts,
    batchSize,
    batchesPerSubmit,
    validationInterval,
    replaySize,
    validationLoss,
    bestValidationLoss,
    epochsWithoutImprovement,
    earlyStopReason,
    nonZeroWeights,
    gpuPhase,
    metrics,
    lossLogValidation
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
  if (labelWorkers !== undefined && collected === undefined) {
    const suffix = labelKind ? ` (${labelKind})` : "";
    const phase = labelPhase === "positions" ? "generating positions" : "encoding positions";
    elements.trainingStatus.textContent = `Run ${trainingCycle + 1}: ${phase} with ${labelWorkers} workers${suffix}.`;
    return;
  }
  if (model) {
    trainingCycle += 1;
    const stopText = earlyStopReason ? ` ${earlyStopReason}.` : "";
    const replayText = formatLossLogValidation(metrics?.lossLogValidation);
    elements.trainingStatus.textContent = `Run ${trainingCycle}: replacing model. Loss ${formatLoss(loss)}, best validation ${formatLoss(bestValidationLoss)}${replayText}.${stopText}`;
    logTrainingMetrics(trainingCycle, metrics);
    let replacement = null;
    try {
      replacement = await replaceActiveModel(model);
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
    const lossLogValidation = await validateTrainingLossLogs(trainingConfig());
    if (metrics && lossLogValidation) {
      metrics.lossLogValidation = lossLogValidation;
    }
    if (!trainingRunning) {
      renderTrainingButtons();
      return;
    }
    if (replacement?.changed === false) {
      const path = replacement.resolvedModelPath ?? replacement.modelPath ?? "model file";
      const hash = replacement.newHash ?? "same hash";
      elements.trainingStatus.textContent = `Run ${trainingCycle}: trained model matched disk (${hash}); no file diff at ${path}${formatLossLogValidation(lossLogValidation)}. Continuing with fresh samples.`;
      setTimeout(runFrontendTrainingCycle, 0);
      return;
    }
    const persisted = `model saved (${replacement?.newHash ?? "new hash"})`;
    elements.trainingStatus.textContent = `Run ${trainingCycle}: ${nonZeroWeights ?? 0} nonzero weights, ${persisted}${formatLossLogValidation(lossLogValidation)}. Restarting.`;
    setTimeout(runFrontendTrainingCycle, 0);
    return;
  }
  if (lossLogValidation) {
    const validation = event.data.lossLogValidation ?? event.data.validation;
    elements.trainingStatus.textContent = `Run ${trainingCycle + 1}: replaying loss logs${formatLossLogValidation(validation)}.`;
    return;
  }
  if (collected !== undefined) {
    const suffix = labelKind ? ` ${labelKind}` : "";
    const workerText = labelWorkers ? ` across ${labelWorkers} worker${labelWorkers === 1 ? "" : "s"}` : "";
    const verb = labelPhase === "positions" ? "generated" : "encoded";
    const noun = labelPhase === "labels" && labelKind === "search" ? " labels" : "";
    elements.trainingStatus.textContent = `Run ${trainingCycle + 1}: ${verb} ${collected}/${sampleCount}${suffix}${noun}${workerText}.`;
    return;
  }
  if (gpuPhase) {
    const selfPlayText = selfPlayWorkers ? `, self-play ${selfPlayWorkers}` : "";
    const searchText = searchWorkers ? `, search ${searchWorkers}` : "";
    elements.trainingStatus.textContent = `Run ${trainingCycle + 1}: GPU training ${bufferSize} samples, batch ${batchSize}${selfPlayText}${searchText}, temp ${formatTemperature(trainingConfig().explorationTemperature)}. ${formatLabelCounts(labelCounts)}.`;
    return;
  }
  if (epoch !== undefined) {
    const stale = epochsWithoutImprovement !== undefined ? `, stale ${epochsWithoutImprovement}` : "";
    const submitGroup = batchesPerSubmit ? `, submit group ${batchesPerSubmit}` : "";
    const validationText = validationInterval ? `, val every ${validationInterval}` : "";
    elements.trainingStatus.textContent = `Run ${trainingCycle + 1}: epoch ${epoch}. Train ${formatLoss(loss)}, validation ${formatLoss(validationLoss)}, best ${formatLoss(bestValidationLoss)}${stale}. Replay ${replaySize ?? "?"}, batch ${batchSize ?? "?"}${submitGroup}${validationText}.`;
  }
}

function validateTrainingLossLogs(config) {
  if (!trainingRunning || (config.lossLogReplay ?? 0) <= 0) {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    const worker = new Worker("./training-worker.js", { type: "module" });
    const id = ++trainingRequestId;
    const cleanup = () => {
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
      worker.removeEventListener("messageerror", handleError);
      worker.terminate();
    };
    const handleMessage = (event) => {
      if (event.data.id !== id) {
        return;
      }
      if (event.data.type === "lossLogValidation") {
        cleanup();
        resolve(event.data.validation ?? null);
        return;
      }
      if (event.data.lossLogValidation) {
        elements.trainingStatus.textContent = `Run ${trainingCycle}: replaying loss logs${formatLossLogValidation(event.data.lossLogValidation)}.`;
      }
      if (event.data.ok === false) {
        cleanup();
        resolve(null);
      }
    };
    const handleError = () => {
      cleanup();
      resolve(null);
    };
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.addEventListener("messageerror", handleError);
    worker.postMessage({
      id,
      type: "validateLossLogs",
      config
    });
  });
}

function logTrainingMetrics(run, metrics) {
  if (!metrics?.phases) {
    return;
  }
  console.groupCollapsed(`Training run ${run}: ${metrics.totalMs ?? "?"} ms`);
  console.table(Object.entries(metrics.phases).map(([phase, ms]) => ({
    phase,
    ms
  })));
  if (metrics.sampleRates && Object.keys(metrics.sampleRates).length > 0) {
    console.table(Object.entries(metrics.sampleRates).map(([source, samplesPerSecond]) => ({
      source,
      samplesPerSecond
    })));
  }
  if (metrics.lossLogValidation) {
    console.table([metrics.lossLogValidation]);
  }
  console.groupEnd();
}

function formatLossLogValidation(validation) {
  if (!validation || validation.checked === 0) {
    return "";
  }
  const state = validation.failed ? "no loss-log decision changes" : "loss-log changes";
  return `, ${state} ${validation.changed}/${validation.checked}`;
}

function formatLoss(loss) {
  return Number.isFinite(loss) ? loss.toFixed(2) : "pending";
}

function formatTemperature(value) {
  return Number.isFinite(value) ? value.toFixed(2).replace(/\.?0+$/, "") : "?";
}

function formatLabelCounts(counts) {
  if (!counts || typeof counts !== "object") {
    return "No source counts";
  }
  return ["search", "outcome", "distilled", "unknown"]
    .filter((key) => counts[key])
    .map((key) => `${key} ${counts[key]}`)
    .join(", ") || "No source counts";
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

  const undone = stagedMoves.length;
  resetStagedClientState();
  elements.message.textContent = undone > 0 ? "Reset staged moves." : "No staged moves to reset.";
  persistLocalGameState();
  render();
});

elements.undoMoveButton.addEventListener("click", async () => {
  if (!engine) {
    elements.message.textContent = "WASM engine is not loaded yet.";
    return;
  }
  if (!canActNow()) {
    elements.message.textContent = `Waiting for ${playerDisplayName(game.turn)}.`;
    return;
  }

  if (stagedMoves.length === 0) {
    elements.message.textContent = "No staged move to undo.";
    return;
  }

  const remaining = stagedMoves.slice(0, -1).map(cloneMove);
  await rebuildStagedClientState(remaining);
  elements.message.textContent = remaining.length === 0
    ? "Select a piece on a latest board."
    : "Undid staged move.";
  persistLocalGameState();
  render();
});

elements.submitTurnButton.addEventListener("click", async () => {
  if (!engine) {
    elements.message.textContent = "WASM engine is not loaded yet.";
    return;
  }
  if (!canActNow()) {
    elements.message.textContent = `Waiting for ${playerDisplayName(game.turn)}.`;
    return;
  }

  const actor = game.turn;
  const message = await submitVisibleTurn(actor);
  if (!message) {
    return;
  }

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

elements.openTrainingButton.addEventListener("click", () => {
  openTrainingModal();
});

elements.closeTrainingButton.addEventListener("click", () => {
  closeTrainingModal();
});

elements.trainingModal.addEventListener("click", (event) => {
  if (event.target === elements.trainingModal) {
    closeTrainingModal();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.trainingModal.hidden) {
    closeTrainingModal();
  }
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
