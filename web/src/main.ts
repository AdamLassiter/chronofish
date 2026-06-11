import { elements } from "./dom.js";
import { capitalize, getLatestBoard, isActiveTimeline, presentTime, samePosition } from "./board.js";
import { renderGame } from "./render.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import { appendNotationLine, postMatchLog, postBotLossLog } from "./match-log.js";
import { readWasmString, writeWasmString } from "./engine-io.js";
import { createTrainingController } from "./training-ui.js";
import { createBotController } from "./bot-controller.js";
import { createEvaluationUi } from "./evaluation-ui.js";
import { wireMainEvents } from "./main-events.js";
import { APP_VERSION } from "./app-version.js";
import type { ChronofishEngine, Color, GameSnapshot, Move, Piece, Position } from "./types.js";

const LOCAL_GAME_STORAGE_KEY = "chronofish.localGameState.v1";
const GPU_MODE_STORAGE_KEY = "chronofish.gpuMode";
const initialSearchParams = new URLSearchParams(window.location.search);
const initialUrlHasRoom = initialSearchParams.has("room");

type Phase = "lobby" | "game" | "review";
type LobbyColor = Color | "spectator";
type Assignment =
  | "local"
  | "human"
  | "open"
  | "bot-fast"
  | "bot-balanced"
  | "bot-expert"
  | "bot-gpu-fast"
  | "bot-gpu-balanced"
  | "bot-gpu-expert"
  | "bot-cpu-fast"
  | "bot-cpu-balanced"
  | "bot-cpu-expert";
type Assignments = Record<Color, Assignment>;

interface BotCredentials {
  color: Color;
  token: string;
}

interface BotEffortConfig {
  label: string;
  displayNames: string[];
  depth: number;
  nodes: number;
  timeMs: number;
}

type BotEffortName = "fast" | "balanced" | "expert";
type BotEffortConfigs = Record<BotEffortName, BotEffortConfig>;

interface MultiplayerState {
  roomId: string;
  token: string;
  color: LobbyColor | "local";
  events: EventSource | null;
  connected: boolean;
}

interface RoomGamePayload {
  phase?: Phase;
  assignments?: Partial<Record<Color, unknown>>;
  notation?: string;
  turns?: Move[][];
  snapshot?: GameSnapshot;
  timelines?: GameSnapshot["timelines"];
}

interface RoomState {
  game?: RoomGamePayload | GameSnapshot;
  players?: Partial<Record<Color, unknown>>;
}

interface RoomResponse {
  room: RoomState;
  color: LobbyColor;
  error?: string;
  version?: string;
}

interface PersistedGameState {
  phase?: Phase;
  assignments?: Partial<Record<Color, unknown>>;
  notation?: string;
  turns?: Move[][];
  stagedMoves?: Move[];
  snapshot?: GameSnapshot;
  committedSnapshot?: GameSnapshot;
  message?: string;
}

interface LegalTargetSelection {
  source: (Position & { piece: Piece }) | null;
  targets: Position[];
}

interface RenderOptions {
  preserveScroll?: boolean;
}

interface ScrollState {
  windowX: number;
  windowY: number;
  multiverseX: number;
  multiverseY: number;
}

let engine: ChronofishEngine | null = null;
let aiParameters: unknown = null;
let aiEffortConfigs: BotEffortConfigs = {
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
    displayNames: ["Kasparadox", "Premovaru Checkamura", "Anandromeda"],
    depth: 5,
    nodes: 200_000,
    timeMs: 250_000
  }
};

const gpuBotDisplayNames: Record<BotEffortName, string[]> = {
  fast: [
    "Stockfish & Chips",
    "Crafty Castler",
    "Fritz Blitz"
  ],
  balanced: [
    "Deep Blue Shift",
    "Leela Timeline Zero",
    "Komodo Chrono"
  ],
  expert: [
    "AlphaZero Hour",
    "Rybka in Time",
    "Houdini's Horizon"
  ]
};

const cpuBotDisplayNames: Record<BotEffortName, string[]> = {
  fast: [
    "Bullet Fischer",
    "Speedrun Steinitz",
    "Blitz Botvinnik"
  ],
  balanced: [
    "Timeline Tal",
    "Causality Capablanca",
    "Multiverse Magnus"
  ],
  expert: [
    "Kasparadox",
    "Premovaru Checkamura",
    "Anandromeda"
  ]
};
let game: GameSnapshot = {
  turn: "white",
  nextTimelineId: 1,
  nextBlackTimelineId: -1,
  checkedRoyals: [],
  timelines: []
};
// Last submitted snapshot. While a turn is staged, rendering compares against
// this so the present line and board status labels do not jump before Submit.
let committedGame = game;
let selected: Position | null = null;
let legalTargets: Position[] = [];
// submittedTurns is replayable room history; stagedMoves is local undo state for
// the current unsubmitted turn only.
let submittedTurns: Move[][] = [];
let submittedNotation = "";
let stagedMoves: Move[] = [];
let legalTargetRequestId = 0;
let phase: Phase = "lobby";
let lastRenderedPhase: Phase = phase;
let lastMatchAlertMessage = "";
let assignments: Assignments = {
  white: normalizeAssignment(localStorage.getItem("chronofish.whitePlayer"), "local"),
  black: normalizeAssignment(localStorage.getItem("chronofish.blackPlayer"), "local")
};
let multiplayer: MultiplayerState = {
  // Room id lives in the URL so sharing the address reconstructs the room.
  roomId: initialSearchParams.get("room") ?? makeRoomId(),
  token: localStorage.getItem("chronofish.playerToken") ?? crypto.randomUUID(),
  color: normalizeLobbyColor(localStorage.getItem("chronofish.playerColor"), "local"),
  events: null,
  connected: false
};
let currentRoom: RoomState | null = null;
const botController = createBotController({
  getEngine: () => engine,
  getPhase: () => phase,
  getAssignments: () => assignments,
  getGame: () => game,
  getStagedMoves: () => stagedMoves,
  getSubmittedTurns: () => submittedTurns,
  getRoomId: () => multiplayer.roomId,
  isBotAssignment,
  botEffortName,
  botEffort,
  botDisplayName,
  cloneGame,
  cloneMove,
  turnSignature,
  postRoom,
  lobbyPayload,
  applyRemoteRoom,
  applyEngineMove,
  submitVisibleTurn,
  concede,
  persistLocalGameState,
  render,
  isMatchOver,
  enterPostMatchReview,
  syncState
});
const evaluationUi = createEvaluationUi({
  getEvaluation: () => {
    if (!engine || !game.timelines.length) {
      return null;
    }
    return JSON.parse(readWasmString(engine, engine.chronofish_evaluation_json())) as {
      score: number;
      source: string;
    };
  }
});
const training = createTrainingController({
  getEngine: () => engine,
  getGame: () => game,
  resetAiWorker: () => botController.resetAiWorker()
});

localStorage.setItem("chronofish.playerToken", multiplayer.token);
elements.appStatus.textContent = `🌐 v${APP_VERSION}`;
elements.roomInput.value = multiplayer.roomId;
elements.whitePlayerSelect.value = assignments.white;
elements.blackPlayerSelect.value = assignments.black;

function makeRoomId(): string {
  return Math.random().toString(36).slice(2, 8);
}

function roomUrl(roomId: string): URL {
  const url = new URL(window.location.href);
  url.searchParams.set("room", roomId);
  return url;
}

function normalizeRoomId(value: string): string {
  return value.trim().replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 48) || makeRoomId();
}

function normalizeLobbyColor(value: unknown, fallback: LobbyColor | "local" = "local"): LobbyColor | "local" {
  return value === "white" || value === "black" || value === "spectator" || value === "local" ? value : fallback;
}

function canControlTurn(): boolean {
  if (!engine || phase !== "game" || isBotAssignment(assignments[game.turn])) {
    return false;
  }

  if (!multiplayer.connected) {
    return assignments[game.turn] === "local";
  }

  return assignments[game.turn] === "human" && multiplayer.color === game.turn;
}

function canActNow(): boolean {
  return phase === "game" && canControlTurn();
}

function hasStagedMoves(): boolean {
  return stagedMoves.length > 0;
}

function setMultiplayerStatus(text: string): void {
  elements.multiplayerStatus.textContent = text;
}

function updateShareLink(): void {
  if (!multiplayer.connected) {
    elements.shareLink.textContent = "";
    return;
  }

  const link = roomUrl(multiplayer.roomId);
  elements.shareLink.innerHTML = `<a href="${link.href}">Share room</a>`;
}

function normalizeAssignment(value: unknown, fallback: Assignment = "local"): Assignment {
  if (value === "bot") {
    return "bot-balanced";
  }
  if (typeof value === "string" && value.startsWith("nn-bot-")) {
    return normalizeAssignment(value.replace("nn-bot-", "bot-"), fallback);
  }
  return [
    "local",
    "human",
    "open",
    "bot-fast",
    "bot-balanced",
    "bot-expert",
    "bot-gpu-fast",
    "bot-gpu-balanced",
    "bot-gpu-expert",
    "bot-cpu-fast",
    "bot-cpu-balanced",
    "bot-cpu-expert"
  ].includes(value as Assignment) ? value as Assignment : fallback;
}

function isBotAssignment(value: unknown): boolean {
  return typeof value === "string" && value.startsWith("bot-");
}

function botEffortName(value: unknown): BotEffortName {
  if (isBotAssignment(value)) {
    const effort = String(value)
      .slice("bot-".length)
      .replace(/^gpu-/, "")
      .replace(/^cpu-/, "");
    if (effort === "fast" || effort === "balanced" || effort === "expert") {
      return effort;
    }
  }
  return "balanced";
}

function botEffort(value: unknown): BotEffortConfig {
  return aiEffortConfigs[botEffortName(value)] ?? aiEffortConfigs.balanced;
}

function botBackendName(value: unknown): "gpu" | "cpu" {
  return typeof value === "string" && value.startsWith("bot-cpu-") ? "cpu" : "gpu";
}

function stableIndex(value: string, count: number): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash) % count;
}

function botDisplayName(color: Color): string {
  const assignment = assignments[color];
  const effortName = botEffortName(assignments[color]);
  const backend = botBackendName(assignment);
  const effort = botEffort(assignment);
  const names = backend === "cpu"
    ? cpuBotDisplayNames[effortName]
    : gpuBotDisplayNames[effortName];
  return names[stableIndex(`${multiplayer.roomId}:${color}:${backend}:${effortName}`, names.length)]
    ?? effort.label
    ?? "Bot";
}

function playerDisplayName(color: Color): string {
  return isBotAssignment(assignments[color]) ? botDisplayName(color) : capitalize(color);
}

function displayGameMessage(message: unknown): string {
  return String(message ?? "")
    .replace(/\bWhite\b/g, playerDisplayName("white"))
    .replace(/\bBlack\b/g, playerDisplayName("black"));
}

function readAssignments(): Assignments {
  return {
    white: normalizeAssignment(elements.whitePlayerSelect.value, "local"),
    black: normalizeAssignment(elements.blackPlayerSelect.value, "local")
  };
}

function writeAssignments(nextAssignments: Partial<Record<Color, unknown>> = {}): void {
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

function gamePayload(nextPhase: Phase = phase): RoomGamePayload {
  return {
    phase: nextPhase,
    assignments,
    notation: submittedNotation,
    snapshot: game
  };
}

function shouldPersistLocalGame(): boolean {
  return !multiplayer.connected && (phase === "game" || phase === "review");
}

function persistLocalGameState(): void {
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

function clearLocalGameState(): void {
  localStorage.removeItem(LOCAL_GAME_STORAGE_KEY);
}

async function restoreLocalGameState(): Promise<boolean> {
  if (!engine || multiplayer.connected || initialUrlHasRoom) {
    return false;
  }

  const saved = localStorage.getItem(LOCAL_GAME_STORAGE_KEY);
  if (!saved) {
    return false;
  }

  try {
    const state = JSON.parse(saved) as PersistedGameState;
    if (!state.phase || !["game", "review"].includes(state.phase)) {
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

function lobbyPayload(): RoomGamePayload {
  return {
    phase: "lobby",
    assignments,
    snapshot: committedGame
  };
}

function isMatchOver(): boolean {
  return phase === "review" || game.result?.terminal === true;
}

function activeEngine(): ChronofishEngine {
  if (!engine) {
    throw new Error("WASM engine is not loaded yet.");
  }
  return engine;
}

function resetEngine(): void {
  // Engine reset clears both visible and committed state plus all local history.
  if (engine) {
    engine.chronofish_reset();
    game = engineSnapshot();
  }
  committedGame = game;
  selected = null;
  legalTargets = [];
  submittedTurns = [];
  submittedNotation = "";
  stagedMoves = [];
  botController.clearDecisionLog();
  lastMatchAlertMessage = "";
}

function engineSnapshot(): GameSnapshot {
  const wasm = activeEngine();
  return JSON.parse(readWasmString(wasm, wasm.chronofish_snapshot_json())) as GameSnapshot;
}

function engineLastMessage(): string {
  const wasm = activeEngine();
  return readWasmString(wasm, wasm.chronofish_last_message());
}

function engineStagedTurnNotation(): string {
  const wasm = activeEngine();
  return readWasmString(wasm, wasm.chronofish_staged_turn_notation());
}

function syncEngineSnapshot(snapshot: GameSnapshot = game): void {
  const wasm = activeEngine();
  const { ptr, len } = writeWasmString(wasm, JSON.stringify(snapshot));
  try {
    if (!wasm.chronofish_load_snapshot_json(ptr, len)) {
      throw new Error(engineLastMessage());
    }
  } finally {
    wasm.chronofish_dealloc(ptr, len);
  }
}

function syncEngineToStagedState(): void {
  const wasm = activeEngine();
  syncEngineSnapshot(committedGame);
  for (const move of stagedMoves) {
    const ok = wasm.chronofish_apply_move(
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

function appendSubmittedNotation(turnNotation: string, actor: Color = game.turn): void {
  if (!turnNotation) {
    return;
  }
  const previous = submittedNotation;
  submittedNotation = appendNotationLine({ submittedNotation, turnNotation });
  if (submittedNotation !== previous) {
    postMatchLog(multiplayer.roomId, submittedNotation.split(/\n/).at(-1) ?? "");
  }
}

function cloneMove(move: Move): Move {
  return {
    from: { ...move.from },
    to: { ...move.to }
  };
}

function cloneGame(snapshot: GameSnapshot): GameSnapshot {
  return JSON.parse(JSON.stringify(snapshot)) as GameSnapshot;
}

function turnSignature(): string {
  return JSON.stringify({
    phase,
    roomId: multiplayer.roomId,
    turn: committedGame.turn,
    submittedTurnCount: submittedTurns.length,
    committedGame
  });
}

function targetFor(position: Position): Position | undefined {
  return legalTargets.find((target) => samePosition(target, position));
}

function legalTargetsFor(position: Position): Promise<LegalTargetSelection> {
  if (!engine) {
    return Promise.resolve({ source: null, targets: [] });
  }
  syncEngineSnapshot(game);
  return Promise.resolve(JSON.parse(readWasmString(engine, engine.chronofish_legal_selection_json(
    position.timelineId,
    position.time,
    position.x,
    position.y
  ))) as LegalTargetSelection);
}

async function applyEngineMove(from: Position, to: Position): Promise<string | null> {
  const wasm = activeEngine();
  let ok: number;
  try {
    syncEngineSnapshot(game);
    ok = wasm.chronofish_apply_move(
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
    elements.message.textContent = errorMessage(error);
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

async function submitVisibleTurn(actor: Color): Promise<string | null> {
  if (stagedMoves.length === 0) {
    elements.message.textContent = "Make at least one move before submitting.";
    return null;
  }

  const wasm = activeEngine();
  let ok: number;
  try {
    syncEngineToStagedState();
    ok = wasm.chronofish_submit_turn();
  } catch (error) {
    elements.message.textContent = errorMessage(error);
    return null;
  }
  const engineMessage = engineLastMessage();
  if (!ok) {
    elements.message.textContent = engineMessage;
    return null;
  }

  const turnNotation = engineStagedTurnNotation();
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

function resetStagedClientState(): void {
  const committed = committedGame;
  game = committed;
  committedGame = committed;
  stagedMoves = [];
  selected = null;
  legalTargets = [];
}

async function rebuildStagedClientState(moves: Move[]): Promise<void> {
  resetStagedClientState();
  for (const move of moves) {
    if (!(await applyEngineMove(move.from, move.to))) {
      break;
    }
  }
}

async function handleSquareClick(position: Position): Promise<void> {
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
    await applyEngineMove(selected, position);
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
    elements.message.textContent = errorMessage(error);
    render({ preserveScroll: true });
    return;
  }
}

function render(options: RenderOptions = {}): void {
  const scrollState = options.preserveScroll ? captureScrollState() : null;
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
  training.renderButtons();

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
  evaluationUi.renderEvaluationBar();

  if (scrollState) {
    restoreScrollState(scrollState);
  }
  evaluationUi.maybeScrollToPresent({
    phase,
    enteredGame,
    nextPresentTime,
    preserveScroll: Boolean(scrollState)
  });
  lastRenderedPhase = phase;
}

function captureScrollState(): ScrollState {
  return {
    windowX: window.scrollX,
    windowY: window.scrollY,
    multiverseX: elements.multiverse?.scrollLeft ?? 0,
    multiverseY: elements.multiverse?.scrollTop ?? 0
  };
}

function restoreScrollState(state: ScrollState): void {
  elements.multiverse.scrollLeft = state.multiverseX;
  elements.multiverse.scrollTop = state.multiverseY;
  window.scrollTo(state.windowX, state.windowY);
}

function setHudCollapsed(collapsed: boolean): void {
  // Preserve the space-saving preference across reloads.
  elements.hud.dataset.collapsed = String(collapsed);
  elements.toggleHudButton.textContent = collapsed ? "Show" : "Hide";
  elements.toggleHudButton.setAttribute("aria-expanded", String(!collapsed));
  localStorage.setItem("chronofish.hudCollapsed", String(collapsed));
}

async function postRoom(action: string, body: unknown): Promise<RoomResponse> {
  const response = await fetch(`/api/rooms/${encodeURIComponent(multiplayer.roomId)}/${action}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body)
  });
  const payload = await response.json() as RoomResponse;

  if (!response.ok) {
    throw new Error(payload.error ?? "Room request failed");
  }

  return payload;
}

function isRoomGamePayload(value: RoomState["game"]): value is RoomGamePayload {
  return Boolean(value && typeof value === "object" && ("phase" in value || "snapshot" in value || "turns" in value || "notation" in value));
}

function applyRemoteRoom(room: RoomState, message = ""): void {
  currentRoom = room;
  botController.resetAiWorker();

  if (isRoomGamePayload(room.game) && room.game.phase) {
    phase = room.game.phase;
  }
  if (isRoomGamePayload(room.game) && room.game.assignments) {
    writeAssignments(room.game.assignments);
  }

  if (isRoomGamePayload(room.game) && room.game.snapshot) {
    game = room.game.snapshot;
    committedGame = game;
    submittedTurns = room.game.turns ?? [];
    submittedNotation = room.game.notation ?? "";
    stagedMoves = [];
  } else if (room.game && "timelines" in room.game) {
    game = room.game as GameSnapshot;
    committedGame = game;
    submittedTurns = [];
    submittedNotation = "";
    stagedMoves = [];
  }

  if (engine && game.timelines.length) {
    syncEngineSnapshot(game);
    game = engineSnapshot();
    committedGame = game;
  }

  selected = null;
  legalTargets = [];
  updateShareLink();
  render();

  if (message) {
    elements.message.textContent = message;
    showMatchDialog(message);
  }

  botController.maybeStartTurn();
}

function connectEvents(): void {
  multiplayer.events?.close();
  multiplayer.events = new EventSource(`/api/rooms/${encodeURIComponent(multiplayer.roomId)}/events`);

  multiplayer.events.addEventListener("message", (event: MessageEvent<string>) => {
    const payload = JSON.parse(event.data) as { type?: string; room: RoomState; message?: string };

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

async function joinRoom(color: LobbyColor): Promise<void> {
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

async function syncState(action: string, message: string, credentials: BotCredentials | null = null): Promise<void> {
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
    elements.message.textContent = errorMessage(error);
  }
}

function showMatchDialog(message: string): void {
  if (!isMatchOver() || lastMatchAlertMessage === message) {
    return;
  }
  lastMatchAlertMessage = message;
  window.alert(message);
}

async function enterPostMatchReview(message: string, credentials: BotCredentials | null = null): Promise<void> {
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

function recordBotLossLog(message: string): void {
  const winner = winnerFromMatchMessage(message);
  if (!winner) {
    return;
  }
  const loser = winner === "white" ? "black" : "white";
  if (!isBotAssignment(assignments[loser]) || isBotAssignment(assignments[winner])) {
    return;
  }
  const decisions = botController.decisionsFor(loser);
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

function winnerFromMatchMessage(message: string): Color | null {
  if (game.result?.winner) {
    return game.result.winner;
  }
  const normalized = String(message ?? "").toLowerCase();
  if (/\bwhite\b.*\bwins\b/.test(normalized)) {
    return "white";
  }
  if (/\bblack\b.*\bwins\b/.test(normalized)) {
    return "black";
  }
  return null;
}

function victoryMessage(loser: Color): string {
  const winner = loser === "white" ? "black" : "white";
  return `${playerDisplayName(winner)} wins. ${playerDisplayName(loser)} conceded.`;
}

async function concede(color: Color = game.turn, credentials: BotCredentials | null = null): Promise<void> {
  const message = victoryMessage(color);
  await enterPostMatchReview(message, credentials ?? { color, token: multiplayer.token });
}

async function syncLobby(message = "Lobby updated."): Promise<void> {
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
    elements.message.textContent = errorMessage(error);
  }
}

function validateAssignments(nextAssignments: Assignments): void {
  for (const color of ["white", "black"] as const) {
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

async function startGame(): Promise<void> {
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
    botController.maybeStartTurn();
    return;
  }

  if (multiplayer.color === "spectator") {
    throw new Error("Spectators cannot start the game.");
  }

  for (const color of botController.colors()) {
    await botController.seat(color);
  }

  await postRoom("state", {
    token: multiplayer.token,
    color: multiplayer.color,
    game: gamePayload("game"),
    message: "Game started."
  });
  elements.message.textContent = "Game started.";
  render();
  botController.maybeStartTurn();
}

async function loadWasmStatus(): Promise<void> {
  try {
    const wasmPath = "./chronofish_engine.wasm";
    const instance = await instantiateChronofishWasm(wasmPath);
    engine = instance.exports as unknown as ChronofishEngine;
    resetEngine();
    const restored = await restoreLocalGameState();
    elements.wasmStatus.textContent = `🧠 v${readWasmString(engine, engine.chronofish_version())}`;
    elements.wasmStatus.dataset.state = "ready";
    if (!restored) {
      elements.message.textContent = "Configure the lobby, then start the game.";
    }
    render();
    if (restored) {
      botController.maybeStartTurn();
    }
  } catch (error) {
    console.error(error);
    elements.wasmStatus.textContent = "🧠 missing";
    elements.wasmStatus.dataset.state = "error";
    elements.message.textContent = "Build the WASM engine first with `cargo build --manifest-path engine/Cargo.toml --target wasm32-unknown-unknown`.";
    render();
  }
}

async function loadServerStatus(): Promise<void> {
  try {
    const [versionResponse, parametersResponse, effortResponse] = await Promise.all([
      fetch("/api/version"),
      fetch("/ai/parameters.json"),
      fetch("/ai/effort.json")
    ]);
    const payload = await versionResponse.json() as { version?: string; error?: string };

    if (!versionResponse.ok) {
      throw new Error(payload.error ?? "Server unavailable");
    }
    if (parametersResponse.ok) {
      aiParameters = await parametersResponse.json();
    }
    if (effortResponse.ok) {
      aiEffortConfigs = await effortResponse.json() as BotEffortConfigs;
    }

    elements.serverStatus.textContent = `🖥 v${payload.version}`;
    elements.serverStatus.dataset.state = "ready";
    await training.loadStatus();
  } catch (error) {
    console.error(error);
    elements.serverStatus.textContent = "🖥 offline";
    elements.serverStatus.dataset.state = "error";
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

wireMainEvents({
  getEngine: () => engine,
  getGame: () => game,
  getStagedMoves: () => stagedMoves,
  canActNow,
  playerDisplayName,
  resetStagedClientState,
  persistLocalGameState,
  render,
  cloneMove,
  rebuildStagedClientState,
  submitVisibleTurn,
  isMatchOver,
  enterPostMatchReview,
  syncState,
  maybeStartBotTurn: () => botController.maybeStartTurn(),
  capitalize,
  concede,
  setHudCollapsed,
  training,
  joinRoom,
  startGame,
  writeAssignments,
  readAssignments,
  syncLobby
});

loadWasmStatus();
loadServerStatus();
setHudCollapsed(localStorage.getItem("chronofish.hudCollapsed") === "true");
render();
