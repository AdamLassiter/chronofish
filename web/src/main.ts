import { elements } from "./dom.js";
import { capitalize, getLatestBoard, isActiveTimeline, presentTime, samePosition } from "./board.js";
import { renderGame } from "./render.js";
import { instantiateChronofishWasm } from "./wasm-loader.js";
import { appendNotationLine, postMatchLog, postBotLossLog } from "./match-log.js";
import { readWasmString, writeWasmString } from "./engine-io.js";
import { createTrainingController } from "./training-ui.js";
import { createBotController } from "./bot-controller.js";
import type { BotDecisionRecord } from "./bot-controller.js";
import { createEvaluationUi } from "./evaluation-ui.js";
import { wireMainEvents } from "./main-events.js";
import { APP_VERSION } from "./app-version.js";
import type { BoardSnapshot, ChronofishEngine, Color, GameSnapshot, GhostBoard, Move, Piece, PlannedArrow, Position } from "./types.js";

const LOCAL_GAME_STORAGE_KEY = "chronofish.localGameState.v1";
const GPU_MODE_STORAGE_KEY = "chronofish.gpuMode";
const CUSTOM_CPU_EFFORT_STORAGE_KEY = "chronofish.customCpuEffort.v1";
const CUSTOM_GPU_EFFORT_STORAGE_KEY = "chronofish.customGpuEffort.v1";
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
  | "bot-gpu-custom"
  | "bot-cpu-fast"
  | "bot-cpu-balanced"
  | "bot-cpu-expert"
  | "bot-cpu-custom";
type Assignments = Record<Color, Assignment>;

interface BotCredentials {
  color: Color;
  token: string;
}

interface BotEffortConfig {
  label: string;
  displayNames: string[];
  depth: number;
  minDepth?: number;
  nodes: number;
  timeMs: number;
  searchStrategy?: "alpha-beta" | "beam";
}

type BotPresetName = "fast" | "balanced" | "expert";
type BotEffortName = BotPresetName | "custom";
type BotEffortConfigs = Partial<Record<BotPresetName, BotEffortConfig>>;

interface CustomCpuEffortConfig extends BotEffortConfig {
  label: "The Chronofish";
  displayNames: ["The Chronofish"];
  minDepth: number;
}

interface CustomGpuEffortConfig extends BotEffortConfig {
  label: "Custom GPU Bot";
  displayNames: ["Custom GPU Bot"];
  minDepth: number;
}

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
  customCpuEffort?: Partial<BotEffortConfig>;
  customGpuEffort?: Partial<BotEffortConfig>;
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
  customCpuEffort?: Partial<BotEffortConfig>;
  customGpuEffort?: Partial<BotEffortConfig>;
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

interface PlannedMoveNode {
  id: string;
  parentId: string | null;
  move: Move;
  beforeSnapshot: GameSnapshot;
  afterSnapshot: GameSnapshot;
  children: string[];
  kind: "planned" | "bot-review";
}

interface PlannedMoveAnchor {
  nodeId: string | null;
  position: Position;
  targets: Position[];
}

interface BotReviewProjection {
  finalGame: GameSnapshot;
  finalCommittedGame: GameSnapshot;
  decision: BotDecisionRecord;
}

interface BotReviewPlanMatch {
  decision: BotDecisionRecord;
  baseSnapshot: GameSnapshot;
  skipTurns: number;
  skipMovesInFirstTurn: number;
}

interface RenderOptions {
  preserveScroll?: boolean;
  focusPosition?: Position;
}

interface ScrollState {
  windowX: number;
  windowY: number;
  multiverseX: number;
  multiverseY: number;
}

let engine: ChronofishEngine | null = null;
let aiParameters: unknown = null;
let cpuEffortConfigs: BotEffortConfigs = {};
let gpuEffortConfigs: BotEffortConfigs = {};

const DEFAULT_CUSTOM_CPU_EFFORT: CustomCpuEffortConfig = {
  label: "The Chronofish",
  displayNames: ["The Chronofish"],
  depth: 4,
  minDepth: 2,
  nodes: 80_000,
  timeMs: 5_000,
  searchStrategy: "alpha-beta"
};
let customCpuEffort: CustomCpuEffortConfig = loadCustomCpuEffort();
const DEFAULT_CUSTOM_GPU_EFFORT: CustomGpuEffortConfig = {
  label: "Custom GPU Bot",
  displayNames: ["Custom GPU Bot"],
  depth: 4,
  minDepth: 2,
  nodes: 80_000,
  timeMs: 5_000
};
let customGpuEffort: CustomGpuEffortConfig = loadCustomGpuEffort();
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
let plannedMoveAnchor: PlannedMoveAnchor | null = null;
let plannedMoveNodes = new Map<string, PlannedMoveNode>();
let plannedMoveRoots: string[] = [];
let nextPlannedMoveNodeId = 1;
let botReviewProjection: BotReviewProjection | null = null;
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
    "bot-gpu-custom",
    "bot-cpu-fast",
    "bot-cpu-balanced",
    "bot-cpu-expert",
    "bot-cpu-custom"
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
    if (effort === "fast" || effort === "balanced" || effort === "expert" || effort === "custom") {
      return effort;
    }
  }
  return "balanced";
}

function botEffort(value: unknown): BotEffortConfig | null {
  if (value === "bot-cpu-custom") {
    return customCpuEffort;
  }
  if (value === "bot-gpu-custom") {
    return customGpuEffort;
  }
  const effortName = botEffortName(value);
  if (effortName === "custom") {
    return botBackendName(value) === "cpu" ? customCpuEffort : customGpuEffort;
  }
  const configs = botBackendName(value) === "cpu" ? cpuEffortConfigs : gpuEffortConfigs;
  return configs[effortName] ?? configs.balanced ?? null;
}

function botBackendName(value: unknown): "gpu" | "cpu" {
  return typeof value === "string" && value.startsWith("bot-cpu-") ? "cpu" : "gpu";
}

function loadCustomCpuEffort(): CustomCpuEffortConfig {
  try {
    return normalizeCustomCpuEffort(JSON.parse(localStorage.getItem(CUSTOM_CPU_EFFORT_STORAGE_KEY) ?? "null"));
  } catch {
    return DEFAULT_CUSTOM_CPU_EFFORT;
  }
}

function loadCustomGpuEffort(): CustomGpuEffortConfig {
  try {
    return normalizeCustomGpuEffort(JSON.parse(localStorage.getItem(CUSTOM_GPU_EFFORT_STORAGE_KEY) ?? "null"));
  } catch {
    return DEFAULT_CUSTOM_GPU_EFFORT;
  }
}

function normalizeCustomCpuEffort(value: unknown): CustomCpuEffortConfig {
  const candidate = value && typeof value === "object" ? value as Partial<BotEffortConfig> : {};
  const depth = clampInteger(candidate.depth, 1, 16, DEFAULT_CUSTOM_CPU_EFFORT.depth);
  return {
    label: "The Chronofish",
    displayNames: ["The Chronofish"],
    depth,
    minDepth: Math.min(depth, clampInteger(candidate.minDepth, 1, 16, DEFAULT_CUSTOM_CPU_EFFORT.minDepth)),
    nodes: clampInteger(candidate.nodes, 1, 1_000_000, DEFAULT_CUSTOM_CPU_EFFORT.nodes),
    timeMs: clampInteger(candidate.timeMs, 1, 600_000, DEFAULT_CUSTOM_CPU_EFFORT.timeMs),
    searchStrategy: candidate.searchStrategy === "beam" ? "beam" : "alpha-beta"
  };
}

function normalizeCustomGpuEffort(value: unknown): CustomGpuEffortConfig {
  const candidate = value && typeof value === "object" ? value as Partial<BotEffortConfig> : {};
  const depth = clampInteger(candidate.depth, 1, 16, DEFAULT_CUSTOM_GPU_EFFORT.depth);
  return {
    label: "Custom GPU Bot",
    displayNames: ["Custom GPU Bot"],
    depth,
    minDepth: Math.min(depth, clampInteger(candidate.minDepth, 1, 16, DEFAULT_CUSTOM_GPU_EFFORT.minDepth)),
    nodes: clampInteger(candidate.nodes, 1, 1_000_000, DEFAULT_CUSTOM_GPU_EFFORT.nodes),
    timeMs: clampInteger(candidate.timeMs, 1, 600_000, DEFAULT_CUSTOM_GPU_EFFORT.timeMs)
  };
}

function clampInteger(value: unknown, min: number, max: number, fallback: number): number {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.round(number)));
}

function saveCustomCpuEffort(next: CustomCpuEffortConfig): void {
  customCpuEffort = next;
  localStorage.setItem(CUSTOM_CPU_EFFORT_STORAGE_KEY, JSON.stringify({
    depth: next.depth,
    minDepth: next.minDepth,
    nodes: next.nodes,
    timeMs: next.timeMs,
    searchStrategy: next.searchStrategy
  }));
}

function syncCustomCpuInputs(): void {
  elements.customCpuDepthInput.value = String(customCpuEffort.depth);
  elements.customCpuMinDepthInput.value = String(customCpuEffort.minDepth);
  elements.customCpuNodesInput.value = String(customCpuEffort.nodes);
  elements.customCpuTimeMsInput.value = String(customCpuEffort.timeMs);
  elements.customCpuSearchStrategyInput.value = customCpuEffort.searchStrategy ?? "alpha-beta";
}

function readCustomCpuInputs(): CustomCpuEffortConfig {
  return normalizeCustomCpuEffort({
    depth: elements.customCpuDepthInput.value,
    minDepth: elements.customCpuMinDepthInput.value,
    nodes: elements.customCpuNodesInput.value,
    timeMs: elements.customCpuTimeMsInput.value,
    searchStrategy: elements.customCpuSearchStrategyInput.value
  });
}

function openCustomCpuModal(): void {
  syncCustomCpuInputs();
  elements.customCpuModal.hidden = false;
  elements.customCpuDepthInput.focus();
}

function closeCustomCpuModal(): void {
  elements.customCpuModal.hidden = true;
}

function applyCustomCpuModal(): void {
  saveCustomCpuEffort(readCustomCpuInputs());
  closeCustomCpuModal();
  elements.message.textContent = "Custom CPU bot configured. The Chronofish is ready.";
  persistLocalGameState();
  void syncLobby();
}

function resetCustomCpuModal(): void {
  elements.customCpuDepthInput.value = String(DEFAULT_CUSTOM_CPU_EFFORT.depth);
  elements.customCpuMinDepthInput.value = String(DEFAULT_CUSTOM_CPU_EFFORT.minDepth);
  elements.customCpuNodesInput.value = String(DEFAULT_CUSTOM_CPU_EFFORT.nodes);
  elements.customCpuTimeMsInput.value = String(DEFAULT_CUSTOM_CPU_EFFORT.timeMs);
  elements.customCpuSearchStrategyInput.value = DEFAULT_CUSTOM_CPU_EFFORT.searchStrategy ?? "alpha-beta";
}

function saveCustomGpuEffort(next: CustomGpuEffortConfig): void {
  customGpuEffort = next;
  localStorage.setItem(CUSTOM_GPU_EFFORT_STORAGE_KEY, JSON.stringify({
    depth: next.depth,
    minDepth: next.minDepth,
    nodes: next.nodes,
    timeMs: next.timeMs
  }));
}

function syncCustomGpuInputs(): void {
  elements.customGpuDepthInput.value = String(customGpuEffort.depth);
  elements.customGpuMinDepthInput.value = String(customGpuEffort.minDepth);
  elements.customGpuNodesInput.value = String(customGpuEffort.nodes);
  elements.customGpuTimeMsInput.value = String(customGpuEffort.timeMs);
}

function readCustomGpuInputs(): CustomGpuEffortConfig {
  return normalizeCustomGpuEffort({
    depth: elements.customGpuDepthInput.value,
    minDepth: elements.customGpuMinDepthInput.value,
    nodes: elements.customGpuNodesInput.value,
    timeMs: elements.customGpuTimeMsInput.value
  });
}

function openCustomGpuModal(): void {
  syncCustomGpuInputs();
  elements.customGpuModal.hidden = false;
  elements.customGpuDepthInput.focus();
}

function closeCustomGpuModal(): void {
  elements.customGpuModal.hidden = true;
}

function applyCustomGpuModal(): void {
  saveCustomGpuEffort(readCustomGpuInputs());
  closeCustomGpuModal();
  elements.message.textContent = "Custom GPU bot configured.";
  persistLocalGameState();
  void syncLobby();
}

function resetCustomGpuModal(): void {
  elements.customGpuDepthInput.value = String(DEFAULT_CUSTOM_GPU_EFFORT.depth);
  elements.customGpuMinDepthInput.value = String(DEFAULT_CUSTOM_GPU_EFFORT.minDepth);
  elements.customGpuNodesInput.value = String(DEFAULT_CUSTOM_GPU_EFFORT.nodes);
  elements.customGpuTimeMsInput.value = String(DEFAULT_CUSTOM_GPU_EFFORT.timeMs);
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
  if (assignment === "bot-cpu-custom") {
    return "The Chronofish";
  }
  if (assignment === "bot-gpu-custom") {
    return customGpuEffort.label;
  }
  const effortName = botEffortName(assignments[color]);
  if (effortName === "custom") {
    return botEffort(assignment)?.label ?? "Bot";
  }
  const backend = botBackendName(assignment);
  const effort = botEffort(assignment);
  const names = effort?.displayNames ?? [];
  return names[stableIndex(`${multiplayer.roomId}:${color}:${backend}:${effortName}`, names.length)]
    ?? effort?.label
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
    customCpuEffort,
    customGpuEffort,
    notation: submittedNotation,
    turns: submittedTurns,
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
    customCpuEffort,
    customGpuEffort,
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

    customCpuEffort = normalizeCustomCpuEffort(state.customCpuEffort ?? customCpuEffort);
    saveCustomCpuEffort(customCpuEffort);
    customGpuEffort = normalizeCustomGpuEffort(state.customGpuEffort ?? customGpuEffort);
    saveCustomGpuEffort(customGpuEffort);
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
    customCpuEffort,
    customGpuEffort,
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
  clearPlannedMoveTree();
  botReviewProjection = null;
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

function sameMove(left: Move, right: Move): boolean {
  return samePosition(left.from, right.from) && samePosition(left.to, right.to);
}

function boardKey(timelineId: number, board: BoardSnapshot): string {
  return `${timelineId}:${board.time}`;
}

function boardSnapshotKey(board: BoardSnapshot): string {
  return JSON.stringify({
    sideToMove: board.sideToMove,
    castling: board.castling,
    enPassant: board.enPassant,
    origin: board.origin,
    board: board.board
  });
}

function boardAt(snapshot: GameSnapshot, timelineId: number, time: number): BoardSnapshot | null {
  return snapshot.timelines
    .find((timeline) => timeline.id === timelineId)
    ?.boards.find((board) => board.time === time) ?? null;
}

function snapshotHasBoard(snapshot: GameSnapshot, position: Position): boolean {
  return Boolean(boardAt(snapshot, position.timelineId, position.time));
}

function plannedBaseSnapshot(nodeId: string | null): GameSnapshot {
  return nodeId ? plannedMoveNodes.get(nodeId)?.afterSnapshot ?? committedGame : committedGame;
}

function restoreVisibleEngineState(): void {
  if (engine) {
    syncEngineSnapshot(game);
  }
}

function previewPlannedMove(beforeSnapshot: GameSnapshot, move: Move): GameSnapshot | null {
  if (!engine) {
    return null;
  }
  const wasm = activeEngine();
  try {
    syncEngineSnapshot(beforeSnapshot);
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
      return null;
    }
    const movedSnapshot = engineSnapshot();
    if (wasm.chronofish_submit_turn()) {
      return engineSnapshot();
    }
    return movedSnapshot;
  } finally {
    restoreVisibleEngineState();
  }
}

function changedGhostBoards(before: GameSnapshot, after: GameSnapshot, nodeId: string, kind: PlannedMoveNode["kind"]): GhostBoard[] {
  const ghosts: GhostBoard[] = [];
  const seen = new Set<string>();
  for (const timeline of after.timelines) {
    for (const board of timeline.boards) {
      const key = boardKey(timeline.id, board);
      const previous = boardAt(before, timeline.id, board.time);
      if (!previous || boardSnapshotKey(previous) !== boardSnapshotKey(board)) {
        const dedupeKey = `${nodeId}:${key}`;
        if (!seen.has(dedupeKey)) {
          ghosts.push({
            nodeId,
            timelineId: timeline.id,
            board,
            kind
          });
          seen.add(dedupeKey);
        }
      }
    }
  }
  return ghosts;
}

function addPlannedMove(parentId: string | null, move: Move, beforeSnapshot: GameSnapshot, kind: PlannedMoveNode["kind"]): PlannedMoveNode | null {
  const afterSnapshot = previewPlannedMove(beforeSnapshot, move);
  if (!afterSnapshot) {
    return null;
  }
  const id = `plan-${nextPlannedMoveNodeId++}`;
  const node: PlannedMoveNode = {
    id,
    parentId,
    move: cloneMove(move),
    beforeSnapshot: cloneGame(beforeSnapshot),
    afterSnapshot: cloneGame(afterSnapshot),
    children: [],
    kind
  };
  plannedMoveNodes.set(id, node);
  if (parentId) {
    plannedMoveNodes.get(parentId)?.children.push(id);
  } else {
    plannedMoveRoots.push(id);
  }
  return node;
}

function clearPlannedMoveTree(): void {
  plannedMoveAnchor = null;
  plannedMoveNodes = new Map();
  plannedMoveRoots = [];
  nextPlannedMoveNodeId = 1;
}

function clearPlannedMoves(): void {
  clearPlannedMoveTree();
  if (botReviewProjection) {
    game = botReviewProjection.finalGame;
    committedGame = botReviewProjection.finalCommittedGame;
    botReviewProjection = null;
    if (engine) {
      syncEngineSnapshot(game);
    }
    elements.message.textContent = "Returned to final position.";
  } else {
    elements.message.textContent = "Cleared planned moves.";
  }
  selected = null;
  legalTargets = [];
  render({ preserveScroll: true });
}

function reachablePlannedNodeIds(roots: string[]): Set<string> {
  const reachable = new Set<string>();
  const stack = [...roots];
  while (stack.length) {
    const id = stack.pop();
    if (!id || reachable.has(id)) {
      continue;
    }
    reachable.add(id);
    stack.push(...(plannedMoveNodes.get(id)?.children ?? []));
  }
  return reachable;
}

function prunePlannedMovesForCommittedTurn(turn: Move[]): void {
  if (turn.length === 0 || plannedMoveRoots.length === 0) {
    return;
  }
  let candidates = plannedMoveRoots
    .map((id) => plannedMoveNodes.get(id))
    .filter((node): node is PlannedMoveNode => Boolean(node));
  let matched: PlannedMoveNode | null = null;
  for (const move of turn) {
    const next = candidates.find((node) => sameMove(node.move, move));
    if (!next) {
      clearPlannedMoveTree();
      return;
    }
    matched = next;
    candidates = next.children
      .map((id) => plannedMoveNodes.get(id))
      .filter((node): node is PlannedMoveNode => Boolean(node));
  }
  plannedMoveAnchor = null;
  plannedMoveRoots = matched?.children.slice() ?? [];
  for (const root of plannedMoveRoots) {
    const node = plannedMoveNodes.get(root);
    if (node) {
      node.parentId = null;
    }
  }
  const reachable = reachablePlannedNodeIds(plannedMoveRoots);
  plannedMoveNodes = new Map(Array.from(plannedMoveNodes.entries()).filter(([id]) => reachable.has(id)));
}

function collectPlannedRenderData(): { plannedArrows: PlannedArrow[]; ghostBoards: GhostBoard[] } {
  const plannedArrows: PlannedArrow[] = [];
  const ghostBoards: GhostBoard[] = [];
  const visit = (id: string) => {
    const node = plannedMoveNodes.get(id);
    if (!node) {
      return;
    }
    plannedArrows.push({
      from: node.move.from,
      to: node.move.to,
      kind: node.kind
    });
    ghostBoards.push(...changedGhostBoards(node.beforeSnapshot, node.afterSnapshot, node.id, node.kind));
    for (const child of node.children) {
      visit(child);
    }
  };
  for (const root of plannedMoveRoots) {
    visit(root);
  }
  return { plannedArrows, ghostBoards };
}

function buildBotReviewPlan(decision: BotDecisionRecord, baseSnapshot: GameSnapshot, skipTurns = 0, skipMovesInFirstTurn = 0): void {
  clearPlannedMoveTree();
  let parentId: string | null = null;
  let snapshot = cloneGame(baseSnapshot);
  for (let turnIndex = skipTurns; turnIndex < decision.principalVariation.length; turnIndex += 1) {
    const turn = decision.principalVariation[turnIndex] ?? [];
    const moves = turnIndex === skipTurns ? turn.slice(skipMovesInFirstTurn) : turn;
    for (const move of moves) {
      const node = addPlannedMove(parentId, move, snapshot, "bot-review");
      if (!node) {
        return;
      }
      parentId = node.id;
      snapshot = node.afterSnapshot;
    }
  }
}

function botReviewPlanForBoard(position: Position, snapshot: GameSnapshot): BotReviewPlanMatch | null {
  const clickedBoard = boardAt(snapshot, position.timelineId, position.time);
  if (!clickedBoard) {
    return null;
  }
  let bestMatch: BotReviewPlanMatch | null = null;
  let bestReplayOffset = Number.POSITIVE_INFINITY;
  for (const decision of botController.allDecisions()) {
    let baseSnapshot: GameSnapshot | null = cloneGame(decision.game);
    let replayOffset = 0;
    for (let turnIndex = 0; turnIndex < decision.principalVariation.length; turnIndex += 1) {
      const turn = decision.principalVariation[turnIndex] ?? [];
      for (let moveIndex = 0; moveIndex < turn.length; moveIndex += 1) {
        if (!baseSnapshot) {
          break;
        }
        const move = turn[moveIndex];
        if (!move) {
          continue;
        }
        const decisionBoard = boardAt(baseSnapshot, move.from.timelineId, move.from.time);
        if (
          move.from.timelineId === position.timelineId
          && move.from.time === position.time
          && decisionBoard
          && boardSnapshotKey(decisionBoard) === boardSnapshotKey(clickedBoard)
        ) {
          const match = {
            decision,
            baseSnapshot: cloneGame(baseSnapshot),
            skipTurns: turnIndex,
            skipMovesInFirstTurn: moveIndex
          };
          if (replayOffset < bestReplayOffset) {
            bestMatch = match;
            bestReplayOffset = replayOffset;
          }
          if (bestReplayOffset === 0) {
            return bestMatch;
          }
        }
        const afterSnapshot = previewPlannedMove(baseSnapshot, move);
        if (!afterSnapshot) {
          baseSnapshot = null;
          break;
        }
        baseSnapshot = afterSnapshot;
        replayOffset += 1;
      }
      if (!baseSnapshot) {
        break;
      }
    }
  }
  return bestMatch;
}

function showBotPlanForPosition(position: Position): boolean {
  if (phase !== "review") {
    return false;
  }
  const reviewSnapshot = botReviewProjection?.finalGame ?? game;
  const match = botReviewPlanForBoard(position, reviewSnapshot);
  if (!match) {
    return false;
  }
  const baseSnapshot = cloneGame(match.baseSnapshot);
  botReviewProjection = {
    finalGame: botReviewProjection?.finalGame ?? cloneGame(reviewSnapshot),
    finalCommittedGame: botReviewProjection?.finalCommittedGame ?? cloneGame(committedGame),
    decision: match.decision
  };
  game = cloneGame(baseSnapshot);
  committedGame = cloneGame(baseSnapshot);
  buildBotReviewPlan(match.decision, baseSnapshot, match.skipTurns, match.skipMovesInFirstTurn);
  elements.message.textContent = `${botDisplayName(match.decision.botColor)} depth ${match.decision.selectedDepth ?? "?"} plan from turn ${match.decision.ply}.`;
  const focusPosition = snapshotHasBoard(baseSnapshot, position)
    ? position
    : match.decision.selectedMoves[0]?.from
    ?? (snapshotHasBoard(baseSnapshot, position) ? position : null);
  render(focusPosition ? { focusPosition } : {});
  return true;
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

function activeLegalTargetsFor(nodeId: string | null = null): Position[] {
  if (!plannedMoveAnchor) {
    return legalTargets;
  }
  return plannedMoveAnchor.nodeId === nodeId ? plannedMoveAnchor.targets : [];
}

function targetFor(position: Position, nodeId: string | null = null): Position | undefined {
  return activeLegalTargetsFor(nodeId).find((target) => samePosition(target, position));
}

function legalTargetsFor(position: Position, snapshot: GameSnapshot = game): Promise<LegalTargetSelection> {
  if (!engine) {
    return Promise.resolve({ source: null, targets: [] });
  }
  syncEngineSnapshot(snapshot);
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
  prunePlannedMovesForCommittedTurn(submitted);
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

async function handlePlannedMoveClick(position: Position, nodeId: string | null): Promise<void> {
  if (!engine) {
    return;
  }
  if (!plannedMoveAnchor) {
    const snapshot = plannedBaseSnapshot(nodeId);
    const requestId = ++legalTargetRequestId;
    selected = null;
    legalTargets = [];
    render({ preserveScroll: true });
    try {
      const selection = await legalTargetsFor(position, snapshot);
      restoreVisibleEngineState();
      if (requestId !== legalTargetRequestId) {
        return;
      }
      if (!selection.source) {
        plannedMoveAnchor = null;
        elements.message.textContent = "Select a piece on a planned board.";
        render({ preserveScroll: true });
        return;
      }
      plannedMoveAnchor = {
        nodeId,
        position,
        targets: selection.targets ?? []
      };
      const piece = selection.source.piece;
      elements.message.textContent = `Planning ${capitalize(piece.color)} ${piece.type}. ${plannedMoveAnchor.targets.length} legal target${plannedMoveAnchor.targets.length === 1 ? "" : "s"}.`;
      render({ preserveScroll: true });
      return;
    } catch (error) {
      restoreVisibleEngineState();
      if (requestId !== legalTargetRequestId) {
        return;
      }
      plannedMoveAnchor = null;
      elements.message.textContent = errorMessage(error);
      render({ preserveScroll: true });
      return;
    }
  }

  const parentId = plannedMoveAnchor.nodeId;
  if (!targetFor(position, parentId)) {
    elements.message.textContent = "Shift-click a highlighted planned target.";
    render({ preserveScroll: true });
    return;
  }
  const move = {
    from: { ...plannedMoveAnchor.position },
    to: { ...position }
  };
  const beforeSnapshot = plannedBaseSnapshot(parentId);
  const node = addPlannedMove(parentId, move, beforeSnapshot, "planned");
  plannedMoveAnchor = null;
  selected = null;
  legalTargets = [];
  elements.message.textContent = node
    ? "Planned move added."
    : "That planned move is not legal from the selected ghost state.";
  render({ preserveScroll: true });
}

async function handleSquareClick(position: Position, event: MouseEvent, nodeId: string | null = null): Promise<void> {
  if (!engine) {
    elements.message.textContent = "Build the WASM engine first with `cargo build --manifest-path engine/Cargo.toml --target wasm32-unknown-unknown`.";
    return;
  }

  if (event.shiftKey && phase !== "review") {
    await handlePlannedMoveClick(position, nodeId);
    return;
  }

  if (phase === "review" && showBotPlanForPosition(position)) {
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
  elements.clearPlansButton.disabled = plannedMoveNodes.size === 0 && !botReviewProjection;
  elements.concedeButton.disabled = !canActNow();
  training.renderButtons();
  const plannedRenderData = collectPlannedRenderData();

  // State and IO live here; renderGame only rebuilds the DOM from supplied data.
  renderGame({
    game,
    presentGame: committedGame,
    selected: plannedMoveAnchor?.position ?? selected,
    legalTargets: activeLegalTargetsFor(plannedMoveAnchor?.nodeId ?? null),
    plannedSelectionNodeId: plannedMoveAnchor?.nodeId ?? null,
    multiplayer,
    elements,
    plannedArrows: plannedRenderData.plannedArrows,
    ghostBoards: plannedRenderData.ghostBoards,
    onSquareClick: handleSquareClick,
    setMultiplayerStatus
  });
  evaluationUi.renderEvaluationBar();

  if (scrollState) {
    restoreScrollState(scrollState);
  }
  if (options.focusPosition) {
    scrollBoardIntoView(options.focusPosition);
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

function scrollBoardIntoView(position: Position): void {
  const key = `${position.timelineId}:${position.time}:${position.x}:${position.y}`;
  const square = Array.from(elements.timelineGrid.querySelectorAll<HTMLElement>("[data-position-key]"))
    .find((candidate) => candidate.dataset.positionKey === key);
  const board = square?.closest<HTMLElement>(".board-card");
  board?.scrollIntoView({
    block: "center",
    inline: "nearest",
    behavior: "smooth"
  });
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
  const previousTurnCount = submittedTurns.length;

  if (isRoomGamePayload(room.game) && room.game.phase) {
    phase = room.game.phase;
  }
  if (isRoomGamePayload(room.game) && room.game.assignments) {
    customCpuEffort = normalizeCustomCpuEffort(room.game.customCpuEffort ?? customCpuEffort);
    saveCustomCpuEffort(customCpuEffort);
    customGpuEffort = normalizeCustomGpuEffort(room.game.customGpuEffort ?? customGpuEffort);
    saveCustomGpuEffort(customGpuEffort);
    writeAssignments(room.game.assignments);
  }

  if (isRoomGamePayload(room.game) && room.game.snapshot) {
    game = room.game.snapshot;
    committedGame = game;
    submittedTurns = room.game.turns ?? [];
    submittedNotation = room.game.notation ?? "";
    stagedMoves = [];
    if (submittedTurns.length > previousTurnCount) {
      prunePlannedMovesForCommittedTurn(submittedTurns.at(-1) ?? []);
    }
  } else if (room.game && "timelines" in room.game) {
    game = room.game as GameSnapshot;
    committedGame = game;
    submittedTurns = [];
    submittedNotation = "";
    stagedMoves = [];
    clearPlannedMoveTree();
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
  plannedMoveAnchor = null;
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
    const [versionResponse, parametersResponse, cpuEffortResponse, gpuEffortResponse] = await Promise.all([
      fetch("/api/version"),
      fetch("/ai/parameters.json"),
      fetch("/ai/effort.json"),
      fetch("/ai/gpu-effort.json")
    ]);
    const payload = await versionResponse.json() as { version?: string; error?: string };

    if (!versionResponse.ok) {
      throw new Error(payload.error ?? "Server unavailable");
    }
    if (parametersResponse.ok) {
      aiParameters = await parametersResponse.json();
    }
    if (cpuEffortResponse.ok) {
      cpuEffortConfigs = await cpuEffortResponse.json() as Record<BotPresetName, BotEffortConfig>;
    }
    if (gpuEffortResponse.ok) {
      gpuEffortConfigs = await gpuEffortResponse.json() as Record<BotPresetName, BotEffortConfig>;
    }

    elements.serverStatus.textContent = `🖥 v${payload.version}`;
    elements.serverStatus.dataset.state = "ready";
    void training.loadStatus();
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
  clearPlannedMoves,
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
  openCustomCpuModal,
  closeCustomCpuModal,
  applyCustomCpuModal,
  resetCustomCpuModal,
  openCustomGpuModal,
  closeCustomGpuModal,
  applyCustomGpuModal,
  resetCustomGpuModal,
  writeAssignments,
  readAssignments,
  syncLobby
});

void (async () => {
  await loadServerStatus();
  await loadWasmStatus();
})();
setHudCollapsed(localStorage.getItem("chronofish.hudCollapsed") === "true");
render();
