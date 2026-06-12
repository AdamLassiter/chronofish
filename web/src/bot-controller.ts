import { elements } from "./dom.js";
import type { Color, GameSnapshot, Move, Piece, PieceType, Position } from "./types.js";

const GPU_MODE_STORAGE_KEY = "chronofish.gpuMode";

type BotColor = Color;
type AiStatus = "ok" | "noLegalTurn" | string;
type BotBackend = "gpu" | "cpu";

interface BotCredentials {
  color: BotColor;
  token: string;
}

interface BotEffort {
  depth?: number;
  nodes?: number;
  timeMs?: number;
}

interface AiChoice {
  moves?: Move[];
  score?: number | null;
  depth?: number | null;
  nodes?: number | null;
  gpuSearch?: string | null;
  cpuSearch?: string | null;
}

type PrincipalVariation = Move[][];

interface AiSearchResult {
  status: AiStatus;
  moves: Move[];
  score?: number | null;
  depth?: number | null;
  nodes?: number | null;
  choices?: AiChoice[];
  principalVariation?: PrincipalVariation;
  gpuSearch?: string | null;
  cpuSearch?: string | null;
  trainingDecision?: BotDecisionRecord | null;
}

interface PendingResult {
  result: AiSearchResult;
  partitionIndex: number | null;
  depth: number;
}

interface PendingSearch {
  id: number;
  botColor: BotColor;
  backend: BotBackend;
  game: GameSnapshot;
  targetDepth: number;
  currentDepth: number;
  workerCount: number;
  nodes: number;
  timeMs: number;
  depthExpected: number;
  depthReceived: number;
  depthResults: PendingResult[];
  bestByDepth: Map<number, AiSearchResult>;
  deadlineAt: number;
  results: PendingResult[];
  errors: string[];
}

interface AiWorkerResponse {
  id: number;
  ok: boolean;
  result?: AiSearchResult;
  error?: string;
  partitionIndex?: number | null;
}

interface RankedBotChoice {
  moves: Move[];
  score?: number | null | undefined;
  depth?: number | null | undefined;
  nodes?: number | null | undefined;
  gpuSearch?: string | null | undefined;
  cpuSearch?: string | null | undefined;
  partitionIndex: number | null;
  selected: boolean;
}

interface BotDecisionChoice {
  moves: Move[];
  score: number | null;
  depth: number | null;
  nodes: number | null;
  gpuSearch: string | null;
  cpuSearch: string | null;
}

export interface BotDecisionRecord {
  ply: number;
  botColor: BotColor;
  effort: string;
  game: GameSnapshot;
  selectedMoves: Move[];
  selectedScore: number | null;
  selectedDepth: number | null;
  selectedNodes: number | null;
  principalVariation: PrincipalVariation;
  choices: BotDecisionChoice[];
}

interface BotState {
  thinking: boolean;
  timeoutId: ReturnType<typeof setTimeout> | null;
  countdownId: ReturnType<typeof setInterval> | null;
  pendingSearch: PendingSearch | null;
  tokens: Partial<Record<BotColor, string>>;
}

interface RoomJoinPayload {
  room: unknown;
}

interface BotControllerOptions {
  getEngine(): unknown;
  getPhase(): string;
  getAssignments(): Record<BotColor, unknown>;
  getGame(): GameSnapshot;
  getStagedMoves(): Move[];
  getSubmittedTurns(): Move[][];
  getRoomId(): string;
  isBotAssignment(value: unknown): boolean;
  botEffortName(value: unknown): string;
  botEffort(value: unknown): BotEffort | null;
  botDisplayName(color: BotColor): string;
  cloneGame(game: GameSnapshot): GameSnapshot;
  cloneMove(move: Move): Move;
  turnSignature(): string;
  postRoom(action: string, body: unknown): Promise<RoomJoinPayload>;
  lobbyPayload(): unknown;
  applyRemoteRoom(room: unknown): void;
  applyEngineMove(from: Position, to: Position): Promise<string | null>;
  submitVisibleTurn(actor: BotColor): Promise<string | null>;
  concede(color: BotColor, credentials?: BotCredentials): void;
  persistLocalGameState(): void;
  render(): void;
  isMatchOver(): boolean;
  enterPostMatchReview(message: string, credentials?: BotCredentials): void;
  syncState(action: string, message: string, credentials?: BotCredentials): void;
}

export interface BotController {
  colors(): BotColor[];
  seat(color: BotColor): Promise<void>;
  maybeStartTurn(): void;
  resetAiWorker(): void;
  moveCredentials(color: BotColor): BotCredentials;
  decisionsFor(color: BotColor): BotDecisionRecord[];
  allDecisions(): BotDecisionRecord[];
  clearDecisionLog(): void;
}

export function createBotController({
  getEngine,
  getPhase,
  getAssignments,
  getGame,
  getStagedMoves,
  getSubmittedTurns,
  getRoomId,
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
}: BotControllerOptions): BotController {
  const message = requireElement(elements.message, "message");
  let botDecisionLog: BotDecisionRecord[] = [];
  let aiWorkers: Worker[] = [];
  let aiRequestId = 0;
  const bot: BotState = {
    thinking: false,
    timeoutId: null,
    countdownId: null,
    pendingSearch: null,
    tokens: {}
  };

  function botToken(color: BotColor): string {
    const key = `chronofish.botToken.${getRoomId()}.${color}`;
    let token = localStorage.getItem(key);
    if (!token) {
      token = crypto.randomUUID();
      localStorage.setItem(key, token);
    }
    return token;
  }

  function botColors(): BotColor[] {
    return (["white", "black"] as BotColor[]).filter((color) => isBotAssignment(getAssignments()[color]));
  }

  function createAiWorker(backend: BotBackend): Worker {
    const worker = new Worker(backend === "cpu" ? "./cpu-ai-worker.js" : "./ai-worker.js", { type: "module" });
    worker.addEventListener("message", handleAiWorkerMessage);
    worker.addEventListener("error", handleAiWorkerError);
    worker.addEventListener("messageerror", handleAiWorkerError);
    aiWorkers.push(worker);
    return worker;
  }

  function handleAiWorkerError(event: ErrorEvent | MessageEvent): void {
    const pending = bot.pendingSearch;
    if (!pending) {
      return;
    }
    handleAiWorkerMessage({
      data: {
        id: pending.id,
        ok: false,
        error: event instanceof ErrorEvent ? event.message : "AI worker failed to load.",
        partitionIndex: null
      }
    } as MessageEvent<AiWorkerResponse>);
  }

  function terminateAiWorkers(): void {
    for (const worker of aiWorkers) {
      worker.terminate();
    }
    aiWorkers = [];
    bot.pendingSearch = null;
  }

  function botSearchWorkerCount(effortName: string, backend: BotBackend): number {
    if (backend === "cpu" || effortName !== "expert") {
      return 1;
    }
    const hardwareThreads = Math.max(1, navigator.hardwareConcurrency ?? 2);
    return Math.max(1, Math.min(2, hardwareThreads - 1));
  }

  function clearBotTimeout(): void {
    if (bot.timeoutId !== null) {
      clearTimeout(bot.timeoutId);
      bot.timeoutId = null;
    }
    if (bot.countdownId !== null) {
      clearInterval(bot.countdownId);
      bot.countdownId = null;
    }
  }

  function formatBotTimeLimit(ms: number): string {
    const seconds = Math.max(0, Math.ceil(ms / 1000));
    return `${seconds}s`;
  }

  function botMoveCredentials(color: BotColor): BotCredentials {
    return { color, token: bot.tokens[color] ?? botToken(color) };
  }

  function updateBotCountdownMessage(id: number): void {
    const pending = bot.pendingSearch;
    if (!bot.thinking || !pending || pending.id !== id) {
      return;
    }
    const remainingMs = Math.max(0, pending.deadlineAt - Date.now());
    const workerText = `${pending.workerCount} worker${pending.workerCount === 1 ? "" : "s"}`;
    const bestDepth = deepestStoredDepth(pending);
    const bestText = bestDepth > 0 ? ` Best depth ${bestDepth}.` : "";
    message.textContent = `${botDisplayName(pending.botColor)} searching depth ${pending.currentDepth}/${pending.targetDepth} across ${workerText}. ${formatBotTimeLimit(remainingMs)} left.${bestText}`;
  }

  async function seatBot(color: BotColor): Promise<void> {
    bot.tokens[color] = botToken(color);
    const payload = await postRoom("join", {
      color,
      token: bot.tokens[color],
      game: lobbyPayload()
    });

    applyRemoteRoom(payload.room);
  }

  function maybeStartBotTurn(): void {
    if (
      !getEngine() ||
      getPhase() !== "game" ||
      !isBotAssignment(getAssignments()[getGame().turn]) ||
      bot.thinking ||
      getStagedMoves().length > 0
    ) {
      return;
    }

    const id = ++aiRequestId;
    const botColor = getGame().turn;
    const effortName = botEffortName(getAssignments()[botColor]);
    const backend = botBackend(getAssignments()[botColor]);
    const effort = botEffort(getAssignments()[botColor]);
    if (!effort) {
      return;
    }
    const timeMs = Math.max(1, effort.timeMs ?? 10_000);
    const workerCount = botSearchWorkerCount(effortName, backend);
    terminateAiWorkers();
    bot.thinking = true;
    bot.pendingSearch = {
      id,
      botColor,
      backend,
      game: cloneGame(getGame()),
      targetDepth: Math.max(1, effort.depth ?? 1),
      currentDepth: 0,
      workerCount,
      nodes: Math.max(1, effort.nodes ?? 64),
      timeMs,
      depthExpected: 0,
      depthReceived: 0,
      depthResults: [],
      bestByDepth: new Map(),
      deadlineAt: Date.now() + timeMs,
      results: [],
      errors: []
    };
    clearBotTimeout();
    bot.timeoutId = setTimeout(() => handleBotTimeout(id, botColor, timeMs), timeMs);
    bot.countdownId = setInterval(() => updateBotCountdownMessage(id), 250);
    launchNextBotDepth(id);
    updateBotCountdownMessage(id);
  }

  function launchNextBotDepth(id: number): void {
    const pending = bot.pendingSearch;
    if (!bot.thinking || !pending || pending.id !== id) {
      return;
    }
    const nextDepth = pending.currentDepth + 1;
    if (nextDepth > pending.targetDepth || Date.now() >= pending.deadlineAt) {
      finishBotSearch(pending, Date.now() >= pending.deadlineAt ? "timeout" : "complete");
      return;
    }
    pending.currentDepth = nextDepth;
    pending.depthExpected = 0;
    pending.depthReceived = 0;
    pending.depthResults = [];
    const remainingMs = Math.max(1, pending.deadlineAt - Date.now());
    const workerTimeMs = botWorkerSearchTimeMs(remainingMs);
    for (let partitionIndex = 0; partitionIndex < pending.workerCount; partitionIndex += 1) {
      try {
        createAiWorker(pending.backend).postMessage({
          id,
          game: getGame(),
          depth: nextDepth,
          nodes: pending.nodes,
          timeMs: workerTimeMs,
          gpuMode: botGpuMode(),
          partitionIndex,
          partitionCount: pending.workerCount
        });
        pending.depthExpected += 1;
      } catch (error: unknown) {
        console.error(error);
        pending.errors.push(errorMessage(error));
      }
    }
    if (pending.depthExpected === 0) {
      pending.depthExpected = 1;
      setTimeout(() => {
        handleAiWorkerMessage({
          data: {
            id,
            ok: false,
            error: "GPU worker search is unavailable.",
            partitionIndex: 0
          }
        } as MessageEvent<AiWorkerResponse>);
      }, 0);
    }
    updateBotCountdownMessage(id);
  }

  function botWorkerSearchTimeMs(timeMs: number): number {
    const margin = Math.min(1000, Math.max(100, Math.floor(timeMs * 0.05)));
    return Math.max(1, timeMs - margin);
  }

  function botGpuMode(): "full" | "hybrid" {
    return localStorage.getItem(GPU_MODE_STORAGE_KEY) === "full" ? "full" : "hybrid";
  }

  function botBackend(value: unknown): BotBackend {
    return typeof value === "string" && value.startsWith("bot-cpu-") ? "cpu" : "gpu";
  }

  function handleBotTimeout(id: number, botColor: BotColor, timeMs: number): void {
    if (id !== aiRequestId || !bot.thinking) {
      return;
    }

    const pending = bot.pendingSearch;
    if (pending) {
      finishBotSearch(pending, "timeout");
    } else {
      aiRequestId += 1;
      bot.thinking = false;
      clearBotTimeout();
      terminateAiWorkers();
      message.textContent = `${botDisplayName(botColor)} found no legal turn in ${formatBotTimeLimit(timeMs)}.`;
      void completeBotTurn(botColor, { status: "noLegalTurn", moves: [] });
    }
  }

  function handleAiWorkerMessage(event: MessageEvent<AiWorkerResponse>): void {
    const { id, ok, result, error, partitionIndex } = event.data;
    if (id !== aiRequestId) {
      return;
    }

    const pending = bot.pendingSearch;
    if (!pending || pending.id !== id) {
      return;
    }

    pending.depthReceived += 1;
    if (ok && result) {
      const receivedDepth = pending.currentDepth;
      const depthResult = { ...result, depth: receivedDepth };
      const entry = { result: depthResult, partitionIndex: partitionIndex ?? null, depth: receivedDepth };
      pending.results.push(entry);
      pending.depthResults.push(entry);
    } else {
      pending.errors.push(error ?? "AI worker returned no result.");
    }

    if (pending.depthReceived < pending.depthExpected) {
      return;
    }

    const depthBest = selectBestAiResult(pending.depthResults.map((entry) => entry.result));
    if (depthBest) {
      pending.bestByDepth.set(pending.currentDepth, depthBest);
    }

    if (pending.currentDepth >= pending.targetDepth || Date.now() >= pending.deadlineAt) {
      finishBotSearch(pending, Date.now() >= pending.deadlineAt ? "timeout" : "complete");
      return;
    }

    launchNextBotDepth(pending.id);
  }

  function finishBotSearch(pending: PendingSearch, reason: "complete" | "timeout"): void {
    if (pending.id !== aiRequestId || !bot.thinking) {
      return;
    }
    const bestResult = selectDeepestStoredResult(pending)
      ?? selectBestAiResult(pending.results.map((entry) => entry.result));
    aiRequestId += 1;
    bot.thinking = false;
    clearBotTimeout();
    logBotSearchChoices(pending, bestResult, reason);
    if (!bestResult && pending.errors.length > 0) {
      terminateAiWorkers();
      message.textContent = `${botDisplayName(pending.botColor)} search failed and conceded: ${pending.errors[0] ?? "unknown error"}`;
      void completeBotTurn(pending.botColor, { status: "noLegalTurn", moves: [] });
      return;
    }
    if (bestResult) {
      bestResult.trainingDecision = buildBotDecisionRecord(pending, bestResult);
    }
    terminateAiWorkers();
    if (bestResult && reason === "timeout") {
      message.textContent = `${botDisplayName(pending.botColor)} used depth ${bestResult.depth ?? deepestStoredDepth(pending)} after ${formatBotTimeLimit(pending.timeMs)}.`;
    }
    void completeBotTurn(pending.botColor, bestResult ?? { status: "noLegalTurn", moves: [] });
  }

  function selectBestAiResult(results: AiSearchResult[]): AiSearchResult | null {
    return results
      .filter((result) => result.status === "ok" && result.moves.length > 0)
      .sort((left, right) => {
        const depth = (right.depth ?? 0) - (left.depth ?? 0);
        if (depth !== 0) {
          return depth;
        }
        const score = (right.score ?? -Infinity) - (left.score ?? -Infinity);
        if (score !== 0) {
          return score;
        }
        return (right.nodes ?? 0) - (left.nodes ?? 0);
      })[0] ?? null;
  }

  function selectDeepestStoredResult(pending: PendingSearch): AiSearchResult | null {
    for (let depth = pending.targetDepth; depth >= 1; depth -= 1) {
      const result = pending.bestByDepth.get(depth);
      if (result) {
        return result;
      }
    }
    return null;
  }

  function deepestStoredDepth(pending: PendingSearch): number {
    return Math.max(0, ...pending.bestByDepth.keys());
  }

  function logBotSearchChoices(pending: PendingSearch | null, selectedResult: AiSearchResult | null, reason: string): void {
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
      moves: choice.moves.map((move) => formatBotMove(move, pending.game)).join(" | "),
      depth: choice.depth ?? "",
      nodes: choice.nodes ?? "",
      worker: choice.partitionIndex ?? "",
      search: choice.cpuSearch ?? choice.gpuSearch ?? ""
    })));
    if (pending.errors.length) {
      console.info("Bot search worker errors", pending.errors);
    }
    console.groupEnd();
  }

  function rankedBotChoices(results: PendingResult[], selectedResult: AiSearchResult | null): RankedBotChoice[] {
    const selectedKey = botMovesKey(selectedResult?.moves ?? []);
    const byMoves = new Map<string, RankedBotChoice>();
    for (const entry of results) {
      const result = entry.result;
      const rawChoices = Array.isArray(result.choices) && result.choices.length
        ? result.choices
        : result.moves.length
          ? [{ moves: result.moves, score: result.score, depth: result.depth, nodes: result.nodes, gpuSearch: result.gpuSearch, cpuSearch: result.cpuSearch }]
          : [];
      for (const choice of rawChoices) {
        const moves = choice.moves ?? [];
        const key = botMovesKey(moves);
        if (!key) {
          continue;
        }
        const current = byMoves.get(key);
        const next: RankedBotChoice = {
          moves,
          score: choice.score,
          depth: choice.depth ?? result.depth,
          nodes: choice.nodes ?? result.nodes,
          gpuSearch: choice.gpuSearch ?? result.gpuSearch,
          cpuSearch: choice.cpuSearch ?? result.cpuSearch,
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

  function buildBotDecisionRecord(pending: PendingSearch | null, result: AiSearchResult): BotDecisionRecord | null {
    if (!pending || result.status !== "ok" || !result.moves.length) {
      return null;
    }
    return {
      ply: getSubmittedTurns().length + 1,
      botColor: pending.botColor,
      effort: botEffortName(getAssignments()[pending.botColor]),
      game: cloneGame(pending.game ?? getGame()),
      selectedMoves: result.moves.map(cloneMove),
      selectedScore: result.score ?? null,
      selectedDepth: result.depth ?? null,
      selectedNodes: result.nodes ?? null,
      principalVariation: normalizePrincipalVariation(result.principalVariation, result.moves),
      choices: rankedBotChoices(pending.results, result).map((choice) => ({
        moves: choice.moves.map(cloneMove),
        score: choice.score ?? null,
        depth: choice.depth ?? null,
        nodes: choice.nodes ?? null,
        gpuSearch: choice.gpuSearch ?? null,
        cpuSearch: choice.cpuSearch ?? null
      }))
    };
  }

  function normalizePrincipalVariation(variation: PrincipalVariation | undefined, fallback: Move[]): PrincipalVariation {
    const cleaned = (variation ?? [])
      .map((turn) => turn.filter((move) => move?.from && move.to).map(cloneMove))
      .filter((turn) => turn.length > 0);
    return cleaned.length ? cleaned : [fallback.map(cloneMove)];
  }

  function recordBotDecision(result: AiSearchResult): void {
    if (result.trainingDecision) {
      botDecisionLog.push(result.trainingDecision);
    }
  }

  function botChoiceScore(choice: Pick<RankedBotChoice, "score">): number {
    return Number.isFinite(choice.score) ? choice.score ?? -Infinity : -Infinity;
  }

  function botMovesKey(moves: Move[]): string {
    return moves.map((move) => formatBotMove(move)).join("|");
  }

  function formatBotMove(move: Move, game: GameSnapshot | null = null): string {
    if (!move.from || !move.to) {
      return "?";
    }
    const piece = game ? botPieceAt(game, move.from) : null;
    const destinationPrefix = move.to.timelineId === move.from.timelineId && move.to.time === move.from.time
      ? ""
      : `T${move.to.time ?? "?"}L${move.to.timelineId ?? ""}`;
    return `${formatBotSquarePrefix(move.from)}${formatBotPieceLetter(piece)}${formatBotCoordinate(move.from)}-${destinationPrefix}${formatBotCoordinate(move.to)}`;
  }

  function formatBotSquarePrefix(square: Position): string {
    return `T${square.time ?? "?"}L${square.timelineId ?? "?"}`;
  }

  function formatBotCoordinate(square: Position): string {
    const file = typeof square.x === "number" ? String.fromCharCode(97 + square.x) : "?";
    const rank = typeof square.y === "number" ? String(8 - square.y) : "?";
    return `${file}${rank}`;
  }

  function formatBotPieceLetter(piece: Piece | null): string {
    if (!piece?.type) {
      return "";
    }
    const letters: Record<PieceType, string> = {
      king: "k",
      commonKing: "k",
      queen: "q",
      royalQueen: "q",
      princess: "s",
      rook: "r",
      bishop: "b",
      unicorn: "u",
      dragon: "d",
      knight: "n",
      pawn: "p",
      brawn: "w"
    };
    return letters[piece.type] ?? "";
  }

  function botPieceAt(game: GameSnapshot, position: Position): Piece | null {
    const timeline = game.timelines.find((candidate) => candidate.id === position.timelineId);
    const board = timeline?.boards.find((candidate) => candidate.time === position.time);
    return board?.board[position.y]?.[position.x] ?? null;
  }

  function formatBotEvaluation(score: number | null | undefined): string {
    if (!Number.isFinite(score)) {
      return "?";
    }
    const finiteScore = score ?? 0;
    return `${finiteScore > 0 ? "+" : ""}${(finiteScore / 100).toFixed(2)}`;
  }

  async function completeBotTurn(botColor: BotColor, result: AiSearchResult): Promise<void> {
    if (!isBotAssignment(getAssignments()[botColor]) || getStagedMoves().length > 0) {
      return;
    }

    if (result.status !== "ok" || result.moves.length === 0) {
      message.textContent = `${botDisplayName(botColor)} found no legal turn and conceded.`;
      await concede(botColor, botMoveCredentials(botColor));
      return;
    }

    recordBotDecision(result);
    const before = turnSignature();
    for (const move of result.moves) {
      if (!(await applyEngineMove(move.from, move.to))) {
        await concede(botColor, botMoveCredentials(botColor));
        return;
      }
    }

    if (before !== turnSignature()) {
      return;
    }

    const turnMessage = await submitVisibleTurn(botColor);
    if (!turnMessage) {
      await concede(botColor, botMoveCredentials(botColor));
      return;
    }

    const botMessage = `${botDisplayName(botColor)} moved. ${turnMessage}`;
    message.textContent = botMessage;
    persistLocalGameState();
    render();
    if (isMatchOver()) {
      await enterPostMatchReview(turnMessage, botMoveCredentials(botColor));
      return;
    }
    await syncState("state", botMessage, botMoveCredentials(botColor));
    maybeStartBotTurn();
  }

  function resetAiWorker(): void {
    aiRequestId += 1;
    bot.thinking = false;
    clearBotTimeout();
    terminateAiWorkers();
  }

  return {
    colors: botColors,
    seat: seatBot,
    maybeStartTurn: maybeStartBotTurn,
    resetAiWorker,
    moveCredentials: botMoveCredentials,
    decisionsFor: (color: BotColor) => botDecisionLog.filter((decision) => decision.botColor === color),
    allDecisions: () => botDecisionLog.slice(),
    clearDecisionLog: () => {
      botDecisionLog = [];
    }
  };
}

function requireElement<T extends Element>(element: T | null, id: string): T {
  if (!element) {
    throw new Error(`Missing required #${id} element.`);
  }
  return element;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
