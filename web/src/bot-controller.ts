import { elements } from "./dom.js";
import type { Color, GameSnapshot, Move, Piece, PieceType, Position } from "./types.js";

const GPU_MODE_STORAGE_KEY = "chronofish.gpuMode";

type BotColor = Color;
type AiStatus = "ok" | "noLegalTurn" | string;

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
}

interface AiSearchResult {
  status: AiStatus;
  moves: Move[];
  score?: number | null;
  depth?: number | null;
  nodes?: number | null;
  choices?: AiChoice[];
  gpuSearch?: string | null;
  trainingDecision?: BotDecisionRecord | null;
}

interface PendingResult {
  result: AiSearchResult;
  partitionIndex: number | null;
}

interface PendingSearch {
  id: number;
  botColor: BotColor;
  game: GameSnapshot;
  expected: number;
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
  partitionIndex: number | null;
  selected: boolean;
}

interface BotDecisionChoice {
  moves: Move[];
  score: number | null;
  depth: number | null;
  nodes: number | null;
  gpuSearch: string | null;
}

interface BotDecisionRecord {
  ply: number;
  botColor: BotColor;
  effort: string;
  game: GameSnapshot;
  selectedMoves: Move[];
  selectedScore: number | null;
  selectedDepth: number | null;
  selectedNodes: number | null;
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
  botEffort(value: unknown): BotEffort;
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
  isMatchOverMessage(message: string): boolean;
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
  isMatchOverMessage,
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

  function createAiWorker(): Worker {
    const worker = new Worker("./ai-worker.js", { type: "module" });
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

  function botSearchWorkerCount(effortName: string): number {
    if (effortName !== "expert") {
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
    const workerText = `${pending.expected} worker${pending.expected === 1 ? "" : "s"}`;
    message.textContent = `${botDisplayName(pending.botColor)} thinking across ${workerText}. ${formatBotTimeLimit(remainingMs)} left.`;
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
    const effort = botEffort(getAssignments()[botColor]);
    const timeMs = Math.max(1, effort.timeMs ?? 10_000);
    const workerTimeMs = botWorkerSearchTimeMs(timeMs);
    const workerCount = botSearchWorkerCount(effortName);
    terminateAiWorkers();
    bot.thinking = true;
    bot.pendingSearch = {
      id,
      botColor,
      game: cloneGame(getGame()),
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
          game: getGame(),
          depth: effort.depth,
          nodes: effort.nodes,
          timeMs: workerTimeMs,
          gpuMode: botGpuMode(),
          partitionIndex,
          partitionCount: workerCount
        });
        bot.pendingSearch.expected += 1;
      } catch (error: unknown) {
        console.error(error);
        bot.pendingSearch.errors.push(errorMessage(error));
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

  function handleBotTimeout(id: number, botColor: BotColor, timeMs: number): void {
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
      message.textContent = `${botDisplayName(botColor)} used the best move found in ${formatBotTimeLimit(timeMs)}.`;
      void completeBotTurn(botColor, bestResult);
      return;
    }

    terminateAiWorkers();
    message.textContent = `${botDisplayName(botColor)} found no legal turn in ${formatBotTimeLimit(timeMs)}.`;
    void completeBotTurn(botColor, { status: "noLegalTurn", moves: [] });
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

    if (ok && result) {
      pending.results.push({ result, partitionIndex: partitionIndex ?? null });
    } else {
      pending.errors.push(error ?? "AI worker returned no result.");
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
      message.textContent = `${botDisplayName(pending.botColor)} search failed: ${pending.errors[0] ?? "unknown error"}`;
      render();
      return;
    }

    if (bestResult) {
      bestResult.trainingDecision = buildBotDecisionRecord(pending, bestResult);
    }
    terminateAiWorkers();
    void completeBotTurn(pending.botColor, bestResult ?? { status: "noLegalTurn", moves: [] });
  }

  function selectBestAiResult(results: AiSearchResult[]): AiSearchResult | null {
    return results
      .filter((result) => result.status === "ok" && result.moves.length > 0)
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
      search: choice.gpuSearch ?? ""
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
          ? [{ moves: result.moves, score: result.score, depth: result.depth, nodes: result.nodes, gpuSearch: result.gpuSearch }]
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
      choices: rankedBotChoices(pending.results, result).map((choice) => ({
        moves: choice.moves.map(cloneMove),
        score: choice.score ?? null,
        depth: choice.depth ?? null,
        nodes: choice.nodes ?? null,
        gpuSearch: choice.gpuSearch ?? null
      }))
    };
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

    const turnMessage = await submitVisibleTurn(botColor);
    if (!turnMessage) {
      concede(botColor, botMoveCredentials(botColor));
      return;
    }

    const botMessage = `${botDisplayName(botColor)} moved. ${turnMessage}`;
    message.textContent = botMessage;
    persistLocalGameState();
    render();
    if (isMatchOverMessage(turnMessage)) {
      enterPostMatchReview(turnMessage, botMoveCredentials(botColor));
      return;
    }
    syncState("state", botMessage, botMoveCredentials(botColor));
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
