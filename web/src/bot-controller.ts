import { elements } from "./dom.js";
import { readWasmString, writeWasmString } from "./engine-io.js";
import type { ChronofishEngine, Color, GameSnapshot, Move, Piece, PieceType, Position } from "./types.js";

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
  minDepth?: number;
  nodes?: number;
  timeMs?: number;
}

interface BotSearchConfig {
  targetDepth: number;
  minDepth: number;
  nodes: number;
  timeMs: number;
}

interface AiChoice {
  moves?: Move[];
  score?: number | null;
  depth?: number | null;
  nodes?: number | null;
  gpuSearch?: string | null;
  cpuSearch?: string | null;
  principalVariation?: PrincipalVariation;
}

type PrincipalVariation = Move[][];

interface AiSearchResult {
  status: AiStatus;
  moves: Move[];
  score?: number | null;
  depth?: number | null;
  nodes?: number | null;
  terminal?: boolean | null;
  resultReason?: string | null;
  gpuTerminal?: boolean | null;
  choices?: AiChoice[];
  principalVariation?: PrincipalVariation;
  gpuSearch?: string | null;
  gpuDiagnostics?: Record<string, number | string | null | undefined> | null;
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
  minDepth: number;
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
  minimumFallbackStarted: boolean;
  incompleteDepthAttempt: boolean;
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
  principalVariation?: PrincipalVariation | undefined;
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
  principalVariation: PrincipalVariation;
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
  getEngine(): ChronofishEngine | null;
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
    terminateAiWorkerInstances();
    bot.pendingSearch = null;
  }

  function terminateAiWorkerInstances(): void {
    for (const worker of aiWorkers) {
      worker.terminate();
    }
    aiWorkers = [];
  }

  function botSearchWorkerCount(_effortName: string, _backend: BotBackend): number {
    // WebGPU work is intentionally serialized through one worker/device queue.
    // Parallel workers create duplicate adapters and compete for the same GPU.
    return 1;
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

  function formatBotCountdown(deadlineAt: number, now = Date.now()): string {
    const deltaMs = deadlineAt - now;
    if (deltaMs >= 0) {
      return `${formatBotTimeLimit(deltaMs)} left`;
    }
    return `${formatBotTimeLimit(-deltaMs)} overtime`;
  }

  function botMoveCredentials(color: BotColor): BotCredentials {
    return { color, token: bot.tokens[color] ?? botToken(color) };
  }

  function updateBotCountdownMessage(id: number): void {
    const pending = bot.pendingSearch;
    if (!bot.thinking || !pending || pending.id !== id) {
      return;
    }
    const workerText = `${pending.workerCount} worker${pending.workerCount === 1 ? "" : "s"}`;
    const bestDepth = deepestStoredDepth(pending);
    const bestText = bestDepth > 0 ? ` Best depth ${bestDepth}.` : "";
    message.textContent = `${botDisplayName(pending.botColor)} searching depth ${pending.currentDepth}/${pending.targetDepth} across ${workerText}. ${formatBotCountdown(pending.deadlineAt)}.${bestText}`;
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
    const searchConfig = botSearchConfig(effort);
    const workerCount = botSearchWorkerCount(effortName, backend);
    terminateAiWorkers();
    bot.thinking = true;
    bot.pendingSearch = {
      id,
      botColor,
      backend,
      game: cloneGame(getGame()),
      targetDepth: searchConfig.targetDepth,
      minDepth: searchConfig.minDepth,
      currentDepth: 0,
      workerCount,
      nodes: searchConfig.nodes,
      timeMs: searchConfig.timeMs,
      depthExpected: 0,
      depthReceived: 0,
      depthResults: [],
      bestByDepth: new Map(),
      deadlineAt: Date.now() + searchConfig.timeMs,
      results: [],
      errors: [],
      minimumFallbackStarted: false,
      incompleteDepthAttempt: false
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
    const nextDepth = nextBotSearchDepth(pending.currentDepth, pending.targetDepth);
    const completedDepth = deepestStoredDepth(pending);
    if (
      nextDepth <= pending.currentDepth
      || (Date.now() >= pending.deadlineAt && completedDepth >= pending.minDepth)
    ) {
      finishBotSearch(pending, Date.now() >= pending.deadlineAt ? "timeout" : "complete");
      return;
    }
    pending.currentDepth = nextDepth;
    pending.depthExpected = 0;
    pending.depthReceived = 0;
    pending.depthResults = [];
    pending.incompleteDepthAttempt = false;
    const remainingMs = pending.currentDepth <= pending.minDepth
      ? pending.timeMs
      : Math.max(1, pending.deadlineAt - Date.now());
    const workerTimeMs = botWorkerSearchTimeMs(remainingMs);
    for (let partitionIndex = 0; partitionIndex < pending.workerCount; partitionIndex += 1) {
      try {
        createAiWorker(pending.backend).postMessage({
          id,
          game: getGame(),
          depth: nextDepth,
          nodes: pending.nodes,
          timeMs: workerTimeMs,
          minDepth: Math.min(nextDepth, pending.minDepth),
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
    return loadedEngine().chronofish_bot_worker_search_time_ms(timeMs);
  }

  function botSearchConfig(effort: BotEffort): BotSearchConfig {
    const engine = loadedEngine();
    const output = engine.chronofish_bot_search_config_json(
      effort.depth ?? Number.NaN,
      effort.minDepth ?? Number.NaN,
      effort.nodes ?? Number.NaN,
      effort.timeMs ?? Number.NaN
    );
    if (!output) {
      throw new Error(readWasmString(engine, engine.chronofish_last_message()));
    }
    return JSON.parse(readWasmString(engine, output)) as BotSearchConfig;
  }

  function nextBotSearchDepth(currentDepth: number, targetDepth: number): number {
    return loadedEngine().chronofish_bot_next_search_depth(currentDepth, targetDepth);
  }

  function completedSearchDepth(result: AiSearchResult, requestedDepth: number): number | null {
    const depth = result.depth ?? requestedDepth;
    const completedDepth = loadedEngine().chronofish_bot_completed_search_depth(
      depth,
      requestedDepth,
      resultEndsInRoyalCapture(result) ? 1 : 0
    );
    return completedDepth > 0 ? completedDepth : null;
  }

  function resultEndsInRoyalCapture(result: AiSearchResult): boolean {
    return result.resultReason === "royal-capture" || result.gpuTerminal === true || result.terminal === true && result.resultReason === "royal-capture";
  }

  function botGpuMode(): "full" | "hybrid" {
    return localStorage.getItem(GPU_MODE_STORAGE_KEY) === "full" ? "full" : "hybrid";
  }

  function botBackend(value: unknown): BotBackend {
    return typeof value === "string" && value.startsWith("bot-cpu-") ? "cpu" : "gpu";
  }

  function loadedEngine(): ChronofishEngine {
    const engine = getEngine();
    if (!engine) {
      throw new Error("WASM engine is not loaded.");
    }
    return engine;
  }

  function handleBotTimeout(id: number, botColor: BotColor, timeMs: number): void {
    if (id !== aiRequestId || !bot.thinking) {
      return;
    }

    const pending = bot.pendingSearch;
    if (pending) {
      bot.timeoutId = null;
      const bestResult = selectDeepestStoredResult(pending)
        ?? selectBestAiResult(pending.results.map((entry) => entry.result));
      if (bestResult && (bestResult.depth ?? 0) >= pending.minDepth) {
        finishBotSearch(pending, "timeout");
        return;
      }
      if (pending.currentDepth <= pending.minDepth && pending.depthReceived < pending.depthExpected) {
        message.textContent = `${botDisplayName(botColor)} reached ${formatBotTimeLimit(timeMs)} and is completing minimum depth ${pending.minDepth} before moving.`;
        return;
      }
      if (pending.backend === "gpu" && !pending.minimumFallbackStarted) {
        startMinimumDepthCpuFallback(pending);
        return;
      }
      message.textContent = `${botDisplayName(botColor)} reached ${formatBotTimeLimit(timeMs)} and is completing depth ${Math.max(1, pending.currentDepth)} before moving.`;
    } else {
      aiRequestId += 1;
      bot.thinking = false;
      clearBotTimeout();
      terminateAiWorkers();
      message.textContent = `${botDisplayName(botColor)} found no legal turn in ${formatBotTimeLimit(timeMs)}.`;
      void completeBotTurn(botColor, { status: "noLegalTurn", moves: [] });
    }
  }

  function startMinimumDepthCpuFallback(pending: PendingSearch): void {
    terminateAiWorkerInstances();
    pending.minimumFallbackStarted = true;
    pending.currentDepth = pending.minDepth;
    pending.depthExpected = 1;
    pending.depthReceived = 0;
    pending.depthResults = [];
    pending.incompleteDepthAttempt = false;
    message.textContent = `${botDisplayName(pending.botColor)} reached ${formatBotTimeLimit(pending.timeMs)} and is completing minimum depth ${pending.minDepth} on the CPU before moving.`;
    try {
      createAiWorker("cpu").postMessage({
        id: pending.id,
        game: getGame(),
        depth: pending.minDepth,
        minDepth: pending.minDepth,
        nodes: pending.nodes,
        timeMs: pending.timeMs,
        partitionIndex: 0,
        partitionCount: 1
      });
    } catch (error: unknown) {
      handleAiWorkerMessage({
        data: {
          id: pending.id,
          ok: false,
          error: errorMessage(error),
          partitionIndex: 0
        }
      } as MessageEvent<AiWorkerResponse>);
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
      const receivedDepth = completedSearchDepth(result, pending.currentDepth);
      if (receivedDepth === null) {
        pending.incompleteDepthAttempt = true;
        pending.errors.push(`AI worker returned incomplete or non-terminal odd depth ${result.depth ?? "unknown"} for requested depth ${pending.currentDepth}.`);
      } else {
        const depthResult = { ...result, depth: receivedDepth };
        const entry = { result: depthResult, partitionIndex: partitionIndex ?? null, depth: receivedDepth };
        pending.results.push(entry);
        pending.depthResults.push(entry);
      }
    } else {
      pending.errors.push(error ?? "AI worker returned no result.");
    }

    if (pending.depthReceived < pending.depthExpected) {
      return;
    }

    for (const entry of pending.depthResults) {
      const existing = pending.bestByDepth.get(entry.depth);
      const depthBest = selectBestAiResult(existing ? [existing, entry.result] : [entry.result]);
      if (depthBest) {
        pending.bestByDepth.set(entry.depth, depthBest);
      }
    }

    const bestResult = selectDeepestStoredResult(pending)
      ?? selectBestAiResult(pending.results.map((entry) => entry.result));
    const reachedTarget = (bestResult?.depth ?? 0) >= pending.targetDepth;
    const reachedTimedMinimum = Date.now() >= pending.deadlineAt
      && (bestResult?.depth ?? 0) >= pending.minDepth;
    if (reachedTarget || reachedTimedMinimum) {
      finishBotSearch(pending, Date.now() >= pending.deadlineAt ? "timeout" : "complete");
      return;
    }

    if (pending.incompleteDepthAttempt && pending.currentDepth >= pending.minDepth) {
      if (pending.backend === "gpu" && !pending.minimumFallbackStarted) {
        startMinimumDepthCpuFallback(pending);
        return;
      }
      if (Date.now() < pending.deadlineAt && nextBotSearchDepth(pending.currentDepth, pending.targetDepth) > pending.currentDepth) {
        launchNextBotDepth(pending.id);
        return;
      }
    }

    if (!bestResult && nextBotSearchDepth(pending.currentDepth, pending.targetDepth) <= pending.currentDepth) {
      if (pending.backend === "gpu" && !pending.minimumFallbackStarted) {
        startMinimumDepthCpuFallback(pending);
        return;
      }
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
    const engine = loadedEngine();
    const { ptr, len } = writeWasmString(engine, JSON.stringify(results));
    try {
      const output = engine.chronofish_bot_select_best_result_json(ptr, len);
      if (!output) {
        throw new Error(readWasmString(engine, engine.chronofish_last_message()));
      }
      return JSON.parse(readWasmString(engine, output)) as AiSearchResult | null;
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
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
      plan: formatBotPlan(choice.principalVariation ?? [choice.moves], pending.game),
      depth: choice.depth ?? "",
      nodes: choice.nodes ?? "",
      worker: choice.partitionIndex ?? "",
      search: choice.cpuSearch ?? choice.gpuSearch ?? ""
    })));
    if (selectedResult?.gpuDiagnostics) {
      console.info("Selected GPU frontier diagnostics", selectedResult.gpuDiagnostics);
    }
    if (pending.errors.length) {
      console.info("Bot search worker errors", pending.errors);
    }
    console.groupEnd();
  }

  function rankedBotChoices(results: PendingResult[], selectedResult: AiSearchResult | null): RankedBotChoice[] {
    const engine = loadedEngine();
    const { ptr, len } = writeWasmString(engine, JSON.stringify({
      results,
      selectedMoves: selectedResult?.moves ?? []
    }));
    try {
      const output = engine.chronofish_bot_ranked_choices_json(ptr, len);
      if (!output) {
        throw new Error(readWasmString(engine, engine.chronofish_last_message()));
      }
      return JSON.parse(readWasmString(engine, output)) as RankedBotChoice[];
    } finally {
      engine.chronofish_dealloc(ptr, len);
    }
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
        cpuSearch: choice.cpuSearch ?? null,
        principalVariation: normalizePrincipalVariation(choice.principalVariation, choice.moves)
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

  function botMovesKey(moves: Move[]): string {
    return moves.map((move) => formatBotMove(move)).join("|");
  }

  function formatBotPlan(variation: PrincipalVariation, game: GameSnapshot | null = null): string {
    return normalizePrincipalVariation(variation, variation[0] ?? [])
      .map((turn, index) => `d${index + 1}: ${turn.map((move) => formatBotMove(move, game)).join(" | ")}`)
      .join(" => ");
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
