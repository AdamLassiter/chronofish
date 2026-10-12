type Side = "white" | "black";
type Ruleset = "standard" | "multiverse" | "multiverse-variant";
type PieceType =
  | "king"
  | "commonKing"
  | "queen"
  | "royalQueen"
  | "princess"
  | "rook"
  | "bishop"
  | "unicorn"
  | "dragon"
  | "knight"
  | "pawn"
  | "brawn";

interface User {
  id: number;
  username: string;
  displayName: string;
  isBot: boolean;
  rating: number;
  gamesPlayed: number;
}

interface Position {
  timelineId: number;
  time: number;
  x: number;
  y: number;
}

interface Movement {
  from: Position;
  to: Position;
  promotion?: PieceType;
}

type PlayerAction =
  | { type: "move"; movement: Movement }
  | { type: "submitTurn" };

interface ConcedeAction {
  type: "concede";
}

interface UndoFailedAiAction {
  type: "undoFailedAiAction";
}

interface AiDecision {
  action: PlayerAction;
  principalVariation: PlayerAction[];
  rootValue: number | null;
}

interface Piece {
  color: Side;
  type: PieceType;
}

interface CastlingRights {
  whiteKingside: boolean;
  whiteQueenside: boolean;
  blackKingside: boolean;
  blackQueenside: boolean;
}

interface EnPassant {
  x: number;
  y: number;
  capturedX: number;
  capturedY: number;
}

type BoardOrigin =
  | { type: "none" }
  | {
      type: "move";
      from: Position;
      to: Position;
      move_type: string;
    };

interface BoardSnapshot {
  time: number;
  sideToMove: Side;
  board: Array<Array<Piece | null>>;
  castling: CastlingRights;
  enPassant: EnPassant | null;
  origin: BoardOrigin;
  halfmoveClock: number;
}

interface Timeline {
  id: number;
  row: number;
  label: string;
  owner: "neutral" | Side;
  boards: BoardSnapshot[];
}

type OutcomeReason =
  | "checkmate"
  | "stalemate"
  | "royal-capture"
  | "threefold-repetition"
  | "fifty-move-rule"
  | "insufficient-material"
  | "no-legal-turn"
  | "concession";

interface GameView {
  ruleset: Ruleset;
  turn: Side;
  outcome: { winner: Side | null; reason: OutcomeReason } | null;
  message: string;
  timelines: Timeline[];
  legalActions: PlayerAction[];
  canSubmit: boolean;
  hasStagedMoves: boolean;
  presentTime: number | null;
  activeTimelines: number[];
  checkedRoyals: Position[];
}

interface GameSummary {
  id: number;
  ruleset: Ruleset;
  status: "active" | "finished";
  white: User;
  black: User;
  winnerId: number | null;
  outcomeReason: string | null;
  version: number;
  createdAt: number;
  updatedAt: number;
  finishedAt: number | null;
}

interface MoveRecord {
  sequence: number;
  actor: User;
  action: PlayerAction | ConcedeAction | UndoFailedAiAction | AiDecision;
  createdAt: number;
}

interface HistoryTurn {
  actor: User;
  entries: MoveRecord[];
}

interface GameDetail {
  summary: GameSummary;
  game: GameView;
  viewerSide: Side | null;
  history: MoveRecord[];
}

interface Session {
  token: string;
  user: User;
}

interface AppState {
  token: string | null;
  user: User | null;
  users: User[];
  current: GameDetail | null;
  selected: Position | null;
  poll: ReturnType<typeof setInterval> | null;
}

interface ApiOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
}

const state: AppState = {
  token: localStorage.getItem("chronofish-token"),
  user: null,
  users: [],
  current: null,
  selected: null,
  poll: null,
};

function element<T extends HTMLElement = HTMLElement>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`Missing required element: ${selector}`);
  return node;
}

function currentGame(): GameDetail {
  if (!state.current) throw new Error("No game is currently open");
  return state.current;
}

function currentUser(): User {
  if (!state.user) throw new Error("No user is signed in");
  return state.user;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function api<T>(path: string, options: ApiOptions = {}): Promise<T> {
  const { body: value, ...requestOptions } = options;
  const headers = new Headers(options.headers);
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  const body = value === undefined ? undefined : JSON.stringify(value);
  if (body !== undefined) headers.set("Content-Type", "application/json");

  const response = await fetch(path, { ...requestOptions, headers, body });
  const payload: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message =
      typeof payload === "object" &&
      payload !== null &&
      "error" in payload &&
      typeof payload.error === "string"
        ? payload.error
        : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return payload as T;
}

function show(view: "auth-view" | "lobby-view" | "game-view"): void {
  for (const id of ["auth-view", "lobby-view", "game-view"])
    element(`#${id}`).hidden = id !== view;
}

function escapeHtml(value: string): string {
  const span = document.createElement("span");
  span.textContent = value;
  return span.innerHTML;
}

function notice(message: string, error = false): void {
  const node = element("#game-notice");
  node.textContent = message;
  node.classList.toggle("error", error);
}

function setIdentity(): void {
  const node = element("#identity");
  if (!state.user) {
    node.replaceChildren();
    return;
  }
  node.innerHTML = `<span><strong>${escapeHtml(state.user.displayName)}</strong><br><small>${state.user.rating} Elo</small></span><button id="logout">Sign out</button>`;
  element<HTMLButtonElement>("#logout").onclick = () => void logout();
}

async function logout(): Promise<void> {
  try {
    await api<{ ok: boolean }>("/api/auth/logout", { method: "POST" });
  } catch {
    // Removing the local token is sufficient if the session already expired.
  }
  localStorage.removeItem("chronofish-token");
  Object.assign(state, { token: null, user: null, current: null });
  stopPolling();
  setIdentity();
  show("auth-view");
}

async function loadLobby(): Promise<void> {
  stopPolling();
  const [games, users, leaders] = await Promise.all([
    api<GameSummary[]>("/api/games"),
    api<User[]>("/api/users"),
    api<User[]>("/api/leaderboard"),
  ]);
  state.users = users;
  const user = currentUser();
  const opponent = element<HTMLSelectElement>(
    "#new-game-form select[name=opponent]",
  );
  opponent.replaceChildren(
    ...users
      .filter((candidate) => candidate.id !== user.id)
      .map((candidate) => {
        const option = document.createElement("option");
        option.value = candidate.username;
        option.textContent = `${candidate.displayName}${candidate.isBot ? " · bot" : ""} (${candidate.rating})`;
        return option;
      }),
  );
  const list = element("#game-list");
  if (games.length === 0)
    list.innerHTML = "<p>No games yet. Challenge a bot or another account.</p>";
  else list.replaceChildren(...games.map(gameCard));
  element<HTMLOListElement>("#leaderboard").replaceChildren(
    ...leaders.map((leader, index) => {
      const item = document.createElement("li");
      item.innerHTML = `<span>${index + 1}</span><span>${escapeHtml(leader.displayName)} ${leader.isBot ? '<b class="bot-badge">bot</b>' : ""}</span><strong>${leader.rating}</strong>`;
      return item;
    }),
  );
  show("lobby-view");
}

function gameCard(game: GameSummary): HTMLButtonElement {
  const button = document.createElement("button");
  button.className = "game-card";
  const user = currentUser();
  const opponent = game.white.id === user.id ? game.black : game.white;
  const result =
    game.status === "active"
      ? "Active"
      : game.winnerId === null
        ? "Draw"
        : game.winnerId === user.id
          ? "Won"
          : "Lost";
  button.innerHTML = `<strong>vs ${escapeHtml(opponent.displayName)}</strong><small>${game.ruleset} · ${new Date(game.updatedAt * 1000).toLocaleString()}</small><span class="result">${result}</span>`;
  button.onclick = () => void openGame(game.id);
  return button;
}

async function openGame(id: number): Promise<void> {
  location.hash = `game/${id}`;
  state.selected = null;
  delete element("#multiverse").dataset.positioned;
  state.current = await api<GameDetail>(`/api/games/${id}`);
  renderGame();
  show("game-view");
  startPolling(id);
}

function startPolling(id: number): void {
  stopPolling();
  state.poll = setInterval(() => {
    void (async () => {
      try {
        const latest = await api<GameDetail>(`/api/games/${id}`);
        if (latest.summary.version !== state.current?.summary.version) {
          state.current = latest;
          state.selected = null;
          renderGame();
        }
      } catch (error) {
        notice(errorMessage(error), true);
      }
    })();
  }, 2000);
}

function stopPolling(): void {
  if (state.poll !== null) clearInterval(state.poll);
  state.poll = null;
}

function moves(): Movement[] {
  return currentGame()
    .game.legalActions.filter(
      (action): action is Extract<PlayerAction, { type: "move" }> =>
        action.type === "move",
    )
    .map((action) => action.movement);
}

function samePosition(a: Position, b: Position): boolean {
  return (
    a.timelineId === b.timelineId &&
    a.time === b.time &&
    a.x === b.x &&
    a.y === b.y
  );
}

function formatPosition(position: Position): string {
  return `T${position.time}L${position.timelineId}:${String.fromCharCode(97 + position.x)}${position.y + 1}`;
}

function formatMove(move: Movement): string {
  return `${formatPosition(move.from)} → ${formatPosition(move.to)}${move.promotion ? `=${move.promotion}` : ""}`;
}

type BoardStatus = "Inactive" | "Future" | "Latest" | "Past";
type SquareHighlight = "arrived" | "move-source" | "move-target" | "check";
type HighlightMap = Map<string, Set<SquareHighlight>>;

interface Point {
  x: number;
  y: number;
}

interface MoveArrow {
  from: Position;
  to: Position;
}

function latestTime(timeline: Timeline): number | undefined {
  return timeline.boards.reduce<number | undefined>(
    (latest, board) =>
      latest === undefined || board.time > latest ? board.time : latest,
    undefined,
  );
}

function currentPresentTime(game: GameView): number {
  if (game.presentTime !== null) return game.presentTime;
  const activeTimes = game.timelines
    .filter((timeline) => game.activeTimelines.includes(timeline.id))
    .map(latestTime)
    .filter((time): time is number => time !== undefined);
  return activeTimes.length > 0 ? Math.min(...activeTimes) : 0;
}

function boardStatus(
  game: GameView,
  timeline: Timeline,
  snapshot: BoardSnapshot,
  presentTime: number,
): BoardStatus {
  if (!game.activeTimelines.includes(timeline.id)) return "Inactive";
  if (snapshot.time > presentTime) return "Future";
  return snapshot.time === latestTime(timeline) ? "Latest" : "Past";
}

function highlightKey(position: Position): string {
  return `${position.timelineId}:${position.time}:${position.x}:${position.y}`;
}

function addHighlight(
  highlights: HighlightMap,
  position: Position,
  type: SquareHighlight,
): void {
  const key = highlightKey(position);
  const existing = highlights.get(key) ?? new Set<SquareHighlight>();
  existing.add(type);
  highlights.set(key, existing);
}

function samePiece(left: Piece | null, right: Piece | null): boolean {
  if (!left || !right) return left === right;
  return left.color === right.color && left.type === right.type;
}

function addArrivalHighlights(
  highlights: HighlightMap,
  timeline: Timeline,
  snapshot: BoardSnapshot,
): void {
  const previous = timeline.boards.find(
    (candidate) => candidate.time === snapshot.time - 1,
  );
  if (!previous) {
    if (
      snapshot.origin.type === "move" &&
      snapshot.origin.move_type !== "source-advance"
    )
      addHighlight(highlights, snapshot.origin.to, "arrived");
    return;
  }
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      const piece = snapshot.board[y]?.[x] ?? null;
      if (piece && !samePiece(piece, previous.board[y]?.[x] ?? null)) {
        addHighlight(
          highlights,
          { timelineId: timeline.id, time: snapshot.time, x, y },
          "arrived",
        );
      }
    }
  }
}

function movementVisuals(game: GameView): {
  highlights: HighlightMap;
  arrows: MoveArrow[];
} {
  const highlights: HighlightMap = new Map();
  const arrows: MoveArrow[] = [];
  for (const timeline of game.timelines) {
    for (const snapshot of timeline.boards) {
      addArrivalHighlights(highlights, timeline, snapshot);
      if (
        snapshot.origin.type === "move" &&
        snapshot.origin.move_type !== "source-advance"
      ) {
        addHighlight(highlights, snapshot.origin.from, "move-source");
        addHighlight(highlights, snapshot.origin.to, "move-target");
        arrows.push({
          from: snapshot.origin.from,
          to: snapshot.origin.to,
        });
      }
    }
  }
  for (const position of game.checkedRoyals)
    addHighlight(highlights, position, "check");
  return { highlights, arrows };
}

function renderGame(): void {
  const { summary, game, viewerSide, history } = currentGame();
  element("#game-title").textContent =
    `${summary.white.displayName} (White) vs ${summary.black.displayName} (Black)`;
  element("#game-status").textContent = game.outcome
    ? `${game.outcome.winner ?? "Nobody"} · ${game.outcome.reason}`
    : `${game.turn} to move · ${game.message}`;
  const mine = viewerSide === game.turn;
  const finished = game.outcome !== null;
  const currentPlayer =
    game.turn === "white" ? summary.white : summary.black;
  element<HTMLButtonElement>("#submit-turn").disabled =
    finished || !mine || !game.canSubmit;
  const retryAi = element<HTMLButtonElement>("#retry-ai");
  retryAi.disabled = finished || !currentPlayer.isBot;
  retryAi.title = currentPlayer.isBot
    ? "Ask the server to try the AI turn again"
    : "The current player is not an AI";
  element<HTMLButtonElement>("#abandon-game").disabled = finished;
  if (finished) stopPolling();
  const presentTime = currentPresentTime(game);
  const maxTime = Math.max(
    presentTime,
    0,
    ...game.timelines.flatMap((timeline) =>
      timeline.boards.map((snapshot) => snapshot.time),
    ),
  );
  const { highlights, arrows } = movementVisuals(game);
  const grid = element("#timeline-grid");
  const targets = state.selected
    ? legalTargets(state.selected).map((move) => move.to)
    : [];
  const timelines = [...game.timelines].sort(
    (left, right) => left.row - right.row || left.id - right.id,
  );
  grid.replaceChildren(
    ...timelines.map((timeline) =>
      renderTimeline(
        game,
        timeline,
        maxTime,
        presentTime,
        targets,
        highlights,
      ),
    ),
  );
  element<HTMLOListElement>("#move-history").replaceChildren(
    ...groupHistory(history).map(renderHistoryTurn),
  );
  requestAnimationFrame(() => {
    renderMoveArrows(grid, arrows);
    const stage = element("#multiverse");
    const origin = grid.querySelector<HTMLElement>('[data-board="0:0"]');
    if (origin && !stage.dataset.positioned) {
      origin.scrollIntoView({ block: "center", inline: "start" });
      stage.dataset.positioned = "true";
    }
  });
}

function renderTimeline(
  game: GameView,
  timeline: Timeline,
  maxTime: number,
  presentTime: number,
  targets: Position[],
  highlights: HighlightMap,
): HTMLElement {
  const row = document.createElement("div");
  row.className = "timeline-row";
  row.dataset.owner = timeline.owner;
  row.dataset.active = String(game.activeTimelines.includes(timeline.id));
  row.style.setProperty("--time-columns", String(maxTime + 1));

  const label = document.createElement("div");
  label.className = "timeline-label";
  label.textContent = `L${timeline.id}`;
  row.append(label);

  const marker = document.createElement("div");
  marker.className = "present-line";
  marker.style.gridColumn = String(presentTime + 2);
  row.append(marker);

  const snapshots = [...timeline.boards].sort(
    (left, right) => left.time - right.time,
  );
  for (const snapshot of snapshots) {
    const board = renderBoard(
      game,
      timeline,
      snapshot,
      presentTime,
      targets,
      highlights,
    );
    board.style.gridColumn = String(snapshot.time + 2);
    row.append(board);
  }
  return row;
}

function renderBoard(
  game: GameView,
  timeline: Timeline,
  snapshot: BoardSnapshot,
  presentTime: number,
  targets: Position[],
  highlights: HighlightMap,
): HTMLElement {
  const card = document.createElement("article");
  card.className = "board-card";
  card.dataset.turn = snapshot.sideToMove;
  card.dataset.latest = String(snapshot.time === latestTime(timeline));
  card.dataset.present = String(snapshot.time === presentTime);
  card.dataset.board = `${timeline.id}:${snapshot.time}`;
  const board = document.createElement("div");
  board.className = "chessboard";
  for (let y = 7; y >= 0; y -= 1) {
    for (let x = 0; x < 8; x += 1) {
      const position: Position = {
        timelineId: timeline.id,
        time: snapshot.time,
        x,
        y,
      };
      const square = document.createElement("button");
      const piece = snapshot.board[y]?.[x] ?? null;
      const squareHighlights = highlights.get(highlightKey(position));
      square.type = "button";
      square.className = "square";
      square.dataset.light = String((x + y) % 2 === 1);
      square.dataset.positionKey = highlightKey(position);
      square.ariaLabel = `${String.fromCharCode(97 + x)}${y + 1}`;
      if (piece) {
        square.textContent = pieceGlyph(piece);
        square.dataset.pieceColor = piece.color;
      }
      if (state.selected && samePosition(state.selected, position))
        square.classList.add("is-selected");
      if (targets.some((target) => samePosition(target, position)))
        square.classList.add("is-target");
      if (squareHighlights?.has("move-source"))
        square.classList.add("was-move-source");
      if (squareHighlights?.has("arrived"))
        square.classList.add("was-arrival");
      if (squareHighlights?.has("move-target"))
        square.classList.add("was-move-target");
      if (squareHighlights?.has("check")) square.classList.add("is-check");
      square.title = `${formatPosition(position)}${piece ? ` ${piece.color} ${piece.type}` : ""}`;
      square.onclick = () => chooseSquare(position);
      board.append(square);
    }
  }
  const footer = document.createElement("footer");
  footer.className = "board-footer";
  footer.innerHTML = `
    <span>${boardStatus(game, timeline, snapshot, presentTime)}</span>
    <span class="board-side-to-move">${snapshot.sideToMove}</span>
    <strong>T${snapshot.time} L${timeline.id}</strong>
  `;
  card.append(board, footer);
  return card;
}

function measureSquareCenters(grid: HTMLElement): Map<string, Point> {
  const gridRect = grid.getBoundingClientRect();
  const centers = new Map<string, Point>();
  for (const square of grid.querySelectorAll<HTMLElement>(
    "[data-position-key]",
  )) {
    const key = square.dataset.positionKey;
    if (!key || centers.has(key)) continue;
    const squareRect = square.getBoundingClientRect();
    centers.set(key, {
      x: squareRect.left - gridRect.left + squareRect.width / 2,
      y: squareRect.top - gridRect.top + squareRect.height / 2,
    });
  }
  return centers;
}

function arrowPath(from: Point, to: Point): string {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);
  if (distance === 0) return "";
  const inset = Math.min(18, distance / 3);
  return `M ${(from.x + (dx / distance) * inset).toFixed(1)} ${(from.y + (dy / distance) * inset).toFixed(1)} L ${(to.x - (dx / distance) * inset).toFixed(1)} ${(to.y - (dy / distance) * inset).toFixed(1)}`;
}

function renderMoveArrows(grid: HTMLElement, arrows: MoveArrow[]): void {
  const oldOverlay = grid.querySelector(".move-arrows");
  oldOverlay?.remove();
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "move-arrows");
  svg.setAttribute("width", String(grid.scrollWidth));
  svg.setAttribute("height", String(grid.scrollHeight));
  svg.setAttribute("viewBox", `0 0 ${grid.scrollWidth} ${grid.scrollHeight}`);
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = `
    <defs>
      <marker id="move-arrow-head" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="3" markerHeight="3" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z"></path>
      </marker>
    </defs>
  `;
  const centers = measureSquareCenters(grid);
  for (const arrow of arrows) {
    const from = centers.get(highlightKey(arrow.from));
    const to = centers.get(highlightKey(arrow.to));
    if (!from || !to) continue;
    const pathData = arrowPath(from, to);
    if (!pathData) continue;
    const path = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path",
    );
    path.setAttribute("class", "move-arrow");
    path.setAttribute("d", pathData);
    path.setAttribute("marker-end", "url(#move-arrow-head)");
    svg.append(path);
  }
  grid.append(svg);
}

const PIECE_GLYPHS: Record<Side, Record<PieceType, string>> = {
  white: {
    king: "♔",
    commonKing: "♔",
    queen: "♕",
    royalQueen: "♕",
    princess: "♖",
    rook: "♖",
    bishop: "♗",
    unicorn: "♘",
    dragon: "♗",
    knight: "♘",
    pawn: "♙",
    brawn: "♙",
  },
  black: {
    king: "♚",
    commonKing: "♚",
    queen: "♛",
    royalQueen: "♛",
    princess: "♜",
    rook: "♜",
    bishop: "♝",
    unicorn: "♞",
    dragon: "♝",
    knight: "♞",
    pawn: "♟",
    brawn: "♟",
  },
};

function pieceGlyph(piece: Piece): string {
  return PIECE_GLYPHS[piece.color][piece.type];
}

function legalTargets(position: Position): Movement[] {
  return moves().filter((move) => samePosition(move.from, position));
}

function chooseSquare(position: Position): void {
  const current = currentGame();
  if (current.viewerSide !== current.game.turn) return;
  if (state.selected) {
    const candidates = legalTargets(state.selected).filter((move) =>
      samePosition(move.to, position),
    );
    if (candidates.length > 0) {
      let movement = candidates[0];
      if (!movement) return;
      if (candidates.length > 1) {
        const choice = prompt(
          `Promote to: ${candidates.map((move) => move.promotion).join(", ")}`,
          movement.promotion,
        );
        movement =
          candidates.find((move) => move.promotion === choice) ?? movement;
      }
      void playMove(movement);
      return;
    }
  }
  state.selected = moves().some((move) => samePosition(move.from, position))
    ? position
    : null;
  renderGame();
}

async function playMove(movement: Movement): Promise<void> {
  try {
    const current = currentGame();
    state.current = await api<GameDetail>(
      `/api/games/${current.summary.id}/moves`,
      {
        method: "POST",
        body: { version: current.summary.version, movement },
      },
    );
    state.selected = null;
    notice("Move accepted.");
    renderGame();
  } catch (error) {
    await recoverGameAfterFailedAction(error);
  }
}

async function recoverGameAfterFailedAction(error: unknown): Promise<void> {
  const message = errorMessage(error);
  const current = state.current;
  if (current) {
    try {
      state.current = await api<GameDetail>(
        `/api/games/${current.summary.id}`,
      );
      state.selected = null;
      renderGame();
    } catch {
      // Preserve the original, more useful action failure below.
    }
  }
  notice(message, true);
}

async function retryAiMove(): Promise<void> {
  const button = element<HTMLButtonElement>("#retry-ai");
  button.disabled = true;
  notice("Requesting another AI attempt…");
  try {
    const current = currentGame();
    state.current = await api<GameDetail>(
      `/api/games/${current.summary.id}/retry-ai`,
      {
        method: "POST",
        body: { version: current.summary.version },
      },
    );
    state.selected = null;
    renderGame();
    notice("The AI move was accepted.");
  } catch (error) {
    await recoverGameAfterFailedAction(error);
  }
}

async function abandonGame(): Promise<void> {
  const current = currentGame();
  if (
    !confirm(
      "Abandon this game? This immediately ends the game and counts as a loss.",
    )
  )
    return;
  element<HTMLButtonElement>("#abandon-game").disabled = true;
  try {
    state.current = await api<GameDetail>(
      `/api/games/${current.summary.id}/abandon`,
      {
        method: "POST",
        body: { version: current.summary.version },
      },
    );
    state.selected = null;
    renderGame();
    notice("Game abandoned. The result has been recorded as a loss.");
  } catch (error) {
    await recoverGameAfterFailedAction(error);
  }
}

function isAiDecision(
  action: PlayerAction | ConcedeAction | UndoFailedAiAction | AiDecision,
): action is AiDecision {
  return "action" in action;
}

function recordedAction(
  entry: MoveRecord,
): PlayerAction | ConcedeAction | UndoFailedAiAction {
  return isAiDecision(entry.action) ? entry.action.action : entry.action;
}

function groupHistory(entries: MoveRecord[]): HistoryTurn[] {
  const turns: HistoryTurn[] = [];
  let current: HistoryTurn | null = null;
  for (const entry of entries) {
    const action = recordedAction(entry);
    if (action.type === "undoFailedAiAction") {
      if (current?.actor.id === entry.actor.id) {
        current.entries.pop();
        if (current.entries.length === 0) {
          turns.pop();
          current = null;
        }
      }
      continue;
    }
    if (!current || current.actor.id !== entry.actor.id) {
      current = { actor: entry.actor, entries: [] };
      turns.push(current);
    }
    current.entries.push(entry);
    if (action.type === "submitTurn" || action.type === "concede")
      current = null;
  }
  return turns;
}

function renderHistoryTurn(turn: HistoryTurn): HTMLLIElement {
  const item = document.createElement("li");
  const actions = turn.entries.map(recordedAction);
  const moves = actions
    .filter(
      (action): action is Extract<PlayerAction, { type: "move" }> =>
        action.type === "move",
    )
    .map((action) => formatMove(action.movement));
  const description = actions.some((action) => action.type === "concede")
    ? "abandoned the game"
    : moves.length > 0
      ? moves.join(" · ")
      : "submitted turn";
  item.textContent = `${turn.actor.displayName}: ${description}`;

  const decisions = turn.entries
    .map((entry) => (isAiDecision(entry.action) ? entry.action : null))
    .filter(
      (decision): decision is AiDecision =>
        decision !== null && decision.principalVariation.length > 0,
    );
  if (decisions.length > 0) {
    const details = document.createElement("details");
    details.className = "ai-analysis";
    const summary = document.createElement("summary");
    summary.textContent = `AI analysis (${decisions.length})`;
    const lines = document.createElement("ol");
    for (const decision of decisions) {
      const line = document.createElement("li");
      line.textContent = decision.principalVariation
        .map((step) =>
          step.type === "submitTurn" ? "submit" : formatMove(step.movement),
        )
        .join(" · ");
      lines.append(line);
    }
    details.append(summary, lines);
    item.append(details);
  }
  return item;
}

function formString(form: FormData, name: string): string {
  const value = form.get(name);
  if (typeof value !== "string") throw new Error(`Missing field: ${name}`);
  return value;
}

document
  .querySelectorAll<HTMLButtonElement>("[data-auth-mode]")
  .forEach((button) => {
    button.onclick = () => {
      document
        .querySelectorAll<HTMLButtonElement>("[data-auth-mode]")
        .forEach((other) =>
          other.classList.toggle("active", other === button),
        );
      const mode = button.dataset.authMode;
      if (mode !== "login" && mode !== "register") return;
      element<HTMLFormElement>("#auth-form").dataset.mode = mode;
      element("#display-name-field").hidden = mode !== "register";
    };
  });

const authForm = element<HTMLFormElement>("#auth-form");
authForm.dataset.mode = "login";
authForm.onsubmit = (event) => {
  event.preventDefault();
  void (async () => {
    const form = new FormData(authForm);
    const mode = authForm.dataset.mode;
    if (mode !== "login" && mode !== "register") return;
    try {
      const displayName = form.get("displayName");
      const session = await api<Session>(`/api/auth/${mode}`, {
        method: "POST",
        body: {
          username: formString(form, "username"),
          password: formString(form, "password"),
          displayName:
            typeof displayName === "string" && displayName.length > 0
              ? displayName
              : undefined,
        },
      });
      state.token = session.token;
      state.user = session.user;
      localStorage.setItem("chronofish-token", session.token);
      setIdentity();
      await loadLobby();
    } catch (error) {
      element("#auth-error").textContent = errorMessage(error);
    }
  })();
};

const newGameForm = element<HTMLFormElement>("#new-game-form");
newGameForm.onsubmit = (event) => {
  event.preventDefault();
  void (async () => {
    try {
      const form = new FormData(newGameForm);
      state.current = await api<GameDetail>("/api/games", {
        method: "POST",
        body: {
          opponent: formString(form, "opponent"),
          ruleset: formString(form, "ruleset"),
          playAs: formString(form, "playAs"),
        },
      });
      delete element("#multiverse").dataset.positioned;
      renderGame();
      show("game-view");
      startPolling(currentGame().summary.id);
    } catch (error) {
      alert(errorMessage(error));
    }
  })();
};

element<HTMLButtonElement>("#back-to-lobby").onclick = () => {
  history.replaceState(null, "", location.pathname);
  void loadLobby();
};
element<HTMLButtonElement>("#refresh-game").onclick = () =>
  void openGame(currentGame().summary.id);
element<HTMLButtonElement>("#retry-ai").onclick = () => void retryAiMove();
element<HTMLButtonElement>("#abandon-game").onclick = () => void abandonGame();
element<HTMLButtonElement>("#submit-turn").onclick = () => {
  void (async () => {
    try {
      const current = currentGame();
      state.current = await api<GameDetail>(
        `/api/games/${current.summary.id}/submit`,
        {
          method: "POST",
          body: { version: current.summary.version },
        },
      );
      state.selected = null;
      renderGame();
    } catch (error) {
      await recoverGameAfterFailedAction(error);
    }
  })();
};

void (async () => {
  if (!state.token) {
    show("auth-view");
    return;
  }
  try {
    state.user = await api<User>("/api/me");
    setIdentity();
    await loadLobby();
    const match = location.hash.match(/^#game\/(\d+)$/);
    if (match?.[1]) await openGame(Number(match[1]));
  } catch {
    await logout();
  }
})();
