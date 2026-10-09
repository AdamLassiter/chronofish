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
  action: PlayerAction | AiDecision;
  createdAt: number;
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

function renderGame(): void {
  const { summary, game, viewerSide, history } = currentGame();
  element("#game-title").textContent =
    `${summary.white.displayName} (White) vs ${summary.black.displayName} (Black)`;
  element("#game-status").textContent = game.outcome
    ? `${game.outcome.winner ?? "Nobody"} · ${game.outcome.reason}`
    : `${game.turn} to move · ${game.message}`;
  const mine = viewerSide === game.turn;
  element<HTMLButtonElement>("#submit-turn").disabled =
    !mine || !game.canSubmit;
  const snapshots = game.timelines.flatMap((timeline) =>
    timeline.boards.map((snapshot) => ({ timeline, snapshot })),
  );
  const times = snapshots.map(({ snapshot }) => snapshot.time).concat(0);
  const rows = game.timelines.map((timeline) => timeline.row).concat(0);
  const [minTime, maxTime, minRow, maxRow] = [
    Math.min(...times),
    Math.max(...times),
    Math.min(...rows),
    Math.max(...rows),
  ];
  const stage = element("#multiverse");
  stage.style.setProperty("--timeline-count", String(maxRow - minRow + 1));
  stage.style.setProperty("--time-count", String(maxTime - minTime + 1));
  const children: HTMLElement[] = [];
  for (let row = minRow; row <= maxRow; row += 1) {
    const label = document.createElement("div");
    label.className = "axis-label timeline";
    label.style.gridColumn = String(row - minRow + 2);
    label.style.gridRow = "1";
    label.textContent = `L${row}`;
    children.push(label);
  }
  for (let time = minTime; time <= maxTime; time += 1) {
    const label = document.createElement("div");
    label.className = "axis-label time";
    label.style.gridColumn = "1";
    label.style.gridRow = String(time - minTime + 2);
    label.textContent = `T${time}`;
    children.push(label);
  }
  for (const item of snapshots)
    children.push(renderBoard(item, minRow, minTime, mine));
  stage.replaceChildren(...children);
  element<HTMLOListElement>("#move-history").replaceChildren(
    ...history.map(renderHistory),
  );
  requestAnimationFrame(() => {
    const origin = stage.querySelector<HTMLElement>('[data-board="0:0"]');
    if (origin && !stage.dataset.positioned) {
      origin.scrollIntoView({ block: "center", inline: "start" });
      stage.dataset.positioned = "true";
    }
  });
}

function renderBoard(
  item: { timeline: Timeline; snapshot: BoardSnapshot },
  minRow: number,
  minTime: number,
  mine: boolean,
): HTMLElement {
  const { timeline, snapshot } = item;
  const game = currentGame().game;
  const latestTime = timeline.boards.at(-1)?.time;
  const active = game.activeTimelines.includes(timeline.id);
  const playable =
    latestTime === snapshot.time &&
    snapshot.time === game.presentTime &&
    snapshot.sideToMove === game.turn;
  const card = document.createElement("article");
  card.className = `board-card${snapshot.time === latestTime ? " latest" : ""}${playable && mine ? " playable" : ""}${active ? "" : " inactive"}`;
  card.style.gridColumn = String(timeline.row - minRow + 2);
  card.style.gridRow = String(snapshot.time - minTime + 2);
  card.dataset.board = `${timeline.id}:${snapshot.time}`;
  const caption = document.createElement("div");
  caption.className = "board-caption";
  caption.innerHTML = `<strong>T${snapshot.time}L${timeline.id}</strong><span>${snapshot.time === latestTime ? "front" : "history"}${active ? "" : " · inactive"}</span>`;
  const board = document.createElement("div");
  board.className = "board";
  const legalMoves = moves();
  for (let y = 7; y >= 0; y -= 1) {
    for (let x = 0; x < 8; x += 1) {
      const position: Position = {
        timelineId: timeline.id,
        time: snapshot.time,
        x,
        y,
      };
      const square = document.createElement("button");
      square.className = `square ${(x + y) % 2 ? "dark" : ""}`;
      const piece = snapshot.board[y]?.[x];
      if (piece)
        square.innerHTML = `<span class="piece ${piece.color}">${pieceGlyph(piece.type)}</span>`;
      if (legalMoves.some((move) => samePosition(move.from, position)))
        square.classList.add("source");
      if (state.selected && samePosition(state.selected, position))
        square.classList.add("selected");
      if (
        state.selected &&
        legalTargets(state.selected).some((move) =>
          samePosition(move.to, position),
        )
      )
        square.classList.add("target");
      square.title = `${formatPosition(position)}${piece ? ` ${piece.color} ${piece.type}` : ""}`;
      square.onclick = () => chooseSquare(position);
      board.append(square);
    }
  }
  card.append(caption, board);
  return card;
}

const PIECE_GLYPHS: Record<PieceType, string> = {
  king: "♚",
  commonKing: "♔",
  queen: "♛",
  royalQueen: "♕",
  rook: "♜",
  princess: "♖",
  bishop: "♝",
  dragon: "◆",
  knight: "♞",
  unicorn: "◇",
  pawn: "♟",
  brawn: "♙",
};

function pieceGlyph(type: PieceType): string {
  return PIECE_GLYPHS[type];
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
    notice(errorMessage(error), true);
  }
}

function isAiDecision(action: PlayerAction | AiDecision): action is AiDecision {
  return "action" in action;
}

function renderHistory(entry: MoveRecord): HTMLLIElement {
  const item = document.createElement("li");
  let decision: AiDecision | null;
  let action: PlayerAction;
  if (isAiDecision(entry.action)) {
    decision = entry.action;
    action = entry.action.action;
  } else {
    decision = null;
    action = entry.action;
  }
  const text =
    action.type === "submitTurn"
      ? "submitted turn"
      : formatMove(action.movement);
  item.textContent = `${entry.sequence}. ${entry.actor.displayName}: ${text}`;
  if (decision && decision.principalVariation.length > 0) {
    const analysis = document.createElement("small");
    analysis.textContent = ` PV: ${decision.principalVariation
      .map((step) =>
        step.type === "submitTurn" ? "submit" : formatMove(step.movement),
      )
      .join(" · ")}`;
    item.append(analysis);
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
      notice(errorMessage(error), true);
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
