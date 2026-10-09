"use strict";
// The API payloads are runtime-owned by the Rust server. A generated schema can
// replace these structural values later without changing the renderer.
const state = { token: localStorage.getItem("chronofish-token"), user: null, users: [], current: null, selected: null, poll: null };
const $ = (selector) => document.querySelector(selector);
async function api(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (state.token)
        headers.Authorization = `Bearer ${state.token}`;
    if (options.body && typeof options.body !== "string") {
        headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(options.body);
    }
    const response = await fetch(path, { ...options, headers });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok)
        throw new Error(payload.error || `Request failed (${response.status})`);
    return payload;
}
function show(view) { for (const id of ["auth-view", "lobby-view", "game-view"])
    $(`#${id}`).hidden = id !== view; }
function escapeHtml(value) { const span = document.createElement("span"); span.textContent = value; return span.innerHTML; }
function notice(message, error = false) { const node = $("#game-notice"); node.textContent = message; node.classList.toggle("error", error); }
function setIdentity() {
    const node = $("#identity");
    if (!state.user)
        return node.replaceChildren();
    node.innerHTML = `<span><strong>${escapeHtml(state.user.displayName)}</strong><br><small>${state.user.rating} Elo</small></span><button id="logout">Sign out</button>`;
    $("#logout").onclick = logout;
}
async function logout() {
    try {
        await api("/api/auth/logout", { method: "POST" });
    }
    catch (_) { /* local expiry is enough */ }
    localStorage.removeItem("chronofish-token");
    Object.assign(state, { token: null, user: null, current: null });
    stopPolling();
    setIdentity();
    show("auth-view");
}
async function loadLobby() {
    stopPolling();
    const [games, users, leaders] = await Promise.all([api("/api/games"), api("/api/users"), api("/api/leaderboard")]);
    state.users = users;
    const opponent = $("#new-game-form select[name=opponent]");
    opponent.replaceChildren(...users.filter((user) => user.id !== state.user.id).map((user) => {
        const option = document.createElement("option");
        option.value = user.username;
        option.textContent = `${user.displayName}${user.isBot ? " · bot" : ""} (${user.rating})`;
        return option;
    }));
    const list = $("#game-list");
    if (!games.length)
        list.innerHTML = "<p>No games yet. Challenge a bot or another account.</p>";
    else
        list.replaceChildren(...games.map(gameCard));
    $("#leaderboard").replaceChildren(...leaders.map((user, index) => {
        const item = document.createElement("li");
        item.innerHTML = `<span>${index + 1}</span><span>${escapeHtml(user.displayName)} ${user.isBot ? '<b class="bot-badge">bot</b>' : ""}</span><strong>${user.rating}</strong>`;
        return item;
    }));
    show("lobby-view");
}
function gameCard(game) {
    const button = document.createElement("button");
    button.className = "game-card";
    const opponent = game.white.id === state.user.id ? game.black : game.white;
    const result = game.status === "active" ? "Active" : game.winnerId == null ? "Draw" : game.winnerId === state.user.id ? "Won" : "Lost";
    button.innerHTML = `<strong>vs ${escapeHtml(opponent.displayName)}</strong><small>${game.ruleset} · ${new Date(game.updatedAt * 1000).toLocaleString()}</small><span class="result">${result}</span>`;
    button.onclick = () => openGame(game.id);
    return button;
}
async function openGame(id) {
    location.hash = `game/${id}`;
    state.selected = null;
    delete $("#multiverse").dataset.positioned;
    state.current = await api(`/api/games/${id}`);
    renderGame();
    show("game-view");
    startPolling(id);
}
function startPolling(id) { stopPolling(); state.poll = setInterval(async () => { try {
    const latest = await api(`/api/games/${id}`);
    if (latest.summary.version !== state.current?.summary.version) {
        state.current = latest;
        state.selected = null;
        renderGame();
    }
}
catch (error) {
    notice(error.message, true);
} }, 2000); }
function stopPolling() { if (state.poll)
    clearInterval(state.poll); state.poll = null; }
function moves() { return state.current.game.legalActions.filter((action) => action.type === "move").map((action) => action.movement); }
function boardKey(position) { return `${position.timelineId}:${position.time}`; }
function samePosition(a, b) { return a.timelineId === b.timelineId && a.time === b.time && a.x === b.x && a.y === b.y; }
function formatPosition(position) { return `T${position.time}L${position.timelineId}:${String.fromCharCode(97 + position.x)}${position.y + 1}`; }
function formatMove(move) { return `${formatPosition(move.from)} → ${formatPosition(move.to)}${move.promotion ? `=${move.promotion}` : ""}`; }
function renderGame() {
    const { summary, game, viewerSide, history } = state.current;
    $("#game-title").textContent = `${summary.white.displayName} (White) vs ${summary.black.displayName} (Black)`;
    $("#game-status").textContent = game.outcome ? `${game.outcome.winner || "Nobody"} · ${game.outcome.reason}` : `${game.turn} to move · ${game.message}`;
    const mine = viewerSide === game.turn;
    $("#submit-turn").disabled = !mine || !game.canSubmit;
    const snapshots = game.timelines.flatMap((timeline) => timeline.boards.map((snapshot) => ({ timeline, snapshot })));
    const times = snapshots.map(({ snapshot }) => snapshot.time).concat(0);
    const rows = game.timelines.map((timeline) => timeline.row).concat(0);
    const [minTime, maxTime, minRow, maxRow] = [Math.min(...times), Math.max(...times), Math.min(...rows), Math.max(...rows)];
    const stage = $("#multiverse");
    stage.style.setProperty("--timeline-count", String(maxRow - minRow + 1));
    stage.style.setProperty("--time-count", String(maxTime - minTime + 1));
    const children = [];
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
    $("#move-history").replaceChildren(...history.map(renderHistory));
    requestAnimationFrame(() => { const origin = stage.querySelector('[data-board="0:0"]'); if (origin && !stage.dataset.positioned) {
        origin.scrollIntoView({ block: "center", inline: "start" });
        stage.dataset.positioned = "true";
    } });
}
function renderBoard({ timeline, snapshot }, minRow, minTime, mine) {
    const latestTime = timeline.boards.at(-1)?.time;
    const active = state.current.game.activeTimelines.includes(timeline.id);
    const playable = latestTime === snapshot.time && snapshot.time === state.current.game.presentTime && snapshot.sideToMove === state.current.game.turn;
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
    for (let y = 7; y >= 0; y -= 1)
        for (let x = 0; x < 8; x += 1) {
            const position = { timelineId: timeline.id, time: snapshot.time, x, y };
            const square = document.createElement("button");
            square.className = `square ${(x + y) % 2 ? "dark" : ""}`;
            const piece = snapshot.board[y][x];
            if (piece)
                square.innerHTML = `<span class="piece ${piece.color}">${pieceGlyph(piece.type)}</span>`;
            if (moves().some((move) => samePosition(move.from, position)))
                square.classList.add("source");
            if (state.selected && samePosition(state.selected, position))
                square.classList.add("selected");
            if (state.selected && legalTargets(state.selected).some((move) => samePosition(move.to, position)))
                square.classList.add("target");
            square.title = `${formatPosition(position)}${piece ? ` ${piece.color} ${piece.type}` : ""}`;
            square.onclick = () => chooseSquare(position);
            board.append(square);
        }
    card.append(caption, board);
    return card;
}
function pieceGlyph(type) { return ({ king: "♚", commonKing: "♔", queen: "♛", royalQueen: "♕", rook: "♜", princess: "♖", bishop: "♝", dragon: "◆", knight: "♞", unicorn: "◇", pawn: "♟", brawn: "♙" })[type] || "?"; }
function legalTargets(position) { return moves().filter((move) => samePosition(move.from, position)); }
function chooseSquare(position) {
    if (state.current.viewerSide !== state.current.game.turn)
        return;
    if (state.selected) {
        const candidates = legalTargets(state.selected).filter((move) => samePosition(move.to, position));
        if (candidates.length) {
            let movement = candidates[0];
            if (candidates.length > 1) {
                const choice = prompt(`Promote to: ${candidates.map((move) => move.promotion).join(", ")}`, candidates[0].promotion);
                movement = candidates.find((move) => move.promotion === choice) || movement;
            }
            playMove(movement);
            return;
        }
    }
    state.selected = moves().some((move) => samePosition(move.from, position)) ? position : null;
    renderGame();
}
async function playMove(movement) { try {
    state.current = await api(`/api/games/${state.current.summary.id}/moves`, { method: "POST", body: { version: state.current.summary.version, movement } });
    state.selected = null;
    notice("Move accepted.");
    renderGame();
}
catch (error) {
    notice(error.message, true);
} }
function renderHistory(entry) {
    const item = document.createElement("li");
    const decision = entry.action.action ? entry.action : null;
    const action = decision?.action || entry.action;
    const text = action.type === "submitTurn" ? "submitted turn" : formatMove(action.movement);
    item.textContent = `${entry.sequence}. ${entry.actor.displayName}: ${text}`;
    if (decision?.principalVariation?.length) {
        const analysis = document.createElement("small");
        analysis.textContent = ` PV: ${decision.principalVariation.map((step) => step.type === "submitTurn" ? "submit" : formatMove(step.movement)).join(" · ")}`;
        item.append(analysis);
    }
    return item;
}
document.querySelectorAll("[data-auth-mode]").forEach((button) => { button.onclick = () => { document.querySelectorAll("[data-auth-mode]").forEach((other) => other.classList.toggle("active", other === button)); $("#auth-form").dataset.mode = button.dataset.authMode; $("#display-name-field").hidden = button.dataset.authMode !== "register"; }; });
$("#auth-form").dataset.mode = "login";
$("#auth-form").onsubmit = async (event) => { event.preventDefault(); const form = new FormData(event.currentTarget); try {
    const session = await api(`/api/auth/${event.currentTarget.dataset.mode}`, { method: "POST", body: { username: form.get("username"), password: form.get("password"), displayName: form.get("displayName") || undefined } });
    state.token = session.token;
    state.user = session.user;
    localStorage.setItem("chronofish-token", state.token);
    setIdentity();
    await loadLobby();
}
catch (error) {
    $("#auth-error").textContent = error.message;
} };
$("#new-game-form").onsubmit = async (event) => { event.preventDefault(); try {
    state.current = await api("/api/games", { method: "POST", body: Object.fromEntries(new FormData(event.currentTarget)) });
    delete $("#multiverse").dataset.positioned;
    renderGame();
    show("game-view");
    startPolling(state.current.summary.id);
}
catch (error) {
    alert(error.message);
} };
$("#back-to-lobby").onclick = () => { history.replaceState(null, "", location.pathname); loadLobby(); };
$("#refresh-game").onclick = () => openGame(state.current.summary.id);
$("#submit-turn").onclick = async () => { try {
    state.current = await api(`/api/games/${state.current.summary.id}/submit`, { method: "POST", body: { version: state.current.summary.version } });
    state.selected = null;
    renderGame();
}
catch (error) {
    notice(error.message, true);
} };
(async () => { if (!state.token)
    return show("auth-view"); try {
    state.user = await api("/api/me");
    setIdentity();
    await loadLobby();
    const match = location.hash.match(/^#game\/(\d+)$/);
    if (match)
        await openGame(Number(match[1]));
}
catch (_) {
    await logout();
} })();
