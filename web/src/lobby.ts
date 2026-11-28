/** Sign-in identity, the lobby, opening games and polling. */

import { api } from "./api.js";
import { renderGame } from "./board.js";
import { element, errorMessage, escapeHtml, notice, show } from "./dom.js";
import { currentUser, state } from "./state.js";
import { GameDetail, GameSummary, Ruleset, User } from "./types.js";

export function setIdentity(): void {
  const node = element("#identity");
  if (!state.user) {
    node.replaceChildren();
    return;
  }
  node.innerHTML = `<span><strong>${escapeHtml(state.user.displayName)}</strong><br><small>${state.user.rating} Elo</small></span><button id="logout">Sign out</button>`;
  element<HTMLButtonElement>("#logout").onclick = () => void logout();
}

export async function logout(): Promise<void> {
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

export function updateOpponents(): void {
  const user = currentUser();
  const opponent = element<HTMLSelectElement>(
    "#new-game-form select[name=opponent]",
  );
  const ruleset = element<HTMLSelectElement>(
    "#new-game-form select[name=ruleset]",
  ).value as Ruleset;
  const selected = opponent.value;
  opponent.replaceChildren(
    ...state.users
      .filter(
        (candidate) =>
          candidate.id !== user.id &&
          (candidate.supportedRulesets?.includes(ruleset) ?? true),
      )
      .map((candidate) => {
        const option = document.createElement("option");
        option.value = candidate.username;
        option.textContent = `${candidate.displayName}${candidate.isBot ? " · bot" : ""}${candidate.referenceRules ? " · reference rules" : ""} (${candidate.rating})`;
        return option;
      }),
  );
  if (
    Array.from(opponent.options).some((option) => option.value === selected)
  ) {
    opponent.value = selected;
  }
}

export async function loadLobby(): Promise<void> {
  stopPolling();
  const [games, users, leaders] = await Promise.all([
    api<GameSummary[]>("/api/games"),
    api<User[]>("/api/users"),
    api<User[]>("/api/leaderboard"),
  ]);
  state.users = users;
  updateOpponents();
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

export function gameCard(game: GameSummary): HTMLButtonElement {
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

export async function openGame(id: number): Promise<void> {
  location.hash = `game/${id}`;
  state.selected = null;
  delete element("#multiverse").dataset.positioned;
  delete element("#multiverse").dataset.presentTime;
  state.current = await api<GameDetail>(`/api/games/${id}`);
  renderGame();
  show("game-view");
  startPolling(id);
}

export function startPolling(id: number): void {
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

export function stopPolling(): void {
  if (state.poll !== null) clearInterval(state.poll);
  state.poll = null;
}
