/** Board input and the move, undo, submit and AI actions. */

import { api } from "./api.js";
import { renderGame } from "./board.js";
import { element, errorMessage, notice } from "./dom.js";
import { legalTargets, moves, samePosition } from "./moves.js";
import { currentGame, requests, state } from "./state.js";
import { GameDetail, Movement, Position } from "./types.js";

export function chooseSquare(position: Position): void {
  const current = currentGame();
  if (requests.turnPending || current.viewerSide !== current.game.turn) return;
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

export async function playMove(movement: Movement): Promise<void> {
  if (requests.turnPending) return;
  requests.turnPending = true;
  notice("Checking legal continuations…");
  renderGame();
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
  } finally {
    requests.turnPending = false;
    renderGame();
  }
}

export async function undoStagedMove(): Promise<void> {
  if (requests.turnPending) return;
  requests.turnPending = true;
  notice("Checking legal continuations…");
  renderGame();
  const button = element<HTMLButtonElement>("#undo-staged-move");
  button.disabled = true;
  try {
    const current = currentGame();
    state.current = await api<GameDetail>(
      `/api/games/${current.summary.id}/undo`,
      {
        method: "POST",
        body: { version: current.summary.version },
      },
    );
    state.selected = null;
    renderGame();
    notice("Staged move undone.");
  } catch (error) {
    await recoverGameAfterFailedAction(error);
  } finally {
    requests.turnPending = false;
    renderGame();
  }
}

export async function submitTurn(): Promise<void> {
  if (requests.turnPending) return;
  requests.turnPending = true;
  notice("Checking the submitted turn…");
  renderGame();
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
    notice(state.current.game.outcome ? "Game over." : "Turn submitted.");
    renderGame();
  } catch (error) {
    await recoverGameAfterFailedAction(error);
  } finally {
    requests.turnPending = false;
    renderGame();
  }
}

export async function recoverGameAfterFailedAction(
  error: unknown,
): Promise<void> {
  const message = errorMessage(error);
  const current = state.current;
  if (current) {
    try {
      state.current = await api<GameDetail>(`/api/games/${current.summary.id}`);
      state.selected = null;
      renderGame();
    } catch {
      // Preserve the original, more useful action failure below.
    }
  }
  notice(message, true);
}

export async function retryAiMove(): Promise<void> {
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

export async function abandonGame(): Promise<void> {
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
