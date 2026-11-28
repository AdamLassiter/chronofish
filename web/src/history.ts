/** The move history drawer. */

import { element } from "./dom.js";
import { formatMove, moves } from "./moves.js";
import {
  AiDecision,
  ConcedeAction,
  HistoryTurn,
  MoveRecord,
  PlayerAction,
  UndoFailedAiAction,
  UndoStagedMoveAction,
} from "./types.js";

export function setHistoryOpen(open: boolean): void {
  element("#history-drawer").hidden = !open;
  element("#toggle-history").setAttribute("aria-expanded", String(open));
}

export function isAiDecision(
  action:
    | PlayerAction
    | ConcedeAction
    | UndoFailedAiAction
    | UndoStagedMoveAction
    | AiDecision,
): action is AiDecision {
  return "action" in action;
}

export function recordedAction(
  entry: MoveRecord,
): PlayerAction | ConcedeAction | UndoFailedAiAction | UndoStagedMoveAction {
  return isAiDecision(entry.action) ? entry.action.action : entry.action;
}

export function groupHistory(entries: MoveRecord[]): HistoryTurn[] {
  const turns: HistoryTurn[] = [];
  let current: HistoryTurn | null = null;
  for (const entry of entries) {
    const action = recordedAction(entry);
    if (
      action.type === "undoFailedAiAction" ||
      action.type === "undoStagedMove"
    ) {
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

export function renderHistoryTurn(turn: HistoryTurn): HTMLLIElement {
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
