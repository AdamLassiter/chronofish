/** Legal-move lookups and move and position formatting. */

import { currentGame } from "./state.js";
import { Movement, PlayerAction, Position } from "./types.js";

export function moves(): Movement[] {
  return currentGame()
    .game.legalActions.filter(
      (action): action is Extract<PlayerAction, { type: "move" }> =>
        action.type === "move",
    )
    .map((action) => action.movement);
}

export function samePosition(a: Position, b: Position): boolean {
  return (
    a.timelineId === b.timelineId &&
    a.time === b.time &&
    a.x === b.x &&
    a.y === b.y
  );
}

export function formatPosition(position: Position): string {
  return `T${position.time}L${position.timelineId}:${String.fromCharCode(97 + position.x)}${position.y + 1}`;
}

export function formatMove(move: Movement): string {
  return `${formatPosition(move.from)} → ${formatPosition(move.to)}${move.promotion ? `=${move.promotion}` : ""}`;
}

export function legalTargets(position: Position): Movement[] {
  return moves().filter((move) => samePosition(move.from, position));
}
