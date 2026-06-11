import type { GameSnapshot, Position, Timeline } from "./types.js";

export function getTimeline(game: GameSnapshot, timelineId: number): Timeline | undefined {
  return game.timelines.find((timeline) => timeline.id === timelineId);
}

export function getBoard(game: GameSnapshot, timelineId: number, time: number) {
  return getTimeline(game, timelineId)?.boards.find((board) => board.time === time) ?? null;
}

export function getLatestBoard(game: GameSnapshot, timelineId: number) {
  const timeline = getTimeline(game, timelineId);
  const first = timeline?.boards[0];
  return first
    ? timeline.boards.reduce((latest, board) => (board.time > latest.time ? board : latest), first)
    : undefined;
}

export function isLatestBoard(game: GameSnapshot, timelineId: number, time: number): boolean {
  return getLatestBoard(game, timelineId)?.time === time;
}

export function sortedTimelines(game: GameSnapshot): Timeline[] {
  // row is the visual/geometric timeline axis; id is only the stable tie-breaker.
  return [...game.timelines].sort((a, b) => a.row - b.row || a.id - b.id);
}

export function sortedBoards(timeline: Timeline) {
  return [...timeline.boards].sort((a, b) => a.time - b.time);
}

export function isActiveTimeline(game: GameSnapshot, timeline: Timeline): boolean {
  if (typeof timeline.active === "boolean") {
    return timeline.active;
  }

  // GPU-originated snapshots do not yet carry Rust's derived metadata.
  if (timeline.owner === "neutral") {
    return true;
  }

  const ids = game.timelines.map((candidate) => candidate.id);
  const minTimeline = Math.min(...ids);
  const maxTimeline = Math.max(...ids);
  const activeDistance = Math.max(0, Math.min(-minTimeline, maxTimeline)) + 1;

  return Math.abs(timeline.id) <= activeDistance;
}

export function presentTime(game: GameSnapshot): number {
  if (Number.isInteger(game.presentTime)) {
    return game.presentTime as number;
  }

  // Keep the WebGPU snapshot path independent from the CPU WASM state.
  const activeLatestTimes = game.timelines
    .filter((timeline) => isActiveTimeline(game, timeline))
    .map((timeline) => getLatestBoard(game, timeline.id)?.time)
    .filter((time): time is number => Number.isInteger(time));

  return activeLatestTimes.length ? Math.min(...activeLatestTimes) : 0;
}

export function samePosition(a: Position, b: Position): boolean {
  return a.timelineId === b.timelineId && a.time === b.time && a.x === b.x && a.y === b.y;
}

export function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
