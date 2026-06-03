export function getTimeline(game, timelineId) {
  return game.timelines.find((timeline) => timeline.id === timelineId);
}

export function getBoard(game, timelineId, time) {
  return getTimeline(game, timelineId)?.boards.find((board) => board.time === time) ?? null;
}

export function getLatestBoard(game, timelineId) {
  const timeline = getTimeline(game, timelineId);
  return timeline?.boards.reduce((latest, board) => (board.time > latest.time ? board : latest), timeline.boards[0]);
}

export function isLatestBoard(game, timelineId, time) {
  return getLatestBoard(game, timelineId)?.time === time;
}

export function sortedTimelines(game) {
  // row is the visual/geometric timeline axis; id is only the stable tie-breaker.
  return [...game.timelines].sort((a, b) => a.row - b.row || a.id - b.id);
}

export function sortedBoards(timeline) {
  return [...timeline.boards].sort((a, b) => a.time - b.time);
}

export function isActiveTimeline(game, timeline) {
  // Mirror the Rust active-timeline rule so labels can be rendered without a
  // round-trip through WASM.
  if (timeline.owner === "neutral") {
    return true;
  }

  const ids = game.timelines.map((candidate) => candidate.id);
  const minTimeline = Math.min(...ids);
  const maxTimeline = Math.max(...ids);
  const activeDistance = Math.max(0, Math.min(-minTimeline, maxTimeline)) + 1;

  return Math.abs(timeline.id) <= activeDistance;
}

export function presentTime(game) {
  // Present time is the earliest latest board among active timelines.
  const activeLatestTimes = game.timelines
    .filter((timeline) => isActiveTimeline(game, timeline))
    .map((timeline) => getLatestBoard(game, timeline.id)?.time)
    .filter((time) => Number.isInteger(time));

  return activeLatestTimes.length ? Math.min(...activeLatestTimes) : 0;
}

export function samePosition(a, b) {
  return a.timelineId === b.timelineId && a.time === b.time && a.x === b.x && a.y === b.y;
}

export function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
