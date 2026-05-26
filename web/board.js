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
  return [...game.timelines].sort((a, b) => a.row - b.row || a.id - b.id);
}

export function sortedBoards(timeline) {
  return [...timeline.boards].sort((a, b) => a.time - b.time);
}

export function isActiveTimeline(game, timeline) {
  if (timeline.owner === "neutral") {
    return true;
  }

  const sameOwnerRank = game.timelines
    .filter((candidate) => candidate.owner === timeline.owner && candidate.id <= timeline.id)
    .length;
  const opponentOwner = timeline.owner === "white" ? "black" : "white";
  const opponentCount = game.timelines.filter((candidate) => candidate.owner === opponentOwner).length;

  return sameOwnerRank <= opponentCount + 1;
}

export function presentTime(game) {
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
