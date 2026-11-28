/** Square highlights and move arrows drawn over the boards. */

import {
  BoardSnapshot,
  GameView,
  HighlightMap,
  MoveArrow,
  Piece,
  Point,
  Position,
  SquareHighlight,
  Timeline,
} from "./types.js";

export function highlightKey(position: Position): string {
  return `${position.timelineId}:${position.time}:${position.x}:${position.y}`;
}

export function addHighlight(
  highlights: HighlightMap,
  position: Position,
  type: SquareHighlight,
): void {
  const key = highlightKey(position);
  const existing = highlights.get(key) ?? new Set<SquareHighlight>();
  existing.add(type);
  highlights.set(key, existing);
}

export function samePiece(left: Piece | null, right: Piece | null): boolean {
  if (!left || !right) return left === right;
  return left.color === right.color && left.type === right.type;
}

export function addArrivalHighlights(
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

export function movementVisuals(game: GameView): {
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

export function measureSquareCenters(grid: HTMLElement): Map<string, Point> {
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

export function arrowPath(from: Point, to: Point): string {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);
  if (distance === 0) return "";
  const inset = Math.min(18, distance / 3);
  return `M ${(from.x + (dx / distance) * inset).toFixed(1)} ${(from.y + (dy / distance) * inset).toFixed(1)} L ${(to.x - (dx / distance) * inset).toFixed(1)} ${(to.y - (dy / distance) * inset).toFixed(1)}`;
}

export function renderMoveArrows(grid: HTMLElement, arrows: MoveArrow[]): void {
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
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "move-arrow");
    path.setAttribute("d", pathData);
    path.setAttribute("marker-end", "url(#move-arrow-head)");
    svg.append(path);
  }
  grid.append(svg);
}
