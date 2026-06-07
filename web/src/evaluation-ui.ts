import { getLatestBoard, isActiveTimeline } from "./board.js";
import { elements } from "./dom.js";
import type { BoardSnapshot, GameSnapshot, PieceType } from "./types.js";

export interface EvaluationUiController {
  renderEvaluationBar(): void;
  maybeScrollToPresent(options: ScrollPresentOptions): void;
}

export interface ScrollPresentOptions {
  phase: string;
  enteredGame: boolean;
  nextPresentTime: number | null;
  preserveScroll: boolean;
}

interface Evaluation {
  score: number;
  source: string;
}

export function createEvaluationUi({ getGame }: { getGame: () => GameSnapshot }): EvaluationUiController {
  let lastScrolledPresentTime: number | null = null;

  function renderEvaluationBar(): void {
    if (!elements.evaluationBar || !elements.evaluationWhite || !elements.evaluationScore) {
      return;
    }
    const evaluation = evaluateClientPosition(getGame());
    if (!evaluation) {
      elements.evaluationBar.hidden = true;
      return;
    }

    const { score, source } = evaluation;
    const whiteShare = 0.5 + 0.5 * normalizedEvaluation(score);
    const whitePercent = Math.max(3, Math.min(97, whiteShare * 100));
    elements.evaluationWhite.style.height = `${whitePercent}%`;
    elements.evaluationScore.textContent = formatEvaluation(score);
    elements.evaluationBar.dataset.leader = score >= 0 ? "white" : "black";
    elements.evaluationBar.title = `White ${formatSignedPawns(score)}. Source: ${source}.`;
    elements.evaluationBar.hidden = false;
  }

  function evaluateClientPosition(position: GameSnapshot): Evaluation | null {
    if (!position.timelines.length) {
      return null;
    }
    let score = 0;
    let boardCount = 0;
    for (const timeline of position.timelines) {
      if (!isActiveTimeline(position, timeline)) {
        continue;
      }
      const board = getLatestBoard(position, timeline.id);
      if (!board?.board) {
        continue;
      }
      boardCount += 1;
      score += evaluateBoardMaterial(board);
      score += board.sideToMove === "white" ? 12 : -12;
    }
    if (boardCount === 0) {
      return null;
    }
    return {
      score: Math.round(score / boardCount),
      source: "client material"
    };
  }

  function evaluateBoardMaterial(board: BoardSnapshot): number {
    let score = 0;
    for (const row of board.board) {
      for (const piece of row) {
        if (!piece) {
          continue;
        }
        const value = pieceValue(piece.type);
        score += piece.color === "white" ? value : -value;
      }
    }
    return score;
  }

  function scrollMultiverseToPresent(): void {
    if (!elements.timelineGrid || !elements.multiverse) {
      return;
    }
    const marker = elements.timelineGrid.querySelector<HTMLElement>('.timeline-row[data-active="true"] .present-line')
      ?? elements.timelineGrid.querySelector<HTMLElement>(".present-line");
    if (!marker) {
      return;
    }

    const markerBox = marker.getBoundingClientRect();
    const containerBox = elements.multiverse.getBoundingClientRect();
    elements.multiverse.scrollTo({
      left: elements.multiverse.scrollLeft
        + markerBox.left
        - containerBox.left
        - containerBox.width / 2
        + markerBox.width / 2,
      top: elements.multiverse.scrollTop
        + markerBox.top
        - containerBox.top
        - containerBox.height / 2
        + markerBox.height / 2,
      behavior: "smooth"
    });
  }

  function maybeScrollToPresent({ phase, enteredGame, nextPresentTime, preserveScroll }: ScrollPresentOptions): void {
    const previousPresentTime = lastScrolledPresentTime;
    if (!preserveScroll && phase === "game" && nextPresentTime !== null && (enteredGame || nextPresentTime !== previousPresentTime)) {
      scrollMultiverseToPresent();
    }
    lastScrolledPresentTime = phase === "game" ? nextPresentTime : null;
  }

  return {
    renderEvaluationBar,
    maybeScrollToPresent
  };
}

function pieceValue(type: PieceType): number {
  return {
    king: 20000,
    commonKing: 10000,
    queen: 900,
    royalQueen: 20000,
    princess: 700,
    rook: 500,
    bishop: 330,
    unicorn: 500,
    dragon: 900,
    knight: 320,
    pawn: 100,
    brawn: 130
  }[type] ?? 0;
}

function formatEvaluation(score: number): string {
  if (Math.abs(score) >= 90000) {
    return score > 0 ? "M" : "-M";
  }
  return formatSignedPawns(score);
}

function normalizedEvaluation(score: number): number {
  const maxCentipawns = 100000;
  const kneeCentipawns = 100;
  const magnitude = Math.min(Math.abs(score), maxCentipawns);
  const scaled = Math.log1p(magnitude / kneeCentipawns)
    / Math.log1p(maxCentipawns / kneeCentipawns);
  return Math.sign(score) * scaled;
}

function formatSignedPawns(score: number): string {
  const pawns = score / 100;
  if (Math.abs(pawns) < 0.05) {
    return "0.0";
  }
  return `${pawns > 0 ? "+" : ""}${pawns.toFixed(1)}`;
}
