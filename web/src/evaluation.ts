/** The engine evaluation bar beside the boards. */

import { api } from "./api.js";
import { element } from "./dom.js";
import { moves } from "./moves.js";
import { Evaluation, GameSummary, Side } from "./types.js";

export let evaluationRequest = 0;

/** Orients the bar for the viewer and fetches a new verdict when the game changes. */
export function renderEvaluationFrame(
  summary: GameSummary,
  viewerSide: Side | null,
): void {
  const bar = element("#eval-bar");
  bar.dataset.flipped = String(viewerSide === "black");
  const key = `${summary.id}:${summary.version}`;
  if (bar.dataset.version === key) return;
  bar.dataset.version = key;
  bar.dataset.pending = "true";
  const request = ++evaluationRequest;
  void (async () => {
    try {
      const evaluation = await api<Evaluation>(
        `/api/games/${summary.id}/evaluation`,
      );
      if (request === evaluationRequest) showEvaluation(bar, evaluation);
    } catch {
      // Leave the previous verdict dimmed; the next game update retries.
      if (request === evaluationRequest) delete bar.dataset.version;
    }
  })();
}

export function showEvaluation(bar: HTMLElement, evaluation: Evaluation): void {
  const score = Math.max(-1, Math.min(1, evaluation.whiteScore));
  bar.querySelector<HTMLElement>(".eval-fill")!.style.height =
    `${((score + 1) / 2) * 100}%`;
  const leader: Side = evaluation.winner ?? (score < 0 ? "black" : "white");
  bar.dataset.leader = leader;
  bar.dataset.pending = "false";
  bar.setAttribute("aria-valuenow", score.toFixed(3));
  let label: string;
  let detail: string;
  if (evaluation.finished) {
    label =
      evaluation.winner === null
        ? "½"
        : evaluation.winner === "white"
          ? "1-0"
          : "0-1";
    detail =
      evaluation.winner === null
        ? "Game drawn"
        : `${evaluation.winner === "white" ? "White" : "Black"} won`;
  } else if (evaluation.winner !== null) {
    label = "#";
    detail = `${evaluation.winner === "white" ? "White" : "Black"} has a forced win`;
  } else {
    const pawns = Math.abs(evaluation.centipawns ?? 0) / 100;
    label = pawns >= 10 ? pawns.toFixed(0) : pawns.toFixed(1);
    const side = (evaluation.centipawns ?? 0) < 0 ? "Black" : "White";
    detail =
      pawns < 0.05
        ? "Level position"
        : `${side} is ahead by ${pawns.toFixed(1)} pawns`;
  }
  bar.querySelector(".eval-score")!.textContent = label;
  bar.setAttribute("aria-valuetext", detail);
  bar.title = `${evaluation.engine}: ${detail}. Evaluates the last submitted turn; staged moves are ignored.`;
}
