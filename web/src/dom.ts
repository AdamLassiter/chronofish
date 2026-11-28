/** DOM lookups, escaping, notices and view switching. */

import { setHistoryOpen } from "./history.js";

export function element<T extends HTMLElement = HTMLElement>(
  selector: string,
): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`Missing required element: ${selector}`);
  return node;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function show(view: "auth-view" | "lobby-view" | "game-view"): void {
  for (const id of ["auth-view", "lobby-view", "game-view"])
    element(`#${id}`).hidden = id !== view;
  // The game view fills the viewport, and only its board pane scrolls.
  document.body.dataset.view = view;
  if (view !== "game-view") setHistoryOpen(false);
}

export function escapeHtml(value: string): string {
  const span = document.createElement("span");
  span.textContent = value;
  return span.innerHTML;
}

export function notice(message: string, error = false): void {
  const node = element("#game-notice");
  node.textContent = message;
  node.title = message;
  node.classList.toggle("error", error);
}

export function formString(form: FormData, name: string): string {
  const value = form.get(name);
  if (typeof value !== "string") throw new Error(`Missing field: ${name}`);
  return value;
}
