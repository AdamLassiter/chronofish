/** Client session state shared across modules. */

import { AppState, GameDetail, User } from "./types.js";

/** In-flight move, undo or submit; blocks further board input. */
export const requests = { turnPending: false };

export const state: AppState = {
  token: localStorage.getItem("chronofish-token"),
  user: null,
  users: [],
  current: null,
  selected: null,
  poll: null,
};

export function currentGame(): GameDetail {
  if (!state.current) throw new Error("No game is currently open");
  return state.current;
}

export function currentUser(): User {
  if (!state.user) throw new Error("No user is signed in");
  return state.user;
}
