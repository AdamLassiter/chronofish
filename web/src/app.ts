/** Entry point: wires the page controls to their actions and starts the session. */

import {
  abandonGame,
  retryAiMove,
  submitTurn,
  undoStagedMove,
} from "./actions.js";
import { api } from "./api.js";
import { renderGame } from "./board.js";
import {
  closeBugReport,
  openBugReport,
  submitBugReport,
} from "./bug-report.js";
import { element, errorMessage, formString, show } from "./dom.js";
import { openHelp } from "./help.js";
import { setHistoryOpen } from "./history.js";
import {
  loadLobby,
  logout,
  openGame,
  setIdentity,
  startPolling,
  updateOpponents,
} from "./lobby.js";
import { currentGame, state } from "./state.js";
import { GameDetail, Session, User } from "./types.js";

document
  .querySelectorAll<HTMLButtonElement>("[data-auth-mode]")
  .forEach((button) => {
    button.onclick = () => {
      document
        .querySelectorAll<HTMLButtonElement>("[data-auth-mode]")
        .forEach((other) => other.classList.toggle("active", other === button));
      const mode = button.dataset.authMode;
      if (mode !== "login" && mode !== "register") return;
      element<HTMLFormElement>("#auth-form").dataset.mode = mode;
      element("#display-name-field").hidden = mode !== "register";
    };
  });

const authForm = element<HTMLFormElement>("#auth-form");

authForm.dataset.mode = "login";

authForm.onsubmit = (event) => {
  event.preventDefault();
  void (async () => {
    const form = new FormData(authForm);
    const mode = authForm.dataset.mode;
    if (mode !== "login" && mode !== "register") return;
    try {
      const displayName = form.get("displayName");
      const session = await api<Session>(`/api/auth/${mode}`, {
        method: "POST",
        body: {
          username: formString(form, "username"),
          password: formString(form, "password"),
          displayName:
            typeof displayName === "string" && displayName.length > 0
              ? displayName
              : undefined,
        },
      });
      state.token = session.token;
      state.user = session.user;
      localStorage.setItem("chronofish-token", session.token);
      setIdentity();
      await loadLobby();
    } catch (error) {
      element("#auth-error").textContent = errorMessage(error);
    }
  })();
};

const newGameForm = element<HTMLFormElement>("#new-game-form");

element<HTMLSelectElement>("#new-game-form select[name=ruleset]").onchange =
  updateOpponents;

newGameForm.onsubmit = (event) => {
  event.preventDefault();
  void (async () => {
    try {
      const form = new FormData(newGameForm);
      state.current = await api<GameDetail>("/api/games", {
        method: "POST",
        body: {
          opponent: formString(form, "opponent"),
          ruleset: formString(form, "ruleset"),
          playAs: formString(form, "playAs"),
        },
      });
      delete element("#multiverse").dataset.positioned;
      delete element("#multiverse").dataset.presentTime;
      renderGame();
      show("game-view");
      startPolling(currentGame().summary.id);
    } catch (error) {
      alert(errorMessage(error));
    }
  })();
};

element<HTMLButtonElement>("#back-to-lobby").onclick = () => {
  history.replaceState(null, "", location.pathname);
  void loadLobby();
};

element<HTMLButtonElement>("#refresh-game").onclick = () =>
  void openGame(currentGame().summary.id);

element<HTMLButtonElement>("#retry-ai").onclick = () => void retryAiMove();

element<HTMLButtonElement>("#undo-staged-move").onclick = () =>
  void undoStagedMove();

element<HTMLButtonElement>("#abandon-game").onclick = () => void abandonGame();

element<HTMLButtonElement>("#raise-bug").onclick = openBugReport;

element<HTMLButtonElement>("#submit-turn").onclick = () => void submitTurn();

const bugReportDialog = element<HTMLDialogElement>("#bug-report-dialog");

const bugReportForm = element<HTMLFormElement>("#bug-report-form");

bugReportForm.onsubmit = (event) => {
  event.preventDefault();
  void submitBugReport(bugReportForm);
};

bugReportDialog.onclose = () => {
  bugReportForm.reset();
  element("#bug-report-error").textContent = "";
};

element<HTMLButtonElement>("#close-bug-report").onclick = closeBugReport;

element<HTMLButtonElement>("#toggle-history").onclick = () =>
  setHistoryOpen(element("#history-drawer").hidden);

element<HTMLButtonElement>("#close-history").onclick = () =>
  setHistoryOpen(false);

document.addEventListener("keydown", (event) => {
  // Open dialogs handle Escape themselves; this closes the non-modal drawer.
  if (event.key === "Escape" && !document.querySelector("dialog[open]"))
    setHistoryOpen(false);
});

element<HTMLButtonElement>("#open-help").onclick = () => void openHelp();

element<HTMLButtonElement>("#close-help").onclick = () =>
  element<HTMLDialogElement>("#help-dialog").close();

element<HTMLButtonElement>("#cancel-bug-report").onclick = closeBugReport;

void (async () => {
  if (!state.token) {
    show("auth-view");
    return;
  }
  try {
    state.user = await api<User>("/api/me");
    setIdentity();
    await loadLobby();
    const match = location.hash.match(/^#game\/(\d+)$/);
    if (match?.[1]) await openGame(Number(match[1]));
  } catch {
    await logout();
  }
})();
