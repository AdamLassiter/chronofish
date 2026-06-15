import { elements } from "./dom.js";
import type { ChronofishEngine, Color, GameSnapshot, Move } from "./types.js";

type LobbyRole = Color | "spectator";

interface TrainingController {
  openModal(): void;
  closeModal(): void;
  start(): void;
  stop(): void;
}

interface MainEventOptions {
  getEngine(): ChronofishEngine | null;
  getGame(): GameSnapshot;
  getStagedMoves(): Move[];
  canActNow(): boolean;
  playerDisplayName(color: Color): string;
  resetStagedClientState(): void;
  persistLocalGameState(): void;
  render(): void;
  cloneMove(move: Move): Move;
  rebuildStagedClientState(moves: Move[]): Promise<void>;
  submitVisibleTurn(actor: Color): Promise<string | null>;
  clearPlannedMoves(): void;
  isMatchOver(): boolean;
  enterPostMatchReview(message: string): void;
  syncState(kind: string, message: string): void;
  maybeStartBotTurn(): void;
  capitalize(value: string): string;
  concede(color: Color): void;
  setHudCollapsed(collapsed: boolean): void;
  training: TrainingController;
  joinRoom(role: LobbyRole): Promise<void>;
  startGame(): Promise<void>;
  openCustomCpuModal(): void;
  closeCustomCpuModal(): void;
  applyCustomCpuModal(): void;
  resetCustomCpuModal(): void;
  openCustomGpuModal(): void;
  closeCustomGpuModal(): void;
  applyCustomGpuModal(): void;
  resetCustomGpuModal(): void;
  writeAssignments(assignments: unknown): void;
  readAssignments(): unknown;
  syncLobby(): void;
}

export function wireMainEvents({
  getEngine,
  getGame,
  getStagedMoves,
  canActNow,
  playerDisplayName,
  resetStagedClientState,
  persistLocalGameState,
  render,
  cloneMove,
  rebuildStagedClientState,
  submitVisibleTurn,
  clearPlannedMoves,
  isMatchOver,
  enterPostMatchReview,
  syncState,
  maybeStartBotTurn,
  capitalize,
  concede,
  setHudCollapsed,
  training,
  joinRoom,
  startGame,
  openCustomCpuModal,
  closeCustomCpuModal,
  applyCustomCpuModal,
  resetCustomCpuModal,
  openCustomGpuModal,
  closeCustomGpuModal,
  applyCustomGpuModal,
  resetCustomGpuModal,
  writeAssignments,
  readAssignments,
  syncLobby
}: MainEventOptions): void {
  const message = requireElement(elements.message, "message");
  const resetButton = requireElement(elements.resetButton, "reset-game");
  const undoMoveButton = requireElement(elements.undoMoveButton, "undo-move");
  const submitTurnButton = requireElement(elements.submitTurnButton, "submit-turn");
  const clearPlansButton = requireElement(elements.clearPlansButton, "clear-plans");
  const concedeButton = requireElement(elements.concedeButton, "concede-game");
  const toggleHudButton = requireElement(elements.toggleHudButton, "toggle-hud");
  const hud = requireElement(elements.hud, "hud");
  const openTrainingButton = requireElement(elements.openTrainingButton, "open-training");
  const closeTrainingButton = requireElement(elements.closeTrainingButton, "close-training");
  const trainingModal = requireElement(elements.trainingModal, "training-modal");
  const joinWhiteButton = requireElement(elements.joinWhiteButton, "join-white");
  const joinBlackButton = requireElement(elements.joinBlackButton, "join-black");
  const joinSpectatorButton = requireElement(elements.joinSpectatorButton, "join-spectator");
  const startGameButton = requireElement(elements.startGameButton, "start-game");
  const startTrainingButton = requireElement(elements.startTrainingButton, "start-training");
  const stopTrainingButton = requireElement(elements.stopTrainingButton, "stop-training");
  const whitePlayerSelect = requireElement(elements.whitePlayerSelect, "white-player");
  const blackPlayerSelect = requireElement(elements.blackPlayerSelect, "black-player");
  const customCpuModal = requireElement(elements.customCpuModal, "custom-cpu-modal");
  const closeCustomCpuButton = requireElement(elements.closeCustomCpuButton, "close-custom-cpu");
  const saveCustomCpuButton = requireElement(elements.saveCustomCpuButton, "save-custom-cpu");
  const resetCustomCpuButton = requireElement(elements.resetCustomCpuButton, "reset-custom-cpu");
  const customGpuModal = requireElement(elements.customGpuModal, "custom-gpu-modal");
  const closeCustomGpuButton = requireElement(elements.closeCustomGpuButton, "close-custom-gpu");
  const saveCustomGpuButton = requireElement(elements.saveCustomGpuButton, "save-custom-gpu");
  const resetCustomGpuButton = requireElement(elements.resetCustomGpuButton, "reset-custom-gpu");

  resetButton.addEventListener("click", () => {
    if (!getEngine()) {
      message.textContent = "WASM getEngine() is not loaded yet.";
      return;
    }
    if (!canActNow()) {
      message.textContent = `Waiting for ${playerDisplayName(getGame().turn)}.`;
      return;
    }

    const undone = getStagedMoves().length;
    resetStagedClientState();
    message.textContent = undone > 0 ? "Reset staged moves." : "No staged moves to reset.";
    persistLocalGameState();
    render();
  });

  undoMoveButton.addEventListener("click", async () => {
    if (!getEngine()) {
      message.textContent = "WASM getEngine() is not loaded yet.";
      return;
    }
    if (!canActNow()) {
      message.textContent = `Waiting for ${playerDisplayName(getGame().turn)}.`;
      return;
    }

    if (getStagedMoves().length === 0) {
      message.textContent = "No staged move to undo.";
      return;
    }

    const remaining = getStagedMoves().slice(0, -1).map(cloneMove);
    await rebuildStagedClientState(remaining);
    message.textContent = remaining.length === 0
      ? "Select a piece on a latest board."
      : "Undid staged move.";
    persistLocalGameState();
    render();
  });

  submitTurnButton.addEventListener("click", async () => {
    if (!getEngine()) {
      message.textContent = "WASM getEngine() is not loaded yet.";
      return;
    }
    if (!canActNow()) {
      message.textContent = `Waiting for ${playerDisplayName(getGame().turn)}.`;
      return;
    }

    const actor = getGame().turn;
    const turnMessage = await submitVisibleTurn(actor);
    if (!turnMessage) {
      return;
    }

    render();
    if (isMatchOver()) {
      enterPostMatchReview(turnMessage);
      return;
    }
    syncState("state", turnMessage);
    maybeStartBotTurn();
  });

  clearPlansButton.addEventListener("click", () => {
    clearPlannedMoves();
  });

  concedeButton.addEventListener("click", () => {
    if (!getEngine()) {
      message.textContent = "WASM getEngine() is not loaded yet.";
      return;
    }
    if (!canActNow()) {
      message.textContent = `Waiting for ${playerDisplayName(getGame().turn)}.`;
      return;
    }

    if (!window.confirm(`${capitalize(getGame().turn)} will concede. Continue?`)) {
      message.textContent = "Concession cancelled.";
      return;
    }

    concede(getGame().turn);
  });

  toggleHudButton.addEventListener("click", () => {
    setHudCollapsed(hud.dataset.collapsed !== "true");
  });

  openTrainingButton.addEventListener("click", () => {
    training.openModal();
  });

  closeTrainingButton.addEventListener("click", () => {
    training.closeModal();
  });

  trainingModal.addEventListener("click", (event) => {
    if (event.target === trainingModal) {
      training.closeModal();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }
    if (!trainingModal.hidden) {
      training.closeModal();
    }
    if (!customCpuModal.hidden) {
      closeCustomCpuModal();
    }
    if (!customGpuModal.hidden) {
      closeCustomGpuModal();
    }
  });

  joinWhiteButton.addEventListener("click", () => {
    joinRoom("white").catch((error: unknown) => {
      message.textContent = errorMessage(error);
    });
  });

  joinBlackButton.addEventListener("click", () => {
    joinRoom("black").catch((error: unknown) => {
      message.textContent = errorMessage(error);
    });
  });

  joinSpectatorButton.addEventListener("click", () => {
    joinRoom("spectator").catch((error: unknown) => {
      message.textContent = errorMessage(error);
    });
  });

  startGameButton.addEventListener("click", () => {
    startGame().catch((error: unknown) => {
      message.textContent = errorMessage(error);
    });
  });

  startTrainingButton.addEventListener("click", () => {
    training.start();
  });

  stopTrainingButton.addEventListener("click", () => {
    training.stop();
  });

  closeCustomCpuButton.addEventListener("click", () => {
    closeCustomCpuModal();
  });

  saveCustomCpuButton.addEventListener("click", () => {
    applyCustomCpuModal();
  });

  resetCustomCpuButton.addEventListener("click", () => {
    resetCustomCpuModal();
  });

  customCpuModal.addEventListener("click", (event) => {
    if (event.target === customCpuModal) {
      closeCustomCpuModal();
    }
  });

  closeCustomGpuButton.addEventListener("click", () => {
    closeCustomGpuModal();
  });

  saveCustomGpuButton.addEventListener("click", () => {
    applyCustomGpuModal();
  });

  resetCustomGpuButton.addEventListener("click", () => {
    resetCustomGpuModal();
  });

  customGpuModal.addEventListener("click", (event) => {
    if (event.target === customGpuModal) {
      closeCustomGpuModal();
    }
  });

  for (const select of [whitePlayerSelect, blackPlayerSelect]) {
    select.addEventListener("change", () => {
      writeAssignments(readAssignments());
      render();
      syncLobby();
      if (select.value === "bot-cpu-custom") {
        openCustomCpuModal();
      }
      if (select.value === "bot-gpu-custom") {
        openCustomGpuModal();
      }
    });
  }
}

function requireElement<T extends Element>(element: T | null, id: string): T {
  if (!element) {
    throw new Error(`Missing required #${id} element.`);
  }
  return element;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
