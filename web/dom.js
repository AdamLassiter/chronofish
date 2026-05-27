// Centralized DOM lookups keep main.js focused on engine/network state and make
// the required index.html ids explicit.
export const elements = {
  timelineGrid: document.querySelector("#timeline-grid"),
  message: document.querySelector("#message"),
  wasmStatus: document.querySelector("#wasm-status"),
  serverStatus: document.querySelector("#server-status"),
  hud: document.querySelector("#hud"),
  toggleHudButton: document.querySelector("#toggle-hud"),
  resetButton: document.querySelector("#reset-game"),
  undoMoveButton: document.querySelector("#undo-move"),
  submitTurnButton: document.querySelector("#submit-turn"),
  roomInput: document.querySelector("#room-id"),
  whitePlayerSelect: document.querySelector("#white-player"),
  blackPlayerSelect: document.querySelector("#black-player"),
  startGameButton: document.querySelector("#start-game"),
  joinWhiteButton: document.querySelector("#join-white"),
  joinBlackButton: document.querySelector("#join-black"),
  joinSpectatorButton: document.querySelector("#join-spectator"),
  multiplayerStatus: document.querySelector("#multiplayer-status"),
  botStatus: document.querySelector("#bot-status"),
  shareLink: document.querySelector("#share-link")
};
