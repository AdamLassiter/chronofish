export function appendNotationLine({ submittedNotation, turnNotation }) {
  if (!turnNotation) {
    return submittedNotation;
  }
  const turnNumber = submittedNotation.trim() === ""
    ? 1
    : submittedNotation.trim().split(/\n+/).length + 1;
  const line = `${turnNumber}. ${turnNotation}`;
  const nextNotation = submittedNotation ? `${submittedNotation}\n${line}` : line;
  console.log(line);
  return nextNotation;
}

export function postMatchLog(roomId, notation = "") {
  // Local games still have a generated room id, so the backend can write one log
  // file per game even when no multiplayer room was joined.
  fetch(`/api/logs/${encodeURIComponent(roomId)}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ notation })
  }).catch(() => {
    // The frontend should remain fully playable without the Rust server.
  });
}

export function postBotLossLog(roomId, payload) {
  fetch(`/api/training/loss-logs/${encodeURIComponent(roomId)}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  }).catch(() => {
    // Loss logs are training aids; gameplay should not depend on them.
  });
}
