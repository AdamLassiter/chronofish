export function readWasmString(engine, ptr) {
  // Rust string exports share one output buffer. Copy the bytes immediately after
  // receiving a pointer because the next engine string call overwrites it.
  const bytes = new Uint8Array(engine.memory.buffer, ptr, engine.chronofish_output_len());
  return new TextDecoder("utf-8").decode(bytes);
}
