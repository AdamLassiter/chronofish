use std::cell::RefCell;

// The engine is intentionally assembled as one Rust module split across smaller
// files. wasm exports, move generation, notation, AI, and tests all share many
// helpers; include! keeps those helpers private without turning them into a wide
// pub(crate) API.
include!("model.rs");
include!("wasm_api.rs");
include!("game.rs");
include!("movegen.rs");
include!("ai.rs");
include!("notation.rs");

// Training is native-only. It uses files, subprocesses, and git, none of which
// should be pulled into the browser/WASM artifact.
#[cfg(not(target_arch = "wasm32"))]
include!("training.rs");

#[cfg(test)]
include!("tests.rs");
