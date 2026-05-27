use std::cell::RefCell;

include!("model.rs");
include!("wasm_api.rs");
include!("game.rs");
include!("movegen.rs");
include!("ai.rs");
include!("notation.rs");

#[cfg(not(target_arch = "wasm32"))]
include!("training.rs");

#[cfg(test)]
include!("tests.rs");
