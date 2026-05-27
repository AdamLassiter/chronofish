use std::cell::RefCell;

include!("model.rs");
include!("wasm_api.rs");
include!("game.rs");
include!("movegen.rs");
include!("ai.rs");
include!("notation.rs");

#[cfg(test)]
include!("tests.rs");
