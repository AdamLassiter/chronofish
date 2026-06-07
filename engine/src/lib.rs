use std::cell::RefCell;

// The engine is intentionally assembled as one Rust module split across smaller
// files. wasm exports, move generation, notation, AI, and tests all share many
// helpers; include! keeps those helpers private without turning them into a wide
// pub(crate) API.
include!("model.rs");
include!("wasm_api.rs");
include!("game.rs");
include!("movegen.rs");
include!("notation.rs");
include!("notation_parser.rs");
#[cfg(not(test))]
include!("ai/types.rs");
#[cfg(not(test))]
include!("ai/effort.rs");
#[cfg(not(test))]
include!("ai/model.rs");
#[cfg(not(test))]
include!("ai/evaluator.rs");
#[cfg(not(test))]
include!("ai/weights.rs");
#[cfg(not(test))]
include!("ai/evaluation.rs");
#[cfg(not(test))]
include!("ai/search.rs");
#[cfg(not(test))]
include!("ai/search_plans.rs");
#[cfg(not(test))]
include!("ai/search_support.rs");
#[cfg(not(test))]
include!("ai/json.rs");

#[cfg(test)]
include!("ai.rs");
#[cfg(test)]
include!("gpu_snapshot.rs");
#[cfg(test)]
include!("notation_replay.rs");

#[cfg(not(target_arch = "wasm32"))]
include!("training.rs");

#[cfg(test)]
include!("tests.rs");
