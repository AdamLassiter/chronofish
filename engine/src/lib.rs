use std::cell::RefCell;

// The engine is intentionally assembled as one Rust module split across smaller
// files. wasm exports, move generation, notation, AI, and tests all share many
// helpers; include! keeps those helpers private without turning them into a wide
// pub(crate) API.
include!("wasm_api.rs");

#[cfg(test)]
include!("ai.rs");
#[cfg(test)]
include!("model.rs");
#[cfg(test)]
include!("game.rs");
#[cfg(test)]
include!("movegen.rs");
#[cfg(test)]
include!("notation.rs");
#[cfg(test)]
include!("gpu_snapshot.rs");
#[cfg(test)]
include!("notation_parser.rs");
#[cfg(test)]
include!("notation_replay.rs");

#[cfg(all(not(target_arch = "wasm32"), test))]
include!("training.rs");

#[cfg(all(not(target_arch = "wasm32"), not(test)))]
pub fn run_training_cli() {
    eprintln!("Native CPU training is disabled; use the browser WebGPU training path.");
    std::process::exit(1);
}

#[cfg(test)]
include!("tests.rs");
