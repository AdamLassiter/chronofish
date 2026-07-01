pub mod cpu;
pub mod gpu;
pub mod model;
pub(crate) use model::*;

pub(crate) mod wasm_api;

mod game;
mod gpu_snapshot;
mod hash;
mod movegen;
mod movegen_piece;
mod notation;

#[cfg(not(target_arch = "wasm32"))]
mod training;
#[cfg(not(target_arch = "wasm32"))]
pub use training::run_training_cli;

#[cfg(test)]
mod tests;
