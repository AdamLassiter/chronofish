mod model;
pub(crate) use model::*;

mod game;
mod hash;
mod movegen;
mod movegen_piece;
mod notation;
pub(crate) use notation::*;
mod ai;
pub(crate) use ai::*;
mod wasm_api;

#[cfg(test)]
mod gpu_snapshot;

#[cfg(not(target_arch = "wasm32"))]
mod training;
#[cfg(not(target_arch = "wasm32"))]
pub use training::run_training_cli;

#[cfg(test)]
mod tests;
