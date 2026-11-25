//! Network-free Chronofish bots, each a different classic game-playing architecture.
//!
//! - [`Grabablanca`]: greedy one-action lookahead.
//! - [`DeepBlueshift`]: iterative-deepening alpha-beta with a transposition table.
//! - [`MonteCarlsen`]: UCT Monte Carlo tree search with heuristic rollouts.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod alphabeta;
pub mod eval;
mod greedy;
mod uct;

pub use alphabeta::{BlueshiftConfig, DeepBlueshift};
pub use greedy::Grabablanca;
pub use uct::{CarlsenConfig, MonteCarlsen};

#[cfg(test)]
mod tests;
