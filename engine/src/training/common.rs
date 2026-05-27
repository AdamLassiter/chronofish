// Native-only genetic training harness for EvalWeights. It plays short matches,
// compares candidate weights against the committed defaults, and can promote a
// statistically significant improvement by patching ai.rs and committing it.
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone)]
struct TrainerConfig {
    generations: usize,
    population: usize,
    depth: i32,
    nodes: usize,
    plies: usize,
    seed: u64,
    time_budget_secs: u64,
    out: Option<String>,
    score: Option<String>,
    score_default: bool,
    train_cycle: bool,
    compare_seeds: Vec<u64>,
    min_wins: usize,
    min_total_delta: i32,
    verify: String,
    ai_src: String,
}

#[derive(Clone)]
struct Lcg {
    // Deterministic tiny RNG: good enough for repeatable mutation/crossover and
    // keeps training independent of extra dependencies.
    state: u64,
}
