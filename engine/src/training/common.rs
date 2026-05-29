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
    max_seconds: Option<u64>,
    out: Option<String>,
    score: Option<String>,
    score_default: bool,
    train_cycle: bool,
    compare_seeds: Vec<u64>,
    min_wins: usize,
    min_total_delta: i32,
    verify: String,
    ai_src: String,
    hall_of_fame: String,
    min_pairs: usize,
    pair_batch: usize,
    max_pairs: usize,
    draw_window: usize,
    draw_rate_limit: f64,
    max_generations_without_candidate: usize,
}

#[derive(Clone)]
struct Lcg {
    // Deterministic tiny RNG: good enough for repeatable mutation/crossover and
    // keeps training independent of extra dependencies.
    state: u64,
}
