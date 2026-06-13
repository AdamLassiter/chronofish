use super::*;
use crate::ai::effort::{ai_effort_config, default_ai_effort};

pub fn run_training_cli() {
    let config = TrainerConfig::from_env(std::env::args().skip(1).collect());

    if config.train_cycle {
        // The top-level ./train script loops this mode until interrupted.
        run_training_cycle(&config);
        return;
    }

    if config.score_default {
        println!(
            "{}",
            fitness(EvalWeights::default_tuned(), &config).summary()
        );
        return;
    }

    if let Some(path) = &config.score {
        let json = std::fs::read_to_string(path).expect("failed to read score weights");
        let weights = EvalWeights::from_json(&json).expect("failed to parse score weights");
        println!("{}", fitness(weights, &config).summary());
        return;
    }

    let weights = train_weights(&config);
    let json = weights.to_json();
    if let Some(path) = &config.out {
        std::fs::write(path, &json).expect("failed to write training output");
    }
    println!("{json}");
}

impl TrainerConfig {
    pub(crate) fn from_env(args: Vec<String>) -> Self {
        // The script-facing CLI is intentionally small, so hand parsing keeps the
        // training harness dependency-free.
        let seed = random_seed();
        let effort = default_ai_effort();
        let mut config = Self {
            effort: "expert".to_string(),
            generations: usize::MAX,
            population: auto_population(),
            training_time_ms: effort.training_time_ms,
            nodes: effort.training_nodes.max(auto_nodes()),
            seed,
            max_seconds: None,
            out: None,
            score: None,
            score_default: false,
            train_cycle: false,
            compare_seeds: default_compare_seeds(seed),
            min_wins: 0,
            min_total_delta: 0,
            verify: "cargo test -q".to_string(),
            ai_src: "engine/models/cpu-v1/parameters.json".to_string(),
            hall_of_fame: "engine/src/ai/hall_of_fame.jsonl".to_string(),
            min_pairs: 12,
            pair_batch: host_parallelism().max(1),
            max_pairs: 48,
            draw_window: 24,
            draw_rate_limit: 0.80,
            max_match_plies: 40,
            max_match_time_ms: 0,
            max_generations_without_candidate: 12,
            finalist_count: auto_finalists(),
            search_strategy: TrainingSearchStrategy::AlphaBeta,
        };
        let mut index = 0;
        let mut compare_seeds_overridden = false;
        while index < args.len() {
            let value = args.get(index + 1).cloned();
            match args[index].as_str() {
                "--train-cycle" => {
                    config.train_cycle = true;
                    index += 1;
                }
                "--generations" => {
                    config.generations = parse_arg(value, config.generations);
                    index += 2;
                }
                "--population" => {
                    config.population = parse_arg(value, config.population);
                    index += 2;
                }
                "--config" | "--effort" => {
                    if let Some(name) = value {
                        config.apply_effort(&name);
                    }
                    index += 2;
                }
                "--depth" => {
                    // Training search is now time bounded. Keep consuming the
                    // retired flag so older scripts do not skew later args.
                    index += 2;
                }
                "--training-time-ms" | "--turn-time-ms" => {
                    config.training_time_ms = parse_arg(value, config.training_time_ms);
                    index += 2;
                }
                "--nodes" => {
                    config.nodes = parse_arg(value, config.nodes);
                    index += 2;
                }
                "--search-strategy" => {
                    if let Some(strategy) = value {
                        config.search_strategy = TrainingSearchStrategy::parse(&strategy)
                            .unwrap_or_else(|message| panic!("{message}"));
                    }
                    index += 2;
                }
                "--plies" => {
                    // Full-match training no longer uses plies, but keep consuming
                    // the flag so older local scripts do not skew later args.
                    index += 2;
                }
                "--seed" => {
                    config.seed = parse_arg(value, config.seed);
                    index += 2;
                }
                "--max-seconds" | "--time-seconds" | "--time-budget" => {
                    config.max_seconds = value.and_then(|raw| raw.parse().ok());
                    index += 2;
                }
                "--out" => {
                    config.out = value;
                    index += 2;
                }
                "--score" => {
                    config.score = value;
                    index += 2;
                }
                "--score-default" => {
                    config.score_default = true;
                    index += 1;
                }
                "--compare-seeds" => {
                    config.compare_seeds =
                        parse_seed_list(value.as_deref()).unwrap_or(config.compare_seeds);
                    compare_seeds_overridden = true;
                    index += 2;
                }
                "--min-wins" => {
                    config.min_wins = parse_arg(value, config.min_wins);
                    index += 2;
                }
                "--min-total-delta" => {
                    config.min_total_delta = parse_arg(value, config.min_total_delta);
                    index += 2;
                }
                "--verify" => {
                    config.verify = value.unwrap_or(config.verify);
                    index += 2;
                }
                "--ai-src" => {
                    config.ai_src = value.unwrap_or(config.ai_src);
                    index += 2;
                }
                "--hall-of-fame" => {
                    config.hall_of_fame = value.unwrap_or(config.hall_of_fame);
                    index += 2;
                }
                "--min-pairs" => {
                    config.min_pairs = parse_arg(value, config.min_pairs);
                    index += 2;
                }
                "--pair-batch" => {
                    config.pair_batch = parse_arg(value, config.pair_batch);
                    index += 2;
                }
                "--max-pairs" => {
                    config.max_pairs = parse_arg(value, config.max_pairs);
                    index += 2;
                }
                "--draw-window" => {
                    config.draw_window = parse_arg(value, config.draw_window);
                    index += 2;
                }
                "--draw-rate-limit" => {
                    config.draw_rate_limit = parse_arg(value, config.draw_rate_limit);
                    index += 2;
                }
                "--max-match-plies" | "--match-plies" => {
                    config.max_match_plies = parse_arg(value, config.max_match_plies);
                    index += 2;
                }
                "--max-match-ms" | "--match-ms" => {
                    config.max_match_time_ms = parse_arg(value, config.max_match_time_ms);
                    index += 2;
                }
                "--max-generations-without-candidate" => {
                    config.max_generations_without_candidate =
                        parse_arg(value, config.max_generations_without_candidate);
                    index += 2;
                }
                "--finalists" => {
                    config.finalist_count = parse_arg(value, config.finalist_count);
                    index += 2;
                }
                _ => index += 1,
            }
        }
        config.population = config.population.max(4);
        config.pair_batch = config.pair_batch.max(1);
        config.min_pairs = config.min_pairs.max(1);
        config.max_pairs = config.max_pairs.max(config.min_pairs);
        config.draw_window = config.draw_window.max(1);
        config.draw_rate_limit = config.draw_rate_limit.clamp(0.0, 1.0);
        config.max_match_plies = config.max_match_plies.max(1);
        config.max_generations_without_candidate = config.max_generations_without_candidate.max(1);
        config.finalist_count = config.finalist_count.clamp(2, config.population);
        if !compare_seeds_overridden {
            config.compare_seeds = default_compare_seeds(config.seed);
        }
        if config.min_wins == 0 {
            config.min_wins = config.compare_seeds.len() * 2 / 3 + 1;
        }
        if config.min_total_delta == 0 {
            config.min_total_delta = (config.compare_seeds.len() as i32) * 50;
        }
        config
    }

    pub(crate) fn with_search(&self, nodes: usize, training_time_ms: u64) -> Self {
        let mut config = self.clone();
        config.nodes = nodes;
        config.training_time_ms = training_time_ms;
        config
    }

    pub(crate) fn screening_search(&self) -> Self {
        let mut config = self.clone();
        config.nodes = (self.nodes / 4).max(20).min(self.nodes);
        config.training_time_ms = (self.training_time_ms / 4)
            .max(1)
            .min(self.training_time_ms);
        config
    }

    pub(crate) fn apply_effort(&mut self, name: &str) {
        if let Some(effort) = ai_effort_config(name) {
            self.effort = name.to_string();
            self.training_time_ms = effort.training_time_ms;
            self.nodes = effort.training_nodes;
        }
    }
}
