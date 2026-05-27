pub fn run_training_cli() {
    let config = TrainerConfig::from_env(std::env::args().skip(1).collect());

    if config.train_cycle {
        // The top-level ./train script loops this mode until interrupted.
        run_training_cycle(&config);
        return;
    }

    if config.score_default {
        println!("{}", fitness(EvalWeights::default_tuned(), &config));
        return;
    }

    if let Some(path) = &config.score {
        let json = std::fs::read_to_string(path).expect("failed to read score weights");
        let weights = EvalWeights::from_json(&json).expect("failed to parse score weights");
        println!("{}", fitness(weights, &config));
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
    fn from_env(args: Vec<String>) -> Self {
        // The script-facing CLI is intentionally small, so hand parsing keeps the
        // training harness dependency-free.
        let seed = random_seed();
        let mut config = Self {
            generations: usize::MAX,
            population: auto_population(),
            depth: 2,
            nodes: auto_nodes(),
            plies: 4,
            seed,
            time_budget_secs: 600,
            out: None,
            score: None,
            score_default: false,
            train_cycle: false,
            compare_seeds: default_compare_seeds(seed),
            min_wins: 0,
            min_total_delta: 0,
            verify: "cargo test -q".to_string(),
            ai_src: "engine/src/ai/parameters.rs".to_string(),
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
                "--depth" => {
                    config.depth = parse_arg(value, config.depth);
                    index += 2;
                }
                "--nodes" => {
                    config.nodes = parse_arg(value, config.nodes);
                    index += 2;
                }
                "--plies" => {
                    config.plies = parse_arg(value, config.plies);
                    index += 2;
                }
                "--seed" => {
                    config.seed = parse_arg(value, config.seed);
                    index += 2;
                }
                "--time-seconds" | "--time-budget" => {
                    config.time_budget_secs = parse_arg(value, config.time_budget_secs);
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
                _ => index += 1,
            }
        }
        config.population = config.population.max(4);
        config.time_budget_secs = config.time_budget_secs.max(1);
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

    fn with_seed(&self, seed: u64) -> Self {
        Self {
            generations: self.generations,
            population: self.population,
            depth: self.depth,
            nodes: self.nodes,
            plies: self.plies,
            seed,
            time_budget_secs: self.time_budget_secs,
            out: self.out.clone(),
            score: self.score.clone(),
            score_default: self.score_default,
            train_cycle: self.train_cycle,
            compare_seeds: self.compare_seeds.clone(),
            min_wins: self.min_wins,
            min_total_delta: self.min_total_delta,
            verify: self.verify.clone(),
            ai_src: self.ai_src.clone(),
        }
    }

    fn with_search(&self, depth: i32, nodes: usize, plies: usize) -> Self {
        let mut config = self.clone();
        config.depth = depth;
        config.nodes = nodes;
        config.plies = plies;
        config
    }
}
