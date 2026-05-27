fn run_training_cycle(config: &TrainerConfig) {
    // Promotion rewrites the parameter include file, so refuse to continue when
    // it already has local edits that should not be mixed with generated tuning.
    if ai_source_is_dirty(&config.ai_src) {
        eprintln!(
            "{} has uncommitted changes; commit or stash before running training",
            config.ai_src
        );
        std::process::exit(1);
    }

    println!(
        "training budget={}s population={} base_depth={} base_nodes={} plies={} seed={}",
        config.time_budget_secs,
        config.population,
        config.depth,
        config.nodes,
        config.plies,
        config.seed
    );
    println!(
        "score note: fitness points are aggregate evaluation margins from short matches; comparison wins/losses are decided by candidate fitness minus baseline fitness per seed"
    );

    let started = SearchInstant::now();
    let deadline = started + std::time::Duration::from_secs(config.time_budget_secs);
    let train_deadline =
        started + std::time::Duration::from_secs((config.time_budget_secs * 3 / 4).max(1));
    let candidate = train_weights_until(config, train_deadline);
    let candidate_json = candidate.to_json();
    let mut wins = 0;
    let mut losses = 0;
    let mut draws = 0;
    let mut total_delta = 0;
    let mut deltas = Vec::new();

    println!("candidate weights: {candidate_json}");
    let comparisons: Vec<(u64, i32, i32)> = config
        .compare_seeds
        .par_iter()
        .copied()
        .filter_map(|seed| {
            if SearchInstant::now() >= deadline {
                return None;
            }
            let baseline_config = config.with_seed(seed);
            let baseline_score =
                fitness_until(EvalWeights::default_tuned(), &baseline_config, deadline);
            let candidate_score = fitness_until(candidate, &baseline_config, deadline);
            Some((seed, candidate_score, baseline_score))
        })
        .collect();
    if comparisons.len() < config.compare_seeds.len() {
        println!("comparison stopped: time budget exhausted");
    }
    for (seed, candidate_score, baseline_score) in comparisons {
        let delta = candidate_score - baseline_score;
        total_delta += delta;
        deltas.push(delta);
        let result = if delta > 0 {
            wins += 1;
            "win"
        } else if delta < 0 {
            losses += 1;
            "loss"
        } else {
            draws += 1;
            "draw"
        };
        println!(
            "seed {seed}: {result} candidate={candidate_score} baseline={baseline_score} delta={delta}"
        );
    }

    let significance = significance(&deltas);
    print_threshold_progress(wins, losses, draws, total_delta, significance, config);
    if should_promote(wins, losses, total_delta, significance, config) {
        promote_weights(candidate, &config.ai_src);
        run_command("cargo", &["fmt"]);
        run_shell(&config.verify);
        run_command("git", &["add", &config.ai_src]);
        run_command("git", &["commit", "-m", "Tune AI evaluation parameters"]);
        println!("promoted candidate and committed updated parameters");
    } else {
        println!("candidate rejected");
    }
}

fn train_weights(config: &TrainerConfig) -> EvalWeights {
    train_weights_until(
        config,
        SearchInstant::now() + std::time::Duration::from_secs(config.time_budget_secs),
    )
}

fn train_weights_until(config: &TrainerConfig, deadline: SearchInstant) -> EvalWeights {
    println!(
        "fitness score = material/tempo/check/present-line heuristic accumulated over {} plies against default and mutated opponents",
        config.plies
    );
    let mut rng = Lcg::new(config.seed);
    let mut population = vec![EvalWeights::default_tuned()];
    while population.len() < config.population {
        population.push(EvalWeights::default_tuned().mutate(&mut rng));
    }

    let mut previous_best: Option<i32> = None;
    let training_started = SearchInstant::now();
    let mut best_seen = EvalWeights::default_tuned();
    let mut best_seen_score = i32::MIN;
    for generation in 0..config.generations {
        if SearchInstant::now() >= deadline {
            break;
        }
        // Each generation is fully rescored against this config seed so best,
        // average, and improvement logs are comparable within the run.
        let started = SearchInstant::now();
        let elapsed = training_started.elapsed().as_secs();
        let third = (config.time_budget_secs / 3).max(1);
        let depth_boost = (elapsed / third).min(2) as i32;
        let search_depth = config.depth + depth_boost;
        let search_nodes = config.nodes * search_depth as usize;
        let search_plies = config.plies + depth_boost as usize * 2;
        let scoring_config = config.with_search(search_depth, search_nodes, search_plies);
        let started_candidates = AtomicUsize::new(0);
        let population_len = population.len();
        let mut scored: Vec<(i32, EvalWeights)> = population
            .par_iter()
            .copied()
            .filter_map(|weights| {
                if SearchInstant::now() >= deadline {
                    return None;
                }
                let index = started_candidates.fetch_add(1, Ordering::Relaxed);
                let remaining = deadline
                    .saturating_duration_since(SearchInstant::now())
                    .as_secs();
                eprintln!(
                    "generation {generation}: scoring candidate {}/{} depth={search_depth} nodes={search_nodes} plies={search_plies} remaining={}s",
                    index + 1,
                    population_len,
                    remaining
                );
                let score = fitness_until(weights, &scoring_config, deadline);
                Some((score, weights))
            })
            .collect();
        for (score, weights) in &scored {
            if *score > best_seen_score {
                best_seen = *weights;
                best_seen_score = *score;
            }
        }
        if scored.is_empty() {
            break;
        }
        scored.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        let best = scored[0].0;
        let worst = scored.last().map_or(best, |entry| entry.0);
        let average =
            scored.iter().map(|entry| entry.0 as i64).sum::<i64>() as f64 / scored.len() as f64;
        let improvement = previous_best.map_or(0, |previous| best - previous);
        previous_best = Some(best);
        let remaining = deadline
            .saturating_duration_since(SearchInstant::now())
            .as_secs();
        eprintln!(
            "generation {generation}: depth={search_depth} nodes={search_nodes} plies={search_plies} best={best} avg={average:.1} worst={worst} improvement={improvement:+} population={} gen_elapsed={:.2}s remaining={}s",
            scored.len(),
            started.elapsed().as_secs_f32(),
            remaining
        );

        let elite = 4.min(scored.len());
        let mut next: Vec<EvalWeights> = scored.iter().take(elite).map(|entry| entry.1).collect();
        while next.len() < config.population {
            let left = tournament(&scored, &mut rng);
            let right = tournament(&scored, &mut rng);
            next.push(EvalWeights::crossover(left, right, &mut rng).mutate(&mut rng));
        }
        population = next;
    }

    best_seen
}

fn fitness(weights: EvalWeights, config: &TrainerConfig) -> i32 {
    fitness_until(
        weights,
        config,
        SearchInstant::now() + std::time::Duration::from_secs(config.time_budget_secs),
    )
}

fn fitness_until(
    weights: EvalWeights,
    config: &TrainerConfig,
    deadline: SearchInstant,
) -> i32 {
    // Score candidates against the committed default and a few nearby mutated
    // opponents so the search improves the current engine instead of overfitting
    // to one self-play lineage.
    let mut rng = Lcg::new(config.seed);
    let mut total = 0;
    let default = EvalWeights::default_tuned();
    total += play_match_until(weights, default, Color::White, config, deadline);
    total += play_match_until(weights, default, Color::Black, config, deadline);

    for _ in 0..3 {
        if SearchInstant::now() >= deadline {
            break;
        }
        let opponent = default.mutate(&mut rng);
        total += play_match_until(weights, opponent, Color::White, config, deadline);
        total += play_match_until(weights, opponent, Color::Black, config, deadline);
    }

    total
}

fn play_match_until(
    weights: EvalWeights,
    opponent: EvalWeights,
    color: Color,
    config: &TrainerConfig,
    deadline: SearchInstant,
) -> i32 {
    // Matches are short by design; the heuristic should learn opening material,
    // tempo, and branch quality before deeper minimax is affordable.
    let mut game = Game::new();
    let mut score = 0;
    for ply in 0..config.plies {
        if SearchInstant::now() >= deadline {
            break;
        }
        let side_weights = if game.turn == color {
            weights
        } else {
            opponent
        };
        let mut context = SearchContext {
            weights: side_weights,
            root_color: game.turn,
            max_nodes: config.nodes,
            nodes: 0,
            deadline: Some(deadline),
            table: std::collections::HashMap::new(),
        };
        let Some((plan, _)) = game.search_root(config.depth, &mut context) else {
            score += if game.turn == color { -10_000 } else { 10_000 };
            break;
        };
        game = plan.game;
        let eval = game.evaluate(color, &weights);
        score += eval / 20 + eval.signum() * (config.plies - ply) as i32;
        if game.is_checkmate(color) || game.is_checkmate(color.opposite()) {
            score += if game.is_checkmate(color) {
                -CHECKMATE_SCORE / 10
            } else {
                CHECKMATE_SCORE / 10
            };
            break;
        }
    }
    score + game.evaluate(color, &weights) / 4
}

fn tournament(scored: &[(i32, EvalWeights)], rng: &mut Lcg) -> EvalWeights {
    // Tournament selection applies pressure toward stronger candidates while
    // preserving enough randomness for weaker genomes to contribute genes.
    let mut best = scored[rng.next_usize(scored.len())];
    for _ in 1..4 {
        let candidate = scored[rng.next_usize(scored.len())];
        if candidate.0 > best.0 {
            best = candidate;
        }
    }
    best.1
}

fn mutate_weight(value: i32, rng: &mut Lcg, spread: i32, min: i32, max: i32) -> i32 {
    let delta = rng.next_usize((spread * 2 + 1) as usize) as i32 - spread;
    (value + delta).clamp(min, max)
}

