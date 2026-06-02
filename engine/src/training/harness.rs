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
        "training config={} max_seconds={} population={} finalists={} base_depth={} base_nodes={} plies={} seed={} min_pairs={} max_pairs={}",
        config.effort,
        config
            .max_seconds
            .map(|seconds| seconds.to_string())
            .unwrap_or_else(|| "none".to_string()),
        config.population,
        config.finalist_count,
        config.depth,
        config.nodes,
        config.plies,
        config.seed,
        config.min_pairs,
        config.max_pairs
    );
    println!(
        "score note: fitness points are aggregate evaluation margins from short matches; comparison wins/losses are decided by candidate fitness minus baseline fitness per seed"
    );

    let deadline = training_deadline(config);
    let hall_of_fame = load_hall_of_fame(&config.hall_of_fame);
    let candidate = train_weights_until(config, deadline);
    if candidate == EvalWeights::default_tuned() {
        println!("candidate inconclusive: selected weights match committed baseline");
        return;
    }
    let candidate_json = candidate.to_json();
    let mut comparison_stats = ComparisonStats::default();
    let mut deltas = Vec::new();
    let mut comparison_match_stats = MatchStats::default();

    println!("candidate weights: {candidate_json}");
    let mut rng = Lcg::new(config.seed ^ 0xadc8_3b19_7f4a_7c15);
    loop {
        if training_expired(deadline) {
            println!("comparison stopped: max seconds exhausted");
            break;
        }
        let remaining_pairs = config.max_pairs.saturating_sub(comparison_stats.played);
        let seeds: Vec<u64> = (0..config.pair_batch.min(remaining_pairs))
            .take_while(|_| !training_expired(deadline))
            .map(|_| rng.next_u64())
            .collect();
        if seeds.is_empty() {
            break;
        }
        let reports: Vec<(u64, PairReport)> = seeds
            .par_iter()
            .copied()
            .map(|seed| (seed, paired_baseline_report(candidate, seed, config, deadline)))
            .collect();
        for (seed, report) in reports {
            comparison_match_stats.add(report.candidate.matches);
            comparison_match_stats.add(report.baseline.matches);
            let delta = report.delta();
        comparison_stats.record(delta);
        deltas.push(delta);
        let result = if delta > 0 {
            "win"
        } else if delta < 0 {
            "loss"
        } else {
            "draw"
        };
        println!(
            "seed {seed}: {result} candidate={} baseline={} delta={delta}",
                report.candidate.summary(),
                report.baseline.summary()
            );
        }
        let significance = significance(&deltas);
        match statistical_decision(comparison_stats, &deltas, significance, config) {
            StatisticalDecision::Promote => {
                println!("sequential comparison accepted");
                break;
            }
            StatisticalDecision::Reject => {
                println!("sequential comparison rejected");
                break;
            }
            StatisticalDecision::Inconclusive => {
                println!("sequential comparison inconclusive");
                break;
            }
            StatisticalDecision::Continue => {}
        }
    }

    let significance = significance(&deltas);
    print_threshold_progress(
        comparison_stats,
        comparison_match_stats,
        significance,
        config,
    );
    if should_promote(comparison_stats, significance, config) {
        promote_weights(candidate, &config.ai_src);
        append_hall_of_fame(&config.hall_of_fame, candidate);
        run_command("cargo", &["fmt"]);
        run_shell(&config.verify);
        run_command("git", &["add", &config.ai_src]);
        if !hall_of_fame.is_empty() || std::path::Path::new(&config.hall_of_fame).is_file() {
            run_command("git", &["add", &config.hall_of_fame]);
        }
        run_command("git", &["commit", "-m", "Tune AI evaluation parameters"]);
        println!("promoted candidate and committed updated parameters");
    } else {
        match statistical_decision(comparison_stats, &deltas, significance, config) {
            StatisticalDecision::Reject => println!("candidate rejected"),
            StatisticalDecision::Inconclusive | StatisticalDecision::Continue => {
                println!("candidate inconclusive")
            }
            StatisticalDecision::Promote => println!("candidate rejected"),
        }
    }
}

fn train_weights(config: &TrainerConfig) -> EvalWeights {
    train_weights_until(config, training_deadline(config))
}

fn train_weights_until(config: &TrainerConfig, deadline: Option<SearchInstant>) -> EvalWeights {
    println!(
        "fitness score = material/tempo/check/present-line heuristic accumulated over {} plies against default and mutated opponents",
        config.plies
    );
    let mut rng = Lcg::new(config.seed);
    let committed = EvalWeights::default_tuned();
    let mut population = vec![committed];
    while population.len() < config.population {
        population.push(committed.mutate(&mut rng));
    }

    let mut previous_best: Option<i32> = None;
    let training_started = SearchInstant::now();
    let mut best_seen = EvalWeights::default_tuned();
    let mut best_seen_score = i32::MIN;
    let mut mutation_scale = 1.0_f32;
    let hall_of_fame = load_hall_of_fame(&config.hall_of_fame);
    let mut generations_without_candidate = 0;
    for generation in 0..config.generations {
        if training_expired(deadline) {
            break;
        }
        // Each generation is fully rescored against this config seed so best,
        // average, and improvement logs are comparable within the run.
        let started = SearchInstant::now();
        let elapsed = training_started.elapsed().as_secs();
        let depth_boost = (elapsed / 600).min(2) as i32;
        let search_depth = config.depth + depth_boost;
        let search_nodes = config.nodes * search_depth as usize;
        let search_plies = config.plies + depth_boost as usize * 2;
        let scoring_config = config.with_search(search_depth, search_nodes, search_plies);
        let screening_config = scoring_config.screening_search();
        let baseline_report =
            fitness_until(EvalWeights::default_tuned(), &scoring_config, deadline);
        let started_candidates = AtomicUsize::new(0);
        let population_len = population.len();
        let mut screened: Vec<(FitnessReport, EvalWeights)> = population
            .par_iter()
            .copied()
            .filter_map(|weights| {
                if training_expired(deadline) {
                    return None;
                }
                let index = started_candidates.fetch_add(1, Ordering::Relaxed);
                let remaining = remaining_seconds(deadline);
                eprintln!(
                    "generation {generation}: screening candidate {}/{} depth={} nodes={} plies={} remaining={}s",
                    index + 1,
                    population_len,
                    screening_config.depth,
                    screening_config.nodes,
                    screening_config.plies,
                    remaining
                );
                let report = fitness_until_with_opponent_limit(weights, &screening_config, deadline, 2);
                Some((report, weights))
            })
            .collect();
        if screened.is_empty() {
            break;
        }
        screened.sort_by_key(|entry| std::cmp::Reverse(entry.0.score));
        let finalist_count = config.finalist_count.min(screened.len()).max(2);
        let finalists: Vec<EvalWeights> = screened
            .iter()
            .take(finalist_count)
            .map(|entry| entry.1)
            .collect();
        let finalist_started = AtomicUsize::new(0);
        let mut scored: Vec<(FitnessReport, EvalWeights)> = finalists
            .par_iter()
            .copied()
            .filter_map(|weights| {
                if training_expired(deadline) {
                    return None;
                }
                let index = finalist_started.fetch_add(1, Ordering::Relaxed);
                let remaining = remaining_seconds(deadline);
                eprintln!(
                    "generation {generation}: scoring finalist {}/{} depth={search_depth} nodes={search_nodes} plies={search_plies} remaining={}s",
                    index + 1,
                    finalist_count,
                    remaining
                );
                let report = fitness_until(weights, &scoring_config, deadline);
                Some((report, weights))
            })
            .collect();
        let mut generation_match_stats = MatchStats::default();
        for (report, weights) in &scored {
            generation_match_stats.add(report.matches);
            if *weights != committed && report.score > best_seen_score {
                best_seen = *weights;
                best_seen_score = report.score;
            }
        }
        if scored.is_empty() {
            break;
        }
        scored.sort_by_key(|entry| std::cmp::Reverse(entry.0.score));
        let mut quality = ComparisonStats::default();
        for (report, _) in &scored {
            quality.record(report.score - baseline_report.score);
        }
        let best = scored[0].0.score;
        let worst = scored.last().map_or(best, |entry| entry.0.score);
        let average = scored.iter().map(|entry| entry.0.score as i64).sum::<i64>() as f64
            / scored.len() as f64;
        let improvement = previous_best.map_or(0, |previous| best - previous);
        previous_best = Some(best);
        let remaining = remaining_seconds(deadline);
        eprintln!(
            "generation {generation}: screen_depth={} screen_nodes={} screen_plies={} depth={search_depth} nodes={search_nodes} plies={search_plies} best={best} avg={average:.1} worst={worst} improvement={improvement:+} screened={} finalists={} matches=({}) better/equal/worse={}/{}/{} baseline={} mutation_scale={mutation_scale:.2} gen_elapsed={:.2}s remaining={}s",
            screening_config.depth,
            screening_config.nodes,
            screening_config.plies,
            screened.len(),
            scored.len(),
            generation_match_stats.summary(),
            quality.wins,
            quality.draws,
            quality.losses,
            baseline_report.summary(),
            started.elapsed().as_secs_f32(),
            remaining
        );

        if quality.wins == 0 {
            mutation_scale = (mutation_scale * 0.75).max(0.25);
            generations_without_candidate += 1;
        } else if quality.wins * 2 > scored.len() {
            mutation_scale = (mutation_scale * 1.15).min(2.0);
            generations_without_candidate = 0;
        } else {
            generations_without_candidate = 0;
        }

        let league_winner = select_league_winner(
            &scored,
            committed,
            &hall_of_fame,
            &scoring_config,
            deadline,
        )
        .unwrap_or_else(|| {
            scored
                .iter()
                .find(|entry| entry.1 != committed)
                .map(|entry| entry.1)
                .unwrap_or_else(|| committed.mutate_with_scale(&mut rng, mutation_scale))
        });
        if best_seen_score <= best {
            best_seen = league_winner;
        }

        let elite = 4.min(scored.len());
        let mut next: Vec<EvalWeights> = scored.iter().take(elite).map(|entry| entry.1).collect();
        while next.len() < config.population {
            let left = tournament(&scored, &mut rng);
            let right = tournament(&scored, &mut rng);
            next.push(
                EvalWeights::crossover(left, right, &mut rng)
                    .mutate_with_scale(&mut rng, mutation_scale),
            );
        }
        population = next;
        if generations_without_candidate >= config.max_generations_without_candidate {
            eprintln!(
                "training stopped: {} generations without a candidate beating baseline",
                generations_without_candidate
            );
            break;
        }
    }

    if best_seen == committed {
        committed.mutate_with_scale(&mut rng, mutation_scale.min(0.5))
    } else {
        best_seen
    }
}

fn fitness(weights: EvalWeights, config: &TrainerConfig) -> FitnessReport {
    fitness_until(weights, config, training_deadline(config))
}

fn fitness_until(
    weights: EvalWeights,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> FitnessReport {
    fitness_until_with_opponent_limit(weights, config, deadline, 4)
}

fn fitness_until_with_opponent_limit(
    weights: EvalWeights,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
    opponent_limit: usize,
) -> FitnessReport {
    // Score candidates from paired seeded starts against the committed default,
    // nearby mutations, and recent promoted weights. Match results are primary;
    // heuristic margins only break ties and guide selection inside noisy draws.
    let mut rng = Lcg::new(config.seed);
    let mut report = FitnessReport::default();
    let default = EvalWeights::default_tuned();
    let mut opponents = vec![default];
    opponents.extend(load_hall_of_fame(&config.hall_of_fame));

    while opponents.len() < 4 {
        opponents.push(default.mutate(&mut rng));
    }

    let work: Vec<(EvalWeights, u64)> = opponents
        .into_iter()
        .take(opponent_limit.max(1))
        .enumerate()
        .map(|(index, opponent)| (opponent, rng.next_u64() ^ ((index as u64) << 32)))
        .collect();
    let pairs: Vec<PairReport> = work
        .par_iter()
        .map(|(opponent, seed)| paired_report(weights, *opponent, *seed, config, deadline))
        .collect();

    for pair in pairs {
        report.matches.add(pair.candidate.matches);
        report.score += pair.delta();
        report.blunders += pair.candidate.blunders;
    }

    report
}

fn play_match_until(
    start: Game,
    weights: EvalWeights,
    opponent: EvalWeights,
    color: Color,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> MatchReport {
    // Matches are short by design; the heuristic should learn opening material,
    // tempo, and branch quality before deeper minimax is affordable.
    let mut game = start;
    let mut score = 0;
    let mut stable_advantage = 0;
    for ply in 0..config.plies {
        if training_expired(deadline) {
            break;
        }
        let side_weights = if game.turn == color {
            weights
        } else {
            opponent
        };
        let mut context = SearchContext::new(side_weights, game.turn, config.nodes, deadline);
        let Some((plan, _)) = game.search_root(config.depth, &mut context, None) else {
            let result = if game.turn == color {
                MatchResult::Loss
            } else {
                MatchResult::Win
            };
            return MatchReport {
                score: score
                    + if result == MatchResult::Win {
                        20_000
                    } else {
                        -20_000
                    },
                result,
                blunder: result == MatchResult::Loss,
            };
        };
        game = plan.game;
        let eval = game.evaluate(color, &weights);
        score += eval / 20 + eval.signum() * (config.plies - ply) as i32;
        if game.terminal_score(color) == Some(CHECKMATE_SCORE) {
            return MatchReport {
                score: score + CHECKMATE_SCORE / 10,
                result: MatchResult::Win,
                blunder: false,
            };
        }
        if game.terminal_score(color) == Some(-CHECKMATE_SCORE) {
            return MatchReport {
                score: score - CHECKMATE_SCORE / 10,
                result: MatchResult::Loss,
                blunder: true,
            };
        }
        stable_advantage = if eval.abs() > 4_000 {
            stable_advantage + eval.signum()
        } else {
            0
        };
        if stable_advantage >= 2 {
            return MatchReport {
                score: score + 10_000,
                result: MatchResult::Win,
                blunder: false,
            };
        }
        if stable_advantage <= -2 {
            return MatchReport {
                score: score - 10_000,
                result: MatchResult::Loss,
                blunder: true,
            };
        }
    }
    let final_score = score + game.evaluate(color, &weights) / 4;
    let result = if final_score > 300 {
        MatchResult::Win
    } else if final_score < -300 {
        MatchResult::Loss
    } else {
        MatchResult::Draw
    };
    MatchReport {
        score: final_score,
        result,
        blunder: false,
    }
}

fn paired_baseline_report(
    candidate: EvalWeights,
    seed: u64,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> PairReport {
    paired_report(
        candidate,
        EvalWeights::default_tuned(),
        seed,
        config,
        deadline,
    )
}

fn paired_report(
    candidate: EvalWeights,
    baseline: EvalWeights,
    seed: u64,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> PairReport {
    let start = seeded_start_position(seed, config, deadline);
    let black_start = start.clone();
    let (candidate_white, candidate_black) = rayon::join(
        || {
            play_match_until(
                start,
                candidate,
                baseline,
                Color::White,
                config,
                deadline,
            )
        },
        || {
            play_match_until(
                black_start,
                candidate,
                baseline,
                Color::Black,
                config,
                deadline,
            )
        },
    );
    let baseline_black = invert_report(candidate_white);
    let baseline_white = invert_report(candidate_black);

    let mut report = PairReport::default();
    report.candidate.add_match(candidate_white);
    report.candidate.add_match(candidate_black);
    report.baseline.add_match(baseline_white);
    report.baseline.add_match(baseline_black);
    report
}

fn invert_report(report: MatchReport) -> MatchReport {
    let result = match report.result {
        MatchResult::Win => MatchResult::Loss,
        MatchResult::Loss => MatchResult::Win,
        MatchResult::Draw => MatchResult::Draw,
    };
    MatchReport {
        score: -report.score,
        result,
        blunder: false,
    }
}

fn select_league_winner(
    scored: &[(FitnessReport, EvalWeights)],
    committed: EvalWeights,
    hall_of_fame: &[EvalWeights],
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> Option<EvalWeights> {
    let contenders: Vec<EvalWeights> = scored
        .iter()
        .filter(|entry| entry.1 != committed)
        .take(3)
        .map(|entry| entry.1)
        .collect();
    if contenders.is_empty() {
        return None;
    }
    let mut opponents = vec![EvalWeights::default_tuned()];
    opponents.extend(hall_of_fame.iter().copied().take(2));
    opponents.extend(contenders.iter().copied());

    let mut results: Vec<(usize, EvalWeights, ComparisonStats)> = contenders
        .par_iter()
        .copied()
        .enumerate()
        .map(|(index, contender)| {
            let mut stats = ComparisonStats::default();
            let reports: Vec<PairReport> = opponents
                .par_iter()
                .copied()
                .enumerate()
                .filter_map(|(opponent_index, opponent)| {
                    if training_expired(deadline) {
                        return None;
                    }
                    let seed = config.seed
                        ^ ((index as u64) << 32)
                        ^ ((opponent_index as u64) << 48)
                        ^ 0xa5a5_5a5a_d3c3_b4b4;
                    Some(paired_report(contender, opponent, seed, config, deadline))
                })
                .collect();
            for report in reports {
                stats.record(report.delta());
            }
            (index, contender, stats)
        })
        .collect();
    results.sort_by_key(|entry| entry.0);
    for (index, _, stats) in &results {
        eprintln!(
            "league candidate {}: pairs={} wins={} losses={} draws={} win_rate={:.1}% elo={:+.0}",
            index + 1,
            stats.played,
            stats.wins,
            stats.losses,
            stats.draws,
            stats.win_rate() * 100.0,
            stats.estimated_elo()
        );
    }
    results
        .into_iter()
        .max_by(|left, right| {
            left.2
                .points
                .partial_cmp(&right.2.points)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.2.total_delta.cmp(&right.2.total_delta))
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|entry| entry.1)
}

fn tournament(scored: &[(FitnessReport, EvalWeights)], rng: &mut Lcg) -> EvalWeights {
    // Tournament selection applies pressure toward stronger candidates while
    // preserving enough randomness for weaker genomes to contribute genes.
    let mut best = scored[rng.next_usize(scored.len())];
    for _ in 1..4 {
        let candidate = scored[rng.next_usize(scored.len())];
        if candidate.0.score > best.0.score {
            best = candidate;
        }
    }
    best.1
}

fn mutate_weight(value: i32, rng: &mut Lcg, spread: i32, min: i32, max: i32) -> i32 {
    let delta = rng.next_usize((spread * 2 + 1) as usize) as i32 - spread;
    (value + delta).clamp(min, max)
}

fn training_deadline(config: &TrainerConfig) -> Option<SearchInstant> {
    config
        .max_seconds
        .map(|seconds| SearchInstant::now() + std::time::Duration::from_secs(seconds.max(1)))
}

fn training_expired(deadline: Option<SearchInstant>) -> bool {
    deadline.is_some_and(|deadline| SearchInstant::now() >= deadline)
}

fn remaining_seconds(deadline: Option<SearchInstant>) -> String {
    deadline.map_or_else(
        || "unbounded".to_string(),
        |deadline| {
            deadline
                .saturating_duration_since(SearchInstant::now())
                .as_secs()
                .to_string()
        },
    )
}
