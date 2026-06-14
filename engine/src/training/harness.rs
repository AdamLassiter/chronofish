use std::sync::atomic::{AtomicUsize, Ordering};

use pretty_log::transient;
use rayon::prelude::*;

use super::*;

pub(crate) fn run_training_cycle(config: &TrainerConfig) {
    // Promotion rewrites the parameter include file, so refuse to continue when
    // it already has local edits that should not be mixed with generated tuning.
    if ai_source_is_dirty(&config.ai_src) {
        pretty_log::fail(format!(
            "{} has uncommitted changes; commit or stash before running training",
            config.ai_src
        ));
        std::process::exit(1);
    }

    training_banner(config);
    pretty_log::phase(
        "Fitness uses full-match evaluation margins; comparisons use candidate minus baseline per seed",
    );

    let deadline = training_deadline(config);
    let hall_of_fame = load_hall_of_fame(&config.hall_of_fame, config.hall_of_fame_entries);
    pretty_log::section("Evolution");
    let candidate = train_weights_until(config, deadline);
    if candidate == EvalWeights::default_tuned() {
        pretty_log::warn("candidate inconclusive: selected weights match committed baseline");
        return;
    }
    let candidate_json = candidate.to_json();
    let mut comparison_stats = ComparisonStats::default();
    let mut deltas = Vec::new();
    let mut comparison_match_stats = MatchStats::default();

    pretty_log::section("Candidate");
    pretty_log::label_value("weights", candidate_json);
    pretty_log::section("Comparison");
    let mut rng = Lcg::new(config.seed ^ 0xadc8_3b19_7f4a_7c15);
    loop {
        if training_expired(deadline) {
            pretty_log::warn("comparison stopped: max seconds exhausted");
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
        let completed = AtomicUsize::new(0);
        let total = seeds.len();
        let reports: Vec<(u64, PairReport)> = seeds
            .par_iter()
            .copied()
            .map(|seed| {
                let report = paired_baseline_report(candidate, seed, config, deadline);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                training_progress(
                    "comparison",
                    done,
                    total,
                    format!("remaining={}", remaining_seconds(deadline)),
                );
                (seed, report)
            })
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
            training_note(format!(
                "seed {seed}: {result} candidate={} baseline={} delta={delta}",
                report.candidate.summary(),
                report.baseline.summary()
            ));
        }
        let significance = significance(&deltas);
        match statistical_decision(comparison_stats, &deltas, significance, config) {
            StatisticalDecision::Promote => {
                pretty_log::success("sequential comparison accepted");
                break;
            }
            StatisticalDecision::Reject => {
                pretty_log::warn("sequential comparison rejected");
                break;
            }
            StatisticalDecision::Inconclusive => {
                pretty_log::warn("sequential comparison inconclusive");
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
        pretty_log::phase("Promoting candidate");
        promote_weights(candidate, &config.ai_src);
        append_hall_of_fame(&config.hall_of_fame, candidate);
        run_command("cargo", &["fmt"]);
        run_shell(&config.verify);
        run_command("git", &["add", &config.ai_src]);
        if !hall_of_fame.is_empty() || std::path::Path::new(&config.hall_of_fame).is_file() {
            run_command("git", &["add", &config.hall_of_fame]);
        }
        run_command("git", &["commit", "-m", "Tune AI evaluation parameters"]);
        pretty_log::success("promoted candidate and committed updated parameters");
    } else {
        match statistical_decision(comparison_stats, &deltas, significance, config) {
            StatisticalDecision::Reject => pretty_log::warn("candidate rejected"),
            StatisticalDecision::Inconclusive | StatisticalDecision::Continue => {
                pretty_log::warn("candidate inconclusive")
            }
            StatisticalDecision::Promote => pretty_log::warn("candidate rejected"),
        }
    }
}

pub(crate) fn train_weights(config: &TrainerConfig) -> EvalWeights {
    train_weights_until(config, training_deadline(config))
}

pub(crate) fn train_weights_until(
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> EvalWeights {
    pretty_log::phase(
        "Scoring candidates across full matches against default and mutated opponents",
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
    let hall_of_fame = load_hall_of_fame(&config.hall_of_fame, config.hall_of_fame_entries);
    let mut generations_without_candidate = 0;
    for generation in 0..config.generations {
        if training_expired(deadline) {
            break;
        }
        // Each generation is fully rescored against this config seed so best,
        // average, and improvement logs are comparable within the run.
        let started = SearchInstant::now();
        let elapsed = training_started.elapsed().as_secs();
        let node_boost = (elapsed / 600).min(2) as usize + 1;
        let search_nodes = config.nodes * node_boost;
        let scoring_config = config.with_search(search_nodes, config.training_time_ms);
        let screening_config = scoring_config.screening_search();
        let baseline_report = fitness_until_named(
            EvalWeights::default_tuned(),
            &format!("generation {generation} committed baseline"),
            &scoring_config,
            deadline,
        );
        let finished_candidates = AtomicUsize::new(0);
        let population_len = population.len();
        let mut screened: Vec<(FitnessReport, EvalWeights)> = population
            .par_iter()
            .copied()
            .enumerate()
            .filter_map(|(index, weights)| {
                if training_expired(deadline) {
                    return None;
                }
                transient(format!(
                    "generation {generation} screening candidate {} start",
                    index + 1
                ));
                let report = fitness_until_with_opponent_limit(
                    weights,
                    &format!("generation {generation} screening candidate {}", index + 1),
                    &screening_config,
                    deadline,
                    screening_config.screening_opponent_variants,
                );
                let done = finished_candidates.fetch_add(1, Ordering::Relaxed) + 1;
                training_progress(
                    &format!("generation {generation} screening"),
                    done,
                    population_len,
                    format!(
                        "turn_ms={} nodes={} remaining={}",
                        screening_config.training_time_ms,
                        screening_config.nodes,
                        remaining_seconds(deadline)
                    ),
                );
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
        let finalist_finished = AtomicUsize::new(0);
        let mut scored: Vec<(FitnessReport, EvalWeights)> = finalists
            .par_iter()
            .copied()
            .enumerate()
            .filter_map(|(index, weights)| {
                if training_expired(deadline) {
                    return None;
                }
                transient(format!(
                    "generation {generation} finalist {} start",
                    index + 1
                ));
                let report = fitness_until_named(
                    weights,
                    &format!("generation {generation} finalist {}", index + 1),
                    &scoring_config,
                    deadline,
                );
                let done = finalist_finished.fetch_add(1, Ordering::Relaxed) + 1;
                training_progress(
                    &format!("generation {generation} finalists"),
                    done,
                    finalist_count,
                    format!(
                        "turn_ms={} nodes={} remaining={}",
                        scoring_config.training_time_ms,
                        search_nodes,
                        remaining_seconds(deadline)
                    ),
                );
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
        training_note(format!(
            "generation {generation}: screen_turn_ms={} screen_nodes={} turn_ms={} nodes={search_nodes} best={best} avg={average:.1} worst={worst} improvement={improvement:+} screened={} finalists={} matches=({}) better/equal/worse={}/{}/{} baseline={} mutation_scale={mutation_scale:.2} gen_elapsed={:.2}s remaining={}",
            screening_config.training_time_ms,
            screening_config.nodes,
            scoring_config.training_time_ms,
            screened.len(),
            scored.len(),
            generation_match_stats.summary(),
            quality.wins,
            quality.draws,
            quality.losses,
            baseline_report.summary(),
            started.elapsed().as_secs_f32(),
            remaining
        ));

        if quality.wins == 0 {
            mutation_scale = (mutation_scale * 0.75).max(0.25);
            generations_without_candidate += 1;
        } else if quality.wins * 2 > scored.len() {
            mutation_scale = (mutation_scale * 1.15).min(2.0);
            generations_without_candidate = 0;
        } else {
            generations_without_candidate = 0;
        }

        let league_winner =
            select_league_winner(&scored, committed, &hall_of_fame, &scoring_config, deadline)
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
            pretty_log::warn(format!(
                "training stopped: {} generations without a candidate beating baseline",
                generations_without_candidate
            ));
            break;
        }
    }

    if best_seen == committed {
        committed.mutate_with_scale(&mut rng, mutation_scale.min(0.5))
    } else {
        best_seen
    }
}

pub(crate) fn fitness(weights: EvalWeights, config: &TrainerConfig) -> FitnessReport {
    fitness_until_named(
        weights,
        "score candidate",
        config,
        training_deadline(config),
    )
}

pub(crate) fn fitness_until_named(
    weights: EvalWeights,
    candidate_label: &str,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> FitnessReport {
    fitness_until_with_opponent_limit(
        weights,
        candidate_label,
        config,
        deadline,
        config.opponent_variants,
    )
}

pub(crate) fn fitness_until_with_opponent_limit(
    weights: EvalWeights,
    candidate_label: &str,
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
    let mut opponents = vec![(default, "committed baseline".to_string())];
    opponents.extend(
        load_hall_of_fame(&config.hall_of_fame, config.hall_of_fame_entries)
            .into_iter()
            .filter(|weights| *weights != default)
            .enumerate()
            .map(|(index, weights)| (weights, format!("hall-of-fame {}", index + 1))),
    );

    while opponents.len() < opponent_limit.max(1) {
        let mutation_index = opponents
            .iter()
            .filter(|(_, label)| label.starts_with("mutated opponent"))
            .count()
            + 1;
        opponents.push((
            default.mutate(&mut rng),
            format!("mutated opponent {mutation_index}"),
        ));
    }

    let work: Vec<(EvalWeights, u64, String)> = opponents
        .into_iter()
        .take(opponent_limit.max(1))
        .enumerate()
        .flat_map(|(index, (opponent, label))| {
            (0..config.rounds_per_variant).map(move |round| {
                (
                    opponent,
                    config.seed
                        ^ ((index as u64) << 32)
                        ^ ((round as u64) << 48)
                        ^ 0x7f4a_7c15_9e37_79b9,
                    format!("{label} round {}", round + 1),
                )
            })
        })
        .collect();
    let pairs: Vec<PairReport> = work
        .par_iter()
        .map(|(opponent, seed, opponent_label)| {
            paired_report(
                weights,
                *opponent,
                *seed,
                candidate_label,
                opponent_label,
                config,
                deadline,
            )
        })
        .collect();

    for pair in pairs {
        report.matches.add(pair.candidate.matches);
        report.score += pair.delta();
        report.blunders += pair.candidate.blunders;
    }

    report
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn play_match_until(
    start: Game,
    weights: EvalWeights,
    opponent: EvalWeights,
    color: Color,
    candidate_label: &str,
    opponent_label: &str,
    match_label: &str,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> MatchReport {
    // Full-match scoring keeps the objective aligned with real game outcomes
    // rather than stopping at a fixed ply horizon.
    transient(format!(
        "{match_label} match-start candidate_color={}",
        color.as_str()
    ));
    let mut game = start;
    let mut score = 0;
    let mut stable_advantage = 0;
    let mut plies_played = 0;
    let match_started = SearchInstant::now();
    let mut total_search_ms = 0u128;
    let mut max_turn_ms = 0u128;
    let mut max_turn_ply = 0;
    let mut max_depth = 0;
    let mut slow_turns = 0;
    let mut fallback_turns = 0;
    let mut capped_turns = 0;
    let mut peak_obligations = 0;
    let mut peak_playable_boards = 0;
    let match_deadline =
        Some(match_started + std::time::Duration::from_millis(max_match_time_ms(config).max(1)));
    loop {
        let turn_deadline = earliest_deadline(deadline, match_deadline);
        if training_expired(turn_deadline) {
            break;
        }
        let mover = game.turn;
        let side_weights = if mover == color { weights } else { opponent };
        transient(format!(
            "{match_label} turn {} {} search-start elapsed={}ms cap={}ms",
            plies_played + 1,
            mover.as_str(),
            SearchInstant::now()
                .duration_since(match_started)
                .as_millis(),
            max_match_time_ms(config)
        ));
        let turn_started = SearchInstant::now();
        let Some(search) = training_turn_search(
            &game,
            side_weights,
            config,
            turn_deadline,
            plies_played as usize,
        ) else {
            let elapsed_ms = SearchInstant::now()
                .duration_since(turn_started)
                .as_millis();
            log_training_match_summary(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms.max(elapsed_ms),
                if elapsed_ms > max_turn_ms {
                    plies_played + 1
                } else {
                    max_turn_ply
                },
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                score,
                if training_expired(match_deadline) {
                    "match-time-cap"
                } else {
                    "turn-timeout"
                },
            );
            let result = if game.turn == color {
                MatchResult::Loss
            } else {
                MatchResult::Win
            };
            return MatchReport {
                score: score
                    + if result == MatchResult::Win {
                        20_000 - plies_played * 10
                    } else {
                        -20_000 + plies_played * 10
                    },
                result,
                blunder: result == MatchResult::Loss,
            };
        };
        let elapsed_ms = SearchInstant::now()
            .duration_since(turn_started)
            .as_millis();
        if elapsed_ms >= slow_training_turn_threshold_ms(config) && elapsed_ms > search.elapsed_ms {
            pretty_log::warn(format!(
                "Slow training turn wall-time · {match_label} · turn {} {} elapsed={}ms search={}ms",
                plies_played + 1,
                mover.as_str(),
                elapsed_ms,
                search.elapsed_ms
            ));
        }
        transient(format!("{match_label} turn {} notation", plies_played + 1));
        let notation = turn_plan_notation(&game, &search.plan);
        transient(format!("{match_label} turn {} apply", plies_played + 1));
        let Some(next_game) = game.apply_turn_plan_for_search(&search.plan) else {
            transient(format!(
                "{match_label} discarded inapplicable {} plan: {notation}",
                mover.as_str()
            ));
            let result = if game.turn == color {
                MatchResult::Loss
            } else {
                MatchResult::Win
            };
            transient(format!(
                "{match_label} match-finish result=inapplicable-plan turns={plies_played} score={score}"
            ));
            return MatchReport {
                score: score
                    + if result == MatchResult::Win {
                        20_000 - plies_played * 10
                    } else {
                        -20_000 + plies_played * 10
                    },
                result,
                blunder: result == MatchResult::Loss,
            };
        };
        game = next_game;
        plies_played += 1;
        total_search_ms += search.elapsed_ms;
        if search.elapsed_ms > max_turn_ms {
            max_turn_ms = search.elapsed_ms;
            max_turn_ply = plies_played;
        }
        max_depth = max_depth.max(search.depth);
        peak_obligations = peak_obligations.max(search.obligations);
        peak_playable_boards = peak_playable_boards.max(search.playable_boards);
        if search.fallback_used {
            fallback_turns += 1;
        }
        if search.capped {
            capped_turns += 1;
        }
        if search.elapsed_ms >= slow_training_turn_threshold_ms(config) {
            slow_turns += 1;
            log_slow_training_turn(match_label, plies_played, mover, &search);
        }
        let mover_label = if mover == color {
            candidate_label
        } else {
            opponent_label
        };
        transient(format!(
            "{match_label} turn {plies_played} {} {mover_label}: {notation} [{}ms d{} n{} p{}/{}]",
            mover.as_str(),
            search.elapsed_ms,
            search.depth,
            search.nodes,
            search.obligations,
            search.playable_boards
        ));
        transient(format!("{match_label} turn {plies_played} terminal-check"));
        if let Some(terminal) = game.terminal_score_until(color, turn_deadline) {
            if terminal == CHECKMATE_SCORE {
                log_training_match_summary(
                    match_label,
                    plies_played,
                    SearchInstant::now()
                        .duration_since(match_started)
                        .as_millis(),
                    total_search_ms,
                    max_turn_ms,
                    max_turn_ply,
                    max_depth,
                    slow_turns,
                    fallback_turns,
                    capped_turns,
                    peak_obligations,
                    peak_playable_boards,
                    score,
                    "win",
                );
                return MatchReport {
                    score: score + CHECKMATE_SCORE / 10 - plies_played * 10,
                    result: MatchResult::Win,
                    blunder: false,
                };
            }
            if terminal == -CHECKMATE_SCORE {
                log_training_match_summary(
                    match_label,
                    plies_played,
                    SearchInstant::now()
                        .duration_since(match_started)
                        .as_millis(),
                    total_search_ms,
                    max_turn_ms,
                    max_turn_ply,
                    max_depth,
                    slow_turns,
                    fallback_turns,
                    capped_turns,
                    peak_obligations,
                    peak_playable_boards,
                    score,
                    "loss",
                );
                return MatchReport {
                    score: score - CHECKMATE_SCORE / 10 + plies_played * 10,
                    result: MatchResult::Loss,
                    blunder: true,
                };
            }
            log_training_match_summary(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms,
                max_turn_ply,
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                score,
                "draw",
            );
            return MatchReport {
                score,
                result: MatchResult::Draw,
                blunder: false,
            };
        }
        transient(format!("{match_label} turn {plies_played} eval"));
        let eval =
            game.evaluate_heuristic_for_nodes_until(color, &weights, config.nodes, turn_deadline);
        score += eval / 20;
        if should_log_training_match_milestone(plies_played) {
            log_training_match_milestone(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms,
                max_turn_ply,
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                eval,
                score,
            );
        }
        stable_advantage = if eval.abs() > 4_000 {
            stable_advantage + eval.signum()
        } else {
            0
        };
        if stable_advantage >= 2 {
            log_training_match_summary(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms,
                max_turn_ply,
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                score,
                "stable-win",
            );
            return MatchReport {
                score: score + 10_000,
                result: MatchResult::Win,
                blunder: false,
            };
        }
        if stable_advantage <= -2 {
            log_training_match_summary(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms,
                max_turn_ply,
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                score,
                "stable-loss",
            );
            return MatchReport {
                score: score - 10_000,
                result: MatchResult::Loss,
                blunder: true,
            };
        }
        if plies_played >= config.max_match_plies {
            let final_score = score + eval / 4;
            let result = adjudicated_match_result(final_score);
            log_training_match_summary(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms,
                max_turn_ply,
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                final_score,
                "ply-cap",
            );
            return MatchReport {
                score: final_score,
                result,
                blunder: false,
            };
        }
    }
    let final_score = score
        + game.evaluate_heuristic_for_nodes_until(
            color,
            &weights,
            config.nodes,
            earliest_deadline(deadline, match_deadline),
        ) / 4;
    let result = adjudicated_match_result(final_score);
    let exit_deadline = earliest_deadline(deadline, match_deadline);
    log_training_match_summary(
        match_label,
        plies_played,
        SearchInstant::now()
            .duration_since(match_started)
            .as_millis(),
        total_search_ms,
        max_turn_ms,
        max_turn_ply,
        max_depth,
        slow_turns,
        fallback_turns,
        capped_turns,
        peak_obligations,
        peak_playable_boards,
        final_score,
        if training_expired(match_deadline) {
            "match-time-cap"
        } else if training_expired(exit_deadline) {
            "deadline"
        } else {
            "adjudicated"
        },
    );
    MatchReport {
        score: final_score,
        result,
        blunder: false,
    }
}

fn adjudicated_match_result(final_score: i32) -> MatchResult {
    if final_score > 300 {
        MatchResult::Win
    } else if final_score < -300 {
        MatchResult::Loss
    } else {
        MatchResult::Draw
    }
}

fn earliest_deadline(
    left: Option<SearchInstant>,
    right: Option<SearchInstant>,
) -> Option<SearchInstant> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
        (None, None) => None,
    }
}

fn slow_training_turn_threshold_ms(config: &TrainerConfig) -> u128 {
    10_000.min((config.training_time_ms.max(1) as u128).max(1))
}

fn should_log_training_match_milestone(plies_played: i32) -> bool {
    plies_played >= 20 && plies_played % 10 == 0
}

fn log_slow_training_turn(
    match_label: &str,
    plies_played: i32,
    mover: Color,
    search: &TrainingSearchOutcome,
) {
    pretty_log::warn(format!(
        "Slow training turn · {match_label} · turn {plies_played} {}",
        mover.as_str()
    ));
    pretty_log::label_value(
        "search",
        format!(
            "{}ms · depth {} · {} nodes · root/child {}/{}",
            search.elapsed_ms,
            search.depth,
            search.nodes,
            search.root_plan_limit,
            search.child_plan_limit
        ),
    );
    pretty_log::label_value(
        "position",
        format!(
            "{} present obligations · {} playable boards",
            search.obligations, search.playable_boards
        ),
    );
    pretty_log::label_value(
        "limits",
        format!(
            "{}{}",
            if search.capped { "capped" } else { "uncapped" },
            if search.fallback_used {
                " · legal-turn fallback"
            } else {
                ""
            }
        ),
    );
    pretty_log::label_value(
        "generated",
        format!(
            "{} moves · {} full plans · {} candidates · {} legality checks",
            search.stats.generated_moves,
            search.stats.generated_plans,
            search.stats.candidate_destinations,
            search.stats.legal_move_attempts
        ),
    );
    pretty_log::label_value(
        "cache",
        format!(
            "eval {}/{} · eval attacks {}/{} caps {} · search attacks {}/{} · tt {} · cutoffs {}",
            search.stats.evaluation_cache_hits,
            search.stats.evaluation_calls,
            search.stats.evaluation_attack_checks,
            search.stats.evaluation_attack_checks + search.stats.evaluation_attack_caps,
            search.stats.evaluation_attack_caps,
            search.stats.attack_cache_hits,
            search.stats.attack_queries,
            search.stats.tt_hits,
            search.stats.beta_cutoffs
        ),
    );
}

#[allow(clippy::too_many_arguments)]
fn log_training_match_milestone(
    match_label: &str,
    plies_played: i32,
    elapsed_ms: u128,
    total_search_ms: u128,
    max_turn_ms: u128,
    max_turn_ply: i32,
    max_depth: i32,
    slow_turns: i32,
    fallback_turns: i32,
    capped_turns: i32,
    peak_obligations: usize,
    peak_playable_boards: usize,
    eval: i32,
    score: i32,
) {
    training_note(format!(
        "long match · {match_label} · turn {plies_played} · elapsed={}ms search={}ms max={}ms@{} depth={} slow={} fallback={} capped={} peak={}/{} eval={} score={}",
        elapsed_ms,
        total_search_ms,
        max_turn_ms,
        max_turn_ply,
        max_depth,
        slow_turns,
        fallback_turns,
        capped_turns,
        peak_obligations,
        peak_playable_boards,
        eval,
        score,
    ));
}

#[allow(clippy::too_many_arguments)]
fn log_training_match_summary(
    match_label: &str,
    plies_played: i32,
    elapsed_ms: u128,
    total_search_ms: u128,
    max_turn_ms: u128,
    max_turn_ply: i32,
    max_depth: i32,
    slow_turns: i32,
    fallback_turns: i32,
    capped_turns: i32,
    peak_obligations: usize,
    peak_playable_boards: usize,
    score: i32,
    reason: &str,
) {
    transient(format!(
        "{match_label} match-finish result={reason} turns={plies_played} score={score}"
    ));
    if plies_played < 20
        && slow_turns == 0
        && fallback_turns == 0
        && !matches!(reason, "match-time-cap" | "turn-timeout" | "deadline")
    {
        return;
    }
    pretty_log::section("Training Match Summary");
    pretty_log::label_value("match", match_label);
    pretty_log::label_value("result", reason);
    pretty_log::label_value("turns", plies_played);
    pretty_log::label_value("elapsed", format!("{elapsed_ms}ms"));
    pretty_log::label_value("search", format!("{total_search_ms}ms total"));
    pretty_log::label_value("slow turns", slow_turns);
    pretty_log::label_value("fallback turns", fallback_turns);
    pretty_log::label_value("capped turns", capped_turns);
    pretty_log::label_value(
        "max turn",
        format!("{max_turn_ms}ms at turn {max_turn_ply}"),
    );
    pretty_log::label_value("max depth", max_depth);
    pretty_log::label_value(
        "peak pressure",
        format!("{peak_obligations} obligations · {peak_playable_boards} playable boards"),
    );
    pretty_log::label_value("score", score);
}

pub(crate) fn turn_plan_notation(start: &Game, plan: &TurnPlan) -> String {
    let mut notation_game = start.clone_for_search();
    for movement in &plan.moves {
        if notation_game.apply_move(movement.from, movement.to) == 0 {
            return plan
                .moves
                .iter()
                .map(|movement| {
                    format!(
                        "{}{} -> {}{}",
                        position_prefix(movement.from),
                        square_name(movement.from),
                        position_prefix(movement.to),
                        square_name(movement.to)
                    )
                })
                .collect::<Vec<_>>()
                .join("/");
        }
    }
    notation_game.staged_turn_notation()
}

pub(crate) fn paired_baseline_report(
    candidate: EvalWeights,
    seed: u64,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> PairReport {
    paired_report(
        candidate,
        EvalWeights::default_tuned(),
        seed,
        "comparison candidate",
        "committed baseline",
        config,
        deadline,
    )
}

pub(crate) fn paired_report(
    candidate: EvalWeights,
    baseline: EvalWeights,
    seed: u64,
    candidate_label: &str,
    opponent_label: &str,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> PairReport {
    transient(format!(
        "{candidate_label} vs {opponent_label} seed={seed} pair-start"
    ));
    let start = seeded_start_position(seed, config, deadline);
    let black_start = start.clone();
    let candidate_white_label =
        format!("{candidate_label} vs {opponent_label} seed={seed} candidate=white");
    let candidate_black_label =
        format!("{candidate_label} vs {opponent_label} seed={seed} candidate=black");
    let (candidate_white, candidate_black) = rayon::join(
        || {
            play_match_until(
                start,
                candidate,
                baseline,
                Color::White,
                candidate_label,
                opponent_label,
                &candidate_white_label,
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
                candidate_label,
                opponent_label,
                &candidate_black_label,
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
    transient(format!(
        "{candidate_label} vs {opponent_label} seed={seed} pair-finish"
    ));
    report
}

pub(crate) fn invert_report(report: MatchReport) -> MatchReport {
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

pub(crate) fn select_league_winner(
    scored: &[(FitnessReport, EvalWeights)],
    committed: EvalWeights,
    hall_of_fame: &[EvalWeights],
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> Option<EvalWeights> {
    let contenders: Vec<EvalWeights> = scored
        .iter()
        .filter(|entry| entry.1 != committed)
        .take(config.league_contenders)
        .map(|entry| entry.1)
        .collect();
    if contenders.is_empty() {
        return None;
    }
    let default = EvalWeights::default_tuned();
    let mut opponents = vec![default];
    opponents.extend(
        hall_of_fame
            .iter()
            .copied()
            .filter(|weights| *weights != default)
            .take(config.league_hall_of_fame_entries),
    );
    opponents.extend(contenders.iter().copied());
    let total_pairs = contenders.len().saturating_mul(opponents.len()).max(1);
    let league_progress = AtomicUsize::new(0);

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
                    let report = paired_report(
                        contender,
                        opponent,
                        seed,
                        &format!("league candidate {}", index + 1),
                        &format!("league opponent {}", opponent_index + 1),
                        config,
                        deadline,
                    );
                    let done = league_progress.fetch_add(1, Ordering::Relaxed) + 1;
                    training_progress(
                        "league",
                        done,
                        total_pairs,
                        format!("remaining={}", remaining_seconds(deadline)),
                    );
                    Some(report)
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
        training_note(format!(
            "league candidate {}: pairs={} wins={} losses={} draws={} win_rate={:.1}% elo={:+.0}",
            index + 1,
            stats.played,
            stats.wins,
            stats.losses,
            stats.draws,
            stats.win_rate() * 100.0,
            stats.estimated_elo()
        ));
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

pub(crate) fn tournament(scored: &[(FitnessReport, EvalWeights)], rng: &mut Lcg) -> EvalWeights {
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

pub(crate) fn mutate_weight(value: i32, rng: &mut Lcg, spread: i32, min: i32, max: i32) -> i32 {
    let delta = rng.next_usize((spread * 2 + 1) as usize) as i32 - spread;
    (value + delta).clamp(min, max)
}

pub(crate) fn training_deadline(config: &TrainerConfig) -> Option<SearchInstant> {
    config
        .max_seconds
        .map(|seconds| SearchInstant::now() + std::time::Duration::from_secs(seconds.max(1)))
}

pub(crate) fn training_expired(deadline: Option<SearchInstant>) -> bool {
    deadline.is_some_and(|deadline| SearchInstant::now() >= deadline)
}

pub(crate) fn remaining_seconds(deadline: Option<SearchInstant>) -> String {
    deadline.map_or_else(
        || "unbounded".to_string(),
        |deadline| {
            let seconds = deadline
                .saturating_duration_since(SearchInstant::now())
                .as_secs();
            format!("{seconds}s")
        },
    )
}
