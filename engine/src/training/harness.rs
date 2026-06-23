use std::sync::atomic::{AtomicUsize, Ordering};

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
    pretty_log::section("Evolution");
    let candidate = train_weights_until(config, deadline);
    compare_and_maybe_promote(candidate, config, deadline);
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
    refill_unique_population(&mut population, config.population, &mut rng, || committed);

    let mut previous_best: Option<i32> = None;
    let training_started = SearchInstant::now();
    let mut best_seen = EvalWeights::default_tuned();
    let mut best_seen_score = i32::MIN;
    let mut mutation_scale = 1.0_f32;
    let hall_of_fame = load_hall_of_fame(&config.hall_of_fame, config.hall_of_fame_entries);
    let mut generations_without_candidate = 0;
    let mut baseline_cache: Option<BaselineFitnessCache> = None;
    let mut fitness_cache: Vec<FitnessCacheEntry> = Vec::new();
    for generation in 0..config.generations {
        if training_expired(deadline) {
            break;
        }
        // Reports are deterministic for a scoring key, so surviving elites can
        // reuse prior work while new mutations remain directly comparable.
        let elapsed = training_started.elapsed().as_secs();
        let node_boost = (elapsed / 600).min(2) as usize + 1;
        let search_nodes = config.nodes * node_boost;
        let scoring_config = config.with_search(search_nodes, config.training_time_ms);
        let screening_config = scoring_config.screening_search();
        let finished_candidates = AtomicUsize::new(0);
        let unique_population = unique_weights(&population);
        let screening_key = baseline_fitness_key(&screening_config);
        let screening_opponent_limit = screening_config.screening_opponent_variants;
        let mut screened: Vec<(FitnessReport, EvalWeights)> = unique_population
            .iter()
            .copied()
            .filter_map(|weights| {
                cached_fitness_report(
                    &fitness_cache,
                    &screening_key,
                    screening_opponent_limit,
                    weights,
                )
                .map(|report| (report, weights))
            })
            .collect();
        let screening_cache_hits = screened.len();
        let screening_jobs: Vec<EvalWeights> = unique_population
            .iter()
            .copied()
            .filter(|weights| {
                cached_fitness_report(
                    &fitness_cache,
                    &screening_key,
                    screening_opponent_limit,
                    *weights,
                )
                .is_none()
            })
            .collect();
        let screening_job_count = screening_jobs.len();
        let screened_misses: Vec<(FitnessReport, EvalWeights)> = screening_jobs
            .par_iter()
            .copied()
            .enumerate()
            .filter_map(|(index, weights)| {
                if training_expired(deadline) {
                    return None;
                }
                let report = fitness_until_with_opponent_limit(
                    weights,
                    &format!("generation {generation} screening candidate {}", index + 1),
                    &screening_config,
                    deadline,
                    screening_opponent_limit,
                );
                let done = finished_candidates.fetch_add(1, Ordering::Relaxed) + 1;
                training_progress(
                    &format!("generation {generation} screening"),
                    done,
                    screening_job_count,
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
        if !training_expired(deadline) {
            for (report, weights) in &screened_misses {
                cache_fitness_report(
                    &mut fitness_cache,
                    screening_key.clone(),
                    screening_opponent_limit,
                    *weights,
                    *report,
                );
            }
        }
        screened.extend(screened_misses);
        if screened.is_empty() {
            break;
        }
        screened.sort_by_key(|entry| std::cmp::Reverse(entry.0.score));
        let finalist_count = config.finalist_count.max(2).min(screened.len());
        let finalists: Vec<EvalWeights> = screened
            .iter()
            .take(finalist_count)
            .map(|entry| entry.1)
            .collect();
        let baseline_key = baseline_fitness_key(&scoring_config);
        let cached_baseline = baseline_cache
            .as_ref()
            .filter(|cached| cached.key == baseline_key)
            .map(|cached| cached.report);
        let scoring_jobs: Vec<EvalWeights> =
            finalist_scoring_jobs(&finalists, committed, cached_baseline.is_some())
                .into_iter()
                .filter(|weights| {
                    *weights == committed
                        || cached_fitness_report(
                            &fitness_cache,
                            &baseline_key,
                            scoring_config.opponent_variants,
                            *weights,
                        )
                        .is_none()
                })
                .collect();
        let finalist_cache_hits = finalists
            .iter()
            .filter(|weights| {
                **weights == committed && cached_baseline.is_some()
                    || **weights != committed
                        && cached_fitness_report(
                            &fitness_cache,
                            &baseline_key,
                            scoring_config.opponent_variants,
                            **weights,
                        )
                        .is_some()
            })
            .count();
        let finalist_finished = AtomicUsize::new(0);
        let scoring_job_count = scoring_jobs.len();
        let scored_jobs: Vec<(FitnessReport, EvalWeights)> = scoring_jobs
            .par_iter()
            .copied()
            .enumerate()
            .filter_map(|(index, weights)| {
                if training_expired(deadline) {
                    return None;
                }
                let label = if weights == committed {
                    format!("generation {generation} committed baseline")
                } else {
                    format!("generation {generation} finalist {}", index + 1)
                };
                let report = fitness_until_named(weights, &label, &scoring_config, deadline);
                let done = finalist_finished.fetch_add(1, Ordering::Relaxed) + 1;
                training_progress(
                    &format!("generation {generation} finalists"),
                    done,
                    scoring_job_count,
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
        if !training_expired(deadline) {
            for (report, weights) in &scored_jobs {
                cache_fitness_report(
                    &mut fitness_cache,
                    baseline_key.clone(),
                    scoring_config.opponent_variants,
                    *weights,
                    *report,
                );
            }
        }
        let Some(baseline_report) = cached_baseline.or_else(|| {
            scored_jobs
                .iter()
                .find(|entry| entry.1 == committed)
                .map(|entry| entry.0)
        }) else {
            break;
        };
        if cached_baseline.is_none() && !training_expired(deadline) {
            baseline_cache = Some(BaselineFitnessCache {
                key: baseline_key.clone(),
                report: baseline_report,
            });
        }
        let mut scored: Vec<(FitnessReport, EvalWeights)> = finalists
            .iter()
            .filter_map(|weights| {
                if *weights == committed {
                    return Some((baseline_report, *weights));
                }
                cached_fitness_report(
                    &fitness_cache,
                    &baseline_key,
                    scoring_config.opponent_variants,
                    *weights,
                )
                .or_else(|| {
                    scored_jobs
                        .iter()
                        .find(|entry| entry.1 == *weights)
                        .map(|entry| entry.0)
                })
                .map(|report| (report, *weights))
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
        let improvement = previous_best.map_or(0, |previous| best - previous);
        previous_best = Some(best);
        let _ = (
            improvement,
            screening_cache_hits,
            finalist_cache_hits,
            generation_match_stats,
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
        let target_population = config.population;
        let mut attempts = 0usize;
        let max_attempts = target_population.saturating_mul(64).max(64);
        while next.len() < target_population && attempts < max_attempts {
            attempts += 1;
            let left = tournament(&scored, &mut rng);
            let right = tournament(&scored, &mut rng);
            let candidate = EvalWeights::crossover(left, right, &mut rng)
                .mutate_with_scale(&mut rng, mutation_scale);
            push_unique_weight(&mut next, candidate);
        }
        if next.len() < target_population {
            pretty_log::warn(format!(
                "generation {generation}: produced {} distinct candidates after {attempts} attempts (target {target_population})",
                next.len()
            ));
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

#[derive(Clone)]
pub(crate) struct BaselineFitnessCache {
    key: BaselineFitnessKey,
    report: FitnessReport,
}

#[derive(Clone)]
pub(crate) struct FitnessCacheEntry {
    key: BaselineFitnessKey,
    opponent_limit: usize,
    weights: EvalWeights,
    report: FitnessReport,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BaselineFitnessKey {
    seed: u64,
    nodes: usize,
    training_time_ms: u64,
    opponent_variants: usize,
    rounds_per_variant: usize,
    hall_of_fame_entries: usize,
    hall_of_fame: String,
    max_match_plies: i32,
    max_match_time_ms: u64,
    search_strategy: TrainingSearchStrategy,
}

pub(crate) fn baseline_fitness_key(config: &TrainerConfig) -> BaselineFitnessKey {
    BaselineFitnessKey {
        seed: config.seed,
        nodes: config.nodes,
        training_time_ms: config.training_time_ms,
        opponent_variants: config.opponent_variants,
        rounds_per_variant: config.rounds_per_variant,
        hall_of_fame_entries: config.hall_of_fame_entries,
        hall_of_fame: config.hall_of_fame.clone(),
        max_match_plies: config.max_match_plies,
        max_match_time_ms: config.max_match_time_ms,
        search_strategy: config.search_strategy,
    }
}

pub(crate) fn finalist_scoring_jobs(
    finalists: &[EvalWeights],
    committed: EvalWeights,
    baseline_cached: bool,
) -> Vec<EvalWeights> {
    let mut jobs = Vec::new();
    for weights in finalists.iter().copied() {
        if weights != committed && !jobs.contains(&weights) {
            jobs.push(weights);
        }
    }
    if !baseline_cached {
        jobs.push(committed);
    }
    jobs
}

pub(crate) fn cached_fitness_report(
    cache: &[FitnessCacheEntry],
    key: &BaselineFitnessKey,
    opponent_limit: usize,
    weights: EvalWeights,
) -> Option<FitnessReport> {
    cache
        .iter()
        .find(|entry| {
            entry.key == *key && entry.opponent_limit == opponent_limit && entry.weights == weights
        })
        .map(|entry| entry.report)
}

pub(crate) fn cache_fitness_report(
    cache: &mut Vec<FitnessCacheEntry>,
    key: BaselineFitnessKey,
    opponent_limit: usize,
    weights: EvalWeights,
    report: FitnessReport,
) {
    if cached_fitness_report(cache, &key, opponent_limit, weights).is_none() {
        cache.push(FitnessCacheEntry {
            key,
            opponent_limit,
            weights,
            report,
        });
    }
}

pub(crate) fn unique_weights(weights: &[EvalWeights]) -> Vec<EvalWeights> {
    let mut unique = Vec::with_capacity(weights.len());
    for weights in weights.iter().copied() {
        push_unique_weight(&mut unique, weights);
    }
    unique
}

fn push_unique_weight(weights: &mut Vec<EvalWeights>, candidate: EvalWeights) -> bool {
    if weights.contains(&candidate) {
        return false;
    }
    weights.push(candidate);
    true
}

fn refill_unique_population(
    population: &mut Vec<EvalWeights>,
    target: usize,
    rng: &mut Lcg,
    parent: impl Fn() -> EvalWeights,
) {
    let mut attempts = 0usize;
    let max_attempts = target.saturating_mul(64).max(64);
    while population.len() < target && attempts < max_attempts {
        attempts += 1;
        push_unique_weight(population, parent().mutate(rng));
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

pub(crate) fn mutate_weight_if(
    value: i32,
    rng: &mut Lcg,
    mutation_divisor: usize,
    spread: i32,
    min: i32,
    max: i32,
) -> i32 {
    if rng.next_usize(mutation_divisor) != 0 {
        return value;
    }
    mutate_weight(value, rng, spread, min, max)
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
