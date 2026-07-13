use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use super::*;
use crate::cpu::{EvalWeights, SearchInstant};

pub(crate) fn run_training_cycle(config: &CpuCliConfig) {
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

pub(crate) fn train_weights(config: &CpuCliConfig) -> EvalWeights {
    train_weights_until(config, training_deadline(config))
}

pub(crate) fn train_weights_until(
    config: &CpuCliConfig,
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
