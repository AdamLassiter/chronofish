use rayon::prelude::*;

use super::*;
use crate::cpu::{EvalWeights, SearchInstant};

pub(crate) fn select_league_winner(
    scored: &[(FitnessReport, EvalWeights)],
    committed: EvalWeights,
    hall_of_fame: &[EvalWeights],
    config: &CpuCliConfig,
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
