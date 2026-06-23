use super::*;

#[derive(Clone)]
pub(crate) struct BaselineFitnessCache {
    pub(crate) key: BaselineFitnessKey,
    pub(crate) report: FitnessReport,
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
