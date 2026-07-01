use super::*;
use crate::cpu::EvalWeights;

pub(crate) fn unique_weights(weights: &[EvalWeights]) -> Vec<EvalWeights> {
    let mut unique = Vec::with_capacity(weights.len());
    for weights in weights.iter().copied() {
        push_unique_weight(&mut unique, weights);
    }
    unique
}

pub(crate) fn push_unique_weight(weights: &mut Vec<EvalWeights>, candidate: EvalWeights) -> bool {
    if weights.contains(&candidate) {
        return false;
    }
    weights.push(candidate);
    true
}

pub(crate) fn refill_unique_population(
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
