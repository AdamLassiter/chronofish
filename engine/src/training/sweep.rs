use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use super::*;

pub(crate) fn run_sweep_training_cycle(config: &TrainerConfig) {
    if ai_source_is_dirty(&config.ai_src) {
        pretty_log::fail(format!(
            "{} has uncommitted changes; commit or stash before running training",
            config.ai_src
        ));
        std::process::exit(1);
    }

    training_banner(config);
    pretty_log::phase("Sweeping CPU evaluation parameters with paired candidate matches");

    let deadline = training_deadline(config);
    let candidate = train_weights_sweep_until(config, deadline);
    compare_and_maybe_promote(candidate, config, deadline);
}

pub(crate) fn train_weights_sweep(config: &TrainerConfig) -> EvalWeights {
    train_weights_sweep_until(config, training_deadline(config))
}

pub(crate) fn train_weights_sweep_until(
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> EvalWeights {
    let parameters = sweep_weight_parameters(&config.sweep_parameter_groups);
    if parameters.is_empty() {
        pretty_log::warn("sweep selected no trainable parameters");
        return EvalWeights::default_tuned();
    }

    let mut current = EvalWeights::default_tuned();
    let mut range_low = config.sweep_range_low;
    let mut range_high = config.sweep_range_high;
    let mut pass = 0usize;

    while !training_expired(deadline) && config.sweep_passes.is_none_or(|limit| pass < limit) {
        let pass_start = current;
        let mut changed = 0usize;
        pretty_log::section(format!("Sweep Pass {}", pass + 1));

        for (parameter_index, parameter) in parameters.iter().copied().enumerate() {
            if training_expired(deadline) {
                break;
            }
            let values = sweep_values(
                parameter,
                parameter.value(current),
                config.sweep_points,
                range_low,
                range_high,
            );
            if values.len() < 2 {
                continue;
            }

            let candidates: Vec<SweepCandidate> = values
                .into_iter()
                .map(|value| SweepCandidate {
                    value,
                    weights: parameter.with_value(current, value),
                })
                .collect();
            let winner = score_sweep_candidates(
                current,
                parameter,
                &candidates,
                pass,
                parameter_index,
                config,
                deadline,
            );

            if winner.value != parameter.value(current) {
                changed += 1;
            }
            current = winner.weights;
            training_task_progress(
                "sweep-pass",
                "sweep pass",
                parameter_index + 1,
                parameters.len(),
                format!(
                    "{}={} score={} remaining={}",
                    parameter.name,
                    winner.value,
                    winner.score,
                    remaining_seconds(deadline)
                ),
            );
        }

        pass += 1;
        training_task_progress(
            "sweep-pass",
            "sweep pass",
            parameters.len(),
            parameters.len(),
            format!(
                "pass {} changed={} remaining={}",
                pass,
                changed,
                remaining_seconds(deadline)
            ),
        );
        finish_training_task("sweep-pass");
        if current == pass_start {
            pretty_log::warn("sweep stopped: pass produced no parameter changes");
            break;
        }
        range_low = shrink_low(range_low, config.sweep_shrink);
        range_high = shrink_high(range_high, config.sweep_shrink);
    }

    current
}

#[derive(Clone, Copy)]
struct SweepCandidate {
    value: i32,
    weights: EvalWeights,
}

#[derive(Clone, Copy)]
pub(crate) struct SweepScore {
    pub(crate) value: i32,
    pub(crate) weights: EvalWeights,
    pub(crate) score: i32,
    pub(crate) blunders: usize,
    pub(crate) movement: i32,
}

fn score_sweep_candidates(
    baseline: EvalWeights,
    parameter: WeightParameter,
    candidates: &[SweepCandidate],
    pass: usize,
    parameter_index: usize,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> SweepScore {
    let original = parameter.value(baseline);
    let seeds = sweep_seeds(config, pass, parameter_index);
    let completed = AtomicUsize::new(0);
    let total = candidates.len();
    let scores: Vec<SweepScore> = candidates
        .par_iter()
        .map(|candidate| {
            let mut score = 0;
            let mut blunders = 0usize;
            for seed in &seeds {
                if training_expired(deadline) {
                    break;
                }
                let report = paired_report(
                    candidate.weights,
                    baseline,
                    *seed,
                    &format!("sweep {}={}", parameter.name, candidate.value),
                    "sweep baseline",
                    config,
                    deadline,
                );
                score += report.delta();
                blunders += report.candidate.blunders;
            }
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            training_progress(
                &format!("sweep {}", parameter.name),
                done,
                total,
                format!("remaining={}", remaining_seconds(deadline)),
            );
            SweepScore {
                value: candidate.value,
                weights: candidate.weights,
                score,
                blunders,
                movement: (candidate.value - original).abs(),
            }
        })
        .collect();

    select_sweep_winner(&scores).unwrap_or(SweepScore {
        value: original,
        weights: baseline,
        score: 0,
        blunders: usize::MAX,
        movement: 0,
    })
}

pub(crate) fn select_sweep_winner(scores: &[SweepScore]) -> Option<SweepScore> {
    scores.iter().copied().max_by(|left, right| {
        left.score
            .cmp(&right.score)
            .then_with(|| right.blunders.cmp(&left.blunders))
            .then_with(|| right.movement.cmp(&left.movement))
    })
}

fn sweep_seeds(config: &TrainerConfig, pass: usize, parameter_index: usize) -> Vec<u64> {
    let mut rng = Lcg::new(
        config.seed
            ^ ((pass as u64) << 32)
            ^ ((parameter_index as u64) << 48)
            ^ 0x51ed_c001_7f4a_7c15,
    );
    (0..config.min_pairs.max(1))
        .map(|_| rng.next_u64())
        .collect()
}

pub(crate) fn sweep_values(
    parameter: WeightParameter,
    value: i32,
    points: usize,
    range_low: f64,
    range_high: f64,
) -> Vec<i32> {
    let points = points.max(3);
    let (low, high) = if value == 0 {
        let step = parameter.zero_step.max(1);
        (-step, step)
    } else if value > 0 {
        (
            ((value as f64) * range_low).round() as i32,
            ((value as f64) * range_high).round() as i32,
        )
    } else {
        (
            ((value as f64) * range_high).round() as i32,
            ((value as f64) * range_low).round() as i32,
        )
    };
    let low = low.clamp(parameter.min, parameter.max);
    let high = high.clamp(parameter.min, parameter.max);
    if low == high {
        return vec![low];
    }

    let mut values = Vec::with_capacity(points);
    for index in 0..points {
        let t = index as f64 / (points - 1) as f64;
        let value = (low as f64 + (high - low) as f64 * t).round() as i32;
        let value = value.clamp(parameter.min, parameter.max);
        if !values.contains(&value) {
            values.push(value);
        }
    }
    values
}

fn shrink_low(low: f64, factor: f64) -> f64 {
    1.0 - (1.0 - low) * factor
}

fn shrink_high(high: f64, factor: f64) -> f64 {
    1.0 + (high - 1.0) * factor
}
