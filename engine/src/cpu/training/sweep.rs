use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use super::*;
use crate::{
    cpu::{EvalWeights, SearchInstant},
    training_runtime::{stable_run_id, ImprovementRecord, PersistenceRequest, PersistenceWorker},
};

pub(crate) fn run_sweep_training_cycle(config: &CpuCliConfig) {
    if ai_source_is_dirty(&config.ai_src) {
        training_log(
            crate::training_runtime::LogLevel::Error,
            "cpu",
            format!(
                "{} has uncommitted changes; commit or stash them before training",
                config.ai_src
            ),
        );
        std::process::exit(1);
    }

    training_banner(config);
    training_log(
        crate::training_runtime::LogLevel::Info,
        "cpu/sweep",
        "evaluating parameter batches against frozen baselines",
    );
    let deadline = training_deadline(config);
    let candidate = train_weights_sweep_until(config, deadline);
    compare_and_maybe_promote(candidate, config, deadline);
}

pub(crate) fn train_weights_sweep(config: &CpuCliConfig) -> EvalWeights {
    train_weights_sweep_until(config, training_deadline(config))
}

pub(crate) fn train_weights_sweep_until(
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
) -> EvalWeights {
    let parameters = sweep_weight_parameters(&config.sweep_parameter_groups);
    if parameters.is_empty() {
        training_log(
            crate::training_runtime::LogLevel::Warn,
            "cpu/sweep",
            "no trainable parameters were selected",
        );
        return EvalWeights::default_tuned();
    }

    let mut current = EvalWeights::default_tuned();
    let mut range_low = config.sweep_range_low;
    let mut range_high = config.sweep_range_high;
    let mut pass = 0usize;
    let run_id = stable_run_id(config.seed);
    let persistence = PersistenceWorker::start(config.improvement_log.clone().into());
    let persistence_sender = persistence.sender();
    register_training_job(
        "cpu-baseline",
        "Frozen CPU baseline",
        "baseline",
        config.seed,
        Vec::new(),
        Some(config.candidate_out.clone().into()),
        [
            ("source".into(), config.ai_src.clone()),
            (
                "match timeout".into(),
                format!("{} ms", max_match_time_ms(config)),
            ),
            (
                "global timeout".into(),
                config
                    .max_seconds
                    .map_or_else(|| "not configured".into(), |seconds| format!("{seconds} s")),
            ),
        ],
    );
    finish_training_task("cpu-baseline");
    register_training_job(
        "cpu-sweep",
        "Coordinate sweep",
        "scheduler",
        config.seed,
        vec!["cpu-baseline".into()],
        Some(config.candidate_out.clone().into()),
        [
            ("parameter jobs".into(), config.parameter_jobs.to_string()),
            ("parameters".into(), parameters.len().to_string()),
            ("candidate points".into(), config.sweep_points.to_string()),
            ("paired seeds".into(), config.min_pairs.max(1).to_string()),
        ],
    );

    while !training_expired(deadline) && config.sweep_passes.is_none_or(|limit| pass < limit) {
        let pass_start = current;
        training_log(
            crate::training_runtime::LogLevel::Info,
            "cpu/sweep",
            format!(
                "starting pass {} range={range_low:.3}:{range_high:.3} remaining={}",
                pass + 1,
                remaining_seconds(deadline)
            ),
        );

        for batch_start in (0..parameters.len()).step_by(config.parameter_jobs) {
            if training_expired(deadline) {
                break;
            }
            let batch_end = (batch_start + config.parameter_jobs).min(parameters.len());
            let baseline = current;
            let winners: Vec<(usize, WeightParameter, SweepScore)> = parameters
                [batch_start..batch_end]
                .par_iter()
                .copied()
                .enumerate()
                .map(|(offset, parameter)| {
                    let parameter_index = batch_start + offset;
                    let values = sweep_values(
                        parameter,
                        parameter.value(baseline),
                        config.sweep_points,
                        range_low,
                        range_high,
                    );
                    let candidates = values
                        .into_iter()
                        .map(|value| SweepCandidate {
                            value,
                            weights: parameter.with_value(baseline, value),
                        })
                        .collect::<Vec<_>>();
                    let winner = if candidates.len() < 2 {
                        SweepScore {
                            value: parameter.value(baseline),
                            weights: baseline,
                            score: 0,
                            blunders: usize::MAX,
                            movement: 0,
                        }
                    } else {
                        score_sweep_candidates(
                            baseline,
                            parameter,
                            &candidates,
                            pass,
                            parameter_index,
                            config,
                            deadline,
                        )
                    };
                    (parameter_index, parameter, winner)
                })
                .collect();

            // Rayon preserves indexed order, but sort explicitly so persistence and
            // merge order remain stable if the implementation changes later.
            let mut winners = winners;
            winners.sort_by_key(|(parameter_index, _, _)| *parameter_index);
            for (parameter_index, parameter, winner) in winners {
                let original = parameter.value(baseline);
                if winner.value != original && !training_expired(deadline) {
                    current = parameter.with_value(current, winner.value);
                    let path = std::path::PathBuf::from(&config.candidate_out);
                    let record = ImprovementRecord::now(
                        &run_id,
                        format!("cpu-pass-{pass}-{}", parameter.name),
                        config.seed,
                        0.0,
                        winner.score as f64,
                        format!(
                            "{} changed from {} to {}",
                            parameter.name, original, winner.value
                        ),
                        path.clone(),
                    );
                    persistence_sender
                        .send(PersistenceRequest::Candidate {
                            path,
                            bytes: current.to_json().into_bytes(),
                            record,
                        })
                        .unwrap_or_else(|error| {
                            panic!("CPU candidate persistence worker stopped: {error}")
                        });
                    training_log(
                        crate::training_runtime::LogLevel::Success,
                        "cpu/sweep",
                        format!(
                            "{} improved {} -> {} score={} candidate={}",
                            parameter.name,
                            original,
                            winner.value,
                            winner.score,
                            config.candidate_out
                        ),
                    );
                } else {
                    training_log(
                        crate::training_runtime::LogLevel::Debug,
                        "cpu/sweep",
                        format!(
                            "{} kept {} score={}",
                            parameter.name, original, winner.score
                        ),
                    );
                }
                training_task_progress(
                    "cpu-sweep",
                    parameter_index + 1,
                    parameters.len(),
                    format!(
                        "pass={} parameter={} best={} score={} remaining={}",
                        pass + 1,
                        parameter.name,
                        winner.value,
                        winner.score,
                        remaining_seconds(deadline)
                    ),
                );
            }
        }

        pass += 1;
        if current == pass_start {
            training_log(
                crate::training_runtime::LogLevel::Info,
                "cpu/sweep",
                format!("stopping after pass {pass}: no parameter changed"),
            );
            break;
        }
        range_low = shrink_low(range_low, config.sweep_shrink);
        range_high = shrink_high(range_high, config.sweep_shrink);
    }

    persistence
        .shutdown()
        .unwrap_or_else(|message| panic!("CPU candidate persistence failed: {message}"));
    finish_training_task("cpu-sweep");
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
    #[allow(dead_code)]
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
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
) -> SweepScore {
    let original = parameter.value(baseline);
    let seeds = sweep_seeds(config, pass, parameter_index);
    let task_id = format!("cpu-pass-{}-parameter-{}", pass + 1, parameter.name);
    let baseline_id = format!("cpu-pass-{}-baseline", pass + 1);
    register_training_job(
        &baseline_id,
        format!("Pass {} frozen baseline", pass + 1),
        "baseline",
        config.seed,
        vec!["cpu-baseline".into()],
        None,
        [("pass".into(), (pass + 1).to_string())],
    );
    finish_training_task(&baseline_id);
    let task_deadline = bounded_training_deadline(
        deadline,
        std::time::Duration::from_millis(max_match_time_ms(config).max(1)),
    );
    register_training_job(
        &task_id,
        format!("Pass {}: {}", pass + 1, parameter.name),
        "coordinate-sweep",
        config.seed,
        vec![baseline_id],
        Some(config.candidate_out.clone().into()),
        [
            ("parameter".into(), parameter.name.into()),
            ("baseline value".into(), original.to_string()),
            (
                "bounds".into(),
                format!("{}..={}", parameter.min, parameter.max),
            ),
            (
                "tested values".into(),
                candidates
                    .iter()
                    .map(|candidate| candidate.value.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            ("paired seeds".into(), seeds.len().to_string()),
            (
                "task timeout".into(),
                format!("{} ms", max_match_time_ms(config)),
            ),
        ],
    );
    let completed = AtomicUsize::new(0);
    let scores: Vec<SweepScore> = candidates
        .par_iter()
        .map(|candidate| {
            let mut score = 0;
            let mut blunders = 0usize;
            for seed in &seeds {
                if training_expired(task_deadline) {
                    break;
                }
                let report = paired_report(
                    candidate.weights,
                    baseline,
                    *seed,
                    &format!("sweep {}={}", parameter.name, candidate.value),
                    "sweep baseline",
                    &task_id,
                    config,
                    task_deadline,
                );
                score += report.delta();
                blunders += report.candidate.blunders;
            }
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            training_task_progress(
                &task_id,
                done,
                candidates.len(),
                format!(
                    "pass={} value={} score={} remaining={}",
                    pass + 1,
                    candidate.value,
                    score,
                    remaining_seconds(task_deadline)
                ),
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

    if crate::training_runtime::cooperative_cancelled() {
        finish_training_task_with_state(
            &task_id,
            crate::training_runtime::JobState::Cancelled,
            Some("cancelled by user".into()),
        );
        return SweepScore {
            value: original,
            weights: baseline,
            score: 0,
            blunders: usize::MAX,
            movement: 0,
        };
    }
    if training_deadline_expired(task_deadline) {
        finish_training_task_with_state(
            &task_id,
            crate::training_runtime::JobState::TimedOut,
            Some(format!(
                "parameter exceeded {} ms",
                max_match_time_ms(config)
            )),
        );
        training_log(
            crate::training_runtime::LogLevel::Warn,
            "cpu/sweep",
            format!(
                "{} timed out; retaining baseline value {}",
                parameter.name, original
            ),
        );
        return SweepScore {
            value: original,
            weights: baseline,
            score: 0,
            blunders: usize::MAX,
            movement: 0,
        };
    }
    finish_training_task(&task_id);
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

fn sweep_seeds(config: &CpuCliConfig, pass: usize, parameter_index: usize) -> Vec<u64> {
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
