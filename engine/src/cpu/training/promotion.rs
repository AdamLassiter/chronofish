use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use super::*;
use crate::cpu::{EvalWeights, SearchInstant};

pub(crate) fn compare_and_maybe_promote(
    candidate: EvalWeights,
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
) {
    if candidate == EvalWeights::default_tuned() {
        training_log(
            crate::training_runtime::LogLevel::Warn,
            "cpu/validation",
            "candidate matches the committed baseline; nothing to validate",
        );
        return;
    }
    let mut comparison_stats = ComparisonStats::default();
    let mut deltas = Vec::new();
    let mut comparison_match_stats = MatchStats::default();
    let hall_of_fame = load_hall_of_fame(&config.hall_of_fame, config.hall_of_fame_entries);
    training_log(
        crate::training_runtime::LogLevel::Info,
        "cpu/validation",
        format!(
            "starting paired validation min_pairs={} max_pairs={} remaining={}",
            config.min_pairs,
            config.max_pairs,
            remaining_seconds(deadline)
        ),
    );

    let mut rng = Lcg::new(config.seed ^ 0xadc8_3b19_7f4a_7c15);
    loop {
        if training_expired(deadline) {
            training_log(
                crate::training_runtime::LogLevel::Warn,
                "cpu/validation",
                "validation stopped because the global deadline or cancellation was reached",
            );
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
        let reports: Vec<(u64, PairReport)> = seeds
            .par_iter()
            .copied()
            .map(|seed| {
                let report = paired_baseline_report(candidate, seed, config, deadline);
                training_progress(
                    "cpu-validation",
                    comparison_stats.played + completed.fetch_add(1, Ordering::Relaxed) + 1,
                    config.max_pairs,
                    format!("remaining={}", remaining_seconds(deadline)),
                );
                (seed, report)
            })
            .collect();
        for (_, report) in reports {
            comparison_match_stats.add(report.candidate.matches);
            comparison_match_stats.add(report.baseline.matches);
            let delta = report.delta();
            comparison_stats.record(delta);
            deltas.push(delta);
        }
        let significance = significance(&deltas);
        match statistical_decision(comparison_stats, &deltas, significance, config) {
            StatisticalDecision::Promote => {
                break;
            }
            StatisticalDecision::Reject => {
                break;
            }
            StatisticalDecision::Inconclusive => {
                break;
            }
            StatisticalDecision::Continue => {}
        }
    }

    let significance = significance(&deltas);
    training_log(
        crate::training_runtime::LogLevel::Info,
        "cpu/validation",
        format!(
            "pairs={} wins={} losses={} draws={} delta={} lower95={:.2} matches={}",
            comparison_stats.played,
            comparison_stats.wins,
            comparison_stats.losses,
            comparison_stats.draws,
            comparison_stats.total_delta,
            significance.lower_95,
            comparison_match_stats.summary()
        ),
    );
    if should_promote(comparison_stats, significance, config) {
        training_log(
            crate::training_runtime::LogLevel::Info,
            "cpu/validation",
            "candidate accepted; running verification and promotion",
        );
        promote_weights(candidate, &config.ai_src);
        append_hall_of_fame(&config.hall_of_fame, candidate);
        run_command("cargo", &["fmt"]);
        run_shell(&config.verify);
        run_command("git", &["add", &config.ai_src]);
        if !hall_of_fame.is_empty() || std::path::Path::new(&config.hall_of_fame).is_file() {
            run_command("git", &["add", &config.hall_of_fame]);
        }
        run_command("git", &["commit", "-m", "Tune AI evaluation parameters"]);
        journal_cpu_validation(candidate, comparison_stats, "promoted", config);
        training_log(
            crate::training_runtime::LogLevel::Success,
            "cpu/validation",
            format!("promoted candidate to {}", config.ai_src),
        );
    } else {
        let outcome = if crate::training_runtime::cooperative_cancelled() {
            "cancelled"
        } else if training_expired(deadline) {
            "timed-out"
        } else {
            "validation-rejected"
        };
        journal_cpu_validation(candidate, comparison_stats, outcome, config);
        training_log(
            if outcome == "cancelled" || outcome == "timed-out" {
                crate::training_runtime::LogLevel::Warn
            } else {
                crate::training_runtime::LogLevel::Info
            },
            "cpu/validation",
            format!("candidate outcome={outcome}"),
        );
    }
    finish_training_task("cpu-validation");
}

fn journal_cpu_validation(
    _candidate: EvalWeights,
    comparison: ComparisonStats,
    outcome: &str,
    config: &CpuCliConfig,
) {
    let path = std::path::PathBuf::from(&config.candidate_out);
    let mut record = crate::training_runtime::ImprovementRecord::now(
        crate::training_runtime::stable_run_id(config.seed),
        "cpu-validation",
        config.seed,
        0.0,
        comparison.total_delta as f64,
        format!(
            "paired validation pairs={} wins={} losses={} draws={}",
            comparison.played, comparison.wins, comparison.losses, comparison.draws
        ),
        path,
    );
    record.outcome = outcome.to_string();
    let worker =
        crate::training_runtime::PersistenceWorker::start(config.improvement_log.clone().into());
    worker
        .sender()
        .send(crate::training_runtime::PersistenceRequest::Journal { record })
        .unwrap_or_else(|error| panic!("CPU validation journal worker stopped: {error}"));
    worker
        .shutdown()
        .unwrap_or_else(|message| panic!("CPU validation journal failed: {message}"));
}
