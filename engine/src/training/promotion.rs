use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use super::*;

pub(crate) fn compare_and_maybe_promote(
    candidate: EvalWeights,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) {
    if candidate == EvalWeights::default_tuned() {
        pretty_log::warn("candidate inconclusive: selected weights match committed baseline");
        return;
    }
    let candidate_json = candidate.to_json();
    let mut comparison_stats = ComparisonStats::default();
    let mut deltas = Vec::new();
    let mut comparison_match_stats = MatchStats::default();
    let hall_of_fame = load_hall_of_fame(&config.hall_of_fame, config.hall_of_fame_entries);

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
