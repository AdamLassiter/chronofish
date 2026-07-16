use rayon::prelude::*;

use super::*;
use crate::cpu::{EvalWeights, SearchInstant};

pub(crate) fn fitness(weights: EvalWeights, config: &CpuCliConfig) -> FitnessReport {
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
    config: &CpuCliConfig,
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
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
    opponent_limit: usize,
) -> FitnessReport {
    let deadline = bounded_training_deadline(
        deadline,
        std::time::Duration::from_millis(max_match_time_ms(config).max(1)),
    );
    let candidate_job = format!(
        "cpu-fitness-{:08x}",
        stable_label_hash(candidate_label, "fitness")
    );
    register_training_job(
        &candidate_job,
        candidate_label,
        "candidate-fitness",
        config.seed,
        vec!["cpu-baseline".into()],
        Some(config.candidate_out.clone().into()),
        [
            ("opponents".into(), opponent_limit.max(1).to_string()),
            (
                "rounds per opponent".into(),
                config.rounds_per_variant.to_string(),
            ),
            (
                "turn budget".into(),
                format!("{} ms", config.training_time_ms),
            ),
            ("node limit".into(), config.nodes.to_string()),
            (
                "task timeout".into(),
                format!("{} ms", max_match_time_ms(config)),
            ),
        ],
    );
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
                &candidate_job,
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

    if crate::training_runtime::cooperative_cancelled() {
        finish_training_task_with_state(
            &candidate_job,
            crate::training_runtime::JobState::Cancelled,
            Some("cancelled by user".into()),
        );
        report.score = i32::MIN / 4;
    } else if training_deadline_expired(deadline) {
        finish_training_task_with_state(
            &candidate_job,
            crate::training_runtime::JobState::TimedOut,
            Some(format!(
                "candidate exceeded {} ms",
                max_match_time_ms(config)
            )),
        );
        report.score = i32::MIN / 4;
    } else {
        finish_training_task(&candidate_job);
    }

    report
}

pub(crate) fn paired_baseline_report(
    candidate: EvalWeights,
    seed: u64,
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
) -> PairReport {
    paired_report(
        candidate,
        EvalWeights::default_tuned(),
        seed,
        "comparison candidate",
        "committed baseline",
        "cpu-validation",
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
    parent_job: &str,
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
) -> PairReport {
    let start = seeded_start_position(seed, config, deadline);
    let black_start = start.clone();
    let identity = stable_label_hash(candidate_label, opponent_label);
    let white_match = format!("cpu-match-{seed}-{identity:08x}-white");
    let black_match = format!("cpu-match-{seed}-{identity:08x}-black");
    let (candidate_white, candidate_black) = rayon::join(
        || {
            play_match_until(
                start,
                candidate,
                baseline,
                Color::White,
                candidate_label,
                opponent_label,
                &white_match,
                parent_job,
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
                &black_match,
                parent_job,
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

fn stable_label_hash(candidate: &str, opponent: &str) -> u32 {
    candidate
        .bytes()
        .chain([0])
        .chain(opponent.bytes())
        .fold(0x811c_9dc5, |hash, byte| {
            (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
        })
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
