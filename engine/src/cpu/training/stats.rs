use super::*;

pub(crate) fn print_threshold_progress(
    comparison_stats: ComparisonStats,
    match_stats: MatchStats,
    significance: Significance,
    config: &TrainerConfig,
) {
    let wins_needed = config.min_wins.saturating_sub(comparison_stats.wins);
    let delta_needed = config
        .min_total_delta
        .saturating_sub(comparison_stats.total_delta);
    pretty_log::section("Comparison Summary");
    pretty_log::label_value(
        "pairs",
        format!(
            "{} wins={} losses={} draws={} points={:.1}/{:.1}",
            comparison_stats.played,
            comparison_stats.wins,
            comparison_stats.losses,
            comparison_stats.draws,
            comparison_stats.points,
            comparison_stats.played as f64,
        ),
    );
    pretty_log::label_value(
        "strength",
        format!(
            "win_rate={:.1}% elo={:+.0}",
            comparison_stats.win_rate() * 100.0,
            comparison_stats.estimated_elo(),
        ),
    );
    pretty_log::label_value(
        "delta",
        format!(
            "total={}/{} lower95={:.1} lower95_win_rate={:.1}%",
            comparison_stats.total_delta,
            config.min_total_delta,
            significance.lower_95,
            comparison_stats.lower_95_win_rate() * 100.0,
        ),
    );
    pretty_log::label_value(
        "stats",
        format!(
            "samples={} mean={:.1} stddev={:.1} stderr={:.1}",
            significance.samples, significance.mean, significance.stddev, significance.stderr,
        ),
    );
    pretty_log::label_value("matches", match_stats.summary());
    pretty_log::label_value(
        "threshold remaining",
        format!("wins={wins_needed} total_delta={delta_needed}"),
    );
}

#[derive(Clone, Copy)]
pub(crate) struct Significance {
    pub(crate) samples: usize,
    pub(crate) mean: f64,
    pub(crate) stddev: f64,
    pub(crate) stderr: f64,
    pub(crate) lower_95: f64,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct ComparisonStats {
    pub(crate) played: usize,
    pub(crate) wins: usize,
    pub(crate) losses: usize,
    pub(crate) draws: usize,
    pub(crate) points: f64,
    pub(crate) total_delta: i32,
}

impl ComparisonStats {
    pub(crate) fn record(&mut self, delta: i32) {
        self.played += 1;
        self.total_delta += delta;
        if delta > 0 {
            self.wins += 1;
            self.points += 1.0;
        } else if delta < 0 {
            self.losses += 1;
        } else {
            self.draws += 1;
            self.points += 0.5;
        }
    }

    pub(crate) fn win_rate(self) -> f64 {
        if self.played == 0 {
            0.0
        } else {
            self.points / self.played as f64
        }
    }

    pub(crate) fn lower_95_win_rate(self) -> f64 {
        if self.played < 2 {
            return 0.0;
        }
        let p = self.win_rate();
        let stderr = (p * (1.0 - p) / self.played as f64).sqrt();
        (p - 1.96 * stderr).max(0.0)
    }

    pub(crate) fn upper_95_win_rate(self) -> f64 {
        if self.played < 2 {
            return 1.0;
        }
        let p = self.win_rate();
        let stderr = (p * (1.0 - p) / self.played as f64).sqrt();
        (p + 1.96 * stderr).min(1.0)
    }

    pub(crate) fn estimated_elo(self) -> f64 {
        let rate = self.win_rate().clamp(0.01, 0.99);
        -400.0 * (1.0 / rate - 1.0).log10()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StatisticalDecision {
    Promote,
    Reject,
    Inconclusive,
    Continue,
}

pub(crate) fn statistical_decision(
    stats: ComparisonStats,
    deltas: &[i32],
    significance: Significance,
    config: &TrainerConfig,
) -> StatisticalDecision {
    if stats.played >= config.min_pairs
        && stats.total_delta > 0
        && significance.lower_95 > 0.0
        && stats.lower_95_win_rate() > 0.5
    {
        return StatisticalDecision::Promote;
    }
    if stats.played >= config.min_pairs && stats.upper_95_win_rate() < 0.5 {
        return StatisticalDecision::Reject;
    }
    if draw_stagnant(deltas, config) || stats.played >= config.max_pairs {
        return StatisticalDecision::Inconclusive;
    }
    StatisticalDecision::Continue
}

pub(crate) fn draw_stagnant(deltas: &[i32], config: &TrainerConfig) -> bool {
    if deltas.len() < config.draw_window {
        return false;
    }
    let window = &deltas[deltas.len() - config.draw_window..];
    let draws = window.iter().filter(|delta| **delta == 0).count();
    let draw_rate = draws as f64 / window.len() as f64;
    let total_delta: i32 = window.iter().sum();
    draw_rate >= config.draw_rate_limit && total_delta.abs() <= config.min_total_delta
}

#[derive(Clone, Copy, Default)]
pub(crate) struct MatchStats {
    pub(crate) played: usize,
    pub(crate) wins: usize,
    pub(crate) losses: usize,
    pub(crate) draws: usize,
}

impl MatchStats {
    pub(crate) fn record_score(&mut self, score: i32) {
        self.played += 1;
        if score > 0 {
            self.wins += 1;
        } else if score < 0 {
            self.losses += 1;
        } else {
            self.draws += 1;
        }
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.played += other.played;
        self.wins += other.wins;
        self.losses += other.losses;
        self.draws += other.draws;
    }

    pub(crate) fn summary(self) -> String {
        format!(
            "played={} wins={} losses={} draws={}",
            self.played, self.wins, self.losses, self.draws
        )
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct FitnessReport {
    pub(crate) score: i32,
    pub(crate) matches: MatchStats,
    pub(crate) blunders: usize,
}

impl FitnessReport {
    pub(crate) fn add_match(&mut self, report: MatchReport) {
        self.score += report.score;
        self.matches.record_score(report.result.score());
        self.blunders += usize::from(report.blunder);
    }

    pub(crate) fn summary(self) -> String {
        format!(
            "score={} {} blunders={}",
            self.score,
            self.matches.summary(),
            self.blunders
        )
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MatchReport {
    pub(crate) score: i32,
    pub(crate) result: MatchResult,
    pub(crate) blunder: bool,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct PairReport {
    pub(crate) candidate: FitnessReport,
    pub(crate) baseline: FitnessReport,
}

impl PairReport {
    pub(crate) fn delta(self) -> i32 {
        self.candidate.score - self.baseline.score
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchResult {
    Win,
    Loss,
    Draw,
}

impl MatchResult {
    pub(crate) fn score(self) -> i32 {
        match self {
            Self::Win => 1,
            Self::Loss => -1,
            Self::Draw => 0,
        }
    }
}

pub(crate) fn significance(values: &[i32]) -> Significance {
    if values.is_empty() {
        return Significance {
            samples: 0,
            mean: 0.0,
            stddev: 0.0,
            stderr: f64::INFINITY,
            lower_95: f64::NEG_INFINITY,
        };
    }

    let samples = values.len();
    let mean = values.iter().map(|value| *value as f64).sum::<f64>() / samples as f64;
    let variance = if samples > 1 {
        values
            .iter()
            .map(|value| {
                let delta = *value as f64 - mean;
                delta * delta
            })
            .sum::<f64>()
            / (samples - 1) as f64
    } else {
        0.0
    };
    let stddev = variance.sqrt();
    let stderr = if samples > 1 {
        stddev / (samples as f64).sqrt()
    } else {
        f64::INFINITY
    };
    let lower_95 = mean - 1.96 * stderr;

    Significance {
        samples,
        mean,
        stddev,
        stderr,
        lower_95,
    }
}

pub(crate) fn should_promote(
    comparison_stats: ComparisonStats,
    significance: Significance,
    config: &TrainerConfig,
) -> bool {
    statistical_decision(comparison_stats, &[], significance, config)
        == StatisticalDecision::Promote
}
