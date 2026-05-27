fn print_threshold_progress(
    wins: usize,
    losses: usize,
    draws: usize,
    total_delta: i32,
    significance: Significance,
    config: &TrainerConfig,
) {
    let wins_needed = config.min_wins.saturating_sub(wins);
    let delta_needed = config.min_total_delta.saturating_sub(total_delta);
    println!(
        "comparison: wins={wins} losses={losses} draws={draws} required_wins={} total_delta={total_delta}/{}",
        config.min_wins, config.min_total_delta
    );
    println!(
        "stats: samples={} mean_delta={:.1} stddev={:.1} stderr={:.1} lower95={:.1}",
        significance.samples,
        significance.mean,
        significance.stddev,
        significance.stderr,
        significance.lower_95
    );
    println!("threshold remaining: wins={wins_needed} total_delta={delta_needed}");
}

#[derive(Clone, Copy)]
struct Significance {
    samples: usize,
    mean: f64,
    stddev: f64,
    stderr: f64,
    lower_95: f64,
}

fn significance(values: &[i32]) -> Significance {
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

fn should_promote(
    wins: usize,
    losses: usize,
    total_delta: i32,
    significance: Significance,
    config: &TrainerConfig,
) -> bool {
    wins >= config.min_wins
        && wins > losses
        && total_delta >= config.min_total_delta
        && significance.samples >= 3
        && significance.lower_95 > 0.0
}
