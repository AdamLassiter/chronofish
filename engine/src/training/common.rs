// Native-only genetic training harness for EvalWeights. It plays full matches,
// compares candidate weights against the committed defaults, and can promote a
// statistically significant improvement by patching ai.rs and committing it.
pub(crate) fn training_banner(config: &TrainerConfig) {
    pretty_log::banner(
        "CPU Training",
        format!("effort={} seed={}", config.effort, config.seed),
    );
    pretty_log::label_value(
        "max seconds",
        config
            .max_seconds
            .map_or_else(|| "none".to_string(), |seconds| seconds.to_string()),
    );
    pretty_log::label_value("population", config.population);
    pretty_log::label_value("finalists", config.finalist_count);
    pretty_log::label_value("pair batch", config.pair_batch);
    pretty_log::label_value("turn time ms", config.training_time_ms);
    pretty_log::label_value("nodes", config.nodes);
    pretty_log::label_value("search", config.search_strategy.as_str());
    pretty_log::label_value("min pairs", config.min_pairs);
    pretty_log::label_value("max pairs", config.max_pairs);
}

pub(crate) const MAX_TRAINING_SEARCH_DEPTH: i32 = 64;

pub(crate) fn training_progress(
    stage: &str,
    current: usize,
    total: usize,
    detail: impl AsRef<str>,
) {
    let percent = if total == 0 {
        0
    } else {
        current.saturating_mul(100) / total.max(1)
    };
    pretty_log::transient(format!(
        "{stage} {current}/{total} ({percent}%) {}",
        detail.as_ref()
    ));
}

pub(crate) fn training_note(msg: impl AsRef<str>) {
    pretty_log::persist(msg);
}

#[derive(Clone)]
pub(crate) struct TrainerConfig {
    pub(crate) effort: String,
    pub(crate) generations: usize,
    pub(crate) population: usize,
    pub(crate) training_time_ms: u64,
    pub(crate) nodes: usize,
    pub(crate) seed: u64,
    pub(crate) max_seconds: Option<u64>,
    pub(crate) out: Option<String>,
    pub(crate) score: Option<String>,
    pub(crate) score_default: bool,
    pub(crate) train_cycle: bool,
    pub(crate) compare_seeds: Vec<u64>,
    pub(crate) min_wins: usize,
    pub(crate) min_total_delta: i32,
    pub(crate) verify: String,
    pub(crate) ai_src: String,
    pub(crate) hall_of_fame: String,
    pub(crate) min_pairs: usize,
    pub(crate) pair_batch: usize,
    pub(crate) max_pairs: usize,
    pub(crate) draw_window: usize,
    pub(crate) draw_rate_limit: f64,
    pub(crate) max_generations_without_candidate: usize,
    pub(crate) finalist_count: usize,
    pub(crate) search_strategy: TrainingSearchStrategy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrainingSearchStrategy {
    AlphaBeta,
    #[cfg(feature = "training-beam-search")]
    Beam,
}

impl TrainingSearchStrategy {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::AlphaBeta => "alpha-beta",
            #[cfg(feature = "training-beam-search")]
            Self::Beam => "beam",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "alpha-beta" | "alphabeta" | "alpha" => Ok(Self::AlphaBeta),
            #[cfg(feature = "training-beam-search")]
            "beam" => Ok(Self::Beam),
            #[cfg(not(feature = "training-beam-search"))]
            "beam" => {
                Err("beam training requires the `training-beam-search` Cargo feature".to_string())
            }
            other => Err(format!("unknown training search strategy `{other}`")),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Lcg {
    // Deterministic tiny RNG: good enough for repeatable mutation/crossover and
    // keeps training independent of extra dependencies.
    pub(crate) state: u64,
}
