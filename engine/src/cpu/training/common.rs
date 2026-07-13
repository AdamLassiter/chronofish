use crate::training_runtime::{self, JobProgress, JobState, LogLevel, TrainingEvent};

pub(crate) fn training_log(level: LogLevel, scope: impl Into<String>, message: impl Into<String>) {
    training_runtime::log(level, scope, message);
}

pub(crate) fn training_banner(config: &CpuCliConfig) {
    training_log(
        LogLevel::Info,
        "cpu",
        format!(
            "starting {:?} training seed={} parameters={} jobs={} turn_ms={} nodes={} max_seconds={}",
            config.training_strategy,
            config.seed,
            super::parameters::sweep_weight_parameters(&config.sweep_parameter_groups).len(),
            config.parameter_jobs,
            config.training_time_ms,
            config.nodes,
            config
                .max_seconds
                .map_or_else(|| "unlimited".to_string(), |value| value.to_string())
        ),
    );
}

pub(crate) fn training_progress(
    stage: &str,
    current: usize,
    total: usize,
    detail: impl Into<String>,
) {
    training_task_progress(stage, current, total, detail);
}

pub(crate) fn training_task_progress(
    key: &str,
    current: usize,
    total: usize,
    detail: impl Into<String>,
) {
    if crate::training_runtime::global_ui_mode() == crate::training_runtime::UiMode::Plain
        && key.starts_with("cpu-match-")
    {
        return;
    }
    training_runtime::render_structured_event(&TrainingEvent::Progress {
        job_id: key.to_string(),
        progress: JobProgress {
            current: current as u64,
            total: total.max(1) as u64,
            detail: detail.into(),
            ..Default::default()
        },
    });
}

pub(crate) fn finish_training_task(key: &str) {
    if crate::training_runtime::global_ui_mode() == crate::training_runtime::UiMode::Plain
        && key.starts_with("cpu-match-")
    {
        return;
    }
    training_runtime::render_structured_event(&TrainingEvent::State {
        job_id: key.to_string(),
        state: JobState::Completed,
        error: None,
    });
}

pub(crate) const MAX_TRAINING_SEARCH_DEPTH: i32 = 64;

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TrainingParameters {
    pub(crate) time_ms: u64,
    pub(crate) nodes: usize,
    pub(crate) candidates: Option<usize>,
    pub(crate) finalists: Option<usize>,
    pub(crate) pair_batch: Option<usize>,
    pub(crate) opponent_variants: usize,
    pub(crate) screening_opponent_variants: usize,
    pub(crate) rounds_per_variant: usize,
    pub(crate) hall_of_fame_entries: usize,
    pub(crate) league_contenders: usize,
    pub(crate) league_hall_of_fame_entries: usize,
    pub(crate) min_pairs: usize,
    pub(crate) max_pairs: usize,
    pub(crate) draw_window: usize,
    pub(crate) draw_rate_limit: f64,
    pub(crate) max_match_plies: i32,
    pub(crate) max_match_time_ms: u64,
    pub(crate) max_generations_without_candidate: usize,
}

#[derive(Clone)]
pub(crate) struct CpuCliConfig {
    pub(crate) generations: usize,
    pub(crate) population: usize,
    pub(crate) training_time_ms: u64,
    pub(crate) nodes: usize,
    pub(crate) seed: u64,
    pub(crate) max_seconds: Option<u64>,
    pub(crate) out: Option<String>,
    pub(crate) ui: crate::training_runtime::UiMode,
    pub(crate) candidate_out: String,
    pub(crate) improvement_log: String,
    pub(crate) score: Option<String>,
    pub(crate) score_default: bool,
    #[allow(dead_code)]
    pub(crate) gpu: crate::gpu::cli::GpuCliConfig,
    pub(crate) cpu_search_snapshot: Option<String>,
    pub(crate) train_cycle: bool,
    pub(crate) training_strategy: CpuTrainingStrategy,
    pub(crate) compare_seeds: Vec<u64>,
    pub(crate) min_wins: usize,
    pub(crate) min_total_delta: i32,
    pub(crate) verify: String,
    pub(crate) ai_src: String,
    pub(crate) hall_of_fame: String,
    pub(crate) opponent_variants: usize,
    pub(crate) screening_opponent_variants: usize,
    pub(crate) rounds_per_variant: usize,
    pub(crate) hall_of_fame_entries: usize,
    pub(crate) league_contenders: usize,
    pub(crate) league_hall_of_fame_entries: usize,
    pub(crate) min_pairs: usize,
    pub(crate) pair_batch: usize,
    pub(crate) max_pairs: usize,
    pub(crate) draw_window: usize,
    pub(crate) draw_rate_limit: f64,
    pub(crate) max_match_plies: i32,
    pub(crate) max_match_time_ms: u64,
    pub(crate) max_generations_without_candidate: usize,
    pub(crate) finalist_count: usize,
    pub(crate) search_strategy: TrainingSearchStrategy,
    pub(crate) sweep_parameter_groups: Vec<SweepParameterGroup>,
    pub(crate) sweep_points: usize,
    pub(crate) sweep_passes: Option<usize>,
    pub(crate) sweep_range_low: f64,
    pub(crate) sweep_range_high: f64,
    pub(crate) sweep_shrink: f64,
    pub(crate) parameter_jobs: usize,
}

pub(crate) fn max_match_time_ms(config: &CpuCliConfig) -> u64 {
    if config.max_match_time_ms > 0 {
        return config.max_match_time_ms;
    }

    config
        .training_time_ms
        .max(1)
        .saturating_mul(config.max_match_plies.max(1) as u64)
        .saturating_mul(60)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CpuTrainingStrategy {
    Sweep,
    Genetic,
}

impl CpuTrainingStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "sweep" | "coordinate" | "coordinate-sweep" => Ok(Self::Sweep),
            "genetic" | "evolution" | "evolutionary" => Ok(Self::Genetic),
            other => Err(format!("unknown CPU training strategy `{other}`")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SweepParameterGroup {
    ClassicBasic,
    AlternateBasic,
    Intermediate,
    Advanced,
}

impl SweepParameterGroup {
    pub(crate) fn parse_list(value: &str) -> Result<Vec<Self>, String> {
        let mut groups = Vec::new();
        for raw in value.split([',', ' ']) {
            let part = raw.trim().to_ascii_lowercase();
            if part.is_empty() {
                continue;
            }
            match part.as_str() {
                "all" => {
                    return Ok(vec![
                        Self::ClassicBasic,
                        Self::AlternateBasic,
                        Self::Intermediate,
                        Self::Advanced,
                    ]);
                }
                "classic-basic" | "classic" => push_unique_group(&mut groups, Self::ClassicBasic),
                "alternate-basic" | "alternate" => {
                    push_unique_group(&mut groups, Self::AlternateBasic)
                }
                "intermediate" => push_unique_group(&mut groups, Self::Intermediate),
                "advanced" => push_unique_group(&mut groups, Self::Advanced),
                other => return Err(format!("unknown sweep parameter group `{other}`")),
            }
        }
        if groups.is_empty() {
            return Err("at least one sweep parameter group is required".to_string());
        }
        Ok(groups)
    }
}

fn push_unique_group(groups: &mut Vec<SweepParameterGroup>, group: SweepParameterGroup) {
    if !groups.contains(&group) {
        groups.push(group);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrainingSearchStrategy {
    AlphaBeta,
    Beam,
}

impl TrainingSearchStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "alpha-beta" | "alphabeta" | "alpha" => Ok(Self::AlphaBeta),
            "beam" => Ok(Self::Beam),
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
