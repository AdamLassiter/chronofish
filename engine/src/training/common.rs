use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
    time::Instant,
};

// Native-only training harness for EvalWeights. It plays full matches, compares
// candidate weights against committed defaults, and can promote a statistically
// significant improvement.
pub(crate) fn training_banner(config: &TrainerConfig) {
    pretty_log::banner(
        "CPU Training",
        format!("parameters=cpu-v1 seed={}", config.seed),
    );
    pretty_log::label_value(
        "max seconds",
        config
            .max_seconds
            .map_or_else(|| "none".to_string(), |seconds| seconds.to_string()),
    );
    pretty_log::label_value("strategy", config.training_strategy.as_str());
    pretty_log::label_value("population", config.population);
    pretty_log::label_value("finalists", config.finalist_count);
    pretty_log::label_value(
        "sweep groups",
        config
            .sweep_parameter_groups
            .iter()
            .map(|group| group.as_str())
            .collect::<Vec<_>>()
            .join(","),
    );
    pretty_log::label_value("sweep points", config.sweep_points);
    pretty_log::label_value(
        "sweep passes",
        config
            .sweep_passes
            .map_or_else(|| "unlimited".to_string(), |passes| passes.to_string()),
    );
    pretty_log::label_value(
        "sweep range",
        format!(
            "{:.3}:{:.3} shrink={:.3}",
            config.sweep_range_low, config.sweep_range_high, config.sweep_shrink
        ),
    );
    pretty_log::label_value("pair batch", config.pair_batch);
    pretty_log::label_value("turn time ms", config.training_time_ms);
    pretty_log::label_value("nodes", config.nodes);
    pretty_log::label_value("search", config.search_strategy.as_str());
    pretty_log::label_value("opponent variants", config.opponent_variants);
    pretty_log::label_value("rounds per variant", config.rounds_per_variant);
    pretty_log::label_value("hall of fame entries", config.hall_of_fame_entries);
    pretty_log::label_value("min pairs", config.min_pairs);
    pretty_log::label_value("max pairs", config.max_pairs);
    pretty_log::label_value("max match plies", config.max_match_plies);
    pretty_log::label_value("max match ms", max_match_time_ms(config));
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

pub(crate) fn training_progress(
    stage: &str,
    current: usize,
    total: usize,
    detail: impl AsRef<str>,
) {
    training_task_progress(stage, stage, current, total, detail);
}

pub(crate) fn training_task_progress(
    key: &str,
    label: &str,
    current: usize,
    total: usize,
    detail: impl AsRef<str>,
) {
    let Ok(mut tasks) = training_progress_tasks().lock() else {
        return;
    };
    let now = Instant::now();
    let task = tasks
        .entry(key.to_string())
        .or_insert_with(|| TrainingProgressTask {
            label: label.to_string(),
            start: now,
            updated: now,
            start_current: current,
            current,
            total,
            detail: String::new(),
        });
    task.label = label.to_string();
    task.updated = now;
    task.current = current.min(total.max(current));
    task.total = total.max(current).max(1);
    task.detail = detail.as_ref().to_string();

    tasks.retain(|_, task| {
        let age = now.saturating_duration_since(task.updated);
        if task.current >= task.total {
            age.as_secs_f32() < 2.0
        } else {
            age.as_secs_f32() < 20.0
        }
    });

    let mut rows: Vec<TrainingProgressRenderRow> = tasks
        .values()
        .map(|task| {
            let elapsed = now
                .saturating_duration_since(task.start)
                .as_secs_f64()
                .max(0.001);
            let completed = task.current.saturating_sub(task.start_current);
            TrainingProgressRenderRow {
                label: task.label.clone(),
                current: task.current,
                total: task.total,
                rate: completed as f64 / elapsed,
                detail: task.detail.clone(),
                updated: task.updated,
            }
        })
        .collect();
    rows.sort_by_key(|row| row.updated);
    rows.truncate(8);

    let pretty_rows: Vec<pretty_log::ProgressRow<'_>> = rows
        .iter()
        .map(|row| pretty_log::ProgressRow {
            label: &row.label,
            current: row.current,
            total: row.total,
            rate: row.rate,
            detail: &row.detail,
        })
        .collect();
    pretty_log::progress(&pretty_rows);
}

pub(crate) fn finish_training_task(key: &str) {
    let Ok(mut tasks) = training_progress_tasks().lock() else {
        return;
    };
    tasks.remove(key);
    let rows: Vec<TrainingProgressRenderRow> = tasks
        .values()
        .map(|task| TrainingProgressRenderRow {
            label: task.label.clone(),
            current: task.current,
            total: task.total,
            rate: 0.0,
            detail: task.detail.clone(),
            updated: task.updated,
        })
        .collect();
    let pretty_rows: Vec<pretty_log::ProgressRow<'_>> = rows
        .iter()
        .take(8)
        .map(|row| pretty_log::ProgressRow {
            label: &row.label,
            current: row.current,
            total: row.total,
            rate: row.rate,
            detail: &row.detail,
        })
        .collect();
    pretty_log::progress(&pretty_rows);
}

#[derive(Clone)]
struct TrainingProgressTask {
    label: String,
    start: Instant,
    updated: Instant,
    start_current: usize,
    current: usize,
    total: usize,
    detail: String,
}

struct TrainingProgressRenderRow {
    label: String,
    current: usize,
    total: usize,
    rate: f64,
    detail: String,
    updated: Instant,
}

fn training_progress_tasks() -> &'static Mutex<HashMap<String, TrainingProgressTask>> {
    static TASKS: OnceLock<Mutex<HashMap<String, TrainingProgressTask>>> = OnceLock::new();
    TASKS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone)]
pub(crate) struct TrainerConfig {
    pub(crate) generations: usize,
    pub(crate) population: usize,
    pub(crate) training_time_ms: u64,
    pub(crate) nodes: usize,
    pub(crate) seed: u64,
    pub(crate) max_seconds: Option<u64>,
    pub(crate) out: Option<String>,
    pub(crate) score: Option<String>,
    pub(crate) score_default: bool,
    pub(crate) gpu_backend_info: bool,
    pub(crate) gpu_compile_shaders: bool,
    pub(crate) gpu_compile_kernels: bool,
    pub(crate) gpu_dispatch_smoke: bool,
    pub(crate) gpu_training_dispatch_smoke: bool,
    pub(crate) gpu_shader_info: bool,
    pub(crate) gpu_value_model_path: String,
    pub(crate) gpu_model_info: Option<String>,
    pub(crate) gpu_model_probe_zero: Option<String>,
    pub(crate) gpu_project_samples: Option<String>,
    pub(crate) gpu_predict_samples: Option<String>,
    pub(crate) gpu_distill_samples: Option<String>,
    pub(crate) gpu_replay_buffer: Option<String>,
    pub(crate) gpu_replay_append: Option<String>,
    pub(crate) gpu_replay_max: usize,
    pub(crate) gpu_search_snapshot: Option<String>,
    pub(crate) gpu_search_depth: Option<i32>,
    pub(crate) gpu_search_min_depth: Option<i32>,
    pub(crate) gpu_sample_search_snapshot: Option<String>,
    pub(crate) gpu_sample_mode: crate::gpu::training::SearchLabelMode,
    pub(crate) gpu_sample_count: usize,
    pub(crate) gpu_sample_max_plies: usize,
    pub(crate) gpu_train_search_snapshot: Option<String>,
    pub(crate) gpu_train_samples: Option<String>,
    pub(crate) gpu_train_projected_samples: Option<String>,
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
}

pub(crate) fn max_match_time_ms(config: &TrainerConfig) -> u64 {
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
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Sweep => "sweep",
            Self::Genetic => "genetic",
        }
    }

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
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ClassicBasic => "classic-basic",
            Self::AlternateBasic => "alternate-basic",
            Self::Intermediate => "intermediate",
            Self::Advanced => "advanced",
        }
    }

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
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::AlphaBeta => "alpha-beta",
            Self::Beam => "beam",
        }
    }

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
