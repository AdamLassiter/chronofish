use clap::Parser;

use super::*;
use crate::cpu::EvalWeights;

#[derive(Parser)]
#[command(
    name = "train-cpu",
    about = "Run native CPU heuristic training and CPU search",
    after_help = "Commands: train (cyclic training), search [SNAPSHOT], score [PARAMETERS]. Existing flat flags remain supported."
)]
struct CpuCliArgs {
    /// Search budgets and output options shared with native GPU training.
    #[command(flatten)]
    common: crate::gpu::cli::CommonCliArgs,
    /// Repeat training cycles until interrupted instead of running one training pass.
    #[arg(long)]
    train_cycle: bool,
    /// Maximum generations to evaluate for genetic training.
    #[arg(long)]
    generations: Option<usize>,
    /// Number of candidate parameter sets in each genetic generation.
    #[arg(long)]
    population: Option<usize>,
    /// CPU training method: `sweep` or `genetic`.
    #[arg(long = "strategy", visible_alias = "training-strategy")]
    strategy: Option<String>,
    /// Comma-separated parameter groups to tune during sweep training.
    #[arg(long = "parameter-groups", visible_alias = "sweep-groups")]
    parameter_groups: Option<String>,
    /// Number of values sampled for each parameter in a sweep pass.
    #[arg(long)]
    sweep_points: Option<usize>,
    /// Number of sweep passes; omit for the configured default.
    #[arg(long)]
    sweep_passes: Option<usize>,
    /// Inclusive multiplier range for sweeps, formatted as `LOW,HIGH`.
    #[arg(long)]
    sweep_range: Option<String>,
    /// Range-shrink multiplier applied between sweep passes.
    #[arg(long)]
    sweep_shrink: Option<f64>,
    /// Search method used to generate training moves: `alpha-beta` or `beam`.
    #[arg(long)]
    search_strategy: Option<String>,
    /// Deterministic random seed for candidate generation and matches.
    #[arg(long)]
    seed: Option<u64>,
    /// Number of heuristic parameters evaluated concurrently in a frozen-baseline batch.
    #[arg(long)]
    parameter_jobs: Option<usize>,
    /// Score the heuristic weights stored in the given JSON file.
    #[arg(long)]
    score: Option<String>,
    /// Score the built-in default heuristic weights and exit.
    #[arg(long)]
    score_default: bool,
    /// Search a snapshot JSON file with the CPU bot; omit the value to use the initial position.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    cpu_search: Option<String>,
    /// Comma-separated seeds for candidate-versus-baseline comparison games.
    #[arg(long)]
    compare_seeds: Option<String>,
    /// Minimum wins required before a candidate can be promoted.
    #[arg(long)]
    min_wins: Option<usize>,
    /// Minimum aggregate score delta required before a candidate can be promoted.
    #[arg(long)]
    min_total_delta: Option<i32>,
    /// Command to run after a candidate is promoted; use an empty value to skip verification.
    #[arg(long)]
    verify: Option<String>,
    /// Path to the current CPU parameter JSON file used as the baseline.
    #[arg(long)]
    ai_src: Option<String>,
    /// Path to the JSONL hall of fame used as historical opponents.
    #[arg(long)]
    hall_of_fame: Option<String>,
    /// Number of baseline opponent variants used in full candidate evaluation.
    #[arg(long)]
    opponent_variants: Option<usize>,
    /// Number of baseline opponent variants used for preliminary screening.
    #[arg(long)]
    screening_opponent_variants: Option<usize>,
    /// Paired games to play for each opponent variant.
    #[arg(long)]
    rounds_per_variant: Option<usize>,
    /// Number of hall-of-fame entries available as opponents.
    #[arg(long)]
    hall_of_fame_entries: Option<usize>,
    /// Number of current-generation candidates eligible for league matches.
    #[arg(long)]
    league_contenders: Option<usize>,
    /// Number of hall-of-fame entries included in the league.
    #[arg(long)]
    league_hall_of_fame_entries: Option<usize>,
    /// Minimum paired matches to play before accepting a decision.
    #[arg(long)]
    min_pairs: Option<usize>,
    /// Paired matches evaluated concurrently in each batch.
    #[arg(long)]
    pair_batch: Option<usize>,
    /// Maximum paired matches before a candidate is accepted or rejected.
    #[arg(long)]
    max_pairs: Option<usize>,
    /// Number of recent game results used to detect draw stagnation.
    #[arg(long)]
    draw_window: Option<usize>,
    /// Draw-rate threshold that stops an unproductive comparison.
    #[arg(long)]
    draw_rate_limit: Option<f64>,
    /// Maximum plies in one comparison game before adjudication.
    #[arg(long = "max-match-plies", visible_alias = "match-plies")]
    max_match_plies: Option<i32>,
    /// Maximum wall-clock time for one comparison game, in milliseconds.
    #[arg(long = "max-match-ms", visible_alias = "match-ms")]
    max_match_ms: Option<u64>,
    /// Stop after this many generations without a promotable candidate.
    #[arg(long)]
    max_generations_without_candidate: Option<usize>,
    /// Number of top candidates retained for final evaluation.
    #[arg(long)]
    finalists: Option<usize>,
    /// Legacy training configuration path; retained for compatibility.
    #[arg(long, hide = true)]
    config: Option<String>,
    /// Legacy runtime effort name; retained for compatibility.
    #[arg(long, hide = true)]
    effort: Option<String>,
    /// Legacy search-depth override; retained for compatibility.
    #[arg(long, hide = true)]
    depth: Option<i32>,
    /// Legacy ply-limit override; retained for compatibility.
    #[arg(long, hide = true)]
    plies: Option<usize>,
}

pub fn run_training_cli() {
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    let config = CpuCliConfig::from_args(CpuCliArgs::parse_from(
        std::iter::once("train-cpu".to_string()).chain(normalize_cpu_command(raw, true)),
    ));
    crate::training_runtime::set_global_ui_mode(config.ui);

    let interactive = config.ui.resolve() == crate::training_runtime::UiMode::Tui
        && !config.score_default
        && config.score.is_none()
        && config.cpu_search_snapshot.is_none();
    if interactive {
        if let Some(output) =
            crate::training_runtime::run_interactive(move || run_training_with_config(config))
                .unwrap_or_else(|message| panic!("interactive CPU training failed: {message}"))
        {
            println!("{output}");
        }
    } else if let Some(output) = run_training_with_config(config) {
        println!("{output}");
    }
}

fn run_training_with_config(config: CpuCliConfig) -> Option<String> {
    if config.train_cycle {
        // The top-level ./train script loops this mode until interrupted.
        match config.training_strategy {
            CpuTrainingStrategy::Sweep => run_sweep_training_cycle(&config),
            CpuTrainingStrategy::Genetic => run_training_cycle(&config),
        }
        return None;
    }

    if config.score_default {
        println!(
            "{}",
            fitness(EvalWeights::default_tuned(), &config).summary()
        );
        return None;
    }

    if config.cpu_search_snapshot.is_some() {
        let request = cpu_search_request(&config);
        let response =
            crate::cpu::search::search(request).unwrap_or_else(|message| panic!("{message}"));
        println!("{}", response.result_json);
        return None;
    }

    if let Some(path) = &config.score {
        let json = std::fs::read_to_string(path).expect("failed to read score weights");
        let weights = EvalWeights::from_json(&json).expect("failed to parse score weights");
        println!("{}", fitness(weights, &config).summary());
        return None;
    }

    let weights = match config.training_strategy {
        CpuTrainingStrategy::Sweep => train_weights_sweep(&config),
        CpuTrainingStrategy::Genetic => train_weights(&config),
    };
    let json = weights.to_json();
    if let Some(path) = &config.out {
        crate::training_runtime::atomic_replace(std::path::Path::new(path), json.as_bytes())
            .expect("failed to atomically write training output");
    }
    Some(json)
}

fn normalize_cpu_command(mut args: Vec<String>, default_cycle: bool) -> Vec<String> {
    match args.first().map(String::as_str) {
        Some("train") => {
            args.remove(0);
            args.insert(0, "--train-cycle".into());
        }
        Some("search") => {
            args.remove(0);
            args.insert(0, "--cpu-search".into());
        }
        Some("score") => {
            args.remove(0);
            if args.first().is_some_and(|arg| !arg.starts_with('-')) {
                args.insert(0, "--score".into());
            } else {
                args.insert(0, "--score-default".into());
            }
        }
        None if default_cycle => args.push("--train-cycle".into()),
        _ => {}
    }
    args
}

impl CpuCliConfig {
    fn from_args(args: CpuCliArgs) -> Self {
        let training = load_training_parameters();
        let seed = args.seed.unwrap_or_else(random_seed);
        let compare_seeds_overridden = args.compare_seeds.is_some();
        let mut config = Self {
            generations: args.generations.unwrap_or(usize::MAX),
            population: args
                .population
                .unwrap_or_else(|| training.candidates.unwrap_or_else(auto_population)),
            training_time_ms: args.common.training_time_ms.unwrap_or(training.time_ms),
            nodes: args.common.nodes.unwrap_or(training.nodes),
            seed,
            max_seconds: args.common.max_seconds,
            out: args.common.out,
            ui: args.common.ui,
            candidate_out: args
                .common
                .candidate_out
                .unwrap_or_else(|| "engine/models/cpu-v1/parameters.candidate.json".to_string()),
            improvement_log: args.common.improvement_log.unwrap_or_else(|| {
                "engine/models/cpu-v1/parameters.improvements.jsonl".to_string()
            }),
            score: args.score,
            score_default: args.score_default,
            gpu: crate::gpu::cli::GpuCliConfig::default(),
            cpu_search_snapshot: args.cpu_search,
            train_cycle: args.train_cycle,
            training_strategy: args
                .strategy
                .as_deref()
                .map(CpuTrainingStrategy::parse)
                .transpose()
                .unwrap_or_else(|message| panic!("{message}"))
                .unwrap_or(CpuTrainingStrategy::Sweep),
            compare_seeds: args
                .compare_seeds
                .as_deref()
                .and_then(|value| parse_seed_list(Some(value)))
                .unwrap_or_else(|| default_compare_seeds(seed)),
            min_wins: args.min_wins.unwrap_or(0),
            min_total_delta: args.min_total_delta.unwrap_or(0),
            verify: args.verify.unwrap_or_else(|| "cargo test -q".to_string()),
            ai_src: args
                .ai_src
                .unwrap_or_else(|| "engine/models/cpu-v1/parameters.json".to_string()),
            hall_of_fame: args.hall_of_fame.unwrap_or_else(default_hall_of_fame_path),
            opponent_variants: args.opponent_variants.unwrap_or(training.opponent_variants),
            screening_opponent_variants: args
                .screening_opponent_variants
                .unwrap_or(training.screening_opponent_variants),
            rounds_per_variant: args
                .rounds_per_variant
                .unwrap_or(training.rounds_per_variant),
            hall_of_fame_entries: args
                .hall_of_fame_entries
                .unwrap_or(training.hall_of_fame_entries),
            league_contenders: args.league_contenders.unwrap_or(training.league_contenders),
            league_hall_of_fame_entries: args
                .league_hall_of_fame_entries
                .unwrap_or(training.league_hall_of_fame_entries),
            min_pairs: args.min_pairs.unwrap_or(training.min_pairs),
            pair_batch: args
                .pair_batch
                .or(training.pair_batch)
                .unwrap_or_else(|| host_parallelism().max(1)),
            max_pairs: args.max_pairs.unwrap_or(training.max_pairs),
            draw_window: args.draw_window.unwrap_or(training.draw_window),
            draw_rate_limit: args.draw_rate_limit.unwrap_or(training.draw_rate_limit),
            max_match_plies: args.max_match_plies.unwrap_or(training.max_match_plies),
            max_match_time_ms: args.max_match_ms.unwrap_or(training.max_match_time_ms),
            max_generations_without_candidate: args
                .max_generations_without_candidate
                .unwrap_or(training.max_generations_without_candidate),
            finalist_count: args
                .finalists
                .or(training.finalists)
                .unwrap_or_else(auto_finalists),
            search_strategy: args
                .search_strategy
                .as_deref()
                .map(TrainingSearchStrategy::parse)
                .transpose()
                .unwrap_or_else(|message| panic!("{message}"))
                .unwrap_or(TrainingSearchStrategy::AlphaBeta),
            sweep_parameter_groups: args
                .parameter_groups
                .as_deref()
                .map(SweepParameterGroup::parse_list)
                .transpose()
                .unwrap_or_else(|message| panic!("{message}"))
                .unwrap_or_else(|| vec![SweepParameterGroup::ClassicBasic]),
            sweep_points: args.sweep_points.unwrap_or(5),
            sweep_passes: args.sweep_passes.or(Some(2)),
            sweep_range_low: 1.0 / 3.0,
            sweep_range_high: 5.0 / 3.0,
            sweep_shrink: args.sweep_shrink.unwrap_or(0.5),
            parameter_jobs: args.parameter_jobs.unwrap_or_else(|| {
                let selected = args
                    .parameter_groups
                    .as_deref()
                    .and_then(|groups| SweepParameterGroup::parse_list(groups).ok())
                    .map(|groups| sweep_weight_parameters(&groups).len())
                    .unwrap_or_else(|| {
                        sweep_weight_parameters(&[SweepParameterGroup::ClassicBasic]).len()
                    });
                selected.min((host_parallelism() / 2).max(1)).max(1)
            }),
        };
        if let Some(range) = args.sweep_range {
            (config.sweep_range_low, config.sweep_range_high) =
                parse_sweep_range(&range).unwrap_or_else(|message| panic!("{message}"));
        }
        config.normalize(compare_seeds_overridden);
        config
    }

    #[cfg(test)]
    pub(crate) fn from_env(args: Vec<String>) -> Self {
        let args = normalize_cpu_command(args, false);
        if args
            .iter()
            .any(|arg| arg.starts_with("--gpu-") || arg.starts_with("--sample-"))
        {
            let mut config = Self::from_args(
                CpuCliArgs::try_parse_from(["train-cpu"])
                    .expect("default CPU training arguments should parse"),
            );
            config.gpu = crate::gpu::cli::GpuCliConfig::from_env(args);
            return config;
        }
        Self::from_args(
            CpuCliArgs::try_parse_from(std::iter::once("train-cpu".to_string()).chain(args))
                .unwrap_or_else(|error| error.exit()),
        )
    }

    fn normalize(&mut self, compare_seeds_overridden: bool) {
        self.population = self.population.max(4);
        self.training_time_ms = self.training_time_ms.max(1);
        self.nodes = self.nodes.max(1);
        self.pair_batch = self.pair_batch.max(1);
        self.opponent_variants = self.opponent_variants.max(1);
        self.screening_opponent_variants = self
            .screening_opponent_variants
            .clamp(1, self.opponent_variants);
        self.rounds_per_variant = self.rounds_per_variant.max(1);
        self.hall_of_fame_entries = self.hall_of_fame_entries.max(1);
        self.league_contenders = self.league_contenders.max(1);
        self.league_hall_of_fame_entries = self.league_hall_of_fame_entries.max(1);
        self.min_pairs = self.min_pairs.max(1);
        self.max_pairs = self.max_pairs.max(self.min_pairs);
        self.draw_window = self.draw_window.max(1);
        self.draw_rate_limit = self.draw_rate_limit.clamp(0.0, 1.0);
        self.max_match_plies = self.max_match_plies.max(1);
        self.max_generations_without_candidate = self.max_generations_without_candidate.max(1);
        self.finalist_count = self.finalist_count.clamp(2, self.population);
        self.sweep_points = self.sweep_points.max(3);
        if self.sweep_range_low <= 0.0 || self.sweep_range_high <= self.sweep_range_low {
            self.sweep_range_low = 1.0 / 3.0;
            self.sweep_range_high = 5.0 / 3.0;
        }
        self.sweep_shrink = self.sweep_shrink.clamp(0.01, 0.99);
        self.parameter_jobs = self.parameter_jobs.max(1);
        if !compare_seeds_overridden {
            self.compare_seeds = default_compare_seeds(self.seed);
        }
        if self.min_wins == 0 {
            self.min_wins = self.compare_seeds.len() * 2 / 3 + 1;
        }
        if self.min_total_delta == 0 {
            self.min_total_delta = (self.compare_seeds.len() as i32) * 50;
        }
    }

    #[cfg(any())]
    #[allow(dead_code)]
    fn from_env_legacy(args: Vec<String>) -> Self {
        // Clap owns the public option schema and diagnostics. The overlay below
        // preserves defaults loaded from training.json and compatibility flags.
        CpuCliArgs::try_parse_from(std::iter::once("train-cpu".to_string()).chain(args.clone()))
            .unwrap_or_else(|error| error.exit());
        let seed = random_seed();
        let training = load_training_parameters();
        let mut config = Self {
            generations: usize::MAX,
            population: training.candidates.unwrap_or_else(auto_population),
            training_time_ms: training.time_ms,
            nodes: training.nodes,
            seed,
            max_seconds: None,
            out: None,
            score: None,
            score_default: false,
            gpu: crate::gpu::cli::GpuCliConfig::default(),
            cpu_search_snapshot: None,
            train_cycle: false,
            training_strategy: CpuTrainingStrategy::Sweep,
            compare_seeds: default_compare_seeds(seed),
            min_wins: 0,
            min_total_delta: 0,
            verify: "cargo test -q".to_string(),
            ai_src: "engine/models/cpu-v1/parameters.json".to_string(),
            hall_of_fame: default_hall_of_fame_path(),
            opponent_variants: training.opponent_variants,
            screening_opponent_variants: training.screening_opponent_variants,
            rounds_per_variant: training.rounds_per_variant,
            hall_of_fame_entries: training.hall_of_fame_entries,
            league_contenders: training.league_contenders,
            league_hall_of_fame_entries: training.league_hall_of_fame_entries,
            min_pairs: training.min_pairs,
            pair_batch: training
                .pair_batch
                .unwrap_or_else(|| host_parallelism().max(1)),
            max_pairs: training.max_pairs,
            draw_window: training.draw_window,
            draw_rate_limit: training.draw_rate_limit,
            max_match_plies: training.max_match_plies,
            max_match_time_ms: training.max_match_time_ms,
            max_generations_without_candidate: training.max_generations_without_candidate,
            finalist_count: training.finalists.unwrap_or_else(auto_finalists),
            search_strategy: TrainingSearchStrategy::AlphaBeta,
            sweep_parameter_groups: vec![SweepParameterGroup::ClassicBasic],
            sweep_points: 5,
            sweep_passes: Some(2),
            sweep_range_low: 1.0 / 3.0,
            sweep_range_high: 5.0 / 3.0,
            sweep_shrink: 0.5,
        };
        let mut index = 0;
        let mut compare_seeds_overridden = false;
        while index < args.len() {
            let value = args.get(index + 1).cloned();
            if let Some(consumed) = config
                .gpu
                .consume_option(args[index].as_str(), value.as_deref())
            {
                index += consumed;
                continue;
            }
            match args[index].as_str() {
                "--train-cycle" => {
                    config.train_cycle = true;
                    index += 1;
                }
                "--generations" => {
                    config.generations = parse_arg(value, config.generations);
                    index += 2;
                }
                "--population" => {
                    config.population = parse_arg(value, config.population);
                    index += 2;
                }
                "--strategy" | "--training-strategy" => {
                    if let Some(strategy) = value {
                        config.training_strategy = CpuTrainingStrategy::parse(&strategy)
                            .unwrap_or_else(|message| panic!("{message}"));
                    }
                    index += 2;
                }
                "--parameter-groups" | "--sweep-groups" => {
                    if let Some(groups) = value {
                        config.sweep_parameter_groups = SweepParameterGroup::parse_list(&groups)
                            .unwrap_or_else(|message| panic!("{message}"));
                    }
                    index += 2;
                }
                "--sweep-points" => {
                    config.sweep_points = parse_arg(value, config.sweep_points);
                    index += 2;
                }
                "--sweep-passes" => {
                    config.sweep_passes = value.and_then(|raw| raw.parse().ok());
                    index += 2;
                }
                "--sweep-range" => {
                    if let Some(range) = value {
                        let (low, high) =
                            parse_sweep_range(&range).unwrap_or_else(|message| panic!("{message}"));
                        config.sweep_range_low = low;
                        config.sweep_range_high = high;
                    }
                    index += 2;
                }
                "--sweep-shrink" => {
                    config.sweep_shrink = parse_arg(value, config.sweep_shrink);
                    index += 2;
                }
                "--config" | "--effort" => {
                    // Training parameters are global. Consume the legacy effort
                    // selector without changing the loaded training config.
                    index += 2;
                }
                "--depth" => {
                    // Training search is now time bounded. Keep consuming the
                    // retired flag so older scripts do not skew later args.
                    index += 2;
                }
                "--training-time-ms" | "--turn-time-ms" => {
                    config.training_time_ms = parse_arg(value, config.training_time_ms);
                    index += 2;
                }
                "--nodes" => {
                    config.nodes = parse_arg(value, config.nodes);
                    index += 2;
                }
                "--search-strategy" => {
                    if let Some(strategy) = value {
                        config.search_strategy = TrainingSearchStrategy::parse(&strategy)
                            .unwrap_or_else(|message| panic!("{message}"));
                    }
                    index += 2;
                }
                "--plies" => {
                    // Full-match training no longer uses plies, but keep consuming
                    // the flag so older local scripts do not skew later args.
                    index += 2;
                }
                "--seed" => {
                    config.seed = parse_arg(value, config.seed);
                    index += 2;
                }
                "--max-seconds" | "--time-seconds" | "--time-budget" => {
                    config.max_seconds = value.and_then(|raw| raw.parse().ok());
                    index += 2;
                }
                "--out" => {
                    config.out = value;
                    index += 2;
                }
                "--score" => {
                    config.score = value;
                    index += 2;
                }
                "--score-default" => {
                    config.score_default = true;
                    index += 1;
                }
                "--cpu-search" => {
                    if matches!(value.as_deref(), Some(next) if !next.starts_with("--")) {
                        config.cpu_search_snapshot = value;
                        index += 2;
                    } else {
                        config.cpu_search_snapshot = Some(String::new());
                        index += 1;
                    }
                }
                "--compare-seeds" => {
                    config.compare_seeds =
                        parse_seed_list(value.as_deref()).unwrap_or(config.compare_seeds);
                    compare_seeds_overridden = true;
                    index += 2;
                }
                "--min-wins" => {
                    config.min_wins = parse_arg(value, config.min_wins);
                    index += 2;
                }
                "--min-total-delta" => {
                    config.min_total_delta = parse_arg(value, config.min_total_delta);
                    index += 2;
                }
                "--verify" => {
                    config.verify = value.unwrap_or(config.verify);
                    index += 2;
                }
                "--ai-src" => {
                    config.ai_src = value.unwrap_or(config.ai_src);
                    index += 2;
                }
                "--hall-of-fame" => {
                    config.hall_of_fame = value.unwrap_or(config.hall_of_fame);
                    index += 2;
                }
                "--opponent-variants" => {
                    config.opponent_variants = parse_arg(value, config.opponent_variants);
                    index += 2;
                }
                "--screening-opponent-variants" => {
                    config.screening_opponent_variants =
                        parse_arg(value, config.screening_opponent_variants);
                    index += 2;
                }
                "--rounds-per-variant" => {
                    config.rounds_per_variant = parse_arg(value, config.rounds_per_variant);
                    index += 2;
                }
                "--hall-of-fame-entries" => {
                    config.hall_of_fame_entries = parse_arg(value, config.hall_of_fame_entries);
                    index += 2;
                }
                "--league-contenders" => {
                    config.league_contenders = parse_arg(value, config.league_contenders);
                    index += 2;
                }
                "--league-hall-of-fame-entries" => {
                    config.league_hall_of_fame_entries =
                        parse_arg(value, config.league_hall_of_fame_entries);
                    index += 2;
                }
                "--min-pairs" => {
                    config.min_pairs = parse_arg(value, config.min_pairs);
                    index += 2;
                }
                "--pair-batch" => {
                    config.pair_batch = parse_arg(value, config.pair_batch);
                    index += 2;
                }
                "--max-pairs" => {
                    config.max_pairs = parse_arg(value, config.max_pairs);
                    index += 2;
                }
                "--draw-window" => {
                    config.draw_window = parse_arg(value, config.draw_window);
                    index += 2;
                }
                "--draw-rate-limit" => {
                    config.draw_rate_limit = parse_arg(value, config.draw_rate_limit);
                    index += 2;
                }
                "--max-match-plies" | "--match-plies" => {
                    config.max_match_plies = parse_arg(value, config.max_match_plies);
                    index += 2;
                }
                "--max-match-ms" | "--match-ms" => {
                    config.max_match_time_ms = parse_arg(value, config.max_match_time_ms);
                    index += 2;
                }
                "--max-generations-without-candidate" => {
                    config.max_generations_without_candidate =
                        parse_arg(value, config.max_generations_without_candidate);
                    index += 2;
                }
                "--finalists" => {
                    config.finalist_count = parse_arg(value, config.finalist_count);
                    index += 2;
                }
                _ => index += 1,
            }
        }
        config.population = config.population.max(4);
        config.training_time_ms = config.training_time_ms.max(1);
        config.nodes = config.nodes.max(1);
        config.gpu.normalize();
        config.pair_batch = config.pair_batch.max(1);
        config.opponent_variants = config.opponent_variants.max(1);
        config.screening_opponent_variants = config
            .screening_opponent_variants
            .clamp(1, config.opponent_variants);
        config.rounds_per_variant = config.rounds_per_variant.max(1);
        config.hall_of_fame_entries = config.hall_of_fame_entries.max(1);
        config.league_contenders = config.league_contenders.max(1);
        config.league_hall_of_fame_entries = config.league_hall_of_fame_entries.max(1);
        config.min_pairs = config.min_pairs.max(1);
        config.max_pairs = config.max_pairs.max(config.min_pairs);
        config.draw_window = config.draw_window.max(1);
        config.draw_rate_limit = config.draw_rate_limit.clamp(0.0, 1.0);
        config.max_match_plies = config.max_match_plies.max(1);
        config.max_generations_without_candidate = config.max_generations_without_candidate.max(1);
        config.finalist_count = config.finalist_count.clamp(2, config.population);
        config.sweep_points = config.sweep_points.max(3);
        if config.sweep_range_low <= 0.0 || config.sweep_range_high <= config.sweep_range_low {
            config.sweep_range_low = 1.0 / 3.0;
            config.sweep_range_high = 5.0 / 3.0;
        }
        config.sweep_shrink = config.sweep_shrink.clamp(0.01, 0.99);
        if !compare_seeds_overridden {
            config.compare_seeds = default_compare_seeds(config.seed);
        }
        if config.min_wins == 0 {
            config.min_wins = config.compare_seeds.len() * 2 / 3 + 1;
        }
        if config.min_total_delta == 0 {
            config.min_total_delta = (config.compare_seeds.len() as i32) * 50;
        }
        config
    }

    pub(crate) fn with_search(&self, nodes: usize, training_time_ms: u64) -> Self {
        let mut config = self.clone();
        config.nodes = nodes;
        config.training_time_ms = training_time_ms;
        config
    }

    pub(crate) fn screening_search(&self) -> Self {
        let mut config = self.clone();
        config.nodes = (self.nodes / 4).max(20).min(self.nodes);
        config.training_time_ms = (self.training_time_ms / 4)
            .max(1)
            .min(self.training_time_ms);
        config
    }
}

fn cpu_search_request(config: &CpuCliConfig) -> crate::cpu::search::CpuSearchRequest {
    let snapshot_json = config.cpu_search_snapshot.as_ref().and_then(|path| {
        if path.is_empty() {
            None
        } else {
            Some(std::fs::read_to_string(path).unwrap_or_else(|error| {
                panic!("failed to read CPU search snapshot {path}: {error}")
            }))
        }
    });
    crate::cpu::search::CpuSearchRequest {
        snapshot_json,
        parameters_json: std::fs::read_to_string(&config.ai_src).ok(),
        depth: crate::cpu::search::DEFAULT_CPU_SEARCH_DEPTH,
        min_depth: Some(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
        nodes: config.nodes.max(1).min(i32::MAX as usize) as i32,
        time_ms: config.training_time_ms.max(1).min(i32::MAX as u64) as i32,
        search_strategy: crate::cpu::search::CpuSearchStrategy::Beam,
    }
}

fn parse_sweep_range(value: &str) -> Result<(f64, f64), String> {
    let Some((low, high)) = value.split_once(':') else {
        return Err("sweep range must use LOW:HIGH".to_string());
    };
    let low = low
        .parse::<f64>()
        .map_err(|_| format!("invalid sweep range low value `{low}`"))?;
    let high = high
        .parse::<f64>()
        .map_err(|_| format!("invalid sweep range high value `{high}`"))?;
    if low <= 0.0 || high <= low {
        return Err("sweep range must satisfy 0 < LOW < HIGH".to_string());
    }
    Ok((low, high))
}

#[cfg(test)]
mod command_tests {
    use super::*;

    #[test]
    fn subcommands_map_to_compatible_flat_options() {
        assert_eq!(
            normalize_cpu_command(vec!["train".into()], false),
            vec!["--train-cycle"]
        );
        assert_eq!(
            normalize_cpu_command(vec!["search".into(), "board.json".into()], false),
            vec!["--cpu-search", "board.json"]
        );
        assert_eq!(
            normalize_cpu_command(vec!["score".into(), "weights.json".into()], false),
            vec!["--score", "weights.json"]
        );
        assert_eq!(
            normalize_cpu_command(vec!["score".into()], false),
            vec!["--score-default"]
        );
    }
}
