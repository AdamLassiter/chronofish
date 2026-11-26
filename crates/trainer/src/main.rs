mod dashboard;
mod setup;

use std::error::Error;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::time::{Duration, Instant};

use chronofish_alphazero::{
    ArenaGameConfig, ArenaReport, EncodedPosition, ModelSize, NetworkConfig, PolicyValueEvaluator,
    PolicyValueNetwork, Prediction, ReplayBuffer, SearchConfig, SelfPlayConfig, SelfPlayGame,
    TrainConfig, estimate_batch_memory_from_shape, estimate_inference_batch_memory,
    estimate_training_batch_memory, play_arena_from_position, play_self_play_game_with_progress,
};
use chronofish_core::{Game, Ruleset, Side};
use chronofish_neural::VulkanTrainingDevice;
use clap::{CommandFactory, FromArgMatches, Parser, ValueEnum};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

const SELF_PLAY_SEED: u64 = 0x5345_4c46_504c_4159;
const ARENA_SEED: u64 = 0x4152_454e_415f_415a;
const ARENA_OPENING_SEED: u64 = 0x4f50_454e_494e_4753;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RuleSelection {
    Standard,
    Multiverse,
    Variant,
    All,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TrainingDevice {
    Cpu,
    Vulkan,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum InferenceDevice {
    /// Match the fitting device (Vulkan fitting also enables Vulkan self-play).
    Auto,
    Cpu,
    Vulkan,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum ModelSizeArgument {
    /// Large for the CPU bot, balanced for the GPU bot.
    Auto,
    Compact,
    Balanced,
    Large,
}

#[derive(Debug, Parser)]
#[allow(clippy::struct_excessive_bools)]
#[command(about = "Train Chronofish's AlphaZero opponent entirely through self-play")]
struct Arguments {
    #[arg(long, default_value = "models/training")]
    work_dir: PathBuf,
    /// Worker threads used for independent self-play and arena games.
    #[arg(long, default_value_t = default_threads())]
    threads: usize,
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    #[arg(long, default_value_t = 8)]
    games: usize,
    #[arg(long, default_value_t = 48)]
    simulations: usize,
    /// Leaves selected by each MCTS search before one inference request.
    #[arg(long, default_value_t = 8)]
    mcts_batch_size: usize,
    /// Positions merged from concurrent games into one network inference batch.
    #[arg(long, default_value_t = 64)]
    inference_batch_size: usize,
    /// Conservative dynamic-tensor budget for one inference batch, in MiB.
    #[arg(long, default_value_t = 1024)]
    inference_memory_budget_mib: usize,
    /// Games in the promotion screen (six balances both sides across all rulesets).
    #[arg(long, default_value_t = 6)]
    arena_games: usize,
    /// Simulations per arena action, independent of self-play search.
    #[arg(long, default_value_t = 16)]
    arena_simulations: usize,
    /// Maximum arena actions per game; unfinished games count as draws.
    #[arg(long, default_value_t = 160)]
    arena_max_actions: usize,
    /// Random opening actions shared by each colour-reversed arena pair. Arena search is
    /// deterministic, so zero replays one game per ruleset and colour.
    #[arg(long, default_value_t = 8)]
    arena_opening_actions: usize,
    /// Print arena starts, periodic action counts, and low-level device identifiers.
    #[arg(long)]
    verbose: bool,
    /// Actions between arena updates when --verbose is enabled; zero disables updates.
    #[arg(long, default_value_t = 25)]
    arena_progress_actions: usize,
    /// Evaluate the saved candidate against best without generating or training.
    #[arg(long)]
    arena_only: bool,
    /// Fit the saved replay only, without self-play or arena evaluation.
    #[arg(long)]
    fit_only: bool,
    /// Initialize and verify the selected fitting device, then exit.
    #[arg(long)]
    check_training_device: bool,
    #[arg(long, default_value_t = 400)]
    max_actions: usize,
    #[arg(long, default_value_t = 100_000)]
    replay_capacity: usize,
    #[arg(long, default_value_t = 4)]
    epochs: usize,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    /// Approximate padded board/action elements allowed in one fitting batch.
    #[arg(long, default_value_t = 262_144)]
    batch_token_budget: usize,
    /// Conservative dynamic-tensor budget for one fitting batch, in MiB.
    #[arg(long, default_value_t = 4096)]
    training_memory_budget_mib: usize,
    /// Maximum expanded replay positions fitted in one iteration.
    #[arg(long, default_value_t = 20_000)]
    training_examples: usize,
    /// Host-memory budget for expanded replay positions, in MiB.
    #[arg(long, default_value_t = 4096)]
    reconstruction_memory_budget_mib: usize,
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f32,
    /// Minimum arena score for promotion. 0.5 accepts ties; use 0.55 for strict improvement.
    #[arg(long, default_value_t = 0.5)]
    promotion_score: f32,
    /// Discard accumulated candidate progress and fit from the incumbent this run.
    #[arg(long)]
    reset_candidate: bool,
    /// Initialize a fresh checkpoint from existing weights.
    #[arg(long)]
    warm_start: Option<PathBuf>,
    /// Carry Adam moments between fits in the candidate checkpoint.
    #[arg(long)]
    persistent_adam: bool,
    /// Network preset. Auto preserves the large CPU model and uses balanced for Vulkan.
    #[arg(long, value_enum, default_value_t = ModelSizeArgument::Auto)]
    model_size: ModelSizeArgument,
    /// Backend used for network fitting.
    #[arg(long, value_enum, default_value_t = TrainingDevice::Cpu)]
    training_device: TrainingDevice,
    /// Backend used for self-play and arena network inference.
    #[arg(long, value_enum, default_value_t = InferenceDevice::Auto)]
    inference_device: InferenceDevice,
    /// Zero-based index among discrete Vulkan GPUs.
    #[arg(long, default_value_t = 0)]
    gpu_index: usize,
    #[arg(long, default_value_t = 0x4855_4749_4e4e)]
    seed: u64,
    #[arg(long, value_enum, default_value_t = RuleSelection::All)]
    ruleset: RuleSelection,
    /// Print plain line-oriented progress instead of the live terminal dashboard.
    #[arg(long)]
    plain: bool,
    /// Start immediately with these arguments instead of opening the interactive setup screen.
    #[arg(long)]
    skip_setup: bool,
}

#[derive(Clone, Copy)]
struct ArenaRunConfig {
    selection: RuleSelection,
    games: usize,
    verbose: bool,
    progress_actions: usize,
    search: SearchConfig,
    inference_batch_size: usize,
    inference_memory_budget_bytes: usize,
    network_config: NetworkConfig,
    max_actions: usize,
    opening_actions: usize,
    seed: u64,
    training_steps: u64,
}

struct GeneratedGame {
    game: SelfPlayGame,
}

struct InferenceRequest {
    queued: Instant,
    positions: Vec<EncodedPosition>,
    response: SyncSender<Vec<Prediction>>,
}

#[derive(Clone)]
struct BatchingEvaluator {
    requests: SyncSender<InferenceRequest>,
    network_config: NetworkConfig,
    memory_budget_bytes: usize,
}

enum FittingDevice {
    Cpu,
    Vulkan(VulkanTrainingDevice),
}

fn section(title: &str) {
    if dashboard::is_active() {
        dashboard::section(title);
    } else {
        println!("\n{title}");
        println!("{}", "-".repeat(title.chars().count()));
    }
}

fn field(label: &str, value: impl std::fmt::Display) {
    if dashboard::is_active() {
        dashboard::field(label, &value.to_string());
    } else {
        println!("  {label:<20} {value}");
    }
}

fn report(text: String) {
    if dashboard::is_active() {
        dashboard::line(text);
    } else {
        println!("{text}");
    }
}

fn warn(text: String) {
    if dashboard::is_active() {
        dashboard::warning(text);
    } else {
        eprintln!("warning: {text}");
    }
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else if seconds < 3_600.0 {
        let minutes = duration.as_secs() / 60;
        let remainder = duration
            .checked_sub(Duration::from_secs(minutes * 60))
            .expect("whole minutes cannot exceed the duration");
        format!("{minutes}m {:.1}s", remainder.as_secs_f64())
    } else {
        let hours = duration.as_secs() / 3_600;
        let minutes = duration.as_secs() / 60 % 60;
        format!("{hours}h {minutes}m")
    }
}

fn add_digit_separators(mut formatted: String) -> String {
    let mut separator = formatted.len();
    while separator > 3 {
        separator -= 3;
        formatted.insert(separator, ',');
    }
    formatted
}

fn format_count(value: usize) -> String {
    add_digit_separators(value.to_string())
}

fn format_steps(value: u64) -> String {
    add_digit_separators(value.to_string())
}

fn counted(value: usize, noun: &str) -> String {
    format!(
        "{} {noun}{}",
        format_count(value),
        if value == 1 { "" } else { "s" }
    )
}

const fn training_device_label(device: TrainingDevice) -> &'static str {
    match device {
        TrainingDevice::Cpu => "CPU",
        TrainingDevice::Vulkan => "Vulkan",
    }
}

const fn inference_device_label(device: InferenceDevice) -> &'static str {
    match device {
        InferenceDevice::Auto => "automatic",
        InferenceDevice::Cpu => "CPU",
        InferenceDevice::Vulkan => "Vulkan",
    }
}

const fn ruleset_label(selection: RuleSelection) -> &'static str {
    match selection {
        RuleSelection::Standard => "Standard Chess",
        RuleSelection::Multiverse => "5D Chess",
        RuleSelection::Variant => "5D Variant Chess",
        RuleSelection::All => "Standard + both 5D armies",
    }
}

fn main() {
    let result = run();
    dashboard::shutdown(result.is_ok());
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let matches = Arguments::command().get_matches();
    let mut arguments = Arguments::from_arg_matches(&matches)?;
    if !arguments.plain
        && !arguments.skip_setup
        && std::io::stdin().is_terminal()
        && std::io::stdout().is_terminal()
    {
        match setup::run(&matches)? {
            Some(chosen) => arguments = chosen,
            None => return Ok(()),
        }
    }
    validate(&arguments)?;
    if !arguments.plain && !arguments.check_training_device {
        dashboard::start();
    }
    dashboard::set_promotion_score(arguments.promotion_score);
    let pool = ThreadPoolBuilder::new()
        .num_threads(arguments.threads)
        .thread_name(|index| format!("chronofish-trainer-{index}"))
        .build()?;
    train(&arguments, &pool)
}

#[allow(clippy::too_many_lines)]
fn train(arguments: &Arguments, pool: &ThreadPool) -> Result<(), Box<dyn Error>> {
    let fitting_device = prepare_fitting_device(arguments)?;
    if arguments.check_training_device {
        field("Status", "device check complete; training was not started");
        return Ok(());
    }
    let mut rng = ChaCha8Rng::seed_from_u64(arguments.seed);
    let best_path = arguments.work_dir.join("best-v2.json");
    let candidate_path = arguments.work_dir.join("candidate-v2.json");
    let replay_path = arguments.work_dir.join("replay-v2.bin.zst");
    warn_about_v1_files(&arguments.work_dir, &best_path, &replay_path);
    let network_config = NetworkConfig::from(resolved_model_size(arguments));
    let mut best = if let Some(source) = &arguments.warm_start {
        if best_path.exists() || candidate_path.exists() || replay_path.exists() {
            return Err("--warm-start requires a fresh training directory".into());
        }
        let mut model = PolicyValueNetwork::load(source)?;
        if model.config() != network_config {
            return Err("--warm-start checkpoint architecture must match --model-size".into());
        }
        model.clear_optimizer();
        model.save(&best_path)?;
        model
    } else {
        load_or_initialize(&best_path, network_config, &mut rng)?
    };
    let search = SearchConfig {
        simulations: arguments.simulations,
        inference_batch_size: arguments.mcts_batch_size,
        ..SearchConfig::default()
    };
    if arguments.arena_only {
        if !candidate_path.exists() {
            return Err(format!(
                "{} does not exist; complete training before using --arena-only",
                candidate_path.display()
            )
            .into());
        }
        let candidate = PolicyValueNetwork::load(&candidate_path)?;
        if candidate.config() != best.config() {
            return Err("candidate and incumbent network shapes differ".into());
        }
        evaluate_and_maybe_promote(
            &candidate,
            &mut best,
            &best_path,
            arguments,
            search,
            pool,
            fitting_device.as_ref(),
        )?;
        return Ok(());
    }
    let mut replay = if replay_path.exists() {
        ReplayBuffer::load(&replay_path)?
    } else {
        ReplayBuffer::default()
    };

    let mut candidate = resume_candidate(&candidate_path, &best, arguments.reset_candidate)?;
    if arguments.reset_candidate {
        candidate.clear_optimizer();
    }
    if arguments.persistent_adam {
        field(
            "Adam state",
            if candidate.has_optimizer() {
                "resuming"
            } else {
                "no saved moments, starting fresh"
            },
        );
    }
    field("Learner steps", format_steps(candidate.training_steps()));
    if arguments.fit_only {
        if replay.is_empty() {
            return Err(format!("{} contains no training positions", replay_path.display()).into());
        }
        fit_candidate(
            arguments,
            pool,
            fitting_device
                .as_ref()
                .expect("fit-only initializes a fitting device"),
            &replay,
            &mut candidate,
            &mut rng,
        )?;
        candidate.save(&candidate_path)?;
        report(format!(
            "fit-only complete; candidate saved to {} (arena and promotion skipped)",
            candidate_path.display()
        ));
        return Ok(());
    }

    for iteration in 0..arguments.iterations {
        let iteration_started = Instant::now();
        dashboard::set_iteration(iteration + 1, arguments.iterations);
        dashboard::set_phase(dashboard::Phase::SelfPlay);
        section(&format!(
            "Iteration {}/{} · Self-play",
            iteration + 1,
            arguments.iterations
        ));
        let self_play_started = Instant::now();
        let mut decisive = 0;
        let mut truncated = 0;
        let generated = match resolved_inference_device(arguments) {
            InferenceDevice::Cpu | InferenceDevice::Auto => generate_self_play(
                arguments,
                &best,
                network_config,
                search,
                iteration,
                replay.len(),
                pool,
            ),
            InferenceDevice::Vulkan => {
                let device = vulkan_device(fitting_device.as_ref())?;
                let evaluator = device.evaluator(&best)?;
                field(
                    "Inference probe",
                    format!("passed · Vulkan · {}", device.info().name),
                );
                generate_self_play(
                    arguments,
                    &evaluator,
                    network_config,
                    search,
                    iteration,
                    replay.len(),
                    pool,
                )
            }
        };
        for generated in generated {
            decisive += usize::from(
                generated
                    .game
                    .outcome
                    .and_then(|outcome| outcome.winner)
                    .is_some(),
            );
            truncated += usize::from(generated.game.truncated);
            replay.extend_game(generated.game, arguments.replay_capacity);
        }
        field(
            "Self-play result",
            format!(
                "{} in replay · {decisive} decisive · {}",
                counted(replay.len(), "position"),
                counted(truncated, "action-limit draw")
            ),
        );
        field(
            "Self-play time",
            format_duration(self_play_started.elapsed()),
        );
        let write_start = Instant::now();
        replay.save(&replay_path)?;
        field("Replay write time", format_duration(write_start.elapsed()));

        fit_candidate(
            arguments,
            pool,
            fitting_device
                .as_ref()
                .expect("training initializes a fitting device"),
            &replay,
            &mut candidate,
            &mut rng,
        )?;
        let write_start = Instant::now();
        candidate.save(&candidate_path)?;
        field(
            "Candidate write time",
            format_duration(write_start.elapsed()),
        );

        let arena_started = Instant::now();
        evaluate_and_maybe_promote(
            &candidate,
            &mut best,
            &best_path,
            arguments,
            search,
            pool,
            fitting_device.as_ref(),
        )?;
        field("Arena time", format_duration(arena_started.elapsed()));
        field(
            "Iteration time",
            format_duration(iteration_started.elapsed()),
        );
    }
    Ok(())
}

fn prepare_fitting_device(arguments: &Arguments) -> Result<Option<FittingDevice>, Box<dyn Error>> {
    if arguments.arena_only && arguments.fit_only {
        return Err("--arena-only and --fit-only cannot be used together".into());
    }
    if arguments.check_training_device && (arguments.arena_only || arguments.fit_only) {
        return Err(
            "--check-training-device cannot be combined with --arena-only or --fit-only".into(),
        );
    }
    let inference_device = resolved_inference_device(arguments);
    if inference_device == InferenceDevice::Vulkan
        && !matches!(arguments.training_device, TrainingDevice::Vulkan)
        && !arguments.fit_only
        && !arguments.check_training_device
    {
        return Err(
            "Vulkan inference currently requires --training-device vulkan so both phases share one verified adapter"
                .into(),
        );
    }
    let resolved_model = resolved_model_size(arguments);
    if !dashboard::is_active() {
        println!("Chronofish Trainer");
        println!("==============");
    }
    section("Configuration");
    field("Work directory", arguments.work_dir.display());
    field("Rules", ruleset_label(arguments.ruleset));
    field(
        "Model",
        format!(
            "{resolved_model:?} ({})",
            if arguments.model_size == ModelSizeArgument::Auto {
                "automatic"
            } else {
                "explicit"
            }
        ),
    );
    field(
        "Compute",
        format!(
            "{} fitting · {} inference",
            training_device_label(arguments.training_device),
            inference_device_label(inference_device)
        ),
    );
    let worker_note = if !arguments.fit_only && arguments.threads > arguments.games.max(1) {
        format!(
            "{} (up to {} concurrent self-play games)",
            arguments.threads, arguments.games
        )
    } else {
        arguments.threads.to_string()
    };
    field("Workers", worker_note);
    field(
        "Search",
        format!(
            "{}/action · MCTS batch {} · inference batch {}",
            counted(arguments.simulations, "simulation"),
            format_count(arguments.mcts_batch_size),
            format_count(arguments.inference_batch_size)
        ),
    );
    field(
        "Memory limits",
        format!(
            "inference {} MiB · fitting {} MiB · replay {} MiB",
            arguments.inference_memory_budget_mib,
            arguments.training_memory_budget_mib,
            arguments.reconstruction_memory_budget_mib
        ),
    );
    field(
        "Training",
        format!(
            "{} sampled · {} · batch ≤ {}",
            counted(arguments.training_examples, "position"),
            counted(arguments.epochs, "epoch"),
            format_count(arguments.batch_size)
        ),
    );
    section("Compute device");
    let fitting_device = if arguments.arena_only && inference_device == InferenceDevice::Cpu {
        field("Fitting", "not initialized (CPU arena-only mode)");
        None
    } else {
        Some(initialize_fitting_device(arguments)?)
    };
    if !arguments.fit_only && !arguments.check_training_device {
        match inference_device {
            InferenceDevice::Cpu | InferenceDevice::Auto => {
                field("Inference", "CPU · Burn ndarray");
            }
            InferenceDevice::Vulkan => {
                field("Inference", "verified Vulkan GPU · self-play and arena");
            }
        }
    }
    Ok(fitting_device)
}

const fn resolved_model_size(arguments: &Arguments) -> ModelSize {
    match arguments.model_size {
        ModelSizeArgument::Auto => match arguments.training_device {
            TrainingDevice::Cpu => ModelSize::Large,
            TrainingDevice::Vulkan => ModelSize::Balanced,
        },
        ModelSizeArgument::Compact => ModelSize::Compact,
        ModelSizeArgument::Balanced => ModelSize::Balanced,
        ModelSizeArgument::Large => ModelSize::Large,
    }
}

const fn resolved_inference_device(arguments: &Arguments) -> InferenceDevice {
    match arguments.inference_device {
        InferenceDevice::Auto => match arguments.training_device {
            TrainingDevice::Cpu => InferenceDevice::Cpu,
            TrainingDevice::Vulkan => InferenceDevice::Vulkan,
        },
        selected => selected,
    }
}

fn vulkan_device(
    fitting_device: Option<&FittingDevice>,
) -> Result<&VulkanTrainingDevice, Box<dyn Error>> {
    match fitting_device {
        Some(FittingDevice::Vulkan(device)) => Ok(device),
        Some(FittingDevice::Cpu) | None => {
            Err("Vulkan inference was selected without an initialized Vulkan device".into())
        }
    }
}

fn initialize_fitting_device(arguments: &Arguments) -> Result<FittingDevice, Box<dyn Error>> {
    match arguments.training_device {
        TrainingDevice::Cpu => {
            field("Fitting", "CPU · Burn ndarray");
            Ok(FittingDevice::Cpu)
        }
        TrainingDevice::Vulkan => {
            field(
                "Initializing",
                format!("Vulkan discrete GPU {}…", arguments.gpu_index),
            );
            let device = VulkanTrainingDevice::initialize(arguments.gpu_index)?;
            let info = device.info();
            field("GPU", &info.name);
            field(
                "Driver",
                format!(
                    "{} · {}",
                    info.driver,
                    if info.driver_info.is_empty() {
                        "version not reported"
                    } else {
                        &info.driver_info
                    }
                ),
            );
            if arguments.verbose {
                field(
                    "Device details",
                    format!(
                        "{} · {} · index {} · vendor 0x{:04x} · device 0x{:04x}",
                        info.device_type,
                        info.backend,
                        info.discrete_gpu_index,
                        info.vendor,
                        info.device
                    ),
                );
            }
            field("Validation", "Vulkan compute probe passed");
            Ok(FittingDevice::Vulkan(device))
        }
    }
}

fn evaluate_and_maybe_promote(
    candidate: &PolicyValueNetwork,
    best: &mut PolicyValueNetwork,
    best_path: &Path,
    arguments: &Arguments,
    search: SearchConfig,
    pool: &ThreadPool,
    fitting_device: Option<&FittingDevice>,
) -> Result<(), Box<dyn Error>> {
    dashboard::set_phase(dashboard::Phase::Arena);
    section("Arena");
    let config = ArenaRunConfig {
        selection: arguments.ruleset,
        games: arguments.arena_games,
        verbose: arguments.verbose,
        progress_actions: arguments.arena_progress_actions,
        search: SearchConfig {
            simulations: arguments.arena_simulations,
            ..search
        },
        inference_batch_size: arguments.inference_batch_size,
        inference_memory_budget_bytes: mib_to_bytes(arguments.inference_memory_budget_mib),
        network_config: candidate.config(),
        max_actions: arguments.arena_max_actions,
        opening_actions: arguments.arena_opening_actions,
        seed: arguments.seed,
        training_steps: candidate.training_steps(),
    };
    let report = match resolved_inference_device(arguments) {
        InferenceDevice::Cpu | InferenceDevice::Auto => evaluate(candidate, best, config, pool),
        InferenceDevice::Vulkan => {
            let device = vulkan_device(fitting_device)?;
            let candidate_evaluator = device.evaluator(candidate)?;
            let incumbent_evaluator = device.evaluator(best)?;
            field(
                "Inference probe",
                format!("passed · Vulkan · {}", device.info().name),
            );
            evaluate(&candidate_evaluator, &incumbent_evaluator, config, pool)
        }
    };
    let score = report.candidate_score();
    let promoted = arguments.arena_games == 0 || score >= arguments.promotion_score;
    dashboard::arena_result(score, promoted);
    field(
        "Match result",
        format!(
            "candidate {} · incumbent {} · draws {} · score {score:.3}",
            report.candidate_wins, report.incumbent_wins, report.draws
        ),
    );
    field(
        "Decision",
        if promoted {
            "candidate promoted"
        } else {
            "incumbent retained"
        },
    );
    if promoted {
        let mut promoted = candidate.clone();
        promoted.clear_optimizer();
        promoted.save(best_path)?;
        *best = promoted;
    }
    Ok(())
}

// Keep the learner independent of the published incumbent, including across restarts.
fn resume_candidate(
    path: &Path,
    best: &PolicyValueNetwork,
    reset: bool,
) -> Result<PolicyValueNetwork, Box<dyn Error>> {
    if reset || !path.exists() {
        return Ok(best.clone());
    }
    let candidate = PolicyValueNetwork::load(path)?;
    if candidate.config() != best.config() {
        return Err("candidate and incumbent network shapes differ; use --reset-candidate to restart the learner".into());
    }
    if candidate.training_steps() < best.training_steps() {
        Ok(best.clone())
    } else {
        Ok(candidate)
    }
}

fn load_or_initialize(
    path: &Path,
    config: NetworkConfig,
    rng: &mut ChaCha8Rng,
) -> Result<PolicyValueNetwork, Box<dyn Error>> {
    if path.exists() {
        let model = PolicyValueNetwork::load(path)?;
        if model.config() != config {
            return Err(format!(
                "checkpoint architecture is {:?}, but --model-size requests {:?}",
                model.config(),
                config
            )
            .into());
        }
        Ok(model)
    } else {
        let model = PolicyValueNetwork::random(config, rng);
        model.save(path)?;
        Ok(model)
    }
}

fn fit_candidate(
    arguments: &Arguments,
    pool: &ThreadPool,
    fitting_device: &FittingDevice,
    replay: &ReplayBuffer,
    candidate: &mut PolicyValueNetwork,
    rng: &mut ChaCha8Rng,
) -> Result<(), Box<dyn Error>> {
    dashboard::set_phase(dashboard::Phase::Fitting);
    section("Fitting");
    let reconstruction_started = Instant::now();
    let sample = replay.sample_examples_in_pool(
        arguments.training_examples,
        mib_to_bytes(arguments.reconstruction_memory_budget_mib),
        rng,
        pool,
    )?;
    let examples = sample.examples;
    field(
        "Replay sample",
        format!(
            "{} · {} MiB host · {} · {} skipped",
            counted(examples.len(), "position"),
            bytes_to_mib_ceil(sample.estimated_bytes),
            format_duration(reconstruction_started.elapsed()),
            sample.skipped_for_memory
        ),
    );
    if examples.is_empty() {
        return Err("the replay sample produced no training positions".into());
    }
    let largest_position = examples
        .iter()
        .map(|example| {
            estimate_training_batch_memory(
                std::slice::from_ref(&example.position),
                candidate.config(),
            )
        })
        .max_by_key(|estimate| estimate.bytes)
        .expect("non-empty replay sample");
    field(
        "Largest position",
        format!(
            "{} · {} · ~{} MiB fitting memory",
            counted(largest_position.max_boards, "board"),
            counted(largest_position.max_actions, "action"),
            bytes_to_mib_ceil(largest_position.bytes)
        ),
    );
    let fit_started = Instant::now();
    let train_config = TrainConfig {
        persistent_adam: arguments.persistent_adam,
        epochs: arguments.epochs,
        batch_size: arguments.batch_size,
        batch_token_budget: arguments.batch_token_budget,
        batch_memory_budget_bytes: mib_to_bytes(arguments.training_memory_budget_mib),
        learning_rate: arguments.learning_rate,
        ..TrainConfig::default()
    };
    match fitting_device {
        FittingDevice::Cpu => field("Backend", "CPU · Burn ndarray"),
        FittingDevice::Vulkan(device) => {
            field("Backend", format!("Vulkan · {}", device.info().name));
        }
    }
    field(
        "Batch limits",
        format!(
            "size {} · {} padded tokens · {} MiB dynamic memory",
            format_count(train_config.batch_size),
            format_count(train_config.batch_token_budget),
            arguments.training_memory_budget_mib
        ),
    );
    field("Status", "training…");
    let metrics = match fitting_device {
        FittingDevice::Cpu => candidate.train(&examples, train_config, rng)?,
        FittingDevice::Vulkan(device) => {
            candidate.train_vulkan(device, &examples, train_config, rng)?
        }
    };
    if metrics.board_slots > 0 {
        field(
            "Spatial boards",
            format!(
                "{} encoded / {} real / {} padded slots across updates",
                metrics.encoded_boards, metrics.real_boards, metrics.board_slots
            ),
        );
    }
    field(
        "Training result",
        format!(
            "{} · {} total steps · policy loss {:.5} · value loss {:.5}",
            counted(metrics.examples, "example"),
            format_steps(candidate.training_steps()),
            metrics.policy_loss,
            metrics.value_loss
        ),
    );
    dashboard::fit_result(metrics.policy_loss, metrics.value_loss);
    field("Fitting time", format_duration(fit_started.elapsed()));
    Ok(())
}

fn warn_about_v1_files(work_dir: &Path, best_v2: &Path, replay_v2: &Path) {
    let old_best = work_dir.join("best.json");
    if !best_v2.exists() && old_best.exists() {
        warn(format!(
            "{} is a version-1 checkpoint and cannot initialize the new multiverse network; creating {}",
            old_best.display(),
            best_v2.display()
        ));
    }
    let old_replay = work_dir.join("replay.json");
    if !replay_v2.exists() && old_replay.exists() {
        warn(format!(
            "{} is a version-1 replay and is intentionally left untouched; starting {}",
            old_replay.display(),
            replay_v2.display()
        ));
    }
}

#[allow(clippy::too_many_lines)]
fn evaluate<E: PolicyValueEvaluator + ?Sized>(
    candidate: &E,
    incumbent: &E,
    config: ArenaRunConfig,
    pool: &ThreadPool,
) -> ArenaReport {
    let schedule = (0..config.games)
        .map(|game_index| {
            let side = candidate_side(config.selection, game_index);
            (selected_ruleset(config.selection, game_index), side)
        })
        .collect::<Vec<_>>();
    dashboard::schedule(
        dashboard::GameKind::Arena,
        &schedule
            .iter()
            .map(|&(ruleset, side)| (ruleset, Some(side)))
            .collect::<Vec<_>>(),
        config.max_actions,
    );
    let live = dashboard::is_active();
    field(
        "Schedule",
        format!(
            "{} · {} · {}/action · {} limit · {}-action paired openings",
            counted(config.games, "game"),
            counted(pool.current_num_threads(), "worker"),
            counted(config.search.simulations, "simulation"),
            counted(config.max_actions, "action"),
            config.opening_actions
        ),
    );
    let completed = Mutex::new(0_usize);
    let results = with_batched_evaluator(
        candidate,
        config.inference_batch_size,
        config.network_config,
        config.inference_memory_budget_bytes,
        |candidate_evaluator| {
            with_batched_evaluator(
                incumbent,
                config.inference_batch_size,
                config.network_config,
                config.inference_memory_budget_bytes,
                |incumbent_evaluator| {
                    pool.install(|| {
                        schedule
                            .par_iter()
                            .enumerate()
                            .map(|(game_index, &(ruleset, candidate_side))| {
                                dashboard::game_started(game_index);
                                if config.verbose && !live {
                                    println!(
                                        "  [arena {}/{}] started · {ruleset} · candidate {candidate_side}",
                                        game_index + 1,
                                        config.games
                                    );
                                }
                                let mut rng = ChaCha8Rng::seed_from_u64(derived_seed(
                                    config.seed,
                                    ARENA_SEED,
                                    config.training_steps,
                                    usize_to_u64(game_index),
                                ));
                                let pair = arena_pair(config.selection, game_index);
                                let opening = arena_opening(
                                    ruleset,
                                    config.opening_actions,
                                    derived_seed(
                                        config.seed,
                                        ARENA_OPENING_SEED,
                                        config.training_steps,
                                        usize_to_u64(pair) | (ruleset_tag(ruleset) << 32),
                                    ),
                                );
                                let result = play_arena_from_position(
                                    candidate_evaluator,
                                    incumbent_evaluator,
                                    opening,
                                    ArenaGameConfig {
                                        ruleset,
                                        candidate_side,
                                        search: config.search,
                                        max_actions: config.max_actions,
                                    },
                                    &mut rng,
                                    |actions, _| {
                                        dashboard::game_progress(game_index, actions);
                                        if config.verbose
                                            && !live
                                            && config.progress_actions > 0
                                            && actions.is_multiple_of(config.progress_actions)
                                        {
                                            println!(
                                                "  [arena {}/{}] {actions} actions searched",
                                                game_index + 1,
                                                config.games
                                            );
                                        }
                                    },
                                );
                                let description = result.outcome.map_or_else(
                                    || "draw (action limit)".to_owned(),
                                    |outcome| match outcome.winner {
                                        Some(winner) => format!("{winner:?} won ({:?})", outcome.reason),
                                        None => format!("draw ({:?})", outcome.reason),
                                    },
                                );
                                dashboard::game_finished(game_index, result.actions, result.outcome);
                                {
                                    let mut finished = completed
                                        .lock()
                                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                                    *finished += 1;
                                    report(format!(
                                        "  [{finished}/{}] arena game {} · {ruleset} · opening {} · candidate {candidate_side} · {} · {description}",
                                        config.games,
                                        game_index + 1,
                                        pair + 1,
                                        counted(result.actions, "action"),
                                    ));
                                }
                                (candidate_side, result.outcome)
                            })
                            .collect::<Vec<_>>()
                    })
                },
            )
        },
    );
    let mut report = ArenaReport::default();
    for (candidate_side, outcome) in results {
        report.record_outcome(candidate_side, outcome);
    }
    report
}

fn generate_self_play<E: PolicyValueEvaluator + ?Sized>(
    arguments: &Arguments,
    best: &E,
    network_config: NetworkConfig,
    search: SearchConfig,
    iteration: usize,
    replay_len: usize,
    pool: &ThreadPool,
) -> Vec<GeneratedGame> {
    dashboard::schedule(
        dashboard::GameKind::SelfPlay,
        &(0..arguments.games)
            .map(|game_index| {
                (
                    selected_ruleset(arguments.ruleset, iteration + game_index),
                    None,
                )
            })
            .collect::<Vec<_>>(),
        arguments.max_actions,
    );
    field(
        "Schedule",
        format!(
            "{} · {} · {}/action · {} limit",
            counted(arguments.games, "game"),
            counted(pool.current_num_threads(), "worker"),
            counted(search.simulations, "simulation"),
            counted(arguments.max_actions, "action")
        ),
    );
    let completed = Mutex::new(0_usize);
    with_batched_evaluator(
        best,
        arguments.inference_batch_size,
        network_config,
        mib_to_bytes(arguments.inference_memory_budget_mib),
        |evaluator| {
            pool.install(|| {
                (0..arguments.games)
                    .into_par_iter()
                    .map(|game_index| {
                        let ruleset = selected_ruleset(arguments.ruleset, iteration + game_index);
                        let mut rng = ChaCha8Rng::seed_from_u64(derived_seed(
                            arguments.seed,
                            SELF_PLAY_SEED,
                            usize_to_u64(iteration) ^ usize_to_u64(replay_len).rotate_left(32),
                            usize_to_u64(game_index),
                        ));
                        dashboard::game_started(game_index);
                        let game = play_self_play_game_with_progress(
                            evaluator,
                            ruleset,
                            SelfPlayConfig {
                                search,
                                max_actions: arguments.max_actions,
                                ..SelfPlayConfig::default()
                            },
                            &mut rng,
                            |actions| dashboard::game_progress(game_index, actions),
                        );
                        dashboard::game_finished(game_index, game.steps.len(), game.outcome);
                        {
                            let mut finished = completed
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            *finished += 1;
                            let result = game.outcome.map_or_else(
                                || "draw · action limit".to_owned(),
                                |outcome| {
                                    format!("{:?} won · {:?}", outcome.winner, outcome.reason)
                                },
                            );
                            report(format!(
                                "  [{finished}/{}] game {} · {ruleset} · {} · {result}",
                                arguments.games,
                                game_index + 1,
                                counted(game.steps.len(), "position"),
                            ));
                        }
                        GeneratedGame { game }
                    })
                    .collect()
            })
        },
    )
}

impl PolicyValueEvaluator for BatchingEvaluator {
    fn predict(&self, position: &EncodedPosition) -> Prediction {
        self.predict_batch(std::slice::from_ref(position))
            .pop()
            .expect("single-position inference returns one result")
    }

    fn predict_batch(&self, positions: &[EncodedPosition]) -> Vec<Prediction> {
        if positions.is_empty() {
            return Vec::new();
        }
        let mut predictions = Vec::with_capacity(positions.len());
        let mut start = 0;
        while start < positions.len() {
            let mut end = start + 1;
            let single =
                estimate_inference_batch_memory(&positions[start..end], self.network_config);
            assert!(
                single.bytes <= self.memory_budget_bytes,
                "one inference position exceeds the configured memory budget: boards={}, actions={}, estimated={} MiB, budget={} MiB",
                single.max_boards,
                single.max_actions,
                bytes_to_mib_ceil(single.bytes),
                bytes_to_mib_ceil(self.memory_budget_bytes)
            );
            while end < positions.len()
                && estimate_inference_batch_memory(&positions[start..=end], self.network_config)
                    .bytes
                    <= self.memory_budget_bytes
            {
                end += 1;
            }
            let (response, results) = mpsc::sync_channel(1);
            self.requests
                .send(InferenceRequest {
                    queued: Instant::now(),
                    positions: positions[start..end].to_vec(),
                    response,
                })
                .expect("inference coordinator remains available");
            predictions.extend(
                results
                    .recv()
                    .expect("inference coordinator returns a result"),
            );
            start = end;
        }
        predictions
    }
}

fn with_batched_evaluator<E: PolicyValueEvaluator + ?Sized, R>(
    model: &E,
    target_batch_size: usize,
    network_config: NetworkConfig,
    memory_budget_bytes: usize,
    operation: impl FnOnce(&BatchingEvaluator) -> R,
) -> R {
    std::thread::scope(|scope| {
        let (requests, receiver) = mpsc::sync_channel(256);
        let evaluator = BatchingEvaluator {
            requests,
            network_config,
            memory_budget_bytes,
        };
        let coordinator = scope.spawn(move || {
            run_inference_coordinator(
                model,
                &receiver,
                target_batch_size.max(1),
                network_config,
                memory_budget_bytes,
            );
        });
        let result = operation(&evaluator);
        drop(evaluator);
        coordinator
            .join()
            .expect("inference coordinator must not panic");
        result
    })
}

fn run_inference_coordinator<E: PolicyValueEvaluator + ?Sized>(
    model: &E,
    receiver: &Receiver<InferenceRequest>,
    target_batch_size: usize,
    network_config: NetworkConfig,
    memory_budget_bytes: usize,
) {
    let mut pending = None;
    let mut batches = 0usize;
    let mut total_positions = 0usize;
    let mut total_requests = 0u32;
    let mut wait = Duration::ZERO;
    let mut compute = Duration::ZERO;
    loop {
        let first = match pending.take() {
            Some(request) => request,
            None => match receiver.recv() {
                Ok(request) => request,
                Err(_) => break,
            },
        };
        let mut count = first.positions.len();
        let (mut max_boards, mut max_actions) = request_dimensions(&first);
        let mut requests = vec![first];
        while count < target_batch_size {
            match receiver.recv_timeout(Duration::from_micros(200)) {
                Ok(request) => {
                    let (request_boards, request_actions) = request_dimensions(&request);
                    let candidate_count = count.saturating_add(request.positions.len());
                    let candidate_boards = max_boards.max(request_boards);
                    let candidate_actions = max_actions.max(request_actions);
                    let estimate = estimate_batch_memory_from_shape(
                        candidate_count,
                        candidate_boards,
                        candidate_actions,
                        network_config,
                        false,
                    );
                    if estimate.bytes > memory_budget_bytes {
                        pending = Some(request);
                        break;
                    }
                    count = candidate_count;
                    max_boards = candidate_boards;
                    max_actions = candidate_actions;
                    requests.push(request);
                }
                Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }
        let mut positions = Vec::with_capacity(count);
        let mut responses = Vec::with_capacity(requests.len());
        batches += 1;
        total_positions += count;
        let mut batch_wait = Duration::ZERO;
        let batch_requests = requests.len();
        for request in requests {
            total_requests += 1;
            batch_wait += request.queued.elapsed();
            responses.push((request.response, request.positions.len()));
            positions.extend(request.positions);
        }
        wait += batch_wait;
        let start = Instant::now();
        let mut predictions = model.predict_batch(&positions).into_iter();
        let batch_compute = start.elapsed();
        compute += batch_compute;
        dashboard::inference_batch(
            count,
            batch_requests,
            target_batch_size,
            batch_wait,
            batch_compute,
        );
        for (response, length) in responses {
            let result = predictions.by_ref().take(length).collect::<Vec<_>>();
            response
                .send(result)
                .expect("inference requester remains available");
        }
        assert!(predictions.next().is_none(), "inference result count");
    }
    if batches > 0 {
        report(format!(
            "  Inference: {total_positions} positions / {batches} batches (target {target_batch_size}), mean queue wait {:?}, compute {}",
            wait / total_requests,
            format_duration(compute)
        ));
    }
}

fn request_dimensions(request: &InferenceRequest) -> (usize, usize) {
    request
        .positions
        .iter()
        .fold((1, 1), |(boards, actions), position| {
            (
                boards.max(position.boards.len()),
                actions.max(position.actions.len()),
            )
        })
}

/// Rejects argument combinations that parse but cannot run.
fn validate(arguments: &Arguments) -> Result<(), String> {
    if arguments.threads == 0 {
        return Err(String::from("--threads must be greater than zero"));
    }
    if arguments.mcts_batch_size == 0 || arguments.inference_batch_size == 0 {
        return Err(String::from(
            "--mcts-batch-size and --inference-batch-size must be greater than zero",
        ));
    }
    if arguments.inference_memory_budget_mib == 0
        || arguments.training_memory_budget_mib == 0
        || arguments.reconstruction_memory_budget_mib == 0
        || arguments.training_examples == 0
    {
        return Err(String::from(
            "memory budgets and --training-examples must be greater than zero",
        ));
    }
    if arguments.simulations == 0
        || arguments.arena_simulations == 0
        || arguments.max_actions == 0
        || arguments.arena_max_actions == 0
    {
        return Err(String::from(
            "search simulations and action limits must be greater than zero",
        ));
    }
    if !arguments.promotion_score.is_finite() || !(0.0..=1.0).contains(&arguments.promotion_score) {
        return Err(String::from(
            "--promotion-score must be finite and between zero and one",
        ));
    }
    Ok(())
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

const fn mib_to_bytes(mib: usize) -> usize {
    mib.saturating_mul(1024 * 1024)
}

const fn bytes_to_mib_ceil(bytes: usize) -> usize {
    bytes.saturating_add(1024 * 1024 - 1) / (1024 * 1024)
}

const fn derived_seed(base: u64, phase: u64, sequence: u64, index: u64) -> u64 {
    splitmix64(base ^ phase ^ sequence.rotate_left(17) ^ index.rotate_left(41))
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

const fn selected_ruleset(selection: RuleSelection, index: usize) -> Ruleset {
    match selection {
        RuleSelection::Standard => Ruleset::Standard,
        RuleSelection::Multiverse => Ruleset::Multiverse,
        RuleSelection::Variant => Ruleset::MultiverseVariant,
        RuleSelection::All => match index % 3 {
            0 => Ruleset::Standard,
            1 => Ruleset::Multiverse,
            _ => Ruleset::MultiverseVariant,
        },
    }
}

const fn ruleset_game_index(selection: RuleSelection, game_index: usize) -> usize {
    match selection {
        RuleSelection::Standard | RuleSelection::Multiverse | RuleSelection::Variant => game_index,
        RuleSelection::All => game_index / 3,
    }
}

/// Index of the colour-reversed pair within its ruleset; both games share one opening.
const fn arena_pair(selection: RuleSelection, game_index: usize) -> usize {
    ruleset_game_index(selection, game_index) / 2
}

const fn ruleset_tag(ruleset: Ruleset) -> u64 {
    match ruleset {
        Ruleset::Standard => 0,
        Ruleset::Multiverse => 1,
        Ruleset::MultiverseVariant => 2,
    }
}

/// Plays up to `actions` uniformly random legal actions, stopping before any that ends the game.
fn arena_opening(ruleset: Ruleset, actions: usize, seed: u64) -> Game {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut game = Game::new(ruleset);
    for _ in 0..actions {
        let legal = game.legal_actions();
        if legal.is_empty() {
            break;
        }
        let mut next = game.clone();
        if next
            .apply_action(legal[rng.random_range(0..legal.len())])
            .is_err()
            || next.outcome().is_some()
        {
            break;
        }
        game = next;
    }
    game
}

const fn candidate_side(selection: RuleSelection, game_index: usize) -> Side {
    if ruleset_game_index(selection, game_index).is_multiple_of(2) {
        Side::White
    } else {
        Side::Black
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use chronofish_alphazero::{encode, play_self_play_game};
    use chronofish_core::Game;
    use clap::Parser;

    use super::*;

    #[derive(Default)]
    struct CountingEvaluator(AtomicUsize);

    impl PolicyValueEvaluator for CountingEvaluator {
        fn predict(&self, position: &EncodedPosition) -> Prediction {
            self.predict_batch(std::slice::from_ref(position))
                .pop()
                .unwrap()
        }

        fn predict_batch(&self, positions: &[EncodedPosition]) -> Vec<Prediction> {
            self.0.fetch_max(positions.len(), Ordering::Relaxed);
            positions
                .iter()
                .map(|position| Prediction {
                    policy: vec![1.0; position.actions.len()],
                    value: 0.0,
                })
                .collect()
        }
    }

    #[test]
    fn command_line_defaults_to_all_rulesets() {
        let arguments = Arguments::try_parse_from(["trainer"]).expect("arguments");
        assert!(matches!(arguments.ruleset, RuleSelection::All));
        assert_eq!(arguments.threads, default_threads());
        assert_eq!(arguments.iterations, 1);
        assert!(matches!(arguments.training_device, TrainingDevice::Cpu));
        assert_eq!(arguments.inference_device, InferenceDevice::Auto);
        assert_eq!(resolved_inference_device(&arguments), InferenceDevice::Cpu);
        assert_eq!(arguments.gpu_index, 0);
        assert!(!arguments.check_training_device);
        assert!(!arguments.verbose);
        assert_eq!(arguments.model_size, ModelSizeArgument::Auto);
        assert_eq!(resolved_model_size(&arguments), ModelSize::Large);
        assert_eq!(arguments.batch_token_budget, 262_144);
        assert_eq!(arguments.mcts_batch_size, 8);
        assert_eq!(arguments.inference_batch_size, 64);
        assert_eq!(arguments.inference_memory_budget_mib, 1024);
        assert_eq!(arguments.training_memory_budget_mib, 4096);
        assert_eq!(arguments.reconstruction_memory_budget_mib, 4096);
        assert_eq!(arguments.training_examples, 20_000);
    }

    #[test]
    fn rejected_candidate_keeps_learning_across_restarts() {
        let mut rng = ChaCha8Rng::seed_from_u64(73);
        let best = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let game = play_self_play_game(
            &best,
            Ruleset::Standard,
            SelfPlayConfig {
                search: SearchConfig {
                    simulations: 1,
                    ..SearchConfig::default()
                },
                max_actions: 1,
                ..SelfPlayConfig::default()
            },
            &mut rng,
        );
        let mut replay = ReplayBuffer::default();
        replay.extend_game(game, 10);
        let examples = replay.examples().unwrap();
        let config = TrainConfig {
            epochs: 1,
            ..TrainConfig::default()
        };
        let mut candidate = best.clone();
        candidate.train(&examples, config, &mut rng).unwrap();
        let path =
            std::env::temp_dir().join(format!("chronofish-learner-{}.json", std::process::id()));
        candidate.save(&path).unwrap();
        let best_path = path.with_extension("best.json");
        let mut incumbent = best.clone();
        let mut arguments = Arguments::try_parse_from([
            "trainer",
            "--ruleset",
            "standard",
            "--arena-games",
            "2",
            "--arena-simulations",
            "1",
            "--arena-max-actions",
            "1",
            "--promotion-score",
            "0.55",
        ])
        .unwrap();
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        evaluate_and_maybe_promote(
            &candidate,
            &mut incumbent,
            &best_path,
            &arguments,
            SearchConfig::default(),
            &pool,
            None,
        )
        .unwrap();
        assert_eq!(incumbent.training_steps(), best.training_steps());
        assert!(!best_path.exists());
        arguments.promotion_score = 0.5;
        evaluate_and_maybe_promote(
            &candidate,
            &mut incumbent,
            &best_path,
            &arguments,
            SearchConfig::default(),
            &pool,
            None,
        )
        .unwrap();
        assert_eq!(
            PolicyValueNetwork::load(&best_path)
                .unwrap()
                .training_steps(),
            candidate.training_steps()
        );
        std::fs::remove_file(best_path).unwrap();
        let mut resumed = resume_candidate(&path, &best, false).unwrap();
        assert_eq!(resumed.training_steps(), candidate.training_steps());
        assert!(resumed.training_steps() > best.training_steps());
        resumed.train(&examples, config, &mut rng).unwrap();
        assert!(resumed.training_steps() > candidate.training_steps());
        assert_eq!(
            resume_candidate(&path, &best, true)
                .unwrap()
                .training_steps(),
            best.training_steps()
        );
        // An older learner must not roll back a more recently published incumbent.
        assert_eq!(
            resume_candidate(&path, &resumed, false)
                .unwrap()
                .training_steps(),
            resumed.training_steps()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn default_arena_accepts_ties_and_balances_every_ruleset() {
        let args = Arguments::try_parse_from(["trainer"]).unwrap();
        assert!((args.promotion_score - 0.5).abs() < f32::EPSILON);
        assert_eq!(args.arena_games, 6);
        assert_eq!(args.arena_simulations, 16);
        assert_eq!(args.arena_max_actions, 160);
        for ruleset in [
            Ruleset::Standard,
            Ruleset::Multiverse,
            Ruleset::MultiverseVariant,
        ] {
            for side in [Side::White, Side::Black] {
                assert_eq!(
                    (0..args.arena_games)
                        .filter(|&i| selected_ruleset(args.ruleset, i) == ruleset
                            && candidate_side(args.ruleset, i) == side)
                        .count(),
                    1
                );
            }
        }
    }

    #[test]
    fn display_helpers_format_counts_and_durations_for_people() {
        assert_eq!(format_count(20_000), "20,000");
        assert_eq!(counted(1, "game"), "1 game");
        assert_eq!(counted(8, "game"), "8 games");
        assert_eq!(format_duration(Duration::from_millis(1_970)), "2.0s");
        assert_eq!(format_duration(Duration::from_secs(159)), "2m 39.0s");
        assert_eq!(format_duration(Duration::from_mins(125)), "2h 5m");
    }

    #[test]
    fn vulkan_device_and_index_are_parsed_without_initializing_hardware() {
        let arguments = Arguments::try_parse_from([
            "trainer",
            "--training-device",
            "vulkan",
            "--gpu-index",
            "2",
            "--check-training-device",
        ])
        .expect("arguments");
        assert!(matches!(arguments.training_device, TrainingDevice::Vulkan));
        assert_eq!(resolved_model_size(&arguments), ModelSize::Balanced);
        assert_eq!(
            resolved_inference_device(&arguments),
            InferenceDevice::Vulkan
        );
        assert_eq!(arguments.gpu_index, 2);
        assert!(arguments.check_training_device);
    }

    #[test]
    fn explicit_model_size_overrides_device_default() {
        let arguments = Arguments::try_parse_from([
            "trainer",
            "--training-device",
            "vulkan",
            "--model-size",
            "large",
        ])
        .expect("arguments");

        assert_eq!(resolved_model_size(&arguments), ModelSize::Large);
    }

    #[test]
    fn obsolete_hidden_option_is_rejected() {
        let error = Arguments::try_parse_from(["trainer", "--hidden", "48"])
            .expect_err("old flat-network option must not be accepted");
        assert!(error.to_string().contains("unexpected argument '--hidden'"));
    }

    #[test]
    fn inference_coordinator_merges_queued_game_requests() {
        let game = Game::new(Ruleset::Standard);
        let position = encode(&game, &game.legal_actions());
        let (requests, receiver) = mpsc::sync_channel(8);
        let mut results = Vec::new();
        for _ in 0..3 {
            let (response, result) = mpsc::sync_channel(1);
            requests
                .send(InferenceRequest {
                    queued: Instant::now(),
                    positions: vec![position.clone()],
                    response,
                })
                .unwrap();
            results.push(result);
        }
        drop(requests);
        let evaluator = CountingEvaluator::default();

        run_inference_coordinator(
            &evaluator,
            &receiver,
            64,
            NetworkConfig::tiny(),
            mib_to_bytes(64),
        );

        assert_eq!(evaluator.0.load(Ordering::Relaxed), 3);
        assert!(
            results
                .into_iter()
                .all(|result| result.recv().unwrap().len() == 1)
        );
    }

    #[test]
    fn inference_memory_budget_splits_one_large_request() {
        let game = Game::new(Ruleset::Standard);
        let position = encode(&game, &game.legal_actions());
        let config = NetworkConfig::tiny();
        let one = estimate_inference_batch_memory(std::slice::from_ref(&position), config);
        let evaluator = CountingEvaluator::default();

        let predictions = with_batched_evaluator(&evaluator, 64, config, one.bytes, |batched| {
            batched.predict_batch(&[position.clone(), position.clone(), position])
        });

        assert_eq!(predictions.len(), 3);
        assert_eq!(evaluator.0.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn arena_pairs_share_an_opening_and_pairs_differ() {
        let opening = |selection, game_index| {
            let ruleset = selected_ruleset(selection, game_index);
            let pair = arena_pair(selection, game_index);
            let game = arena_opening(
                ruleset,
                8,
                derived_seed(
                    1,
                    ARENA_OPENING_SEED,
                    0,
                    usize_to_u64(pair) | (ruleset_tag(ruleset) << 32),
                ),
            );
            assert!(game.outcome().is_none());
            format!("{:?}", game.timelines())
        };
        for selection in [RuleSelection::Standard, RuleSelection::All] {
            let games = if matches!(selection, RuleSelection::All) {
                12
            } else {
                4
            };
            let openings = (0..games)
                .map(|index| opening(selection, index))
                .collect::<Vec<_>>();
            for first in 0..games {
                for second in first + 1..games {
                    let same_ruleset =
                        selected_ruleset(selection, first) == selected_ruleset(selection, second);
                    let same_pair = same_ruleset
                        && arena_pair(selection, first) == arena_pair(selection, second);
                    assert_eq!(
                        openings[first] == openings[second],
                        same_pair,
                        "{first} {second}"
                    );
                    if same_pair {
                        assert_ne!(
                            candidate_side(selection, first),
                            candidate_side(selection, second)
                        );
                    }
                }
            }
        }
        let start = format!("{:?}", Game::new(Ruleset::Standard).timelines());
        assert_eq!(
            format!("{:?}", arena_opening(Ruleset::Standard, 0, 9).timelines()),
            start
        );
    }

    #[test]
    fn mixed_arena_alternates_colours_within_each_ruleset() {
        let sides = (0..8)
            .map(|index| candidate_side(RuleSelection::All, index))
            .collect::<Vec<_>>();
        assert_eq!(
            sides,
            vec![
                Side::White,
                Side::White,
                Side::White,
                Side::Black,
                Side::Black,
                Side::Black,
                Side::White,
                Side::White,
            ]
        );
    }
}
