mod api;
mod db;
mod web;

use std::path::Path;

use axum::Router;
use chronofish_alphazero::{
    AlphaNowBot, ChronofishCpuBot, ChronofishGpuBot, PolicyValueNetwork, SearchConfig,
};
use chronofish_classic::{
    BlueshiftLevel, BobbyFissureBot, CarlsenConfig, DeepBlueshift, Grabablanca, MonteCarlsen,
};
use db::Database;

pub use api::AppState;

impl AppState {
    /// Opens the `SQLite` database, applies migrations, and registers built-in bots.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or initialized.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let database = Database::open(path)?;
        let search = SearchConfig {
            simulations: env_parse("CHRONOFISH_AZ_SIMULATIONS")?.unwrap_or(48),
            ..SearchConfig::default()
        };
        let cpu_network = load_network(
            std::env::var_os("CHRONOFISH_AZ_MODEL"),
            "models/training/best-v2.json",
            "Chronofish CPU and AlphaNow",
        )?;
        let muninn = cpu_network.clone().map_or_else(
            || ChronofishCpuBot::bootstrap(search, 0x4d55_4e49_4e4e),
            |network| ChronofishCpuBot::new(network, search, 0x4d55_4e49_4e4e),
        );
        let alpha_now = cpu_network.map_or_else(
            || AlphaNowBot::bootstrap(0x4e4f_5741_4c50_4841),
            AlphaNowBot::new,
        );
        let chronofish = load_network(
            std::env::var_os("CHRONOFISH_GPU_MODEL"),
            "models/training-gpu/best-v2.json",
            "Chronofish GPU",
        )?
        .map_or_else(
            || ChronofishGpuBot::bootstrap(search, 0x4855_4749_4e4e),
            |network| ChronofishGpuBot::new(network, search, 0x4855_4749_4e4e),
        );
        let reference_policy = load_network(
            std::env::var_os("CHRONOFISH_REFERENCE_AZ_MODEL"),
            "models/training/best-reference-v3.json",
            "Chronofish CPU and AlphaNow (5D)",
        )?;
        let reference_now = reference_policy.clone().map_or_else(
            || AlphaNowBot::bootstrap(0x4e4f_5741_4c50_4841),
            AlphaNowBot::new,
        );
        let reference_muninn = reference_policy.map_or_else(
            || ChronofishCpuBot::bootstrap(search, 0x4d55_4e49_4e4e),
            |model| ChronofishCpuBot::new(model, search, 0x4d55_4e49_4e4e),
        );
        let reference_accelerated = load_network(
            std::env::var_os("CHRONOFISH_REFERENCE_GPU_MODEL"),
            "models/training-gpu/best-reference-v3.json",
            "Chronofish GPU (5D)",
        )?
        .map_or_else(
            || ChronofishGpuBot::bootstrap(search, 0x4855_4749_4e4e),
            |model| ChronofishGpuBot::new(model, search, 0x4855_4749_4e4e),
        );
        let blueshift = DeepBlueshift::level(BlueshiftLevel::Hard).with_max_nodes(
            env_parse("CHRONOFISH_BLUESHIFT_NODES")?.unwrap_or(BlueshiftLevel::Hard.max_nodes()),
        );
        let carlsen = MonteCarlsen::new(
            CarlsenConfig {
                iterations: env_parse("CHRONOFISH_CARLSEN_ITERATIONS")?
                    .unwrap_or(CarlsenConfig::default().iterations),
                ..CarlsenConfig::default()
            },
            0x4d41_474e_5553_4d43,
        );
        Self::new(
            database,
            vec![
                Box::new(BobbyFissureBot::default()),
                Box::new(Grabablanca::default()),
                Box::new(DeepBlueshift::level(BlueshiftLevel::Easy)),
                Box::new(DeepBlueshift::level(BlueshiftLevel::Medium)),
                Box::new(blueshift),
                Box::new(carlsen),
                Box::new(RulesetBot {
                    standard: alpha_now,
                    reference: reference_now,
                }),
                Box::new(RulesetBot {
                    standard: muninn,
                    reference: reference_muninn,
                }),
                Box::new(RulesetBot {
                    standard: chronofish,
                    reference: reference_accelerated,
                }),
            ],
        )
    }
}

struct RulesetBot<B> {
    standard: B,
    reference: B,
}

impl<B: chronofish_core::AiPlayer> chronofish_core::AiPlayer for RulesetBot<B> {
    fn profile(&self) -> chronofish_core::AiProfile {
        self.standard.profile()
    }

    fn choose_action(
        &mut self,
        game: &chronofish_core::Game,
    ) -> Result<chronofish_core::AiDecision, String> {
        if game.ruleset() == chronofish_core::Ruleset::Standard {
            self.standard.choose_action(game)
        } else {
            self.reference.choose_action(game)
        }
    }
}

/// Starting Elo for each bot account: the lower bound of its 95% interval from the
/// standard-chess calibration against Stockfish 1320/1600/2000
/// (`chronofish-benchmarks --tournament --calibrate`, 2026-11-25, one opening per pair).
/// Lower bounds keep new accounts from being overrated while the intervals are wide.
/// Uncalibrated bots (the Chronofish networks) start at the default 1500.
pub(crate) fn seed_rating(username: &str) -> i32 {
    match username {
        "bot-bobby-fissure" => 995,
        "bot-alpha-now" => 1034,
        "bot-grabablanca" => 1152,
        "bot-monte-carlsen" => 1159,
        "bot-deep-blueshift-easy" => 1263,
        "bot-deep-blueshift-medium" => 1416,
        "bot-deep-blueshift" => 1517,
        _ => 1500,
    }
}

fn env_parse<T: std::str::FromStr>(name: &str) -> Result<Option<T>, Box<dyn std::error::Error>>
where
    T::Err: std::error::Error + 'static,
{
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse())
        .transpose()?)
}

/// Loads a checkpoint from an explicit path or the default location, or `None` to bootstrap.
fn load_network(
    explicit: Option<std::ffi::OsString>,
    default_path: &str,
    bots: &str,
) -> Result<Option<PolicyValueNetwork>, Box<dyn std::error::Error>> {
    if let Some(path) = explicit {
        return Ok(Some(PolicyValueNetwork::load(path)?));
    }
    if Path::new(default_path).exists() {
        return Ok(Some(PolicyValueNetwork::load(default_path)?));
    }
    tracing::warn!("{default_path} is absent; {bots} use an untrained bootstrap network");
    Ok(None)
}

pub fn build_app(state: AppState) -> Router {
    api::router(state)
}
