mod api;
mod db;
mod web;

use std::path::Path;

use axum::Router;
use chronofish_alphazero::{
    AlphaNowBot, ChronofishCpuBot, ChronofishGpuBot, PolicyValueNetwork, SearchConfig,
};
use chronofish_classic::{
    BlueshiftConfig, CarlsenConfig, DeepBlueshift, Grabablanca, MonteCarlsen,
};
use chronofish_random_bot::BobbyFissureBot;
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
        let blueshift = DeepBlueshift::new(BlueshiftConfig {
            max_nodes: env_parse("CHRONOFISH_BLUESHIFT_NODES")?
                .unwrap_or(BlueshiftConfig::default().max_nodes),
            ..BlueshiftConfig::default()
        });
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
                Box::new(blueshift),
                Box::new(carlsen),
                Box::new(alpha_now),
                Box::new(muninn),
                Box::new(chronofish),
            ],
        )
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
