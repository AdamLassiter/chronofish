mod api;
mod db;
mod web;

use std::path::Path;

use axum::Router;
use chronofish_alphazero::{ChronofishCpuBot, ChronofishGpuBot, SearchConfig};
use chronofish_random_bot::RavenBot;
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
            simulations: std::env::var("CHRONOFISH_AZ_SIMULATIONS")
                .ok()
                .map(|value| value.parse())
                .transpose()?
                .unwrap_or(48),
            ..SearchConfig::default()
        };
        let muninn = if let Some(path) = std::env::var_os("CHRONOFISH_AZ_MODEL") {
            ChronofishCpuBot::load(path, search, 0x4d55_4e49_4e4e)?
        } else {
            let default_path = Path::new("models/training/best-v2.json");
            if default_path.exists() {
                ChronofishCpuBot::load(default_path, search, 0x4d55_4e49_4e4e)?
            } else {
                tracing::warn!(
                    "models/training/best-v2.json is absent; Chronofish CPU is using an untrained bootstrap network"
                );
                ChronofishCpuBot::bootstrap(search, 0x4d55_4e49_4e4e)
            }
        };
        let chronofish = if let Some(path) = std::env::var_os("CHRONOFISH_GPU_MODEL") {
            ChronofishGpuBot::load(path, search, 0x4855_4749_4e4e)?
        } else {
            let default_path = Path::new("models/training-gpu/best-v2.json");
            if default_path.exists() {
                ChronofishGpuBot::load(default_path, search, 0x4855_4749_4e4e)?
            } else {
                tracing::warn!(
                    "models/training-gpu/best-v2.json is absent; Chronofish GPU is using an untrained bootstrap network"
                );
                ChronofishGpuBot::bootstrap(search, 0x4855_4749_4e4e)
            }
        };
        Self::new(
            database,
            vec![
                Box::new(RavenBot::default()),
                Box::new(muninn),
                Box::new(chronofish),
            ],
        )
    }
}

pub fn build_app(state: AppState) -> Router {
    api::router(state)
}
