use std::path::Path;

use chronofish_core::{AiDecision, AiPlayer, AiProfile, Game};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::network::{ModelError, NetworkConfig, PolicyValueNetwork};
use crate::search::{Mcts, SearchConfig};

struct AlphaZeroBot {
    model: PolicyValueNetwork,
    search: SearchConfig,
    rng: ChaCha8Rng,
}

impl AlphaZeroBot {
    fn new(model: PolicyValueNetwork, search: SearchConfig, seed: u64) -> Self {
        Self {
            model,
            search,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    fn load(path: impl AsRef<Path>, search: SearchConfig, seed: u64) -> Result<Self, ModelError> {
        Ok(Self::new(PolicyValueNetwork::load(path)?, search, seed))
    }

    fn bootstrap(search: SearchConfig, seed: u64, config: NetworkConfig) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let model = PolicyValueNetwork::random(config, &mut rng);
        Self::new(model, search, seed ^ 0xa17e_20a0)
    }

    fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String> {
        let result = Mcts::new(&self.model, self.search).search(game, false, &mut self.rng);
        let action = result
            .best_action()
            .ok_or_else(|| "no legal action is available".to_owned())?;
        Ok(AiDecision {
            action,
            principal_variation: result.principal_variation,
            root_value: Some(result.root_value),
        })
    }
}

macro_rules! define_alpha_zero_bot {
    ($name:ident, $username:literal, $display_name:literal, $bootstrap_config:expr) => {
        pub struct $name(AlphaZeroBot);

        impl $name {
            #[must_use]
            pub fn new(model: PolicyValueNetwork, search: SearchConfig, seed: u64) -> Self {
                Self(AlphaZeroBot::new(model, search, seed))
            }

            /// Loads a trained checkpoint for inference.
            ///
            /// # Errors
            ///
            /// Returns an error if the checkpoint cannot be read, decoded, or does not
            /// match the supported network format.
            pub fn load(
                path: impl AsRef<Path>,
                search: SearchConfig,
                seed: u64,
            ) -> Result<Self, ModelError> {
                Ok(Self(AlphaZeroBot::load(path, search, seed)?))
            }

            /// Creates an untrained network, useful only as a fallback.
            #[must_use]
            pub fn bootstrap(search: SearchConfig, seed: u64) -> Self {
                Self::bootstrap_with_config(search, seed, $bootstrap_config)
            }

            #[doc(hidden)]
            #[must_use]
            pub fn bootstrap_with_config(
                search: SearchConfig,
                seed: u64,
                config: NetworkConfig,
            ) -> Self {
                Self(AlphaZeroBot::bootstrap(search, seed, config))
            }

            #[cfg(test)]
            fn network_config(&self) -> NetworkConfig {
                self.0.model.config()
            }
        }

        impl AiPlayer for $name {
            fn profile(&self) -> AiProfile {
                AiProfile {
                    username: $username,
                    display_name: $display_name,
                }
            }

            fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String> {
                self.0.choose_action(game)
            }
        }
    };
}

define_alpha_zero_bot!(
    ChronofishCpuBot,
    "bot-chronofish-cpu",
    "Chronofish CPU",
    NetworkConfig::large()
);
define_alpha_zero_bot!(
    ChronofishGpuBot,
    "bot-chronofish-gpu",
    "Chronofish GPU",
    NetworkConfig::balanced()
);

#[cfg(test)]
mod tests {
    use chronofish_core::{Game, Ruleset};

    use super::*;

    #[test]
    fn both_named_models_return_authoritatively_legal_actions() {
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        let game = Game::new(Ruleset::Standard);
        let search = SearchConfig {
            simulations: 3,
            ..SearchConfig::default()
        };
        let model = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let bots: Vec<Box<dyn AiPlayer>> = vec![
            Box::new(ChronofishCpuBot::new(model.clone(), search, 31)),
            Box::new(ChronofishGpuBot::new(model, search, 37)),
        ];
        let profiles = bots.iter().map(|bot| bot.profile()).collect::<Vec<_>>();
        assert_eq!(profiles[0].username, "bot-chronofish-cpu");
        assert_eq!(profiles[1].username, "bot-chronofish-gpu");
        for mut bot in bots {
            let action = bot.choose_action(&game).expect("bot action");
            let mut next = game.clone();
            next.apply_action(action.action).expect("legal bot action");
        }
    }

    #[test]
    fn named_bootstraps_use_their_training_architectures() {
        let search = SearchConfig::default();

        assert_eq!(
            ChronofishCpuBot::bootstrap(search, 31).network_config(),
            NetworkConfig::large()
        );
        assert_eq!(
            ChronofishGpuBot::bootstrap(search, 37).network_config(),
            NetworkConfig::balanced()
        );
    }
}
