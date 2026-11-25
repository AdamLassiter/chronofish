use std::path::Path;

use chronofish_core::{AiDecision, AiPlayer, AiProfile, Game};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::encoding::encode;
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

/// Chronofish: `AlphaZero` MCTS guided by the trained policy-value network.
///
/// Training may run on the CPU or GPU backend; the bot always runs inference on the
/// CPU. A checkpoint carries its own architecture, so any trained network loads.
pub struct ChronofishBot(AlphaZeroBot);

impl ChronofishBot {
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

    /// Creates an untrained network with the default training architecture, useful
    /// only as a fallback.
    #[must_use]
    pub fn bootstrap(search: SearchConfig, seed: u64) -> Self {
        Self(AlphaZeroBot::bootstrap(
            search,
            seed,
            NetworkConfig::large(),
        ))
    }

    #[cfg(test)]
    fn network_config(&self) -> NetworkConfig {
        self.0.model.config()
    }
}

impl AiPlayer for ChronofishBot {
    fn profile(&self) -> AiProfile {
        AiProfile {
            username: "bot-chronofish",
            display_name: "Chronofish",
        }
    }

    fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String> {
        self.0.choose_action(game)
    }
}

/// `AlphaNow`: trusts the network's first instinct and never looks ahead.
///
/// Plays the highest-prior action of the policy head with no tree search, isolating
/// how much of the `AlphaZero` bots' strength comes from search rather than the network.
pub struct AlphaNowBot {
    model: PolicyValueNetwork,
}

impl AlphaNowBot {
    #[must_use]
    pub const fn new(model: PolicyValueNetwork) -> Self {
        Self { model }
    }

    /// Loads a trained checkpoint for inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint cannot be read or decoded.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        Ok(Self::new(PolicyValueNetwork::load(path)?))
    }

    /// Creates an untrained network, useful only as a fallback.
    #[must_use]
    pub fn bootstrap(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Self::new(PolicyValueNetwork::random(NetworkConfig::large(), &mut rng))
    }
}

impl AiPlayer for AlphaNowBot {
    fn profile(&self) -> AiProfile {
        AiProfile {
            username: "bot-alpha-now",
            display_name: "AlphaNow",
        }
    }

    fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String> {
        let actions = game.legal_actions();
        if actions.is_empty() {
            return Err("no legal action is available".to_owned());
        }
        let prediction = self.model.predict(&encode(game, &actions));
        let best = prediction
            .policy
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map_or(0, |(index, _)| index);
        let action = actions[best.min(actions.len() - 1)];
        Ok(AiDecision {
            action,
            principal_variation: vec![action],
            root_value: Some(prediction.value),
        })
    }
}

#[cfg(test)]
mod tests {
    use chronofish_core::{Game, Ruleset};

    use super::*;

    #[test]
    fn chronofish_returns_authoritatively_legal_actions() {
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        let game = Game::new(Ruleset::Standard);
        let search = SearchConfig {
            simulations: 3,
            ..SearchConfig::default()
        };
        let model = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let mut bot = ChronofishBot::new(model, search, 31);
        assert_eq!(bot.profile().username, "bot-chronofish");
        let action = bot.choose_action(&game).expect("bot action");
        let mut next = game.clone();
        next.apply_action(action.action).expect("legal bot action");
    }

    #[test]
    fn alpha_now_plays_a_legal_policy_action_in_every_ruleset() {
        let mut rng = ChaCha8Rng::seed_from_u64(43);
        let mut bot = AlphaNowBot::new(PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng));
        assert_eq!(bot.profile().username, "bot-alpha-now");
        for ruleset in [
            Ruleset::Standard,
            Ruleset::Multiverse,
            Ruleset::MultiverseVariant,
        ] {
            let game = Game::new(ruleset);
            let decision = bot.choose_action(&game).expect("bot action");
            let mut next = game.clone();
            next.apply_action(decision.action)
                .expect("legal policy action");
        }
    }

    #[test]
    fn bootstrap_uses_the_default_training_architecture() {
        assert_eq!(
            ChronofishBot::bootstrap(SearchConfig::default(), 31).network_config(),
            NetworkConfig::large()
        );
    }
}
