//! Grabablanca: grabs whatever looks best right now, one action deep.

use chronofish_core::{AiDecision, AiPlayer, AiProfile, Game};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::eval::{relative, squash};

/// Greedy one-action bot over the handcrafted evaluation.
pub struct Grabablanca {
    rng: ChaCha8Rng,
}

impl Grabablanca {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl Default for Grabablanca {
    fn default() -> Self {
        Self::new(0x4341_5041_424c_4e43)
    }
}

impl AiPlayer for Grabablanca {
    fn profile(&self) -> AiProfile {
        AiProfile {
            username: "bot-grabablanca",
            display_name: "Grabablanca",
        }
    }

    fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String> {
        let mover = game.turn();
        let mut best = Vec::new();
        let mut best_score = i32::MIN;
        for action in game.legal_actions() {
            let mut child = game.clone();
            if child.apply_action(action).is_err() {
                continue;
            }
            let score = relative(&child, mover);
            if score > best_score {
                best_score = score;
                best.clear();
            }
            if score == best_score {
                best.push(action);
            }
        }
        if best.is_empty() {
            return Err("no legal action is available".to_owned());
        }
        let action = best[self.rng.random_range(0..best.len())];
        Ok(AiDecision {
            action,
            principal_variation: vec![action],
            root_value: Some(squash(best_score)),
        })
    }
}
