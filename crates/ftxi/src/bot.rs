use chronofish_5d_rules::core::action::Action;
use chronofish_core::{AiDecision, AiPlayer, AiProfile, Game, ReferenceBotTurn, Ruleset};
use rand::{SeedableRng, rngs::StdRng};

use crate::{LeafEvaluation, search_until};

pub struct LokiBot {
    rng: StdRng,
    turn: ReferenceBotTurn,
}

impl Default for LokiBot {
    fn default() -> Self {
        Self::new(0x4c4f_4b49)
    }
}

impl LokiBot {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            turn: ReferenceBotTurn::default(),
        }
    }
}

impl AiPlayer for LokiBot {
    fn profile(&self) -> AiProfile {
        AiProfile {
            username: "bot-loki",
            display_name: "Loki",
        }
    }

    fn supports_ruleset(&self, ruleset: Ruleset) -> bool {
        ruleset != Ruleset::Standard
    }

    fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String> {
        self.turn.choose(game, |state| {
            let started = std::time::Instant::now();
            search_until(
                state,
                128,
                LeafEvaluation::ZeroCaptureCheckWidening,
                &mut self.rng,
                || started.elapsed() >= std::time::Duration::from_millis(500),
            )
            .map(|turn| Action::from_seq(&turn, state).moves().to_vec())
        })
    }
}
