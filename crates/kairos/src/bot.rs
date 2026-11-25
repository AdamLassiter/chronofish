use std::sync::atomic::AtomicBool;

use chronofish_core::{AiDecision, AiPlayer, AiProfile, Game, ReferenceBotTurn, Ruleset};

use crate::engine::{Limits, ab::AlphaBeta};

#[derive(Default)]
pub struct KairosBot {
    engine: AlphaBeta,
    turn: ReferenceBotTurn,
}

impl AiPlayer for KairosBot {
    fn profile(&self) -> AiProfile {
        AiProfile {
            username: "bot-kairos",
            display_name: "Kairos",
        }
    }

    fn supports_ruleset(&self, ruleset: Ruleset) -> bool {
        ruleset != Ruleset::Standard
    }

    fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String> {
        self.turn.choose(game, |state| {
            self.engine
                .find_best_move(
                    state,
                    &Limits {
                        nodes: Some(20_000),
                        movetime: Some(500),
                        ..Limits::default()
                    },
                    &AtomicBool::new(false),
                    &mut |_| {},
                )
                .map(|action| action.moves().to_vec())
        })
    }
}
