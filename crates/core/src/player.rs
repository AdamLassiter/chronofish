use serde::{Deserialize, Serialize};

use crate::{Color, Game, GameOutcome, PlayerAction, Position, Ruleset, Timeline};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameView {
    pub ruleset: Ruleset,
    pub turn: Color,
    pub outcome: Option<GameOutcome>,
    pub message: String,
    pub timelines: Vec<Timeline>,
    pub legal_actions: Vec<PlayerAction>,
    pub can_submit: bool,
    pub has_staged_moves: bool,
    pub present_time: Option<i32>,
    pub active_timelines: Vec<i32>,
    pub checked_royals: Vec<Position>,
}

impl GameView {
    #[must_use]
    pub fn from_game(game: &Game) -> Self {
        Self {
            ruleset: game.ruleset(),
            turn: game.turn(),
            outcome: game.outcome(),
            message: game.message().to_owned(),
            timelines: game.timelines().to_vec(),
            legal_actions: game.legal_actions(),
            can_submit: game.can_submit(),
            has_staged_moves: game.has_staged_moves(),
            present_time: game.present_time(),
            active_timelines: game
                .timelines()
                .iter()
                .filter(|timeline| game.is_active_timeline(timeline.id))
                .map(|timeline| timeline.id)
                .collect(),
            checked_royals: game.checked_royal_positions(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AiProfile {
    pub username: &'static str,
    pub display_name: &'static str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiDecision {
    pub action: PlayerAction,
    #[serde(default)]
    pub principal_variation: Vec<PlayerAction>,
    pub root_value: Option<f32>,
}

pub trait AiPlayer: Send {
    fn profile(&self) -> AiProfile;
    fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String>;
}
