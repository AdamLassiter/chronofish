use crate::{AiDecision, Game, Ruleset};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AiProfile {
    pub username: &'static str,
    pub display_name: &'static str,
}

pub trait AiPlayer: Send {
    fn profile(&self) -> AiProfile;
    fn supports_ruleset(&self, _ruleset: Ruleset) -> bool {
        true
    }
    fn requires_reference_rules(&self) -> bool {
        false
    }
    /// Choose the next action in the current position.
    ///
    /// # Errors
    /// Returns an error when the bot cannot choose an action.
    fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String>;
}
