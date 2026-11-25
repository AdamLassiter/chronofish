use chronofish_core::{Game, PlayerAction, Position};

/// Formats a move in 5DUCI long algebraic notation. This maps Chronofish's
/// initial half-turn to T1. Moves within a board omit the destination board.
#[must_use]
pub fn format_lan(action: PlayerAction) -> String {
    let PlayerAction::Move { movement } = action else {
        return "submit".into();
    };
    let board = |position: Position| {
        format!(
            "({}T{})",
            position.timeline_id,
            position.time.div_euclid(2) + 1
        )
    };
    let square =
        |position: Position| format!("{}{}", char::from(b'a' + position.x as u8), position.y + 1);
    let mut text = board(movement.from);
    text.push_str(&square(movement.from));
    if movement.from.timeline_id != movement.to.timeline_id
        || movement.from.time != movement.to.time
    {
        text.push_str(&board(movement.to));
    }
    text.push_str(&square(movement.to));
    if let Some(promotion) = movement.promotion {
        text.push(promotion.notation_letter());
    }
    text
}

/// Replays whitespace-separated 5DUCI moves according to Chronofish rules.
///
/// # Errors
///
/// Returns the token number and text if a move is illegal or ambiguous.
pub fn replay_lan(initial: &Game, moves: &str) -> Result<Game, String> {
    decode_lan(initial, moves).map(|(game, _)| game)
}

pub fn decode_lan(initial: &Game, moves: &str) -> Result<(Game, Vec<PlayerAction>), String> {
    let mut game = initial.clone();
    let mut actions = Vec::new();
    for (index, token) in moves.split_whitespace().enumerate() {
        let mut matches = game.legal_actions().into_iter().filter(|&action| {
            let formatted = format_lan(action);
            if token == formatted {
                return true;
            }
            let PlayerAction::Move { movement } = action else {
                return false;
            };
            if movement.from.timeline_id != movement.to.timeline_id
                || movement.from.time != movement.to.time
            {
                return false;
            }
            let prefix_length = formatted
                .find(')')
                .expect("LAN board ends with a parenthesis")
                + 1;
            let (board, squares) = formatted.split_at(prefix_length);
            token == format!("{board}{}{board}{}", &squares[..2], &squares[2..])
        });
        let error = || format!("illegal or ambiguous LAN token {}: {token}", index + 1);
        let action = matches.next().ok_or_else(error)?;
        if matches.next().is_some() {
            return Err(error());
        }
        game.apply_action(action)
            .map_err(|err| format!("LAN token {}: {err}", index + 1))?;
        actions.push(action);
    }
    Ok((game, actions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronofish_core::{Color, Ruleset};

    #[test]
    fn replay_rejects_exposing_a_royal() {
        let initial = Game::new(Ruleset::Multiverse);
        let history =
            "(0T1)e2(0T1)e4 submit (0T1)d7d5 submit (0T2)f1b5 submit (0T2)a7a6 submit (0T3)b5e8";
        assert!(replay_lan(&initial, history).is_err());
        assert_eq!(initial.turn(), Color::White);
        assert!(replay_lan(&initial, "(0T1)e2e5").is_err());
    }

    #[test]
    fn replays_a_temporal_branch() {
        let initial = Game::new(Ruleset::Multiverse);
        let history = "(0T1)g1f3 submit (0T1)g8f6 submit (0T2)f3(0T1)d3 submit";
        let game = replay_lan(&initial, history).unwrap();
        assert_eq!(game.timelines().len(), 2);
    }
}
