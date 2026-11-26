use crate::{decode_lan, pgn::parse_5dpgn};
use chronofish_5d_turngen::core::variants::state_from_setup;
use chronofish_core::{Game, PlayerAction};

pub fn replay_5dpgn(initial: &Game, text: &str) -> Result<(Game, Vec<PlayerAction>), String> {
    let standard = chronofish_core::reference_setup(initial.ruleset())
        .ok_or("5DPGN replay requires a 5D mode")?;
    let parsed = parse_5dpgn(text, Some(&standard))?;
    let reference = state_from_setup(&standard)?;
    let origin = state_from_setup(&parsed.setup)?;
    if parsed.setup.even
        || origin.get_board_size() != reference.get_board_size()
        || origin.get_boards() != reference.get_boards()
        || origin.get_present() != reference.get_present()
        || origin.get_promotion_options() != reference.get_promotion_options()
    {
        return Err(
            "5DPGN replay requires the selected 5D starting setup and promotion rules".into(),
        );
    }

    let mut game = initial.clone();
    let mut actions = Vec::new();
    for (index, action) in parsed.root.mainline().into_iter().enumerate() {
        let (next, moves) = decode_lan(&game, &action.to_string())
            .map_err(|error| format!("5DPGN turn {}: {error}", index + 1))?;
        game = next;
        actions.extend(moves);
        if game.outcome().is_none() {
            game.submit_turn()
                .map_err(|error| format!("5DPGN turn {}: {error}", index + 1))?;
            actions.push(PlayerAction::SubmitTurn);
        }
    }
    Ok((game, actions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronofish_core::Ruleset;

    #[test]
    fn round_trips_the_alternate_army() {
        let game = Game::new(Ruleset::MultiverseVariant);
        let (_, actions) = decode_lan(&game, "(0T1)d2d3 submit").unwrap();
        let text = crate::export_5dpgn_for(game.ruleset(), &actions).unwrap();
        let (_, decoded) = replay_5dpgn(&game, &text).unwrap();
        assert_eq!(actions, decoded);
    }

    #[test]
    fn imports_a_branch_into_chronofish() {
        let initial = Game::new(Ruleset::Multiverse);
        let text = "[Board \"Standard\"]\n[Mode \"5D\"]\n1. Nf3 / Nf6 2. (0T2)Nf3>>(0T1)d3";
        let (game, actions) = replay_5dpgn(&initial, text).unwrap();
        let exported = crate::export_5dpgn(&actions).unwrap();
        let (_, reimported) = replay_5dpgn(&initial, &exported).unwrap();
        assert_eq!(actions, reimported);
        assert_eq!(game.timelines().len(), 2);
        assert!(!game.has_staged_moves());
        assert!(replay_5dpgn(&initial, "[Board \"Standard - Turn Zero\"]\n1. e4").is_err());
    }
}
