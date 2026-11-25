use chronofish_core::{AiPlayer, Game, Move, PlayerAction, Position, Ruleset};
use chronofish_ftxi::LokiBot;
use kairos::KairosBot;

fn play(game: &mut Game, time: i32, from: (i32, i32), to: (i32, i32, i32)) {
    let position = |time, x, y| Position {
        timeline_id: 0,
        time,
        x,
        y,
    };
    game.apply_move(Move::new(
        position(time, from.0, from.1),
        position(to.0, to.1, to.2),
    ))
    .expect("fixture move");
    game.submit_turn().expect("fixture turn");
}

#[test]
fn both_bots_emit_verified_reference_turns() {
    for mut bot in [
        Box::new(KairosBot::default()) as Box<dyn AiPlayer>,
        Box::new(LokiBot::default()),
    ] {
        assert!(!bot.supports_ruleset(Ruleset::Standard));
        assert!(bot.choose_action(&Game::new(Ruleset::Standard)).is_err());
        for ruleset in [Ruleset::Multiverse, Ruleset::MultiverseVariant] {
            assert!(bot.supports_ruleset(ruleset));
            let mut game = Game::new(ruleset);
            if ruleset == Ruleset::Multiverse {
                play(&mut game, 0, (6, 0), (0, 5, 2));
                play(&mut game, 1, (6, 7), (1, 5, 5));
                play(&mut game, 2, (5, 2), (0, 3, 2));
            }

            let before = game.turn();
            let plan = bot.choose_action(&game).expect("bot plan");
            assert_eq!(plan.principal_variation.first(), Some(&plan.action));
            assert_eq!(
                plan.principal_variation.last(),
                Some(&PlayerAction::SubmitTurn)
            );
            for (index, action) in plan.principal_variation.into_iter().enumerate() {
                if index > 0 {
                    assert_eq!(
                        bot.choose_action(&game).expect("remaining plan").action,
                        action
                    );
                }
                assert!(game.legal_actions().contains(&action), "{action:?}");
                game.apply_action(action).expect("verified action");
            }
            assert_ne!(game.turn(), before);
        }
    }
}
