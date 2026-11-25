use chronofish_core::{AiPlayer, Game, PlayerAction, Ruleset};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    BlueshiftConfig, BlueshiftLevel, CarlsenConfig, DeepBlueshift, Grabablanca, MonteCarlsen,
};

/// Plays the move between two squares of the present board, submitting if the turn stays open.
fn play(game: &mut Game, from: (i32, i32), to: (i32, i32)) {
    let action = game
        .legal_actions()
        .into_iter()
        .find(|action| {
            matches!(action, PlayerAction::Move { movement }
                if (movement.from.x, movement.from.y) == from
                    && (movement.to.x, movement.to.y) == to
                    && movement.from.timeline_id == movement.to.timeline_id
                    && movement.from.time == movement.to.time)
        })
        .unwrap_or_else(|| panic!("{from:?} -> {to:?} is not legal"));
    let mover = game.turn();
    game.apply_action(action).unwrap();
    if game.turn() == mover && game.legal_actions().contains(&PlayerAction::SubmitTurn) {
        game.submit_turn().unwrap();
    }
}

fn squares(action: PlayerAction) -> Option<((i32, i32), (i32, i32))> {
    match action {
        PlayerAction::Move { movement } => Some((
            (movement.from.x, movement.from.y),
            (movement.to.x, movement.to.y),
        )),
        PlayerAction::SubmitTurn => None,
    }
}

/// 1. f3 e5 2. g4, with Black to play Qh4#.
fn fools_mate() -> Game {
    let mut game = Game::new(Ruleset::Standard);
    play(&mut game, (5, 1), (5, 2));
    play(&mut game, (4, 6), (4, 4));
    play(&mut game, (6, 1), (6, 3));
    game
}

/// 1. e4 d6 2. Qg4, leaving the queen to the c8 bishop.
fn hanging_queen() -> Game {
    let mut game = Game::new(Ruleset::Standard);
    play(&mut game, (4, 1), (4, 3));
    play(&mut game, (3, 6), (3, 5));
    play(&mut game, (3, 0), (6, 3));
    game
}

fn random_opening(ruleset: Ruleset, seed: u64, actions: usize) -> Game {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut game = Game::new(ruleset);
    for _ in 0..actions {
        let legal = game.legal_actions();
        if legal.is_empty() {
            break;
        }
        let mut next = game.clone();
        next.apply_action(legal[rng.random_range(0..legal.len())])
            .unwrap();
        if next.outcome().is_some() {
            break;
        }
        game = next;
    }
    game
}

fn small_bots() -> Vec<Box<dyn AiPlayer>> {
    vec![
        Box::new(Grabablanca::new(3)),
        Box::new(DeepBlueshift::new(BlueshiftConfig {
            max_nodes: 2_000,
            ..BlueshiftConfig::default()
        })),
        Box::new(MonteCarlsen::new(
            CarlsenConfig {
                iterations: 60,
                rollout_actions: 4,
                ..CarlsenConfig::default()
            },
            5,
        )),
    ]
}

#[test]
fn every_bot_returns_legal_actions_in_every_ruleset() {
    for ruleset in [
        Ruleset::Standard,
        Ruleset::Multiverse,
        Ruleset::MultiverseVariant,
    ] {
        for game in [Game::new(ruleset), random_opening(ruleset, 41, 10)] {
            for mut bot in small_bots() {
                let decision = bot.choose_action(&game).expect("bot action");
                let mut next = game.clone();
                next.apply_action(decision.action).unwrap_or_else(|error| {
                    panic!("{} in {ruleset}: {error}", bot.profile().username)
                });
                assert_eq!(decision.principal_variation.first(), Some(&decision.action));
            }
        }
    }
}

#[test]
fn search_bots_find_fools_mate() {
    let game = fools_mate();
    let mut blueshift = DeepBlueshift::new(BlueshiftConfig {
        max_nodes: 20_000,
        ..BlueshiftConfig::default()
    });
    let mut carlsen = MonteCarlsen::new(
        CarlsenConfig {
            iterations: 400,
            rollout_actions: 4,
            ..CarlsenConfig::default()
        },
        11,
    );
    let mut greedy = Grabablanca::new(1);
    for bot in [
        &mut blueshift as &mut dyn AiPlayer,
        &mut carlsen,
        &mut greedy,
    ] {
        let decision = bot.choose_action(&game).unwrap();
        assert_eq!(
            squares(decision.action),
            Some(((3, 7), (7, 3))),
            "{} missed Qh4#",
            bot.profile().username
        );
        let mut next = game.clone();
        next.apply_action(decision.action).unwrap();
        assert!(
            next.outcome()
                .is_some_and(|outcome| outcome.winner.is_some())
        );
    }
}

#[test]
fn greedy_and_alpha_beta_take_a_hanging_queen() {
    let game = hanging_queen();
    let mut greedy = Grabablanca::new(9);
    let mut blueshift = DeepBlueshift::new(BlueshiftConfig {
        max_nodes: 5_000,
        ..BlueshiftConfig::default()
    });
    for bot in [&mut greedy as &mut dyn AiPlayer, &mut blueshift] {
        let decision = bot.choose_action(&game).unwrap();
        assert_eq!(
            squares(decision.action),
            Some(((2, 7), (6, 3))),
            "{} left the queen",
            bot.profile().username
        );
    }
}

#[test]
fn alpha_beta_is_deterministic_under_a_node_budget() {
    let game = random_opening(Ruleset::Multiverse, 7, 8);
    let config = BlueshiftConfig {
        max_nodes: 1_500,
        ..BlueshiftConfig::default()
    };
    let first = DeepBlueshift::new(config).choose_action(&game).unwrap();
    let second = DeepBlueshift::new(config).choose_action(&game).unwrap();
    assert_eq!(first.action, second.action);
    assert_eq!(first.principal_variation, second.principal_variation);
}

#[test]
fn bots_have_distinct_accounts() {
    let names = small_bots()
        .iter()
        .map(|bot| bot.profile().username)
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        ["bot-grabablanca", "bot-deep-blueshift", "bot-monte-carlsen"]
    );
}

#[test]
fn blueshift_levels_have_rising_budgets_and_their_own_accounts() {
    assert_eq!(
        BlueshiftLevel::Hard.max_nodes(),
        BlueshiftConfig::default().max_nodes
    );
    assert!(
        BlueshiftLevel::ALL
            .windows(2)
            .all(|pair| pair[0].max_nodes() < pair[1].max_nodes())
    );
    let names = BlueshiftLevel::ALL.map(|level| DeepBlueshift::level(level).profile().username);
    assert_eq!(
        names,
        [
            "bot-deep-blueshift-easy",
            "bot-deep-blueshift-medium",
            "bot-deep-blueshift"
        ]
    );
    assert_eq!(
        DeepBlueshift::default().profile().username,
        "bot-deep-blueshift"
    );
}
