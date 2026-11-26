#![cfg(feature = "chronofish")]

use std::collections::BTreeSet;

use chronofish_5d_turngen::core::{
    action::FullMove,
    hypercuboid::HcInfo,
    state::State,
    variants::{default_variant, state_from_setup},
};
use chronofish_core::{Game, PlayerAction, Ruleset};
use chronofish_notation::{apply_uci_moves, format_lan, replay_lan};

fn states(history: &str) -> (Game, State) {
    let game = replay_lan(&Game::new_reference(Ruleset::Multiverse), history).unwrap();
    let setup = default_variant("Standard").unwrap();
    let mut reference = state_from_setup(&setup).unwrap();
    apply_uci_moves(&mut reference, history).unwrap();
    (game, reference)
}

fn native_moves(game: &Game) -> BTreeSet<String> {
    game.legal_actions()
        .into_iter()
        .filter(|action| *action != PlayerAction::SubmitTurn)
        .map(format_lan)
        .collect()
}

fn reference_candidates(reference: &State) -> BTreeSet<String> {
    let mut candidates = BTreeSet::new();
    for from in reference.get_movable_pieces() {
        let mut targets = Vec::new();
        reference.gen_piece_move(from, &mut targets);
        for to in targets {
            candidates.insert(FullMove::new(from, to).to_string());
        }
    }
    candidates
}

#[test]
fn single_front_candidate_moves_match_through_the_opening() {
    for history in ["", "(0T1)g1f3 submit", "(0T1)g1f3 submit (0T1)g8f6 submit"] {
        let (game, reference) = states(history);
        assert_eq!(
            native_moves(&game),
            reference_candidates(&reference),
            "{history}"
        );
    }
}

#[test]
fn candidate_with_royal_exposure_is_excluded_from_live_play() {
    let (game, reference) = states("(0T1)e2e4 submit (0T1)d7d5 submit (0T2)f1b5 submit");
    let ignoring_check = "(0T2)a7a6";
    assert!(!native_moves(&game).contains(ignoring_check));
    assert!(reference_candidates(&reference).contains(ignoring_check));
    let (generator, space) = HcInfo::build_hc(&reference);
    assert!(!generator.search(&reference, space).any(|turn| {
        turn.iter()
            .any(|movement| movement.to_string() == ignoring_check)
    }));
    assert!(replay_lan(&game, "(0T2)a7a6 submit").is_err());
}

#[test]
fn optional_future_front_is_playable_in_live_games() {
    let (game, reference) = states("(0T1)g1f3 submit (0T1)g8f6 submit (0T2)f3(0T1)d3 submit");
    let native = native_moves(&game);
    let upstream = reference_candidates(&reference);
    let present_front = |moves: &BTreeSet<String>| {
        moves
            .iter()
            .filter(|movement| movement.starts_with("(1T1)"))
            .cloned()
            .collect::<BTreeSet<_>>()
    };
    assert!(!present_front(&native).is_empty());
    assert_eq!(present_front(&native), present_front(&upstream));
    assert!(native.is_subset(&upstream));
    assert!(upstream.contains("(0T2)a7a6"));
    assert!(native.contains("(0T2)a7a6"));
    let (generator, space) = HcInfo::build_hc(&reference);
    assert!(generator.search(&reference, space).any(|turn| {
        turn.iter()
            .any(|movement| movement.to_string() == "(0T2)a7a6")
    }));
}

#[test]
fn reference_history_rejects_royal_exposure_at_submission() {
    let (_, reference) = states("(0T1)e2e4 submit (0T1)d7d5 submit (0T2)f1b5 submit");
    let replay = chronofish_notation::replay_checked_uci;
    assert!(replay(&reference, "(0T2)a7a6").is_ok());
    assert!(replay(&reference, "(0T2)a7a6 submit").is_err());
    assert!(replay(&reference, "(0T2)c7c6 submit").is_ok());
}

#[test]
fn reference_draft_offers_only_moves_with_a_safe_completion() {
    use chronofish_5d_turngen::core::{action::ExtMove, referee::draft_choices};
    use std::time::Duration;

    let (_, state) = states("(0T1)e2e4 submit (0T1)d7d5 submit (0T2)f1b5 submit");
    let bad = ExtMove::parse("(0T2)a7a6").unwrap();
    let good = ExtMove::parse("(0T2)c7c6").unwrap();
    let budget = Duration::from_secs(2);
    let root = draft_choices(&state, &[], budget).unwrap();
    assert!(!root.moves.contains(&bad));
    assert!(root.moves.contains(&good));
    assert!(root.complete);
    assert!(draft_choices(&state, &[good], budget).unwrap().can_submit);
    assert!(!draft_choices(&state, &[bad], budget).unwrap().can_submit);
}
