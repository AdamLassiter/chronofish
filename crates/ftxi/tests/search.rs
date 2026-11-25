use chronofish_5d_rules::core::{action::Action, hypercuboid::HcInfo, variants::startpos};
use chronofish_ftxi::{LeafEvaluation, flat_ucb, monkey, search};
use chronofish_notation::apply_uci_moves;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn fixed_seed_search_returns_a_legal_turn_after_branching() {
    let mut position = startpos();
    apply_uci_moves(
        &mut position,
        "(0T1)g1f3 submit (0T1)g8f6 submit (0T2)f3(0T1)d3 submit",
    )
    .unwrap();
    assert_ne!(position.get_lines_range(), (0, 0));
    let original_boards = position.get_boards();
    let (context, space) = HcInfo::build_hc(&position);
    let legal: Vec<_> = context
        .search(&position, space)
        .map(|seq| Action::from_seq(&seq, &position).to_string())
        .collect();

    for evaluation in [
        LeafEvaluation::Zero,
        LeafEvaluation::ZeroCapture,
        LeafEvaluation::ZeroCaptureCheckWidening,
        LeafEvaluation::Rollout { max_turns: 2 },
        LeafEvaluation::WeightedRollout { max_turns: 2 },
    ] {
        let choose = || {
            let mut rng = StdRng::seed_from_u64(42);
            let sequence = search(&position, 40, evaluation, &mut rng).unwrap();
            Action::from_seq(&sequence, &position).to_string()
        };
        let selected = choose();
        assert!(legal.contains(&selected));
        assert_eq!(selected, choose());
        assert_eq!(position.get_boards(), original_boards);
    }
    let mut rng = StdRng::seed_from_u64(42);
    let flat = flat_ucb(
        &position,
        80,
        LeafEvaluation::WeightedRollout { max_turns: 2 },
        &mut rng,
    )
    .unwrap();
    let random = monkey(&position, &mut rng).unwrap();
    for turn in [flat, random] {
        assert!(legal.contains(&Action::from_seq(&turn, &position).to_string()));
    }
    assert_eq!(position.get_boards(), original_boards);
}
