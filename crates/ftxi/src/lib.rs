//! Adversarial UCT over submitted 5D turns, using the shared hypercuboid generator.
//!
//! The reference is ftxi's `5dchess_engine` at `664277d`, particularly
//! src/engine/uct.h. Selection shares statistics across prefixes of legal hypercuboid points.

mod bot;
pub use bot::LokiBot;
mod linear;
pub use linear::LinearProfile;

use chronofish_5d_rules_reference::core::{
    action::{MoveSeq, special},
    bitboard::{bishop_copy_mask, knight_jump1_attack, rook_copy_mask},
    check_position::{BoardOverlay, CheckKind, CheckPosition},
    hypercuboid::{Entry, HcCursor, HcInfo, has_physical_check},
    piece::{NO_PIECE, piece_name, to_white},
    state::State,
};
use chronofish_5d_rules_reference::misc::geometry::{Point, Slice};
use chronofish_5d_rules_reference::misc::integer_set::{Index, IntegerSet};
use chronofish_5d_rules_reference::misc::ordering::{AxisOrders, HcOrdering};
use rand::{Rng, seq::SliceRandom};

/// Static evaluation or a bounded random rollout at a newly explored node.
#[derive(Clone, Copy, Debug)]
pub enum LeafEvaluation {
    Zero,
    ZeroCapture,
    ZeroCaptureCheckWidening,
    Rollout {
        max_turns: usize,
    },
    WeightedRollout {
        max_turns: usize,
    },
    Linear {
        max_turns: usize,
        profile: LinearProfile,
        weighted: bool,
    },
}

#[derive(Default)]
struct Node {
    children: Vec<(MoveSeq, Node)>,
    prefixes: Prefix,
    frontier: Option<Frontier>,
    visits: u32,
}

#[derive(Default)]
struct Prefix {
    edges: Vec<(Index, Prefix)>,
    turn: Option<usize>,
    exhausted: bool,
    visits: u32,
    reward: f64,
}

impl Prefix {
    fn insert(&mut self, point: &[Index], turn: usize, reward: f64) {
        self.record(reward);
        if let Some((&coordinate, rest)) = point.split_first() {
            let index = self
                .edges
                .iter()
                .position(|(key, _)| *key == coordinate)
                .unwrap_or_else(|| {
                    self.edges.push((coordinate, Self::default()));
                    self.edges.len() - 1
                });
            self.edges[index].1.insert(rest, turn, reward);
        } else {
            self.turn = Some(turn);
        }
    }

    fn record(&mut self, reward: f64) {
        self.visits += 1;
        self.reward += reward;
    }

    fn best_child(&self, black: bool) -> Option<usize> {
        let exploration = (2.0 * f64::from(self.visits.max(1)).ln()).sqrt();
        let score = |child: &Self| {
            let visits = f64::from(child.visits);
            let mean = child.reward / visits;
            (if black { -mean } else { mean }) + exploration / visits.sqrt()
        };
        self.edges
            .iter()
            .enumerate()
            .max_by(|(_, (_, a)), (_, (_, b))| score(a).total_cmp(&score(b)))
            .map(|(index, _)| index)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn select(
        &mut self,
        frontier: &mut Frontier,
        state: &State,
        widening: bool,
        coordinates: &mut Point,
        path: &mut Vec<usize>,
    ) -> Option<Selection> {
        if let Some(turn) = self.turn {
            return Some(Selection::Existing(turn));
        }
        let limit = (2.0 * f64::from(self.visits).sqrt()).ceil().max(1.0) as usize;
        if !self.exhausted && (!widening || self.edges.len() < limit) {
            if let Some((turn, point)) = frontier.discover(state, coordinates, &self.edges) {
                return Some(Selection::New(turn, point));
            }
            self.exhausted = true;
        }
        let index = self.best_child(state.get_present().1)?;
        let (coordinate, child) = &mut self.edges[index];
        coordinates.push(*coordinate);
        path.push(index);
        child.select(frontier, state, widening, coordinates, path)
    }

    fn record_path(&mut self, path: &[usize], reward: f64) {
        self.record(reward);
        if let Some((&index, rest)) = path.split_first() {
            self.edges[index].1.record_path(rest, reward);
        }
    }
}

/// Search a position with a fixed simulation budget.
///
/// Returns the selected sequence before submission.
#[must_use]
pub fn search(
    position: &State,
    simulations: u32,
    evaluation: LeafEvaluation,
    rng: &mut impl Rng,
) -> Option<MoveSeq> {
    search_until(position, simulations, evaluation, rng, || false)
}

/// Runs search until the given cancellation check returns true.
#[must_use]
pub fn search_until(
    position: &State,
    simulations: u32,
    evaluation: LeafEvaluation,
    rng: &mut impl Rng,
    stop: impl Fn() -> bool,
) -> Option<MoveSeq> {
    let mut root = Node::default();
    let mut state = position.clone();
    state.enable_undo();
    for _ in 0..simulations {
        if stop() {
            break;
        }
        let mark = state.mark();
        simulate(&mut root, &mut state, evaluation, rng, &stop);
        state.unmake(mark);
    }
    root.children
        .into_iter()
        .max_by_key(|(_, child)| child.visits)
        .map(|(action, _)| action)
}

/// Runs ftxi's flat UCB policy over root turns.
///
/// Each newly discovered turn gets a rollout before previously visited turns
/// compete by UCB. Enumeration stops once the rollout budget is spent.
#[must_use]
pub fn flat_ucb(
    position: &State,
    simulations: u32,
    evaluation: LeafEvaluation,
    rng: &mut impl Rng,
) -> Option<MoveSeq> {
    flat_ucb_until(position, simulations, evaluation, rng, || false)
}

/// Runs `flat_ucb` until the given cancellation check returns true.
#[must_use]
pub fn flat_ucb_until(
    position: &State,
    simulations: u32,
    evaluation: LeafEvaluation,
    rng: &mut impl Rng,
    stop: impl Fn() -> bool,
) -> Option<MoveSeq> {
    let (context, space) = HcInfo::build_hc(position);
    let mut legal = context.search(position, space).fuse();
    let mut candidates: Vec<(MoveSeq, u32, f64)> = Vec::new();
    let mut state = position.clone();
    state.enable_undo();
    let mark = state.mark();
    let sign = if position.get_present().1 { -1.0 } else { 1.0 };

    for sample in 0..simulations {
        if stop() {
            break;
        }
        let selected = if let Some(turn) = legal.next() {
            candidates.push((turn, 0, 0.0));
            candidates.len() - 1
        } else {
            let exploration = (2.0 * f64::from(sample.max(1)).ln()).sqrt();
            let score = |(_, visits, reward): &(MoveSeq, u32, f64)| {
                let visits = f64::from(*visits);
                sign * reward / visits + exploration / visits.sqrt()
            };
            candidates
                .iter()
                .enumerate()
                .rev()
                .max_by(|(_, a), (_, b)| score(a).total_cmp(&score(b)))?
                .0
        };
        let (turn, visits, reward) = &mut candidates[selected];
        apply(&mut state, turn);
        *reward += evaluate(&mut state, evaluation, rng, &stop);
        *visits += 1;
        state.unmake(mark);
    }
    candidates
        .into_iter()
        .rev()
        .max_by_key(|(_, visits, _)| *visits)
        .map(|(turn, _, _)| turn)
}

/// Uniformly samples one complete legal turn, as in ftxi's Monkey engine.
#[must_use]
pub fn monkey(position: &State, rng: &mut impl Rng) -> Option<MoveSeq> {
    monkey_until(position, rng, || false)
}

/// Runs monkey until the given cancellation check returns true.
#[must_use]
pub fn monkey_until(
    position: &State,
    rng: &mut impl Rng,
    stop: impl Fn() -> bool,
) -> Option<MoveSeq> {
    let (context, space) = HcInfo::build_hc(position);
    let mut selected = None;
    for (index, turn) in context.search(position, space).enumerate() {
        if stop() {
            break;
        }
        if rng.random_range(0..=index) == 0 {
            selected = Some(turn);
        }
    }
    selected
}

enum Selection {
    New(MoveSeq, Point),
    Existing(usize),
}

struct Frontier {
    context: HcInfo,
    cursor: HcCursor,
}

impl Frontier {
    fn discover(
        &mut self,
        state: &State,
        prefix: &[Index],
        children: &[(Index, Prefix)],
    ) -> Option<(MoveSeq, Point)> {
        let axis = prefix.len();
        let mut region = Slice::new();
        for (axis, &coordinate) in prefix.iter().enumerate() {
            region.fix_axis(axis, IntegerSet::singleton(coordinate));
        }
        if axis < self.context.dimension() && !children.is_empty() {
            let mut choices = IntegerSet::from_bits(&self.context.get_universe()[axis]);
            for (coordinate, _) in children {
                choices.erase(*coordinate);
            }
            if choices.is_empty() {
                return None;
            }
            region.fix_axis(axis, choices);
        }

        // Search only the selected region. Keep its complement available to
        // sibling prefixes and retain global invalid-region memoization.
        let mut outside = Vec::new();
        if !region.fixed.is_empty() {
            let all = std::mem::take(&mut self.cursor.ss.hcs);
            for mut cube in all {
                if !cube.intersects(&region) {
                    outside.push(cube);
                    continue;
                }
                cube.remove_slice_with(&region, |part| outside.push(part));
                for (axis, allowed) in &region.fixed {
                    cube[*axis].and_assign(allowed);
                }
                self.cursor.ss.push_back(cube);
            }
        }
        let turn = self.cursor.next(&self.context, state);
        self.cursor.ss.hcs.extend(outside);
        turn.map(|turn| (turn, self.cursor.last_point().clone()))
    }
}

fn simulate(
    node: &mut Node,
    state: &mut State,
    evaluation: LeafEvaluation,
    rng: &mut impl Rng,
    stop: &impl Fn() -> bool,
) -> f64 {
    let frontier = node.frontier.get_or_insert_with(|| {
        let (context, space) = HcInfo::build_hc(state);
        let order = axis_order(state, &context, evaluation, rng);
        Frontier {
            context,
            cursor: HcCursor::new(space, order),
        }
    });
    let widening = matches!(evaluation, LeafEvaluation::ZeroCaptureCheckWidening);
    let mut path = Vec::new();
    let choice = node
        .prefixes
        .select(frontier, state, widening, &mut Point::new(), &mut path);
    let reward = match choice {
        Some(Selection::New(action, point)) => {
            apply(state, &action);
            let reward = evaluate(state, evaluation, rng, stop);
            node.prefixes.insert(&point, node.children.len(), reward);
            node.children.push((
                action,
                Node {
                    visits: 1,
                    ..Node::default()
                },
            ));
            reward
        }
        Some(Selection::Existing(index)) => {
            let (action, child) = &mut node.children[index];
            apply(state, action);
            let reward = simulate(child, state, evaluation, rng, stop);
            node.prefixes.record_path(&path, reward);
            reward
        }
        None => terminal_reward(state),
    };
    node.visits += 1;
    reward
}

fn axis_order(
    state: &State,
    context: &HcInfo,
    evaluation: LeafEvaluation,
    rng: &mut impl Rng,
) -> HcOrdering {
    let captures = matches!(
        evaluation,
        LeafEvaluation::ZeroCapture | LeafEvaluation::ZeroCaptureCheckWidening
    );
    let checks = matches!(evaluation, LeafEvaluation::ZeroCaptureCheckWidening);
    let mut check_view = checks.then(|| CheckPosition::for_move_candidates(state));
    let branch = context.first_branch_axis();
    let lists = (0..context.dimension().min(branch + 1))
        .map(|axis| {
            let mut list = context.get_universe()[axis].iter().collect::<Vec<_>>();
            list.shuffle(rng);
            if captures {
                list.sort_by_cached_key(|&coordinate| {
                    let victim = match context.entry(axis, coordinate) {
                        Entry::Physical { victim, .. } | Entry::Arriving { victim, .. } => *victim,
                        _ => return 0,
                    };
                    let capture = i32::from(victim != NO_PIECE) * 480;
                    let check = i32::from(check_view.as_mut().is_some_and(|view| {
                        coordinate_gives_check(state, context, axis, coordinate, view)
                    })) * 300;
                    -(capture + check)
                });
            }
            list
        })
        .collect();
    let mut orders = AxisOrders::new(lists);
    orders.shared = branch;
    HcOrdering::Precomputed(orders)
}

fn coordinate_gives_check<'a>(
    state: &'a State,
    context: &'a HcInfo,
    axis: usize,
    coordinate: Index,
    view: &mut CheckPosition<'a>,
) -> bool {
    let Some(boards) = context.get_move_boards(axis, coordinate) else {
        return false;
    };
    let black = state.get_present().1;
    if has_physical_check(boards.result, !black)
        || boards
            .departure
            .is_some_and(|board| has_physical_check(board, !black))
    {
        return true;
    }
    let arrival = BoardOverlay {
        l: boards.mv.new_position(state).l(),
        t: boards.mv.to.t() + i32::from(black),
        c: !black,
        value: boards.result,
    };
    let departure = boards.departure.map(|value| BoardOverlay {
        l: boards.mv.from.l(),
        t: boards.mv.from.t() + i32::from(black),
        c: !black,
        value,
    });
    view.gives_check(CheckKind::Superphysical, arrival, black, departure)
}

fn apply(state: &mut State, action: &MoveSeq) {
    for (index, movement) in action.iter().copied().enumerate() {
        assert!(state.apply_promo::<true>(movement, action.promo(index)));
    }
    state.submit::<true>();
}

fn evaluate(
    state: &mut State,
    evaluation: LeafEvaluation,
    rng: &mut impl Rng,
    stop: &impl Fn() -> bool,
) -> f64 {
    let turns = match evaluation {
        LeafEvaluation::Zero
        | LeafEvaluation::ZeroCapture
        | LeafEvaluation::ZeroCaptureCheckWidening => {
            let (context, space) = HcInfo::build_hc(state);
            return if context.search(state, space).next().is_some() {
                0.0
            } else {
                terminal_reward(state)
            };
        }
        LeafEvaluation::Rollout { max_turns }
        | LeafEvaluation::WeightedRollout { max_turns }
        | LeafEvaluation::Linear { max_turns, .. } => max_turns,
    };
    let weighted = matches!(
        evaluation,
        LeafEvaluation::WeightedRollout { .. } | LeafEvaluation::Linear { weighted: true, .. }
    );
    for _ in 0..turns {
        if stop() {
            return 0.0;
        }
        let action = {
            let (context, space) = HcInfo::build_hc(state);
            let order = rollout_order(state, &context, weighted, rng);
            HcCursor::new(space, order).next(&context, state)
        };
        let Some(action) = action else {
            return terminal_reward(state);
        };
        apply(state, &action);
    }
    if let LeafEvaluation::Linear { profile, .. } = evaluation {
        let (context, _) = HcInfo::build_hc(state);
        linear::evaluate(state, &context, profile)
    } else {
        0.0
    }
}

fn rollout_order(
    state: &State,
    context: &HcInfo,
    weighted: bool,
    rng: &mut impl Rng,
) -> HcOrdering {
    let mut view = weighted.then(|| CheckPosition::for_move_candidates(state));
    let lists = (0..context.dimension())
        .map(|axis| {
            let mut coordinates = context.get_universe()[axis].iter().collect::<Vec<_>>();
            if let Some(view) = &mut view {
                // Exponential races sample a size-biased permutation without
                // rebuilding a categorical distribution after every draw.
                let mut keys = coordinates
                    .iter()
                    .map(|&coordinate| {
                        let score = rollout_score(state, context, axis, coordinate, view);
                        let uniform = rng.random::<f64>().max(f64::MIN_POSITIVE);
                        (coordinate, (-uniform.ln()).ln() - score / 1200.0)
                    })
                    .collect::<Vec<_>>();
                keys.sort_by(|(_, a), (_, b)| a.total_cmp(b));
                for (slot, (coordinate, _)) in coordinates.iter_mut().zip(keys) {
                    *slot = coordinate;
                }
            } else {
                coordinates.shuffle(rng);
            }
            coordinates
        })
        .collect();
    HcOrdering::Precomputed(AxisOrders::new(lists))
}

fn rollout_score<'a>(
    state: &'a State,
    context: &'a HcInfo,
    axis: usize,
    coordinate: Index,
    view: &mut CheckPosition<'a>,
) -> f64 {
    let Some(boards) = context.get_move_boards(axis, coordinate) else {
        return 0.0;
    };
    let movement = boards.mv;
    let capture = match to_white(piece_name(movement.captured_piece(state))) {
        b'Q' | b'Y' => 1440.0,
        b'S' => 800.0,
        b'D' | b'U' => 450.0,
        b'B' | b'N' => 350.0,
        b'R' | b'C' => 300.0,
        b'W' => 120.0,
        b'P' => 100.0,
        _ => 0.0,
    };
    let mut score = 1.1 * capture;
    let kind = movement.move_type(state);
    let mover = to_white(piece_name(movement.moved_piece(state)));
    if matches!(mover, b'K' | b'C') {
        let board = boards.result;
        let enemy = if state.get_present().1 {
            board.white()
        } else {
            board.black()
        };
        let square = movement.to.xy();
        let latent = mover == b'K'
            && enemy
                & ((knight_jump1_attack(square) & board.knight())
                    | (rook_copy_mask(square, 1) & board.lbishop())
                    | (bishop_copy_mask(square, 1) & board.lunicorn()))
                != 0;
        score -= if latent { 50_000.0 } else { 1200.0 };
        if kind & special::BRANCHING != 0 {
            score -= 8000.0;
        }
    }
    if kind & special::SUPERPHYSICAL != 0 {
        score -= 700.0;
    }
    if coordinate_gives_check(state, context, axis, coordinate, view) {
        score += 1200.0;
    }
    score
}

fn terminal_reward(state: &State) -> f64 {
    if state.has_phantom_check() {
        if state.get_present().1 { 1.0 } else { -1.0 }
    } else {
        0.0
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::*;

    #[test]
    fn upstream_check_profile_counts_cross_board_threats() {
        use chronofish_5d_rules_reference::core::variants::startpos;
        use chronofish_notation::apply_uci_moves;

        let mut state = startpos();
        let moves = concat!(
            "(0T1)e2e3 submit (0T1)a7a6 submit (0T2)d1f3 submit (0T2)a6a5 submit ",
            "(0T3)a2a3 submit (0T3)b7b6 submit (0T4)a3a4 submit (0T4)c8b7 submit ",
            "(0T5)f3(0T1)f7 submit (1T1)e8f7 submit (1T2)g1h3 submit (1T2)g8f6 submit ",
            "(1T3)h3g5 submit (1T3)f7e8 submit (1T4)c2c3 submit (1T4)a7a6 submit",
        );
        apply_uci_moves(&mut state, moves).unwrap();
        let (context, _) = HcInfo::build_hc(&state);
        let mut view = CheckPosition::for_move_candidates(&state);
        let mut local = 0;
        let mut remote_only = 0;
        let mut weight_score_sum = 0.0;
        let defender = !state.get_present().1;
        for axis in 0..context.dimension() {
            for coordinate in &context.get_universe()[axis] {
                weight_score_sum += rollout_score(&state, &context, axis, coordinate, &mut view);
                let Some(boards) = context.get_move_boards(axis, coordinate) else {
                    continue;
                };
                if has_physical_check(boards.result, defender)
                    || boards
                        .departure
                        .is_some_and(|board| has_physical_check(board, defender))
                {
                    local += 1;
                } else if coordinate_gives_check(&state, &context, axis, coordinate, &mut view) {
                    remote_only += 1;
                }
            }
        }
        // Counts recorded from semimove_feature at upstream commit 664277d.
        assert_eq!((local, remote_only), (2, 2));
        assert!(
            (weight_score_sum + 50_570.0).abs() < 0.01,
            "{weight_score_sum}"
        );
    }

    #[test]
    fn restricted_frontier_preserves_sibling_turns() {
        use chronofish_5d_rules_reference::core::variants::startpos;
        use chronofish_notation::apply_uci_moves;
        use std::collections::BTreeSet;

        let mut state = startpos();
        apply_uci_moves(
            &mut state,
            "(0T1)g1f3 submit (0T1)g8f6 submit (0T2)f3(0T1)d3 submit",
        )
        .unwrap();
        let (context, space) = HcInfo::build_hc(&state);
        let mut baseline = HcCursor::new(space.clone(), HcOrdering::Natural);
        let mut expected = BTreeSet::new();
        while baseline.next(&context, &state).is_some() {
            expected.insert(baseline.last_point().clone());
        }
        let coordinate = expected.first().unwrap()[0];
        assert!(expected.iter().any(|point| point[0] != coordinate));
        let mut frontier = Frontier {
            context,
            cursor: HcCursor::new(space, HcOrdering::Natural),
        };
        let mut found = BTreeSet::new();
        while let Some((_, point)) = frontier.discover(&state, &[coordinate], &[]) {
            assert_eq!(point[0], coordinate);
            assert!(found.insert(point));
        }
        assert!(!found.is_empty());
        while let Some((_, point)) = frontier.discover(&state, &[], &[]) {
            assert_ne!(point[0], coordinate);
            assert!(found.insert(point));
        }
        assert_eq!(found, expected);
    }

    #[test]
    fn coordinate_prefixes_share_visits_and_respect_the_mover() {
        let mut tree = Prefix::default();
        tree.insert(&[1, 2], 0, 1.0);
        tree.insert(&[1, 3], 1, 1.0);
        tree.insert(&[4, 1], 2, -1.0);

        assert_eq!(tree.best_child(false), Some(0));
        tree.record_path(&[0, 1], 0.5);
        assert_eq!(tree.visits, 4);
        assert_eq!(tree.edges[0].1.visits, 3);
        assert_eq!(tree.edges[0].1.reward, 2.5);
        assert_eq!(tree.best_child(true), Some(1));
    }
}
