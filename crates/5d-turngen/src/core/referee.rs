//! Checked application of reference 5D turns.

use super::{action::ExtMove, state::State};

/// Apply one move of an action being built; Err explains why it is not playable.
pub fn apply_draft_move(s: &mut State, mv: ExtMove) -> Result<(), String> {
    let player = s.get_present().1;
    let p = mv.fm.from;
    if s.get_board_ptr(p.l(), p.t(), player).is_none()
        || s.get_timeline_end(p.l()) != (p.t(), player)
    {
        return Err(format!("{} is not from a playable board", mv));
    }
    let n = s
        .normalize_promotion(mv)
        .ok_or_else(|| format!("bad promotion in {}", mv))?;
    if !s.apply_ext::<false>(n) {
        return Err(format!("illegal move {}", mv));
    }
    Ok(())
}

/// Play a whole action from `s` (the sprt referee's rules); returns the position after it.
pub fn play_action(s: &State, moves: &[ExtMove]) -> Result<State, String> {
    let mut c = s.clone();
    for &m in moves {
        apply_draft_move(&mut c, m)?;
    }
    submit_draft(&c)
}

/// Commit a staged position once the Present advances and all royals are safe.
pub fn submit_draft(draft: &State) -> Result<State, String> {
    let mut c = draft.clone();
    if !c.submit::<false>() {
        return Err(
            "the action does not move the present: every board on the present must be played"
                .into(),
        );
    }
    if c.first_check(c.get_present().1).is_some() {
        return Err("the action leaves a royal piece in check".into());
    }
    Ok(c)
}

use super::hypercuboid::{self, HcInfo};
use crate::misc::geometry::SearchSpace;
use std::collections::HashSet;
use std::sync::Arc;

/// Whether the side to move has a legal action; None if the search ran out of `budget`.
pub fn has_legal_action(s: &State, budget: std::time::Duration) -> Option<bool> {
    let (w, ss) = HcInfo::build_hc(s);
    let stop = std::sync::atomic::AtomicBool::new(false);
    let _guard = hypercuboid::interrupt_on(&stop, std::time::Instant::now().checked_add(budget));
    if w.search(s, ss).next().is_some() {
        Some(true)
    } else if hypercuboid::interrupted() {
        None
    } else {
        Some(false)
    }
}

/// Result of a reference position check. Unknown means enumeration timed out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionStatus {
    Playing,
    Unknown,
    Checkmate,
    Stalemate,
    Exiled { black: bool },
}

pub fn position_status(state: &State, budget: std::time::Duration) -> PositionStatus {
    classify_position(state, has_legal_action(state, budget))
}

fn classify_position(state: &State, legal: Option<bool>) -> PositionStatus {
    match legal {
        Some(true) => PositionStatus::Playing,
        None => PositionStatus::Unknown,
        Some(false) if state.has_phantom_check() => PositionStatus::Checkmate,
        Some(false) => match state.exiled_kings_loser() {
            Some(black) => PositionStatus::Exiled { black },
            None => PositionStatus::Stalemate,
        },
    }
}

const DIRECT_SEARCH_LIMIT: usize = 2048;

/// Geometry shared by all ordered drafts of one immutable submitted position.
pub struct DraftSearch {
    state: Arc<State>,
    context: HcInfo,
    space: SearchSpace,
}

impl DraftSearch {
    pub fn new(state: Arc<State>) -> Self {
        let (context, space) = HcInfo::build_hc(&state);
        Self {
            state,
            context,
            space,
        }
    }

    pub fn state(&self) -> &Arc<State> {
        &self.state
    }

    pub fn status(&self) -> PositionStatus {
        let _guard = hypercuboid::interrupt_off();
        let exists = self
            .context
            .search(&self.state, self.space.clone())
            .next()
            .is_some();
        classify_position(&self.state, Some(exists))
    }

    pub fn choices(&self, draft: &[ExtMove]) -> Result<DraftChoices, String> {
        self.query(draft, Query::All)
    }

    /// Whether `movement` is among [`Self::choices`] for `draft`, proving only that move.
    pub fn allows_move(&self, draft: &[ExtMove], movement: ExtMove) -> Result<bool, String> {
        let normalized = {
            let mut position = self.state.as_ref().clone();
            for &m in draft {
                apply_draft_move(&mut position, m)?;
            }
            position.normalize_promotion(movement)
        };
        let Some(normalized) = normalized else {
            return Ok(false);
        };
        Ok(self
            .query(draft, Query::Move(normalized))?
            .moves
            .contains(&normalized))
    }

    /// Whether [`Self::choices`] for `draft` allows submitting it, proving only that.
    pub fn allows_submit(&self, draft: &[ExtMove]) -> Result<bool, String> {
        Ok(self.query(draft, Query::Submit)?.can_submit)
    }

    fn query(&self, draft: &[ExtMove], query: Query) -> Result<DraftChoices, String> {
        collect_draft_choices(
            &self.state,
            draft,
            std::time::Duration::MAX,
            Some(DIRECT_SEARCH_LIMIT),
            Some((&self.context, &self.space)),
            query,
        )
    }
}

/// What a draft search must decide: every choice, one move, or submission.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Query {
    All,
    Move(ExtMove),
    Submit,
}

impl Query {
    fn settled(self, moves: &std::collections::BTreeSet<ExtMove>, can_submit: bool) -> bool {
        match self {
            Self::All => false,
            Self::Move(movement) => moves.contains(&movement),
            Self::Submit => can_submit,
        }
    }
}

/// Choices proven to extend an ordered draft to a legal reference turn.
#[derive(Debug, Default)]
pub struct DraftChoices {
    pub moves: Vec<ExtMove>,
    pub can_submit: bool,
    /// False when the deadline prevented a complete enumeration.
    pub complete: bool,
}

/// Enumerate the moves that have a legal completion from the submitted position.
/// The draft and optional next move are replayed in their requested order.
/// The remainder is checked in generator order first, then in alternative orders.
pub fn draft_choices(
    submitted: &State,
    draft: &[ExtMove],
    budget: std::time::Duration,
) -> Result<DraftChoices, String> {
    collect_draft_choices(
        submitted,
        draft,
        budget,
        Some(DIRECT_SEARCH_LIMIT),
        None,
        Query::All,
    )
}

fn collect_draft_choices(
    submitted: &State,
    draft: &[ExtMove],
    budget: std::time::Duration,
    direct_limit: Option<usize>,
    prepared: Option<(&HcInfo, &SearchSpace)>,
    query: Query,
) -> Result<DraftChoices, String> {
    use super::action::Action;
    use std::collections::BTreeSet;

    let mut position = submitted.clone();
    let mut normalized_draft = Vec::with_capacity(draft.len());
    for &movement in draft {
        let normalized = position
            .normalize_promotion(movement)
            .ok_or_else(|| format!("bad promotion in {movement}"))?;
        apply_draft_move(&mut position, normalized)?;
        normalized_draft.push(normalized);
    }
    let mut moves = BTreeSet::new();
    let mut can_submit = false;
    let stop = std::sync::atomic::AtomicBool::new(false);
    let _guard = hypercuboid::interrupt_on(&stop, std::time::Instant::now().checked_add(budget));
    let built;
    let (context, mut space) = if let Some((context, space)) = prepared {
        (context, space.clone())
    } else {
        let (context, space) = HcInfo::build_hc(submitted);
        built = context;
        (&built, space)
    };
    let restrict_search = direct_limit.is_some();
    use super::hypercuboid::Entry;
    // Each move uses its source timeline exactly once. Restrict that axis before
    // searching for completions of an already ordered prefix.
    if restrict_search {
        for movement in &normalized_draft {
            if let Some(axis) = context.line_to_axis(movement.fm.from.l()) {
                for hc in &mut space.hcs {
                    hc[axis].erase_if(|i| match context.entry(axis, i) {
                        Entry::Physical { m, .. } => *m != movement.fm,
                        Entry::Departing { from, .. } => *from != movement.fm.from,
                        _ => true,
                    });
                }
            }
        }
    }
    if restrict_search && !draft.is_empty() {
        let (lo, hi) = position.get_lines_range();
        let (old_lo, old_hi) = submitted.get_lines_range();
        for line in lo..=hi {
            if !(old_lo..=old_hi).contains(&line)
                || position.get_timeline_end(line) != submitted.get_timeline_end(line)
            {
                restrict_board(&mut space, context, &position, line);
            }
        }
    }
    space.prune_empty();
    let inspect = |sequence: &super::action::MoveSeq,
                   moves: &mut BTreeSet<ExtMove>,
                   can_submit: &mut bool|
     -> Result<(), String> {
        let action = Action::from_seq(sequence, submitted);
        let mut remaining = action.moves().to_vec();
        for movement in &normalized_draft {
            if let Some(index) = remaining.iter().position(|candidate| candidate == movement) {
                remaining.remove(index);
            } else {
                break;
            }
        }
        if remaining.len() + normalized_draft.len() != action.len() {
            return Ok(());
        }

        if remaining.is_empty() && *can_submit
            || (!remaining.is_empty() && remaining.iter().all(|m| moves.contains(m)))
        {
            return Ok(());
        }
        // A generated one-move turn is already a proof at the untouched root.
        if normalized_draft.is_empty() && remaining.len() == 1 {
            moves.insert(remaining[0]);
            return Ok(());
        }
        if restrict_search {
            // Spatial moves on distinct boards commute. The legal generated point
            // proves every ordering, including the already staged prefix.
            if action
                .moves()
                .iter()
                .all(|m| m.fm.from.tl() == m.fm.to.tl())
            {
                *can_submit |= remaining.is_empty();
                moves.extend(remaining);
                return Ok(());
            }
            if let Some(suffix) = action.moves().strip_prefix(normalized_draft.as_slice()) {
                if let Some(first) = suffix.first() {
                    moves.insert(*first);
                } else {
                    *can_submit = true;
                }
                if remaining.iter().all(|m| moves.contains(m)) {
                    return Ok(());
                }
            }
        }
        let expected = play_action(submitted, action.moves())?;
        if remaining.is_empty() {
            *can_submit |=
                submit_draft(&position).is_ok_and(|state| state.hash() == expected.hash());
            return Ok(());
        }
        // The generator canonicalizes independent moves. Offer each possible
        // first move if playing it first yields the same reference position.
        for index in 0..remaining.len() {
            if moves.contains(&remaining[index]) {
                continue;
            }
            let mut next = position.clone();
            if apply_draft_move(&mut next, remaining[index]).is_err() {
                continue;
            }
            let mut rest = remaining.clone();
            let choice = rest.remove(index);
            if completion_matches(&next, &mut rest, &expected) {
                moves.insert(choice);
            }
        }
        Ok(())
    };
    // Small spaces are cheaper to traverse directly. For wide positions find
    // one witness per move-bearing coordinate, skipping its other completions.
    if !restrict_search
        || space.hcs.iter().all(|hc| {
            (0..context.dimension()).fold(1usize, |v, a| v.saturating_mul(hc[a].len()))
                < direct_limit.unwrap_or_default()
        })
    {
        for seq in context.search(submitted, space) {
            inspect(&seq, &mut moves, &mut can_submit)?;
            if query.settled(&moves, can_submit) {
                break;
            }
        }
    } else {
        // A large product of candidate coordinates can still contain only
        // one or two legal turns (for example, when evading check).
        let mut initial = context.search(submitted, space.clone());
        for _ in 0..4 {
            let Some(seq) = initial.next() else {
                return Ok(DraftChoices {
                    moves: moves.into_iter().collect(),
                    can_submit,
                    complete: !hypercuboid::interrupted(),
                });
            };
            inspect(&seq, &mut moves, &mut can_submit)?;
            if query.settled(&moves, can_submit) {
                return Ok(DraftChoices {
                    moves: moves.into_iter().collect(),
                    can_submit,
                    complete: !hypercuboid::interrupted(),
                });
            }
        }
        let coordinate_move = |axis, i| {
            let raw = match context.entry(axis, i) {
                Entry::Physical { m, promo, .. } => ExtMove::new(*m, *promo),
                Entry::Arriving { m, .. } => ExtMove::new(*m, super::piece::NO_PIECE),
                _ => return None,
            };
            submitted.normalize_promotion(raw)
        };
        // A submit witness must contain exactly the moves already staged.
        let mut finished = space.clone();
        for hc in &mut finished.hcs {
            for axis in 0..context.dimension() {
                hc[axis].erase_if(|i| {
                    coordinate_move(axis, i).is_some_and(|m| !normalized_draft.contains(&m))
                });
            }
        }
        finished.prune_empty();
        if matches!(query, Query::Move(_)) {
            finished.hcs.clear();
        }
        for seq in context.search(submitted, finished) {
            inspect(&seq, &mut moves, &mut can_submit)?;
            if can_submit {
                break;
            }
        }
        let axes = if query == Query::Submit {
            0
        } else {
            context.dimension()
        };
        for axis in 0..axes {
            for i in context.get_universe()[axis].iter() {
                if hypercuboid::interrupted() || query.settled(&moves, can_submit) {
                    break;
                }
                let Some(choice) = coordinate_move(axis, i) else {
                    continue;
                };
                if moves.contains(&choice) || normalized_draft.contains(&choice) {
                    continue;
                }
                if matches!(query, Query::Move(target) if target != choice) {
                    continue;
                }
                let mut next = position.clone();
                if apply_draft_move(&mut next, choice).is_err() {
                    continue;
                }
                let mut pinned = space.clone();
                pinned.hcs.retain_mut(|hc| {
                    if !hc[axis].contains(i) {
                        return false;
                    }
                    hc[axis].assign_singleton(i);
                    true
                });
                // Every board appended by the ordered prefix must occur in
                // the witness. This also fixes the order of new branch lines.
                let destination = if next.new_line() != position.new_line() {
                    position.new_line()
                } else {
                    choice.fm.to.l()
                };
                restrict_board(&mut pinned, context, &next, choice.fm.from.l());
                if destination != choice.fm.from.l() {
                    restrict_board(&mut pinned, context, &next, destination);
                }
                pinned.prune_empty();
                for seq in context.search(submitted, pinned) {
                    inspect(&seq, &mut moves, &mut can_submit)?;
                    if moves.contains(&choice) {
                        break;
                    }
                }
            }
        }
    }
    Ok(DraftChoices {
        moves: moves.into_iter().collect(),
        can_submit,
        complete: !hypercuboid::interrupted(),
    })
}

fn restrict_board(
    space: &mut crate::misc::geometry::SearchSpace,
    context: &HcInfo,
    position: &State,
    line: i32,
) {
    let Some(axis) = context.line_to_axis(line) else {
        space.hcs.clear();
        return;
    };
    let turn = position.get_timeline_end(line);
    let board = position.board(line, turn.0, turn.1);
    for hc in &mut space.hcs {
        hc[axis].erase_if(|coord| {
            let entry = context.entry(axis, coord);
            if axis >= context.first_branch_axis() {
                let super::hypercuboid::Entry::Arriving { m, .. } = entry else {
                    return true;
                };
                if super::multiverse::next_turn((m.to.t(), position.player())) != turn {
                    return true;
                }
            }
            entry.is_null() || entry.board() != board
        });
    }
}

fn completion_matches(position: &State, remaining: &mut Vec<ExtMove>, expected: &State) -> bool {
    completion_matches_from(position, remaining, expected, &mut HashSet::new())
}

/// Whether some order of `remaining` played from `position` reproduces `expected`.
///
/// Different orders of commuting moves reach the same intermediate position, so a
/// position already shown to fail with the same remaining moves is not searched again.
/// That bounds the work by the distinct intermediate positions (at most 2^n) rather
/// than the n! orders.
fn completion_matches_from(
    position: &State,
    remaining: &mut Vec<ExtMove>,
    expected: &State,
    failed: &mut HashSet<(u64, Vec<ExtMove>)>,
) -> bool {
    if hypercuboid::interrupted() {
        return false;
    }
    if remaining.is_empty() {
        return submit_draft(position).is_ok_and(|result| {
            result.hash() == expected.hash()
                && result.get_present() == expected.get_present()
                && result.multiverse().same_boards(expected.multiverse())
        });
    }
    let key = if remaining.len() > 2 {
        let mut set = remaining.clone();
        set.sort_unstable();
        let key = (position.hash(), set);
        if failed.contains(&key) {
            return false;
        }
        Some(key)
    } else {
        None
    };
    for index in 0..remaining.len() {
        let mut next = position.clone();
        let movement = remaining.remove(index);
        let matched = apply_draft_move(&mut next, movement).is_ok()
            && completion_matches_from(&next, remaining, expected, failed);
        remaining.insert(index, movement);
        if matched {
            return true;
        }
    }
    if let Some(key) = key {
        failed.insert(key);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::super::variants::{default_variant, state_from_setup};
    use super::*;

    #[test]
    fn constrained_choices_match_full_enumeration_after_a_branch() {
        let mut state = state_from_setup(&default_variant("Standard").unwrap()).unwrap();
        for token in ["(0T1)g1f3", "(0T1)g8f6", "(0T2)f3(0T1)d3"] {
            state = play_action(&state, &[ExtMove::parse(token).unwrap()]).unwrap();
        }
        let check = |prefix: &[ExtMove]| {
            let budget = std::time::Duration::MAX;
            let full =
                collect_draft_choices(&state, prefix, budget, None, None, Query::All).unwrap();
            let fast =
                collect_draft_choices(&state, prefix, budget, Some(0), None, Query::All).unwrap();
            assert!(full.complete && fast.complete);
            assert_eq!(
                (fast.can_submit, &fast.moves),
                (full.can_submit, &full.moves)
            );
            fast.moves
        };
        for movement in check(&[]) {
            check(&[movement]);
        }
    }

    /// Positions along seeded random games that branch often, kept to those whose
    /// complete turns can still be enumerated quickly.
    fn branching_positions(seed: u64, turns: usize, max_turns: usize) -> Vec<State> {
        use super::super::action::Action;
        let mut seed = seed | 1;
        let mut rand = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            usize::try_from(seed % n as u64).unwrap()
        };
        let lines = |s: &State| {
            let (low, high) = s.get_lines_range();
            high - low
        };
        let mut state = state_from_setup(&default_variant("Standard").unwrap()).unwrap();
        let mut out = Vec::new();
        for _ in 0..turns {
            let (context, space) = HcInfo::build_hc(&state);
            let options = context
                .search(&state, space)
                .take(max_turns + 1)
                .map(|sequence| Action::from_seq(&sequence, &state))
                .collect::<Vec<_>>();
            if options.is_empty() {
                break;
            }
            if options.len() <= max_turns {
                out.push(state.clone());
            }
            let now = lines(&state);
            let branching = options
                .iter()
                .filter(|a| play_action(&state, a.moves()).is_ok_and(|n| lines(&n) > now))
                .collect::<Vec<_>>();
            let action = if !branching.is_empty() && rand(3) != 0 {
                branching[rand(branching.len())]
            } else {
                &options[rand(options.len())]
            };
            state = play_action(&state, action.moves()).unwrap();
        }
        out
    }

    /// The restricted and witness searches must offer exactly what the exhaustive
    /// enumeration offers, for the root and for staged prefixes of one and two moves.
    #[test]
    fn fast_choices_match_full_enumeration_along_branching_games() {
        let budget = std::time::Duration::MAX;
        let mut checked = 0;
        for seed in [3, 11, 29, 47] {
            for state in branching_positions(seed, 24, 3000) {
                let check = |prefix: &[ExtMove]| {
                    let full =
                        collect_draft_choices(&state, prefix, budget, None, None, Query::All)
                            .unwrap();
                    for limit in [0, DIRECT_SEARCH_LIMIT, usize::MAX] {
                        let fast = collect_draft_choices(
                            &state,
                            prefix,
                            budget,
                            Some(limit),
                            None,
                            Query::All,
                        )
                        .unwrap();
                        assert_eq!(
                            (fast.can_submit, &fast.moves),
                            (full.can_submit, &full.moves),
                            "limit {limit} prefix {prefix:?} at {}",
                            state.show_fen()
                        );
                    }
                    // Single-move and submit queries must agree with the full set,
                    // including for pseudo-legal moves the full set rejects.
                    let search = DraftSearch::new(Arc::new(state.clone()));
                    assert_eq!(search.allows_submit(prefix).unwrap(), full.can_submit);
                    let witness_submit =
                        collect_draft_choices(&state, prefix, budget, Some(0), None, Query::Submit)
                            .unwrap()
                            .can_submit;
                    assert_eq!(witness_submit, full.can_submit, "witness submit");
                    let mut staged = state.clone();
                    for &m in prefix {
                        apply_draft_move(&mut staged, m).unwrap();
                    }
                    let mut targets = Vec::new();
                    for from in staged.get_movable_pieces() {
                        targets.clear();
                        staged.gen_piece_move(from, &mut targets);
                        for &to in &targets {
                            let movement =
                                ExtMove::new(super::super::action::FullMove::new(from, to), 0);
                            let Some(movement) = staged.normalize_promotion(movement) else {
                                continue;
                            };
                            let witness = collect_draft_choices(
                                &state,
                                prefix,
                                budget,
                                Some(0),
                                None,
                                Query::Move(movement),
                            )
                            .unwrap()
                            .moves
                            .contains(&movement);
                            assert_eq!(
                                witness,
                                full.moves.contains(&movement),
                                "witness {movement}"
                            );
                            assert_eq!(
                                search.allows_move(prefix, movement).unwrap(),
                                full.moves.contains(&movement),
                                "{movement} after {prefix:?} at {}",
                                state.show_fen()
                            );
                        }
                    }
                    full.moves
                };
                let root = check(&[]);
                checked += 1;
                for &first in root.iter().step_by(root.len().div_ceil(6).max(1)) {
                    let second = check(&[first]);
                    checked += 1;
                    if let Some(&next) = second.first() {
                        check(&[first, next]);
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 100, "only {checked} drafts checked");
    }
}
