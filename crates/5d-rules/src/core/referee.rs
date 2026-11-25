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
    match has_legal_action(state, budget) {
        Some(true) => PositionStatus::Playing,
        None => PositionStatus::Unknown,
        Some(false) if state.has_phantom_check() => PositionStatus::Checkmate,
        Some(false) => match state.exiled_kings_loser() {
            Some(black) => PositionStatus::Exiled { black },
            None => PositionStatus::Stalemate,
        },
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
    let (context, space) = HcInfo::build_hc(submitted);
    let mut complete = true;
    for sequence in context.search(submitted, space) {
        if hypercuboid::interrupted() {
            complete = false;
            break;
        }
        let action = Action::from_seq(&sequence, submitted);
        let mut remaining = action.moves().to_vec();
        for movement in &normalized_draft {
            if let Some(index) = remaining.iter().position(|candidate| candidate == movement) {
                remaining.remove(index);
            } else {
                break;
            }
        }
        if remaining.len() + normalized_draft.len() != action.len() {
            continue;
        }

        if remaining.is_empty() && can_submit
            || (!remaining.is_empty() && remaining.iter().all(|m| moves.contains(m)))
        {
            continue;
        }
        // A generated one-move turn is already a proof at the untouched root.
        if normalized_draft.is_empty() && remaining.len() == 1 {
            moves.insert(remaining[0]);
            continue;
        }
        let expected = play_action(submitted, action.moves())?;
        if remaining.is_empty() {
            can_submit |=
                submit_draft(&position).is_ok_and(|state| state.hash() == expected.hash());
            continue;
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
    }
    Ok(DraftChoices {
        moves: moves.into_iter().collect(),
        can_submit,
        complete: complete && !hypercuboid::interrupted(),
    })
}

fn completion_matches(position: &State, remaining: &mut Vec<ExtMove>, expected: &State) -> bool {
    if hypercuboid::interrupted() {
        return false;
    }
    if remaining.is_empty() {
        return submit_draft(position).is_ok_and(|result| {
            result.hash() == expected.hash()
                && result.get_present() == expected.get_present()
                && result.multiverse().get_boards(true) == expected.multiverse().get_boards(true)
        });
    }
    for index in 0..remaining.len() {
        let mut next = position.clone();
        let movement = remaining.remove(index);
        let matched = apply_draft_move(&mut next, movement).is_ok()
            && completion_matches(&next, remaining, expected);
        remaining.insert(index, movement);
        if matched {
            return true;
        }
    }
    false
}
