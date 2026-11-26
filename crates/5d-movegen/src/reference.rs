use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use chronofish_5d_turngen::core::{
    action::{ExtMove, FullMove},
    referee::{DraftSearch, PositionStatus, apply_draft_move, play_action, submit_draft},
    state::State,
    variants::{default_variant, state_from_setup},
    vec4::Vec4,
};
use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct ReferenceGame {
    turns: Vec<Vec<Move>>,
    draft: Vec<Move>,
    #[serde(skip)]
    submitted: OnceLock<Arc<State>>,
    #[serde(skip)]
    draft_cache: OnceLock<Arc<State>>,
    #[serde(skip)]
    choices: OnceLock<Arc<[PlayerAction]>>,
}

impl std::fmt::Debug for ReferenceGame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReferenceGame")
            .field("turns", &self.turns)
            .field("draft", &self.draft)
            .finish_non_exhaustive()
    }
}

fn color(black: bool) -> Color {
    if black { Color::Black } else { Color::White }
}

fn time(turn: (i32, bool)) -> i32 {
    (turn.0 - 1) * 2 + i32::from(turn.1)
}

fn position(square: Vec4, black: bool) -> Position {
    Position {
        timeline_id: square.l(),
        time: time((square.t(), black)),
        x: square.x(),
        y: square.y(),
    }
}

fn piece(code: u8) -> Option<PieceType> {
    Some(match (code & 127).to_ascii_uppercase() {
        b'K' => PieceType::King,
        b'Q' => PieceType::Queen,
        b'R' => PieceType::Rook,
        b'B' => PieceType::Bishop,
        b'N' => PieceType::Knight,
        b'P' => PieceType::Pawn,
        b'W' => PieceType::Brawn,
        b'S' => PieceType::Princess,
        b'D' => PieceType::Dragon,
        b'U' => PieceType::Unicorn,
        b'Y' => PieceType::RoyalQueen,
        b'C' => PieceType::CommonKing,
        _ => return None,
    })
}

fn reference_move(movement: Move) -> ExtMove {
    let square = |p: Position| Vec4::new(p.x, p.y, p.time.div_euclid(2) + 1, p.timeline_id);
    ExtMove::new(
        FullMove::new(square(movement.from), square(movement.to)),
        movement.promotion.map_or(0, |p| p.notation_letter() as u8),
    )
}

/// The starting position for a Chronofish reference-rules game.
#[must_use]
pub fn reference_setup(
    rules: Ruleset,
) -> Option<chronofish_5d_turngen::core::variants::VariantSetup> {
    if rules == Ruleset::Standard {
        return None;
    }
    let mut setup = default_variant("Standard")?;
    if rules == Ruleset::MultiverseVariant {
        setup.boards[0].fen =
            "sudycdus/w*w*w*w*w*w*w*w*/8/8/8/8/W*W*W*W*W*W*W*W*/SUDYCDUS".to_string();
    }
    Some(setup)
}

impl ReferenceGame {
    pub(crate) fn new() -> Self {
        Self {
            turns: Vec::new(),
            draft: Vec::new(),
            submitted: OnceLock::new(),
            choices: OnceLock::new(),
            draft_cache: OnceLock::new(),
        }
    }

    fn submitted(&self, rules: Ruleset) -> &Arc<State> {
        self.submitted.get_or_init(|| {
            let mut state = state_from_setup(&reference_setup(rules).expect("5D setup"))
                .expect("valid reference setup");
            for moves in &self.turns {
                let moves = moves
                    .iter()
                    .copied()
                    .map(reference_move)
                    .collect::<Vec<_>>();
                state = play_action(&state, &moves).expect("persisted reference turn");
            }
            Arc::new(state)
        })
    }

    fn with_search<T>(&self, rules: Ruleset, query: impl FnOnce(&DraftSearch) -> T) -> T {
        // Cursor scratch is local to the search thread. Sharing the immutable position
        // across Game clones keeps consecutive drafts on the same compiled geometry.
        thread_local! {
            static SEARCH: std::cell::RefCell<Option<DraftSearch>> = const { std::cell::RefCell::new(None) };
        }
        let state = self.submitted(rules);
        SEARCH.with_borrow_mut(|slot| {
            if slot
                .as_ref()
                .is_none_or(|search| !Arc::ptr_eq(search.state(), state))
            {
                *slot = Some(DraftSearch::new(Arc::clone(state)));
            }
            query(slot.as_ref().expect("initialized search"))
        })
    }

    fn draft_state(&self, rules: Ruleset) -> &State {
        if self.draft.is_empty() {
            return self.submitted(rules);
        }
        self.draft_cache.get_or_init(|| {
            let mut state = self.submitted(rules).as_ref().clone();
            for movement in &self.draft {
                apply_draft_move(&mut state, reference_move(*movement))
                    .expect("persisted reference draft");
            }
            Arc::new(state)
        })
    }

    /// Whether `action` is among [`Self::actions`]. Uses the cached choices when they exist;
    /// otherwise proves only this action, which is far cheaper than every choice.
    pub(crate) fn allows(&self, rules: Ruleset, action: PlayerAction) -> bool {
        if let Some(actions) = self.choices.get() {
            return actions.contains(&action);
        }
        let draft = self
            .draft
            .iter()
            .copied()
            .map(reference_move)
            .collect::<Vec<_>>();
        match action {
            PlayerAction::SubmitTurn => self
                .with_search(rules, |search| search.allows_submit(&draft))
                .unwrap_or(false),
            PlayerAction::Move { movement } => {
                let candidate = reference_move(movement);
                // Only the exact spelling `actions` would list: the side's ply parity and
                // an explicit promotion piece where one is required.
                let black = self.submitted(rules).player();
                if position(candidate.fm.from, black) != movement.from
                    || position(candidate.fm.to, black) != movement.to
                    || self
                        .draft_state(rules)
                        .normalize_promotion(candidate)
                        .is_none_or(|normalized| normalized != candidate)
                {
                    return false;
                }
                self.with_search(rules, |search| search.allows_move(&draft, candidate))
                    .unwrap_or(false)
            }
        }
    }

    pub(crate) fn actions(&self, rules: Ruleset) -> &[PlayerAction] {
        self.choices
            .get_or_init(|| self.compute_actions(rules).into())
    }

    fn compute_actions(&self, rules: Ruleset) -> Vec<PlayerAction> {
        let state = self.submitted(rules);
        let draft = self
            .draft
            .iter()
            .copied()
            .map(reference_move)
            .collect::<Vec<_>>();
        let choices = self
            .with_search(rules, |search| search.choices(&draft))
            .expect("reference draft");
        let black = state.player();
        let mut actions = choices
            .moves
            .into_iter()
            .map(|m| PlayerAction::Move {
                movement: Move {
                    from: position(m.fm.from, black),
                    to: position(m.fm.to, black),
                    promotion: piece(m.promote_to),
                },
            })
            .collect::<Vec<_>>();
        if choices.can_submit {
            actions.push(PlayerAction::SubmitTurn);
        }
        actions
    }

    pub(crate) fn describe(&self, rules: Ruleset, movement: Move) -> Option<MoveDescription> {
        let state = self.draft_state(rules);
        let fm = reference_move(movement).fm;
        let kind = fm.move_type(&state);
        use chronofish_5d_turngen::core::action::special as flags;
        Some(MoveDescription {
            piece: piece(fm.moved_piece(&state))?,
            capture: fm.captured_piece(&state) != 0,
            castle: kind & flags::CASTLE != 0,
            creates_timeline: kind & flags::BRANCHING != 0,
        })
    }

    pub(crate) fn staged(&self) -> bool {
        !self.draft.is_empty()
    }
    pub(crate) fn present(&self, rules: Ruleset, submitted: bool) -> i32 {
        if submitted {
            time(self.submitted(rules).get_present())
        } else {
            time(self.draft_state(rules).apparent_present())
        }
    }
    pub(crate) fn active(&self, rules: Ruleset, line: i32) -> bool {
        let (first, last) = self.draft_state(rules).get_active_range();
        (first..=last).contains(&line)
    }

    pub(crate) fn checks(&self, rules: Ruleset) -> Vec<Position> {
        let state = self.draft_state(rules);
        [false, true]
            .into_iter()
            .flat_map(|black| {
                state
                    .find_checks(black)
                    .into_iter()
                    .map(move |m| position(m.to, black))
            })
            .collect()
    }
}

impl Game {
    /// A copy whose reference choices must be recomputed, for testing uncached paths.
    #[cfg(test)]
    pub(crate) fn clone_without_reference_cache(&self) -> Self {
        let mut copy = self.clone();
        if let Some(reference) = copy.reference.as_mut() {
            reference.choices.take();
        }
        copy
    }

    pub(crate) fn apply_reference(&mut self, action: PlayerAction) -> Result<(), MoveError> {
        if self.result.is_some() {
            return Err(MoveError::GameOver);
        }
        let reference = self.reference.as_ref().expect("reference mode");
        if !reference.allows(self.ruleset, action) {
            return Err(match action {
                PlayerAction::SubmitTurn => MoveError::CannotSubmit,
                _ => MoveError::IllegalMove,
            });
        }
        self.stage_reference(action)
    }

    /// Applies `action` without proving it is a listed choice. The referee still
    /// rejects a submission that does not advance the Present or leaves a royal in check.
    fn stage_reference(&mut self, action: PlayerAction) -> Result<(), MoveError> {
        let reference = self.reference.as_mut().expect("reference mode");
        let mut origin = None;
        match action {
            PlayerAction::Move { movement } => {
                origin = Some(movement);
                reference.draft.push(movement);
            }
            PlayerAction::SubmitTurn => {
                let next = submit_draft(reference.draft_state(self.ruleset))
                    .map_err(|_| MoveError::CannotSubmit)?;
                reference.turns.push(std::mem::take(&mut reference.draft));
                reference.submitted = OnceLock::from(Arc::new(next));
            }
        }
        reference.choices.take();
        reference.draft_cache.take();
        // This prefix already has a proven completion. If every source
        // board has advanced, that completion can only be submission.
        if origin.is_some() {
            let state = reference.draft_state(self.ruleset);
            let (lo, hi) = state.get_lines_range();
            if (lo..=hi).all(|l| state.get_timeline_end(l).1 != state.player()) {
                reference.choices = OnceLock::from(Arc::from([PlayerAction::SubmitTurn]));
            }
        }
        self.refresh_reference(origin);
        Ok(())
    }

    pub(crate) fn undo_reference(&mut self) -> Result<(), MoveError> {
        if self.result.is_some() {
            return Err(MoveError::GameOver);
        }
        let reference = self.reference.as_mut().expect("reference mode");
        reference.draft.pop().ok_or(MoveError::IllegalMove)?;
        reference.choices.take();
        reference.draft_cache.take();
        self.refresh_reference(None);
        Ok(())
    }

    pub(crate) fn refresh_reference(&mut self, movement: Option<Move>) {
        let reference = self.reference.as_ref().expect("reference mode");
        let state = reference.draft_state(self.ruleset);
        self.turn = color(state.player());
        let (low, high) = state.get_lines_range();
        let mut previous = std::mem::take(&mut self.timelines)
            .into_iter()
            .map(|timeline| (timeline.id, timeline))
            .collect::<BTreeMap<_, _>>();
        for l in low..=high {
            let mut timeline = previous.remove(&l).unwrap_or_else(|| Timeline {
                id: l,
                row: l,
                label: format!("T{l}"),
                owner: match l.cmp(&0) {
                    std::cmp::Ordering::Less => TimelineOwner::Black,
                    std::cmp::Ordering::Greater => TimelineOwner::White,
                    std::cmp::Ordering::Equal => TimelineOwner::Neutral,
                },
                accepted: false,
                boards: Vec::new(),
            });
            let end = time(state.get_timeline_end(l));
            // Reference moves append boards. Undo removes a suffix of the timeline.
            timeline
                .boards
                .truncate(timeline.boards.partition_point(|b| b.time <= end));
            let start = timeline
                .boards
                .last()
                .map_or_else(|| time(state.get_timeline_start(l)), |b| b.time + 1);
            for new_time in start..=end {
                let black = new_time.rem_euclid(2) != 0;
                let t = new_time.div_euclid(2) + 1;
                let Some(board) = state.get_board_ptr(l, t, black) else {
                    continue;
                };

                let origin = movement
                    .map(|m| Origin::Move {
                        from: m.from,
                        to: m.to,
                        move_type: if m.from.timeline_id == m.to.timeline_id
                            && m.from.time == m.to.time
                        {
                            "standard"
                        } else {
                            "branch"
                        }
                        .to_owned(),
                    })
                    .unwrap_or(Origin::None);
                let cells = std::array::from_fn(|y| {
                    std::array::from_fn(|x| {
                        let code = board.get_piece((y * 8 + x) as i32);
                        piece(code).map(|piece_type| Piece {
                            color: if (code & 127).is_ascii_lowercase() {
                                Color::Black
                            } else {
                                Color::White
                            },
                            piece_type,
                        })
                    })
                });
                timeline.boards.push(BoardSnapshot {
                    time: new_time,
                    side_to_move: color(black),
                    board: cells,
                    castling: CastlingRights {
                        white_kingside: board.umove() & (1 << 4 | 1 << 7) == (1 << 4 | 1 << 7),
                        white_queenside: board.umove() & (1 << 4 | 1) == (1 << 4 | 1),
                        black_kingside: board.umove() & (1 << 60 | 1 << 63) == (1 << 60 | 1 << 63),
                        black_queenside: board.umove() & (1 << 60 | 1 << 56) == (1 << 60 | 1 << 56),
                    },
                    en_passant: None,
                    origin,
                    halfmove_clock: 0,
                });
            }
            if !timeline.boards.is_empty() {
                self.timelines.push(timeline);
            }
        }
        self.position_hash = state.hash();
        if !reference.staged() {
            self.result = match reference.with_search(self.ruleset, DraftSearch::status) {
                PositionStatus::Checkmate => Some(GameOutcome {
                    winner: Some(self.turn.opposite()),
                    reason: OutcomeReason::Checkmate,
                }),
                PositionStatus::Stalemate => Some(GameOutcome {
                    winner: None,
                    reason: OutcomeReason::Stalemate,
                }),
                PositionStatus::Exiled { black } => Some(GameOutcome {
                    winner: Some(color(!black)),
                    reason: OutcomeReason::NoLegalTurn,
                }),
                _ => None,
            };
        }
        self.last_message = if reference.staged() {
            "Turn in progress.".into()
        } else {
            format!("{} to move.", self.turn)
        };
    }
}

/// A verified whole-turn decision, returned one staged action at a time.
#[derive(Default)]
pub struct ReferenceBotTurn {
    pending: std::collections::VecDeque<(u64, PlayerAction)>,
}

impl ReferenceBotTurn {
    pub fn choose(
        &mut self,
        game: &Game,
        search: impl FnOnce(&State) -> Option<Vec<ExtMove>>,
    ) -> Result<AiDecision, String> {
        let reference = game
            .reference
            .as_ref()
            .ok_or("this bot requires reference 5D rules")?;
        if game.outcome().is_some() {
            return Err("the game is already over".into());
        }
        if self
            .pending
            .front()
            .is_none_or(|(hash, _)| *hash != game.position_hash())
        {
            self.pending.clear();
            let state = reference.submitted(game.ruleset());
            // Continue from authoritative choices when a draft was restored or the search ran out
            // of time before returning a turn.
            let moves = if reference.staged() {
                None
            } else {
                search(state)
            };
            let Some(moves) = moves else {
                let actions = game.legal_actions();
                let action = if actions.contains(&PlayerAction::SubmitTurn) {
                    Some(PlayerAction::SubmitTurn)
                } else {
                    actions.first().copied()
                }
                .ok_or("the position has no legal continuation")?;
                return Ok(AiDecision {
                    action,
                    principal_variation: vec![action],
                    root_value: None,
                });
            };
            let mut probe = game.clone();
            let mut pending = std::collections::VecDeque::new();
            for movement in moves {
                let action = PlayerAction::Move {
                    movement: Move {
                        from: position(movement.fm.from, state.player()),
                        to: position(movement.fm.to, state.player()),
                        promotion: piece(movement.promote_to),
                    },
                };
                pending.push_back((probe.position_hash(), action));
                // The game proves each action when the bot plays it; the probe only
                // records the positions in between.
                if let PlayerAction::Move { movement } = action {
                    apply_draft_move(
                        &mut probe
                            .reference
                            .as_ref()
                            .expect("reference mode")
                            .draft_state(game.ruleset())
                            .clone(),
                        reference_move(movement),
                    )
                    .map_err(|error| error.to_string())?;
                }
                probe
                    .stage_reference(action)
                    .map_err(|error| error.to_string())?;
            }
            pending.push_back((probe.position_hash(), PlayerAction::SubmitTurn));
            probe
                .stage_reference(PlayerAction::SubmitTurn)
                .map_err(|error| error.to_string())?;
            self.pending = pending;
        }
        let principal_variation = self.pending.iter().map(|(_, action)| *action).collect();
        let (_, action) = self.pending.pop_front().ok_or("missing verified action")?;
        Ok(AiDecision {
            action,
            principal_variation,
            root_value: None,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiDecision {
    pub action: PlayerAction,
    #[serde(default)]
    pub principal_variation: Vec<PlayerAction>,
    pub root_value: Option<f32>,
}
