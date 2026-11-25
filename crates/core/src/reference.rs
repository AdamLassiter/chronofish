use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
    time::Duration,
};

use chronofish_5d_rules::core::{
    action::{ExtMove, FullMove},
    referee::{
        PositionStatus, apply_draft_move, draft_choices, play_action, position_status, submit_draft,
    },
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
    choices: OnceLock<Vec<PlayerAction>>,
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
) -> Option<chronofish_5d_rules::core::variants::VariantSetup> {
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
        }
    }

    fn submitted(&self, rules: Ruleset) -> &State {
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

    fn draft_state(&self, rules: Ruleset) -> State {
        let mut state = self.submitted(rules).clone();
        for movement in &self.draft {
            apply_draft_move(&mut state, reference_move(*movement))
                .expect("persisted reference draft");
        }
        state
    }

    pub(crate) fn actions(&self, rules: Ruleset) -> Vec<PlayerAction> {
        self.choices
            .get_or_init(|| self.compute_actions(rules))
            .clone()
    }

    fn compute_actions(&self, rules: Ruleset) -> Vec<PlayerAction> {
        let state = self.submitted(rules);
        let draft = self
            .draft
            .iter()
            .copied()
            .map(reference_move)
            .collect::<Vec<_>>();
        let choices = draft_choices(state, &draft, Duration::MAX).expect("reference draft");
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
        use chronofish_5d_rules::core::action::special as flags;
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
    pub(crate) fn apply_reference(&mut self, action: PlayerAction) -> Result<(), MoveError> {
        if self.result.is_some() {
            return Err(MoveError::GameOver);
        }
        let reference = self.reference.as_mut().expect("reference mode");
        if !reference.actions(self.ruleset).contains(&action) {
            return Err(match action {
                PlayerAction::SubmitTurn => MoveError::CannotSubmit,
                _ => MoveError::IllegalMove,
            });
        }
        let mut origin = None;
        match action {
            PlayerAction::Move { movement } => {
                origin = Some(movement);
                reference.draft.push(movement);
            }
            PlayerAction::SubmitTurn => {
                let next = submit_draft(&reference.draft_state(self.ruleset))
                    .map_err(|_| MoveError::CannotSubmit)?;
                reference.turns.push(std::mem::take(&mut reference.draft));
                reference.submitted = OnceLock::from(Arc::new(next));
            }
        }
        reference.choices.take();
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
        self.refresh_reference(None);
        Ok(())
    }

    pub(crate) fn refresh_reference(&mut self, movement: Option<Move>) {
        let reference = self.reference.as_ref().expect("reference mode");
        let state = reference.draft_state(self.ruleset);
        self.turn = color(state.player());
        let mut timelines = BTreeMap::new();
        let (low, high) = state.get_lines_range();
        let boards = (low..=high).flat_map(|line| {
            (time(state.get_timeline_start(line))..=time(state.get_timeline_end(line)))
                .map(move |ply| (line, ply))
        });
        for (l, new_time) in boards {
            let black = new_time.rem_euclid(2) != 0;
            let t = new_time.div_euclid(2) + 1;
            let Some(board) = state.get_board_ptr(l, t, black) else {
                continue;
            };

            let old = self.board(l, new_time);
            let origin = old
                .map(|b| b.origin.clone())
                .or_else(|| {
                    movement.map(|m| Origin::Move {
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
            timelines
                .entry(l)
                .or_insert_with(|| Timeline {
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
                })
                .boards
                .push(BoardSnapshot {
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
        self.timelines = timelines.into_values().collect();
        self.position_hash = state.hash();
        if !reference.staged() {
            self.result = match position_status(&state, Duration::MAX) {
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
