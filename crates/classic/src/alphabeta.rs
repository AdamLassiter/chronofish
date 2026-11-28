//! Deep Blueshift: a classic alpha-beta engine racing towards the present.
//!
//! Search runs over *actions* rather than plies. A 5D turn is several staged moves
//! followed by `SubmitTurn`, while standard chess submits automatically, so the score
//! is negated only when an action hands the turn to the opponent.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use chronofish_core::{AiDecision, AiPlayer, AiProfile, Color, Game, PlayerAction};

use crate::eval::{MATE, MATE_THRESHOLD, capture_score, relative, squash};

const INFINITY: i32 = MATE + 1;
const TABLE_LIMIT: usize = 1 << 20;

#[derive(Clone, Copy, Debug)]
pub struct BlueshiftConfig {
    /// Hard cap on nodes per decision; keeps move time bounded and deterministic.
    pub max_nodes: usize,
    /// Deepest iterative-deepening iteration, in actions.
    pub max_depth: u32,
    /// Extra capture-only actions searched beyond the nominal depth.
    pub quiescence_depth: u32,
    /// Optional wall-clock cap per decision, for interactive callers. Bots leave it
    /// unset so that their play stays deterministic.
    pub time_limit: Option<Duration>,
}

impl Default for BlueshiftConfig {
    fn default() -> Self {
        Self {
            max_nodes: 80_000,
            max_depth: 8,
            quiescence_depth: 4,
            time_limit: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct Entry {
    depth: u32,
    score: i32,
    bound: Bound,
    best: Option<PlayerAction>,
}

type TableKey = (u64, Color);

fn key(game: &Game) -> TableKey {
    (game.position_hash(), game.turn())
}

/// Difficulty presets: the same engine with a smaller or larger node budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlueshiftLevel {
    Easy,
    Medium,
    Hard,
}

impl BlueshiftLevel {
    pub const ALL: [Self; 3] = [Self::Easy, Self::Medium, Self::Hard];

    /// Node budget per decision. Hard is the default configuration.
    #[must_use]
    pub const fn max_nodes(self) -> usize {
        match self {
            Self::Easy => 1_000,
            Self::Medium => 10_000,
            Self::Hard => 80_000,
        }
    }

    #[must_use]
    pub const fn profile(self) -> AiProfile {
        match self {
            Self::Easy => AiProfile {
                username: "bot-deep-blueshift-easy",
                display_name: "Deep Blueshift (easy)",
            },
            Self::Medium => AiProfile {
                username: "bot-deep-blueshift-medium",
                display_name: "Deep Blueshift (medium)",
            },
            Self::Hard => AiProfile {
                username: "bot-deep-blueshift",
                display_name: "Deep Blueshift",
            },
        }
    }
}

/// Iterative-deepening negamax with a transposition table and quiescence search.
pub struct DeepBlueshift {
    config: BlueshiftConfig,
    profile: AiProfile,
    table: HashMap<TableKey, Entry>,
}

impl DeepBlueshift {
    /// A custom configuration, playing under the Hard account.
    #[must_use]
    pub fn new(config: BlueshiftConfig) -> Self {
        Self {
            config,
            profile: BlueshiftLevel::Hard.profile(),
            table: HashMap::new(),
        }
    }

    /// A difficulty preset with its own account.
    #[must_use]
    pub fn level(level: BlueshiftLevel) -> Self {
        Self {
            profile: level.profile(),
            ..Self::new(BlueshiftConfig {
                max_nodes: level.max_nodes(),
                ..BlueshiftConfig::default()
            })
        }
    }

    /// Keeps a preset's account while overriding its node budget.
    #[must_use]
    pub const fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.config.max_nodes = max_nodes;
        self
    }

    /// Also stops each search once `limit` has elapsed, keeping the deepest finished result.
    #[must_use]
    pub const fn with_time_limit(mut self, limit: Duration) -> Self {
        self.config.time_limit = Some(limit);
        self
    }
}

impl Default for DeepBlueshift {
    fn default() -> Self {
        Self::new(BlueshiftConfig::default())
    }
}

struct Search<'a> {
    config: BlueshiftConfig,
    table: &'a mut HashMap<TableKey, Entry>,
    nodes: usize,
    aborted: bool,
    deadline: Option<Instant>,
}

#[derive(Clone, Copy)]
enum Mode {
    Full(u32),
    Quiet(u32),
}

impl Search<'_> {
    fn tick(&mut self) -> bool {
        self.nodes += 1;
        // Reading the clock every node would cost more than a cheap node.
        if self.nodes > self.config.max_nodes
            || (self.nodes.is_multiple_of(64)
                && self.deadline.is_some_and(|end| Instant::now() >= end))
        {
            self.aborted = true;
        }
        self.aborted
    }

    /// Orders actions: table move, captures by MVV-LVA, quiet moves, then submission.
    fn ordered(
        game: &Game,
        preferred: Option<PlayerAction>,
        captures_only: bool,
    ) -> Vec<PlayerAction> {
        let mut scored = game
            .legal_actions()
            .into_iter()
            .filter_map(|action| {
                let capture = capture_score(game, action);
                if captures_only && capture.is_none() {
                    return None;
                }
                let order = if Some(action) == preferred {
                    i32::MAX
                } else if let Some(capture) = capture {
                    1_000_000 + capture
                } else if action == PlayerAction::SubmitTurn {
                    -1
                } else {
                    0
                };
                Some((order, action))
            })
            .collect::<Vec<_>>();
        // Stable sort keeps move generation order among equals, so search stays deterministic.
        scored.sort_by_key(|(order, _)| std::cmp::Reverse(*order));
        scored.into_iter().map(|(_, action)| action).collect()
    }

    /// Score of playing `action` from `game`, from the mover's point of view.
    fn child(
        &mut self,
        game: &Game,
        action: PlayerAction,
        mode: Mode,
        ply: u32,
        alpha: i32,
        beta: i32,
    ) -> Option<i32> {
        let mover = game.turn();
        let mut child = game.clone();
        child.apply_action(action).ok()?;
        let ply_penalty = i32::try_from(ply + 1).unwrap_or(i32::MAX);
        if let Some(outcome) = child.outcome() {
            return Some(match outcome.winner {
                Some(winner) if winner == mover => MATE - ply_penalty,
                Some(_) => -(MATE - ply_penalty),
                None => 0,
            });
        }
        let same_side = child.turn() == mover;
        let (child_alpha, child_beta) = if same_side {
            (alpha, beta)
        } else {
            (-beta, -alpha)
        };
        let score = match mode {
            Mode::Full(depth) => self.negamax(&child, depth, ply + 1, child_alpha, child_beta),
            Mode::Quiet(depth) => self.quiescence(&child, depth, ply + 1, child_alpha, child_beta),
        };
        Some(if same_side { score } else { -score })
    }

    fn quiescence(&mut self, game: &Game, depth: u32, ply: u32, mut alpha: i32, beta: i32) -> i32 {
        if self.tick() {
            return 0;
        }
        let stand = relative(game, game.turn());
        if stand >= beta || depth == 0 {
            return stand;
        }
        alpha = alpha.max(stand);
        let mut best = stand;
        for action in Self::ordered(game, None, true) {
            let Some(score) = self.child(game, action, Mode::Quiet(depth - 1), ply, alpha, beta)
            else {
                continue;
            };
            if self.aborted {
                return 0;
            }
            best = best.max(score);
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        best
    }

    fn negamax(&mut self, game: &Game, depth: u32, ply: u32, mut alpha: i32, mut beta: i32) -> i32 {
        if depth == 0 {
            return self.quiescence(game, self.config.quiescence_depth, ply, alpha, beta);
        }
        if self.tick() {
            return 0;
        }
        let key = key(game);
        let entry = self.table.get(&key).copied();
        if let Some(entry) = entry.filter(|entry| entry.depth >= depth) {
            match entry.bound {
                Bound::Exact => return entry.score,
                Bound::Lower => alpha = alpha.max(entry.score),
                Bound::Upper => beta = beta.min(entry.score),
            }
            if alpha >= beta {
                return entry.score;
            }
        }
        let original_alpha = alpha;
        let mut best = -INFINITY;
        let mut best_action = None;
        for action in Self::ordered(game, entry.and_then(|entry| entry.best), false) {
            let Some(score) = self.child(game, action, Mode::Full(depth - 1), ply, alpha, beta)
            else {
                continue;
            };
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
                best_action = Some(action);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        if best_action.is_none() {
            // No playable action but no recorded outcome: fall back to the static view.
            return relative(game, game.turn());
        }
        let bound = if best <= original_alpha {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        if self.table.len() >= TABLE_LIMIT {
            self.table.clear();
        }
        self.table.insert(
            key,
            Entry {
                depth,
                score: best,
                bound,
                best: best_action,
            },
        );
        best
    }

    /// One root iteration. Returns the best action and score, or `None` if aborted.
    fn root(
        &mut self,
        game: &Game,
        depth: u32,
        previous: Option<PlayerAction>,
    ) -> Option<(PlayerAction, i32)> {
        let mut alpha = -INFINITY;
        let mut best = None;
        for action in Self::ordered(game, previous, false) {
            let Some(score) = self.child(game, action, Mode::Full(depth - 1), 0, alpha, INFINITY)
            else {
                continue;
            };
            if self.aborted {
                // A partial first iteration still beats an arbitrary fallback.
                return if previous.is_none() { best } else { None };
            }
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((action, score));
            }
            alpha = alpha.max(score);
        }
        if let Some((action, score)) = best {
            self.table.insert(
                key(game),
                Entry {
                    depth,
                    score,
                    bound: Bound::Exact,
                    best: Some(action),
                },
            );
        }
        best
    }
}

impl DeepBlueshift {
    fn principal_variation(
        &self,
        game: &Game,
        first: PlayerAction,
        limit: u32,
    ) -> Vec<PlayerAction> {
        let mut line = vec![first];
        let mut position = game.clone();
        if position.apply_action(first).is_err() {
            return line;
        }
        while line.len() < limit as usize && position.outcome().is_none() {
            let Some(action) = self.table.get(&key(&position)).and_then(|entry| entry.best) else {
                break;
            };
            if position.apply_action(action).is_err() {
                break;
            }
            line.push(action);
        }
        line
    }
}

impl AiPlayer for DeepBlueshift {
    fn profile(&self) -> AiProfile {
        self.profile
    }

    fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String> {
        let fallback = game
            .legal_actions()
            .first()
            .copied()
            .ok_or_else(|| "no legal action is available".to_owned())?;
        let (action, score, depth) = self.search(game).unwrap_or((fallback, 0, 1));
        Ok(AiDecision {
            action,
            principal_variation: self.principal_variation(game, action, depth),
            root_value: Some(squash(score)),
        })
    }
}

impl DeepBlueshift {
    /// Searches `game` and scores it in centipawns for the side to move.
    ///
    /// Scores at or beyond [`MATE_THRESHOLD`] in magnitude are forced results. If the
    /// budget runs out before the first iteration finishes, this falls back to the
    /// static evaluation. A finished game scores `None`.
    pub fn evaluate(&mut self, game: &Game) -> Option<i32> {
        if game.outcome().is_some() {
            return None;
        }
        Some(
            self.search(game)
                .map_or_else(|| relative(game, game.turn()), |(_, score, _)| score),
        )
    }

    /// Iterative deepening up to the configured depth and node budget.
    fn search(&mut self, game: &Game) -> Option<(PlayerAction, i32, u32)> {
        let mut search = Search {
            config: self.config,
            table: &mut self.table,
            nodes: 0,
            aborted: false,
            deadline: self.config.time_limit.map(|limit| Instant::now() + limit),
        };
        let mut best: Option<(PlayerAction, i32, u32)> = None;
        for depth in 1..=self.config.max_depth.max(1) {
            match search.root(game, depth, best.map(|(action, _, _)| action)) {
                Some((action, score)) => {
                    best = Some((action, score, depth));
                    if search.aborted || score.abs() >= MATE_THRESHOLD {
                        break;
                    }
                }
                None => break,
            }
        }
        best
    }
}
