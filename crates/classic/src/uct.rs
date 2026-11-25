//! Monte Carlsen: UCT tree search with cheap heuristic rollouts and no neural network.
//!
//! Every edge remembers which colour chose it and values are kept from White's point of
//! view, so turns made of several staged actions back up correctly without sign games.

use chronofish_core::{AiDecision, AiPlayer, AiProfile, Color, Game, PlayerAction};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::eval::{capture_score, evaluate};

#[derive(Clone, Copy, Debug)]
pub struct CarlsenConfig {
    /// Tree iterations per decision; each one expands a node and plays one rollout.
    pub iterations: usize,
    /// Rollout length in actions before the static evaluation takes over.
    pub rollout_actions: usize,
    /// UCT exploration constant.
    pub exploration: f64,
    /// Probability that a rollout prefers an available capture over a random action.
    pub capture_bias: f64,
}

impl Default for CarlsenConfig {
    fn default() -> Self {
        Self {
            iterations: 1200,
            rollout_actions: 12,
            exploration: 1.4,
            capture_bias: 0.7,
        }
    }
}

struct Node {
    /// Colour that chose the action leading here; `None` at the root.
    mover: Option<Color>,
    action: Option<PlayerAction>,
    children: Vec<usize>,
    untried: Vec<PlayerAction>,
    visits: u32,
    /// Sum of values from the mover's point of view.
    value: f64,
}

/// UCT Monte Carlo tree search over the handcrafted evaluation.
pub struct MonteCarlsen {
    config: CarlsenConfig,
    rng: ChaCha8Rng,
}

impl MonteCarlsen {
    #[must_use]
    pub fn new(config: CarlsenConfig, seed: u64) -> Self {
        Self {
            config,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    fn white_value(game: &Game) -> Option<f64> {
        game.outcome().map(|outcome| match outcome.winner {
            Some(Color::White) => 1.0,
            Some(Color::Black) => -1.0,
            None => 0.0,
        })
    }

    fn rollout(&mut self, mut game: Game) -> f64 {
        for _ in 0..self.config.rollout_actions {
            if let Some(value) = Self::white_value(&game) {
                return value;
            }
            let actions = game.legal_actions();
            if actions.is_empty() {
                break;
            }
            let captures = actions
                .iter()
                .copied()
                .filter(|&action| capture_score(&game, action).is_some())
                .collect::<Vec<_>>();
            let action = if !captures.is_empty() && self.rng.random_bool(self.config.capture_bias) {
                captures[self.rng.random_range(0..captures.len())]
            } else {
                actions[self.rng.random_range(0..actions.len())]
            };
            if game.apply_action(action).is_err() {
                break;
            }
        }
        Self::white_value(&game).unwrap_or_else(|| (f64::from(evaluate(&game)) / 400.0).tanh())
    }

    fn select(&self, nodes: &[Node], parent: usize) -> usize {
        let log_parent = f64::from(nodes[parent].visits.max(1)).ln();
        let score = |child: usize| {
            let node = &nodes[child];
            let visits = f64::from(node.visits.max(1));
            node.value / visits + self.config.exploration * (log_parent / visits).sqrt()
        };
        nodes[parent]
            .children
            .iter()
            .copied()
            .max_by(|&left, &right| score(left).total_cmp(&score(right)))
            .expect("selection only visits expanded nodes")
    }

    fn untried(&mut self, game: &Game) -> Vec<PlayerAction> {
        if game.outcome().is_some() {
            return Vec::new();
        }
        let mut actions = game.legal_actions();
        actions.shuffle(&mut self.rng);
        actions
    }
}

impl Default for MonteCarlsen {
    fn default() -> Self {
        Self::new(CarlsenConfig::default(), 0x4d41_474e_5553_4d43)
    }
}

impl AiPlayer for MonteCarlsen {
    fn profile(&self) -> AiProfile {
        AiProfile {
            username: "bot-monte-carlsen",
            display_name: "Monte Carlsen",
        }
    }

    fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String> {
        let root_actions = self.untried(game);
        if root_actions.is_empty() {
            return Err("no legal action is available".to_owned());
        }
        let mut nodes = vec![Node {
            mover: None,
            action: None,
            children: Vec::new(),
            untried: root_actions,
            visits: 0,
            value: 0.0,
        }];
        for _ in 0..self.config.iterations.max(1) {
            let mut position = game.clone();
            let mut path = vec![0];
            let mut current = 0;
            while nodes[current].untried.is_empty() && !nodes[current].children.is_empty() {
                current = self.select(&nodes, current);
                let action = nodes[current].action.expect("children carry actions");
                position
                    .apply_action(action)
                    .expect("tree actions were legal when expanded");
                path.push(current);
            }
            while let Some(action) = nodes[current].untried.pop() {
                let mover = position.turn();
                let mut child = position.clone();
                if child.apply_action(action).is_err() {
                    continue;
                }
                let untried = self.untried(&child);
                nodes.push(Node {
                    mover: Some(mover),
                    action: Some(action),
                    children: Vec::new(),
                    untried,
                    visits: 0,
                    value: 0.0,
                });
                let index = nodes.len() - 1;
                nodes[current].children.push(index);
                path.push(index);
                position = child;
                break;
            }
            let value = self.rollout(position);
            for &index in &path {
                let node = &mut nodes[index];
                node.visits += 1;
                node.value += match node.mover {
                    Some(Color::White) => value,
                    Some(Color::Black) => -value,
                    None => 0.0,
                };
            }
        }
        let most_visited = |parent: usize| {
            nodes[parent]
                .children
                .iter()
                .copied()
                .max_by_key(|&child| nodes[child].visits)
        };
        let best = most_visited(0).ok_or_else(|| "search expanded no legal action".to_owned())?;
        let mut principal_variation = Vec::new();
        let mut cursor = Some(best);
        while let Some(index) = cursor {
            principal_variation.push(nodes[index].action.expect("children carry actions"));
            cursor = most_visited(index).filter(|&child| nodes[child].visits > 1);
        }
        let root_value = nodes[best].value / f64::from(nodes[best].visits.max(1));
        Ok(AiDecision {
            action: nodes[best].action.expect("children carry actions"),
            principal_variation,
            root_value: Some(root_value as f32),
        })
    }
}
