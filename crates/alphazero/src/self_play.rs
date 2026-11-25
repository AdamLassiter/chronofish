use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use chronofish_core::{Game, GameOutcome, PlayerAction, Ruleset, Side};
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::encoding::encode;
use crate::network::{PolicyValueEvaluator, PolicyValueNetwork, TrainingExample};
use crate::search::{Mcts, SearchConfig};

#[derive(Clone, Copy, Debug)]
pub struct SelfPlayConfig {
    pub search: SearchConfig,
    pub max_actions: usize,
    pub exploration_actions: usize,
    pub temperature: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            search: SearchConfig {
                simulations: 64,
                ..SearchConfig::default()
            },
            max_actions: 600,
            exploration_actions: 30,
            temperature: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SelfPlayGame {
    pub ruleset: Ruleset,
    pub steps: Vec<ReplayStep>,
    pub outcome: Option<GameOutcome>,
    pub truncated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayStep {
    pub action: PlayerAction,
    pub policy: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReplayGame {
    #[serde(skip)]
    validated: std::sync::OnceLock<()>,
    ruleset: Ruleset,
    steps: Vec<ReplayStep>,
    outcome: Option<GameOutcome>,
    truncated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayBuffer {
    format_version: u32,
    games: Vec<ReplayGame>,
}

#[derive(Debug)]
pub struct SampledReplay {
    pub examples: Vec<TrainingExample>,
    pub estimated_bytes: usize,
    pub skipped_for_memory: usize,
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self {
            format_version: 2,
            games: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaReport {
    pub candidate_wins: usize,
    pub incumbent_wins: usize,
    pub draws: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaProgress {
    pub game: usize,
    pub games: usize,
    pub ruleset: Ruleset,
    pub candidate_side: Side,
    pub actions: usize,
    pub completed: bool,
    pub outcome: Option<GameOutcome>,
}

#[derive(Clone, Copy, Debug)]
pub struct ArenaGameConfig {
    pub ruleset: Ruleset,
    pub candidate_side: Side,
    pub search: SearchConfig,
    pub max_actions: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaGameResult {
    pub outcome: Option<GameOutcome>,
    pub actions: usize,
}

impl ArenaReport {
    /// Record terminal draws and action-limit draws without crediting either player.
    pub fn record_outcome(&mut self, candidate_side: Side, outcome: Option<GameOutcome>) {
        match outcome.and_then(|outcome| outcome.winner) {
            Some(winner) if winner == candidate_side => self.candidate_wins += 1,
            Some(_) => self.incumbent_wins += 1,
            None => self.draws += 1,
        }
    }

    #[must_use]
    pub fn candidate_score(self) -> f32 {
        let games = self.candidate_wins + self.incumbent_wins + self.draws;
        if games == 0 {
            0.0
        } else {
            (self.candidate_wins as f32 + self.draws as f32 * 0.5) / games as f32
        }
    }
}

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("replay I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("replay data is invalid: {0}")]
    Codec(String),
    #[error("unsupported replay format version {0}")]
    Version(u32),
    #[error("replay trajectory is invalid: {0}")]
    Trajectory(String),
    #[error(
        "a replay position cannot fit the reconstruction memory budget: estimated={estimated_mib} MiB, budget={budget_mib} MiB"
    )]
    MemoryBudget {
        estimated_mib: usize,
        budget_mib: usize,
    },
}

#[must_use]
pub fn play_self_play_game<E: PolicyValueEvaluator + ?Sized>(
    model: &E,
    ruleset: Ruleset,
    config: SelfPlayConfig,
    rng: &mut impl Rng,
) -> SelfPlayGame {
    play_self_play_game_with_progress(model, ruleset, config, rng, |_| {})
}

/// Plays one self-play game, reporting the number of recorded actions after each one.
#[must_use]
pub fn play_self_play_game_with_progress<E: PolicyValueEvaluator + ?Sized>(
    model: &E,
    ruleset: Ruleset,
    config: SelfPlayConfig,
    rng: &mut impl Rng,
    mut progress: impl FnMut(usize),
) -> SelfPlayGame {
    let mut game = Game::new(ruleset);
    let mut steps = Vec::new();
    for action_index in 0..config.max_actions {
        if game.outcome().is_some() {
            break;
        }
        let search = Mcts::new(model, config.search).search(&game, true, rng);
        if search.actions.is_empty() {
            break;
        }
        let temperature = if action_index < config.exploration_actions {
            config.temperature
        } else {
            0.0
        };
        let Some(action) = search.sample_action(temperature, rng) else {
            break;
        };
        steps.push(ReplayStep {
            action,
            policy: search.policy,
        });
        if game.apply_action(action).is_err() {
            break;
        }
        progress(steps.len());
    }
    let outcome = game.outcome();
    SelfPlayGame {
        ruleset,
        steps,
        outcome,
        truncated: outcome.is_none(),
    }
}

#[must_use]
pub fn run_arena(
    candidate: &PolicyValueNetwork,
    incumbent: &PolicyValueNetwork,
    ruleset: Ruleset,
    games: usize,
    search: SearchConfig,
    max_actions: usize,
    rng: &mut impl Rng,
) -> ArenaReport {
    run_arena_schedule(
        candidate,
        incumbent,
        &vec![ruleset; games],
        search,
        max_actions,
        rng,
    )
}

#[must_use]
pub fn run_arena_schedule(
    candidate: &PolicyValueNetwork,
    incumbent: &PolicyValueNetwork,
    rulesets: &[Ruleset],
    search: SearchConfig,
    max_actions: usize,
    rng: &mut impl Rng,
) -> ArenaReport {
    run_arena_schedule_with_progress(
        candidate,
        incumbent,
        rulesets,
        search,
        max_actions,
        rng,
        |_| {},
    )
}

#[must_use]
pub fn run_arena_schedule_with_progress(
    candidate: &PolicyValueNetwork,
    incumbent: &PolicyValueNetwork,
    rulesets: &[Ruleset],
    search: SearchConfig,
    max_actions: usize,
    rng: &mut impl Rng,
    mut progress: impl FnMut(ArenaProgress),
) -> ArenaReport {
    let mut report = ArenaReport::default();
    for (game_index, &ruleset) in rulesets.iter().enumerate() {
        let candidate_side = if game_index % 2 == 0 {
            Side::White
        } else {
            Side::Black
        };
        let mut actions = 0;
        progress(ArenaProgress {
            game: game_index + 1,
            games: rulesets.len(),
            ruleset,
            candidate_side,
            actions,
            completed: false,
            outcome: None,
        });
        let result = play_arena_game(
            candidate,
            incumbent,
            ArenaGameConfig {
                ruleset,
                candidate_side,
                search,
                max_actions,
            },
            rng,
            |current_actions, outcome| {
                actions = current_actions;
                progress(ArenaProgress {
                    game: game_index + 1,
                    games: rulesets.len(),
                    ruleset,
                    candidate_side,
                    actions,
                    completed: false,
                    outcome,
                });
            },
        );
        report.record_outcome(candidate_side, result.outcome);
        progress(ArenaProgress {
            game: game_index + 1,
            games: rulesets.len(),
            ruleset,
            candidate_side,
            actions: result.actions,
            completed: true,
            outcome: result.outcome,
        });
    }
    report
}

#[must_use]
pub fn play_arena_game<E: PolicyValueEvaluator + ?Sized>(
    candidate: &E,
    incumbent: &E,
    config: ArenaGameConfig,
    rng: &mut impl Rng,
    progress: impl FnMut(usize, Option<GameOutcome>),
) -> ArenaGameResult {
    play_arena_from_position(
        candidate,
        incumbent,
        Game::new(config.ruleset),
        config,
        rng,
        progress,
    )
}

/// Plays an arena game from a supplied opening with the same search as the trainer.
#[must_use]
pub fn play_arena_from_position<E: PolicyValueEvaluator + ?Sized>(
    candidate: &E,
    incumbent: &E,
    mut game: Game,
    config: ArenaGameConfig,
    rng: &mut impl Rng,
    mut progress: impl FnMut(usize, Option<GameOutcome>),
) -> ArenaGameResult {
    let mut actions = 0;
    for _ in 0..config.max_actions {
        if game.outcome().is_some() {
            break;
        }
        let model = if game.turn() == config.candidate_side {
            candidate
        } else {
            incumbent
        };
        let result = Mcts::new(model, config.search).search(&game, false, rng);
        let Some(action) = result.best_action() else {
            break;
        };
        if game.apply_action(action).is_err() {
            break;
        }
        actions += 1;
        progress(actions, game.outcome());
    }
    ArenaGameResult {
        outcome: game.outcome(),
        actions,
    }
}

impl ReplayStep {
    fn checked_actions(&self, game: &Game, index: usize) -> Result<Vec<PlayerAction>, ReplayError> {
        let actions = game.legal_actions();
        if actions.len() != self.policy.len() {
            return Err(ReplayError::Trajectory(format!(
                "step {index} has {} policy entries for {} legal actions",
                self.policy.len(),
                actions.len()
            )));
        }
        if !actions.contains(&self.action) {
            return Err(ReplayError::Trajectory(format!(
                "step {index} selects an action that is no longer legal"
            )));
        }
        Ok(actions)
    }

    fn apply_to(&self, game: &mut Game, index: usize) -> Result<(), ReplayError> {
        game.apply_action(self.action).map_err(|error| {
            ReplayError::Trajectory(format!("step {index} cannot be applied: {error}"))
        })
    }
}

impl ReplayGame {
    fn validate(&self) -> Result<(), ReplayError> {
        let mut game = Game::new(self.ruleset);
        for (index, step) in self.steps.iter().enumerate() {
            step.checked_actions(&game, index)?;
            step.apply_to(&mut game, index)?;
        }
        let _ = self.validated.set(());
        Ok(())
    }
}

impl ReplayBuffer {
    #[must_use]
    pub fn len(&self) -> usize {
        self.games.iter().map(|game| game.steps.len()).sum()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.games.is_empty()
    }

    /// Reconstructs encoded training positions from compact authoritative trajectories.
    ///
    /// # Errors
    ///
    /// Returns an error if a stored policy/action no longer matches authoritative rules.
    pub fn examples(&self) -> Result<Vec<TrainingExample>, ReplayError> {
        let mut rng = rand::rng();
        Ok(self
            .sample_examples(self.len(), usize::MAX, &mut rng)?
            .examples)
    }

    /// Reconstructs a random, host-memory-bounded sample of replay positions.
    /// Each trajectory is validated in full on its first sample pass.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid trajectories or when one selected position
    /// cannot fit the memory budget by itself.
    pub fn sample_examples(
        &self,
        max_examples: usize,
        max_bytes: usize,
        rng: &mut impl Rng,
    ) -> Result<SampledReplay, ReplayError> {
        self.sample_examples_impl(max_examples, max_bytes, rng, None)
    }

    /// Samples in the same order as the serial sampler, using the supplied pool
    /// for first-use trajectory checks.
    ///
    /// # Errors
    ///
    /// Returns the same trajectory or memory errors as [`Self::sample_examples`].
    pub fn sample_examples_in_pool(
        &self,
        max_examples: usize,
        max_bytes: usize,
        rng: &mut impl Rng,
        pool: &rayon::ThreadPool,
    ) -> Result<SampledReplay, ReplayError> {
        self.sample_examples_impl(max_examples, max_bytes, rng, Some(pool))
    }

    fn sample_examples_impl(
        &self,
        max_examples: usize,
        max_bytes: usize,
        rng: &mut impl Rng,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<SampledReplay, ReplayError> {
        let mut selected = (0..self.len()).collect::<Vec<_>>();
        selected.shuffle(rng);
        selected.truncate(max_examples.min(selected.len()));
        let mut selected = selected.into_iter().collect::<HashSet<_>>();
        let mut game_order = (0..self.games.len()).collect::<Vec<_>>();
        if max_examples < self.len() || max_bytes != usize::MAX {
            game_order.shuffle(rng);
        }
        if let Some(pool) = pool.filter(|pool| pool.current_num_threads() > 1) {
            let unchecked: Vec<_> = game_order
                .iter()
                .map(|&i| &self.games[i])
                .filter(|g| g.validated.get().is_none())
                .collect();
            if unchecked.len() > 1 {
                let first_error = pool.install(|| {
                    unchecked
                        .par_iter()
                        .find_map_first(|replay| replay.validate().err())
                });
                if let Some(error) = first_error {
                    return Err(error);
                }
            }
        }
        let mut game_offsets = Vec::with_capacity(self.games.len());
        let mut offset = 0_usize;
        for replay in &self.games {
            game_offsets.push(offset);
            offset = offset.saturating_add(replay.steps.len());
        }
        let mut examples = Vec::with_capacity(max_examples.min(self.len()));
        let mut estimated_bytes = 0_usize;
        let mut skipped_for_memory = 0_usize;
        let mut smallest_rejected = usize::MAX;
        for game_index in game_order {
            let replay = &self.games[game_index];
            let validated = replay.validated.get().is_some();
            let count = if validated {
                (0..replay.steps.len())
                    .rfind(|i| selected.contains(&(game_offsets[game_index] + i)))
                    .map_or(0, |i| i + 1)
            } else {
                replay.steps.len()
            };
            if count == 0 {
                continue;
            }
            let mut game = Game::new(replay.ruleset);
            for (index, step) in replay.steps[..count].iter().enumerate() {
                let global_index = game_offsets[game_index].saturating_add(index);
                let sampled = selected.remove(&global_index);
                if sampled || !validated {
                    let actions = step.checked_actions(&game, index)?;
                    if sampled {
                        let estimate = estimate_example_bytes(&game, actions.len());
                        if estimated_bytes.saturating_add(estimate) <= max_bytes {
                            let side = game.turn();
                            examples.push(TrainingExample {
                                position: encode(&game, &actions),
                                policy: step.policy.clone(),
                                value: replay.outcome.map_or(0.0, |outcome| {
                                    outcome.winner.map_or(0.0, |winner| {
                                        if winner == side { 1.0 } else { -1.0 }
                                    })
                                }),
                            });
                            estimated_bytes = estimated_bytes.saturating_add(estimate);
                        } else {
                            skipped_for_memory += 1;
                            smallest_rejected = smallest_rejected.min(estimate);
                        }
                    }
                }
                step.apply_to(&mut game, index)?;
            }
            let _ = replay.validated.set(());
        }
        if examples.is_empty() && skipped_for_memory > 0 {
            return Err(ReplayError::MemoryBudget {
                estimated_mib: bytes_to_mib_ceil(smallest_rejected),
                budget_mib: bytes_to_mib_ceil(max_bytes),
            });
        }
        Ok(SampledReplay {
            examples,
            estimated_bytes,
            skipped_for_memory,
        })
    }

    pub fn extend_game(&mut self, game: SelfPlayGame, capacity: usize) {
        self.format_version = 2;
        self.games.push(ReplayGame {
            validated: std::sync::OnceLock::new(),
            ruleset: game.ruleset,
            steps: game.steps,
            outcome: game.outcome,
            truncated: game.truncated,
        });
        while self.games.len() > 1 && self.len() > capacity {
            self.games.remove(0);
        }
    }

    /// Saves the replay window atomically as `MessagePack` data compressed with zstd.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination cannot be created or serialized.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ReplayError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("zst.tmp");
        let writer = BufWriter::new(File::create(&temporary)?);
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;
        rmp_serde::encode::write_named(&mut encoder, self)
            .map_err(|error| ReplayError::Codec(error.to_string()))?;
        let mut writer = encoder.finish()?;
        writer.flush()?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    /// Loads a version-2 compressed replay window.
    ///
    /// # Errors
    ///
    /// Returns an error if the source cannot be read or decoded.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        let reader = BufReader::new(File::open(path)?);
        let decoder = zstd::stream::read::Decoder::new(reader)?;
        let replay: Self =
            rmp_serde::from_read(decoder).map_err(|error| ReplayError::Codec(error.to_string()))?;
        if replay.format_version != 2 {
            return Err(ReplayError::Version(replay.format_version));
        }
        Ok(replay)
    }
}

fn estimate_example_bytes(game: &Game, actions: usize) -> usize {
    let boards = if game.ruleset() == Ruleset::Standard {
        1
    } else {
        game.timelines()
            .iter()
            .map(|timeline| timeline.boards.len())
            .sum::<usize>()
            .max(1)
    };
    // Deliberately includes allocator/Vec overhead above the serialized f32 data.
    boards
        .saturating_mul(4 * 1024)
        .saturating_add(actions.saturating_mul(192))
        .saturating_add(1024)
}

const fn bytes_to_mib_ceil(bytes: usize) -> usize {
    bytes.saturating_add(1024 * 1024 - 1) / (1024 * 1024)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::network::NetworkConfig;

    use super::*;

    #[test]
    fn short_self_play_game_produces_aligned_examples_and_draw_targets() {
        let mut rng = ChaCha8Rng::seed_from_u64(23);
        let model = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let game = play_self_play_game(
            &model,
            Ruleset::Standard,
            SelfPlayConfig {
                search: SearchConfig {
                    simulations: 2,
                    ..SearchConfig::default()
                },
                max_actions: 2,
                exploration_actions: 2,
                temperature: 1.0,
            },
            &mut rng,
        );
        assert_eq!(game.steps.len(), 2);
        assert!(game.truncated);
        let mut replay = ReplayBuffer::default();
        replay.extend_game(game, 10);
        for example in replay.examples().unwrap() {
            assert_eq!(example.position.actions.len(), example.policy.len());
            assert!((example.policy.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
            assert!(example.value.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn replay_capacity_discards_oldest_complete_games_and_round_trips() {
        let make_game = || {
            let game = Game::new(Ruleset::Standard);
            let actions = game.legal_actions();
            SelfPlayGame {
                ruleset: Ruleset::Standard,
                steps: vec![ReplayStep {
                    action: actions[0],
                    policy: vec![1.0 / actions.len() as f32; actions.len()],
                }],
                outcome: None,
                truncated: true,
            }
        };
        let mut replay = ReplayBuffer::default();
        replay.extend_game(make_game(), 2);
        replay.extend_game(make_game(), 2);
        replay.extend_game(make_game(), 2);
        assert_eq!(replay.len(), 2);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("replay-v2.bin.zst");
        replay.save(&path).unwrap();
        let loaded = ReplayBuffer::load(path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.examples().unwrap().len(), 2);
    }

    #[test]
    fn cached_trajectory_sampling_preserves_examples_and_validates_unsampled_games() {
        let mut rng = ChaCha8Rng::seed_from_u64(118);
        let model = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let mut replay = ReplayBuffer::default();
        for ruleset in [
            Ruleset::Standard,
            Ruleset::Multiverse,
            Ruleset::MultiverseVariant,
        ] {
            let game = play_self_play_game(
                &model,
                ruleset,
                SelfPlayConfig {
                    search: SearchConfig {
                        simulations: 2,
                        ..SearchConfig::default()
                    },
                    max_actions: 8,
                    ..SelfPlayConfig::default()
                },
                &mut rng,
            );
            replay.extend_game(game, 100);
        }
        let parallel = replay.clone();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let sample = || {
            replay
                .sample_examples(12, 40 * 1024, &mut ChaCha8Rng::seed_from_u64(78))
                .unwrap()
        };
        let first = sample();
        assert!(first.skipped_for_memory > 0);
        let pooled = parallel
            .sample_examples_in_pool(12, 40 * 1024, &mut ChaCha8Rng::seed_from_u64(78), &pool)
            .unwrap();
        let second = sample();
        for other in [pooled, second] {
            assert_eq!(first.estimated_bytes, other.estimated_bytes);
            assert_eq!(first.skipped_for_memory, other.skipped_for_memory);
            assert_eq!(first.examples.len(), other.examples.len());
            for (a, b) in first.examples.iter().zip(other.examples) {
                assert_eq!(a.position, b.position);
                assert_eq!(a.policy, b.policy);
                assert_eq!(a.value.to_bits(), b.value.to_bits());
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid.zst");
        replay.games[0].steps[2].policy.clear();
        replay.games[1].steps[7].policy.clear();
        replay.save(&path).unwrap();
        let loaded = ReplayBuffer::load(path).unwrap();
        for seed in 0..8 {
            let mut serial_rng = ChaCha8Rng::seed_from_u64(seed);
            let mut parallel_rng = serial_rng.clone();
            let serial_error = loaded
                .sample_examples(0, usize::MAX, &mut serial_rng)
                .unwrap_err();
            let parallel_error = loaded
                .sample_examples_in_pool(0, usize::MAX, &mut parallel_rng, &pool)
                .unwrap_err();
            assert!(matches!(serial_error, ReplayError::Trajectory(_)));
            assert_eq!(serial_error.to_string(), parallel_error.to_string());
            assert!(
                loaded.games[..2]
                    .iter()
                    .all(|game| game.validated.get().is_none())
            );
        }
    }

    #[test]
    fn replay_sampling_bounds_expanded_positions_and_host_memory() {
        let game = Game::new(Ruleset::Standard);
        let actions = game.legal_actions();
        let replay_game = SelfPlayGame {
            ruleset: Ruleset::Standard,
            steps: vec![ReplayStep {
                action: actions[0],
                policy: vec![1.0 / actions.len() as f32; actions.len()],
            }],
            outcome: None,
            truncated: true,
        };
        let mut replay = ReplayBuffer::default();
        replay.extend_game(replay_game.clone(), 10);
        replay.extend_game(replay_game, 10);
        let mut rng = ChaCha8Rng::seed_from_u64(41);

        let sample = replay.sample_examples(1, 1024 * 1024, &mut rng).unwrap();
        assert_eq!(sample.examples.len(), 1);
        assert!(sample.estimated_bytes <= 1024 * 1024);

        let error = replay.sample_examples(1, 1, &mut rng).unwrap_err();
        assert!(matches!(error, ReplayError::MemoryBudget { .. }));
    }

    #[test]
    fn arena_scores_terminal_and_truncated_draws_equally() {
        let mut report = ArenaReport::default();
        report.record_outcome(Side::White, None);
        report.record_outcome(
            Side::Black,
            Some(GameOutcome {
                winner: None,
                reason: chronofish_core::OutcomeReason::Stalemate,
            }),
        );
        assert_eq!(report.draws, 2);
        assert!((report.candidate_score() - 0.5).abs() < f32::EPSILON);
        report.record_outcome(
            Side::Black,
            Some(GameOutcome {
                winner: Some(Side::Black),
                reason: chronofish_core::OutcomeReason::Checkmate,
            }),
        );
        report.record_outcome(
            Side::Black,
            Some(GameOutcome {
                winner: Some(Side::White),
                reason: chronofish_core::OutcomeReason::Checkmate,
            }),
        );
        assert_eq!(report.candidate_wins, 1);
        assert_eq!(report.incumbent_wins, 1);
        assert!((report.candidate_score() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn arena_progress_reports_actions_and_alternating_candidate_sides() {
        let mut rng = ChaCha8Rng::seed_from_u64(37);
        let candidate = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let incumbent = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let mut events = Vec::new();
        let report = run_arena_schedule_with_progress(
            &candidate,
            &incumbent,
            &[Ruleset::Standard, Ruleset::Multiverse],
            SearchConfig {
                simulations: 1,
                ..SearchConfig::default()
            },
            1,
            &mut rng,
            |event| events.push(event),
        );
        assert_eq!(report.draws, 2);
        let completed = events
            .iter()
            .filter(|event| event.completed)
            .collect::<Vec<_>>();
        assert_eq!(completed.len(), 2);
        assert_eq!(completed[0].candidate_side, Side::White);
        assert_eq!(completed[1].candidate_side, Side::Black);
        assert_eq!(completed[0].actions, 1);
        assert_eq!(completed[1].actions, 1);
    }
}
