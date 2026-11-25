//! Round-robin bot tournament: every pair plays each opening twice with colours reversed.
//!
//! Ratings come from a logistic maximum-likelihood fit. Without anchors they are centred
//! on zero; with anchors (calibrated Stockfish levels, or fixed reference bots) the
//! anchored players are held at their known Elo and everyone else is placed on that
//! scale. Confidence intervals bootstrap colour-reversed opening pairs, the unit that
//! is actually independent.
//!
//! `--calibrate` carries a standard-chess scale into 5D through a ladder of Deep
//! Blueshift node budgets: the ladder is rated against Stockfish in standard chess, then
//! held at those ratings while everyone plays 5D.

// Counts here are small game tallies, far below the precision limits of f64.
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use chronofish_alphazero::{AlphaNowBot, ChronofishCpuBot, PolicyValueNetwork, SearchConfig};
use chronofish_classic::eval::evaluate;
use chronofish_classic::{
    BlueshiftConfig, BobbyFissureBot, CarlsenConfig, DeepBlueshift, Grabablanca, MonteCarlsen,
};
use chronofish_core::{AiPlayer, Color, Game, Ruleset};
use clap::{Parser, ValueEnum};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;

use crate::uci::{Strength, UciEngine};

const ROSTER: [&str; 8] = [
    "fissure",
    "grabablanca",
    "blueshift",
    "carlsen",
    "alpha-now",
    "cpu",
    "kairos",
    "loki",
];
const LADDER: [&str; 5] = [
    "blueshift-1k",
    "blueshift-3k",
    "blueshift-10k",
    "blueshift-30k",
    "blueshift-80k",
];
const REFERENCES: [&str; 3] = ["sf-1320", "sf-1600", "sf-2000"];
const STOCKFISH_ELO: std::ops::RangeInclusive<u32> = 1320..=3190;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RuleSelection {
    Standard,
    Multiverse,
    Variant,
    /// Both 5D rulesets.
    #[value(name = "5d")]
    FiveD,
    All,
}

impl RuleSelection {
    fn rulesets(self) -> Vec<Ruleset> {
        match self {
            Self::Standard => vec![Ruleset::Standard],
            Self::Multiverse => vec![Ruleset::Multiverse],
            Self::Variant => vec![Ruleset::MultiverseVariant],
            Self::FiveD => vec![Ruleset::Multiverse, Ruleset::MultiverseVariant],
            Self::All => vec![
                Ruleset::Standard,
                Ruleset::Multiverse,
                Ruleset::MultiverseVariant,
            ],
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "chronofish-benchmarks --tournament",
    about = "Round-robin tournament between Chronofish bots"
)]
#[allow(clippy::struct_excessive_bools)]
struct Arguments {
    /// Comma-separated bots: fissure, grabablanca, blueshift[-<nodes>[k]], carlsen,
    /// alpha-now, cpu, kairos, loki, and with --stockfish sf-<elo> (1320–3190) or sf-skill<0-20>.
    #[arg(long, value_delimiter = ',', default_values_t = ROSTER.map(str::to_owned))]
    bots: Vec<String>,
    /// Random openings per ruleset; each pair plays every opening from both sides.
    #[arg(long, default_value_t = 2)]
    openings: usize,
    /// Random actions in each opening.
    #[arg(long, default_value_t = 12)]
    opening_actions: usize,
    #[arg(long, value_enum, default_value_t = RuleSelection::All)]
    ruleset: RuleSelection,
    /// Actions after the opening before a game is adjudicated or drawn.
    #[arg(long, default_value_t = 160)]
    max_actions: usize,
    /// At the action limit, the side ahead by this many centipawns (handcrafted eval)
    /// wins; otherwise the game is drawn. Zero scores every unfinished game as a draw.
    #[arg(long, default_value_t = 400)]
    adjudicate_cp: i32,
    /// Checkpoint for the network bots (alpha-now, cpu); untrained if omitted.
    #[arg(long)]
    model: Option<PathBuf>,
    /// MCTS simulations per action for the cpu bot.
    #[arg(long, default_value_t = 48)]
    simulations: usize,
    /// UCI engine binary for sf-* bots; defaults to $STOCKFISH.
    #[arg(long)]
    stockfish: Option<PathBuf>,
    /// Thinking time per move for UCI engines, in milliseconds.
    #[arg(long, default_value_t = 100)]
    uci_movetime: u64,
    /// Fix ratings: name=elo pairs. Stockfish Elo levels are anchored automatically.
    #[arg(long, value_delimiter = ',')]
    anchor: Vec<String>,
    /// Bootstrap resamples of opening pairs for 95% intervals; zero disables them.
    #[arg(long, default_value_t = 200)]
    bootstrap: usize,
    /// Rate the ladder against Stockfish in standard chess, then rate every bot in 5D
    /// against the ladder held at those ratings.
    #[arg(long)]
    calibrate: bool,
    /// Ladder rungs for --calibrate, weakest first.
    #[arg(long, value_delimiter = ',', default_values_t = LADDER.map(str::to_owned))]
    ladder: Vec<String>,
    /// Stockfish anchors for --calibrate.
    #[arg(long, value_delimiter = ',', default_values_t = REFERENCES.map(str::to_owned))]
    references: Vec<String>,
    /// Write one row per game to this CSV file.
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Write the fitted ratings and intervals to this JSON file.
    #[arg(long)]
    save_ratings: Option<PathBuf>,
    #[arg(long, default_value_t = 0x544f_5552_4e45_59)]
    seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bot {
    Fissure,
    Grabablanca,
    Blueshift(usize),
    Carlsen,
    AlphaNow,
    Cpu,
    Kairos,
    Loki,
    Stockfish(Strength),
}

impl Bot {
    fn parse(name: &str) -> Result<Self, String> {
        let unknown = || {
            format!(
                "unknown bot {name}; choose from {}, blueshift-<nodes>, sf-<elo>, sf-skill<level>",
                ROSTER.join(", ")
            )
        };
        Ok(match name {
            "fissure" => Self::Fissure,
            "grabablanca" => Self::Grabablanca,
            "blueshift" => Self::Blueshift(BlueshiftConfig::default().max_nodes),
            "carlsen" => Self::Carlsen,
            "alpha-now" => Self::AlphaNow,
            "cpu" => Self::Cpu,
            "kairos" => Self::Kairos,
            "loki" => Self::Loki,
            _ => {
                if let Some(budget) = name.strip_prefix("blueshift-") {
                    let (digits, scale) = budget
                        .strip_suffix('k')
                        .map_or((budget, 1), |digits| (digits, 1_000));
                    let nodes = digits.parse::<usize>().map_err(|_| unknown())?;
                    if nodes == 0 {
                        return Err(format!("{name}: the node budget must be positive"));
                    }
                    Self::Blueshift(nodes * scale)
                } else if let Some(level) = name.strip_prefix("sf-skill") {
                    match level.parse::<u8>() {
                        Ok(level) if level <= 20 => Self::Stockfish(Strength::Skill(level)),
                        _ => return Err(format!("{name}: skill levels run from 0 to 20")),
                    }
                } else if let Some(elo) = name.strip_prefix("sf-") {
                    match elo.parse::<u32>() {
                        Ok(elo) if STOCKFISH_ELO.contains(&elo) => {
                            Self::Stockfish(Strength::Elo(elo))
                        }
                        _ => {
                            return Err(format!(
                                "{name}: Stockfish Elo levels run from {} to {}; use sf-skill<0-20> below that",
                                STOCKFISH_ELO.start(),
                                STOCKFISH_ELO.end()
                            ));
                        }
                    }
                } else {
                    return Err(unknown());
                }
            }
        })
    }

    /// Canonical short name, so `blueshift` and `blueshift-80k` are the same player.
    fn key(self) -> String {
        match self {
            Self::Fissure => "fissure".to_owned(),
            Self::Grabablanca => "grabablanca".to_owned(),
            Self::Blueshift(nodes) if nodes % 1_000 == 0 => format!("blueshift-{}k", nodes / 1_000),
            Self::Blueshift(nodes) => format!("blueshift-{nodes}"),
            Self::Carlsen => "carlsen".to_owned(),
            Self::AlphaNow => "alpha-now".to_owned(),
            Self::Cpu => "cpu".to_owned(),
            Self::Kairos => "kairos".to_owned(),
            Self::Loki => "loki".to_owned(),
            Self::Stockfish(Strength::Elo(elo)) => format!("sf-{elo}"),
            Self::Stockfish(Strength::Skill(level)) => format!("sf-skill{level}"),
        }
    }

    fn label(self, trained: bool) -> String {
        let untrained = if trained { "" } else { " (untrained)" };
        match self {
            Self::Fissure => "Bobby Fissure".to_owned(),
            Self::Grabablanca => "Grabablanca".to_owned(),
            Self::Blueshift(nodes) if nodes == BlueshiftConfig::default().max_nodes => {
                "Deep Blueshift".to_owned()
            }
            Self::Blueshift(nodes) if nodes % 1_000 == 0 => {
                format!("Deep Blueshift {}k", nodes / 1_000)
            }
            Self::Blueshift(nodes) => format!("Deep Blueshift {nodes}"),
            Self::Carlsen => "Monte Carlsen".to_owned(),
            Self::AlphaNow => format!("AlphaNow{untrained}"),
            Self::Cpu => format!("Chronofish CPU{untrained}"),
            Self::Kairos => "Kairos".to_owned(),
            Self::Loki => "Loki".to_owned(),
            Self::Stockfish(Strength::Elo(elo)) => format!("Stockfish {elo}"),
            Self::Stockfish(Strength::Skill(level)) => format!("Stockfish skill {level}"),
        }
    }

    const fn supports(self, ruleset: Ruleset) -> bool {
        match self {
            Self::Kairos | Self::Loki => !matches!(ruleset, Ruleset::Standard),
            Self::Stockfish(_) => matches!(ruleset, Ruleset::Standard),
            _ => true,
        }
    }

    /// Orthodox engines cannot play 5D chess.
    const fn standard_only(self) -> bool {
        matches!(self, Self::Stockfish(_))
    }

    fn calibrated_elo(self) -> Option<f64> {
        match self {
            Self::Stockfish(Strength::Elo(elo)) => Some(f64::from(elo)),
            _ => None,
        }
    }
}

fn parse_bots(names: &[String]) -> Result<Vec<Bot>, String> {
    names.iter().map(|name| Bot::parse(name)).collect()
}

struct Context {
    model: Option<PolicyValueNetwork>,
    simulations: usize,
    stockfish: Option<PathBuf>,
    movetime_ms: u64,
    /// Per-game CSV rows, written as each game finishes so an interrupted run keeps them.
    csv: Option<Mutex<BufWriter<File>>>,
}

impl Context {
    fn make_bot(&self, bot: Bot, seed: u64) -> Result<Box<dyn AiPlayer>, String> {
        let search = SearchConfig {
            simulations: self.simulations,
            ..SearchConfig::default()
        };
        let model = self.model.as_ref();
        Ok(match bot {
            Bot::Fissure => Box::new(BobbyFissureBot::new(seed | 1)),
            Bot::Grabablanca => Box::new(Grabablanca::new(seed)),
            Bot::Blueshift(max_nodes) => Box::new(DeepBlueshift::new(BlueshiftConfig {
                max_nodes,
                ..BlueshiftConfig::default()
            })),
            Bot::Carlsen => Box::new(MonteCarlsen::new(CarlsenConfig::default(), seed)),
            Bot::AlphaNow => Box::new(model.map_or_else(
                || AlphaNowBot::bootstrap(seed),
                |model| AlphaNowBot::new(model.clone()),
            )),
            Bot::Cpu => Box::new(model.map_or_else(
                || ChronofishCpuBot::bootstrap(search, seed),
                |model| ChronofishCpuBot::new(model.clone(), search, seed),
            )),
            Bot::Kairos => Box::new(kairos::KairosBot::default()),
            Bot::Loki => Box::new(chronofish_ftxi::LokiBot::new(seed)),
            Bot::Stockfish(strength) => {
                let path = self
                    .stockfish
                    .as_deref()
                    .ok_or_else(|| "sf-* bots need --stockfish or $STOCKFISH".to_owned())?;
                Box::new(UciEngine::spawn(path, strength, self.movetime_ms)?)
            }
        })
    }

    /// Starts one engine up front so a bad path fails before any game is played.
    fn check(&self, bots: &[Bot]) -> Result<(), String> {
        if let Some(&bot) = bots.iter().find(|bot| bot.standard_only()) {
            self.make_bot(bot, 0).map(drop)?;
        }
        Ok(())
    }

    fn label(&self, bot: Bot) -> String {
        bot.label(self.model.is_some())
    }
}

#[derive(Clone, Copy)]
struct Schedule {
    openings: usize,
    opening_actions: usize,
    max_actions: usize,
    adjudicate_cp: i32,
    seed: u64,
}

/// Plays up to `actions` random legal actions, stopping before any that ends the game.
fn random_opening(ruleset: Ruleset, actions: usize, seed: u64) -> Game {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut game = Game::new(ruleset);
    for _ in 0..actions {
        let legal = game.legal_actions();
        if legal.is_empty() {
            break;
        }
        let mut next = game.clone();
        let index = usize::try_from(rng.next_u64() % legal.len() as u64).unwrap_or(0);
        if next.apply_action(legal[index]).is_err() || next.outcome().is_some() {
            break;
        }
        game = next;
    }
    game
}

#[derive(Clone, Copy)]
struct Pairing {
    ruleset: Ruleset,
    opening: usize,
    white: usize,
    black: usize,
}

impl Pairing {
    /// Both colours of one opening between one pair: the independent unit for resampling.
    fn group(self) -> Group {
        (
            self.ruleset,
            self.opening,
            self.white.min(self.black),
            self.white.max(self.black),
        )
    }
}

/// `(white, black, white's score)` for the rating fit.
type Outcome = (usize, usize, f64);
/// Ruleset, opening and the unordered pair of players.
type Group = (Ruleset, usize, usize, usize);

struct GameResult {
    /// White's score: 1, 0.5, or 0.
    white_score: f64,
    actions: usize,
    reason: String,
    thinking: [Duration; 2],
    moves: [usize; 2],
}

struct Played {
    pairing: Pairing,
    result: GameResult,
}

fn play(
    white: &mut (dyn AiPlayer + 'static),
    black: &mut (dyn AiPlayer + 'static),
    mut game: Game,
    max_actions: usize,
    adjudicate_cp: i32,
) -> GameResult {
    let mut thinking = [Duration::ZERO; 2];
    let mut moves = [0; 2];
    let forfeit = |loser: Color, reason: String, actions, thinking, moves| GameResult {
        white_score: if loser == Color::White { 0.0 } else { 1.0 },
        actions,
        reason,
        thinking,
        moves,
    };
    for actions in 0..max_actions {
        if game.outcome().is_some() {
            break;
        }
        let to_move = game.turn();
        let side = usize::from(to_move == Color::Black);
        let bot = if side == 0 { &mut *white } else { &mut *black };
        let started = Instant::now();
        let decision = bot.choose_action(&game);
        thinking[side] += started.elapsed();
        moves[side] += 1;
        let action = match decision {
            Ok(decision) => decision.action,
            Err(error) => {
                return forfeit(
                    to_move,
                    format!("forfeit: {error}"),
                    actions,
                    thinking,
                    moves,
                );
            }
        };
        if let Err(error) = game.apply_action(action) {
            return forfeit(
                to_move,
                format!("forfeit: illegal action ({error})"),
                actions,
                thinking,
                moves,
            );
        }
    }
    let actions = moves[0] + moves[1];
    let (white_score, reason) = match game.outcome() {
        Some(outcome) => (
            match outcome.winner {
                Some(Color::White) => 1.0,
                Some(Color::Black) => 0.0,
                None => 0.5,
            },
            format!("{:?}", outcome.reason),
        ),
        None => adjudicate(&game, adjudicate_cp),
    };
    GameResult {
        white_score,
        actions,
        reason,
        thinking,
        moves,
    }
}

/// Scores an unfinished game by the handcrafted evaluation, from White's point of view.
fn adjudicate(game: &Game, threshold_cp: i32) -> (f64, String) {
    if threshold_cp <= 0 {
        return (0.5, "action limit".to_owned());
    }
    let eval = evaluate(game);
    if eval >= threshold_cp {
        (1.0, format!("adjudicated {eval:+}cp"))
    } else if eval <= -threshold_cp {
        (0.0, format!("adjudicated {eval:+}cp"))
    } else {
        (0.5, format!("action limit {eval:+}cp"))
    }
}

/// Plays every pair from both sides of each opening. Orthodox engines sit out 5D rulesets.
fn round_robin(
    players: &[Bot],
    rulesets: &[Ruleset],
    context: &Context,
    schedule: Schedule,
) -> Vec<Played> {
    let mut pairings = Vec::new();
    for &ruleset in rulesets {
        for opening in 0..schedule.openings {
            for first in 0..players.len() {
                for second in first + 1..players.len() {
                    if !players[first].supports(ruleset) || !players[second].supports(ruleset) {
                        continue;
                    }
                    for (white, black) in [(first, second), (second, first)] {
                        pairings.push(Pairing {
                            ruleset,
                            opening,
                            white,
                            black,
                        });
                    }
                }
            }
        }
    }
    eprintln!(
        "{} games · {} bots · {} openings per ruleset · {} action limit",
        pairings.len(),
        players.len(),
        schedule.openings,
        schedule.max_actions
    );
    let finished = std::sync::atomic::AtomicUsize::new(0);
    pairings
        .par_iter()
        .enumerate()
        .map(|(index, &pairing)| {
            let ruleset_tag = match pairing.ruleset {
                Ruleset::Standard => 0,
                Ruleset::Multiverse => 1,
                Ruleset::MultiverseVariant => 2,
            };
            let opening = random_opening(
                pairing.ruleset,
                schedule.opening_actions,
                schedule.seed ^ (ruleset_tag << 48) ^ (pairing.opening as u64) << 8,
            );
            let game_seed = schedule.seed.rotate_left(17) ^ index as u64;
            let white = context.make_bot(players[pairing.white], game_seed);
            let black = context.make_bot(players[pairing.black], game_seed ^ 1);
            let result = match (white, black) {
                (Ok(mut white), Ok(mut black)) => play(
                    white.as_mut(),
                    black.as_mut(),
                    opening,
                    schedule.max_actions,
                    schedule.adjudicate_cp,
                ),
                (white, _) => GameResult {
                    white_score: if white.is_err() { 0.0 } else { 1.0 },
                    actions: 0,
                    reason: "forfeit: engine failed to start".to_owned(),
                    thinking: [Duration::ZERO; 2],
                    moves: [0; 2],
                },
            };
            let done = finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            eprintln!(
                "  [{done}/{}] {} · opening {} · {} vs {} · {}-{} · {} actions · {}",
                pairings.len(),
                pairing.ruleset,
                pairing.opening + 1,
                context.label(players[pairing.white]),
                context.label(players[pairing.black]),
                format_score(result.white_score),
                format_score(1.0 - result.white_score),
                result.actions,
                result.reason
            );
            if let Some(csv) = &context.csv {
                let mut file = csv
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let row = writeln!(
                    file,
                    "{},{},{},{},{},{},{}",
                    pairing.ruleset,
                    pairing.opening + 1,
                    players[pairing.white].key(),
                    players[pairing.black].key(),
                    result.white_score,
                    result.actions,
                    result.reason
                )
                .and_then(|()| file.flush());
                if let Err(error) = row {
                    eprintln!("warning: could not write the CSV row: {error}");
                }
            }
            Played { pairing, result }
        })
        .collect()
}

/// Maximum-likelihood Elo with one virtual draw per pairing so perfect scores stay finite.
///
/// Each result is `(player, opponent, player's score)`. Players with an anchor are held
/// at it; without any anchor the ratings are centred on zero.
fn elo(players: usize, results: &[(usize, usize, f64)], anchors: &[Option<f64>]) -> Vec<f64> {
    let mut pairs = results
        .iter()
        .map(|&(a, b, _)| (a.min(b), a.max(b)))
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs.dedup();
    let games = results
        .iter()
        .copied()
        .chain(pairs.into_iter().map(|(a, b)| (a, b, 0.5)))
        .collect::<Vec<_>>();
    let anchored = anchors.iter().any(Option::is_some);
    let known = anchors.iter().flatten().copied().collect::<Vec<_>>();
    let start = if known.is_empty() {
        0.0
    } else {
        known.iter().sum::<f64>() / known.len() as f64
    };
    let mut ratings = (0..players)
        .map(|player| anchors.get(player).copied().flatten().unwrap_or(start))
        .collect::<Vec<_>>();
    for _ in 0..5_000 {
        let mut actual = vec![0.0; players];
        let mut expected = vec![0.0; players];
        let mut counts = vec![0.0; players];
        for &(a, b, score) in &games {
            let expectation = 1.0 / (1.0 + 10f64.powf((ratings[b] - ratings[a]) / 400.0));
            actual[a] += score;
            actual[b] += 1.0 - score;
            expected[a] += expectation;
            expected[b] += 1.0 - expectation;
            counts[a] += 1.0;
            counts[b] += 1.0;
        }
        let mut largest_step = 0.0_f64;
        for player in 0..players {
            if counts[player] > 0.0 && anchors.get(player).copied().flatten().is_none() {
                let step = 400.0 * (actual[player] - expected[player]) / counts[player];
                ratings[player] += step;
                largest_step = largest_step.max(step.abs());
            }
        }
        if !anchored {
            let mean = ratings.iter().sum::<f64>() / players.max(1) as f64;
            for rating in &mut ratings {
                *rating -= mean;
            }
        }
        if largest_step < 1e-4 {
            break;
        }
    }
    ratings
}

/// 95% intervals from refitting on resampled opening pairs.
fn bootstrap(
    players: usize,
    played: &[Played],
    anchors: &[Option<f64>],
    rounds: usize,
    seed: u64,
) -> Option<Vec<(f64, f64)>> {
    if rounds == 0 {
        return None;
    }
    let mut groups: HashMap<Group, Vec<Outcome>> = HashMap::new();
    for game in played {
        groups.entry(game.pairing.group()).or_default().push((
            game.pairing.white,
            game.pairing.black,
            game.result.white_score,
        ));
    }
    let mut groups = groups.into_iter().collect::<Vec<_>>();
    groups.sort_unstable_by_key(|(key, _)| (key.0.to_string(), key.1, key.2, key.3));
    let groups = groups
        .into_iter()
        .map(|(_, games)| games)
        .collect::<Vec<_>>();
    if groups.is_empty() {
        return None;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut samples = vec![Vec::with_capacity(rounds); players];
    for _ in 0..rounds {
        let mut results = Vec::new();
        for _ in 0..groups.len() {
            let pick = usize::try_from(rng.next_u64() % groups.len() as u64).unwrap_or(0);
            results.extend_from_slice(&groups[pick]);
        }
        for (player, rating) in elo(players, &results, anchors).into_iter().enumerate() {
            samples[player].push(rating);
        }
    }
    Some(
        samples
            .into_iter()
            .map(|mut ratings| {
                ratings.sort_by(f64::total_cmp);
                let at = |quantile: f64| {
                    let index = (quantile * (ratings.len() - 1) as f64).round();
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    ratings[index as usize]
                };
                (at(0.025), at(0.975))
            })
            .collect(),
    )
}

fn results_for_fit(played: &[Played]) -> Vec<(usize, usize, f64)> {
    played
        .iter()
        .map(|game| {
            (
                game.pairing.white,
                game.pairing.black,
                game.result.white_score,
            )
        })
        .collect()
}

fn format_score(score: f64) -> String {
    if score.fract() == 0.0 {
        format!("{score:.0}")
    } else {
        format!("{score:.1}")
    }
}

/// A fitted rating table for one set of players and games.
struct Ratings {
    values: Vec<f64>,
    intervals: Option<Vec<(f64, f64)>>,
    anchors: Vec<Option<f64>>,
}

impl Ratings {
    fn fit(
        players: usize,
        played: &[Played],
        anchors: Vec<Option<f64>>,
        rounds: usize,
        seed: u64,
    ) -> Self {
        let ratings = elo(players, &results_for_fit(played), &anchors);
        let intervals = bootstrap(players, played, &anchors, rounds, seed);
        Self {
            values: ratings,
            intervals,
            anchors,
        }
    }

    fn anchored(&self) -> bool {
        self.anchors.iter().any(Option::is_some)
    }

    fn cell(&self, player: usize) -> String {
        let rating = if self.anchored() {
            format!("{:.0}", self.values[player])
        } else {
            format!("{:+.0}", self.values[player])
        };
        if self.anchors[player].is_some() {
            return format!("{rating} anchor");
        }
        match &self.intervals {
            Some(intervals) => {
                let (low, high) = intervals[player];
                format!("{rating} ±{:.0}", (high - low) / 2.0)
            }
            None => rating,
        }
    }

    fn json(&self, keys: &[String]) -> String {
        let mut entries = Vec::new();
        for (player, key) in keys.iter().enumerate() {
            let mut entry = format!("\"{key}\": {{\"elo\": {:.1}", self.values[player]);
            if let Some(intervals) = &self.intervals {
                let (low, high) = intervals[player];
                let _ = write!(entry, ", \"low\": {low:.1}, \"high\": {high:.1}");
            }
            let _ = write!(entry, ", \"anchor\": {}}}", self.anchors[player].is_some());
            entries.push(entry);
        }
        format!("{{{}}}", entries.join(", "))
    }
}

fn standings(
    title: &str,
    labels: &[String],
    played: &[Played],
    ratings: &Ratings,
) -> Result<String, std::fmt::Error> {
    let players = labels.len();
    let mut table = vec![vec![(0.0_f64, 0_usize); players]; players];
    let mut record = vec![[0_usize; 3]; players];
    let mut thinking = vec![(Duration::ZERO, 0_usize); players];
    let mut adjudicated = 0;
    for game in played {
        let (white, black) = (game.pairing.white, game.pairing.black);
        let score = game.result.white_score;
        table[white][black].0 += score;
        table[white][black].1 += 1;
        table[black][white].0 += 1.0 - score;
        table[black][white].1 += 1;
        let outcome = |score: f64| {
            if score > 0.5 {
                0
            } else if score < 0.5 {
                2
            } else {
                1
            }
        };
        record[white][outcome(score)] += 1;
        record[black][outcome(1.0 - score)] += 1;
        for (player, side) in [(white, 0), (black, 1)] {
            thinking[player].0 += game.result.thinking[side];
            thinking[player].1 += game.result.moves[side];
        }
        adjudicated += usize::from(game.result.reason.starts_with("adjudicated"));
    }
    let mut order = (0..players).collect::<Vec<_>>();
    order.sort_by(|&a, &b| ratings.values[b].total_cmp(&ratings.values[a]));
    let width = labels.iter().map(String::len).max().unwrap_or(4).max(4);
    let mut output = String::new();
    writeln!(output, "\n{title}")?;
    writeln!(
        output,
        "  {:<width$}  {:>5}  {:>4} {:>4} {:>4}  {:>6}  {:>12}  {:>9}",
        "Bot", "Games", "W", "D", "L", "Score", "Elo", "ms/action"
    )?;
    for &player in &order {
        let [wins, draws, losses] = record[player];
        let games = wins + draws + losses;
        let score = (wins as f64 + draws as f64 * 0.5) / games.max(1) as f64;
        let (time, moves) = thinking[player];
        writeln!(
            output,
            "  {:<width$}  {games:>5}  {wins:>4} {draws:>4} {losses:>4}  {:>5.1}%  {:>12}  {:>9.1}",
            labels[player],
            score * 100.0,
            ratings.cell(player),
            time.as_secs_f64() * 1_000.0 / moves.max(1) as f64,
        )?;
    }
    if ratings.intervals.is_some() {
        writeln!(
            output,
            "  ± is half the 95% bootstrap interval over opening pairs."
        )?;
    }
    if adjudicated > 0 {
        writeln!(
            output,
            "  {adjudicated} of {} games were adjudicated by evaluation at the action limit.",
            played.len()
        )?;
    }
    writeln!(output, "\nCrosstable (row's score against column)")?;
    write!(output, "  {:<width$}", "")?;
    for index in 0..order.len() {
        write!(output, "  {:>7}", format!("#{}", index + 1))?;
    }
    writeln!(output)?;
    for (rank, &row) in order.iter().enumerate() {
        write!(
            output,
            "  {:<width$}",
            format!("#{} {}", rank + 1, labels[row])
        )?;
        for &column in &order {
            let cell = if row == column {
                "—".to_owned()
            } else {
                let (score, games) = table[row][column];
                if games == 0 {
                    "·".to_owned()
                } else {
                    format!("{}/{games}", format_score(score))
                }
            };
            write!(output, "  {cell:>7}")?;
        }
        writeln!(output)?;
    }
    Ok(output)
}

fn csv_file(path: &Path) -> std::io::Result<BufWriter<File>> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(
        file,
        "ruleset,opening,white,black,white_score,actions,reason"
    )?;
    Ok(file)
}

/// Explicit `name=elo` anchors, keyed by canonical bot name.
fn parse_anchors(entries: &[String]) -> Result<HashMap<String, f64>, String> {
    entries
        .iter()
        .map(|entry| {
            let (name, elo) = entry
                .split_once('=')
                .ok_or_else(|| format!("--anchor {entry}: expected name=elo"))?;
            let elo = elo
                .parse::<f64>()
                .map_err(|_| format!("--anchor {entry}: {elo} is not a number"))?;
            Ok((Bot::parse(name)?.key(), elo))
        })
        .collect()
}

fn anchors_for(players: &[Bot], explicit: &HashMap<String, f64>) -> Vec<Option<f64>> {
    players
        .iter()
        .map(|bot| {
            explicit
                .get(&bot.key())
                .copied()
                .or_else(|| bot.calibrated_elo())
        })
        .collect()
}

/// Removes repeated players (`blueshift` and `blueshift-80k` are one bot), keeping order.
fn dedupe(players: impl IntoIterator<Item = Bot>) -> Vec<Bot> {
    let mut seen = Vec::new();
    for bot in players {
        if !seen.contains(&bot) {
            seen.push(bot);
        }
    }
    seen
}

pub fn run(arguments: impl IntoIterator<Item = String>) -> Result<(), Box<dyn Error>> {
    let arguments = match Arguments::try_parse_from(
        std::iter::once("tournament".to_owned()).chain(arguments),
    ) {
        Ok(arguments) => arguments,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            error.print()?;
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let context = Context {
        model: arguments
            .model
            .as_ref()
            .map(PolicyValueNetwork::load)
            .transpose()?,
        simulations: arguments.simulations,
        stockfish: arguments
            .stockfish
            .clone()
            .or_else(|| std::env::var_os("STOCKFISH").map(PathBuf::from)),
        movetime_ms: arguments.uci_movetime,
        csv: arguments
            .csv
            .as_deref()
            .map(csv_file)
            .transpose()?
            .map(Mutex::new),
    };
    let schedule = Schedule {
        openings: arguments.openings,
        opening_actions: arguments.opening_actions,
        max_actions: arguments.max_actions,
        adjudicate_cp: arguments.adjudicate_cp,
        seed: arguments.seed,
    };
    let explicit = parse_anchors(&arguments.anchor)?;
    if arguments.calibrate {
        return calibrate(&arguments, &context, schedule, &explicit);
    }
    let rulesets = arguments.ruleset.rulesets();
    let players = dedupe(parse_bots(&arguments.bots)?)
        .into_iter()
        .filter(|bot| rulesets.iter().any(|&ruleset| bot.supports(ruleset)))
        .collect::<Vec<_>>();
    if players.len() < 2 {
        return Err("a tournament needs at least two distinct bots".into());
    }
    context.check(&players)?;
    let started = Instant::now();
    let played = round_robin(&players, &rulesets, &context, schedule);
    let ratings = Ratings::fit(
        players.len(),
        &played,
        anchors_for(&players, &explicit),
        arguments.bootstrap,
        arguments.seed,
    );
    let labels = players
        .iter()
        .map(|&bot| context.label(bot))
        .collect::<Vec<_>>();
    let keys = players.iter().map(|bot| bot.key()).collect::<Vec<_>>();
    print!(
        "{}",
        standings(
            &format!("Standings ({:.1?})", started.elapsed()),
            &labels,
            &played,
            &ratings
        )?
    );
    if let Some(path) = &arguments.csv {
        eprintln!("wrote {}", path.display());
    }
    if let Some(path) = &arguments.save_ratings {
        std::fs::write(path, format!("{{\"ratings\": {}}}\n", ratings.json(&keys)))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn calibrate(
    arguments: &Arguments,
    context: &Context,
    schedule: Schedule,
    explicit: &HashMap<String, f64>,
) -> Result<(), Box<dyn Error>> {
    let ladder = dedupe(parse_bots(&arguments.ladder)?);
    if ladder.len() < 2 || ladder.iter().any(|bot| bot.standard_only()) {
        return Err("--ladder needs at least two rungs that can play 5D chess".into());
    }
    let references = dedupe(parse_bots(&arguments.references)?);
    if !references.iter().any(|bot| bot.calibrated_elo().is_some())
        && !references
            .iter()
            .any(|bot| explicit.contains_key(&bot.key()))
    {
        return Err("--references needs at least one calibrated sf-<elo> level".into());
    }
    let bots = parse_bots(&arguments.bots)?
        .into_iter()
        .filter(|bot| !ladder.contains(bot) && !references.contains(bot))
        .collect::<Vec<_>>();
    let started = Instant::now();

    // Stage 1: standard chess, on the Stockfish scale.
    let standard_players = dedupe(
        ladder
            .iter()
            .chain(&references)
            .chain(&bots)
            .copied()
            .filter(|bot| bot.supports(Ruleset::Standard)),
    );
    context.check(&standard_players)?;
    let standard_played = round_robin(&standard_players, &[Ruleset::Standard], context, schedule);
    let standard = Ratings::fit(
        standard_players.len(),
        &standard_played,
        anchors_for(&standard_players, explicit),
        arguments.bootstrap,
        arguments.seed,
    );
    let standard_labels = standard_players
        .iter()
        .map(|&bot| context.label(bot))
        .collect::<Vec<_>>();
    print!(
        "{}",
        standings(
            "Stage 1 · standard chess, anchored to Stockfish",
            &standard_labels,
            &standard_played,
            &standard
        )?
    );

    // Stage 2: 5D chess, with the ladder held at its standard-chess ratings.
    let rated = |players: &[Bot], bot: &Bot| {
        players
            .iter()
            .position(|other| other == bot)
            .map(|index| standard.values[index])
    };
    let multiverse_players = dedupe(
        ladder
            .iter()
            .chain(bots.iter().filter(|bot| !bot.standard_only()))
            .copied(),
    );
    let multiverse_played = round_robin(
        &multiverse_players,
        &[Ruleset::Multiverse, Ruleset::MultiverseVariant],
        context,
        Schedule {
            seed: schedule.seed ^ 0x5d,
            ..schedule
        },
    );
    let ladder_anchors = multiverse_players
        .iter()
        .map(|bot| {
            ladder
                .contains(bot)
                .then(|| rated(&standard_players, bot))
                .flatten()
        })
        .collect::<Vec<_>>();
    let multiverse = Ratings::fit(
        multiverse_players.len(),
        &multiverse_played,
        ladder_anchors,
        arguments.bootstrap,
        arguments.seed,
    );
    let multiverse_labels = multiverse_players
        .iter()
        .map(|&bot| context.label(bot))
        .collect::<Vec<_>>();
    print!(
        "{}",
        standings(
            "Stage 2 · 5D chess, anchored to the ladder's standard-chess ratings",
            &multiverse_labels,
            &multiverse_played,
            &multiverse
        )?
    );

    let (report, ratio) = transfer_check(
        &ladder,
        context,
        &standard_players,
        &standard,
        &multiverse_players,
        &multiverse_played,
        arguments.seed,
    )?;
    let mut output = report;
    writeln!(output, "\nCalibration took {:.1?}.", started.elapsed())?;
    print!("{output}");

    let standard_keys = standard_players
        .iter()
        .map(|bot| bot.key())
        .collect::<Vec<_>>();
    let multiverse_keys = multiverse_players
        .iter()
        .map(|bot| bot.key())
        .collect::<Vec<_>>();
    if let Some(path) = &arguments.csv {
        eprintln!("wrote {}", path.display());
    }
    if let Some(path) = &arguments.save_ratings {
        std::fs::write(
            path,
            format!(
                "{{\"standard\": {}, \"5d\": {}, \"spread_ratio\": {ratio:.3}}}\n",
                standard.json(&standard_keys),
                multiverse.json(&multiverse_keys)
            ),
        )?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

/// Refits 5D with only the middle rung fixed and compares the ladder's spacing with
/// standard chess. Similar spacing means the transferred scale is credible.
fn transfer_check(
    ladder: &[Bot],
    context: &Context,
    standard_players: &[Bot],
    standard: &Ratings,
    multiverse_players: &[Bot],
    multiverse_played: &[Played],
    seed: u64,
) -> Result<(String, f64), std::fmt::Error> {
    let rated = |players: &[Bot], bot: &Bot| rated_in(players, &standard.values, bot);
    let middle = ladder[ladder.len() / 2];
    let free = Ratings::fit(
        multiverse_players.len(),
        multiverse_played,
        multiverse_players
            .iter()
            .map(|bot| {
                (*bot == middle)
                    .then(|| rated(standard_players, bot))
                    .flatten()
            })
            .collect(),
        0,
        seed,
    );
    let mut output = String::new();
    writeln!(
        output,
        "\nTransfer check · ladder spacing with only {} fixed",
        context.label(middle)
    )?;
    writeln!(
        output,
        "  {:<22}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Rung", "Standard", "5D free", "Std gap", "5D gap"
    )?;
    let mut previous: Option<(f64, f64)> = None;
    let mut spans = (0.0, 0.0);
    let mut monotonic = true;
    for rung in ladder {
        let standard_rating = rated(standard_players, rung).unwrap_or(f64::NAN);
        let free_rating = rated_in(multiverse_players, &free.values, rung).unwrap_or(f64::NAN);
        let gaps = previous.map(|(standard_before, free_before)| {
            (standard_rating - standard_before, free_rating - free_before)
        });
        if let Some((standard_gap, free_gap)) = gaps {
            spans.0 += standard_gap;
            spans.1 += free_gap;
            monotonic &= standard_gap > 0.0 && free_gap > 0.0;
        }
        let gap =
            |value: Option<f64>| value.map_or_else(|| "—".to_owned(), |gap| format!("{gap:+.0}"));
        writeln!(
            output,
            "  {:<22}  {standard_rating:>8.0}  {free_rating:>8.0}  {:>8}  {:>8}",
            context.label(*rung),
            gap(gaps.map(|gaps| gaps.0)),
            gap(gaps.map(|gaps| gaps.1)),
        )?;
        previous = Some((standard_rating, free_rating));
    }
    let ratio = spans.1 / spans.0;
    let verdict = if !monotonic {
        "the ladder is not monotonic in both games; treat 5D ratings as ordinal only"
    } else if (0.67..=1.5).contains(&ratio) {
        "the ladder spaces similarly in both games, so the transferred scale is credible"
    } else {
        "the ladder spaces differently in 5D; transferred ratings are compressed or stretched"
    };
    writeln!(output, "  5D/standard spread ratio {ratio:.2}: {verdict}.")?;
    Ok((output, ratio))
}

fn rated_in(players: &[Bot], ratings: &[f64], bot: &Bot) -> Option<f64> {
    players
        .iter()
        .position(|other| other == bot)
        .map(|index| ratings[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(strengths: &[f64], games_per_pair: usize) -> Vec<(usize, usize, f64)> {
        // Deterministic results whose score fractions match the Elo expectation.
        let mut results = Vec::new();
        for a in 0..strengths.len() {
            for b in a + 1..strengths.len() {
                let expected = 1.0 / (1.0 + 10f64.powf((strengths[b] - strengths[a]) / 400.0));
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let wins = (expected * games_per_pair as f64).round() as usize;
                for game in 0..games_per_pair {
                    results.push((a, b, if game < wins { 1.0 } else { 0.0 }));
                }
            }
        }
        results
    }

    #[test]
    fn elo_orders_players_by_strength_and_centres_on_zero() {
        let ratings = elo(3, &synthetic(&[200.0, 0.0, -200.0], 40), &[None; 3]);
        assert!(
            ratings[0] > ratings[1] && ratings[1] > ratings[2],
            "{ratings:?}"
        );
        assert!(ratings.iter().sum::<f64>().abs() < 1e-6);
    }

    #[test]
    fn even_results_give_equal_ratings() {
        let ratings = elo(2, &[(0, 1, 1.0), (0, 1, 0.0), (1, 0, 0.5)], &[None, None]);
        assert!((ratings[0] - ratings[1]).abs() < 1e-6, "{ratings:?}");
    }

    #[test]
    fn anchors_stay_fixed_and_place_others_on_their_scale() {
        let truth = [1_800.0, 1_500.0, 1_200.0];
        let results = synthetic(&truth, 200);
        let ratings = elo(3, &results, &[Some(1_800.0), None, Some(1_200.0)]);
        assert!((ratings[0] - 1_800.0).abs() < 1e-9 && (ratings[2] - 1_200.0).abs() < 1e-9);
        assert!((ratings[1] - 1_500.0).abs() < 25.0, "{ratings:?}");
    }

    #[test]
    fn bots_parse_to_canonical_keys() {
        assert_eq!(Bot::parse("blueshift").unwrap().key(), "blueshift-80k");
        assert_eq!(Bot::parse("blueshift-2500").unwrap(), Bot::Blueshift(2_500));
        assert_eq!(
            Bot::parse("sf-1600").unwrap().calibrated_elo(),
            Some(1_600.0)
        );
        assert!(Bot::parse("sf-skill3").unwrap().standard_only());
        for bad in [
            "sf-1000",
            "sf-skill21",
            "blueshift-0",
            "blueshift-x",
            "kasparov",
        ] {
            assert!(Bot::parse(bad).is_err(), "{bad}");
        }
        assert_eq!(
            dedupe([Bot::parse("blueshift").unwrap(), Bot::Blueshift(80_000)]).len(),
            1
        );
    }

    #[test]
    fn adjudication_awards_large_material_leads() {
        let mut game = Game::new(Ruleset::Standard);
        assert_eq!(adjudicate(&game, 400).0, 0.5);
        assert_eq!(adjudicate(&game, 0).1, "action limit");
        // 1. e4 d5 2. exd5 Qxd5 3. Nc3 Qxg2 4. Bxg2: Black is a queen down.
        for (from, to) in [
            ("e2", "e4"),
            ("d7", "d5"),
            ("e4", "d5"),
            ("d8", "d5"),
            ("b1", "c3"),
            ("d5", "g2"),
            ("f1", "g2"),
        ] {
            let parse = |square: &str| {
                let bytes = square.as_bytes();
                (i32::from(bytes[0] - b'a'), i32::from(bytes[1] - b'1'))
            };
            let action = game
                .legal_actions()
                .into_iter()
                .find(|action| {
                    matches!(action, chronofish_core::PlayerAction::Move { movement }
                        if (movement.from.x, movement.from.y) == parse(from)
                            && (movement.to.x, movement.to.y) == parse(to))
                })
                .unwrap();
            game.apply_action(action).unwrap();
        }
        let (score, reason) = adjudicate(&game, 400);
        assert_eq!(score, 1.0, "{reason}");
        assert!(reason.starts_with("adjudicated"));
    }

    fn fake_played(results: &[(usize, usize, f64)]) -> Vec<Played> {
        results
            .iter()
            .enumerate()
            .map(|(index, &(white, black, white_score))| Played {
                pairing: Pairing {
                    ruleset: Ruleset::Standard,
                    opening: index / 2,
                    white,
                    black,
                },
                result: GameResult {
                    white_score,
                    actions: 0,
                    reason: String::new(),
                    thinking: [Duration::ZERO; 2],
                    moves: [0; 2],
                },
            })
            .collect()
    }

    #[test]
    fn bootstrap_intervals_are_deterministic_and_contain_the_fit() {
        let played = fake_played(&synthetic(&[100.0, 0.0, -100.0], 20));
        let anchors = [None; 3];
        let first = bootstrap(3, &played, &anchors, 100, 9).unwrap();
        let second = bootstrap(3, &played, &anchors, 100, 9).unwrap();
        assert_eq!(first, second);
        let ratings = elo(3, &results_for_fit(&played), &anchors);
        for (rating, (low, high)) in ratings.iter().zip(&first) {
            assert!(
                low <= rating && rating <= high,
                "{rating} not in {low}..{high}"
            );
            assert!(high > low);
        }
        assert!(bootstrap(3, &played, &anchors, 0, 9).is_none());
    }

    #[test]
    fn a_tiny_tournament_completes() {
        run([
            "--bots",
            "fissure,grabablanca",
            "--openings",
            "1",
            "--ruleset",
            "standard",
            "--max-actions",
            "6",
            "--bootstrap",
            "20",
        ]
        .map(str::to_owned))
        .unwrap();
    }

    #[test]
    fn unknown_bots_and_missing_engines_are_rejected() {
        assert!(run(["--bots", "fissure,kasparov"].map(str::to_owned)).is_err());
        assert!(
            run([
                "--bots",
                "fissure,sf-1500",
                "--stockfish",
                "/nonexistent/stockfish"
            ]
            .map(str::to_owned))
            .is_err()
        );
    }
}
