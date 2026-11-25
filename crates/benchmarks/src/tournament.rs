//! Round-robin bot tournament: every pair plays each opening twice with colours reversed.

// Counts here are small game tallies, far below the precision limits of f64.
#![allow(clippy::cast_precision_loss)]

use std::error::Error;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use chronofish_alphazero::{AlphaNowBot, ChronofishCpuBot, PolicyValueNetwork, SearchConfig};
use chronofish_classic::{CarlsenConfig, DeepBlueshift, Grabablanca, MonteCarlsen};
use chronofish_core::{AiPlayer, Color, Game, Ruleset};
use chronofish_random_bot::BobbyFissureBot;
use clap::{Parser, ValueEnum};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;

const ROSTER: [&str; 6] = [
    "fissure",
    "grabablanca",
    "blueshift",
    "carlsen",
    "alpha-now",
    "cpu",
];

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RuleSelection {
    Standard,
    Multiverse,
    Variant,
    All,
}

impl RuleSelection {
    fn rulesets(self) -> Vec<Ruleset> {
        match self {
            Self::Standard => vec![Ruleset::Standard],
            Self::Multiverse => vec![Ruleset::Multiverse],
            Self::Variant => vec![Ruleset::MultiverseVariant],
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
struct Arguments {
    /// Comma-separated bots: fissure, grabablanca, blueshift, carlsen, alpha-now, cpu.
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
    /// Actions after the opening before a game is scored as a draw.
    #[arg(long, default_value_t = 160)]
    max_actions: usize,
    /// Checkpoint for the network bots (alpha-now, cpu); untrained if omitted.
    #[arg(long)]
    model: Option<PathBuf>,
    /// MCTS simulations per action for the cpu bot.
    #[arg(long, default_value_t = 48)]
    simulations: usize,
    /// Write one row per game to this CSV file.
    #[arg(long)]
    csv: Option<PathBuf>,
    #[arg(long, default_value_t = 0x544f_5552_4e45_59)]
    seed: u64,
}

fn display_name(name: &str) -> &'static str {
    match name {
        "fissure" => "Bobby Fissure",
        "grabablanca" => "Grabablanca",
        "blueshift" => "Deep Blueshift",
        "carlsen" => "Monte Carlsen",
        "alpha-now" => "AlphaNow",
        "cpu" => "Chronofish CPU",
        _ => "unknown",
    }
}

fn make_bot(
    name: &str,
    seed: u64,
    model: Option<&PolicyValueNetwork>,
    simulations: usize,
) -> Box<dyn AiPlayer> {
    let search = SearchConfig {
        simulations,
        ..SearchConfig::default()
    };
    match name {
        "fissure" => Box::new(BobbyFissureBot::new(seed | 1)),
        "grabablanca" => Box::new(Grabablanca::new(seed)),
        "blueshift" => Box::new(DeepBlueshift::default()),
        "carlsen" => Box::new(MonteCarlsen::new(CarlsenConfig::default(), seed)),
        "alpha-now" => Box::new(model.map_or_else(
            || AlphaNowBot::bootstrap(seed),
            |model| AlphaNowBot::new(model.clone()),
        )),
        "cpu" => Box::new(model.map_or_else(
            || ChronofishCpuBot::bootstrap(search, seed),
            |model| ChronofishCpuBot::new(model.clone(), search, seed),
        )),
        other => unreachable!("unvalidated bot {other}"),
    }
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

struct Pairing {
    ruleset: Ruleset,
    opening: usize,
    white: usize,
    black: usize,
}

struct GameResult {
    /// White's score: 1, 0.5, or 0.
    white_score: f64,
    actions: usize,
    reason: String,
    thinking: [Duration; 2],
    moves: [usize; 2],
}

fn play(
    white: &mut (dyn AiPlayer + 'static),
    black: &mut (dyn AiPlayer + 'static),
    mut game: Game,
    max_actions: usize,
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
    match game.outcome() {
        Some(outcome) => GameResult {
            white_score: match outcome.winner {
                Some(Color::White) => 1.0,
                Some(Color::Black) => 0.0,
                None => 0.5,
            },
            actions,
            reason: format!("{:?}", outcome.reason),
            thinking,
            moves,
        },
        None => GameResult {
            white_score: 0.5,
            actions,
            reason: "action limit".to_owned(),
            thinking,
            moves,
        },
    }
}

/// Maximum-likelihood Elo with one virtual draw per pairing so perfect scores stay finite.
///
/// Each result is `(player, opponent, player's score)`. Ratings are centred on zero.
fn elo(players: usize, results: &[(usize, usize, f64)]) -> Vec<f64> {
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
    let mut ratings = vec![0.0_f64; players];
    for _ in 0..2_000 {
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
        for player in 0..players {
            if counts[player] > 0.0 {
                ratings[player] += 400.0 * (actual[player] - expected[player]) / counts[player];
            }
        }
        let mean = ratings.iter().sum::<f64>() / players.max(1) as f64;
        for rating in &mut ratings {
            *rating -= mean;
        }
    }
    ratings
}

fn format_score(score: f64) -> String {
    if score.fract() == 0.0 {
        format!("{score:.0}")
    } else {
        format!("{score:.1}")
    }
}

#[allow(clippy::too_many_lines)]
pub fn run(arguments: impl IntoIterator<Item = String>) -> Result<(), Box<dyn Error>> {
    let arguments =
        Arguments::try_parse_from(std::iter::once("tournament".to_owned()).chain(arguments))?;
    if let Some(unknown) = arguments
        .bots
        .iter()
        .find(|bot| !ROSTER.contains(&bot.as_str()))
    {
        return Err(format!("unknown bot {unknown}; choose from {}", ROSTER.join(", ")).into());
    }
    if arguments.bots.len() < 2 {
        return Err("a tournament needs at least two bots".into());
    }
    let model = arguments
        .model
        .as_ref()
        .map(PolicyValueNetwork::load)
        .transpose()?;
    let bots = &arguments.bots;
    let label = |index: usize| {
        let name = display_name(&bots[index]);
        if model.is_none() && matches!(bots[index].as_str(), "alpha-now" | "cpu") {
            format!("{name} (untrained)")
        } else {
            name.to_owned()
        }
    };
    let mut pairings = Vec::new();
    for ruleset in arguments.ruleset.rulesets() {
        for opening in 0..arguments.openings {
            for first in 0..bots.len() {
                for second in first + 1..bots.len() {
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
        bots.len(),
        arguments.openings,
        arguments.max_actions
    );
    let started = Instant::now();
    let finished = std::sync::atomic::AtomicUsize::new(0);
    let results = pairings
        .par_iter()
        .enumerate()
        .map(|(index, pairing)| {
            let ruleset_tag = match pairing.ruleset {
                Ruleset::Standard => 0,
                Ruleset::Multiverse => 1,
                Ruleset::MultiverseVariant => 2,
            };
            let opening = random_opening(
                pairing.ruleset,
                arguments.opening_actions,
                arguments.seed ^ (ruleset_tag << 48) ^ (pairing.opening as u64) << 8,
            );
            let game_seed = arguments.seed.rotate_left(17) ^ index as u64;
            let mut white = make_bot(
                &bots[pairing.white],
                game_seed,
                model.as_ref(),
                arguments.simulations,
            );
            let mut black = make_bot(
                &bots[pairing.black],
                game_seed ^ 1,
                model.as_ref(),
                arguments.simulations,
            );
            let result = play(
                white.as_mut(),
                black.as_mut(),
                opening,
                arguments.max_actions,
            );
            let done = finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            eprintln!(
                "  [{done}/{}] {} · opening {} · {} vs {} · {}-{} · {} actions · {}",
                pairings.len(),
                pairing.ruleset,
                pairing.opening + 1,
                label(pairing.white),
                label(pairing.black),
                format_score(result.white_score),
                format_score(1.0 - result.white_score),
                result.actions,
                result.reason
            );
            result
        })
        .collect::<Vec<_>>();

    let players = bots.len();
    let mut table = vec![vec![(0.0_f64, 0_usize); players]; players];
    let mut record = vec![[0_usize; 3]; players];
    let mut thinking = vec![(Duration::ZERO, 0_usize); players];
    let mut elo_results = Vec::with_capacity(results.len());
    for (pairing, result) in pairings.iter().zip(&results) {
        let (white, black) = (pairing.white, pairing.black);
        table[white][black].0 += result.white_score;
        table[white][black].1 += 1;
        table[black][white].0 += 1.0 - result.white_score;
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
        record[white][outcome(result.white_score)] += 1;
        record[black][outcome(1.0 - result.white_score)] += 1;
        for (player, side) in [(white, 0), (black, 1)] {
            thinking[player].0 += result.thinking[side];
            thinking[player].1 += result.moves[side];
        }
        elo_results.push((white, black, result.white_score));
    }
    let ratings = elo(players, &elo_results);
    let mut order = (0..players).collect::<Vec<_>>();
    order.sort_by(|&a, &b| ratings[b].total_cmp(&ratings[a]));

    let width = (0..players)
        .map(|index| label(index).len())
        .max()
        .unwrap_or(4)
        .max(4);
    let mut output = String::new();
    writeln!(output, "\nStandings ({:.1?})", started.elapsed())?;
    writeln!(
        output,
        "  {:<width$}  {:>5}  {:>4} {:>4} {:>4}  {:>6}  {:>6}  {:>9}",
        "Bot", "Games", "W", "D", "L", "Score", "Elo", "ms/action"
    )?;
    for &player in &order {
        let [wins, draws, losses] = record[player];
        let games = wins + draws + losses;
        let score = (wins as f64 + draws as f64 * 0.5) / games.max(1) as f64;
        let (time, moves) = thinking[player];
        writeln!(
            output,
            "  {:<width$}  {games:>5}  {wins:>4} {draws:>4} {losses:>4}  {:>5.1}%  {:>+6.0}  {:>9.1}",
            label(player),
            score * 100.0,
            ratings[player],
            time.as_secs_f64() * 1_000.0 / moves.max(1) as f64,
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
            format!("#{} {}", rank + 1, label(row))
        )?;
        for &column in &order {
            let cell = if row == column {
                "—".to_owned()
            } else {
                let (score, games) = table[row][column];
                format!("{}/{games}", format_score(score))
            };
            write!(output, "  {cell:>7}")?;
        }
        writeln!(output)?;
    }
    print!("{output}");

    if let Some(path) = &arguments.csv {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(
            file,
            "ruleset,opening,white,black,white_score,actions,reason"
        )?;
        for (pairing, result) in pairings.iter().zip(&results) {
            writeln!(
                file,
                "{},{},{},{},{},{},{}",
                pairing.ruleset,
                pairing.opening + 1,
                bots[pairing.white],
                bots[pairing.black],
                result.white_score,
                result.actions,
                result.reason
            )?;
        }
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elo_orders_players_by_strength_and_centres_on_zero() {
        // 0 beats 1 three games in four; 1 beats 2 likewise; 0 sweeps 2.
        let mut results = Vec::new();
        for (a, b, wins) in [(0, 1, 3), (1, 2, 3), (0, 2, 4)] {
            for game in 0..4 {
                results.push((a, b, if game < wins { 1.0 } else { 0.0 }));
            }
        }
        let ratings = elo(3, &results);
        assert!(
            ratings[0] > ratings[1] && ratings[1] > ratings[2],
            "{ratings:?}"
        );
        assert!(ratings.iter().sum::<f64>().abs() < 1e-6);
        assert!(ratings.iter().all(|rating| rating.is_finite()));
    }

    #[test]
    fn even_results_give_equal_ratings() {
        let ratings = elo(2, &[(0, 1, 1.0), (0, 1, 0.0), (1, 0, 0.5)]);
        assert!((ratings[0] - ratings[1]).abs() < 1e-6, "{ratings:?}");
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
        ]
        .map(str::to_owned))
        .unwrap();
    }

    #[test]
    fn unknown_bots_are_rejected() {
        assert!(run(["--bots", "fissure,kasparov"].map(str::to_owned)).is_err());
    }
}
