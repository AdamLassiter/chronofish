use std::io::{self, Write};

use std::path::PathBuf;

use chronofish_alphazero::{AlphaNowBot, ChronofishBot, PolicyValueNetwork, SearchConfig};
use chronofish_classic::{
    BlueshiftLevel, BobbyFissureBot, DeepBlueshift, Grabablanca, MonteCarlsen,
};
use chronofish_core::{AiPlayer, Game, Move, PieceType, PlayerAction, Position, Ruleset};
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Standard,
    Multiverse,
    Variant,
}

impl From<Mode> for Ruleset {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::Standard => Self::Standard,
            Mode::Multiverse => Self::Multiverse,
            Mode::Variant => Self::MultiverseVariant,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum Opponent {
    #[default]
    Human,
    /// Chronofish: `AlphaZero` MCTS guided by the trained network.
    #[value(alias = "cpu")]
    Chronofish,
    /// Bobby Fissure: uniformly random legal actions.
    Fissure,
    /// Kairos: alpha-beta over complete 5D turns (5D modes only).
    Kairos,
    /// Loki: Monte Carlo tree search over complete 5D turns (5D modes only).
    Loki,
    /// Grabablanca: greedy one-action lookahead.
    Grabablanca,
    /// Deep Blueshift: alpha-beta search with a handcrafted evaluation.
    Blueshift,
    /// Deep Blueshift with a 1k-node budget.
    BlueshiftEasy,
    /// Deep Blueshift with a 10k-node budget.
    BlueshiftMedium,
    /// Monte Carlsen: UCT tree search with heuristic rollouts.
    Carlsen,
    /// `AlphaNow`: the network's top policy action, no search.
    AlphaNow,
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Debug terminal client for standard and 5D Chronofish"
)]
struct Cli {
    #[arg(long, value_enum, default_value_t = Mode::Multiverse)]
    mode: Mode,
    #[arg(long, value_enum, default_value_t)]
    opponent: Opponent,
    #[arg(long, default_value_t = 48)]
    ai_simulations: usize,
    /// Checkpoint for the network bots (chronofish, alpha-now); untrained if omitted.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Start from a file containing a 5DUCI move history.
    #[arg(long, conflicts_with = "pgn_history")]
    lan_history: Option<PathBuf>,
    /// Replay the main line of a 5DPGN file from the single-board 5D start.
    #[arg(long)]
    pgn_history: Option<PathBuf>,
    /// Write the submitted main line as 5DPGN when the session ends (Multiverse mode).
    #[arg(long)]
    export_pgn: Option<PathBuf>,
}

/// Builds the selected computer opponent, or `None` for a human-only game.
fn make_opponent(cli: &Cli) -> Result<Option<Box<dyn AiPlayer>>, Box<dyn std::error::Error>> {
    let search = SearchConfig {
        simulations: cli.ai_simulations,
        ..SearchConfig::default()
    };
    let model = cli
        .model
        .as_ref()
        .map(PolicyValueNetwork::load)
        .transpose()?;
    Ok(match cli.opponent {
        Opponent::Human => None,
        Opponent::Chronofish => Some(Box::new(model.map_or_else(
            || ChronofishBot::bootstrap(search, 0x0043_5055),
            |model| ChronofishBot::new(model, search, 0x0043_5055),
        ))),
        Opponent::AlphaNow => Some(Box::new(
            model.map_or_else(|| AlphaNowBot::bootstrap(0x004e_4f57), AlphaNowBot::new),
        )),
        Opponent::Fissure => Some(Box::new(BobbyFissureBot::default())),
        Opponent::Kairos => Some(Box::new(kairos::KairosBot::default())),
        Opponent::Loki => Some(Box::new(chronofish_ftxi::LokiBot::default())),
        Opponent::Grabablanca => Some(Box::new(Grabablanca::default())),
        Opponent::Blueshift => Some(Box::new(DeepBlueshift::level(BlueshiftLevel::Hard))),
        Opponent::BlueshiftEasy => Some(Box::new(DeepBlueshift::level(BlueshiftLevel::Easy))),
        Opponent::BlueshiftMedium => Some(Box::new(DeepBlueshift::level(BlueshiftLevel::Medium))),
        Opponent::Carlsen => Some(Box::new(MonteCarlsen::default())),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let mut bot = make_opponent(&cli)?;
    let mut game = Game::new(cli.mode.into());
    if let Some(player) = &bot
        && !player.supports_ruleset(game.ruleset())
    {
        return Err(format!(
            "{} cannot play {}",
            player.profile().display_name,
            game.ruleset()
        )
        .into());
    }
    if cli.export_pgn.is_some() && game.ruleset() == Ruleset::Standard {
        return Err("--export-pgn requires a 5D mode".into());
    }
    let mut history = Vec::new();
    if let Some(path) = cli.lan_history {
        (game, history) = chronofish_notation::decode_lan(&game, &std::fs::read_to_string(path)?)?;
    }
    if let Some(path) = cli.pgn_history {
        (game, history) =
            chronofish_notation::replay_5dpgn(&game, &std::fs::read_to_string(path)?)?;
    }

    println!("Chronofish — {}", game.ruleset());
    println!("Commands: move <from> <to> [promotion], submit, undo, legal, boards, help, quit");
    println!("Coordinates use TnLm:square, for example T0L0:e2 T0L0:e4.");
    loop {
        render(&game);
        if game.outcome().is_some() {
            break;
        }
        if let Some(player) = bot
            .as_mut()
            .filter(|_| game.turn() == chronofish_core::Color::Black)
        {
            let decision = player.choose_action(&game).map_err(io::Error::other)?;
            println!(
                "{} plays {:?} (value {:?})",
                player.profile().display_name,
                decision.action,
                decision.root_value
            );
            game.apply_action(decision.action)?;
            history.push(decision.action);
            continue;
        }
        print!("> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let words = line.split_whitespace().collect::<Vec<_>>();
        match words.as_slice() {
            [] => {}
            ["quit" | "q"] => break,
            ["help" | "h"] => {
                println!("move T0L0:e2 T0L0:e4 | submit | undo | legal | boards | quit");
            }
            ["boards"] => render_all(&game),
            ["legal"] => {
                for action in game.legal_actions() {
                    println!("  {action:?}");
                }
            }
            ["submit"] => match game.submit_turn() {
                Ok(()) => history.push(PlayerAction::SubmitTurn),
                Err(error) => println!("error: {error}"),
            },
            ["undo"] => match game.undo_staged_move() {
                Ok(()) => {
                    history.pop();
                }
                Err(error) => println!("error: {error}"),
            },
            ["move" | "m", from, to] => {
                if let Some(action) = play(&mut game, from, to, None) {
                    history.push(action);
                }
            }
            ["move" | "m", from, to, promotion] => {
                if let Some(action) = play(&mut game, from, to, Some(promotion)) {
                    history.push(action);
                }
            }
            _ => println!("Unknown command. Type help."),
        }
    }
    if let Some(path) = cli.export_pgn {
        let text = chronofish_notation::export_5dpgn_for(game.ruleset(), &history)?;
        let mut file = std::fs::File::create_new(path)?;
        file.write_all(text.as_bytes())?;
    }
    Ok(())
}

fn play(game: &mut Game, from: &str, to: &str, promotion: Option<&&str>) -> Option<PlayerAction> {
    let result = parse_position(from).and_then(|from| {
        let to = parse_position(to)?;
        let promotion = promotion.map(|value| parse_promotion(value)).transpose()?;
        let action = PlayerAction::Move {
            movement: Move {
                from,
                to,
                promotion,
            },
        };
        game.apply_action(action)
            .map_err(|error| error.to_string())?;
        Ok(action)
    });
    match result {
        Ok(action) => Some(action),
        Err(error) => {
            println!("error: {error}");
            None
        }
    }
}

fn parse_position(value: &str) -> Result<Position, String> {
    let (board, square) = value
        .split_once(':')
        .ok_or_else(|| "expected TnLm:square".to_string())?;
    let board = board
        .strip_prefix('T')
        .ok_or_else(|| "board must begin with T".to_string())?;
    let (time, timeline) = board
        .split_once('L')
        .ok_or_else(|| "board must contain L".to_string())?;
    let bytes = square.as_bytes();
    if bytes.len() != 2 || !(b'a'..=b'h').contains(&bytes[0]) || !(b'1'..=b'8').contains(&bytes[1])
    {
        return Err("square must be a1 through h8".to_string());
    }
    Ok(Position {
        timeline_id: timeline.parse().map_err(|_| "invalid timeline")?,
        time: time.parse().map_err(|_| "invalid time")?,
        x: i32::from(bytes[0] - b'a'),
        y: i32::from(bytes[1] - b'1'),
    })
}

fn parse_promotion(value: &str) -> Result<PieceType, String> {
    match value.to_ascii_lowercase().as_str() {
        "q" | "queen" => Ok(PieceType::Queen),
        "r" | "rook" => Ok(PieceType::Rook),
        "b" | "bishop" => Ok(PieceType::Bishop),
        "n" | "knight" => Ok(PieceType::Knight),
        "rq" | "royal-queen" => Ok(PieceType::RoyalQueen),
        "p" | "princess" => Ok(PieceType::Princess),
        "d" | "dragon" => Ok(PieceType::Dragon),
        "u" | "unicorn" => Ok(PieceType::Unicorn),
        _ => Err("unknown promotion piece".to_string()),
    }
}

fn render(game: &Game) {
    println!("\n{} — {}", game.turn(), game.message());
    for timeline in game
        .timelines()
        .iter()
        .filter(|timeline| game.is_active_timeline(timeline.id))
    {
        if let Some(board) = timeline
            .boards
            .last()
            .filter(|board| board.time == game.present_time().unwrap_or_default())
        {
            render_board(timeline.id, board);
        }
    }
    if game.can_submit() {
        println!("Turn is complete; enter submit.");
    }
}

fn render_all(game: &Game) {
    for timeline in game.timelines() {
        for board in &timeline.boards {
            render_board(timeline.id, board);
        }
    }
}

fn render_board(timeline: i32, board: &chronofish_core::BoardSnapshot) {
    println!("T{}L{} ({})", board.time, timeline, board.side_to_move);
    for y in (0..8).rev() {
        print!("{} ", y + 1);
        for square in &board.board[y] {
            print!("{} ", square.map_or('.', piece_char));
        }
        println!();
    }
    println!("  a b c d e f g h");
}

fn piece_char(piece: chronofish_core::Piece) -> char {
    let value = match piece.piece_type {
        PieceType::King => 'k',
        PieceType::CommonKing => 'c',
        PieceType::Queen => 'q',
        PieceType::RoyalQueen => 'y',
        PieceType::Princess => 's',
        PieceType::Rook => 'r',
        PieceType::Bishop => 'b',
        PieceType::Unicorn => 'u',
        PieceType::Dragon => 'd',
        PieceType::Knight => 'n',
        PieceType::Pawn => 'p',
        PieceType::Brawn => 'w',
    };
    if piece.color == chronofish_core::Color::White {
        value.to_ascii_uppercase()
    } else {
        value
    }
}
