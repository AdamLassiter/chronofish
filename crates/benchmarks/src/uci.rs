//! Orthodox UCI engines (Stockfish) as standard-chess opponents and rating anchors.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use chronofish_core::{AiDecision, AiPlayer, AiProfile, Game, PieceType, PlayerAction};

/// How a UCI engine's strength is limited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strength {
    /// `UCI_LimitStrength` at a calibrated Elo (Stockfish accepts 1320–3190).
    Elo(u32),
    /// `Skill Level` 0–20: weaker than the Elo floor, but uncalibrated.
    Skill(u8),
}

pub struct UciEngine {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    movetime_ms: u64,
}

impl UciEngine {
    /// Starts the engine, applies the strength limit, and begins a new game.
    ///
    /// # Errors
    ///
    /// Returns an error if the process cannot start or does not speak UCI.
    pub fn spawn(path: &Path, strength: Strength, movetime_ms: u64) -> Result<Self, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("cannot start {}: {error}", path.display()))?;
        let input = child.stdin.take().expect("stdin is piped");
        let output = BufReader::new(child.stdout.take().expect("stdout is piped"));
        let mut engine = Self {
            child,
            input,
            output,
            movetime_ms,
        };
        engine.send("uci")?;
        engine.wait_for("uciok")?;
        engine.send("setoption name Threads value 1")?;
        engine.send("setoption name Hash value 16")?;
        match strength {
            Strength::Elo(elo) => {
                engine.send("setoption name UCI_LimitStrength value true")?;
                engine.send(&format!("setoption name UCI_Elo value {elo}"))?;
            }
            Strength::Skill(level) => {
                engine.send(&format!("setoption name Skill Level value {level}"))?;
            }
        }
        engine.send("ucinewgame")?;
        engine.send("isready")?;
        engine.wait_for("readyok")?;
        Ok(engine)
    }

    fn send(&mut self, command: &str) -> Result<(), String> {
        writeln!(self.input, "{command}")
            .and_then(|()| self.input.flush())
            .map_err(|error| format!("engine closed its input: {error}"))
    }

    /// Reads lines until one starts with `prefix`, returning that line.
    fn wait_for(&mut self, prefix: &str) -> Result<String, String> {
        let mut line = String::new();
        loop {
            line.clear();
            let read = self
                .output
                .read_line(&mut line)
                .map_err(|error| format!("engine output failed: {error}"))?;
            if read == 0 {
                return Err(format!("engine exited before '{prefix}'"));
            }
            if line.trim_start().starts_with(prefix) {
                return Ok(line.trim().to_owned());
            }
        }
    }
}

impl Drop for UciEngine {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Parses `e2e4` or `e7e8q` into board squares and an optional promotion piece.
/// A board square as `(file, rank)`, both zero-based.
type Square = (i32, i32);

fn parse_uci_move(text: &str) -> Option<(Square, Square, Option<PieceType>)> {
    let bytes = text.as_bytes();
    if !(4..=5).contains(&bytes.len()) {
        return None;
    }
    let square = |file: u8, rank: u8| {
        ((b'a'..=b'h').contains(&file) && (b'1'..=b'8').contains(&rank))
            .then(|| (i32::from(file - b'a'), i32::from(rank - b'1')))
    };
    let promotion = match bytes.get(4) {
        None => None,
        Some(b'q') => Some(PieceType::Queen),
        Some(b'r') => Some(PieceType::Rook),
        Some(b'b') => Some(PieceType::Bishop),
        Some(b'n') => Some(PieceType::Knight),
        Some(_) => return None,
    };
    Some((
        square(bytes[0], bytes[1])?,
        square(bytes[2], bytes[3])?,
        promotion,
    ))
}

/// Finds the legal action matching a UCI move; castling is a two-square king move in both.
fn find_action(game: &Game, text: &str) -> Option<PlayerAction> {
    let (from, to, promotion) = parse_uci_move(text)?;
    let candidates = game
        .legal_actions()
        .into_iter()
        .filter(|action| {
            matches!(action, PlayerAction::Move { movement }
                if (movement.from.x, movement.from.y) == from
                    && (movement.to.x, movement.to.y) == to)
        })
        .collect::<Vec<_>>();
    let promotion_of = |action: &PlayerAction| match action {
        PlayerAction::Move { movement } => movement.promotion,
        PlayerAction::SubmitTurn => None,
    };
    candidates
        .iter()
        .find(|action| promotion_of(action) == promotion)
        .or_else(|| {
            // Rules engines that auto-queen expose a single unpromoted pawn move.
            (promotion == Some(PieceType::Queen))
                .then(|| {
                    candidates
                        .iter()
                        .find(|action| promotion_of(action).is_none())
                })
                .flatten()
        })
        .copied()
}

impl AiPlayer for UciEngine {
    fn profile(&self) -> AiProfile {
        AiProfile {
            username: "bot-uci-engine",
            display_name: "UCI engine",
        }
    }

    fn choose_action(&mut self, game: &Game) -> Result<AiDecision, String> {
        let fen = game
            .fen()
            .ok_or_else(|| "UCI engines only play standard chess".to_owned())?;
        self.send(&format!("position fen {fen}"))?;
        self.send(&format!("go movetime {}", self.movetime_ms))?;
        let line = self.wait_for("bestmove")?;
        let text = line.split_whitespace().nth(1).unwrap_or("(none)");
        let action = find_action(game, text)
            .ok_or_else(|| format!("engine move {text} is not legal here ({fen})"))?;
        Ok(AiDecision {
            action,
            principal_variation: vec![action],
            root_value: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use chronofish_core::Ruleset;

    use super::*;

    fn apply(game: &mut Game, text: &str) {
        let action = find_action(game, text).unwrap_or_else(|| panic!("{text} not found"));
        game.apply_action(action).unwrap();
    }

    #[test]
    fn uci_moves_parse_and_reject_garbage() {
        assert_eq!(parse_uci_move("e2e4"), Some(((4, 1), (4, 3), None)));
        assert_eq!(
            parse_uci_move("a7a8n"),
            Some(((0, 6), (0, 7), Some(PieceType::Knight)))
        );
        for bad in ["", "(none)", "e2e9", "e7e8k", "e2e4qq"] {
            assert_eq!(parse_uci_move(bad), None, "{bad}");
        }
    }

    #[test]
    fn castling_and_promotion_map_onto_legal_actions() {
        let mut game = Game::new(Ruleset::Standard);
        for text in ["e2e4", "a7a5", "g1f3", "a5a4", "f1e2", "a4a3", "e1g1"] {
            apply(&mut game, text);
        }
        assert!(
            game.fen()
                .unwrap()
                .starts_with("rnbqkbnr/1ppppppp/8/8/4P3/p4N2/PPPPBPPP/RNBQ1RK1 b")
        );
        for text in ["a3b2", "d2d3", "b2a1q"] {
            apply(&mut game, text);
        }
        let fen = game.fen().unwrap();
        assert!(
            fen.starts_with("rnbqkbnr/1ppppppp/8/8/4P3/3P1N2/P1P1BPPP/qNBQ1RK1 w"),
            "{fen}"
        );
        assert!(find_action(&game, "e1g1").is_none());
    }

    /// Runs only with a real engine: `STOCKFISH=/usr/games/stockfish cargo test -- --ignored`.
    #[test]
    #[ignore = "needs a Stockfish binary in $STOCKFISH"]
    fn stockfish_plays_a_legal_game() {
        let path = std::env::var("STOCKFISH").expect("set STOCKFISH");
        let mut engine = UciEngine::spawn(Path::new(&path), Strength::Elo(1320), 20).unwrap();
        let mut game = Game::new(Ruleset::Standard);
        for _ in 0..40 {
            if game.outcome().is_some() {
                break;
            }
            let decision = engine.choose_action(&game).unwrap();
            game.apply_action(decision.action).unwrap();
        }
    }
}
