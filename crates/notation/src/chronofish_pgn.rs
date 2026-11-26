use std::fmt::Write;

use chronofish_core::{Color, Game, Move, PieceType, PlayerAction, Position, Ruleset};

/// Writes the submitted main line from the single-board Standard 5D start.
/// A final draft is omitted.
///
/// # Errors
///
/// Returns an error if an action cannot be applied to the position.
pub fn export_5dpgn(actions: &[PlayerAction]) -> Result<String, String> {
    export_5dpgn_for(Ruleset::Multiverse, actions)
}

/// Write the submitted main line for the selected 5D ruleset.
pub fn export_5dpgn_for(rules: Ruleset, actions: &[PlayerAction]) -> Result<String, String> {
    export_5dpgn_from(Game::new(rules), actions)
}

/// Write the submitted main line using the supplied starting game and its rules.
///
/// # Errors
/// Returns an error for a non-5D start or an action that cannot be applied.
pub fn export_5dpgn_from(mut game: Game, actions: &[PlayerAction]) -> Result<String, String> {
    let rules = game.ruleset();
    let setup =
        chronofish_core::reference_setup(rules).ok_or("5DPGN export requires a 5D ruleset")?;
    let mut out = if rules == Ruleset::Multiverse {
        String::from("[Board \"Standard\"]\n[Mode \"5D\"]\n\n")
    } else {
        format!(
            "[Board \"Custom\"]\n[Mode \"5D\"]\n[Size \"8x8\"]\n[{}:0:1:w]\n\n",
            setup.boards[0].fen
        )
    };
    let mut draft = Vec::new();
    let mut turn = 1;
    for (index, &action) in actions.iter().enumerate() {
        let side = game.turn();
        if let PlayerAction::Move { movement } = action {
            draft.push(
                move_text(&game, movement)
                    .ok_or_else(|| format!("invalid move at action {}", index + 1))?,
            );
        }
        game.apply_action(action)
            .map_err(|err| format!("action {}: {err}", index + 1))?;
        if action == PlayerAction::SubmitTurn || game.outcome().is_some() {
            let color = if side == Color::White { 'w' } else { 'b' };
            writeln!(out, "{turn}{color}. {}", draft.join(" ")).expect("write to string");
            draft.clear();
            if side == Color::Black {
                turn += 1;
            }
        }
    }
    if let Some(outcome) = game.outcome() {
        let result = match outcome.winner {
            Some(Color::White) => "1-0",
            Some(Color::Black) => "0-1",
            None => "1/2-1/2",
        };
        writeln!(out, "{result}").expect("write to string");
    }
    Ok(out)
}

fn move_text(game: &Game, movement: Move) -> Option<String> {
    let Move {
        from,
        to,
        promotion,
    } = movement;
    let description = game.describe_move(movement)?;
    let board = |p: Position| format!("({}T{})", p.timeline_id, p.time.div_euclid(2) + 1);
    let square = |p: Position| format!("{}{}", char::from(b'a' + p.x as u8), p.y + 1);
    let mut text = board(from);
    if description.castle {
        text.push_str(if to.x > from.x { "O-O" } else { "O-O-O" });
        return Some(text);
    }
    let jump = from.timeline_id != to.timeline_id || from.time != to.time;
    if jump || description.piece != PieceType::Pawn {
        text.push(description.piece.notation_letter());
    }
    text.push_str(&square(from));
    if jump {
        let branch = description.creates_timeline;
        text.push_str(if branch { ">>" } else { ">" });
    }
    if description.capture {
        text.push('x');
    }
    if jump {
        text.push_str(&board(to));
    }
    text.push_str(&square(to));
    if let Some(piece) = promotion {
        text.push('=');
        text.push(piece.notation_letter());
    }
    Some(text)
}
