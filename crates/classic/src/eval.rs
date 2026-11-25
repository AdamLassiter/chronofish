//! Handcrafted, network-free position evaluation shared by the classic bots.

use chronofish_core::{Color, Game, Piece, PieceType, PlayerAction, Position};

/// Score of a won game, in centipawns. Search subtracts the ply so faster mates rank higher.
pub const MATE: i32 = 1_000_000;

/// Scores closer to [`MATE`] than this are forced wins or losses rather than evaluations.
pub const MATE_THRESHOLD: i32 = MATE - 10_000;

const CHECK_PENALTY: i32 = 60;
const TIMELINE_BONUS: i32 = 20;

/// Material value in centipawns. Royal pieces are priceless and score zero; losing them ends the game.
#[must_use]
pub const fn piece_value(piece_type: PieceType) -> i32 {
    match piece_type {
        PieceType::Pawn => 100,
        PieceType::Brawn => 120,
        PieceType::Knight | PieceType::CommonKing => 300,
        PieceType::Bishop => 325,
        PieceType::Dragon => 225,
        PieceType::Unicorn => 275,
        PieceType::Rook => 500,
        PieceType::Princess => 800,
        PieceType::Queen | PieceType::RoyalQueen => 900,
        PieceType::King => 0,
    }
}

const fn sign(color: Color) -> i32 {
    match color {
        Color::White => 1,
        Color::Black => -1,
    }
}

/// Small piece-square bonus: centralise minor pieces and advance pawns.
fn placement(piece: Piece, x: i32, y: i32) -> i32 {
    match piece.piece_type {
        PieceType::Knight | PieceType::Bishop | PieceType::Unicorn | PieceType::Dragon => {
            let distance = (2 * x - 7).abs().max((2 * y - 7).abs());
            (7 - distance) * 3
        }
        PieceType::Pawn | PieceType::Brawn => {
            let advanced = match piece.color {
                Color::White => y - 1,
                Color::Black => 6 - y,
            };
            advanced.max(0) * 6
        }
        _ => 0,
    }
}

/// Evaluation in centipawns from White's point of view.
#[must_use]
pub fn evaluate(game: &Game) -> i32 {
    if let Some(outcome) = game.outcome() {
        return outcome.winner.map_or(0, |winner| sign(winner) * MATE);
    }
    let mut score = 0;
    for timeline in game.timelines() {
        let Some(board) = timeline.boards.last() else {
            continue;
        };
        // Pieces on inactive timelines still matter, but cannot influence the present.
        let weight = if game.is_active_timeline(timeline.id) {
            2
        } else {
            1
        };
        let mut material = 0;
        for (y, row) in (0..).zip(&board.board) {
            for (x, square) in (0..).zip(row) {
                if let Some(piece) = square {
                    material += sign(piece.color)
                        * (piece_value(piece.piece_type) + placement(*piece, x, y));
                }
            }
        }
        score += material * weight / 2;
        score += match timeline.owner {
            chronofish_core::TimelineOwner::White => TIMELINE_BONUS,
            chronofish_core::TimelineOwner::Black => -TIMELINE_BONUS,
            chronofish_core::TimelineOwner::Neutral => 0,
        };
    }
    for royal in game.checked_royals() {
        if let Some(piece) = piece_at(game, royal) {
            score -= sign(piece.color) * CHECK_PENALTY;
        }
    }
    score
}

/// Evaluation from `side`'s point of view.
#[must_use]
pub fn relative(game: &Game, side: Color) -> i32 {
    sign(side) * evaluate(game)
}

/// The piece on any board in the multiverse, including past boards.
#[must_use]
pub fn piece_at(game: &Game, position: Position) -> Option<Piece> {
    let timeline = game
        .timelines()
        .iter()
        .find(|timeline| timeline.id == position.timeline_id)?;
    let board = timeline
        .boards
        .iter()
        .rev()
        .find(|board| board.time == position.time)?;
    let x = usize::try_from(position.x).ok()?;
    let y = usize::try_from(position.y).ok()?;
    *board.board.get(y)?.get(x)?
}

/// Most-valuable-victim, least-valuable-attacker ordering key for captures.
#[must_use]
pub fn capture_score(game: &Game, action: PlayerAction) -> Option<i32> {
    let PlayerAction::Move { movement } = action else {
        return None;
    };
    let victim = piece_at(game, movement.to).filter(|piece| piece.color != game.turn())?;
    let attacker = piece_at(game, movement.from).map_or(0, |piece| piece_value(piece.piece_type));
    let victim_value = match victim.piece_type {
        PieceType::King | PieceType::RoyalQueen => 10_000,
        other => piece_value(other),
    };
    Some(victim_value * 16 - attacker / 10)
}

/// Maps a centipawn score onto [-1, 1] for `AiDecision::root_value`.
#[must_use]
pub fn squash(score: i32) -> f32 {
    if score >= MATE_THRESHOLD {
        1.0
    } else if score <= -MATE_THRESHOLD {
        -1.0
    } else {
        (f64::from(score) / 400.0).tanh() as f32
    }
}

#[cfg(test)]
mod tests {
    use chronofish_core::Ruleset;

    use super::*;

    #[test]
    fn opening_positions_are_balanced() {
        for ruleset in [
            Ruleset::Standard,
            Ruleset::Multiverse,
            Ruleset::MultiverseVariant,
        ] {
            assert_eq!(evaluate(&Game::new(ruleset)), 0, "{ruleset}");
        }
    }

    #[test]
    fn squash_saturates_at_mate() {
        assert!((squash(MATE - 3) - 1.0).abs() < f32::EPSILON);
        assert!(squash(0).abs() < f32::EPSILON);
        assert!(squash(-300) < 0.0);
    }
}
