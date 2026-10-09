use std::collections::BTreeMap;

use chronofish_core::{
    BOARD_SIZE, Color, Game, PieceType, PlayerAction, Position, Ruleset, TimelineOwner,
};
use serde::{Deserialize, Serialize};

pub const BOARD_PLANES: usize = 24;
pub const BOARD_METADATA_FEATURES: usize = 15;
pub const GLOBAL_FEATURES: usize = 9;
pub const ACTION_FEATURES: usize = 32;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BoardCoordinate {
    pub timeline: i32,
    pub time: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EncodedBoard {
    pub coordinate: BoardCoordinate,
    pub planes: Vec<f32>,
    pub metadata: [f32; BOARD_METADATA_FEATURES],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EncodedAction {
    pub features: [f32; ACTION_FEATURES],
    pub source_board: Option<usize>,
    pub destination_board: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EncodedPosition {
    pub global: [f32; GLOBAL_FEATURES],
    pub boards: Vec<EncodedBoard>,
    pub actions: Vec<EncodedAction>,
}

#[must_use]
pub fn encode(game: &Game, actions: &[PlayerAction]) -> EncodedPosition {
    let present = game.present_time().unwrap_or_default();
    let board_count = game
        .timelines()
        .iter()
        .map(|timeline| timeline.boards.len())
        .sum::<usize>();
    let active_count = game
        .timelines()
        .iter()
        .filter(|timeline| game.is_active_timeline(timeline.id))
        .count();
    let global = [
        side_sign(game.turn()),
        f32::from(game.ruleset() != Ruleset::Standard),
        f32::from(game.ruleset() == Ruleset::MultiverseVariant),
        f32::from(game.can_submit()),
        f32::from(game.has_staged_moves()),
        bounded(present),
        (board_count as f32).ln_1p() / 6.0,
        (game.timelines().len() as f32).ln_1p() / 4.0,
        (active_count as f32).ln_1p() / 4.0,
    ];

    let mut boards = Vec::new();
    for timeline in game.timelines() {
        let snapshots: &[chronofish_core::BoardSnapshot] = if game.ruleset() == Ruleset::Standard {
            timeline
                .boards
                .last()
                .map(std::slice::from_ref)
                .unwrap_or_default()
        } else {
            &timeline.boards
        };
        for snapshot in snapshots {
            let coordinate = BoardCoordinate {
                timeline: timeline.id,
                time: snapshot.time,
            };
            let mut planes = vec![0.0; BOARD_PLANES * BOARD_SIZE * BOARD_SIZE];
            for (y, rank) in snapshot.board.iter().enumerate() {
                for (x, piece) in rank.iter().enumerate() {
                    if let Some(piece) = piece {
                        let plane = usize::from(piece.color == Color::Black) * 12
                            + piece_index(piece.piece_type);
                        planes[plane * BOARD_SIZE * BOARD_SIZE + y * BOARD_SIZE + x] = 1.0;
                    }
                }
            }
            let latest = timeline
                .boards
                .last()
                .is_some_and(|board| board.time == snapshot.time);
            let metadata = [
                bounded(snapshot.time - present),
                bounded(timeline.row),
                side_sign(snapshot.side_to_move),
                f32::from(latest),
                f32::from(game.is_active_timeline(timeline.id)),
                f32::from(latest && snapshot.time == present),
                f32::from(timeline.owner == TimelineOwner::White),
                f32::from(timeline.owner == TimelineOwner::Black),
                f32::from(timeline.owner == TimelineOwner::Neutral),
                f32::from(snapshot.castling.white_kingside),
                f32::from(snapshot.castling.white_queenside),
                f32::from(snapshot.castling.black_kingside),
                f32::from(snapshot.castling.black_queenside),
                f32::from(snapshot.en_passant.is_some()),
                (f32::from(snapshot.halfmove_clock) / 100.0).min(1.0),
            ];
            boards.push(EncodedBoard {
                coordinate,
                planes,
                metadata,
            });
        }
    }
    boards.sort_by_key(|board| (board.coordinate.timeline, board.coordinate.time));
    let indices = boards
        .iter()
        .enumerate()
        .map(|(index, board)| (board.coordinate, index))
        .collect::<BTreeMap<_, _>>();
    let actions = actions
        .iter()
        .copied()
        .map(|action| encode_action(game, action, &indices))
        .collect();
    EncodedPosition {
        global,
        boards,
        actions,
    }
}

#[must_use]
pub fn encode_action(
    game: &Game,
    action: PlayerAction,
    board_indices: &BTreeMap<BoardCoordinate, usize>,
) -> EncodedAction {
    let mut features = [0.0; ACTION_FEATURES];
    features[0] = 1.0;
    features[1] = side_sign(game.turn());
    let (source_board, destination_board) = match action {
        PlayerAction::SubmitTurn => {
            features[2] = 1.0;
            (None, None)
        }
        PlayerAction::Move { movement } => {
            features[3] = 1.0;
            if let Some(piece) = piece_at(game, movement.from) {
                features[4 + piece_index(piece.piece_type)] = 1.0;
            }
            features[16] = unit_square(movement.from.x);
            features[17] = unit_square(movement.from.y);
            features[18] = unit_square(movement.to.x);
            features[19] = unit_square(movement.to.y);
            features[20] = bounded(movement.from.time - game.present_time().unwrap_or_default());
            features[21] = bounded(movement.from.timeline_id);
            features[22] = bounded(movement.to.time - movement.from.time);
            features[23] = bounded(movement.to.timeline_id - movement.from.timeline_id);
            features[24] = f32::from(
                movement.from.timeline_id != movement.to.timeline_id
                    || movement.from.time != movement.to.time,
            );
            features[25] =
                f32::from(movement.from.x == movement.to.x && movement.from.y == movement.to.y);
            features[26] =
                f32::from(matches!(movement.to.x, 0 | 7) || matches!(movement.to.y, 0 | 7));
            features[27] = f32::from(movement.promotion.is_some());
            if let Some(promotion) = movement.promotion {
                features[28 + promotion_group(promotion)] = 1.0;
            }
            let from = BoardCoordinate {
                timeline: movement.from.timeline_id,
                time: movement.from.time,
            };
            let to = BoardCoordinate {
                timeline: movement.to.timeline_id,
                time: movement.to.time,
            };
            (
                board_indices.get(&from).copied(),
                board_indices.get(&to).copied(),
            )
        }
    };
    EncodedAction {
        features,
        source_board,
        destination_board,
    }
}

fn piece_at(game: &Game, position: Position) -> Option<chronofish_core::Piece> {
    let x = usize::try_from(position.x).ok()?;
    let y = usize::try_from(position.y).ok()?;
    game.timelines()
        .iter()
        .find(|timeline| timeline.id == position.timeline_id)?
        .boards
        .iter()
        .find(|board| board.time == position.time)?
        .board
        .get(y)?
        .get(x)
        .copied()
        .flatten()
}

const fn piece_index(piece: PieceType) -> usize {
    match piece {
        PieceType::King => 0,
        PieceType::CommonKing => 1,
        PieceType::Queen => 2,
        PieceType::RoyalQueen => 3,
        PieceType::Princess => 4,
        PieceType::Rook => 5,
        PieceType::Bishop => 6,
        PieceType::Unicorn => 7,
        PieceType::Dragon => 8,
        PieceType::Knight => 9,
        PieceType::Pawn => 10,
        PieceType::Brawn => 11,
    }
}

const fn promotion_group(piece: PieceType) -> usize {
    match piece {
        PieceType::Queen | PieceType::RoyalQueen => 0,
        PieceType::Rook | PieceType::Princess => 1,
        PieceType::Bishop | PieceType::Dragon => 2,
        _ => 3,
    }
}

const fn side_sign(side: Color) -> f32 {
    match side {
        Color::White => 1.0,
        Color::Black => -1.0,
    }
}
fn unit_square(value: i32) -> f32 {
    value as f32 / 3.5 - 1.0
}
fn bounded(value: i32) -> f32 {
    (value as f32 / 8.0).tanh()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_rulesets_encode_with_64_square_piece_planes() {
        for ruleset in [
            Ruleset::Standard,
            Ruleset::Multiverse,
            Ruleset::MultiverseVariant,
        ] {
            let game = Game::new(ruleset);
            let actions = game.legal_actions();
            let encoded = encode(&game, &actions);
            assert_eq!(encoded.boards.len(), 1);
            assert_eq!(encoded.boards[0].planes.len(), BOARD_PLANES * 64);
            assert_eq!(encoded.actions.len(), actions.len());
        }
    }
}
