//! 5D chess notation and replay adapters.

#[cfg(feature = "chronofish")]
mod chronofish;
#[cfg(feature = "chronofish")]
mod chronofish_pgn;
#[cfg(feature = "chronofish")]
mod lan;
#[cfg(feature = "chronofish")]
pub mod legacy;
pub mod pgn;
mod position;

#[cfg(feature = "chronofish")]
pub use chronofish::replay_5dpgn;
#[cfg(feature = "chronofish")]
pub use chronofish_pgn::{export_5dpgn, export_5dpgn_for};
#[cfg(feature = "chronofish")]
pub use lan::{decode_lan, format_lan, replay_lan};
pub use position::{
    apply_uci_moves, parse_board_fen_metadata, parse_g_line, position_from_uci, replay_checked_uci,
};
