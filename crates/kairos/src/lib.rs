//! kairos: a 5D Chess With Multiverse Time Travel engine.
#![allow(dropping_copy_types)]
pub mod core {
    pub use chronofish_5d_rules_reference::core::*;
    pub mod variants {
        pub use chronofish_5d_rules_reference::core::variants::*;
        pub use chronofish_notation::{
            apply_uci_moves, parse_board_fen_metadata, position_from_uci,
        };
    }
    pub mod perft {
        pub use chronofish_5d_rules_reference::core::perft::*;
        pub use chronofish_notation::parse_g_line;
    }
}
mod bot;
pub use bot::KairosBot;
pub mod engine;
pub mod misc;
