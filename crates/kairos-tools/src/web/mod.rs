//! Support for the web front end: JSON, game records, 5dpgn notation, the analysis game tree
//! and the JSON command interface used by the WASM build.

pub mod api;
pub mod json;
pub use chronofish_notation::pgn;
pub mod record;
pub mod tree;
