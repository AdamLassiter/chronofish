//! Analysis tools and fixtures inherited from Kairos.
#![allow(dropping_copy_types)]
pub use kairos::core;
pub mod engine {
    pub use crate::uci;
    pub use kairos::engine::*;
}
pub mod misc {
    pub use kairos::misc::*;
    pub mod tactics;
}
pub mod uci;
pub mod web;
