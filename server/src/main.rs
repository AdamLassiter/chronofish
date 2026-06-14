// The server is deliberately thin: it serves the static frontend/WASM artifact
// and stores ephemeral multiplayer room snapshots for browser synchronization.
use std::{
    collections::HashMap,
    convert::Infallible,
    env,
    fs::{self, OpenOptions},
    io::Write,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    extract::{Path as AxumPath, State},
    http::{header, HeaderMap, HeaderValue, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
        Response,
    },
    routing::{any, get, post},
    Json,
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use uuid::Uuid;

include!("state.rs");
include!("rooms.rs");
include!("logging.rs");
include!("routes.rs");
include!("static_files.rs");
include!("bootstrap.rs");

#[cfg(test)]
mod static_file_tests {
    use super::*;

    #[test]
    fn favicon_is_compiled_into_the_server() {
        let asset = embedded_static_asset("/favicon.svg").expect("favicon should be embedded");
        assert_eq!(asset.content_type, "image/svg+xml");
        assert!(asset.bytes.starts_with(b"<svg") || asset.bytes.starts_with(b"<?xml"));
        assert!(asset.bytes.len() > 100);
    }

    #[test]
    fn unknown_assets_are_not_treated_as_embedded() {
        assert!(embedded_static_asset("/styles.css").is_none());
    }
}
