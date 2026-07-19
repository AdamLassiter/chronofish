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

    #[test]
    fn gpu_shaders_are_compiled_into_the_server() {
        let search = embedded_static_asset("/shaders/search/frontier_expand.wgsl")
            .expect("search shader should be embedded");
        let training = embedded_static_asset("/shaders/training/project_features.wgsl")
            .expect("training shader should be embedded");

        assert_eq!(search.content_type, "text/plain; charset=utf-8");
        assert_eq!(training.content_type, "text/plain; charset=utf-8");
        assert!(search.bytes.starts_with(b"struct ") || search.bytes.starts_with(b"const "));
        assert!(training.bytes.starts_with(b"struct ") || training.bytes.starts_with(b"const "));
    }

    #[test]
    fn gpu_effort_is_served_from_the_gpu_model_directory() {
        let root = workspace_root();
        let path = resolve_request_path(&root, "/ai/gpu-effort.json")
            .expect("GPU effort configuration should exist");
        assert!(path.ends_with("engine/models/gpu-v1/effort.json"));
    }

    #[test]
    fn gpu_value_model_is_served_from_the_gpu_model_directory() {
        let root = workspace_root();
        let path = resolve_request_path(&root, "/ai/value-model.cfnn")
            .expect("GPU value model should exist");
        assert!(path.ends_with("engine/models/gpu-v1/value-model.cfnn"));
    }
}
