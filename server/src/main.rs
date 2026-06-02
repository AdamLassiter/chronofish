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
    http::{header, HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode},
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
