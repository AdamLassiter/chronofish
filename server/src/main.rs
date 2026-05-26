use std::{
    collections::HashMap,
    convert::Infallible,
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{Path as AxumPath, State},
    http::{header, HeaderValue, Method, StatusCode},
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
include!("routes.rs");
include!("static_files.rs");
include!("bootstrap.rs");
