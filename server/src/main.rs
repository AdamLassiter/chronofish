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

#[derive(Clone)]
struct AppState {
    rooms: Arc<Mutex<HashMap<String, Room>>>,
    root: Arc<PathBuf>,
}

#[derive(Clone)]
struct Room {
    id: String,
    game: Option<Value>,
    players: Players,
    updated_at: u128,
    events: broadcast::Sender<ServerEvent>,
}

#[derive(Clone, Default)]
struct Players {
    white: Option<String>,
    black: Option<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PublicRoom {
    id: String,
    game: Option<Value>,
    players: PublicPlayers,
    updated_at: u128,
}

#[derive(Clone, Serialize)]
struct PublicPlayers {
    white: bool,
    black: bool,
}

#[derive(Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum ServerEvent {
    Sync {
        room: PublicRoom,
    },
    Players {
        room: PublicRoom,
    },
    State {
        color: String,
        message: String,
        room: PublicRoom,
    },
    Reset {
        color: String,
        message: String,
        room: PublicRoom,
    },
}

#[derive(Deserialize)]
struct RoomBody {
    color: Option<String>,
    token: Option<String>,
    game: Option<Value>,
    message: Option<String>,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    room: Option<PublicRoom>,
}

#[tokio::main]
async fn main() {
    let port = env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(5173);
    let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let root = workspace_root();

    let state = AppState {
        rooms: Arc::new(Mutex::new(HashMap::new())),
        root: Arc::new(root),
    };

    let app = Router::new()
        .route("/api/version", get(server_version))
        .route("/api/rooms/:room_id", get(get_room))
        .route("/api/rooms/:room_id/events", get(room_events))
        .route("/api/rooms/:room_id/join", post(join_room))
        .route("/api/rooms/:room_id/state", post(update_room_state))
        .route("/api/rooms/:room_id/reset", post(reset_room))
        .route("/api/*path", any(unknown_api_route))
        .fallback(static_file)
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .unwrap_or_else(|error| panic!("Invalid bind address {host}:{port}: {error}"));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|error| panic!("Failed to bind {addr}: {error}"));

    println!("Chronofish dev server listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server failed");
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("server crate should live under the workspace root")
        .to_path_buf()
}

async fn server_version() -> impl IntoResponse {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

async fn get_room(
    State(state): State<AppState>,
    AxumPath(room_id): AxumPath<String>,
) -> impl IntoResponse {
    let room_id = sanitize_room_id(&room_id);
    let mut rooms = state.rooms.lock().expect("room mutex poisoned");
    let room = get_or_create_room(&mut rooms, &room_id);
    Json(public_room(room))
}

async fn room_events(
    State(state): State<AppState>,
    AxumPath(room_id): AxumPath<String>,
) -> impl IntoResponse {
    let room_id = sanitize_room_id(&room_id);
    let (initial, receiver) = {
        let mut rooms = state.rooms.lock().expect("room mutex poisoned");
        let room = get_or_create_room(&mut rooms, &room_id);
        (public_room(room), room.events.subscribe())
    };

    let initial: Vec<Result<Event, Infallible>> = vec![Ok(Event::default()
        .json_data(ServerEvent::Sync { room: initial })
        .expect("server event should serialize"))];
    let updates = BroadcastStream::new(receiver).filter_map(|result| {
        result.ok().map(|event| {
            Ok(Event::default()
                .json_data(event)
                .expect("server event should serialize"))
        })
    });

    Sse::new(tokio_stream::iter(initial).chain(updates))
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn join_room(
    State(state): State<AppState>,
    AxumPath(room_id): AxumPath<String>,
    Json(body): Json<RoomBody>,
) -> Response {
    let room_id = sanitize_room_id(&room_id);
    let token = body
        .token
        .filter(|token| !token.is_empty())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let requested_color = body.color.as_deref().unwrap_or("spectator");

    let mut rooms = state.rooms.lock().expect("room mutex poisoned");
    let room = get_or_create_room(&mut rooms, &room_id);
    let color = match seat_player(room, requested_color, &token) {
        Ok(color) => color,
        Err(reason) => {
            return json_error(StatusCode::CONFLICT, reason, Some(public_room(room)));
        }
    };

    if room.game.is_none() {
        room.game = body.game;
    }

    let public = public_room(room);
    let _ = room.events.send(ServerEvent::Players {
        room: public.clone(),
    });

    Json(json!({
        "ok": true,
        "token": token,
        "color": color,
        "room": public
    }))
    .into_response()
}

async fn update_room_state(
    State(state): State<AppState>,
    AxumPath(room_id): AxumPath<String>,
    Json(body): Json<RoomBody>,
) -> Response {
    mutate_game_room(state, room_id, body, "state")
}

async fn reset_room(
    State(state): State<AppState>,
    AxumPath(room_id): AxumPath<String>,
    Json(body): Json<RoomBody>,
) -> Response {
    mutate_game_room(state, room_id, body, "reset")
}

async fn unknown_api_route() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorBody {
            error: "Unknown API route".to_string(),
            room: None,
        }),
    )
}

fn mutate_game_room(state: AppState, room_id: String, body: RoomBody, action: &str) -> Response {
    let room_id = sanitize_room_id(&room_id);
    let color = body.color.unwrap_or_default();
    let token = body.token.unwrap_or_default();

    let mut rooms = state.rooms.lock().expect("room mutex poisoned");
    let room = get_or_create_room(&mut rooms, &room_id);

    if !is_seated(room, &color, &token) {
        let verb = if action == "reset" { "reset" } else { "update" };
        return json_error(
            StatusCode::FORBIDDEN,
            format!("Only a seated player can {verb} the game"),
            None,
        );
    }

    room.game = body.game;
    room.updated_at = now_millis();
    let public = public_room(room);

    let event = if action == "reset" {
        ServerEvent::Reset {
            color: color.clone(),
            message: format!("{color} reset the room."),
            room: public.clone(),
        }
    } else {
        ServerEvent::State {
            color: color.clone(),
            message: body.message.unwrap_or_default(),
            room: public.clone(),
        }
    };
    let _ = room.events.send(event);

    Json(public).into_response()
}

async fn static_file(
    State(state): State<AppState>,
    method: Method,
    uri: axum::http::Uri,
) -> Response {
    if method != Method::GET && method != Method::HEAD {
        return (
            StatusCode::METHOD_NOT_ALLOWED,
            Json(ErrorBody {
                error: "Method not allowed".to_string(),
                room: None,
            }),
        )
            .into_response();
    }

    match resolve_request_path(&state.root, uri.path()) {
        Some(path) => match tokio::fs::read(&path).await {
            Ok(bytes) => {
                let mut response = bytes.into_response();
                response.headers_mut().insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static(content_type(&path)),
                );
                response
            }
            Err(_) => (StatusCode::NOT_FOUND, "Not found").into_response(),
        },
        None => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("css") => "text/css; charset=utf-8",
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    }
}

fn resolve_request_path(root: &Path, request_path: &str) -> Option<PathBuf> {
    let requested = percent_decode_path(request_path)?;
    let requested = if requested == "/" {
        PathBuf::from("web/index.html")
    } else {
        PathBuf::from(requested.trim_start_matches('/'))
    };

    let direct = root.join(&requested);
    if is_safe_existing_path(root, &direct) {
        return Some(direct);
    }

    let web_root = root.join("web");
    let web_path = web_root.join(&requested);
    if is_safe_existing_path(&web_root, &web_path) {
        return Some(web_path);
    }

    None
}

fn is_safe_existing_path(root: &Path, path: &Path) -> bool {
    let Ok(root) = root.canonicalize() else {
        return false;
    };
    let Ok(path) = path.canonicalize() else {
        return false;
    };

    path.starts_with(root) && path.is_file()
}

fn percent_decode_path(path: &str) -> Option<String> {
    let mut decoded = String::with_capacity(path.len());
    let mut bytes = path.bytes();

    while let Some(byte) = bytes.next() {
        if byte != b'%' {
            decoded.push(byte as char);
            continue;
        }

        let high = hex_value(bytes.next()?)?;
        let low = hex_value(bytes.next()?)?;
        decoded.push((high * 16 + low) as char);
    }

    Some(decoded)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn sanitize_room_id(room_id: &str) -> String {
    let sanitized: String = room_id
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || *character == '_' || *character == '-'
        })
        .take(48)
        .collect();

    if sanitized.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        sanitized
    }
}

fn get_or_create_room<'a>(rooms: &'a mut HashMap<String, Room>, room_id: &str) -> &'a mut Room {
    rooms.entry(room_id.to_string()).or_insert_with(|| {
        let (events, _) = broadcast::channel(128);
        Room {
            id: room_id.to_string(),
            game: None,
            players: Players::default(),
            updated_at: now_millis(),
            events,
        }
    })
}

fn public_room(room: &Room) -> PublicRoom {
    PublicRoom {
        id: room.id.clone(),
        game: room.game.clone(),
        players: PublicPlayers {
            white: room.players.white.is_some(),
            black: room.players.black.is_some(),
        },
        updated_at: room.updated_at,
    }
}

fn seat_player(room: &mut Room, color: &str, token: &str) -> Result<String, String> {
    let seat = match color {
        "white" => &mut room.players.white,
        "black" => &mut room.players.black,
        _ => return Ok("spectator".to_string()),
    };

    if seat.as_deref().is_some_and(|seated| seated != token) {
        return Err(format!("{color} is already occupied"));
    }

    *seat = Some(token.to_string());
    room.updated_at = now_millis();
    Ok(color.to_string())
}

fn is_seated(room: &Room, color: &str, token: &str) -> bool {
    match color {
        "white" => room.players.white.as_deref() == Some(token),
        "black" => room.players.black.as_deref() == Some(token),
        _ => false,
    }
}

fn json_error(status: StatusCode, error: String, room: Option<PublicRoom>) -> Response {
    (status, Json(ErrorBody { error, room })).into_response()
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after Unix epoch")
        .as_millis()
}
