#[derive(Clone)]
struct AppState {
    // rooms is the in-memory multiplayer store; root points at the workspace for
    // static file and WASM artifact lookup.
    rooms: Arc<Mutex<HashMap<String, Room>>>,
    root: Arc<PathBuf>,
    log_root: Arc<PathBuf>,
}

#[derive(Clone)]
struct Room {
    // The game payload is intentionally opaque JSON. Browser engines enforce
    // legality; the server is only a synchronization relay.
    id: String,
    game: Option<Value>,
    players: Players,
    updated_at: u128,
    events: broadcast::Sender<ServerEvent>,
}

#[derive(Clone, Default)]
struct Players {
    // Tokens, not sockets, own seats so a reconnecting tab can reclaim its color.
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
    // Reused by join/state/reset routes; optional fields let each route provide
    // only the data it needs.
    color: Option<String>,
    token: Option<String>,
    game: Option<Value>,
    message: Option<String>,
}

#[derive(Deserialize)]
struct LogBody {
    notation: Option<String>,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    room: Option<PublicRoom>,
}
