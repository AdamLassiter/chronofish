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
