#[tokio::main]
async fn main() {
    let port = env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(5173);
    let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let root = workspace_root();

    // One shared AppState is cloned into every handler. Room operations are small
    // and synchronous, so a mutex around the room map is enough for this server.
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
    // Static serving needs the workspace root so it can find both web/ and
    // Cargo's target/wasm32-unknown-unknown output.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("server crate should live under the workspace root")
        .to_path_buf()
}
