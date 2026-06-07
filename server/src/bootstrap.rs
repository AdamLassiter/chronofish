#[tokio::main]
async fn main() {
    let port = env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(5173);
    let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let root = workspace_root();
    let log_root = root.join("logs");

    // One shared AppState is cloned into every handler. Room operations are small
    // and synchronous, so a mutex around the room map is enough for this server.
    let state = AppState {
        rooms: Arc::new(Mutex::new(HashMap::new())),
        root: Arc::new(root),
        log_root: Arc::new(log_root),
    };

    // Axum 0.8 uses `{name}` and `{*name}` route captures; the old `:name`
    // syntax now fails at router construction.
    let api = training_routes(Router::new())
        .route("/api/version", get(server_version))
        .route("/api/rooms/{room_id}", get(get_room))
        .route("/api/rooms/{room_id}/events", get(room_events))
        .route("/api/rooms/{room_id}/join", post(join_room))
        .route("/api/rooms/{room_id}/state", post(update_room_state))
        .route("/api/rooms/{room_id}/reset", post(reset_room))
        .route("/api/logs/{room_id}", post(log_match_event))
        .route("/api/{*path}", any(unknown_api_route))
        .layer(middleware::from_fn(no_store_middleware));

    let app = Router::new()
        .merge(api)
        .fallback(static_file)
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .unwrap_or_else(|error| panic!("Invalid bind address {host}:{port}: {error}"));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|error| panic!("Failed to bind {addr}: {error}"));

    eprintln!("Chronofish dev server listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server failed");
}

async fn no_store_middleware(request: Request<Body>, next: Next) -> Response {
    let mut response = next.run(request).await;
    apply_no_store_headers(&mut response);
    response
}

fn apply_no_store_headers(response: &mut Response) {
    let headers = response.headers_mut();
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    headers.insert(header::PRAGMA, HeaderValue::from_static("no-cache"));
    headers.insert(header::EXPIRES, HeaderValue::from_static("0"));
}

#[cfg(feature = "frontend-training")]
fn training_routes(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/api/training/status", get(training_status))
        .route("/api/training/model", get(get_training_model).put(put_training_model))
        .route(
            "/api/training/cpu-parameters",
            get(get_training_cpu_parameters).put(put_training_cpu_parameters),
        )
        .route("/api/training/loss-logs", get(list_training_loss_logs))
        .route("/api/training/loss-logs/{room_id}", post(post_training_loss_log))
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
}

#[cfg(not(feature = "frontend-training"))]
fn training_routes(router: Router<AppState>) -> Router<AppState> {
    router
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
