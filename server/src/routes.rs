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

    // Emit one initial sync event before live updates so clients can render the
    // current room state without racing a separate GET request.
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
        // The first join can seed the room with the local engine snapshot. Later
        // joins preserve whatever state the room already holds.
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

async fn log_match_event(
    State(state): State<AppState>,
    AxumPath(room_id): AxumPath<String>,
    Json(body): Json<LogBody>,
) -> impl IntoResponse {
    let room_id = sanitize_room_id(&room_id);
    log_room_line(&state, &room_id, body.notation.as_deref().unwrap_or_default());
    Json(json!({ "ok": true }))
}

#[cfg(feature = "frontend-training")]
async fn training_status(State(state): State<AppState>) -> impl IntoResponse {
    let path = active_training_model_path(&state);
    let existing = std::fs::read(&path).ok();
    let metadata = std::fs::metadata(&path).ok();
    let cpu_path = active_cpu_parameters_path(&state);
    let cpu_existing = std::fs::read(&cpu_path).ok();
    let cpu_metadata = std::fs::metadata(&cpu_path).ok();
    Json(json!({
        "enabled": true,
        "modelPath": "engine/models/gpu-v1/value-model.cfnn",
        "resolvedModelPath": path.display().to_string(),
        "modelPresent": metadata.is_some(),
        "modelBytes": metadata.as_ref().map(|metadata| metadata.len()),
        "modelHash": existing.as_deref().map(training_model_hash),
        "cpuParametersPath": "engine/models/cpu-v1/parameters.json",
        "resolvedCpuParametersPath": cpu_path.display().to_string(),
        "cpuParametersPresent": cpu_metadata.is_some(),
        "cpuParametersBytes": cpu_metadata.as_ref().map(|metadata| metadata.len()),
        "cpuParametersHash": cpu_existing.as_deref().map(training_model_hash),
        "updatedAt": metadata
            .and_then(|metadata| metadata.modified().ok())
            .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_millis())
    }))
}

#[cfg(feature = "frontend-training")]
async fn get_training_model(State(state): State<AppState>) -> Response {
    let path = active_training_model_path(&state);
    match std::fs::read(&path) {
        Ok(bytes) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/octet-stream")],
            bytes,
        )
            .into_response(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: "No active training model has been saved.".to_string(),
                room: None,
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("Failed to read active training model: {error}"),
                room: None,
            }),
        )
            .into_response(),
    }
}

#[cfg(feature = "frontend-training")]
async fn put_training_model(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    if let Err(error) = validate_training_model(&body) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody { error, room: None }),
        )
            .into_response();
    }

    let path = active_training_model_path(&state);
    let existing = std::fs::read(&path).ok();
    let old_hash = existing.as_deref().map(training_model_hash);
    let new_hash = training_model_hash(&body);
    if existing.as_deref() == Some(body.as_ref()) {
        return Json(json!({
            "ok": true,
            "changed": false,
            "modelPath": "engine/models/gpu-v1/value-model.cfnn",
            "resolvedModelPath": path.display().to_string(),
            "modelBytes": body.len(),
            "oldHash": old_hash,
            "newHash": new_hash,
            "reason": "uploaded model bytes match the active model on disk",
            "updatedAt": now_millis()
        }))
        .into_response();
    }

    if let Some(parent) = path.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody {
                    error: format!("Failed to create model directory: {error}"),
                    room: None,
                }),
            )
                .into_response();
        }
    }

    if path.is_file() {
        let backup = path.with_file_name(format!("value-model.{}.bak.cfnn", now_millis()));
        if let Err(error) = std::fs::copy(&path, &backup) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody {
                    error: format!("Failed to back up active model: {error}"),
                    room: None,
                }),
            )
                .into_response();
        }
    }

    let tmp = path.with_file_name("value-model.tmp.cfnn");
    if let Err(error) = write_atomic(&tmp, &path, &body) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("Failed to replace active model: {error}"),
                room: None,
            }),
        )
            .into_response();
    }
    let disk_hash = std::fs::read(&path)
        .ok()
        .as_deref()
        .map(training_model_hash);

    Json(json!({
        "ok": true,
        "changed": true,
        "modelPath": "engine/models/gpu-v1/value-model.cfnn",
        "resolvedModelPath": path.display().to_string(),
        "modelBytes": body.len(),
        "oldHash": old_hash,
        "newHash": new_hash,
        "diskHash": disk_hash,
        "updatedAt": now_millis()
    }))
    .into_response()
}

#[cfg(feature = "frontend-training")]
async fn get_training_cpu_parameters(State(state): State<AppState>) -> Response {
    let path = active_cpu_parameters_path(&state);
    match std::fs::read(&path) {
        Ok(bytes) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json; charset=utf-8")],
            bytes,
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("Failed to read CPU parameters: {error}"),
                room: None,
            }),
        )
            .into_response(),
    }
}

#[cfg(feature = "frontend-training")]
async fn put_training_cpu_parameters(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    if let Err(error) = validate_cpu_parameters(&body) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody { error, room: None }),
        )
            .into_response();
    }

    let path = active_cpu_parameters_path(&state);
    let existing = std::fs::read(&path).ok();
    let old_hash = existing.as_deref().map(training_model_hash);
    let new_hash = training_model_hash(&body);
    if existing.as_deref() == Some(body.as_ref()) {
        return Json(json!({
            "ok": true,
            "changed": false,
            "modelPath": "engine/models/cpu-v1/parameters.json",
            "resolvedModelPath": path.display().to_string(),
            "modelBytes": body.len(),
            "oldHash": old_hash,
            "newHash": new_hash,
            "reason": "uploaded CPU parameters match the active parameters on disk",
            "updatedAt": now_millis()
        }))
        .into_response();
    }

    if let Some(parent) = path.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody {
                    error: format!("Failed to create CPU parameters directory: {error}"),
                    room: None,
                }),
            )
                .into_response();
        }
    }

    if path.is_file() {
        let backup = path.with_file_name(format!("parameters.{}.bak.json", now_millis()));
        if let Err(error) = std::fs::copy(&path, &backup) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody {
                    error: format!("Failed to back up active CPU parameters: {error}"),
                    room: None,
                }),
            )
                .into_response();
        }
    }

    let tmp = path.with_file_name("parameters.tmp.json");
    if let Err(error) = write_atomic(&tmp, &path, &body) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("Failed to replace active CPU parameters: {error}"),
                room: None,
            }),
        )
            .into_response();
    }
    let disk_hash = std::fs::read(&path)
        .ok()
        .as_deref()
        .map(training_model_hash);

    Json(json!({
        "ok": true,
        "changed": true,
        "modelPath": "engine/models/cpu-v1/parameters.json",
        "resolvedModelPath": path.display().to_string(),
        "modelBytes": body.len(),
        "oldHash": old_hash,
        "newHash": new_hash,
        "diskHash": disk_hash,
        "updatedAt": now_millis()
    }))
    .into_response()
}

#[cfg(feature = "frontend-training")]
async fn post_training_loss_log(
    State(state): State<AppState>,
    AxumPath(room_id): AxumPath<String>,
    Json(body): Json<Value>,
) -> Response {
    let room_id = sanitize_room_id(&room_id);
    let dir = training_loss_log_dir(&state);
    if let Err(error) = std::fs::create_dir_all(&dir) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("Failed to create loss-log directory: {error}"),
                room: None,
            }),
        )
            .into_response();
    }

    let path = dir.join(format!("{}-{}.json", now_millis(), room_id));
    let bytes = match serde_json::to_vec_pretty(&body) {
        Ok(bytes) => bytes,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: format!("Invalid loss log: {error}"),
                    room: None,
                }),
            )
                .into_response();
        }
    };
    match std::fs::write(&path, bytes) {
        Ok(()) => Json(json!({
            "ok": true,
            "path": path.display().to_string()
        }))
        .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("Failed to write loss log: {error}"),
                room: None,
            }),
        )
            .into_response(),
    }
}

#[cfg(feature = "frontend-training")]
async fn list_training_loss_logs(State(state): State<AppState>) -> impl IntoResponse {
    let dir = training_loss_log_dir(&state);
    let mut entries = Vec::new();
    if let Ok(read_dir) = std::fs::read_dir(&dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let Ok(mut value) = serde_json::from_slice::<Value>(&bytes) else {
                continue;
            };
            if let Some(object) = value.as_object_mut() {
                object.insert(
                    "logPath".to_string(),
                    Value::String(path.display().to_string()),
                );
            }
            entries.push(value);
        }
    }
    entries.sort_by(|left, right| {
        right
            .get("recordedAt")
            .and_then(Value::as_u64)
            .cmp(&left.get("recordedAt").and_then(Value::as_u64))
    });
    Json(json!({ "ok": true, "logs": entries }))
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

    // Seat ownership is the only write permission check. The server deliberately
    // does not revalidate chess legality.
    if !is_seated(room, &color, &token) {
        let verb = if action == "reset" { "reset" } else { "update" };
        return json_error(
            StatusCode::FORBIDDEN,
            format!("Only a seated player can {verb} the game"),
            None,
        );
    }

    let message = body.message.unwrap_or_default();
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
            message,
            room: public.clone(),
        }
    };
    let _ = room.events.send(event);

    Json(public).into_response()
}

#[cfg(feature = "frontend-training")]
fn active_training_model_path(state: &AppState) -> PathBuf {
    state.root.join("engine/models/gpu-v1/value-model.cfnn")
}

#[cfg(feature = "frontend-training")]
fn active_cpu_parameters_path(state: &AppState) -> PathBuf {
    state.root.join("engine/models/cpu-v1/parameters.json")
}

#[cfg(feature = "frontend-training")]
fn training_loss_log_dir(state: &AppState) -> PathBuf {
    state.root.join("logs/training-losses")
}

#[cfg(feature = "frontend-training")]
fn validate_training_model(model: &[u8]) -> Result<(), String> {
    if model.len() < 40 || &model[0..4] != b"CFNN" {
        return Err("Model must use the compact CFNN binary format.".to_string());
    }
    let version = u32::from_le_bytes(model[4..8].try_into().expect("slice length checked"));
    if version != 1 {
        return Err("Unsupported compact model version.".to_string());
    }
    if model.len() > 64 * 1024 * 1024 {
        return Err("Model is too large.".to_string());
    }
    Ok(())
}

#[cfg(feature = "frontend-training")]
fn validate_cpu_parameters(parameters: &[u8]) -> Result<(), String> {
    if parameters.len() > 1024 * 1024 {
        return Err("CPU parameters JSON is too large.".to_string());
    }
    let value: Value = serde_json::from_slice(parameters)
        .map_err(|error| format!("CPU parameters must be valid JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "CPU parameters must be a JSON object.".to_string())?;
    for required in [
        "king",
        "queen",
        "pawn",
        "mobility",
        "royal_capture_threat",
        "royal_capture_setup",
    ] {
        if !object.get(required).is_some_and(Value::is_number) {
            return Err(format!("CPU parameters missing numeric field `{required}`."));
        }
    }
    Ok(())
}

#[cfg(feature = "frontend-training")]
fn training_model_hash(model: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in model {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(feature = "frontend-training")]
fn write_atomic(tmp: &Path, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    {
        let mut file = std::fs::File::create(tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(tmp, path)
}
