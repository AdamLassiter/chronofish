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
