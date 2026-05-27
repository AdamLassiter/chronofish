fn sanitize_room_id(room_id: &str) -> String {
    // Room ids are URL path segments, so keep a conservative portable alphabet
    // and cap length to avoid noisy accidental ids.
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
    // Reusing the same token lets a browser reconnect without losing its seat.
    if game_started(room) && color != "spectator" && !is_seated(room, color, token) {
        return Err("game already started; join as spectator".to_string());
    }

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
    // Spectators can observe room state but cannot mutate it.
    match color {
        "white" => room.players.white.as_deref() == Some(token),
        "black" => room.players.black.as_deref() == Some(token),
        _ => false,
    }
}

fn json_error(status: StatusCode, error: String, room: Option<PublicRoom>) -> Response {
    (status, Json(ErrorBody { error, room })).into_response()
}

fn game_started(room: &Room) -> bool {
    room.game
        .as_ref()
        .and_then(|game| game.get("phase"))
        .and_then(Value::as_str)
        == Some("game")
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after Unix epoch")
        .as_millis()
}
