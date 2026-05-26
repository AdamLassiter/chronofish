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
    let requested = if requested == "/" || requested.is_empty() {
        PathBuf::from("index.html")
    } else {
        PathBuf::from(requested.trim_start_matches('/'))
    };

    if requested == Path::new("chronofish_engine.wasm") {
        return wasm_path(root);
    }

    let web_root = root.join("web");
    let web_path = web_root.join(&requested);
    if is_safe_existing_path(&web_root, &web_path) {
        return Some(web_path);
    }

    None
}

fn wasm_path(root: &Path) -> Option<PathBuf> {
    [
        root.join("target/wasm32-unknown-unknown/debug/chronofish_engine.wasm"),
        root.join("target/wasm32-unknown-unknown/release/chronofish_engine.wasm"),
    ]
    .into_iter()
    .find(|path| path.is_file())
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
