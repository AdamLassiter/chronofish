async fn static_file(
    State(state): State<AppState>,
    method: Method,
    headers: HeaderMap,
    uri: axum::http::Uri,
) -> Response {
    // Minimal static server: GET/HEAD only, no directory listings, and every
    // resolved frontend path must stay under the generated web/dist tree.
    if method != Method::GET && method != Method::HEAD {
        let mut response = (
            StatusCode::METHOD_NOT_ALLOWED,
            Json(ErrorBody {
                error: "Method not allowed".to_string(),
                room: None,
            }),
        )
            .into_response();
        apply_no_store_headers(&mut response);
        return response;
    }

    if let Some(asset) = embedded_static_asset(uri.path()) {
        return static_bytes_response(method, &headers, asset.bytes, asset.content_type);
    }

    match resolve_request_path(&state.root, uri.path()) {
        Some(path) => match tokio::fs::read(&path).await {
            Ok(bytes) => static_bytes_response(method, &headers, &bytes, content_type(&path)),
            Err(_) => static_no_store_error(StatusCode::NOT_FOUND, "Not found"),
        },
        None => static_no_store_error(StatusCode::NOT_FOUND, "Not found"),
    }
}

struct EmbeddedStaticAsset {
    bytes: &'static [u8],
    content_type: &'static str,
}

const FAVICON_SVG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../logo.svg"));

fn embedded_static_asset(request_path: &str) -> Option<EmbeddedStaticAsset> {
    match request_path {
        "/favicon.svg" => Some(EmbeddedStaticAsset {
            bytes: FAVICON_SVG,
            content_type: "image/svg+xml",
        }),
        _ => None,
    }
}

fn static_bytes_response(
    method: Method,
    headers: &HeaderMap,
    bytes: &[u8],
    content_type: &'static str,
) -> Response {
    let etag = asset_etag(bytes);
    let mut response = if etag_matches(headers.get(header::IF_NONE_MATCH), &etag) {
        StatusCode::NOT_MODIFIED.into_response()
    } else if method == Method::HEAD {
        ().into_response()
    } else {
        bytes.to_vec().into_response()
    };
    apply_static_headers(&mut response, content_type, &etag);
    response
}

const STATIC_CONTENT_SECURITY_POLICY: &str = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; child-src 'self' blob:; connect-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'self'";
const STATIC_CACHE_CONTROL: &str = "no-cache, max-age=0, must-revalidate";

fn apply_static_headers(response: &mut Response, content_type: &'static str, etag: &str) {
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(content_type),
    );
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(STATIC_CONTENT_SECURITY_POLICY),
    );
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(STATIC_CACHE_CONTROL),
    );
    headers.insert(header::ETAG, HeaderValue::from_str(etag).expect("valid etag"));
}

fn asset_etag(bytes: &[u8]) -> String {
    format!("\"cf-{:x}-{:016x}\"", bytes.len(), fnv1a64(bytes))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn etag_matches(header: Option<&HeaderValue>, etag: &str) -> bool {
    header
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(',')
                .any(|candidate| matches!(candidate.trim(), "*") || candidate.trim() == etag)
        })
}

fn static_no_store_error(status: StatusCode, message: &'static str) -> Response {
    let mut response = (status, message).into_response();
    apply_no_store_headers(&mut response);
    response
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("css") => "text/css; charset=utf-8",
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
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
        // The frontend imports a stable filename; Cargo writes the actual file in
        // target/wasm32-unknown-unknown/{debug,release}.
        return wasm_path(root);
    }

    if requested == Path::new("ai/parameters.json") {
        let path = root.join("engine/models/cpu-v1/parameters.json");
        return path.is_file().then_some(path);
    }

    if requested == Path::new("ai/effort.json") {
        let path = root.join("engine/models/cpu-v1/effort.json");
        return path.is_file().then_some(path);
    }

    if requested == Path::new("ai/gpu-effort.json") {
        let path = root.join("engine/models/gpu-v1/effort.json");
        return path.is_file().then_some(path);
    }

    if requested == Path::new("ai/value-model.cfnn") {
        let path = root.join("engine/models/gpu-v1/value-model.cfnn");
        return path.is_file().then_some(path);
    }

    if requested == Path::new("ai/training.json") {
        let path = root.join("engine/models/cpu-v1/training.json");
        return path.is_file().then_some(path);
    }

    let web_root = root.join("web/dist");
    let web_path = web_root.join(&requested);
    if is_safe_existing_path(&web_root, &web_path) {
        return Some(web_path);
    }

    None
}

fn wasm_path(root: &Path) -> Option<PathBuf> {
    [
        root.join("target/wasm32-unknown-unknown/release/chronofish_engine.wasm"),
        root.join("target/wasm32-unknown-unknown/debug/chronofish_engine.wasm"),
    ]
    .into_iter()
    .find(|path| path.is_file())
}

fn is_safe_existing_path(root: &Path, path: &Path) -> bool {
    // Canonicalizing both sides prevents "../" and symlink escapes from web/.
    let Ok(root) = root.canonicalize() else {
        return false;
    };
    let Ok(path) = path.canonicalize() else {
        return false;
    };

    path.starts_with(root) && path.is_file()
}

fn percent_decode_path(path: &str) -> Option<String> {
    // Decode browser path escapes and reject malformed percent sequences.
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
