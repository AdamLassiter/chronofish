use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};

const INDEX: &str = include_str!("../../../web/dist/index.html");
const APP: &str = include_str!("../../../web/dist/app.js");
const STYLES: &str = include_str!("../../../web/dist/styles.css");
const FAVICON: &str = include_str!("../../../web/dist/favicon.svg");

pub(crate) async fn index() -> Response {
    asset("text/html; charset=utf-8", INDEX)
}

pub(crate) async fn app() -> Response {
    asset("text/javascript; charset=utf-8", APP)
}

pub(crate) async fn styles() -> Response {
    asset("text/css; charset=utf-8", STYLES)
}

pub(crate) async fn favicon() -> Response {
    asset("image/svg+xml", FAVICON)
}

pub(crate) async fn not_found() -> Response {
    (StatusCode::NOT_FOUND, "Not found").into_response()
}

fn asset(content_type: &'static str, body: &'static str) -> Response {
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-cache"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (
                header::CONTENT_SECURITY_POLICY,
                "default-src 'self'; connect-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'",
            ),
        ],
        body,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_client_preserves_the_horizontal_timeline_game_view() {
        assert!(INDEX.contains("/assets/app.js"));
        assert!(INDEX.contains("/assets/styles.css"));
        assert!(INDEX.contains("id=\"multiverse\""));
        assert!(INDEX.contains("id=\"timeline-grid\""));
        assert!(APP.contains("timeline-row"));
        assert!(APP.contains("present-line"));
        assert!(APP.contains("move-arrows"));
        assert!(APP.contains("snapshot.time + 2"));
        assert!(APP.contains("groupHistory"));
        assert!(APP.contains("AI analysis"));
        assert!(STYLES.contains("--board-width: 282px"));
        assert!(STYLES.contains("repeat(var(--time-columns), var(--board-width))"));
        assert!(STYLES.contains("min-height: 352px"));
        assert!(FAVICON.contains("<svg"));
    }
}
