use axum::extract::Path;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};

macro_rules! script {
    ($name:literal) => {
        ($name, include_str!(concat!("../../../web/dist/", $name)))
    };
}

const INDEX: &str = include_str!("../../../web/dist/index.html");

/// The client's ES modules, as emitted by `npm run build`. `app.js` is the entry point
/// and imports the rest. A test checks that this lists every module in `web/dist`.
const SCRIPTS: &[(&str, &str)] = &[
    script!("actions.js"),
    script!("api.js"),
    script!("app.js"),
    script!("board.js"),
    script!("bug-report.js"),
    script!("dom.js"),
    script!("evaluation.js"),
    script!("help.js"),
    script!("history.js"),
    script!("lobby.js"),
    script!("moves.js"),
    script!("scroll.js"),
    script!("state.js"),
    script!("types.js"),
    script!("visuals.js"),
];
const STYLES: &str = include_str!("../../../web/dist/styles.css");
const FAVICON: &str = include_str!("../../../web/dist/favicon.svg");
/// The rules document, shown in the client's Help dialog.
const RULES: &str = include_str!("../../../RULES.md");

pub(crate) async fn index() -> Response {
    asset("text/html; charset=utf-8", INDEX)
}

pub(crate) async fn script(Path(file): Path<String>) -> Response {
    SCRIPTS.iter().find(|(name, _)| *name == file).map_or_else(
        || (StatusCode::NOT_FOUND, "Not found").into_response(),
        |(_, body)| asset("text/javascript; charset=utf-8", body),
    )
}

pub(crate) async fn styles() -> Response {
    asset("text/css; charset=utf-8", STYLES)
}

pub(crate) async fn favicon() -> Response {
    asset("image/svg+xml", FAVICON)
}

pub(crate) async fn rules() -> Response {
    asset("text/markdown; charset=utf-8", RULES)
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

    /// Every client module, concatenated, for checking what the client contains.
    fn app() -> String {
        SCRIPTS.iter().map(|(_, body)| *body).collect()
    }

    #[test]
    fn every_built_client_module_is_served() {
        let dist = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/dist");
        let mut built = std::fs::read_dir(dist)
            .expect("web/dist exists; run npm run build")
            .map(|entry| {
                entry
                    .expect("dist entry")
                    .file_name()
                    .into_string()
                    .expect("utf-8")
            })
            .filter(|name| {
                std::path::Path::new(name)
                    .extension()
                    .is_some_and(|e| e == "js")
            })
            .collect::<Vec<_>>();
        built.sort();
        let served = SCRIPTS
            .iter()
            .map(|(name, _)| (*name).to_owned())
            .collect::<Vec<_>>();
        assert_eq!(built, served, "update SCRIPTS in web.rs to match web/dist");
    }

    #[test]
    fn embedded_client_preserves_the_horizontal_timeline_game_view() {
        let app = app();
        assert!(INDEX.contains("/assets/app.js"));
        assert!(INDEX.contains("/assets/styles.css"));
        assert!(INDEX.contains("id=\"multiverse\""));
        assert!(INDEX.contains("id=\"timeline-grid\""));
        assert!(INDEX.contains("id=\"undo-staged-move\""));
        assert!(INDEX.contains("id=\"raise-bug\""));
        assert!(INDEX.contains("id=\"bug-report-dialog\""));
        assert!(INDEX.contains("Expected behaviour"));
        assert!(INDEX.contains("Actual behaviour"));
        assert!(INDEX.contains("id=\"history-drawer\""));
        assert!(INDEX.contains("id=\"help-dialog\""));
        assert!(INDEX.contains("Board colours"));
        assert!(app.contains("timeline-row"));
        assert!(app.contains("present-line"));
        assert!(app.contains("Required"));
        assert!(app.contains("Optional"));
        assert!(app.contains("Inactive"));
        assert!(app.contains("submittedPresentTime"));
        assert!(app.contains("undoStagedMove"));
        assert!(app.contains("Turn submitted."));
        assert!(app.contains("submitBugReport"));
        assert!(app.contains("expectedBehavior"));
        assert!(app.contains("move-arrows"));
        assert!(app.contains("snapshot.time + 1"));
        assert!(app.contains("renderRules"));
        assert!(app.contains("dataset.view"));
        assert!(app.contains("groupHistory"));
        assert!(app.contains("AI analysis"));
        assert!(STYLES.contains("--board-width: 282px"));
        assert!(STYLES.contains("repeat(var(--time-columns), var(--board-width))"));
        assert!(STYLES.contains("100dvh"));
        assert!(RULES.starts_with("# Rules"));
        assert!(FAVICON.contains("<svg"));
    }
}
