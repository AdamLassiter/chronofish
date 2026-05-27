// Runtime AI search. Training-only mutation/scoring/promotion code lives in
// training.rs so wasm gets a deterministic search surface without file or git
// automation.
const CHECKMATE_SCORE: i32 = 1_000_000;
const MAX_TURN_PLANS: usize = 32;
const MAX_MOVES_PER_NODE: usize = 24;
const MAX_QUIESCENCE_DEPTH: i32 = 2;

type SearchInstant = wasm_timer::Instant;

#[derive(Clone)]
struct MoveStep {
    from: Position,
    to: Position,
}

#[derive(Clone)]
struct TurnPlan {
    moves: Vec<MoveStep>,
    game: Game,
    score_hint: i32,
}

#[derive(Clone)]
struct AiSearchResult {
    moves: Vec<MoveStep>,
    score: i32,
    depth: i32,
    nodes: usize,
    status: &'static str,
}

#[derive(Clone, Copy)]
struct EvalWeights {
    king: i32,
    common_king: i32,
    queen: i32,
    royal_queen: i32,
    princess: i32,
    rook: i32,
    bishop: i32,
    unicorn: i32,
    dragon: i32,
    knight: i32,
    pawn: i32,
    brawn: i32,
    check_penalty: i32,
    active_timeline: i32,
    inactive_timeline: i32,
    present_progress: i32,
    mobility: i32,
    branch_penalty: i32,
    advancement: i32,
    centrality: i32,
    defended_piece: i32,
    attacked_piece: i32,
    hanging_piece: i32,
    royal_threat: i32,
    temporal_threat: i32,
    pincer_threat: i32,
    timeline_pincer: i32,
    historical_pincer: i32,
    frontier_tempo: i32,
    present_anchor: i32,
    development: i32,
    branch_attack: i32,
    check_bonus: i32,
    royal_capture_threat: i32,
    royal_escape_pressure: i32,
    forcing_move_pressure: i32,
    own_royal_exposure: i32,
    fork_pressure: i32,
    board_control: i32,
    piece_activity: i32,
    pawn_structure: i32,
    timeline_economy: i32,
    present_tempo: i32,
    royal_shelter: i32,
    space_advantage: i32,
}

#[derive(Default)]
struct AttackSummary {
    count: i32,
    temporal_count: i32,
    timeline_count: i32,
    time_count: i32,
}

struct SearchContext {
    // The node budget is shared across iterative-deepening branches.
    weights: EvalWeights,
    root_color: Color,
    max_nodes: usize,
    nodes: usize,
    deadline: Option<SearchInstant>,
    table: std::collections::HashMap<String, SearchEntry>,
}

#[derive(Clone, Copy)]
struct SearchEntry {
    depth: i32,
    score: i32,
}
