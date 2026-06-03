// Runtime AI search. Training-only mutation/scoring/promotion code lives in
// training.rs so wasm gets a deterministic search surface without file or git
// automation.
#[cfg(test)]
const CHECKMATE_SCORE: i32 = 1_000_000;
#[cfg(test)]
const MAX_TURN_PLANS: usize = 32;
#[cfg(test)]
const MAX_MOVES_PER_NODE: usize = 24;
#[cfg(test)]
const REQUIRED_MOVES_PER_BOARD: usize = 4;
#[cfg(test)]
const MAX_QUIESCENCE_DEPTH: i32 = 2;
#[cfg(test)]
const ASPIRATION_WINDOW: i32 = 400;
#[cfg(test)]
const LATE_MOVE_REDUCTION_AFTER: usize = 8;
#[cfg(test)]
const HISTORY_BONUS: i32 = 32;

#[cfg(test)]
type SearchInstant = wasm_timer::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg(test)]
struct MoveStep {
    from: Position,
    to: Position,
}

#[derive(Clone)]
#[cfg(test)]
struct TurnPlan {
    moves: Vec<MoveStep>,
    game: Game,
    score_hint: i32,
}

#[derive(Clone)]
#[cfg(test)]
struct AiSearchResult {
    moves: Vec<MoveStep>,
    score: i32,
    depth: i32,
    nodes: usize,
    status: &'static str,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct AiEffort {
    label: String,
    display_names: Vec<String>,
    depth: i32,
    nodes: usize,
    time_ms: u64,
    training_depth: i32,
    training_nodes: usize,
    training_plies: usize,
}

#[derive(Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
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
    #[serde(default = "default_royal_capture_setup")]
    royal_capture_setup: i32,
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

#[allow(dead_code)]
fn default_royal_capture_setup() -> i32 {
    900
}

#[derive(Default)]
#[allow(dead_code)]
struct AttackSummary {
    count: i32,
    temporal_count: i32,
    timeline_count: i32,
    time_count: i32,
}

#[cfg(test)]
struct SearchContext {
    // The node budget is shared across iterative-deepening branches.
    weights: EvalWeights,
    evaluator: ValueEvaluator,
    root_color: Color,
    max_nodes: usize,
    nodes: usize,
    deadline: Option<SearchInstant>,
    options: SearchOptions,
    table: std::collections::HashMap<u64, SearchEntry>,
    turn_plan_cache: std::collections::HashMap<u64, Vec<TurnPlan>>,
    killers: Vec<[Option<MoveStep>; 2]>,
    history: std::collections::HashMap<u64, i32>,
    stats: SearchStats,
}

#[derive(Clone, Copy)]
#[cfg(test)]
struct SearchEntry {
    depth: i32,
    score: i32,
    best_move: Option<MoveStep>,
}

#[derive(Clone, Copy)]
#[cfg(test)]
struct SearchOptions {
    tt_best_move: bool,
    killer_moves: bool,
    history_heuristic: bool,
    direct_quiescence: bool,
    late_move_reduction: bool,
    aspiration_windows: bool,
    capture_sanity: bool,
    turn_plan_cache: bool,
}

#[derive(Clone, Copy, Default)]
#[cfg(test)]
struct SearchStats {
    expensive_order_probes: usize,
    turn_plan_cache_hits: usize,
    tt_hits: usize,
    beta_cutoffs: usize,
    reduced_searches: usize,
    aspiration_researches: usize,
}

#[derive(Clone)]
#[cfg(test)]
struct SearchPerfSample {
    label: &'static str,
    elapsed_micros: u128,
    nodes: usize,
    stats: SearchStats,
}
