// Runtime AI search. Training-only mutation/scoring/promotion code lives in
// training.rs so wasm gets a deterministic search surface without file or git
// automation.
const CHECKMATE_SCORE: i32 = 1_000_000;
const MAX_TURN_PLANS: usize = 32;
const MAX_MOVES_PER_NODE: usize = 24;
const REQUIRED_MOVES_PER_BOARD: usize = 4;
const MAX_QUIESCENCE_DEPTH: i32 = 2;
const ASPIRATION_WINDOW: i32 = 400;
const LATE_MOVE_REDUCTION_AFTER: usize = 8;
const HISTORY_BONUS: i32 = 32;

type SearchInstant = wasm_timer::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    #[serde(default = "default_royal_weight")]
    king: i32,
    #[serde(default = "default_centiqueen")]
    common_king: i32,
    #[serde(default = "default_centiqueen")]
    queen: i32,
    #[serde(default = "default_royal_weight")]
    royal_queen: i32,
    #[serde(default = "default_centibishop")]
    princess: i32,
    #[serde(default = "default_centirook")]
    rook: i32,
    #[serde(default = "default_centibishop")]
    bishop: i32,
    #[serde(default = "default_centirook")]
    unicorn: i32,
    #[serde(default = "default_centibishop")]
    dragon: i32,
    #[serde(default = "default_centiknight")]
    knight: i32,
    #[serde(default = "default_centipawn")]
    pawn: i32,
    #[serde(default = "default_centipawn")]
    brawn: i32,
    #[serde(default = "default_zero_weight")]
    check_penalty: i32,
    #[serde(default = "default_zero_weight")]
    active_timeline: i32,
    #[serde(default = "default_zero_weight")]
    inactive_timeline: i32,
    #[serde(default = "default_zero_weight")]
    present_progress: i32,
    #[serde(default = "default_zero_weight")]
    mobility: i32,
    #[serde(default = "default_zero_weight")]
    branch_penalty: i32,
    #[serde(default = "default_zero_weight")]
    advancement: i32,
    #[serde(default = "default_zero_weight")]
    centrality: i32,
    #[serde(default = "default_zero_weight")]
    defended_piece: i32,
    #[serde(default = "default_zero_weight")]
    attacked_piece: i32,
    #[serde(default = "default_zero_weight")]
    hanging_piece: i32,
    #[serde(default = "default_zero_weight")]
    royal_threat: i32,
    #[serde(default = "default_zero_weight")]
    temporal_threat: i32,
    #[serde(default = "default_zero_weight")]
    pincer_threat: i32,
    #[serde(default = "default_zero_weight")]
    timeline_pincer: i32,
    #[serde(default = "default_zero_weight")]
    historical_pincer: i32,
    #[serde(default = "default_zero_weight")]
    frontier_tempo: i32,
    #[serde(default = "default_zero_weight")]
    present_anchor: i32,
    #[serde(default = "default_zero_weight")]
    development: i32,
    #[serde(default = "default_zero_weight")]
    branch_attack: i32,
    #[serde(default = "default_zero_weight")]
    check_bonus: i32,
    #[serde(default = "default_zero_weight")]
    royal_capture_threat: i32,
    #[serde(default = "default_zero_weight")]
    royal_capture_setup: i32,
    #[serde(default = "default_zero_weight")]
    royal_escape_pressure: i32,
    #[serde(default = "default_zero_weight")]
    forcing_move_pressure: i32,
    #[serde(default = "default_zero_weight")]
    own_royal_exposure: i32,
    #[serde(default = "default_zero_weight")]
    fork_pressure: i32,
    #[serde(default = "default_zero_weight")]
    board_control: i32,
    #[serde(default = "default_zero_weight")]
    piece_activity: i32,
    #[serde(default = "default_zero_weight")]
    pawn_structure: i32,
    #[serde(default = "default_zero_weight")]
    timeline_economy: i32,
    #[serde(default = "default_zero_weight")]
    present_tempo: i32,
    #[serde(default = "default_zero_weight")]
    royal_shelter: i32,
    #[serde(default = "default_zero_weight")]
    space_advantage: i32,
    #[serde(default = "default_zero_weight")]
    mandatory_move_burden: i32,
    #[serde(default = "default_zero_weight")]
    turn_completion_safety: i32,
    #[serde(default = "default_zero_weight")]
    present_zugzwang: i32,
    #[serde(default = "default_zero_weight")]
    weakest_royal_safety: i32,
    #[serde(default = "default_zero_weight")]
    royal_liability_count: i32,
    #[serde(default = "default_zero_weight")]
    multi_royal_attack: i32,
    #[serde(default = "default_zero_weight")]
    defensive_bandwidth: i32,
    #[serde(default = "default_zero_weight")]
    threat_overload: i32,
    #[serde(default = "default_zero_weight")]
    active_branch_capacity: i32,
    #[serde(default = "default_zero_weight")]
    latent_timeline_reactivation: i32,
    #[serde(default = "default_zero_weight")]
    inactive_material_quality: i32,
    #[serde(default = "default_zero_weight")]
    branch_payload: i32,
    #[serde(default = "default_zero_weight")]
    branch_waste: i32,
    #[serde(default = "default_zero_weight")]
    timeline_compaction: i32,
    #[serde(default = "default_zero_weight")]
    frontier_material: i32,
    #[serde(default = "default_zero_weight")]
    historical_access: i32,
    #[serde(default = "default_zero_weight")]
    temporal_lane_control: i32,
    #[serde(default = "default_zero_weight")]
    temporal_pin: i32,
    #[serde(default = "default_zero_weight")]
    temporal_skewer: i32,
    #[serde(default = "default_zero_weight")]
    causal_battery: i32,
    #[serde(default = "default_zero_weight")]
    arrival_square_safety: i32,
    #[serde(default = "default_zero_weight")]
    source_board_abandonment: i32,
    #[serde(default = "default_zero_weight")]
    piece_temporal_flexibility: i32,
    #[serde(default = "default_zero_weight")]
    dimension_coverage_balance: i32,
    #[serde(default = "default_zero_weight")]
    promotion_timeline_choice: i32,
    #[serde(default = "default_zero_weight")]
    promotion_with_check: i32,
    #[serde(default = "default_zero_weight")]
    past_royal_vulnerability: i32,
    #[serde(default = "default_zero_weight")]
    safe_haven_boards: i32,
    #[serde(default = "default_zero_weight")]
    escape_branch_potential: i32,
    #[serde(default = "default_zero_weight", rename = "mateNetDepth12")]
    mate_net_depth_1_2: i32,
    #[serde(default = "default_zero_weight")]
    anti_mate_resources: i32,
    #[serde(default = "default_zero_weight")]
    checking_move_quality: i32,
    #[serde(default = "default_zero_weight")]
    search_volatility: i32,
    #[serde(default = "default_zero_weight")]
    timeline_repetition_risk: i32,
    #[serde(default = "default_zero_weight")]
    phase_by_multiverse_size: i32,
    #[serde(default = "default_zero_weight")]
    royal_distance_in_4d: i32,
    #[serde(default = "default_zero_weight")]
    board_importance_weight: i32,
}

#[allow(dead_code)]
fn default_royal_weight() -> i32 {
    i16::MAX as i32
}

#[allow(dead_code)]
fn default_centiqueen() -> i32 {
    900
}

#[allow(dead_code)]
fn default_centirook() -> i32 {
    500
}

#[allow(dead_code)]
fn default_centibishop() -> i32 {
    300
}

#[allow(dead_code)]
fn default_centiknight() -> i32 {
    250
}

#[allow(dead_code)]
fn default_centipawn() -> i32 {
    100
}

#[allow(dead_code)]
fn default_zero_weight() -> i32 {
    0
}

#[derive(Default)]
#[allow(dead_code)]
struct AttackSummary {
    count: i32,
    temporal_count: i32,
    timeline_count: i32,
    time_count: i32,
}

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
struct SearchEntry {
    depth: i32,
    score: i32,
    best_move: Option<MoveStep>,
}

#[derive(Clone, Copy)]
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
struct SearchStats {
    expensive_order_probes: usize,
    turn_plan_cache_hits: usize,
    tt_hits: usize,
    beta_cutoffs: usize,
    reduced_searches: usize,
    aspiration_researches: usize,
}

#[derive(Clone)]
struct SearchPerfSample {
    label: &'static str,
    elapsed_micros: u128,
    nodes: usize,
    stats: SearchStats,
}
