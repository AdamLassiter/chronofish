use super::*;

// Runtime AI search. Training-only mutation/scoring/promotion code lives in
// training.rs so wasm gets a deterministic search surface without file or git
// automation.
pub(crate) const CHECKMATE_SCORE: i32 = 1_000_000;
pub(crate) const MAX_TURN_PLANS: usize = 32;
pub(crate) const MAX_ROOT_TURN_PLANS: usize = 16;
pub(crate) const MAX_CHILD_TURN_PLANS: usize = 8;
pub(crate) const FAST_ROOT_TURN_PLANS: usize = 8;
pub(crate) const FAST_CHILD_TURN_PLANS: usize = 3;
pub(crate) const FAST_SEARCH_NODE_THRESHOLD: usize = 5_000;
pub(crate) const MAX_MOVES_PER_NODE: usize = 24;
#[allow(dead_code)]
pub(crate) const REQUIRED_MOVES_PER_BOARD: usize = 4;
pub(crate) const MAX_QUIESCENCE_DEPTH: i32 = 1;
pub(crate) const MAX_QUIESCENCE_MOVES: usize = 3;
pub(crate) const ASPIRATION_WINDOW: i32 = 400;
pub(crate) const LATE_MOVE_REDUCTION_AFTER: usize = 8;
pub(crate) const HISTORY_BONUS: i32 = 32;

pub(crate) type SearchInstant = wasm_timer::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MoveStep {
    pub(crate) from: Position,
    pub(crate) to: Position,
}

#[derive(Clone)]
pub(crate) struct TurnPlan {
    pub(crate) moves: Vec<MoveStep>,
    pub(crate) score_hint: i32,
}

#[derive(Clone)]
pub(crate) struct AiSearchResult {
    pub(crate) moves: Vec<MoveStep>,
    pub(crate) score: i32,
    pub(crate) depth: i32,
    pub(crate) nodes: usize,
    pub(crate) status: &'static str,
    pub(crate) principal_variation: Vec<Vec<MoveStep>>,
    pub(crate) terminal_royal_capture: bool,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AiEffort {
    pub(crate) label: String,
    pub(crate) display_names: Vec<String>,
    pub(crate) depth: i32,
    #[serde(default = "default_min_ai_search_depth")]
    pub(crate) min_depth: i32,
    pub(crate) nodes: usize,
    pub(crate) time_ms: u64,
    #[serde(default = "default_cpu_search_strategy")]
    pub(crate) search_strategy: crate::cpu::search::CpuSearchStrategy,
}

pub(crate) const fn default_cpu_search_strategy() -> crate::cpu::search::CpuSearchStrategy {
    crate::cpu::search::CpuSearchStrategy::Beam
}

pub(crate) const fn default_min_ai_search_depth() -> i32 {
    2
}

#[derive(Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EvalWeights {
    #[serde(default = "default_royal_weight")]
    pub(crate) king: i32,
    #[serde(default = "default_centiqueen")]
    pub(crate) common_king: i32,
    #[serde(default = "default_centiqueen")]
    pub(crate) queen: i32,
    #[serde(default = "default_royal_weight")]
    pub(crate) royal_queen: i32,
    #[serde(default = "default_centibishop")]
    pub(crate) princess: i32,
    #[serde(default = "default_centirook")]
    pub(crate) rook: i32,
    #[serde(default = "default_centibishop")]
    pub(crate) bishop: i32,
    #[serde(default = "default_centirook")]
    pub(crate) unicorn: i32,
    #[serde(default = "default_centibishop")]
    pub(crate) dragon: i32,
    #[serde(default = "default_centiknight")]
    pub(crate) knight: i32,
    #[serde(default = "default_centipawn")]
    pub(crate) pawn: i32,
    #[serde(default = "default_centipawn")]
    pub(crate) brawn: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) check_penalty: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) active_timeline: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) inactive_timeline: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) present_progress: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) mobility: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) branch_penalty: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) advancement: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) centrality: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) defended_piece: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) attacked_piece: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) hanging_piece: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) royal_threat: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) temporal_threat: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) pincer_threat: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) timeline_pincer: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) historical_pincer: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) frontier_tempo: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) present_anchor: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) development: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) branch_attack: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) check_bonus: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) royal_capture_threat: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) royal_capture_setup: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) royal_escape_pressure: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) forcing_move_pressure: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) own_royal_exposure: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) fork_pressure: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) board_control: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) piece_activity: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) pawn_structure: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) timeline_economy: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) present_tempo: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) royal_shelter: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) space_advantage: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) mandatory_move_burden: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) turn_completion_safety: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) present_zugzwang: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) weakest_royal_safety: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) royal_liability_count: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) multi_royal_attack: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) defensive_bandwidth: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) threat_overload: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) active_branch_capacity: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) latent_timeline_reactivation: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) inactive_material_quality: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) branch_payload: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) branch_waste: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) timeline_compaction: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) frontier_material: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) historical_access: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) temporal_lane_control: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) temporal_pin: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) temporal_skewer: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) causal_battery: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) arrival_square_safety: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) source_board_abandonment: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) piece_temporal_flexibility: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) dimension_coverage_balance: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) promotion_timeline_choice: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) promotion_with_check: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) past_royal_vulnerability: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) safe_haven_boards: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) escape_branch_potential: i32,
    #[serde(default = "default_zero_weight", rename = "mateNetDepth12")]
    pub(crate) mate_net_depth_1_2: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) anti_mate_resources: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) checking_move_quality: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) search_volatility: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) timeline_repetition_risk: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) phase_by_multiverse_size: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) royal_distance_in_4d: i32,
    #[serde(default = "default_zero_weight")]
    pub(crate) board_importance_weight: i32,
}

#[allow(dead_code)]
pub(crate) fn default_royal_weight() -> i32 {
    i16::MAX as i32
}

#[allow(dead_code)]
pub(crate) fn default_centiqueen() -> i32 {
    900
}

#[allow(dead_code)]
pub(crate) fn default_centirook() -> i32 {
    500
}

#[allow(dead_code)]
pub(crate) fn default_centibishop() -> i32 {
    300
}

#[allow(dead_code)]
pub(crate) fn default_centiknight() -> i32 {
    250
}

#[allow(dead_code)]
pub(crate) fn default_centipawn() -> i32 {
    100
}

#[allow(dead_code)]
pub(crate) fn default_zero_weight() -> i32 {
    0
}

#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct AttackSummary {
    pub(crate) count: i32,
    pub(crate) temporal_count: i32,
    pub(crate) timeline_count: i32,
    pub(crate) time_count: i32,
}

pub(crate) struct LatestPositionView {
    pub(crate) pieces: Vec<(Position, Piece)>,
    pub(crate) board_positions: Vec<Position>,
    pub(crate) white_royals: Vec<(Position, Piece)>,
    pub(crate) black_royals: Vec<(Position, Piece)>,
}

impl LatestPositionView {
    pub(crate) fn royals(&self, color: Color) -> &[(Position, Piece)] {
        match color {
            Color::White => &self.white_royals,
            Color::Black => &self.black_royals,
        }
    }
}

pub(crate) struct SearchContext {
    // The node budget is shared across iterative-deepening branches.
    pub(crate) weights: EvalWeights,
    pub(crate) evaluator: ValueEvaluator,
    pub(crate) root_color: Color,
    pub(crate) max_nodes: usize,
    pub(crate) nodes: usize,
    pub(crate) deadline: Option<SearchInstant>,
    pub(crate) options: SearchOptions,
    pub(crate) root_plan_cap: Option<usize>,
    pub(crate) child_plan_cap: Option<usize>,
    pub(crate) evaluation_limits: Option<EvaluationLimits>,
    pub(crate) table: TranspositionTable,
    pub(crate) evaluation_cache: EvaluationCache,
    pub(crate) turn_plan_cache: std::collections::HashMap<u64, Vec<TurnPlan>>,
    pub(crate) attack_cache: std::collections::HashMap<u64, bool>,
    pub(crate) killers: Vec<[Option<MoveStep>; 2]>,
    pub(crate) history: std::collections::HashMap<u64, i32>,
    pub(crate) stats: SearchStats,
}

#[derive(Clone, Copy)]
pub(crate) struct EvaluationLimits {
    pub(crate) turn_moves: usize,
    pub(crate) completion_results: usize,
    pub(crate) zugzwang_moves_per_board: usize,
    pub(crate) setup_results: usize,
    pub(crate) setup_probes: usize,
    pub(crate) attack_checks: usize,
    pub(crate) deadline: Option<SearchInstant>,
}

impl EvaluationLimits {
    pub(crate) const FULL: Self = Self {
        turn_moves: 48,
        completion_results: 6,
        zugzwang_moves_per_board: usize::MAX,
        setup_results: 48,
        setup_probes: usize::MAX,
        attack_checks: usize::MAX,
        deadline: None,
    };

    pub(crate) fn for_nodes(max_nodes: usize) -> Self {
        if max_nodes <= FAST_SEARCH_NODE_THRESHOLD {
            Self {
                turn_moves: 8,
                completion_results: 2,
                zugzwang_moves_per_board: 4,
                setup_results: 4,
                setup_probes: 96,
                attack_checks: 256,
                deadline: None,
            }
        } else {
            Self {
                turn_moves: 24,
                completion_results: 4,
                zugzwang_moves_per_board: 8,
                setup_results: 8,
                setup_probes: 256,
                attack_checks: 16_384,
                deadline: None,
            }
        }
    }

    pub(crate) fn training_late_game(max_nodes: usize) -> Self {
        let mut limits = Self::for_nodes(max_nodes);
        limits.turn_moves = limits.turn_moves.min(8);
        limits.completion_results = limits.completion_results.min(2);
        limits.zugzwang_moves_per_board = limits.zugzwang_moves_per_board.min(4);
        limits.setup_results = limits.setup_results.min(4);
        limits.setup_probes = limits.setup_probes.min(96);
        limits.attack_checks = limits.attack_checks.min(4_096);
        limits
    }

    pub(crate) fn training_fast_late_game(max_nodes: usize) -> Self {
        let mut limits = Self::training_late_game(max_nodes);
        limits.turn_moves = limits.turn_moves.min(4);
        limits.completion_results = limits.completion_results.min(1);
        limits.zugzwang_moves_per_board = limits.zugzwang_moves_per_board.min(2);
        limits.setup_results = limits.setup_results.min(2);
        limits.setup_probes = limits.setup_probes.min(32);
        limits.attack_checks = limits.attack_checks.min(1_024);
        limits
    }

    pub(crate) fn with_deadline(mut self, deadline: Option<SearchInstant>) -> Self {
        self.deadline = deadline;
        self
    }
}

#[derive(Default)]
pub(crate) struct EvaluationStats {
    pub(crate) calls: usize,
    pub(crate) turn_moves: usize,
    pub(crate) setup_probes: usize,
    pub(crate) attack_checks: usize,
    pub(crate) attack_caps: usize,
    pub(crate) clones: usize,
}

pub(crate) struct EvaluationCache {
    pub(crate) slots: Vec<Option<EvaluationSlot>>,
    pub(crate) mask: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct EvaluationSlot {
    pub(crate) key: u64,
    pub(crate) score: i32,
}

pub(crate) struct TranspositionTable {
    pub(crate) slots: Vec<Option<TranspositionSlot>>,
    pub(crate) mask: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct TranspositionSlot {
    pub(crate) key: u64,
    pub(crate) entry: SearchEntry,
}

#[derive(Clone, Copy)]
pub(crate) struct SearchEntry {
    pub(crate) depth: i32,
    pub(crate) score: i32,
    pub(crate) bound: SearchBound,
    pub(crate) best_move: Option<MoveStep>,
}

#[derive(Clone, Copy)]
pub(crate) enum SearchBound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
pub(crate) struct SearchOptions {
    pub(crate) tt_best_move: bool,
    pub(crate) killer_moves: bool,
    pub(crate) history_heuristic: bool,
    pub(crate) direct_quiescence: bool,
    pub(crate) late_move_reduction: bool,
    pub(crate) aspiration_windows: bool,
    pub(crate) capture_sanity: bool,
    pub(crate) turn_plan_cache: bool,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct SearchStats {
    pub(crate) generated_moves: usize,
    pub(crate) generated_plans: usize,
    pub(crate) candidate_destinations: usize,
    pub(crate) legal_move_attempts: usize,
    pub(crate) attack_queries: usize,
    pub(crate) attack_cache_hits: usize,
    pub(crate) search_clones: usize,
    pub(crate) expensive_order_probes: usize,
    pub(crate) turn_plan_cache_hits: usize,
    pub(crate) tt_hits: usize,
    pub(crate) beta_cutoffs: usize,
    pub(crate) reduced_searches: usize,
    pub(crate) aspiration_researches: usize,
    pub(crate) evaluation_calls: usize,
    pub(crate) evaluation_cache_hits: usize,
    pub(crate) evaluated_turn_moves: usize,
    pub(crate) evaluation_setup_probes: usize,
    pub(crate) evaluation_attack_checks: usize,
    pub(crate) evaluation_attack_caps: usize,
    pub(crate) evaluation_clones: usize,
}

#[derive(Clone)]
pub(crate) struct SearchPerfSample {
    pub(crate) label: &'static str,
    pub(crate) elapsed_micros: u128,
    pub(crate) nodes: usize,
    pub(crate) stats: SearchStats,
}
