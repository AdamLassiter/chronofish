use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};

use crate::{cpu::EvalWeights, wasm_api::parse_game_snapshot, Game};

pub const DEFAULT_CPU_SEARCH_DEPTH: i32 = 2;
pub const DEFAULT_CPU_SEARCH_NODES: i32 = 1_024;
pub const DEFAULT_CPU_SEARCH_TIME_MS: i32 = 10_000;
pub const CPU_MOVE_AGREEMENT_BONUS: i32 = 25;
pub const CPU_TRAINING_WIN_SCORE: i32 = 100_000;
pub const MAX_CPU_TRAINING_CANDIDATES: usize = 256;
pub const MAX_CPU_TRAINING_ELITES: usize = 4;

pub type CpuParameters = Vec<(String, i32)>;

#[derive(Clone, Debug, PartialEq)]
pub struct CpuScoredCandidate {
    pub parameters: CpuParameters,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CpuFitnessEntry {
    pub key: String,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CpuCandidateScoringPlan {
    pub unique_candidates: Vec<CpuParameters>,
    pub cached_scores: Vec<CpuScoredCandidate>,
    pub uncached_candidates: Vec<CpuParameters>,
    pub cache_hits: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CpuTrainingMove {
    pub from_timeline_id: i32,
    pub from_time: i32,
    pub from_x: i32,
    pub from_y: i32,
    pub to_timeline_id: i32,
    pub to_time: i32,
    pub to_x: i32,
    pub to_y: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CpuReferenceScoreDelta {
    pub score: i32,
    pub near_draw: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuScreeningTrainingConfig {
    pub cpu_depth: i32,
    pub depth: i32,
    pub cpu_nodes: i32,
    pub nodes: i32,
    pub cpu_training_time_ms: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuTrainingPositionSearchConfig {
    pub depth: i32,
    pub nodes: i32,
}

#[derive(Clone, Debug)]
pub struct CpuSearchRequest {
    pub snapshot_json: Option<String>,
    pub parameters_json: Option<String>,
    pub depth: i32,
    pub min_depth: Option<i32>,
    pub nodes: i32,
    pub time_ms: i32,
}

impl Default for CpuSearchRequest {
    fn default() -> Self {
        Self {
            snapshot_json: None,
            parameters_json: None,
            depth: DEFAULT_CPU_SEARCH_DEPTH,
            min_depth: Some(crate::Game::DEFAULT_MIN_AI_SEARCH_DEPTH),
            nodes: DEFAULT_CPU_SEARCH_NODES,
            time_ms: DEFAULT_CPU_SEARCH_TIME_MS,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuSearchResponse {
    pub result_json: String,
    pub cpu_search: &'static str,
}

pub fn search(request: CpuSearchRequest) -> Result<CpuSearchResponse, String> {
    if let Some(parameters_json) = request.parameters_json.as_deref() {
        EvalWeights::set_active_from_json(parameters_json)
            .map_err(|error| format!("Invalid CPU search parameters: {error}"))?;
    }
    let game = match request.snapshot_json.as_deref() {
        Some(snapshot) => parse_game_snapshot(snapshot)?,
        None => Game::new(),
    };
    let depth = request.depth.max(1);
    let nodes = request.nodes.max(1);
    let time_ms = request.time_ms.max(1);
    let result_json = match request.min_depth {
        Some(min_depth) => {
            game.ai_turn_timed_min_depth_json(depth, min_depth.max(1), nodes, time_ms)
        }
        None => game.ai_turn_timed_json(depth, nodes, time_ms),
    };
    Ok(CpuSearchResponse {
        result_json,
        cpu_search: "heuristic",
    })
}

pub fn cpu_parameters_key(parameters: &[(String, i32)]) -> String {
    let mut entries: Vec<_> = parameters.iter().collect();
    entries.sort_by(|(left, _), (right, _)| left.cmp(right));
    entries
        .into_iter()
        .map(|(key, value)| format!("{key}:{value}"))
        .collect::<Vec<_>>()
        .join("|")
}

pub fn unique_cpu_parameters(values: &[CpuParameters]) -> Vec<CpuParameters> {
    let mut seen = BTreeSet::new();
    let mut unique = Vec::new();
    for parameters in values {
        let key = cpu_parameters_key(parameters);
        if seen.insert(key) {
            unique.push(parameters.clone());
        }
    }
    unique
}

pub fn cpu_reference_worker_count(
    game_count: usize,
    requested_workers: usize,
    pair_batch: usize,
) -> usize {
    game_count
        .min(requested_workers.max(1))
        .min(pair_batch.max(1))
}

pub fn cpu_training_position_worker_count(target: usize, cpu_workers: usize) -> usize {
    target.min(cpu_workers.max(1))
}

pub fn cpu_label_worker_count(position_count: usize, cpu_workers: usize) -> usize {
    position_count.min(cpu_workers.max(1))
}

pub fn cpu_candidate_worker_count(
    candidate_count: usize,
    cpu_workers: usize,
    pair_batch: usize,
) -> usize {
    candidate_count
        .min(cpu_workers.max(1))
        .min(pair_batch.max(1))
}

pub fn cpu_training_candidate_count(cpu_candidates: usize) -> usize {
    cpu_candidates.clamp(1, MAX_CPU_TRAINING_CANDIDATES)
}

pub fn cpu_screening_game_count(
    sample_game_count: usize,
    cpu_screening_opponent_variants: usize,
) -> usize {
    if sample_game_count == 0 {
        0
    } else {
        sample_game_count.min(cpu_screening_opponent_variants.max(1))
    }
}

pub fn cpu_training_finalist_target(
    population_len: usize,
    cpu_finalists: usize,
    cpu_pair_batch: usize,
    screened_len: usize,
) -> usize {
    if population_len == 0 {
        return 0;
    }
    let screened_or_population = if screened_len > 0 {
        screened_len
    } else {
        population_len
    };
    population_len.min(cpu_finalists.max(cpu_pair_batch.min(screened_or_population)))
}

pub fn cpu_training_elite_count(cpu_finalists: usize) -> usize {
    cpu_finalists.clamp(1, MAX_CPU_TRAINING_ELITES)
}

pub fn cpu_training_candidate_improved(
    candidate_score: f64,
    baseline_score: f64,
    best_candidate_score: f64,
) -> bool {
    candidate_score.is_finite()
        && candidate_score > baseline_score
        && candidate_score > best_candidate_score
}

pub fn cpu_training_next_stagnation(generations_without_candidate: usize, improved: bool) -> usize {
    if improved {
        0
    } else {
        generations_without_candidate.saturating_add(1)
    }
}

pub fn cpu_training_should_continue(
    now_ms: f64,
    deadline_at_ms: f64,
    generations_without_candidate: usize,
    max_generations_without_candidate: usize,
) -> bool {
    now_ms < deadline_at_ms && generations_without_candidate < max_generations_without_candidate
}

pub fn rank_cpu_scored_candidates(
    mut candidates: Vec<CpuScoredCandidate>,
) -> Vec<CpuScoredCandidate> {
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
    candidates
}

pub fn cpu_training_elites(
    candidates: &[CpuScoredCandidate],
    baseline: &CpuParameters,
    cpu_finalists: usize,
) -> Vec<CpuParameters> {
    let baseline_key = cpu_parameters_key(baseline);
    rank_cpu_scored_candidates(candidates.to_vec())
        .into_iter()
        .filter(|entry| cpu_parameters_key(&entry.parameters) != baseline_key)
        .take(cpu_training_elite_count(cpu_finalists))
        .map(|entry| entry.parameters)
        .collect()
}

pub fn cpu_training_finalist_candidates(
    baseline: &CpuParameters,
    screened: &[CpuScoredCandidate],
    target: usize,
) -> Vec<CpuParameters> {
    let mut candidates = Vec::with_capacity(target.saturating_add(1));
    candidates.push(baseline.clone());
    candidates.extend(
        rank_cpu_scored_candidates(screened.to_vec())
            .into_iter()
            .take(target)
            .map(|entry| entry.parameters),
    );
    unique_cpu_parameters(&candidates)
}

pub fn cpu_candidate_scoring_plan(
    candidates: &[CpuParameters],
    fitness: &[CpuFitnessEntry],
) -> CpuCandidateScoringPlan {
    let unique_candidates = unique_cpu_parameters(candidates);
    let fitness_by_key = fitness
        .iter()
        .map(|entry| (entry.key.clone(), entry.score))
        .collect::<BTreeMap<_, _>>();
    let mut cached_scores = Vec::new();
    let mut uncached_candidates = Vec::new();
    for parameters in &unique_candidates {
        let key = cpu_parameters_key(parameters);
        if let Some(score) = fitness_by_key.get(&key) {
            cached_scores.push(CpuScoredCandidate {
                parameters: parameters.clone(),
                score: *score,
            });
        } else {
            uncached_candidates.push(parameters.clone());
        }
    }
    let cache_hits = cached_scores.len();
    CpuCandidateScoringPlan {
        unique_candidates,
        cached_scores,
        uncached_candidates,
        cache_hits,
    }
}

pub fn breed_cpu_population(
    baseline: &[(String, i32)],
    elites: &[CpuParameters],
    target: usize,
    seed: u32,
    generation: u32,
    stagnation: u32,
) -> Vec<CpuParameters> {
    let mut candidates = Vec::with_capacity(elites.len() + 1);
    candidates.push(baseline.to_vec());
    candidates.extend_from_slice(elites);
    let parents = unique_cpu_parameters(&candidates);
    let initial = target.min(parents.len()).max(1);
    let mut population = parents.iter().take(initial).cloned().collect::<Vec<_>>();
    let mutation_scale = (1.0 + stagnation as f64 * 0.4).min(3.0);
    let max_attempts = target.saturating_mul(64).max(64);
    for attempt in 0..max_attempts {
        if population.len() >= target {
            break;
        }
        let left = parents
            .get((attempt + generation as usize) % parents.len())
            .map(Vec::as_slice)
            .unwrap_or(baseline);
        let right = parents
            .get((attempt * 5 + generation as usize + 1) % parents.len())
            .map(Vec::as_slice)
            .unwrap_or(baseline);
        let child_seed = seed
            ^ (generation + 1).wrapping_mul(0x9e37_79b1)
            ^ ((attempt as u32) + 1).wrapping_mul(0x85eb_ca6b);
        let crossed = crossover_cpu_parameters(left, right, child_seed ^ 0xc2b2_ae35);
        let child = mutate_cpu_parameters(&crossed, child_seed, mutation_scale);
        let child_key = cpu_parameters_key(&child);
        if !population
            .iter()
            .any(|candidate| cpu_parameters_key(candidate) == child_key)
        {
            population.push(child);
        }
    }
    population
}

pub fn mutate_cpu_parameters(base: &[(String, i32)], seed: u32, scale: f64) -> CpuParameters {
    let mut rng = JsLcg::new(seed, 1_664_525, 1_013_904_223);
    let mut next = base.to_vec();
    let mut mutable = base
        .iter()
        .enumerate()
        .filter(|(_, (key, _))| key != "king" && key != "royal_queen")
        .map(|(index, (key, value))| (index, key.clone(), *value))
        .collect::<Vec<_>>();
    if mutable.is_empty() {
        return next;
    }
    for index in (1..mutable.len()).rev() {
        let swap_index = (rng.next() * (index + 1) as f64).floor() as usize;
        mutable.swap(index, swap_index);
    }
    let broad_mutation = rng.next() < 0.125;
    let sparse_target = js_round((1.0 + rng.next() * 3.0) * scale.max(0.25).sqrt())
        .max(1)
        .min(mutable.len() as i32) as usize;
    let mutation_target = if broad_mutation {
        sparse_target
            .max(((mutable.len() as f64) * (0.2 * scale.max(1.0)).min(0.8)).ceil() as usize)
    } else {
        sparse_target
    };
    let mut changed = 0;
    for (index, _, value) in &mutable {
        if changed >= mutation_target {
            break;
        }
        let spread = js_round(value.abs() as f64 * 0.08 * scale.max(0.25)).max(1);
        let mut delta = js_round((rng.next() * 2.0 - 1.0) * spread as f64);
        if delta == 0 {
            delta = if rng.next() < 0.5 { -1 } else { 1 };
        }
        let mutated = (*value + delta).clamp(-10_000, 10_000);
        if mutated != *value {
            next[*index].1 = mutated;
            changed += 1;
        }
    }
    if changed == 0 {
        let (index, _, value) = mutable[0];
        next[index].1 = if value >= 10_000 {
            value - 1
        } else {
            value + 1
        };
    }
    next
}

pub fn crossover_cpu_parameters(
    left: &[(String, i32)],
    right: &[(String, i32)],
    seed: u32,
) -> CpuParameters {
    let mut rng = JsLcg::new(seed, 1_103_515_245, 12_345);
    let mut child = Vec::new();
    let mut keys = Vec::<String>::new();
    for (key, _) in left.iter().chain(right.iter()) {
        if !keys.iter().any(|existing| existing == key) {
            keys.push(key.clone());
        }
    }
    for key in keys {
        let left_value = parameter_value(left, &key);
        let right_value = parameter_value(right, &key);
        let value = match (left_value, right_value) {
            (Some(left), Some(_)) if key == "king" || key == "royal_queen" => left,
            (Some(left), Some(right)) => {
                let blend = rng.next();
                js_round(left as f64 * blend + right as f64 * (1.0 - blend))
            }
            (Some(left), None) => left,
            (None, Some(right)) => right,
            (None, None) => 0,
        };
        child.push((key, value));
    }
    child
}

pub fn cpu_match_turn_time_ms(
    cpu_training_time_ms: i32,
    now_ms: f64,
    deadline_at_ms: f64,
    remaining_searches: usize,
) -> i32 {
    let remaining = remaining_searches.max(1) as f64;
    let available = ((deadline_at_ms - now_ms) / remaining).floor() as i32;
    available.clamp(1, cpu_training_time_ms.max(1))
}

pub fn cpu_paired_match_deadline_ms(
    now_ms: f64,
    deadline_at_ms: f64,
    total_matches: usize,
    completed_matches: usize,
) -> f64 {
    let remaining_matches = total_matches.saturating_sub(completed_matches).max(1) as f64;
    let slice_ms = ((deadline_at_ms - now_ms) / remaining_matches).max(1.0);
    deadline_at_ms.min(now_ms + slice_ms)
}

pub fn cpu_paired_match_average_score(score: f64, completed_matches: usize) -> f64 {
    if completed_matches == 0 {
        f64::NAN
    } else {
        score / completed_matches as f64
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cpu_training_position_target(
    samples: usize,
    training_mode_count: usize,
    cpu_opponent_variants: usize,
    cpu_screening_opponent_variants: usize,
    cpu_rounds_per_variant: usize,
    cpu_league_contenders: usize,
    cpu_league_hall_of_fame_entries: usize,
    cpu_hall_of_fame_entries: i32,
    cpu_min_pairs: usize,
    cpu_max_pairs: usize,
    cpu_max_match_plies: usize,
) -> usize {
    let variant_pairs =
        (cpu_opponent_variants + cpu_screening_opponent_variants) * cpu_rounds_per_variant;
    let league_pairs = cpu_league_contenders * cpu_league_hall_of_fame_entries.max(1);
    let hall_pairs = cpu_hall_of_fame_entries.max(0) as usize;
    let requested = cpu_min_pairs.max(variant_pairs + league_pairs + hall_pairs);
    let capped_pairs = cpu_min_pairs
        .max(cpu_max_pairs)
        .min(requested)
        .min(cpu_max_match_plies);
    mode_label_target(samples, training_mode_count, capped_pairs.max(1))
}

pub fn cpu_training_budget_ms(
    cpu_train_seconds: u64,
    cpu_training_time_ms: u64,
    cpu_max_match_plies: usize,
    cpu_max_match_time_ms: u64,
) -> u64 {
    let fallback_ms = (cpu_train_seconds * 1_000)
        .min(cpu_training_time_ms * cpu_max_match_plies.max(1) as u64 * 60);
    let budget_ms = if cpu_max_match_time_ms > 0 {
        cpu_max_match_time_ms.min(fallback_ms)
    } else {
        fallback_ms
    };
    budget_ms.max(1_000)
}

pub fn cpu_training_position_search_config(
    cpu_depth: i32,
    cpu_nodes: i32,
) -> CpuTrainingPositionSearchConfig {
    CpuTrainingPositionSearchConfig {
        depth: cpu_depth.clamp(1, 2),
        nodes: cpu_nodes.clamp(1, 512),
    }
}

pub fn cpu_screening_training_config(
    cpu_depth: i32,
    depth: i32,
    cpu_nodes: i32,
    nodes: i32,
    cpu_training_time_ms: i32,
) -> CpuScreeningTrainingConfig {
    CpuScreeningTrainingConfig {
        cpu_depth: screening_depth(cpu_depth),
        depth: screening_depth(depth),
        cpu_nodes: screening_quarter(cpu_nodes),
        nodes: screening_quarter(nodes),
        cpu_training_time_ms: screening_quarter(cpu_training_time_ms),
    }
}

fn screening_depth(value: i32) -> i32 {
    value.clamp(1, 2)
}

fn screening_quarter(value: i32) -> i32 {
    let upper = (((value as f64) / 4.0).ceil() as i32).max(1);
    value.clamp(1, upper)
}

pub fn mode_label_target(samples: usize, training_mode_count: usize, divisor: usize) -> usize {
    if training_mode_count <= 1 {
        samples
    } else {
        samples.div_ceil(divisor.max(1)).max(1)
    }
}

pub fn cpu_search_label_weight(training_mode_count: usize) -> f32 {
    if training_mode_count > 1 {
        1.1
    } else {
        1.0
    }
}

pub fn cpu_reference_comparison_count(game_count: usize, reference_count: usize) -> usize {
    game_count.min(if reference_count == 0 {
        game_count
    } else {
        reference_count
    })
}

pub fn cpu_reference_should_continue(
    now_ms: f64,
    deadline_at_ms: f64,
    compared: usize,
    max_match_plies: usize,
) -> bool {
    now_ms < deadline_at_ms && compared < max_match_plies
}

pub fn move_agreement_bonus(left: &[CpuTrainingMove], right: &[CpuTrainingMove]) -> i32 {
    let left_key = bot_training_moves_key(left);
    let right_key = bot_training_moves_key(right);
    if !left_key.is_empty() && left_key == right_key {
        CPU_MOVE_AGREEMENT_BONUS
    } else {
        0
    }
}

pub fn cpu_reference_score_delta(
    candidate_score: i32,
    reference_score: i32,
    candidate_moves: &[CpuTrainingMove],
    reference_moves: &[CpuTrainingMove],
    draw_window: i32,
) -> CpuReferenceScoreDelta {
    let delta = candidate_score - reference_score;
    CpuReferenceScoreDelta {
        score: delta + move_agreement_bonus(candidate_moves, reference_moves),
        near_draw: delta.abs() <= draw_window.max(0),
    }
}

pub fn cpu_reference_candidate_average(
    score: i32,
    compared: usize,
    near_draws: usize,
    draw_rate_limit: f32,
) -> f32 {
    let compared = compared.max(1);
    let average = score as f32 / compared as f32;
    let near_draw_rate = near_draws as f32 / compared as f32;
    if near_draw_rate > draw_rate_limit {
        average * 0.5
    } else {
        average
    }
}

pub fn cpu_training_no_move_score(candidate_turn: bool) -> i32 {
    if candidate_turn {
        -CPU_TRAINING_WIN_SCORE
    } else {
        CPU_TRAINING_WIN_SCORE
    }
}

pub fn cpu_training_winner_score(winner: Option<&str>, candidate_color: &str) -> i32 {
    match winner {
        Some(winner) if winner == candidate_color => CPU_TRAINING_WIN_SCORE,
        Some(_) => -CPU_TRAINING_WIN_SCORE,
        None => 0,
    }
}

pub fn cpu_training_adjudication_score(
    current_turn: &str,
    candidate_color: &str,
    baseline_score: i32,
) -> i32 {
    if current_turn == candidate_color {
        baseline_score
    } else {
        -baseline_score
    }
}

pub fn bot_training_moves_key(moves: &[CpuTrainingMove]) -> String {
    moves
        .iter()
        .map(|movement| {
            format!(
                "{},{},{},{},{},{},{},{}",
                movement.from_timeline_id,
                movement.from_time,
                movement.from_x,
                movement.from_y,
                movement.to_timeline_id,
                movement.to_time,
                movement.to_x,
                movement.to_y
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn parameter_value(parameters: &[(String, i32)], key: &str) -> Option<i32> {
    parameters
        .iter()
        .find(|(parameter_key, _)| parameter_key == key)
        .map(|(_, value)| *value)
}

fn js_round(value: f64) -> i32 {
    (value + 0.5).floor() as i32
}

struct JsLcg {
    state: u32,
    multiplier: u32,
    increment: u32,
}

impl JsLcg {
    fn new(seed: u32, multiplier: u32, increment: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
            multiplier,
            increment,
        }
    }

    fn next(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(self.multiplier)
            .wrapping_add(self.increment);
        self.state as f64 / u32::MAX as f64
    }
}
