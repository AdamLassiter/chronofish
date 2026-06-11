use super::*;

impl SearchContext {
    pub(crate) fn new(
        weights: EvalWeights,
        root_color: Color,
        max_nodes: usize,
        deadline: Option<SearchInstant>,
    ) -> Self {
        Self {
            weights,
            evaluator: ValueEvaluator::heuristic(),
            root_color,
            max_nodes,
            nodes: 0,
            deadline,
            options: SearchOptions::optimized(),
            table: TranspositionTable::new(max_nodes),
            evaluation_cache: EvaluationCache::new(max_nodes),
            turn_plan_cache: std::collections::HashMap::new(),
            killers: vec![[None, None]; 16],
            history: std::collections::HashMap::new(),
            stats: SearchStats::default(),
        }
    }

    pub(crate) fn expired(&self) -> bool {
        deadline_expired(self.deadline)
    }

    pub(crate) fn exhausted(&self) -> bool {
        self.nodes >= self.max_nodes || self.expired()
    }

    pub(crate) fn root_plan_limit(&self) -> usize {
        if self.max_nodes <= FAST_SEARCH_NODE_THRESHOLD {
            FAST_ROOT_TURN_PLANS
        } else {
            MAX_ROOT_TURN_PLANS
        }
    }

    pub(crate) fn child_plan_limit(&self) -> usize {
        if self.max_nodes <= FAST_SEARCH_NODE_THRESHOLD {
            FAST_CHILD_TURN_PLANS
        } else {
            MAX_CHILD_TURN_PLANS
        }
    }

    pub(crate) fn quiescence_depth(&self) -> i32 {
        if self.max_nodes <= FAST_SEARCH_NODE_THRESHOLD {
            0
        } else {
            MAX_QUIESCENCE_DEPTH
        }
    }

    pub(crate) fn use_aspiration_windows(&self) -> bool {
        self.options.aspiration_windows && self.max_nodes > FAST_SEARCH_NODE_THRESHOLD
    }

    pub(crate) fn charge_move_generation(&mut self, count: usize) {
        self.stats.generated_moves += count;
        self.nodes = self.nodes.saturating_add(count).min(self.max_nodes);
    }

    pub(crate) fn charge_clone(&mut self) -> bool {
        if self.exhausted() {
            return false;
        }
        self.stats.search_clones += 1;
        self.nodes += 1;
        true
    }

    pub(crate) fn charge_move_application(&mut self) -> bool {
        if self.exhausted() {
            return false;
        }
        self.nodes += 1;
        true
    }

    pub(crate) fn record_generated_plan(&mut self) {
        self.stats.generated_plans += 1;
    }

    pub(crate) fn evaluate(&mut self, game: &Game, color: Color) -> i32 {
        let key = game.position_hash ^ color_hash(color).rotate_left(29);
        if let Some(score) = self.evaluation_cache.get(key) {
            self.stats.evaluation_cache_hits += 1;
            return score;
        }
        let mut evaluation_stats = EvaluationStats::default();
        let score = self.evaluator.evaluate_with_limits(
            game,
            color,
            &self.weights,
            EvaluationLimits::for_nodes(self.max_nodes),
            &mut evaluation_stats,
        );
        self.stats.evaluation_calls += evaluation_stats.calls;
        self.stats.evaluated_turn_moves += evaluation_stats.turn_moves;
        self.stats.evaluation_setup_probes += evaluation_stats.setup_probes;
        self.stats.evaluation_clones += evaluation_stats.clones;
        self.evaluation_cache.insert(key, score);
        score
    }

    pub(crate) fn record_cutoff(&mut self, depth: i32, movement: Option<MoveStep>) {
        self.stats.beta_cutoffs += 1;
        let Some(movement) = movement else {
            return;
        };
        if self.options.killer_moves {
            let index = depth.max(0) as usize;
            if index >= self.killers.len() {
                self.killers.resize(index + 1, [None, None]);
            }
            if self.killers[index][0] != Some(movement) {
                self.killers[index][1] = self.killers[index][0];
                self.killers[index][0] = Some(movement);
            }
        }
        if self.options.history_heuristic {
            *self.history.entry(move_hash(movement)).or_default() += HISTORY_BONUS * depth.max(1);
        }
    }
}

impl EvaluationCache {
    pub(crate) fn new(max_nodes: usize) -> Self {
        let capacity = max_nodes.clamp(512, 32_768).next_power_of_two();
        Self {
            slots: vec![None; capacity],
            mask: capacity - 1,
        }
    }

    pub(crate) fn get(&self, key: u64) -> Option<i32> {
        self.slots[key as usize & self.mask]
            .filter(|slot| slot.key == key)
            .map(|slot| slot.score)
    }

    pub(crate) fn insert(&mut self, key: u64, score: i32) {
        let index = key as usize & self.mask;
        self.slots[index] = Some(EvaluationSlot { key, score });
    }
}

impl TranspositionTable {
    pub(crate) fn new(max_nodes: usize) -> Self {
        let capacity = max_nodes.clamp(1_024, 65_536).next_power_of_two();
        Self {
            slots: vec![None; capacity],
            mask: capacity - 1,
        }
    }

    pub(crate) fn get(&self, key: &u64) -> Option<&SearchEntry> {
        self.slots[*key as usize & self.mask]
            .as_ref()
            .filter(|slot| slot.key == *key)
            .map(|slot| &slot.entry)
    }

    pub(crate) fn insert(&mut self, key: u64, entry: SearchEntry) {
        let index = key as usize & self.mask;
        let replace =
            self.slots[index].is_none_or(|slot| slot.key == key || entry.depth >= slot.entry.depth);
        if replace {
            self.slots[index] = Some(TranspositionSlot { key, entry });
        }
    }
}

impl SearchOptions {
    pub(crate) fn optimized() -> Self {
        Self {
            tt_best_move: true,
            killer_moves: true,
            history_heuristic: true,
            direct_quiescence: true,
            late_move_reduction: true,
            aspiration_windows: true,
            capture_sanity: true,
            turn_plan_cache: true,
        }
    }

    pub(crate) fn minimal() -> Self {
        Self {
            tt_best_move: false,
            killer_moves: false,
            history_heuristic: false,
            direct_quiescence: false,
            late_move_reduction: false,
            aspiration_windows: false,
            capture_sanity: false,
            turn_plan_cache: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn baseline() -> Self {
        Self::minimal()
    }
}

impl SearchPerfSample {
    pub(crate) fn summary_score(&self) -> u128 {
        self.elapsed_micros
            + self.nodes as u128
            + self.stats.generated_moves as u128
            + self.stats.generated_plans as u128
            + self.stats.search_clones as u128
            + self.stats.turn_plan_cache_hits as u128
            + self.stats.tt_hits as u128
            + self.stats.beta_cutoffs as u128
            + self.stats.reduced_searches as u128
            + self.stats.aspiration_researches as u128
            + self.stats.expensive_order_probes as u128
            + self.stats.evaluation_calls as u128
            + self.stats.evaluation_cache_hits as u128
            + self.stats.evaluated_turn_moves as u128
            + self.stats.evaluation_setup_probes as u128
            + self.stats.evaluation_clones as u128
            + self.label.len() as u128
    }
}

pub(crate) fn deadline_expired(deadline: Option<SearchInstant>) -> bool {
    deadline.is_some_and(|deadline| SearchInstant::now() >= deadline)
}

pub(crate) fn search_deadline(millis: i32) -> Option<SearchInstant> {
    (millis > 0).then(|| SearchInstant::now() + std::time::Duration::from_millis(millis as u64))
}

pub(crate) fn move_hash(movement: MoveStep) -> u64 {
    let mut hash = mix64(0x1234_5678_90ab_cdef);
    hash_position(&mut hash, movement.from);
    hash_position(&mut hash, movement.to);
    hash
}

pub(crate) fn hash_position(hash: &mut u64, position: Position) {
    hash_combine(hash, position.timeline_id as u64);
    hash_combine(hash, position.time as u64);
    hash_combine(hash, position.x as u64);
    hash_combine(hash, position.y as u64);
}

pub(crate) fn color_hash(color: Color) -> u64 {
    match color {
        Color::White => 1,
        Color::Black => 2,
    }
}

pub(crate) fn hash_combine(hash: &mut u64, value: u64) {
    *hash ^= mix64(
        value
            .wrapping_add(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(*hash << 6)
            .wrapping_add(*hash >> 2),
    );
}

pub(crate) fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
