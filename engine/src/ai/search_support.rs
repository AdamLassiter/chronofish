impl SearchContext {
    fn new(
        weights: EvalWeights,
        root_color: Color,
        max_nodes: usize,
        deadline: Option<SearchInstant>,
    ) -> Self {
        Self {
            weights,
            root_color,
            max_nodes,
            nodes: 0,
            deadline,
            fast_eval: true,
            options: SearchOptions::optimized(),
            table: std::collections::HashMap::new(),
            turn_plan_cache: std::collections::HashMap::new(),
            killers: vec![[None, None]; 16],
            history: std::collections::HashMap::new(),
            stats: SearchStats::default(),
        }
    }

    fn expired(&self) -> bool {
        deadline_expired(self.deadline)
    }

    fn exhausted(&self) -> bool {
        self.nodes >= self.max_nodes || self.expired()
    }

    fn record_cutoff(&mut self, depth: i32, movement: Option<MoveStep>) {
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

impl SearchOptions {
    fn optimized() -> Self {
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

    #[cfg(test)]
    fn baseline() -> Self {
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
}

impl SearchPerfSample {
    fn summary_score(&self) -> u128 {
        self.elapsed_micros
            + self.nodes as u128
            + self.stats.turn_plan_cache_hits as u128
            + self.stats.tt_hits as u128
            + self.stats.beta_cutoffs as u128
            + self.stats.reduced_searches as u128
            + self.stats.aspiration_researches as u128
            + self.stats.expensive_order_probes as u128
            + self.label.len() as u128
    }
}

fn deadline_expired(deadline: Option<SearchInstant>) -> bool {
    deadline.is_some_and(|deadline| SearchInstant::now() >= deadline)
}

fn search_deadline(millis: i32) -> Option<SearchInstant> {
    (millis > 0)
        .then(|| SearchInstant::now() + std::time::Duration::from_millis(millis as u64))
}

fn move_hash(movement: MoveStep) -> u64 {
    let mut hash = mix64(0x1234_5678_90ab_cdef);
    hash_position(&mut hash, movement.from);
    hash_position(&mut hash, movement.to);
    hash
}

fn hash_position(hash: &mut u64, position: Position) {
    hash_combine(hash, position.timeline_id as u64);
    hash_combine(hash, position.time as u64);
    hash_combine(hash, position.x as u64);
    hash_combine(hash, position.y as u64);
}

fn piece_hash(piece: Piece) -> u64 {
    (color_hash(piece.color) << 8) ^ piece_type_hash(piece.piece_type)
}

fn color_hash(color: Color) -> u64 {
    match color {
        Color::White => 1,
        Color::Black => 2,
    }
}

fn owner_hash(owner: TimelineOwner) -> u64 {
    match owner {
        TimelineOwner::Neutral => 0,
        TimelineOwner::White => 1,
        TimelineOwner::Black => 2,
    }
}

fn piece_type_hash(piece_type: PieceType) -> u64 {
    match piece_type {
        PieceType::King => 1,
        PieceType::CommonKing => 2,
        PieceType::Queen => 3,
        PieceType::RoyalQueen => 4,
        PieceType::Princess => 5,
        PieceType::Rook => 6,
        PieceType::Bishop => 7,
        PieceType::Unicorn => 8,
        PieceType::Dragon => 9,
        PieceType::Knight => 10,
        PieceType::Pawn => 11,
        PieceType::Brawn => 12,
    }
}

fn castling_hash(castling: CastlingRights) -> u64 {
    (castling.white_kingside as u64)
        | ((castling.white_queenside as u64) << 1)
        | ((castling.black_kingside as u64) << 2)
        | ((castling.black_queenside as u64) << 3)
}

fn hash_combine(hash: &mut u64, value: u64) {
    *hash ^= mix64(
        value
            .wrapping_add(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(*hash << 6)
            .wrapping_add(*hash >> 2),
    );
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
