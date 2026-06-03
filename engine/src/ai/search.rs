struct TurnPlanBuildContext<'a> {
    color: Color,
    weights: &'a EvalWeights,
    deadline: Option<SearchInstant>,
    move_limit: usize,
    capture_sanity: bool,
}

impl Game {
    #[allow(dead_code)]
    fn ai_turn_json(&self, max_depth: i32, max_nodes: i32) -> String {
        self.best_ai_turn(max_depth, max_nodes, None).to_json()
    }

    #[allow(dead_code)]
    fn ai_turn_timed_json(&self, max_depth: i32, max_nodes: i32, millis: i32) -> String {
        self.best_ai_turn(max_depth, max_nodes, search_deadline(millis))
            .to_json()
    }

    fn best_ai_turn(
        &self,
        max_depth: i32,
        max_nodes: i32,
        deadline: Option<SearchInstant>,
    ) -> AiSearchResult {
        self.best_ai_turn_with_options(
            max_depth,
            max_nodes,
            deadline,
            SearchOptions::optimized(),
            None,
        )
        .0
    }

    fn best_ai_turn_with_options(
        &self,
        max_depth: i32,
        max_nodes: i32,
        deadline: Option<SearchInstant>,
        options: SearchOptions,
        label: Option<&'static str>,
    ) -> (AiSearchResult, Option<SearchPerfSample>) {
        self.best_ai_turn_with_value_evaluator(
            max_depth,
            max_nodes,
            deadline,
            options,
            ValueEvaluator::heuristic(),
            label,
        )
    }

    fn best_ai_turn_with_value_evaluator(
        &self,
        max_depth: i32,
        max_nodes: i32,
        deadline: Option<SearchInstant>,
        options: SearchOptions,
        evaluator: ValueEvaluator,
        label: Option<&'static str>,
    ) -> (AiSearchResult, Option<SearchPerfSample>) {
        let started = SearchInstant::now();
        let depth = max_depth.max(1);
        let nodes = max_nodes.max(1) as usize;
        let weights = EvalWeights::default_tuned();
        let mut context = SearchContext::new(weights, self.turn, nodes, deadline);
        context.options = options;
        context.evaluator = evaluator;
        context.killers.resize((depth as usize).saturating_add(3), [None, None]);

        // Check evasions are tactically forced and should not wait behind the
        // full multiverse turn planner. In heavily branched positions the full
        // planner can spend its budget proving long alternatives while a simple
        // capture/block already saves the royal piece.
        if let Some(plan) = self.immediate_check_escape_plan(&mut context) {
            let score = plan.score_hint;
            let result = AiSearchResult {
                moves: plan.moves,
                score,
                depth: 1,
                nodes: context.nodes,
                status: "ok",
            };
            let sample = label.map(|label| SearchPerfSample {
                label,
                elapsed_micros: SearchInstant::now().duration_since(started).as_micros(),
                nodes: context.nodes,
                stats: context.stats,
            });
            if let Some(sample) = &sample {
                let _ = sample.summary_score();
            }
            return (result, sample);
        }

        let mut best = AiSearchResult {
            moves: Vec::new(),
            score: 0,
            depth: 0,
            nodes: 0,
            status: "noLegalTurn",
        };

        // Iterative deepening preserves a usable shallower answer if the node
        // limit is hit before the requested depth completes.
        let mut previous_score = 0;
        for current_depth in 1..=depth {
            let window = if context.options.aspiration_windows && current_depth > 1 {
                Some((
                    previous_score - ASPIRATION_WINDOW,
                    previous_score + ASPIRATION_WINDOW,
                ))
            } else {
                None
            };
            let Some((plan, score)) = self.search_root(current_depth, &mut context, window) else {
                break;
            };
            previous_score = score;
            best = AiSearchResult {
                moves: plan.moves,
                score,
                depth: current_depth,
                nodes: context.nodes,
                status: "ok",
            };
            if context.exhausted() || score.abs() >= CHECKMATE_SCORE / 2 {
                break;
            }
        }

        best.nodes = context.nodes;
        let sample = label.map(|label| SearchPerfSample {
            label,
            elapsed_micros: SearchInstant::now().duration_since(started).as_micros(),
            nodes: context.nodes,
            stats: context.stats,
        });
        if let Some(sample) = &sample {
            let _ = sample.summary_score();
        }
        (best, sample)
    }

    #[allow(dead_code)]
    fn best_ai_turn_partitioned(
        &self,
        max_depth: i32,
        max_nodes: i32,
        deadline: Option<SearchInstant>,
        partition_index: usize,
        partition_count: usize,
    ) -> AiSearchResult {
        self.best_ai_turn_partitioned_with_value_evaluator(
            max_depth,
            max_nodes,
            deadline,
            partition_index,
            partition_count,
            ValueEvaluator::heuristic(),
        )
    }

    #[allow(dead_code)]
    fn best_ai_turn_partitioned_with_value_evaluator(
        &self,
        max_depth: i32,
        max_nodes: i32,
        deadline: Option<SearchInstant>,
        partition_index: usize,
        partition_count: usize,
        evaluator: ValueEvaluator,
    ) -> AiSearchResult {
        let depth = max_depth.max(1);
        let nodes = max_nodes.max(1) as usize;
        let weights = EvalWeights::default_tuned();
        let mut context = SearchContext::new(weights, self.turn, nodes, deadline);
        context.options = SearchOptions::optimized();
        context.evaluator = evaluator;
        context.killers.resize((depth as usize).saturating_add(3), [None, None]);

        if let Some(plan) = self.immediate_check_escape_plan(&mut context) {
            return AiSearchResult {
                moves: plan.moves,
                score: plan.score_hint,
                depth: 1,
                nodes: context.nodes,
                status: "ok",
            };
        }

        let partition_count = partition_count.max(1);
        let partition_index = partition_index.min(partition_count - 1);
        let partition = Some((partition_index, partition_count));
        let mut best = AiSearchResult {
            moves: Vec::new(),
            score: 0,
            depth: 0,
            nodes: 0,
            status: "noLegalTurn",
        };
        let mut previous_score = 0;
        for current_depth in 1..=depth {
            let window = if context.options.aspiration_windows && current_depth > 1 {
                Some((
                    previous_score - ASPIRATION_WINDOW,
                    previous_score + ASPIRATION_WINDOW,
                ))
            } else {
                None
            };
            let Some((plan, score)) =
                self.search_root_partitioned(current_depth, &mut context, window, partition)
            else {
                break;
            };
            previous_score = score;
            best = AiSearchResult {
                moves: plan.moves,
                score,
                depth: current_depth,
                nodes: context.nodes,
                status: "ok",
            };
            if context.exhausted() || score.abs() >= CHECKMATE_SCORE / 2 {
                break;
            }
        }

        best.nodes = context.nodes;
        best
    }

    fn search_root(
        &self,
        depth: i32,
        context: &mut SearchContext,
        window: Option<(i32, i32)>,
    ) -> Option<(TurnPlan, i32)> {
        self.search_root_partitioned(depth, context, window, None)
    }

    fn search_root_partitioned(
        &self,
        depth: i32,
        context: &mut SearchContext,
        window: Option<(i32, i32)>,
        partition: Option<(usize, usize)>,
    ) -> Option<(TurnPlan, i32)> {
        if context.expired() {
            return None;
        }

        let mut plans = self.legal_turn_plans_with_context(context);
        if let Some((partition_index, partition_count)) = partition {
            plans = plans
                .into_iter()
                .enumerate()
                .filter_map(|(index, plan)| {
                    (index % partition_count == partition_index).then_some(plan)
                })
                .collect();
        }
        let (mut alpha, beta) = window.unwrap_or((-CHECKMATE_SCORE * 2, CHECKMATE_SCORE * 2));
        let mut best = self.search_root_with_bounds(depth, context, plans.clone(), alpha, beta);

        if let (Some((_, score)), Some((low, high))) = (&best, window) {
            if *score <= low || *score >= high {
                context.stats.aspiration_researches += 1;
                alpha = -CHECKMATE_SCORE * 2;
                best = self.search_root_with_bounds(depth, context, plans, alpha, CHECKMATE_SCORE * 2);
            }
        }

        best
    }

    fn search_root_with_bounds(
        &self,
        depth: i32,
        context: &mut SearchContext,
        plans: Vec<TurnPlan>,
        mut alpha: i32,
        beta: i32,
    ) -> Option<(TurnPlan, i32)> {
        let mut best: Option<(TurnPlan, i32)> = None;

        for plan in plans {
            if context.exhausted() {
                break;
            }
            let score = plan
                .game
                .alpha_beta(depth - 1, alpha, beta, context.root_color, context);
            // Deterministic tie-breaking matters for repeatable self-play and
            // stable worker output in the frontend.
            let replace = best.as_ref().is_none_or(|(current, current_score)| {
                score > *current_score
                    || score == *current_score && Self::turn_plan_cmp(&plan, current).is_lt()
            });
            if replace {
                alpha = alpha.max(score);
                best = Some((plan, score));
            }
        }

        best
    }

    fn alpha_beta(
        &self,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        maximizing_color: Color,
        context: &mut SearchContext,
    ) -> i32 {
        // Children are whole submitted turn plans, not individual piece moves.
        context.nodes += 1;
        if let Some(score) = self.terminal_score(maximizing_color) {
            return score;
        }
        if context.exhausted() {
            return context.evaluator.evaluate(self, maximizing_color, &context.weights);
        }

        if depth <= 0 {
            return self.quiescence(
                -CHECKMATE_SCORE * 2,
                CHECKMATE_SCORE * 2,
                maximizing_color,
                context,
                MAX_QUIESCENCE_DEPTH,
            );
        }

        let key = self.search_key(depth, maximizing_color);
        if let Some(entry) = context.table.get(&key) {
            if entry.depth >= depth {
                context.stats.tt_hits += 1;
                return entry.score;
            }
        }

        let plans = self.legal_turn_plans_with_context(context);
        if plans.is_empty() {
            return context.evaluator.evaluate(self, maximizing_color, &context.weights);
        }

        let result = if self.turn == maximizing_color {
            let mut best = -CHECKMATE_SCORE * 2;
            let mut best_move = None;
            for (index, plan) in plans.iter().enumerate() {
                let child_depth = self.child_search_depth(depth, index, plan, context);
                let mut score = if index > 0 {
                    plan.game.alpha_beta(
                        child_depth,
                        alpha,
                        alpha + 1,
                        maximizing_color,
                        context,
                    )
                } else {
                    plan.game
                        .alpha_beta(child_depth, alpha, beta, maximizing_color, context)
                };
                if (child_depth < depth - 1 || index > 0) && score > alpha && score < beta {
                    score = plan.game.alpha_beta(
                        depth - 1,
                        alpha,
                        beta,
                        maximizing_color,
                        context,
                    );
                }
                if score > best {
                    best_move = plan.moves.first().copied();
                }
                best = best.max(score);
                alpha = alpha.max(best);
                if beta <= alpha || context.exhausted() {
                    context.record_cutoff(depth, plan.moves.first().copied());
                    break;
                }
            }
            context.table.insert(
                key,
                SearchEntry {
                    depth,
                    score: best,
                    best_move,
                },
            );
            best
        } else {
            let mut best = CHECKMATE_SCORE * 2;
            let mut best_move = None;
            for (index, plan) in plans.iter().enumerate() {
                let child_depth = self.child_search_depth(depth, index, plan, context);
                let mut score = if index > 0 {
                    plan.game.alpha_beta(
                        child_depth,
                        beta - 1,
                        beta,
                        maximizing_color,
                        context,
                    )
                } else {
                    plan.game
                        .alpha_beta(child_depth, alpha, beta, maximizing_color, context)
                };
                if (child_depth < depth - 1 || index > 0) && score > alpha && score < beta {
                    score = plan.game.alpha_beta(
                        depth - 1,
                        alpha,
                        beta,
                        maximizing_color,
                        context,
                    );
                }
                if score < best {
                    best_move = plan.moves.first().copied();
                }
                best = best.min(score);
                beta = beta.min(best);
                if beta <= alpha || context.exhausted() {
                    context.record_cutoff(depth, plan.moves.first().copied());
                    break;
                }
            }
            context.table.insert(
                key,
                SearchEntry {
                    depth,
                    score: best,
                    best_move,
                },
            );
            best
        };
        result
    }

    fn quiescence(
        &self,
        mut alpha: i32,
        mut beta: i32,
        maximizing_color: Color,
        context: &mut SearchContext,
        depth: i32,
    ) -> i32 {
        if let Some(score) = self.terminal_score(maximizing_color) {
            return score;
        }
        let stand_pat = context.evaluator.evaluate(self, maximizing_color, &context.weights);
        if depth <= 0 || context.exhausted() {
            return stand_pat;
        }
        let moves: Vec<MoveStep> = if context.options.direct_quiescence {
            self.forcing_moves_until(&context.weights, context.deadline)
        } else {
            let weights = context.weights;
            self.legal_single_moves_until(&context.weights, context.deadline)
                .into_iter()
                .filter(|movement| self.is_forcing_move(movement, &weights, context))
                .take(6)
                .collect()
        };

        if self.turn == maximizing_color {
            if stand_pat >= beta {
                return beta;
            }
            alpha = alpha.max(stand_pat);

            let mut best = stand_pat;
            for movement in moves {
                if context.exhausted() {
                    break;
                }
                context.nodes += 1;
                let mut next = self.clone_for_search();
                if !next.apply_move_for_search(movement.from, movement.to) {
                    continue;
                }
                let score = next.quiescence(alpha, beta, maximizing_color, context, depth - 1);
                best = best.max(score);
                alpha = alpha.max(best);
                if alpha >= beta {
                    break;
                }
            }
            best
        } else {
            if stand_pat <= alpha {
                return alpha;
            }
            beta = beta.min(stand_pat);

            let mut best = stand_pat;
            for movement in moves {
                if context.exhausted() {
                    break;
                }
                context.nodes += 1;
                let mut next = self.clone_for_search();
                if !next.apply_move_for_search(movement.from, movement.to) {
                    continue;
                }
                let score = next.quiescence(alpha, beta, maximizing_color, context, depth - 1);
                best = best.min(score);
                beta = beta.min(best);
                if alpha >= beta {
                    break;
                }
            }
            best
        }
    }

    #[allow(dead_code)]
    fn legal_single_moves(&self, weights: &EvalWeights) -> Vec<MoveStep> {
        self.legal_single_moves_until(weights, None)
    }

    fn legal_single_moves_until(
        &self,
        weights: &EvalWeights,
        deadline: Option<SearchInstant>,
    ) -> Vec<MoveStep> {
        let mut moves = Vec::new();
        for timeline in &self.timelines {
            if !self.is_active_timeline(timeline.id) {
                continue;
            }
            for board in &timeline.boards {
                if deadline_expired(deadline) {
                    return moves;
                }
                if !self.is_latest_board(timeline.id, board.time) || board.side_to_move != self.turn
                {
                    continue;
                }
                for y in 0..8 {
                    for x in 0..8 {
                        let from = Position {
                            timeline_id: timeline.id,
                            time: board.time,
                            x,
                            y,
                        };
                        if !self
                            .piece_at(from)
                            .is_some_and(|piece| piece.color == self.turn)
                        {
                            continue;
                        }
                        for target_timeline in &self.timelines {
                            for target_board in &target_timeline.boards {
                                if deadline_expired(deadline) {
                                    return moves;
                                }
                                for target_y in 0..8 {
                                    for target_x in 0..8 {
                                        let to = Position {
                                            timeline_id: target_timeline.id,
                                            time: target_board.time,
                                            x: target_x,
                                            y: target_y,
                                        };
                                        let Some((piece, move_kind)) =
                                            self.legal_move_kind(from, to)
                                        else {
                                            continue;
                                        };
                                        if self.allows_search_move(from, to, piece, move_kind) {
                                            moves.push(MoveStep { from, to });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Order likely tactical/progress moves first using cheap facts only.
        // Deep tactical probes are reserved for the small set of moves that
        // survive turn-plan construction.
        moves.sort_by(|left, right| {
            self.cheap_move_order_score(right, weights)
                .cmp(&self.cheap_move_order_score(left, weights))
                .then_with(|| Self::move_cmp(left, right))
        });
        moves
    }

    fn forcing_moves_until(
        &self,
        weights: &EvalWeights,
        deadline: Option<SearchInstant>,
    ) -> Vec<MoveStep> {
        let mut moves = Vec::new();
        for movement in self.legal_single_moves_until(weights, deadline) {
            if deadline_expired(deadline) {
                break;
            }
            if self.piece_at(movement.to).is_some()
                || movement.from.timeline_id != movement.to.timeline_id
                || movement.from.time != movement.to.time
                || self.move_creates_royal_capture_setup(movement, weights)
            {
                moves.push(movement);
            }
            if moves.len() >= 6 {
                break;
            }
        }
        moves
    }

    fn cheap_move_order_score(&self, movement: &MoveStep, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        let is_branch = movement.from.timeline_id != movement.to.timeline_id
            || movement.from.time != movement.to.time;
        if let Some(piece) = self.piece_at(movement.to) {
            score += weights.piece_value(piece.piece_type) * 4;
            if Self::is_royal_piece(piece.piece_type) {
                score += CHECKMATE_SCORE / 4;
            }
        }
        if is_branch {
            score -= weights.branch_penalty;
        }
        if self
            .present_board()
            .is_some_and(|board| movement.from.time <= board.time)
        {
            score += weights.present_progress;
        }
        score += self.quiet_development_order_score(*movement, weights);
        score
    }

    fn move_order_score(&self, movement: &MoveStep, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        let captures_piece = self.piece_at(movement.to).is_some();
        if let Some(piece) = self.piece_at(movement.to) {
            score += weights.piece_value(piece.piece_type) * 4;
            if Self::is_royal_piece(piece.piece_type) {
                score += CHECKMATE_SCORE / 4;
            }
        }
        let is_branch = movement.from.timeline_id != movement.to.timeline_id
            || movement.from.time != movement.to.time;
        if is_branch {
            score -= weights.branch_penalty;
            if captures_piece {
                score += weights.branch_attack;
            }
        }
        if self
            .present_board()
            .is_some_and(|board| movement.from.time <= board.time)
        {
            score += weights.present_progress;
        }
        score += self.quiet_development_order_score(*movement, weights);
        let should_probe_tactics = captures_piece || is_branch;
        if self.move_creates_royal_capture_setup(*movement, weights) {
            score += weights.royal_capture_setup;
        }
        if should_probe_tactics {
            if let Some(piece) = self.piece_at(movement.from) {
                let mut next = self.clone_for_search();
                if next.apply_move_for_search(movement.from, movement.to) {
                    if next.royal_capture_available(piece.color) {
                        score += CHECKMATE_SCORE / 2;
                    }
                    score += next.royal_capture_setup_pressure_for_limited(piece.color, weights, 8);
                    if next.is_in_check(piece.color.opposite()) {
                        score += weights.check_bonus;
                        if is_branch {
                            score += weights.branch_attack;
                        }
                    }
                    score += next.fork_pressure_for(piece.color, weights) / 2;
                    score += next.forcing_pressure_for(piece.color, weights) / 2;
                    score -= next.royal_safety_for(piece.color, weights).min(0).abs() / 2;
                    let arrival = Position {
                        timeline_id: movement.to.timeline_id,
                        time: movement.to.time + 1,
                        x: movement.to.x,
                        y: movement.to.y,
                    };
                    if is_branch && next.attack_summary(arrival, piece.color).count >= 2 {
                        score += weights.branch_attack;
                    }
                }
            }
        }
        score
    }

    fn quiet_development_order_score(&self, movement: MoveStep, weights: &EvalWeights) -> i32 {
        if self.piece_at(movement.to).is_some() {
            return 0;
        }

        let Some(piece) = self.piece_at(movement.from) else {
            return 0;
        };
        if matches!(
            piece.piece_type,
            PieceType::Pawn | PieceType::Brawn | PieceType::King | PieceType::RoyalQueen
        ) {
            return 0;
        }

        let mut score = 0;
        let development_gain = development(piece.color, piece.piece_type, movement.to.y)
            - development(piece.color, piece.piece_type, movement.from.y);
        if development_gain > 0 {
            // Quiet piece development is often the only way to make later
            // forcing and temporal tactics visible before the search cutoff.
            score += development_gain * weights.development * 8;
        }

        let centrality_gain =
            centrality(movement.to.x, movement.to.y) - centrality(movement.from.x, movement.from.y);
        if centrality_gain > 0 {
            score += centrality_gain * weights.centrality / 2;
        }

        score += match piece.piece_type {
            PieceType::Queen | PieceType::Princess => weights.piece_activity * 3,
            PieceType::Rook | PieceType::Bishop | PieceType::Knight => weights.piece_activity * 2,
            PieceType::Unicorn | PieceType::Dragon => weights.piece_activity * 3,
            PieceType::CommonKing => weights.piece_activity,
            PieceType::Pawn | PieceType::Brawn | PieceType::King | PieceType::RoyalQueen => 0,
        };

        score
    }

    fn move_creates_royal_capture_setup(
        &self,
        movement: MoveStep,
        weights: &EvalWeights,
    ) -> bool {
        if weights.royal_capture_setup == 0
            || self
                .piece_at(movement.to)
                .is_some_and(|piece| Self::is_royal_piece(piece.piece_type))
        {
            return false;
        }
        let Some(piece) = self.piece_at(movement.from) else {
            return false;
        };
        let current_corridor_pressure =
            self.temporal_royal_corridor_pressure_for(piece.color, weights);
        let mut next = self.clone_for_search();
        next.apply_move_for_search(movement.from, movement.to)
            && (next.royal_capture_available(piece.color)
                || next.temporal_royal_corridor_pressure_for(piece.color, weights)
                    > current_corridor_pressure)
    }

    fn submit_turn_for_search(&mut self) -> bool {
        // Search mirrors user submission but returns a bool rather than writing a
        // user-facing status message.
        let Some(present_side) = self.present_board().map(|board| board.side_to_move) else {
            return false;
        };
        if self.has_pending_present_board(self.turn) {
            return false;
        }
        let royal_capture_by = self.staged_royal_capture_by;
        self.turn = present_side;
        self.staged_turn.clear();
        self.staged_notation.clear();
        self.staged_royal_capture_by = royal_capture_by;
        true
    }

    fn apply_move_for_search(&mut self, from: Position, to: Position) -> bool {
        let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
            return false;
        };
        if !self.allows_search_move(from, to, piece, move_kind) {
            return false;
        }

        let captured = self.captured_piece(to, move_kind);
        self.record_staged_capture(piece.color, captured);
        self.apply_move_unchecked(from, to, piece, move_kind);
        true
    }

    fn turn_plan_tactical_score(
        &self,
        moves: &[MoveStep],
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
        let mut score = 0;
        for movement in moves {
            score += self.move_order_score(movement, weights);
        }
        let mut next = self.clone_for_search();
        for movement in moves {
            let _ = next.apply_move_for_search(movement.from, movement.to);
        }
        if next.is_in_check(color.opposite()) {
            score += weights.check_bonus;
        }
        score
    }

    fn is_forcing_move(
        &self,
        movement: &MoveStep,
        weights: &EvalWeights,
        context: &mut SearchContext,
    ) -> bool {
        self.piece_at(movement.to).is_some()
            || movement.from.timeline_id != movement.to.timeline_id
            || movement.from.time != movement.to.time
            || {
                context.stats.expensive_order_probes += 1;
                self.move_order_score(movement, weights) >= weights.check_bonus
            }
    }

    fn search_key(&self, depth: i32, maximizing_color: Color) -> u64 {
        let mut hash = mix64(0x8a5c_7d13_9e37_79b9);
        hash_combine(&mut hash, depth as u64);
        hash_combine(&mut hash, color_hash(self.turn));
        hash_combine(&mut hash, color_hash(maximizing_color));
        for timeline in &self.timelines {
            hash_combine(&mut hash, timeline.id as u64);
            hash_combine(&mut hash, timeline.row as u64);
            hash_combine(&mut hash, owner_hash(timeline.owner));
            hash_combine(&mut hash, self.is_active_timeline(timeline.id) as u64);
            for board in &timeline.boards {
                hash_combine(&mut hash, board.time as u64);
                hash_combine(&mut hash, color_hash(board.side_to_move));
                hash_combine(&mut hash, castling_hash(board.castling));
                if let Some(en_passant) = board.en_passant {
                    hash_combine(&mut hash, en_passant.x as u64);
                    hash_combine(&mut hash, en_passant.y as u64);
                    hash_combine(&mut hash, en_passant.captured_x as u64);
                    hash_combine(&mut hash, en_passant.captured_y as u64);
                }
                for y in 0..8 {
                    for x in 0..8 {
                        if let Some(piece) = board.board[y][x] {
                            hash_combine(&mut hash, piece_hash(piece));
                            hash_combine(&mut hash, ((x as u64) << 3) | y as u64);
                        }
                    }
                }
            }
        }
        hash
    }

    fn turn_plan_cache_key(&self) -> u64 {
        self.search_key(0, self.turn)
    }

    fn turn_plan_cmp(left: &TurnPlan, right: &TurnPlan) -> std::cmp::Ordering {
        left.moves.len().cmp(&right.moves.len()).then_with(|| {
            left.moves
                .iter()
                .zip(&right.moves)
                .map(|(left_move, right_move)| Self::move_cmp(left_move, right_move))
                .find(|ordering| !ordering.is_eq())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn move_cmp(left: &MoveStep, right: &MoveStep) -> std::cmp::Ordering {
        position_key(left.from)
            .cmp(&position_key(right.from))
            .then_with(|| position_key(left.to).cmp(&position_key(right.to)))
    }

    fn child_search_depth(
        &self,
        depth: i32,
        index: usize,
        plan: &TurnPlan,
        context: &mut SearchContext,
    ) -> i32 {
        let mover = self.turn;
        let creates_or_answers_royal_setup =
            plan.game.royal_capture_available(mover)
                || plan
                    .game
                    .royal_capture_setup_pressure_for_limited(mover, &context.weights, 12)
                    > 0
                || plan
                    .game
                    .temporal_royal_corridor_pressure_for(mover, &context.weights)
                    > self.temporal_royal_corridor_pressure_for(mover, &context.weights)
                || plan.game.royal_capture_setup_pressure_for_limited(
                    mover.opposite(),
                    &context.weights,
                    12,
                ) < self.royal_capture_setup_pressure_for_limited(
                    mover.opposite(),
                    &context.weights,
                    12,
                );

        let reduced = context.options.late_move_reduction
            && depth > 2
            && index >= LATE_MOVE_REDUCTION_AFTER
            && !creates_or_answers_royal_setup
            && plan.moves.iter().all(|movement| {
                movement.from.timeline_id == movement.to.timeline_id
                    && movement.from.time == movement.to.time
                    && plan.game.piece_at(movement.to).is_none()
            });

        let base_depth = if reduced {
            context.stats.reduced_searches += 1;
            depth - 2
        } else {
            depth - 1
        };

        if creates_or_answers_royal_setup {
            (base_depth + 1).min(depth)
        } else {
            base_depth
        }
    }
}
