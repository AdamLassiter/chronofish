use super::*;

impl Game {
    pub(crate) const DEFAULT_MIN_AI_SEARCH_DEPTH: i32 = default_min_ai_search_depth();

    #[allow(dead_code)]
    pub(crate) fn ai_turn_json(&self, max_depth: i32, max_nodes: i32) -> String {
        self.best_ai_turn(max_depth, max_nodes, None).to_json()
    }

    #[allow(dead_code)]
    pub(crate) fn ai_turn_timed_json(&self, max_depth: i32, max_nodes: i32, millis: i32) -> String {
        self.best_ai_turn(max_depth, max_nodes, search_deadline(millis))
            .to_json()
    }

    #[allow(dead_code)]
    pub(crate) fn ai_turn_timed_min_depth_json(
        &self,
        max_depth: i32,
        min_depth: i32,
        max_nodes: i32,
        millis: i32,
    ) -> String {
        self.best_ai_turn_with_min_depth(max_depth, min_depth, max_nodes, search_deadline(millis))
            .to_json()
    }

    pub(crate) fn best_ai_turn(
        &self,
        max_depth: i32,
        max_nodes: i32,
        deadline: Option<SearchInstant>,
    ) -> AiSearchResult {
        self.best_ai_turn_with_min_depth(
            max_depth,
            Self::DEFAULT_MIN_AI_SEARCH_DEPTH,
            max_nodes,
            deadline,
        )
    }

    pub(crate) fn best_ai_turn_with_min_depth(
        &self,
        max_depth: i32,
        min_depth: i32,
        max_nodes: i32,
        deadline: Option<SearchInstant>,
    ) -> AiSearchResult {
        self.best_ai_turn_with_options_min_depth(
            max_depth,
            min_depth,
            max_nodes,
            deadline,
            SearchOptions::optimized(),
            None,
        )
        .0
    }

    #[allow(dead_code)]
    pub(crate) fn best_ai_turn_with_options(
        &self,
        max_depth: i32,
        max_nodes: i32,
        deadline: Option<SearchInstant>,
        options: SearchOptions,
        label: Option<&'static str>,
    ) -> (AiSearchResult, Option<SearchPerfSample>) {
        self.best_ai_turn_with_options_min_depth(
            max_depth,
            Self::DEFAULT_MIN_AI_SEARCH_DEPTH,
            max_nodes,
            deadline,
            options,
            label,
        )
    }

    pub(crate) fn best_ai_turn_with_options_min_depth(
        &self,
        max_depth: i32,
        min_depth: i32,
        max_nodes: i32,
        deadline: Option<SearchInstant>,
        options: SearchOptions,
        label: Option<&'static str>,
    ) -> (AiSearchResult, Option<SearchPerfSample>) {
        self.best_ai_turn_with_value_evaluator_min_depth(
            max_depth,
            min_depth,
            max_nodes,
            deadline,
            options,
            ValueEvaluator::heuristic(),
            label,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn best_ai_turn_with_value_evaluator(
        &self,
        max_depth: i32,
        max_nodes: i32,
        deadline: Option<SearchInstant>,
        options: SearchOptions,
        evaluator: ValueEvaluator,
        label: Option<&'static str>,
    ) -> (AiSearchResult, Option<SearchPerfSample>) {
        self.best_ai_turn_with_value_evaluator_min_depth(
            max_depth,
            Self::DEFAULT_MIN_AI_SEARCH_DEPTH,
            max_nodes,
            deadline,
            options,
            evaluator,
            label,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn best_ai_turn_with_value_evaluator_min_depth(
        &self,
        max_depth: i32,
        min_depth: i32,
        max_nodes: i32,
        deadline: Option<SearchInstant>,
        options: SearchOptions,
        evaluator: ValueEvaluator,
        label: Option<&'static str>,
    ) -> (AiSearchResult, Option<SearchPerfSample>) {
        let started = SearchInstant::now();
        let depth = max_depth.max(1);
        let min_completed_depth = depth.min(min_depth.max(1));
        let nodes = max_nodes.max(1) as usize;
        let weights = EvalWeights::active_tuned();
        let mut context = SearchContext::new(weights, self.turn, nodes, deadline);
        context.options = options;
        context.evaluator = evaluator;
        context
            .killers
            .resize((depth as usize).saturating_add(3), [None, None]);

        // Check evasions are tactically forced and should not wait behind the
        // full multiverse turn planner. In heavily branched positions the full
        // planner can spend its budget proving long alternatives while a simple
        // capture/block already saves the royal piece.
        if let Some(plan) = self.immediate_check_escape_plan(&mut context) {
            let score = plan.score_hint;
            let terminal_royal_capture = self.turn_plan_ends_in_royal_capture(&plan.moves);
            let result = AiSearchResult {
                principal_variation: vec![plan.moves.clone()],
                moves: plan.moves,
                score,
                depth: min_completed_depth,
                nodes: context.nodes,
                status: "ok",
                terminal_royal_capture,
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
            principal_variation: Vec::new(),
            terminal_royal_capture: false,
        };

        // Iterative deepening preserves a usable shallower answer if the node
        // limit is hit before the requested depth completes.
        let requested_deadline = context.deadline;
        let mut previous_score = 0;
        for current_depth in 1..=depth {
            context.deadline = if current_depth <= min_completed_depth {
                None
            } else {
                requested_deadline
            };
            let window = if context.use_aspiration_windows() && current_depth > 1 {
                Some((
                    previous_score - ASPIRATION_WINDOW,
                    previous_score + ASPIRATION_WINDOW,
                ))
            } else {
                None
            };
            let Some((plan, score, principal_variation)) =
                self.search_root_staged_with_pv(current_depth, &mut context, window, None)
            else {
                break;
            };
            previous_score = score;
            best = AiSearchResult {
                principal_variation,
                terminal_royal_capture: self.turn_plan_ends_in_royal_capture(&plan.moves),
                moves: plan.moves,
                score,
                depth: current_depth,
                nodes: context.nodes,
                status: "ok",
            };
            if (context.exhausted() && current_depth >= min_completed_depth)
                || score.abs() >= CHECKMATE_SCORE / 2
            {
                break;
            }
        }

        if best.status == "noLegalTurn" {
            let mut fallback =
                SearchContext::new(weights, self.turn, nodes.max(MAX_TURN_PLANS), None);
            fallback.options = SearchOptions::minimal();
            let plan_limit = fallback.root_plan_limit();
            if let Some(plan) = self
                .legal_turn_plans_with_context(&mut fallback, plan_limit)
                .into_iter()
                .next()
            {
                best = AiSearchResult {
                    principal_variation: vec![plan.moves.clone()],
                    terminal_royal_capture: self.turn_plan_ends_in_royal_capture(&plan.moves),
                    moves: plan.moves,
                    score: plan.score_hint,
                    depth: 1,
                    nodes: context.nodes + fallback.nodes,
                    status: "ok",
                };
            }
        }

        best.nodes = best.nodes.max(context.nodes);
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
    pub(crate) fn best_ai_turn_partitioned(
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
    pub(crate) fn best_ai_turn_partitioned_with_value_evaluator(
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
        let weights = EvalWeights::active_tuned();
        let mut context = SearchContext::new(weights, self.turn, nodes, deadline);
        context.options = SearchOptions::optimized();
        context.evaluator = evaluator;
        context
            .killers
            .resize((depth as usize).saturating_add(3), [None, None]);

        if let Some(plan) = self.immediate_check_escape_plan(&mut context) {
            return AiSearchResult {
                principal_variation: vec![plan.moves.clone()],
                terminal_royal_capture: self.turn_plan_ends_in_royal_capture(&plan.moves),
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
            principal_variation: Vec::new(),
            terminal_royal_capture: false,
        };
        let mut previous_score = 0;
        for current_depth in 1..=depth {
            let window = if context.use_aspiration_windows() && current_depth > 1 {
                Some((
                    previous_score - ASPIRATION_WINDOW,
                    previous_score + ASPIRATION_WINDOW,
                ))
            } else {
                None
            };
            let Some((plan, score, principal_variation)) =
                self.search_root_staged_with_pv(current_depth, &mut context, window, partition)
            else {
                break;
            };
            previous_score = score;
            best = AiSearchResult {
                principal_variation,
                terminal_royal_capture: self.turn_plan_ends_in_royal_capture(&plan.moves),
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

    pub(crate) fn search_root(
        &self,
        depth: i32,
        context: &mut SearchContext,
        window: Option<(i32, i32)>,
    ) -> Option<(TurnPlan, i32)> {
        self.search_root_partitioned(depth, context, window, None)
    }

    pub(crate) fn search_root_partitioned(
        &self,
        depth: i32,
        context: &mut SearchContext,
        window: Option<(i32, i32)>,
        partition: Option<(usize, usize)>,
    ) -> Option<(TurnPlan, i32)> {
        self.search_root_staged(depth, context, window, partition)
    }

    #[allow(dead_code)]
    pub(crate) fn search_root_with_bounds(
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
            if !context.charge_clone() {
                break;
            }
            let Some(child) = self.apply_turn_plan_for_search(&plan) else {
                continue;
            };
            let score = child.alpha_beta(depth - 1, alpha, beta, context.root_color, context);
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

    #[allow(dead_code)]
    pub(crate) fn alpha_beta_plan_based(
        &self,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        maximizing_color: Color,
        context: &mut SearchContext,
    ) -> i32 {
        // Children are whole submitted turn plans, not individual piece moves.
        context.nodes += 1;
        if let Some(score) = self.terminal_score_until(maximizing_color, context.deadline) {
            return score;
        }
        if context.exhausted() {
            return context.evaluate(self, maximizing_color);
        }

        if depth <= 0 {
            let mut game = self.clone_for_search();
            return game.quiescence(
                -CHECKMATE_SCORE * 2,
                CHECKMATE_SCORE * 2,
                maximizing_color,
                context,
                context.quiescence_depth(),
            );
        }

        let original_alpha = alpha;
        let original_beta = beta;
        let key = self.search_key(maximizing_color);
        if let Some(entry) = context.table.get(&key) {
            if entry.depth >= depth {
                let usable = match entry.bound {
                    SearchBound::Exact => true,
                    SearchBound::Lower => entry.score >= beta,
                    SearchBound::Upper => entry.score <= alpha,
                };
                if usable {
                    context.stats.tt_hits += 1;
                    return entry.score;
                }
            }
        }

        let plan_limit = context.child_plan_limit();
        let plans = self.legal_turn_plans_with_context(context, plan_limit);
        if plans.is_empty() {
            return context.evaluate(self, maximizing_color);
        }

        let result = if self.turn == maximizing_color {
            let mut best = -CHECKMATE_SCORE * 2;
            let mut best_move = None;
            for (index, plan) in plans.iter().enumerate() {
                if !context.charge_clone() {
                    break;
                }
                let Some(child) = self.apply_turn_plan_for_search(plan) else {
                    continue;
                };
                let child_depth = self.child_search_depth(depth, index, plan, &child, context);
                let mut score = if index > 0 {
                    child.alpha_beta(child_depth, alpha, alpha + 1, maximizing_color, context)
                } else {
                    child.alpha_beta(child_depth, alpha, beta, maximizing_color, context)
                };
                if (child_depth < depth - 1 || index > 0) && score > alpha && score < beta {
                    score = child.alpha_beta(depth - 1, alpha, beta, maximizing_color, context);
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
            let bound = if best <= original_alpha {
                SearchBound::Upper
            } else if best >= original_beta {
                SearchBound::Lower
            } else {
                SearchBound::Exact
            };
            context.table.insert(
                key,
                SearchEntry {
                    depth,
                    score: best,
                    bound,
                    best_move,
                },
            );
            best
        } else {
            let mut best = CHECKMATE_SCORE * 2;
            let mut best_move = None;
            for (index, plan) in plans.iter().enumerate() {
                if !context.charge_clone() {
                    break;
                }
                let Some(child) = self.apply_turn_plan_for_search(plan) else {
                    continue;
                };
                let child_depth = self.child_search_depth(depth, index, plan, &child, context);
                let mut score = if index > 0 {
                    child.alpha_beta(child_depth, beta - 1, beta, maximizing_color, context)
                } else {
                    child.alpha_beta(child_depth, alpha, beta, maximizing_color, context)
                };
                if (child_depth < depth - 1 || index > 0) && score > alpha && score < beta {
                    score = child.alpha_beta(depth - 1, alpha, beta, maximizing_color, context);
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
            let bound = if best <= original_alpha {
                SearchBound::Upper
            } else if best >= original_beta {
                SearchBound::Lower
            } else {
                SearchBound::Exact
            };
            context.table.insert(
                key,
                SearchEntry {
                    depth,
                    score: best,
                    bound,
                    best_move,
                },
            );
            best
        };
        result
    }

    pub(crate) fn quiescence(
        &mut self,
        mut alpha: i32,
        mut beta: i32,
        maximizing_color: Color,
        context: &mut SearchContext,
        depth: i32,
    ) -> i32 {
        if let Some(score) = self.terminal_score_until(maximizing_color, context.deadline) {
            return score;
        }
        let stand_pat = context.evaluate(self, maximizing_color);
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
                .take(MAX_QUIESCENCE_MOVES)
                .collect()
        };
        context.charge_move_generation(moves.len());

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
                if !context.charge_move_application() {
                    break;
                }
                let Some(undo) = self.make_search_move(movement) else {
                    continue;
                };
                let score = self.quiescence(alpha, beta, maximizing_color, context, depth - 1);
                self.unmake_search_move(undo);
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
                if !context.charge_move_application() {
                    break;
                }
                let Some(undo) = self.make_search_move(movement) else {
                    continue;
                };
                let score = self.quiescence(alpha, beta, maximizing_color, context, depth - 1);
                self.unmake_search_move(undo);
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
    pub(crate) fn legal_single_moves(&self, weights: &EvalWeights) -> Vec<MoveStep> {
        self.legal_single_moves_until(weights, None)
    }

    pub(crate) fn legal_single_moves_until(
        &self,
        weights: &EvalWeights,
        deadline: Option<SearchInstant>,
    ) -> Vec<MoveStep> {
        let mut moves = Vec::new();
        for timeline in &self.timelines {
            if !self.is_active_timeline(timeline.id) {
                continue;
            }
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            if deadline_expired(deadline) {
                return moves;
            }
            if board.side_to_move != self.turn {
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
                    let Some(piece) = self.piece_at(from).filter(|piece| piece.color == self.turn)
                    else {
                        continue;
                    };
                    self.for_each_piece_candidate_destination(from, piece, |to| {
                        if deadline_expired(deadline) {
                            return false;
                        }
                        let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
                            return true;
                        };
                        if self.allows_search_move(from, to, piece, move_kind) {
                            push_unique_move(&mut moves, MoveStep { from, to });
                        }
                        true
                    });
                    if deadline_expired(deadline) {
                        return self.order_moves(moves, weights);
                    }
                }
            }
        }

        // Order likely tactical/progress moves first using cheap facts only.
        // Deep tactical probes are reserved for the small set of moves that
        // survive turn-plan construction.
        self.order_moves(moves, weights)
    }

    pub(crate) fn legal_single_moves_for_board_until(
        &self,
        timeline_id: i32,
        time: i32,
        weights: &EvalWeights,
        deadline: Option<SearchInstant>,
    ) -> Vec<MoveStep> {
        let Some(board) = self.board(timeline_id, time) else {
            return Vec::new();
        };
        if !self.is_active_timeline(timeline_id)
            || !self.is_latest_board(timeline_id, time)
            || board.side_to_move != self.turn
            || self.present_time() != Some(time)
        {
            return Vec::new();
        }

        let mut moves = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                let from = Position {
                    timeline_id,
                    time,
                    x,
                    y,
                };
                let Some(piece) = self.piece_at(from).filter(|piece| piece.color == self.turn)
                else {
                    continue;
                };
                self.for_each_piece_candidate_destination(from, piece, |to| {
                    if deadline_expired(deadline) {
                        return false;
                    }
                    let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
                        return true;
                    };
                    if self.allows_search_move(from, to, piece, move_kind) {
                        push_unique_move(&mut moves, MoveStep { from, to });
                    }
                    true
                });
                if deadline_expired(deadline) {
                    return self.order_moves(moves, weights);
                }
            }
        }
        self.order_moves(moves, weights)
    }

    pub(crate) fn legal_single_moves_for_board_limited_until(
        &self,
        timeline_id: i32,
        time: i32,
        context: &mut SearchContext,
        limit: usize,
    ) -> Vec<MoveStep> {
        if limit == 0 {
            return Vec::new();
        }
        let Some(board) = self.board(timeline_id, time) else {
            return Vec::new();
        };
        if !self.is_active_timeline(timeline_id)
            || !self.is_latest_board(timeline_id, time)
            || board.side_to_move != self.turn
            || self.present_time() != Some(time)
        {
            return Vec::new();
        }

        let weights = context.weights;
        let deadline = context.deadline;
        let mut scored: Vec<(i32, MoveStep)> = Vec::new();
        let mut candidate_destinations = 0;
        let mut legal_move_attempts = 0;
        let mut expired = false;

        'squares: for y in 0..8 {
            for x in 0..8 {
                let from = Position {
                    timeline_id,
                    time,
                    x,
                    y,
                };
                let Some(piece) = self.piece_at(from).filter(|piece| piece.color == self.turn)
                else {
                    continue;
                };
                self.for_each_piece_candidate_destination(from, piece, |to| {
                    candidate_destinations += 1;
                    if deadline_expired(deadline) {
                        expired = true;
                        return false;
                    }
                    legal_move_attempts += 1;
                    let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
                        return true;
                    };
                    if self.allows_search_move(from, to, piece, move_kind) {
                        self.push_limited_ordered_move(
                            &mut scored,
                            MoveStep { from, to },
                            piece,
                            move_kind,
                            &weights,
                            limit,
                        );
                    }
                    true
                });
                if expired {
                    break 'squares;
                }
            }
        }

        context.stats.candidate_destinations += candidate_destinations;
        context.stats.legal_move_attempts += legal_move_attempts;
        scored.into_iter().map(|(_, movement)| movement).collect()
    }

    fn push_limited_ordered_move(
        &self,
        scored: &mut Vec<(i32, MoveStep)>,
        movement: MoveStep,
        piece: Piece,
        move_kind: MoveKind,
        weights: &EvalWeights,
        limit: usize,
    ) {
        if scored
            .iter()
            .any(|(_, existing)| existing.from == movement.from && existing.to == movement.to)
        {
            return;
        }
        let tier_bonus = if self.present_obligation_count(self.turn) > 1 {
            0
        } else {
            self.search_move_tier_bonus(movement, piece, move_kind, weights)
        };
        let score = self.cheap_move_order_score(&movement, weights) + tier_bonus;
        let insert_at = scored
            .iter()
            .position(|(existing_score, existing)| {
                score > *existing_score
                    || score == *existing_score && Self::move_cmp(&movement, existing).is_lt()
            })
            .unwrap_or(scored.len());
        if insert_at >= limit && scored.len() >= limit {
            return;
        }
        scored.insert(insert_at, (score, movement));
        if scored.len() > limit {
            scored.pop();
        }
        let Some((inserted_bucket, cap)) =
            self.temporal_dominance_bucket(movement, piece, move_kind, weights)
        else {
            return;
        };
        self.truncate_temporal_dominance(scored, weights, inserted_bucket, cap);
    }

    fn search_move_tier_bonus(
        &self,
        movement: MoveStep,
        piece: Piece,
        move_kind: MoveKind,
        weights: &EvalWeights,
    ) -> i32 {
        match self.search_move_tier(movement, piece, move_kind, weights) {
            SearchMoveTier::Forced => CHECKMATE_SCORE / 3,
            SearchMoveTier::Tactical => 80_000,
            SearchMoveTier::Strategic => 8_000,
            SearchMoveTier::Quiet => 0,
        }
    }

    fn search_move_tier(
        &self,
        movement: MoveStep,
        piece: Piece,
        move_kind: MoveKind,
        weights: &EvalWeights,
    ) -> SearchMoveTier {
        if self.is_in_check(piece.color) {
            return SearchMoveTier::Forced;
        }
        if self
            .piece_at(movement.to)
            .is_some_and(|victim| Self::is_royal_piece(victim.piece_type))
        {
            return SearchMoveTier::Forced;
        }
        let mut setup_probe = None;
        if self
            .piece_at(movement.to)
            .is_some_and(|victim| weights.piece_value(victim.piece_type) >= weights.rook)
            || matches!(move_kind, MoveKind::Branch)
            || self.move_creates_royal_capture_setup_with_probe(movement, weights, &mut setup_probe)
        {
            return SearchMoveTier::Tactical;
        }
        if self.tactical_dependency_order_score(movement, weights) > 0
            || self.quiet_development_order_score(movement, weights) > 0
        {
            return SearchMoveTier::Strategic;
        }
        SearchMoveTier::Quiet
    }

    fn truncate_temporal_dominance(
        &self,
        scored: &mut Vec<(i32, MoveStep)>,
        weights: &EvalWeights,
        inserted_bucket: TemporalDominanceBucket,
        cap: usize,
    ) {
        let mut seen = 0usize;
        scored.retain(|(_, movement)| {
            let Some((piece, move_kind)) = self.legal_move_kind(movement.from, movement.to) else {
                return true;
            };
            let Some((bucket, _)) =
                self.temporal_dominance_bucket(*movement, piece, move_kind, weights)
            else {
                return true;
            };
            if bucket != inserted_bucket {
                return true;
            }
            seen += 1;
            seen <= cap
        });
    }

    fn temporal_dominance_bucket(
        &self,
        movement: MoveStep,
        piece: Piece,
        move_kind: MoveKind,
        weights: &EvalWeights,
    ) -> Option<(TemporalDominanceBucket, usize)> {
        if piece.piece_type != PieceType::Queen
            || movement.from.timeline_id == movement.to.timeline_id
                && movement.from.time == movement.to.time
        {
            return None;
        }
        if self
            .piece_at(movement.to)
            .is_some_and(|victim| Self::is_royal_piece(victim.piece_type))
        {
            return None;
        }
        let intent = self.queen_temporal_intent(movement, piece, move_kind, weights);
        let branch_class = if matches!(move_kind, MoveKind::Branch) {
            TemporalBranchClass::Branch
        } else {
            TemporalBranchClass::SameTimeline
        };
        let cap = match intent {
            QueenTemporalIntent::CaptureHighValue => 16,
            QueenTemporalIntent::CaptureLowValue => 8,
            QueenTemporalIntent::CreateBranch => 8,
            QueenTemporalIntent::DefendRoyal => 8,
            QueenTemporalIntent::RetreatFromAttack => 4,
            QueenTemporalIntent::ImproveMobility => 4,
            QueenTemporalIntent::QuietReposition => 2,
        };
        Some((
            TemporalDominanceBucket {
                source_timeline: movement.from.timeline_id,
                source_time: movement.from.time,
                target_time_cmp: movement.to.time.cmp(&movement.from.time),
                branch_class,
                intent,
            },
            cap,
        ))
    }

    fn queen_temporal_intent(
        &self,
        movement: MoveStep,
        piece: Piece,
        move_kind: MoveKind,
        weights: &EvalWeights,
    ) -> QueenTemporalIntent {
        if let Some(victim) = self.piece_at(movement.to) {
            return if weights.piece_value(victim.piece_type) >= weights.rook {
                QueenTemporalIntent::CaptureHighValue
            } else {
                QueenTemporalIntent::CaptureLowValue
            };
        }
        if matches!(move_kind, MoveKind::Branch) {
            return QueenTemporalIntent::CreateBranch;
        }
        if self
            .royal_piece_positions(piece.color)
            .into_iter()
            .any(|royal| same_board(royal, movement.to) && board_distance(royal, movement.to) <= 2)
        {
            return QueenTemporalIntent::DefendRoyal;
        }
        if self.is_square_attacked(movement.from, piece.color.opposite()) {
            return QueenTemporalIntent::RetreatFromAttack;
        }
        if centrality(movement.to.x, movement.to.y) > centrality(movement.from.x, movement.from.y) {
            return QueenTemporalIntent::ImproveMobility;
        }
        QueenTemporalIntent::QuietReposition
    }

    pub(crate) fn order_moves(&self, moves: Vec<MoveStep>, weights: &EvalWeights) -> Vec<MoveStep> {
        let mut scored: Vec<(i32, MoveStep)> = moves
            .into_iter()
            .map(|movement| (self.cheap_move_order_score(&movement, weights), movement))
            .collect();
        scored.sort_by(|(left_score, left), (right_score, right)| {
            right_score
                .cmp(left_score)
                .then_with(|| Self::move_cmp(left, right))
        });
        scored.into_iter().map(|(_, movement)| movement).collect()
    }

    #[cfg(test)]
    pub(crate) fn piece_candidate_destinations(
        &self,
        from: Position,
        piece: Piece,
    ) -> Vec<Position> {
        let mut targets = Vec::new();
        self.for_each_piece_candidate_destination(from, piece, |target| {
            targets.push(target);
            true
        });
        targets.sort_by_key(|position| position_key(*position));
        targets.dedup();
        targets
    }

    pub(crate) fn for_each_piece_candidate_destination(
        &self,
        from: Position,
        piece: Piece,
        mut visit: impl FnMut(Position) -> bool,
    ) {
        match piece.piece_type {
            PieceType::Pawn => {
                self.visit_pawn_candidates(from, piece.color, false, &mut visit);
            }
            PieceType::Brawn => {
                self.visit_pawn_candidates(from, piece.color, true, &mut visit);
            }
            PieceType::Knight => {
                for long_axis in 0..4 {
                    for short_axis in 0..4 {
                        if long_axis == short_axis {
                            continue;
                        }
                        for long_sign in [-1, 1] {
                            for short_sign in [-1, 1] {
                                let mut offset = [0; 4];
                                offset[long_axis] = long_sign * 2;
                                offset[short_axis] = short_sign;
                                if let Some(target) = self.offset_target(from, offset) {
                                    if !visit(target) {
                                        return;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            PieceType::King | PieceType::CommonKing => {
                if !self.visit_direction_targets(from, 1, 4, 1, &mut visit) {
                    return;
                }
                if piece.piece_type == PieceType::King {
                    for offset in [[2, 0, 0, 0], [-2, 0, 0, 0]] {
                        if let Some(target) = self.offset_target(from, offset) {
                            if !visit(target) {
                                return;
                            }
                        }
                    }
                }
            }
            PieceType::Rook => {
                self.visit_slider_targets(from, 1, 1, &mut visit);
            }
            PieceType::Bishop => {
                self.visit_slider_targets(from, 2, 2, &mut visit);
            }
            PieceType::Unicorn => {
                self.visit_slider_targets(from, 3, 3, &mut visit);
            }
            PieceType::Dragon => {
                self.visit_slider_targets(from, 4, 4, &mut visit);
            }
            PieceType::Princess => {
                self.visit_slider_targets(from, 1, 2, &mut visit);
            }
            PieceType::Queen | PieceType::RoyalQueen => {
                self.visit_slider_targets(from, 1, 4, &mut visit);
            }
        }
    }

    pub(crate) fn visit_pawn_candidates(
        &self,
        from: Position,
        color: Color,
        brawn: bool,
        visit: &mut impl FnMut(Position) -> bool,
    ) -> bool {
        let forward = if color == Color::White { 1 } else { -1 };
        for offset in [
            [0, forward, 0, 0],
            [0, forward * 2, 0, 0],
            [-1, forward, 0, 0],
            [1, forward, 0, 0],
            [0, 0, 0, forward],
            [0, 0, 0, forward * 2],
            [0, 0, -1, forward],
            [0, 0, 1, forward],
        ] {
            if let Some(target) = self.offset_target(from, offset) {
                if !visit(target) {
                    return false;
                }
            }
        }
        if brawn {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dt in -1..=1 {
                        for dl in -1..=1 {
                            let offset = [dx, dy, dt, dl];
                            let changed = offset.iter().filter(|value| **value != 0).count();
                            if changed >= 2
                                && (dy == forward || dl == forward)
                                && dy != -forward
                                && dl != -forward
                            {
                                if let Some(target) = self.offset_target(from, offset) {
                                    if !visit(target) {
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        true
    }

    pub(crate) fn visit_slider_targets(
        &self,
        from: Position,
        min_axes: usize,
        max_axes: usize,
        visit: &mut impl FnMut(Position) -> bool,
    ) -> bool {
        self.visit_direction_targets(from, min_axes, max_axes, self.max_ray_distance(from), visit)
    }

    pub(crate) fn visit_direction_targets(
        &self,
        from: Position,
        min_axes: usize,
        max_axes: usize,
        max_distance: i32,
        visit: &mut impl FnMut(Position) -> bool,
    ) -> bool {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dt in -1..=1 {
                    for dl in -1..=1 {
                        let direction = [dx, dy, dt, dl];
                        let axes = direction.iter().filter(|value| **value != 0).count();
                        if axes < min_axes || axes > max_axes {
                            continue;
                        }
                        for distance in 1..=max_distance {
                            let offset = direction.map(|value| value * distance);
                            let Some(target) = self.offset_target(from, offset) else {
                                break;
                            };
                            if !visit(target) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }

    pub(crate) fn offset_target(
        &self,
        from: Position,
        [dx, dy, dt, dl]: [i32; 4],
    ) -> Option<Position> {
        let x = from.x + dx;
        let y = from.y + dy;
        if !Self::in_bounds(x, y) {
            return None;
        }
        let from_row = self
            .timeline(from.timeline_id)
            .map_or(0, |timeline| timeline.row);
        let timeline = self
            .timelines
            .iter()
            .find(|timeline| timeline.row == from_row + dl)?;
        let time = from.time + dt * 2;
        self.board(timeline.id, time).is_some().then_some(Position {
            timeline_id: timeline.id,
            time,
            x,
            y,
        })
    }

    pub(crate) fn max_ray_distance(&self, from: Position) -> i32 {
        let from_row = self
            .timeline(from.timeline_id)
            .map_or(0, |timeline| timeline.row);
        self.timelines
            .iter()
            .flat_map(|timeline| {
                timeline.boards.iter().map(move |board| {
                    (timeline.row - from_row)
                        .abs()
                        .max((board.time - from.time).abs() / 2)
                })
            })
            .max()
            .unwrap_or(0)
            .max(7)
    }

    pub(crate) fn forcing_moves_until(
        &self,
        weights: &EvalWeights,
        deadline: Option<SearchInstant>,
    ) -> Vec<MoveStep> {
        let mut moves = Vec::new();
        let Some((timeline_id, time)) = self.next_pending_board_key(self.turn) else {
            return moves;
        };
        let mut setup_probe = None;
        for movement in
            self.legal_single_moves_for_board_until(timeline_id, time, weights, deadline)
        {
            if deadline_expired(deadline) {
                break;
            }
            if self.piece_at(movement.to).is_some()
                || movement.from.timeline_id != movement.to.timeline_id
                || movement.from.time != movement.to.time
                || self.move_creates_royal_capture_setup_with_probe(
                    movement,
                    weights,
                    &mut setup_probe,
                )
            {
                moves.push(movement);
            }
            if moves.len() >= MAX_QUIESCENCE_MOVES {
                break;
            }
        }
        moves
    }

    fn move_creates_royal_capture_setup_with_probe(
        &self,
        movement: MoveStep,
        weights: &EvalWeights,
        probe: &mut Option<Game>,
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
        let search = probe.get_or_insert_with(|| self.clone_for_search());
        let Some(undo) = search.make_search_move(movement) else {
            return false;
        };
        let creates_setup = search.royal_capture_available(piece.color)
            || search.temporal_royal_corridor_pressure_for(piece.color, weights)
                > current_corridor_pressure;
        search.unmake_search_move(undo);
        creates_setup
    }

    pub(crate) fn cheap_move_order_score(&self, movement: &MoveStep, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        let is_branch = movement.from.timeline_id != movement.to.timeline_id
            || movement.from.time != movement.to.time;
        if let Some(victim) = self.piece_at(movement.to) {
            let victim_value = weights.piece_value(victim.piece_type);
            score += victim_value * 4;
            if is_branch {
                if let Some(attacker) = self.piece_at(movement.from) {
                    score -= (weights.piece_value(attacker.piece_type) - victim_value).max(0);
                }
            }
            if Self::is_royal_piece(victim.piece_type) {
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
        score += self.tactical_dependency_order_score(*movement, weights);
        score += self.quiet_development_order_score(*movement, weights);
        score
    }

    pub(crate) fn tactical_dependency_order_score(
        &self,
        movement: MoveStep,
        weights: &EvalWeights,
    ) -> i32 {
        let Some(piece) = self.piece_at(movement.from) else {
            return 0;
        };
        let mut score = 0;
        let source_touches_royal_board = self
            .royal_piece_positions(piece.color)
            .into_iter()
            .chain(self.royal_piece_positions(piece.color.opposite()))
            .any(|royal| {
                same_board(royal, movement.from) && board_distance(royal, movement.from) <= 2
            });
        let destination_touches_royal_board = self
            .royal_piece_positions(piece.color)
            .into_iter()
            .chain(self.royal_piece_positions(piece.color.opposite()))
            .any(|royal| same_board(royal, movement.to) && board_distance(royal, movement.to) <= 2);
        if source_touches_royal_board {
            score += weights.royal_shelter.max(1) * 2;
        }
        if destination_touches_royal_board {
            score += weights.royal_threat.max(weights.royal_shelter).max(1) * 2;
        }
        if let Some(victim) = self.piece_at(movement.to) {
            if weights.piece_value(victim.piece_type) >= weights.rook {
                score += weights.fork_pressure.max(1) * 4;
            }
            return score;
        }
        if movement.from.timeline_id != movement.to.timeline_id
            || movement.from.time != movement.to.time
        {
            return score;
        }
        if !source_touches_royal_board && !destination_touches_royal_board {
            score -= weights.piece_activity.max(1);
        }
        score
    }

    pub(crate) fn quiet_development_order_score(
        &self,
        movement: MoveStep,
        weights: &EvalWeights,
    ) -> i32 {
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

    pub(crate) fn submit_turn_for_search(&mut self) -> bool {
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

    fn turn_plan_ends_in_royal_capture(&self, moves: &[MoveStep]) -> bool {
        if moves.is_empty() {
            return false;
        }
        let mut replay = self.clone();
        for movement in moves {
            if replay.apply_move(movement.from, movement.to) == 0 {
                return false;
            }
        }
        replay.submit_turn() == 1
            && replay
                .result
                .is_some_and(|result| result.reason == GameResultReason::RoyalCapture)
    }

    pub(crate) fn apply_move_for_search(&mut self, from: Position, to: Position) -> bool {
        let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
            return false;
        };
        if !self.allows_search_move(from, to, piece, move_kind) {
            return false;
        }

        self.apply_legal_move_for_search(from, to, piece, move_kind);
        true
    }

    fn apply_legal_move_for_search(
        &mut self,
        from: Position,
        to: Position,
        piece: Piece,
        move_kind: MoveKind,
    ) {
        let captured = self.captured_piece(to, move_kind);
        self.record_staged_capture(piece.color, captured);
        self.apply_move_unchecked(from, to, piece, move_kind);
    }

    pub(crate) fn make_search_move(&mut self, movement: MoveStep) -> Option<SearchUndo> {
        let (piece, move_kind) = self.legal_move_kind(movement.from, movement.to)?;
        if !self.allows_search_move(movement.from, movement.to, piece, move_kind) {
            return None;
        }

        let source_board_len = (
            movement.from.timeline_id,
            self.timeline(movement.from.timeline_id)?.boards.len(),
        );
        let target_board_len = if matches!(move_kind, MoveKind::Branch)
            && !self.is_latest_board(movement.to.timeline_id, movement.to.time)
        {
            None
        } else if matches!(move_kind, MoveKind::Branch)
            && movement.to.timeline_id != movement.from.timeline_id
        {
            Some((
                movement.to.timeline_id,
                self.timeline(movement.to.timeline_id)?.boards.len(),
            ))
        } else {
            None
        };
        let undo = SearchUndo {
            timeline_count: self.timelines.len(),
            source_board_len,
            target_board_len,
            next_timeline_id: self.next_timeline_id,
            next_black_timeline_id: self.next_black_timeline_id,
            staged_royal_capture_by: self.staged_royal_capture_by,
            position_hash: self.position_hash,
        };
        self.apply_legal_move_for_search(movement.from, movement.to, piece, move_kind);
        Some(undo)
    }

    pub(crate) fn unmake_search_move(&mut self, undo: SearchUndo) {
        self.timelines.truncate(undo.timeline_count);
        if let Some(timeline) = self.timeline_mut(undo.source_board_len.0) {
            timeline.boards.truncate(undo.source_board_len.1);
        }
        if let Some((timeline_id, board_len)) = undo.target_board_len {
            if let Some(timeline) = self.timeline_mut(timeline_id) {
                timeline.boards.truncate(board_len);
            }
        }
        self.next_timeline_id = undo.next_timeline_id;
        self.next_black_timeline_id = undo.next_black_timeline_id;
        self.staged_royal_capture_by = undo.staged_royal_capture_by;
        self.position_hash = undo.position_hash;
    }

    pub(crate) fn apply_turn_plan_for_search(&self, plan: &TurnPlan) -> Option<Game> {
        let mut game = self.clone_for_search();
        for movement in &plan.moves {
            if !game.apply_move_for_search(movement.from, movement.to) {
                return None;
            }
        }
        game.submit_turn_for_search().then_some(game)
    }

    pub(crate) fn turn_plan_tactical_score_from_result(
        &self,
        result: &Game,
        moves: &[MoveStep],
        color: Color,
        weights: &EvalWeights,
    ) -> i32 {
        let mut score = moves
            .first()
            .map_or(0, |movement| self.cheap_move_order_score(movement, weights));
        if result.is_in_check(color.opposite()) {
            score += weights.check_bonus;
        }
        score
    }

    pub(crate) fn is_forcing_move(
        &self,
        movement: &MoveStep,
        _weights: &EvalWeights,
        _context: &mut SearchContext,
    ) -> bool {
        self.piece_at(movement.to).is_some()
            || movement.from.timeline_id != movement.to.timeline_id
            || movement.from.time != movement.to.time
    }

    pub(crate) fn search_key(&self, maximizing_color: Color) -> u64 {
        let mut hash = self.position_hash;
        hash_combine(&mut hash, color_hash(self.turn));
        hash_combine(&mut hash, color_hash(maximizing_color));
        hash
    }

    pub(crate) fn turn_plan_cache_key(&self) -> u64 {
        self.search_key(self.turn)
    }

    pub(crate) fn turn_plan_cmp(left: &TurnPlan, right: &TurnPlan) -> std::cmp::Ordering {
        left.moves.len().cmp(&right.moves.len()).then_with(|| {
            left.moves
                .iter()
                .zip(&right.moves)
                .map(|(left_move, right_move)| Self::move_cmp(left_move, right_move))
                .find(|ordering| !ordering.is_eq())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub(crate) fn move_cmp(left: &MoveStep, right: &MoveStep) -> std::cmp::Ordering {
        position_key(left.from)
            .cmp(&position_key(right.from))
            .then_with(|| position_key(left.to).cmp(&position_key(right.to)))
    }

    pub(crate) fn child_search_depth(
        &self,
        depth: i32,
        index: usize,
        plan: &TurnPlan,
        child: &Game,
        context: &mut SearchContext,
    ) -> i32 {
        if !context.options.late_move_reduction {
            return depth - 1;
        }

        let mover = self.turn;
        let creates_or_answers_royal_setup = child.royal_capture_available(mover)
            || child.royal_capture_setup_pressure_for_limited(mover, &context.weights, 12) > 0
            || child.temporal_royal_corridor_pressure_for(mover, &context.weights)
                > self.temporal_royal_corridor_pressure_for(mover, &context.weights)
            || child.royal_capture_setup_pressure_for_limited(
                mover.opposite(),
                &context.weights,
                12,
            ) < self.royal_capture_setup_pressure_for_limited(
                mover.opposite(),
                &context.weights,
                12,
            );

        let reduced = depth > 2
            && index >= LATE_MOVE_REDUCTION_AFTER
            && !creates_or_answers_royal_setup
            && plan.moves.iter().all(|movement| {
                movement.from.timeline_id == movement.to.timeline_id
                    && movement.from.time == movement.to.time
                    && child.piece_at(movement.to).is_none()
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

#[derive(Clone, Copy, PartialEq, Eq)]
struct TemporalDominanceBucket {
    source_timeline: i32,
    source_time: i32,
    target_time_cmp: std::cmp::Ordering,
    branch_class: TemporalBranchClass,
    intent: QueenTemporalIntent,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TemporalBranchClass {
    SameTimeline,
    Branch,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QueenTemporalIntent {
    CaptureHighValue,
    CaptureLowValue,
    CreateBranch,
    DefendRoyal,
    RetreatFromAttack,
    ImproveMobility,
    QuietReposition,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchMoveTier {
    Forced,
    Tactical,
    Strategic,
    Quiet,
}

fn push_unique_move(moves: &mut Vec<MoveStep>, movement: MoveStep) {
    if !moves
        .iter()
        .any(|existing| existing.from == movement.from && existing.to == movement.to)
    {
        moves.push(movement);
    }
}

fn same_board(left: Position, right: Position) -> bool {
    left.timeline_id == right.timeline_id && left.time == right.time
}

fn board_distance(left: Position, right: Position) -> i32 {
    (left.x - right.x).abs().max((left.y - right.y).abs())
}
