use super::*;

impl Game {
    pub(crate) fn search_root_staged(
        &self,
        depth: i32,
        context: &mut SearchContext,
        window: Option<(i32, i32)>,
        partition: Option<(usize, usize)>,
    ) -> Option<(TurnPlan, i32)> {
        if context.expired() {
            return None;
        }
        let (alpha, beta) = window.unwrap_or((-CHECKMATE_SCORE * 2, CHECKMATE_SCORE * 2));
        let mut game = self.clone_for_search();
        let mut best = game.search_pending_root(depth, alpha, beta, context, partition, true);
        if let (Some((_, score)), Some((low, high))) = (&best, window) {
            if *score <= low || *score >= high {
                context.stats.aspiration_researches += 1;
                let mut game = self.clone_for_search();
                best = game.search_pending_root(
                    depth,
                    -CHECKMATE_SCORE * 2,
                    CHECKMATE_SCORE * 2,
                    context,
                    partition,
                    true,
                );
            }
        }
        best.map(|(moves, score)| {
            let score_hint = moves.first().map_or(0, |movement| {
                self.cheap_move_order_score(movement, &context.weights)
            });
            (TurnPlan { moves, score_hint }, score)
        })
    }

    pub(crate) fn search_root_staged_with_pv(
        &self,
        depth: i32,
        context: &mut SearchContext,
        window: Option<(i32, i32)>,
        partition: Option<(usize, usize)>,
    ) -> Option<(TurnPlan, i32, Vec<Vec<MoveStep>>)> {
        if context.expired() {
            return None;
        }
        let (alpha, beta) = window.unwrap_or((-CHECKMATE_SCORE * 2, CHECKMATE_SCORE * 2));
        let mut game = self.clone_for_search();
        let mut best =
            game.search_pending_root_with_pv(depth, alpha, beta, context, partition, true);
        if let (Some((_, score, _)), Some((low, high))) = (&best, window) {
            if *score <= low || *score >= high {
                context.stats.aspiration_researches += 1;
                let mut game = self.clone_for_search();
                best = game.search_pending_root_with_pv(
                    depth,
                    -CHECKMATE_SCORE * 2,
                    CHECKMATE_SCORE * 2,
                    context,
                    partition,
                    true,
                );
            }
        }
        best.map(|(moves, score, mut future)| {
            let score_hint = moves.first().map_or(0, |movement| {
                self.cheap_move_order_score(movement, &context.weights)
            });
            let mut principal_variation = vec![moves.clone()];
            principal_variation.append(&mut future);
            (TurnPlan { moves, score_hint }, score, principal_variation)
        })
    }

    fn search_pending_root(
        &mut self,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        context: &mut SearchContext,
        partition: Option<(usize, usize)>,
        root_level: bool,
    ) -> Option<(Vec<MoveStep>, i32)> {
        if let Some(score) = self.terminal_score_until(context.root_color, context.deadline) {
            return Some((Vec::new(), score));
        }
        if !self.has_pending_present_board(self.turn) {
            let original_turn = self.turn;
            if !self.submit_turn_for_search() {
                return None;
            }
            let score =
                self.alpha_beta_in_place(depth - 1, alpha, beta, context.root_color, context);
            self.turn = original_turn;
            return Some((Vec::new(), score));
        }
        if context.exhausted() {
            return None;
        }

        let limit = self.staged_move_limit(context, root_level, depth);
        let generation_limit = self.staged_generation_limit(limit, root_level);
        let mut moves = self.prioritized_turn_moves(self.turn, context, generation_limit);
        if root_level {
            moves = self.root_move_beam(moves, limit, context);
        }
        if root_level {
            if let Some((partition_index, partition_count)) = partition {
                moves = moves
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, movement)| {
                        (index % partition_count == partition_index).then_some(movement)
                    })
                    .collect();
            }
        }
        moves.truncate(context.max_nodes.saturating_sub(context.nodes));
        context.charge_move_generation(moves.len());
        let maximizing = self.turn == context.root_color;
        let use_pvs = self.use_staged_pvs();
        let mut best: Option<(Vec<MoveStep>, i32)> = None;
        for (index, movement) in moves.into_iter().enumerate() {
            if self.should_prune_quiet_temporal_branch(movement, depth, context) {
                continue;
            }
            if !context.charge_move_application() {
                break;
            }
            let Some(undo) = self.make_search_move(movement) else {
                continue;
            };
            let mut result = if use_pvs && index > 0 && alpha + 1 < beta {
                if maximizing {
                    self.search_pending_root(depth, alpha, alpha + 1, context, None, false)
                } else {
                    self.search_pending_root(depth, beta - 1, beta, context, None, false)
                }
            } else {
                self.search_pending_root(depth, alpha, beta, context, None, false)
            };
            if let Some((_, score)) = &result {
                if use_pvs && index > 0 && *score > alpha && *score < beta {
                    result = self.search_pending_root(depth, alpha, beta, context, None, false);
                }
            }
            self.unmake_search_move(undo);
            let Some((suffix, score)) = result else {
                continue;
            };
            let mut line = Vec::with_capacity(suffix.len() + 1);
            line.push(movement);
            line.extend(suffix);
            let score = if root_level {
                self.verified_root_score(&line, score, context)
            } else {
                score
            };
            let replace = best.as_ref().is_none_or(|(best_moves, best_score)| {
                score > *best_score && maximizing
                    || score < *best_score && !maximizing
                    || score == *best_score
                        && Self::move_cmp(&movement, best_moves.first().unwrap_or(&movement))
                            .is_lt()
            });
            if replace {
                best = Some((line, score));
            }
            if maximizing {
                alpha = alpha.max(score);
            } else {
                beta = beta.min(score);
            }
            if alpha >= beta || context.exhausted() {
                context.record_cutoff(depth, Some(movement));
                break;
            }
        }
        best
    }

    fn search_pending_root_with_pv(
        &mut self,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        context: &mut SearchContext,
        partition: Option<(usize, usize)>,
        root_level: bool,
    ) -> Option<(Vec<MoveStep>, i32, Vec<Vec<MoveStep>>)> {
        if let Some(score) = self.terminal_score_until(context.root_color, context.deadline) {
            return Some((Vec::new(), score, Vec::new()));
        }
        if !self.has_pending_present_board(self.turn) {
            let original_turn = self.turn;
            if !self.submit_turn_for_search() {
                return None;
            }
            let (score, principal_variation) =
                self.alpha_beta_line_in_place(depth - 1, alpha, beta, context.root_color, context);
            self.turn = original_turn;
            return Some((Vec::new(), score, principal_variation));
        }
        if context.exhausted() {
            return None;
        }

        let limit = self.staged_move_limit(context, root_level, depth);
        let generation_limit = self.staged_generation_limit(limit, root_level);
        let mut moves = self.prioritized_turn_moves(self.turn, context, generation_limit);
        if root_level {
            moves = self.root_move_beam(moves, limit, context);
        }
        if root_level {
            if let Some((partition_index, partition_count)) = partition {
                moves = moves
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, movement)| {
                        (index % partition_count == partition_index).then_some(movement)
                    })
                    .collect();
            }
        }
        moves.truncate(context.max_nodes.saturating_sub(context.nodes));
        context.charge_move_generation(moves.len());
        let maximizing = self.turn == context.root_color;
        let use_pvs = self.use_staged_pvs();
        let mut best: Option<(Vec<MoveStep>, i32, Vec<Vec<MoveStep>>)> = None;
        for (index, movement) in moves.into_iter().enumerate() {
            if self.should_prune_quiet_temporal_branch(movement, depth, context) {
                continue;
            }
            if !context.charge_move_application() {
                break;
            }
            let Some(undo) = self.make_search_move(movement) else {
                continue;
            };
            let mut result = if use_pvs && index > 0 && alpha + 1 < beta {
                if maximizing {
                    self.search_pending_root(depth, alpha, alpha + 1, context, None, false)
                } else {
                    self.search_pending_root(depth, beta - 1, beta, context, None, false)
                }
                .map(|(suffix, score)| (suffix, score, Vec::new()))
            } else {
                self.search_pending_root_with_pv(depth, alpha, beta, context, None, false)
            };
            if let Some((_, score, _)) = &result {
                if use_pvs && index > 0 && *score > alpha && *score < beta {
                    result =
                        self.search_pending_root_with_pv(depth, alpha, beta, context, None, false);
                }
            }
            self.unmake_search_move(undo);
            let Some((suffix, score, principal_variation)) = result else {
                continue;
            };
            let mut line = Vec::with_capacity(suffix.len() + 1);
            line.push(movement);
            line.extend(suffix);
            let score = if root_level {
                self.verified_root_score(&line, score, context)
            } else {
                score
            };
            let replace = best.as_ref().is_none_or(|(best_moves, best_score, _)| {
                score > *best_score && maximizing
                    || score < *best_score && !maximizing
                    || score == *best_score
                        && Self::move_cmp(&movement, best_moves.first().unwrap_or(&movement))
                            .is_lt()
            });
            if replace {
                best = Some((line, score, principal_variation));
            }
            if maximizing {
                alpha = alpha.max(score);
            } else {
                beta = beta.min(score);
            }
            if alpha >= beta || context.exhausted() {
                context.record_cutoff(depth, Some(movement));
                break;
            }
        }
        best
    }

    pub(crate) fn alpha_beta(
        &self,
        depth: i32,
        alpha: i32,
        beta: i32,
        maximizing_color: Color,
        context: &mut SearchContext,
    ) -> i32 {
        let mut game = self.clone_for_search();
        game.alpha_beta_in_place(depth, alpha, beta, maximizing_color, context)
    }

    fn alpha_beta_in_place(
        &mut self,
        depth: i32,
        alpha: i32,
        beta: i32,
        maximizing_color: Color,
        context: &mut SearchContext,
    ) -> i32 {
        context.nodes += 1;
        if context.exhausted() {
            return context.evaluate(self, maximizing_color);
        }
        if depth <= 0 {
            if self.present_obligation_count(self.turn) >= 4 {
                return context.evaluate(self, maximizing_color);
            }
            return self.quiescence(
                -CHECKMATE_SCORE * 2,
                CHECKMATE_SCORE * 2,
                maximizing_color,
                context,
                context.quiescence_depth(),
            );
        }

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
        let score = self
            .search_pending_score(depth, alpha, beta, maximizing_color, context)
            .unwrap_or_else(|| context.evaluate(self, maximizing_color));
        let bound = if score <= alpha {
            SearchBound::Upper
        } else if score >= beta {
            SearchBound::Lower
        } else {
            SearchBound::Exact
        };
        context.table.insert(
            key,
            SearchEntry {
                depth,
                score,
                bound,
                best_move: None,
            },
        );
        score
    }

    fn alpha_beta_line_in_place(
        &mut self,
        depth: i32,
        alpha: i32,
        beta: i32,
        maximizing_color: Color,
        context: &mut SearchContext,
    ) -> (i32, Vec<Vec<MoveStep>>) {
        context.nodes += 1;
        if context.exhausted() {
            return (context.evaluate(self, maximizing_color), Vec::new());
        }
        if depth <= 0 {
            let score = if self.present_obligation_count(self.turn) >= 4 {
                context.evaluate(self, maximizing_color)
            } else {
                self.quiescence(
                    -CHECKMATE_SCORE * 2,
                    CHECKMATE_SCORE * 2,
                    maximizing_color,
                    context,
                    context.quiescence_depth(),
                )
            };
            return (score, Vec::new());
        }

        let original_alpha = alpha;
        let original_beta = beta;
        let Some((moves, score, mut future)) =
            self.search_pending_root_with_pv(depth, alpha, beta, context, None, false)
        else {
            return (context.evaluate(self, maximizing_color), Vec::new());
        };
        let bound = if score <= original_alpha {
            SearchBound::Upper
        } else if score >= original_beta {
            SearchBound::Lower
        } else {
            SearchBound::Exact
        };
        context.table.insert(
            self.search_key(maximizing_color),
            SearchEntry {
                depth,
                score,
                bound,
                best_move: moves.first().copied(),
            },
        );
        if moves.is_empty() {
            (score, future)
        } else {
            let mut principal_variation = vec![moves];
            principal_variation.append(&mut future);
            (score, principal_variation)
        }
    }

    fn search_pending_score(
        &mut self,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        maximizing_color: Color,
        context: &mut SearchContext,
    ) -> Option<i32> {
        if let Some(score) = self.terminal_score_until(maximizing_color, context.deadline) {
            return Some(score);
        }
        if !self.has_pending_present_board(self.turn) {
            let original_turn = self.turn;
            if !self.submit_turn_for_search() {
                return None;
            }
            let score = self.alpha_beta_in_place(depth - 1, alpha, beta, maximizing_color, context);
            self.turn = original_turn;
            return Some(score);
        }

        let limit = self.staged_move_limit(context, false, depth);
        let mut moves = self.prioritized_turn_moves(self.turn, context, limit);
        moves.truncate(context.max_nodes.saturating_sub(context.nodes));
        context.charge_move_generation(moves.len());
        let maximizing = self.turn == maximizing_color;
        let mut best = if maximizing {
            -CHECKMATE_SCORE * 2
        } else {
            CHECKMATE_SCORE * 2
        };
        let mut found = false;
        let use_pvs = self.use_staged_pvs();
        for (index, movement) in moves.into_iter().enumerate() {
            if self.should_prune_quiet_temporal_branch(movement, depth, context) {
                continue;
            }
            if !context.charge_move_application() {
                break;
            }
            let Some(undo) = self.make_search_move(movement) else {
                continue;
            };
            let mut score = if use_pvs && index > 0 && alpha + 1 < beta {
                if maximizing {
                    self.search_pending_score(depth, alpha, alpha + 1, maximizing_color, context)
                } else {
                    self.search_pending_score(depth, beta - 1, beta, maximizing_color, context)
                }
            } else {
                self.search_pending_score(depth, alpha, beta, maximizing_color, context)
            };
            if let Some(probe_score) = score {
                if use_pvs && index > 0 && probe_score > alpha && probe_score < beta {
                    score =
                        self.search_pending_score(depth, alpha, beta, maximizing_color, context);
                }
            }
            self.unmake_search_move(undo);
            let Some(score) = score else {
                continue;
            };
            found = true;
            if maximizing {
                best = best.max(score);
                alpha = alpha.max(best);
            } else {
                best = best.min(score);
                beta = beta.min(best);
            }
            if alpha >= beta || context.exhausted() {
                context.record_cutoff(depth, Some(movement));
                break;
            }
        }
        found.then_some(best)
    }

    fn staged_move_limit(
        &self,
        context: &mut SearchContext,
        root_level: bool,
        depth: i32,
    ) -> usize {
        let obligations = self.present_obligation_count(self.turn);
        if root_level && obligations >= 4 {
            return context.root_plan_limit().min(2);
        }
        if !root_level {
            match obligations {
                3.. => return 1,
                2 => return context.child_plan_limit().min(3),
                _ => {}
            }
        }

        let mode = self.search_pressure_mode(self.turn, context);
        if root_level {
            return match mode {
                SearchPressureMode::Panic => context.root_plan_limit(),
                SearchPressureMode::Tactical => {
                    context.root_plan_limit().min(depth_move_limit(depth, 12))
                }
                SearchPressureMode::Quiet => {
                    context.root_plan_limit().min(depth_move_limit(depth, 8))
                }
            };
        }

        match mode {
            SearchPressureMode::Panic => context.child_plan_limit(),
            SearchPressureMode::Tactical => {
                context.child_plan_limit().min(depth_move_limit(depth, 6))
            }
            SearchPressureMode::Quiet => context.child_plan_limit().min(depth_move_limit(depth, 3)),
        }
    }

    fn use_staged_pvs(&self) -> bool {
        self.present_obligation_count(self.turn) <= 1
    }

    fn staged_generation_limit(&self, limit: usize, root_level: bool) -> usize {
        if root_level
            && !self.is_in_check(self.turn)
            && self.present_obligation_count(self.turn) < 4
        {
            limit.saturating_mul(4).max(limit)
        } else {
            limit
        }
    }

    fn root_move_beam(
        &self,
        moves: Vec<MoveStep>,
        limit: usize,
        context: &SearchContext,
    ) -> Vec<MoveStep> {
        if moves.len() <= limit
            || self.is_in_check(self.turn)
            || self.present_obligation_count(self.turn) >= 4
        {
            return moves.into_iter().take(limit).collect();
        }

        let mut beam = Vec::with_capacity(limit);
        for (intent, cap) in [
            (RootMoveIntent::RoyalCapture, 8),
            (RootMoveIntent::HighValueCapture, 8),
            (RootMoveIntent::Temporal, 8),
            (RootMoveIntent::RoyalGraph, 8),
            (RootMoveIntent::QuietStrategic, 8),
            (RootMoveIntent::Other, 4),
        ] {
            for movement in moves
                .iter()
                .copied()
                .filter(|movement| self.root_move_intent(*movement, context) == intent)
                .take(cap)
            {
                if beam.len() >= limit {
                    return beam;
                }
                if !beam.contains(&movement) {
                    beam.push(movement);
                }
            }
        }

        for movement in moves {
            if beam.len() >= limit {
                break;
            }
            if !beam.contains(&movement) {
                beam.push(movement);
            }
        }
        beam
    }

    fn root_move_intent(&self, movement: MoveStep, context: &SearchContext) -> RootMoveIntent {
        if self
            .piece_at(movement.to)
            .is_some_and(|piece| Self::is_royal_piece(piece.piece_type))
        {
            return RootMoveIntent::RoyalCapture;
        }
        if self.piece_at(movement.to).is_some_and(|piece| {
            context.weights.piece_value(piece.piece_type) >= context.weights.rook
        }) {
            return RootMoveIntent::HighValueCapture;
        }
        if movement.from.timeline_id != movement.to.timeline_id
            || movement.from.time != movement.to.time
        {
            return RootMoveIntent::Temporal;
        }
        if self.tactical_dependency_order_score(movement, &context.weights) > 0 {
            return RootMoveIntent::RoyalGraph;
        }
        if self.quiet_development_order_score(movement, &context.weights) > 0 {
            return RootMoveIntent::QuietStrategic;
        }
        RootMoveIntent::Other
    }

    fn verified_root_score(&self, line: &[MoveStep], score: i32, context: &SearchContext) -> i32 {
        if self.present_obligation_count(self.turn) >= 4 {
            return score;
        }
        let plan = TurnPlan {
            moves: line.to_vec(),
            score_hint: 0,
        };
        let Some(child) = self.apply_turn_plan_for_search(&plan) else {
            return score;
        };
        let opponent = child.turn;
        let mut refutation_penalty = 0;
        if child.royal_capture_available(opponent) {
            refutation_penalty += CHECKMATE_SCORE / 3;
        }
        if child.royal_capture_setup_pressure_for_limited(opponent, &context.weights, 8) > 0 {
            refutation_penalty += 20_000;
        }
        if !child
            .forcing_moves_until(&context.weights, context.deadline)
            .is_empty()
        {
            refutation_penalty += 4_000;
        }
        if refutation_penalty == 0 {
            return score;
        }
        if self.turn == context.root_color {
            score - refutation_penalty
        } else {
            score + refutation_penalty
        }
    }

    fn search_pressure_mode(
        &self,
        color: Color,
        context: &mut SearchContext,
    ) -> SearchPressureMode {
        let obligations = self.present_obligation_count(color);
        if obligations >= 4 || self.is_in_check(color) {
            return SearchPressureMode::Panic;
        }

        let opponent = color.opposite();
        if self.royal_capture_available(opponent)
            || self.royal_capture_setup_pressure_for_limited(opponent, &context.weights, 8) > 0
        {
            return SearchPressureMode::Panic;
        }

        if obligations >= 2
            || self.royal_capture_available(color)
            || self.royal_capture_setup_pressure_for_limited(color, &context.weights, 8) > 0
        {
            return SearchPressureMode::Tactical;
        }

        SearchPressureMode::Quiet
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchPressureMode {
    Panic,
    Tactical,
    Quiet,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RootMoveIntent {
    RoyalCapture,
    HighValueCapture,
    Temporal,
    RoyalGraph,
    QuietStrategic,
    Other,
}

fn depth_move_limit(depth: i32, quiet_cap: usize) -> usize {
    match depth {
        d if d >= 5 => quiet_cap,
        4 => quiet_cap.saturating_add(2),
        3 => quiet_cap.saturating_add(4),
        2 => quiet_cap.saturating_add(8),
        _ => usize::MAX,
    }
}
