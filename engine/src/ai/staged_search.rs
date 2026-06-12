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

        let limit = self.staged_move_limit(context, root_level);
        let mut moves =
            self.prioritized_turn_moves(self.turn, &context.weights, context.deadline, limit);
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
        let mut best: Option<(Vec<MoveStep>, i32)> = None;
        for movement in moves {
            if !context.charge_move_application() {
                break;
            }
            let Some(undo) = self.make_search_move(movement) else {
                continue;
            };
            let result = self.search_pending_root(depth, alpha, beta, context, None, false);
            self.unmake_search_move(undo);
            let Some((suffix, score)) = result else {
                continue;
            };
            let replace = best.as_ref().is_none_or(|(best_moves, best_score)| {
                score > *best_score && maximizing
                    || score < *best_score && !maximizing
                    || score == *best_score
                        && Self::move_cmp(&movement, best_moves.first().unwrap_or(&movement))
                            .is_lt()
            });
            if replace {
                let mut line = Vec::with_capacity(suffix.len() + 1);
                line.push(movement);
                line.extend(suffix);
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

        let limit = self.staged_move_limit(context, root_level);
        let mut moves =
            self.prioritized_turn_moves(self.turn, &context.weights, context.deadline, limit);
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
        let mut best: Option<(Vec<MoveStep>, i32, Vec<Vec<MoveStep>>)> = None;
        for movement in moves {
            if !context.charge_move_application() {
                break;
            }
            let Some(undo) = self.make_search_move(movement) else {
                continue;
            };
            let result = self.search_pending_root_with_pv(depth, alpha, beta, context, None, false);
            self.unmake_search_move(undo);
            let Some((suffix, score, principal_variation)) = result else {
                continue;
            };
            let replace = best.as_ref().is_none_or(|(best_moves, best_score, _)| {
                score > *best_score && maximizing
                    || score < *best_score && !maximizing
                    || score == *best_score
                        && Self::move_cmp(&movement, best_moves.first().unwrap_or(&movement))
                            .is_lt()
            });
            if replace {
                let mut line = Vec::with_capacity(suffix.len() + 1);
                line.push(movement);
                line.extend(suffix);
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
        if let Some(score) = self.terminal_score_until(maximizing_color, context.deadline) {
            return score;
        }
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
        if let Some(score) = self.terminal_score_until(maximizing_color, context.deadline) {
            return (score, Vec::new());
        }
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

        let limit = self.staged_move_limit(context, false);
        let mut moves =
            self.prioritized_turn_moves(self.turn, &context.weights, context.deadline, limit);
        moves.truncate(context.max_nodes.saturating_sub(context.nodes));
        context.charge_move_generation(moves.len());
        let maximizing = self.turn == maximizing_color;
        let mut best = if maximizing {
            -CHECKMATE_SCORE * 2
        } else {
            CHECKMATE_SCORE * 2
        };
        let mut found = false;
        for movement in moves {
            if !context.charge_move_application() {
                break;
            }
            let Some(undo) = self.make_search_move(movement) else {
                continue;
            };
            let score = self.search_pending_score(depth, alpha, beta, maximizing_color, context);
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

    fn staged_move_limit(&self, context: &SearchContext, root_level: bool) -> usize {
        if root_level {
            return if self.present_obligation_count(self.turn) >= 4 {
                context.root_plan_limit().min(2)
            } else {
                context.root_plan_limit()
            };
        }

        match self.present_obligation_count(self.turn) {
            3.. => 1,
            2 => context.child_plan_limit().min(3),
            _ => context.child_plan_limit(),
        }
    }
}
