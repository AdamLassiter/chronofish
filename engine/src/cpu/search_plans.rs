use super::*;

impl Game {
    pub(crate) fn legal_turn_plans_with_context(
        &self,
        context: &mut SearchContext,
        plan_limit: usize,
    ) -> Vec<TurnPlan> {
        let cache_key = self.turn_plan_cache_key() ^ mix64(plan_limit as u64);
        if context.options.turn_plan_cache {
            if let Some(plans) = context.turn_plan_cache.get(&cache_key) {
                context.stats.turn_plan_cache_hits += 1;
                return plans.clone();
            }
        }

        let color = self.turn;
        // A side may need to move once on every active latest board before the
        // present line flips. The old fixed cap made the bot report no legal
        // turn as soon as the multiverse grew past four active boards.
        let max_depth = self
            .timelines
            .iter()
            .filter_map(|timeline| timeline.boards.last().map(|board| (timeline, board)))
            .filter(|(timeline, board)| {
                self.is_active_timeline(timeline.id) && board.side_to_move == color
            })
            .count()
            + 1;
        let mut plans = Vec::new();
        if !context.charge_clone() {
            return plans;
        }
        let mut working = self.clone_for_search();
        let mut prefix = Vec::new();

        for move_limit in [MAX_MOVES_PER_NODE, MAX_MOVES_PER_NODE * 4] {
            if deadline_expired(context.deadline) {
                break;
            }
            working.collect_turn_plans(
                max_depth,
                &mut prefix,
                &mut plans,
                context,
                move_limit,
                plan_limit,
            );
            if !plans.is_empty() {
                break;
            }
        }

        plans.sort_by(|left, right| {
            right
                .score_hint
                .cmp(&left.score_hint)
                .then_with(|| Self::turn_plan_cmp(left, right))
        });
        self.apply_search_ordering(&mut plans, context);
        plans.truncate(plan_limit);
        if context.options.turn_plan_cache {
            context.turn_plan_cache.insert(cache_key, plans.clone());
        }
        plans
    }

    pub(crate) fn collect_turn_plans(
        &mut self,
        depth_left: usize,
        prefix: &mut Vec<MoveStep>,
        plans: &mut Vec<TurnPlan>,
        context: &mut SearchContext,
        move_limit: usize,
        plan_limit: usize,
    ) {
        if plans.len() >= plan_limit || depth_left == 0 || context.exhausted() {
            return;
        }

        // Once the present line belongs to the opponent, this staged prefix is a
        // complete legal turn.
        let color = self.turn;
        if !prefix.is_empty() && !self.has_pending_present_board(color) {
            if self.staged_royal_capture_by == Some(color.opposite()) {
                return;
            }
            let score_hint =
                self.turn_plan_tactical_score_from_result(self, prefix, color, &context.weights)
                    - prefix.len() as i32;
            plans.push(TurnPlan {
                moves: prefix.clone(),
                score_hint,
            });
            context.record_generated_plan();
            return;
        }

        let mut moves = self.prioritized_turn_moves(color, context, move_limit);
        moves.truncate(context.max_nodes.saturating_sub(context.nodes));
        context.charge_move_generation(moves.len());
        let weights = context.weights;
        for movement in moves {
            if context.exhausted() {
                break;
            }
            if self.should_prune_quiet_temporal_branch(movement, depth_left as i32, context) {
                continue;
            }
            if context.options.capture_sanity
                && move_limit == MAX_MOVES_PER_NODE
                && self.is_likely_bad_capture_with_context(movement, &weights, context)
            {
                continue;
            }
            let Some(undo) = self.make_search_move(movement) else {
                continue;
            };
            prefix.push(movement);
            self.collect_turn_plans(
                depth_left - 1,
                prefix,
                plans,
                context,
                move_limit,
                plan_limit,
            );
            prefix.pop();
            self.unmake_search_move(undo);
            if plans.len() >= plan_limit {
                break;
            }
        }
    }

    pub(crate) fn prioritized_turn_moves(
        &self,
        color: Color,
        context: &mut SearchContext,
        soft_limit: usize,
    ) -> Vec<MoveStep> {
        let Some((timeline_id, time)) = self.next_pending_board_key(color) else {
            return Vec::new();
        };
        let moves = self.legal_single_moves_for_board_limited_until(
            timeline_id,
            time,
            context,
            soft_limit.max(1),
        );
        if context.is_in_check_cached(self, color) {
            let evasions = self.check_evasion_moves(
                timeline_id,
                time,
                self.legal_single_moves_for_board_until(
                    timeline_id,
                    time,
                    &context.weights,
                    context.deadline,
                ),
                color,
            );
            if !evasions.is_empty() {
                return evasions;
            }
            return self.legal_single_moves_for_board_until(
                timeline_id,
                time,
                &context.weights,
                context.deadline,
            );
        }
        if moves.is_empty() {
            return self
                .legal_single_moves_for_board_until(
                    timeline_id,
                    time,
                    &context.weights,
                    context.deadline,
                )
                .into_iter()
                .take(soft_limit)
                .collect();
        }
        moves
    }

    fn check_evasion_moves(
        &self,
        _timeline_id: i32,
        _time: i32,
        moves: Vec<MoveStep>,
        color: Color,
    ) -> Vec<MoveStep> {
        moves
            .into_iter()
            .filter(|movement| {
                let mut probe = self.clone_for_search();
                let Some(undo) = probe.make_search_move(*movement) else {
                    return false;
                };
                let escaped = !probe.is_in_check(color);
                probe.unmake_search_move(undo);
                escaped
            })
            .collect()
    }

    pub(crate) fn prioritized_turn_moves_for_evaluation(
        &self,
        color: Color,
        weights: &EvalWeights,
        deadline: Option<SearchInstant>,
        soft_limit: usize,
    ) -> Vec<MoveStep> {
        let Some((timeline_id, time)) = self.next_pending_board_key(color) else {
            return Vec::new();
        };
        let moves = self.legal_single_moves_for_board_until(timeline_id, time, weights, deadline);
        if self.is_in_check(color) {
            return moves;
        }
        moves.into_iter().take(soft_limit).collect()
    }

    pub(crate) fn apply_search_ordering(&self, plans: &mut [TurnPlan], context: &SearchContext) {
        let key = self.search_key(context.root_color);
        let tt_move = context
            .options
            .tt_best_move
            .then(|| context.table.get(&key).and_then(|entry| entry.best_move))
            .flatten();
        let mut order = plans
            .iter()
            .enumerate()
            .map(|(index, plan)| (self.plan_order_bonus(plan, tt_move, context), index))
            .collect::<Vec<_>>();
        order.sort_by(|(left_bonus, left_index), (right_bonus, right_index)| {
            right_bonus.cmp(left_bonus).then_with(|| {
                plans[*right_index]
                    .score_hint
                    .cmp(&plans[*left_index].score_hint)
                    .then_with(|| Self::turn_plan_cmp(&plans[*left_index], &plans[*right_index]))
            })
        });
        let mut target_by_original = vec![0; order.len()];
        for (target_index, (_, original_index)) in order.into_iter().enumerate() {
            target_by_original[original_index] = target_index;
        }
        for original_index in 0..target_by_original.len() {
            while target_by_original[original_index] != original_index {
                let target_index = target_by_original[original_index];
                plans.swap(original_index, target_index);
                target_by_original.swap(original_index, target_index);
            }
        }
    }

    pub(crate) fn plan_order_bonus(
        &self,
        plan: &TurnPlan,
        tt_move: Option<MoveStep>,
        context: &SearchContext,
    ) -> i32 {
        let Some(first) = plan.moves.first().copied() else {
            return 0;
        };
        let mut bonus = 0;
        if context.options.tt_best_move && tt_move == Some(first) {
            bonus += CHECKMATE_SCORE / 8;
        }
        if context.options.killer_moves {
            for killers in &context.killers {
                if killers.contains(&Some(first)) {
                    bonus += 8_000;
                    break;
                }
            }
        }
        if context.options.history_heuristic {
            bonus += context.history.get(&move_hash(first)).copied().unwrap_or(0);
        }
        bonus
    }

    pub(crate) fn is_likely_bad_capture_with_context(
        &self,
        movement: MoveStep,
        weights: &EvalWeights,
        context: &mut SearchContext,
    ) -> bool {
        let Some(attacker) = self.piece_at(movement.from) else {
            return false;
        };
        let Some(victim) = self.piece_at(movement.to) else {
            return false;
        };
        if Self::is_royal_piece(victim.piece_type) {
            return false;
        }
        weights.piece_value(attacker.piece_type) > weights.piece_value(victim.piece_type) * 2
            && context.is_square_attacked_cached(self, movement.to, attacker.color.opposite())
    }

    pub(crate) fn should_prune_quiet_temporal_branch(
        &self,
        movement: MoveStep,
        depth: i32,
        context: &mut SearchContext,
    ) -> bool {
        if depth <= 1 {
            return false;
        }
        if movement.from.timeline_id == movement.to.timeline_id
            && movement.from.time == movement.to.time
        {
            return false;
        }
        if self.piece_at(movement.to).is_some() {
            return false;
        }
        let Some(piece) = self.piece_at(movement.from) else {
            return false;
        };
        if Self::is_royal_piece(piece.piece_type) {
            return false;
        }
        if self.present_time() == Some(movement.to.time) {
            return false;
        }
        if context.move_is_search_suggested(movement, self.search_key(context.root_color)) {
            return false;
        }
        if context.is_in_check_cached(self, self.turn) {
            return false;
        }
        self.cheap_move_order_score(&movement, &context.weights) <= 0
    }

    pub(crate) fn playable_board_keys(&self, color: Color) -> Vec<(i32, i32)> {
        self.timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| {
                timeline
                    .boards
                    .last()
                    .filter(|board| board.side_to_move == color)
                    .map(|board| (timeline.id, board.time))
            })
            .collect()
    }

    pub(crate) fn next_pending_board_key(&self, color: Color) -> Option<(i32, i32)> {
        let present_time = self.present_time()?;
        let checked = self
            .royal_piece_positions(color)
            .into_iter()
            .find(|position| {
                position.time == present_time
                    && self.is_active_timeline(position.timeline_id)
                    && self
                        .board(position.timeline_id, position.time)
                        .is_some_and(|board| board.side_to_move == color)
                    && self.is_square_attacked(*position, color.opposite())
            });
        if let Some(position) = checked {
            return Some((position.timeline_id, position.time));
        }

        self.playable_board_keys(color)
            .into_iter()
            .filter(|(_, time)| *time == present_time)
            .min()
    }
}
