impl Game {
    fn legal_turn_plans_with_context(&self, context: &mut SearchContext) -> Vec<TurnPlan> {
        let cache_key = self.turn_plan_cache_key();
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
            .filter_map(|timeline| {
                timeline
                    .boards
                    .iter()
                    .max_by_key(|board| board.time)
                    .map(|board| (timeline, board))
            })
            .filter(|(timeline, board)| {
                self.is_active_timeline(timeline.id) && board.side_to_move == color
            })
            .count()
            + 1;
        let mut plans = Vec::new();

        for move_limit in [MAX_MOVES_PER_NODE, MAX_MOVES_PER_NODE * 4] {
            if deadline_expired(context.deadline) {
                break;
            }
            let build_context = TurnPlanBuildContext {
                color,
                weights: &context.weights,
                deadline: context.deadline,
                move_limit,
                capture_sanity: context.options.capture_sanity,
            };
            self.collect_turn_plans(max_depth, Vec::new(), &mut plans, &build_context);
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
        plans.truncate(MAX_TURN_PLANS);
        if context.options.turn_plan_cache {
            context.turn_plan_cache.insert(cache_key, plans.clone());
        }
        plans
    }

    fn collect_turn_plans(
        &self,
        depth_left: usize,
        prefix: Vec<MoveStep>,
        plans: &mut Vec<TurnPlan>,
        context: &TurnPlanBuildContext<'_>,
    ) {
        if plans.len() >= MAX_TURN_PLANS || depth_left == 0 || deadline_expired(context.deadline) {
            return;
        }

        // Once the present line belongs to the opponent, this staged prefix is a
        // complete legal turn.
        if !prefix.is_empty() && !self.has_pending_present_board(context.color) {
            let mut submitted = self.clone_for_search();
            if submitted.submit_turn_for_search() {
                let score_hint = submitted.evaluate_fast(context.color, context.weights)
                    + self.turn_plan_tactical_score(&prefix, context.color, context.weights)
                    - prefix.len() as i32;
                plans.push(TurnPlan {
                    moves: prefix,
                    game: submitted,
                    score_hint,
                });
            }
            return;
        }

        let moves = self.prioritized_turn_moves(
            context.color,
            context.weights,
            context.deadline,
            context.move_limit,
        );
        for movement in moves {
            if context.capture_sanity
                && context.move_limit == MAX_MOVES_PER_NODE
                && self.is_likely_bad_capture(movement, context.weights)
            {
                continue;
            }
            let mut next = self.clone_for_search();
            if !next.apply_move_for_search(movement.from, movement.to) {
                continue;
            }
            let mut next_prefix = prefix.clone();
            next_prefix.push(movement);
            next.collect_turn_plans(depth_left - 1, next_prefix, plans, context);
            if plans.len() >= MAX_TURN_PLANS {
                break;
            }
        }
    }

    fn prioritized_turn_moves(
        &self,
        color: Color,
        weights: &EvalWeights,
        deadline: Option<SearchInstant>,
        soft_limit: usize,
    ) -> Vec<MoveStep> {
        let moves = self.legal_single_moves_until(weights, deadline);
        if moves.len() <= soft_limit {
            return moves;
        }

        let mut selected = Vec::new();
        for movement in moves.iter().take(soft_limit) {
            Self::push_unique_move(&mut selected, movement);
        }

        // Search ordering can heavily favor captures, branches, and knights. A
        // whole turn still needs coverage over every playable latest board, so
        // keep a few locally best moves from each board even when they rank below
        // the global cutoff.
        for (timeline_id, time) in self.playable_board_keys(color) {
            let mut added = 0;
            for movement in &moves {
                if movement.from.timeline_id != timeline_id || movement.from.time != time {
                    continue;
                }
                if Self::push_unique_move(&mut selected, movement) {
                    added += 1;
                }
                if added >= REQUIRED_MOVES_PER_BOARD {
                    break;
                }
            }
        }

        selected
    }

    fn apply_search_ordering(&self, plans: &mut [TurnPlan], context: &SearchContext) {
        let key = self.search_key(0, context.root_color);
        let tt_move = context
            .options
            .tt_best_move
            .then(|| context.table.get(&key).and_then(|entry| entry.best_move))
            .flatten();
        plans.sort_by(|left, right| {
            self.plan_order_bonus(right, tt_move, context)
                .cmp(&self.plan_order_bonus(left, tt_move, context))
                .then_with(|| {
                    right
                        .score_hint
                        .cmp(&left.score_hint)
                        .then_with(|| Self::turn_plan_cmp(left, right))
                })
        });
    }

    fn plan_order_bonus(
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

    fn is_likely_bad_capture(&self, movement: MoveStep, weights: &EvalWeights) -> bool {
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
            && self.is_square_attacked(movement.to, attacker.color.opposite())
    }

    fn playable_board_keys(&self, color: Color) -> Vec<(i32, i32)> {
        self.timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| {
                timeline
                    .boards
                    .iter()
                    .max_by_key(|board| board.time)
                    .filter(|board| board.side_to_move == color)
                    .map(|board| (timeline.id, board.time))
            })
            .collect()
    }

    fn push_unique_move(selected: &mut Vec<MoveStep>, movement: &MoveStep) -> bool {
        if selected.iter().any(|existing| {
            position_key(existing.from) == position_key(movement.from)
                && position_key(existing.to) == position_key(movement.to)
        }) {
            return false;
        }
        selected.push(*movement);
        true
    }
}
