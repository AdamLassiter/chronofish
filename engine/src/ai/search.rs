impl Game {
    fn ai_turn_json(&self, max_depth: i32, max_nodes: i32) -> String {
        self.best_ai_turn(max_depth, max_nodes, None).to_json()
    }

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
        let depth = max_depth.max(1);
        let nodes = max_nodes.max(1) as usize;
        let weights = EvalWeights::default_tuned();
        let mut context = SearchContext {
            weights,
            root_color: self.turn,
            max_nodes: nodes,
            nodes: 0,
            deadline,
            table: std::collections::HashMap::new(),
        };
        let mut best = AiSearchResult {
            moves: Vec::new(),
            score: 0,
            depth: 0,
            nodes: 0,
            status: "noLegalTurn",
        };

        // Iterative deepening preserves a usable shallower answer if the node
        // limit is hit before the requested depth completes.
        for current_depth in 1..=depth {
            let Some((plan, score)) = self.search_root(current_depth, &mut context) else {
                break;
            };
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

    fn search_root(&self, depth: i32, context: &mut SearchContext) -> Option<(TurnPlan, i32)> {
        if context.expired() {
            return None;
        }

        let plans = self.legal_turn_plans_until(&context.weights, context.deadline);
        let mut best: Option<(TurnPlan, i32)> = None;
        let mut alpha = -CHECKMATE_SCORE * 2;
        let beta = CHECKMATE_SCORE * 2;

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
        if context.exhausted() {
            return self.evaluate(maximizing_color, &context.weights);
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
                return entry.score;
            }
        }

        let plans = self.legal_turn_plans_until(&context.weights, context.deadline);
        if plans.is_empty() {
            return self.evaluate(maximizing_color, &context.weights);
        }

        let result = if self.turn == maximizing_color {
            let mut best = -CHECKMATE_SCORE * 2;
            for plan in plans {
                let score = plan
                    .game
                    .alpha_beta(depth - 1, alpha, beta, maximizing_color, context);
                best = best.max(score);
                alpha = alpha.max(best);
                if beta <= alpha || context.exhausted() {
                    break;
                }
            }
            best
        } else {
            let mut best = CHECKMATE_SCORE * 2;
            for plan in plans {
                let score = plan
                    .game
                    .alpha_beta(depth - 1, alpha, beta, maximizing_color, context);
                best = best.min(score);
                beta = beta.min(best);
                if beta <= alpha || context.exhausted() {
                    break;
                }
            }
            best
        };

        context.table.insert(
            key,
            SearchEntry {
                depth,
                score: result,
            },
        );
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
        let stand_pat = self.evaluate(maximizing_color, &context.weights);
        if depth <= 0 || context.exhausted() {
            return stand_pat;
        }
        let moves: Vec<MoveStep> = self
            .legal_single_moves_until(&context.weights, context.deadline)
            .into_iter()
            .filter(|movement| self.is_forcing_move(movement, &context.weights))
            .take(6)
            .collect();

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

    fn legal_turn_plans_until(
        &self,
        weights: &EvalWeights,
        deadline: Option<SearchInstant>,
    ) -> Vec<TurnPlan> {
        let color = self.turn;
        // A side may need to move on several active timelines before the present
        // line flips. Cap that expansion to keep browser AI responsive.
        let max_depth = (self
            .timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .count()
            + 1)
        .min(4);
        let mut plans = Vec::new();
        self.collect_turn_plans(color, max_depth, Vec::new(), &mut plans, weights, deadline);
        plans.sort_by(|left, right| {
            right
                .score_hint
                .cmp(&left.score_hint)
                .then_with(|| Self::turn_plan_cmp(left, right))
        });
        plans.truncate(MAX_TURN_PLANS);
        plans
    }

    fn collect_turn_plans(
        &self,
        color: Color,
        depth_left: usize,
        prefix: Vec<MoveStep>,
        plans: &mut Vec<TurnPlan>,
        weights: &EvalWeights,
        deadline: Option<SearchInstant>,
    ) {
        if plans.len() >= MAX_TURN_PLANS || depth_left == 0 || deadline_expired(deadline) {
            return;
        }

        // Once the present line belongs to the opponent, this staged prefix is a
        // complete legal turn.
        if !prefix.is_empty()
            && self
                .present_board()
                .is_some_and(|board| board.side_to_move != color)
        {
            let mut submitted = self.clone_for_search();
            if submitted.submit_turn_for_search() {
                let score_hint = submitted.evaluate(color, weights)
                    + self.turn_plan_tactical_score(&prefix, color, weights)
                    - prefix.len() as i32;
                plans.push(TurnPlan {
                    moves: prefix,
                    game: submitted,
                    score_hint,
                });
            }
            return;
        }

        let mut moves = self.legal_single_moves_until(weights, deadline);
        moves.truncate(MAX_MOVES_PER_NODE);
        for movement in moves {
            let mut next = self.clone_for_search();
            if !next.apply_move_for_search(movement.from, movement.to) {
                continue;
            }
            let mut next_prefix = prefix.clone();
            next_prefix.push(movement);
            next.collect_turn_plans(color, depth_left - 1, next_prefix, plans, weights, deadline);
            if plans.len() >= MAX_TURN_PLANS {
                break;
            }
        }
    }

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
                                        if self.can_move_to(from, to) {
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

        // Order likely tactical/progress moves first to improve alpha-beta
        // pruning. The sort does not change legality.
        moves.sort_by(|left, right| {
            self.move_order_score(right, weights)
                .cmp(&self.move_order_score(left, weights))
                .then_with(|| Self::move_cmp(left, right))
        });
        moves
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
        let should_probe_tactics = captures_piece || is_branch;
        if should_probe_tactics {
            if let Some(piece) = self.piece_at(movement.from) {
                let mut next = self.clone_for_search();
                if next.apply_move_for_search(movement.from, movement.to) {
                    if next.royal_capture_available(piece.color) {
                        score += CHECKMATE_SCORE / 2;
                    }
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

    fn submit_turn_for_search(&mut self) -> bool {
        // Search mirrors user submission but returns a bool rather than writing a
        // user-facing status message.
        let Some(present_side) = self.present_board().map(|board| board.side_to_move) else {
            return false;
        };
        if present_side == self.turn {
            return false;
        }
        self.turn = present_side;
        self.staged_turn.clear();
        true
    }

    fn apply_move_for_search(&mut self, from: Position, to: Position) -> bool {
        let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
            return false;
        };

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

    fn is_forcing_move(&self, movement: &MoveStep, weights: &EvalWeights) -> bool {
        self.piece_at(movement.to).is_some()
            || movement.from.timeline_id != movement.to.timeline_id
            || movement.from.time != movement.to.time
            || self.move_order_score(movement, weights) >= weights.check_bonus
    }

    fn search_key(&self, depth: i32, maximizing_color: Color) -> String {
        let mut parts = vec![format!(
            "d{depth}:t{}:m{}",
            self.turn.as_str(),
            maximizing_color.as_str()
        )];
        for timeline in &self.timelines {
            let Some(board) = timeline.boards.iter().max_by_key(|board| board.time) else {
                continue;
            };
            parts.push(format!(
                "L{}@{}:{}",
                timeline.id,
                board.time,
                board.side_to_move.as_str()
            ));
            for y in 0..8 {
                for x in 0..8 {
                    if let Some(piece) = board.board[y][x] {
                        parts.push(format!(
                            "{}{}{}{}",
                            x,
                            y,
                            piece.color.as_str().as_bytes()[0] as char,
                            piece.piece_type.as_str()
                        ));
                    }
                }
            }
        }
        parts.join("|")
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
}

impl SearchContext {
    fn expired(&self) -> bool {
        deadline_expired(self.deadline)
    }

    fn exhausted(&self) -> bool {
        self.nodes >= self.max_nodes || self.expired()
    }
}

fn deadline_expired(deadline: Option<SearchInstant>) -> bool {
    deadline.is_some_and(|deadline| SearchInstant::now() >= deadline)
}

fn search_deadline(millis: i32) -> Option<SearchInstant> {
    (millis > 0)
        .then(|| SearchInstant::now() + std::time::Duration::from_millis(millis as u64))
}
