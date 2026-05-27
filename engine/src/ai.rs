// Runtime AI search. Training-only mutation/scoring/promotion code lives in
// training.rs so wasm gets a deterministic search surface without file or git
// automation.
const CHECKMATE_SCORE: i32 = 1_000_000;
const MAX_TURN_PLANS: usize = 32;
const MAX_MOVES_PER_NODE: usize = 24;

#[derive(Clone)]
struct MoveStep {
    from: Position,
    to: Position,
}

#[derive(Clone)]
struct TurnPlan {
    moves: Vec<MoveStep>,
    game: Game,
    score_hint: i32,
}

#[derive(Clone)]
struct AiSearchResult {
    moves: Vec<MoveStep>,
    score: i32,
    depth: i32,
    nodes: usize,
    status: &'static str,
}

#[derive(Clone, Copy)]
struct EvalWeights {
    king: i32,
    common_king: i32,
    queen: i32,
    royal_queen: i32,
    princess: i32,
    rook: i32,
    bishop: i32,
    unicorn: i32,
    dragon: i32,
    knight: i32,
    pawn: i32,
    brawn: i32,
    check_penalty: i32,
    active_timeline: i32,
    inactive_timeline: i32,
    present_progress: i32,
    mobility: i32,
    branch_penalty: i32,
    advancement: i32,
    centrality: i32,
}

struct SearchContext {
    // The node budget is shared across iterative-deepening branches.
    weights: EvalWeights,
    root_color: Color,
    max_nodes: usize,
    nodes: usize,
    deadline: Option<std::time::Instant>,
}

impl Game {
    fn ai_turn_json(&self, max_depth: i32, max_nodes: i32) -> String {
        self.best_ai_turn(max_depth, max_nodes).to_json()
    }

    fn best_ai_turn(&self, max_depth: i32, max_nodes: i32) -> AiSearchResult {
        let depth = max_depth.max(1);
        let nodes = max_nodes.max(1) as usize;
        let weights = EvalWeights::default_tuned();
        let mut context = SearchContext {
            weights,
            root_color: self.turn,
            max_nodes: nodes,
            nodes: 0,
            deadline: None,
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
            if context.nodes >= context.max_nodes || score.abs() >= CHECKMATE_SCORE / 2 {
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
        if context.exhausted() || depth <= 0 {
            return self.evaluate(maximizing_color, &context.weights);
        }

        let plans = self.legal_turn_plans_until(&context.weights, context.deadline);
        if plans.is_empty() {
            return self.evaluate(maximizing_color, &context.weights);
        }

        if self.turn == maximizing_color {
            let mut best = -CHECKMATE_SCORE * 2;
            for plan in plans {
                let score =
                    plan.game
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
                let score =
                    plan.game
                        .alpha_beta(depth - 1, alpha, beta, maximizing_color, context);
                best = best.min(score);
                beta = beta.min(best);
                if beta <= alpha || context.exhausted() {
                    break;
                }
            }
            best
        }
    }

    fn legal_turn_plans_until(
        &self,
        weights: &EvalWeights,
        deadline: Option<std::time::Instant>,
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
        deadline: Option<std::time::Instant>,
    ) {
        if plans.len() >= MAX_TURN_PLANS || depth_left == 0 || deadline_expired(deadline) {
            return;
        }

        // Once the present line belongs to the opponent, this staged prefix is a
        // complete legal turn.
        if !prefix.is_empty() && self.present_board().is_some_and(|board| board.side_to_move != color)
        {
            let mut submitted = self.clone_for_search();
            if submitted.submit_turn_for_search() {
                let score_hint = submitted.evaluate(color, weights) - prefix.len() as i32;
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
            next.collect_turn_plans(
                color,
                depth_left - 1,
                next_prefix,
                plans,
                weights,
                deadline,
            );
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
        deadline: Option<std::time::Instant>,
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
                        if !self.piece_at(from).is_some_and(|piece| piece.color == self.turn) {
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
        if let Some(piece) = self.piece_at(movement.to) {
            score += weights.piece_value(piece.piece_type) * 4;
        }
        if movement.from.timeline_id != movement.to.timeline_id || movement.from.time != movement.to.time {
            score -= weights.branch_penalty;
        }
        if self
            .present_board()
            .is_some_and(|board| movement.from.time <= board.time)
        {
            score += weights.present_progress;
        }
        score
    }

    fn evaluate(&self, color: Color, weights: &EvalWeights) -> i32 {
        // Only latest boards contain live material. Historical boards are context
        // for time-travel legality, not extra material to score.
        let mut score = 0;
        for timeline in &self.timelines {
            let active = self.is_active_timeline(timeline.id);
            score += if active {
                weights.active_timeline
            } else {
                weights.inactive_timeline
            } * owner_factor(timeline.owner, color);

            for board in &timeline.boards {
                if !self.is_latest_board(timeline.id, board.time) {
                    continue;
                }
                for (y, rank) in board.board.iter().enumerate() {
                    for (x, piece) in rank.iter().enumerate() {
                        let Some(piece) = piece else {
                            continue;
                        };
                        let value = weights.piece_value(piece.piece_type);
                        let positional = weights.advancement * advancement(piece.color, y as i32)
                            + weights.centrality * centrality(x as i32, y as i32);
                        score += if piece.color == color {
                            value + positional
                        } else {
                            -value - positional
                        };
                    }
                }
            }
        }

        if self.is_in_check(color) {
            score -= weights.check_penalty;
        }
        if self.is_in_check(color.opposite()) {
            score += weights.check_penalty;
        }
        score + self.present_progress(color) * weights.present_progress
            + if weights.mobility == 0 {
                0
            } else {
                self.mobility_balance(color) * weights.mobility
            }
    }

    fn submit_turn_for_search(&mut self) -> bool {
        // Search mirrors user submission but returns a bool rather than writing a
        // user-facing status message.
        let Some(present_side) = self.present_board().map(|board| board.side_to_move) else {
            return false;
        };
        if present_side == self.turn || self.is_in_check(self.turn) {
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

    fn present_progress(&self, color: Color) -> i32 {
        let Some(present) = self.present_board() else {
            return 0;
        };
        let latest_sum: i32 = self
            .timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| self.latest_time(timeline.id))
            .sum();
        let factor = if present.side_to_move == color { 1 } else { -1 };
        // Reward advancing active timelines while this color controls the
        // present line; penalize the same advance when it hands tempo away.
        factor * (latest_sum - present.time)
    }

    fn mobility_balance(&self, color: Color) -> i32 {
        let mut own = self.clone_for_search();
        own.turn = color;
        let mut opponent = self.clone_for_search();
        opponent.turn = color.opposite();
        own.legal_single_moves(&EvalWeights::default_tuned()).len() as i32
            - opponent
                .legal_single_moves(&EvalWeights::default_tuned())
                .len() as i32
    }

    fn turn_plan_cmp(left: &TurnPlan, right: &TurnPlan) -> std::cmp::Ordering {
        left.moves
            .len()
            .cmp(&right.moves.len())
            .then_with(|| {
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

fn deadline_expired(deadline: Option<std::time::Instant>) -> bool {
    deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline)
}

impl AiSearchResult {
    fn to_json(&self) -> String {
        format!(
            "{{\"moves\":[{}],\"score\":{},\"depth\":{},\"nodes\":{},\"status\":\"{}\"}}",
            self.moves
                .iter()
                .map(move_step_json)
                .collect::<Vec<_>>()
                .join(","),
            self.score,
            self.depth,
            self.nodes,
            self.status
        )
    }
}

impl EvalWeights {
    fn default_tuned() -> Self {
        // Committed training data lives in a dedicated include target so the
        // trainer never edits this type definition.
        include!("ai_parameters.rs")
    }

    fn piece_value(self, piece_type: PieceType) -> i32 {
        match piece_type {
            PieceType::King => self.king,
            PieceType::CommonKing => self.common_king,
            PieceType::Queen => self.queen,
            PieceType::RoyalQueen => self.royal_queen,
            PieceType::Princess => self.princess,
            PieceType::Rook => self.rook,
            PieceType::Bishop => self.bishop,
            PieceType::Unicorn => self.unicorn,
            PieceType::Dragon => self.dragon,
            PieceType::Knight => self.knight,
            PieceType::Pawn => self.pawn,
            PieceType::Brawn => self.brawn,
        }
    }
}

fn owner_factor(owner: TimelineOwner, color: Color) -> i32 {
    match owner {
        TimelineOwner::Neutral => 0,
        TimelineOwner::White => {
            if color == Color::White {
                1
            } else {
                -1
            }
        }
        TimelineOwner::Black => {
            if color == Color::Black {
                1
            } else {
                -1
            }
        }
    }
}

fn advancement(color: Color, y: i32) -> i32 {
    match color {
        Color::White => y,
        Color::Black => 7 - y,
    }
}

fn centrality(x: i32, y: i32) -> i32 {
    14 - ((2 * x - 7).abs() + (2 * y - 7).abs())
}

fn position_key(position: Position) -> (i32, i32, i32, i32) {
    (position.timeline_id, position.time, position.y, position.x)
}

fn move_step_json(step: &MoveStep) -> String {
    format!(
        "{{\"from\":{},\"to\":{}}}",
        position_json(step.from),
        position_json(step.to)
    )
}
