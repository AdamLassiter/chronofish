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
    weights: EvalWeights,
    root_color: Color,
    max_nodes: usize,
    nodes: usize,
}

struct TrainerConfig {
    generations: usize,
    population: usize,
    depth: i32,
    nodes: usize,
    plies: usize,
    seed: u64,
    out: Option<String>,
    score: Option<String>,
    score_default: bool,
}

#[derive(Clone)]
struct Lcg {
    state: u64,
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
        };
        let mut best = AiSearchResult {
            moves: Vec::new(),
            score: 0,
            depth: 0,
            nodes: 0,
            status: "noLegalTurn",
        };

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
        let plans = self.legal_turn_plans(&context.weights);
        let mut best: Option<(TurnPlan, i32)> = None;
        let mut alpha = -CHECKMATE_SCORE * 2;
        let beta = CHECKMATE_SCORE * 2;

        for plan in plans {
            if context.nodes >= context.max_nodes {
                break;
            }
            let score = plan
                .game
                .alpha_beta(depth - 1, alpha, beta, context.root_color, context);
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
        context.nodes += 1;
        if context.nodes >= context.max_nodes || depth <= 0 {
            return self.evaluate(maximizing_color, &context.weights);
        }

        let plans = self.legal_turn_plans(&context.weights);
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
                if beta <= alpha || context.nodes >= context.max_nodes {
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
                if beta <= alpha || context.nodes >= context.max_nodes {
                    break;
                }
            }
            best
        }
    }

    fn legal_turn_plans(&self, weights: &EvalWeights) -> Vec<TurnPlan> {
        let color = self.turn;
        let max_depth = (self
            .timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .count()
            + 1)
            .min(4);
        let mut plans = Vec::new();
        self.collect_turn_plans(color, max_depth, Vec::new(), &mut plans, weights);
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
    ) {
        if plans.len() >= MAX_TURN_PLANS || depth_left == 0 {
            return;
        }

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

        let mut moves = self.legal_single_moves(weights);
        moves.truncate(MAX_MOVES_PER_NODE);
        for movement in moves {
            let mut next = self.clone_for_search();
            if !next.apply_move_for_search(movement.from, movement.to) {
                continue;
            }
            let mut next_prefix = prefix.clone();
            next_prefix.push(movement);
            next.collect_turn_plans(color, depth_left - 1, next_prefix, plans, weights);
            if plans.len() >= MAX_TURN_PLANS {
                break;
            }
        }
    }

    fn legal_single_moves(&self, weights: &EvalWeights) -> Vec<MoveStep> {
        let mut moves = Vec::new();
        for timeline in &self.timelines {
            for board in &timeline.boards {
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
        Self {
            king: 20_000,
            common_king: 200,
            queen: 1_108,
            royal_queen: 20_500,
            princess: 894,
            rook: 508,
            bishop: 353,
            unicorn: 486,
            dragon: 432,
            knight: 430,
            pawn: 71,
            brawn: 143,
            check_penalty: 467,
            active_timeline: 53,
            inactive_timeline: -42,
            present_progress: 22,
            mobility: 0,
            branch_penalty: 6,
            advancement: 1,
            centrality: 8,
        }
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

    fn mutate(self, rng: &mut Lcg) -> Self {
        Self {
            king: self.king,
            common_king: mutate_weight(self.common_king, rng, 80, 0, 2_000),
            queen: mutate_weight(self.queen, rng, 160, 100, 3_000),
            royal_queen: self.royal_queen,
            princess: mutate_weight(self.princess, rng, 140, 100, 3_000),
            rook: mutate_weight(self.rook, rng, 100, 100, 2_000),
            bishop: mutate_weight(self.bishop, rng, 80, 50, 2_000),
            unicorn: mutate_weight(self.unicorn, rng, 90, 50, 2_000),
            dragon: mutate_weight(self.dragon, rng, 120, 50, 2_000),
            knight: mutate_weight(self.knight, rng, 80, 50, 2_000),
            pawn: mutate_weight(self.pawn, rng, 30, 10, 600),
            brawn: mutate_weight(self.brawn, rng, 30, 10, 600),
            check_penalty: mutate_weight(self.check_penalty, rng, 120, 0, 3_000),
            active_timeline: mutate_weight(self.active_timeline, rng, 20, -500, 500),
            inactive_timeline: mutate_weight(self.inactive_timeline, rng, 20, -500, 500),
            present_progress: mutate_weight(self.present_progress, rng, 10, 0, 200),
            mobility: mutate_weight(self.mobility, rng, 4, 0, 80),
            branch_penalty: mutate_weight(self.branch_penalty, rng, 10, 0, 300),
            advancement: mutate_weight(self.advancement, rng, 4, 0, 80),
            centrality: mutate_weight(self.centrality, rng, 4, 0, 80),
        }
    }

    fn crossover(left: Self, right: Self, rng: &mut Lcg) -> Self {
        macro_rules! pick {
            ($field:ident) => {
                if rng.next_bool() {
                    left.$field
                } else {
                    right.$field
                }
            };
        }
        Self {
            king: left.king,
            common_king: pick!(common_king),
            queen: pick!(queen),
            royal_queen: left.royal_queen,
            princess: pick!(princess),
            rook: pick!(rook),
            bishop: pick!(bishop),
            unicorn: pick!(unicorn),
            dragon: pick!(dragon),
            knight: pick!(knight),
            pawn: pick!(pawn),
            brawn: pick!(brawn),
            check_penalty: pick!(check_penalty),
            active_timeline: pick!(active_timeline),
            inactive_timeline: pick!(inactive_timeline),
            present_progress: pick!(present_progress),
            mobility: pick!(mobility),
            branch_penalty: pick!(branch_penalty),
            advancement: pick!(advancement),
            centrality: pick!(centrality),
        }
    }

    fn to_json(self) -> String {
        format!(
            "{{\"king\":{},\"commonKing\":{},\"queen\":{},\"royalQueen\":{},\"princess\":{},\"rook\":{},\"bishop\":{},\"unicorn\":{},\"dragon\":{},\"knight\":{},\"pawn\":{},\"brawn\":{},\"checkPenalty\":{},\"activeTimeline\":{},\"inactiveTimeline\":{},\"presentProgress\":{},\"mobility\":{},\"branchPenalty\":{},\"advancement\":{},\"centrality\":{}}}",
            self.king,
            self.common_king,
            self.queen,
            self.royal_queen,
            self.princess,
            self.rook,
            self.bishop,
            self.unicorn,
            self.dragon,
            self.knight,
            self.pawn,
            self.brawn,
            self.check_penalty,
            self.active_timeline,
            self.inactive_timeline,
            self.present_progress,
            self.mobility,
            self.branch_penalty,
            self.advancement,
            self.centrality
        )
    }

    fn from_json(value: &str) -> Result<Self, String> {
        Ok(Self {
            king: json_i32(value, "king")?,
            common_king: json_i32(value, "commonKing")?,
            queen: json_i32(value, "queen")?,
            royal_queen: json_i32(value, "royalQueen")?,
            princess: json_i32(value, "princess")?,
            rook: json_i32(value, "rook")?,
            bishop: json_i32(value, "bishop")?,
            unicorn: json_i32(value, "unicorn")?,
            dragon: json_i32(value, "dragon")?,
            knight: json_i32(value, "knight")?,
            pawn: json_i32(value, "pawn")?,
            brawn: json_i32(value, "brawn")?,
            check_penalty: json_i32(value, "checkPenalty")?,
            active_timeline: json_i32(value, "activeTimeline")?,
            inactive_timeline: json_i32(value, "inactiveTimeline")?,
            present_progress: json_i32(value, "presentProgress")?,
            mobility: json_i32(value, "mobility")?,
            branch_penalty: json_i32(value, "branchPenalty")?,
            advancement: json_i32(value, "advancement")?,
            centrality: json_i32(value, "centrality")?,
        })
    }
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper.max(1)
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

pub fn run_training_cli() {
    let config = TrainerConfig::from_env(std::env::args().skip(1).collect());

    if config.score_default {
        println!("{}", fitness(EvalWeights::default_tuned(), &config));
        return;
    }

    if let Some(path) = &config.score {
        let json = std::fs::read_to_string(path).expect("failed to read score weights");
        let weights = EvalWeights::from_json(&json).expect("failed to parse score weights");
        println!("{}", fitness(weights, &config));
        return;
    }

    let weights = train_weights(&config);
    let json = weights.to_json();
    if let Some(path) = &config.out {
        std::fs::write(path, &json).expect("failed to write training output");
    }
    println!("{json}");
}

impl TrainerConfig {
    fn from_env(args: Vec<String>) -> Self {
        let mut config = Self {
            generations: 50,
            population: 32,
            depth: 1,
            nodes: 5_000,
            plies: 12,
            seed: 1,
            out: None,
            score: None,
            score_default: false,
        };
        let mut index = 0;
        while index < args.len() {
            let value = args.get(index + 1).cloned();
            match args[index].as_str() {
                "--generations" => {
                    config.generations = parse_arg(value, config.generations);
                    index += 2;
                }
                "--population" => {
                    config.population = parse_arg(value, config.population);
                    index += 2;
                }
                "--depth" => {
                    config.depth = parse_arg(value, config.depth);
                    index += 2;
                }
                "--nodes" => {
                    config.nodes = parse_arg(value, config.nodes);
                    index += 2;
                }
                "--plies" => {
                    config.plies = parse_arg(value, config.plies);
                    index += 2;
                }
                "--seed" => {
                    config.seed = parse_arg(value, config.seed);
                    index += 2;
                }
                "--out" => {
                    config.out = value;
                    index += 2;
                }
                "--score" => {
                    config.score = value;
                    index += 2;
                }
                "--score-default" => {
                    config.score_default = true;
                    index += 1;
                }
                _ => index += 1,
            }
        }
        config.population = config.population.max(4);
        config
    }
}

fn train_weights(config: &TrainerConfig) -> EvalWeights {
    let mut rng = Lcg::new(config.seed);
    let mut population = vec![EvalWeights::default_tuned()];
    while population.len() < config.population {
        population.push(EvalWeights::default_tuned().mutate(&mut rng));
    }

    for generation in 0..config.generations {
        let started = std::time::Instant::now();
        let mut scored: Vec<(i32, EvalWeights)> = population
            .iter()
            .copied()
            .map(|weights| (fitness(weights, config), weights))
            .collect();
        scored.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        eprintln!(
            "generation {generation}: best {} in {:.2}s",
            scored[0].0,
            started.elapsed().as_secs_f32()
        );

        let elite = 4.min(scored.len());
        let mut next: Vec<EvalWeights> = scored.iter().take(elite).map(|entry| entry.1).collect();
        while next.len() < config.population {
            let left = tournament(&scored, &mut rng);
            let right = tournament(&scored, &mut rng);
            next.push(EvalWeights::crossover(left, right, &mut rng).mutate(&mut rng));
        }
        population = next;
    }

    population
        .into_iter()
        .max_by_key(|weights| fitness(*weights, config))
        .unwrap_or_else(EvalWeights::default_tuned)
}

fn fitness(weights: EvalWeights, config: &TrainerConfig) -> i32 {
    let mut rng = Lcg::new(config.seed);
    let mut total = 0;
    let default = EvalWeights::default_tuned();
    total += play_match(weights, default, Color::White, config);
    total += play_match(weights, default, Color::Black, config);

    for _ in 0..3 {
        let opponent = default.mutate(&mut rng);
        total += play_match(weights, opponent, Color::White, config);
        total += play_match(weights, opponent, Color::Black, config);
    }

    total
}

fn play_match(weights: EvalWeights, opponent: EvalWeights, color: Color, config: &TrainerConfig) -> i32 {
    let mut game = Game::new();
    let mut score = 0;
    for ply in 0..config.plies {
        let side_weights = if game.turn == color { weights } else { opponent };
        let mut context = SearchContext {
            weights: side_weights,
            root_color: game.turn,
            max_nodes: config.nodes,
            nodes: 0,
        };
        let Some((plan, _)) = game.search_root(config.depth, &mut context) else {
            score += if game.turn == color { -10_000 } else { 10_000 };
            break;
        };
        game = plan.game;
        let eval = game.evaluate(color, &weights);
        score += eval / 20 + eval.signum() * (config.plies - ply) as i32;
        if game.is_checkmate(color) || game.is_checkmate(color.opposite()) {
            score += if game.is_checkmate(color) {
                -CHECKMATE_SCORE / 10
            } else {
                CHECKMATE_SCORE / 10
            };
            break;
        }
    }
    score + game.evaluate(color, &weights) / 4
}

fn tournament(scored: &[(i32, EvalWeights)], rng: &mut Lcg) -> EvalWeights {
    let mut best = scored[rng.next_usize(scored.len())];
    for _ in 1..4 {
        let candidate = scored[rng.next_usize(scored.len())];
        if candidate.0 > best.0 {
            best = candidate;
        }
    }
    best.1
}

fn mutate_weight(value: i32, rng: &mut Lcg, spread: i32, min: i32, max: i32) -> i32 {
    let delta = rng.next_usize((spread * 2 + 1) as usize) as i32 - spread;
    (value + delta).clamp(min, max)
}

fn parse_arg<T: std::str::FromStr>(value: Option<String>, fallback: T) -> T {
    value.and_then(|raw| raw.parse().ok()).unwrap_or(fallback)
}

fn json_i32(value: &str, key: &str) -> Result<i32, String> {
    let needle = format!("\"{key}\":");
    let Some(start) = value.find(&needle).map(|index| index + needle.len()) else {
        return Err(format!("missing key {key}"));
    };
    let tail = &value[start..];
    let end = tail
        .find(|character: char| !character.is_ascii_digit() && character != '-')
        .unwrap_or(tail.len());
    tail[..end]
        .parse()
        .map_err(|_| format!("invalid integer for {key}"))
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
