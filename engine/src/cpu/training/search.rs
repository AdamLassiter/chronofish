use super::*;
use crate::cpu::{
    EvalWeights,
    EvaluationLimits,
    SearchContext,
    SearchInstant,
    SearchOptions,
    TurnPlan,
    ASPIRATION_WINDOW,
    CHECKMATE_SCORE,
};

pub(crate) struct TrainingSearchOutcome {
    pub(crate) plan: TurnPlan,
    pub(crate) depth: i32,
    pub(crate) obligations: usize,
    pub(crate) playable_boards: usize,
}

#[allow(dead_code)]
pub(crate) fn training_turn_plan(
    game: &Game,
    weights: EvalWeights,
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
) -> Option<TurnPlan> {
    training_turn_search(game, weights, config, deadline, 0).map(|outcome| outcome.plan)
}

pub(crate) fn training_turn_search(
    game: &Game,
    weights: EvalWeights,
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
    plies_played: usize,
) -> Option<TrainingSearchOutcome> {
    match config.search_strategy {
        TrainingSearchStrategy::AlphaBeta => {
            alpha_beta_training_turn_search(game, weights, config, deadline, plies_played)
        }
        TrainingSearchStrategy::Beam => {
            beam_training_turn_search(game, weights, config, deadline, plies_played)
        }
    }
}

fn alpha_beta_training_turn_search(
    game: &Game,
    weights: EvalWeights,
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
    plies_played: usize,
) -> Option<TrainingSearchOutcome> {
    let deadline = training_turn_deadline(config, deadline);
    let mut context = SearchContext::new(weights, game.turn, config.nodes, deadline);
    context.options = SearchOptions::training();
    let profile = apply_training_search_profile(game, &mut context, plies_played);
    context.killers.resize(
        (MAX_TRAINING_SEARCH_DEPTH as usize).saturating_add(3),
        [None, None],
    );
    let capped = context.root_plan_cap.is_some() || context.child_plan_cap.is_some();
    let fallback = if capped {
        let mut fallback_context = SearchContext::new(weights, game.turn, config.nodes, deadline);
        fallback_context.options = SearchOptions::minimal();
        fallback_context.evaluation_limits = Some(EvaluationLimits::training_fast_late_game(
            fallback_context.max_nodes,
        ));
        fallback_training_turn_plan(game, &mut fallback_context)
    } else {
        None
    };

    let (mut best, mut best_depth) = run_alpha_beta_training_search(game, &mut context);
    if best.is_none() {
        best = fallback;
        best_depth = 0;
    }
    best.map(|plan| TrainingSearchOutcome {
        plan,
        depth: best_depth,
        obligations: profile.obligations,
        playable_boards: profile.playable_boards,
    })
}

fn fallback_training_turn_plan(game: &Game, context: &mut SearchContext) -> Option<TurnPlan> {
    let plan_limit = context.root_plan_limit();
    game.legal_turn_plans_with_context(context, plan_limit)
        .into_iter()
        .filter(|plan| game.apply_turn_plan_for_search(plan).is_some())
        .max_by(|left, right| {
            left.score_hint
                .cmp(&right.score_hint)
                .then_with(|| Game::turn_plan_cmp(right, left))
        })
}

fn run_alpha_beta_training_search(
    game: &Game,
    context: &mut SearchContext,
) -> (Option<TurnPlan>, i32) {
    let mut best = None;
    let mut best_depth = 0;
    let mut previous_score = 0;
    for depth in 1..=MAX_TRAINING_SEARCH_DEPTH {
        if context.exhausted() {
            break;
        }
        let window = if context.use_aspiration_windows() && depth > 1 {
            Some((
                previous_score - ASPIRATION_WINDOW,
                previous_score + ASPIRATION_WINDOW,
            ))
        } else {
            None
        };
        let Some((plan, score)) = game.search_root(depth, &mut *context, window) else {
            break;
        };
        if game.apply_turn_plan_for_search(&plan).is_none() {
            if context.exhausted() {
                break;
            }
            continue;
        }
        previous_score = score;
        best = Some(plan);
        best_depth = depth;
        if context.exhausted() || score.abs() >= CHECKMATE_SCORE / 2 {
            break;
        }
    }
    (best, best_depth)
}

fn training_turn_deadline(
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
) -> Option<SearchInstant> {
    let turn_deadline = (config.training_time_ms > 0)
        .then(|| SearchInstant::now() + std::time::Duration::from_millis(config.training_time_ms));
    match (turn_deadline, deadline) {
        (Some(turn_deadline), Some(training_deadline)) => {
            Some(if turn_deadline <= training_deadline {
                turn_deadline
            } else {
                training_deadline
            })
        }
        (Some(turn_deadline), None) => Some(turn_deadline),
        (None, Some(training_deadline)) => Some(training_deadline),
        (None, None) => None,
    }
}

pub(crate) struct TrainingSearchProfile {
    pub(crate) obligations: usize,
    pub(crate) playable_boards: usize,
}

pub(crate) fn apply_training_search_profile(
    game: &Game,
    context: &mut SearchContext,
    plies_played: usize,
) -> TrainingSearchProfile {
    let obligations = game.present_obligation_count(game.turn).max(0) as usize;
    let playable_boards = game.playable_board_keys(game.turn).len();
    if obligations >= 4 || playable_boards >= 4 || plies_played >= 24 {
        context.root_plan_cap = Some(2);
        context.child_plan_cap = Some(1);
        context.evaluation_limits =
            Some(EvaluationLimits::training_fast_late_game(context.max_nodes));
    } else if obligations >= 3 || playable_boards >= 3 {
        context.root_plan_cap = Some(4);
        context.child_plan_cap = Some(2);
        context.evaluation_limits = Some(EvaluationLimits::training_late_game(context.max_nodes));
    }
    TrainingSearchProfile {
        obligations,
        playable_boards,
    }
}

fn beam_training_turn_search(
    game: &Game,
    weights: EvalWeights,
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
    plies_played: usize,
) -> Option<TrainingSearchOutcome> {
    let mut context = SearchContext::new(
        weights,
        game.turn,
        config.nodes,
        training_turn_deadline(config, deadline),
    );
    context.options = SearchOptions::minimal();
    let profile = apply_training_search_profile(game, &mut context, plies_played);
    let limit = context.root_plan_limit();
    game.legal_turn_plans_with_context(&mut context, limit)
        .into_iter()
        .filter_map(|plan| {
            game.apply_turn_plan_for_search(&plan)
                .map(|child| (plan, child))
        })
        .max_by(|(left, left_child), (right, right_child)| {
            let left_score =
                left_child.evaluate_heuristic_for_nodes(game.turn, &weights, config.nodes)
                    + left.score_hint;
            let right_score =
                right_child.evaluate_heuristic_for_nodes(game.turn, &weights, config.nodes)
                    + right.score_hint;
            left_score
                .cmp(&right_score)
                .then_with(|| Game::turn_plan_cmp(right, left))
        })
        .map(|(plan, _)| TrainingSearchOutcome {
            plan,
            depth: 1,
            obligations: profile.obligations,
            playable_boards: profile.playable_boards,
        })
}
