use super::*;

pub(crate) fn training_turn_plan(
    game: &Game,
    weights: EvalWeights,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> Option<TurnPlan> {
    match config.search_strategy {
        TrainingSearchStrategy::AlphaBeta => {
            alpha_beta_training_turn_plan(game, weights, config, deadline)
        }
        #[cfg(feature = "training-beam-search")]
        TrainingSearchStrategy::Beam => beam_training_turn_plan(game, weights, config, deadline),
    }
}

fn alpha_beta_training_turn_plan(
    game: &Game,
    weights: EvalWeights,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> Option<TurnPlan> {
    let deadline = training_turn_deadline(config, deadline);
    let mut context = SearchContext::new(weights, game.turn, config.nodes, deadline);
    context.killers.resize(
        (MAX_TRAINING_SEARCH_DEPTH as usize).saturating_add(3),
        [None, None],
    );

    let mut best = None;
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
        let Some((plan, score)) = game.search_root(depth, &mut context, window) else {
            break;
        };
        previous_score = score;
        best = Some(plan);
        if context.exhausted() || score.abs() >= CHECKMATE_SCORE / 2 {
            break;
        }
    }
    best
}

fn training_turn_deadline(
    config: &TrainerConfig,
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

#[cfg(feature = "training-beam-search")]
fn beam_training_turn_plan(
    game: &Game,
    weights: EvalWeights,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> Option<TurnPlan> {
    let mut context = SearchContext::new(
        weights,
        game.turn,
        config.nodes,
        training_turn_deadline(config, deadline),
    );
    context.options = SearchOptions::minimal();
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
        .map(|(plan, _)| plan)
}
