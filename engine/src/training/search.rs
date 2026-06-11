use super::*;

pub(crate) fn training_turn_plan(
    game: &Game,
    weights: EvalWeights,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> Option<TurnPlan> {
    match config.search_strategy {
        TrainingSearchStrategy::AlphaBeta => {
            let mut context = SearchContext::new(weights, game.turn, config.nodes, deadline);
            game.search_root(config.depth, &mut context, None)
                .map(|(plan, _)| plan)
        }
        #[cfg(feature = "training-beam-search")]
        TrainingSearchStrategy::Beam => beam_training_turn_plan(game, weights, config, deadline),
    }
}

#[cfg(feature = "training-beam-search")]
fn beam_training_turn_plan(
    game: &Game,
    weights: EvalWeights,
    config: &TrainerConfig,
    deadline: Option<SearchInstant>,
) -> Option<TurnPlan> {
    let mut context = SearchContext::new(weights, game.turn, config.nodes, deadline);
    context.options = SearchOptions::minimal();
    let limit = context.root_plan_limit();
    game.legal_turn_plans_with_context(&mut context, limit)
        .into_iter()
        .filter_map(|plan| {
            game.apply_turn_plan_for_search(&plan)
                .map(|child| (plan, child))
        })
        .max_by(|(left, left_child), (right, right_child)| {
            let left_score = left_child.evaluate(game.turn, &weights) + left.score_hint;
            let right_score = right_child.evaluate(game.turn, &weights) + right.score_hint;
            left_score
                .cmp(&right_score)
                .then_with(|| Game::turn_plan_cmp(right, left))
        })
        .map(|(plan, _)| plan)
}
