use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn play_match_until(
    start: Game,
    weights: EvalWeights,
    opponent: EvalWeights,
    color: Color,
    _candidate_label: &str,
    _opponent_label: &str,
    match_label: &str,
    config: &CpuCliConfig,
    deadline: Option<SearchInstant>,
) -> MatchReport {
    // Full-match scoring keeps the objective aligned with real game outcomes
    // rather than stopping at a fixed ply horizon.
    let mut game = start;
    let mut score = 0;
    let mut stable_advantage = 0;
    let mut plies_played = 0;
    let match_started = SearchInstant::now();
    let mut max_depth = 0;
    let mut peak_obligations = 0;
    let mut peak_playable_boards = 0;
    let match_deadline =
        Some(match_started + std::time::Duration::from_millis(max_match_time_ms(config).max(1)));
    training_task_progress(
        match_label,
        0,
        config.max_match_plies.max(1) as usize,
        format!("starting color={}", color.as_str()),
    );
    let _task_guard = MatchTaskGuard(match_label);
    loop {
        let turn_deadline = earliest_deadline(deadline, match_deadline);
        if training_expired(turn_deadline) {
            break;
        }
        let mover = game.turn;
        let side_weights = if mover == color { weights } else { opponent };
        let turn_started = SearchInstant::now();
        let Some(search) = training_turn_search(
            &game,
            side_weights,
            config,
            turn_deadline,
            plies_played as usize,
        ) else {
            return MatchReport {
                score,
                result: MatchResult::Draw,
                blunder: false,
            };
        };
        let elapsed_ms = SearchInstant::now()
            .duration_since(turn_started)
            .as_millis();
        let _elapsed_ms = elapsed_ms;
        let Some(next_game) = game.apply_turn_plan_for_search(&search.plan) else {
            let result = if game.turn == color {
                MatchResult::Loss
            } else {
                MatchResult::Win
            };
            return MatchReport {
                score: score
                    + if result == MatchResult::Win {
                        20_000 - plies_played * 10
                    } else {
                        -20_000 + plies_played * 10
                    },
                result,
                blunder: result == MatchResult::Loss,
            };
        };
        game = next_game;
        plies_played += 1;
        max_depth = max_depth.max(search.depth);
        peak_obligations = peak_obligations.max(search.obligations);
        peak_playable_boards = peak_playable_boards.max(search.playable_boards);
        if plies_played == 1 || plies_played % 4 == 0 {
            training_task_progress(
                match_label,
                plies_played as usize,
                config.max_match_plies.max(1) as usize,
                format!(
                    "depth={} obligations={} boards={} elapsed_ms={elapsed_ms}",
                    max_depth, peak_obligations, peak_playable_boards
                ),
            );
        }
        if let Some(terminal) = game.terminal_score_until(color, turn_deadline) {
            if terminal == CHECKMATE_SCORE {
                return MatchReport {
                    score: score + CHECKMATE_SCORE / 10 - plies_played * 10,
                    result: MatchResult::Win,
                    blunder: false,
                };
            }
            if terminal == -CHECKMATE_SCORE {
                return MatchReport {
                    score: score - CHECKMATE_SCORE / 10 + plies_played * 10,
                    result: MatchResult::Loss,
                    blunder: true,
                };
            }
            return MatchReport {
                score,
                result: MatchResult::Draw,
                blunder: false,
            };
        }
        let eval =
            game.evaluate_heuristic_for_nodes_until(color, &weights, config.nodes, turn_deadline);
        score += eval / 20;
        stable_advantage = if eval.abs() > 4_000 {
            stable_advantage + eval.signum()
        } else {
            0
        };
        if stable_advantage >= 2 {
            return MatchReport {
                score: score + 10_000,
                result: MatchResult::Win,
                blunder: false,
            };
        }
        if stable_advantage <= -2 {
            return MatchReport {
                score: score - 10_000,
                result: MatchResult::Loss,
                blunder: true,
            };
        }
        if plies_played >= config.max_match_plies {
            let final_score = score + eval / 4;
            let result = adjudicated_match_result(final_score);
            return MatchReport {
                score: final_score,
                result,
                blunder: false,
            };
        }
    }
    let final_score = score
        + game.evaluate_heuristic_for_nodes_until(
            color,
            &weights,
            config.nodes,
            earliest_deadline(deadline, match_deadline),
        ) / 4;
    let result = adjudicated_match_result(final_score);
    MatchReport {
        score: final_score,
        result,
        blunder: false,
    }
}

struct MatchTaskGuard<'a>(&'a str);

impl Drop for MatchTaskGuard<'_> {
    fn drop(&mut self) {
        finish_training_task(self.0);
    }
}

fn adjudicated_match_result(final_score: i32) -> MatchResult {
    if final_score > 300 {
        MatchResult::Win
    } else if final_score < -300 {
        MatchResult::Loss
    } else {
        MatchResult::Draw
    }
}

fn earliest_deadline(
    left: Option<SearchInstant>,
    right: Option<SearchInstant>,
) -> Option<SearchInstant> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
        (None, None) => None,
    }
}

#[cfg(test)]
pub(crate) fn turn_plan_notation(start: &Game, plan: &TurnPlan) -> String {
    let mut notation_game = start.clone_for_search();
    for movement in &plan.moves {
        if notation_game.apply_move(movement.from, movement.to) == 0 {
            return plan
                .moves
                .iter()
                .map(|movement| {
                    use crate::notation::{position_prefix, square_name};

                    format!(
                        "{}{} -> {}{}",
                        position_prefix(movement.from),
                        square_name(movement.from),
                        position_prefix(movement.to),
                        square_name(movement.to)
                    )
                })
                .collect::<Vec<_>>()
                .join("/");
        }
    }
    notation_game.staged_turn_notation()
}
