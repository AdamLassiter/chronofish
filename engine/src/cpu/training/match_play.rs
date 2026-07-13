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
    let mut total_search_ms = 0u128;
    let mut max_turn_ms = 0u128;
    let mut max_turn_ply = 0;
    let mut max_depth = 0;
    let mut slow_turns = 0;
    let mut fallback_turns = 0;
    let mut capped_turns = 0;
    let mut peak_obligations = 0;
    let mut peak_playable_boards = 0;
    let match_deadline =
        Some(match_started + std::time::Duration::from_millis(max_match_time_ms(config).max(1)));
    training_task_progress(
        match_label,
        "match",
        0,
        config.max_match_plies.max(1) as usize,
        format!("{} {}", color.as_str(), short_training_label(match_label)),
    );
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
            let elapsed_ms = SearchInstant::now()
                .duration_since(turn_started)
                .as_millis();
            log_training_match_summary(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms.max(elapsed_ms),
                if elapsed_ms > max_turn_ms {
                    plies_played + 1
                } else {
                    max_turn_ply
                },
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                score,
                if training_expired(match_deadline) {
                    "match-time-cap"
                } else {
                    "turn-timeout"
                },
            );
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
            finish_training_task(match_label);
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
        total_search_ms += search.elapsed_ms;
        if search.elapsed_ms > max_turn_ms {
            max_turn_ms = search.elapsed_ms;
            max_turn_ply = plies_played;
        }
        max_depth = max_depth.max(search.depth);
        peak_obligations = peak_obligations.max(search.obligations);
        peak_playable_boards = peak_playable_boards.max(search.playable_boards);
        if search.fallback_used {
            fallback_turns += 1;
        }
        if search.capped {
            capped_turns += 1;
        }
        if search.elapsed_ms >= slow_training_turn_threshold_ms(config) {
            slow_turns += 1;
            log_slow_training_turn(match_label, plies_played, mover, &search);
        }
        training_task_progress(
            match_label,
            "match",
            plies_played as usize,
            config.max_match_plies.max(1) as usize,
            format!(
                "{} d{} n{} {}",
                mover.as_str(),
                search.depth,
                search.nodes,
                short_training_label(match_label)
            ),
        );
        if let Some(terminal) = game.terminal_score_until(color, turn_deadline) {
            if terminal == CHECKMATE_SCORE {
                log_training_match_summary(
                    match_label,
                    plies_played,
                    SearchInstant::now()
                        .duration_since(match_started)
                        .as_millis(),
                    total_search_ms,
                    max_turn_ms,
                    max_turn_ply,
                    max_depth,
                    slow_turns,
                    fallback_turns,
                    capped_turns,
                    peak_obligations,
                    peak_playable_boards,
                    score,
                    "win",
                );
                return MatchReport {
                    score: score + CHECKMATE_SCORE / 10 - plies_played * 10,
                    result: MatchResult::Win,
                    blunder: false,
                };
            }
            if terminal == -CHECKMATE_SCORE {
                log_training_match_summary(
                    match_label,
                    plies_played,
                    SearchInstant::now()
                        .duration_since(match_started)
                        .as_millis(),
                    total_search_ms,
                    max_turn_ms,
                    max_turn_ply,
                    max_depth,
                    slow_turns,
                    fallback_turns,
                    capped_turns,
                    peak_obligations,
                    peak_playable_boards,
                    score,
                    "loss",
                );
                return MatchReport {
                    score: score - CHECKMATE_SCORE / 10 + plies_played * 10,
                    result: MatchResult::Loss,
                    blunder: true,
                };
            }
            log_training_match_summary(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms,
                max_turn_ply,
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                score,
                "draw",
            );
            return MatchReport {
                score,
                result: MatchResult::Draw,
                blunder: false,
            };
        }
        let eval =
            game.evaluate_heuristic_for_nodes_until(color, &weights, config.nodes, turn_deadline);
        score += eval / 20;
        if should_log_training_match_milestone(plies_played) {
            log_training_match_milestone(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms,
                max_turn_ply,
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                eval,
                score,
            );
        }
        stable_advantage = if eval.abs() > 4_000 {
            stable_advantage + eval.signum()
        } else {
            0
        };
        if stable_advantage >= 2 {
            log_training_match_summary(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms,
                max_turn_ply,
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                score,
                "stable-win",
            );
            return MatchReport {
                score: score + 10_000,
                result: MatchResult::Win,
                blunder: false,
            };
        }
        if stable_advantage <= -2 {
            log_training_match_summary(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms,
                max_turn_ply,
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                score,
                "stable-loss",
            );
            return MatchReport {
                score: score - 10_000,
                result: MatchResult::Loss,
                blunder: true,
            };
        }
        if plies_played >= config.max_match_plies {
            let final_score = score + eval / 4;
            let result = adjudicated_match_result(final_score);
            log_training_match_summary(
                match_label,
                plies_played,
                SearchInstant::now()
                    .duration_since(match_started)
                    .as_millis(),
                total_search_ms,
                max_turn_ms,
                max_turn_ply,
                max_depth,
                slow_turns,
                fallback_turns,
                capped_turns,
                peak_obligations,
                peak_playable_boards,
                final_score,
                "ply-cap",
            );
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
    let exit_deadline = earliest_deadline(deadline, match_deadline);
    log_training_match_summary(
        match_label,
        plies_played,
        SearchInstant::now()
            .duration_since(match_started)
            .as_millis(),
        total_search_ms,
        max_turn_ms,
        max_turn_ply,
        max_depth,
        slow_turns,
        fallback_turns,
        capped_turns,
        peak_obligations,
        peak_playable_boards,
        final_score,
        if training_expired(match_deadline) {
            "match-time-cap"
        } else if training_expired(exit_deadline) {
            "deadline"
        } else {
            "adjudicated"
        },
    );
    MatchReport {
        score: final_score,
        result,
        blunder: false,
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

fn slow_training_turn_threshold_ms(config: &CpuCliConfig) -> u128 {
    10_000.min((config.training_time_ms.max(1) as u128).max(1))
}

fn short_training_label(label: &str) -> String {
    let mut parts = label.split(" seed=");
    parts.next().unwrap_or(label).replace("sweep ", "")
}

fn should_log_training_match_milestone(plies_played: i32) -> bool {
    plies_played >= 20 && plies_played % 10 == 0
}

fn log_slow_training_turn(
    _match_label: &str,
    _plies_played: i32,
    _mover: Color,
    _search: &TrainingSearchOutcome,
) {
}

#[allow(clippy::too_many_arguments)]
fn log_training_match_milestone(
    _match_label: &str,
    _plies_played: i32,
    _elapsed_ms: u128,
    _total_search_ms: u128,
    _max_turn_ms: u128,
    _max_turn_ply: i32,
    _max_depth: i32,
    _slow_turns: i32,
    _fallback_turns: i32,
    _capped_turns: i32,
    _peak_obligations: usize,
    _peak_playable_boards: usize,
    _eval: i32,
    _score: i32,
) {
}

#[allow(clippy::too_many_arguments)]
fn log_training_match_summary(
    match_label: &str,
    _plies_played: i32,
    _elapsed_ms: u128,
    _total_search_ms: u128,
    _max_turn_ms: u128,
    _max_turn_ply: i32,
    _max_depth: i32,
    _slow_turns: i32,
    _fallback_turns: i32,
    _capped_turns: i32,
    _peak_obligations: usize,
    _peak_playable_boards: usize,
    _score: i32,
    _reason: &str,
) {
    finish_training_task(match_label);
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
