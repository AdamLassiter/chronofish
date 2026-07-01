use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use crate::{cpu::{CHECKMATE_SCORE, EvalWeights, EvaluationLimits, EvaluationStats, SearchContext, SearchOptions, SearchStats}, *};

// Run with:
// CHRONOFISH_PERF_REPS=20 cargo test -p chronofish-engine perf_tests:: --release -- --ignored --nocapture --test-threads=1

const COMPLEX_LOG: &str = r#"
1. T0L0d2-T0L0d4
2. T1L0d7-T1L0d5
3. T2L0d1-T2L0d3
4. T3L0h7-T3L0h5
5. T4L0d3-T4L0e3
6. T5L0d8-T5L0d6
7. T6L0h2-T6L0h4
8. T7L0d6-T7L0b4
9. T8L0c2-T8L0c3
10. T9L0b4-T9L0c4
11. T10L0e3-T10L0e5
12. T11L0c4-T11L0b5
13. T12L0e5-T12L0c7
14. T13L0c8-T13L0e6
15. T14L0c7-T14L0e5
16. T15L0e6-T15L0d7
17. T16L0e5-T16L0e3
18. T17L0b5-T17L0c4
19. T18L0e3-T18L0e5
20. T19L0b8-T19L0c6
21. T20L0e5-T20L0e3
22. T21L0c4-T21L0b5
23. T22L0e3-T22L0f4
24. T23L0b5-T23L0c4
25. T24L0f4-T24L0e3
26. T25L0c4-T25L0b5
27. T26L0e3-T26L0f4
28. T27L0b5-T27L0c4
29. T28L0f4-T28L0e3
30. T29L0c4-T29L0b5
30. T29L0c4-T29L0b5
"#;

#[derive(Clone)]
struct NamedPosition {
    name: String,
    game: Game,
}

#[derive(Default)]
struct CountStats {
    values: Vec<usize>,
}

struct TimingStats {
    runs: usize,
    total: Duration,
    min: Duration,
    max: Duration,
}

#[test]
#[ignore = "performance diagnostics; run with --ignored --nocapture"]
fn perf_move_generation_and_branching_stats() {
    let positions = fixture_positions();
    let weights = EvalWeights::active_tuned();
    let reps = perf_reps(20);

    println!("\nmove generation and branching stats");
    println!("positions={} reps={reps}", positions.len());
    for position in &positions {
        let mut move_counts = CountStats::default();
        let mut candidate_counts = CountStats::default();
        let mut legal_attempt_counts = CountStats::default();
        let mut turn_plan_counts = CountStats::default();

        let timing = time_repeated(reps, || {
            let moves = position.game.legal_single_moves_until(&weights, None);
            black_box(moves.len());
        });

        for sample in sampled_games(position) {
            let moves = sample.legal_single_moves_until(&weights, None);
            move_counts.push(moves.len());

            let mut context = SearchContext::new(weights, sample.turn, 10_000, None);
            for (timeline_id, time) in sample.playable_board_keys(sample.turn) {
                let _ = sample.legal_single_moves_for_board_limited_until(
                    timeline_id,
                    time,
                    &mut context,
                    usize::MAX / 4,
                );
            }
            candidate_counts.push(context.stats.candidate_destinations);
            legal_attempt_counts.push(context.stats.legal_move_attempts);

            let mut plan_context = SearchContext::new(weights, sample.turn, 10_000, None);
            let plan_limit = plan_context.root_plan_limit();
            turn_plan_counts.push(
                sample
                    .legal_turn_plans_with_context(&mut plan_context, plan_limit)
                    .len(),
            );
        }

        print_position_shape(position);
        print_timing("legal_single_moves", &timing);
        print_counts("legal_moves", &move_counts);
        print_counts("candidate_destinations", &candidate_counts);
        print_counts("legal_move_attempts", &legal_attempt_counts);
        print_counts("turn_plans", &turn_plan_counts);
    }
}

#[test]
#[ignore = "performance diagnostics; run with --ignored --nocapture"]
fn perf_simple_and_complex_evaluation() {
    let positions = fixture_positions();
    let weights = EvalWeights::active_tuned();
    let reps = perf_reps(20);

    println!("\nsimple and complex evaluation time");
    println!("positions={} reps={reps}", positions.len());
    for position in &positions {
        let simple_timing = time_repeated(reps, || {
            let mut stats = EvaluationStats::default();
            black_box(position.game.evaluate_heuristic_with_limits(
                position.game.turn,
                &weights,
                EvaluationLimits::for_nodes(64),
                &mut stats,
            ));
            black_box(stats.calls + stats.setup_probes + stats.attack_checks + stats.clones);
        });
        let complex_timing = time_repeated(reps, || {
            let mut stats = EvaluationStats::default();
            black_box(position.game.evaluate_heuristic_with_limits(
                position.game.turn,
                &weights,
                EvaluationLimits::FULL,
                &mut stats,
            ));
            black_box(stats.calls + stats.setup_probes + stats.attack_checks + stats.clones);
        });
        let mut full_stats = EvaluationStats::default();
        let full_score = position.game.evaluate_heuristic_with_limits(
            position.game.turn,
            &weights,
            EvaluationLimits::FULL,
            &mut full_stats,
        );

        print_position_shape(position);
        print_timing("bounded_evaluation", &simple_timing);
        print_timing("full_evaluation", &complex_timing);
        println!(
            "full_eval score={} calls={} turn_moves={} setup_probes={} attack_checks={} attack_caps={} clones={}",
            full_score,
            full_stats.calls,
            full_stats.turn_moves,
            full_stats.setup_probes,
            full_stats.attack_checks,
            full_stats.attack_caps,
            full_stats.clones,
        );
    }
}

#[test]
#[ignore = "performance diagnostics; run with --ignored --nocapture"]
fn perf_quiescence_time() {
    let positions = fixture_positions();
    let weights = EvalWeights::active_tuned();
    let reps = perf_reps(20);

    println!("\nquiescence time");
    println!("positions={} reps={reps}", positions.len());
    for position in &positions {
        let timing = time_repeated(reps, || {
            let mut game = position.game.clone_for_search();
            let mut context = SearchContext::new(weights, game.turn, 10_000, None);
            let score = game.quiescence(
                -CHECKMATE_SCORE * 2,
                CHECKMATE_SCORE * 2,
                position.game.turn,
                &mut context,
                2,
            );
            black_box(score);
            black_box(context.stats.generated_moves + context.stats.evaluation_calls);
        });

        let mut game = position.game.clone_for_search();
        let mut context = SearchContext::new(weights, game.turn, 10_000, None);
        let score = game.quiescence(
            -CHECKMATE_SCORE * 2,
            CHECKMATE_SCORE * 2,
            position.game.turn,
            &mut context,
            2,
        );

        print_position_shape(position);
        print_timing("quiescence_depth_2", &timing);
        print_search_stats("quiescence", score, context.nodes, &context.stats);
    }
}

#[test]
#[ignore = "performance diagnostics; run with --ignored --nocapture"]
fn perf_turn_plan_generation_time() {
    let positions = fixture_positions();
    let weights = EvalWeights::active_tuned();
    let reps = perf_reps(20);

    println!("\nturn plan generation time");
    println!("positions={} reps={reps}", positions.len());
    for position in &positions {
        let timing = time_repeated(reps, || {
            let mut context = SearchContext::new(weights, position.game.turn, 10_000, None);
            let plan_limit = context.root_plan_limit();
            let plans = position
                .game
                .legal_turn_plans_with_context(&mut context, plan_limit);
            black_box(plans.len());
            black_box(context.stats.generated_moves + context.stats.generated_plans);
        });

        let mut context = SearchContext::new(weights, position.game.turn, 10_000, None);
        let plan_limit = context.root_plan_limit();
        let plans = position
            .game
            .legal_turn_plans_with_context(&mut context, plan_limit);

        print_position_shape(position);
        print_timing("legal_turn_plans", &timing);
        print_search_stats(
            "turn_plans",
            plans.len() as i32,
            context.nodes,
            &context.stats,
        );
    }
}

#[test]
#[ignore = "performance diagnostics; run with --ignored --nocapture"]
fn perf_shallow_search_stats() {
    let positions = representative_positions();

    println!("\nshallow search stats");
    println!("positions={} depth=2 nodes=20000", positions.len());
    for position in &positions {
        let (result, sample) = position.game.best_ai_turn_with_options_min_depth(
            2,
            1,
            20_000,
            None,
            SearchOptions::optimized(),
            Some("perf_shallow_search"),
        );

        print_position_shape(position);
        println!(
            "search result status={} depth={} score={} moves={} nodes={}",
            result.status,
            result.depth,
            result.score,
            result.moves.len(),
            result.nodes,
        );
        if let Some(sample) = sample {
            println!(
                "search_perf elapsed_us={} nodes={} nps={:.0}",
                sample.elapsed_micros,
                sample.nodes,
                rate_per_second(sample.nodes, sample.elapsed_micros),
            );
            print_search_stats("search", result.score, sample.nodes, &sample.stats);
        }
    }
}

fn fixture_positions() -> Vec<NamedPosition> {
    let mut positions = Vec::new();
    positions.push(NamedPosition {
        name: "initial".to_string(),
        game: Game::new(),
    });

    let mut tactical = Game::new();
    tactical
        .load_notation(
            "1. T0L0e2Pe4\n\
             2. T1L0d7pd5\n\
             3. T2L0d1Qh5\n\
             4. T3L0g8nf6",
        )
        .expect("tactical fixture should replay");
    positions.push(NamedPosition {
        name: "tactical_midgame".to_string(),
        game: tactical,
    });

    positions.extend(replay_legacy_log_positions(COMPLEX_LOG, 11));
    positions
}

fn representative_positions() -> Vec<NamedPosition> {
    let positions = fixture_positions();
    let mut representative = Vec::new();
    if let Some(position) = positions.iter().find(|position| position.name == "initial") {
        representative.push(position.clone());
    }
    if let Some(position) = positions
        .iter()
        .find(|position| position.name == "tactical_midgame")
    {
        representative.push(position.clone());
    }
    if let Some(position) = positions
        .iter()
        .rev()
        .find(|position| position.name.starts_with("complex_log_move_"))
    {
        representative.push(position.clone());
    }
    representative
}

fn sampled_games(position: &NamedPosition) -> Vec<Game> {
    if position.name == "complex_log_move_11" {
        replay_legacy_log_positions(COMPLEX_LOG, 11)
            .into_iter()
            .map(|position| position.game)
            .collect()
    } else {
        vec![position.game.clone_for_search()]
    }
}

fn replay_legacy_log_positions(log: &str, max_turns: usize) -> Vec<NamedPosition> {
    let mut game = Game::new();
    let mut positions = Vec::new();
    let mut previous_turn = 0;

    for line in log.lines() {
        let Some((turn, moves)) = parse_legacy_log_line(line) else {
            continue;
        };
        if turn <= previous_turn || turn > max_turns {
            break;
        }
        previous_turn = turn;

        let replayed = moves
            .split('/')
            .map(str::trim)
            .filter(|move_text| !move_text.is_empty())
            .all(|move_text| apply_legacy_move(&mut game, move_text).is_ok());
        if !replayed || game.submit_turn() == 0 {
            break;
        }

        positions.push(NamedPosition {
            name: format!("complex_log_move_{turn}"),
            game: game.clone_for_search(),
        });
    }

    positions
}

fn parse_legacy_log_line(line: &str) -> Option<(usize, &str)> {
    let line = line.trim();
    let (turn, moves) = line.split_once('.')?;
    Some((turn.trim().parse().ok()?, moves.trim()))
}

fn apply_legacy_move(game: &mut Game, move_text: &str) -> Result<(), String> {
    let (from, to) = parse_legacy_move(move_text)?;
    if game.apply_move(from, to) == 0 {
        return Err(game.last_message.clone());
    }
    Ok(())
}

fn parse_legacy_move(move_text: &str) -> Result<(Position, Position), String> {
    let (from, to) = move_text
        .split_once('-')
        .ok_or_else(|| format!("missing move separator in `{move_text}`"))?;
    Ok((parse_legacy_position(from)?, parse_legacy_position(to)?))
}

fn parse_legacy_position(text: &str) -> Result<Position, String> {
    let chars: Vec<char> = text.trim().chars().collect();
    let mut index = 0;
    let time = parse_signed_prefixed_i32(&chars, &mut index, 'T')?;
    let timeline_id = parse_signed_prefixed_i32(&chars, &mut index, 'L')?;
    let file = *chars
        .get(index)
        .ok_or_else(|| format!("missing file in `{text}`"))?;
    index += 1;
    let rank = *chars
        .get(index)
        .ok_or_else(|| format!("missing rank in `{text}`"))?;
    index += 1;
    if index != chars.len() {
        return Err(format!("unexpected suffix in `{text}`"));
    }
    let x = (file as u8).wrapping_sub(b'a') as i32;
    let y = (rank as u8).wrapping_sub(b'1') as i32;
    if !Game::in_bounds(x, y) {
        return Err(format!("out of bounds square in `{text}`"));
    }
    Ok(Position {
        timeline_id,
        time,
        x,
        y,
    })
}

fn parse_signed_prefixed_i32(
    chars: &[char],
    index: &mut usize,
    prefix: char,
) -> Result<i32, String> {
    if chars.get(*index) != Some(&prefix) {
        return Err(format!("expected `{prefix}`"));
    }
    *index += 1;
    let sign = if chars.get(*index) == Some(&'-') {
        *index += 1;
        -1
    } else {
        1
    };
    let start = *index;
    while chars
        .get(*index)
        .is_some_and(|character| character.is_ascii_digit())
    {
        *index += 1;
    }
    if start == *index {
        return Err(format!("missing integer after `{prefix}`"));
    }
    let value = chars[start..*index]
        .iter()
        .collect::<String>()
        .parse::<i32>()
        .map_err(|error| error.to_string())?;
    Ok(value * sign)
}

fn time_repeated(reps: usize, mut run: impl FnMut()) -> TimingStats {
    for _ in 0..3 {
        run();
    }

    let mut total = Duration::ZERO;
    let mut min = Duration::MAX;
    let mut max = Duration::ZERO;
    for _ in 0..reps {
        let started = Instant::now();
        run();
        let elapsed = started.elapsed();
        total += elapsed;
        min = min.min(elapsed);
        max = max.max(elapsed);
    }

    TimingStats {
        runs: reps,
        total,
        min,
        max,
    }
}

fn perf_reps(default: usize) -> usize {
    env::var("CHRONOFISH_PERF_REPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
        .max(1)
}

impl CountStats {
    fn push(&mut self, value: usize) {
        self.values.push(value);
    }

    fn sorted(&self) -> Vec<usize> {
        let mut values = self.values.clone();
        values.sort_unstable();
        values
    }

    fn average(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<usize>() as f64 / self.values.len() as f64
    }
}

fn print_position_shape(position: &NamedPosition) {
    let total_boards: usize = position
        .game
        .timelines
        .iter()
        .map(|timeline| timeline.boards.len())
        .sum();
    let active_timelines = position
        .game
        .timelines
        .iter()
        .filter(|timeline| position.game.is_active_timeline(timeline.id))
        .count();
    let playable_boards = position.game.playable_board_keys(position.game.turn).len();
    println!(
        "\nposition={} turn={:?} timelines={} active_timelines={} total_boards={} playable_boards={}",
        position.name,
        position.game.turn,
        position.game.timelines.len(),
        active_timelines,
        total_boards,
        playable_boards,
    );
}

fn print_timing(label: &str, stats: &TimingStats) {
    let avg_us = stats.total.as_micros() as f64 / stats.runs as f64;
    println!(
        "{label} avg_us={avg_us:.1} min_us={} max_us={}",
        stats.min.as_micros(),
        stats.max.as_micros(),
    );
}

fn print_counts(label: &str, stats: &CountStats) {
    let values = stats.sorted();
    if values.is_empty() {
        println!("{label} count=0");
        return;
    }
    let median = percentile(&values, 50);
    let p90 = percentile(&values, 90);
    println!(
        "{label} samples={} avg={:.1} median={} p90={} max={}",
        values.len(),
        stats.average(),
        median,
        p90,
        values[values.len() - 1],
    );
}

fn print_search_stats(label: &str, score: i32, nodes: usize, stats: &SearchStats) {
    println!(
        "{label} score={} nodes={} generated_moves={} generated_plans={} candidate_destinations={} legal_move_attempts={} attack_queries={} attack_cache_hits={} search_clones={} turn_plan_cache_hits={} tt_hits={} beta_cutoffs={} reduced_searches={} aspiration_researches={} expensive_order_probes={} evaluation_calls={} evaluation_cache_hits={} evaluated_turn_moves={} evaluation_setup_probes={} evaluation_attack_checks={} evaluation_attack_caps={} evaluation_clones={}",
        score,
        nodes,
        stats.generated_moves,
        stats.generated_plans,
        stats.candidate_destinations,
        stats.legal_move_attempts,
        stats.attack_queries,
        stats.attack_cache_hits,
        stats.search_clones,
        stats.turn_plan_cache_hits,
        stats.tt_hits,
        stats.beta_cutoffs,
        stats.reduced_searches,
        stats.aspiration_researches,
        stats.expensive_order_probes,
        stats.evaluation_calls,
        stats.evaluation_cache_hits,
        stats.evaluated_turn_moves,
        stats.evaluation_setup_probes,
        stats.evaluation_attack_checks,
        stats.evaluation_attack_caps,
        stats.evaluation_clones,
    );
}

fn percentile(sorted: &[usize], percentile: usize) -> usize {
    let index = ((sorted.len() - 1) * percentile) / 100;
    sorted[index]
}

fn rate_per_second(count: usize, elapsed_micros: u128) -> f64 {
    if elapsed_micros == 0 {
        return 0.0;
    }
    count as f64 * 1_000_000.0 / elapsed_micros as f64
}
