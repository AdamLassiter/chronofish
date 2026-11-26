//! Port of the 64-feature evaluator in ftxi's linear.cpp and statistics.cpp.

use chronofish_5d_rules_reference::core::{hypercuboid::HcInfo, state::State};

#[derive(Clone, Copy, Debug)]
pub enum LinearProfile {
    Default,
    Trained,
}

#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
const TRAINED: [f32; 64] = [
    -0.00123027945,
    -0.00600768765,
    0.00338658318,
    0.00632669684,
    -0.00422760518,
    -0.00573036913,
    -0.00573036913,
    -0.00573036913,
    -0.0115035931,
    0.000595524965,
    0.0781893134,
    0.0359656215,
    0.175541759,
    -0.0855880827,
    -0.115588054,
    0.284411639,
    0.288696021,
    -0.00817218516,
    -0.00686811749,
    -0.0119211152,
    -0.0104583567,
    -0.0165845193,
    -0.0165845193,
    -0.0165845193,
    -0.00623198925,
    0.0197074413,
    0.0567287579,
    0.137343943,
    0.0912609622,
    0.046411965,
    0.0314119607,
    0.231411979,
    0.171710014,
    0.0163448583,
    0.0168772414,
    0.015377258,
    0.0141730411,
    0.0118304137,
    0.0118304137,
    0.0118304137,
    0.0140779829,
    0.027899893,
    0.0103068734,
    0.0221800804,
    0.00934366416,
    -0.00264071906,
    -0.00714072073,
    0.0528592542,
    0.0637308881,
    -0.0115995659,
    -0.0115995659,
    0.0,
    -0.0118690757,
    -0.00865917187,
    0.0151735367,
    -0.0135565149,
    0.0168984998,
    0.064030014,
    0.222902104,
    0.00271896203,
    0.0168984998,
    0.064030014,
    0.0140336938,
    0.0197210461,
];

pub(crate) fn evaluate(state: &State, context: &HcInfo, profile: LinearProfile) -> f64 {
    let features = features(state, context);
    let weights = match profile {
        LinearProfile::Default => default_weights(),
        LinearProfile::Trained => TRAINED,
    };
    let score: f32 = features.iter().zip(weights).map(|(x, w)| x * w).sum();
    let white_sign = if state.get_present().1 { -1.0 } else { 1.0 };
    white_sign * f64::from(score.tanh())
}

fn default_weights() -> [f32; 64] {
    let mut weights = [0.0; 64];
    let piece_weights = [0.05, 0.15, 0.25, 0.15, 0.08, 0.05, 0.45, 0.30];
    for (group, scale) in [1.0, 0.5, 0.15].into_iter().enumerate() {
        let offset = 9 + group * 16;
        for (index, weight) in piece_weights.into_iter().enumerate() {
            weights[offset + index] = weight * scale;
        }
    }
    weights[58] = 0.25;
    weights[62..].fill(0.04);
    weights
}

// Match the numeric conversions in the upstream feature schema.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn features(state: &State, context: &HcInfo) -> [f32; 64] {
    let mut values = [0.0; 64];
    values[0] = 1.0;
    let status = state.get_timeline_status();
    for (group, lines) in [&status.mandatory, &status.optional, &status.unplayable]
        .into_iter()
        .enumerate()
    {
        let offset = 1 + group * 16;
        for &line in lines {
            let (time, black) = state.get_timeline_end(line);
            let board = state.board(line, time, black);
            let (friendly, hostile) = if black {
                (board.black(), board.white())
            } else {
                (board.white(), board.black())
            };
            let masks = [
                board.lpawn(),
                board.lknight(),
                board.lrook(),
                board.lbishop(),
                board.lunicorn(),
                board.ldragon(),
                board.queen(),
                board.royal(),
            ];
            for (index, mask) in masks.into_iter().enumerate() {
                values[offset + index] += mask.count_ones() as f32;
                values[offset + 8 + index] += ((mask & friendly).count_ones() as i32
                    - (mask & hostile).count_ones() as i32)
                    as f32;
            }
        }
    }

    let (lower, upper) = state.get_lines_range();
    let (origin_lower, origin_upper) = state.get_initial_lines_range();
    let (active_lower, active_upper) = state.get_active_range();
    let total = upper - lower + 1;
    let active = active_upper - active_lower + 1;
    let white_created = upper - origin_upper;
    let black_created = origin_lower - lower;
    let white_active = (active_upper.min(upper) - active_lower.max(origin_upper + 1) + 1).max(0);
    let black_active = (active_upper.min(origin_lower - 1) - active_lower.max(lower) + 1).max(0);
    let order = |white, black| {
        if state.get_present().1 {
            (black, white)
        } else {
            (white, black)
        }
    };
    let (friendly, hostile) = order(white_created, black_created);
    let (friendly_active, hostile_active) = order(white_active, black_active);
    let advantage = hostile - friendly;
    let mandatory = status.mandatory.len() as i32;
    let optional = status.optional.len() as i32;
    let unplayable = status.unplayable.len() as i32;
    values[49..62].copy_from_slice(
        &[
            total,
            active,
            total - active,
            mandatory,
            optional,
            unplayable,
            mandatory + optional,
            friendly,
            hostile,
            advantage,
            (advantage + 1).max(0),
            friendly_active,
            hostile_active,
        ]
        .map(|x| x as f32),
    );

    let mut log_volume = 0.0_f64;
    let mut log_existing_axes = 0.0_f64;
    for axis in 0..context.dimension() {
        let log_size = (context.get_universe()[axis].len() as f64).ln();
        log_volume += log_size;
        if axis < context.first_branch_axis() {
            log_existing_axes += log_size;
        }
    }
    values[62] = log_volume as f32;
    values[63] = log_existing_axes as f32;
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronofish_5d_rules_reference::core::variants::startpos;
    use chronofish_notation::apply_uci_moves;

    #[test]
    fn matches_upstream_linear_features_and_scores() {
        let histories = [
            "",
            "(0T1)e2e4 submit",
            "(0T1)g1f3 submit (0T1)g8f6 submit (0T2)f3(0T1)d3 submit",
        ];
        // Captured from linear_cutoff_evaluation at ftxi commit 664277d.
        let reference = include_str!("../tests/data/linear-reference.txt");
        assert_eq!(histories.len(), reference.lines().count());
        for (index, (history, row)) in histories.into_iter().zip(reference.lines()).enumerate() {
            let mut state = startpos();
            apply_uci_moves(&mut state, history).unwrap();
            let (context, _) = HcInfo::build_hc(&state);
            let expected = row
                .split_whitespace()
                .map(|token| token.parse::<f64>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(expected.len(), 66);
            let actual = features(&state, &context)
                .map(f64::from)
                .into_iter()
                .chain([
                    evaluate(&state, &context, LinearProfile::Default),
                    evaluate(&state, &context, LinearProfile::Trained),
                ]);
            for (column, (actual, expected)) in actual.zip(expected).enumerate() {
                assert!(
                    (actual - expected).abs() < 0.000_002,
                    "position {index}, column {column}: got {actual}, expected {expected}"
                );
            }
        }
    }
}
