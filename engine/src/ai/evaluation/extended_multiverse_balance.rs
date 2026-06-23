use super::*;

#[allow(dead_code)]
impl Game {
    #[allow(dead_code)]
    pub(crate) fn extended_multiverse_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut stats = EvaluationStats::default();
        self.extended_multiverse_balance_with_limits(
            color,
            weights,
            EvaluationLimits::FULL,
            &mut stats,
        )
    }

    pub(crate) fn extended_multiverse_balance_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> i32 {
        let own_turn = self.turn_feature_summary_with_limits(color, weights, limits, stats);
        let opp_turn =
            self.turn_feature_summary_with_limits(color.opposite(), weights, limits, stats);
        let own_scores = self.individual_royal_safety_scores_with_limits(color, limits, stats);
        let opp_scores =
            self.individual_royal_safety_scores_with_limits(color.opposite(), limits, stats);
        let own_weakest = own_scores
            .iter()
            .copied()
            .min()
            .map(|worst| (worst * 2 + own_scores.iter().sum::<i32>() / own_scores.len() as i32) / 3)
            .unwrap_or(0);
        let opp_weakest = opp_scores
            .iter()
            .copied()
            .min()
            .map(|worst| (worst * 2 + opp_scores.iter().sum::<i32>() / opp_scores.len() as i32) / 3)
            .unwrap_or(0);
        let own_urgent_threats = self.urgent_threat_count_for_with_limits(color, limits, stats);
        let opp_urgent_threats =
            self.urgent_threat_count_for_with_limits(color.opposite(), limits, stats);
        let own_temporal_lane =
            self.temporal_lane_control_score_for_with_limits(color, limits, stats);
        let opp_temporal_lane =
            self.temporal_lane_control_score_for_with_limits(color.opposite(), limits, stats);
        let own_royal_distance = self.royal_distance_score_for(color);
        let opp_royal_distance = self.royal_distance_score_for(color.opposite());
        let multiverse_size =
            self.active_timeline_count() + self.playable_board_keys(color).len() as i32;
        let opening_factor = (6 - multiverse_size).max(0);
        let temporal_factor = (multiverse_size - 2).max(0);

        (opp_turn.obligations - own_turn.obligations) * weights.mandatory_move_burden
            + ((own_turn.completion_count - 2) - (opp_turn.completion_count - 2))
                * weights.turn_completion_safety
            + (opp_turn.zugzwang_boards - own_turn.zugzwang_boards) * weights.present_zugzwang
            + (own_weakest - opp_weakest) * weights.weakest_royal_safety
            + (self.royal_liability_score_for_with_limits(color.opposite(), limits, stats)
                - self.royal_liability_score_for_with_limits(color, limits, stats))
                * weights.royal_liability_count
            + (self.multi_royal_attack_score_for_with_limits(color, limits, stats)
                - self.multi_royal_attack_score_for_with_limits(color.opposite(), limits, stats))
                * weights.multi_royal_attack
            + (own_turn.safe_move_count - opp_turn.safe_move_count) * weights.defensive_bandwidth
            + ((own_urgent_threats - opp_turn.safe_move_count - opp_turn.obligations)
                - (opp_urgent_threats - own_turn.safe_move_count - own_turn.obligations))
                * weights.threat_overload
            + (self.active_branch_capacity_score_for(color)
                - self.active_branch_capacity_score_for(color.opposite()))
                * weights.active_branch_capacity
            + (self.latent_timeline_reactivation_score_for(color, weights)
                - self.latent_timeline_reactivation_score_for(color.opposite(), weights))
                * weights.latent_timeline_reactivation
            + (self.inactive_material_score_for(color, weights)
                - self.inactive_material_score_for(color.opposite(), weights))
                * weights.inactive_material_quality
            + (own_turn.branch_payload - opp_turn.branch_payload) * weights.branch_payload
            + (opp_turn.branch_waste - own_turn.branch_waste) * weights.branch_waste
            + (self.timeline_compaction_score_for(color, weights)
                - self.timeline_compaction_score_for(color.opposite(), weights))
                * weights.timeline_compaction
            + (self.latest_material_for(color, weights)
                - self.latest_material_for(color.opposite(), weights))
                * weights.frontier_material
            + (self.historical_access_score_for_with_limits(color, limits, stats)
                - self.historical_access_score_for_with_limits(color.opposite(), limits, stats))
                * weights.historical_access
            + (own_temporal_lane - opp_temporal_lane) * weights.temporal_lane_control
            + (self.temporal_pin_score_for_with_limits(color, weights, limits, stats)
                - self.temporal_pin_score_for_with_limits(color.opposite(), weights, limits, stats))
                * weights.temporal_pin
            + (self.temporal_skewer_score_for_with_limits(color, weights, limits, stats)
                - self.temporal_skewer_score_for_with_limits(
                    color.opposite(),
                    weights,
                    limits,
                    stats,
                ))
                * weights.temporal_skewer
            + (self.causal_battery_score_for_with_limits(color, limits, stats)
                - self.causal_battery_score_for_with_limits(color.opposite(), limits, stats))
                * weights.causal_battery
            + (own_turn.safe_arrivals - opp_turn.safe_arrivals) * weights.arrival_square_safety
            + (opp_turn.source_abandonment - own_turn.source_abandonment)
                * weights.source_board_abandonment
            + (self.piece_temporal_flexibility_score_for_with_limits(color, limits, stats)
                - self.piece_temporal_flexibility_score_for_with_limits(
                    color.opposite(),
                    limits,
                    stats,
                ))
                * weights.piece_temporal_flexibility
            + (self.dimension_coverage_score_for_with_limits(color, limits, stats)
                - self.dimension_coverage_score_for_with_limits(color.opposite(), limits, stats))
                * weights.dimension_coverage_balance
            + (own_turn.promotion_choices - opp_turn.promotion_choices)
                * weights.promotion_timeline_choice
            + (own_turn.promotion_checks - opp_turn.promotion_checks) * weights.promotion_with_check
            + (self.past_royal_vulnerability_score_for_with_limits(color.opposite(), limits, stats)
                - self.past_royal_vulnerability_score_for_with_limits(color, limits, stats))
                * weights.past_royal_vulnerability
            + (self.safe_haven_board_score_for(color)
                - self.safe_haven_board_score_for(color.opposite()))
                * weights.safe_haven_boards
            + (own_turn.escape_branches - opp_turn.escape_branches)
                * weights.escape_branch_potential
            + (own_turn.mate_nets - opp_turn.mate_nets) * weights.mate_net_depth_1_2
            + (own_turn.anti_mate_resources - opp_turn.anti_mate_resources)
                * weights.anti_mate_resources
            + (own_turn.check_quality - opp_turn.check_quality) * weights.checking_move_quality
            + ((own_turn.volatility + own_urgent_threats)
                - (opp_turn.volatility + opp_urgent_threats))
                * weights.search_volatility
            + (self.timeline_repetition_risk_score_for(color.opposite())
                - self.timeline_repetition_risk_score_for(color))
                * weights.timeline_repetition_risk
            + ((self.development_count_for(color) - self.development_count_for(color.opposite()))
                * opening_factor
                + ((own_temporal_lane - opp_temporal_lane)
                    + (own_royal_distance - opp_royal_distance) / 2)
                    * temporal_factor)
                * weights.phase_by_multiverse_size
            + (own_royal_distance - opp_royal_distance) * weights.royal_distance_in_4d
            + (self.board_importance_material_score_for(color, weights)
                - self.board_importance_material_score_for(color.opposite(), weights))
                * weights.board_importance_weight
    }

    #[allow(dead_code)]
    pub(crate) fn turn_feature_summary(
        &self,
        color: Color,
        weights: &EvalWeights,
    ) -> TurnFeatureSummary {
        let mut stats = EvaluationStats::default();
        self.turn_feature_summary_with_limits(color, weights, EvaluationLimits::FULL, &mut stats)
    }

    pub(crate) fn turn_feature_summary_with_limits(
        &self,
        color: Color,
        weights: &EvalWeights,
        limits: EvaluationLimits,
        stats: &mut EvaluationStats,
    ) -> TurnFeatureSummary {
        let obligations = self.present_obligation_count(color);
        let mut summary = TurnFeatureSummary {
            obligations,
            ..TurnFeatureSummary::default()
        };
        if obligations == 0 {
            return summary;
        }
        if deadline_expired(limits.deadline) {
            return summary;
        }

        let mut search = self.clone_for_search();
        stats.clones += 1;
        search.turn = color;
        summary.completion_count = search.estimated_turn_completion_count_with_limit_until(
            color,
            weights,
            limits.completion_results,
            limits.deadline,
        ) as i32;
        if deadline_expired(limits.deadline) {
            return summary;
        }
        let current_material = search.latest_material_for(color, weights);
        let current_royal_safety = search.royal_safety_for(color, weights);
        let current_temporal_pressure =
            search.opponent_temporal_tactic_pressure_with_limits(color, weights, limits, stats);
        let current_timeline_economy = search.timeline_economy_for(color, weights);
        let current_active_timelines = search.active_timeline_count();
        let current_royal_capture_setup = search.royal_capture_setup_pressure_bounded(
            color,
            weights,
            limits,
            limits.setup_results,
            limits.setup_probes,
            stats,
        );
        let present_time = search.present_time();

        if let Some(present_time) = present_time {
            for (timeline_id, time) in search.playable_board_keys(color) {
                if time != present_time {
                    continue;
                }
                let moves = search.legal_single_moves_from_board(
                    timeline_id,
                    time,
                    color,
                    weights,
                    limits.zugzwang_moves_per_board,
                    limits.deadline,
                );
                let mut all_moves_are_bad = !moves.is_empty();
                for movement in moves {
                    if deadline_expired(limits.deadline) {
                        return summary;
                    }
                    let Some(undo) = search.make_search_move(movement) else {
                        all_moves_are_bad = false;
                        break;
                    };
                    stats.turn_moves += 1;
                    let move_is_bad = search.royal_safety_for(color, weights)
                        < current_royal_safety
                        || search.latest_material_for(color, weights) < current_material
                        || search.opponent_temporal_tactic_pressure_with_limits(
                            color, weights, limits, stats,
                        ) > current_temporal_pressure;
                    search.unmake_search_move(undo);
                    if !move_is_bad {
                        all_moves_are_bad = false;
                        break;
                    }
                }
                if all_moves_are_bad {
                    summary.zugzwang_boards += 1;
                }
            }
        }

        for movement in
            search.current_turn_moves_for(color, weights, limits.turn_moves, limits.deadline)
        {
            if deadline_expired(limits.deadline) {
                return summary;
            }
            let Some((piece, _move_kind)) = search.legal_move_kind(movement.from, movement.to)
            else {
                continue;
            };
            let is_temporal = movement.from.timeline_id != movement.to.timeline_id
                || movement.from.time != movement.to.time;
            let capture_bonus = search
                .piece_at(movement.to)
                .map(|target| weights.piece_value(target.piece_type) / 100)
                .unwrap_or(0);
            let Some(undo) = search.make_search_move(movement) else {
                continue;
            };
            stats.turn_moves += 1;

            let next_royal_safety = search.royal_safety_for(color, weights);
            let next_material = search.latest_material_for(color, weights);
            let next_temporal_pressure =
                search.opponent_temporal_tactic_pressure_with_limits(color, weights, limits, stats);
            let gives_check = search.is_in_check(color.opposite());
            let makes_mate_net = search.royal_capture_available(color)
                || search.royal_capture_setup_pressure_bounded(
                    color,
                    weights,
                    limits,
                    limits.setup_results,
                    limits.setup_probes,
                    stats,
                ) > current_royal_capture_setup;
            let next_active_timelines = search.active_timeline_count();
            let is_safe_move = next_royal_safety >= current_royal_safety
                && next_material >= current_material
                && next_temporal_pressure <= current_temporal_pressure;
            if is_safe_move {
                summary.safe_move_count += 1;
            }
            if gives_check {
                summary.checking_moves += 1;
            }
            if makes_mate_net {
                summary.mate_nets += 1;
            }

            if is_temporal {
                summary.branch_moves += 1;
                let source_abandonment =
                    self.source_material_abandonment_cost(movement.from, piece, weights);
                let payload = capture_bonus
                    + (next_royal_safety - current_royal_safety).max(0) / 50
                    + gives_check as i32
                    + makes_mate_net as i32
                    + (next_active_timelines > current_active_timelines) as i32;
                if payload > 0 {
                    summary.branch_payload += payload;
                } else {
                    summary.branch_waste += 1;
                }
                if next_royal_safety > current_royal_safety {
                    summary.escape_branches += 1;
                }
                if let Some(arrival) =
                    search.latest_arrival_position(color, movement.to.x, movement.to.y)
                {
                    if !search.is_square_attacked(arrival, color.opposite()) {
                        summary.safe_arrivals += 1;
                    }
                }
                if movement.from.time == present_time.unwrap_or(movement.from.time)
                    && next_royal_safety < current_royal_safety
                {
                    summary.source_abandonment +=
                        (current_royal_safety - next_royal_safety) / 50 + 1;
                }
                summary.source_abandonment += source_abandonment;
            }

            if gives_check {
                let quality = 1
                    + (next_royal_safety >= current_royal_safety) as i32
                    + (search.timeline_economy_for(color, weights) >= current_timeline_economy)
                        as i32
                    + makes_mate_net as i32;
                summary.check_quality += quality;
            }

            if matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
                && (movement.to.y == 0 || movement.to.y == 7)
            {
                summary.promotion_choices += 1 + is_temporal as i32;
                if gives_check || makes_mate_net {
                    summary.promotion_checks += 1;
                }
            }

            summary.volatility += is_temporal as i32
                + gives_check as i32
                + makes_mate_net as i32
                + (next_active_timelines > current_active_timelines) as i32;
            search.unmake_search_move(undo);
        }

        summary.anti_mate_resources = summary.completion_count
            + summary.safe_move_count
            + summary.escape_branches
            + self.safe_haven_board_score_for(color);
        summary
    }

    pub(crate) fn present_obligation_count(&self, color: Color) -> i32 {
        let Some(present_time) = self.present_time() else {
            return 0;
        };
        self.timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| timeline.boards.last())
            .filter(|board| board.time == present_time && board.side_to_move == color)
            .count() as i32
    }

    #[allow(dead_code)]
    pub(crate) fn estimated_turn_completion_count(
        &self,
        color: Color,
        weights: &EvalWeights,
        limit: usize,
    ) -> usize {
        let mut search = self.clone_for_search();
        search.estimated_turn_completion_count_with_limit(color, weights, limit)
    }

    pub(crate) fn estimated_turn_completion_count_with_limit(
        &mut self,
        color: Color,
        weights: &EvalWeights,
        limit: usize,
    ) -> usize {
        self.estimated_turn_completion_count_with_limit_until(color, weights, limit, None)
    }

    pub(crate) fn estimated_turn_completion_count_with_limit_until(
        &mut self,
        color: Color,
        weights: &EvalWeights,
        limit: usize,
        deadline: Option<SearchInstant>,
    ) -> usize {
        if limit == 0 || deadline_expired(deadline) || !self.has_pending_present_board(color) {
            return 0;
        }
        let max_depth = self
            .timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .count()
            + 1;
        self.estimated_turn_completion_count_at_depth(color, weights, 0, max_depth, limit, deadline)
    }

    pub(crate) fn estimated_turn_completion_count_at_depth(
        &mut self,
        color: Color,
        weights: &EvalWeights,
        depth: usize,
        max_depth: usize,
        limit: usize,
        deadline: Option<SearchInstant>,
    ) -> usize {
        if limit == 0 || deadline_expired(deadline) {
            return 0;
        }
        if !self.has_pending_present_board(color) {
            return (!self.is_in_check(color)) as usize;
        }
        if depth >= max_depth {
            return 0;
        }

        let mut total = 0;
        for movement in
            self.prioritized_turn_moves_for_evaluation(color, weights, deadline, MAX_MOVES_PER_NODE)
        {
            if deadline_expired(deadline) {
                break;
            }
            let Some(undo) = self.make_search_move(movement) else {
                continue;
            };
            total += self.estimated_turn_completion_count_at_depth(
                color,
                weights,
                depth + 1,
                max_depth,
                limit - total,
                deadline,
            );
            self.unmake_search_move(undo);
            if total >= limit {
                break;
            }
        }
        total
    }

    pub(crate) fn current_turn_moves_for(
        &mut self,
        color: Color,
        weights: &EvalWeights,
        limit: usize,
        deadline: Option<SearchInstant>,
    ) -> Vec<MoveStep> {
        if deadline_expired(deadline) || !self.has_pending_present_board(color) {
            return Vec::new();
        }
        let previous_turn = self.turn;
        self.turn = color;
        if limit == EvaluationLimits::FULL.turn_moves {
            let mut moves = self.legal_single_moves_until(weights, deadline);
            moves.truncate(limit);
            self.turn = previous_turn;
            return moves;
        }
        let moves = self.sampled_current_turn_moves(weights, limit, deadline);
        self.turn = previous_turn;
        moves
    }

    fn sampled_current_turn_moves(
        &self,
        weights: &EvalWeights,
        limit: usize,
        deadline: Option<SearchInstant>,
    ) -> Vec<MoveStep> {
        let mut moves = Vec::with_capacity(limit);
        for timeline in &self.timelines {
            if deadline_expired(deadline) {
                return self.order_moves(moves, weights);
            }
            if !self.is_active_timeline(timeline.id) {
                continue;
            }
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            if board.side_to_move != self.turn {
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
                    let Some(piece) = self.piece_at(from).filter(|piece| piece.color == self.turn)
                    else {
                        continue;
                    };
                    self.for_each_piece_candidate_destination(from, piece, |to| {
                        if deadline_expired(deadline) {
                            return false;
                        }
                        let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
                            return true;
                        };
                        if self.allows_search_move(from, to, piece, move_kind) {
                            let movement = MoveStep { from, to };
                            if !moves.contains(&movement) {
                                moves.push(movement);
                            }
                            if moves.len() >= limit {
                                return false;
                            }
                        }
                        true
                    });
                    if deadline_expired(deadline) || moves.len() >= limit {
                        return self.order_moves(moves, weights);
                    }
                }
            }
        }
        self.order_moves(moves, weights)
    }

    pub(crate) fn legal_single_moves_from_board(
        &self,
        timeline_id: i32,
        time: i32,
        color: Color,
        weights: &EvalWeights,
        limit: usize,
        deadline: Option<SearchInstant>,
    ) -> Vec<MoveStep> {
        let Some(board) = self.board(timeline_id, time) else {
            return Vec::new();
        };
        if !self.is_active_timeline(timeline_id)
            || !self.is_latest_board(timeline_id, time)
            || board.side_to_move != color
        {
            return Vec::new();
        }

        let mut moves = Vec::new();
        for y in 0..8 {
            if deadline_expired(deadline) {
                return self.order_moves(moves, weights);
            }
            for x in 0..8 {
                let from = Position {
                    timeline_id,
                    time,
                    x,
                    y,
                };
                let Some(piece) = self.piece_at(from) else {
                    continue;
                };
                if piece.color != color {
                    continue;
                }
                self.for_each_piece_candidate_destination(from, piece, |to| {
                    if deadline_expired(deadline) {
                        return false;
                    }
                    let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
                        return true;
                    };
                    if self.allows_search_move(from, to, piece, move_kind) {
                        let movement = MoveStep { from, to };
                        if !moves.contains(&movement) {
                            moves.push(movement);
                        }
                        if moves.len() >= limit {
                            return false;
                        }
                    }
                    true
                });
                if deadline_expired(deadline) || moves.len() >= limit {
                    return self.order_moves(moves, weights);
                }
            }
        }
        self.order_moves(moves, weights)
    }
}
