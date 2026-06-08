impl Game {
    fn extended_multiverse_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        let own_turn = self.turn_feature_summary(color, weights);
        let opp_turn = self.turn_feature_summary(color.opposite(), weights);
        let own_scores = self.individual_royal_safety_scores(color);
        let opp_scores = self.individual_royal_safety_scores(color.opposite());
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
        let own_urgent_threats = self.urgent_threat_count_for(color);
        let opp_urgent_threats = self.urgent_threat_count_for(color.opposite());
        let own_temporal_lane = self.temporal_lane_control_score_for(color);
        let opp_temporal_lane = self.temporal_lane_control_score_for(color.opposite());
        let own_royal_distance = self.royal_distance_score_for(color);
        let opp_royal_distance = self.royal_distance_score_for(color.opposite());
        let multiverse_size = self.active_timeline_count() + self.playable_board_keys(color).len() as i32;
        let opening_factor = (6 - multiverse_size).max(0);
        let temporal_factor = (multiverse_size - 2).max(0);

        (opp_turn.obligations - own_turn.obligations) * weights.mandatory_move_burden
            + ((own_turn.completion_count - 2) - (opp_turn.completion_count - 2))
                * weights.turn_completion_safety
            + (opp_turn.zugzwang_boards - own_turn.zugzwang_boards) * weights.present_zugzwang
            + (own_weakest - opp_weakest) * weights.weakest_royal_safety
            + (self.royal_liability_score_for(color.opposite()) - self.royal_liability_score_for(color))
                * weights.royal_liability_count
            + (self.multi_royal_attack_score_for(color)
                - self.multi_royal_attack_score_for(color.opposite()))
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
            + (self.historical_access_score_for(color)
                - self.historical_access_score_for(color.opposite()))
                * weights.historical_access
            + (own_temporal_lane - opp_temporal_lane) * weights.temporal_lane_control
            + (self.temporal_pin_score_for(color, weights)
                - self.temporal_pin_score_for(color.opposite(), weights))
                * weights.temporal_pin
            + (self.temporal_skewer_score_for(color, weights)
                - self.temporal_skewer_score_for(color.opposite(), weights))
                * weights.temporal_skewer
            + (self.causal_battery_score_for(color) - self.causal_battery_score_for(color.opposite()))
                * weights.causal_battery
            + (own_turn.safe_arrivals - opp_turn.safe_arrivals) * weights.arrival_square_safety
            + (opp_turn.source_abandonment - own_turn.source_abandonment)
                * weights.source_board_abandonment
            + (self.piece_temporal_flexibility_score_for(color)
                - self.piece_temporal_flexibility_score_for(color.opposite()))
                * weights.piece_temporal_flexibility
            + (self.dimension_coverage_score_for(color)
                - self.dimension_coverage_score_for(color.opposite()))
                * weights.dimension_coverage_balance
            + (own_turn.promotion_choices - opp_turn.promotion_choices)
                * weights.promotion_timeline_choice
            + (own_turn.promotion_checks - opp_turn.promotion_checks) * weights.promotion_with_check
            + (self.past_royal_vulnerability_score_for(color.opposite())
                - self.past_royal_vulnerability_score_for(color))
                * weights.past_royal_vulnerability
            + (self.safe_haven_board_score_for(color) - self.safe_haven_board_score_for(color.opposite()))
                * weights.safe_haven_boards
            + (own_turn.escape_branches - opp_turn.escape_branches) * weights.escape_branch_potential
            + (own_turn.mate_nets - opp_turn.mate_nets) * weights.mate_net_depth_1_2
            + (own_turn.anti_mate_resources - opp_turn.anti_mate_resources)
                * weights.anti_mate_resources
            + (own_turn.check_quality - opp_turn.check_quality) * weights.checking_move_quality
            + ((own_turn.volatility + own_urgent_threats) - (opp_turn.volatility + opp_urgent_threats))
                * weights.search_volatility
            + (self.timeline_repetition_risk_score_for(color.opposite())
                - self.timeline_repetition_risk_score_for(color))
                * weights.timeline_repetition_risk
            + ((self.development_count_for(color) - self.development_count_for(color.opposite()))
                * opening_factor
                + ((own_temporal_lane - opp_temporal_lane) + (own_royal_distance - opp_royal_distance) / 2)
                    * temporal_factor)
                * weights.phase_by_multiverse_size
            + (own_royal_distance - opp_royal_distance) * weights.royal_distance_in_4d
            + (self.board_importance_material_score_for(color, weights)
                - self.board_importance_material_score_for(color.opposite(), weights))
                * weights.board_importance_weight
    }

    fn turn_feature_summary(&self, color: Color, weights: &EvalWeights) -> TurnFeatureSummary {
        let obligations = self.present_obligation_count(color);
        let mut summary = TurnFeatureSummary {
            obligations,
            ..TurnFeatureSummary::default()
        };
        if obligations == 0 {
            return summary;
        }

        let mut search = self.clone_for_search();
        search.turn = color;
        summary.completion_count = search.estimated_turn_completion_count(color, weights, 6) as i32;
        let current_material = search.latest_material_for(color, weights);
        let current_royal_safety = search.royal_safety_for(color, weights);
        let current_temporal_pressure = search.opponent_temporal_tactic_pressure(color, weights);
        let current_timeline_economy = search.timeline_economy_for(color, weights);
        let current_active_timelines = search.active_timeline_count();

        if let Some(present_time) = search.present_time() {
            for (timeline_id, time) in search.playable_board_keys(color) {
                if time != present_time {
                    continue;
                }
                let moves = search.legal_single_moves_from_board(timeline_id, time, color, weights);
                if !moves.is_empty()
                    && moves.iter().all(|movement| {
                        let mut next = search.clone_for_search();
                        next.apply_move_for_search(movement.from, movement.to)
                            && (next.royal_safety_for(color, weights) < current_royal_safety
                                || next.latest_material_for(color, weights) < current_material
                                || next.opponent_temporal_tactic_pressure(color, weights)
                                    > current_temporal_pressure)
                    })
                {
                    summary.zugzwang_boards += 1;
                }
            }
        }

        for movement in search.current_turn_moves_for(color, weights, 48) {
            let Some((piece, _move_kind)) = search.legal_move_kind(movement.from, movement.to) else {
                continue;
            };
            let is_temporal = movement.from.timeline_id != movement.to.timeline_id
                || movement.from.time != movement.to.time;
            let mut next = search.clone_for_search();
            if !next.apply_move_for_search(movement.from, movement.to) {
                continue;
            }

            let next_royal_safety = next.royal_safety_for(color, weights);
            let next_material = next.latest_material_for(color, weights);
            let next_temporal_pressure = next.opponent_temporal_tactic_pressure(color, weights);
            let gives_check = next.is_in_check(color.opposite());
            let makes_mate_net = next.royal_capture_available(color)
                || next.royal_capture_setup_pressure_for_limited(color, weights, 8)
                    > search.royal_capture_setup_pressure_for_limited(color, weights, 8);
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
                let capture_bonus = search
                    .piece_at(movement.to)
                    .map(|target| weights.piece_value(target.piece_type) / 100)
                    .unwrap_or(0);
                let payload = capture_bonus
                    + (next_royal_safety - current_royal_safety).max(0) / 50
                    + gives_check as i32
                    + makes_mate_net as i32
                    + (next.active_timeline_count() > current_active_timelines) as i32;
                if payload > 0 {
                    summary.branch_payload += payload;
                } else {
                    summary.branch_waste += 1;
                }
                if next_royal_safety > current_royal_safety {
                    summary.escape_branches += 1;
                }
                if let Some(arrival) = next.latest_arrival_position(color, movement.to.x, movement.to.y) {
                    if !next.is_square_attacked(arrival, color.opposite()) {
                        summary.safe_arrivals += 1;
                    }
                }
                if movement.from.time == search.present_time().unwrap_or(movement.from.time)
                    && next_royal_safety < current_royal_safety
                {
                    summary.source_abandonment += (current_royal_safety - next_royal_safety) / 50 + 1;
                }
            }

            if gives_check {
                let quality = 1
                    + (next_royal_safety >= current_royal_safety) as i32
                    + (next.timeline_economy_for(color, weights) >= current_timeline_economy) as i32
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
                + (next.active_timeline_count() > current_active_timelines) as i32;
        }

        summary.anti_mate_resources =
            summary.completion_count + summary.safe_move_count + summary.escape_branches
                + self.safe_haven_board_score_for(color);
        summary
    }

    fn present_obligation_count(&self, color: Color) -> i32 {
        let Some(present_time) = self.present_time() else {
            return 0;
        };
        self.timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| timeline.boards.iter().max_by_key(|board| board.time))
            .filter(|board| board.time == present_time && board.side_to_move == color)
            .count() as i32
    }

    fn estimated_turn_completion_count(
        &self,
        color: Color,
        weights: &EvalWeights,
        limit: usize,
    ) -> usize {
        if limit == 0 || !self.has_pending_present_board(color) {
            return 0;
        }
        let max_depth = self
            .timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .count()
            + 1;
        self.estimated_turn_completion_count_at_depth(color, weights, 0, max_depth, limit)
    }

    fn estimated_turn_completion_count_at_depth(
        &self,
        color: Color,
        weights: &EvalWeights,
        depth: usize,
        max_depth: usize,
        limit: usize,
    ) -> usize {
        if limit == 0 {
            return 0;
        }
        if !self.has_pending_present_board(color) {
            return (!self.is_in_check(color)) as usize;
        }
        if depth >= max_depth {
            return 0;
        }

        let mut total = 0;
        for movement in self.prioritized_turn_moves(color, weights, None, MAX_MOVES_PER_NODE) {
            let mut next = self.clone_for_search();
            if !next.apply_move_for_search(movement.from, movement.to) {
                continue;
            }
            total += next.estimated_turn_completion_count_at_depth(
                color,
                weights,
                depth + 1,
                max_depth,
                limit - total,
            );
            if total >= limit {
                break;
            }
        }
        total
    }

    fn current_turn_moves_for(&self, color: Color, weights: &EvalWeights, limit: usize) -> Vec<MoveStep> {
        if !self.has_pending_present_board(color) {
            return Vec::new();
        }
        let mut search = self.clone_for_search();
        search.turn = color;
        let mut moves = search.legal_single_moves_until(weights, None);
        moves.truncate(limit);
        moves
    }

    fn legal_single_moves_from_board(
        &self,
        timeline_id: i32,
        time: i32,
        color: Color,
        weights: &EvalWeights,
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
            for x in 0..8 {
                let from = Position { timeline_id, time, x, y };
                let Some(piece) = self.piece_at(from) else {
                    continue;
                };
                if piece.color != color {
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
                                let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
                                    continue;
                                };
                                if self.allows_search_move(from, to, piece, move_kind) {
                                    moves.push(MoveStep { from, to });
                                }
                            }
                        }
                    }
                }
            }
        }
        moves.sort_by(|left, right| {
            self.cheap_move_order_score(right, weights)
                .cmp(&self.cheap_move_order_score(left, weights))
                .then_with(|| Self::move_cmp(left, right))
        });
        moves
    }

    fn latest_material_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.latest_pieces()
            .into_iter()
            .filter(|(_, piece)| piece.color == color)
            .map(|(_, piece)| weights.piece_value(piece.piece_type))
            .sum()
    }

    fn opponent_temporal_tactic_pressure(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.royal_capture_pressure_for(color.opposite(), weights)
            + self.temporal_royal_corridor_pressure_for(color.opposite(), weights)
    }

    fn royal_liability_score_for(&self, color: Color) -> i32 {
        let royal_scores = self.individual_royal_safety_scores(color);
        royal_scores.iter().filter(|score| **score < 0).count() as i32
            + royal_scores.len().saturating_sub(1) as i32
    }

    fn multi_royal_attack_score_for(&self, color: Color) -> i32 {
        let enemy_royals = self.royal_pieces(color.opposite());
        self.latest_pieces()
            .into_iter()
            .filter(|(_, piece)| piece.color == color)
            .map(|(from, piece)| {
                enemy_royals
                    .iter()
                    .filter(|(target, _)| self.attacks_square(piece, from, *target))
                    .count() as i32
            })
            .filter(|count| *count >= 2)
            .map(|count| count - 1)
            .sum()
    }

    fn urgent_threat_count_for(&self, color: Color) -> i32 {
        let enemy_royals = self.royal_pieces(color.opposite()).len() as i32;
        let enemy_hanging = self
            .latest_pieces()
            .into_iter()
            .filter(|(position, piece)| {
                piece.color == color.opposite()
                    && self.attack_summary(*position, color).count > 0
                    && self.attack_summary(*position, color.opposite()).count == 0
            })
            .count() as i32;
        enemy_royals + enemy_hanging
    }

    fn active_branch_capacity_score_for(&self, color: Color) -> i32 {
        let min_timeline = self.timelines.iter().map(|timeline| timeline.id).min().unwrap_or(0);
        let max_timeline = self.timelines.iter().map(|timeline| timeline.id).max().unwrap_or(0);
        let active_distance = (-min_timeline).min(max_timeline).max(0) + 1;
        let frontier = match color {
            Color::White => max_timeline.max(0),
            Color::Black => (-min_timeline).max(0),
        };
        (active_distance - frontier).max(0)
    }

    fn latent_timeline_reactivation_score_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let min_timeline = self.timelines.iter().map(|timeline| timeline.id).min().unwrap_or(0);
        let max_timeline = self.timelines.iter().map(|timeline| timeline.id).max().unwrap_or(0);
        let active_distance = (-min_timeline).min(max_timeline).max(0) + 1;
        let owner = TimelineOwner::from_color(color);
        self.timelines
            .iter()
            .filter(|timeline| timeline.owner == owner && !self.is_active_timeline(timeline.id))
            .map(|timeline| {
                let distance = (timeline.id.abs() - active_distance).max(1);
                self.timeline_material(timeline.id, weights) / distance
            })
            .sum()
    }

    fn inactive_material_score_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let owner = TimelineOwner::from_color(color);
        self.timelines
            .iter()
            .filter(|timeline| timeline.owner == owner && !self.is_active_timeline(timeline.id))
            .map(|timeline| self.timeline_material(timeline.id, weights) / 100)
            .sum()
    }

    fn timeline_material(&self, timeline_id: i32, weights: &EvalWeights) -> i32 {
        let Some(timeline) = self.timeline(timeline_id) else {
            return 0;
        };
        timeline
            .boards
            .iter()
            .flat_map(|board| board.board.iter().flatten())
            .filter_map(|piece| *piece)
            .map(|piece| weights.piece_value(piece.piece_type))
            .sum()
    }

    fn timeline_compaction_score_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let active_material: i32 = self
            .latest_pieces()
            .into_iter()
            .filter(|(position, piece)| piece.color == color && self.is_active_timeline(position.timeline_id))
            .map(|(_, piece)| weights.piece_value(piece.piece_type) / 100)
            .sum();
        let inactive_material = self.inactive_material_score_for(color, weights);
        let present_material = self.present_board_material(color, weights).max(0);
        active_material + present_material - inactive_material
    }

    fn present_board_material(&self, color: Color, weights: &EvalWeights) -> i32 {
        let Some(present_time) = self.present_time() else {
            return 0;
        };
        self.timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| {
                timeline
                    .boards
                    .iter()
                    .max_by_key(|board| board.time)
                    .map(|board| (timeline, board))
            })
            .filter(|(_, board)| board.time == present_time)
            .map(|(_, board)| {
                board.board
                    .iter()
                    .flatten()
                    .filter_map(|piece| *piece)
                    .map(|piece| {
                        if piece.color == color {
                            weights.piece_value(piece.piece_type) / 100
                        } else {
                            -weights.piece_value(piece.piece_type) / 100
                        }
                    })
                    .sum::<i32>()
            })
            .sum()
    }

    fn historical_access_score_for(&self, color: Color) -> i32 {
        let mut score = 0;
        for (from, piece) in self.latest_pieces() {
            if piece.color != color {
                continue;
            }
            for timeline in &self.timelines {
                for board in &timeline.boards {
                    if self.is_latest_board(timeline.id, board.time) {
                        continue;
                    }
                    for y in 0..8 {
                        for x in 0..8 {
                            let target = Position {
                                timeline_id: timeline.id,
                                time: board.time,
                                x: x as i32,
                                y: y as i32,
                            };
                            if self.attacks_square(piece, from, target) {
                                score += 1;
                                if board.board[y][x]
                                    .is_some_and(|target_piece| Self::is_royal_piece(target_piece.piece_type))
                                {
                                    score += 2;
                                }
                            }
                        }
                    }
                }
            }
        }
        score
    }

    fn temporal_lane_control_score_for(&self, color: Color) -> i32 {
        self.latest_pieces()
            .into_iter()
            .filter(|(_, piece)| piece.color == color)
            .map(|(position, piece)| self.temporal_open_line_count(position, piece))
            .sum()
    }

    fn temporal_open_line_count(&self, position: Position, piece: Piece) -> i32 {
        if !self.is_temporal_slider(piece.piece_type) {
            return 0;
        }
        let directions: &[(i32, i32, i32, i32)] = &[
            (0, 0, 2, 0),
            (0, 0, -2, 0),
            (0, 0, 0, 1),
            (0, 0, 0, -1),
            (1, 0, 2, 0),
            (-1, 0, 2, 0),
            (0, 1, 2, 0),
            (0, -1, 2, 0),
            (1, 0, 0, 1),
            (-1, 0, 0, -1),
        ];
        directions
            .iter()
            .filter(|(dx, dy, dt, dl)| {
                self.first_step_on_line(position, *dx, *dy, *dt, *dl).is_some_and(|target| {
                    self.piece_at(target).is_none() && self.attacks_square(piece, position, target)
                })
            })
            .count() as i32
    }

    fn temporal_pin_score_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.temporal_xray_score_for(color, weights, true)
    }

    fn temporal_skewer_score_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.temporal_xray_score_for(color, weights, false)
    }

    fn temporal_xray_score_for(&self, color: Color, weights: &EvalWeights, pin_mode: bool) -> i32 {
        let enemy_royals = self.royal_pieces(color.opposite());
        let mut score = 0;
        for (from, piece) in self.latest_pieces() {
            if piece.color != color || !self.is_temporal_slider(piece.piece_type) {
                continue;
            }
            for (target, victim) in self.latest_pieces() {
                if victim.color != color.opposite() || Self::is_royal_piece(victim.piece_type) {
                    continue;
                }
                if !self.attacks_square(piece, from, target) {
                    continue;
                }
                let delta = self.movement_delta(from, target);
                if delta.t == 0 && delta.l == 0 {
                    continue;
                }
                let mut cleared = self.clone_for_search();
                cleared.clear_piece_at(target);
                for (royal_position, royal_piece) in &enemy_royals {
                    if !cleared.attacks_square(piece, from, *royal_position) {
                        continue;
                    }
                    if pin_mode {
                        score += 1;
                    } else if weights.piece_value(royal_piece.piece_type)
                        > weights.piece_value(victim.piece_type)
                    {
                        score += 1;
                    }
                }
            }
        }
        score
    }

    fn causal_battery_score_for(&self, color: Color) -> i32 {
        let enemy_royals = self.royal_pieces(color.opposite());
        let own_pieces = self.latest_pieces();
        let mut score = 0;
        for (front_pos, front_piece) in &own_pieces {
            if front_piece.color != color || !self.is_temporal_slider(front_piece.piece_type) {
                continue;
            }
            if !enemy_royals
                .iter()
                .any(|(royal_pos, _)| self.attacks_square(*front_piece, *front_pos, *royal_pos))
            {
                continue;
            }
            for (rear_pos, rear_piece) in &own_pieces {
                if rear_piece.color != color || !self.is_temporal_slider(rear_piece.piece_type) {
                    continue;
                }
                let delta = self.movement_delta(*rear_pos, *front_pos);
                if (delta.t != 0 || delta.l != 0)
                    && self.attacks_square(*rear_piece, *rear_pos, *front_pos)
                {
                    score += 1;
                }
            }
        }
        score
    }

    fn is_temporal_slider(&self, piece_type: PieceType) -> bool {
        matches!(
            piece_type,
            PieceType::Rook
                | PieceType::Bishop
                | PieceType::Unicorn
                | PieceType::Dragon
                | PieceType::Queen
                | PieceType::RoyalQueen
                | PieceType::Princess
        )
    }

    fn piece_temporal_flexibility_score_for(&self, color: Color) -> i32 {
        self.latest_pieces()
            .into_iter()
            .filter(|(_, piece)| piece.color == color)
            .map(|(position, piece)| {
                let mut spatial = false;
                let mut temporal = false;
                for target in self.latest_board_positions() {
                    if !self.attacks_square(piece, position, target) {
                        continue;
                    }
                    let delta = self.movement_delta(position, target);
                    spatial |= delta.x != 0 || delta.y != 0;
                    temporal |= delta.t != 0 || delta.l != 0;
                }
                (spatial && temporal) as i32
            })
            .sum()
    }

    fn dimension_coverage_score_for(&self, color: Color) -> i32 {
        let mut x = 0;
        let mut y = 0;
        let mut t = 0;
        let mut l = 0;
        for (position, piece) in self.latest_pieces() {
            if piece.color != color {
                continue;
            }
            for target in self.latest_board_positions() {
                if !self.attacks_square(piece, position, target) {
                    continue;
                }
                let delta = self.movement_delta(position, target);
                x += (delta.x != 0) as i32;
                y += (delta.y != 0) as i32;
                t += (delta.t != 0) as i32;
                l += (delta.l != 0) as i32;
            }
        }
        [x, y, t, l].into_iter().min().unwrap_or(0)
    }

    fn past_royal_vulnerability_score_for(&self, color: Color) -> i32 {
        self.royal_pieces(color)
            .into_iter()
            .filter(|(position, _)| !self.is_latest_board(position.timeline_id, position.time))
            .map(|(position, _)| self.attack_summary(position, color.opposite()).count)
            .sum()
    }

    fn safe_haven_board_score_for(&self, color: Color) -> i32 {
        self.royal_pieces(color)
            .into_iter()
            .filter(|(position, _)| self.is_latest_board(position.timeline_id, position.time))
            .map(|(position, _)| {
                let shield = self.royal_shield_count(position, color);
                let escapes = self.royal_escape_count(position, color);
                let active = self.is_active_timeline(position.timeline_id) as i32;
                (shield + escapes + active).saturating_sub(1)
            })
            .sum()
    }

    fn royal_distance_score_for(&self, color: Color) -> i32 {
        let enemy_royals = self.royal_pieces(color.opposite());
        self.latest_pieces()
            .into_iter()
            .filter(|(_, piece)| piece.color == color)
            .map(|(from, piece)| {
                enemy_royals
                    .iter()
                    .map(|(target, _)| {
                        let distance = tactical_distance(self.movement_delta(from, *target)).max(1);
                        weights_for_tropism(piece.piece_type) / distance
                    })
                    .max()
                    .unwrap_or(0)
            })
            .sum()
    }

    fn board_importance_material_score_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.timelines
            .iter()
            .flat_map(|timeline| timeline.boards.iter().map(move |board| (timeline.id, board)))
            .map(|(timeline_id, board)| {
                self.board_importance(timeline_id, board)
                    * board
                        .board
                        .iter()
                        .flatten()
                        .filter_map(|piece| *piece)
                        .map(|piece| {
                            if piece.color == color {
                                weights.piece_value(piece.piece_type) / 200
                            } else {
                                -weights.piece_value(piece.piece_type) / 200
                            }
                        })
                        .sum::<i32>()
            })
            .sum()
    }

    fn board_importance(&self, timeline_id: i32, board: &BoardSnapshot) -> i32 {
        let latest = self.is_latest_board(timeline_id, board.time) as i32;
        let active = self.is_active_timeline(timeline_id) as i32;
        let present_distance = self
            .present_time()
            .map(|present| 4 - (board.time - present).abs().min(3))
            .unwrap_or(1);
        let royal_count = board
            .board
            .iter()
            .flatten()
            .filter(|piece| piece.is_some_and(|piece| Self::is_royal_piece(piece.piece_type)))
            .count() as i32;
        latest * 3 + active * 2 + present_distance + royal_count.max(1)
    }

    fn timeline_repetition_risk_score_for(&self, color: Color) -> i32 {
        let owner = TimelineOwner::from_color(color);
        let inactive = self
            .timelines
            .iter()
            .filter(|timeline| timeline.owner == owner && !self.is_active_timeline(timeline.id))
            .count() as i32;
        let mut counts = std::collections::HashMap::new();
        let repeated = self
            .timelines
            .iter()
            .filter(|timeline| timeline.owner == owner)
            .filter_map(|timeline| timeline.boards.iter().max_by_key(|board| board.time))
            .map(|board| {
                let key = Self::board_repetition_key(board);
                let count = counts.entry(key).or_insert(0);
                *count += 1;
                (*count > 1) as i32
            })
            .sum::<i32>();
        inactive + repeated
    }

    fn development_count_for(&self, color: Color) -> i32 {
        self.latest_pieces()
            .into_iter()
            .filter(|(position, piece)| {
                piece.color == color && development(color, piece.piece_type, position.y) > 0
            })
            .count() as i32
    }
}
