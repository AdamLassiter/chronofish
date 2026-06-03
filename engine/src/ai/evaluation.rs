#[allow(dead_code)]
impl Game {
    fn evaluate(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.evaluate_heuristic(color, weights)
    }

    fn evaluate_heuristic(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.pruned_for_evaluation()
            .evaluate_heuristic_without_pruning(color, weights)
    }

    fn evaluate_heuristic_without_pruning(&self, color: Color, weights: &EvalWeights) -> i32 {
        if let Some(score) = self.terminal_score(color) {
            return score;
        }

        // Only latest boards contain live material. Historical boards are context
        // for time-travel legality, not extra material to score.
        let mut score = 0;
        for timeline in &self.timelines {
            let active = self.is_active_timeline(timeline.id);
            score += if active {
                weights.active_timeline
            } else {
                weights.inactive_timeline
            } * owner_factor(timeline.owner, color);

            let Some(board) = timeline.boards.iter().max_by_key(|board| board.time) else {
                continue;
            };
            for (y, rank) in board.board.iter().enumerate() {
                for (x, piece) in rank.iter().enumerate() {
                    let Some(piece) = piece else {
                        continue;
                    };
                    let value = weights.piece_value(piece.piece_type);
                    let positional = weights.advancement * advancement(piece.color, y as i32)
                        + weights.centrality * centrality(x as i32, y as i32);
                    let development =
                        weights.development * development(piece.color, piece.piece_type, y as i32);
                    score += if piece.color == color {
                        value + positional + development
                    } else {
                        -value - positional - development
                    };
                }
            }
        }

        if self.is_in_check(color) {
            score -= weights.check_penalty;
        }
        if self.is_in_check(color.opposite()) {
            score += weights.check_penalty;
        }
        score
            + self.present_progress(color) * weights.present_progress
            + self.strategic_balance(color, weights)
            + self.timeline_coordination(color, weights)
            + self.royal_capture_pressure(color, weights)
            + self.temporal_royal_corridor_balance(color, weights)
            + self.royal_capture_setup_balance(color, weights)
            + self.royal_safety_balance(color, weights)
            + self.fork_pressure_balance(color, weights)
            + self.forcing_pressure_balance(color, weights)
            + self.board_control_balance(color, weights)
            + self.piece_activity_balance(color, weights)
            + self.pawn_structure_balance(color, weights)
            + self.timeline_economy_balance(color, weights)
            + self.present_tempo_balance(color, weights)
            + self.royal_shelter_balance(color, weights)
            + self.space_advantage_balance(color, weights)
            + if weights.mobility == 0 {
                0
            } else {
                self.mobility_balance(color) * weights.mobility
            }
    }

    fn terminal_score(&self, color: Color) -> Option<i32> {
        if self.staged_royal_capture_by == Some(color) {
            Some(CHECKMATE_SCORE)
        } else if self.staged_royal_capture_by == Some(color.opposite()) {
            Some(-CHECKMATE_SCORE)
        } else if self.has_threefold_repetition() || self.is_classic_stalemate(self.turn) {
            Some(0)
        } else {
            None
        }
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
        // Reward advancing active timelines while this color controls the
        // present line; penalize the same advance when it hands tempo away.
        factor * (latest_sum - present.time)
    }

    fn mobility_balance(&self, color: Color) -> i32 {
        self.legal_single_move_count_for(color)
            - self.legal_single_move_count_for(color.opposite())
    }

    fn legal_single_move_count_for(&self, color: Color) -> i32 {
        let mut count = 0;
        for timeline in &self.timelines {
            if !self.is_active_timeline(timeline.id) {
                continue;
            }
            for board in &timeline.boards {
                if !self.is_latest_board(timeline.id, board.time) || board.side_to_move != color {
                    continue;
                }
                for y in 0..8 {
                    for x in 0..8 {
                        let Some(piece) = board.board[y][x] else {
                            continue;
                        };
                        if piece.color != color {
                            continue;
                        }
                        let from = Position {
                            timeline_id: timeline.id,
                            time: board.time,
                            x: x as i32,
                            y: y as i32,
                        };
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
                                        let Some((piece, move_kind)) =
                                            self.legal_move_kind_for_turn(color, from, to)
                                        else {
                                            continue;
                                        };
                                        if self.allows_search_move(from, to, piece, move_kind) {
                                            count += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        count
    }

    fn legal_move_kind_for_turn(
        &self,
        turn: Color,
        from: Position,
        to: Position,
    ) -> Option<(Piece, MoveKind)> {
        if !Self::in_bounds(from.x, from.y) || !Self::in_bounds(to.x, to.y) {
            return None;
        }

        let source_board = self.board(from.timeline_id, from.time)?;
        let target_board = self.board(to.timeline_id, to.time)?;
        let piece = source_board.board[from.y as usize][from.x as usize]?;
        let same_board = from.timeline_id == to.timeline_id && from.time == to.time;

        if !self.is_present_source_board(from)
            || !self.is_latest_board(from.timeline_id, from.time)
            || source_board.side_to_move != turn
            || piece.color != turn
        {
            return None;
        }

        if !same_board && target_board.side_to_move != piece.color {
            return None;
        }

        if target_board.board[to.y as usize][to.x as usize]
            .is_some_and(|target| target.color == piece.color)
        {
            return None;
        }

        self.move_kind_for(piece, from, to)
            .map(|kind| (piece, kind))
    }

    fn strategic_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (position, piece) in self.latest_pieces() {
            let attackers = self.attack_summary(position, piece.color.opposite());
            let defenders = self.attack_summary(position, piece.color);
            let value = weights.piece_value(piece.piece_type);
            let mut piece_score = 0;

            if defenders.count > 0 {
                piece_score += weights.defended_piece;
            }
            if attackers.count > 0 {
                piece_score -= weights.attacked_piece + value / 32;
            }
            if attackers.count > 0 && defenders.count == 0 {
                piece_score -= weights.hanging_piece + value / 16;
            }
            if attackers.count > 0 && Self::is_royal_piece(piece.piece_type) {
                piece_score -= weights.royal_threat;
            }
            if attackers.temporal_count > 0 {
                piece_score -= weights.temporal_threat * attackers.temporal_count;
            }
            if attackers.count >= 2 {
                piece_score -= weights.pincer_threat * (attackers.count - 1);
            }
            if attackers.timeline_count >= 2 {
                piece_score -= weights.timeline_pincer * (attackers.timeline_count - 1);
            }
            if attackers.time_count >= 2 {
                piece_score -= weights.historical_pincer * (attackers.time_count - 1);
            }

            score += if piece.color == color {
                piece_score
            } else {
                -piece_score
            };
        }
        score
    }

    fn board_control_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.board_control_for(color, weights) - self.board_control_for(color.opposite(), weights)
    }

    fn board_control_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (from, piece) in self.latest_pieces() {
            if piece.color != color {
                continue;
            }
            let mut controlled = 0;
            let mut central = 0;
            let mut royal_zone = 0;
            for target in self.latest_board_positions() {
                if !self.attacks_square(piece, from, target) {
                    continue;
                }
                controlled += 1;
                central += centrality(target.x, target.y).max(0);
                if self.near_enemy_royal(target, color) {
                    royal_zone += 1;
                }
            }
            score += controlled * weights.board_control
                + central * weights.board_control / 8
                + royal_zone * weights.royal_threat / 4;
        }
        score
    }

    fn piece_activity_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.piece_activity_for(color, weights) - self.piece_activity_for(color.opposite(), weights)
    }

    fn piece_activity_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (position, piece) in self.latest_pieces() {
            if piece.color != color || Self::is_royal_piece(piece.piece_type) {
                continue;
            }
            let mobility = self.pseudo_attack_count(position, piece).min(24);
            let activity = match piece.piece_type {
                PieceType::Pawn | PieceType::Brawn => mobility / 2,
                PieceType::Knight => mobility,
                PieceType::Bishop | PieceType::Rook | PieceType::Unicorn | PieceType::Dragon => {
                    mobility + self.open_line_count(position, piece) * 2
                }
                PieceType::Queen | PieceType::Princess | PieceType::RoyalQueen => mobility + 2,
                PieceType::King | PieceType::CommonKing => 0,
            };
            score += activity * weights.piece_activity;
            if mobility <= 1 {
                score -= weights.piece_activity * 3;
            }
        }
        score
    }

    fn pawn_structure_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.pawn_structure_for(color, weights) - self.pawn_structure_for(color.opposite(), weights)
    }

    fn pawn_structure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (position, piece) in self.latest_pieces() {
            if piece.color != color || !matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
            {
                continue;
            }
            let forward = if color == Color::White { 1 } else { -1 };
            let advance = advancement(color, position.y);
            score += advance * weights.space_advantage;
            if self.is_passed_pawn(position, color) {
                score += weights.pawn_structure * (advance + 1);
            }
            if self.is_supported_pawn(position, color) {
                score += weights.pawn_structure;
            }
            if self.is_isolated_pawn(position, color) {
                score -= weights.pawn_structure;
            }
            let ahead = Position {
                timeline_id: position.timeline_id,
                time: position.time,
                x: position.x,
                y: position.y + forward,
            };
            if Self::in_bounds(ahead.x, ahead.y) && self.piece_at(ahead).is_some() {
                score -= weights.pawn_structure;
            }
        }
        score
    }

    fn timeline_economy_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.timeline_economy_for(color, weights) - self.timeline_economy_for(color.opposite(), weights)
    }

    fn timeline_economy_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let owner = match color {
            Color::White => TimelineOwner::White,
            Color::Black => TimelineOwner::Black,
        };
        let own_active = self
            .timelines
            .iter()
            .filter(|timeline| timeline.owner == owner && self.is_active_timeline(timeline.id))
            .count() as i32;
        let own_inactive = self
            .timelines
            .iter()
            .filter(|timeline| timeline.owner == owner && !self.is_active_timeline(timeline.id))
            .count() as i32;
        let active_material: i32 = self
            .latest_pieces()
            .into_iter()
            .filter(|(position, piece)| piece.color == color && self.is_active_timeline(position.timeline_id))
            .map(|(_, piece)| weights.piece_value(piece.piece_type) / 200)
            .sum();
        own_active * weights.timeline_economy + active_material - own_inactive * weights.timeline_economy * 2
    }

    fn present_tempo_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        let Some(present) = self.present_board() else {
            return 0;
        };
        let factor = if present.side_to_move == color { 1 } else { -1 };
        let spread = self.timeline_time_spread();
        factor * (weights.present_tempo * (spread + 1))
    }

    fn royal_shelter_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.royal_shelter_for(color, weights) - self.royal_shelter_for(color.opposite(), weights)
    }

    fn royal_shelter_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (position, _) in self.royal_pieces(color) {
            if !self.is_latest_board(position.timeline_id, position.time) {
                continue;
            }
            let forward = if color == Color::White { 1 } else { -1 };
            for dx in -1..=1 {
                let shield = Position {
                    timeline_id: position.timeline_id,
                    time: position.time,
                    x: position.x + dx,
                    y: position.y + forward,
                };
                if Self::in_bounds(shield.x, shield.y)
                    && self.piece_at(shield).is_some_and(|piece| {
                        piece.color == color && matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
                    })
                {
                    score += weights.royal_shelter;
                } else {
                    score -= weights.royal_shelter / 2;
                }
            }
        }
        score
    }

    fn space_advantage_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.space_advantage_for(color, weights) - self.space_advantage_for(color.opposite(), weights)
    }

    fn space_advantage_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.latest_pieces()
            .into_iter()
            .filter(|(_, piece)| piece.color == color)
            .map(|(position, piece)| {
                let value_scale = if matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn) {
                    2
                } else {
                    1
                };
                advancement(color, position.y) * weights.space_advantage * value_scale
            })
            .sum()
    }

    fn royal_capture_pressure(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.royal_capture_pressure_for(color, weights)
            - self.royal_capture_pressure_for(color.opposite(), weights)
    }

    fn royal_capture_pressure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        let royal_targets = self.royal_pieces(color.opposite());
        for (from, piece) in self.latest_pieces() {
            if piece.color != color {
                continue;
            }
            for (target, _) in &royal_targets {
                if self.attacks_square(piece, from, *target) {
                    let distance = tactical_distance(self.movement_delta(from, *target));
                    let urgency = 6_i32.saturating_sub(distance.min(6)).max(1);
                    score += weights.royal_capture_threat * urgency;
                    if from.timeline_id != target.timeline_id || from.time != target.time {
                        score += weights.temporal_threat * urgency;
                    }
                }
            }
        }
        score
    }

    fn royal_capture_setup_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.royal_capture_setup_pressure_for(color, weights)
            - self.royal_capture_setup_pressure_for(color.opposite(), weights)
    }

    fn royal_capture_setup_pressure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.royal_capture_setup_pressure_for_limited(color, weights, 48)
    }

    fn royal_capture_setup_pressure_for_limited(
        &self,
        color: Color,
        weights: &EvalWeights,
        limit: usize,
    ) -> i32 {
        if weights.royal_capture_setup == 0 || self.royal_capture_available(color) {
            return 0;
        }

        let mut score = 0;
        let mut counted = 0;

        for (from, piece) in self.latest_pieces() {
            if piece.color != color || matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
            {
                continue;
            }

            for y in 0..8 {
                for x in 0..8 {
                    if counted >= limit {
                        break;
                    }
                    let to = Position {
                        timeline_id: from.timeline_id,
                        time: from.time,
                        x,
                        y,
                    };
                    if self
                        .piece_at(to)
                        .is_some_and(|target| target.color == color)
                        || self.move_kind_for(piece, from, to).is_none()
                    {
                        continue;
                    }

                    let arrival = Position {
                        time: from.time + 1,
                        ..to
                    };
                    let corridor_pressure =
                        self.temporal_royal_corridor_from(piece, arrival, weights);
                    if corridor_pressure <= 0 {
                        continue;
                    }

                    counted += 1;
                    let major_piece_bonus = match piece.piece_type {
                        PieceType::Queen | PieceType::RoyalQueen => weights.royal_capture_setup / 2,
                        PieceType::Bishop
                        | PieceType::Rook
                        | PieceType::Unicorn
                        | PieceType::Dragon
                        | PieceType::Princess => weights.royal_capture_setup / 3,
                        _ => 0,
                    };
                    let capture_bonus = self
                        .piece_at(to)
                        .map(|target| weights.piece_value(target.piece_type) / 20)
                        .unwrap_or(0);

                    score += weights.royal_capture_setup
                        + major_piece_bonus
                        + capture_bonus
                        + corridor_pressure;
                }
            }
            if counted >= limit {
                break;
            }
        }

        score
    }

    fn temporal_royal_corridor_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.temporal_royal_corridor_pressure_for(color, weights)
            - self.temporal_royal_corridor_pressure_for(color.opposite(), weights)
    }

    fn temporal_royal_corridor_pressure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        if weights.royal_capture_setup == 0 {
            return 0;
        }

        let royal_targets = self.royal_pieces(color.opposite());
        let mut score = 0;
        for (from, piece) in self.latest_pieces() {
            if piece.color != color || matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
            {
                continue;
            }
            score += self.temporal_royal_corridor_from_with_targets(
                piece,
                from,
                &royal_targets,
                weights,
            );
        }

        score
    }

    fn temporal_royal_corridor_from(
        &self,
        piece: Piece,
        from: Position,
        weights: &EvalWeights,
    ) -> i32 {
        let royal_targets = self.royal_pieces(piece.color.opposite());
        self.temporal_royal_corridor_from_with_targets(piece, from, &royal_targets, weights)
    }

    fn temporal_royal_corridor_from_with_targets(
        &self,
        piece: Piece,
        from: Position,
        royal_targets: &[(Position, Piece)],
        weights: &EvalWeights,
    ) -> i32 {
        let mut score = 0;
        for (target, _) in royal_targets {
            if from.timeline_id == target.timeline_id
                && from.time == target.time
                && from.x == target.x
                && from.y == target.y
            {
                continue;
            }

            for wait in 1..=4 {
                let future_from = Position {
                    time: from.time + wait,
                    ..from
                };
                if future_from.time <= target.time {
                    continue;
                }
                if !self.attacks_square(piece, future_from, *target) {
                    continue;
                }

                let urgency = 5 - wait;
                let fixed_target_bonus = if self.is_latest_board(target.timeline_id, target.time) {
                    0
                } else {
                    weights.temporal_threat * 2
                };
                let piece_bonus = match piece.piece_type {
                    PieceType::Queen | PieceType::RoyalQueen => weights.royal_capture_setup / 2,
                    PieceType::Bishop
                    | PieceType::Rook
                    | PieceType::Unicorn
                    | PieceType::Dragon
                    | PieceType::Princess => weights.royal_capture_setup / 3,
                    _ => weights.royal_capture_setup / 6,
                };
                score +=
                    weights.royal_capture_setup * urgency / 2 + piece_bonus + fixed_target_bonus;
                break;
            }
        }
        score
    }

    fn royal_safety_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.royal_safety_for(color, weights) - self.royal_safety_for(color.opposite(), weights)
    }

    fn royal_safety_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (position, _) in self.royal_pieces(color) {
            if !self.is_latest_board(position.timeline_id, position.time) {
                continue;
            }
            let attackers = self.attack_summary(position, color.opposite());
            let defenders = self.attack_summary(position, color);
            let escapes = self.royal_escape_count(position, color);
            score -= attackers.count * weights.own_royal_exposure;
            score -= attackers.temporal_count * weights.royal_capture_threat;
            score += defenders.count * weights.defended_piece;
            score += escapes * weights.royal_escape_pressure;
            if attackers.count > 0 && escapes == 0 {
                score -= weights.check_penalty / 2;
            }
        }
        score
    }

    fn royal_escape_count(&self, position: Position, color: Color) -> i32 {
        let mut search = self.clone_for_search();
        search.turn = color;
        let mut escapes = 0;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let to = Position {
                    timeline_id: position.timeline_id,
                    time: position.time,
                    x: position.x + dx,
                    y: position.y + dy,
                };
                if !Self::in_bounds(to.x, to.y)
                    || search
                        .piece_at(to)
                        .is_some_and(|piece| piece.color == color)
                    || search.is_square_attacked(to, color.opposite())
                {
                    continue;
                }
                if search.legal_move_kind(position, to).is_some() {
                    escapes += 1;
                }
            }
        }
        escapes
    }

    fn fork_pressure_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.fork_pressure_for(color, weights) - self.fork_pressure_for(color.opposite(), weights)
    }

    fn fork_pressure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let enemies: Vec<(Position, Piece)> = self
            .latest_pieces()
            .into_iter()
            .filter(|(_, piece)| piece.color == color.opposite())
            .collect();
        let mut score = 0;
        for (from, piece) in self.latest_pieces() {
            if piece.color != color {
                continue;
            }
            let mut threatened = 0;
            let mut value_sum = 0;
            let mut royal = false;
            for (target, enemy) in &enemies {
                if !self.attacks_square(piece, from, *target) {
                    continue;
                }
                threatened += 1;
                value_sum += weights.piece_value(enemy.piece_type);
                royal |= Self::is_royal_piece(enemy.piece_type);
            }
            if threatened >= 2 {
                score += weights.fork_pressure * (threatened - 1) + value_sum / 24;
                if royal {
                    score += weights.royal_threat;
                }
            }
        }
        score
    }

    fn forcing_pressure_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.forcing_pressure_for(color, weights)
            - self.forcing_pressure_for(color.opposite(), weights)
    }

    fn forcing_pressure_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let mut score = 0;
        for (position, piece) in self.latest_pieces() {
            if piece.color != color.opposite() {
                continue;
            }
            let attackers = self.attack_summary(position, color);
            if attackers.count == 0 {
                continue;
            }
            let value = weights.piece_value(piece.piece_type);
            score += attackers.count * weights.forcing_move_pressure + value / 48;
            if Self::is_royal_piece(piece.piece_type) {
                score += weights.royal_threat + attackers.temporal_count * weights.temporal_threat;
            }
            if attackers.timeline_count >= 2 || attackers.time_count >= 2 {
                score += weights.pincer_threat
                    + weights.timeline_pincer * (attackers.timeline_count - 1).max(0)
                    + weights.historical_pincer * (attackers.time_count - 1).max(0);
            }
        }
        score
    }

    fn timeline_coordination(&self, color: Color, weights: &EvalWeights) -> i32 {
        let Some(present) = self.present_board() else {
            return 0;
        };
        let mut score = 0;
        for timeline in &self.timelines {
            if !self.is_active_timeline(timeline.id) {
                continue;
            }
            let Some(board) = timeline.boards.iter().max_by_key(|board| board.time) else {
                continue;
            };
            let side = if board.side_to_move == color { 1 } else { -1 };
            score += side * weights.frontier_tempo;
            if board.time == present.time {
                score += side * weights.present_anchor;
            }
        }
        score
    }

    fn latest_pieces(&self) -> Vec<(Position, Piece)> {
        let mut pieces = Vec::new();
        for timeline in &self.timelines {
            let Some(board) = timeline.boards.iter().max_by_key(|board| board.time) else {
                continue;
            };
            for y in 0..8 {
                for x in 0..8 {
                    if let Some(piece) = board.board[y][x] {
                        pieces.push((
                            Position {
                                timeline_id: timeline.id,
                                time: board.time,
                                x: x as i32,
                                y: y as i32,
                            },
                            piece,
                        ));
                    }
                }
            }
        }
        pieces
    }

    fn latest_board_positions(&self) -> Vec<Position> {
        let mut positions = Vec::new();
        for timeline in &self.timelines {
            let Some(board) = timeline.boards.iter().max_by_key(|board| board.time) else {
                continue;
            };
            for y in 0..8 {
                for x in 0..8 {
                    positions.push(Position {
                        timeline_id: timeline.id,
                        time: board.time,
                        x,
                        y,
                    });
                }
            }
        }
        positions
    }

    fn near_enemy_royal(&self, target: Position, color: Color) -> bool {
        self.royal_pieces(color.opposite())
            .into_iter()
            .filter(|(position, _)| self.is_latest_board(position.timeline_id, position.time))
            .any(|(position, _)| {
                let delta = self.movement_delta(target, position);
                delta.x.abs().max(delta.y.abs()).max(delta.t.abs()).max(delta.l.abs()) <= 2
            })
    }

    fn pseudo_attack_count(&self, position: Position, piece: Piece) -> i32 {
        self.latest_board_positions()
            .into_iter()
            .filter(|target| self.attacks_square(piece, position, *target))
            .count() as i32
    }

    fn open_line_count(&self, position: Position, piece: Piece) -> i32 {
        let directions: &[(i32, i32, i32, i32)] = &[
            (1, 0, 0, 0),
            (-1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, -1, 0, 0),
            (1, 1, 0, 0),
            (1, -1, 0, 0),
            (-1, 1, 0, 0),
            (-1, -1, 0, 0),
            (0, 0, 2, 0),
            (0, 0, -2, 0),
            (0, 0, 0, 1),
            (0, 0, 0, -1),
        ];
        directions
            .iter()
            .filter(|(dx, dy, dt, dl)| {
                self.first_step_on_line(position, *dx, *dy, *dt, *dl)
                    .is_some_and(|target| {
                        self.piece_at(target).is_none()
                            && self.attacks_square(piece, position, target)
                    })
            })
            .count() as i32
    }

    fn first_step_on_line(
        &self,
        position: Position,
        dx: i32,
        dy: i32,
        dt: i32,
        dl: i32,
    ) -> Option<Position> {
        let from_row = self.timeline(position.timeline_id)?.row;
        let target_row = from_row + dl;
        let target_timeline_id = if dl == 0 {
            position.timeline_id
        } else {
            self.timelines
                .iter()
                .find(|timeline| timeline.row == target_row)?
                .id
        };
        let target = Position {
            timeline_id: target_timeline_id,
            time: position.time + dt,
            x: position.x + dx,
            y: position.y + dy,
        };
        (Self::in_bounds(target.x, target.y) && self.board(target.timeline_id, target.time).is_some())
            .then_some(target)
    }

    fn is_passed_pawn(&self, position: Position, color: Color) -> bool {
        let direction = if color == Color::White { 1 } else { -1 };
        let Some(board) = self.board(position.timeline_id, position.time) else {
            return false;
        };
        let mut y = position.y + direction;
        while Self::in_bounds(position.x, y) {
            for x in position.x - 1..=position.x + 1 {
                if Self::in_bounds(x, y)
                    && board.board[y as usize][x as usize].is_some_and(|piece| {
                        piece.color == color.opposite()
                            && matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
                    })
                {
                    return false;
                }
            }
            y += direction;
        }
        true
    }

    fn is_supported_pawn(&self, position: Position, color: Color) -> bool {
        let behind = if color == Color::White { -1 } else { 1 };
        [-1, 1].into_iter().any(|dx| {
            let supporter = Position {
                timeline_id: position.timeline_id,
                time: position.time,
                x: position.x + dx,
                y: position.y + behind,
            };
            Self::in_bounds(supporter.x, supporter.y)
                && self.piece_at(supporter).is_some_and(|piece| {
                    piece.color == color && matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
                })
        })
    }

    fn is_isolated_pawn(&self, position: Position, color: Color) -> bool {
        let Some(board) = self.board(position.timeline_id, position.time) else {
            return false;
        };
        ![-1, 1].into_iter().any(|dx| {
            let file = position.x + dx;
            Self::in_bounds(file, position.y)
                && (0..8).any(|y| {
                    board.board[y][file as usize].is_some_and(|piece| {
                        piece.color == color
                            && matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
                    })
                })
        })
    }

    fn timeline_time_spread(&self) -> i32 {
        let mut latest_times = self
            .timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| self.latest_time(timeline.id));
        let Some(first) = latest_times.next() else {
            return 0;
        };
        let (min, max) = latest_times.fold((first, first), |(min, max), time| {
            (min.min(time), max.max(time))
        });
        max - min
    }

    fn attack_summary(&self, target: Position, by_color: Color) -> AttackSummary {
        let mut summary = AttackSummary::default();
        let mut timelines = Vec::new();
        let mut times = Vec::new();

        for (from, piece) in self.latest_pieces() {
            if piece.color != by_color
                || from.timeline_id == target.timeline_id
                    && from.time == target.time
                    && from.x == target.x
                    && from.y == target.y
                || !self.attacks_square(piece, from, target)
            {
                continue;
            }

            summary.count += 1;
            if from.timeline_id != target.timeline_id || from.time != target.time {
                summary.temporal_count += 1;
            }
            if !timelines.contains(&from.timeline_id) {
                timelines.push(from.timeline_id);
            }
            if !times.contains(&from.time) {
                times.push(from.time);
            }
        }

        summary.timeline_count = timelines.len() as i32;
        summary.time_count = times.len() as i32;
        summary
    }

}
