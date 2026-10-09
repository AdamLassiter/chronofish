use crate::{
    hash::{board_position_hash, timeline_position_hash},
    notation::{
        en_passant_after_move, piece_json, position_json, promote_if_needed, update_castling_rights,
    },
    *,
};

impl Game {
    /// Creates a fresh game at T0L0 using the selected army and dimension rules.
    #[must_use]
    pub fn new(ruleset: Ruleset) -> Self {
        let mut board = [[None; 8]; 8];
        let (back_rank, pawn) = if ruleset == Ruleset::MultiverseVariant {
            (
                [
                    PieceType::Princess,
                    PieceType::Unicorn,
                    PieceType::Dragon,
                    PieceType::RoyalQueen,
                    PieceType::CommonKing,
                    PieceType::Dragon,
                    PieceType::Unicorn,
                    PieceType::Princess,
                ],
                PieceType::Brawn,
            )
        } else {
            (
                [
                    PieceType::Rook,
                    PieceType::Knight,
                    PieceType::Bishop,
                    PieceType::Queen,
                    PieceType::King,
                    PieceType::Bishop,
                    PieceType::Knight,
                    PieceType::Rook,
                ],
                PieceType::Pawn,
            )
        };

        for x in 0..8 {
            board[0][x] = Some(Piece {
                color: Color::White,
                piece_type: back_rank[x],
            });
            board[1][x] = Some(Piece {
                color: Color::White,
                piece_type: pawn,
            });
            board[6][x] = Some(Piece {
                color: Color::Black,
                piece_type: pawn,
            });
            board[7][x] = Some(Piece {
                color: Color::Black,
                piece_type: back_rank[x],
            });
        }

        let mut game = Self {
            ruleset,
            turn: Color::White,
            timelines: vec![Timeline {
                id: 0,
                row: 0,
                label: "Sacred T0".to_string(),
                owner: TimelineOwner::Neutral,
                boards: vec![BoardSnapshot {
                    time: 0,
                    side_to_move: Color::White,
                    board,
                    castling: CastlingRights::new(),
                    en_passant: None,
                    origin: Origin::None,
                    halfmove_clock: 0,
                }],
            }],
            next_timeline_id: 1,
            next_black_timeline_id: -1,
            staged_turn: Vec::new(),
            staged_notation: Vec::new(),
            staged_royal_capture_by: None,
            result: None,
            last_message: "Select a white piece on a latest board.".to_string(),
            position_hash: 0,
            repetition_history: Vec::new(),
        };
        game.position_hash = game.recompute_position_hash();
        game.repetition_history.push(game.position_hash);
        game
    }

    #[must_use]
    pub const fn ruleset(&self) -> Ruleset {
        self.ruleset
    }

    #[must_use]
    pub const fn turn(&self) -> Color {
        self.turn
    }

    #[must_use]
    pub const fn outcome(&self) -> Option<GameOutcome> {
        self.result
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.last_message
    }

    #[must_use]
    pub fn timelines(&self) -> &[Timeline] {
        &self.timelines
    }

    #[must_use]
    pub fn has_staged_moves(&self) -> bool {
        !self.staged_turn.is_empty()
    }

    #[must_use]
    pub fn can_submit(&self) -> bool {
        self.ruleset != Ruleset::Standard
            && !self.staged_turn.is_empty()
            && !self.has_pending_present_board(self.turn)
            && !self.is_in_check(self.turn)
    }

    /// Returns the complete authoritative action set for the current turn.
    #[must_use]
    pub fn legal_actions(&self) -> Vec<PlayerAction> {
        if self.result.is_some() {
            return Vec::new();
        }
        let mut actions = Vec::new();
        for timeline in &self.timelines {
            if !self.is_active_timeline(timeline.id) {
                continue;
            }
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            if board.side_to_move != self.turn || self.present_time() != Some(board.time) {
                continue;
            }
            for y in 0..BOARD_SIZE_I32 {
                for x in 0..BOARD_SIZE_I32 {
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
                        let Some((candidate, kind)) = self.legal_move_kind(from, to) else {
                            return true;
                        };
                        if !self.allows_search_move(from, to, candidate, kind) {
                            return true;
                        }
                        if self.ruleset == Ruleset::Standard
                            && self
                                .piece_at(to)
                                .is_some_and(|target| Self::is_royal_piece(target.piece_type))
                        {
                            return true;
                        }
                        let promotions: &[Option<PieceType>] =
                            if Self::requires_promotion(piece, to) {
                                if piece.piece_type == PieceType::Brawn {
                                    &[
                                        Some(PieceType::RoyalQueen),
                                        Some(PieceType::Princess),
                                        Some(PieceType::Dragon),
                                        Some(PieceType::Unicorn),
                                    ]
                                } else {
                                    &[
                                        Some(PieceType::Queen),
                                        Some(PieceType::Rook),
                                        Some(PieceType::Bishop),
                                        Some(PieceType::Knight),
                                    ]
                                }
                            } else {
                                &[None]
                            };
                        for &promotion in promotions {
                            let movement = Move {
                                from,
                                to,
                                promotion,
                            };
                            if self.ruleset != Ruleset::Standard
                                || self.move_preserves_royal_safety(movement)
                            {
                                actions.push(PlayerAction::Move { movement });
                            }
                        }
                        true
                    });
                }
            }
        }
        actions.sort_by_key(|action| match action {
            PlayerAction::Move { movement } => (
                0,
                movement.from,
                movement.to,
                movement
                    .promotion
                    .map_or(0, Self::repetition_piece_type_code),
            ),
            PlayerAction::SubmitTurn => (
                1,
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 0,
                    y: 0,
                },
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 0,
                    y: 0,
                },
                0,
            ),
        });
        actions.dedup();
        if self.can_submit() {
            actions.push(PlayerAction::SubmitTurn);
        }
        actions
    }

    fn move_preserves_royal_safety(&self, movement: Move) -> bool {
        let mut probe = self.clone_for_search();
        let mover = probe.turn;
        if probe.apply_move_coords(movement.from, movement.to) == 0 {
            return false;
        }
        !probe.is_in_check(mover)
    }

    fn requires_promotion(piece: Piece, to: Position) -> bool {
        matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn)
            && ((piece.color == Color::White && to.y == 7)
                || (piece.color == Color::Black && to.y == 0))
    }

    /// Applies one move. Standard chess completes the turn immediately; 5D
    /// games retain the move until every board at the Present has advanced.
    pub fn apply_move(&mut self, movement: Move) -> Result<(), MoveError> {
        if self.result.is_some() {
            return Err(MoveError::GameOver);
        }
        let piece = self.piece_at(movement.from).ok_or(MoveError::IllegalMove)?;
        let promotion_required = Self::requires_promotion(piece, movement.to);
        if promotion_required && movement.promotion.is_none() {
            return Err(MoveError::PromotionRequired);
        }
        if let Some(promotion) = movement.promotion {
            let valid = if piece.piece_type == PieceType::Brawn {
                matches!(
                    promotion,
                    PieceType::RoyalQueen
                        | PieceType::Princess
                        | PieceType::Dragon
                        | PieceType::Unicorn
                )
            } else {
                matches!(
                    promotion,
                    PieceType::Queen | PieceType::Rook | PieceType::Bishop | PieceType::Knight
                )
            };
            if !promotion_required || !valid {
                return Err(MoveError::InvalidPromotion);
            }
        }
        if self.ruleset == Ruleset::Standard
            && self
                .piece_at(movement.to)
                .is_some_and(|target| Self::is_royal_piece(target.piece_type))
        {
            return Err(MoveError::IllegalMove);
        }
        let checkpoint = self.checkpoint();
        let mover = self.turn;
        if self.apply_move_coords(movement.from, movement.to) == 0 {
            return Err(MoveError::IllegalMove);
        }
        if self.ruleset == Ruleset::Standard && self.is_in_check(mover) {
            let _ = self.staged_turn.pop();
            self.restore(checkpoint);
            return Err(MoveError::IllegalMove);
        }
        if let Some(promoted) = movement.promotion {
            self.replace_moved_piece(movement, promoted);
        }
        if self.ruleset != Ruleset::Standard {
            if let Some(winner) = self.staged_royal_capture_by {
                self.staged_turn.clear();
                self.staged_notation.clear();
                self.staged_royal_capture_by = None;
                self.result = Some(GameResult {
                    winner: Some(winner),
                    reason: GameResultReason::RoyalCapture,
                });
                self.last_message = format!("{} wins by royal capture.", winner.capitalized());
                return Ok(());
            }
        }
        if self.ruleset == Ruleset::Standard && self.submit_turn_raw() == 0 {
            return Err(MoveError::CannotSubmit);
        }
        Ok(())
    }

    pub fn validate_move(&self, movement: Move) -> Result<(), MoveError> {
        if self
            .legal_actions()
            .contains(&PlayerAction::Move { movement })
        {
            Ok(())
        } else {
            Err(MoveError::IllegalMove)
        }
    }

    fn replace_moved_piece(&mut self, movement: Move, promoted: PieceType) {
        for timeline in &mut self.timelines {
            if let Some(board) = timeline.boards.last_mut() {
                if matches!(&board.origin, Origin::Move { from, to, .. } if *from == movement.from && *to == movement.to)
                {
                    if let Some(piece) =
                        board.board[movement.to.y as usize][movement.to.x as usize].as_mut()
                    {
                        piece.piece_type = promoted;
                    }
                }
            }
        }
        self.position_hash = self.recompute_position_hash();
    }

    pub fn apply_action(&mut self, action: PlayerAction) -> Result<(), MoveError> {
        match action {
            PlayerAction::Move { movement } => self.apply_move(movement),
            PlayerAction::SubmitTurn => self.submit_turn(),
        }
    }

    pub fn submit_turn(&mut self) -> Result<(), MoveError> {
        (self.submit_turn_raw() == 1)
            .then_some(())
            .ok_or(MoveError::CannotSubmit)
    }

    pub fn undo_staged_move(&mut self) -> Result<(), MoveError> {
        (self.undo_staged_move_raw() == 1)
            .then_some(())
            .ok_or(MoveError::IllegalMove)
    }

    pub fn concede(&mut self, side: Color) -> Result<(), MoveError> {
        if self.result.is_some() {
            return Err(MoveError::GameOver);
        }
        self.result = Some(GameOutcome {
            winner: Some(side.opposite()),
            reason: OutcomeReason::Concession,
        });
        self.last_message = format!("{} resigned.", side.capitalized());
        Ok(())
    }

    pub(crate) fn turn_color(&self) -> Color {
        self.turn
    }

    pub(crate) fn position_key(&self) -> String {
        format!("{:016x}", self.position_hash)
    }

    pub(crate) fn timeline(&self, timeline_id: i32) -> Option<&Timeline> {
        self.timelines
            .iter()
            .find(|timeline| timeline.id == timeline_id)
    }

    pub(crate) fn timeline_mut(&mut self, timeline_id: i32) -> Option<&mut Timeline> {
        self.timelines
            .iter_mut()
            .find(|timeline| timeline.id == timeline_id)
    }

    #[inline]
    pub(crate) fn board(&self, timeline_id: i32, time: i32) -> Option<&BoardSnapshot> {
        let boards = &self.timeline(timeline_id)?.boards;
        if boards.last().is_some_and(|board| board.time == time) {
            return boards.last();
        }
        boards
            .binary_search_by_key(&time, |board| board.time)
            .ok()
            .map(|index| &boards[index])
    }

    pub(crate) fn latest_time(&self, timeline_id: i32) -> Option<i32> {
        self.timeline(timeline_id)?
            .boards
            .last()
            .map(|board| board.time)
    }

    pub(crate) fn is_latest_board(&self, timeline_id: i32, time: i32) -> bool {
        self.latest_time(timeline_id) == Some(time)
    }

    #[inline]
    pub(crate) fn piece_at(&self, position: Position) -> Option<Piece> {
        if !Self::in_bounds(position.x, position.y) {
            return None;
        }

        self.board(position.timeline_id, position.time)?.board[position.y as usize]
            [position.x as usize]
    }

    pub(crate) fn can_move_to(&self, from: Position, to: Position) -> bool {
        self.legal_move_kind(from, to).is_some()
    }

    #[allow(dead_code)]
    pub(crate) fn pruned_for_evaluation(&self) -> Self {
        let active_timeline_ids: Vec<i32> = self
            .timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .map(|timeline| timeline.id)
            .collect();
        let mut pruned = self.clone_for_search();
        pruned
            .timelines
            .retain(|timeline| active_timeline_ids.contains(&timeline.id));
        pruned
    }

    pub(crate) fn allows_search_move(
        &self,
        from: Position,
        to: Position,
        piece: Piece,
        move_kind: MoveKind,
    ) -> bool {
        if !self.is_active_timeline(from.timeline_id) {
            return false;
        }

        if matches!(move_kind, MoveKind::Branch) && !self.is_latest_board(to.timeline_id, to.time) {
            let branch_timeline_id = match piece.color {
                Color::White => self.next_timeline_id,
                Color::Black => self.next_black_timeline_id,
            };
            return self.would_be_active_timeline(branch_timeline_id)
                || Self::is_royal_piece(piece.piece_type);
        }

        true
    }

    pub(crate) fn would_be_active_timeline(&self, timeline_id: i32) -> bool {
        let min_timeline = self
            .timelines
            .iter()
            .map(|timeline| timeline.id)
            .chain(std::iter::once(timeline_id))
            .min()
            .unwrap_or(0);
        let max_timeline = self
            .timelines
            .iter()
            .map(|timeline| timeline.id)
            .chain(std::iter::once(timeline_id))
            .max()
            .unwrap_or(0);
        let active_distance = (-min_timeline).min(max_timeline).max(0) + 1;

        timeline_id.abs() <= active_distance
    }

    // Shared legality gate for UI highlighting, user moves, and AI search. It
    // checks turn ownership, latest-board source rules, destination turn rules
    // for time travel, and friendly occupancy before piece geometry runs.
    pub(crate) fn legal_move_kind(
        &self,
        from: Position,
        to: Position,
    ) -> Option<(Piece, MoveKind)> {
        if !Self::in_bounds(from.x, from.y) || !Self::in_bounds(to.x, to.y) {
            return None;
        }

        let source_board = self.board(from.timeline_id, from.time)?;
        let target_board = self.board(to.timeline_id, to.time)?;
        let piece = self.piece_at(from)?;
        let same_board = from.timeline_id == to.timeline_id && from.time == to.time;

        if !self.is_present_source_board(from)
            || !self.is_latest_board(from.timeline_id, from.time)
            || source_board.side_to_move != self.turn
            || piece.color != self.turn
        {
            return None;
        }

        if !same_board && target_board.side_to_move != piece.color {
            return None;
        }

        if self
            .piece_at(to)
            .is_some_and(|target| target.color == piece.color)
        {
            return None;
        }

        self.move_kind_for(piece, from, to)
            .map(|kind| (piece, kind))
    }

    /// Geometry and destination validation for search generators that have
    /// already proved the source is active, latest, present, and owned by the
    /// side to move.
    pub(crate) fn legal_move_kind_from_valid_source(
        &self,
        from: Position,
        to: Position,
        piece: Piece,
    ) -> Option<MoveKind> {
        if !Self::in_bounds(to.x, to.y) {
            return None;
        }
        let target_board = self.board(to.timeline_id, to.time)?;
        let same_board = from.timeline_id == to.timeline_id && from.time == to.time;
        if !same_board && target_board.side_to_move != piece.color {
            return None;
        }
        if target_board.board[to.y as usize][to.x as usize]
            .is_some_and(|target| target.color == piece.color)
        {
            return None;
        }
        self.move_kind_for(piece, from, to)
    }

    #[allow(dead_code)]
    pub(crate) fn legal_targets_json(&self, from: Position) -> String {
        let mut targets = Vec::new();

        for timeline in &self.timelines {
            for board in &timeline.boards {
                for y in 0..8 {
                    for x in 0..8 {
                        let to = Position {
                            timeline_id: timeline.id,
                            time: board.time,
                            x,
                            y,
                        };

                        if self.can_move_to(from, to) {
                            targets.push(position_json(to));
                        }
                    }
                }
            }
        }

        format!("[{}]", targets.join(","))
    }

    #[allow(dead_code)]
    pub(crate) fn legal_selection_json(&self, from: Position) -> String {
        let source = self
            .board(from.timeline_id, from.time)
            .and_then(|board| {
                let piece = self.piece_at(from)?;
                (self.is_present_source_board(from)
                    && self.is_latest_board(from.timeline_id, from.time)
                    && board.side_to_move == self.turn
                    && piece.color == self.turn)
                    .then_some(piece)
            })
            .map_or_else(
                || "null".to_string(),
                |piece| {
                    format!(
                        "{{\"timelineId\":{},\"time\":{},\"x\":{},\"y\":{},\"piece\":{}}}",
                        from.timeline_id,
                        from.time,
                        from.x,
                        from.y,
                        piece_json(&Some(piece))
                    )
                },
            );

        format!(
            "{{\"source\":{source},\"targets\":{}}}",
            self.legal_targets_json(from)
        )
    }

    pub(crate) fn is_present_source_board(&self, from: Position) -> bool {
        self.present_time() == Some(from.time)
    }

    #[allow(dead_code)]
    fn apply_move_coords(&mut self, from: Position, to: Position) -> i32 {
        let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
            self.last_message = "Illegal move.".to_string();
            return 0;
        };

        // Moves are staged until submit_turn accepts the whole turn. This allows
        // a side to make the multiple moves needed to advance the present line.
        let captured = self.captured_piece(to, move_kind);
        let notation = self.move_notation(piece, from, to, move_kind);
        self.staged_turn.push(self.checkpoint());
        self.record_staged_capture(piece.color, captured);
        self.apply_move_unchecked(from, to, piece, move_kind);
        self.staged_notation
            .push(self.finish_move_notation(notation, piece.color));
        self.last_message = self.move_message(
            piece,
            from,
            to,
            matches!(
                move_kind,
                MoveKind::Standard | MoveKind::Castle { .. } | MoveKind::EnPassant { .. }
            ),
        );
        1
    }

    pub(crate) fn clone_for_search(&self) -> Self {
        self.clone()
    }

    // Capture enough state to undo one staged move.
    #[allow(dead_code)]
    pub(crate) fn checkpoint(&self) -> GameCheckpoint {
        GameCheckpoint {
            turn: self.turn,
            timelines: self.timelines.clone(),
            next_timeline_id: self.next_timeline_id,
            next_black_timeline_id: self.next_black_timeline_id,
            staged_notation: self.staged_notation.clone(),
            staged_royal_capture_by: self.staged_royal_capture_by,
            last_message: self.last_message.clone(),
            position_hash: self.position_hash,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn restore(&mut self, checkpoint: GameCheckpoint) {
        self.turn = checkpoint.turn;
        self.timelines = checkpoint.timelines;
        self.next_timeline_id = checkpoint.next_timeline_id;
        self.next_black_timeline_id = checkpoint.next_black_timeline_id;
        self.staged_notation = checkpoint.staged_notation;
        self.staged_royal_capture_by = checkpoint.staged_royal_capture_by;
        self.last_message = checkpoint.last_message;
        self.position_hash = checkpoint.position_hash;
    }

    #[allow(dead_code)]
    fn undo_staged_move_raw(&mut self) -> i32 {
        let Some(checkpoint) = self.staged_turn.pop() else {
            self.last_message = "No staged move to undo.".to_string();
            return 0;
        };

        self.restore(checkpoint);
        self.last_message = if self.staged_turn.is_empty() {
            format!("{} to move.", self.turn.capitalized())
        } else {
            "Undid staged move.".to_string()
        };
        1
    }

    #[allow(dead_code)]
    fn submit_turn_raw(&mut self) -> i32 {
        if self.staged_turn.is_empty() {
            self.last_message = "Make at least one move before submitting.".to_string();
            return 0;
        }

        // Turn passing follows the present line, not a simple alternating
        // single-move clock. The player keeps moving until the earliest active
        // latest board is now waiting for the opponent.
        let Some(present_side) = self.present_board().map(|board| board.side_to_move) else {
            self.last_message = "No active present board.".to_string();
            return 0;
        };

        if let Some(winner) = self.staged_royal_capture_by {
            self.turn = present_side;
            self.staged_turn.clear();
            self.staged_notation.clear();
            self.staged_royal_capture_by = None;
            self.result = Some(GameResult {
                winner: Some(winner),
                reason: GameResultReason::RoyalCapture,
            });
            self.last_message = format!("{} wins by royal capture.", winner.capitalized());
            return 1;
        }

        if self.has_pending_present_board(self.turn) {
            self.last_message =
                "Make moves until the present line reaches the opponent's turn.".to_string();
            return 0;
        }

        if self.is_in_check(self.turn) {
            self.last_message =
                "The turn cannot be submitted while a royal piece is in check.".to_string();
            return 0;
        }

        self.turn = present_side;
        self.staged_turn.clear();
        self.staged_notation.clear();
        self.staged_royal_capture_by = None;

        self.repetition_history.push(self.position_hash);

        if self.is_checkmate(self.turn) {
            let winner = self.turn.opposite();
            self.result = Some(GameResult {
                winner: Some(winner),
                reason: GameResultReason::Checkmate,
            });
            self.last_message = format!("{} wins by checkmate.", winner.capitalized());
            return 1;
        }

        if self.ruleset == Ruleset::Standard
            && self
                .present_board()
                .is_some_and(|board| board.halfmove_clock >= 100)
        {
            self.result = Some(GameResult {
                winner: None,
                reason: GameResultReason::FiftyMoveRule,
            });
            self.last_message = "Draw by the fifty-move rule.".to_string();
            return 1;
        }

        if self.ruleset == Ruleset::Standard && self.has_insufficient_material() {
            self.result = Some(GameResult {
                winner: None,
                reason: GameResultReason::InsufficientMaterial,
            });
            self.last_message = "Draw by insufficient material.".to_string();
            return 1;
        }

        if self.has_threefold_repetition() {
            self.result = Some(GameResult {
                winner: None,
                reason: GameResultReason::ThreefoldRepetition,
            });
            self.last_message = "Stalemate by threefold repetition.".to_string();
            return 1;
        }

        if self.is_standard_stalemate(self.turn) {
            self.result = Some(GameResult {
                winner: None,
                reason: GameResultReason::Stalemate,
            });
            self.last_message = "Stalemate.".to_string();
            return 1;
        }

        let suffix = if self.is_in_check(self.turn) {
            " Check."
        } else {
            ""
        };
        self.last_message = format!("{} to move.{}", self.turn.capitalized(), suffix);
        1
    }

    pub(crate) fn has_threefold_repetition(&self) -> bool {
        const SMALL_REPETITION_HISTORY: usize = 24;
        self.timelines.iter().any(|timeline| {
            if timeline.boards.len() <= SMALL_REPETITION_HISTORY {
                for (index, board) in timeline.boards.iter().enumerate() {
                    let key = Self::board_repetition_key_array(board);
                    let mut count = 1;
                    for later in &timeline.boards[index + 1..] {
                        if Self::board_repetition_key_array(later) == key {
                            count += 1;
                            if count >= 3 {
                                return true;
                            }
                        }
                    }
                }
                return false;
            }
            let mut counts = std::collections::HashMap::new();
            timeline.boards.iter().any(|board| {
                let count = counts
                    .entry(Self::board_repetition_key_array(board))
                    .or_insert(0);
                *count += 1;
                *count >= 3
            })
        })
    }

    fn has_insufficient_material(&self) -> bool {
        let Some(board) = self.present_board() else {
            return false;
        };
        let non_royals = board
            .board
            .iter()
            .enumerate()
            .flat_map(|(y, rank)| {
                rank.iter().enumerate().filter_map(move |(x, piece)| {
                    piece
                        .filter(|piece| !Self::is_royal_piece(piece.piece_type))
                        .map(|piece| (piece, x, y))
                })
            })
            .collect::<Vec<_>>();
        if non_royals.is_empty()
            || (non_royals.len() == 1
                && matches!(
                    non_royals[0].0.piece_type,
                    PieceType::Bishop | PieceType::Knight
                ))
        {
            return true;
        }
        let bishop_square_color = (non_royals[0].1 + non_royals[0].2) % 2;
        non_royals.iter().all(|(piece, x, y)| {
            piece.piece_type == PieceType::Bishop && (x + y) % 2 == bishop_square_color
        })
    }

    pub(crate) fn board_repetition_key_array(board: &BoardSnapshot) -> [i32; 70] {
        let mut key = [0; 70];
        key[0] = Self::repetition_color_code(board.side_to_move);
        key[1] = Self::repetition_castling_code(board.castling);
        if let Some(en_passant) = board.en_passant {
            key[2] = en_passant.x;
            key[3] = en_passant.y;
            key[4] = en_passant.captured_x;
            key[5] = en_passant.captured_y;
        } else {
            key[2..6].fill(-1);
        }
        let mut index = 6;
        for row in &board.board {
            for square in row {
                key[index] = Self::repetition_piece_code(*square);
                index += 1;
            }
        }
        key
    }

    pub(crate) fn board_repetition_key(board: &BoardSnapshot) -> Vec<i32> {
        let mut key = Vec::with_capacity(70);
        key.push(Self::repetition_color_code(board.side_to_move));
        key.push(Self::repetition_castling_code(board.castling));
        if let Some(en_passant) = board.en_passant {
            key.extend([
                en_passant.x,
                en_passant.y,
                en_passant.captured_x,
                en_passant.captured_y,
            ]);
        } else {
            key.extend([-1, -1, -1, -1]);
        }
        for row in board.board.iter() {
            for square in row.iter() {
                key.push(Self::repetition_piece_code(*square));
            }
        }
        key
    }

    pub(crate) fn repetition_color_code(color: Color) -> i32 {
        match color {
            Color::White => 0,
            Color::Black => 1,
        }
    }

    pub(crate) fn repetition_piece_type_code(piece_type: PieceType) -> i32 {
        match piece_type {
            PieceType::King => 1,
            PieceType::CommonKing => 2,
            PieceType::Queen => 3,
            PieceType::RoyalQueen => 4,
            PieceType::Princess => 5,
            PieceType::Rook => 6,
            PieceType::Bishop => 7,
            PieceType::Unicorn => 8,
            PieceType::Dragon => 9,
            PieceType::Knight => 10,
            PieceType::Pawn => 11,
            PieceType::Brawn => 12,
        }
    }

    pub(crate) fn repetition_piece_code(piece: Option<Piece>) -> i32 {
        piece.map_or(0, |piece| {
            Self::repetition_piece_type_code(piece.piece_type)
                | (Self::repetition_color_code(piece.color) << 8)
        })
    }

    pub(crate) fn repetition_castling_code(castling: CastlingRights) -> i32 {
        castling.white_kingside as i32
            | ((castling.white_queenside as i32) << 1)
            | ((castling.black_kingside as i32) << 2)
            | ((castling.black_queenside as i32) << 3)
    }

    pub(crate) fn record_staged_capture(&mut self, color: Color, captured: Option<Piece>) {
        if captured.is_some_and(|piece| Self::is_royal_piece(piece.piece_type)) {
            self.staged_royal_capture_by = Some(color);
        }
    }

    pub(crate) fn apply_move_unchecked(
        &mut self,
        from: Position,
        to: Position,
        piece: Piece,
        move_kind: MoveKind,
    ) {
        let source_board = self
            .board(from.timeline_id, from.time)
            .expect("legal move has source board")
            .clone();
        let target_board = self
            .board(to.timeline_id, to.time)
            .expect("legal move has target board")
            .clone();
        let source_row = self
            .timeline(from.timeline_id)
            .expect("legal move has source timeline")
            .row;
        let next_turn = self.turn.opposite();
        let is_branch = matches!(move_kind, MoveKind::Branch);
        let is_capture = self.captured_piece(to, move_kind).is_some();
        let halfmove_clock =
            if matches!(piece.piece_type, PieceType::Pawn | PieceType::Brawn) || is_capture {
                0
            } else {
                source_board.halfmove_clock.saturating_add(1)
            };

        if !is_branch {
            // Same-board moves append one successor snapshot and carry forward
            // transient state such as castling rights and en-passant eligibility.
            let mut next_board = source_board.board;
            next_board[from.y as usize][from.x as usize] = None;
            if let MoveKind::EnPassant {
                captured_x,
                captured_y,
            } = move_kind
            {
                next_board[captured_y as usize][captured_x as usize] = None;
            }
            next_board[to.y as usize][to.x as usize] = Some(promote_if_needed(piece, to.y));
            if let MoveKind::Castle {
                rook_from_x,
                rook_to_x,
            } = move_kind
            {
                let rook = next_board[from.y as usize][rook_from_x as usize];
                next_board[from.y as usize][rook_from_x as usize] = None;
                next_board[from.y as usize][rook_to_x as usize] = rook;
            }

            let mut castling = source_board.castling;
            update_castling_rights(&mut castling, piece, from, to, source_board.board);

            self.push_board_snapshot(
                from.timeline_id,
                BoardSnapshot {
                    time: from.time + 1,
                    side_to_move: next_turn,
                    board: next_board,
                    castling,
                    en_passant: en_passant_after_move(piece, from, to, move_kind),
                    origin: Origin::Move {
                        from,
                        to,
                        move_type: move_kind.name().to_string(),
                    },
                    halfmove_clock,
                },
            );
        } else {
            // Time/timeline moves advance the source line without the piece, then
            // place it on the destination. Targeting a historical board creates a
            // new branch owned by the moving color.
            let target_is_historical = !self.is_latest_board(to.timeline_id, to.time);
            let destination_timeline_id = if target_is_historical {
                self.place_timeline(piece.color, source_row)
            } else {
                to.timeline_id
            };

            let mut advanced_source = source_board.board;
            advanced_source[from.y as usize][from.x as usize] = None;
            let mut source_castling = source_board.castling;
            update_castling_rights(&mut source_castling, piece, from, to, source_board.board);
            self.push_board_snapshot(
                from.timeline_id,
                BoardSnapshot {
                    time: from.time + 1,
                    side_to_move: next_turn,
                    board: advanced_source,
                    castling: source_castling,
                    en_passant: None,
                    origin: Origin::Move {
                        from,
                        to,
                        move_type: "source-advance".to_string(),
                    },
                    halfmove_clock,
                },
            );

            let mut branch_board = target_board.board;
            branch_board[to.y as usize][to.x as usize] = Some(promote_if_needed(piece, to.y));
            let mut target_castling = target_board.castling;
            update_castling_rights(&mut target_castling, piece, from, to, target_board.board);
            self.push_board_snapshot(
                destination_timeline_id,
                BoardSnapshot {
                    time: to.time + 1,
                    side_to_move: next_turn,
                    board: branch_board,
                    castling: target_castling,
                    en_passant: None,
                    origin: Origin::Move {
                        from,
                        to,
                        move_type: if target_is_historical {
                            "branch".to_string()
                        } else {
                            "cross-board".to_string()
                        },
                    },
                    halfmove_clock,
                },
            );
        }
    }

    pub(crate) fn place_timeline(&mut self, owner: Color, source_row: i32) -> i32 {
        // White-created ids are positive and black-created ids are negative,
        // matching RULES.md notation while row supplies the visual L-axis.
        let direction = if owner == Color::White { 1 } else { -1 };
        let mut row = source_row + direction;

        while self.timelines.iter().any(|timeline| timeline.row == row) {
            row += direction;
        }

        let id = if owner == Color::White {
            let id = self.next_timeline_id;
            self.next_timeline_id += 1;
            id
        } else {
            let id = self.next_black_timeline_id;
            self.next_black_timeline_id -= 1;
            id
        };
        let timeline = Timeline {
            id,
            row,
            label: format!(
                "{} T{}",
                if owner == Color::White {
                    "White"
                } else {
                    "Black"
                },
                id
            ),
            owner: TimelineOwner::from_color(owner),
            boards: Vec::new(),
        };
        self.position_hash ^= timeline_position_hash(&timeline);
        self.timelines.push(timeline);
        id
    }

    pub(crate) fn push_board_snapshot(&mut self, timeline_id: i32, board: BoardSnapshot) {
        self.position_hash ^= board_position_hash(timeline_id, &board);
        self.timeline_mut(timeline_id)
            .expect("timeline exists for appended board")
            .boards
            .push(board);
    }

    pub(crate) fn present_board(&self) -> Option<&BoardSnapshot> {
        // The present line is the earliest latest-board among active timelines.
        // Inactive timelines do not hold the turn hostage.
        let active_distance = self.active_timeline_distance();
        self.timelines
            .iter()
            .filter(|timeline| Self::timeline_is_active(timeline, active_distance))
            .filter_map(|timeline| timeline.boards.last())
            .min_by_key(|board| board.time)
    }

    pub fn present_time(&self) -> Option<i32> {
        self.present_board().map(|board| board.time)
    }

    pub(crate) fn has_pending_present_board(&self, color: Color) -> bool {
        let active_distance = self.active_timeline_distance();
        let mut present_time = None;
        let mut pending = false;
        for timeline in &self.timelines {
            if !Self::timeline_is_active(timeline, active_distance) {
                continue;
            }
            let Some(board) = timeline.boards.last() else {
                continue;
            };
            match present_time {
                None => {
                    present_time = Some(board.time);
                    pending = board.side_to_move == color;
                }
                Some(time) if board.time < time => {
                    present_time = Some(board.time);
                    pending = board.side_to_move == color;
                }
                Some(time) if board.time == time => {
                    pending |= board.side_to_move == color;
                }
                Some(_) => {}
            }
        }
        pending
    }

    pub fn is_active_timeline(&self, timeline_id: i32) -> bool {
        let Some(timeline) = self.timeline(timeline_id) else {
            return false;
        };
        Self::timeline_is_active(timeline, self.active_timeline_distance())
    }

    pub(crate) fn active_timeline_distance(&self) -> i32 {
        let min_timeline = self
            .timelines
            .iter()
            .map(|timeline| timeline.id)
            .min()
            .unwrap_or(0);
        let max_timeline = self
            .timelines
            .iter()
            .map(|timeline| timeline.id)
            .max()
            .unwrap_or(0);
        (-min_timeline).min(max_timeline).max(0) + 1
    }

    pub(crate) fn timeline_is_active(timeline: &Timeline, active_distance: i32) -> bool {
        timeline.owner == TimelineOwner::Neutral || timeline.id.abs() <= active_distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(time: i32, timeline_id: i32, square: &str) -> Position {
        let bytes = square.as_bytes();
        Position {
            timeline_id,
            time,
            x: i32::from(bytes[0] - b'a'),
            y: i32::from(bytes[1] - b'1'),
        }
    }

    fn play(game: &mut Game, time: i32, from: &str, to: &str) {
        game.apply_move(Move::new(position(time, 0, from), position(time, 0, to)))
            .expect("legal move");
    }

    #[test]
    fn standard_opening_has_twenty_moves_and_advances_immediately() {
        let mut game = Game::new(Ruleset::Standard);
        assert_eq!(game.legal_actions().len(), 20);
        assert!(!game.legal_actions().contains(&PlayerAction::SubmitTurn));
        play(&mut game, 0, "e2", "e4");
        assert_eq!(game.turn(), Color::Black);
        assert_eq!(game.present_time(), Some(1));
        assert!(!game.has_staged_moves());
    }

    #[test]
    fn fools_mate_is_recognized_as_checkmate() {
        let mut game = Game::new(Ruleset::Standard);
        play(&mut game, 0, "f2", "f3");
        play(&mut game, 1, "e7", "e5");
        play(&mut game, 2, "g2", "g4");
        play(&mut game, 3, "d8", "h4");
        assert_eq!(
            game.outcome(),
            Some(GameOutcome {
                winner: Some(Color::Black),
                reason: OutcomeReason::Checkmate
            })
        );
        assert!(game.legal_actions().is_empty());
    }

    #[test]
    fn en_passant_removes_the_pawn_that_was_passed() {
        let mut game = Game::new(Ruleset::Standard);
        play(&mut game, 0, "e2", "e4");
        play(&mut game, 1, "a7", "a6");
        play(&mut game, 2, "e4", "e5");
        play(&mut game, 3, "d7", "d5");
        play(&mut game, 4, "e5", "d6");
        let board = game.present_board().expect("board");
        assert!(board.board[4][3].is_none());
        assert_eq!(
            board.board[5][3],
            Some(Piece {
                color: Color::White,
                piece_type: PieceType::Pawn
            })
        );
    }

    #[test]
    fn orthodox_castling_moves_both_king_and_rook() {
        let mut game = Game::new(Ruleset::Standard);
        play(&mut game, 0, "e2", "e4");
        play(&mut game, 1, "a7", "a6");
        play(&mut game, 2, "g1", "f3");
        play(&mut game, 3, "a6", "a5");
        play(&mut game, 4, "f1", "e2");
        play(&mut game, 5, "a5", "a4");
        play(&mut game, 6, "e1", "g1");
        let board = game.present_board().expect("board after castling");
        assert_eq!(
            board.board[0][6],
            Some(Piece {
                color: Color::White,
                piece_type: PieceType::King
            })
        );
        assert_eq!(
            board.board[0][5],
            Some(Piece {
                color: Color::White,
                piece_type: PieceType::Rook
            })
        );
        assert!(board.board[0][4].is_none());
        assert!(board.board[0][7].is_none());
    }

    #[test]
    fn promotion_uses_the_requested_orthodox_piece() {
        let mut game = Game::new(Ruleset::Standard);
        let board = game.timelines[0].boards.last_mut().expect("initial board");
        board.board = [[None; BOARD_SIZE]; BOARD_SIZE];
        board.board[0][4] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::King,
        });
        board.board[7][4] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::King,
        });
        board.board[6][0] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Pawn,
        });
        board.castling = CastlingRights {
            white_kingside: false,
            white_queenside: false,
            black_kingside: false,
            black_queenside: false,
        };
        game.position_hash = game.recompute_position_hash();
        let movement = Move {
            from: position(0, 0, "a7"),
            to: position(0, 0, "a8"),
            promotion: Some(PieceType::Knight),
        };
        assert!(
            game.legal_actions()
                .contains(&PlayerAction::Move { movement })
        );
        game.apply_move(movement).expect("promotion");
        assert_eq!(
            game.present_board().expect("board").board[7][0],
            Some(Piece {
                color: Color::White,
                piece_type: PieceType::Knight
            })
        );
    }

    #[test]
    fn five_d_move_to_history_creates_a_owned_timeline() {
        let mut game = Game::new(Ruleset::Multiverse);
        play(&mut game, 0, "g1", "f3");
        game.submit_turn().expect("white turn complete");
        play(&mut game, 1, "g8", "f6");
        game.submit_turn().expect("black turn complete");
        let movement = Move::new(position(2, 0, "f3"), position(0, 0, "d3"));
        assert!(
            game.legal_actions()
                .contains(&PlayerAction::Move { movement })
        );
        game.apply_move(movement).expect("temporal knight move");
        assert_eq!(game.timelines.len(), 2);
        let branch = game.timeline(1).expect("white branch");
        assert_eq!(branch.owner, TimelineOwner::White);
        assert_eq!(branch.boards.last().expect("branch board").time, 1);
    }

    #[test]
    fn staged_five_d_turn_survives_persistence() {
        let mut game = Game::new(Ruleset::Multiverse);
        play(&mut game, 0, "e2", "e4");
        let json = serde_json::to_string(&game).expect("serialize game");
        let mut loaded: Game = serde_json::from_str(&json).expect("deserialize game");
        assert!(loaded.has_staged_moves());
        assert!(loaded.can_submit());
        loaded.submit_turn().expect("submit persisted turn");
        assert_eq!(loaded.turn(), Color::Black);
    }

    #[test]
    fn variant_setup_uses_the_alternate_army_and_royal_queen() {
        let game = Game::new(Ruleset::MultiverseVariant);
        let board = game.present_board().expect("initial board");
        assert_eq!(
            board.board[0][0].expect("piece").piece_type,
            PieceType::Princess
        );
        assert_eq!(
            board.board[0][3].expect("piece").piece_type,
            PieceType::RoyalQueen
        );
        assert_eq!(
            board.board[0][4].expect("piece").piece_type,
            PieceType::CommonKing
        );
        assert_eq!(
            board.board[1][0].expect("piece").piece_type,
            PieceType::Brawn
        );
        assert!(Game::is_royal_piece(PieceType::RoyalQueen));
        assert!(!Game::is_royal_piece(PieceType::CommonKing));
    }
}
