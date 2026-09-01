use crate::*;

impl Game {
    pub(crate) fn is_square_attacked(&self, target: Position, by_color: Color) -> bool {
        self.timelines.iter().any(|timeline| {
            let Some(board) = timeline.boards.last() else {
                return false;
            };
            board.board.iter().enumerate().any(|(y, rank)| {
                rank.iter().enumerate().any(|(x, piece)| {
                    let Some(piece) = piece else {
                        return false;
                    };
                    if piece.color != by_color {
                        return false;
                    }
                    let from = Position {
                        timeline_id: timeline.id,
                        time: board.time,
                        x: x as i32,
                        y: y as i32,
                    };
                    self.attacks_square(*piece, from, target)
                })
            })
        })
    }

    pub(crate) fn attacks_square(&self, piece: Piece, from: Position, target: Position) -> bool {
        if !Self::in_bounds(target.x, target.y)
            || from.timeline_id == target.timeline_id
                && from.time == target.time
                && from.x == target.x
                && from.y == target.y
        {
            return false;
        }

        // Cross-time attacks use the same constraint as moves: the target board
        // must be one where the attacking color was to play.
        if (from.timeline_id != target.timeline_id || from.time != target.time)
            && self
                .board(target.timeline_id, target.time)
                .is_some_and(|board| board.side_to_move != piece.color)
        {
            return false;
        }

        let delta = self.movement_delta(from, target);
        let distances = [delta.x.abs(), delta.y.abs(), delta.t.abs(), delta.l.abs()];
        let (non_zero, non_zero_len) = Self::non_zero_distances(distances);

        self.piece_attacks(
            piece,
            from,
            target,
            delta,
            distances,
            &non_zero[..non_zero_len],
        )
    }

    pub(crate) fn move_kind_for(
        &self,
        piece: Piece,
        from: Position,
        to: Position,
    ) -> Option<MoveKind> {
        let delta = self.movement_delta(from, to);
        let distances = [delta.x.abs(), delta.y.abs(), delta.t.abs(), delta.l.abs()];
        let (non_zero, non_zero_len) = Self::non_zero_distances(distances);
        let non_zero = &non_zero[..non_zero_len];

        if non_zero_len == 0 {
            return None;
        }

        if piece.piece_type == PieceType::King {
            if let Some(castle) = self.castle_kind(piece, from, to, delta) {
                return Some(castle);
            }
        }

        // Pawns and brawns are asymmetric, so they produce MoveKind directly
        // instead of going through the generic attack-shape matcher.
        let legal = match piece.piece_type {
            PieceType::Pawn => return self.pawn_move_kind(piece, from, to, delta),
            PieceType::Brawn => return self.brawn_move_kind(piece, from, to, delta),
            _ => self.piece_attacks(piece, from, to, delta, distances, non_zero),
        };

        if !legal {
            return None;
        }

        if from.timeline_id == to.timeline_id && from.time == to.time {
            Some(MoveKind::Standard)
        } else {
            Some(MoveKind::Branch)
        }
    }

    pub(crate) fn pawn_move_kind(
        &self,
        piece: Piece,
        from: Position,
        to: Position,
        delta: Delta,
    ) -> Option<MoveKind> {
        let destination = self.piece_at(to);
        let forward = if piece.color == Color::White { 1 } else { -1 };
        let has_moved = if piece.color == Color::White {
            from.y != 1
        } else {
            from.y != 6
        };
        let same_board = from.timeline_id == to.timeline_id && from.time == to.time;
        let source_board = same_board
            .then(|| self.board(from.timeline_id, from.time))
            .flatten();

        // Orthodox forward movement and captures are same-board only.
        if same_board && delta.x == 0 && delta.y == forward && destination.is_none() {
            return Some(MoveKind::Standard);
        }

        // En-passant reads a target stored on the current snapshot. Any later
        // snapshot clears it, giving the required one-turn window.
        if same_board
            && delta.x == 0
            && delta.y == forward * 2
            && !has_moved
            && destination.is_none()
        {
            return source_board.and_then(|source_board| {
                source_board.board[(from.y + forward) as usize][from.x as usize]
                    .is_none()
                    .then_some(MoveKind::Standard)
            });
        }

        if same_board
            && delta.x.abs() == 1
            && delta.y == forward
            && destination.is_some_and(|target| target.color != piece.color)
        {
            return Some(MoveKind::Standard);
        }

        let timeline_forward = if piece.color == Color::White { 1 } else { -1 };
        if !same_board
            && delta.x == 0
            && delta.y == 0
            && delta.t == 0
            && (delta.l == timeline_forward || delta.l == timeline_forward * 2 && !has_moved)
            && destination.is_none()
            && self.is_path_clear(piece, from, to)
        {
            return Some(MoveKind::Branch);
        }

        if same_board && delta.x.abs() == 1 && delta.y == forward && destination.is_none() {
            if let Some(target) = source_board
                .and_then(|source_board| source_board.en_passant)
                .filter(|target| target.x == to.x && target.y == to.y)
            {
                return Some(MoveKind::EnPassant {
                    captured_x: target.captured_x,
                    captured_y: target.captured_y,
                });
            }
        }

        // Pawns can also capture through one time step and one timeline step in
        // their forward timeline direction.
        (delta.t.abs() == 1
            && delta.l == timeline_forward
            && delta.x == 0
            && delta.y == 0
            && destination.is_some_and(|target| target.color != piece.color))
        .then_some(MoveKind::Branch)
    }

    pub(crate) fn brawn_move_kind(
        &self,
        piece: Piece,
        from: Position,
        to: Position,
        delta: Delta,
    ) -> Option<MoveKind> {
        let destination = self.piece_at(to);

        // Brawns inherit pawn non-captures, but capture diagonally across two or
        // more changed axes in the four-dimensional movement space.
        if destination.is_some_and(|target| target.color != piece.color)
            && Self::is_brawn_capture(piece, delta)
        {
            return Some(
                if from.timeline_id == to.timeline_id && from.time == to.time {
                    MoveKind::Standard
                } else {
                    MoveKind::Branch
                },
            );
        }

        self.pawn_move_kind(piece, from, to, delta)
    }

    pub(crate) fn piece_attacks(
        &self,
        piece: Piece,
        from: Position,
        target: Position,
        delta: Delta,
        distances: [i32; 4],
        non_zero: &[i32],
    ) -> bool {
        // Sliding variant pieces are described by how many axes change at the
        // same distance: rook=1, bishop=2, unicorn=3, dragon=4.
        match piece.piece_type {
            PieceType::Pawn => Self::is_pawn_capture(piece, delta),
            PieceType::Brawn => Self::is_brawn_capture(piece, delta),
            PieceType::Knight => Self::is_knight_move(distances),
            PieceType::King | PieceType::CommonKing => Self::is_king_move(non_zero),
            PieceType::Rook => {
                Self::is_rook_move(non_zero) && self.is_path_clear(piece, from, target)
            }
            PieceType::Bishop => {
                Self::is_bishop_move(non_zero) && self.is_path_clear(piece, from, target)
            }
            PieceType::Unicorn => {
                Self::is_unicorn_move(non_zero) && self.is_path_clear(piece, from, target)
            }
            PieceType::Dragon => {
                Self::is_dragon_move(non_zero) && self.is_path_clear(piece, from, target)
            }
            PieceType::Princess => {
                (Self::is_rook_move(non_zero) || Self::is_bishop_move(non_zero))
                    && self.is_path_clear(piece, from, target)
            }
            PieceType::Queen | PieceType::RoyalQueen => {
                Self::is_queen_move(non_zero) && self.is_path_clear(piece, from, target)
            }
        }
    }

    pub(crate) fn is_royal_piece(piece_type: PieceType) -> bool {
        matches!(piece_type, PieceType::King | PieceType::RoyalQueen)
    }

    pub(crate) fn is_pawn_capture(piece: Piece, delta: Delta) -> bool {
        let forward = if piece.color == Color::White { 1 } else { -1 };
        let timeline_forward = if piece.color == Color::White { 1 } else { -1 };
        (delta.x.abs() == 1 && delta.y == forward && delta.t == 0 && delta.l == 0)
            || (delta.t.abs() == 1 && delta.l == timeline_forward && delta.x == 0 && delta.y == 0)
    }

    pub(crate) fn is_brawn_capture(piece: Piece, delta: Delta) -> bool {
        let forward = if piece.color == Color::White { 1 } else { -1 };
        let timeline_forward = if piece.color == Color::White { 1 } else { -1 };
        let distances = [delta.x.abs(), delta.y.abs(), delta.t.abs(), delta.l.abs()];
        let mut non_zero = 0;
        for distance in distances {
            if distance > 1 {
                return false;
            }
            non_zero += (distance > 0) as i32;
        }

        non_zero >= 2
            && (delta.y == forward || delta.l == timeline_forward)
            && delta.y != -forward
            && delta.l != -timeline_forward
    }

    pub(crate) fn is_knight_move(distances: [i32; 4]) -> bool {
        let mut twos = 0;
        let mut ones = 0;
        let mut zeros = 0;
        for distance in distances {
            match distance {
                0 => zeros += 1,
                1 => ones += 1,
                2 => twos += 1,
                _ => return false,
            }
        }
        twos == 1 && ones == 1 && zeros == 2
    }

    pub(crate) fn is_king_move(non_zero: &[i32]) -> bool {
        matches!(non_zero, [] | [1] | [1, 1] | [1, 1, 1] | [1, 1, 1, 1])
    }

    pub(crate) fn is_rook_move(non_zero: &[i32]) -> bool {
        non_zero.len() == 1
    }

    pub(crate) fn is_bishop_move(non_zero: &[i32]) -> bool {
        non_zero.len() == 2 && Self::same_distance(non_zero)
    }

    pub(crate) fn is_unicorn_move(non_zero: &[i32]) -> bool {
        non_zero.len() == 3 && Self::same_distance(non_zero)
    }

    pub(crate) fn is_dragon_move(non_zero: &[i32]) -> bool {
        non_zero.len() == 4 && Self::same_distance(non_zero)
    }

    pub(crate) fn is_queen_move(non_zero: &[i32]) -> bool {
        !non_zero.is_empty() && Self::same_distance(non_zero)
    }

    pub(crate) fn same_distance(non_zero: &[i32]) -> bool {
        match non_zero {
            [] => false,
            [_] => true,
            [a, b] => a == b,
            [a, b, c] => a == b && a == c,
            [a, b, c, d] => a == b && a == c && a == d,
            _ => false,
        }
    }

    #[inline(always)]
    pub(crate) fn non_zero_distances(distances: [i32; 4]) -> ([i32; 4], usize) {
        let mut non_zero = [0; 4];
        let mut len = 0;
        for distance in distances {
            if distance > 0 {
                non_zero[len] = distance;
                len += 1;
            }
        }
        (non_zero, len)
    }

    pub(crate) fn castle_kind(
        &self,
        piece: Piece,
        from: Position,
        to: Position,
        delta: Delta,
    ) -> Option<MoveKind> {
        // Castling remains an orthodox same-board move; cross-time castling is
        // not legal. Rights already encode whether the king/rook moved earlier.
        if from.timeline_id != to.timeline_id
            || from.time != to.time
            || delta.y != 0
            || delta.t != 0
            || delta.l != 0
            || delta.x.abs() != 2
        {
            return None;
        }

        let board = self.board(from.timeline_id, from.time)?;
        let (home_y, kingside, queenside) = match piece.color {
            Color::White => (
                0,
                board.castling.white_kingside,
                board.castling.white_queenside,
            ),
            Color::Black => (
                7,
                board.castling.black_kingside,
                board.castling.black_queenside,
            ),
        };
        if from.x != 4 || from.y != home_y || to.y != home_y || self.piece_at(to).is_some() {
            return None;
        }

        let (rook_from_x, rook_to_x, clear_files, right) = if delta.x == 2 {
            (7, 5, [5, 6, -1], kingside)
        } else {
            (0, 3, [1, 2, 3], queenside)
        };
        if !right {
            return None;
        }
        if board.board[home_y as usize][rook_from_x as usize]
            != Some(Piece {
                color: piece.color,
                piece_type: PieceType::Rook,
            })
        {
            return None;
        }
        if clear_files
            .iter()
            .filter(|file| **file >= 0)
            .any(|file| board.board[home_y as usize][*file as usize].is_some())
        {
            return None;
        }

        Some(MoveKind::Castle {
            rook_from_x,
            rook_to_x,
        })
    }

    pub(crate) fn is_path_clear(&self, piece: Piece, from: Position, to: Position) -> bool {
        if from.timeline_id == to.timeline_id && from.time == to.time {
            let delta_x = to.x - from.x;
            let delta_y = to.y - from.y;
            let distance = delta_x.abs().max(delta_y.abs());
            if distance <= 1 {
                return true;
            }
            let Some(board) = self.board(from.timeline_id, from.time) else {
                return false;
            };
            let step_x = delta_x.signum();
            let step_y = delta_y.signum();
            return (1..distance).all(|step| {
                let x = from.x + step_x * step;
                let y = from.y + step_y * step;
                Self::in_bounds(x, y) && board.board[y as usize][x as usize].is_none()
            });
        }

        // Path walking follows timeline rows rather than ids. Ids encode
        // ownership/notation; rows encode geometry.
        let delta = self.movement_delta(from, to);
        let raw_delta = self.axis_delta(from, to);
        let distance = delta
            .x
            .abs()
            .max(delta.y.abs())
            .max(delta.t.abs())
            .max(delta.l.abs());
        let step_x = delta.x.signum();
        let step_y = delta.y.signum();
        let step_t = if distance == 0 {
            0
        } else {
            raw_delta.t / distance
        };
        let step_l = delta.l.signum();
        let from_row = self
            .timeline(from.timeline_id)
            .map_or(0, |timeline| timeline.row);

        for i in 1..distance {
            let Some(timeline) = self
                .timelines
                .iter()
                .find(|timeline| timeline.row == from_row + step_l * i)
            else {
                return false;
            };
            let Some(board) = self.board(timeline.id, from.time + step_t * i) else {
                return false;
            };
            // Passing through time only considers boards where this color was to
            // play; opponent-turn boards are not valid waypoints or blockers.
            if step_t != 0 && board.side_to_move != piece.color {
                continue;
            }
            let x = from.x + step_x * i;
            let y = from.y + step_y * i;

            if !Self::in_bounds(x, y) || board.board[y as usize][x as usize].is_some() {
                return false;
            }
        }

        true
    }

    pub(crate) fn axis_delta(&self, from: Position, to: Position) -> Delta {
        let timeline_delta = if from.timeline_id == to.timeline_id {
            0
        } else {
            let from_row = self
                .timeline(from.timeline_id)
                .map_or(0, |timeline| timeline.row);
            let to_row = self
                .timeline(to.timeline_id)
                .map_or(0, |timeline| timeline.row);
            to_row - from_row
        };

        Delta {
            x: to.x - from.x,
            y: to.y - from.y,
            t: to.time - from.time,
            l: timeline_delta,
        }
    }

    pub(crate) fn movement_delta(&self, from: Position, to: Position) -> Delta {
        let mut delta = self.axis_delta(from, to);
        // Cross-board time movement advances every other board because legal
        // destinations are restricted to boards where the mover was to play.
        if (from.timeline_id != to.timeline_id || from.time != to.time) && delta.t % 2 == 0 {
            delta.t /= 2;
        }
        delta
    }
}
