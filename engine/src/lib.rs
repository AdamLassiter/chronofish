use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Color {
    White,
    Black,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PieceType {
    King,
    Queen,
    Rook,
    Bishop,
    Knight,
    Pawn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Piece {
    color: Color,
    piece_type: PieceType,
}

#[derive(Clone)]
struct BoardSnapshot {
    time: i32,
    side_to_move: Color,
    board: [[Option<Piece>; 8]; 8],
    castling: CastlingRights,
    en_passant: Option<EnPassant>,
    origin: Origin,
}

#[derive(Clone)]
enum Origin {
    None,
    Move {
        from: Position,
        to: Position,
        move_type: &'static str,
    },
}

#[derive(Clone)]
struct Timeline {
    id: i32,
    row: i32,
    label: String,
    owner: TimelineOwner,
    boards: Vec<BoardSnapshot>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TimelineOwner {
    Neutral,
    White,
    Black,
}

#[derive(Clone, Copy)]
struct Position {
    timeline_id: i32,
    time: i32,
    x: i32,
    y: i32,
}

#[derive(Clone)]
struct Game {
    turn: Color,
    timelines: Vec<Timeline>,
    next_timeline_id: i32,
    last_message: String,
}

#[derive(Clone, Copy)]
struct CastlingRights {
    white_kingside: bool,
    white_queenside: bool,
    black_kingside: bool,
    black_queenside: bool,
}

#[derive(Clone, Copy)]
struct EnPassant {
    x: i32,
    y: i32,
    captured_x: i32,
    captured_y: i32,
}

#[derive(Clone, Copy)]
struct Delta {
    x: i32,
    y: i32,
    t: i32,
    l: i32,
}

#[derive(Clone, Copy)]
enum MoveKind {
    Standard,
    Branch,
    Castle { rook_from_x: i32, rook_to_x: i32 },
    EnPassant { captured_x: i32, captured_y: i32 },
}

thread_local! {
    static GAME: RefCell<Option<Game>> = const { RefCell::new(None) };
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[no_mangle]
pub extern "C" fn chronofish_version() -> *const u8 {
    set_output(env!("CARGO_PKG_VERSION").into())
}

#[no_mangle]
pub extern "C" fn chronofish_reset() {
    GAME.with(|game| {
        *game.borrow_mut() = Some(Game::new());
    });
}

#[no_mangle]
pub extern "C" fn chronofish_snapshot_json() -> *const u8 {
    let json = with_game(|game| game.to_json());
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_legal_targets_json(
    from_timeline_id: i32,
    from_time: i32,
    from_x: i32,
    from_y: i32,
) -> *const u8 {
    let from = Position {
        timeline_id: from_timeline_id,
        time: from_time,
        x: from_x,
        y: from_y,
    };
    let json = with_game(|game| game.legal_targets_json(from));
    set_output(json)
}

#[no_mangle]
pub extern "C" fn chronofish_apply_move(
    from_timeline_id: i32,
    from_time: i32,
    from_x: i32,
    from_y: i32,
    to_timeline_id: i32,
    to_time: i32,
    to_x: i32,
    to_y: i32,
) -> i32 {
    let from = Position {
        timeline_id: from_timeline_id,
        time: from_time,
        x: from_x,
        y: from_y,
    };
    let to = Position {
        timeline_id: to_timeline_id,
        time: to_time,
        x: to_x,
        y: to_y,
    };

    with_game_mut(|game| game.apply_move(from, to))
}

#[no_mangle]
pub extern "C" fn chronofish_last_message() -> *const u8 {
    let message = with_game(|game| game.last_message.clone());
    set_output(message)
}

#[no_mangle]
pub extern "C" fn chronofish_output_len() -> usize {
    OUTPUT.with(|output| output.borrow().len())
}

fn set_output(value: String) -> *const u8 {
    OUTPUT.with(|output| {
        let mut output = output.borrow_mut();
        *output = value.into_bytes();
        output.as_ptr()
    })
}

fn with_game<T>(callback: impl FnOnce(&Game) -> T) -> T {
    GAME.with(|game| {
        let mut game = game.borrow_mut();

        if game.is_none() {
            *game = Some(Game::new());
        }

        callback(game.as_ref().expect("game initialized"))
    })
}

fn with_game_mut<T>(callback: impl FnOnce(&mut Game) -> T) -> T {
    GAME.with(|game| {
        let mut game = game.borrow_mut();

        if game.is_none() {
            *game = Some(Game::new());
        }

        callback(game.as_mut().expect("game initialized"))
    })
}

impl Game {
    fn new() -> Self {
        let mut board = [[None; 8]; 8];
        let back_rank = [
            PieceType::Rook,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Queen,
            PieceType::King,
            PieceType::Bishop,
            PieceType::Knight,
            PieceType::Rook,
        ];

        for x in 0..8 {
            board[0][x] = Some(Piece {
                color: Color::White,
                piece_type: back_rank[x],
            });
            board[1][x] = Some(Piece {
                color: Color::White,
                piece_type: PieceType::Pawn,
            });
            board[6][x] = Some(Piece {
                color: Color::Black,
                piece_type: PieceType::Pawn,
            });
            board[7][x] = Some(Piece {
                color: Color::Black,
                piece_type: back_rank[x],
            });
        }

        Self {
            turn: Color::White,
            timelines: vec![Timeline {
                id: 0,
                row: 0,
                label: "Sacred Timeline".to_string(),
                owner: TimelineOwner::Neutral,
                boards: vec![BoardSnapshot {
                    time: 0,
                    side_to_move: Color::White,
                    board,
                    castling: CastlingRights::new(),
                    en_passant: None,
                    origin: Origin::None,
                }],
            }],
            next_timeline_id: 1,
            last_message: "Select a white piece on a latest board.".to_string(),
        }
    }

    fn timeline(&self, timeline_id: i32) -> Option<&Timeline> {
        self.timelines
            .iter()
            .find(|timeline| timeline.id == timeline_id)
    }

    fn timeline_mut(&mut self, timeline_id: i32) -> Option<&mut Timeline> {
        self.timelines
            .iter_mut()
            .find(|timeline| timeline.id == timeline_id)
    }

    fn board(&self, timeline_id: i32, time: i32) -> Option<&BoardSnapshot> {
        self.timeline(timeline_id)?
            .boards
            .iter()
            .find(|board| board.time == time)
    }

    fn latest_time(&self, timeline_id: i32) -> Option<i32> {
        self.timeline(timeline_id)?
            .boards
            .iter()
            .map(|board| board.time)
            .max()
    }

    fn is_latest_board(&self, timeline_id: i32, time: i32) -> bool {
        self.latest_time(timeline_id) == Some(time)
    }

    fn piece_at(&self, position: Position) -> Option<Piece> {
        if !Self::in_bounds(position.x, position.y) {
            return None;
        }

        self.board(position.timeline_id, position.time)?.board[position.y as usize]
            [position.x as usize]
    }

    fn can_move_to(&self, from: Position, to: Position) -> bool {
        let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
            return false;
        };

        let mut next = self.clone_for_search();
        next.apply_move_unchecked(from, to, piece, move_kind);
        !next.is_in_check(piece.color)
    }

    fn legal_move_kind(&self, from: Position, to: Position) -> Option<(Piece, MoveKind)> {
        if !Self::in_bounds(from.x, from.y) || !Self::in_bounds(to.x, to.y) {
            return None;
        }

        let source_board = self.board(from.timeline_id, from.time)?;
        let target_board = self.board(to.timeline_id, to.time)?;
        let piece = self.piece_at(from)?;
        let same_board = from.timeline_id == to.timeline_id && from.time == to.time;

        if !self.is_latest_board(from.timeline_id, from.time)
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

    fn legal_targets_json(&self, from: Position) -> String {
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

    fn apply_move(&mut self, from: Position, to: Position) -> i32 {
        let Some((piece, move_kind)) = self.legal_move_kind(from, to) else {
            self.last_message = "Illegal move.".to_string();
            return 0;
        };

        let mut next = self.clone_for_search();
        next.apply_move_unchecked(from, to, piece, move_kind);
        if next.is_in_check(piece.color) {
            self.last_message = "Illegal move: king would be in check.".to_string();
            return 0;
        }

        self.apply_move_unchecked(from, to, piece, move_kind);
        self.recompute_turn_from_present();
        let opponent = piece.color.opposite();
        let suffix = if self.is_checkmate(opponent) {
            " Checkmate."
        } else if self.is_in_check(opponent) {
            " Check."
        } else {
            ""
        };
        self.last_message = format!(
            "{}{}",
            self.move_message(
                piece,
                from,
                to,
                matches!(
                    move_kind,
                    MoveKind::Standard | MoveKind::Castle { .. } | MoveKind::EnPassant { .. }
                )
            ),
            suffix
        );
        1
    }

    fn clone_for_search(&self) -> Self {
        self.clone()
    }

    fn apply_move_unchecked(
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

        if !is_branch {
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

            self.timeline_mut(from.timeline_id)
                .expect("source timeline exists")
                .boards
                .push(BoardSnapshot {
                    time: from.time + 1,
                    side_to_move: next_turn,
                    board: next_board,
                    castling,
                    en_passant: en_passant_after_move(piece, from, to, move_kind),
                    origin: Origin::Move {
                        from,
                        to,
                        move_type: move_kind.name(),
                    },
                });
        } else {
            let destination_timeline_id = self.place_timeline(piece.color, source_row);

            let mut advanced_source = source_board.board;
            advanced_source[from.y as usize][from.x as usize] = None;
            let mut source_castling = source_board.castling;
            update_castling_rights(&mut source_castling, piece, from, to, source_board.board);
            self.timeline_mut(from.timeline_id)
                .expect("source timeline exists")
                .boards
                .push(BoardSnapshot {
                    time: from.time + 1,
                    side_to_move: next_turn,
                    board: advanced_source,
                    castling: source_castling,
                    en_passant: None,
                    origin: Origin::Move {
                        from,
                        to,
                        move_type: "source-advance",
                    },
                });

            let mut branch_board = target_board.board;
            branch_board[to.y as usize][to.x as usize] = Some(promote_if_needed(piece, to.y));
            let mut target_castling = target_board.castling;
            update_castling_rights(&mut target_castling, piece, from, to, target_board.board);
            self.timeline_mut(destination_timeline_id)
                .expect("destination timeline exists")
                .boards
                .push(BoardSnapshot {
                    time: to.time + 1,
                    side_to_move: next_turn,
                    board: branch_board,
                    castling: target_castling,
                    en_passant: None,
                    origin: Origin::Move {
                        from,
                        to,
                        move_type: "branch",
                    },
                });
        }
    }

    fn place_timeline(&mut self, owner: Color, source_row: i32) -> i32 {
        let direction = if owner == Color::White { 1 } else { -1 };
        let mut row = source_row + direction;

        while self.timelines.iter().any(|timeline| timeline.row == row) {
            row += direction;
        }

        let id = self.next_timeline_id;
        self.next_timeline_id += 1;
        self.timelines.push(Timeline {
            id,
            row,
            label: format!(
                "{} branch {}",
                if owner == Color::White {
                    "White"
                } else {
                    "Black"
                },
                id
            ),
            owner: TimelineOwner::from_color(owner),
            boards: Vec::new(),
        });
        id
    }

    fn recompute_turn_from_present(&mut self) {
        if let Some(board) = self.present_board() {
            self.turn = board.side_to_move;
        }
    }

    fn present_board(&self) -> Option<&BoardSnapshot> {
        self.timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| timeline.boards.iter().max_by_key(|board| board.time))
            .min_by_key(|board| board.time)
    }

    fn is_active_timeline(&self, timeline_id: i32) -> bool {
        let Some(timeline) = self.timeline(timeline_id) else {
            return false;
        };
        match timeline.owner {
            TimelineOwner::Neutral => true,
            TimelineOwner::White | TimelineOwner::Black => {
                let same_owner = self
                    .timelines
                    .iter()
                    .filter(|candidate| {
                        candidate.owner == timeline.owner && candidate.id <= timeline.id
                    })
                    .count();
                let opponent_owner = match timeline.owner {
                    TimelineOwner::White => TimelineOwner::Black,
                    TimelineOwner::Black => TimelineOwner::White,
                    TimelineOwner::Neutral => TimelineOwner::Neutral,
                };
                let opponent_count = self
                    .timelines
                    .iter()
                    .filter(|candidate| candidate.owner == opponent_owner)
                    .count();
                same_owner <= opponent_count + 1
            }
        }
    }

    fn is_in_check(&self, color: Color) -> bool {
        let kings = self.king_positions(color);
        kings
            .iter()
            .any(|king| self.is_square_attacked(*king, color.opposite()))
    }

    fn is_checkmate(&self, color: Color) -> bool {
        if !self.is_in_check(color) {
            return false;
        }

        let mut search = self.clone_for_search();
        search.turn = color;
        !search.has_legal_move(color)
    }

    fn has_legal_move(&self, color: Color) -> bool {
        self.timelines.iter().any(|timeline| {
            timeline.boards.iter().any(|board| {
                self.is_latest_board(timeline.id, board.time)
                    && board.side_to_move == color
                    && board.board.iter().enumerate().any(|(y, rank)| {
                        rank.iter().enumerate().any(|(x, piece)| {
                            piece.is_some_and(|piece| piece.color == color)
                                && self.has_target_from(Position {
                                    timeline_id: timeline.id,
                                    time: board.time,
                                    x: x as i32,
                                    y: y as i32,
                                })
                        })
                    })
            })
        })
    }

    fn has_target_from(&self, from: Position) -> bool {
        self.timelines.iter().any(|timeline| {
            timeline.boards.iter().any(|board| {
                (0..8).any(|y| {
                    (0..8).any(|x| {
                        self.can_move_to(
                            from,
                            Position {
                                timeline_id: timeline.id,
                                time: board.time,
                                x,
                                y,
                            },
                        )
                    })
                })
            })
        })
    }

    fn king_positions(&self, color: Color) -> Vec<Position> {
        let mut positions = Vec::new();
        for timeline in &self.timelines {
            for board in &timeline.boards {
                if !self.is_latest_board(timeline.id, board.time) {
                    continue;
                }
                for y in 0..8 {
                    for x in 0..8 {
                        if board.board[y][x]
                            == Some(Piece {
                                color,
                                piece_type: PieceType::King,
                            })
                        {
                            positions.push(Position {
                                timeline_id: timeline.id,
                                time: board.time,
                                x: x as i32,
                                y: y as i32,
                            });
                        }
                    }
                }
            }
        }
        positions
    }

    fn is_square_attacked(&self, target: Position, by_color: Color) -> bool {
        self.timelines.iter().any(|timeline| {
            timeline.boards.iter().any(|board| {
                self.is_latest_board(timeline.id, board.time)
                    && board.board.iter().enumerate().any(|(y, rank)| {
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
        })
    }

    fn attacks_square(&self, piece: Piece, from: Position, target: Position) -> bool {
        if !Self::in_bounds(target.x, target.y)
            || from.timeline_id == target.timeline_id
                && from.time == target.time
                && from.x == target.x
                && from.y == target.y
        {
            return false;
        }

        if (from.timeline_id != target.timeline_id || from.time != target.time)
            && self
                .board(target.timeline_id, target.time)
                .is_some_and(|board| board.side_to_move != piece.color)
        {
            return false;
        }

        let delta = self.movement_delta(from, target);
        let distances = [delta.x.abs(), delta.y.abs(), delta.t.abs(), delta.l.abs()];
        let non_zero: Vec<i32> = distances
            .iter()
            .copied()
            .filter(|distance| *distance > 0)
            .collect();

        match piece.piece_type {
            PieceType::Pawn => {
                let forward = if piece.color == Color::White { 1 } else { -1 };
                let timeline_forward = if piece.color == Color::White { 1 } else { -1 };
                (from.timeline_id == target.timeline_id
                    && from.time == target.time
                    && delta.x.abs() == 1
                    && delta.y == forward)
                    || (delta.t.abs() == 1
                        && delta.l == timeline_forward
                        && delta.x == 0
                        && delta.y == 0)
            }
            PieceType::Knight => {
                let mut sorted = distances;
                sorted.sort_by(|a, b| b.cmp(a));
                sorted[0] == 2 && sorted[1] == 1 && sorted[2] == 0 && sorted[3] == 0
            }
            PieceType::King => non_zero.iter().all(|distance| *distance == 1),
            PieceType::Rook => non_zero.len() == 1 && self.is_path_clear(piece, from, target),
            PieceType::Bishop => {
                non_zero.len() == 2
                    && non_zero[0] == non_zero[1]
                    && self.is_path_clear(piece, from, target)
            }
            PieceType::Queen => {
                !non_zero.is_empty()
                    && non_zero.iter().all(|distance| *distance == non_zero[0])
                    && self.is_path_clear(piece, from, target)
            }
        }
    }

    fn move_kind_for(&self, piece: Piece, from: Position, to: Position) -> Option<MoveKind> {
        let delta = self.movement_delta(from, to);
        let distances = [delta.x.abs(), delta.y.abs(), delta.t.abs(), delta.l.abs()];
        let non_zero: Vec<i32> = distances
            .iter()
            .copied()
            .filter(|distance| *distance > 0)
            .collect();

        if non_zero.is_empty() {
            return None;
        }

        if piece.piece_type == PieceType::King {
            if let Some(castle) = self.castle_kind(piece, from, to, delta) {
                return Some(castle);
            }
        }

        let legal = match piece.piece_type {
            PieceType::Rook => non_zero.len() == 1 && self.is_path_clear(piece, from, to),
            PieceType::Bishop => {
                non_zero.len() == 2
                    && non_zero[0] == non_zero[1]
                    && self.is_path_clear(piece, from, to)
            }
            PieceType::Queen => {
                non_zero.iter().all(|distance| *distance == non_zero[0])
                    && self.is_path_clear(piece, from, to)
            }
            PieceType::King => non_zero.iter().all(|distance| *distance == 1),
            PieceType::Knight => {
                let mut sorted = distances;
                sorted.sort_by(|a, b| b.cmp(a));
                sorted[0] == 2 && sorted[1] == 1 && sorted[2] == 0 && sorted[3] == 0
            }
            PieceType::Pawn => return self.pawn_move_kind(piece, from, to, delta),
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

    fn pawn_move_kind(
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

        if same_board && delta.x == 0 && delta.y == forward && destination.is_none() {
            return Some(MoveKind::Standard);
        }

        if same_board
            && delta.x == 0
            && delta.y == forward * 2
            && !has_moved
            && destination.is_none()
        {
            return self.board(from.timeline_id, from.time).and_then(|board| {
                board.board[(from.y + forward) as usize][from.x as usize]
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

        if same_board
            && delta.x.abs() == 1
            && delta.y == forward
            && destination.is_none()
            && self
                .board(from.timeline_id, from.time)
                .and_then(|board| board.en_passant)
                .is_some_and(|target| target.x == to.x && target.y == to.y)
        {
            let target = self
                .board(from.timeline_id, from.time)
                .and_then(|board| board.en_passant)
                .expect("checked en passant target");
            return Some(MoveKind::EnPassant {
                captured_x: target.captured_x,
                captured_y: target.captured_y,
            });
        }

        let timeline_forward = if piece.color == Color::White { 1 } else { -1 };
        (delta.t.abs() == 1
            && delta.l == timeline_forward
            && delta.x == 0
            && delta.y == 0
            && destination.is_some_and(|target| target.color != piece.color))
        .then_some(MoveKind::Branch)
    }

    fn castle_kind(
        &self,
        piece: Piece,
        from: Position,
        to: Position,
        delta: Delta,
    ) -> Option<MoveKind> {
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

    fn is_path_clear(&self, piece: Piece, from: Position, to: Position) -> bool {
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

    fn axis_delta(&self, from: Position, to: Position) -> Delta {
        let from_row = self
            .timeline(from.timeline_id)
            .map_or(0, |timeline| timeline.row);
        let to_row = self
            .timeline(to.timeline_id)
            .map_or(0, |timeline| timeline.row);

        Delta {
            x: to.x - from.x,
            y: to.y - from.y,
            t: to.time - from.time,
            l: to_row - from_row,
        }
    }

    fn movement_delta(&self, from: Position, to: Position) -> Delta {
        let mut delta = self.axis_delta(from, to);
        if (from.timeline_id != to.timeline_id || from.time != to.time) && delta.t % 2 == 0 {
            delta.t /= 2;
        }
        delta
    }

    fn move_message(
        &self,
        piece: Piece,
        from: Position,
        to: Position,
        is_standard_move: bool,
    ) -> String {
        let move_name = format!(
            "{}{} to {}{}",
            file_name(from.x),
            from.y + 1,
            file_name(to.x),
            to.y + 1
        );

        if is_standard_move {
            format!("{} played {}.", piece.color.capitalized(), move_name)
        } else {
            format!("{} branched {}.", piece.color.capitalized(), move_name)
        }
    }

    fn to_json(&self) -> String {
        let mut timelines = self.timelines.clone();
        timelines.sort_by(|left, right| left.row.cmp(&right.row).then(left.id.cmp(&right.id)));

        format!(
            "{{\"turn\":\"{}\",\"timelines\":[{}],\"nextTimelineId\":{}}}",
            self.turn.as_str(),
            timelines
                .iter()
                .map(Timeline::to_json)
                .collect::<Vec<_>>()
                .join(","),
            self.next_timeline_id
        )
    }

    fn in_bounds(x: i32, y: i32) -> bool {
        (0..8).contains(&x) && (0..8).contains(&y)
    }
}

impl Timeline {
    fn to_json(&self) -> String {
        let mut boards = self.boards.clone();
        boards.sort_by_key(|board| board.time);

        format!(
            "{{\"id\":{},\"row\":{},\"label\":\"{}\",\"owner\":\"{}\",\"boards\":[{}]}}",
            self.id,
            self.row,
            escape_json(&self.label),
            self.owner.as_str(),
            boards
                .iter()
                .map(|board| board.to_json(self.id))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

impl BoardSnapshot {
    fn to_json(&self, timeline_id: i32) -> String {
        let ranks = self
            .board
            .iter()
            .map(|rank| {
                format!(
                    "[{}]",
                    rank.iter().map(piece_json).collect::<Vec<_>>().join(",")
                )
            })
            .collect::<Vec<_>>()
            .join(",");

        format!(
            "{{\"id\":\"{}:{}\",\"time\":{},\"sideToMove\":\"{}\",\"board\":[{}],\"origin\":{}}}",
            timeline_id,
            self.time,
            self.time,
            self.side_to_move.as_str(),
            ranks,
            self.origin.to_json()
        )
    }
}

impl Origin {
    fn to_json(&self) -> String {
        match self {
            Origin::None => "null".to_string(),
            Origin::Move {
                from,
                to,
                move_type,
            } => format!(
                "{{\"from\":{},\"to\":{},\"type\":\"{}\"}}",
                position_json(*from),
                position_json(*to),
                move_type
            ),
        }
    }
}

impl Color {
    fn opposite(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Color::White => "white",
            Color::Black => "black",
        }
    }

    fn capitalized(self) -> &'static str {
        match self {
            Color::White => "White",
            Color::Black => "Black",
        }
    }
}

impl PieceType {
    fn as_str(self) -> &'static str {
        match self {
            PieceType::King => "king",
            PieceType::Queen => "queen",
            PieceType::Rook => "rook",
            PieceType::Bishop => "bishop",
            PieceType::Knight => "knight",
            PieceType::Pawn => "pawn",
        }
    }
}

impl TimelineOwner {
    fn from_color(color: Color) -> Self {
        match color {
            Color::White => TimelineOwner::White,
            Color::Black => TimelineOwner::Black,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            TimelineOwner::Neutral => "neutral",
            TimelineOwner::White => "white",
            TimelineOwner::Black => "black",
        }
    }
}

impl CastlingRights {
    fn new() -> Self {
        Self {
            white_kingside: true,
            white_queenside: true,
            black_kingside: true,
            black_queenside: true,
        }
    }
}

impl MoveKind {
    fn name(self) -> &'static str {
        match self {
            MoveKind::Standard => "standard",
            MoveKind::Branch => "branch",
            MoveKind::Castle { .. } => "castle",
            MoveKind::EnPassant { .. } => "en-passant",
        }
    }
}

fn en_passant_after_move(
    piece: Piece,
    from: Position,
    to: Position,
    move_kind: MoveKind,
) -> Option<EnPassant> {
    if piece.piece_type != PieceType::Pawn || !matches!(move_kind, MoveKind::Standard) {
        return None;
    }

    let forward = if piece.color == Color::White { 1 } else { -1 };
    (from.x == to.x && to.y - from.y == forward * 2).then_some(EnPassant {
        x: from.x,
        y: from.y + forward,
        captured_x: to.x,
        captured_y: to.y,
    })
}

fn update_castling_rights(
    castling: &mut CastlingRights,
    piece: Piece,
    from: Position,
    to: Position,
    board_before: [[Option<Piece>; 8]; 8],
) {
    match (piece.color, piece.piece_type) {
        (Color::White, PieceType::King) => {
            castling.white_kingside = false;
            castling.white_queenside = false;
        }
        (Color::Black, PieceType::King) => {
            castling.black_kingside = false;
            castling.black_queenside = false;
        }
        (Color::White, PieceType::Rook) if from.y == 0 && from.x == 0 => {
            castling.white_queenside = false;
        }
        (Color::White, PieceType::Rook) if from.y == 0 && from.x == 7 => {
            castling.white_kingside = false;
        }
        (Color::Black, PieceType::Rook) if from.y == 7 && from.x == 0 => {
            castling.black_queenside = false;
        }
        (Color::Black, PieceType::Rook) if from.y == 7 && from.x == 7 => {
            castling.black_kingside = false;
        }
        _ => {}
    }

    match board_before[to.y as usize][to.x as usize] {
        Some(Piece {
            color: Color::White,
            piece_type: PieceType::Rook,
        }) if to.y == 0 && to.x == 0 => castling.white_queenside = false,
        Some(Piece {
            color: Color::White,
            piece_type: PieceType::Rook,
        }) if to.y == 0 && to.x == 7 => castling.white_kingside = false,
        Some(Piece {
            color: Color::Black,
            piece_type: PieceType::Rook,
        }) if to.y == 7 && to.x == 0 => castling.black_queenside = false,
        Some(Piece {
            color: Color::Black,
            piece_type: PieceType::Rook,
        }) if to.y == 7 && to.x == 7 => castling.black_kingside = false,
        _ => {}
    }
}

fn promote_if_needed(piece: Piece, y: i32) -> Piece {
    if piece.piece_type != PieceType::Pawn {
        return piece;
    }

    if (piece.color == Color::White && y == 7) || (piece.color == Color::Black && y == 0) {
        Piece {
            color: piece.color,
            piece_type: PieceType::Queen,
        }
    } else {
        piece
    }
}

fn piece_json(piece: &Option<Piece>) -> String {
    match piece {
        Some(piece) => format!(
            "{{\"color\":\"{}\",\"type\":\"{}\"}}",
            piece.color.as_str(),
            piece.piece_type.as_str()
        ),
        None => "null".to_string(),
    }
}

fn position_json(position: Position) -> String {
    format!(
        "{{\"timelineId\":{},\"time\":{},\"x\":{},\"y\":{}}}",
        position.timeline_id, position.time, position.x, position.y
    )
}

fn file_name(x: i32) -> &'static str {
    match x {
        0 => "a",
        1 => "b",
        2 => "c",
        3 => "d",
        4 => "e",
        5 => "f",
        6 => "g",
        7 => "h",
        _ => "?",
    }
}

fn escape_json(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_board_with_kings() -> [[Option<Piece>; 8]; 8] {
        let mut board = [[None; 8]; 8];
        board[0][4] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::King,
        });
        board[7][4] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::King,
        });
        board
    }

    fn snapshot(time: i32, side_to_move: Color, board: [[Option<Piece>; 8]; 8]) -> BoardSnapshot {
        BoardSnapshot {
            time,
            side_to_move,
            board,
            castling: CastlingRights::new(),
            en_passant: None,
            origin: Origin::None,
        }
    }

    #[test]
    fn starts_with_white_to_move() {
        let game = Game::new();
        assert_eq!(game.turn.as_str(), "white");
        assert_eq!(game.timelines.len(), 1);
    }

    #[test]
    fn standard_move_advances_main_timeline() {
        let mut game = Game::new();
        assert_eq!(
            game.apply_move(
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 4,
                    y: 1
                },
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 4,
                    y: 3
                },
            ),
            1
        );

        assert_eq!(game.latest_time(0), Some(1));
        assert_eq!(game.turn.as_str(), "black");
        assert!(game.board(0, 1).expect("new board").board[1][4].is_none());
    }

    #[test]
    fn branch_move_advances_source_and_destination() {
        let mut game = Game::new();
        let empty = [[None; 8]; 8];
        let mut latest = empty;
        latest[0][0] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Rook,
        });
        latest[0][4] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::King,
        });
        latest[7][4] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::King,
        });
        game.timelines[0].boards = vec![
            BoardSnapshot {
                time: 0,
                side_to_move: Color::White,
                board: empty,
                castling: CastlingRights::new(),
                en_passant: None,
                origin: Origin::None,
            },
            BoardSnapshot {
                time: 1,
                side_to_move: Color::Black,
                board: empty,
                castling: CastlingRights::new(),
                en_passant: None,
                origin: Origin::None,
            },
            BoardSnapshot {
                time: 2,
                side_to_move: Color::White,
                board: latest,
                castling: CastlingRights::new(),
                en_passant: None,
                origin: Origin::None,
            },
        ];

        assert_eq!(
            game.apply_move(
                Position {
                    timeline_id: 0,
                    time: 2,
                    x: 0,
                    y: 0,
                },
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 0,
                    y: 0,
                },
            ),
            1
        );

        assert_eq!(game.timelines.len(), 2);
        assert!(game.board(0, 3).is_some());
        assert!(game.board(1, 1).is_some());
    }

    #[test]
    fn time_travel_only_targets_boards_where_mover_is_to_play() {
        let mut game = Game::new();
        let empty = [[None; 8]; 8];
        let mut latest = empty;
        latest[0][0] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Rook,
        });
        latest[0][4] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::King,
        });
        latest[7][4] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::King,
        });
        game.timelines[0].boards = vec![
            BoardSnapshot {
                time: 0,
                side_to_move: Color::White,
                board: empty,
                castling: CastlingRights::new(),
                en_passant: None,
                origin: Origin::None,
            },
            BoardSnapshot {
                time: 1,
                side_to_move: Color::Black,
                board: empty,
                castling: CastlingRights::new(),
                en_passant: None,
                origin: Origin::None,
            },
            BoardSnapshot {
                time: 2,
                side_to_move: Color::White,
                board: latest,
                castling: CastlingRights::new(),
                en_passant: None,
                origin: Origin::None,
            },
        ];

        assert!(!game.can_move_to(
            Position {
                timeline_id: 0,
                time: 2,
                x: 0,
                y: 0,
            },
            Position {
                timeline_id: 0,
                time: 1,
                x: 0,
                y: 0,
            },
        ));
    }

    #[test]
    fn time_travel_distance_counts_same_color_boards() {
        let mut game = Game::new();
        let empty = [[None; 8]; 8];
        let mut latest = empty;
        latest[0][4] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::King,
        });
        latest[7][4] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::King,
        });
        game.timelines[0].boards = vec![
            snapshot(0, Color::White, empty),
            snapshot(1, Color::Black, empty),
            snapshot(2, Color::White, latest),
        ];

        assert!(game.can_move_to(
            Position {
                timeline_id: 0,
                time: 2,
                x: 4,
                y: 0,
            },
            Position {
                timeline_id: 0,
                time: 0,
                x: 3,
                y: 0,
            },
        ));
    }

    #[test]
    fn en_passant_is_available_only_on_the_immediate_reply_board() {
        let mut game = Game::new();
        let mut board = empty_board_with_kings();
        board[4][4] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Pawn,
        });
        board[6][3] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::Pawn,
        });
        game.turn = Color::Black;
        game.timelines[0].boards = vec![snapshot(0, Color::Black, board)];

        assert_eq!(
            game.apply_move(
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 3,
                    y: 6,
                },
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 3,
                    y: 4,
                },
            ),
            1
        );
        assert!(game.can_move_to(
            Position {
                timeline_id: 0,
                time: 1,
                x: 4,
                y: 4,
            },
            Position {
                timeline_id: 0,
                time: 1,
                x: 3,
                y: 5,
            },
        ));

        assert_eq!(
            game.apply_move(
                Position {
                    timeline_id: 0,
                    time: 1,
                    x: 4,
                    y: 4,
                },
                Position {
                    timeline_id: 0,
                    time: 1,
                    x: 3,
                    y: 5,
                },
            ),
            1
        );
        let board_after = game.board(0, 2).expect("en passant result");
        assert!(board_after.board[4][3].is_none());
        assert_eq!(
            board_after.board[5][3],
            Some(Piece {
                color: Color::White,
                piece_type: PieceType::Pawn,
            })
        );
        assert!(board_after.en_passant.is_none());
    }

    #[test]
    fn castling_moves_the_rook_and_expires_rights() {
        let mut game = Game::new();
        let mut board = [[None; 8]; 8];
        board[0][4] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::King,
        });
        board[0][7] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Rook,
        });
        board[7][4] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::King,
        });
        game.timelines[0].boards = vec![snapshot(0, Color::White, board)];

        assert_eq!(
            game.apply_move(
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 4,
                    y: 0,
                },
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 6,
                    y: 0,
                },
            ),
            1
        );

        let board_after = game.board(0, 1).expect("castled board");
        assert_eq!(
            board_after.board[0][6],
            Some(Piece {
                color: Color::White,
                piece_type: PieceType::King,
            })
        );
        assert_eq!(
            board_after.board[0][5],
            Some(Piece {
                color: Color::White,
                piece_type: PieceType::Rook,
            })
        );
        assert!(!board_after.castling.white_kingside);
        assert!(!board_after.castling.white_queenside);
    }

    #[test]
    fn present_line_keeps_turn_until_leftmost_active_board_advances() {
        let mut game = Game::new();
        let mut board_a = empty_board_with_kings();
        board_a[1][0] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Pawn,
        });
        let mut board_b = empty_board_with_kings();
        board_b[1][1] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Pawn,
        });
        game.timelines.push(Timeline {
            id: 1,
            row: 1,
            label: "White branch 1".to_string(),
            owner: TimelineOwner::White,
            boards: vec![snapshot(0, Color::White, board_b)],
        });
        game.next_timeline_id = 2;
        game.timelines[0].boards = vec![snapshot(0, Color::White, board_a)];

        assert_eq!(
            game.apply_move(
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 0,
                    y: 1,
                },
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 0,
                    y: 2,
                },
            ),
            1
        );
        assert_eq!(game.turn, Color::White);

        assert_eq!(
            game.apply_move(
                Position {
                    timeline_id: 1,
                    time: 0,
                    x: 1,
                    y: 1,
                },
                Position {
                    timeline_id: 1,
                    time: 0,
                    x: 1,
                    y: 2,
                },
            ),
            1
        );
        assert_eq!(game.turn, Color::Black);
    }

    #[test]
    fn moves_that_leave_king_in_check_are_rejected() {
        let mut game = Game::new();
        let mut board = [[None; 8]; 8];
        board[0][4] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::King,
        });
        board[1][4] = Some(Piece {
            color: Color::White,
            piece_type: PieceType::Rook,
        });
        board[7][4] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::Rook,
        });
        board[7][0] = Some(Piece {
            color: Color::Black,
            piece_type: PieceType::King,
        });
        game.timelines[0].boards = vec![snapshot(0, Color::White, board)];

        assert_eq!(
            game.apply_move(
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 4,
                    y: 1,
                },
                Position {
                    timeline_id: 0,
                    time: 0,
                    x: 5,
                    y: 1,
                },
            ),
            0
        );
    }
}
