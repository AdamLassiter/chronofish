const NEURAL_MAX_BOARDS: usize = 16;
const NEURAL_BOARD_PLANES: usize = 32;
const NEURAL_BOARD_SQUARES: usize = 64;
const NEURAL_INPUT_SIZE: usize = NEURAL_MAX_BOARDS * NEURAL_BOARD_PLANES * NEURAL_BOARD_SQUARES;
const DEFAULT_PROJECTION_SEED: u32 = 2_166_136_261;
const COMPACT_MODEL_MAGIC: &[u8; 4] = b"CFNN";

#[derive(Clone)]
struct HeuristicEvaluator;

#[derive(Clone)]
#[allow(dead_code)]
struct NeuralEvaluator {
    model_path: Option<String>,
    model: Option<NeuralLinearModel>,
}

#[derive(Clone)]
struct HybridEvaluator {
    heuristic_weight: i32,
    neural_weight: i32,
    neural: NeuralEvaluator,
}

#[derive(Clone)]
enum ValueEvaluator {
    Heuristic(HeuristicEvaluator),
    Neural(NeuralEvaluator),
    Hybrid(HybridEvaluator),
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
struct NeuralLinearModel {
    bias: f32,
    scale: f32,
    #[serde(default)]
    feature_weights: Vec<f32>,
    #[serde(default)]
    projection_size: usize,
    #[serde(default = "default_projection_seed")]
    projection_seed: u32,
    #[serde(default)]
    hidden_layers: Vec<usize>,
    #[serde(default)]
    hidden_weights: Vec<f32>,
}

struct NeuralEncodedPosition {
    values: Vec<f32>,
    board_count: usize,
}

impl HeuristicEvaluator {
    fn evaluate(&self, game: &Game, color: Color, weights: &EvalWeights) -> i32 {
        game.evaluate_heuristic(color, weights)
    }
}

impl NeuralEvaluator {
    fn missing_model(path: Option<String>) -> Self {
        Self { model_path: path, model: None }
    }

    fn from_model(model: NeuralLinearModel) -> Self {
        Self {
            model_path: None,
            model: Some(model),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(dead_code)]
    fn from_path(path: impl Into<String>) -> Self {
        let path = path.into();
        let model = std::fs::read_to_string(&path)
            .ok()
            .and_then(|json| serde_json::from_str(&json).ok());
        Self {
            model_path: Some(path),
            model,
        }
    }

    fn predict(&self, game: &Game, color: Color) -> Option<i32> {
        let model = self.model.as_ref()?;
        let encoded = game.encode_neural_position(color);
        let mut score = model.bias;
        if model.projection_size > 0 && !model.hidden_layers.is_empty() {
            let projected = project_neural_features(
                &encoded.values,
                model.projection_size,
                model.projection_seed,
            );
            let hidden = evaluate_hidden_layers(&projected, &model.hidden_layers, &model.hidden_weights)?;
            for (value, weight) in hidden.iter().zip(model.feature_weights.iter()) {
                score += value * weight;
            }
            if model.feature_weights.len() > hidden.len() {
                score += model.feature_weights[hidden.len()];
            }
        } else if model.projection_size > 0 && model.feature_weights.len() == model.projection_size {
            let projected = project_neural_features(
                &encoded.values,
                model.projection_size,
                model.projection_seed,
            );
            for (value, weight) in projected.iter().zip(model.feature_weights.iter()) {
                score += value * weight;
            }
        } else {
            for (value, weight) in encoded.values.iter().zip(model.feature_weights.iter()) {
                score += value * weight;
            }
        }
        let board_scale = encoded.board_count.max(1) as f32;
        Some(
            (score * model.scale / board_scale)
                .round()
                .clamp(-(CHECKMATE_SCORE as f32), CHECKMATE_SCORE as f32) as i32,
        )
    }

    fn evaluate(&self, game: &Game, color: Color, weights: &EvalWeights) -> i32 {
        self.predict(game, color)
            .unwrap_or_else(|| HeuristicEvaluator.evaluate(game, color, weights))
    }

    #[allow(dead_code)]
    fn is_available(&self) -> bool {
        self.model.is_some()
    }

    #[allow(dead_code)]
    fn model_path(&self) -> Option<&str> {
        self.model_path.as_deref()
    }
}

impl HybridEvaluator {
    fn evaluate(&self, game: &Game, color: Color, weights: &EvalWeights) -> i32 {
        let heuristic = HeuristicEvaluator.evaluate(game, color, weights);
        let Some(neural) = self.neural.predict(game, color) else {
            return heuristic;
        };
        let total_weight = (self.heuristic_weight + self.neural_weight).max(1);
        (heuristic * self.heuristic_weight + neural * self.neural_weight) / total_weight
    }
}

impl ValueEvaluator {
    fn heuristic() -> Self {
        Self::Heuristic(HeuristicEvaluator)
    }

    #[allow(dead_code)]
    fn neural(model_path: Option<String>) -> Self {
        Self::Neural(NeuralEvaluator::missing_model(model_path))
    }

    fn hybrid_from_model(model: NeuralLinearModel, heuristic_weight: i32, neural_weight: i32) -> Self {
        Self::Hybrid(HybridEvaluator {
            heuristic_weight,
            neural_weight,
            neural: NeuralEvaluator::from_model(model),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(dead_code)]
    fn neural_from_path(path: impl Into<String>) -> Self {
        Self::Neural(NeuralEvaluator::from_path(path))
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(dead_code)]
    fn hybrid_from_path(path: impl Into<String>, heuristic_weight: i32, neural_weight: i32) -> Self {
        Self::Hybrid(HybridEvaluator {
            heuristic_weight,
            neural_weight,
            neural: NeuralEvaluator::from_path(path),
        })
    }

    fn evaluate(&self, game: &Game, color: Color, weights: &EvalWeights) -> i32 {
        match self {
            Self::Heuristic(evaluator) => evaluator.evaluate(game, color, weights),
            Self::Neural(evaluator) => evaluator.evaluate(game, color, weights),
            Self::Hybrid(evaluator) => evaluator.evaluate(game, color, weights),
        }
    }

    #[allow(dead_code)]
    fn neural_available(&self) -> bool {
        match self {
            Self::Heuristic(_) => false,
            Self::Neural(evaluator) => evaluator.is_available(),
            Self::Hybrid(evaluator) => evaluator.neural.is_available(),
        }
    }

    #[allow(dead_code)]
    fn model_path(&self) -> Option<&str> {
        match self {
            Self::Heuristic(_) => None,
            Self::Neural(evaluator) => evaluator.model_path(),
            Self::Hybrid(evaluator) => evaluator.neural.model_path(),
        }
    }
}

impl Game {
    fn encode_neural_position(&self, color: Color) -> NeuralEncodedPosition {
        let mut values = vec![0.0; NEURAL_INPUT_SIZE];
        let selected = self.neural_board_selection();
        let board_count = selected.len();
        let present_time = self.present_board().map(|board| board.time).unwrap_or(0);

        for (board_index, (timeline_index, board_index_in_timeline)) in
            selected.into_iter().enumerate()
        {
            let timeline = &self.timelines[timeline_index];
            let board = &timeline.boards[board_index_in_timeline];
            let latest = self.is_latest_board(timeline.id, board.time);
            let active = self.is_active_timeline(timeline.id);
            let owner_sign = match timeline.owner {
                TimelineOwner::Neutral => 0.0,
                TimelineOwner::White => relative_color_value(Color::White, color),
                TimelineOwner::Black => relative_color_value(Color::Black, color),
            };
            let time_distance = ((board.time - present_time).clamp(-16, 16) as f32) / 16.0;

            for y in 0..8 {
                for x in 0..8 {
                    let square = y * 8 + x;
                    if let Some(piece) = board.board[y][x] {
                        let plane = neural_piece_plane(piece);
                        values[neural_feature_index(board_index, plane, square)] = 1.0;
                    }
                    values[neural_feature_index(board_index, 24, square)] =
                        relative_color_value(board.side_to_move, color);
                    values[neural_feature_index(board_index, 25, square)] =
                        if active { 1.0 } else { 0.0 };
                    values[neural_feature_index(board_index, 26, square)] =
                        if latest { 1.0 } else { 0.0 };
                    values[neural_feature_index(board_index, 27, square)] =
                        if board.time == present_time { 1.0 } else { 0.0 };
                    values[neural_feature_index(board_index, 28, square)] = owner_sign;
                    values[neural_feature_index(board_index, 29, square)] = time_distance;
                    values[neural_feature_index(board_index, 30, square)] = 1.0;
                    values[neural_feature_index(board_index, 31, square)] = if latest
                        && (self.is_in_check(color) || self.is_in_check(color.opposite()))
                    {
                        1.0
                    } else {
                        0.0
                    };
                }
            }
        }

        NeuralEncodedPosition { values, board_count }
    }

    fn neural_board_selection(&self) -> Vec<(usize, usize)> {
        let mut boards = Vec::new();
        for (timeline_index, timeline) in self.timelines.iter().enumerate() {
            for (board_index, board) in timeline.boards.iter().enumerate() {
                let latest = self.is_latest_board(timeline.id, board.time);
                let active = self.is_active_timeline(timeline.id);
                let has_royal = board.board.iter().flatten().any(|piece| {
                    piece.is_some_and(|piece| Self::is_royal_piece(piece.piece_type))
                });
                let has_recent_origin = matches!(board.origin, Origin::Move { .. });
                if latest || has_royal || has_recent_origin {
                    let category = match (latest, active) {
                        (true, true) => 0,
                        (true, false) => 1,
                        (false, _) if has_royal => 2,
                        _ => 3,
                    };
                    boards.push((
                        category,
                        -board.time,
                        timeline.id.abs(),
                        timeline.id,
                        timeline_index,
                        board_index,
                    ));
                }
            }
        }
        boards.sort();
        boards
            .into_iter()
            .take(NEURAL_MAX_BOARDS)
            .map(|(_, _, _, _, timeline_index, board_index)| (timeline_index, board_index))
            .collect()
    }
}

fn neural_feature_index(board: usize, plane: usize, square: usize) -> usize {
    board * NEURAL_BOARD_PLANES * NEURAL_BOARD_SQUARES + plane * NEURAL_BOARD_SQUARES + square
}

fn relative_color_value(color: Color, perspective: Color) -> f32 {
    if color == perspective {
        1.0
    } else {
        -1.0
    }
}

fn neural_piece_plane(piece: Piece) -> usize {
    let offset = if piece.color == Color::White { 0 } else { 12 };
    offset
        + match piece.piece_type {
            PieceType::King => 0,
            PieceType::CommonKing => 1,
            PieceType::Queen => 2,
            PieceType::RoyalQueen => 3,
            PieceType::Princess => 4,
            PieceType::Rook => 5,
            PieceType::Bishop => 6,
            PieceType::Unicorn => 7,
            PieceType::Dragon => 8,
            PieceType::Knight => 9,
            PieceType::Pawn => 10,
            PieceType::Brawn => 11,
        }
}

fn default_projection_seed() -> u32 {
    DEFAULT_PROJECTION_SEED
}

fn project_neural_features(values: &[f32], projection_size: usize, seed: u32) -> Vec<f32> {
    let active: Vec<(usize, f32)> = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value != 0.0)
        .collect();
    if active.is_empty() || projection_size == 0 {
        return vec![0.0; projection_size];
    }
    let scale = (active.len() as f32).sqrt();
    let mut projected = vec![0.0; projection_size];
    for (raw_index, value) in active {
        for (projection_index, projected_value) in projected.iter_mut().enumerate() {
            let sign = if projection_hash(raw_index as u32, projection_index as u32, seed) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            *projected_value += value * sign / scale;
        }
    }
    projected
}

fn projection_hash(raw_index: u32, projection_index: u32, seed: u32) -> u32 {
    let mut hash = seed ^ raw_index;
    hash = hash.wrapping_mul(16_777_619);
    hash ^= projection_index;
    hash = hash.wrapping_mul(16_777_619);
    hash ^= hash >> 16;
    hash
}

fn evaluate_hidden_layers(
    input: &[f32],
    hidden_layers: &[usize],
    hidden_weights: &[f32],
) -> Option<Vec<f32>> {
    let mut cursor = 0;
    let mut values = input.to_vec();
    for &layer_size in hidden_layers {
        let required = values.len().checked_add(1)?.checked_mul(layer_size)?;
        if hidden_weights.len().saturating_sub(cursor) < required {
            return None;
        }
        let mut next = vec![0.0; layer_size];
        for (output, next_value) in next.iter_mut().enumerate().take(layer_size) {
            let row = cursor + output * (values.len() + 1);
            let mut sum = hidden_weights[row + values.len()];
            for (input_index, value) in values.iter().enumerate() {
                sum += value * hidden_weights[row + input_index];
            }
            *next_value = sum.max(0.0);
        }
        cursor += required;
        values = next;
    }
    Some(values)
}

fn parse_compact_neural_model(bytes: &[u8]) -> Option<NeuralLinearModel> {
    if bytes.len() < 32 || &bytes[0..4] != COMPACT_MODEL_MAGIC {
        return None;
    }
    let mut cursor = 4;
    let version = read_u32(bytes, &mut cursor)?;
    if version != 1 {
        return None;
    }
    let projection_size = read_u32(bytes, &mut cursor)? as usize;
    let projection_seed = read_u32(bytes, &mut cursor)?;
    let layer_count = read_u32(bytes, &mut cursor)? as usize;
    let output_size = read_u32(bytes, &mut cursor)? as usize;
    let scale = read_f32(bytes, &mut cursor)?;
    let bias = read_f32(bytes, &mut cursor)?;
    let mut hidden_layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        hidden_layers.push(read_u32(bytes, &mut cursor)? as usize);
    }
    let hidden_weight_count = read_u32(bytes, &mut cursor)? as usize;
    let mut hidden_weights = Vec::with_capacity(hidden_weight_count);
    for _ in 0..hidden_weight_count {
        hidden_weights.push(read_f32(bytes, &mut cursor)?);
    }
    let mut feature_weights = Vec::with_capacity(output_size);
    for _ in 0..output_size {
        feature_weights.push(read_f32(bytes, &mut cursor)?);
    }
    Some(NeuralLinearModel {
        bias,
        scale,
        feature_weights,
        projection_size,
        projection_seed,
        hidden_layers,
        hidden_weights,
    })
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Option<u32> {
    let end = cursor.checked_add(4)?;
    let value = u32::from_le_bytes(bytes.get(*cursor..end)?.try_into().ok()?);
    *cursor = end;
    Some(value)
}

fn read_f32(bytes: &[u8], cursor: &mut usize) -> Option<f32> {
    Some(f32::from_bits(read_u32(bytes, cursor)?))
}
