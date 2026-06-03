const DEFAULT_PROJECTION_SEED: u32 = 2_166_136_261;

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

fn default_projection_seed() -> u32 {
    DEFAULT_PROJECTION_SEED
}
