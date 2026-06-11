pub(crate) const DEFAULT_PROJECTION_SEED: u32 = 2_166_136_261;

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub(crate) struct NeuralLinearModel {
    pub(crate) bias: f32,
    pub(crate) scale: f32,
    #[serde(default)]
    pub(crate) feature_weights: Vec<f32>,
    #[serde(default)]
    pub(crate) projection_size: usize,
    #[serde(default = "default_projection_seed")]
    pub(crate) projection_seed: u32,
    #[serde(default)]
    pub(crate) hidden_layers: Vec<usize>,
    #[serde(default)]
    pub(crate) hidden_weights: Vec<f32>,
}

pub(crate) fn default_projection_seed() -> u32 {
    DEFAULT_PROJECTION_SEED
}
