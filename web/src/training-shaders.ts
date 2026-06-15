import project_features from "./shaders/project_features.wgsl";
import forward_layer from "./shaders/forward_layer.wgsl";
import forward_indexed_layer from "./shaders/forward_indexed_layer.wgsl";
import forward_output from "./shaders/forward_output.wgsl";
import output_delta from "./shaders/output_delta.wgsl";
import hidden_delta from "./shaders/hidden_delta.wgsl";
import hidden3_delta from "./shaders/hidden3_delta.wgsl";
import apply_layer from "./shaders/apply_layer.wgsl";
import apply_indexed_layer from "./shaders/apply_indexed_layer.wgsl";
import apply_output from "./shaders/apply_output.wgsl";
import policy from "./shaders/policy.wgsl";
import frontier_neural from "./shaders/frontier_neural.wgsl";

export const PROJECT_FEATURES_SHADER = project_features;

export const FORWARD_LAYER_SHADER = forward_layer;

export const FORWARD_INDEXED_LAYER_SHADER = forward_indexed_layer;

export const FORWARD_OUTPUT_SHADER = forward_output;

export const OUTPUT_DELTA_SHADER = output_delta;

export const HIDDEN_DELTA_SHADER = hidden_delta;

export const HIDDEN3_DELTA_SHADER = hidden3_delta;

export const APPLY_LAYER_SHADER = apply_layer;

export const APPLY_INDEXED_LAYER_SHADER = apply_indexed_layer;

export const APPLY_OUTPUT_SHADER = apply_output;

export const POLICY_SHADER = policy;
export const FRONTIER_NEURAL_SHADER = frontier_neural;
