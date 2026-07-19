import { loadShader } from "./shader-loader.js";

export interface TrainingShaders {
  projectFeatures: string;
  forwardLayer: string;
  forwardIndexedLayer: string;
  forwardOutput: string;
  outputDelta: string;
  hiddenDelta: string;
  hidden3Delta: string;
  applyLayer: string;
  applyIndexedLayer: string;
  applyOutput: string;
  policy: string;
  policyLoss: string;
  reduceLoss: string;
  frontierPolicy: string;
  frontierNeural: string;
}

let shadersPromise: Promise<TrainingShaders> | undefined;

export function loadTrainingShaders(): Promise<TrainingShaders> {
  shadersPromise ??= Promise.all([
    loadShader("/shaders/training/project_features.wgsl"),
    loadShader("/shaders/training/forward_layer.wgsl"),
    loadShader("/shaders/training/forward_indexed_layer.wgsl"),
    loadShader("/shaders/training/forward_output.wgsl"),
    loadShader("/shaders/training/output_delta.wgsl"),
    loadShader("/shaders/training/hidden_delta.wgsl"),
    loadShader("/shaders/training/hidden3_delta.wgsl"),
    loadShader("/shaders/training/apply_layer.wgsl"),
    loadShader("/shaders/training/apply_indexed_layer.wgsl"),
    loadShader("/shaders/training/apply_output.wgsl"),
    loadShader("/shaders/training/policy.wgsl"),
    loadShader("/shaders/training/policy_loss.wgsl"),
    loadShader("/shaders/training/reduce_loss.wgsl"),
    loadShader("/shaders/search/frontier_policy.wgsl"),
    loadShader("/shaders/search/frontier_neural.wgsl")
  ]).then(([
    projectFeatures,
    forwardLayer,
    forwardIndexedLayer,
    forwardOutput,
    outputDelta,
    hiddenDelta,
    hidden3Delta,
    applyLayer,
    applyIndexedLayer,
    applyOutput,
    policy,
    policyLoss,
    reduceLoss,
    frontierPolicy,
    frontierNeural
  ]) => ({
    projectFeatures,
    forwardLayer,
    forwardIndexedLayer,
    forwardOutput,
    outputDelta,
    hiddenDelta,
    hidden3Delta,
    applyLayer,
    applyIndexedLayer,
    applyOutput,
    policy,
    policyLoss,
    reduceLoss,
    frontierPolicy,
    frontierNeural
  }));
  return shadersPromise;
}
