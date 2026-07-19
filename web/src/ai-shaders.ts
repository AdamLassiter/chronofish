import { loadShader } from "./shader-loader.js";

export interface AiShaders {
  turnStatus: string;
  frontierSelect: string;
  frontierState: string;
  frontierExpand: string;
  movegen: string;
  reply: string;
  mutate: string;
}

let shadersPromise: Promise<AiShaders> | undefined;

export function loadAiShaders(): Promise<AiShaders> {
  shadersPromise ??= Promise.all([
    loadShader("/shaders/search/turn_status.wgsl"),
    loadShader("/shaders/search/frontier_select.wgsl"),
    loadShader("/shaders/search/frontier_state.wgsl"),
    loadShader("/shaders/search/frontier_expand.wgsl"),
    loadShader("/shaders/search/movegen.wgsl"),
    loadShader("/shaders/search/reply.wgsl"),
    loadShader("/shaders/search/mutate.wgsl")
  ]).then(([turnStatus, frontierSelect, frontierState, frontierExpand, movegen, reply, mutate]) => ({
    turnStatus,
    frontierSelect,
    frontierState,
    frontierExpand,
    movegen,
    reply,
    mutate
  }));
  return shadersPromise;
}
