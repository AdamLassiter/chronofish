import turn_status from "../../engine/src/gpu/search/shaders/turn_status.wgsl";
import frontier_select from "../../engine/src/gpu/search/shaders/frontier_select.wgsl";
import frontier_state from "../../engine/src/gpu/search/shaders/frontier_state.wgsl";
import frontier_expand from "../../engine/src/gpu/search/shaders/frontier_expand.wgsl";
import movegen from "../../engine/src/gpu/search/shaders/movegen.wgsl";
import reply from "../../engine/src/gpu/search/shaders/reply.wgsl";
import mutate from "../../engine/src/gpu/search/shaders/mutate.wgsl";

export const GPU_TURN_STATUS_SHADER = turn_status;
export const GPU_FRONTIER_SELECT_SHADER = frontier_select;
export const GPU_FRONTIER_STATE_SHADER = frontier_state;
export const GPU_FRONTIER_EXPAND_SHADER = frontier_expand;

export const GPU_MOVEGEN_SHADER = movegen;

export const GPU_REPLY_SHADER = reply;

export const GPU_MUTATE_SHADER = mutate;
