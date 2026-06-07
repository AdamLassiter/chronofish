import turn_status from "./shaders/turn_status.wgsl";
import movegen from "./shaders/movegen.wgsl";
import reply from "./shaders/reply.wgsl";
import mutate from "./shaders/mutate.wgsl";

export const GPU_TURN_STATUS_SHADER = turn_status;

export const GPU_MOVEGEN_SHADER = movegen;

export const GPU_REPLY_SHADER = reply;

export const GPU_MUTATE_SHADER = mutate;
