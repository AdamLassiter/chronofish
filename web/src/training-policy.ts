import type { Move } from "./types.js";

export const POLICY_BUCKETS = 257;

export function policyBucket(move: Move | null | undefined): number | null {
  if (!move) {
    return null;
  }
  const values = [
    (move.to?.timelineId ?? 0) - (move.from?.timelineId ?? 0),
    (move.to?.time ?? 0) - (move.from?.time ?? 0),
    (move.to?.x ?? 0) - (move.from?.x ?? 0),
    (move.to?.y ?? 0) - (move.from?.y ?? 0),
    move.from?.x ?? 0,
    move.from?.y ?? 0
  ];
  let hash = 2166136261;
  for (const value of values) {
    const bits = value >>> 0;
    for (let shift = 0; shift < 32; shift += 8) {
      hash ^= (bits >>> shift) & 0xff;
      hash = Math.imul(hash, 16777619) >>> 0;
    }
  }
  return hash % POLICY_BUCKETS;
}
