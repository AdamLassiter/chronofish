//! Perft over legal actions (one action = one ply).

use super::hypercuboid::HcInfo;
use super::state::State;

/// Leaf count at `depth`; `cap` (if non-zero) limits the actions expanded per node.
pub fn perft(s: &State, depth: u32, cap: usize) -> u64 {
    let (w, ss) = HcInfo::build_hc(s);
    let mut total = 0u64;
    for (n, mv) in w.search(s, ss).enumerate() {
        if cap != 0 && n >= cap {
            break;
        }
        if depth <= 1 {
            total += 1;
            continue;
        }
        let mut t = s.clone();
        for &m in &mv {
            t.apply_move::<true>(m);
        }
        t.submit::<true>();
        total += perft(&t, depth - 1, cap);
    }
    total
}
