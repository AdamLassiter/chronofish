use super::*;

impl Game {
    pub(crate) fn timeline_coordination(&self, color: Color, weights: &EvalWeights) -> i32 {
        let Some(present) = self.present_board() else {
            return 0;
        };
        let mut score = 0;
        for timeline in &self.timelines {
            if !self.is_active_timeline(timeline.id) {
                continue;
            }
            let Some(board) = timeline.boards.iter().max_by_key(|board| board.time) else {
                continue;
            };
            let side = if board.side_to_move == color { 1 } else { -1 };
            score += side * weights.frontier_tempo;
            if board.time == present.time {
                score += side * weights.present_anchor;
            }
        }
        score
    }
}
