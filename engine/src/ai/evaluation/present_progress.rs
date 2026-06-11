use super::*;

impl Game {
    pub(crate) fn present_progress(&self, color: Color) -> i32 {
        let Some(present) = self.present_board() else {
            return 0;
        };
        let latest_sum: i32 = self
            .timelines
            .iter()
            .filter(|timeline| self.is_active_timeline(timeline.id))
            .filter_map(|timeline| self.latest_time(timeline.id))
            .sum();
        let factor = if present.side_to_move == color { 1 } else { -1 };
        factor * (latest_sum - present.time)
    }
}
