use super::*;

impl Game {
    pub(crate) fn timeline_economy_balance(&self, color: Color, weights: &EvalWeights) -> i32 {
        self.timeline_economy_for(color, weights)
            - self.timeline_economy_for(color.opposite(), weights)
    }

    pub(crate) fn timeline_economy_for(&self, color: Color, weights: &EvalWeights) -> i32 {
        let owner = match color {
            Color::White => TimelineOwner::White,
            Color::Black => TimelineOwner::Black,
        };
        let own_active = self
            .timelines
            .iter()
            .filter(|timeline| timeline.owner == owner && self.is_active_timeline(timeline.id))
            .count() as i32;
        let own_inactive = self
            .timelines
            .iter()
            .filter(|timeline| timeline.owner == owner && !self.is_active_timeline(timeline.id))
            .count() as i32;
        let active_material: i32 = self
            .latest_pieces()
            .into_iter()
            .filter(|(position, piece)| {
                piece.color == color && self.is_active_timeline(position.timeline_id)
            })
            .map(|(_, piece)| weights.piece_value(piece.piece_type) / 200)
            .sum();
        own_active * weights.timeline_economy + active_material
            - own_inactive * weights.timeline_economy * 2
    }
}
