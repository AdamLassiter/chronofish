impl AiSearchResult {
    fn to_json(&self) -> String {
        format!(
            "{{\"moves\":[{}],\"score\":{},\"depth\":{},\"nodes\":{},\"status\":\"{}\"}}",
            self.moves
                .iter()
                .map(move_step_json)
                .collect::<Vec<_>>()
                .join(","),
            self.score,
            self.depth,
            self.nodes,
            self.status
        )
    }
}
fn move_step_json(step: &MoveStep) -> String {
    format!(
        "{{\"from\":{},\"to\":{}}}",
        position_json(step.from),
        position_json(step.to)
    )
}
