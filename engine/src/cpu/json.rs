use super::*;
use crate::notation::position_json;

impl Game {
    pub(crate) fn evaluation_json(&self) -> String {
        let score = self.evaluate_heuristic(Color::White, &EvalWeights::active_tuned());
        format!("{{\"score\":{score},\"source\":\"engine heuristic\"}}")
    }
}

impl AiSearchResult {
    pub(crate) fn to_json(&self) -> String {
        format!(
            "{{\"moves\":[{}],\"score\":{},\"depth\":{},\"nodes\":{},\"status\":\"{}\",\"terminal\":{},\"resultReason\":{},\"principalVariation\":[{}]}}",
            self.moves
                .iter()
                .map(move_step_json)
                .collect::<Vec<_>>()
                .join(","),
            self.score,
            self.depth,
            self.nodes,
            self.status,
            self.terminal_royal_capture,
            if self.terminal_royal_capture {
                "\"royal-capture\""
            } else {
                "null"
            },
            self.principal_variation
                .iter()
                .map(|turn| format!(
                    "[{}]",
                    turn.iter()
                        .map(move_step_json)
                        .collect::<Vec<_>>()
                        .join(",")
                ))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}
pub(crate) fn move_step_json(step: &MoveStep) -> String {
    format!(
        "{{\"from\":{},\"to\":{}}}",
        position_json(step.from),
        position_json(step.to)
    )
}
