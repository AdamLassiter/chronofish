#[allow(dead_code)]
fn ai_effort_config(name: &str) -> Option<AiEffort> {
    let configs: std::collections::BTreeMap<String, AiEffort> =
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/cpu-v1/effort.json"
        )))
            .expect("committed AI effort configs should be valid JSON");
    configs.get(name).cloned()
}

#[allow(dead_code)]
fn default_ai_effort() -> AiEffort {
    ai_effort_config("expert").expect("expert AI effort config should exist")
}
