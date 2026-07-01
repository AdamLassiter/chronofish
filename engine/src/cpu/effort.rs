use super::*;

#[allow(dead_code)]
pub(crate) fn ai_effort_config(name: &str) -> Option<AiEffort> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let json =
            std::fs::read_to_string(super::weights::cpu_model_dir().join("effort.json")).ok()?;
        let configs =
            serde_json::from_str::<std::collections::BTreeMap<String, AiEffort>>(&json).ok()?;
        configs.get(name).cloned()
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = name;
        None
    }
}

#[allow(dead_code)]
pub(crate) fn default_ai_effort() -> AiEffort {
    ai_effort_config("expert").expect("runtime expert AI effort config should exist")
}
