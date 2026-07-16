use super::*;
use crate::cpu::SearchInstant;

pub(crate) fn training_deadline(config: &CpuCliConfig) -> Option<SearchInstant> {
    config
        .max_seconds
        .map(|seconds| SearchInstant::now() + std::time::Duration::from_secs(seconds.max(1)))
}

pub(crate) fn training_expired(deadline: Option<SearchInstant>) -> bool {
    matches!(
        crate::training_runtime::cooperative_checkpoint(),
        crate::training_runtime::Checkpoint::Cancelled
    ) || deadline.is_some_and(|deadline| SearchInstant::now() >= deadline)
}

pub(crate) fn training_deadline_expired(deadline: Option<SearchInstant>) -> bool {
    deadline.is_some_and(|deadline| SearchInstant::now() >= deadline)
}

pub(crate) fn bounded_training_deadline(
    deadline: Option<SearchInstant>,
    budget: std::time::Duration,
) -> Option<SearchInstant> {
    let task_deadline = SearchInstant::now() + budget;
    match deadline {
        Some(global) => Some(global.min(task_deadline)),
        None => Some(task_deadline),
    }
}

pub(crate) fn remaining_seconds(deadline: Option<SearchInstant>) -> String {
    deadline.map_or_else(
        || "unlimited".to_string(),
        |deadline| {
            let now = SearchInstant::now();
            if now >= deadline {
                "0".to_string()
            } else {
                deadline.duration_since(now).as_secs().to_string()
            }
        },
    )
}
