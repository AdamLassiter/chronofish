use super::*;
use crate::cpu::SearchInstant;

pub(crate) fn training_deadline(config: &CpuCliConfig) -> Option<SearchInstant> {
    config
        .max_seconds
        .map(|seconds| SearchInstant::now() + std::time::Duration::from_secs(seconds.max(1)))
}

pub(crate) fn training_expired(deadline: Option<SearchInstant>) -> bool {
    deadline.is_some_and(|deadline| SearchInstant::now() >= deadline)
}

pub(crate) fn remaining_seconds(deadline: Option<SearchInstant>) -> String {
    deadline.map_or_else(
        || "unbounded".to_string(),
        |deadline| {
            let seconds = deadline
                .saturating_duration_since(SearchInstant::now())
                .as_secs();
            format!("{seconds}s")
        },
    )
}
