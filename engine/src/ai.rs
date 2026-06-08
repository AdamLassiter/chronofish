include!("ai/types.rs");
include!("ai/effort.rs");
include!("ai/model.rs");

#[cfg(any(test, doctest))]
include!("ai/evaluator.rs");
#[cfg(any(test, doctest))]
#[path = "ai/evaluation/mod.rs"]
mod evaluation;
#[cfg(any(test, doctest))]
include!("ai/weights.rs");
#[cfg(any(test, doctest))]
include!("ai/search.rs");
#[cfg(any(test, doctest))]
include!("ai/search_plans.rs");
#[cfg(any(test, doctest))]
include!("ai/search_support.rs");
#[cfg(any(test, doctest))]
include!("ai/json.rs");
