#[cfg(not(target_arch = "wasm32"))]
fn main() {
    // Keep the binary as a thin entry point; the native-only training harness
    // owns parsing, logging, comparison, and optional promotion.
    chronofish_engine::run_training_cli();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
