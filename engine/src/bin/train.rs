#[cfg(not(target_arch = "wasm32"))]
fn main() {
    chronofish_engine::run_training_cli();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
