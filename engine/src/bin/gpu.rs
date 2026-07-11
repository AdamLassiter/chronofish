#[cfg(not(target_arch = "wasm32"))]
fn main() {
    chronofish_engine::gpu::cli::run_gpu_cli();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
