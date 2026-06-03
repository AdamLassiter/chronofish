// The browser talks to the engine through a deliberately small C ABI. Game
// calculations are handled by WebGPU workers; this module only keeps version and
// output buffers for the remaining WASM boundary.
thread_local! {
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[no_mangle]
pub extern "C" fn chronofish_version() -> *const u8 {
    // Compile-time crate version, so the frontend reports the version of the
    // actual WASM artifact it loaded.
    set_output(env!("CARGO_PKG_VERSION").into())
}

#[no_mangle]
pub extern "C" fn chronofish_output_len() -> usize {
    OUTPUT.with(|output| output.borrow().len())
}

fn set_output(value: String) -> *const u8 {
    // Pointers returned by exports remain valid only until the next exported
    // string call rewrites OUTPUT.
    set_output_bytes(value.into_bytes())
}

fn set_output_bytes(value: Vec<u8>) -> *const u8 {
    // Pointers returned by exports remain valid only until the next exported
    // output call rewrites OUTPUT.
    OUTPUT.with(|output| {
        let mut output = output.borrow_mut();
        *output = value;
        output.as_ptr()
    })
}
