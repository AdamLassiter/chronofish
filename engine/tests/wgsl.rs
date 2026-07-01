use std::collections::HashMap;

use chronofish_engine::gpu::{search, training, GpuKernel, WgslShader};

#[test]
fn engine_gpu_wgsl_shaders_parse() {
    for shader in search::SHADERS.iter().chain(training::SHADERS.iter()) {
        validate_shader(shader);
    }
}

#[test]
fn engine_gpu_kernel_descriptors_match_wgsl_entrypoints() {
    let shaders = search::SHADERS
        .iter()
        .chain(training::SHADERS.iter())
        .map(|shader| (shader.name, *shader))
        .collect::<HashMap<_, _>>();
    for kernel in search::KERNELS.iter().chain(training::KERNELS.iter()) {
        validate_kernel(kernel, &shaders);
    }
}

fn validate_shader(shader: &WgslShader) {
    let module = naga::front::wgsl::parse_str(shader.source)
        .unwrap_or_else(|error| panic!("{} failed to parse: {error}", shader.name));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{} failed validation: {error:?}", shader.name));
}

fn validate_kernel(kernel: &GpuKernel, shaders: &HashMap<&'static str, WgslShader>) {
    let shader = shaders.get(kernel.shader).unwrap_or_else(|| {
        panic!(
            "{} references missing shader {}",
            kernel.label, kernel.shader
        )
    });
    let module = naga::front::wgsl::parse_str(shader.source)
        .unwrap_or_else(|error| panic!("{} failed to parse: {error}", shader.name));
    assert!(
        module
            .entry_points
            .iter()
            .any(|entry| entry.name == kernel.entry_point),
        "{} references missing entry point {} in {}",
        kernel.label,
        kernel.entry_point,
        kernel.shader
    );
}
