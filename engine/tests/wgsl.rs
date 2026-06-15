use std::{fs, path::PathBuf};

#[test]
fn frontend_wgsl_shaders_parse() {
    let shader_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../web/src/shaders");
    let entries = fs::read_dir(&shader_root).expect("read shader directory");
    for entry in entries {
        let path = entry.expect("read shader entry").path();
        if path.extension().and_then(|value| value.to_str()) != Some("wgsl") {
            continue;
        }
        let source = fs::read_to_string(&path).expect("read WGSL shader");
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{} failed to parse: {error}", path.display()));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{} failed validation: {error:?}", path.display()));
    }
}
