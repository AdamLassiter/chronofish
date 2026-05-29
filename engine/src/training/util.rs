fn parse_arg<T: std::str::FromStr>(value: Option<String>, fallback: T) -> T {
    value.and_then(|raw| raw.parse().ok()).unwrap_or(fallback)
}

fn parse_seed_list(value: Option<&str>) -> Option<Vec<u64>> {
    let seeds: Vec<u64> = value?
        .split([',', ' '])
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse().ok())
        .collect();
    (!seeds.is_empty()).then_some(seeds)
}

fn auto_population() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(8)
        .clamp(4, 8)
}

fn auto_nodes() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get() * 20)
        .unwrap_or(160)
        .clamp(80, 240)
}

fn random_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(1)
}

fn default_compare_seeds(seed: u64) -> Vec<u64> {
    let mut rng = Lcg::new(seed ^ 0x9e37_79b9_7f4a_7c15);
    (0..9).map(|_| rng.next_u64()).collect()
}

fn promote_weights(weights: EvalWeights, ai_src: &str) {
    // Runtime weights live in a small JSON include file. Overwriting the whole
    // file is less clever than field patching and avoids ever touching types.
    let json = serde_json::to_string_pretty(&weights).expect("EvalWeights should serialize");
    std::fs::write(ai_src, format!("{json}\n")).expect("failed to write AI parameters");
}

fn load_hall_of_fame(path: &str) -> Vec<EvalWeights> {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    raw.lines()
        .rev()
        .filter_map(|line| EvalWeights::from_json(line).ok())
        .take(4)
        .collect()
}

fn append_hall_of_fame(path: &str, weights: EvalWeights) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).expect("failed to create hall-of-fame directory");
    }
    let mut line = weights.to_json();
    line.push('\n');
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    use std::io::Write;
    options
        .open(path)
        .and_then(|mut file| file.write_all(line.as_bytes()))
        .expect("failed to append hall-of-fame weights");
}

fn ai_source_is_dirty(ai_src: &str) -> bool {
    !std::process::Command::new("git")
        .args(["diff", "--quiet", "--", ai_src])
        .status()
        .is_ok_and(|status| status.success())
        || !std::process::Command::new("git")
            .args(["diff", "--cached", "--quiet", "--", ai_src])
            .status()
            .is_ok_and(|status| status.success())
}

fn run_command(command: &str, args: &[&str]) {
    let status = std::process::Command::new(command)
        .args(args)
        .status()
        .unwrap_or_else(|error| panic!("failed to run {command}: {error}"));
    if !status.success() {
        panic!("{command} failed with status {status}");
    }
}

fn run_shell(command: &str) {
    let status = std::process::Command::new("sh")
        .args(["-c", command])
        .status()
        .unwrap_or_else(|error| panic!("failed to run verification command: {error}"));
    if !status.success() {
        panic!("verification failed with status {status}");
    }
}
