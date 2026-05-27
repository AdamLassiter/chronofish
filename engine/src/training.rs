// Native-only genetic training harness for EvalWeights. It plays short matches,
// compares candidate weights against the committed defaults, and can promote a
// statistically significant improvement by patching ai.rs and committing it.
#[derive(Clone)]
struct TrainerConfig {
    generations: usize,
    population: usize,
    depth: i32,
    nodes: usize,
    plies: usize,
    seed: u64,
    time_budget_secs: u64,
    out: Option<String>,
    score: Option<String>,
    score_default: bool,
    train_cycle: bool,
    compare_seeds: Vec<u64>,
    min_wins: usize,
    min_total_delta: i32,
    verify: String,
    ai_src: String,
}

#[derive(Clone)]
struct Lcg {
    // Deterministic tiny RNG: good enough for repeatable mutation/crossover and
    // keeps training independent of extra dependencies.
    state: u64,
}

pub fn run_training_cli() {
    let config = TrainerConfig::from_env(std::env::args().skip(1).collect());

    if config.train_cycle {
        // The top-level ./train script loops this mode until interrupted.
        run_training_cycle(&config);
        return;
    }

    if config.score_default {
        println!("{}", fitness(EvalWeights::default_tuned(), &config));
        return;
    }

    if let Some(path) = &config.score {
        let json = std::fs::read_to_string(path).expect("failed to read score weights");
        let weights = EvalWeights::from_json(&json).expect("failed to parse score weights");
        println!("{}", fitness(weights, &config));
        return;
    }

    let weights = train_weights(&config);
    let json = weights.to_json();
    if let Some(path) = &config.out {
        std::fs::write(path, &json).expect("failed to write training output");
    }
    println!("{json}");
}

impl TrainerConfig {
    fn from_env(args: Vec<String>) -> Self {
        // The script-facing CLI is intentionally small, so hand parsing keeps the
        // training harness dependency-free.
        let seed = random_seed();
        let mut config = Self {
            generations: usize::MAX,
            population: auto_population(),
            depth: 1,
            nodes: auto_nodes(),
            plies: 12,
            seed,
            time_budget_secs: 300,
            out: None,
            score: None,
            score_default: false,
            train_cycle: false,
            compare_seeds: default_compare_seeds(seed),
            min_wins: 0,
            min_total_delta: 0,
            verify: "cargo test -q".to_string(),
            ai_src: "engine/src/ai_parameters.rs".to_string(),
        };
        let mut index = 0;
        let mut compare_seeds_overridden = false;
        while index < args.len() {
            let value = args.get(index + 1).cloned();
            match args[index].as_str() {
                "--train-cycle" => {
                    config.train_cycle = true;
                    index += 1;
                }
                "--generations" => {
                    config.generations = parse_arg(value, config.generations);
                    index += 2;
                }
                "--population" => {
                    config.population = parse_arg(value, config.population);
                    index += 2;
                }
                "--depth" => {
                    config.depth = parse_arg(value, config.depth);
                    index += 2;
                }
                "--nodes" => {
                    config.nodes = parse_arg(value, config.nodes);
                    index += 2;
                }
                "--plies" => {
                    config.plies = parse_arg(value, config.plies);
                    index += 2;
                }
                "--seed" => {
                    config.seed = parse_arg(value, config.seed);
                    index += 2;
                }
                "--time-seconds" | "--time-budget" => {
                    config.time_budget_secs = parse_arg(value, config.time_budget_secs);
                    index += 2;
                }
                "--out" => {
                    config.out = value;
                    index += 2;
                }
                "--score" => {
                    config.score = value;
                    index += 2;
                }
                "--score-default" => {
                    config.score_default = true;
                    index += 1;
                }
                "--compare-seeds" => {
                    config.compare_seeds =
                        parse_seed_list(value.as_deref()).unwrap_or(config.compare_seeds);
                    compare_seeds_overridden = true;
                    index += 2;
                }
                "--min-wins" => {
                    config.min_wins = parse_arg(value, config.min_wins);
                    index += 2;
                }
                "--min-total-delta" => {
                    config.min_total_delta = parse_arg(value, config.min_total_delta);
                    index += 2;
                }
                "--verify" => {
                    config.verify = value.unwrap_or(config.verify);
                    index += 2;
                }
                "--ai-src" => {
                    config.ai_src = value.unwrap_or(config.ai_src);
                    index += 2;
                }
                _ => index += 1,
            }
        }
        config.population = config.population.max(4);
        config.time_budget_secs = config.time_budget_secs.max(1);
        if !compare_seeds_overridden {
            config.compare_seeds = default_compare_seeds(config.seed);
        }
        if config.min_wins == 0 {
            config.min_wins = config.compare_seeds.len() * 2 / 3 + 1;
        }
        if config.min_total_delta == 0 {
            config.min_total_delta = (config.compare_seeds.len() as i32) * 50;
        }
        config
    }

    fn with_seed(&self, seed: u64) -> Self {
        Self {
            generations: self.generations,
            population: self.population,
            depth: self.depth,
            nodes: self.nodes,
            plies: self.plies,
            seed,
            time_budget_secs: self.time_budget_secs,
            out: self.out.clone(),
            score: self.score.clone(),
            score_default: self.score_default,
            train_cycle: self.train_cycle,
            compare_seeds: self.compare_seeds.clone(),
            min_wins: self.min_wins,
            min_total_delta: self.min_total_delta,
            verify: self.verify.clone(),
            ai_src: self.ai_src.clone(),
        }
    }

    fn with_search(&self, depth: i32, nodes: usize, plies: usize) -> Self {
        let mut config = self.clone();
        config.depth = depth;
        config.nodes = nodes;
        config.plies = plies;
        config
    }
}

impl EvalWeights {
    fn mutate(self, rng: &mut Lcg) -> Self {
        // Keep royal values fixed so training cannot discover incentives that
        // trade away the king-shaped objective for short-term material.
        Self {
            king: self.king,
            common_king: mutate_weight(self.common_king, rng, 80, 0, 2_000),
            queen: mutate_weight(self.queen, rng, 160, 100, 3_000),
            royal_queen: self.royal_queen,
            princess: mutate_weight(self.princess, rng, 140, 100, 3_000),
            rook: mutate_weight(self.rook, rng, 100, 100, 2_000),
            bishop: mutate_weight(self.bishop, rng, 80, 50, 2_000),
            unicorn: mutate_weight(self.unicorn, rng, 90, 50, 2_000),
            dragon: mutate_weight(self.dragon, rng, 120, 50, 2_000),
            knight: mutate_weight(self.knight, rng, 80, 50, 2_000),
            pawn: mutate_weight(self.pawn, rng, 30, 10, 600),
            brawn: mutate_weight(self.brawn, rng, 30, 10, 600),
            check_penalty: mutate_weight(self.check_penalty, rng, 120, 0, 3_000),
            active_timeline: mutate_weight(self.active_timeline, rng, 20, -500, 500),
            inactive_timeline: mutate_weight(self.inactive_timeline, rng, 20, -500, 500),
            present_progress: mutate_weight(self.present_progress, rng, 10, 0, 200),
            mobility: mutate_weight(self.mobility, rng, 4, 0, 80),
            branch_penalty: mutate_weight(self.branch_penalty, rng, 10, 0, 300),
            advancement: mutate_weight(self.advancement, rng, 4, 0, 80),
            centrality: mutate_weight(self.centrality, rng, 4, 0, 80),
        }
    }

    fn crossover(left: Self, right: Self, rng: &mut Lcg) -> Self {
        // Uniform crossover lets each parameter independently come from either
        // parent, which fits this compact, flat genome.
        macro_rules! pick {
            ($field:ident) => {
                if rng.next_bool() {
                    left.$field
                } else {
                    right.$field
                }
            };
        }
        Self {
            king: left.king,
            common_king: pick!(common_king),
            queen: pick!(queen),
            royal_queen: left.royal_queen,
            princess: pick!(princess),
            rook: pick!(rook),
            bishop: pick!(bishop),
            unicorn: pick!(unicorn),
            dragon: pick!(dragon),
            knight: pick!(knight),
            pawn: pick!(pawn),
            brawn: pick!(brawn),
            check_penalty: pick!(check_penalty),
            active_timeline: pick!(active_timeline),
            inactive_timeline: pick!(inactive_timeline),
            present_progress: pick!(present_progress),
            mobility: pick!(mobility),
            branch_penalty: pick!(branch_penalty),
            advancement: pick!(advancement),
            centrality: pick!(centrality),
        }
    }

    fn to_json(self) -> String {
        format!(
            "{{\"king\":{},\"commonKing\":{},\"queen\":{},\"royalQueen\":{},\"princess\":{},\"rook\":{},\"bishop\":{},\"unicorn\":{},\"dragon\":{},\"knight\":{},\"pawn\":{},\"brawn\":{},\"checkPenalty\":{},\"activeTimeline\":{},\"inactiveTimeline\":{},\"presentProgress\":{},\"mobility\":{},\"branchPenalty\":{},\"advancement\":{},\"centrality\":{}}}",
            self.king,
            self.common_king,
            self.queen,
            self.royal_queen,
            self.princess,
            self.rook,
            self.bishop,
            self.unicorn,
            self.dragon,
            self.knight,
            self.pawn,
            self.brawn,
            self.check_penalty,
            self.active_timeline,
            self.inactive_timeline,
            self.present_progress,
            self.mobility,
            self.branch_penalty,
            self.advancement,
            self.centrality
        )
    }

    fn from_json(value: &str) -> Result<Self, String> {
        Ok(Self {
            king: json_i32(value, "king")?,
            common_king: json_i32(value, "commonKing")?,
            queen: json_i32(value, "queen")?,
            royal_queen: json_i32(value, "royalQueen")?,
            princess: json_i32(value, "princess")?,
            rook: json_i32(value, "rook")?,
            bishop: json_i32(value, "bishop")?,
            unicorn: json_i32(value, "unicorn")?,
            dragon: json_i32(value, "dragon")?,
            knight: json_i32(value, "knight")?,
            pawn: json_i32(value, "pawn")?,
            brawn: json_i32(value, "brawn")?,
            check_penalty: json_i32(value, "checkPenalty")?,
            active_timeline: json_i32(value, "activeTimeline")?,
            inactive_timeline: json_i32(value, "inactiveTimeline")?,
            present_progress: json_i32(value, "presentProgress")?,
            mobility: json_i32(value, "mobility")?,
            branch_penalty: json_i32(value, "branchPenalty")?,
            advancement: json_i32(value, "advancement")?,
            centrality: json_i32(value, "centrality")?,
        })
    }

    fn to_rust_parameters(self) -> String {
        format!(
            "Self {{\n    king: {},\n    common_king: {},\n    queen: {},\n    royal_queen: {},\n    princess: {},\n    rook: {},\n    bishop: {},\n    unicorn: {},\n    dragon: {},\n    knight: {},\n    pawn: {},\n    brawn: {},\n    check_penalty: {},\n    active_timeline: {},\n    inactive_timeline: {},\n    present_progress: {},\n    mobility: {},\n    branch_penalty: {},\n    advancement: {},\n    centrality: {},\n}}\n",
            self.king,
            self.common_king,
            self.queen,
            self.royal_queen,
            self.princess,
            self.rook,
            self.bishop,
            self.unicorn,
            self.dragon,
            self.knight,
            self.pawn,
            self.brawn,
            self.check_penalty,
            self.active_timeline,
            self.inactive_timeline,
            self.present_progress,
            self.mobility,
            self.branch_penalty,
            self.advancement,
            self.centrality
        )
    }
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper.max(1)
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

fn run_training_cycle(config: &TrainerConfig) {
    // Promotion rewrites ai.rs, so refuse to continue when that file already has
    // local edits that should not be mixed with generated tuning changes.
    if ai_source_is_dirty(&config.ai_src) {
        eprintln!(
            "{} has uncommitted changes; commit or stash before running training",
            config.ai_src
        );
        std::process::exit(1);
    }

    println!(
        "training budget={}s population={} base_depth={} base_nodes={} plies={} seed={}",
        config.time_budget_secs, config.population, config.depth, config.nodes, config.plies, config.seed
    );
    println!(
        "score note: fitness points are aggregate evaluation margins from short matches; comparison wins/losses are decided by candidate fitness minus baseline fitness per seed"
    );

    let candidate = train_weights(config);
    let candidate_json = candidate.to_json();
    let mut wins = 0;
    let mut losses = 0;
    let mut draws = 0;
    let mut total_delta = 0;

    println!("candidate weights: {candidate_json}");
    for seed in &config.compare_seeds {
        let baseline_config = config.with_seed(*seed);
        let baseline_score = fitness(EvalWeights::default_tuned(), &baseline_config);
        let candidate_score = fitness(candidate, &baseline_config);
        let delta = candidate_score - baseline_score;
        total_delta += delta;
        let result = if delta > 0 {
            wins += 1;
            "win"
        } else if delta < 0 {
            losses += 1;
            "loss"
        } else {
            draws += 1;
            "draw"
        };
        println!(
            "seed {seed}: {result} candidate={candidate_score} baseline={baseline_score} delta={delta}"
        );
    }

    print_threshold_progress(wins, losses, draws, total_delta, config);
    if wins >= config.min_wins && total_delta >= config.min_total_delta {
        promote_weights(candidate, &config.ai_src);
        run_command("cargo", &["fmt"]);
        run_shell(&config.verify);
        run_command("git", &["add", &config.ai_src]);
        run_command("git", &["commit", "-m", "Tune AI evaluation parameters"]);
        println!("promoted candidate and committed updated parameters");
    } else {
        println!("candidate rejected");
    }
}

fn train_weights(config: &TrainerConfig) -> EvalWeights {
    println!(
        "fitness score = material/tempo/check/present-line heuristic accumulated over {} plies against default and mutated opponents",
        config.plies
    );
    let mut rng = Lcg::new(config.seed);
    let mut population = vec![EvalWeights::default_tuned()];
    while population.len() < config.population {
        population.push(EvalWeights::default_tuned().mutate(&mut rng));
    }

    let mut previous_best: Option<i32> = None;
    let training_started = std::time::Instant::now();
    let deadline = training_started + std::time::Duration::from_secs(config.time_budget_secs);
    for generation in 0..config.generations {
        if std::time::Instant::now() >= deadline {
            break;
        }
        // Each generation is fully rescored against this config seed so best,
        // average, and improvement logs are comparable within the run.
        let started = std::time::Instant::now();
        let elapsed = training_started.elapsed().as_secs();
        let third = (config.time_budget_secs / 3).max(1);
        let depth_boost = (elapsed / third).min(2) as i32;
        let search_depth = config.depth + depth_boost;
        let search_nodes = config.nodes * search_depth as usize;
        let search_plies = config.plies + depth_boost as usize * 2;
        let scoring_config = config.with_search(search_depth, search_nodes, search_plies);
        let mut scored: Vec<(i32, EvalWeights)> = population
            .iter()
            .copied()
            .map(|weights| (fitness(weights, &scoring_config), weights))
            .collect();
        scored.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        let best = scored[0].0;
        let worst = scored.last().map_or(best, |entry| entry.0);
        let average =
            scored.iter().map(|entry| entry.0 as i64).sum::<i64>() as f64 / scored.len() as f64;
        let improvement = previous_best.map_or(0, |previous| best - previous);
        previous_best = Some(best);
        let remaining = deadline
            .saturating_duration_since(std::time::Instant::now())
            .as_secs();
        eprintln!(
            "generation {generation}: depth={search_depth} nodes={search_nodes} plies={search_plies} best={best} avg={average:.1} worst={worst} improvement={improvement:+} population={} gen_elapsed={:.2}s remaining={}s",
            scored.len(),
            started.elapsed().as_secs_f32(),
            remaining
        );

        let elite = 4.min(scored.len());
        let mut next: Vec<EvalWeights> = scored.iter().take(elite).map(|entry| entry.1).collect();
        while next.len() < config.population {
            let left = tournament(&scored, &mut rng);
            let right = tournament(&scored, &mut rng);
            next.push(EvalWeights::crossover(left, right, &mut rng).mutate(&mut rng));
        }
        population = next;
    }

    population
        .into_iter()
        .max_by_key(|weights| fitness(*weights, config))
        .unwrap_or_else(EvalWeights::default_tuned)
}

fn fitness(weights: EvalWeights, config: &TrainerConfig) -> i32 {
    // Score candidates against the committed default and a few nearby mutated
    // opponents so the search improves the current engine instead of overfitting
    // to one self-play lineage.
    let mut rng = Lcg::new(config.seed);
    let mut total = 0;
    let default = EvalWeights::default_tuned();
    total += play_match(weights, default, Color::White, config);
    total += play_match(weights, default, Color::Black, config);

    for _ in 0..3 {
        let opponent = default.mutate(&mut rng);
        total += play_match(weights, opponent, Color::White, config);
        total += play_match(weights, opponent, Color::Black, config);
    }

    total
}

fn play_match(weights: EvalWeights, opponent: EvalWeights, color: Color, config: &TrainerConfig) -> i32 {
    // Matches are short by design; the heuristic should learn opening material,
    // tempo, and branch quality before deeper minimax is affordable.
    let mut game = Game::new();
    let mut score = 0;
    for ply in 0..config.plies {
        let side_weights = if game.turn == color { weights } else { opponent };
        let mut context = SearchContext {
            weights: side_weights,
            root_color: game.turn,
            max_nodes: config.nodes,
            nodes: 0,
        };
        let Some((plan, _)) = game.search_root(config.depth, &mut context) else {
            score += if game.turn == color { -10_000 } else { 10_000 };
            break;
        };
        game = plan.game;
        let eval = game.evaluate(color, &weights);
        score += eval / 20 + eval.signum() * (config.plies - ply) as i32;
        if game.is_checkmate(color) || game.is_checkmate(color.opposite()) {
            score += if game.is_checkmate(color) {
                -CHECKMATE_SCORE / 10
            } else {
                CHECKMATE_SCORE / 10
            };
            break;
        }
    }
    score + game.evaluate(color, &weights) / 4
}

fn tournament(scored: &[(i32, EvalWeights)], rng: &mut Lcg) -> EvalWeights {
    // Tournament selection applies pressure toward stronger candidates while
    // preserving enough randomness for weaker genomes to contribute genes.
    let mut best = scored[rng.next_usize(scored.len())];
    for _ in 1..4 {
        let candidate = scored[rng.next_usize(scored.len())];
        if candidate.0 > best.0 {
            best = candidate;
        }
    }
    best.1
}

fn mutate_weight(value: i32, rng: &mut Lcg, spread: i32, min: i32, max: i32) -> i32 {
    let delta = rng.next_usize((spread * 2 + 1) as usize) as i32 - spread;
    (value + delta).clamp(min, max)
}

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

fn json_i32(value: &str, key: &str) -> Result<i32, String> {
    let needle = format!("\"{key}\":");
    let Some(start) = value.find(&needle).map(|index| index + needle.len()) else {
        return Err(format!("missing key {key}"));
    };
    let tail = &value[start..];
    let end = tail
        .find(|character: char| !character.is_ascii_digit() && character != '-')
        .unwrap_or(tail.len());
    tail[..end]
        .parse()
        .map_err(|_| format!("invalid integer for {key}"))
}

fn print_threshold_progress(
    wins: usize,
    losses: usize,
    draws: usize,
    total_delta: i32,
    config: &TrainerConfig,
) {
    let wins_needed = config.min_wins.saturating_sub(wins);
    let delta_needed = config.min_total_delta.saturating_sub(total_delta);
    println!(
        "comparison: wins={wins} losses={losses} draws={draws} required_wins={} total_delta={total_delta}/{}",
        config.min_wins, config.min_total_delta
    );
    println!("threshold remaining: wins={wins_needed} total_delta={delta_needed}");
}

fn auto_population() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get() * 4)
        .unwrap_or(16)
        .clamp(8, 64)
}

fn auto_nodes() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get() * 500)
        .unwrap_or(4_000)
        .clamp(1_000, 12_000)
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
    // Runtime weights live in a small include file. Overwriting the whole file is
    // less clever than field patching and avoids ever touching EvalWeights types.
    std::fs::write(ai_src, weights.to_rust_parameters()).expect("failed to write AI parameters");
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
