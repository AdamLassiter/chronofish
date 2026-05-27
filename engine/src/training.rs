struct TrainerConfig {
    generations: usize,
    population: usize,
    depth: i32,
    nodes: usize,
    plies: usize,
    seed: u64,
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
    state: u64,
}

pub fn run_training_cli() {
    let config = TrainerConfig::from_env(std::env::args().skip(1).collect());

    if config.train_cycle {
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
        let mut config = Self {
            generations: 50,
            population: 32,
            depth: 1,
            nodes: 5_000,
            plies: 12,
            seed: 1,
            out: None,
            score: None,
            score_default: false,
            train_cycle: false,
            compare_seeds: vec![101, 202, 303, 404, 505],
            min_wins: 4,
            min_total_delta: 250,
            verify: "cargo test -q".to_string(),
            ai_src: "engine/src/ai.rs".to_string(),
        };
        let mut index = 0;
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
}

impl EvalWeights {
    fn mutate(self, rng: &mut Lcg) -> Self {
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

    fn fields(self) -> [(&'static str, i32); 20] {
        [
            ("king", self.king),
            ("common_king", self.common_king),
            ("queen", self.queen),
            ("royal_queen", self.royal_queen),
            ("princess", self.princess),
            ("rook", self.rook),
            ("bishop", self.bishop),
            ("unicorn", self.unicorn),
            ("dragon", self.dragon),
            ("knight", self.knight),
            ("pawn", self.pawn),
            ("brawn", self.brawn),
            ("check_penalty", self.check_penalty),
            ("active_timeline", self.active_timeline),
            ("inactive_timeline", self.inactive_timeline),
            ("present_progress", self.present_progress),
            ("mobility", self.mobility),
            ("branch_penalty", self.branch_penalty),
            ("advancement", self.advancement),
            ("centrality", self.centrality),
        ]
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
    if ai_source_is_dirty(&config.ai_src) {
        eprintln!(
            "{} has uncommitted changes; commit or stash before running training",
            config.ai_src
        );
        std::process::exit(1);
    }

    let candidate = train_weights(config);
    let candidate_json = candidate.to_json();
    let mut wins = 0;
    let mut total_delta = 0;

    println!("candidate weights: {candidate_json}");
    for seed in &config.compare_seeds {
        let baseline_config = config.with_seed(*seed);
        let baseline_score = fitness(EvalWeights::default_tuned(), &baseline_config);
        let candidate_score = fitness(candidate, &baseline_config);
        let delta = candidate_score - baseline_score;
        total_delta += delta;
        if delta > 0 {
            wins += 1;
        }
        println!("seed {seed}: candidate={candidate_score} baseline={baseline_score} delta={delta}");
    }

    print_threshold_progress(wins, total_delta, config);
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
    let mut rng = Lcg::new(config.seed);
    let mut population = vec![EvalWeights::default_tuned()];
    while population.len() < config.population {
        population.push(EvalWeights::default_tuned().mutate(&mut rng));
    }

    let mut previous_best: Option<i32> = None;
    for generation in 0..config.generations {
        let started = std::time::Instant::now();
        let mut scored: Vec<(i32, EvalWeights)> = population
            .iter()
            .copied()
            .map(|weights| (fitness(weights, config), weights))
            .collect();
        scored.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        let best = scored[0].0;
        let worst = scored.last().map_or(best, |entry| entry.0);
        let average =
            scored.iter().map(|entry| entry.0 as i64).sum::<i64>() as f64 / scored.len() as f64;
        let improvement = previous_best.map_or(0, |previous| best - previous);
        previous_best = Some(best);
        eprintln!(
            "generation {generation}: best={best} avg={average:.1} worst={worst} improvement={improvement:+} population={} elapsed={:.2}s",
            scored.len(),
            started.elapsed().as_secs_f32()
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

fn print_threshold_progress(wins: usize, total_delta: i32, config: &TrainerConfig) {
    let wins_needed = config.min_wins.saturating_sub(wins);
    let delta_needed = config.min_total_delta.saturating_sub(total_delta);
    println!(
        "comparison: wins={wins}/{} total_delta={total_delta}/{}",
        config.min_wins, config.min_total_delta
    );
    println!("threshold remaining: wins={wins_needed} total_delta={delta_needed}");
}

fn promote_weights(weights: EvalWeights, ai_src: &str) {
    let mut source = std::fs::read_to_string(ai_src).expect("failed to read AI source");
    for (rust_name, value) in weights.fields() {
        source = replace_weight_field(&source, rust_name, value);
    }
    std::fs::write(ai_src, source).expect("failed to write AI source");
}

fn replace_weight_field(source: &str, field: &str, value: i32) -> String {
    let needle = format!("{field}:");
    let Some(start) = source.find(&needle).map(|index| index + needle.len()) else {
        panic!("missing weight field {field}");
    };
    let prefix = &source[..start];
    let tail = &source[start..];
    let digits_start = tail
        .find(|character: char| character == '-' || character.is_ascii_digit())
        .expect("missing weight value");
    let digits_end = digits_start
        + tail[digits_start..]
            .find(|character: char| !character.is_ascii_digit() && character != '-')
            .unwrap_or(tail.len() - digits_start);
    format!("{}{}{}", prefix, &tail[..digits_start], value) + &tail[digits_end..]
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
