# Chronofish Benchmarks

Provides a release-mode throughput check, bot tournaments, and paired model matches.
Run commands from the workspace root.

```sh
cargo run --release -p chronofish-benchmarks -- --check
```

## Tournaments

Compare bots with a round-robin tournament. Every pair plays each random opening from both sides. The tournament prints standings with W/D/L, Elo and time per action, followed by a crosstable:

```sh
cargo run --release -p chronofish-benchmarks -- --tournament \
  --openings 2 --ruleset all --max-actions 80 --model models/training/best-v2.json
```

Pairings against Kairos or Loki use reference rules. Use `--bots kairos,loki --ruleset multiverse` for a head-to-head match. Use `--bots` to choose a subset and `--csv` to save one row per game. `blueshift-<nodes>` (for example `blueshift-10k`) adds Deep Blueshift with a different node budget.

The Elo column has a ± value: half the 95% bootstrap interval, from resampling colour-reversed opening pairs (`--bootstrap N`, default 200). Treat two bots whose intervals overlap as unresolved, and add openings to separate them. When a game reaches the action limit, the handcrafted evaluation scores it. If one side leads by `--adjudicate-cp` centipawns (default 400), that side wins; otherwise the game is a draw. `--adjudicate-cp 0` scores every unfinished game as a draw.

### Ratings on a recognised scale

Without anchors, ratings are centred on zero and shift whenever the roster changes. Stockfish at calibrated strengths gives fixed points on the standard-chess scale. Install it (`sudo apt install stockfish`), then pass `--stockfish /usr/games/stockfish` or set `$STOCKFISH`:

- `sf-<elo>` is Stockfish limited to that Elo, from 1320 to 3190. These levels are anchored automatically.
- `sf-skill<0-20>` uses Stockfish's Skill Level, which reaches below the 1320 floor. These levels are rated, not anchored.
- `--anchor name=elo,...` fixes any other bot at a known rating.
- `--uci-movetime` sets the engine's time per move (default 100 ms).

Stockfish's Elo calibration was made at a longer time control against a CCRL-style pool, so the scale is approximate. Orthodox engines only play standard chess, and they sit out 5D pairings.

`--calibrate` carries the scale into 5D in two stages:

```sh
cargo run --release -p chronofish-benchmarks -- --tournament --calibrate \
  --stockfish /usr/games/stockfish --openings 4 --model models/training/best-v2.json \
  --save-ratings ratings.json
```

1. **Standard chess.** A ladder of Deep Blueshift budgets (`--ladder`, default 1k to 80k nodes) plays the Stockfish references (`--references`, default 1320/1600/2000) and the bots. The ladder is rated on the Stockfish scale.
2. **5D chess.** The ladder is held at those ratings while it plays the bots in both 5D rulesets. The resulting 5D numbers mean "plays 5D about as well as a standard-chess player of this strength plays standard chess". That is a mapping through one engine family, not a true equivalence.

A transfer check then refits 5D with only the middle rung fixed and compares the ladder's spacing in both games. A 5D/standard spread ratio near 1 means the bridge is credible. A ratio far from 1, or a ladder that is not monotonic, means the 5D ratings should be read as ordinal only.

## Equal-time training experiments

The existing script targets v2 artifact names. Use it only with trainers and
snapshots that agree on those filenames and rules. Reference-v3 comparisons
need the script's artifact handling updated.

For an equal-time comparison, build the trainer and benchmarks in release mode and retain a copy of the baseline trainer before making changes. Point the experiment at a saved training directory containing the best model, candidate, and replay. Choose a new output directory outside the snapshot.

```sh
python3 scripts/strength_per_hour.py --baseline /path/to/baseline-trainer \
  --trainer target/release/chronofish-trainer --benchmark target/release/chronofish-benchmarks \
  --snapshot models/training --output /tmp/chronofish-strength-cpu --device cpu --model-size balanced
```

Each backend takes nine training hours plus evaluation. The script copies the starting data, runs three seeds for each training configuration, and evaluates the last saved candidate at the deadline. Each match uses six held-out openings per ruleset with colours reversed. Per-run files include trainer timings, game outcomes, and a bootstrap interval over opening pairs. Use `--device vulkan` on a discrete GPU for the Vulkan training comparison. Both use the same CPU match evaluator.
