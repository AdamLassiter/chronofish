# Chronofish

![Chronofish](logo.svg)

Chronofish is a playable Rust/WASM prototype for **5D Chess with Multiverse
Time Travel**. The rules engine lives in Rust, compiles to WebAssembly for the
browser, and is served by a small Rust backend that also provides in-memory
multiplayer rooms.

The project currently includes:

- a Rust engine crate with game state, legal moves, turn submission, checkmate
  detection, alpha-beta bot search, and a native training harness;
- a TypeScript frontend with local play, online rooms, spectators, CPU and
  WebGPU bots, bot-vs-human games, bot-vs-bot games, and browser training tools;
- a Rust `axum` server that serves `web/dist`, the built engine WASM, and room
  APIs;
- a working rules reference in [`RULES.md`](RULES.md).

## Repository Layout

```text
chronofish/
  engine/                  Rust engine crate and native trainer
    src/ai/                AI search, evaluation, weights, and parameters
    src/notation/          Notation formatting, parsing, and replay
    src/training/          Native genetic training harness
    models/cpu-v1/         CPU effort, evaluation, training, and hall-of-fame data
    models/gpu-v1/         Compact browser value model and backups
  pretty-log/              Terminal output helper used by native training
  server/                  Rust static file and multiplayer room server
  web/                     Browser frontend npm project
    src/                   TypeScript, CSS, HTML, workers, and WGSL shaders
    scripts/               Build and source-check scripts
    tests/                 Frontend functional tests
    dist/                  Generated frontend bundle (ignored)
  logs/                    Match and browser-training logs (ignored)
  RULES.md                 Rules reference
  run                      Build the app and start the normal server
  train                    Build the app and start browser training mode
  train-cpu                Run native CPU heuristic tuning
  profile-cpu              Profile native CPU tuning with cargo-flamegraph
```

## Development

Local development requires a stable Rust toolchain, Node.js 22 with npm, and
the Rust WASM target. Install the target once:

```sh
rustup target add wasm32-unknown-unknown
```

Run the test suite:

```sh
cargo test
```

Start the app:

```sh
./run
```

Then open <http://localhost:5173>. The script builds the engine for
`wasm32-unknown-unknown` in release mode, installs the locked frontend
dependencies, builds `web/dist`, and starts the Rust server. Override the bind
address for LAN play:

```sh
HOST=0.0.0.0 PORT=5173 ./run
```

Build and run the hosting container:

```sh
docker build -t chronofish .
docker run --rm -p 5173:5173 chronofish
```

Build and run the training-enabled container:

```sh
docker build -f Dockerfile.training -t chronofish-training .
docker volume create chronofish-models
docker run --rm -p 5173:5173 -v chronofish-models:/app/engine/models/gpu-v1 chronofish-training
```

The training image exposes the frontend model replacement endpoints by compiling
the server with `frontend-training`. Use the regular `Dockerfile` for public
hosting.

The same examples are available through Docker Compose:

```sh
docker compose up chronofish
docker compose --profile training up chronofish-training
```

The regular service binds to <http://localhost:5173>. The training service binds
to <http://localhost:5174> and persists trained models in the
`chronofish-models` volume. Both services persist match logs in the
`chronofish-logs` volume.

With the training service running, benchmark one bounded end-to-end cycle:

```sh
CHRONOFISH_BROWSER=/path/to/chromium npm --prefix web run training:benchmark -- --url http://127.0.0.1:5174
CHRONOFISH_BROWSER=/path/to/chromium npm --prefix web run training:benchmark:cpu -- --url http://127.0.0.1:5174
```

The benchmark prints JSON with adapter information, total time, phase timings,
sample rates, validation losses, checkpoint decisions, and replay sizes.

Useful checks before committing:

```sh
cargo fmt
cargo check --workspace --all-targets
cargo test -q
cargo clippy -- -D warnings
cargo build --release --manifest-path engine/Cargo.toml --target wasm32-unknown-unknown
npm --prefix web run check
npm --prefix web run lint
npm --prefix web test
npm --prefix web run build
```

GPU frontier validation needs a real WebGPU browser in addition to the static
shader and TypeScript checks. With the app already served, run:

```sh
CHRONOFISH_BROWSER=/path/to/chrome npm --prefix web run gpu:smoke -- --url http://localhost:5173
```

The smoke harness runs one-, three-, and five-board full GPU searches through
`ai-worker.js` and requires the returned turns to pass the worker's WASM replay
validation with one frontier readback. It also runs tactical branch, capture,
castling, en-passant, promotion, terminal-pressure, stale-generation,
device-loss/rebuild, and median latency gates by default: full resident frontier
search must be no more than 10% slower on the one-board fixture and at least 2x
faster on the three- and five-board fixtures than the legacy hybrid GPU path.
Use `--skip-performance-gates` for a legality-only browser smoke run while
debugging.

## Playing

The frontend starts in a lobby. A game can be configured as:

- local multiplayer, where one browser controls both sides;
- online multiplayer, with separate white/black seats and spectators;
- human vs CPU or WebGPU bot;
- bot vs bot.

During a turn, moves are staged until submitted. `Undo` removes the most recent
staged move, while `Reset` clears the current turn's staged moves. Checkmate and
concessions leave the game in a post-match review state until dismissed.

GPU bot modes require a browser with WebGPU support. CPU bot modes run the Rust
search engine through WASM and also expose a custom depth, node, and time preset.

The server stores rooms in memory, so restarting it clears all rooms. Match
notation is appended to `logs/<room-id>.log`.

## Engine

The engine is one crate organized as normal Rust modules. Internal APIs use
`pub(crate)` where sibling modules need to collaborate, while the public surface
stays limited to the WASM C ABI and native training entrypoint:

- `model.rs` defines core state types;
- `game.rs` applies moves, staging, submission, timelines, castling, en-passant,
  and present-line rules;
- `movegen.rs` implements attacks and legal movement across board, time, and
  timeline axes;
- `notation/` contains notation formatting, parsing, and replay;
- `ai/search.rs` implements iterative deepening, alpha-beta, quiescence search,
  move ordering, and candidate generation, while `ai/staged_search.rs` searches
  multi-board turns incrementally;
- `ai/evaluation/` scores material, timelines, safety, tactics, and 5D
  strategic features;
- `engine/models/cpu-v1/parameters.json` contains the active CPU heuristic
  evaluation weights, and `/ai/parameters.json` serves that same file;
- `engine/models/cpu-v1/effort.json` contains CPU bot effort presets and is
  served at `/ai/effort.json`;
- `engine/models/gpu-v1/effort.json` contains GPU bot effort presets and is
  served at `/ai/gpu-effort.json`;
- `wasm_api.rs` exposes the C ABI consumed by the frontend.

The default setup still uses orthodox pieces, but the engine models the variant
pieces from the rules reference so they can be introduced later without changing
the representation.

## Training

Chronofish has two separate training workflows.

### Browser Training

Start the server with model-replacement and loss-log endpoints enabled:

```sh
./train
```

Open <http://localhost:5173> and use the Training control. The browser UI can:

- train the compact GPU value model from any selected mix of GPU search, CPU
  heuristic, self-play, and distillation modes;
- mutate and score CPU heuristic parameters from selected GPU search, CPU
  heuristic, and self modes;
- upload accepted GPU models to
  `engine/models/gpu-v1/value-model.cfnn`;
- upload accepted CPU parameters to
  `engine/models/cpu-v1/parameters.json`;
- write training-loss replay data under `logs/training-losses/`.

Browser GPU search and value-model training require WebGPU. The normal `./run`
server does not expose writable training endpoints.

### Native GPU CLI

The Rust training binary can run GPU search, sample collection, and value/policy
training without starting Chromium. The `gpu` form of `./train` enables the
engine's `neural-wgpu` feature:

```sh
# Search the default starting position, or pass a game snapshot JSON file.
./train gpu --gpu-search --gpu-search-depth 2 --nodes 4096
./train gpu --gpu-search fixtures/position.json --gpu-model engine/models/gpu-v1/value-model.cfnn

# Generate search labels, then train and write a replacement model.
./train gpu --gpu-sample-search fixtures/position.json --gpu-sample-count 128 --out /tmp/chronofish-samples.json
./train gpu --gpu-train-samples /tmp/chronofish-samples.json --gpu-model engine/models/gpu-v1/value-model.cfnn --out /tmp/value-model.cfnn

# Collect labels and train in one command.
./train gpu --gpu-train-search fixtures/position.json --gpu-sample-count 128 --gpu-model engine/models/gpu-v1/value-model.cfnn --out /tmp/value-model.cfnn
```

Tune native value and policy optimization with `--gpu-learning-rate`,
`--gpu-epochs`, `--gpu-weight-decay`, and `--gpu-momentum`.

The native WGPU adapter must be available for GPU projection and training. Use
`./train gpu --gpu-backend-info`, `--gpu-compile-shaders`, or
`--gpu-dispatch-smoke` to diagnose the local GPU path. Search results and
training samples use the same engine JSON contracts as the browser bindings.
In `--gpu-sample-mode search` (the default), CLI labels are produced by the
native GPU search API; the explicit CPU, curriculum, tactical, distillation,
outcome, and duel modes retain their corresponding engine-side collection
strategies.

GPU model optimization is staged by replay diversity. Replays with at most 32
unique positions train the value and policy heads on CPU over cached hidden
features. Larger replays use WebGPU, and hidden-layer backpropagation starts at
256 unique positions. This avoids overfitting and full-network GPU work before
the replay can support it. Both backends use normalized momentum to smooth noisy
minibatch gradients without changing the configured long-run learning-rate
scale. Value targets use `[-1, 1]` for the engine's `[-20000, 20000]` score
range, and inference restores that scale before frontier minimax.
CFNN v4 records the bounded `tanh` output activation explicitly; older v1-v3
models retain their linear-output interpretation.
Replay and device-sized working-set truncation reserve at least 25% of capacity
for policy-labelled positions when enough are available, and the UI reports the
resulting policy sample count.
Worker-side caps match the detected GPU profile, allowing up to 16,384 fresh
labels, batch entries, and validation steps plus 16 parallel label workers.

To replace a corrupt or intentionally reset compact model with the deterministic
finite initializer:

```sh
npm --prefix web run model:initialize
```

### Native CPU Training

Run native heuristic tuning with:

```sh
./train-cpu --max-seconds 3600
```

The CPU trainer is native-only and lives under `engine/src/training/`. By
default it runs a coordinate sweep over selected evaluation parameters, scores
linear candidate values in paired self-play matches, commits each local winner
to the in-memory pass result, shrinks the range, and repeats. The accumulated
sweep winner is still verified against the committed baseline before promotion.

The default sweep trains the `classic-basic` parameter group. Use
`--parameter-groups classic-basic`, `--parameter-groups alternate-basic`,
`--parameter-groups classic-basic,intermediate`, `--parameter-groups advanced`,
or `--parameter-groups all` to change the field set. `--sweep-points N`
controls the candidate grid per parameter, `--sweep-passes N` caps passes
(default 2), `--sweep-range LOW:HIGH` controls the initial multiplicative range
for non-zero values, and `--sweep-shrink F` narrows the range after each pass.
Royal objective weights stay fixed. The previous sparse-mutation genetic
trainer remains available with `--strategy genetic`.

Candidate scoring and seed comparisons use Rayon parallel iterators, so training
uses available CPU cores without launching extra trainer processes. Fitness uses
paired candidate/baseline matches from identical seeded starts, mixes in tactical
mate-training positions, tracks win-rate confidence and Elo-style estimates, and
keeps recent promoted weights in a JSONL hall of fame. `./train-cpu` is evidence
bounded rather than time bounded by default: it runs comparison pairs until a
candidate is promoted, rejected, or marked inconclusive because it hit the pair
or draw-stagnation caps. Individual self-play matches are also adjudicated after
a bounded number of plies so unresolved games cannot occupy worker threads
forever; override this with `--max-match-plies N` or `--max-match-ms N` when
needed. Set `--max-seconds N` for an optional wall-clock safety limit. If a
candidate is promoted, the trainer rewrites
`engine/models/cpu-v1/parameters.json`, appends the candidate to the hall of
fame at `engine/models/cpu-v1/hall_of_fame.jsonl`, runs verification, and commits
the updated data. Training-mode servers also
expose these CPU parameters over `/api/training/cpu-parameters` for GET/PUT.

Native training uses `engine/models/cpu-v1/training.json`. Its defaults match
the former `fast` effort training values and do not change with the runtime bot effort. The
legacy `--config` and `--effort` arguments are accepted but no longer alter
training parameters.

For a short smoke run:

```sh
cargo run -q --manifest-path engine/Cargo.toml --bin train -- \
  --sweep-passes 1 --sweep-points 3 --training-time-ms 100 --nodes 20 \
  --min-pairs 2 --max-pairs 4 --max-seconds 20 --verify true
```

Alpha-beta is the default training search. Select the alternative beam strategy
through the trainer CLI:

```sh
cargo run -q --manifest-path engine/Cargo.toml \
  --bin train -- --search-strategy beam --max-seconds 20
```

Profile the native workflow with
[`cargo-flamegraph`](https://github.com/flamegraph-rs/flamegraph):

```sh
./profile-cpu --max-seconds 60
```

The default output is `flamegraph.svg`.

## AI Effort Presets

`engine/models/cpu-v1/effort.json` contains CPU runtime bot presets shared by
the Rust engine and frontend via `/ai/effort.json`.
`engine/models/gpu-v1/effort.json` contains the corresponding GPU presets,
including minimum depth, and is served via `/ai/gpu-effort.json`.

| Preset | Runtime purpose |
| --- | --- |
| `fast` | Low latency bot turns for quick local play. |
| `balanced` | Default interactive strength/speed tradeoff. |
| `expert` | Highest included browser bot effort. |

`engine/models/cpu-v1/training.json` owns search time/nodes, candidate
population and finalists for the genetic strategy, parallel pair batch, opponent
variant counts, rounds per variant, hall-of-fame depth, league composition,
promotion pair limits, draw threshold, match bounds, and candidate-stagnation
limit. The `candidates` field controls the population scored by
`--strategy genetic`. A `null` candidate count, finalist count, or pair batch
selects the host-derived automatic value. CLI flags remain available as later
overrides; for example,
`--rounds-per-variant 3` plays three paired rounds against every selected
baseline, hall-of-fame, or mutated opponent variant.

## AI Parameters

`engine/models/cpu-v1/parameters.json` is deserialized into an `EvalWeights`
value. Larger positive weights generally make the bot care more about that
feature. Some fields are piece values, while others scale positional, tactical,
or multiverse-specific terms. The tables below use Rust's `snake_case` field
names; the JSON file uses their Serde `camelCase` equivalents.

### Classic Basic Parameters

Classic Basic includes the classic chess piece material values below plus every
non-material heuristic listed under Shared Basic.

| Heuristic | Meaning |
| --- | --- |
| `king` | Material value for a royal king. Kept extremely high so king capture dominates normal material. |
| `queen` | Material value for a queen. |
| `rook` | Material value for a rook. |
| `bishop` | Material value for a bishop. |
| `knight` | Material value for a knight. |
| `pawn` | Material value for a pawn. |

### Alternate Basic Parameters

Alternate Basic includes the alternate chess piece material values below plus
every non-material heuristic listed under Shared Basic.

| Heuristic | Meaning |
| --- | --- |
| `common_king` | Material value for a non-royal common king variant. |
| `royal_queen` | Material value for a royal queen variant. |
| `princess` | Material value for the rook+bishop style princess variant. |
| `unicorn` | Material value for a three-axis diagonal slider. |
| `dragon` | Material value for a four-axis diagonal slider. |
| `brawn` | Material value for the brawn pawn variant. |

### Shared Basic Parameters

Shared Basic heuristics are included when training either `classic-basic` or
`alternate-basic`.

| Heuristic | Meaning |
| --- | --- |
| `check_penalty` | Penalty for own check and bonus for checking the opponent. |
| `active_timeline` | Value assigned to owning an active timeline. |
| `inactive_timeline` | Value assigned through timeline ownership when a timeline is inactive. |
| `present_progress` | Rewards control of the present line and progress across active timeline fronts. |
| `mobility` | Scales legal single-move count advantage. |
| `branch_penalty` | Cost applied to branch/time-travel moves during move ordering. |
| `advancement` | Rewards pieces, especially pawns, for advancing toward promotion/pressure. |
| `centrality` | Rewards pieces placed near central files/ranks. |
| `defended_piece` | Bonus when a live piece has friendly defenders. |
| `attacked_piece` | Penalty when a live piece is attacked. |
| `hanging_piece` | Extra penalty when an attacked piece has no defenders. |
| `royal_threat` | Extra value for attacks or controlled squares near royal pieces. |

### Intermediate Parameters

| Heuristic | Meaning |
| --- | --- |
| `temporal_threat` | Extra value for threats that cross time or timelines. |
| `pincer_threat` | Rewards multiple attackers converging on one target. |
| `timeline_pincer` | Rewards threats arriving from multiple timelines. |
| `historical_pincer` | Rewards threats arriving from multiple historical times. |
| `frontier_tempo` | Rewards active timeline fronts where the side to move is favorable. |
| `present_anchor` | Rewards active boards aligned with the current present board. |
| `development` | Rewards non-pawn, non-royal pieces leaving their home rank. |
| `branch_attack` | Bonus for tactically useful branch moves, especially attacking branches. |
| `check_bonus` | Move-ordering and evaluation bonus for checking lines. |
| `royal_capture_threat` | Rewards positions where a royal piece can be captured, including temporal capture paths. |
| `royal_capture_setup` | Rewards one-move setup moves that would create a royal capture threat, such as queen moves that line up a later time-travel mate. |
| `royal_escape_pressure` | Rewards own royal escape squares and penalizes boxed-in royals. |
| `forcing_move_pressure` | Rewards attacks that force replies, captures, or urgent defense. |
| `own_royal_exposure` | Penalty for attacks against the bot's own royal pieces. |
| `fork_pressure` | Rewards attacks that fork multiple valuable or royal targets. |
| `board_control` | Rewards controlled squares on latest boards, with extra value for central control. |
| `piece_activity` | Rewards active pieces with many attack lines and open lanes. |
| `pawn_structure` | Rewards passed/supported pawns and penalizes isolated or blocked pawns. |
| `timeline_economy` | Rewards useful active owned timelines and penalizes excess inactive branches. |
| `present_tempo` | Rewards favorable side-to-move tempo relative to spread across active timelines. |
| `royal_shelter` | Rewards pawn cover around royal pieces and penalizes missing shelter. |
| `space_advantage` | Rewards advanced space, weighted more heavily for pawns/brawns. |

### Advanced Parameters

| Heuristic | Meaning |
| --- | --- |
| `mandatory_move_burden` | Penalise positions where you must make many awkward moves before the Present passes back. Reward positions where the opponent has many present-board obligations. |
| `turn_completion_safety` | Estimate how many legal full-turn completion sequences exist. A side with only one or two safe ways to finish its turn is close to tactical collapse. |
| `present_zugzwang` | Penalise active present boards where every legal move worsens royal safety, loses material, or opens a temporal tactic. |
| `weakest_royal_safety` | Use a soft-min over all royal pieces rather than an average. In 5D, one doomed king/royal queen can lose the game even if the other royals are safe. |
| `royal_liability_count` | Slightly penalise having many exposed royal pieces. Extra timelines can mean extra kings/royal queens to defend, not just extra material. |
| `multi_royal_attack` | Reward threats that attack two or more royal pieces across different boards/timelines, especially when no single reply can cover all of them. |
| `defensive_bandwidth` | Estimate how many independent threats a side can answer during its current turn. This is the defensive counterpart to forcing pressure. |
| `threat_overload` | Reward positions where the opponent has more urgent threats than mandatory/available moves to address them. |
| `active_branch_capacity` | Reward having the ability to create another active timeline before your own branches become inactive; penalise being branch-saturated. Inactive timelines are explicitly a balance mechanism and do not affect the Present. |
| `latent_timeline_reactivation` | Penalise inactive enemy timelines that could become active again if timeline counts change. They may be optional now but dangerous later. |
| `inactive_material_quality` | Score material/threats on inactive timelines at a discounted value, but increase the discount if they are close to becoming active. |
| `branch_payload` | For a branch/time-travel move, evaluate the quality of the newly created frontier: material, royal safety, threats, and whether the moved piece lands with purpose. |
| `branch_waste` | Penalise branches that create a new timeline but do not create threats, improve safety, win material, or shift Present pressure. |
| `timeline_compaction` | Reward having threats concentrated on the timelines that actually matter to the Present; penalise useful-looking pieces stranded on irrelevant branches. |
| `frontier_material` | Evaluate material only on latest/playable boards separately from historical material. Old-board material may be tactically relevant, but frontier material is what can move now. |
| `historical_access` | Reward pieces that can reach important past boards, especially past boards containing vulnerable royals, promotion paths, or branchable tactical positions. |
| `temporal_lane_control` | Reward open lanes along T/L axes and temporal diagonals for rooks, bishops, unicorns, dragons, queens, and royal queens. |
| `temporal_pin` | Detect pieces pinned through time/timelines because moving them would expose a royal to capture on another board. |
| `temporal_skewer` | Reward attacks where a high-value or royal piece is behind another piece along a temporal ray. |
| `causal_battery` | Reward two-piece batteries that line up through time/timeline dimensions, especially queen/unicorn/dragon batteries aimed at royal fronts. |
| `arrival_square_safety` | For temporal moves, evaluate whether the destination board square is safe after arrival. Time-travel attacks can look strong but simply strand a piece. |
| `source_board_abandonment` | Penalise temporal moves that remove a key defender from the source frontier, especially if that source board still matters this turn. |
| `piece_temporal_flexibility` | Reward pieces that have useful moves in both spatial and temporal dimensions, not merely many legal moves. |
| `dimension_coverage_balance` | Reward armies that control x/y/T/L threats in a balanced way. A side with only spatial pressure may be blind to temporal tactics. |
| `promotion_timeline_choice` | Reward pawns/brawns whose advancement creates multiple promotion or branching options across timelines, not just rank progress. |
| `promotion_with_check` | Extra reward for promotions that immediately create royal threats across time or timelines. |
| `past_royal_vulnerability` | Penalise royal pieces sitting on historical boards that are reachable by enemy time travel, even if they are not threatened on the current frontier. |
| `safe_haven_boards` | Reward having boards/timelines where a royal can retreat or branch defensively without creating a losing inactive timeline. |
| `escape_branch_potential` | Reward legal time-travel/branch moves that can rescue a royal from a future attack. |
| `mate_net_depth_1_2` | A shallow specialised search feature: count whether the side has one-turn or two-turn royal-capture nets, even if the static attack map undervalues them. |
| `anti_mate_resources` | Count defensive resources against known 5D mate patterns: capture attacker, move royal, block temporal ray, branch away, or shift Present. |
| `checking_move_quality` | Separate good checks from bad checks. Penalise checks that create useless branches or lose timeline economy; reward checks that reduce legal full-turn completions. |
| `search_volatility` | Mark positions with many royal threats, branch moves, or timeline activations as tactically volatile, encouraging deeper/quiescence search there. |
| `timeline_repetition_risk` | Penalise positions likely to create aimless branch proliferation without progress, especially if your evaluation otherwise overvalues material copies. |
| `phase_by_multiverse_size` | Taper weights based on number of active timelines/frontier boards, not just material. Opening-like development matters less once temporal tactics dominate. |
| `royal_distance_in_4d` | Tropism-style distance from attacking pieces to enemy royals using 4D movement distance, piece-specific. This is broader than “near royal squares”. |
| `board_importance_weight` | Weight each board by active/inactive status, present distance, side to move, royal presence, and tactical volatility before summing local features. |

## Rules Reference

The implementation follows [`RULES.md`](RULES.md). When changing rules, update
that file first, then add focused Rust tests for the affected move generation,
turn submission, checkmate, or timeline behavior.

## License

### The MIT License (MIT)

Copyright © 2026 Adam Lassiter

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
