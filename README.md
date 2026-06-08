# Chronofish

![Chronofish](logo.svg)

Chronofish is a playable Rust/WASM prototype for **5D Chess with Multiverse
Time Travel**. The rules engine lives in Rust, compiles to WebAssembly for the
browser, and is served by a small Rust backend that also provides in-memory
multiplayer rooms.

The project currently includes:

- a Rust engine crate with game state, legal moves, turn submission, checkmate
  detection, alpha-beta bot search, and a native training harness;
- a dependency-free JavaScript frontend npm package for local play, online
  rooms, spectators, bot-vs-human, and bot-vs-bot games;
- a Rust `axum` server that serves `web/dist`, the built engine WASM, and room
  APIs;
- a working rules reference in [`RULES.md`](RULES.md).

## Repository Layout

```text
chronofish/
  engine/                  Rust engine crate and native trainer
    src/ai/                AI search, evaluation, weights, and parameters
    src/training/          Native genetic training harness
  server/                  Rust static file and multiplayer room server
  web/                     Browser frontend npm project
    src/                   Frontend source files
    tests/                 Frontend functional tests
    dist/                  Generated frontend bundle (ignored)
  RULES.md                 Rules reference
  run                      Build WASM and start the backend
  train                    Repeatedly run training/promotion cycles
```

## Development

Install the WASM target once:

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
`wasm32-unknown-unknown`, builds the frontend into `web/dist`, and starts the
Rust server. Override the bind address for LAN play:

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

Useful checks before committing:

```sh
cargo fmt
cargo test -q
cargo clippy -- -D warnings
cargo build --release --manifest-path engine/Cargo.toml --target wasm32-unknown-unknown
npm --prefix web run check
npm --prefix web test
npm --prefix web run build
```

## Playing

The frontend starts in a lobby. A game can be configured as:

- local multiplayer, where one browser controls both sides;
- online multiplayer, with separate white/black seats and spectators;
- human vs bot;
- bot vs bot.

During a turn, moves are staged until submitted. `Undo` removes the most recent
staged move, while `Reset` clears the current turn's staged moves. Checkmate and
concessions leave the game in a post-match review state until dismissed.

The server stores rooms in memory. Restarting it clears all rooms.

## Engine

The engine is one crate split into small include files so private helpers can be
shared without exposing a broad crate API:

- `model.rs` defines core state types;
- `game.rs` applies moves, staging, submission, timelines, castling, en-passant,
  and present-line rules;
- `movegen.rs` implements attacks and legal movement across board, time, and
  timeline axes;
- `ai/search.rs` implements iterative deepening, alpha-beta, quiescence search,
  move ordering, and turn-plan generation;
- `ai/evaluation/` scores material, timelines, safety, tactics, and 5D
  strategic features;
- `engine/models/cpu-v1/parameters.json` contains the active CPU heuristic
  evaluation weights, and `/ai/parameters.json` serves that same file;
- `wasm_api.rs` exposes the C ABI consumed by the frontend.

The default setup still uses orthodox pieces, but the engine models the variant
pieces from the rules reference so they can be introduced later without changing
the representation.

## Training

Run the frontend GPU training server with:

```sh
./train
```

Run native CPU heuristic tuning with:

```sh
./train-cpu --max-seconds 3600
```

The CPU trainer is native-only and lives under `engine/src/training/`. It mutates
the evaluation weights, scores candidates in short self-play matches, verifies a
candidate against the committed baseline, and only promotes it when paired match
evidence clears the configured confidence thresholds.

Candidate scoring and seed comparisons use Rayon parallel iterators, so training
uses available CPU cores without launching extra trainer processes. Fitness uses
paired candidate/baseline matches from identical seeded starts, mixes in tactical
mate-training positions, tracks win-rate confidence and Elo-style estimates, and
keeps recent promoted weights in a JSONL hall of fame. `./train-cpu` is evidence
bounded rather than time bounded by default: it runs comparison pairs until a
candidate is promoted, rejected, or marked inconclusive because it hit the pair
or draw-stagnation caps. Set `TRAIN_MAX_SECONDS` for an optional wall-clock
safety limit. If a candidate is promoted, the trainer rewrites
`engine/models/cpu-v1/parameters.json`, appends the candidate to the hall of
fame, runs verification, and commits the updated data. Training-mode servers also
expose these CPU parameters over `/api/training/cpu-parameters` for GET/PUT.

Training uses the shared AI effort presets from `engine/models/cpu-v1/effort.json`.
`./train` defaults to `expert`; set `TRAIN_CONFIG=fast`, `TRAIN_CONFIG=balanced`,
or pass `--config fast|balanced|expert` to run another preset.

For a short smoke run:

```sh
cargo run -q --manifest-path engine/Cargo.toml --bin train -- \
  --config fast --generations 1 --population 4 --depth 1 --nodes 20 --plies 1 \
  --min-pairs 4 --max-pairs 8 --max-seconds 20
```

## AI Effort Presets

`engine/models/cpu-v1/effort.json` is shared by the Rust engine/trainer and the
frontend via `/ai/effort.json`.

| Preset | Runtime purpose | Training purpose |
| --- | --- | --- |
| `fast` | Low latency bot turns for quick local play. | Small search budget for smoke checks. |
| `balanced` | Default interactive strength/speed tradeoff. | Moderate self-play search. |
| `expert` | Highest included browser bot effort. | Default trainer configuration. |

## AI Parameters

`engine/models/cpu-v1/parameters.json` is deserialized into an `EvalWeights` value. Larger positive
weights generally make the bot care more about that feature. Some fields are
piece values, while others scale positional, tactical, or multiverse-specific
terms.

### Basic Parameters

| Heuristic | Meaning |
| --- | --- |
| `king` | Material value for a royal king. Kept extremely high so king capture dominates normal material. |
| `common_king` | Material value for a non-royal common king variant. |
| `queen` | Material value for a queen. |
| `royal_queen` | Material value for a royal queen variant. |
| `princess` | Material value for the rook+bishop style princess variant. |
| `rook` | Material value for a rook. |
| `bishop` | Material value for a bishop. |
| `unicorn` | Material value for a three-axis diagonal slider. |
| `dragon` | Material value for a four-axis diagonal slider. |
| `knight` | Material value for a knight. |
| `pawn` | Material value for a pawn. |
| `brawn` | Material value for the brawn pawn variant. |
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
