# Chronofish

![Chronofish](logo.svg)

Chronofish is a playable Rust/WASM prototype for **5D Chess with Multiverse
Time Travel**. The rules engine lives in Rust, compiles to WebAssembly for the
browser, and is served by a small Rust backend that also provides in-memory
multiplayer rooms.

The project currently includes:

- a Rust engine crate with game state, legal moves, turn submission, checkmate
  detection, alpha-beta bot search, and a native training harness;
- a dependency-free JavaScript frontend for local play, online rooms,
  spectators, bot-vs-human, and bot-vs-bot games;
- a Rust `axum` server that serves `web/`, the built engine WASM, and room APIs;
- a working rules reference in [`RULES.md`](RULES.md).

## Repository Layout

```text
chronofish/
  engine/                  Rust engine crate and native trainer
    src/ai/                AI search, evaluation, weights, and parameters
    src/training/          Native genetic training harness
  server/                  Rust static file and multiplayer room server
  web/                     Browser frontend and WASM loader
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
`wasm32-unknown-unknown` and starts the Rust server. Override the bind address
for LAN play:

```sh
HOST=0.0.0.0 PORT=5173 ./run
```

Useful checks before committing:

```sh
cargo fmt
cargo test -q
cargo clippy -- -D warnings
cargo build --release --manifest-path engine/Cargo.toml --target wasm32-unknown-unknown
node --check web/main.js
node --check web/ai-worker.js
node --check web/wasm-loader.js
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
- `ai/evaluation.rs` scores material, timelines, safety, tactics, and 5D
  strategic features;
- `ai/parameters.json` contains the committed tuned evaluation weights and is
  also served to the frontend at `/ai/parameters.json`;
- `wasm_api.rs` exposes the C ABI consumed by the frontend.

The default setup still uses orthodox pieces, but the engine models the variant
pieces from the rules reference so they can be introduced later without changing
the representation.

## Training

Run a continuous training cycle with:

```sh
./train
```

The trainer is native-only and lives under `engine/src/training/`. It mutates
the evaluation weights, scores candidates in short self-play matches, verifies a
candidate against the committed baseline, and only promotes it when paired match
evidence clears the configured confidence thresholds.

Candidate scoring and seed comparisons use Rayon parallel iterators, so training
uses available CPU cores without launching extra trainer processes. Fitness uses
paired candidate/baseline matches from identical seeded starts, mixes in tactical
mate-training positions, tracks win-rate confidence and Elo-style estimates, and
keeps recent promoted weights in a JSONL hall of fame. `./train` is evidence
bounded rather than time bounded by default: it runs comparison pairs until a
candidate is promoted, rejected, or marked inconclusive because it hit the pair
or draw-stagnation caps. Set `TRAIN_MAX_SECONDS` for an optional wall-clock
safety limit. If a candidate is promoted, the trainer rewrites
`engine/src/ai/parameters.json`, appends the candidate to the hall of fame, runs
verification, and commits the updated data.

Training uses the shared AI effort presets from `engine/src/ai/effort.json`.
`./train` defaults to `expert`; set `TRAIN_CONFIG=fast`, `TRAIN_CONFIG=balanced`,
or pass `--config fast|balanced|expert` to run another preset.

For a short smoke run:

```sh
cargo run -q --manifest-path engine/Cargo.toml --bin train -- \
  --config fast --generations 1 --population 4 --depth 1 --nodes 20 --plies 1 \
  --min-pairs 4 --max-pairs 8 --max-seconds 20
```

## AI Effort Presets

`engine/src/ai/effort.json` is shared by the Rust engine/trainer and the
frontend via `/ai/effort.json`.

| Preset | Runtime purpose | Training purpose |
| --- | --- | --- |
| `fast` | Low latency bot turns for quick local play. | Small search budget for smoke checks. |
| `balanced` | Default interactive strength/speed tradeoff. | Moderate self-play search. |
| `expert` | Highest included browser bot effort. | Default trainer configuration. |

## AI Parameters

`engine/src/ai/parameters.json` is deserialized into an `EvalWeights` value. Larger positive
weights generally make the bot care more about that feature. Some fields are
piece values, while others scale positional, tactical, or multiverse-specific
terms.

| Field | Meaning |
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

## Rules Reference

The implementation follows [`RULES.md`](RULES.md). When changing rules, update
that file first, then add focused Rust tests for the affected move generation,
turn submission, checkmate, or timeline behavior.

## License

No license has been selected yet.
