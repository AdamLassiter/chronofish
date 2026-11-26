# Chronofish

Chronofish is a Rust chess engine and web app for orthodox chess and 5D chess
with time travel and branching timelines. The two 5D modes use Chronofish’s move-first
rules engine and offer an orthodox or alternate starting army. See the
[game rules](RULES.md).

## Quick start

Run these commands from the workspace root. The Rust workspace requires 1.92
or newer.

```sh
cargo run -p chronofish-tui -- --mode multiverse
docker compose up --build
cargo test --workspace --all-targets --locked
```

The web app is at [localhost:3000/app](http://localhost:3000/app) with the
default configuration. See the [server guide](crates/server/README.md) for a
local build and configuration, or the [trainer guide](crates/trainer/README.md)
to train a model.

## Workspace

| Crate | Purpose |
| --- | --- |
| [core](crates/core/README.md) | Game API and orthodox rules |
| [5d-movegen](crates/5d-movegen/README.md) | Default move generation and game state |
| [5d-turngen](crates/5d-turngen/README.md) | Kairos-derived whole-turn engine and referee |
| [notation](crates/notation/README.md) | 5DPGN, 5DFEN, and move histories |
| [classic](crates/classic/README.md) | Classical Chronofish bots |
| [neural](crates/neural/README.md) | Model, encoding, and inference |
| [alphazero](crates/alphazero/README.md) | Neural search and self-play |
| [ftxi](crates/ftxi/README.md) | Loki 5D Monte Carlo search engine |
| [kairos](crates/kairos/README.md) | Kairos 5D alpha-beta engine |
| [kairos-tools](crates/kairos-tools/README.md) | Engine tools and reusable 5D puzzle fixtures |
| [trainer](crates/trainer/README.md) | Training, hardware setup, and promotion |
| [benchmarks](crates/benchmarks/README.md) | Throughput, tournaments, and ratings |
| [server](crates/server/README.md) | Web app and persistent games |
| [tui](crates/tui/README.md) | Local terminal play |

Chronofish is licensed under [GNU GPL v3.0](LICENSE.md).
