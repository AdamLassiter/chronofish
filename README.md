# Chronofish

Chronofish is a Rust engine for standard chess and chess played across time and branching timelines. It provides one authoritative rules crate, AlphaZero-style CPU and Vulkan/WGPU bots, self-play training, a debug terminal client, and a persistent web server.

The implementation is a clean break from Chronofish's former alpha-beta, Stockfish curriculum, WASM, and browser-training stack. Old checkpoints are intentionally incompatible with the new neural representation.

## Rulesets

- `standard`: ordinary 8×8 chess, including check, checkmate, stalemate, castling, en passant, promotion, repetition, the fifty-move rule, and insufficient material.
- `multiverse`: 5D chess with the orthodox king, queen, rook, bishop, knight, and pawn army.
- `multiverse-variant`: the alternate royal queen, common king, princess, dragon, unicorn, and brawn army described in [RULES.md](RULES.md).

The 5D engine retains the Present, active-timeline, branching, and multi-move turn rules in [RULES.md](RULES.md). A move into a historical board creates a new timeline; turns are submitted only after every required board at the Present has advanced.

## Build and test

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p chronofish-tui -- --mode multiverse
```

The TUI accepts commands such as `move T0L0:e2 T0L0:e4`, `submit`, `undo`, `legal`, and `boards`.

## Training

CPU training:

```sh
cargo run --release -p chronofish-trainer -- \
  --work-dir models/training --ruleset all --training-device cpu
```

Vulkan training (including AMD GPUs through the installed Vulkan driver):

```powershell
.\chronofish-trainer.exe --work-dir models\training-gpu --ruleset all --training-device vulkan
```

Use `--check-training-device` to initialize and verify the selected device without beginning a run. The trainer prints the resolved adapter, fitting and inference backends, parallelism, memory budgets, and phase timings at startup. CPU and Vulkan checkpoints share the same architecture and file format, but old pre-rewrite models do not.

Built-in server opponents are exposed as **Chronofish CPU** and **Chronofish GPU**. The GPU name describes the checkpoint/training path; inference uses the backend selected by the executable and its feature set.

## Web server

```sh
cd web
npm ci
npm run build
cd ..
cargo run --release -p chronofish-server
```

The server listens on `127.0.0.1:3000` by default and serves the built browser client at `http://127.0.0.1:3000/app` (and `/`). Static client assets are embedded from `web/dist` under `/assets`. It stores accounts, games, history, and Elo in `chronofish.sqlite3`. Configure it with `CHRONOFISH_ADDR`, `CHRONOFISH_DATABASE`, `CHRONOFISH_AZ_MODEL`, `CHRONOFISH_GPU_MODEL`, and `CHRONOFISH_AZ_SIMULATIONS`.

The browser client is server-authoritative. It renders every board in the multiverse, highlights legal planning moves, and records a bot's principal variation with its move for post-game review.

## Containers

```sh
docker compose up --build
```

The compose example persists SQLite state in a named volume. GitHub and Gitea workflows run formatting, Clippy, tests, the throughput check, build Linux and Windows trainer artifacts, and publish the server image on pushes.
