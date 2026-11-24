# Chronofish

Chronofish is a Rust engine for standard chess and chess played across time and branching timelines. It provides one authoritative rules crate, AlphaZero-style CPU and Vulkan/WGPU bots, self-play training, a debug terminal client, and a persistent web server.

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

Use `--check-training-device` to initialize and verify the selected device without beginning a run. The trainer prints the resolved adapter, fitting and inference backends, parallelism, memory budgets, and phase timings at startup. CPU and Vulkan checkpoints share the same architecture and file format.

Training keeps a persistent learner in `candidate-v2.json`, separate from the published `best-v2.json`. Every round and `--fit-only` run continues that learner, even after a failed promotion or a restart. Use `--reset-candidate` to start fitting from the incumbent again. A candidate older than the incumbent is automatically replaced by the incumbent.

The default promotion screen plays six games (each ruleset once per colour), with 16 simulations per action and a 160-action limit. Self-play retains its independent 48-simulation, 400-action budget. This reduces the arena's maximum simulation budget from 76,800 to 15,360, though actual elapsed time depends on game complexity and hardware. Adjust `--arena-simulations`, `--arena-max-actions`, and `--arena-games` separately from self-play. Unfinished games count as draws.

The default `--promotion-score 0.5` accepts tied scores so draw-heavy early training can advance. Promotion means passing this small screening match, not demonstrated strength improvement. Use `--promotion-score 0.55 --arena-games 12 --arena-simulations 48 --arena-max-actions 400` for a more expensive strict-improvement screen. Failed screens retain the incumbent without throwing away learner updates. `--arena-only` screens the saved candidate without fitting; `--arena-games 0` publishes without a screen.

Use multiple iterations in one invocation to reuse validated replay trajectories. The first fit checks the full replay, validating independent games on the trainer’s worker pool. Later fits reconstruct only the positions needed for sampling. The trainer reports replay and checkpoint write times, inference batch occupancy, queue wait, and compute time. CPU fitting encodes each distinct board layout once per batch. The “Spatial boards” line totals the layouts encoded, real board occurrences, and padded board slots across optimizer updates. These counts show how much work repeated 5D history saves on your replay.

Add `--persistent-adam` to experiment with retaining optimizer moments and update counts between fits. The candidate checkpoint stores the moments atomically alongside the weights and verifies their weight digest on load. Older candidates start with fresh moments. Fitting without the flag drops saved moments, and `--reset-candidate` starts a fresh optimizer from the incumbent.

For an equal-time comparison, build the trainer and benchmarks in release mode and retain a copy of the baseline trainer before making changes. Point the experiment at a saved training directory containing the best model, candidate, and replay. Choose a new output directory outside the snapshot.

```sh
python3 scripts/strength_per_hour.py --baseline /path/to/baseline-trainer \
  --trainer target/release/chronofish-trainer --benchmark target/release/chronofish-benchmarks \
  --snapshot models/training --output /tmp/chronofish-strength-cpu --device cpu --model-size balanced
```

Each backend takes nine training hours plus evaluation. The script copies the starting data, runs three seeds for each training configuration, and evaluates the last saved candidate at the deadline. Each match uses six held-out openings per ruleset with colours reversed. Per-run files include trainer timings, game outcomes, and a bootstrap interval over opening pairs. Use `--device vulkan` on a discrete GPU for the Vulkan training comparison. Both use the same CPU match evaluator.

Built-in server opponents are exposed as **Chronofish CPU** and **Chronofish GPU**. The GPU name describes the checkpoint/training path; inference uses the backend selected by the executable and its feature set.

## Web server

```sh
cd web
npm ci
npm run build
cd ..
cargo run --release -p chronofish-server
```

The server listens on `127.0.0.1:3000` by default and serves the built browser client at `http://127.0.0.1:3000/app` (and `/`). Static client assets are embedded from `web/dist` under `/assets`. It stores accounts, games, history, Elo, and player bug reports in `chronofish.sqlite3`. Configure it with `CHRONOFISH_ADDR`, `CHRONOFISH_DATABASE`, `CHRONOFISH_AZ_MODEL`, `CHRONOFISH_GPU_MODEL`, and `CHRONOFISH_AZ_SIMULATIONS`.

The browser client is server-authoritative. It renders every board in the multiverse, highlights legal planning moves, and records a bot's principal variation with its move for post-game review.

Set `CHRONOFISH_BUG_ADMIN_TOKEN` to a secret random string to enable bug-report administration. Players submit reports from an open game using their normal session; every report captures the exact internal game state, game summary, full action history with the state after every action, reporter, game and server versions, and the expected and actual behaviour. The management endpoints accept the configured secret as an `Authorization: Bearer` token:

```sh
curl -H "Authorization: Bearer $CHRONOFISH_BUG_ADMIN_TOKEN" \
  http://127.0.0.1:3000/api/admin/bugs
curl -X POST -H "Authorization: Bearer $CHRONOFISH_BUG_ADMIN_TOKEN" \
  http://127.0.0.1:3000/api/admin/bugs/123/close
```

`GET /api/admin/bugs` returns unresolved reports and all captured diagnostic fields. `POST /api/admin/bugs/{id}/close` marks one report closed; administration returns `503 Service Unavailable` when no token is configured.

## Containers

```sh
docker compose up --build
```

The compose example persists SQLite state in a named volume. GitHub and Gitea workflows run formatting, Clippy, tests, the throughput check, build Linux and Windows trainer artifacts, and publish the server image on pushes.
