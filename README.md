# Chronofish

Chronofish is a Rust engine for standard chess and chess played across time and branching timelines. It provides one authoritative rules crate, AlphaZero-style CPU and Vulkan/WGPU bots, classic alpha-beta, rollout and greedy bots, self-play training, a debug terminal client, and a persistent web server.

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

## Bots

Every bot implements `AiPlayer`. Each one isolates a different part of how an engine plays.

| Bot | Username | Architecture | Starting Elo |
|---|---|---|---|
| Bobby Fissure | `bot-bobby-fissure` | Uniformly random legal actions; the baseline (formerly Raven) | 995 |
| Grabablanca | `bot-grabablanca` | Greedy: the best handcrafted evaluation one action ahead | 1152 |
| Deep Blueshift (easy) | `bot-deep-blueshift-easy` | Deep Blueshift with a 1k-node budget | 1263 |
| Deep Blueshift (medium) | `bot-deep-blueshift-medium` | Deep Blueshift with a 10k-node budget | 1416 |
| Deep Blueshift | `bot-deep-blueshift` | Iterative-deepening alpha-beta with a transposition table, quiescence and an 80k-node budget | 1517 |
| Monte Carlsen | `bot-monte-carlsen` | UCT tree search with capture-biased random rollouts; no network | 1159 |
| AlphaNow | `bot-alpha-now` | The AlphaZero network's top policy action, with no search | 1034 |
| Chronofish CPU / GPU | `bot-chronofish-cpu` / `-gpu` | AlphaZero MCTS guided by the trained network | 1500 |

A new bot account starts at the lower bound of its 95% interval from the standard-chess calibration against Stockfish (see below). Using the lower bound keeps new accounts from being overrated while the intervals are still wide. A bot account is reseeded each time the server starts, but only until it plays its first rated game; after that it keeps the rating it has earned. The Chronofish networks are not calibrated yet and start at 1500. The seeds live in `seed_rating` in `crates/server/src/lib.rs`.

The classic bots live in `crates/classic` and share one material-and-placement evaluation that is aware of active timelines. Their search runs over actions rather than plies, because a 5D turn is several staged moves followed by a submission. The server registers every bot. AlphaNow shares Chronofish CPU's checkpoint. In the TUI, choose a bot with `--opponent fissure|grabablanca|blueshift-easy|blueshift-medium|blueshift|carlsen|alpha-now|cpu|gpu`, and pass `--model` for a trained network.

Compare bots with a round-robin tournament. Every pair plays each random opening from both sides. The tournament prints standings with W/D/L, Elo and time per action, followed by a crosstable:

```sh
cargo run --release -p chronofish-benchmarks -- --tournament \
  --openings 2 --ruleset all --max-actions 80 --model models/training/best-v2.json
```

Use `--bots` to choose a subset and `--csv` to save one row per game. `blueshift-<nodes>` (for example `blueshift-10k`) adds Deep Blueshift with a different node budget.

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

## Training

Run the trainer with no arguments:

```sh
cargo run --release -p chronofish-trainer
```

In an interactive terminal it first opens a setup screen. The screen lists every parameter by group (run, self-play, training, arena, hardware and mode), with a description, the default, and a suggested value where the default is a poor choice for real training. An Outlook pane estimates the cost of an iteration and how many times each self-play position is refitted. Above about 4 refits, the value head starts memorising game results instead of learning to judge positions.

- Use `↑`/`↓` to move.
- `Enter` edits a value or toggles an option, and `←`/`→` cycles choices.
- `s`/`S` applies the suggested value for one parameter or for all of them.
- `d`/`D` restores the default for one parameter or for all of them.
- `F5` starts the run, and the live dashboard opens.

Arguments on the command line pre-fill the form. Values that differ from the defaults are saved to `trainer-settings.txt` in the work directory and pre-filled on the next run. One-shot modes are not saved. `--skip-setup` starts straight away with the given arguments, and piped or redirected runs (scripts, CI) skip the screen automatically:

```powershell
.\chronofish-trainer.exe --skip-setup --work-dir models\training-gpu --ruleset all --training-device vulkan
```

Once training starts, the live dashboard shows every scheduled self-play and arena game with its progress and result, the arena scoreboard and promotion outlook, action throughput, inference batching, and the result of each iteration. Press `q` to quit. When the run ends, the full log is printed to stdout. Output that is piped or redirected stays plain text, and `--plain` forces plain text in a terminal.

Arena search is deterministic, so each colour-reversed pair of arena games starts from its own random opening (`--arena-opening-actions`, default 8). Both games in a pair use the same opening, so the comparison stays fair. Openings change as training progresses.

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

The server listens on `127.0.0.1:3000` by default and serves the built browser client at `http://127.0.0.1:3000/app` (and `/`). Static client assets are embedded from `web/dist` under `/assets`. It stores accounts, games, history, Elo, and player bug reports in `chronofish.sqlite3`. Configure it with `CHRONOFISH_ADDR`, `CHRONOFISH_DATABASE`, `CHRONOFISH_AZ_MODEL`, `CHRONOFISH_GPU_MODEL`, `CHRONOFISH_AZ_SIMULATIONS`, `CHRONOFISH_BLUESHIFT_NODES`, and `CHRONOFISH_CARLSEN_ITERATIONS`.

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

## Other 5D engines

The workspace includes Kairos and a Rust implementation of ftxi's adversarial
UCT search. Both depend on `chronofish-5d-rules` for whole-turn move generation
and on `chronofish-notation` for game interchange. Chronofish's own game
state and legal actions live in `chronofish-core`. Bobby Fissure and the
other native classical bots live in `chronofish-classic`.
Under those rules, search generates complete turns that protect the moving
side's royals. Chronofish Multiverse uses the same shared referee.

```sh
cargo run --release -p kairos --bin kairos
cargo run --release -p chronofish-ftxi -- 5duci zero-capture-check-pw 42
cargo run --release -p chronofish-ftxi -- 128 4 42 pgn game.5dpgn
cargo run --release -p chronofish-ftxi -- 128 4 42 lan '(0T1)e2e4' submit
```

The ftxi command takes a simulation budget, rollout turn limit, and seed. A
rollout limit of zero uses static zero evaluation with terminal scoring.
Use `weighted:4` in place of `4` for weighted coordinate rollouts. The
`zero-capture` mode orders hypercuboid coordinates by captures.
`zero-capture-check-pw` also favors checks on and across boards and expands children
under a visit-dependent limit. The tree shares UCT statistics across prefixes of legal hypercuboid points
and uses seeded tie-breaking. A resumable cursor completes turns inside the selected coordinate prefix
and keeps unsearched sibling regions available for later visits. `linear:N`, `linear-trained:N`, and `linear-weighted:N`
use the upstream 64-feature evaluation after at most N rollout turns.
`flat:N` and `flat-weighted:N` allocate rollouts among root turns only.
`monkey` samples uniformly across all complete legal turns and ignores the
simulation budget. In 5DUCI mode, send `position startpos` or a 5DFEN setup,
then `go nodes N` or `go movetime MS`. `stop` returns the best turn found so
far, or `nobestmove` if search has yet to find one.
It prints the selected turn in both 5DUCI long algebraic notation and 5DPGN.
Kairos speaks 5DUCI and its `ktools pgn2g` command converts the main line of a
5DPGN file to a 5DFEN setup and coordinate move history.

The Chronofish terminal client accepts `--lan-history FILE` to replay a
whitespace-separated list of 5DUCI moves. Use `submit` after each complete 5D
turn. For 5DPGN, use `--pgn-history FILE` with the default Multiverse mode.
This replays the main line from the selected single-board 5D setup. Each move
and submission is checked by Chronofish after the shared parser resolves the notation. Add
`--export-pgn FILE` to save the submitted main line on exit. The export
omits any pending draft and requires a new output path.

### Engine and variant references

- ftxi shares immutable historical boards across search copies. Kairos uses
  compact bitboards and an undo log to remove appended boards and branches
  after search. Chronofish's existing board snapshots also carry castling,
  en passant, and historical origin metadata. A useful first experiment is
  to reuse its existing search undo for standard move generation. It now
  reuses one scratch game for all king-safety checks in an action list.
  In two paired release runs over eight positions after action 140, each with
  48 AlphaZero simulations and batch size 8, mean time outside the evaluator
  fell from 21–23 ms to about 6 ms per action. End-to-end results ranged from
  230–245 ms before and 229–248 ms after. Inference variation dominated the
  total. Shared history may still reduce clone cost in branched 5D searches.
  The current evidence favors keeping that as a separate experiment.
- adri326's variants repository supplies 5DPGN and 5DFEN examples for import
  tests. Custom sizes, even starting timelines and changes to promotion rules
  make it a good fixture source for the Kairos and ftxi analysis paths. Any
  Chronofish import needs to validate those settings explicitly.
  [Variants catalog](https://github.com/adri326/5dchess-variants)
- adri326's notation specification is useful for interchange. Its separate
  Rust analysis tool documents another lazy whole-turn search, and the
  upstream README identifies its default branch as unmaintained.
  [Notation](https://github.com/adri326/5dchess-notation) ·
  [Analysis tools](https://github.com/adri326/5dchess-tools)
- Kairos's Turn Zero opening guide and f7-sacrifice lines provide start
  positions for fixed tactical comparisons. Check expected outcomes under
  the receiving engine's rules before using a line as a training target.
  [Kairos theory](https://github.com/FirePlank/5d-chess-engine/tree/main/docs/theory)

### Live 5D rules

New Chronofish 5D games use the shared reference engine. Every offered draft
move has a checked completion, and Submit advances the Present only when the
mover's royals are safe. Orthodox Standard Chess uses the native rules path.
Training runs that include 5D use the `reference-v3` checkpoint and replay
filenames. Standard-only training keeps the v2 filenames. For a fresh 5D run,
`--warm-start PATH` explicitly imports model weights into the reference checkpoint.

The server loads 5D neural weights from `best-reference-v3.json` in the CPU
and GPU model directories. Set `CHRONOFISH_REFERENCE_AZ_MODEL` or
`CHRONOFISH_REFERENCE_GPU_MODEL` to select a different 5D checkpoint.
