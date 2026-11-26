# Chronofish Trainer

Runs self-play, model fitting, and candidate promotion.

## Run the trainer

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

Training keeps a persistent candidate separate from the incumbent best model. Every round and `--fit-only` run continues that learner, even after a failed promotion or a restart. Use `--reset-candidate` to start fitting from the incumbent again. A candidate older than the incumbent is automatically replaced by the incumbent.

The default promotion screen plays six games (each ruleset once per colour), with 16 simulations per action and a 160-action limit. Self-play retains its independent 48-simulation, 400-action budget. This reduces the arena's maximum simulation budget from 76,800 to 15,360, though actual elapsed time depends on game complexity and hardware. Adjust `--arena-simulations`, `--arena-max-actions`, and `--arena-games` separately from self-play. Unfinished games count as draws.

The default `--promotion-score 0.5` accepts tied scores so draw-heavy early training can advance. Promotion means passing this small screening match, not demonstrated strength improvement. Use `--promotion-score 0.55 --arena-games 12 --arena-simulations 48 --arena-max-actions 400` for a more expensive strict-improvement screen. Failed screens retain the incumbent without throwing away learner updates. `--arena-only` screens the saved candidate without fitting; `--arena-games 0` publishes without a screen.

Use multiple iterations in one invocation to reuse validated replay trajectories. The first fit checks the full replay, validating independent games on the trainer’s worker pool. Later fits reconstruct only the positions needed for sampling. The trainer reports replay and checkpoint write times, inference batch occupancy, queue wait, and compute time. CPU fitting encodes each distinct board layout once per batch. The “Spatial boards” line totals the layouts encoded, real board occurrences, and padded board slots across optimizer updates. These counts show how much work repeated 5D history saves on your replay.

Add `--persistent-adam` to experiment with retaining optimizer moments and update counts between fits. The candidate checkpoint stores the moments atomically alongside the weights and verifies their weight digest on load. Older candidates start with fresh moments. Fitting without the flag drops saved moments, and `--reset-candidate` starts a fresh optimizer from the incumbent.

## Checkpoints and compatibility

All trainer modes use `best-v2.json`, `candidate-v2.json`, and
`replay-v2.bin.zst` with Chronofish’s move-first generator. The
`reference-v3` files belong to the reference-rules run. To reuse its model
weights, pass `--warm-start PATH` in a fresh work directory. The optimizer and
replay start fresh.

For tournament evaluation and equal-time experiments, see the [benchmark guide](../benchmarks/README.md).
