# Chronofish Terminal Client

Play locally against a human or built-in bot.

```sh
cargo run -p chronofish-tui -- --mode multiverse --opponent blueshift
```

Modes are `standard`, `multiverse`, and `variant`. The terminal accepts
`move T0L0:e2 T0L0:e4`, `submit`, `undo`, `legal`, `boards`, and `quit`.
Choose an opponent with `--opponent fissure|grabablanca|blueshift-easy|blueshift-medium|blueshift|carlsen|alpha-now|cpu|gpu|kairos|loki`. Kairos and Loki support both 5D modes.
Pass `--model FILE` for a neural checkpoint.

## Game files

The Chronofish terminal client accepts `--lan-history FILE` to replay a
whitespace-separated list of 5DUCI moves. Use `submit` after each complete 5D
turn. For 5DPGN, use `--pgn-history FILE` in the 5D mode matching the file’s starting army.
This replays the main line from the selected single-board 5D setup. Each move
and submission is checked by Chronofish after the shared parser resolves the notation. Add
`--export-pgn FILE` to save the submitted main line on exit. The export
omits any pending draft and requires a new output path.
