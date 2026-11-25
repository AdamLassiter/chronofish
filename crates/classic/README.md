# Chronofish Classic Bots

Implements the classical `AiPlayer` opponents used by the server, terminal, and
benchmarks. Their handcrafted material and placement evaluation accounts for
active timelines. In 5D, search explores staged actions and submission.

| Bot | Method |
|---|---|
| Bobby Fissure | Seeded random legal actions |
| Grabablanca | Greedy one-action lookahead |
| Deep Blueshift | Iterative-deepening alpha-beta, quiescence, and a transposition table |
| Monte Carlsen | UCT with capture-biased rollouts |

Deep Blueshift ships with Easy, Medium, and Hard budgets. See the [benchmark
guide](../benchmarks/README.md) for tournament and custom-budget examples.

```sh
cargo test -p chronofish-classic
```
