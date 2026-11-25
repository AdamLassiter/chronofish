# Chronofish AlphaZero

Owns policy-guided MCTS, the CPU and GPU-labelled bot adapters, AlphaNow's
policy-only play, self-play, arenas, and replay targets. It uses the ordered
legal action set from [Chronofish Core](../core/README.md) and model inference
from [Chronofish Neural](../neural/README.md). Staged 5D moves keep the mover's
value perspective until submission.

Use [the trainer](../trainer/README.md) to run a training loop or
[the benchmarks](../benchmarks/README.md) to compare opponents.

```sh
cargo test -p chronofish-alphazero
```
