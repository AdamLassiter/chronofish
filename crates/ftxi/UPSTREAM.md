# ftxi upstream

The Rust engine follows [ftxi/5dchess_engine at 664277d](https://github.com/ftxi/5dchess_engine/tree/664277de910a3bc3b146d9d5f27da7de490f3ed4).
See the accompanying LICENSE for the BSD-2-Clause notice.

The port uses the Rust state and hypercuboid cursor extracted from Kairos
into `chronofish-5d-turngen` for legal complete turns. It implements adversarial UCT across coordinate prefixes, progressive
widening, flat UCB, random and weighted rollouts, the default and frozen
linear evaluators, and uniform random play. Its RNG and cursor differ from
the upstream C++ implementations, so the same seed can select different
turns. Regression fixtures compare linear features and scores, check counts,
and the aggregate rollout score with the upstream C++ implementation.
