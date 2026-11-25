# Kairos tools and shared fixtures

This workspace crate houses the standalone 5DUCI interfaces, analysis APIs,
puzzle generator, tuning tools and match runner. The corpus in `data/` and
`tests/data/` includes tactical puzzles, sacrifice positions, mate regressions,
perft positions, variant setups and opening lines. The `.g` files use 5DUCI
position histories and can be loaded through
[`chronofish_notation::parse_g_line`](../notation/README.md) for other bot tests.
Annotated source files preserve the original solutions and provenance.

From the workspace root:

```sh
cargo test -p kairos-tools
cargo run --release -p kairos-tools --bin kairos
cargo run --release -p kairos-tools --bin ktools -- tactics crates/kairos-tools/data/tactics.g 2000 1
cargo run --release -p kairos-tools --bin ktools -- perft 3 < crates/kairos-tools/data/perft.g
cargo run --release -p kairos-tools --bin puzzlegen -- --help
```

The standalone Kairos engine accepts `5duci`, `isready`, `position startpos`
and `go movetime 1000`. Legacy scripts are preserved in `scripts/`. Scripts
written for the upstream checkout may require upstream paths or CI setup.

## ftxi CLI

```sh
cargo run --release -p kairos-tools --bin ftxi -- 5duci zero-capture-check-pw 42
cargo run --release -p kairos-tools --bin ftxi -- 128 4 42 pgn game.5dpgn
cargo run --release -p kairos-tools --bin ftxi -- 128 4 42 lan '(0T1)e2e4' submit
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
