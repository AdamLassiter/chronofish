# Chronofish 5D turn generation

Shared reference rules for Chronofish, Kairos, and ftxi. The crate owns bitboard
state, timeline activity, move generation, and lazy hypercuboid enumeration of
complete turns. The referee checks that a submitted turn advances the Present
and protects the mover's royals. Its draft API finds individually playable
moves with legal completions.

Every draft choice is proven: a move is offered only when a generated complete
turn contains it and the staged order reproduces that turn. Large positions are
answered with one witness search per candidate instead of enumerating every
turn, and the order check skips intermediate positions already shown to fail.
Proofs still cost far more than listing moves, so the reference rules are
several times slower than [Chronofish 5D Rules](../5d-movegen/README.md). They
are used only in games against bots that require them.

The `prof` feature instruments rules search. See [upstream attribution](NOTICE)
and [Chronofish's rules](../../RULES.md).

```sh
cargo test -p chronofish-5d-turngen
```

The [Kairos perft tests](../kairos-tools/tests/perft.rs) also cover this crate.
`fast_choices_match_full_enumeration_along_branching_games` checks the witness
search against full enumeration along seeded branching games.
