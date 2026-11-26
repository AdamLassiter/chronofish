# Chronofish 5D Rules

Shared reference rules for Chronofish, Kairos, and ftxi. The crate owns bitboard
state, timeline activity, move generation, and lazy hypercuboid enumeration of
complete turns. The referee checks that a submitted turn advances the Present
and protects the mover's royals. Its draft API finds individually playable
moves with legal completions.

The `prof` feature instruments rules search. See [upstream attribution](NOTICE)
and [Chronofish's rules](../../RULES.md).

```sh
cargo test -p chronofish-5d-rules-reference
```

The [Kairos perft tests](../kairos/tests/perft.rs) also cover this crate.
