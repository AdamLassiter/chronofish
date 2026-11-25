# Chronofish Core

Owns the application-facing `Game`, `PlayerAction`, `GameView`, and `AiPlayer`
interfaces. Standard Chess uses the native move generator. New 5D games
delegate legality and adjudication to [the reference rules crate](../5d-rules/README.md)
and expose staged moves, undo, submission, and board views through the same API.

For interchange formats, use [chronofish-notation](../notation/README.md).
See the [rules guide](../../RULES.md) for gameplay.

```sh
cargo test -p chronofish-core
```
