# Chronofish 5D rules

Owns the game state and move-first generator used by default in Standard Chess,
both 5D modes, and AlphaZero training. It handles staged moves, timeline
creation and acceptance, submission, undo, and game outcomes.

Explicit reference matches and previously saved reference games use the
[Kairos-derived rules](../5d-rules-reference/README.md).

```sh
cargo test -p chronofish-5d-rules
```
