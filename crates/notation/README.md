# Chronofish Notation

Provides 5DPGN parsing and formatting, 5DFEN setup parsing, and 5DUCI move
history replay. With the `chronofish` feature it also replays and exports
Chronofish 5D main lines and exposes the original move parser under `legacy`.
Game replay validates actions through the receiving engine's rules.

The [5DPGN specification](https://github.com/adri326/5dchess-notation) and
[variant catalog](https://github.com/adri326/5dchess-variants) provide format
examples and fixture sources. See [upstream attribution](NOTICE).

```sh
cargo test -p chronofish-notation --features chronofish
```
