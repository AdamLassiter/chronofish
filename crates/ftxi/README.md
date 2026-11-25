# Loki

Loki is Chronofish’s Rust port of ftxi’s 5D chess engine. It uses Monte Carlo
tree search over the same reference-rules action space that Kairos searches
with alpha-beta. By default Loki prioritizes captures and checks as it widens
the tree and searches for up to 128 simulations or 500 ms per turn.

`LokiBot` implements `chronofish_core::AiPlayer` and supports both 5D modes.
See [upstream attribution and port differences](UPSTREAM.md) and the
[standalone CLI](../kairos-tools/README.md#ftxi-cli).
