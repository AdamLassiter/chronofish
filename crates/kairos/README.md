# Kairos

Kairos searches complete 5D chess turns with iterative deepening, principal
variation search, a transposition table, move ordering and tactical quiescence.
It shares the [reference rules](../5d-rules-reference/README.md) with Loki. The engine
embeds its evaluation networks and Turn Zero opening book from `data/`.

`KairosBot` implements `chronofish_core::AiPlayer` and supports both 5D modes.
The Chronofish adapter searches with a 20,000-node and 500 ms budget per turn.
See [upstream provenance](UPSTREAM.md) and [standalone tools and fixtures](../kairos-tools/README.md).
