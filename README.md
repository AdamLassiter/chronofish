# Chronofish

Chronofish is an experimental Stockfish-like engine for **5D Chess with
Multiverse Time Travel**.

The goal is to build a fast, inspectable engine that can search legal 5D Chess
positions, evaluate tactical and strategic features across timelines, and expose
that engine through a browser-based interface.

## Project Status

This repository is at the planning/scaffolding stage.

Current contents:

- `engine/` - Rust engine crate that compiles to WebAssembly.
- `web/` - dependency-free JavaScript frontend.
- `server/` - Rust static file and multiplayer room server.
- `RULES.md` - working rules reference for the game model.
- `README.md` - project overview and intended architecture.

Current implementation:

- Rust/WASM game core for the current prototype rules, including game setup,
  legal-target generation, move application, branching, promotion, and JSON
  snapshots for display.
- JavaScript frontend that loads the WASM module and renders a basic 5D Chess
  prototype. Display, selection UI, and multiplayer room controls stay in the
  frontend; chess state and rules live in Rust.
- Rust backend for multiplayer rooms with white/black seats, spectators, and
  live game-state updates.

Planned implementation:

- board representation, legal move generation, search, and evaluation;
- position visualization, move entry, analysis, and bot play.

## Why Rust and WASM?

5D Chess has a much larger state space than orthodox chess. A useful engine
needs to make many speculative moves across boards, timelines, and turns while
keeping memory layout predictable.

Rust is intended for the heavy lifting because it gives the engine:

- explicit ownership over large game trees and board snapshots;
- good performance for search, hashing, and move generation;
- a path to native tooling for tests and benchmarks;
- straightforward compilation to WebAssembly for the frontend.

The JavaScript layer should stay focused on interaction, rendering, and
integration. Engine-critical logic should live in Rust wherever practical.

## Intended Architecture

```text
chronofish/
  engine/          Rust engine crate compiled to WASM
  server/          Rust static file and multiplayer room server
  web/             JavaScript frontend
  RULES.md         Rules reference
  README.md        Project overview
```

### Rust Engine

The engine should eventually own:

- canonical game-state representation;
- legal move generation across space, time, and timelines;
- turn submission logic for multi-move 5D turns;
- check, checkmate, and stalemate detection;
- position hashing and repetition/state caching;
- search algorithms;
- evaluation functions;
- WASM bindings for the frontend.

The current browser integration already follows this boundary: JavaScript asks
the WASM module for legal targets, applies moves through Rust, and renders the
JSON snapshot returned by the engine.

### JavaScript Frontend

The frontend should eventually provide:

- visual navigation across boards, turns, and timelines;
- legal move previews;
- current-turn composition before submit;
- engine analysis display;
- bot-vs-human and bot-vs-bot workflows;
- import/export formats once a stable notation is chosen.

## Engine Direction

Chronofish is inspired by traditional chess engines, but 5D Chess changes the
shape of the problem.

Important design questions include:

- how to represent timelines and inactive timelines efficiently;
- how to generate moves without repeatedly cloning the entire multiverse;
- how to model the present line and forced turn completion;
- how to detect attacks against kings across time and timelines;
- how to evaluate material, tempo, timeline activity, king safety, and threats;
- how to keep search useful when legal turn sequences may contain multiple
  individual moves.

Early development should prioritize correctness over search strength. A weak
engine with complete legal move generation is more valuable than a faster engine
with unsound rules.

## Planned Milestones

1. Define a minimal internal notation for boards, timelines, pieces, and moves.
2. Implement orthodox chess movement on a single board.
3. Generalize piece movement across the turn and timeline axes.
4. Implement playable boards, active timelines, and present-line turn rules.
5. Add legal turn generation and king-safety validation.
6. Add perft-style test positions for move-generation correctness.
7. Add a simple static evaluator.
8. Add alpha-beta search or another baseline search strategy.
9. Compile the engine to WebAssembly.
10. Build the first browser UI around the WASM engine.

## Development

Install the Rust WASM target once:

```sh
rustup target add wasm32-unknown-unknown
```

Run Rust tests:

```sh
cargo test
```

Build the WASM module:

```sh
npm run build:wasm
```

Start the local frontend:

```sh
npm run dev
```

Then open <http://localhost:5173>.

For multiplayer across machines, run the same server on an address reachable by
both players:

```sh
HOST=0.0.0.0 PORT=5173 npm run dev
```

Open the site, enter a room ID, and choose `Join White`, `Join Black`, or
`Spectate`. The share link in the multiplayer bar keeps the room ID in the URL.

The multiplayer backend is currently in-memory. Restarting the server clears all
rooms, and production hosting should put it behind HTTPS plus a real persistence
layer before treating games as durable.

## Rules Reference

The working rules summary lives in [`RULES.md`](RULES.md). That file should be
treated as the starting point for implementation, test fixtures, and engine
semantics.

## License

No license has been selected yet.
