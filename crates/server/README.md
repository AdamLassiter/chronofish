# Chronofish Server

Serves the browser client and owns accounts, games, history, ratings, and
player reports. The server validates all moves and runs built-in opponents.
Run the following commands from the workspace root.

## Run locally

```sh
cd web
npm ci
npm run build
cd ..
cargo run --release -p chronofish-server
```

The server listens on `127.0.0.1:3000` by default and serves the built browser client at `http://127.0.0.1:3000/app` (and `/`). Static client assets are embedded from `web/dist` under `/assets`. It stores accounts, games, history, Elo, and player bug reports in `chronofish.sqlite3`. Configure it with `CHRONOFISH_ADDR`, `CHRONOFISH_DATABASE`, `CHRONOFISH_AZ_MODEL`, `CHRONOFISH_REFERENCE_AZ_MODEL`, `CHRONOFISH_AZ_SIMULATIONS`, `CHRONOFISH_BLUESHIFT_NODES`, and `CHRONOFISH_CARLSEN_ITERATIONS`.

The browser client is server-authoritative. It renders every board in the multiverse, highlights legal planning moves, and records a bot's principal variation with its move for post-game review.

Set `CHRONOFISH_BUG_ADMIN_TOKEN` to a secret random string to enable bug-report administration. Players submit reports from an open game using their normal session; every report captures the exact internal game state, game summary, full action history with the state after every action, reporter, game and server versions, and the expected and actual behaviour. The management endpoints accept the configured secret as an `Authorization: Bearer` token:

```sh
curl -H "Authorization: Bearer $CHRONOFISH_BUG_ADMIN_TOKEN" \
  http://127.0.0.1:3000/api/admin/bugs
curl -X POST -H "Authorization: Bearer $CHRONOFISH_BUG_ADMIN_TOKEN" \
  http://127.0.0.1:3000/api/admin/bugs/123/close
```

`GET /api/admin/bugs` returns unresolved reports and all captured diagnostic fields. `POST /api/admin/bugs/{id}/close` marks one report closed; administration returns `503 Service Unavailable` when no token is configured.

## Containers

```sh
docker compose up --build
```

The compose example persists SQLite state in a named volume. GitHub and Gitea workflows run formatting, Clippy, tests, the throughput check, build Linux and Windows trainer artifacts, and publish the server image on pushes.

## Built-in opponents

| Bot | Username | Architecture | Starting Elo |
|---|---|---|---|
| Bobby Fissure | `bot-bobby-fissure` | Uniformly random legal actions; the baseline (formerly Raven) | 995 |
| Grabablanca | `bot-grabablanca` | Greedy: the best handcrafted evaluation one action ahead | 1152 |
| Kairos | `bot-kairos` | Alpha-beta over complete 5D turns | 1500 |
| Loki | `bot-loki` | Monte Carlo tree search over complete 5D turns | 1500 |
| Deep Blueshift (easy) | `bot-deep-blueshift-easy` | Deep Blueshift with a 1k-node budget | 1263 |
| Deep Blueshift (medium) | `bot-deep-blueshift-medium` | Deep Blueshift with a 10k-node budget | 1416 |
| Deep Blueshift | `bot-deep-blueshift` | Iterative-deepening alpha-beta with a transposition table, quiescence and an 80k-node budget | 1517 |
| Monte Carlsen | `bot-monte-carlsen` | UCT tree search with capture-biased random rollouts; no network | 1159 |
| AlphaNow | `bot-alpha-now` | The AlphaZero network's top policy action, with no search | 1034 |
| Chronofish | `bot-chronofish` | AlphaZero MCTS guided by the trained network, with CPU inference | 1500 |

A new bot account starts at the lower bound of its 95% interval from the [standard-chess calibration against Stockfish](../benchmarks/README.md#ratings-on-a-recognised-scale). Using the lower bound keeps new accounts from being overrated while the intervals are still wide. A bot account is reseeded each time the server starts, but only until it plays its first rated game; after that it keeps the rating it has earned. Uncalibrated bots, including Kairos, Loki and Chronofish, start at 1500. The seeds live in `seed_rating` in [the server library](src/lib.rs).

Chronofish and AlphaNow share checkpoint weights for each engine. Kairos and Loki start explicit reference-rules matches. Saved reference matches keep their engine. Default matches use
`models/training/best-v2.json` (`CHRONOFISH_AZ_MODEL`) and reference matches use
`models/training/best-reference-v3.json` (`CHRONOFISH_REFERENCE_AZ_MODEL`). A
network trained on either the CPU or the GPU backend works, since a checkpoint
carries its architecture; the server always runs inference on the CPU.

Chronofish replaces the earlier Chronofish CPU and GPU bots. On startup the
`bot-chronofish-cpu` account is renamed to `bot-chronofish` and keeps its rating
and games. A bot account the server no longer registers, such as
`bot-chronofish-gpu`, stays on the leaderboard with its history but cannot be
challenged.
