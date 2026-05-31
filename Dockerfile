# syntax=docker/dockerfile:1

FROM rust:1-bookworm AS builder

WORKDIR /app

RUN rustup target add wasm32-unknown-unknown

COPY Cargo.toml Cargo.lock ./
COPY engine/Cargo.toml engine/Cargo.toml
COPY server/Cargo.toml server/Cargo.toml
COPY engine/src engine/src
COPY server/src server/src

RUN cargo build --release --manifest-path engine/Cargo.toml --target wasm32-unknown-unknown
RUN cargo build --release -p chronofish-server

FROM debian:bookworm-slim AS runtime

WORKDIR /app

ENV HOST=0.0.0.0
ENV PORT=5173

COPY --from=builder /app/target/release/chronofish-server /usr/local/bin/chronofish-server
COPY --from=builder /app/target/wasm32-unknown-unknown/release/chronofish_engine.wasm /app/target/wasm32-unknown-unknown/release/chronofish_engine.wasm
COPY web /app/web
COPY engine/src/ai/parameters.json /app/engine/src/ai/parameters.json
COPY engine/src/ai/effort.json /app/engine/src/ai/effort.json

EXPOSE 5173

CMD ["chronofish-server"]
