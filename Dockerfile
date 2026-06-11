# syntax=docker/dockerfile:1

FROM rust:1-bookworm AS builder

WORKDIR /app

RUN rustup target add wasm32-unknown-unknown
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY engine/Cargo.toml engine/Cargo.toml
COPY server/Cargo.toml server/Cargo.toml
COPY pretty-log/Cargo.toml pretty-log/Cargo.toml
COPY web/package.json web/package.json
COPY web/package-lock.json web/package-lock.json
COPY web/scripts web/scripts
COPY web/src web/src
COPY engine/src engine/src
COPY server/src server/src
COPY pretty-log/src pretty-log/src

RUN npm --prefix web ci
RUN npm --prefix web run build
RUN cargo build --release --manifest-path engine/Cargo.toml --target wasm32-unknown-unknown
RUN cargo build --release -p chronofish-server

FROM debian:bookworm-slim AS runtime

WORKDIR /app

ENV HOST=0.0.0.0
ENV PORT=5173

COPY --from=builder /app/target/release/chronofish-server /usr/local/bin/chronofish-server
COPY --from=builder /app/target/wasm32-unknown-unknown/release/chronofish_engine.wasm /app/target/wasm32-unknown-unknown/release/chronofish_engine.wasm
COPY --from=builder /app/web/dist /app/web/dist

RUN mkdir -p /app/engine/models/gpu-v1 /app/engine/models/cpu-v1
VOLUME ["/app/engine/models/gpu-v1", "/app/engine/models/cpu-v1"]

EXPOSE 5173

CMD ["chronofish-server"]
