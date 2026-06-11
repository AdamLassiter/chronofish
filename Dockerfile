# syntax=docker/dockerfile:1

FROM node:22-bookworm-slim AS web-builder

WORKDIR /app

COPY web/package.json web/package.json
COPY web/package-lock.json web/package-lock.json
COPY web/scripts web/scripts
COPY web/src web/src

RUN npm --prefix web ci
RUN npm --prefix web run build

FROM rust:1-bookworm AS builder

WORKDIR /app

RUN rustup target add wasm32-unknown-unknown

COPY Cargo.toml Cargo.lock ./
COPY engine/Cargo.toml engine/Cargo.toml
COPY server/Cargo.toml server/Cargo.toml
COPY pretty-log/Cargo.toml pretty-log/Cargo.toml
COPY engine/src engine/src
COPY server/src server/src
COPY pretty-log/src pretty-log/src

RUN cargo build --release --manifest-path engine/Cargo.toml --target wasm32-unknown-unknown
RUN cargo build --release -p chronofish-server

FROM debian:bookworm-slim AS runtime

WORKDIR /app

ENV HOST=0.0.0.0
ENV PORT=5173
ENV CHRONOFISH_CPU_MODEL_DIR=/app/engine/models/cpu-v1

COPY --from=builder /app/target/release/chronofish-server /usr/local/bin/chronofish-server
COPY --from=builder /app/target/wasm32-unknown-unknown/release/chronofish_engine.wasm /app/target/wasm32-unknown-unknown/release/chronofish_engine.wasm
COPY --from=web-builder /app/web/dist /app/web/dist

RUN mkdir -p /app/engine/models/gpu-v1 /app/engine/models/cpu-v1
VOLUME ["/app/engine/models/gpu-v1", "/app/engine/models/cpu-v1"]

EXPOSE 5173

CMD ["chronofish-server"]
