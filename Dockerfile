# syntax=docker/dockerfile:1.7

FROM rust:1.92-bookworm AS builder

WORKDIR /src
COPY . .
RUN cargo build --locked --release -p chronofish-server

FROM debian:bookworm-slim AS runtime

LABEL org.opencontainers.image.title="Chronofish" \
      org.opencontainers.image.description="Standard and five-dimensional chess server" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 chronofish \
    && useradd --uid 10001 --gid chronofish --no-create-home --shell /usr/sbin/nologin chronofish \
    && install --directory --owner chronofish --group chronofish /data

COPY --from=builder /src/target/release/chronofish-server /usr/local/bin/chronofish-server

USER chronofish:chronofish
WORKDIR /data

ENV CHRONOFISH_ADDR=0.0.0.0:3000 \
    CHRONOFISH_DATABASE=/data/chronofish.sqlite3 \
    RUST_LOG=info

VOLUME ["/data"]
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent --show-error http://127.0.0.1:3000/api/health >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/chronofish-server"]
