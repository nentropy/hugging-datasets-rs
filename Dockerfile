FROM rust:1.69 as builder
WORKDIR /usr/src/app
COPY Cargo.toml Cargo.lock ./
RUN cargo fetch
COPY ./src ./src
COPY ./hugging_datasets ./hugging_datasets
RUN cargo build --release

FROM debian:buster-slim
WORKDIR /usr/local/bin
COPY --from=builder /usr/src/app/target/release/my_ml_project .
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN chmod +x ./my_ml_project
EXPOSE 8080
CMD ["./","src","lib.rs"]
