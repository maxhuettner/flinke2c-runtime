FROM rust:slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_TERM_COLOR=always

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        git \
        libibumad-dev \
        libibverbs-dev \
        librdmacm-dev \
        linux-libc-dev \
        pkg-config \
        rdma-core \
        cmake clang lld pkg-config \
        gcc-x86-64-linux-gnu g++-x86-64-linux-gnu \
        libc6-dev-amd64-cross
    && rm -rf /var/lib/apt/lists/*

RUN curl -LO https://ziglang.org/download/0.15.2/zig-aarch64-linux-0.15.2.tar.xz
RUN tar -xf zig-aarch64-linux-0.15.2.tar.xz -C /usr/local/bin --strip-components=1
RUN rm zig-aarch64-linux-0.15.2.tar.xz

WORKDIR /workspaces/flinke2c-runtime

CMD ["bash"]
