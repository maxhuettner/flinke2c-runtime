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
        libc6-dev-amd64-cross \
        curl \
        apt-transport-https \
        gpg \
        wget \
        && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://packages.adoptium.net/artifactory/api/gpg/key/public \
    | gpg --dearmor -o /usr/share/keyrings/adoptium.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb $(. /etc/os-release && echo $VERSION_CODENAME) main" \
    > /etc/apt/sources.list.d/adoptium.list
RUN apt-get update && apt-get install -y --no-install-recommends temurin-25-jdk && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN rustup target add x86_64-unknown-linux-gnu
RUN rustup component add clippy
RUN rustup component add rustfmt
RUN curl -LO https://ziglang.org/download/0.15.2/zig-aarch64-linux-0.15.2.tar.xz
RUN tar -xf zig-aarch64-linux-0.15.2.tar.xz -C /usr/local/bin --strip-components=1
RUN rm zig-aarch64-linux-0.15.2.tar.xz
RUN cargo install --locked cargo-zigbuild

ENV JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-arm64

WORKDIR /workspaces/flinke2c-runtime

CMD ["bash"]
