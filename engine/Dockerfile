# Writing engine, built against a pinned llama.cpp with CUDA. The host driver (580) supports up to CUDA 13.0.
ARG CUDA_VERSION=13.0.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt server.cpp ./
# 89 = Ada (RTX 4090). Static libs so the runtime image needs only the binary.
RUN cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --target writing-engine

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/writing-engine /usr/local/bin/writing-engine
COPY docker-entrypoint.sh check-model-update.sh /usr/local/bin/
EXPOSE 52409
ENTRYPOINT ["docker-entrypoint.sh"]
