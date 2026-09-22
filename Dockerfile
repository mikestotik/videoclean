FROM oven/bun:1 AS webui-build
WORKDIR /build
COPY webui/package.json webui/bun.lock ./
RUN bun install --frozen-lockfile
COPY webui/ ./
RUN bun run build

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.11 \
    UV_PROJECT_ENVIRONMENT=/opt/videoclean \
    VIRTUAL_ENV=/opt/videoclean \
    PATH="/opt/videoclean/bin:/usr/local/bin:${PATH}" \
    VIDEOCLEAN_DATA_DIR=/root/.videoclean \
    HF_HOME=/root/.cache/huggingface \
    VIDEOCLEAN_PORT=7860 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    SAM2_BUILD_CUDA=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        ffmpeg \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY videoclean ./videoclean
COPY server ./server
COPY scripts/start.sh ./scripts/start.sh

# pyproject pins torch==2.2.2 for local hardware; the server image installs
# cu124 wheels (>=2.5) for sam2-video instead. torch is excluded from the
# sync so it is downloaded only once. Do not download HF weights at build time.
# Split into layers: deps -> torch -> sam2, so code-only rebuilds reuse them.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra gpu --extra lama --extra web --no-dev \
        --no-install-package torch --no-install-package torchvision
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/videoclean/bin/python \
        --index-url https://download.pytorch.org/whl/cu124 \
        "torch>=2.5" torchvision
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/videoclean/bin/python \
        "git+https://github.com/facebookresearch/sam2.git" \
    && chmod +x /app/scripts/start.sh

COPY --from=webui-build /server/static_dist ./server/static_dist

EXPOSE 7860

CMD ["/app/scripts/start.sh"]
