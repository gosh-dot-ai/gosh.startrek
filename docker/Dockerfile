# gosh-memory Docker image
# Builds a minimal image with all runtime dependencies including SQLCipher.
#
# Build:
#   docker build -t gosh-memory:latest -f docker/Dockerfile .
#
# Run (via gosh CLI):
#   gosh memory setup local --runtime docker --data-dir /path/to/data

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlcipher-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install deps + entry point
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir . pysqlcipher3

# ── Runtime stage ───────────────────────────────────────────────────────

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlcipher1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv with all deps and entry point
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source — needed because setuptools auto-discovery doesn't install
# the `src` package into site-packages (it treats `src` as a src-layout root).
# PYTHONPATH makes `from src.cli import main` work.
COPY src/ /app/src/
ENV PYTHONPATH="/app"

WORKDIR /app

EXPOSE 8765

ENTRYPOINT ["gosh-memory"]
