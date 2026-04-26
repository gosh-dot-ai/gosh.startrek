# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

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
