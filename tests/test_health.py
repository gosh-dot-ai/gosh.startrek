# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import pytest


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health must return 200 with {"status": "ok"}."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        from src.mcp_server import create_app
        app = create_app(app_data_dir=tmp)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
