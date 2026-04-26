# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import pytest
from tests._auth_helpers import bootstrap_harness


@pytest.mark.asyncio
async def test_unauthorized_without_token():
    """Routes other than /health must return 401 without valid token."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        from src.mcp_server import SERVER_TOKEN, create_app
        app = create_app(app_data_dir=tmp)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            # /health should work without token
            r = await c.get("/health")
            assert r.status_code == 200

            # /admin/reload should require token
            r = await c.post("/admin/reload", json={"key": "x"})
            assert r.status_code == 401

            # /api/v1/admin/memory/init should require token
            r = await c.post("/api/v1/admin/memory/init", json={"key": "x"})
            assert r.status_code == 401

            # /mcp/sse should require token
            # (SSE endpoint returns streaming, but 401 comes first)
            r = await c.get("/mcp/sse")
            assert r.status_code == 401


@pytest.mark.asyncio
async def test_authorized_with_valid_token():
    """Routes accept requests with valid server token + persisted admin bearer."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        import src.mcp_server as mcp_mod
        with pytest.MonkeyPatch.context() as mp:
            harness = bootstrap_harness(mp, tmp)
            app = mcp_mod.create_app(app_data_dir=tmp)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                headers = {
                    "x-server-token": mcp_mod.SERVER_TOKEN,
                    "Authorization": f"Bearer {harness.admin_token}",
                }

                r = await c.post("/admin/reload", json={"key": "nonexistent"},
                                 headers=headers)
                assert r.status_code == 200
                assert r.json()["status"] == "reloaded"

                init = await c.post(
                    "/api/v1/admin/memory/init",
                    json={"key": "init-key"},
                    headers=headers,
                )
                assert init.status_code == 200
                assert init.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_valid_server_token_without_principal_token_is_denied():
    """Perimeter token alone must not authorize protected admin routes."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        import src.mcp_server as mcp_mod
        with pytest.MonkeyPatch.context() as mp:
            bootstrap_harness(mp, tmp)
            app = mcp_mod.create_app(app_data_dir=tmp)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(
                    "/admin/reload",
                    json={"key": "nonexistent"},
                    headers={"x-server-token": mcp_mod.SERVER_TOKEN},
                )
                assert r.status_code == 401
                assert r.json()["code"] == "AUTH_REQUIRED"


@pytest.mark.asyncio
async def test_valid_server_token_with_invalid_principal_token_is_denied():
    """Perimeter token must not bless an invalid bearer token."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        import src.mcp_server as mcp_mod
        with pytest.MonkeyPatch.context() as mp:
            bootstrap_harness(mp, tmp)
            app = mcp_mod.create_app(app_data_dir=tmp)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(
                    "/admin/reload",
                    json={"key": "nonexistent"},
                    headers={
                        "x-server-token": mcp_mod.SERVER_TOKEN,
                        "Authorization": "Bearer not-a-real-token",
                    },
                )
                assert r.status_code == 403
                assert r.json()["code"] == "INVALID_TOKEN"


@pytest.mark.asyncio
async def test_valid_principal_token_with_missing_or_invalid_server_token_is_denied():
    """When perimeter token is configured, valid bearer auth still needs it."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        import src.mcp_server as mcp_mod
        with pytest.MonkeyPatch.context() as mp:
            harness = bootstrap_harness(mp, tmp)
            app = mcp_mod.create_app(app_data_dir=tmp)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                missing = await c.post(
                    "/admin/reload",
                    json={"key": "nonexistent"},
                    headers={"Authorization": f"Bearer {harness.admin_token}"},
                )
                invalid = await c.post(
                    "/admin/reload",
                    json={"key": "nonexistent"},
                    headers={
                        "x-server-token": "wrong-token-value",
                        "Authorization": f"Bearer {harness.admin_token}",
                    },
                )
                assert missing.status_code == 401
                assert invalid.status_code == 401


@pytest.mark.asyncio
async def test_x_server_token_does_not_elevate_caller_role():
    """Perimeter token must not change caller identity or grant admin role."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        import src.mcp_server as mcp_mod
        with pytest.MonkeyPatch.context() as mp:
            harness = bootstrap_harness(mp, tmp)
            agent_token = harness.issue("agent:alice", kind="agent")
            app = mcp_mod.create_app(app_data_dir=tmp)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(
                    "/admin/reload",
                    json={"key": "nonexistent"},
                    headers={
                        "x-server-token": mcp_mod.SERVER_TOKEN,
                        "Authorization": f"Bearer {agent_token}",
                    },
                )
                assert r.status_code == 403
                assert r.json()["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_wrong_token_rejected():
    """Wrong token should get 401."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        from src.mcp_server import create_app
        app = create_app(app_data_dir=tmp)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            headers = {"x-server-token": "wrong-token-value"}
            r = await c.post("/admin/reload", json={"key": "x"},
                             headers=headers)
            assert r.status_code == 401
