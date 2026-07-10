from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastmcp import Client, FastMCP
from fastmcp.server.auth import StaticTokenVerifier

from colab_codex_adapter.broker import (
    BrokerCoordinator,
    BrokerState,
    broker_is_healthy,
)


async def test_multiple_proxies_share_one_authenticated_broker(
    tmp_path: Path, unused_tcp_port: int
) -> None:
    state_file = tmp_path / "broker.json"
    lock_file = tmp_path / "broker.lock"
    owner = BrokerCoordinator(
        port=unused_tcp_port,
        state_file=state_file,
        lock_file=lock_file,
        startup_timeout=2.0,
    )
    state = await owner.claim()
    assert owner.is_owner is True

    verifier = StaticTokenVerifier(
        {state.token: {"client_id": "test-proxy", "scopes": []}}
    )
    backend = FastMCP("broker-test", auth=verifier)

    @backend.tool()
    async def shared_identity() -> dict[str, int]:
        return {"owner_pid": os.getpid()}

    server_task = asyncio.create_task(
        backend.run_http_async(
            show_banner=False,
            host="127.0.0.1",
            port=unused_tcp_port,
            json_response=True,
            stateless_http=True,
        )
    )
    try:
        await owner.wait_until_healthy(state, server_task)
        owner.publish(state)

        child = BrokerCoordinator(
            port=unused_tcp_port,
            state_file=state_file,
            lock_file=lock_file,
            startup_timeout=2.0,
        )
        shared_state = await child.claim()

        assert child.is_owner is False
        assert shared_state == state
        assert state_file.stat().st_mode & 0o777 == 0o600

        proxy = FastMCP.as_proxy(Client(state.endpoint, auth=state.token))
        async with Client(proxy) as client:
            names = {tool.name for tool in await client.list_tools()}
            result = await client.call_tool("shared_identity", {})

        assert names == {"shared_identity"}
        assert result.data == {"owner_pid": os.getpid()}

        invalid = BrokerState(
            endpoint=state.endpoint,
            token="invalid",
            owner_pid=state.owner_pid,
            started_at=state.started_at,
        )
        assert await broker_is_healthy(invalid) is False
    finally:
        owner.release(state)
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    assert not state_file.exists()


async def test_new_owner_removes_stale_state(tmp_path: Path, unused_tcp_port: int) -> None:
    state_file = tmp_path / "broker.json"
    lock_file = tmp_path / "broker.lock"
    state_file.write_text(
        json.dumps(
            {
                "endpoint": "http://127.0.0.1:1/mcp",
                "token": "stale",
                "owner_pid": 999_999,
                "started_at": 1,
            }
        ),
        encoding="utf-8",
    )
    coordinator = BrokerCoordinator(
        port=unused_tcp_port,
        state_file=state_file,
        lock_file=lock_file,
        startup_timeout=1.0,
    )

    state = await coordinator.claim()

    assert coordinator.is_owner is True
    assert state.owner_pid == os.getpid()
    assert not state_file.exists()
    coordinator.release(state)
