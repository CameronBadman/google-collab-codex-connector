from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import os
import secrets
import signal
import time
import uuid
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import IO, Awaitable, Callable

from .broker import (
    BROKER_HEALTH_TIMEOUT_SECONDS,
    BROKER_PROTOCOL_VERSION,
    BROKER_START_TIMEOUT_SECONDS,
    BROKER_STATE_VERSION,
    DEFAULT_BROKER_FACTORY,
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_LOCK_FILE,
    DEFAULT_BROKER_PORT,
    DEFAULT_BROKER_STATE_FILE,
    BrokerLaunchConfig,
    BrokerState,
    _next_generation,
    _try_lock,
    _unlock,
    broker_is_healthy,
    broker_process_start_identity,
    read_broker_state,
    stopped_state,
    write_broker_state,
)

BrokerServeCallback = Callable[[BrokerState], Awaitable[None]]


def load_serve_callback(specification: str) -> BrokerServeCallback:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Broker factory must use the form 'module:async_callable'")
    target = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(target):
        raise TypeError(f"Broker factory {specification!r} is not callable")
    return target


async def _acquire_lifetime_lock(
    path: Path, *, deadline: float
) -> IO[str]:
    while time.monotonic() < deadline:
        handle = _try_lock(path)
        if handle is not None:
            return handle
        await asyncio.sleep(0.05)
    raise TimeoutError("Timed out acquiring detached broker lifetime lock")


def _recovery_token(previous: BrokerState | None, config: BrokerLaunchConfig) -> str:
    if (
        previous is not None
        and previous.protocol_version == config.protocol_version
        and previous.endpoint == config.endpoint
        and previous.token
    ):
        return previous.token
    return secrets.token_urlsafe(32)


def _recovery_service_instance_id(
    previous: BrokerState | None, config: BrokerLaunchConfig
) -> str:
    if previous is not None and previous.protocol_version == config.protocol_version:
        return previous.service_instance_id or previous.owner_id or uuid.uuid4().hex
    return uuid.uuid4().hex


async def run_broker_daemon(
    config: BrokerLaunchConfig,
    serve: BrokerServeCallback,
) -> None:
    """Own the broker lifetime lock and run an injected HTTP MCP backend."""

    deadline = time.monotonic() + config.startup_timeout
    lock_handle = await _acquire_lifetime_lock(config.lock_file, deadline=deadline)
    previous = read_broker_state(config.state_file)
    now = time.time()
    state = BrokerState(
        endpoint=config.endpoint,
        token=_recovery_token(previous, config),
        owner_pid=os.getpid(),
        service_instance_id=_recovery_service_instance_id(previous, config),
        owner_id=uuid.uuid4().hex,
        owner_process_start=broker_process_start_identity(os.getpid()) or "",
        previous_owner_id=previous.owner_id if previous is not None else "",
        previous_generation=(
            previous.generation if previous is not None else None
        ),
        generation=_next_generation(previous),
        version=BROKER_STATE_VERSION,
        protocol_version=config.protocol_version,
        status="starting",
        started_at=now,
        updated_at=now,
    )
    write_broker_state(config.state_file, state)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signum, stop_event.set)
            installed_signals.append(signum)

    server_task = asyncio.create_task(serve(state), name="colab-broker-backend")
    terminal_status = "stopped"
    try:
        while time.monotonic() < deadline:
            if server_task.done():
                await server_task
                raise RuntimeError("Detached broker backend stopped during startup")
            if await broker_is_healthy(state, timeout=config.health_timeout):
                state = replace(state, status="ready", updated_at=time.time())
                write_broker_state(config.state_file, state)
                break
            await asyncio.sleep(0.05)
        else:
            raise TimeoutError("Detached broker backend did not become healthy")

        stop_task = asyncio.create_task(stop_event.wait(), name="colab-broker-stop")
        done, _ = await asyncio.wait(
            {server_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if server_task in done:
            await server_task
        stop_task.cancel()
        with suppress(asyncio.CancelledError):
            await stop_task
    except BaseException:
        terminal_status = "failed"
        raise
    finally:
        if not server_task.done():
            server_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await server_task
        current = read_broker_state(config.state_file)
        if current is not None and current.owner_id == state.owner_id:
            write_broker_state(config.state_file, stopped_state(state, terminal_status))
        for signum in installed_signals:
            with suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(signum)
        _unlock(lock_handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detached Colab MCP broker")
    parser.add_argument("--host", default=DEFAULT_BROKER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_BROKER_PORT)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_BROKER_STATE_FILE)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_BROKER_LOCK_FILE)
    parser.add_argument("--factory", default=DEFAULT_BROKER_FACTORY)
    parser.add_argument(
        "--startup-timeout", type=float, default=BROKER_START_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--health-timeout", type=float, default=BROKER_HEALTH_TIMEOUT_SECONDS
    )
    parser.add_argument("--protocol-version", default=BROKER_PROTOCOL_VERSION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    callback = load_serve_callback(args.factory)
    if not inspect.iscoroutinefunction(callback):
        raise TypeError("Broker factory must be an async callable")
    config = BrokerLaunchConfig(
        host=args.host,
        port=args.port,
        state_file=args.state_file,
        lock_file=args.lock_file,
        factory=args.factory,
        startup_timeout=args.startup_timeout,
        health_timeout=args.health_timeout,
        protocol_version=args.protocol_version,
    )
    asyncio.run(run_broker_daemon(config, callback))


if __name__ == "__main__":
    main()
