"""Tests for llm-tool-timeout."""

from __future__ import annotations

import asyncio
import time

import pytest

from llm_tool_timeout import TimeoutRegistry, ToolTimeoutError, with_timeout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fast_fn(x: int) -> int:
    return x * 2


def slow_fn(duration: float = 5.0) -> str:
    time.sleep(duration)
    return "done"


async def async_fast(x: int) -> int:
    await asyncio.sleep(0)
    return x + 1


async def async_slow(duration: float = 5.0) -> str:
    await asyncio.sleep(duration)
    return "done"


# ---------------------------------------------------------------------------
# with_timeout — sync
# ---------------------------------------------------------------------------


def test_sync_fast_fn_passes():
    wrapped = with_timeout(2.0)(fast_fn)
    assert wrapped(3) == 6


def test_sync_timeout_raises():
    wrapped = with_timeout(0.05)(slow_fn)
    with pytest.raises(ToolTimeoutError) as exc_info:
        wrapped(0.5)
    err = exc_info.value
    assert err.tool_name == "slow_fn"
    assert err.limit_seconds == 0.05
    assert err.elapsed_seconds >= 0.05


def test_sync_timeout_error_message():
    wrapped = with_timeout(0.05)(slow_fn)
    with pytest.raises(ToolTimeoutError, match="slow_fn"):
        wrapped(0.5)


def test_sync_custom_tool_name():
    wrapped = with_timeout(0.05, tool_name="my_tool")(slow_fn)
    with pytest.raises(ToolTimeoutError) as exc_info:
        wrapped(0.5)
    assert exc_info.value.tool_name == "my_tool"


def test_sync_preserves_return_value():
    @with_timeout(1.0)
    def add(a: int, b: int) -> int:
        return a + b

    assert add(3, 4) == 7


def test_sync_propagates_exception():
    @with_timeout(1.0)
    def explode() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        explode()


def test_sync_preserves_functools_metadata():
    @with_timeout(1.0)
    def my_tool(x: int) -> int:
        """Does something."""
        return x

    assert my_tool.__name__ == "my_tool"
    assert my_tool.__doc__ == "Does something."


def test_sync_kwargs_passed():
    @with_timeout(1.0)
    def greet(name: str = "world") -> str:
        return f"hello {name}"

    assert greet(name="alice") == "hello alice"


# ---------------------------------------------------------------------------
# with_timeout — async
# ---------------------------------------------------------------------------


def test_async_fast_passes():
    wrapped = with_timeout(2.0)(async_fast)

    async def run():
        return await wrapped(5)

    assert asyncio.run(run()) == 6


def test_async_timeout_raises():
    wrapped = with_timeout(0.05)(async_slow)

    async def run():
        await wrapped(5.0)

    with pytest.raises(ToolTimeoutError) as exc_info:
        asyncio.run(run())
    err = exc_info.value
    assert err.tool_name == "async_slow"
    assert err.limit_seconds == 0.05


def test_async_custom_tool_name():
    wrapped = with_timeout(0.05, tool_name="fetch")(async_slow)

    async def run():
        await wrapped(5.0)

    with pytest.raises(ToolTimeoutError) as exc_info:
        asyncio.run(run())
    assert exc_info.value.tool_name == "fetch"


def test_async_propagates_exception():
    @with_timeout(1.0)
    async def fail() -> None:
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError, match="async boom"):
        asyncio.run(fail())


def test_async_preserves_functools_metadata():
    @with_timeout(1.0)
    async def async_tool(x: int) -> int:
        """Async doc."""
        return x

    assert async_tool.__name__ == "async_tool"
    assert async_tool.__doc__ == "Async doc."


def test_async_kwargs_passed():
    @with_timeout(1.0)
    async def async_greet(name: str = "world") -> str:
        return f"hi {name}"

    assert asyncio.run(async_greet(name="bob")) == "hi bob"


# ---------------------------------------------------------------------------
# with_timeout — validation
# ---------------------------------------------------------------------------


def test_zero_seconds_raises():
    with pytest.raises(ValueError, match="seconds must be > 0"):
        with_timeout(0.0)


def test_negative_seconds_raises():
    with pytest.raises(ValueError, match="seconds must be > 0"):
        with_timeout(-1.0)


# ---------------------------------------------------------------------------
# ToolTimeoutError attributes
# ---------------------------------------------------------------------------


def test_error_has_all_attributes():
    err = ToolTimeoutError("my_fn", 5.0, 5.12)
    assert err.tool_name == "my_fn"
    assert err.limit_seconds == 5.0
    assert err.elapsed_seconds == 5.12


def test_error_str_includes_tool_name():
    err = ToolTimeoutError("web_search", 3.0, 3.5)
    assert "web_search" in str(err)
    assert "3.0" in str(err)


def test_error_is_exception():
    err = ToolTimeoutError("fn", 1.0, 1.1)
    assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# TimeoutRegistry
# ---------------------------------------------------------------------------


def test_registry_default_timeout():
    reg = TimeoutRegistry(default_seconds=1.0)
    assert reg.get_limit("anything") == 1.0


def test_registry_per_tool_override():
    reg = TimeoutRegistry(default_seconds=10.0)
    reg.set("fast_tool", 2.0)
    assert reg.get_limit("fast_tool") == 2.0
    assert reg.get_limit("other_tool") == 10.0


def test_registry_set_returns_self():
    reg = TimeoutRegistry()
    result = reg.set("t", 5.0)
    assert result is reg


def test_registry_chaining():
    reg = TimeoutRegistry(default_seconds=30.0)
    reg.set("a", 5.0).set("b", 10.0).set("c", 15.0)
    assert reg.get_limit("a") == 5.0
    assert reg.get_limit("b") == 10.0
    assert reg.get_limit("c") == 15.0


def test_registry_wrap_fast_fn():
    reg = TimeoutRegistry(default_seconds=2.0)
    wrapped = reg.wrap(fast_fn)
    assert wrapped(4) == 8


def test_registry_wrap_uses_fn_name():
    reg = TimeoutRegistry(default_seconds=0.05)
    wrapped = reg.wrap(slow_fn)
    with pytest.raises(ToolTimeoutError) as exc_info:
        wrapped(5.0)
    assert exc_info.value.tool_name == "slow_fn"


def test_registry_wrap_tool_name_override():
    reg = TimeoutRegistry(default_seconds=0.05)
    wrapped = reg.wrap(slow_fn, tool_name="scraper")
    with pytest.raises(ToolTimeoutError) as exc_info:
        wrapped(5.0)
    assert exc_info.value.tool_name == "scraper"


def test_registry_wrap_applies_per_tool_limit():
    reg = TimeoutRegistry(default_seconds=5.0)
    reg.set("slow_fn", 0.05)
    wrapped = reg.wrap(slow_fn)
    with pytest.raises(ToolTimeoutError):
        wrapped(5.0)


def test_registry_wrap_all_returns_dict():
    reg = TimeoutRegistry(default_seconds=2.0)
    result = reg.wrap_all({"double": fast_fn})
    assert "double" in result
    assert result["double"](5) == 10


def test_registry_wrap_all_multiple():
    def add_one(x: int) -> int:
        return x + 1

    def mul_two(x: int) -> int:
        return x * 2

    reg = TimeoutRegistry(default_seconds=1.0)
    wrapped = reg.wrap_all({"add": add_one, "mul": mul_two})
    assert wrapped["add"](3) == 4
    assert wrapped["mul"](3) == 6


def test_registry_wrap_all_uses_dict_key_as_name():
    reg = TimeoutRegistry(default_seconds=0.05)
    wrapped = reg.wrap_all({"my_slow": slow_fn})
    with pytest.raises(ToolTimeoutError) as exc_info:
        wrapped["my_slow"](5.0)
    assert exc_info.value.tool_name == "my_slow"


def test_registry_default_invalid():
    with pytest.raises(ValueError, match="default_seconds must be > 0"):
        TimeoutRegistry(default_seconds=0.0)


def test_registry_set_invalid():
    reg = TimeoutRegistry()
    with pytest.raises(ValueError, match="seconds must be > 0"):
        reg.set("t", -1.0)


# ---------------------------------------------------------------------------
# Async in registry
# ---------------------------------------------------------------------------


def test_registry_wrap_async():
    reg = TimeoutRegistry(default_seconds=2.0)
    wrapped = reg.wrap(async_fast)

    async def run():
        return await wrapped(7)

    assert asyncio.run(run()) == 8


def test_registry_wrap_async_timeout():
    reg = TimeoutRegistry(default_seconds=0.05)
    wrapped = reg.wrap(async_slow)

    async def run():
        await wrapped(5.0)

    with pytest.raises(ToolTimeoutError):
        asyncio.run(run())
