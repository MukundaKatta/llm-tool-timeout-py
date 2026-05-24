"""Timeout wrappers for LLM agent tool functions.

Both sync and async variants are supported via a single ``with_timeout``
decorator factory.

Sync implementation note
------------------------
Python threads cannot be forcibly killed, so the underlying sync function
continues running in a daemon thread after a timeout is raised.  The
*caller* receives ``ToolTimeoutError`` immediately; the daemon thread is
cleaned up when the process exits.  For agent tool use this is almost
always acceptable — the model will not see the result and can move on.

Async implementation
--------------------
Uses ``asyncio.wait_for`` which cancels the coroutine cleanly via the
standard cancellation protocol.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
import time
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ToolTimeoutError(Exception):
    """Raised when a tool call exceeds its time limit.

    Attributes:
        tool_name: Name of the tool that timed out.
        limit_seconds: The configured timeout in seconds.
        elapsed_seconds: Actual elapsed time before the error was raised.
    """

    def __init__(self, tool_name: str, limit_seconds: float, elapsed_seconds: float) -> None:
        self.tool_name = tool_name
        self.limit_seconds = limit_seconds
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f"Tool {tool_name!r} timed out after {elapsed_seconds:.3f}s (limit: {limit_seconds}s)"
        )


# ---------------------------------------------------------------------------
# Core decorator
# ---------------------------------------------------------------------------


def with_timeout(
    seconds: float,
    *,
    tool_name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory that enforces a wall-clock timeout on a tool function.

    Works with both regular functions and ``async def`` coroutines.  The
    decorated function raises :exc:`ToolTimeoutError` if the call does not
    complete within *seconds*.

    Args:
        seconds: Maximum allowed execution time in seconds.
        tool_name: Override for the tool name used in the error message.
            Defaults to the wrapped function's ``__name__``.

    Returns:
        A decorator that wraps a callable with timeout enforcement.

    Example::

        @with_timeout(5.0)
        def lookup(query: str) -> str:
            return slow_search(query)

        @with_timeout(10.0)
        async def fetch(url: str) -> str:
            return await http_get(url)
    """
    if seconds <= 0:
        raise ValueError(f"seconds must be > 0, got {seconds!r}")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = tool_name or fn.__name__

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.monotonic()
                try:
                    return await asyncio.wait_for(fn(*args, **kwargs), timeout=seconds)
                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - start
                    raise ToolTimeoutError(name, seconds, elapsed) from None

            return async_wrapper

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                result_box: list[Any] = []
                error_box: list[BaseException] = []

                def _run() -> None:
                    try:
                        result_box.append(fn(*args, **kwargs))
                    except BaseException as exc:  # noqa: BLE001
                        error_box.append(exc)

                thread = threading.Thread(target=_run, daemon=True)
                start = time.monotonic()
                thread.start()
                thread.join(timeout=seconds)
                elapsed = time.monotonic() - start

                if thread.is_alive():
                    # Thread is still running; raise timeout to caller.
                    # The daemon thread will be reaped when the process exits.
                    raise ToolTimeoutError(name, seconds, elapsed)

                if error_box:
                    raise error_box[0]

                return result_box[0]

            return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TimeoutRegistry:
    """Apply per-tool or default timeouts to many functions at once.

    Example::

        reg = TimeoutRegistry(default_seconds=30.0)
        reg.set("slow_scraper", 120.0)
        reg.set("quick_lookup", 5.0)

        wrapped = reg.wrap_all(tool_dict)
    """

    def __init__(self, default_seconds: float = 30.0) -> None:
        if default_seconds <= 0:
            raise ValueError(f"default_seconds must be > 0, got {default_seconds!r}")
        self._default = default_seconds
        self._per_tool: dict[str, float] = {}

    def set(self, tool_name: str, seconds: float) -> TimeoutRegistry:
        """Set a per-tool timeout override.

        Args:
            tool_name: Name of the tool function.
            seconds: Timeout in seconds for this tool.

        Returns:
            ``self`` for chaining.
        """
        if seconds <= 0:
            raise ValueError(f"seconds must be > 0, got {seconds!r}")
        self._per_tool[tool_name] = seconds
        return self

    def get_limit(self, tool_name: str) -> float:
        """Return the effective timeout for *tool_name* (default if not set)."""
        return self._per_tool.get(tool_name, self._default)

    def wrap(
        self,
        fn: Callable[..., Any],
        *,
        tool_name: str | None = None,
    ) -> Callable[..., Any]:
        """Wrap a single function with the registry's timeout for its name.

        Args:
            fn: Callable to wrap.
            tool_name: Override for the lookup key and error label.
                Defaults to ``fn.__name__``.

        Returns:
            Timeout-wrapped callable.
        """
        name = tool_name or fn.__name__
        limit = self._per_tool.get(name, self._default)
        return with_timeout(limit, tool_name=name)(fn)

    def wrap_all(
        self,
        tools: dict[str, Callable[..., Any]],
    ) -> dict[str, Callable[..., Any]]:
        """Wrap every callable in *tools* using registry timeouts.

        Args:
            tools: Mapping of tool name → callable.

        Returns:
            New dict with the same keys and timeout-wrapped callables.
        """
        return {name: self.wrap(fn, tool_name=name) for name, fn in tools.items()}
