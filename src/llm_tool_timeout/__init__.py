"""llm-tool-timeout: sync and async timeout wrappers for LLM agent tools.

Quick start::

    from llm_tool_timeout import with_timeout, ToolTimeoutError

    @with_timeout(5.0)
    def search_web(query: str) -> str:
        ...  # may hang

    @with_timeout(10.0)
    async def fetch_doc(url: str) -> str:
        ...  # awaitable, uses asyncio.wait_for internally

Registry for bulk configuration::

    from llm_tool_timeout import TimeoutRegistry

    reg = TimeoutRegistry(default_seconds=30.0)
    reg.set("slow_tool", 60.0)
    wrapped = reg.wrap_all({"search": search_fn, "slow_tool": slow_fn})
"""

from .core import TimeoutRegistry, ToolTimeoutError, with_timeout

__all__ = ["ToolTimeoutError", "TimeoutRegistry", "with_timeout"]
__version__ = "0.1.0"
