# llm-tool-timeout

Sync and async timeout enforcement for LLM agent tool functions. Zero dependencies.

```python
from llm_tool_timeout import with_timeout, ToolTimeoutError, TimeoutRegistry

@with_timeout(5.0)
def search_web(query: str) -> str:
    ...  # may hang

@with_timeout(10.0)
async def fetch_doc(url: str) -> str:
    ...  # cancelled cleanly via asyncio

# Or use a registry for bulk config
reg = TimeoutRegistry(default_seconds=30.0)
reg.set("slow_scraper", 120.0)
wrapped = reg.wrap_all({"search": search_web, "slow_scraper": slow_fn})
```

## Install

```bash
pip install llm-tool-timeout
```

## How it works

- **Async functions**: `asyncio.wait_for` — coroutine is cancelled cleanly.
- **Sync functions**: daemon thread — caller gets `ToolTimeoutError` immediately; the thread finishes in the background (can't be forcibly killed in Python).

## API

### `with_timeout(seconds, *, tool_name=None)`

Decorator factory. Works on both `def` and `async def`.

### `ToolTimeoutError`

Raised on timeout. Has `.tool_name`, `.limit_seconds`, `.elapsed_seconds`.

### `TimeoutRegistry`

```python
reg = TimeoutRegistry(default_seconds=30.0)
reg.set("tool_name", 5.0)      # per-tool override, chainable
reg.get_limit("tool_name")     # → 5.0
reg.wrap(fn)                   # → wrapped callable
reg.wrap_all({"name": fn})     # → dict of wrapped callables
```

## License

MIT
