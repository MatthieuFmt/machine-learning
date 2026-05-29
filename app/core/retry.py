"""Retry avec backoff exponentiel. Obligatoire pour toute I/O (cf. Règle 11)."""
from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            "retry",
                            extra={"context": {
                                "fn": fn.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "error": str(e),
                                "next_delay_s": delay,
                            }},
                        )
                        time.sleep(delay)
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator
