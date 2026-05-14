import pytest
from app.core.retry import retry_with_backoff


def test_success_first_try():
    calls = []
    @retry_with_backoff(max_attempts=3, base_delay=0.01)
    def f():
        calls.append(1)
        return "ok"
    assert f() == "ok"
    assert len(calls) == 1


def test_retry_then_success():
    calls = []
    @retry_with_backoff(max_attempts=3, base_delay=0.01)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("flaky")
        return "ok"
    assert f() == "ok"
    assert len(calls) == 3


def test_max_attempts_then_raise():
    @retry_with_backoff(max_attempts=2, base_delay=0.01)
    def f():
        raise RuntimeError("nope")
    with pytest.raises(RuntimeError, match="nope"):
        f()
