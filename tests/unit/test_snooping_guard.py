import json
import pytest
from app.testing.snooping_guard import (
    is_locked, read_oos, lock, check_unlocked, TestSetSnoopingError, LOCK_PATH
)


def test_lifecycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not is_locked()
    check_unlocked()  # no-op
    read_oos("07", "H06", sharpe=1.2, n_trades=40)
    assert LOCK_PATH.exists()
    state = json.loads(LOCK_PATH.read_text())
    assert state["n_reads"] == 1
    lock("18")
    assert is_locked()
    with pytest.raises(TestSetSnoopingError):
        check_unlocked()
