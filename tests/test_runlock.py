"""Tests for the cross-process run lock."""

import os

import pytest

import runlock
from runlock import RunLockBusy, current_run_pid, run_lock


def test_lock_records_pid_and_clears_it():
    """While held, the lock file carries the live PID; after release it is empty."""
    assert current_run_pid() is None
    with run_lock():
        assert current_run_pid() == os.getpid()
    assert current_run_pid() is None


def test_second_acquisition_is_rejected():
    """A second acquirer (a distinct open file description) gets RunLockBusy."""
    with run_lock():
        with pytest.raises(RunLockBusy):
            with run_lock():
                pass


def test_stale_pid_is_reported_as_not_running():
    """current_run_pid returns None when the recorded PID is dead."""
    runlock.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    runlock.LOCK_PATH.write_text("999999999", encoding="utf-8")  # not a live PID
    assert current_run_pid() is None
