"""Circuit breaker tests — open/half-open/closed state transitions."""

import time
import pytest
from governance.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_under_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_requests(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.allow_request() is False

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        cb.allow_request()  # transitions to HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        cb.allow_request()  # transitions to HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_allows_only_one_probe(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.allow_request() is True   # first probe allowed
        assert cb.allow_request() is False  # second caller blocked

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # count reset after success
