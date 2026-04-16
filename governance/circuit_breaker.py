"""
Circuit breaker for LLM inference calls.

Prevents cascading failures when Ollama/vLLM is down or overloaded.
Three states:
  CLOSED  — requests pass through normally
  OPEN    — requests rejected immediately (return fallback)
  HALF_OPEN — one probe request allowed to test recovery
"""

import time
from enum import Enum

from loguru import logger


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    def allow_request(self) -> bool:
        current = self.state
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.HALF_OPEN:
            self._state = CircuitState.HALF_OPEN
            return True
        return False  # OPEN

    def record_success(self):
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        logger.debug("Circuit breaker: closed (success)")

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker: OPEN after {self._failure_count} failures. "
                f"Recovery in {self.recovery_timeout}s."
            )
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker: re-opened (half-open probe failed)")
