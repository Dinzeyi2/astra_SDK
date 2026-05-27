"""
Fail-closed enforcement engine — CodeAstra SDK v2.0.0.

PRINCIPLE: When in doubt, block. Data must never reach an LLM if
CodeAstra cannot protect it. A 503 or timeout is always safer than
sending a real SSN, salary, or medical record to an AI model.

This module provides:
  - FailClosedGuard: context manager + decorator that wraps any protection call
  - apply_fail_closed(): monkey-patches BlindAgentMiddleware with hard stops
  - SAFE_ABORT_MESSAGE: what the agent sees when execution is aborted
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

from .exceptions import (
    FailClosedError,
    ExecutionAbortedError,
    TokenizationError,
)

log = logging.getLogger("codeastra.fail_closed")

F = TypeVar("F", bound=Callable)

# What the agent receives instead of real data when protection fails.
# Short, unambiguous, contains no sensitive information.
SAFE_ABORT_MESSAGE: str = (
    "[CODEASTRA:ABORT] Data protection failed. "
    "Execution blocked by fail-closed policy. "
    "No sensitive data was processed. "
    "Contact your administrator."
)

# Global kill-switch: once tripped, ALL agent calls are blocked until reset.
_GLOBAL_KILL: bool = False
_KILL_REASON: str = ""


def trip_global_kill(reason: str = "") -> None:
    """
    Globally block all agent executions — call when a catastrophic protection
    failure occurs. All subsequent middleware calls will raise ExecutionAbortedError.
    Reset with reset_global_kill() after investigating.
    """
    global _GLOBAL_KILL, _KILL_REASON
    _GLOBAL_KILL = True
    _KILL_REASON = reason or "Unspecified protection failure"
    log.critical("codeastra.fail_closed.KILL_SWITCH_TRIPPED reason=%s", _KILL_REASON)


def reset_global_kill() -> None:
    """Reset the global kill-switch after investigation."""
    global _GLOBAL_KILL, _KILL_REASON
    _GLOBAL_KILL = False
    _KILL_REASON = ""
    log.info("codeastra.fail_closed.kill_switch_reset")


def is_kill_active() -> bool:
    return _GLOBAL_KILL


class FailClosedGuard:
    """
    Context manager / decorator that enforces fail-closed on any operation.

    Usage:
        with FailClosedGuard("tokenize_patient_record"):
            tokens = client.tokenize({"ssn": "123-45-6789"})

        # As decorator:
        @FailClosedGuard.wrap("protect_salary")
        def get_salary():
            return client.tokenize({"salary": 95000})

    If the wrapped code raises ANY exception (including network errors,
    timeouts, or API errors), FailClosedGuard raises FailClosedError and
    the original exception is attached as __cause__.

    The real data is never forwarded to the caller on failure.
    """

    def __init__(
        self,
        label: str = "operation",
        *,
        reraise_as: type = FailClosedError,
        log_sensitive: bool = False,
    ):
        self.label = label
        self._reraise_as = reraise_as
        self._log_sensitive = log_sensitive
        self._t0: float = 0.0

    def __enter__(self):
        if _GLOBAL_KILL:
            raise ExecutionAbortedError(
                f"CodeAstra global kill-switch is active: {_KILL_REASON}. "
                "All agent executions are blocked."
            )
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False  # no error, proceed normally

        # If already a CodeAstra fail-closed error, don't re-wrap
        if isinstance(exc_val, FailClosedError):
            log.error(
                "codeastra.fail_closed.%s.already_blocked label=%s error=%s",
                self._reraise_as.__name__, self.label, str(exc_val)[:120],
            )
            return False  # let it propagate as-is

        # Wrap ANY other exception as a fail-closed error
        duration_ms = int((time.monotonic() - self._t0) * 1000)
        log.error(
            "codeastra.fail_closed.BLOCKED label=%s exc=%s duration_ms=%d",
            self.label, type(exc_val).__name__, duration_ms,
        )
        raise self._reraise_as(
            f"CodeAstra '{self.label}' failed after {duration_ms}ms — "
            f"data blocked by fail-closed policy. "
            f"Root cause: {type(exc_val).__name__}: {str(exc_val)[:200]}"
        ) from exc_val

    @classmethod
    def wrap(cls, label: str, reraise_as: type = FailClosedError):
        """Decorator version of FailClosedGuard."""
        def decorator(fn: F) -> F:
            if _is_async(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    with cls(label, reraise_as=reraise_as):
                        return await fn(*args, **kwargs)
                return async_wrapper  # type: ignore
            else:
                @functools.wraps(fn)
                def sync_wrapper(*args, **kwargs):
                    with cls(label, reraise_as=reraise_as):
                        return fn(*args, **kwargs)
                return sync_wrapper  # type: ignore
        return decorator


def _is_async(fn: Callable) -> bool:
    import asyncio, inspect
    return asyncio.iscoroutinefunction(fn) or inspect.iscoroutinefunction(fn)


def guard_tokenize(fn: F) -> F:
    """Wrap a tokenization method with fail-closed enforcement."""
    from .exceptions import TokenizationError
    return FailClosedGuard.wrap(f"tokenize:{fn.__name__}", reraise_as=TokenizationError)(fn)


def guard_smpc(fn: F) -> F:
    """Wrap an SMPC method with fail-closed enforcement."""
    from .exceptions import SmpcError
    return FailClosedGuard.wrap(f"smpc:{fn.__name__}", reraise_as=SmpcError)(fn)


def guard_fhe(fn: F) -> F:
    """Wrap an FHE method with fail-closed enforcement."""
    from .exceptions import HEError
    return FailClosedGuard.wrap(f"fhe:{fn.__name__}", reraise_as=HEError)(fn)


def guard_vault_compute(fn: F) -> F:
    """Wrap a vault compute method with fail-closed enforcement."""
    from .exceptions import VaultComputeError
    return FailClosedGuard.wrap(f"vault_compute:{fn.__name__}", reraise_as=VaultComputeError)(fn)


def apply_fail_closed_to_middleware(middleware_cls=None) -> None:
    """
    Monkey-patch BlindAgentMiddleware so that ANY unhandled exception
    in _blind_output or _ablind_output raises ExecutionAbortedError
    instead of silently returning the unprotected data.

    Called automatically from __init__.py on import.
    """
    if middleware_cls is None:
        try:
            from .middleware import BlindAgentMiddleware
            middleware_cls = BlindAgentMiddleware
        except ImportError:
            return

    _orig_blind = getattr(middleware_cls, "_blind_output", None)
    _orig_ablind = getattr(middleware_cls, "_ablind_output", None)

    if _orig_blind:
        @functools.wraps(_orig_blind)
        def _safe_blind_output(self, output: Any) -> Any:
            if _GLOBAL_KILL:
                raise ExecutionAbortedError(
                    f"Global kill-switch active: {_KILL_REASON}"
                )
            try:
                return _orig_blind(self, output)
            except FailClosedError:
                raise
            except Exception as exc:
                if getattr(self, "_fail_closed", True):
                    raise ExecutionAbortedError(
                        f"Output protection failed — execution aborted. "
                        f"Data NOT forwarded to agent. Error: {exc}"
                    ) from exc
                log.warning(
                    "codeastra.fail_closed.SKIPPED (fail_closed=False) error=%s", exc
                )
                return output  # only if user explicitly set fail_closed=False

        middleware_cls._blind_output = _safe_blind_output

    if _orig_ablind:
        @functools.wraps(_orig_ablind)
        async def _safe_ablind_output(self, output: Any) -> Any:
            if _GLOBAL_KILL:
                raise ExecutionAbortedError(
                    f"Global kill-switch active: {_KILL_REASON}"
                )
            try:
                return await _orig_ablind(self, output)
            except FailClosedError:
                raise
            except Exception as exc:
                if getattr(self, "_fail_closed", True):
                    raise ExecutionAbortedError(
                        f"Async output protection failed — execution aborted. "
                        f"Error: {exc}"
                    ) from exc
                return output

        middleware_cls._ablind_output = _safe_ablind_output
