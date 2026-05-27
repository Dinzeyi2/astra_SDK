"""
CodeAstra exception hierarchy.

All FailClosedError subclasses are HARD STOPS — they abort execution
and must never be silently caught. Data never reaches the LLM.
"""
from __future__ import annotations


class CodeAstraError(Exception):
    """Base for all CodeAstra exceptions."""


# ── Fail-closed (hard stops) ──────────────────────────────────────────────────

class FailClosedError(CodeAstraError):
    """
    CodeAstra protection failed — execution aborted.

    This is a HARD STOP. When raised, data has been blocked from reaching
    the LLM. Never silence this exception without understanding what it means.

    In fail_closed=True mode (default), any protection failure raises this.
    The agent framework receives an abort signal, not the real data.
    """


class TokenizationError(FailClosedError):
    """Tokenization of sensitive data failed — agent execution blocked."""


class VaultError(FailClosedError):
    """Vault operation failed — execution aborted."""


class SmpcError(FailClosedError):
    """SMPC operation failed — data never reconstructed, execution aborted."""


class HEError(FailClosedError):
    """Homomorphic encryption operation failed — execution aborted."""


class VaultComputeError(FailClosedError):
    """Vault compute operation failed — execution aborted."""


class ExecutionAbortedError(FailClosedError):
    """
    Agent execution forcibly aborted by CodeAstra.

    Raised when the middleware detects that continuing would expose
    real sensitive data to the LLM or external systems.
    """


# ── Non-fatal errors ──────────────────────────────────────────────────────────

class AuthError(CodeAstraError):
    """API key missing, invalid, or revoked."""


class RateLimitError(CodeAstraError):
    """API rate limit exceeded. Back off and retry."""
    def __init__(self, msg: str = "", retry_after: float = 60.0):
        super().__init__(msg or f"Rate limited. Retry after {retry_after}s.")
        self.retry_after = retry_after


class NetworkError(CodeAstraError):
    """Network connectivity error reaching the CodeAstra API."""


class ConfigurationError(CodeAstraError):
    """SDK misconfigured — check API key, base_url, and env vars."""


class SMPCNotReady(CodeAstraError):
    """SMPC store not found or expired. Store value before computing."""


class HENotInstalled(HEError):
    """
    TenSEAL not installed. Client-side HE requires it.

    Install with:
        pip install codeastra[fhe]
        # or
        pip install tenseal>=0.3.16
    """
