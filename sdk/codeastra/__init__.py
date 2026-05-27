"""
CodeAstra SDK v2.0.0 — Enterprise Privacy for AI Agents.

Two lines.  Any agent becomes blind.  Fail-closed by default.

Quick start:
    from codeastra import CodeAstraClient, BlindAgentMiddleware

    client = CodeAstraClient(api_key="sk-guard-xxx")

    # Tokenize sensitive fields
    tokens = client.tokenize({"ssn": "123-45-6789", "salary": 95000})

    # SMPC — server never sees the value
    bundle = client.smpc_store(95000.0, label="salary")
    result = client.smpc_compute("percentage_of", {"value": bundle.bundle_id}, {"rate": 7.5})
    print(result.result)   # "7125.00"

    # FHE — compute on ciphertext, secret key never leaves client
    result = client.fhe_full_compute(95000.0, "percentage_of", {"rate": 7.5})
    print(result)          # 7125.0

    # Vault Compute — 155 operations on tokenized values
    result = client.vault_compute("percentage_of", {"value": tokens["salary"]}, {"rate": 7.5})
    print(result.result)   # "$7,125.00"

    # Blind agent — two lines makes any agent blind
    agent = BlindAgentMiddleware(your_langchain_agent, api_key="sk-guard-xxx")
    result = agent.invoke({"input": "Schedule patient appointment"})

Fail-closed guarantee:
    If CodeAstra cannot protect data (network error, auth failure, timeout),
    execution is ABORTED.  Data never reaches the LLM.  This is the default.
    Use fail_closed=False ONLY if you explicitly accept the risk.
"""
from .client import CodeAstraClient
from .middleware import BlindAgentMiddleware
from .wrappers import blind_tool, BlindCrewAIAgent, BlindAutoGPTAgent

from .exceptions import (
    # Base
    CodeAstraError,
    # Fail-closed hard stops
    FailClosedError,
    TokenizationError,
    VaultError,
    SmpcError,
    HEError,
    HENotInstalled,
    VaultComputeError,
    ExecutionAbortedError,
    # Non-fatal
    AuthError,
    RateLimitError,
    NetworkError,
    ConfigurationError,
    SMPCNotReady,
)

from .types import (
    FHESession,
    FHEResult,
    SMPCBundle,
    SMPCResult,
    VaultComputeResult,
    VaultCohortResult,
    ThinkingToken,
    SMPCOperation,
    FHEOperation,
)

from .fhe import he_available

from .fail_closed import (
    FailClosedGuard,
    trip_global_kill,
    reset_global_kill,
    is_kill_active,
    guard_tokenize,
    guard_smpc,
    guard_fhe,
    guard_vault_compute,
    SAFE_ABORT_MESSAGE,
)

__version__ = "2.1.0"

# Apply fail-closed enforcement to BlindAgentMiddleware on import.
# This wraps _blind_output and _ablind_output so that ANY unhandled exception
# raises ExecutionAbortedError instead of silently returning unprotected data.
try:
    from .fail_closed import apply_fail_closed_to_middleware
    apply_fail_closed_to_middleware()
except Exception:
    pass  # Middleware import failure should not break the whole SDK

__all__ = [
    # Client
    "CodeAstraClient",
    # Middleware
    "BlindAgentMiddleware",
    "blind_tool",
    "BlindCrewAIAgent",
    "BlindAutoGPTAgent",
    # Exceptions — fail-closed hard stops
    "CodeAstraError",
    "FailClosedError",
    "TokenizationError",
    "VaultError",
    "SmpcError",
    "HEError",
    "HENotInstalled",
    "VaultComputeError",
    "ExecutionAbortedError",
    # Exceptions — non-fatal
    "AuthError",
    "RateLimitError",
    "NetworkError",
    "ConfigurationError",
    "SMPCNotReady",
    # Return types
    "FHESession",
    "FHEResult",
    "SMPCBundle",
    "SMPCResult",
    "VaultComputeResult",
    "VaultCohortResult",
    "ThinkingToken",
    "SMPCOperation",
    "FHEOperation",
    # FHE helpers
    "he_available",
    # Fail-closed utilities
    "FailClosedGuard",
    "trip_global_kill",
    "reset_global_kill",
    "is_kill_active",
    "guard_tokenize",
    "guard_smpc",
    "guard_fhe",
    "guard_vault_compute",
    "SAFE_ABORT_MESSAGE",
    # Version
    "__version__",
]
