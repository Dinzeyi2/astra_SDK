"""
Return-type dataclasses for CodeAstra SDK v2.0.0.
All fields match what the server returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ── FHE ───────────────────────────────────────────────────────────────────────

@dataclass
class FHESession:
    """
    A live FHE keypair session.

    secret_context_b64 is the private key — store it safely and never send
    it to the server. public_context_b64 is safe to send for computation.
    """
    key_id: str
    public_context_b64: str
    secret_context_b64: str
    preset: str = "standard"
    slots: int = 8192
    max_depth: int = 4

    def public_only(self) -> "FHESession":
        """Return a copy with secret key stripped — safe to log or serialize."""
        return FHESession(
            key_id=self.key_id,
            public_context_b64=self.public_context_b64,
            secret_context_b64="<redacted>",
            preset=self.preset,
            slots=self.slots,
            max_depth=self.max_depth,
        )


@dataclass
class FHEResult:
    """Result of a server-side FHE computation."""
    encrypted_result_b64: str       # still encrypted — decrypt with your secret key
    operation: str
    key_id: str
    duration_ms: int = 0
    server_saw_plaintext: bool = False  # always False for HE

    def decrypt(self, session: "FHESession") -> float:
        """Convenience: decrypt this result in one call."""
        import base64
        try:
            import tenseal as ts
        except ImportError:
            from .exceptions import HENotInstalled
            raise HENotInstalled()
        ctx = ts.context_from(base64.b64decode(session.secret_context_b64))
        enc = ts.lazy_ckks_vector_from(base64.b64decode(self.encrypted_result_b64))
        enc.link_context(ctx)
        return enc.decrypt()[0]

    def decrypt_batch(self, session: "FHESession") -> list[float]:
        """Decrypt batch result."""
        import base64
        try:
            import tenseal as ts
        except ImportError:
            from .exceptions import HENotInstalled
            raise HENotInstalled()
        ctx = ts.context_from(base64.b64decode(session.secret_context_b64))
        enc = ts.lazy_ckks_vector_from(base64.b64decode(self.encrypted_result_b64))
        enc.link_context(ctx)
        return enc.decrypt()


# ── SMPC ──────────────────────────────────────────────────────────────────────

@dataclass
class SMPCBundle:
    """A stored SMPC share bundle — use bundle_id in smpc_compute()."""
    bundle_id: str
    n_parties: int = 3
    threshold: int = 2
    label: str = ""
    party_mode: str = "virtual"
    expires_in_seconds: int = 7200


@dataclass
class SMPCResult:
    """Result of an SMPC computation."""
    operation: str
    result: str                         # formatted result (e.g. "$297,348.24")
    result_raw: Optional[float]         # raw numeric result (None if wiped)
    protocol: str                       # "linear_one", "linear_two", "mul_two", "comparison"
    party_mode: str                     # "virtual" or "distributed"
    n_parties: int
    threshold: int
    wiped: bool                         # shares wiped after computation
    bundle_ids_consumed: list[str]
    duration_ms: int = 0
    audit_id: str = ""
    inputs_seen_by_compute_node: bool = True  # True in virtual mode
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# ── Vault Compute ─────────────────────────────────────────────────────────────

@dataclass
class VaultComputeResult:
    """Result of a vault compute operation."""
    operation: str
    result: str                         # formatted result
    result_raw: Optional[float]
    tokens_consumed: list[str]
    wiped: bool
    duration_ms: int = 0
    error: Optional[str] = None
    server_saw_plaintext: bool = True   # vault compute reconstructs transiently

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class VaultCohortResult:
    """Result of a vault cohort computation (aggregate over many tokens)."""
    operation: str
    result: str
    result_raw: Optional[float]
    token_count: int
    dp_applied: bool = False
    dp_epsilon: Optional[float] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# ── Token info ────────────────────────────────────────────────────────────────

@dataclass
class ThinkingToken:
    token_id: str
    data_type: str
    facts: dict
    cohort_id: Optional[str] = None
    signal_conditions: list = field(default_factory=list)
    ttl_hours: int = 720


@dataclass
class SMPCOperation:
    name: str
    description: str
    category: str
    protocol: str
    protocol_security: str
    required_tokens: list[str]
    required_plaintext: list[str]
    output_format: str
    example: dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class FHEOperation:
    name: str
    description: str
    category: str
    encrypted_inputs: list[str]
    plaintext_params: list[str]
    depth: int
    batched: bool = False
    example: dict = field(default_factory=dict)
