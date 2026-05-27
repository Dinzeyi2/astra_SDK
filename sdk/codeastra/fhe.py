"""
FHE (Fully Homomorphic Encryption) client-side helpers — CodeAstra SDK v2.0.0.

Handles local encrypt / decrypt so users never need to touch TenSEAL directly.
The secret key NEVER leaves the client. The server only sees ciphertext.

Protocol:
  1. client.fhe_setup()          → FHESession (public + secret context)
  2. client.fhe_encrypt(value)   → base64 ciphertext  (local, no network)
  3. client.fhe_compute(...)     → base64 result       (server computes on ciphertext)
  4. client.fhe_decrypt(result)  → float               (local, no network)

Or one-shot:
  client.fhe_full_compute(value, "percentage_of", {"rate": 7.0}, session)
  → 297348.24  (server saw zero plaintext)
"""
from __future__ import annotations

import base64
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import FHESession

_TS_AVAILABLE: bool | None = None  # None = not checked yet


def _tenseal():
    """Import TenSEAL with a clear error if missing."""
    global _TS_AVAILABLE
    try:
        import tenseal as ts
        _TS_AVAILABLE = True
        return ts
    except ImportError:
        _TS_AVAILABLE = False
        from .exceptions import HENotInstalled
        raise HENotInstalled()


def he_available() -> bool:
    """Return True if TenSEAL is installed and usable."""
    global _TS_AVAILABLE
    if _TS_AVAILABLE is None:
        try:
            import tenseal  # noqa: F401
            _TS_AVAILABLE = True
        except ImportError:
            _TS_AVAILABLE = False
    return _TS_AVAILABLE  # type: ignore


def encrypt_value(value: float, session: "FHESession") -> str:
    """
    Encrypt a single float value locally using the session's public context.

    Returns a base64-encoded ciphertext string ready to send to the server.
    TenSEAL (SEAL CKKS) is used locally — the secret key never leaves this process.
    """
    ts = _tenseal()
    ctx = ts.context_from(base64.b64decode(session.public_context_b64))
    vec = ts.ckks_vector(ctx, [value])
    return base64.b64encode(vec.serialize()).decode()


def encrypt_batch(values: list[float], session: "FHESession") -> str:
    """
    Encrypt a batch of floats into a single CKKS ciphertext (SIMD).
    Max batch size = session.slots (typically 4096 or 8192).
    Pad with zeros if needed before calling.
    """
    ts = _tenseal()
    ctx = ts.context_from(base64.b64decode(session.public_context_b64))
    vec = ts.ckks_vector(ctx, values)
    return base64.b64encode(vec.serialize()).decode()


def decrypt_value(encrypted_b64: str, session: "FHESession") -> float:
    """
    Decrypt a single-slot CKKS ciphertext returned by the server.
    Returns the plaintext float. Uses the secret context locally.
    """
    ts = _tenseal()
    ctx = ts.context_from(base64.b64decode(session.secret_context_b64))
    vec = ts.lazy_ckks_vector_from(base64.b64decode(encrypted_b64))
    vec.link_context(ctx)
    return vec.decrypt()[0]


def decrypt_batch(encrypted_b64: str, session: "FHESession") -> list[float]:
    """
    Decrypt a multi-slot CKKS ciphertext. Returns all decrypted values.
    """
    ts = _tenseal()
    ctx = ts.context_from(base64.b64decode(session.secret_context_b64))
    vec = ts.lazy_ckks_vector_from(base64.b64decode(encrypted_b64))
    vec.link_context(ctx)
    return vec.decrypt()
