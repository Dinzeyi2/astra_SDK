"""
CodeAstraClient v2.0.1 — enterprise-grade privacy SDK.

Bug fixes vs v2.0.0:
  - FHE paths corrected:  /fhe/* → /executor/compute/fhe/*
  - Vault Compute paths corrected: /vault/compute → /executor/compute,
    /vault/cohort/compute → /executor/compute/cohort
  - FHE compute now sends required public_context_b64 field
  - SMPC store ttl converted hours→seconds (server field is ttl_seconds)
  - vault_cohort_compute signature fixed: token_list (list) + epsilon, not dict + dp_epsilon

New in v2.0.1 — 110+ missing enterprise endpoints added:
  - Auth:       login, me
  - Workspace:  create, invite, keys, members, usage
  - Sessions:   CRUD, drift, protect, cert lifecycle
  - HITL:       list, get, decide (human-in-the-loop)
  - Guardrails: topics, grounding, scan-output, semantic-topics, events, stats
  - Proxy:      chat v2 / v3 (LLM pass-through with guardrails)
  - Policies:   CRUD + synthesize (AI-generated policies)
  - Agents:     register, revoke, honey-tools
  - Platform agents: full CRUD, memory, channels, skills, workspace, schedules
  - Audit:      export (JSON/CSV), stats, timeseries, by-tool, shadow
  - Alerts:     list, resolve
  - Webhooks:   CRUD
  - Vault wipe: token / session / cohort / proof / stats
  - Rate limits: CRUD
  - Security:   injection monitoring, Falco, Trivy, SOC2
  - Compliance: reports, evidence, stack status, policy generation
  - Legal:      BAA generate / status
  - TEE:        status
  - Recipes:    CRUD + test + templates
  - Badge:      SVG, verify, impressions
  - FHE:        key management (list / get / delete), batch compute, status
  - Vault compute: audit, status, operations list/get
  - SMPC:       parties, audit, status, operation detail
  - Platform:   stats, skills, security stats, on-prem agents
  - Metrics:    Prometheus export
"""
from __future__ import annotations

import re
import os
import json
import hmac
import time
import socket
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Any, Optional, List, Dict

import httpx

from .exceptions import (
    AuthError,
    RateLimitError,
    NetworkError,
    ConfigurationError,
)
from .types import (
    FHESession,
    FHEResult,
    SMPCBundle,
    SMPCResult,
    VaultComputeResult,
    VaultCohortResult,
    SMPCOperation,
    FHEOperation,
)

log = logging.getLogger("codeastra.client")

TOKEN_RE  = re.compile(r'\[CVT:[A-Z]+:[A-F0-9]+\]')
CDT_RE    = re.compile(r'cdt_[A-Z]+_[bto]_[a-f0-9]+')
THT_RE    = re.compile(r'tht_[A-Z]+_[a-f0-9]+')
ANY_TOKEN = re.compile(
    r'(\[CVT:[A-Z]+:[A-F0-9]+\]|cdt_[A-Z]+_[bto]_[a-f0-9]+'
    r'|tht_[A-Z]+_[a-f0-9]+|tok_[A-Z]+_[a-f0-9]+)'
)

_DEFAULT_BASE   = "https://app.codeastra.dev"
_ONPREM_DEFAULT = "http://localhost:4000"
_MAX_RETRIES    = 3
_RETRY_STATUSES = {500, 502, 503, 504}


def _detect_environment() -> str:
    env_mode = os.environ.get("CODEASTRA_MODE", "").lower()
    if env_mode in ("cloud", "onprem", "hybrid"):
        return env_mode
    try:
        s = socket.create_connection(("localhost", 4000), timeout=1)
        s.close()
        return "onprem"
    except Exception:
        pass
    return "cloud"


def _get_base_url(mode: str, base_url: str | None = None) -> str:
    if base_url:
        return base_url.rstrip("/")
    if mode in ("onprem", "hybrid"):
        return os.environ.get("CODEASTRA_ONPREM_URL", _ONPREM_DEFAULT)
    return _DEFAULT_BASE


class CodeAstraClient:
    """
    Full-featured CodeAstra client — v2.0.1.

    Quickstart:
        from codeastra import CodeAstraClient

        client = CodeAstraClient(api_key="sk-guard-xxx")

        # Tokenize sensitive fields
        tokens = client.tokenize({"ssn": "123-45-6789", "salary": 95000})

        # SMPC — compute on secret shares (no party ever sees the value)
        bundle = client.smpc_store(95000.0, label="salary")
        result = client.smpc_compute("percentage_of", {"value": bundle.bundle_id}, {"rate": 7.5})
        print(result.result)    # "7125.00" — server never saw 95000

        # FHE — compute on ciphertext (server sees zero plaintext)
        result = client.fhe_full_compute(95000.0, "percentage_of", {"rate": 7.5})
        print(result)           # 7125.0 — decrypted locally

        # Vault Compute — 155+ operations on tokens
        result = client.vault_compute("percentage_of", {"value": tokens["salary"]}, {"rate": 7.5})
        print(result.result)    # "$7,125.00"

        # Blind agent — two lines makes any agent blind
        from codeastra import BlindAgentMiddleware
        agent = BlindAgentMiddleware(your_langchain_agent, api_key="sk-guard-xxx")
        result = agent.invoke({"input": "Schedule patient appointment"})
    """

    def __init__(
        self,
        api_key:      str | None  = None,
        base_url:     str | None  = None,
        agent_id:     str         = "sdk-agent",
        timeout:      float       = 30.0,
        executor_url: str | None  = None,
        mode:         str         = "auto",
        zero_log:     bool        = False,
        onprem_dir:   str         = "./codeastra-onprem",
        verbose:      bool        = False,
        max_retries:  int         = _MAX_RETRIES,
    ):
        if not api_key:
            api_key = os.environ.get("CODEASTRA_API_KEY")
        if not api_key:
            api_key = self._auto_signup()

        if mode == "auto":
            mode = _detect_environment()

        self.api_key      = api_key
        self.agent_id     = agent_id
        self.mode         = mode
        self.zero_log     = zero_log
        self._verbose     = verbose
        self._timeout     = timeout
        self._onprem_dir  = Path(onprem_dir)
        self._max_retries = max_retries
        self.base_url     = _get_base_url(mode, base_url)

        self._headers = {
            "X-API-Key":    api_key,
            "Content-Type": "application/json",
        }
        if zero_log:
            self._headers["X-Zero-Log"] = "true"

        self._sync_client:  httpx.Client | None      = None
        self._async_client: httpx.AsyncClient | None = None

        if verbose:
            log.info("CodeAstra v2.0.1 mode=%s base=%s", mode, self.base_url)

        if mode in ("onprem", "hybrid"):
            self._setup_onprem(mode)

        if executor_url:
            self._executor_url = executor_url
            try:
                self._post("/agent/executor", {
                    "execution_url": executor_url,
                    "action_type":   "*",
                    "agent_id":      agent_id,
                    "description":   f"Auto-registered by SDK agent {agent_id} ({mode})",
                })
            except Exception as e:
                if verbose:
                    log.warning("Executor registration skipped: %s", e)

    # ── Auto-signup ───────────────────────────────────────────────────────────

    def _auto_signup(self) -> str:
        creds_path = Path.home() / ".codeastra" / "credentials"
        if creds_path.exists():
            try:
                data = json.loads(creds_path.read_text())
                key  = data.get("api_key")
                if key:
                    return key
            except Exception:
                pass
        import uuid
        email    = os.environ.get("CODEASTRA_EMAIL",    f"user-{uuid.uuid4().hex[:8]}@codeastra.local")
        password = os.environ.get("CODEASTRA_PASSWORD", uuid.uuid4().hex)
        name     = os.environ.get("CODEASTRA_NAME",     f"SDK User {uuid.uuid4().hex[:6]}")
        try:
            r = httpx.post(f"{_DEFAULT_BASE}/auth/signup",
                           json={"name": name, "email": email, "password": password},
                           timeout=10)
            if r.is_success:
                data    = r.json()
                api_key = data.get("api_key")
                if api_key:
                    creds_path.parent.mkdir(parents=True, exist_ok=True)
                    creds_path.write_text(json.dumps(
                        {"api_key": api_key, "email": email, "password": password}))
                    log.info("CodeAstra account created. Key saved to %s", creds_path)
                    return api_key
        except Exception:
            pass
        raise ConfigurationError(
            "No API key. Set CODEASTRA_API_KEY or pass api_key= "
            "or sign up at https://app.codeastra.dev"
        )

    # ── On-premise setup ──────────────────────────────────────────────────────

    def _setup_onprem(self, mode: str) -> None:
        setup_sh = self._onprem_dir / "setup.sh"
        if setup_sh.exists():
            return
        try:
            resp  = self._post("/onprem/generate", {
                "deployment_mode": "docker", "llm_provider": "ollama",
                "llm_model": "llama3", "air_gapped": mode != "hybrid",
                "name": f"codeastra-{self.agent_id}",
            })
            files = resp.get("files", {})
            if files:
                self._onprem_dir.mkdir(parents=True, exist_ok=True)
                for filename, content in files.items():
                    (self._onprem_dir / filename).write_text(content)
                if setup_sh.exists():
                    setup_sh.chmod(0o755)
                log.info("On-premise package ready: %s", self._onprem_dir)
        except Exception as e:
            log.warning("On-premise setup warning: %s — falling back to cloud", e)
            self.base_url = _DEFAULT_BASE
            self.mode     = "cloud"

    # ── HTTP primitives with retry / backoff ──────────────────────────────────

    def _get_sync(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                headers=self._headers, timeout=self._timeout,
            )
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                headers=self._headers, timeout=self._timeout,
            )
        return self._async_client

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        client = self._get_sync()
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(self._max_retries + 1):
            try:
                r = getattr(client, method)(url, **kwargs)
                if r.status_code == 429:
                    raise RateLimitError(retry_after=float(r.headers.get("Retry-After", 60.0)))
                if r.status_code == 401:
                    raise AuthError("API key missing, invalid, or revoked")
                r.raise_for_status()
                return r.json()
            except (RateLimitError, AuthError):
                raise
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRY_STATUSES:
                    raise
                last_exc = exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = NetworkError(str(exc))
            if attempt < self._max_retries:
                time.sleep(min(2 ** attempt, 30))
        raise last_exc

    async def _arequest(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        client = self._get_async_client()
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(self._max_retries + 1):
            try:
                r = await getattr(client, method)(url, **kwargs)
                if r.status_code == 429:
                    raise RateLimitError(retry_after=float(r.headers.get("Retry-After", 60.0)))
                if r.status_code == 401:
                    raise AuthError("API key missing, invalid, or revoked")
                r.raise_for_status()
                return r.json()
            except (RateLimitError, AuthError):
                raise
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRY_STATUSES:
                    raise
                last_exc = exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = NetworkError(str(exc))
            if attempt < self._max_retries:
                await asyncio.sleep(min(2 ** attempt, 30))
        raise last_exc

    def _post(self, path: str, body: dict) -> dict:
        return self._request("post", path, json=body)

    def _get(self, path: str, params: dict | None = None) -> dict:
        return self._request("get", path, params=params or {})

    def _delete(self, path: str) -> dict:
        return self._request("delete", path)

    def _patch(self, path: str, body: dict) -> dict:
        return self._request("patch", path, json=body)

    def _put(self, path: str, body: dict) -> dict:
        return self._request("put", path, json=body)

    async def _apost(self, path: str, body: dict) -> dict:
        return await self._arequest("post", path, json=body)

    async def _aget(self, path: str, params: dict | None = None) -> dict:
        return await self._arequest("get", path, params=params or {})

    async def _adelete(self, path: str) -> dict:
        return await self._arequest("delete", path)

    # ══════════════════════════════════════════════════════════════════════════
    # AUTH
    # ══════════════════════════════════════════════════════════════════════════

    def auth_login(self, email: str, password: str) -> dict:
        """Login and get API key."""
        return self._post("/auth/login", {"email": email, "password": password})

    def auth_me(self) -> dict:
        """Get current authenticated user / workspace info."""
        return self._get("/auth/me")

    # ══════════════════════════════════════════════════════════════════════════
    # WORKSPACE
    # ══════════════════════════════════════════════════════════════════════════

    def workspace_create(self, name: str, plan: str = "starter") -> dict:
        """Create a new workspace."""
        return self._post("/workspace/create", {"name": name, "plan": plan})

    def workspace_me(self) -> dict:
        """Get current workspace details."""
        return self._get("/workspace/me")

    def workspace_invite(self, email: str, role: str = "member") -> dict:
        """Invite a user to the workspace."""
        return self._post("/workspace/invite", {"email": email, "role": role})

    def workspace_members(self) -> dict:
        """List all workspace members."""
        return self._get("/workspace/members")

    def workspace_usage(self) -> dict:
        """Get workspace usage statistics."""
        return self._get("/workspace/usage")

    def workspace_keys_bootstrap(self, name: str = "default") -> dict:
        """Bootstrap initial API keys for a new workspace."""
        return self._post("/workspace/keys/bootstrap", {"name": name})

    def workspace_keys_generate(self, name: str, scopes: list | None = None) -> dict:
        """Generate a new API key for the workspace."""
        body: dict[str, Any] = {"name": name}
        if scopes: body["scopes"] = scopes
        return self._post("/workspace/keys/generate", body)

    def workspace_keys_delete(self, key_id: str) -> dict:
        """Revoke a workspace API key."""
        return self._delete(f"/workspace/keys/{key_id}")

    def workspace_keys_verify(self, api_key: str) -> dict:
        """Verify an API key belongs to this workspace."""
        return self._post("/workspace/keys/verify", {"api_key": api_key})

    # ══════════════════════════════════════════════════════════════════════════
    # INVOKE (single-shot protected LLM call)
    # ══════════════════════════════════════════════════════════════════════════

    def invoke(
        self,
        prompt:         str,
        model:          str = "claude-sonnet-4-6",
        system:         str | None = None,
        classification: str = "pii",
        session_id:     str | None = None,
    ) -> dict:
        """
        Invoke an LLM with automatic PII protection.
        CodeAstra tokenizes sensitive data before it reaches the model.

        Returns the LLM response along with a protection report.
        """
        body: dict[str, Any] = {
            "prompt": prompt, "model": model, "classification": classification,
        }
        if system:     body["system"]     = system
        if session_id: body["session_id"] = session_id
        return self._post("/invoke", body)

    # ══════════════════════════════════════════════════════════════════════════
    # SESSIONS
    # ══════════════════════════════════════════════════════════════════════════

    def session_create(
        self,
        name:           str | None = None,
        classification: str        = "pii",
        agent_id:       str | None = None,
    ) -> dict:
        """Create a protected session for multi-turn conversations."""
        body: dict[str, Any] = {"classification": classification}
        if name:     body["name"]     = name
        if agent_id: body["agent_id"] = agent_id
        return self._post("/sessions", body)

    def session_list(self, limit: int = 50) -> dict:
        """List all sessions."""
        return self._get("/sessions", {"limit": limit})

    def session_get(self, session_id: str) -> dict:
        """Get session details."""
        return self._get(f"/sessions/{session_id}")

    def session_delete(self, session_id: str) -> dict:
        """Delete a session and wipe its tokens."""
        return self._delete(f"/sessions/{session_id}")

    def session_drift(self, session_id: str) -> dict:
        """
        Detect prompt-injection drift within a session.
        Returns drift score and flagged patterns.
        """
        return self._get(f"/sessions/{session_id}/drift")

    def session_protect(self, session_id: str, content: Any) -> dict:
        """Protect content within an existing session context."""
        return self._post(f"/sessions/{session_id}/protect", {"content": content})

    def session_cert_create(self, session_id: str) -> dict:
        """Issue a tamper-evident certificate for a session."""
        return self._post(f"/sessions/{session_id}/cert", {})

    def session_cert_get(self, session_id: str) -> dict:
        """Get the certificate for a session."""
        return self._get(f"/sessions/{session_id}/cert")

    def session_cert_verify(self, session_id: str) -> dict:
        """Verify a session certificate has not been tampered with."""
        return self._post(f"/sessions/{session_id}/cert/verify", {})

    def session_cert_revoke(self, session_id: str) -> dict:
        """Revoke a session certificate."""
        return self._post(f"/sessions/{session_id}/cert/revoke", {})

    def vault_wipe_session(self, session_id: str) -> dict:
        """Wipe all vault tokens associated with a session."""
        return self._post(f"/vault/wipe/session/{session_id}", {})

    # ══════════════════════════════════════════════════════════════════════════
    # HITL — Human-in-the-Loop
    # ══════════════════════════════════════════════════════════════════════════

    def hitl_list(self, status: str | None = None, limit: int = 50) -> dict:
        """
        List pending HITL decisions.
        When an agent action requires human approval, it appears here.

        Args:
            status: "pending" | "approved" | "rejected" | None (all)
        """
        params: dict[str, Any] = {"limit": limit}
        if status: params["status"] = status
        return self._get("/hitl", params)

    def hitl_get(self, hitl_id: str) -> dict:
        """Get details of a specific HITL decision request."""
        return self._get(f"/hitl/{hitl_id}")

    def hitl_decide(
        self,
        hitl_id:   str,
        decision:  str,
        reason:    str | None = None,
        reviewer:  str | None = None,
    ) -> dict:
        """
        Approve or reject a human-in-the-loop decision.

        Args:
            hitl_id:  The HITL decision ID
            decision: "approve" or "reject"
            reason:   Optional audit reason
            reviewer: Optional reviewer identifier
        """
        body: dict[str, Any] = {"decision": decision}
        if reason:   body["reason"]   = reason
        if reviewer: body["reviewer"] = reviewer
        return self._post(f"/hitl/{hitl_id}/decide", body)

    # ══════════════════════════════════════════════════════════════════════════
    # RATE LIMITS
    # ══════════════════════════════════════════════════════════════════════════

    def rate_limit_create(
        self,
        name:     str,
        limit:    int,
        window:   str = "1h",
        scope:    str = "tenant",
    ) -> dict:
        """Create a rate limit rule (e.g. 1000 calls per hour)."""
        return self._post("/rate-limits", {
            "name": name, "limit": limit, "window": window, "scope": scope,
        })

    def rate_limit_list(self) -> dict:
        """List all rate limit rules."""
        return self._get("/rate-limits")

    def rate_limit_delete(self, limit_id: str) -> dict:
        """Delete a rate limit rule."""
        return self._delete(f"/rate-limits/{limit_id}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECURITY — Injection monitoring
    # ══════════════════════════════════════════════════════════════════════════

    def security_injections(
        self,
        limit:      int         = 50,
        session_id: str | None  = None,
        severity:   str | None  = None,
    ) -> dict:
        """
        List detected prompt-injection attempts.
        Returns flagged events with severity, pattern matched, and session context.
        """
        params: dict[str, Any] = {"limit": limit}
        if session_id: params["session_id"] = session_id
        if severity:   params["severity"]   = severity
        return self._get("/security/injections", params)

    def security_injections_stats(self) -> dict:
        """Aggregate injection detection statistics."""
        return self._get("/security/injections/stats")

    def security_falco_rules(self) -> dict:
        """Get active Falco runtime security rules."""
        return self._get("/security/falco/rules")

    def security_trivy_latest(self) -> dict:
        """Get latest Trivy vulnerability scan results."""
        return self._get("/security/trivy/latest")

    def security_soc2_evidence(self) -> dict:
        """Collect SOC2 evidence package."""
        return self._get("/security/soc2/evidence")

    def security_soc2_setup(self) -> dict:
        """Get SOC2 setup guidance for your deployment."""
        return self._get("/security/soc2/setup")

    # ══════════════════════════════════════════════════════════════════════════
    # GUARDRAILS
    # ══════════════════════════════════════════════════════════════════════════

    def guardrail_topic_add(self, name: str, description: str, action: str = "block") -> dict:
        """
        Add a topic guardrail — block or warn when the agent discusses this topic.

        Args:
            name:        Topic name, e.g. "competitor_products"
            description: What this topic covers
            action:      "block" | "warn" | "log"
        """
        return self._post("/guardrails/topics", {
            "name": name, "description": description, "action": action,
        })

    def guardrail_topics_list(self) -> dict:
        """List all topic guardrails."""
        return self._get("/guardrails/topics")

    def guardrail_topic_delete(self, name: str) -> dict:
        """Remove a topic guardrail."""
        return self._delete(f"/guardrails/topics/{name}")

    def guardrail_grounding_set(self, documents: list, strict: bool = False) -> dict:
        """Set grounding documents — agent must stay factually grounded to these."""
        return self._post("/guardrails/grounding", {
            "documents": documents, "strict": strict,
        })

    def guardrail_grounding_get(self) -> dict:
        """Get current grounding configuration."""
        return self._get("/guardrails/grounding")

    def guardrail_scan_output(self, output: str, session_id: str | None = None) -> dict:
        """
        Scan agent output for policy violations before returning to user.
        Returns pass/fail with flagged sections.
        """
        body: dict[str, Any] = {"output": output}
        if session_id: body["session_id"] = session_id
        return self._post("/guardrails/scan-output", body)

    def guardrail_events(self, limit: int = 50) -> dict:
        """List guardrail trigger events."""
        return self._get("/guardrails/events", {"limit": limit})

    def guardrail_stats(self) -> dict:
        """Aggregate guardrail statistics."""
        return self._get("/guardrails/stats")

    def guardrail_semantic_topic_add(
        self,
        name:        str,
        examples:    list[str],
        threshold:   float      = 0.8,
        action:      str        = "block",
    ) -> dict:
        """
        Add a semantic topic guardrail — blocks similar topics via embeddings,
        not just keyword matching.

        Args:
            name:      Topic name
            examples:  Example sentences representing this topic
            threshold: Similarity threshold 0–1 (default 0.8)
            action:    "block" | "warn" | "log"
        """
        return self._post("/guardrails/semantic-topics", {
            "name": name, "examples": examples,
            "threshold": threshold, "action": action,
        })

    def guardrail_semantic_topics_list(self) -> dict:
        """List all semantic topic guardrails."""
        return self._get("/guardrails/semantic-topics")

    def guardrail_semantic_topic_delete(self, name: str) -> dict:
        """Remove a semantic topic guardrail."""
        return self._delete(f"/guardrails/semantic-topics/{name}")

    def guardrail_semantic_topic_update_threshold(self, name: str, threshold: float) -> dict:
        """Update the similarity threshold for a semantic guardrail."""
        return self._patch(f"/guardrails/semantic-topics/{name}/threshold",
                           {"threshold": threshold})

    def guardrail_semantic_topic_test(self, text: str, topic: str | None = None) -> dict:
        """Test text against semantic guardrails."""
        body: dict[str, Any] = {"text": text}
        if topic: body["topic"] = topic
        return self._post("/guardrails/semantic-topics/test", body)

    def guardrail_semantic_topics_stats(self) -> dict:
        """Semantic guardrail trigger statistics."""
        return self._get("/guardrails/semantic-topics/stats")

    # ══════════════════════════════════════════════════════════════════════════
    # PROXY / CHAT — protected LLM pass-through
    # ══════════════════════════════════════════════════════════════════════════

    def proxy_chat_v2(
        self,
        messages:       list[dict],
        model:          str        = "claude-sonnet-4-6",
        system:         str | None = None,
        classification: str        = "pii",
        session_id:     str | None = None,
        max_tokens:     int        = 2048,
    ) -> dict:
        """
        Protected LLM chat — CodeAstra tokenizes PII in messages before they
        reach the model, and de-tokenizes after.  The model never sees real data.

        Compatible with OpenAI chat format.
        """
        body: dict[str, Any] = {
            "messages": messages, "model": model,
            "classification": classification, "max_tokens": max_tokens,
        }
        if system:     body["system"]     = system
        if session_id: body["session_id"] = session_id
        return self._post("/proxy/chat/v2", body)

    def proxy_chat_v3(
        self,
        messages:       list[dict],
        model:          str        = "claude-sonnet-4-6",
        system:         str | None = None,
        classification: str        = "pii",
        session_id:     str | None = None,
        max_tokens:     int        = 2048,
        guardrails:     bool       = True,
    ) -> dict:
        """
        v3 protected chat — adds guardrail scanning on output.
        Blocks policy-violating responses before they reach the user.
        """
        body: dict[str, Any] = {
            "messages": messages, "model": model,
            "classification": classification, "max_tokens": max_tokens,
            "guardrails": guardrails,
        }
        if system:     body["system"]     = system
        if session_id: body["session_id"] = session_id
        return self._post("/proxy/chat/v3", body)

    # ══════════════════════════════════════════════════════════════════════════
    # POLICIES
    # ══════════════════════════════════════════════════════════════════════════

    def policy_create(self, name: str, rules: list[dict], description: str = "") -> dict:
        """Create a data protection policy with explicit rules."""
        return self._post("/policies", {
            "name": name, "rules": rules, "description": description,
        })

    def policy_list(self) -> dict:
        """List all policies."""
        return self._get("/policies")

    def policy_get(self, name: str) -> dict:
        """Get a policy by name."""
        return self._get(f"/policies/{name}")

    def policy_delete(self, name: str) -> dict:
        """Delete a policy."""
        return self._delete(f"/policies/{name}")

    def policy_history(self, name: str) -> dict:
        """Get the version history of a policy."""
        return self._get(f"/policies/{name}/history")

    def policy_synthesize(
        self,
        description:    str,
        industry:       str | None = None,
        compliance:     list | None = None,
    ) -> dict:
        """
        AI-generated policy synthesis.
        Describe what you want in plain English — CodeAstra writes the policy.

        Args:
            description: e.g. "Block any PII from reaching the LLM in a HIPAA context"
            industry:    "healthcare" | "finance" | "legal" | ...
            compliance:  ["hipaa", "pci-dss", "gdpr", ...]
        """
        body: dict[str, Any] = {"description": description}
        if industry:   body["industry"]   = industry
        if compliance: body["compliance"] = compliance
        return self._post("/policies/synthesize", body)

    def policy_synthesize_activate(self, policy_id: str) -> dict:
        """Activate a synthesized policy."""
        return self._post(f"/policies/synthesize/{policy_id}/activate", {})

    def policy_synthesize_list(self) -> dict:
        """List synthesized policies."""
        return self._get("/policies/synthesize")

    def policy_synthesize_get(self, policy_id: str) -> dict:
        """Get a synthesized policy by ID."""
        return self._get(f"/policies/synthesize/{policy_id}")

    def policy_synthesize_delete(self, policy_id: str) -> dict:
        """Delete a synthesized policy."""
        return self._delete(f"/policies/synthesize/{policy_id}")

    # ══════════════════════════════════════════════════════════════════════════
    # AGENTS (key management + honey tools)
    # ══════════════════════════════════════════════════════════════════════════

    def agent_register(
        self,
        name:         str,
        description:  str         = "",
        capabilities: list | None = None,
    ) -> dict:
        """Register an agent with CodeAstra for key-based auth."""
        body: dict[str, Any] = {"name": name, "description": description}
        if capabilities: body["capabilities"] = capabilities
        return self._post("/agents", body)

    def agent_list(self) -> dict:
        """List all registered agents."""
        return self._get("/agents")

    def agent_delete(self, agent_id: str) -> dict:
        """Deregister an agent."""
        return self._delete(f"/agents/{agent_id}")

    def agent_register_key(self, agent_id: str, key_name: str = "default") -> dict:
        """Issue a signing key for an agent."""
        return self._post(f"/agents/{agent_id}/register-key", {"key_name": key_name})

    def agent_revoke(self, agent_id: str, reason: str = "") -> dict:
        """Revoke all keys for an agent."""
        return self._post(f"/agents/{agent_id}/revoke", {"reason": reason})

    def agent_honey_tools_add(
        self,
        agent_id: str,
        tools:    list[dict],
    ) -> dict:
        """
        Add honey-tools to an agent — fake tools that trigger alerts when
        an attacker tries to use them (prompt-injection detection).
        """
        return self._post(f"/agents/{agent_id}/honey-tools", {"tools": tools})

    def agent_honey_tools_list(self, agent_id: str) -> dict:
        """List honey-tools for an agent."""
        return self._get(f"/agents/{agent_id}/honey-tools")

    def agent_honey_tools_delete(self, agent_id: str) -> dict:
        """Remove all honey-tools from an agent."""
        return self._delete(f"/agents/{agent_id}/honey-tools")

    def agents_stats(self) -> dict:
        """Agent overview statistics."""
        return self._get("/agents/stats/overview")

    def agents_templates(self) -> dict:
        """List agent templates."""
        return self._get("/agents/templates")

    # ══════════════════════════════════════════════════════════════════════════
    # PLATFORM AGENTS (full agent platform with memory, channels, skills)
    # ══════════════════════════════════════════════════════════════════════════

    def platform_agent_create(self, name: str, config: dict) -> dict:
        """Create a full platform agent with memory, channels, and skills."""
        return self._post("/platform/agents", {"name": name, **config})

    def platform_agent_list(self) -> dict:
        """List all platform agents."""
        return self._get("/platform/agents")

    def platform_agent_get(self, agent_id: str) -> dict:
        """Get platform agent details."""
        return self._get(f"/platform/agents/{agent_id}")

    def platform_agent_run(self, agent_id: str, input: str, **kwargs) -> dict:
        """Run a platform agent."""
        return self._post(f"/platform/agents/{agent_id}/run", {"input": input, **kwargs})

    def platform_agent_update(self, agent_id: str, config: dict) -> dict:
        """Update platform agent configuration."""
        return self._post(f"/platform/agents/{agent_id}/update", config)

    def platform_agent_delete(self, agent_id: str) -> dict:
        """Delete a platform agent."""
        return self._delete(f"/platform/agents/{agent_id}")

    def platform_agent_memory_get(self, agent_id: str) -> dict:
        """Get agent memory."""
        return self._get(f"/platform/agents/{agent_id}/memory")

    def platform_agent_memory_add(self, agent_id: str, memory: dict) -> dict:
        """Add to agent memory."""
        return self._post(f"/platform/agents/{agent_id}/memory", memory)

    def platform_agent_memory_clear(self, agent_id: str) -> dict:
        """Clear agent memory."""
        return self._delete(f"/platform/agents/{agent_id}/memory")

    def platform_agent_channels_add(self, agent_id: str, channel: dict) -> dict:
        """Add a channel (Slack, email, webhook) to an agent."""
        return self._post(f"/platform/agents/{agent_id}/channels", channel)

    def platform_agent_channels_list(self, agent_id: str) -> dict:
        """List agent channels."""
        return self._get(f"/platform/agents/{agent_id}/channels")

    def platform_agent_skill_add(self, agent_id: str, skill_name: str) -> dict:
        """Add a skill to a platform agent."""
        return self._post(f"/platform/agents/{agent_id}/skills", {"skill_name": skill_name})

    def platform_agent_skill_run(
        self, agent_id: str, skill_name: str, params: dict | None = None,
    ) -> dict:
        """Run a specific skill on a platform agent."""
        return self._post(f"/platform/agents/{agent_id}/skills/{skill_name}/run", params or {})

    def platform_agent_schedule_create(self, agent_id: str, schedule: dict) -> dict:
        """Create a scheduled run for a platform agent."""
        return self._post(f"/platform/agents/{agent_id}/schedules", schedule)

    def platform_agent_schedules_list(self, agent_id: str) -> dict:
        """List agent schedules."""
        return self._get(f"/platform/agents/{agent_id}/schedules")

    def platform_agent_sensitive_data_add(self, agent_id: str, data: dict) -> dict:
        """Store sensitive data for use by a platform agent (tokenized)."""
        return self._post(f"/platform/agents/{agent_id}/sensitive-data", data)

    def platform_stats(self) -> dict:
        """Platform-wide statistics."""
        return self._get("/platform/stats")

    def platform_skills_enterprise(self) -> dict:
        """List all enterprise platform skills."""
        return self._get("/platform/skills/enterprise")

    def platform_security_stats(self) -> dict:
        """Platform security statistics."""
        return self._get("/platform/security/stats")

    # ══════════════════════════════════════════════════════════════════════════
    # AUDIT (full)
    # ══════════════════════════════════════════════════════════════════════════

    def audit_list(
        self,
        limit:      int        = 50,
        token:      str | None = None,
        agent_id:   str | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Full audit log with filtering."""
        params: dict[str, Any] = {"limit": limit}
        if token:      params["token"]      = token
        if agent_id:   params["agent_id"]   = agent_id
        if session_id: params["session_id"] = session_id
        return self._get("/audit", params)

    def audit_stats(self) -> dict:
        """Aggregate audit statistics."""
        return self._get("/audit/stats")

    def audit_stats_timeseries(
        self, period: str = "24h", granularity: str = "1h",
    ) -> dict:
        """Time-series audit statistics."""
        return self._get("/audit/stats/timeseries",
                         {"period": period, "granularity": granularity})

    def audit_stats_by_tool(self) -> dict:
        """Audit statistics broken down by tool."""
        return self._get("/audit/stats/by-tool")

    def audit_export_json(self, limit: int = 1000) -> dict:
        """Export audit log as JSON."""
        return self._get("/audit/export/json", {"limit": limit})

    def audit_export_csv(self, limit: int = 1000) -> str:
        """Export audit log as CSV string."""
        url = f"{self.base_url}/audit/export/csv"
        r = self._get_sync().get(url, params={"limit": limit})
        r.raise_for_status()
        return r.text

    def audit_shadow(
        self,
        content:    Any,
        session_id: str | None = None,
    ) -> dict:
        """
        Shadow audit — log a protected operation for compliance without
        tokenizing or blocking. Used for audit-only pipelines.
        """
        body: dict[str, Any] = {"content": content}
        if session_id: body["session_id"] = session_id
        return self._post("/audit/shadow", body)

    def audit_shadow_history(self, limit: int = 50) -> dict:
        """Get shadow audit history."""
        return self._get("/audit/shadow/history", {"limit": limit})

    def verify_audit(self) -> dict:
        """Verify audit log integrity."""
        try:
            return self._get("/audit/secure/verify")
        except Exception as e:
            return {"verified": False, "error": str(e)}

    def export_audit(self, output_path: str = "audit_report.json") -> str:
        """Export and save audit report to a local file."""
        try:
            data = self._get("/audit/secure/export")
            Path(output_path).write_text(json.dumps(data, indent=2))
            return output_path
        except Exception as e:
            return str(e)

    # ══════════════════════════════════════════════════════════════════════════
    # ALERTS
    # ══════════════════════════════════════════════════════════════════════════

    def alerts_list(self, status: str | None = None, limit: int = 50) -> dict:
        """List security alerts."""
        params: dict[str, Any] = {"limit": limit}
        if status: params["status"] = status
        return self._get("/alerts", params)

    def alert_resolve(self, alert_id: str, note: str = "") -> dict:
        """Resolve a security alert."""
        return self._post(f"/alerts/{alert_id}/resolve", {"note": note})

    # ══════════════════════════════════════════════════════════════════════════
    # WEBHOOKS
    # ══════════════════════════════════════════════════════════════════════════

    def webhook_create(
        self,
        url:    str,
        events: list[str],
        secret: str | None = None,
    ) -> dict:
        """
        Register a webhook to receive CodeAstra events.

        Events: "tokenization", "vault_access", "policy_violation",
                "injection_detected", "hitl_required", "alert"
        """
        body: dict[str, Any] = {"url": url, "events": events}
        if secret: body["secret"] = secret
        return self._post("/webhooks", body)

    def webhook_list(self) -> dict:
        """List all webhooks."""
        return self._get("/webhooks")

    def webhook_delete(self, webhook_id: str) -> dict:
        """Remove a webhook."""
        return self._delete(f"/webhooks/{webhook_id}")

    # ══════════════════════════════════════════════════════════════════════════
    # VAULT WIPE
    # ══════════════════════════════════════════════════════════════════════════

    def vault_wipe_token(self, token_id: str) -> dict:
        """
        Permanently wipe a vault token and its real value.
        Irreversible — use only when the data is no longer needed.
        """
        return self._post(f"/vault/wipe/{token_id}", {})

    def vault_wipe_cohort(self, cohort_id: str) -> dict:
        """Wipe all tokens in a cohort."""
        return self._post(f"/vault/wipe/cohort/{cohort_id}", {})

    def vault_wipe_proof(self) -> dict:
        """Get cryptographic proof that recent wipes completed."""
        return self._get("/vault/wipe/proof")

    def vault_wipe_stats(self) -> dict:
        """Wipe statistics — total wiped, pending wipes, etc."""
        return self._get("/vault/wipe/stats")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPLIANCE
    # ══════════════════════════════════════════════════════════════════════════

    def compliance_report(
        self,
        frameworks: list[str] | None = None,
        period:     str              = "30d",
    ) -> dict:
        """
        Generate a compliance report.

        Args:
            frameworks: ["hipaa", "pci-dss", "gdpr", "soc2"] or None (all)
            period:     "7d" | "30d" | "90d" | "1y"
        """
        body: dict[str, Any] = {"period": period}
        if frameworks: body["frameworks"] = frameworks
        return self._post("/compliance/report", body)

    def compliance_report_preview(self) -> dict:
        """Preview compliance posture without generating full report."""
        return self._get("/compliance/report/preview")

    def compliance_report_full(self) -> dict:
        """Get the full compliance report with all evidence."""
        return self._get("/compliance/report/full")

    def compliance_verify(self, report_id: str) -> dict:
        """Verify a compliance report is authentic."""
        return self._get(f"/compliance/verify/{report_id}")

    def compliance_stack_status(self) -> dict:
        """Status of all integrated compliance stack components."""
        return self._get("/compliance/stack/status")

    def compliance_evidence_collect(self) -> dict:
        """Collect all available compliance evidence."""
        return self._get("/compliance/evidence/collect")

    def compliance_sync(self, targets: list[str] | None = None) -> dict:
        """Sync compliance data to integrated tools (Vanta, Drata, etc.)."""
        body: dict[str, Any] = {}
        if targets: body["targets"] = targets
        return self._post("/compliance/sync", body)

    def compliance_policies_generate(self, framework: str, context: dict | None = None) -> dict:
        """
        AI-generate compliance policies for a framework.

        Args:
            framework: "hipaa" | "pci-dss" | "gdpr" | "soc2" | "iso27001"
        """
        body: dict[str, Any] = {"framework": framework}
        if context: body["context"] = context
        return self._post("/compliance/policies/generate", body)

    def compliance_policies_list(self) -> dict:
        """List all generated compliance policies."""
        return self._get("/compliance/policies/list")

    # ══════════════════════════════════════════════════════════════════════════
    # LEGAL — BAA (Business Associate Agreement)
    # ══════════════════════════════════════════════════════════════════════════

    def legal_baa_generate(
        self,
        company_name:    str,
        contact_email:   str,
        effective_date:  str | None = None,
    ) -> dict:
        """
        Generate a HIPAA Business Associate Agreement (BAA).

        Args:
            company_name:   Your company's legal name
            contact_email:  Signatory email
            effective_date: ISO date string (defaults to today)
        """
        body: dict[str, Any] = {
            "company_name": company_name, "contact_email": contact_email,
        }
        if effective_date: body["effective_date"] = effective_date
        return self._post("/legal/baa/generate", body)

    def legal_baa_status(self) -> dict:
        """Get status of your BAA (signed, pending, expired)."""
        return self._get("/legal/baa/status")

    # ══════════════════════════════════════════════════════════════════════════
    # TEE — Trusted Execution Environment
    # ══════════════════════════════════════════════════════════════════════════

    def tee_status(self) -> dict:
        """Get TEE (Trusted Execution Environment) status and attestation."""
        return self._get("/tee/status")

    # ══════════════════════════════════════════════════════════════════════════
    # RECIPES — automated protection pipelines
    # ══════════════════════════════════════════════════════════════════════════

    def recipe_create(self, name: str, steps: list[dict], description: str = "") -> dict:
        """Create a protection recipe — a reusable pipeline of protection steps."""
        return self._post("/recipes", {
            "name": name, "steps": steps, "description": description,
        })

    def recipe_list(self) -> dict:
        """List all recipes."""
        return self._get("/recipes")

    def recipe_templates(self) -> dict:
        """List built-in recipe templates."""
        return self._get("/recipes/templates")

    def recipe_from_template(self, template_id: str, overrides: dict | None = None) -> dict:
        """Create a recipe from a template."""
        body: dict[str, Any] = {"template_id": template_id}
        if overrides: body.update(overrides)
        return self._post("/recipes/from-template", body)

    def recipe_get(self, recipe_id: str) -> dict:
        """Get a recipe."""
        return self._get(f"/recipes/{recipe_id}")

    def recipe_update(self, recipe_id: str, updates: dict) -> dict:
        """Update a recipe."""
        return self._put(f"/recipes/{recipe_id}", updates)

    def recipe_delete(self, recipe_id: str) -> dict:
        """Delete a recipe."""
        return self._delete(f"/recipes/{recipe_id}")

    def recipe_test(self, recipe_id: str, sample_input: Any) -> dict:
        """Test a recipe against sample input."""
        return self._post(f"/recipes/{recipe_id}/test", {"input": sample_input})

    def recipe_test_all(self) -> dict:
        """Run all recipe tests."""
        return self._post("/recipes/test/all", {})

    def recipe_stats(self) -> dict:
        """Recipe match statistics."""
        return self._get("/recipes/stats/matches")

    # ══════════════════════════════════════════════════════════════════════════
    # BADGE — trust badge for your product
    # ══════════════════════════════════════════════════════════════════════════

    def badge_verify(self, token: str | None = None) -> dict:
        """Verify a CodeAstra trust badge."""
        body: dict[str, Any] = {}
        if token: body["token"] = token
        return self._post("/badge/verify", body)

    def badge_impressions(self) -> dict:
        """Get badge impression statistics."""
        return self._get("/badge/impressions")

    # ══════════════════════════════════════════════════════════════════════════
    # METRICS
    # ══════════════════════════════════════════════════════════════════════════

    def metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        url = f"{self.base_url}/metrics/prometheus"
        r = self._get_sync().get(url)
        r.raise_for_status()
        return r.text

    # ══════════════════════════════════════════════════════════════════════════
    # SMPC — Secure Multi-Party Computation  (BUG FIX: ttl_hours → ttl_seconds)
    # ══════════════════════════════════════════════════════════════════════════

    def smpc_store(
        self,
        value:      float,
        label:      str  = "",
        n_parties:  int  = 3,
        threshold:  int  = 2,
        ttl_hours:  int  = 2,
        party_mode: str  = "virtual",
    ) -> SMPCBundle:
        """
        Secret-share a single value across n_parties using Shamir's Secret Sharing.
        threshold parties are needed to reconstruct.  No single party ever sees the value.

        Args:
            value:      The sensitive number to protect
            label:      Human label for audit logs (no PII)
            n_parties:  Total party count (default 3)
            threshold:  Shares needed to reconstruct (default 2)
            ttl_hours:  Share lifetime in hours (default 2)
            party_mode: "virtual" (in-process) or "distributed" (separate Railway services)

        Returns:
            SMPCBundle with bundle_id — pass to smpc_compute()

        Example:
            bundle = client.smpc_store(95000.0, label="annual_salary")
            result = client.smpc_compute(
                "percentage_of",
                bundles={"value": bundle.bundle_id},
                params={"rate": 7.5},
            )
            print(result.result)   # "7125.00"
        """
        resp = self._post("/smpc/store", {
            "value":       value,
            "label":       label,
            "n_parties":   n_parties,
            "threshold":   threshold,
            "ttl_seconds": ttl_hours * 3600,   # ← server field is ttl_seconds
        })
        return SMPCBundle(
            bundle_id          = resp["bundle_id"],
            n_parties          = resp.get("n_parties", n_parties),
            threshold          = resp.get("threshold", threshold),
            label              = resp.get("label", label),
            party_mode         = party_mode,
            expires_in_seconds = ttl_hours * 3600,
        )

    def smpc_compute(
        self,
        operation:  str,
        bundles:    dict[str, str],
        params:     dict[str, Any] | None = None,
        party_mode: str                   = "virtual",
        wipe_after: bool                  = True,
        currency:   str                   = "USD",
        precision:  int                   = 2,
    ) -> SMPCResult:
        """
        Run an SMPC operation on secret-shared values.

        Args:
            operation:  e.g. "percentage_of", "sum_two", "multiply_two"
            bundles:    {"value": bundle.bundle_id} or {"value1": ..., "value2": ...}
            params:     Non-sensitive plaintext params, e.g. {"rate": 7.5}
            party_mode: "virtual" or "distributed"
            wipe_after: Wipe shares after computation (default True)

        Example (two-value multiply):
            b1 = client.smpc_store(1500.0, label="hours")
            b2 = client.smpc_store(75.0,   label="rate")
            r  = client.smpc_compute("multiply_two",
                                     {"value1": b1.bundle_id, "value2": b2.bundle_id})
            print(r.result)   # "112500.00"
        """
        resp = self._post("/smpc/compute", {
            "operation":        operation,
            "bundles":          bundles,
            "plaintext_params": params or {},
            "wipe_after":       wipe_after,
            "currency":         currency,
            "precision":        precision,
        })
        return SMPCResult(
            operation                   = resp.get("operation", operation),
            result                      = resp.get("result", ""),
            result_raw                  = resp.get("result_raw"),
            protocol                    = resp.get("protocol", ""),
            party_mode                  = resp.get("party_mode", party_mode),
            n_parties                   = resp.get("n_parties", 3),
            threshold                   = resp.get("threshold", 2),
            wiped                       = resp.get("wiped", True),
            bundle_ids_consumed         = resp.get("bundle_ids_consumed", list(bundles.values())),
            duration_ms                 = resp.get("duration_ms", 0),
            audit_id                    = resp.get("audit_id", ""),
            inputs_seen_by_compute_node = resp.get("inputs_seen_by_compute_node", True),
            error                       = resp.get("error"),
        )

    def smpc_list_operations(self, category: str | None = None) -> list[SMPCOperation]:
        """List all available SMPC operations."""
        params: dict[str, Any] = {}
        if category: params["category"] = category
        resp = self._get("/smpc/operations", params)
        return [
            SMPCOperation(
                name               = o["name"],
                description        = o.get("description", ""),
                category           = o.get("category", ""),
                protocol           = o.get("protocol", ""),
                protocol_security  = o.get("protocol_security", ""),
                required_tokens    = o.get("required_tokens", []),
                required_plaintext = o.get("required_plaintext", []),
                output_format      = o.get("output_format", ""),
                example            = o.get("example", {}),
                notes              = o.get("notes", ""),
            )
            for o in resp.get("operations", [])
        ]

    def smpc_operation_get(self, operation_name: str) -> dict:
        """Get details of a single SMPC operation."""
        return self._get(f"/smpc/operations/{operation_name}")

    def smpc_parties(self) -> dict:
        """Get status of all SMPC party nodes."""
        return self._get("/smpc/parties")

    def smpc_audit(self, limit: int = 50) -> dict:
        """Get SMPC computation audit log."""
        return self._get("/smpc/audit", {"limit": limit})

    def smpc_status(self) -> dict:
        """SMPC system status and configuration."""
        return self._get("/smpc/status")

    async def asmpc_store(
        self,
        value:      float,
        label:      str = "",
        n_parties:  int = 3,
        threshold:  int = 2,
        ttl_hours:  int = 2,
        party_mode: str = "virtual",
    ) -> SMPCBundle:
        """Async version of smpc_store."""
        resp = await self._apost("/smpc/store", {
            "value": value, "label": label, "n_parties": n_parties,
            "threshold": threshold, "ttl_seconds": ttl_hours * 3600,
        })
        return SMPCBundle(
            bundle_id=resp["bundle_id"], n_parties=resp.get("n_parties", n_parties),
            threshold=resp.get("threshold", threshold), label=resp.get("label", label),
            party_mode=party_mode, expires_in_seconds=ttl_hours * 3600,
        )

    async def asmpc_compute(
        self,
        operation:  str,
        bundles:    dict[str, str],
        params:     dict[str, Any] | None = None,
        party_mode: str                   = "virtual",
    ) -> SMPCResult:
        """Async version of smpc_compute."""
        resp = await self._apost("/smpc/compute", {
            "operation": operation, "bundles": bundles,
            "plaintext_params": params or {},
        })
        return SMPCResult(
            operation=resp.get("operation", operation), result=resp.get("result", ""),
            result_raw=resp.get("result_raw"), protocol=resp.get("protocol", ""),
            party_mode=resp.get("party_mode", party_mode), n_parties=resp.get("n_parties", 3),
            threshold=resp.get("threshold", 2), wiped=resp.get("wiped", True),
            bundle_ids_consumed=resp.get("bundle_ids_consumed", list(bundles.values())),
            duration_ms=resp.get("duration_ms", 0), audit_id=resp.get("audit_id", ""),
            inputs_seen_by_compute_node=resp.get("inputs_seen_by_compute_node", True),
            error=resp.get("error"),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FHE — Fully Homomorphic Encryption  (BUG FIX: correct API paths)
    #
    # Server paths:  /executor/compute/fhe/setup  (was wrongly /fhe/setup)
    #                /executor/compute/fhe         (was wrongly /fhe/compute)
    #                /executor/compute/fhe/operations (was wrongly /fhe/operations)
    # Also fixed:    fhe_compute now sends required public_context_b64
    # ══════════════════════════════════════════════════════════════════════════

    def fhe_setup(self, preset: str = "standard", mode: str = "stateless") -> FHESession:
        """
        Generate an FHE keypair (CKKS scheme via TenSEAL).
        The secret key is returned once and NEVER stored server-side.

        Args:
            preset: "fast" (depth 2, ~1ms) | "standard" (depth 4, ~3ms) | "deep" (depth 7, ~15ms)
            mode:   "stateless" (default) — server never holds secret key

        Security:
            secret_context_b64 contains the private key.
            Never log it, never send it to the server.
            Use session.public_only() for safe logging.
        """
        resp = self._post("/executor/compute/fhe/setup", {"preset": preset, "mode": mode})
        return FHESession(
            key_id             = resp["key_id"],
            public_context_b64 = resp["public_context_b64"],
            secret_context_b64 = resp["secret_context_b64"],
            preset             = resp.get("preset", preset),
            slots              = resp.get("slots", resp.get("slot_count", 8192)),
            max_depth          = resp.get("max_depth", 4),
        )

    def fhe_encrypt(self, value: float, session: FHESession) -> str:
        """
        Encrypt a single float locally using TenSEAL (CKKS).
        No network call.  Requires: pip install codeastra[fhe]
        """
        from .fhe import encrypt_value
        return encrypt_value(value, session)

    def fhe_encrypt_batch(self, values: list[float], session: FHESession) -> str:
        """Encrypt a batch of floats into a single CKKS ciphertext (SIMD)."""
        from .fhe import encrypt_batch
        return encrypt_batch(values, session)

    def fhe_decrypt(self, result: FHEResult, session: FHESession) -> float:
        """Decrypt an FHEResult locally.  No network — pure client-side TenSEAL."""
        return result.decrypt(session)

    def fhe_decrypt_batch(self, result: FHEResult, session: FHESession) -> list[float]:
        """Decrypt a batch FHEResult locally."""
        return result.decrypt_batch(session)

    def fhe_compute(
        self,
        operation:        str,
        encrypted_inputs: dict[str, str],
        plaintext_params: dict[str, Any] | None = None,
        key_id:           str                   = "",
        public_context_b64: str                 = "",
    ) -> FHEResult:
        """
        Run a homomorphic operation on the server.
        The server receives only ciphertext — it never decrypts.

        Args:
            operation:          e.g. "percentage_of", "add_constant", "multiply_constant"
            encrypted_inputs:   {"value": base64_ciphertext}
            plaintext_params:   {"rate": 7.5} — non-sensitive params
            key_id:             From FHESession.key_id
            public_context_b64: From FHESession.public_context_b64 (required by server)
        """
        resp = self._post("/executor/compute/fhe", {
            "operation":          operation,
            "encrypted_inputs":   encrypted_inputs,
            "plaintext_params":   plaintext_params or {},
            "key_id":             key_id,
            "public_context_b64": public_context_b64,   # ← required field, was missing
        })
        return FHEResult(
            encrypted_result_b64 = resp["encrypted_result_b64"],
            operation            = resp.get("operation", operation),
            key_id               = resp.get("key_id", key_id),
            duration_ms          = resp.get("duration_ms", 0),
            server_saw_plaintext = resp.get("server_saw_plaintext", False),
        )

    def fhe_compute_batch(
        self,
        operation:        str,
        encrypted_inputs: dict[str, str],
        plaintext_params: dict[str, Any] | None = None,
        key_id:           str                   = "",
        public_context_b64: str                 = "",
    ) -> FHEResult:
        """
        SIMD batch computation — pack thousands of values into one ciphertext.
        Same privacy guarantee as fhe_compute, much higher throughput.
        """
        resp = self._post("/executor/compute/fhe/batch", {
            "operation":          operation,
            "encrypted_inputs":   encrypted_inputs,
            "plaintext_params":   plaintext_params or {},
            "key_id":             key_id,
            "public_context_b64": public_context_b64,
        })
        return FHEResult(
            encrypted_result_b64=resp["encrypted_result_b64"],
            operation=resp.get("operation", operation),
            key_id=resp.get("key_id", key_id),
            duration_ms=resp.get("duration_ms", 0),
            server_saw_plaintext=resp.get("server_saw_plaintext", False),
        )

    def fhe_full_compute(
        self,
        value:     float,
        operation: str,
        params:    dict[str, Any] | None = None,
        session:   FHESession | None     = None,
        preset:    str                   = "standard",
    ) -> float:
        """
        One-shot FHE: encrypt locally → compute on server → decrypt locally.
        The server sees zero plaintext throughout.

        Example:
            result = client.fhe_full_compute(95000.0, "percentage_of", {"rate": 7.5})
            print(result)   # 7125.0  — server saw only ciphertext
        """
        if session is None:
            session = self.fhe_setup(preset=preset)
        ciphertext = self.fhe_encrypt(value, session)
        he_result  = self.fhe_compute(
            operation          = operation,
            encrypted_inputs   = {"value": ciphertext},
            plaintext_params   = params or {},
            key_id             = session.key_id,
            public_context_b64 = session.public_context_b64,
        )
        return self.fhe_decrypt(he_result, session)

    def fhe_list_operations(self, category: str | None = None) -> list[FHEOperation]:
        """List all FHE operations supported by the server."""
        params: dict[str, Any] = {}
        if category: params["category"] = category
        resp = self._get("/executor/compute/fhe/operations", params)
        return [
            FHEOperation(
                name             = o["name"],
                description      = o.get("description", ""),
                category         = o.get("category", ""),
                encrypted_inputs = o.get("encrypted_inputs", []),
                plaintext_params = o.get("plaintext_params", []),
                depth            = o.get("depth", 1),
                batched          = o.get("batched", False),
                example          = o.get("example", {}),
            )
            for o in resp.get("operations", [])
        ]

    def fhe_keys_list(self) -> dict:
        """List all FHE key IDs (server stores public context only)."""
        return self._get("/executor/compute/fhe/keys")

    def fhe_key_get(self, key_id: str) -> dict:
        """Get FHE key metadata."""
        return self._get(f"/executor/compute/fhe/keys/{key_id}")

    def fhe_key_delete(self, key_id: str) -> dict:
        """Delete a stored FHE key."""
        return self._delete(f"/executor/compute/fhe/keys/{key_id}")

    def fhe_status(self) -> dict:
        """FHE system status — available operations, presets, performance benchmarks."""
        return self._get("/executor/compute/fhe/status")

    async def afhe_setup(self, preset: str = "standard") -> FHESession:
        """Async version of fhe_setup."""
        resp = await self._apost("/executor/compute/fhe/setup", {"preset": preset})
        return FHESession(
            key_id=resp["key_id"], public_context_b64=resp["public_context_b64"],
            secret_context_b64=resp["secret_context_b64"],
            preset=resp.get("preset", preset),
            slots=resp.get("slots", resp.get("slot_count", 8192)),
            max_depth=resp.get("max_depth", 4),
        )

    async def afhe_compute(
        self,
        operation:          str,
        encrypted_inputs:   dict[str, str],
        plaintext_params:   dict[str, Any] | None = None,
        key_id:             str                   = "",
        public_context_b64: str                   = "",
    ) -> FHEResult:
        """Async version of fhe_compute."""
        resp = await self._apost("/executor/compute/fhe", {
            "operation": operation, "encrypted_inputs": encrypted_inputs,
            "plaintext_params": plaintext_params or {}, "key_id": key_id,
            "public_context_b64": public_context_b64,
        })
        return FHEResult(
            encrypted_result_b64=resp["encrypted_result_b64"],
            operation=resp.get("operation", operation), key_id=resp.get("key_id", key_id),
            duration_ms=resp.get("duration_ms", 0),
            server_saw_plaintext=resp.get("server_saw_plaintext", False),
        )

    async def afhe_full_compute(
        self,
        value:     float,
        operation: str,
        params:    dict[str, Any] | None = None,
        session:   FHESession | None     = None,
        preset:    str                   = "standard",
    ) -> float:
        """Async version of fhe_full_compute."""
        if session is None:
            session = await self.afhe_setup(preset=preset)
        ciphertext = self.fhe_encrypt(value, session)
        he_result  = await self.afhe_compute(
            operation=operation, encrypted_inputs={"value": ciphertext},
            plaintext_params=params or {}, key_id=session.key_id,
            public_context_b64=session.public_context_b64,
        )
        return self.fhe_decrypt(he_result, session)

    # ══════════════════════════════════════════════════════════════════════════
    # VAULT COMPUTE  (BUG FIX: correct API paths)
    #
    # Server paths:  /executor/compute        (was wrongly /vault/compute)
    #                /executor/compute/cohort  (was wrongly /vault/cohort/compute)
    # Also fixed:    cohort uses token_list (list) + epsilon, not tokens dict + dp_epsilon
    # ══════════════════════════════════════════════════════════════════════════

    def vault_compute(
        self,
        operation:        str,
        tokens:           dict[str, str],
        plaintext_params: dict[str, Any] | None = None,
        wipe_after:       bool                  = True,
        currency:         str                   = "USD",
        session_id:       str | None            = None,
    ) -> VaultComputeResult:
        """
        Compute on vault-tokenized values.  The vault resolves tokens to real
        values transiently, computes, then wipes.  The agent never sees plaintext.

        Args:
            operation:        e.g. "percentage_of", "compound_interest", "format_ssn"
            tokens:           {"principal": "[CVT:AMT:A1B2]", "rate": "[CVT:RATE:C3D4]"}
            plaintext_params: Non-sensitive params, e.g. {"periods": 12}
            wipe_after:       Wipe tokens after compute (default True)
            currency:         Currency for formatted output (default "USD")

        Security:
            server_saw_plaintext=True — vault reconstructs transiently.
            For zero-knowledge compute use FHE or SMPC.

        Example:
            tokens = client.tokenize({"salary": 95000})
            result = client.vault_compute(
                "percentage_of",
                tokens={"value": tokens["salary"]},
                plaintext_params={"rate": 7.5},
            )
            print(result.result)   # "$7,125.00"
            print(result.wiped)    # True
        """
        body: dict[str, Any] = {
            "operation":        operation,
            "tokens":           tokens,
            "plaintext_params": plaintext_params or {},
            "wipe_after":       wipe_after,
            "currency":         currency,
        }
        if session_id: body["session_id"] = session_id
        resp = self._post("/executor/compute", body)
        return VaultComputeResult(
            operation            = resp.get("operation", operation),
            result               = resp.get("result", ""),
            result_raw           = resp.get("result_raw"),
            tokens_consumed      = list(tokens.values()),
            wiped                = resp.get("wiped", True),
            duration_ms          = resp.get("duration_ms", 0),
            error                = resp.get("error"),
            server_saw_plaintext = not resp.get("inputs_seen_by_llm", False),
        )

    def vault_compute_operations(self, category: str | None = None) -> dict:
        """List all 155+ vault compute operations."""
        params: dict[str, Any] = {}
        if category: params["category"] = category
        return self._get("/executor/compute/operations", params)

    def vault_compute_operation_get(self, operation_name: str) -> dict:
        """Get details of a single vault compute operation."""
        return self._get(f"/executor/compute/operations/{operation_name}")

    def vault_compute_audit(self, limit: int = 50, session_id: str | None = None) -> dict:
        """Get vault compute audit log (no real values — token IDs and results only)."""
        params: dict[str, Any] = {"limit": limit}
        if session_id: params["session_id"] = session_id
        return self._get("/executor/compute/audit", params)

    def vault_compute_status(self) -> dict:
        """Vault compute system status and capabilities overview."""
        return self._get("/executor/compute/status")

    def vault_cohort_compute(
        self,
        operation:   str,
        token_list:  list[str],
        cohort_id:   str | None  = None,
        epsilon:     float       = 1.0,
        sensitivity: float | None = None,
        wipe_after:  bool        = True,
        session_id:  str | None  = None,
    ) -> VaultCohortResult:
        """
        Differential-privacy aggregate computation over a list of vault tokens.
        Applies Laplace/Gaussian noise calibrated to the epsilon privacy budget.

        Args:
            operation:   e.g. "cohort_average", "cohort_sum", "cohort_median"
            token_list:  List of CVT tokens (NOT a dict — a flat list)
            cohort_id:   Optional cohort identifier for filtering
            epsilon:     DP privacy budget (smaller = more private, default 1.0)
            sensitivity: Optional sensitivity override
            wipe_after:  Wipe tokens after compute (default True)

        Example:
            tokens = [client.tokenize({"salary": s})["salary"] for s in [90000, 95000, 105000]]
            result = client.vault_cohort_compute(
                "cohort_average", token_list=tokens, epsilon=1.0,
            )
            print(result.result)      # "$96,412.00 (DP-protected, ε=1.0)"
            print(result.dp_applied)  # True
        """
        body: dict[str, Any] = {
            "operation":  operation,
            "token_list": token_list,         # ← server field is token_list (list), not tokens dict
            "epsilon":    epsilon,            # ← server field is epsilon, not dp_epsilon
            "wipe_after": wipe_after,
        }
        if cohort_id:   body["cohort_id"]   = cohort_id
        if sensitivity: body["sensitivity"] = sensitivity
        if session_id:  body["session_id"]  = session_id
        resp = self._post("/executor/compute/cohort", body)
        return VaultCohortResult(
            operation   = resp.get("operation", operation),
            result      = resp.get("result", ""),
            result_raw  = resp.get("result_raw"),
            token_count = len(token_list),
            dp_applied  = True,
            dp_epsilon  = epsilon,
            error       = resp.get("error"),
        )

    async def avault_compute(
        self,
        operation:        str,
        tokens:           dict[str, str],
        plaintext_params: dict[str, Any] | None = None,
        wipe_after:       bool                  = True,
    ) -> VaultComputeResult:
        """Async version of vault_compute."""
        resp = await self._apost("/executor/compute", {
            "operation": operation, "tokens": tokens,
            "plaintext_params": plaintext_params or {}, "wipe_after": wipe_after,
        })
        return VaultComputeResult(
            operation=resp.get("operation", operation), result=resp.get("result", ""),
            result_raw=resp.get("result_raw"), tokens_consumed=list(tokens.values()),
            wiped=resp.get("wiped", True), duration_ms=resp.get("duration_ms", 0),
            error=resp.get("error"),
            server_saw_plaintext=not resp.get("inputs_seen_by_llm", False),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PROTECT TEXT
    # ══════════════════════════════════════════════════════════════════════════

    def protect_text(self, text: str, classification: str = "pii") -> str:
        """
        Tokenize any free-text string — emails, SSNs, card numbers, names.
        Agent receives protected text; never the original.
        """
        resp = self._post("/protect/text", {"text": text, "classification": classification})
        return resp.get("protected_text", text)

    def protect_text_full(self, text: str, classification: str = "pii") -> dict:
        """Tokenize text and return full response with entity list."""
        return self._post("/protect/text", {"text": text, "classification": classification})

    async def aprotect_text(self, text: str, classification: str = "pii") -> str:
        """Async version of protect_text."""
        resp = await self._apost("/protect/text", {"text": text, "classification": classification})
        return resp.get("protected_text", text)

    async def aprotect_text_full(self, text: str, classification: str = "pii") -> dict:
        """Async version of protect_text_full."""
        return await self._apost("/protect/text", {"text": text, "classification": classification})

    # ══════════════════════════════════════════════════════════════════════════
    # VAULT RESOLVE
    # ══════════════════════════════════════════════════════════════════════════

    def vault_resolve(self, token: str) -> dict:
        """Resolve a token to its real value.  For trusted executors only."""
        return self._post("/vault/resolve", {"token": token})

    def vault_resolve_batch(self, tokens: list[str]) -> dict:
        """Resolve multiple tokens at once."""
        return self._post("/vault/resolve-batch", {"tokens": tokens})

    async def avault_resolve(self, token: str) -> dict:
        return await self._apost("/vault/resolve", {"token": token})

    async def avault_resolve_batch(self, tokens: list[str]) -> dict:
        return await self._apost("/vault/resolve-batch", {"tokens": tokens})

    # ══════════════════════════════════════════════════════════════════════════
    # THINKING TOKENS
    # ══════════════════════════════════════════════════════════════════════════

    def think_mint(
        self,
        real_value:        str,
        data_type:         str,
        facts:             dict,
        cohort_id:         str | None  = None,
        signal_conditions: list | None = None,
        ttl_hours:         int         = 720,
    ) -> dict:
        """Mint a ThinkingToken.  Real value goes to vault — never seen again."""
        body: dict[str, Any] = {
            "real_value": real_value, "data_type": data_type,
            "facts": facts, "ttl_hours": ttl_hours,
        }
        if cohort_id:         body["cohort_id"]         = cohort_id
        if signal_conditions: body["signal_conditions"] = signal_conditions
        return self._post("/think/mint", body)

    def think_mint_batch(self, tokens: list) -> dict:
        return self._post("/think/mint/batch", {"tokens": tokens})

    def think_query(
        self,
        query:           str,
        cohort_id:       str | None = None,
        include_reasons: bool       = False,
        top_k:           int | None = None,
    ) -> dict:
        body: dict[str, Any] = {"query": query, "include_reasons": include_reasons}
        if cohort_id: body["cohort_id"] = cohort_id
        if top_k:     body["top_k"]     = top_k
        return self._post("/think/query", body)

    def think_query_cohort(self, query: str, cohort_id: str, **kwargs) -> dict:
        return self.think_query(query, cohort_id=cohort_id, **kwargs)

    def think_signal(self, cohort_id: str) -> dict:
        return self._post("/think/signal", {"cohort_id": cohort_id})

    def think_get(self, token_id: str) -> dict:
        return self._get(f"/think/{token_id}")

    def think_memory(self, token_id: str) -> dict:
        return self._get(f"/think/{token_id}/memory")

    def think_evolve(self, token_id: str) -> dict:
        return self._post(f"/think/{token_id}/evolve", {})

    def think_audit(self, token_id: str) -> dict:
        return self._get(f"/think/{token_id}/audit")

    def think_revoke(self, token_id: str) -> dict:
        return self._delete(f"/think/{token_id}")

    def think_stats(self) -> dict:
        return self._get("/think/stats")

    def think_ollama_status(self) -> dict:
        return self._get("/think/ollama/status")

    async def athink_mint(
        self, real_value: str, data_type: str, facts: dict,
        cohort_id: str | None = None, signal_conditions: list | None = None,
        ttl_hours: int = 720,
    ) -> dict:
        body: dict[str, Any] = {
            "real_value": real_value, "data_type": data_type,
            "facts": facts, "ttl_hours": ttl_hours,
        }
        if cohort_id:         body["cohort_id"]         = cohort_id
        if signal_conditions: body["signal_conditions"] = signal_conditions
        return await self._apost("/think/mint", body)

    async def athink_mint_batch(self, tokens: list) -> dict:
        return await self._apost("/think/mint/batch", {"tokens": tokens})

    async def athink_query(
        self, query: str, cohort_id: str | None = None, include_reasons: bool = False,
    ) -> dict:
        body: dict[str, Any] = {"query": query, "include_reasons": include_reasons}
        if cohort_id: body["cohort_id"] = cohort_id
        return await self._apost("/think/query", body)

    async def athink_signal(self, cohort_id: str) -> dict:
        return await self._apost("/think/signal", {"cohort_id": cohort_id})

    # ══════════════════════════════════════════════════════════════════════════
    # THINKING EXECUTOR
    # ══════════════════════════════════════════════════════════════════════════

    def executor_register_integration(self, name: str, config: dict) -> dict:
        return self._post("/executor/integrations", {"name": name, "config": config})

    def executor_list_integrations(self) -> dict:
        return self._get("/executor/integrations")

    def executor_delete_integration(self, integration_id: str) -> dict:
        return self._delete(f"/executor/integrations/{integration_id}")

    def executor_register_rule(
        self,
        name:            str,
        integration:     str,
        action_type:     str,
        conditions:      list,
        action_template: dict | None = None,
        priority:        int         = 5,
    ) -> dict:
        body: dict[str, Any] = {
            "name": name, "integration": integration, "action_type": action_type,
            "conditions": conditions, "priority": priority,
        }
        if action_template: body["action_template"] = action_template
        return self._post("/executor/rules", body)

    def executor_list_rules(self) -> dict:
        return self._get("/executor/rules")

    def executor_delete_rule(self, rule_id: str) -> dict:
        return self._delete(f"/executor/rules/{rule_id}")

    def executor_run(self, token_id: str, dry_run: bool = False) -> dict:
        return self._post("/executor/run", {"token_id": token_id, "dry_run": dry_run})

    def executor_run_cohort(
        self, cohort_id: str, dry_run: bool = False, limit: int | None = None,
    ) -> dict:
        body: dict[str, Any] = {"cohort_id": cohort_id, "dry_run": dry_run}
        if limit: body["limit"] = limit
        return self._post("/executor/run/cohort", body)

    def executor_log(self, limit: int = 50, token_id: str | None = None) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if token_id: params["token_id"] = token_id
        return self._get("/executor/log", params)

    def executor_learning(self) -> dict:
        return self._get("/executor/learning")

    def executor_status(self) -> dict:
        return self._get("/executor/status")

    def executor_supported(self) -> dict:
        return self._get("/executor/supported")

    async def aexecutor_run(self, token_id: str, dry_run: bool = False) -> dict:
        return await self._apost("/executor/run", {"token_id": token_id, "dry_run": dry_run})

    async def aexecutor_run_cohort(self, cohort_id: str, dry_run: bool = False) -> dict:
        return await self._apost("/executor/run/cohort",
                                  {"cohort_id": cohort_id, "dry_run": dry_run})

    # ══════════════════════════════════════════════════════════════════════════
    # OMEGA TOKENS
    # ══════════════════════════════════════════════════════════════════════════

    def omega_mint(
        self,
        real_value:        str,
        data_type:         str,
        facts:             dict | None = None,
        allowed_actions:   list | None = None,
        allowed_targets:   list | None = None,
        signal_conditions: list | None = None,
        pipeline_id:       str | None  = None,
        ttl_hours:         int         = 720,
        max_uses:          int | None  = None,
    ) -> dict:
        body: dict[str, Any] = {
            "real_value": real_value, "data_type": data_type, "ttl_hours": ttl_hours,
        }
        if facts:             body["facts"]             = facts
        if allowed_actions:   body["allowed_actions"]   = allowed_actions
        if allowed_targets:   body["allowed_targets"]   = allowed_targets
        if signal_conditions: body["signal_conditions"] = signal_conditions
        if pipeline_id:       body["pipeline_id"]       = pipeline_id
        if max_uses:          body["max_uses"]           = max_uses
        return self._post("/omega/mint", body)

    def omega_mint_batch(self, tokens: list) -> dict:
        return self._post("/omega/mint/batch", {"tokens": tokens})

    def omega_get(self, token_id: str) -> dict:
        return self._get(f"/omega/{token_id}")

    def omega_execute(
        self, token_id: str, action_type: str,
        target_url: str | None = None, field_name: str | None = None,
    ) -> dict:
        return self._post(f"/omega/{token_id}/execute", {
            "action_type": action_type, "target_url": target_url, "field_name": field_name,
        })

    def omega_proof(self, token_id: str, fact: str) -> dict:
        return self._get(f"/omega/{token_id}/proof", {"fact": fact})

    def omega_audit(self, token_id: str) -> dict:
        return self._get(f"/omega/{token_id}/audit")

    def omega_revoke(self, token_id: str) -> dict:
        return self._delete(f"/omega/{token_id}")

    # ══════════════════════════════════════════════════════════════════════════
    # CORE TOKENIZE / EXECUTE / GRANT  (preserved from v1.x)
    # ══════════════════════════════════════════════════════════════════════════

    def tokenize(self, data: dict, classification: str = "pii", ttl_hours: int = 24) -> dict:
        resp = self._post("/vault/store", {
            "data": data, "agent_id": self.agent_id,
            "classification": classification, "ttl_hours": ttl_hours,
        })
        return resp.get("tokens", {})

    def execute(self, action_type: str, params: dict, pipeline_id: str | None = None) -> dict:
        body = {"agent_id": self.agent_id, "action_type": action_type, "params": params}
        if pipeline_id:
            body["pipeline_id"] = pipeline_id
            return self._post("/pipeline/action", body)
        return self._post("/agent/action", body)

    def grant(
        self,
        receiving_agent: str,
        tokens:          list,
        allowed_actions: list       = [],
        pipeline_id:     str | None = None,
        purpose:         str | None = None,
    ) -> dict:
        return self._post("/vault/grant", {
            "granting_agent": self.agent_id, "receiving_agent": receiving_agent,
            "tokens": tokens, "allowed_actions": allowed_actions,
            "pipeline_id": pipeline_id, "purpose": purpose,
        })

    def audit(self, pipeline_id: str | None = None, token: str | None = None) -> list:
        params = {}
        if pipeline_id: params["pipeline_id"] = pipeline_id
        if token:       params["token"]        = token
        return self._get("/pipeline/audit", params).get("audit", [])

    def stats(self) -> dict:
        return self._get("/vault/stats")

    async def atokenize(self, data: dict, classification: str = "pii", ttl_hours: int = 24) -> dict:
        resp = await self._apost("/vault/store", {
            "data": data, "agent_id": self.agent_id,
            "classification": classification, "ttl_hours": ttl_hours,
        })
        return resp.get("tokens", {})

    async def aexecute(self, action_type: str, params: dict, pipeline_id: str | None = None) -> dict:
        body: dict[str, Any] = {
            "agent_id": self.agent_id, "action_type": action_type, "params": params,
        }
        if pipeline_id:
            body["pipeline_id"] = pipeline_id
            return await self._apost("/pipeline/action", body)
        return await self._apost("/agent/action", body)

    async def agrant(
        self,
        receiving_agent: str,
        tokens:          list,
        allowed_actions: list       = [],
        pipeline_id:     str | None = None,
    ) -> dict:
        return await self._apost("/vault/grant", {
            "granting_agent": self.agent_id, "receiving_agent": receiving_agent,
            "tokens": tokens, "allowed_actions": allowed_actions, "pipeline_id": pipeline_id,
        })

    # ── Smart Tokens ──────────────────────────────────────────────────────────

    def smart_tokenize(
        self, real_value: str, data_type: str, allowed_actions: list = [],
        allowed_targets: list = [], allowed_fields: list = [],
        max_uses: int = 1, ttl_seconds: int = 86400, semantic_label: str | None = None,
    ) -> dict:
        return self._post("/vault/smart-token", {
            "real_value": real_value, "data_type": data_type, "agent_id": self.agent_id,
            "allowed_actions": allowed_actions, "allowed_targets": allowed_targets,
            "allowed_fields": allowed_fields, "max_uses": max_uses,
            "ttl_seconds": ttl_seconds, "semantic_label": semantic_label,
        })

    def smart_tokenize_batch(self, tokens: list) -> list:
        return self._post("/vault/smart-token/batch", {
            "agent_id": self.agent_id, "tokens": tokens,
        }).get("tokens", [])

    def smart_token_info(self, token_id: str) -> dict:
        return self._get(f"/vault/smart-token/{token_id}")

    def smart_token_execute(
        self, token_id: str, action_type: str | None = None,
        target_url: str | None = None, field_name: str | None = None,
    ) -> dict:
        return self._post("/vault/smart-token/execute", {
            "token_id": token_id, "action_type": action_type,
            "target_url": target_url, "field_name": field_name, "agent_id": self.agent_id,
        })

    def smart_token_revoke(self, token_id: str, reason: str = "manual") -> dict:
        try:
            return self._get(f"/vault/smart-token/{token_id}/revoke")
        except Exception:
            return self._post(f"/vault/smart-token/{token_id}/revoke", {"reason": reason})

    def smart_token_audit(self, token_id: str) -> list:
        return self._get(f"/vault/smart-token/{token_id}/audit").get("audit", [])

    def smart_token_types(self) -> list:
        return self._get("/vault/smart-token-types").get("types", [])

    async def asmart_tokenize(
        self, real_value: str, data_type: str, allowed_actions: list = [],
        allowed_targets: list = [], allowed_fields: list = [],
        max_uses: int = 1, ttl_seconds: int = 86400,
    ) -> dict:
        return await self._apost("/vault/smart-token", {
            "real_value": real_value, "data_type": data_type, "agent_id": self.agent_id,
            "allowed_actions": allowed_actions, "allowed_targets": allowed_targets,
            "allowed_fields": allowed_fields, "max_uses": max_uses, "ttl_seconds": ttl_seconds,
        })

    async def asmart_token_execute(
        self, token_id: str, action_type: str | None = None,
        target_url: str | None = None, field_name: str | None = None,
    ) -> dict:
        return await self._apost("/vault/smart-token/execute", {
            "token_id": token_id, "action_type": action_type,
            "target_url": target_url, "field_name": field_name, "agent_id": self.agent_id,
        })

    # ── Blind RAG ─────────────────────────────────────────────────────────────

    def rag_ingest(
        self, content: dict, doc_type: str, title: str | None = None,
        source: str | None = None, classification: str = "pii",
    ) -> dict:
        return self._post("/rag/ingest", {
            "content": content, "doc_type": doc_type, "agent_id": self.agent_id,
            "title": title, "source": source, "classification": classification,
        })

    def rag_ingest_batch(self, documents: list) -> dict:
        return self._post("/rag/ingest/batch", {"agent_id": self.agent_id, "documents": documents})

    def rag_search(
        self, query: str, doc_type: str | None = None,
        top_k: int = 5, min_score: float = 0.3,
    ) -> dict:
        body: dict[str, Any] = {"query": query, "top_k": top_k, "min_score": min_score}
        if doc_type: body["doc_type"] = doc_type
        return self._post("/rag/search", body)

    def rag_delete(self, doc_id: str) -> dict:
        return self._delete(f"/rag/document/{doc_id}")

    def rag_stats(self) -> dict:
        return self._get("/rag/stats")

    async def arag_ingest(
        self, content: dict, doc_type: str,
        title: str | None = None, classification: str = "pii",
    ) -> dict:
        return await self._apost("/rag/ingest", {
            "content": content, "doc_type": doc_type,
            "agent_id": self.agent_id, "title": title, "classification": classification,
        })

    async def arag_search(
        self, query: str, doc_type: str | None = None,
        top_k: int = 5, min_score: float = 0.3,
    ) -> dict:
        body: dict[str, Any] = {"query": query, "top_k": top_k, "min_score": min_score}
        if doc_type: body["doc_type"] = doc_type
        return await self._apost("/rag/search", body)

    # ── Policy ────────────────────────────────────────────────────────────────

    def register_sensitive_type(
        self, fields: list, prefixes: list = [], doc_types: list = [],
    ) -> dict:
        return self._post("/policy/sensitivity/fields", {
            "fields": fields, "prefixes": prefixes, "doc_types": doc_types,
        })

    def set_sensitivity_policy(
        self,
        sensitive_fields:      list | None = None,
        sensitive_prefixes:    list | None = None,
        sensitive_doc_types:   list | None = None,
        field_classifications: dict | None = None,
        strict_mode:           bool | None = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if sensitive_fields      is not None: body["sensitive_fields"]      = sensitive_fields
        if sensitive_prefixes    is not None: body["sensitive_prefixes"]    = sensitive_prefixes
        if sensitive_doc_types   is not None: body["sensitive_doc_types"]   = sensitive_doc_types
        if field_classifications is not None: body["field_classifications"] = field_classifications
        if strict_mode           is not None: body["strict_mode"]           = strict_mode
        return self._post("/policy/sensitivity", body)

    def get_sensitivity_policy(self) -> dict:
        return self._get("/policy/sensitivity")

    def test_sensitivity(
        self, content: dict, field_policy: dict = {},
        sensitive_fields: list = [], tokenize_all: bool = False,
    ) -> dict:
        return self._post("/policy/sensitivity/test", {
            "content": content, "field_policy": field_policy,
            "sensitive_fields": sensitive_fields, "tokenize_all": tokenize_all,
        })

    def set_context(
        self,
        industry:               str | None = None,
        data_scope:             str | None = None,
        classification_level:   str | None = None,
        extra_sensitive_fields: list       = [],
        safe_fields:            list       = [],
        strict_mode:            bool       = False,
    ) -> dict:
        return self._post("/policy/context", {
            "industry": industry, "data_scope": data_scope,
            "classification_level": classification_level,
            "extra_sensitive_fields": extra_sensitive_fields,
            "safe_fields": safe_fields, "context_strict_mode": strict_mode,
        })

    def set_anonymity(
        self,
        k_minimum:          int         = 5,
        suppress_singleton: bool        = True,
        auto_bucket:        bool        = True,
        detect_narrowing:   bool        = True,
        quasi_identifiers:  list | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "k_minimum": k_minimum, "suppress_singleton": suppress_singleton,
            "auto_bucket": auto_bucket, "detect_narrowing": detect_narrowing,
        }
        if quasi_identifiers is not None: body["quasi_identifiers"] = quasi_identifiers
        return self._post("/policy/anonymity", body)

    def test_context(self, content: dict, context: dict, field_policy: dict = {}) -> dict:
        return self._post("/policy/context/test", {
            "content": content, "context": context, "field_policy": field_policy,
        })

    def smart_ingest(
        self, content: dict, doc_type: str, field_policy: dict = {},
        sensitive_fields: list = [], tokenize_all: bool = False,
        title: str | None = None, classification: str = "pii",
    ) -> dict:
        return self._post("/rag/ingest", {
            "content": content, "doc_type": doc_type, "agent_id": self.agent_id,
            "title": title, "classification": classification,
            "field_policy": field_policy, "sensitive_fields": sensitive_fields,
            "tokenize_all": tokenize_all,
        })

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def extract_tokens(obj: Any) -> list:
        text = json.dumps(obj) if not isinstance(obj, str) else obj
        return ANY_TOKEN.findall(text)

    @staticmethod
    def contains_token(val: Any) -> bool:
        text = json.dumps(val) if not isinstance(val, str) else str(val)
        return bool(ANY_TOKEN.search(text))

    @staticmethod
    def is_token(val: str) -> bool:
        return bool(ANY_TOKEN.fullmatch(val.strip()))

    @staticmethod
    def verify_executor_call(payload: str, signature: str, secret: str) -> bool:
        expected = "sha256=" + hmac.new(
            secret.encode(),
            payload.encode() if isinstance(payload, str) else payload,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def set_zero_log(self, enabled: bool = True) -> None:
        self.zero_log = enabled
        if enabled:
            self._headers["X-Zero-Log"] = "true"
        else:
            self._headers.pop("X-Zero-Log", None)
        self._sync_client  = None
        self._async_client = None

    def info(self) -> dict:
        return {
            "version":  "2.0.1",
            "mode":     self.mode,
            "base_url": self.base_url,
            "agent_id": self.agent_id,
            "zero_log": self.zero_log,
        }

    def close(self) -> None:
        if self._sync_client:  self._sync_client.close()

    async def aclose(self) -> None:
        if self._async_client: await self._async_client.aclose()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()
    async def __aenter__(self): return self
    async def __aexit__(self, *_): await self.aclose()

    def __repr__(self) -> str:
        return f"CodeAstraClient(v2.0.1, mode={self.mode!r}, agent_id={self.agent_id!r})"

