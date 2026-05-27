"""
BlindAgentMiddleware v2.0.0 — drop-in middleware for LangChain, CrewAI, AutoGPT.

Two lines.  Any agent becomes blind.  And if CodeAstra fails, execution is
ABORTED — data never reaches the LLM.

    from codeastra import BlindAgentMiddleware
    agent = BlindAgentMiddleware(your_langchain_agent, api_key="sk-guard-xxx")

How it works:
  1. Scans every prompt for raw PII/PHI/PCI before passing to the agent
  2. Intercepts every tool result before the agent sees it
  3. Tokenizes sensitive fields — stores real values in CodeAstra vault
  4. Returns tokens to the agent — agent reasons on tokens, never real data
  5. Scans agent output for any leaked values and catches them too

Fail-closed guarantee:
  If tokenization fails for any reason (network, auth, timeout, server error),
  _blind_output raises ExecutionAbortedError instead of silently returning
  the unprotected data.  No sensitive data ever reaches the LLM on failure.
  Set fail_closed=False ONLY if you explicitly accept the risk.

Supports: LangChain AgentExecutor, CrewAI Agent, AutoGPT-style run() agents,
          any object with .run() / .invoke() / .chat() / .step()
"""
from __future__ import annotations

import re
import json
import logging
import functools
from typing import Any, Callable, Optional

from .client import CodeAstraClient, TOKEN_RE
from .exceptions import ExecutionAbortedError, FailClosedError

log = logging.getLogger("codeastra.middleware")

# Fields that trigger automatic tokenization when found in tool output
_PII_FIELDS = {
    "name", "first_name", "last_name", "full_name",
    "email", "email_address",
    "phone", "phone_number", "mobile",
    "ssn", "social_security", "social_security_number",
    "dob", "date_of_birth", "birthday",
    "address", "street", "zip", "postal_code",
    "credit_card", "card_number", "cvv", "expiry",
    "mrn", "patient_id", "npi",
    "account_number", "routing_number", "iban",
    "passport", "license", "drivers_license",
    "ip", "ip_address", "mac_address",
    "username", "user_id", "employee_id",
}

_PHI_FIELDS = {
    "diagnosis", "icd_code", "medication", "prescription",
    "allergy", "lab_result", "test_result", "condition",
    "treatment", "procedure", "insurance_id", "member_id",
}

_PCI_FIELDS = {
    "card_number", "credit_card", "cvv", "expiry",
    "account_number", "routing_number",
}

_ALL_SENSITIVE = _PII_FIELDS | _PHI_FIELDS | _PCI_FIELDS


def _classify(fields: set) -> str:
    if fields & _PCI_FIELDS: return "pci"
    if fields & _PHI_FIELDS: return "phi"
    return "pii"


def _extract_sensitive(obj: Any) -> dict:
    """Walk a dict/str/list and extract fields that look sensitive."""
    found: dict = {}

    def _walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                key = k.lower().replace(" ", "_").replace("-", "_")
                if key in _ALL_SENSITIVE:
                    if isinstance(v, str) and v and not TOKEN_RE.fullmatch(v.strip()):
                        found[k] = v
                else:
                    _walk(v)
        elif isinstance(o, list):
            for item in o:
                _walk(item)

    if isinstance(obj, dict):
        _walk(obj)
    elif isinstance(obj, str):
        try:
            _walk(json.loads(obj))
        except Exception:
            pass
    return found


def _tokenize_in_place(obj: Any, token_map: dict) -> Any:
    """Replace real values with tokens throughout a nested object."""
    if isinstance(obj, str):
        for real, token in token_map.items():
            obj = obj.replace(str(real), token)
        return obj
    elif isinstance(obj, dict):
        return {k: _tokenize_in_place(v, token_map) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_tokenize_in_place(i, token_map) for i in obj]
    return obj


# Regex patterns for detecting raw PII in free text
_re = re
_PATTERNS = {
    "ssn": _re.compile(
        r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b'
    ),
    "credit_card": _re.compile(
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?'
        r'|5[1-5][0-9]{14}'
        r'|3[47][0-9]{13}'
        r'|6(?:011|5[0-9]{2})[0-9]{12}'
        r'|(?:2131|1800|35\d{3})\d{11})\b'
    ),
    "email": _re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
    "phone": _re.compile(
        r'(?:\+?1[-.\ s]?)?(?:\(?\d{3}\)?[-.\ s]?)\d{3}[-.\ s]?\d{4}\b'
    ),
    "dob": _re.compile(
        r'\b(?:0[1-9]|1[0-2])[\/\-](?:0[1-9]|[12]\d|3[01])[\/\-](?:19|20)\d{2}\b'
        r'|\b(?:19|20)\d{2}[\/\-](?:0[1-9]|1[0-2])[\/\-](?:0[1-9]|[12]\d|3[01])\b'
    ),
    "mrn": _re.compile(r'\bMRN[-:\s]*[A-Z0-9][-A-Z0-9]{2,14}\b', _re.IGNORECASE),
    "ip_address": _re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    ),
    "clearance": _re.compile(
        r'\b(?:TOP\s+SECRET|TS)(?:/(?:SCI|SAP|SI|TK|HCS|G|NOFORN))?\b'
        r'|\bSECRET(?:/(?:SAP|SCI|NOFORN|REL))?\b'
        r'|\bCONFIDENTIAL(?:/NOFORN)?\b', _re.IGNORECASE
    ),
    "operation_code": _re.compile(
        r'\bOP-[A-Z0-9]{3,20}\b|\bOPERATION\s+[A-Z]{3,20}\b', _re.IGNORECASE
    ),
    "asset_id": _re.compile(r'\b(?:ASSET|FAC|FACILITY)-[A-Z0-9][-A-Z0-9]{2,14}\b', _re.IGNORECASE),
    "employee_id": _re.compile(r'\bEMP-[0-9]{4,8}\b', _re.IGNORECASE),
    "case_ref": _re.compile(r'\b(?:LEGAL|CASE|MATTER)-[A-Z0-9][-A-Z0-9]{2,20}\b', _re.IGNORECASE),
    "account_ref": _re.compile(r'\bACC-[0-9]{6,12}\b', _re.IGNORECASE),
    "financial_amount": _re.compile(
        r'\$\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b|\$\d{4,}(?:\.\d{2})?\b'
    ),
}


def _luhn_check(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _scan_text_for_pii(text: str) -> dict:
    """Scan free text for raw PII/PHI/PCI patterns."""
    found: dict = {}
    if not isinstance(text, str):
        return found
    for field, pattern in _PATTERNS.items():
        matches = pattern.findall(text)
        for i, match in enumerate(matches):
            val = match.strip() if isinstance(match, str) else match[0].strip()
            if not val or TOKEN_RE.search(val):
                continue
            if field == "credit_card":
                if not _luhn_check(re.sub(r'\D', '', val)):
                    continue
            key = f"{field}_{i}" if i > 0 else field
            found[key] = val
    return found


def _scan_obj_for_pii(obj: Any) -> dict:
    """Scan any object for raw PII in free text."""
    if isinstance(obj, str):
        return _scan_text_for_pii(obj)
    elif isinstance(obj, dict):
        combined: dict = {}
        for v in obj.values():
            combined.update(_scan_obj_for_pii(v))
        return combined
    elif isinstance(obj, list):
        combined = {}
        for item in obj:
            combined.update(_scan_obj_for_pii(item))
        return combined
    return {}


def _blind_text(text: str, token_map: dict) -> str:
    if not isinstance(text, str):
        return text
    for real, token in token_map.items():
        if real and real in text:
            text = text.replace(real, token)
    return text


def _blind_any(obj: Any, token_map: dict) -> Any:
    if isinstance(obj, str):
        return _blind_text(obj, token_map)
    elif isinstance(obj, dict):
        return {k: _blind_any(v, token_map) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_blind_any(i, token_map) for i in obj]
    return obj


class BlindAgentMiddleware:
    """
    Drop-in middleware that makes any agent framework blind to real data.

    Works with:
      - LangChain:  AgentExecutor, RunnableAgent, Chain
      - CrewAI:     Agent, Crew
      - AutoGPT:    any object with .run() / .step()
      - Generic:    anything with .run() / .invoke() / .chat()

    Fail-closed by default:
      If tokenization fails, the call raises ExecutionAbortedError instead of
      silently returning unprotected data.  Data never reaches the LLM on error.

    Usage:
        agent = BlindAgentMiddleware(langchain_executor, api_key="sk-guard-xxx")
        result = agent.invoke({"input": "Schedule appointment for patient"})
    """

    def __init__(
        self,
        agent:          Any,
        api_key:        str | None           = None,
        agent_id:       str                  = "sdk-agent",
        base_url:       str | None           = None,
        classification: str                  = "pii",
        pipeline_id:    str | None           = None,
        on_tokenize:    Callable | None      = None,
        verbose:        bool                 = False,
        mode:           str                  = "auto",
        zero_log:       bool                 = False,
        executor_url:   str | None           = None,
        onprem_dir:     str                  = "./codeastra-onprem",
        fail_closed:    bool                 = True,
    ):
        self._agent          = agent
        self._client         = CodeAstraClient(
            api_key=api_key, base_url=base_url, agent_id=agent_id,
            mode=mode, zero_log=zero_log, executor_url=executor_url,
            onprem_dir=onprem_dir, verbose=verbose,
        )
        self._classification = classification
        self._pipeline_id    = pipeline_id
        self._on_tokenize    = on_tokenize
        self._verbose        = verbose
        self._fail_closed    = fail_closed   # read by apply_fail_closed_to_middleware

        self._session_tokens: dict = {}
        self._value_to_token: dict = {}

        self._patch_tools()

    # ── tool patching ─────────────────────────────────────────────────────────

    def _patch_tools(self) -> None:
        agent = self._agent

        if hasattr(agent, "tools") and isinstance(agent.tools, list):
            for i, tool in enumerate(agent.tools):
                agent.tools[i] = self._wrap_tool(tool)
            if self._verbose:
                log.info("CodeAstra: patched %d LangChain tools", len(agent.tools))

        if hasattr(agent, "steps"):
            for step in agent.steps:
                if hasattr(step, "tool"):
                    step.tool = self._wrap_tool(step.tool)

        if hasattr(agent, "agent") and hasattr(agent.agent, "tools"):
            tools = agent.agent.tools
            for i, tool in enumerate(tools):
                tools[i] = self._wrap_tool(tool)

    def _wrap_tool(self, tool: Any) -> Any:
        if hasattr(tool, "_run"):
            original_run  = tool._run
            original_arun = getattr(tool, "_arun", None)

            @functools.wraps(original_run)
            def patched_run(*args, **kwargs):
                result = original_run(*args, **kwargs)
                return self._blind_output(result)

            tool._run = patched_run

            if original_arun:
                @functools.wraps(original_arun)
                async def patched_arun(*args, **kwargs):
                    result = await original_arun(*args, **kwargs)
                    return await self._ablind_output(result)
                tool._arun = patched_arun

            return tool

        if callable(tool):
            @functools.wraps(tool)
            def wrapped(*args, **kwargs):
                result = tool(*args, **kwargs)
                return self._blind_output(result)
            return wrapped

        return tool

    # ── core blindness logic — FAIL-CLOSED ────────────────────────────────────
    #
    # These methods intentionally do NOT catch exceptions from tokenize().
    # apply_fail_closed_to_middleware() wraps them so that ANY exception
    # becomes ExecutionAbortedError when _fail_closed=True (default).
    # If you see an ExecutionAbortedError, that means CodeAstra correctly
    # stopped data from reaching your LLM when something went wrong.

    def _blind_output(self, output: Any) -> Any:
        """
        Tokenize all sensitive fields in tool output.
        Raises on failure when fail_closed=True (default).
        Never silently returns unprotected data.
        """
        sensitive = _extract_sensitive(output)
        if not sensitive:
            return output

        classification = _classify(set(k.lower() for k in sensitive))
        tokens = self._client.tokenize(sensitive, classification=classification)

        for field, token in tokens.items():
            real_val = sensitive.get(field)
            if real_val:
                self._value_to_token[str(real_val)] = token
            self._session_tokens[field] = token

        if self._on_tokenize:
            for field, token in tokens.items():
                try: self._on_tokenize(field, token)
                except Exception: pass

        if self._verbose:
            log.info("CodeAstra: tokenized %d field(s): %s", len(tokens), list(tokens.keys()))

        return _tokenize_in_place(output, self._value_to_token)

    async def _ablind_output(self, output: Any) -> Any:
        """Async version of _blind_output.  Fail-closed by default."""
        sensitive = _extract_sensitive(output)
        if not sensitive:
            return output

        classification = _classify(set(k.lower() for k in sensitive))
        tokens = await self._client.atokenize(sensitive, classification=classification)

        for field, token in tokens.items():
            real_val = sensitive.get(field)
            if real_val:
                self._value_to_token[str(real_val)] = token
            self._session_tokens[field] = token

        if self._verbose:
            log.info("CodeAstra: tokenized %d field(s): %s", len(tokens), list(tokens.keys()))

        return _tokenize_in_place(output, self._value_to_token)

    def _scan_and_blind_input(self, *args, **kwargs) -> tuple:
        """
        Scan all input args/kwargs for raw PII before passing to the agent.
        Raises on failure (fail-closed via outer wrapper).
        """
        all_text = json.dumps(list(args)) + json.dumps(kwargs)
        raw_pii  = _scan_obj_for_pii(all_text)

        if not raw_pii:
            return args, kwargs

        classification = _classify(set(k.split("_")[0] for k in raw_pii))
        minted = self._client.tokenize(raw_pii, classification=classification)
        for field, token in minted.items():
            real_val = raw_pii.get(field)
            if real_val:
                self._value_to_token[real_val] = token
            self._session_tokens[field] = token

        if self._verbose:
            log.info("CodeAstra input scan: tokenized %d value(s): %s",
                     len(minted), list(minted.keys()))

        new_args   = tuple(_blind_any(a, self._value_to_token) for a in args)
        new_kwargs = {k: _blind_any(v, self._value_to_token) for k, v in kwargs.items()}
        return new_args, new_kwargs

    def _scan_output(self, result: Any) -> Any:
        """Second-pass output scan — catches any real values that leaked through."""
        if self._value_to_token:
            result = _blind_any(result, self._value_to_token)

        new_pii = _scan_obj_for_pii(result)
        if new_pii:
            classification = _classify(set(k.split("_")[0] for k in new_pii))
            minted = self._client.tokenize(new_pii, classification=classification)
            for field, token in minted.items():
                real_val = new_pii.get(field)
                if real_val:
                    self._value_to_token[real_val] = token
                self._session_tokens[field] = token
            result = _blind_any(result, self._value_to_token)
            if self._verbose:
                log.info("CodeAstra output gate: caught %d leaked value(s): %s",
                         len(minted), list(minted.keys()))
        return result

    async def _ascan_output(self, result: Any) -> Any:
        """Async version of _scan_output."""
        if self._value_to_token:
            result = _blind_any(result, self._value_to_token)

        new_pii = _scan_obj_for_pii(result)
        if new_pii:
            classification = _classify(set(k.split("_")[0] for k in new_pii))
            minted = await self._client.atokenize(new_pii, classification=classification)
            for field, token in minted.items():
                real_val = new_pii.get(field)
                if real_val:
                    self._value_to_token[real_val] = token
                self._session_tokens[field] = token
            result = _blind_any(result, self._value_to_token)
            if self._verbose:
                log.info("CodeAstra output gate (async): caught %d leaked value(s)",
                         len(minted))
        return result

    # ── proxy methods ─────────────────────────────────────────────────────────

    def run(self, *args, **kwargs):
        args, kwargs = self._scan_and_blind_input(*args, **kwargs)
        result = self._agent.run(*args, **kwargs)
        result = self._blind_output(result)
        return self._scan_output(result)

    def invoke(self, *args, **kwargs):
        args, kwargs = self._scan_and_blind_input(*args, **kwargs)
        result = self._agent.invoke(*args, **kwargs)
        if isinstance(result, dict) and "output" in result:
            result["output"] = self._blind_output(result["output"])
            result["output"] = self._scan_output(result["output"])
            return result
        result = self._blind_output(result)
        return self._scan_output(result)

    def chat(self, *args, **kwargs):
        args, kwargs = self._scan_and_blind_input(*args, **kwargs)
        result = self._agent.chat(*args, **kwargs)
        result = self._blind_output(result)
        return self._scan_output(result)

    async def arun(self, *args, **kwargs):
        args, kwargs = self._scan_and_blind_input(*args, **kwargs)
        result = await self._agent.arun(*args, **kwargs)
        result = await self._ablind_output(result)
        return await self._ascan_output(result)

    async def ainvoke(self, *args, **kwargs):
        args, kwargs = self._scan_and_blind_input(*args, **kwargs)
        result = await self._agent.ainvoke(*args, **kwargs)
        if isinstance(result, dict) and "output" in result:
            result["output"] = await self._ablind_output(result["output"])
            result["output"] = await self._ascan_output(result["output"])
            return result
        result = await self._ablind_output(result)
        return await self._ascan_output(result)

    # ── pipeline: grant tokens to next agent ─────────────────────────────────

    def grant_to(
        self,
        next_agent_id:   str,
        allowed_actions: list[str] = [],
        purpose:         str | None = None,
    ) -> dict:
        """Grant all session tokens to the next agent in the pipeline."""
        tokens = list(self._session_tokens.values())
        if not tokens:
            return {"granted": False, "error": "No tokens minted this session"}
        return self._client.grant(
            receiving_agent=next_agent_id, tokens=tokens,
            allowed_actions=allowed_actions, pipeline_id=self._pipeline_id, purpose=purpose,
        )

    async def agrant_to(self, next_agent_id: str, allowed_actions: list[str] = []) -> dict:
        tokens = list(self._session_tokens.values())
        if not tokens:
            return {"granted": False, "error": "No tokens minted this session"}
        return await self._client.agrant(
            receiving_agent=next_agent_id, tokens=tokens,
            allowed_actions=allowed_actions, pipeline_id=self._pipeline_id,
        )

    def execute(self, action_type: str, params: dict) -> dict:
        return self._client.execute(action_type, params, self._pipeline_id)

    async def aexecute(self, action_type: str, params: dict) -> dict:
        return await self._client.aexecute(action_type, params, self._pipeline_id)

    def audit(self) -> list:
        return self._client.audit(pipeline_id=self._pipeline_id)

    @property
    def tokens(self) -> dict:
        return dict(self._session_tokens)

    @property
    def token_count(self) -> int:
        return len(self._session_tokens)

    @staticmethod
    def scan_text(text: str) -> dict:
        """Scan free text for raw PII patterns.  Returns matches for review."""
        return _scan_text_for_pii(text)

    def __getattr__(self, name: str):
        return getattr(self._agent, name)

    def __repr__(self) -> str:
        return (
            f"BlindAgentMiddleware(agent={type(self._agent).__name__}, "
            f"agent_id={self._client.agent_id!r}, "
            f"tokens_minted={self.token_count}, "
            f"fail_closed={self._fail_closed})"
        )

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._client.aclose()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()
    async def __aenter__(self): return self
    async def __aexit__(self, *_): await self.aclose()
