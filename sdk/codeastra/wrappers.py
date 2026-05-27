"""
Framework-specific wrappers and decorators — v2.0.0.

blind_tool         — decorator for individual LangChain/CrewAI tools (fail-closed)
BlindCrewAIAgent   — CrewAI-specific wrapper with crew-level pipeline support
BlindAutoGPTAgent  — AutoGPT-style wrapper with .step() interception
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from .client import CodeAstraClient
from .middleware import (
    BlindAgentMiddleware,
    _extract_sensitive,
    _tokenize_in_place,
    _scan_text_for_pii,
)
from .exceptions import FailClosedError, ExecutionAbortedError

log = logging.getLogger("codeastra.wrappers")


# ── @blind_tool decorator ─────────────────────────────────────────────────────

def blind_tool(
    api_key:        str,
    agent_id:       str  = "sdk-agent",
    base_url:       str  = "https://app.codeastra.dev",
    classification: str  = "pii",
    fail_closed:    bool = True,
):
    """
    Decorator that makes a single tool function blind.  Fail-closed by default.

    Any sensitive data in the return value is tokenized before the agent sees it.
    If tokenization fails and fail_closed=True (default), raises ExecutionAbortedError
    instead of silently returning the unprotected value.

    Usage:
        @blind_tool(api_key="sk-guard-xxx", classification="phi")
        def get_patient_record(patient_id: str) -> dict:
            return db.get_patient(patient_id)
            # Agent receives tokens, not real patient data

        # As a LangChain tool:
        @tool
        @blind_tool(api_key="sk-guard-xxx")
        def lookup_patient(patient_id: str) -> str:
            return fetch_from_ehr(patient_id)
    """
    _client = CodeAstraClient(api_key, base_url, agent_id)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            sensitive = _extract_sensitive(result)
            if not sensitive:
                return result
            try:
                tokens     = _client.tokenize(sensitive, classification=classification)
                val_to_tok = {str(sensitive[k]): v for k, v in tokens.items()}
                return _tokenize_in_place(result, val_to_tok)
            except FailClosedError:
                raise
            except Exception as exc:
                if fail_closed:
                    raise ExecutionAbortedError(
                        f"blind_tool({fn.__name__}) tokenization failed — data blocked. "
                        f"Error: {exc}"
                    ) from exc
                log.warning(
                    "codeastra.blind_tool.FAIL_OPEN fn=%s error=%s — returning unprotected",
                    fn.__name__, exc,
                )
                return result

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            result = await fn(*args, **kwargs)
            sensitive = _extract_sensitive(result)
            if not sensitive:
                return result
            try:
                tokens     = await _client.atokenize(sensitive, classification=classification)
                val_to_tok = {str(sensitive[k]): v for k, v in tokens.items()}
                return _tokenize_in_place(result, val_to_tok)
            except FailClosedError:
                raise
            except Exception as exc:
                if fail_closed:
                    raise ExecutionAbortedError(
                        f"blind_tool({fn.__name__}) async tokenization failed — data blocked. "
                        f"Error: {exc}"
                    ) from exc
                log.warning(
                    "codeastra.blind_tool.FAIL_OPEN fn=%s error=%s", fn.__name__, exc,
                )
                return result

        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper

    return decorator


# ── BlindCrewAIAgent ──────────────────────────────────────────────────────────

class BlindCrewAIAgent(BlindAgentMiddleware):
    """
    CrewAI-specific blind wrapper.  Fail-closed by default.

    Usage:
        from crewai import Agent, Task, Crew
        from codeastra import BlindCrewAIAgent

        intake_agent = Agent(role="intake", tools=[ehr_tool])
        blind_intake = BlindCrewAIAgent(
            intake_agent,
            api_key     = "sk-guard-xxx",
            agent_id    = "intake-agent",
            pipeline_id = "patient_intake_001",
        )

        # In your Crew, use blind_intake instead of intake_agent
        # All tool outputs are tokenized. Agent reasons on tokens only.
        blind_intake.grant_to("scheduling-agent", ["schedule_appointment"])
    """

    def kickoff(self, *args, **kwargs):
        """Proxy CrewAI Crew.kickoff()"""
        if hasattr(self._agent, "kickoff"):
            result = self._agent.kickoff(*args, **kwargs)
            return self._blind_output(result)
        return self.run(*args, **kwargs)

    async def akickoff(self, *args, **kwargs):
        if hasattr(self._agent, "akickoff"):
            result = await self._agent.akickoff(*args, **kwargs)
            return await self._ablind_output(result)
        return await self.arun(*args, **kwargs)

    def execute_task(self, task: Any, *args, **kwargs):
        """Intercept CrewAI task execution."""
        if hasattr(self._agent, "execute_task"):
            result = self._agent.execute_task(task, *args, **kwargs)
            return self._blind_output(result)
        return self.run(*args, **kwargs)


# ── BlindAutoGPTAgent ─────────────────────────────────────────────────────────

class BlindAutoGPTAgent(BlindAgentMiddleware):
    """
    AutoGPT-style blind wrapper.  Intercepts .step() and .run() calls.

    Usage:
        from codeastra import BlindAutoGPTAgent

        agent = BlindAutoGPTAgent(
            your_autogpt_agent,
            api_key  = "sk-guard-xxx",
            agent_id = "autogpt-agent",
        )
        while not agent.is_done():
            agent.step()
    """

    def step(self, *args, **kwargs):
        """Intercept single step execution."""
        if hasattr(self._agent, "step"):
            result = self._agent.step(*args, **kwargs)
            return self._blind_output(result)
        raise AttributeError("Underlying agent has no .step() method")

    async def astep(self, *args, **kwargs):
        if hasattr(self._agent, "astep"):
            result = await self._agent.astep(*args, **kwargs)
            return await self._ablind_output(result)
        raise AttributeError("Underlying agent has no .astep() method")

    def is_done(self) -> bool:
        if hasattr(self._agent, "is_done"):
            return self._agent.is_done()
        return False
